use std::{
    collections::{BTreeMap, HashMap},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU8, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use atomic_refcell::AtomicRefCell;

use anyhow::{Context, Result};
use nice_plug_core::{
    audio_setup::BufferConfig,
    params::{InternalParamMut, Param, Params, internals::ParamPtr, range::IntRange},
    plugin::{
        ParamValue, Plugin, PluginState, PreparedFields, RestoreContext, RestoreOrigin,
        StateRefusal,
    },
};

/// A state object travelling to the audio thread for a restore at the end of a process call,
/// and back to its origin with the outcome.
///
/// `prepared` was staged off the audio thread, before this travelled anywhere (T1 maintainer
/// fix, section 8, S-28): under [`Plugin::ATOMIC_STATE_RESTORE`] it is the plugin's own
/// [`Plugin::prepare_fields()`] result; otherwise [`NoFields`], unused by the non-atomic path.
/// It rides inside this struct, which the audio thread only ever borrows in place in the
/// [`RestoreMailbox`] and never owns, so dropping (and so freeing) it happens on the restoring
/// thread, never at commit time.
pub(crate) struct PendingRestore<P: Plugin> {
    pub state: PluginState,
    pub origin: RestoreOrigin,
    pub outcome: core::result::Result<(), StateRefusal>,
    pub prepared: Box<dyn PreparedFields<P>>,
}

/// The hand-off between the one thread restoring a state and the audio thread, which commits it
/// at the end of a process call (T1 round-3 fix; pool round-3 codex, astra). The pinned wrappers
/// used a zero-capacity crossbeam channel to hand a state over and the same (later a one-slot)
/// channel to hand it back. Crossbeam's zero-capacity `try_recv()` takes the channel's own
/// `std::sync::Mutex` on every call -- every process call, restore or not -- and may allocate
/// (hence the pinned `permit_alloc` and its `FIXME`), and a `try_send()` towards a receiver
/// blocked in `recv()` takes that receiver's waker mutex and unparks its thread: a lock, a
/// possible wait and a system call reachable from the audio callback.
///
/// The mailbox is one slot and a phase. The audio side's whole job is
/// [`service()`][Self::service]: one atomic load when nothing is posted; otherwise one
/// compare-and-swap to claim the posted restore, an `AtomicRefCell` borrow (one atomic operation,
/// never a wait, and never contended under the protocol), the commit in place, and one atomic
/// store to publish it. It never allocates, frees, locks, waits, wakes a thread, reads a clock or
/// logs. Everything that can wait -- polling for the outcome, the deadline, admitting the next
/// restorer -- is on the restoring side.
pub(crate) struct RestoreMailbox<T> {
    /// [`EMPTY`], [`POSTED`], [`COMMITTING`] or [`DONE`].
    phase: AtomicU8,
    /// The restore in flight: written by the restoring side only before `POSTED` and after
    /// `DONE` (or after it took an unclaimed one back), borrowed by the audio side only while
    /// `COMMITTING`.
    slot: AtomicRefCell<Option<T>>,
    /// One restorer at a time, so the one slot is never overwritten. Only restoring threads
    /// ever take this lock; the audio side never touches it.
    restorers: parking_lot::Mutex<()>,
}

/// Nothing in flight.
const EMPTY: u8 = 0;
/// A restore is in the slot, waiting for the audio side.
const POSTED: u8 = 1;
/// The audio side claimed the restore and is committing it.
const COMMITTING: u8 = 2;
/// The audio side committed it; the restoring side takes it back.
const DONE: u8 = 3;

/// The restoring side's first, and longest, wait between two looks at the phase.
const FIRST_NAP: Duration = Duration::from_micros(50);
const LONGEST_NAP: Duration = Duration::from_millis(1);

impl<T> RestoreMailbox<T> {
    pub fn new() -> Self {
        Self {
            phase: AtomicU8::new(EMPTY),
            slot: AtomicRefCell::new(None),
            restorers: parking_lot::Mutex::new(()),
        }
    }

    /// Restoring side: become the one restorer, waiting for a restore already in flight to
    /// finish -- at most `within` (`None` if it is still in flight then; the atomic contract
    /// refuses with [`StateRefusal::BUSY`]), or for as long as it takes when `within` is `None`
    /// (a plugin without the atomic contract: the pinned wrappers made a second restorer wait,
    /// never refused it; T1 round-4 fix). Hold the guard while the mailbox is in use. Never
    /// called by the audio thread.
    pub fn restorer(&self, within: Option<Duration>) -> Option<parking_lot::MutexGuard<'_, ()>> {
        match within {
            Some(within) => self.restorers.try_lock_for(within),
            None => Some(self.restorers.lock()),
        }
    }

    /// Restoring side, holding [`restorer()`][Self::restorer]: hand `item` to the audio side and
    /// wait for its commit. `Ok` carries it back as the commit left it. With `take_within`, an
    /// item the audio side has not claimed by then comes back untouched as `Err` (the caller
    /// decides whether to commit it itself); once claimed, the commit is always waited for.
    /// Never called by the audio thread: this polls, with naps.
    pub fn hand_to_audio(&self, item: T, take_within: Option<Duration>) -> Result<T, T> {
        debug_assert_eq!(self.phase.load(Ordering::Acquire), EMPTY);
        *self.slot.borrow_mut() = Some(item);
        self.phase.store(POSTED, Ordering::Release);

        let posted = Instant::now();
        let mut nap = FIRST_NAP;
        loop {
            let phase = self.phase.load(Ordering::Acquire);
            if phase == DONE {
                return Ok(self.take_back());
            }
            let expired = take_within.is_some_and(|limit| posted.elapsed() >= limit);
            // Still unclaimed past the limit: take it back. If the audio side claimed it in the
            // meantime the exchange fails, and its commit is waited for like any other.
            if phase == POSTED
                && expired
                && self
                    .phase
                    .compare_exchange(POSTED, EMPTY, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                let item = self.slot.borrow_mut().take();
                return Err(item.expect("an unclaimed restore is still in the slot"));
            }
            thread::sleep(nap);
            nap = (nap * 2).min(LONGEST_NAP);
        }
    }

    /// The committed item, and the mailbox free again.
    fn take_back(&self) -> T {
        let item = self.slot.borrow_mut().take();
        self.phase.store(EMPTY, Ordering::Release);
        item.expect("a committed restore is in the slot")
    }

    /// Audio side, at the end of a process call: commit a posted restore in place, if there is
    /// one. One atomic load when there is none. The item is only borrowed here; it goes back to
    /// the restoring side, which drops it.
    pub fn service(&self, commit: impl FnOnce(&mut T)) {
        if self.phase.load(Ordering::Relaxed) != POSTED
            || self
                .phase
                .compare_exchange(POSTED, COMMITTING, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
        {
            return;
        }
        match self.slot.try_borrow_mut() {
            Ok(mut slot) => {
                if let Some(item) = slot.as_mut() {
                    commit(item);
                }
            }
            // Unreachable under the protocol (the restoring side borrows only outside
            // COMMITTING); never a wait or a panic here: leave it posted for the next call.
            Err(_) => {
                self.phase.store(POSTED, Ordering::Release);
                return;
            }
        }
        self.phase.store(DONE, Ordering::Release);
    }
}

/// What a calling-thread atomic commit returns instead of committing while a process call is in
/// flight (T1 round-4 fix; pool round-4 gemini, astra, flash). CLAP and VST3 take the plugin per
/// slice, so its lock alone is free between two slices of one call; the wrappers' restore entry
/// points treat this as "hand it to the audio thread, whose call ends shortly", never as a
/// refusal of the state.
pub(crate) const CALL_IN_FLIGHT: StateRefusal = StateRefusal::new(
    "a process call is in flight; the restore is handed to its end",
    0,
);

/// Raised by a CLAP or VST3 process call from before its first slice until it returns, between
/// its slices included, and lowered on every way out (T1 round-4 fix). One atomic store each
/// way; nothing else. A calling-thread commit reads the flag after taking the plugin's lock:
/// the call raised it before its first slice took (and released) that lock, so a commit that
/// gets the lock between two slices always sees it.
pub(crate) struct CallInFlight<'a>(&'a AtomicBool);

impl<'a> CallInFlight<'a> {
    pub fn enter(flag: &'a AtomicBool) -> Self {
        flag.store(true, Ordering::SeqCst);
        Self(flag)
    }
}

impl Drop for CallInFlight<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

/// Whether [`apply_params()`] may log what it cannot apply. The atomic commit is
/// [`Diagnostics::Silent`]: validation already refused every value it could not apply, and the
/// one diagnostic left (an unknown id) is reported off the audio thread by
/// [`report_unknown_params()`] before the hand-off (T1 round-3 fix).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Diagnostics {
    /// Pristine: warn (debug builds only) from whichever thread commits.
    Log,
    /// Nothing at all from the committing thread.
    Silent,
}

/// The one diagnostic an accepted atomic restore can still meet -- a saved parameter id the
/// plugin no longer has, which `apply_params()` skips -- logged on the restoring thread before
/// the hand-off instead of from the committing (audio) thread (T1 round-3 fix). Debug builds
/// only, like the pristine warning it replaces.
pub(crate) fn report_unknown_params(
    state: &PluginState,
    params_getter: impl Fn(&str) -> Option<ParamPtr>,
) {
    #[cfg(debug_assertions)]
    for param_id_str in state.params.keys() {
        if params_getter(param_id_str.as_str()).is_none() {
            crate::nice_warn!("Unknown parameter: {}", param_id_str);
        }
    }
    #[cfg(not(debug_assertions))]
    let _ = (state, params_getter);
}

/// Create a parameters iterator from the hashtables stored in the plugin wrappers. This avoids
/// having to call `.param_map()` again, which may include expensive user written code.
pub(crate) fn make_params_iter<'a>(
    param_by_hash: &'a HashMap<u32, ParamPtr>,
    param_id_to_hash: &'a HashMap<String, u32>,
) -> impl IntoIterator<Item = (&'a String, ParamPtr)> {
    param_id_to_hash.iter().filter_map(|(param_id_str, hash)| {
        let param_ptr = param_by_hash.get(hash)?;
        Some((param_id_str, *param_ptr))
    })
}

/// Create a getter function that gets a parameter from the hashtables stored in the plugin by
/// string ID.
pub(crate) fn make_params_getter<'a>(
    param_by_hash: &'a HashMap<u32, ParamPtr>,
    param_id_to_hash: &'a HashMap<String, u32>,
) -> impl Fn(&str) -> Option<ParamPtr> + 'a {
    |param_id_str| {
        param_id_to_hash
            .get(param_id_str)
            .and_then(|hash| param_by_hash.get(hash))
            .copied()
    }
}

/// Serialize a plugin's state to a state object. This is separate from [`serialize_json()`] to
/// allow passing the raw object directly to the plugin. The parameters are not pulled directly from
/// `plugin_params` by default to avoid unnecessary allocations in the `.param_map()` method, as the
/// plugin wrappers will already have a list of parameters handy. See [`make_params_iter()`].
pub(crate) unsafe fn serialize_object<'a, P: Plugin>(
    plugin_params: Arc<dyn Params>,
    params_iter: impl IntoIterator<Item = (&'a String, ParamPtr)>,
) -> PluginState {
    // We'll serialize parameter values as a simple `string_param_id: display_value` map.
    // NOTE: If the plugin is being modulated (and the plugin is a CLAP plugin in Bitwig Studio),
    //       then this should save the values without any modulation applied to it
    let params: BTreeMap<_, _> = unsafe {
        params_iter
            .into_iter()
            .map(|(param_id_str, param_ptr)| match param_ptr {
                ParamPtr::FloatParam(p) => (
                    param_id_str.clone(),
                    ParamValue::F32((*p).unmodulated_plain_value()),
                ),
                ParamPtr::IntParam(p) => (
                    param_id_str.clone(),
                    ParamValue::I32((*p).unmodulated_plain_value()),
                ),
                ParamPtr::BoolParam(p) => (
                    param_id_str.clone(),
                    ParamValue::Bool((*p).unmodulated_plain_value()),
                ),
                ParamPtr::EnumParam(p) => (
                    // Enums are either serialized based on the active variant's index (which may not be
                    // the same as the discriminator), or a custom set stable string ID. The latter
                    // allows the variants to be reordered.
                    param_id_str.clone(),
                    match (*p).unmodulated_plain_id() {
                        Some(id) => ParamValue::String(id.to_owned()),
                        None => ParamValue::I32((*p).unmodulated_plain_value()),
                    },
                ),
            })
            .collect()
    };

    // The plugin can also persist arbitrary fields alongside its parameters. This is useful for
    // storing things like sample data.
    let fields = plugin_params.serialize_fields();

    PluginState {
        version: String::from(P::VERSION),
        params,
        fields,
    }
}

/// Serialize a plugin's state to a vector containing JSON data. This can (and should) be shared
/// across plugin formats. If the `zstd` feature is enabled, then the state will be compressed using
/// Zstandard.
pub(crate) unsafe fn serialize_json<'a, P: Plugin>(
    plugin_params: Arc<dyn Params>,
    params_iter: impl IntoIterator<Item = (&'a String, ParamPtr)>,
) -> Result<Vec<u8>> {
    let plugin_state = unsafe { serialize_object::<P>(plugin_params, params_iter) };
    let json = serde_json::to_vec(&plugin_state).context("Could not format as JSON")?;

    #[cfg(feature = "zstd")]
    {
        let compressed = zstd::encode_all(json.as_slice(), zstd::DEFAULT_COMPRESSION_LEVEL)
            .context("Could not compress state")?;

        let state_bytes = json.len();
        let compressed_state_bytes = compressed.len();
        let compression_ratio = compressed_state_bytes as f32 / state_bytes as f32 * 100.0;
        nice_plug_core::nice_trace!(
            "Compressed {state_bytes} bytes of state to {compressed_state_bytes} bytes \
             ({compression_ratio:.1}% compression ratio)"
        );

        Ok(compressed)
    }
    #[cfg(not(feature = "zstd"))]
    {
        Ok(json)
    }
}

/// Deserialize a plugin's state from a [`PluginState`] object. This is used to allow the plugin to
/// do its own internal preset management. Returns the plugin's refusal if it would not accept
/// the state, in which case nothing was written.
///
/// This uses a parameter getter function to avoid having to rebuild the parameter map, which may
/// include expensive user written code. See [`make_params_getter()`].
///
/// Make sure to reactivate plugin after deserializing the state so it can react to the new
/// parameter values. The smoothers have already been reset by this function.
///
/// The non-atomic path: `Plugin::filter_state()` runs here, on the committing thread, first, as
/// pinned (T1 round-4 fix; the maintainer delta had moved it to the restoring thread for every
/// plugin, a behaviour change for plugins that ignore the new contract). The atomic path runs it
/// once, off the audio thread, in `Wrapper::restore_state_object()` instead. With a `plugin`
/// instance (the all-or-nothing contract,
/// [`Plugin::ATOMIC_STATE_RESTORE`]) the instance's [`Plugin::preflight_state()`] sees the whole
/// state before any write and [`Plugin::state_restored()`] runs after the last one; the only
/// allocation permitted in between is the plugin's own [`Params::deserialize_fields()`]. Used by
/// the non-atomic restore path; the atomic path's commit is
/// [`commit_atomic_restore()`][commit_atomic_restore], which never reaches
/// `deserialize_fields()`.
pub(crate) unsafe fn deserialize_object<P: Plugin>(
    state: &mut PluginState,
    plugin: Option<&mut P>,
    plugin_params: Arc<dyn Params>,
    params_getter: impl Fn(&str) -> Option<ParamPtr>,
    current_buffer_config: Option<&BufferConfig>,
    restore: RestoreContext,
) -> core::result::Result<(), StateRefusal> {
    P::filter_state(state);

    // The whole state before anything is written: a refusal leaves everything untouched
    if let Some(plugin) = &plugin {
        plugin.preflight_state(state, restore)?;
    }

    let sample_rate = current_buffer_config.map(|c| c.sample_rate);
    unsafe { apply_params(state, &params_getter, sample_rate, Diagnostics::Log) };

    // The plugin can also persist arbitrary fields alongside its parameters. This is useful for
    // storing things like sample data. This is the plugin's own code; it keeps the allocation
    // permit the wrapper always gave it.
    crate::util::permit_alloc(|| plugin_params.deserialize_fields(&state.fields));

    // Everything is written: the plugin baselines what it must before the next process slice
    if let Some(plugin) = plugin {
        plugin.state_restored(restore);
    }

    Ok(())
}

/// The parameter-writing loop `deserialize_object()` and `commit_atomic_restore()` share: no
/// allocation outside the `debug_assertions` warning paths, which only run on a malformed state
/// and only with [`Diagnostics::Log`] (the atomic commit passes [`Diagnostics::Silent`]).
unsafe fn apply_params(
    state: &PluginState,
    params_getter: &impl Fn(&str) -> Option<ParamPtr>,
    sample_rate: Option<f32>,
    diagnostics: Diagnostics,
) {
    #[cfg(not(debug_assertions))]
    let _ = diagnostics;
    for (param_id_str, param_value) in &state.params {
        let param_ptr = match params_getter(param_id_str.as_str()) {
            Some(ptr) => ptr,
            None => {
                #[cfg(debug_assertions)]
                if diagnostics == Diagnostics::Log {
                    crate::util::permit_alloc(|| {
                        crate::nice_warn!("Unknown parameter: {}", param_id_str)
                    });
                }
                continue;
            }
        };

        unsafe {
            match (param_ptr, param_value) {
                (ParamPtr::FloatParam(p), ParamValue::F32(v)) => {
                    (*p)._internal_set_plain_value(*v);
                }
                (ParamPtr::IntParam(p), ParamValue::I32(v)) => {
                    (*p)._internal_set_plain_value(*v);
                }
                (ParamPtr::BoolParam(p), ParamValue::Bool(v)) => {
                    (*p)._internal_set_plain_value(*v);
                }
                // Enums are either serialized based on the active variant's index (which may not be the
                // same as the discriminator), or a custom set stable string ID. The latter allows the
                // variants to be reordered.
                (ParamPtr::EnumParam(p), ParamValue::I32(variant_idx)) => {
                    (*p)._internal_set_plain_value(*variant_idx);
                }
                (ParamPtr::EnumParam(p), ParamValue::String(id)) => {
                    let deserialized_enum = (*p).set_from_id(id);
                    #[cfg(debug_assertions)]
                    if !deserialized_enum && diagnostics == Diagnostics::Log {
                        crate::util::permit_alloc(|| {
                            crate::nice_warn!(
                                "Unknown ID {:?} for enum parameter \"{}\"",
                                id,
                                param_id_str,
                            )
                        });
                    }
                    #[cfg(not(debug_assertions))]
                    let _ = deserialized_enum;
                }
                (param_ptr, param_value) => {
                    #[cfg(debug_assertions)]
                    if diagnostics == Diagnostics::Log {
                        crate::util::permit_alloc(|| {
                            crate::nice_warn!(
                                "Invalid serialized value {:?} for parameter \"{}\" ({:?})",
                                param_value,
                                param_id_str,
                                param_ptr,
                            )
                        });
                    }
                    #[cfg(not(debug_assertions))]
                    let _ = (param_ptr, param_value);
                }
            }
        }

        // Make sure everything starts out in sync
        if let Some(sample_rate) = sample_rate {
            unsafe { param_ptr._internal_update_smoother(sample_rate, true) };
        }
    }
}

/// Commit an atomic restore (section 8, S-28; T1 maintainer fix). `state` has already been
/// through `Plugin::filter_state()` and `prepared` was already built by `Plugin::prepare_fields()`
/// off the audio thread, in `Wrapper::restore_state_object()`, so everything here is
/// allocation/lock/wait/I-O free and silent: the value validation, the preflight, every parameter
/// write (`apply_params()`, shared with the non-atomic path, here with
/// [`Diagnostics::Silent`]) and finally `prepared.commit()` -- the only field-related work left,
/// and the one place `Params::deserialize_fields()` is deliberately never called. Mirrors
/// `deserialize_object()`'s order otherwise (preflight, parameters, fields, `state_restored()`).
///
/// Adapted with attribution from the Codex T1 candidate's `prepare_state_fields()` /
/// `PreparedStateFields<P>` split (commit `087cb9c`), which first proved this split keeps an
/// accepted live restore's field commit allocation-free; Claude's first delta here instead
/// reached `deserialize_fields()` from the audio thread, a confirmed gap recorded in the
/// maintainer's check.
pub(crate) unsafe fn commit_atomic_restore<P: Plugin>(
    state: &PluginState,
    plugin: &mut P,
    params_getter: impl Fn(&str) -> Option<ParamPtr>,
    current_buffer_config: Option<&BufferConfig>,
    prepared: &dyn PreparedFields<P>,
    restore: RestoreContext,
) -> core::result::Result<(), StateRefusal> {
    validate_param_payloads(state, &params_getter)?;
    plugin.preflight_state(state, restore)?;

    let sample_rate = current_buffer_config.map(|c| c.sample_rate);
    unsafe { apply_params(state, &params_getter, sample_rate, Diagnostics::Silent) };

    prepared.commit(plugin);

    plugin.state_restored(restore);

    Ok(())
}

/// Every `state.params` entry whose id `params_getter` recognises must carry a legal value of
/// that parameter (section 8: "Cold restore validates count and parameter payloads before
/// activation"; "a refused restore does not apply a partial parameter/count state") -- the
/// all-or-nothing half of that sentence `apply_params()` alone cannot provide: on a value it
/// cannot apply it logs (debug builds only) and silently keeps that one parameter's prior value
/// instead of refusing, so the rest of an otherwise-valid restore would still commit around it.
/// This runs first, before `Plugin::preflight_state()` and before any write, so a malformed
/// parameter payload refuses the whole restore the same way a malformed `deck_count` already
/// does.
///
/// "Legal" is exactly what `apply_params()` can apply as a value the parameter can hold (T1
/// round-3 fix; round 2 checked only the kind, so an enum id the enum does not have still passed
/// and was skipped, the pool's counterexample): the kind must match, a string id must name one
/// of the enum's own variants, an enum index must lie in `0..variants`, an integer in its range,
/// and a float must be finite. A finite float outside its nominal range keeps pristine's
/// tolerance on purpose: a float parameter's own normalised arithmetic can land a stored value an
/// ulp past its bounds, so an exact bound check would refuse states the plugin saved itself.
///
/// An id `params_getter` does not recognise is left to `apply_params()`'s own, unchanged
/// tolerance (an unknown id is not a malformed one, matching pristine). No allocation, no lock:
/// `params_getter` is the wrapper's own pre-built lookup, not a fresh `Params::param_map()`, and
/// every check reads the parameter's own static description.
fn validate_param_payloads(
    state: &PluginState,
    params_getter: &impl Fn(&str) -> Option<ParamPtr>,
) -> core::result::Result<(), StateRefusal> {
    for (param_id_str, param_value) in &state.params {
        let Some(param_ptr) = params_getter(param_id_str.as_str()) else {
            continue;
        };
        // SAFETY: `params_getter` only hands out pointers into the plugin's own `Params`
        // object, which the wrapper keeps alive for as long as the wrapper itself.
        if let Some(detail) = unsafe { illegal_param_value(param_ptr, param_value) } {
            return Err(StateRefusal {
                detail,
                ..StateRefusal::MALFORMED_PARAM
            });
        }
    }
    Ok(())
}

/// Why `value` is no legal value of the parameter behind `param_ptr` (a
/// [`StateRefusal::MALFORMED_PARAM`] detail), or `None` if `apply_params()` can apply it as one.
///
/// # Safety
///
/// `param_ptr` must point at a live parameter.
unsafe fn illegal_param_value(param_ptr: ParamPtr, value: &ParamValue) -> Option<u64> {
    let out_of_range = |legal: bool| (!legal).then_some(StateRefusal::PARAM_OUT_OF_RANGE);
    unsafe {
        match (param_ptr, value) {
            (ParamPtr::FloatParam(_), ParamValue::F32(v)) => {
                (!v.is_finite()).then_some(StateRefusal::PARAM_NOT_FINITE)
            }
            (ParamPtr::IntParam(p), ParamValue::I32(v)) => {
                out_of_range(int_range_contains((*p).range(), *v))
            }
            (ParamPtr::BoolParam(_), ParamValue::Bool(_)) => None,
            (ParamPtr::EnumParam(p), ParamValue::I32(index)) => {
                out_of_range(usize::try_from(*index).is_ok_and(|index| index < (*p).len()))
            }
            (ParamPtr::EnumParam(p), ParamValue::String(id)) => (*p)
                .index_of_id(id)
                .is_none()
                .then_some(StateRefusal::PARAM_UNKNOWN_ENUM_ID),
            _ => Some(StateRefusal::PARAM_WRONG_KIND),
        }
    }
}

/// Whether `value` lies within an integer parameter's range, bounds included.
fn int_range_contains(range: IntRange, value: i32) -> bool {
    match range {
        IntRange::Linear { min, max } => (min..=max).contains(&value),
        IntRange::Reversed(range) => int_range_contains(*range, value),
    }
}

/// Deserialize a plugin's state from a vector containing (compressed) JSON data. Doesn't load the
/// plugin state since doing so should be accompanied by calls to `Plugin::init()` and
/// `Plugin::reset()`, and this way all of that behavior can be encapsulated so it can be reused in
/// multiple places. The returned state object can be passed to [`deserialize_object()`].
pub(crate) unsafe fn deserialize_json(state: &[u8]) -> Option<PluginState> {
    #[cfg(feature = "zstd")]
    let result: Option<PluginState> = match zstd::decode_all(state) {
        Ok(decompressed) => match serde_json::from_slice(decompressed.as_slice()) {
            Ok(s) => {
                let state_bytes = decompressed.len();
                let compressed_state_bytes = state.len();
                let compression_ratio = compressed_state_bytes as f32 / state_bytes as f32 * 100.0;
                crate::nice_trace!(
                    "Inflated {compressed_state_bytes} bytes of state to {state_bytes} bytes \
                     ({compression_ratio:.1}% compression ratio)"
                );

                Some(s)
            }
            Err(err) => {
                crate::nice_debug_assert_failure!("Error while deserializing state: {}", err);
                None
            }
        },
        // Uncompressed state files can still be loaded after enabling this feature to prevent
        // breaking existing plugin instances
        Err(zstd_err) => match serde_json::from_slice(state) {
            Ok(s) => {
                crate::nice_trace!("Older uncompressed state found");
                Some(s)
            }
            Err(json_err) => {
                crate::nice_debug_assert_failure!(
                    "Error while deserializing state as either compressed or uncompressed state: \
                     {}, {}",
                    zstd_err,
                    json_err
                );
                None
            }
        },
    };

    #[cfg(not(feature = "zstd"))]
    let result: Option<PluginState> = match serde_json::from_slice(state) {
        Ok(s) => Some(s),
        Err(err) => {
            crate::nice_debug_assert_failure!("Error while deserializing state: {}", err);
            None
        }
    };

    result
}

/// T1 round-3 (pool round-3 codex, astra): the hand-off between a restoring thread and the audio
/// thread. The pinned wrappers used a zero-capacity crossbeam channel, whose `try_recv()` takes
/// the channel's own `std::sync::Mutex` on every process call (and may allocate, hence the
/// pinned `permit_alloc`), and a return channel whose `try_send()` takes the waiting receiver's
/// waker mutex and unparks it. The mailbox's audio side is atomics only.
#[cfg(test)]
mod tests {
    use super::RestoreMailbox;
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};

    /// A restore stand-in: what travels, and the outcome the committing side writes into it.
    #[derive(Debug, PartialEq)]
    struct Item {
        payload: u32,
        committed_by_audio: bool,
    }

    fn item(payload: u32) -> Item {
        Item {
            payload,
            committed_by_audio: false,
        }
    }

    /// Nothing posted: the audio side's service is a no-op and never runs the commit.
    #[test]
    fn restore_mailbox_idle_service_never_commits() {
        let mailbox = RestoreMailbox::<Item>::new();
        for _ in 0..3 {
            mailbox.service(|_| panic!("nothing was posted"));
        }
    }

    /// A posted restore is committed by the audio side's next service, in place, and comes back
    /// to the restoring thread with what the commit wrote -- the object itself is never dropped
    /// on the audio side.
    #[test]
    fn restore_mailbox_hands_a_restore_to_the_audio_side_and_back() {
        let mailbox = Arc::new(RestoreMailbox::<Item>::new());
        let restorer = {
            let mailbox = mailbox.clone();
            thread::spawn(move || {
                let _one = mailbox
                    .restorer(Some(Duration::from_secs(1)))
                    .expect("no other restore in flight");
                mailbox.hand_to_audio(item(7), None)
            })
        };
        let started = Instant::now();
        let mut commits = 0;
        while !restorer.is_finished() {
            mailbox.service(|pending| {
                pending.committed_by_audio = true;
                commits += 1;
            });
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "never came back"
            );
            thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(
            restorer.join().unwrap(),
            Ok(Item {
                payload: 7,
                committed_by_audio: true
            })
        );
        assert_eq!(commits, 1, "committed exactly once");
        mailbox.service(|_| panic!("the mailbox is empty again"));
    }

    /// A restore the audio side never takes within the limit comes back untouched (the caller
    /// then commits it itself, as the pinned one-second host-load fallback does), and the
    /// mailbox is free for the next one.
    #[test]
    fn restore_mailbox_gives_back_a_restore_the_audio_side_never_took() {
        let mailbox = RestoreMailbox::<Item>::new();
        let _one = mailbox.restorer(Some(Duration::from_secs(1))).unwrap();
        let back = mailbox.hand_to_audio(item(3), Some(Duration::from_millis(20)));
        assert_eq!(back, Err(item(3)), "untaken: returned as it was");
        mailbox.service(|_| panic!("a restore given back is not committed later"));
    }

    /// One restorer at a time: while one restore is in flight a second restoring thread waits at
    /// most its limit and then gets nothing (the wrappers refuse with `StateRefusal::BUSY`)
    /// instead of overwriting the first one's slot.
    #[test]
    fn restore_mailbox_admits_one_restorer_at_a_time() {
        let mailbox = Arc::new(RestoreMailbox::<Item>::new());
        let first = mailbox.restorer(Some(Duration::from_secs(1))).unwrap();
        let second = {
            let mailbox = mailbox.clone();
            thread::spawn(move || {
                mailbox
                    .restorer(Some(Duration::from_millis(20)))
                    .map(|_guard| ())
                    .is_some()
            })
        };
        assert!(!second.join().unwrap(), "a second restorer is turned away");
        drop(first);
        assert!(mailbox.restorer(Some(Duration::from_millis(20))).is_some());
    }

    /// Without a limit -- a plugin that does not take the atomic contract -- a second restorer
    /// waits for the first and then gets in: never turned away, as the pinned wrappers never
    /// turned one away (T1 round-4).
    #[test]
    fn restore_mailbox_without_a_limit_waits_instead_of_refusing() {
        let mailbox = Arc::new(RestoreMailbox::<Item>::new());
        let first = mailbox.restorer(None).unwrap();
        let second = {
            let mailbox = mailbox.clone();
            thread::spawn(move || mailbox.restorer(None).map(|_guard| ()).is_some())
        };
        thread::sleep(Duration::from_millis(50));
        assert!(!second.is_finished(), "the second restorer waits");
        drop(first);
        assert!(second.join().unwrap(), "and gets in once the first is done");
    }
}
