//! Utilities for saving a [`crate::plugin::Plugin`]'s state. The actual state object is also exposed
//! to plugins through the [`GuiContext`][crate::prelude::GuiContext].

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// These state objects are also exposed directly to the plugin so it can do its own internal preset
// management

/// A plain, unnormalized value for a parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamValue {
    F32(f32),
    I32(i32),
    Bool(bool),
    /// Only used for enum parameters that have the `#[id = "..."]` attribute set.
    String(String),
}

/// A plugin's state so it can be restored at a later point. This object can be serialized and
/// deserialized using serde.
///
/// The fields are stored as `BTreeMap`s so the order in the serialized file is consistent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginState {
    /// The plugin version this state was saved with. Right now this is not used, but later versions
    /// of nice-plug may allow you to modify the plugin state object directly before it is loaded to
    /// allow migrating plugin states between breaking parameter changes.
    ///
    /// # Notes
    ///
    /// If the saved state is very old, then this field may be empty.
    #[serde(default)]
    pub version: String,

    /// The plugin's parameter values. These are stored unnormalized. This means the old values will
    /// be recalled when when the parameter's range gets increased. Doing so may still mess with
    /// parameter automation though, depending on how the host implements that.
    pub params: BTreeMap<String, ParamValue>,
    /// Arbitrary fields that should be persisted together with the plugin's parameters. Any field
    /// on the [`Params`][crate::params::Params] struct that's annotated with `#[persist =
    /// "stable_name"]` will be persisted this way.
    ///
    /// The individual fields are also serialized as JSON so they can safely be restored
    /// independently of the other fields.
    pub fields: BTreeMap<String, String>,
}

/// Why a plugin refused an incoming state in
/// [`Plugin::preflight_state()`][crate::plugin::Plugin::preflight_state()]: plain `Copy` data
/// that travels back to the restore's origin without allocating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateRefusal {
    /// A short static reason, shown to the person and logged off the audio thread.
    pub reason: &'static str,
    /// Plugin-defined detail, for example a bit mask of the decks that blocked the restore.
    pub detail: u64,
}

impl StateRefusal {
    /// A refusal with a reason and detail.
    pub const fn new(reason: &'static str, detail: u64) -> Self {
        Self { reason, detail }
    }

    /// The wrapper could not get exclusive access to the plugin for an atomic restore (the host
    /// activates or deactivates it concurrently); nothing was written.
    pub const BUSY: Self = Self::new("the plugin is busy; nothing was restored", 0);

    /// A known parameter's saved value is no legal value of that parameter (section 8: "Cold
    /// restore validates count and parameter payloads before activation"); nothing was written.
    /// `detail` says which check failed: [`Self::PARAM_WRONG_KIND`],
    /// [`Self::PARAM_UNKNOWN_ENUM_ID`], [`Self::PARAM_OUT_OF_RANGE`] or
    /// [`Self::PARAM_NOT_FINITE`]. An unrecognised parameter id is not this -- that is left to the
    /// wrapper's existing, pristine tolerance (skipped; warned about in debug builds, off the
    /// audio thread).
    pub const MALFORMED_PARAM: Self = Self::new(
        "a parameter's saved value is not a legal value of that parameter's own type",
        0,
    );

    /// [`Self::MALFORMED_PARAM`] detail: the value is not the parameter's own kind (a float
    /// parameter restored from a bool, say).
    pub const PARAM_WRONG_KIND: u64 = 1;
    /// [`Self::MALFORMED_PARAM`] detail: an enum parameter restored from a string id none of its
    /// variants has (or an enum without stable ids restored from a string at all).
    pub const PARAM_UNKNOWN_ENUM_ID: u64 = 2;
    /// [`Self::MALFORMED_PARAM`] detail: an integer outside its parameter's range, or an enum
    /// index outside `0..variants`.
    pub const PARAM_OUT_OF_RANGE: u64 = 3;
    /// [`Self::MALFORMED_PARAM`] detail: a float parameter restored from NaN or an infinity.
    pub const PARAM_NOT_FINITE: u64 = 4;
}

/// Where a state restore comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestoreOrigin {
    /// The host loaded state (`clap_plugin_state.load`, `IComponent::setState`).
    Host,
    /// The plugin's own GUI or preset management.
    Gui,
}

/// The thread committing a state restore.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestoreThread {
    /// The thread that called the host or GUI entry point.
    Caller,
    /// The audio thread, at the end of a process call.
    Audio,
}

/// Where a state restore comes from and on which thread it is being committed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestoreContext {
    /// The origin.
    pub origin: RestoreOrigin,
    /// The committing thread.
    pub thread: RestoreThread,
}

/// Persisted fields staged for an atomic restore (section 8, S-28), built off the audio thread by
/// [`crate::plugin::Plugin::prepare_fields()`] so the only work left for the committing thread is
/// [`commit()`][Self::commit()] -- the sole place a live restore writes persistent fields when
/// [`Plugin::ATOMIC_STATE_RESTORE`][crate::plugin::Plugin::ATOMIC_STATE_RESTORE] is set.
///
/// `&self`, not a consuming `self`: the wrapper keeps the staged value alive inside the
/// in-flight restore and hands the whole thing back to its origin afterwards, exactly like it
/// already does for the incoming [`PluginState`], so freeing it happens off the audio thread too.
///
/// Adapted with attribution from the Codex T1 candidate's `PreparedStateFields<P>` (commit
/// `087cb9c`), the prepare-off-audio / commit-on-audio split that candidate used to keep an
/// accepted live restore's field commit allocation-free.
///
/// `Sync` as well as `Send`: the staged value lives inside `PendingRestore`, which in turn sits
/// behind an `AtomicRefCell` the wrapper itself must be `Sync` to hold (it is captured by
/// `Arc<dyn Fn(...) + Send + Sync>` background-task closures elsewhere in the fork).
pub trait PreparedFields<P>: Send + Sync {
    /// Apply the staged fields to `plugin`. Runs on the committing thread -- the audio thread
    /// while processing, under the atomic contract. Must not allocate, deallocate, lock in a way
    /// that can wait, perform I/O or panic.
    fn commit(&self, plugin: &mut P);
}

/// Nothing staged, nothing to commit: what a plugin that does not override
/// [`Plugin::prepare_fields()`][crate::plugin::Plugin::prepare_fields()] already satisfies,
/// including every plugin with no persisted fields.
pub struct NoFields;

impl<P> PreparedFields<P> for NoFields {
    fn commit(&self, _plugin: &mut P) {}
}
