use atomic_refcell::AtomicRefCell;
use crossbeam::atomic::AtomicCell;
use nice_plug_core::audio_setup::{AudioIOLayout, BufferConfig, ProcessMode};
#[cfg(feature = "editor")]
use nice_plug_core::context::gui::GuiContext;
use nice_plug_core::context::process::{HostCall, Transport};
#[cfg(feature = "editor")]
use nice_plug_core::editor::dpi::Size;
#[cfg(feature = "editor")]
use nice_plug_core::editor::{Editor, SpawnedEditor};
use nice_plug_core::midi::{MidiConfig, PluginNoteEvent};
use nice_plug_core::params::internals::ParamPtr;
use nice_plug_core::params::{ParamFlags, Params};
#[cfg(feature = "editor")]
use nice_plug_core::plugin::TrackInfo;
use nice_plug_core::plugin::{
    NoFields, Plugin, PluginState, PreparedFields, ProcessStatus, RestoreContext, RestoreOrigin,
    RestoreThread, StateRefusal, TaskExecutor,
};
use parking_lot::Mutex;
#[cfg(feature = "editor")]
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use try_lock::TryLock;
#[cfg(feature = "editor")]
use vst3::ComWrapper;
use vst3::Steinberg::Vst::{IComponentHandler, IComponentHandlerTrait, IEventList, RestartFlags_};
use vst3::Steinberg::{kInvalidArgument, kResultOk, tresult};
use vst3::{ComPtr, ComRef};

use super::context::{WrapperActivateContext, WrapperProcessContext};
use super::note_expressions::NoteExpressionController;
use super::param_units::ParamUnits;
use super::util::{VST3_MIDI_PARAMS_END, VST3_MIDI_PARAMS_START};
use crate::event_loop::{EventLoop, MainThreadExecutor, OsEventLoop};
use crate::util::permit_alloc;
use crate::wrapper::state;
use crate::wrapper::util::buffer_management::BufferManager;
use crate::wrapper::util::hash_param_id;
use crate::wrapper::vst3::Vst3Plugin;
#[cfg(feature = "editor")]
use crate::wrapper::vst3::context::WrapperGuiContext;
#[cfg(feature = "editor")]
use crate::wrapper::vst3::view::WrapperView;

/// Identity for one handler connection, without ownership of the host object.
pub(super) struct HandlerIdentity(std::sync::atomic::AtomicU8);
impl HandlerIdentity {
    fn pending() -> Self {
        Self(std::sync::atomic::AtomicU8::new(0))
    }
    fn ready() -> Self {
        Self(std::sync::atomic::AtomicU8::new(1))
    }
    fn invalidate(&self) {
        self.0.store(2, Ordering::Release);
    }
    #[cfg(feature = "editor")]
    fn is_ready(&self) -> bool {
        self.0.load(Ordering::Acquire) == 1
    }
}
pub(super) struct HandlerConnection {
    pub handler: ComPtr<IComponentHandler>,
}
struct HandlerSlot {
    identity: Arc<HandlerIdentity>,
    handler: Option<Arc<HandlerConnection>>,
}
impl HandlerSlot {
    fn empty() -> Self {
        Self {
            identity: Arc::new(HandlerIdentity::ready()),
            handler: None,
        }
    }
}

/// The actual wrapper bits. We need this as an `Arc<T>` so we can safely use our event loop API.
/// Since we can't combine that with VST3's interior reference counting this just has to be moved to
/// its own struct.
pub(crate) struct WrapperInner<P: Vst3Plugin> {
    /// The wrapped plugin instance.
    pub plugin: TryLock<P>,
    /// The plugin's background task executor closure.
    pub task_executor: Mutex<TaskExecutor<P>>,
    /// The plugin's parameters. These are fetched once during initialization. That way the
    /// `ParamPtr`s are guaranteed to live at least as long as this object and we can interact with
    /// the `Params` object without having to acquire a lock on `plugin`.
    pub params: Arc<dyn Params>,
    /// The plugin's editor, if it has one. This object does not do anything on its own, but we need
    /// to instantiate this in advance so we don't need to lock the entire [`Plugin`] object when
    /// creating an editor. Wrapped in an `AtomicRefCell` because it needs to be initialized late.
    #[allow(clippy::type_complexity)]
    #[cfg(feature = "editor")]
    pub editor: AtomicRefCell<Option<Arc<Mutex<P::Editor>>>>,
    #[allow(clippy::type_complexity)]
    // My personal record for craziest Rust type!!!!
    #[cfg(feature = "editor")]
    pub editor_window:
        Arc<AtomicRefCell<Option<fragile::Fragile<SpawnedEditor<<P::Editor as Editor>::Handle>>>>>,

    /// The host's [`IComponentHandler`] instance, if passed through
    /// [`IEditController::set_component_handler`].
    component_handler: AtomicRefCell<HandlerSlot>,
    pub(super) dirty_initialized: AtomicBool,
    #[cfg(feature = "editor")]
    pub(super) dirty_in_flight: AtomicBool,
    #[cfg(feature = "editor")]
    pub(super) dirty_lease: AtomicRefCell<Option<Arc<crate::wrapper::dirty::EditorLease>>>,

    /// Our own [`IPlugView`] instance.
    #[cfg(feature = "editor")]
    pub plug_view: RwLock<Option<ComWrapper<WrapperView<P>>>>,

    /// Whether the editor is currently open. This is changed by the attached and removed functions on IPlugViewTrait
    #[cfg(feature = "editor")]
    pub is_editor_open: AtomicBool,

    /// A realtime-safe task queue so the plugin can schedule tasks that need to be run later on the
    /// GUI thread. This field should not be used directly for posting tasks. This should be done
    /// through [`Self::schedule_gui()`] instead. That method posts the task to the host's
    /// `IRunLoop` instead of it's available.
    ///
    /// This AtomicRefCell+Option is only needed because it has to be initialized late. There is no
    /// reason to mutably borrow the event loop, so reads will never be contested.
    ///
    /// TODO: Is there a better type for Send+Sync late initialization?
    pub event_loop: AtomicRefCell<Option<OsEventLoop<Task<P>, Self>>>,

    /// Whether the plugin is currently processing audio. In other words, the last state
    /// `IAudioProcessor::setActive()` has been called with.
    pub is_processing: AtomicBool,
    /// The current audio IO layout. Modified through `IAudioProcessor::setBusArrangements()` after
    /// matching the proposed bus arrangement to one of the supported ones. The plugin's first audio
    /// IO layout is chosen as the default. Because of the way VST3 works it's not possible to
    /// change the number of busses from that default, only the channel counts can change.
    pub current_audio_io_layout: AtomicCell<AudioIOLayout>,
    /// SDK activation state, independent of declarations and reset/processing state.
    pub active_input_buses: Vec<AtomicBool>,
    pub active_output_buses: Vec<AtomicBool>,
    /// The current buffer configuration, containing the sample rate and the maximum block size.
    /// Will be set in `IAudioProcessor::setupProcessing()`.
    pub current_buffer_config: AtomicCell<Option<BufferConfig>>,
    /// The current audio processing mode. Set in `IAudioProcessor::setup_processing()`.
    pub current_process_mode: AtomicCell<ProcessMode>,
    /// The most recently reported track information. Hosts may send partial updates (e.g.
    /// Ableton), so this is used to merge successive [`IInfoListener::setChannelContextInfos()`]
    /// calls.
    #[cfg(feature = "editor")]
    pub current_track_info: AtomicRefCell<TrackInfo>,

    /// The last process status returned by the plugin. This is used for tail handling.
    pub last_process_status: AtomicCell<ProcessStatus>,
    /// Serial of the host's process calls, rising by one per `IAudioProcessor::process()` call
    /// for the lifetime of this instance. Reported to the plugin through
    /// [`ProcessContext::host_call()`][nice_plug_core::context::process::ProcessContext::host_call()].
    pub host_call_serial: AtomicU64,
    /// Raised while a process call is in flight, between its slices included (T1 round-4 fix):
    /// a calling-thread atomic commit hands its restore to the call's end instead of landing
    /// between two slices. See [`state::CallInFlight`].
    pub in_call: AtomicBool,
    /// The current latency in samples, as set by the plugin through the [`ActivateContext`] and the
    /// [`ProcessContext`].
    pub current_latency: AtomicU32,
    /// A data structure that helps manage and create buffers for all of the plugin's inputs and
    /// outputs based on channel pointers provided by the host.
    pub buffer_manager: AtomicRefCell<BufferManager>,
    /// The incoming events for the plugin, if `P::ACCEPTS_MIDI` is set. If
    /// `P::SAMPLE_ACCURATE_AUTOMATION`, this is also read in lockstep with the parameter change
    /// block splitting.
    ///
    /// NOTE: Because with VST3 MIDI CC messages are sent as parameter changes and VST3 does not
    ///       interleave parameter changes and note events, this queue has to be sorted when
    ///       creating the process context
    pub input_note_events: AtomicRefCell<VecDeque<PluginNoteEvent<P>>>,
    /// VST3 has several useful predefined note expressions, but for some reason they are the only
    /// note event type that don't have MIDI note ID and channel fields. So we need to keep track of
    /// the most recent VST3 note IDs we've seen, and then map those back to MIDI note IDs and
    /// channels as needed.
    pub note_expression_controller: AtomicRefCell<NoteExpressionController>,
    /// Unprocessed parameter changes and note events sent by the host during a process call.
    /// Parameter changes are sent as separate queues for each parameter, and note events are in
    /// another queue on top of that. And if `P::MIDI_INPUT >= MidiConfig::MidiCCs`, then we can
    /// also receive MIDI CC messages through special parameter changes. On top of that, we also
    /// support sample accurate automation through block splitting if
    /// `P::SAMPLE_ACCURATE_AUTOMATION` is set. To account for all of this, we'll read all of the
    /// parameter changes and events into a vector at the start of the process call, sort it, and
    /// then do the block splitting based on that. Note events need to have their timing adjusted to
    /// match the block start, since they're all read upfront.
    pub process_events: AtomicRefCell<Vec<ProcessEvent<P>>>,
    /// The plugin is able to restore state through a method on the `GuiContext`, and the host loads
    /// state. To avoid changing parameters mid-processing and running into garbled data, while
    /// processing the restore happens at the end of a processing call: the restoring thread posts
    /// it here, the audio thread commits it in place, and the restoring thread takes it back with
    /// its outcome and frees it. The audio side is atomics only (T1 round-3 fix; the pinned
    /// zero-capacity channel locked a mutex on every process call): see
    /// [`state::RestoreMailbox`].
    pub restores: state::RestoreMailbox<state::PendingRestore<P>>,

    /// The keys from `param_map` in a stable order.
    pub param_hashes: Vec<u32>,
    /// A mapping from parameter ID hashes (obtained from the string parameter IDs) to pointers to
    /// parameters belonging to the plugin. These addresses will remain stable as long as the
    /// `params` object does not get deallocated.
    pub param_by_hash: HashMap<u32, ParamPtr>,
    /// Mappings from parameter hashes to string parameter IDs. Used for notifying the plugin's
    /// editor about parameter changes.
    pub param_id_by_hash: HashMap<u32, String>,
    pub param_units: ParamUnits,
    /// Mappings from string parameter identifiers to parameter hashes. Useful for debug logging
    /// and when storing and restoring plugin state.
    pub param_id_to_hash: HashMap<String, u32>,
    /// The inverse mapping from [`param_by_hash`][Self::param_by_hash]. This is needed to be able
    /// to have an ergonomic parameter setting API that uses references to the parameters instead of
    /// having to add a setter function to the parameter (or even worse, have it be completely
    /// untyped).
    pub param_ptr_to_hash: HashMap<ParamPtr, u32>,
}

/// Tasks that can be sent from the plugin to be executed on the main thread in a non-blocking
/// realtime-safe way (either a random thread or `IRunLoop` on Linux, the OS' message loop on
/// Windows and macOS).
#[allow(clippy::enum_variant_names)]
pub enum Task<P: Plugin> {
    /// Execute one of the plugin's background tasks.
    PluginTask(P::BackgroundTask),
    /// Inform the plugin that one or more parameter values have changed.
    #[cfg(feature = "editor")]
    StateChanged,
    /// Inform the plugin that one parameter's value has changed. This uses the parameter hashes
    /// since the task will be created from the audio thread.
    #[cfg(feature = "editor")]
    ParameterValueChanged(u32, f32),
    /// Trigger a restart with the given restart flags. This is a bit set of the flags from
    /// [`vst3::Steinberg::Vst::RestartFlags`].
    TriggerRestart(i32),
    // Request the editor to be resized according to its current size. Right now there is no way to
    // handle "denied resize" requests yet.
    #[cfg(feature = "editor")]
    RequestResize { size: Size, scale_factor: f64 },
    #[cfg(feature = "editor")]
    CallMainThread,
}

/// VST3 makes audio processing pretty complicated. In order to support both block splitting for
/// sample accurate automation and MIDI CC handling through parameters we need to put all parameter
/// changes and (translated) note events into a sorted array first.
#[derive(Debug, PartialEq)]
pub enum ProcessEvent<P: Plugin> {
    /// An incoming parameter change sent by the host. This will only be used when sample accurate
    /// automation has been enabled, and the parameters are only updated when we process this
    /// spooled event at the start of a block.
    ParameterChange {
        /// The event's sample offset within the buffer. Used for sorting.
        timing: u32,
        /// The parameter's hash, as used everywhere else.
        hash: u32,
        /// The normalized values, as provided by the host.
        normalized_value: f32,
    },
    /// An incoming parameter change sent by the host. This will only be used when sample accurate
    /// automation has been enabled, and the parameters are only updated when we process this
    /// spooled event at the start of a block.
    ///
    /// The timing stored within the note event needs to have the block start index subtraced from
    /// it. make sure to subtract the block start index with [`NoteEvent::subtract_timing()`] before
    /// putting this into the input event queue.
    NoteEvent(PluginNoteEvent<P>),
}

impl<P: Plugin> ProcessEvent<P> {
    #[inline]
    pub(crate) fn timing(&self) -> u32 {
        match self {
            ProcessEvent::ParameterChange { timing, .. } => *timing,
            ProcessEvent::NoteEvent(event) => event.timing(),
        }
    }
}

impl<P: Vst3Plugin> WrapperInner<P> {
    #[allow(unused_unsafe)]
    pub fn new() -> Arc<Self> {
        let mut plugin = P::default();
        let task_executor = Mutex::new(plugin.task_executor());

        // This is a mapping from the parameter IDs specified by the plugin to pointers to those
        // parameters. These pointers are assumed to be safe to dereference as long as
        // `wrapper.plugin` is alive. The plugin API identifiers these parameters by hashes, which
        // we'll calculate from the string ID specified by the plugin. These parameters should also
        // remain in the same order as the one returned by the plugin.
        let params = plugin.params();
        let param_id_hashes_ptrs_groups: Vec<_> = params
            .param_map()
            .into_iter()
            .map(|(id, ptr, group)| {
                let hash = hash_param_id(&id);
                (id, hash, ptr, group)
            })
            .collect();
        if cfg!(debug_assertions) {
            let param_map = params.param_map();
            let param_ids: HashSet<_> = param_id_hashes_ptrs_groups
                .iter()
                .map(|(id, _, _, _)| id.clone())
                .collect();
            crate::nice_debug_assert_eq!(
                param_map.len(),
                param_ids.len(),
                "The plugin has duplicate parameter IDs, weird things may happen. Consider using \
                 6 character parameter IDs to avoid collisions."
            );

            let mut bypass_param_exists = false;
            for (id, hash, ptr, _) in &param_id_hashes_ptrs_groups {
                let flags = unsafe { ptr.flags() };
                let is_bypass = flags.contains(ParamFlags::BYPASS);

                if is_bypass && bypass_param_exists {
                    crate::nice_debug_assert_failure!(
                        "Duplicate bypass parameters found, the host will only use the first one"
                    );
                }

                bypass_param_exists |= is_bypass;

                if P::MIDI_INPUT >= MidiConfig::MidiCCs
                    && (VST3_MIDI_PARAMS_START..VST3_MIDI_PARAMS_END).contains(hash)
                {
                    crate::nice_debug_assert_failure!(
                        "Parameter '{}' collides with an automatically generated MIDI CC \
                         parameter, consider giving it a different ID",
                        id
                    );
                }
            }
        }

        let param_hashes = param_id_hashes_ptrs_groups
            .iter()
            .map(|(_, hash, _, _)| *hash)
            .collect();
        let param_by_hash = param_id_hashes_ptrs_groups
            .iter()
            .map(|(_, hash, ptr, _)| (*hash, *ptr))
            .collect();
        let param_id_by_hash = param_id_hashes_ptrs_groups
            .iter()
            .map(|(id, hash, _, _)| (*hash, id.clone()))
            .collect();
        let param_units = ParamUnits::from_param_groups(
            param_id_hashes_ptrs_groups
                .iter()
                .map(|(_, hash, _, group_name)| (*hash, group_name.as_str())),
        )
        .expect("Inconsistent parameter groups");
        let param_id_to_hash = param_id_hashes_ptrs_groups
            .iter()
            .map(|(id, hash, _, _)| (id.clone(), *hash))
            .collect();
        let param_ptr_to_hash = param_id_hashes_ptrs_groups
            .into_iter()
            .map(|(_, hash, ptr, _)| (ptr, hash))
            .collect();

        let input_note_events = if P::MIDI_INPUT == MidiConfig::None {
            AtomicRefCell::new(VecDeque::new())
        } else {
            AtomicRefCell::new(VecDeque::with_capacity(P::INPUT_EVENT_CAPACITY))
        };

        let wrapper = Arc::new(Self {
            plugin: TryLock::new(plugin),
            task_executor,
            params,
            // Initialized later as it needs a reference to the wrapper for the async executor
            #[cfg(feature = "editor")]
            editor: AtomicRefCell::new(None),
            #[cfg(feature = "editor")]
            editor_window: Arc::new(AtomicRefCell::new(None)),

            component_handler: AtomicRefCell::new(HandlerSlot::empty()),
            dirty_initialized: AtomicBool::new(false),
            #[cfg(feature = "editor")]
            dirty_in_flight: AtomicBool::new(false),
            #[cfg(feature = "editor")]
            dirty_lease: AtomicRefCell::new(None),

            #[cfg(feature = "editor")]
            plug_view: RwLock::new(None),

            #[cfg(feature = "editor")]
            is_editor_open: AtomicBool::new(false),

            event_loop: AtomicRefCell::new(None),

            is_processing: AtomicBool::new(false),
            // Some hosts, like the current version of Bitwig and Ardour at the time of writing,
            // will try using the plugin's default not yet initialized bus arrangement. Because of
            // that, we'll always initialize this configuration even before the host requests a
            // channel layout.
            current_audio_io_layout: AtomicCell::new(
                P::AUDIO_IO_LAYOUTS.first().copied().unwrap_or_default(),
            ),
            active_input_buses: (0..P::AUDIO_IO_LAYOUTS
                .iter()
                .map(|layout| {
                    usize::from(layout.main_input_channels.is_some()) + layout.aux_input_ports.len()
                })
                .max()
                .unwrap_or(0))
                .map(|_| AtomicBool::new(true))
                .collect(),
            active_output_buses: (0..P::AUDIO_IO_LAYOUTS
                .iter()
                .map(|layout| {
                    usize::from(layout.main_output_channels.is_some())
                        + layout.aux_output_ports.len()
                })
                .max()
                .unwrap_or(0))
                .map(|_| AtomicBool::new(true))
                .collect(),
            current_buffer_config: AtomicCell::new(None),
            current_process_mode: AtomicCell::new(ProcessMode::Realtime),
            #[cfg(feature = "editor")]
            current_track_info: AtomicRefCell::new(TrackInfo::default()),
            last_process_status: AtomicCell::new(ProcessStatus::Normal),
            host_call_serial: AtomicU64::new(0),
            in_call: AtomicBool::new(false),
            current_latency: AtomicU32::new(0),
            // This is initialized just before calling `Plugin::activate()` so that during the
            // process call buffers can be initialized without any allocations
            buffer_manager: AtomicRefCell::new(BufferManager::for_audio_io_layout(
                0,
                AudioIOLayout::default(),
            )),
            input_note_events,
            note_expression_controller: AtomicRefCell::new(NoteExpressionController::default()),
            process_events: AtomicRefCell::new(Vec::with_capacity(P::INPUT_EVENT_CAPACITY)),
            restores: state::RestoreMailbox::new(),

            param_hashes,
            param_by_hash,
            param_id_by_hash,
            param_units,
            param_id_to_hash,
            param_ptr_to_hash,
        });

        // FIXME: Right now this is safe, but if we are going to have a singleton main thread queue
        //        serving multiple plugin instances, Arc can't be used because its reference count
        //        is separate from the internal COM-style reference count.
        *wrapper.event_loop.borrow_mut() =
            Some(OsEventLoop::new_and_spawn(Arc::downgrade(&wrapper)));

        // The editor also needs to be initialized later so the Async executor can work.

        #[cfg(feature = "editor")]
        {
            *wrapper.editor.borrow_mut() = wrapper
                .plugin
                .try_lock()
                .unwrap()
                .editor(nice_plug_core::context::gui::AsyncExecutor::new(
                    Arc::new({
                        let wrapper = Arc::downgrade(&wrapper);
                        move |task| {
                            let wrapper = match wrapper.upgrade() {
                                Some(wrapper) => wrapper,
                                None => return,
                            };

                            let task_posted = wrapper.schedule_background(Task::PluginTask(task));
                            crate::nice_debug_assert!(
                                task_posted,
                                "The task queue is full, dropping task..."
                            );
                        }
                    }),
                    Arc::new({
                        let wrapper = Arc::downgrade(&wrapper);
                        move |task| {
                            let wrapper = match wrapper.upgrade() {
                                Some(wrapper) => wrapper,
                                None => return,
                            };

                            let task_posted = wrapper.schedule_gui(Task::PluginTask(task));
                            crate::nice_debug_assert!(
                                task_posted,
                                "The task queue is full, dropping task..."
                            );
                        }
                    }),
                ))
                .map(|editor| Arc::new(Mutex::new(editor)));
        }

        wrapper
    }

    pub(super) fn snapshot_handler(&self) -> Option<Arc<HandlerConnection>> {
        self.component_handler.borrow().handler.clone()
    }

    #[cfg(feature = "editor")]
    pub(super) fn connection_identity(&self) -> Arc<HandlerIdentity> {
        self.component_handler.borrow().identity.clone()
    }

    fn identity_current(&self, identity: &Arc<HandlerIdentity>) -> bool {
        self.dirty_initialized.load(Ordering::Acquire)
            && identity.0.load(Ordering::Acquire) != 2
            && Arc::ptr_eq(&self.component_handler.borrow().identity, identity)
    }

    #[cfg(feature = "editor")]
    pub(super) fn connection_ready(&self, identity: &Arc<HandlerIdentity>) -> bool {
        identity.is_ready() && self.identity_current(identity)
    }

    #[cfg(feature = "editor")]
    pub(super) fn close_dirty_editor(&self) {
        let lease = self.dirty_lease.borrow_mut().take();
        if let Some(lease) = lease {
            lease.close();
        }
    }

    pub(super) fn terminate_dirty(&self) {
        self.dirty_initialized.store(false, Ordering::Release);
        #[cfg(feature = "editor")]
        self.close_dirty_editor();
        let old = {
            let mut slot = self.component_handler.borrow_mut();
            slot.identity.invalidate();
            let replacement = HandlerSlot::empty();
            replacement.identity.invalidate();
            std::mem::replace(&mut *slot, replacement)
        };
        // Host Release may reenter; no handler or editor cell is borrowed.
        drop(old);
    }

    pub(super) fn initialize_dirty(&self) {
        self.terminate_dirty();
        let old = std::mem::replace(
            &mut *self.component_handler.borrow_mut(),
            HandlerSlot::empty(),
        );
        self.dirty_initialized.store(true, Ordering::Release);
        drop(old);
    }

    pub(super) unsafe fn replace_component_handler(&self, handler: *mut IComponentHandler) {
        if !self.dirty_initialized.load(Ordering::Acquire) {
            return;
        }
        let identity = Arc::new(HandlerIdentity::pending());
        let old = {
            let mut slot = self.component_handler.borrow_mut();
            slot.identity.invalidate();
            std::mem::replace(
                &mut *slot,
                HandlerSlot {
                    identity: identity.clone(),
                    handler: None,
                },
            )
        };
        drop(old);
        if !self.identity_current(&identity) {
            return;
        }
        let incoming = unsafe { vst3::ComRef::from_raw(handler) }.map(|handler| {
            Arc::new(HandlerConnection {
                handler: handler.to_com_ptr(),
            })
        });
        // AddRef may have replaced/disconnected/terminated this instance.
        if !self.identity_current(&identity) {
            drop(incoming);
            return;
        }
        let displaced = {
            let mut slot = self.component_handler.borrow_mut();
            let old = std::mem::replace(&mut slot.handler, incoming);
            identity.0.store(1, Ordering::Release);
            old
        };
        drop(displaced);
    }

    #[cfg(feature = "editor")]
    pub fn make_gui_context(
        self: Arc<Self>,
        dirty_lease: Arc<crate::wrapper::dirty::EditorLease>,
        dirty_connection: Arc<HandlerIdentity>,
    ) -> GuiContext {
        GuiContext::new(Arc::new(WrapperGuiContext {
            inner: Arc::downgrade(&self),
            dirty_lease,
            dirty_connection,
            #[cfg(debug_assertions)]
            param_gesture_checker: Default::default(),
        }))
    }

    /// # Note
    ///
    /// The lock on the plugin must be dropped before this object is dropped to avoid deadlocks
    /// caused by reentrant function calls.
    pub fn make_activate_context(&self) -> WrapperActivateContext<'_, P> {
        WrapperActivateContext {
            inner: self,
            pending_requests: Default::default(),
        }
    }

    pub fn make_process_context<'a>(
        &'a self,
        transport: Transport,
        host_call: HostCall,
        host_out_events: Option<ComRef<'a, IEventList>>,
        total_buffer_len: usize,
        current_sample_idx: usize,
    ) -> WrapperProcessContext<'a, P> {
        WrapperProcessContext {
            inner: self,
            input_events_guard: self.input_note_events.borrow_mut(),
            transport,
            host_call,
            host_out_events,
            total_buffer_len: total_buffer_len as u32,
            current_sample_idx: current_sample_idx as u32,
        }
    }

    /// Posts the task to the background task queue using [`EventLoop::schedule_background()`] so it
    /// can be run in the background without blocking either the GUI or the audio thread.
    ///
    /// If the task queue is full, then this will return false.
    #[must_use]
    pub fn schedule_background(&self, task: Task<P>) -> bool {
        let event_loop = self.event_loop.borrow();
        let event_loop = event_loop.as_ref().unwrap();
        event_loop.schedule_background(task)
    }

    /// Either posts the task to the task queue using [`EventLoop::schedule_gui()`] so it can be
    /// delegated to the main thread, executes the task directly if this is the main thread, or runs
    /// the task on the host's `IRunLoop` if the GUI is open and it exposes one.
    ///
    /// If the task queue is full, then this will return false.
    #[must_use]
    pub fn schedule_gui(&self, task: Task<P>) -> bool {
        let event_loop = self.event_loop.borrow();
        let event_loop = event_loop.as_ref().unwrap();
        if event_loop.is_main_thread() {
            self.execute(task, true);
            true
        } else {
            // If the editor is open, and the host exposes the `IRunLoop` interface, then we'll run
            // the task on the host's GUI thread using that interface. Otherwise we'll use the
            // regular event loop. If the editor gets dropped while there's still outstanding work
            // left in the run loop task queue, then those tasks will be posted to the regular event
            // loop so no work is lost.
            #[cfg(feature = "editor")]
            if self.is_editor_open.load(Ordering::Relaxed) {
                match self
                    .plug_view
                    .read()
                    .clone()
                    .unwrap()
                    .do_maybe_in_run_loop(task)
                {
                    Ok(()) => true,
                    Err(task) => event_loop.schedule_gui(task),
                }
            } else {
                event_loop.schedule_gui(task)
            }

            #[cfg(not(feature = "editor"))]
            event_loop.schedule_gui(task)
        }
    }

    /// Get a parameter's ID based on a `ParamPtr`. Used in the `GuiContext` implementation for the
    /// gesture checks.
    #[allow(unused)]
    pub fn param_id_from_ptr(&self, param: ParamPtr) -> Option<&str> {
        self.param_ptr_to_hash
            .get(&param)
            .and_then(|hash| self.param_id_by_hash.get(hash))
            .map(|s| s.as_str())
    }

    /// Convenience function for setting a value for a parameter as triggered by a VST3 parameter
    /// update. The same rate is for updating parameter smoothing.
    ///
    /// After calling this function, you should call
    /// [`notify_param_values_changed()`][Self::notify_param_values_changed()] to allow the editor
    /// to update itself. This needs to be done separately so you can process parameter changes in
    /// batches.
    pub fn set_normalized_value_by_hash(
        &self,
        hash: u32,
        normalized_value: f32,
        sample_rate: Option<f32>,
    ) -> tresult {
        if !normalized_value.is_finite() {
            return kInvalidArgument;
        }

        match self.param_by_hash.get(&hash) {
            Some(param_ptr) => {
                apply_parameter_value(param_ptr, normalized_value, sample_rate, || {
                    #[cfg(feature = "editor")]
                    {
                        let task_posted =
                            self.schedule_gui(Task::ParameterValueChanged(hash, normalized_value));
                        crate::nice_debug_assert!(
                            task_posted,
                            "The task queue is full, dropping task..."
                        );
                    }
                });

                kResultOk
            }
            _ => kInvalidArgument,
        }
    }

    /// Get the plugin's state object, may be called by the plugin's GUI as part of its own preset
    /// management. The wrapper doesn't use these functions and serializes and deserializes directly
    /// the JSON in the relevant plugin API methods instead.
    #[cfg(feature = "editor")]
    pub fn get_state_object(&self) -> PluginState {
        unsafe {
            state::serialize_object::<P>(
                self.params.clone(),
                state::make_params_iter(&self.param_by_hash, &self.param_id_to_hash),
            )
        }
    }

    /// The pristine signature (T1 maintainer fix): a caller that never heard of
    /// [`StateRefusal`] still compiles against this fork unchanged. Discards the refusal after
    /// logging it.
    pub fn set_state_object_from_gui(&self, state: PluginState) {
        if let Err(refusal) = self.try_set_state_object_from_gui(state) {
            crate::nice_error!(
                "Refused to restore state: {} (detail {})",
                refusal.reason,
                refusal.detail
            );
        }
    }

    /// Update the plugin's internal state, called by the plugin itself from the GUI thread. To
    /// prevent corrupting data and changing parameters during processing the actual state is only
    /// updated at the end of the audio processing cycle. Returns the plugin's refusal, if any
    /// (see [`Plugin::ATOMIC_STATE_RESTORE`]); nothing is written then.
    pub fn try_set_state_object_from_gui(&self, state: PluginState) -> Result<(), StateRefusal> {
        self.restore_state_object(state, RestoreOrigin::Gui)
    }

    /// Restore a state object from the host or the GUI. While the plugin is processing, a GUI
    /// restore (and, under [`Plugin::ATOMIC_STATE_RESTORE`], a host load as well) is handed to
    /// the audio thread, which commits it at the end of its process call and hands the object
    /// back with the outcome; otherwise the calling thread commits it right here.
    ///
    /// `Plugin::filter_state()` and, under the atomic contract, `Plugin::prepare_fields()` both
    /// run right here, off the audio thread, before any threading decision (T1 maintainer fix):
    /// whichever thread ends up committing only has to call `commit_restore()`.
    ///
    /// One restore at a time (T1 round-3 fix): a second restoring thread waits up to a second for
    /// the first to finish and is refused with [`StateRefusal::BUSY`] otherwise, nothing written.
    /// Everything the audio thread used to do besides the commit itself -- the hand-back, posting
    /// `Task::StateChanged`, the unknown-parameter warning -- happens here, on this thread.
    pub fn restore_state_object(
        &self,
        mut state: PluginState,
        origin: RestoreOrigin,
    ) -> Result<(), StateRefusal> {
        let (outcome, committed_on_audio) = {
            // Under the atomic contract one restore at a time (T1 round-3 fix): a second restoring
            // thread waits up to a second for the first, then is refused with nothing written.
            // A plugin without the contract keeps the pinned behaviour (T1 round-4 fix): its
            // restorers only queue for the mailbox, below, and never get refused.
            let _atomic_restorer = if P::ATOMIC_STATE_RESTORE {
                match self
                    .restores
                    .restorer(Some(std::time::Duration::from_secs(1)))
                {
                    Some(guard) => Some(guard),
                    None => return Err(StateRefusal::BUSY),
                }
            } else {
                None
            };

            let prepared: Box<dyn PreparedFields<P>> = if P::ATOMIC_STATE_RESTORE {
                // Off the audio thread, before any threading decision (T1 maintainer fix)
                P::filter_state(&mut state);
                state::report_unknown_params(
                    &state,
                    state::make_params_getter(&self.param_by_hash, &self.param_id_to_hash),
                );
                P::prepare_fields(&state.fields)
            } else {
                // As pinned: `filter_state()` runs on the committing thread, in
                // `state::deserialize_object()` (T1 round-4 fix)
                Box::new(NoFields)
            };
            let mut pending = state::PendingRestore {
                state,
                origin,
                outcome: Ok(()),
                prepared,
            };
            let mut via_audio = origin == RestoreOrigin::Gui || P::ATOMIC_STATE_RESTORE;

            // Use a loop and timeouts to handle the super rare edge case when this function gets
            // called between a process call and the host disabling the plugin
            loop {
                if via_audio && self.is_processing.load(Ordering::SeqCst) {
                    // If the plugin is currently processing audio, then we'll perform the restore
                    // operation at the end of the audio call: the audio thread commits it in place
                    // and this thread takes it back, with its outcome, and frees it. Without the
                    // atomic contract a second restorer waits here for the mailbox for as long as
                    // the first needs it, as the pinned shared channel made it wait.
                    let _restorer = if P::ATOMIC_STATE_RESTORE {
                        None
                    } else {
                        self.restores.restorer(None)
                    };
                    match self
                        .restores
                        .hand_to_audio(pending, Some(std::time::Duration::from_secs(1)))
                    {
                        Ok(returned) => break (returned.outcome, true),
                        Err(untaken) => {
                            pending = untaken;
                            // A host load must not wait on a processing loop that is not running
                            // (some hosts keep a bypassed plugin "processing" without calling it):
                            // commit it on this thread after all. Under the atomic contract a GUI
                            // restore too (T1 round-5 fix): only a call with audio commits on the
                            // audio thread, so while only buffer-less calls run nothing else
                            // would; and this thread's commit is safe while processing, since it
                            // takes the plugin's lock and never lands inside a call.
                            if origin == RestoreOrigin::Host || P::ATOMIC_STATE_RESTORE {
                                via_audio = false;
                            }
                            continue;
                        }
                    }
                } else {
                    // Otherwise we'll set the state right here and now, on the calling thread --
                    // unless a process call is between two of its slices (T1 round-4 fix): then
                    // the call's own end commits it, through the mailbox, like any audio-thread
                    // restore.
                    let restore = RestoreContext {
                        origin,
                        thread: RestoreThread::Caller,
                    };
                    match self.commit_restore(&mut pending.state, &*pending.prepared, restore) {
                        Err(refusal) if refusal == state::CALL_IN_FLIGHT => {
                            via_audio = true;
                            std::thread::sleep(std::time::Duration::from_millis(1));
                            continue;
                        }
                        outcome => break (outcome, false),
                    }
                }
            }
        };

        // Under the atomic contract the audio thread posts nothing itself (T1 round-3 fix); without
        // it the audio thread already posted this, as pinned
        #[cfg(feature = "editor")]
        if committed_on_audio && outcome.is_ok() && P::ATOMIC_STATE_RESTORE {
            let task_posted = self.schedule_gui(Task::StateChanged);
            crate::nice_debug_assert!(task_posted, "The task queue is full, dropping task...");
        }
        #[cfg(not(feature = "editor"))]
        let _ = committed_on_audio;

        if origin == RestoreOrigin::Gui && outcome.is_ok() {
            // After the state has been updated, notify the host about the new parameter values
            let task_posted = self
                .event_loop
                .borrow()
                .as_ref()
                .unwrap()
                .schedule_gui(Task::TriggerRestart(RestartFlags_::kParamValuesChanged));
            crate::nice_debug_assert!(task_posted, "The task queue is full, dropping task...");
        }

        outcome
    }

    pub fn set_latency_samples(&self, samples: u32) {
        // Only trigger a restart if it's actually needed
        let old_latency = self.current_latency.swap(samples, Ordering::SeqCst);
        if old_latency != samples {
            self.request_restart();
        }
    }

    pub fn request_restart(&self) {
        let task_posted = self.schedule_gui(Task::TriggerRestart(RestartFlags_::kLatencyChanged));
        crate::nice_debug_assert!(task_posted, "The task queue is full, dropping task...");
    }

    /// The pristine signature (T1 maintainer fix): a caller using only what the framework
    /// exposed before this delta still compiles and still gets a correct restore. Stages fields
    /// itself, on whichever thread calls this, so call it off the audio thread; the normal path
    /// (`restore_state_object()`) stages them before any threading decision instead. Kept for
    /// source compatibility; `WrapperInner` is `pub(crate)` (pristine too), so nothing outside
    /// this crate can reach it, and nothing inside calls it either (same as pristine, where it
    /// had no caller besides the drain loop, which now calls `commit_restore()` instead).
    #[allow(dead_code)]
    pub fn set_state_inner(&self, state: &mut PluginState) -> bool {
        let prepared: Box<dyn PreparedFields<P>> = if P::ATOMIC_STATE_RESTORE {
            P::filter_state(state);
            P::prepare_fields(&state.fields)
        } else {
            // `filter_state()` runs in `state::deserialize_object()`, as pinned
            Box::new(NoFields)
        };
        let restore = RestoreContext {
            origin: RestoreOrigin::Gui,
            thread: RestoreThread::Caller,
        };
        self.commit_restore(state, &*prepared, restore).is_ok()
    }

    /// The atomic commit with the plugin already held by the caller (T1 round-4 fix; pool
    /// round-4 terra): a process call commits a restore handed to it inside its last slice, under
    /// that slice's own plugin guard, so the audio thread takes no lock of its own for it. Posts
    /// nothing; the restoring thread does (T1 round-3 fix).
    pub fn commit_restore_holding(
        &self,
        plugin: &mut P,
        state: &mut PluginState,
        prepared: &dyn PreparedFields<P>,
        restore: RestoreContext,
    ) -> Result<(), StateRefusal> {
        let params_getter = state::make_params_getter(&self.param_by_hash, &self.param_id_to_hash);
        let buffer_config = self.current_buffer_config.load();
        unsafe {
            state::commit_atomic_restore::<P>(
                state,
                plugin,
                params_getter,
                buffer_config.as_ref(),
                prepared,
                restore,
            )
        }
    }

    /// Immediately set the plugin state. Returns the plugin's refusal if it would not accept the
    /// state (see [`Plugin::ATOMIC_STATE_RESTORE`]), in which case nothing was written. The
    /// plugin state is set from a couple places, so this function aims to deduplicate that.
    /// Without the atomic contract this includes a `permit_alloc()` around the deserialization
    /// for the use case where `set_state_object_from_gui()` was called while the plugin is
    /// processing audio. `prepared` was already built off the audio thread by
    /// `Plugin::prepare_fields()` (T1 maintainer fix, section 8, S-28); under the atomic contract
    /// it is the only thing that writes persistent fields, so `Params::deserialize_fields()` is
    /// never reached from here.
    ///
    /// Implicitly emits `Task::StateChanged` when committed on the calling thread; after an
    /// audio-thread commit the restoring thread posts it instead (T1 round-3 fix).
    ///
    /// # Notes
    ///
    /// `self.plugin` must _not_ be locked while calling this function or it will deadlock.
    pub fn commit_restore(
        &self,
        state: &mut PluginState,
        prepared: &dyn PreparedFields<P>,
        restore: RestoreContext,
    ) -> Result<(), StateRefusal> {
        let params_getter = state::make_params_getter(&self.param_by_hash, &self.param_id_to_hash);
        let buffer_config = self.current_buffer_config.load();
        if P::ATOMIC_STATE_RESTORE {
            // The plugin's hooks see the live instance, so this needs the plugin lock: free
            // between process calls and while not processing, taken only if the host activates
            // or deactivates concurrently, in which case nothing is written. Never waited for.
            let Some(mut plugin) = self.plugin.try_lock() else {
                return Err(StateRefusal::BUSY);
            };
            // Read under the plugin's lock (T1 round-4 fix): a call that is between two slices
            // raised the flag before its first slice took and released this lock.
            if restore.thread == RestoreThread::Caller && self.in_call.load(Ordering::SeqCst) {
                return Err(state::CALL_IN_FLIGHT);
            }
            self.commit_restore_holding(&mut plugin, state, prepared, restore)?;
        } else {
            // FIXME: This is obviously not realtime-safe, but loading presets without doing this
            //        could lead to inconsistencies. `state::deserialize_object()` normally never
            //        allocates, but if the plugin has persistent non-parameter data then its
            //        `deserialize_fields()` implementation may still allocate.
            permit_alloc(|| unsafe {
                state::deserialize_object::<P>(
                    state,
                    None,
                    self.params.clone(),
                    params_getter,
                    buffer_config.as_ref(),
                    restore,
                )
            })?;
        }

        // Reinitialize the plugin after loading state so it can respond to the new parameter
        // values. Under the atomic contract from the calling thread only: after an audio-thread
        // commit, `restore_state_object()` posts this on the restoring thread (T1 round-3 fix), so
        // nothing here touches the event loop or the host's run loop from the audio thread.
        // Without the contract wherever the commit runs, as pinned (T1 round-4 fix).
        #[cfg(feature = "editor")]
        if restore.thread == RestoreThread::Caller || !P::ATOMIC_STATE_RESTORE {
            let task_posted = self.schedule_gui(Task::StateChanged);
            crate::nice_debug_assert!(task_posted, "The task queue is full, dropping task...");
        }

        Ok(())
    }
}

impl<P: Vst3Plugin> MainThreadExecutor<Task<P>> for WrapperInner<P> {
    fn execute(&self, task: Task<P>, is_gui_thread: bool) {
        // This function is always called from the main thread
        match task {
            Task::PluginTask(task) => (self.task_executor.lock())(task),
            #[cfg(feature = "editor")]
            Task::StateChanged => {
                use nice_plug_core::editor::EditorHandle;

                if let Some(editor_window) = self.editor_window.borrow().as_ref() {
                    editor_window.get().handle.state_changed();
                }
            }
            #[cfg(feature = "editor")]
            Task::ParameterValueChanged(param_hash, normalized_value) => {
                use nice_plug_core::editor::EditorHandle;

                if let Some(editor_window) = self.editor_window.borrow().as_ref() {
                    let param_id = &self.param_id_by_hash[&param_hash];
                    editor_window
                        .get()
                        .handle
                        .param_value_changed(param_id, normalized_value);
                }
            }
            Task::TriggerRestart(flags) => match self.snapshot_handler().as_ref() {
                Some(handler) => unsafe {
                    crate::nice_debug_assert!(is_gui_thread);
                    let result = handler.handler.restartComponent(flags);
                    crate::nice_debug_assert_eq!(
                        result,
                        kResultOk,
                        "Failed the restart request call with flags '{:?}'",
                        flags
                    );
                },
                None => crate::nice_debug_assert_failure!("Component handler not yet set"),
            },
            #[cfg(feature = "editor")]
            Task::RequestResize { size, scale_factor } => {
                if self.is_editor_open.load(Ordering::SeqCst) {
                    unsafe {
                        crate::nice_debug_assert!(is_gui_thread);
                        let _success = WrapperView::request_resize(
                            &self.plug_view.read().clone().unwrap(),
                            size,
                            scale_factor,
                        );
                    }
                } else {
                    crate::nice_debug_assert_failure!("Can't resize a closed editor");
                }
            }
            #[cfg(feature = "editor")]
            Task::CallMainThread => {
                use nice_plug_core::editor::EditorHandle;

                if let Some(editor_window) = self.editor_window.borrow().as_ref() {
                    let editor_window = editor_window.get();
                    editor_window
                        .handle
                        .host_main_thread_callback(&editor_window.window)
                }
            }
        }
    }
}

// The real value-delivery block, shared with its headless test. Callers retain
// ownership of the parameter while this private function uses its pointer.
fn apply_parameter_value(
    param: &ParamPtr,
    normalized: f32,
    sample_rate: Option<f32>,
    notify: impl FnOnce(),
) {
    // Independent relaxed reads, not a claim of a coherent multi-atomic snapshot.
    let base_plain = unsafe { param.unmodulated_plain_value() };
    let base_normalized = unsafe { param.unmodulated_normalized_value() };
    let output_changed = unsafe { param._internal_set_normalized_value(normalized) };
    let base_changed = base_plain != unsafe { param.unmodulated_plain_value() }
        || base_normalized != unsafe { param.unmodulated_normalized_value() };
    if output_changed && let Some(rate) = sample_rate {
        unsafe { param._internal_update_smoother(rate, false) };
    }
    if output_changed || base_changed {
        notify();
    }
}

#[cfg(test)]
mod p3_delivery_tests {
    use super::apply_parameter_value;
    use nice_plug_core::params::{
        FloatParam, InternalParamMut, Param,
        range::{DecibelZeroRange, FloatRange},
        smoothing::SmoothingStyle,
    };
    use std::cell::Cell;

    #[test]
    fn base_edit_notifies_without_smoother_restart() {
        for range in [
            FloatRange::Linear { min: 0.0, max: 1.0 },
            FloatRange::DecibelsWithZero(DecibelZeroRange::new(-60.0, 6.0).unwrap()),
        ] {
            for offset in [-1.0, 1.0] {
                let p = FloatParam::new("level", range.unnormalize(0.5), range)
                    .with_smoother(SmoothingStyle::Linear(10.0));
                unsafe {
                    p._internal_update_smoother(48000.0, true);
                }
                assert!(unsafe { p._internal_modulate_value(offset) });
                unsafe {
                    p._internal_update_smoother(48000.0, false);
                }
                for _ in 0..73 {
                    p.smoothed.next();
                }
                let remaining = p.smoothed.steps_left();
                let before = p.smoothed.previous_value();
                let notifications = Cell::new(0);
                apply_parameter_value(&p.as_ptr(), 0.25, Some(48000.0), || {
                    notifications.set(notifications.get() + 1)
                });
                assert_eq!(
                    notifications.get(),
                    1,
                    "base-only edit must notify the editor"
                );
                assert_eq!(p.unmodulated_plain_value(), range.unnormalize(0.25));
                assert_eq!(
                    p.smoothed.steps_left(),
                    remaining,
                    "base-only edit must not restart ramp"
                );
                assert_eq!(p.smoothed.previous_value(), before);
                apply_parameter_value(&p.as_ptr(), 0.25, Some(48000.0), || {
                    notifications.set(notifications.get() + 1)
                });
                assert_eq!(notifications.get(), 1, "duplicate must not notify");
                assert_eq!(p.smoothed.steps_left(), remaining);
                for _ in 0..remaining {
                    p.smoothed.next();
                }
                assert!(!p.smoothed.is_smoothing());
                assert_eq!(p.smoothed.previous_value(), p.value());
                unsafe {
                    p._internal_modulate_value(0.0);
                }
                apply_parameter_value(&p.as_ptr(), 0.7, Some(48000.0), || {
                    notifications.set(notifications.get() + 1)
                });
                assert_eq!(notifications.get(), 2);
                assert_eq!(p.smoothed.steps_left(), 480);
            }
        }
    }
}
