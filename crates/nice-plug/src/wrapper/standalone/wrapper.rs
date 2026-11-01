use atomic_refcell::AtomicRefCell;
use crossbeam::queue::ArrayQueue;
use nice_plug_core::audio_setup::{AudioIOLayout, BufferConfig, ProcessMode};
#[cfg(feature = "editor")]
use nice_plug_core::context::gui::GuiContext;
use nice_plug_core::context::process::{HostCall, HostSegment, Transport};
#[cfg(feature = "editor")]
use nice_plug_core::editor::{Editor, EditorHandle, SpawnedEditor};
use nice_plug_core::midi::PluginNoteEvent;
use nice_plug_core::params::internals::ParamPtr;
use nice_plug_core::params::{ParamFlags, Params};
use nice_plug_core::plugin::{
    NoFields, Plugin, PluginState, PreparedFields, ProcessStatus, RestoreContext, RestoreOrigin,
    RestoreThread, StateRefusal, TaskExecutor,
};
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
#[cfg(feature = "editor")]
use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::thread;

use super::backend::Backend;
use super::config::WrapperConfig;
use super::context::{WrapperActivateContext, WrapperProcessContext};
use crate::event_loop::{EventLoop, MainThreadExecutor, OsEventLoop};
use crate::util::permit_alloc;
#[cfg(feature = "editor")]
use crate::wrapper::standalone::context::WrapperGuiContext;
use crate::wrapper::state;
use crate::wrapper::util::process_wrapper;

/// How many parameter changes we can store in our unprocessed parameter change queue. Storing more
/// than this many parameters at a time will cause changes to get lost.
const EVENT_QUEUE_CAPACITY: usize = 2048;

pub struct Wrapper<P: Plugin, B: Backend<P>> {
    backend: AtomicRefCell<B>,

    /// The wrapped plugin instance.
    plugin: Mutex<P>,
    /// The plugin's background task executor closure. Tasks scheduled by the plugin will be
    /// executed on the GUI or background thread using this function.
    pub task_executor: Mutex<TaskExecutor<P>>,
    /// The plugin's parameters. These are fetched once during initialization. That way the
    /// `ParamPtr`s are guaranteed to live at least as long as this object and we can interact with
    /// the `Params` object without having to acquire a lock on `plugin`.
    params: Arc<dyn Params>,
    /// The plugin's editor, if it has one. This object does not do anything on its own, but we need
    /// to instantiate this in advance so we don't need to lock the entire [`Plugin`] object when
    /// creating an editor. Wrapped in an `AtomicRefCell` because it needs to be initialized late.
    #[allow(clippy::type_complexity)]
    #[cfg(feature = "editor")]
    pub editor: AtomicRefCell<Option<Arc<Mutex<P::Editor>>>>,
    #[allow(clippy::type_complexity)]
    #[cfg(feature = "editor")]
    pub editor_handle: Mutex<Option<<P::Editor as Editor>::Handle>>,
    close_requested: Arc<AtomicBool>,

    /// A realtime-safe task queue so the plugin can schedule tasks that need to be run later on the
    /// GUI thread. See the same field in the VST3 wrapper for more information on why this looks
    /// the way it does.
    event_loop: AtomicRefCell<Option<OsEventLoop<Task<P>, Self>>>,

    /// This is used to grab the DPI scaling config. Not used on macOS.
    #[allow(unused)]
    config: WrapperConfig,

    /// A mapping from parameter pointers to string parameter IDs. This is used as part of
    /// `Task::ParamValueChanged` to send a parameter change event to the editor from the GUI
    /// thread. This is also used to check whether the `ParamPtr` for an incoming parameter change
    /// actually belongs to a registered parameter.
    param_ptr_to_id: HashMap<ParamPtr, String>,
    /// A mapping from parameter string IDs to parameter pointers. Used for serialization and
    /// deserialization.
    param_id_to_ptr: HashMap<String, ParamPtr>,

    /// The bus and buffer configurations are static for the standalone target.
    audio_io_layout: AudioIOLayout,
    buffer_config: BufferConfig,

    /// Parameter changes that have been output by the GUI that have not yet been set in the plugin.
    /// This queue will be flushed at the end of every processing cycle, just like in the plugin
    /// versions.
    unprocessed_param_changes: ArrayQueue<(ParamPtr, f32)>,
    /// The plugin is able to restore state through a method on the `GuiContext`. To avoid changing
    /// parameters mid-processing and running into garbled data, the restore happens at the end of
    /// a processing call: the GUI thread posts it here, the audio thread commits it in place, and
    /// the GUI thread takes it back with its outcome and frees it. The audio side is atomics only
    /// (T1 round-3 fix; the pinned zero-capacity channel locked a mutex on every callback): see
    /// [`state::RestoreMailbox`].
    restores: state::RestoreMailbox<state::PendingRestore<P>>,
    /// The current latency in samples, as set by the plugin through the [`ActivateContext`] and the
    /// [`ProcessContext`]. This value may not be used depending on the audio backend, but it's
    /// still kept track of to avoid firing debug assertions multiple times for the same latency
    /// value.
    current_latency: AtomicU32,
    /// Serial of the backend's audio callbacks, rising by one per callback for the lifetime of
    /// this instance. Reported to the plugin through
    /// [`ProcessContext::host_call()`][nice_plug_core::context::process::ProcessContext::host_call()].
    host_call_serial: AtomicU64,
    /// Set from the moment an accepted atomic restore commits until the restoring thread has
    /// reinitialized the plugin (T1 round-3 fix): meanwhile the audio callback outputs silence and
    /// never touches the plugin mutex the restoring thread holds for `activate()` + `reset()`, so
    /// it neither waits for that lock nor processes the plugin against its stale activation.
    reinit_pending: AtomicBool,
}

/// What the pinned (non-atomic) restore path reports when the plugin's own `activate()` failed on
/// the reinitialization after it: pristine's `false`. The atomic path never reports it -- its
/// state is applied either way, and a refusal there means nothing was written (T1 round-4).
const REINIT_FAILED: StateRefusal =
    StateRefusal::new("the plugin failed to reinitialize after the restore", 0);

/// Tasks that can be sent from the plugin to be executed on the main thread in a non-blocking
/// realtime-safe way (either a random thread or `IRunLoop` on Linux, the OS' message loop on
/// Windows and macOS).
#[allow(clippy::enum_variant_names)]
pub enum Task<P: Plugin> {
    /// Execute one of the plugin's background tasks.
    PluginTask(P::BackgroundTask),
    #[cfg(feature = "editor")]
    StateChanged,
    /// Inform the plugin that one parameter's value has changed. This uses the parameter hashes
    /// since the task will be created from the audio thread. We don't have parameter hashes here
    /// like in the plugin APIs, so we'll just use the `ParamPtr`s directly. These are used to index
    /// the hashmaps stored on `Wrapper`.
    #[cfg(feature = "editor")]
    ParameterValueChanged(ParamPtr, f32),
}

/// Errors that may arise while initializing the wrapped plugins.
#[derive(Debug, thiserror::Error)]
pub enum WrapperError {
    /// The plugin returned `false` during activation.
    #[error("The plugin returned false during activation")]
    ActivationFailed,
    #[cfg(feature = "editor")]
    #[error("{0}")]
    WindowError(#[from] Box<dyn Error>),
}

impl<P: Plugin, B: Backend<P>> MainThreadExecutor<Task<P>> for Wrapper<P, B> {
    fn execute(&self, task: Task<P>, _is_gui_thread: bool) {
        match task {
            Task::PluginTask(task) => (self.task_executor.lock())(task),
            #[cfg(feature = "editor")]
            Task::StateChanged => {
                if let Some(editor_handle) = self.editor_handle.lock().as_ref() {
                    editor_handle.state_changed();
                }
            }
            #[cfg(feature = "editor")]
            Task::ParameterValueChanged(param_ptr, normalized_value) => {
                if let Some(editor) = self.editor_handle.lock().as_ref() {
                    let param_id = &self.param_ptr_to_id[&param_ptr];
                    editor.param_value_changed(param_id, normalized_value);
                }
            }
        }
    }
}

impl<P: Plugin, B: Backend<P>> Wrapper<P, B> {
    /// Instantiate a new instance of the standalone wrapper. Returns an error if the plugin does
    /// not accept the IO configuration from the wrapper config.
    pub fn new(backend: B, config: WrapperConfig) -> Result<Arc<Self>, WrapperError> {
        // The backend has already queried this, so this will never cause the program to exit
        // TODO: Do the validation and parsing in the argument parser so this value can be stored on
        //       the config itself. Right now clap doesn't support this.
        let audio_io_layout = config.audio_io_layout_or_exit::<P>();

        let mut plugin = P::default();
        let task_executor = Mutex::new(plugin.task_executor());
        let params = plugin.params();

        // For consistency's sake we'll include the same assertions as the other backends
        // TODO: Move these common checks to a function instead of repeating them in every wrapper
        let param_map = params.param_map();
        if cfg!(debug_assertions) {
            let param_ids: HashSet<_> = param_map.iter().map(|(id, _, _)| id.clone()).collect();
            crate::nice_debug_assert_eq!(
                param_map.len(),
                param_ids.len(),
                "The plugin has duplicate parameter IDs, weird things may happen. Consider using \
                 6 character parameter IDs to avoid collisions."
            );

            let mut bypass_param_exists = false;
            for (_, ptr, _) in &param_map {
                let flags = unsafe { ptr.flags() };
                let is_bypass = flags.contains(ParamFlags::BYPASS);

                if is_bypass && bypass_param_exists {
                    crate::nice_debug_assert_failure!(
                        "Duplicate bypass parameters found, the host will only use the first one"
                    );
                }

                bypass_param_exists |= is_bypass;
            }
        }

        let wrapper = Arc::new(Wrapper {
            backend: AtomicRefCell::new(backend),

            plugin: Mutex::new(plugin),
            task_executor,
            params,
            // Initialized later as it needs a reference to the wrapper for the async executor
            #[cfg(feature = "editor")]
            editor: AtomicRefCell::new(None),
            #[cfg(feature = "editor")]
            editor_handle: Mutex::new(None),
            close_requested: Arc::new(AtomicBool::new(false)),

            // Also initialized later as it also needs a reference to the wrapper
            event_loop: AtomicRefCell::new(None),

            param_ptr_to_id: param_map
                .iter()
                .map(|(param_id, param_ptr, _)| (*param_ptr, param_id.clone()))
                .collect(),
            param_id_to_ptr: param_map
                .into_iter()
                .map(|(param_id, param_ptr, _)| (param_id, param_ptr))
                .collect(),

            audio_io_layout,
            buffer_config: BufferConfig {
                sample_rate: config.sample_rate,
                min_buffer_size: None,
                max_buffer_size: config.period_size,
                // TODO: Detect JACK freewheeling and report it here
                process_mode: ProcessMode::Realtime,
            },
            config,

            unprocessed_param_changes: ArrayQueue::new(EVENT_QUEUE_CAPACITY),
            restores: state::RestoreMailbox::new(),
            current_latency: AtomicU32::new(0),
            host_call_serial: AtomicU64::new(0),
            reinit_pending: AtomicBool::new(false),
        });

        *wrapper.event_loop.borrow_mut() =
            Some(OsEventLoop::new_and_spawn(Arc::downgrade(&wrapper)));

        // The editor needs to be initialized later so the Async executor can work.
        #[cfg(feature = "editor")]
        {
            *wrapper.editor.borrow_mut() = wrapper
                .plugin
                .lock()
                .editor(nice_plug_core::context::gui::AsyncExecutor::new(
                    Arc::new({
                        let wrapper = wrapper.clone();

                        move |task| {
                            let task_posted = wrapper.schedule_background(Task::PluginTask(task));
                            crate::nice_debug_assert!(
                                task_posted,
                                "The task queue is full, dropping task..."
                            );
                        }
                    }),
                    Arc::new({
                        let wrapper = wrapper.clone();

                        move |task| {
                            let task_posted = wrapper.schedule_gui(Task::PluginTask(task));
                            crate::nice_debug_assert!(
                                task_posted,
                                "The task queue is full, dropping task..."
                            );
                        }
                    }),
                ))
                .map(|editor| {
                    // Safety: This is only called in standalone mode, and only if the user has
                    // enabled the `unsafe_standalone_dpi_fix` feature.
                    #[cfg(feature = "unsafe_standalone_dpi_fix")]
                    unsafe {
                        editor.assume_standalone_in_process();
                    }

                    Arc::new(Mutex::new(editor))
                });
        }

        // Before initializing the plugin, make sure all smoothers are set the the default values
        for param in wrapper.param_id_to_ptr.values() {
            unsafe { param._internal_update_smoother(wrapper.buffer_config.sample_rate, true) };
        }

        {
            let mut plugin = wrapper.plugin.lock();
            if !plugin.activate(
                &wrapper.audio_io_layout,
                &wrapper.buffer_config,
                &mut wrapper.make_activate_context(),
            ) {
                return Err(WrapperError::ActivationFailed);
            }
            process_wrapper(|| plugin.reset());
        }

        Ok(wrapper)
    }

    /// Open the editor, start processing audio, and block this thread until the editor is closed.
    /// If the plugin does not have an editor, then this will block until SIGINT is received.
    ///
    /// Will return an error if the plugin threw an error during audio processing or if the editor
    /// could not be opened.
    pub fn run(self: Arc<Self>) -> Result<(), WrapperError> {
        let close_gui_requested = Arc::clone(&self.close_requested);
        if let Err(e) = ctrlc::set_handler(move || {
            close_gui_requested.store(true, Ordering::Relaxed);
        }) {
            crate::nice_error!("Error setting close signal callback {}", e);
        }

        // We'll spawn a separate thread to handle IO and to process audio. This audio thread should
        // terminate together with this function.
        let terminate_audio_thread = Arc::new(AtomicBool::new(false));
        let close_gui_requested = Arc::clone(&self.close_requested);
        let audio_thread = {
            let this = self.clone();
            let terminate_audio_thread = terminate_audio_thread.clone();
            thread::spawn(move || {
                this.run_audio_thread(terminate_audio_thread, close_gui_requested)
            })
        };

        #[allow(unused_mut)]
        let mut gui_spawned = false;

        #[cfg(feature = "editor")]
        let res = match self.editor.borrow().clone() {
            Some(editor) => {
                let context = self.clone().make_gui_context();

                match editor.lock().spawn(None, false, None, context, None) {
                    Ok(editor_window) => {
                        let SpawnedEditor {
                            handle: instance,
                            window,
                        } = editor_window;

                        gui_spawned = true;
                        *self.editor_handle.lock() = Some(instance);

                        <P::Editor as Editor>::Handle::run_until_closed(window)
                            .map_err(|e| WrapperError::WindowError(e.into()))
                    }
                    Err(e) => Err(WrapperError::WindowError(e)),
                }
            }
            None => Ok(()),
        };

        #[cfg(not(feature = "editor"))]
        let res = Ok(());

        if !gui_spawned {
            crate::nice_log!("No GUI available for {}, blocking indefinitely...", P::NAME);

            loop {
                if self.close_requested.swap(false, Ordering::Relaxed) {
                    break;
                }

                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }

        terminate_audio_thread.store(true, Ordering::SeqCst);

        let _ = audio_thread.join();

        // Some plugins may use this to clean up resources. Should not be needed for the standalone
        // application, but it seems like a good idea to stay consistent.
        self.plugin.lock().deactivate();

        res
    }

    /// Get a parameter's ID based on a `ParamPtr`. Used in the `GuiContext` implementation for the
    /// gesture checks.
    #[allow(unused)]
    pub fn param_id_from_ptr(&self, param: ParamPtr) -> Option<&str> {
        self.param_ptr_to_id.get(&param).map(|s| s.as_str())
    }

    /// Set a parameter based on a `ParamPtr`. The value will be updated at the end of the next
    /// processing cycle, and this won't do anything if the parameter has not been registered by the
    /// plugin.
    ///
    /// This returns false if the parameter was not set because the `ParamPtr` was either unknown or
    /// the queue is full.
    #[cfg(feature = "editor")]
    pub fn set_parameter(&self, param: ParamPtr, normalized: f32) -> bool {
        if !self.param_ptr_to_id.contains_key(&param) {
            return false;
        }

        let push_successful = self
            .unprocessed_param_changes
            .push((param, normalized))
            .is_ok();
        crate::nice_debug_assert!(push_successful, "The parameter change queue was full");

        push_successful
    }

    /// Get the plugin's state object, may be called by the plugin's GUI as part of its own preset
    /// management. The wrapper doesn't use these functions and serializes and deserializes directly
    /// the JSON in the relevant plugin API methods instead.
    #[cfg(feature = "editor")]
    pub fn get_state_object(&self) -> PluginState {
        unsafe {
            state::serialize_object::<P>(
                self.params.clone(),
                self.param_id_to_ptr
                    .iter()
                    .map(|(param_id, param_ptr)| (param_id, *param_ptr)),
            )
        }
    }

    /// The pristine signature (T1 maintainer fix): a caller that never heard of
    /// [`StateRefusal`] still compiles against this fork unchanged. Kept for source
    /// compatibility; nothing in this crate calls it (pristine's own single method is what
    /// `try_set_state_object_from_gui()` replaced as the one `context.rs` calls now).
    #[cfg(feature = "editor")]
    #[allow(dead_code)]
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
    /// updated at the end of the audio processing cycle.
    ///
    /// `Plugin::filter_state()` and, under the atomic contract, `Plugin::prepare_fields()` both
    /// run right here, off the audio thread (T1 maintainer fix): the audio thread only has to
    /// call `commit_restore()`.
    ///
    /// One restore at a time (T1 round-3 fix): a second restoring thread waits up to a second for
    /// the first to finish and is refused with [`StateRefusal::BUSY`] otherwise, nothing written.
    /// Everything the audio thread used to do besides the commit itself -- the hand-back, posting
    /// `Task::StateChanged`, the unknown-parameter warning -- happens here, on this thread.
    #[cfg(feature = "editor")]
    pub fn try_set_state_object_from_gui(
        &self,
        mut state: PluginState,
    ) -> Result<(), StateRefusal> {
        // Under the atomic contract one restore at a time, refused after a second (T1 round-3
        // fix); without it a second restorer waits for as long as the first needs the mailbox,
        // as the pinned blocking channel made it wait (T1 round-4 fix).
        let within = P::ATOMIC_STATE_RESTORE.then_some(std::time::Duration::from_secs(1));
        let Some(_restorer) = self.restores.restorer(within) else {
            return Err(StateRefusal::BUSY);
        };

        let prepared: Box<dyn PreparedFields<P>> = if P::ATOMIC_STATE_RESTORE {
            // Off the audio thread, before the hand-off (T1 maintainer fix); without the contract
            // it runs on the committing thread, in `state::deserialize_object()`, as pinned
            P::filter_state(&mut state);
            state::report_unknown_params(&state, |param_id: &str| {
                self.param_id_to_ptr.get(param_id).copied()
            });
            P::prepare_fields(&state.fields)
        } else {
            Box::new(NoFields)
        };
        let pending = state::PendingRestore {
            state,
            origin: RestoreOrigin::Gui,
            outcome: Ok(()),
            prepared,
        };
        // The audio thread always runs while the standalone does, so this waits for it as the
        // pinned blocking `send()` did: the state is committed at the end of the next callback,
        // in place, and comes back here with its outcome to be freed.
        let outcome = match self.restores.hand_to_audio(pending, None) {
            Ok(returned) => returned.outcome,
            Err(_) => unreachable!("no deadline was given"),
        };
        // An accepted atomic restore left its reinitialization to this thread (T1 round-3 fix);
        // the non-atomic path reinitialized on the audio thread, as pinned.
        // An accepted restore committed under the atomic contract left its reinitialization to
        // this thread (T1 round-3 fix). The state is applied either way, so the outcome stays
        // accepted even if the plugin's own `activate()` then fails (T1 round-4 fix): a refusal
        // means nothing was written. The failure is logged by `reinitialize_after_restore()`.
        if outcome.is_ok() && P::ATOMIC_STATE_RESTORE {
            self.reinitialize_after_restore();
        }

        if outcome.is_ok() && P::ATOMIC_STATE_RESTORE {
            // What the audio thread no longer posts itself under the atomic contract (T1 round-3
            // fix); without it the audio thread posted it, as pinned
            let task_posted = self.schedule_gui(Task::StateChanged);
            crate::nice_debug_assert!(task_posted, "The task queue is full, dropping task...");
        }

        outcome
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

    /// Posts the task to the task queue using [`EventLoop::schedule_gui()`] so it can be delegated
    /// to the main thread. The task is run directly if this is the GUI thread.
    ///
    /// If the task queue is full, then this will return false.
    #[must_use]
    pub fn schedule_gui(&self, task: Task<P>) -> bool {
        let event_loop = self.event_loop.borrow();
        let event_loop = event_loop.as_ref().unwrap();
        event_loop.schedule_gui(task)
    }

    pub fn set_latency_samples(&self, samples: u32) {
        // This should only change the value if it's actually needed
        let old_latency = self.current_latency.swap(samples, Ordering::SeqCst);
        if old_latency != samples {
            // None of the backends actually support this at the moment
            crate::nice_debug_assert_failure!(
                "Standalones currently don't support latency reporting"
            );
        }
    }

    /// The audio thread. This should be called from another thread, and it will run until
    /// `should_terminate` is `true`.
    fn run_audio_thread(
        self: Arc<Self>,
        should_terminate: Arc<AtomicBool>,
        close_gui_requested: Arc<AtomicBool>,
    ) {
        self.clone().backend.borrow_mut().run(
            move |buffer, aux, transport, input_events, output_events| {
                // TODO: This process wrapper should actually be in the backends (since the backends
                //       should also not allocate in their audio callbacks), but that's a bit more
                //       error prone
                process_wrapper(|| {
                    if should_terminate.load(Ordering::SeqCst) {
                        return false;
                    }

                    let sample_rate = self.buffer_config.sample_rate;
                    // One callback is one host call: a single segment anchored at frame 0
                    let host_call = HostCall {
                        serial: self.host_call_serial.fetch_add(1, Ordering::Relaxed) + 1,
                        frames: buffer.samples() as u32,
                        slice_offset: 0,
                        segment: HostSegment::from_transport(&transport),
                    };
                    // Never a wait on the plugin mutex here (T1 round-3 fix; pinned: a blocking
                    // `lock()`). While a restore's reinitialization is pending, or another
                    // thread holds the plugin for any reason, this callback outputs silence and
                    // returns, like CLAP's `try_lock()` in its process call. While the audio
                    // thread runs, the only other holder of the mutex is that
                    // reinitialization, on the restoring thread.
                    let plugin = if self.reinit_pending.load(Ordering::Acquire) {
                        None
                    } else {
                        self.plugin.try_lock()
                    };
                    let Some(mut plugin) = plugin else {
                        for channel in buffer.as_slice() {
                            channel.fill(0.0);
                        }
                        for output in aux.outputs.iter_mut() {
                            for channel in output.as_slice() {
                                channel.fill(0.0);
                            }
                        }
                        return true;
                    };
                    if let ProcessStatus::Error(err) = plugin.process(
                        buffer,
                        aux,
                        &mut self.make_process_context(
                            transport,
                            host_call,
                            input_events,
                            output_events,
                        ),
                    ) {
                        crate::nice_error!("The plugin returned an error while processing:");
                        crate::nice_error!("{}", err);

                        close_gui_requested.store(true, Ordering::Relaxed);

                        return false;
                    }

                    // Any output note events are now in a vector that can be processed by the
                    // audio/MIDI backend

                    // We'll always write these events to the first sample, so even when we add note
                    // output we shouldn't have to think about interleaving events here
                    while let Some((param_ptr, normalized_value)) =
                        self.unprocessed_param_changes.pop()
                    {
                        apply_parameter_value(
                            &param_ptr,
                            normalized_value,
                            Some(sample_rate),
                            || {
                                #[cfg(feature = "editor")]
                                {
                                    let task_posted = self.schedule_gui(
                                        Task::ParameterValueChanged(param_ptr, normalized_value),
                                    );
                                    crate::nice_debug_assert!(
                                        task_posted,
                                        "The task queue is full, dropping task..."
                                    );
                                }
                            },
                        );
                    }

                    // After processing audio, we'll check if the editor has handed us a state to
                    // restore. We'll restore that here on the audio thread to prevent changing the
                    // values during the process call. One atomic load when nothing is pending; the
                    // state is committed in place and freed by the thread that handed it over (T1
                    // round-3 fix: this used to be a zero-capacity channel's `try_recv()`, which
                    // locks a mutex on every call, under a `permit_alloc`).
                    //
                    // Under the atomic contract the commit runs under this callback's own plugin
                    // guard (T1 round-4 fix; pool round-4 terra): the audio thread takes no lock of
                    // its own for it. Without the contract the pinned path runs once the guard is
                    // gone, since its reinitialization takes the plugin's lock itself.
                    if P::ATOMIC_STATE_RESTORE {
                        self.restores.service(|pending| {
                            let restore = RestoreContext {
                                origin: pending.origin,
                                thread: RestoreThread::Audio,
                            };
                            pending.outcome = self.commit_restore_holding(
                                &mut plugin,
                                &mut pending.state,
                                &*pending.prepared,
                                restore,
                            );
                        });
                        drop(plugin);
                    } else {
                        drop(plugin);
                        self.restores.service(|pending| {
                            let restore = RestoreContext {
                                origin: pending.origin,
                                thread: RestoreThread::Audio,
                            };
                            pending.outcome = self.commit_restore(
                                &mut pending.state,
                                &*pending.prepared,
                                restore,
                            );
                        });
                    }

                    true
                })
            },
        );
    }

    #[cfg(feature = "editor")]
    fn make_gui_context(self: Arc<Self>) -> GuiContext {
        GuiContext::new(Arc::new(WrapperGuiContext {
            wrapper: Arc::downgrade(&self),
            #[cfg(debug_assertions)]
            param_gesture_checker: Default::default(),
        }))
    }

    fn make_activate_context(&self) -> WrapperActivateContext<'_, P, B> {
        WrapperActivateContext { wrapper: self }
    }

    fn make_process_context<'a>(
        &'a self,
        transport: Transport,
        host_call: HostCall,
        input_events: &'a [PluginNoteEvent<P>],
        output_events: &'a mut Vec<PluginNoteEvent<P>>,
    ) -> WrapperProcessContext<'a, P, B> {
        WrapperProcessContext {
            wrapper: self,
            input_events,
            input_events_idx: 0,
            output_events,
            transport,
            host_call,
        }
    }

    /// The pristine signature (T1 maintainer fix): stages fields itself, on whichever thread
    /// calls this, so call it off the audio thread; the normal path
    /// (`try_set_state_object_from_gui()`) stages them before any threading decision instead.
    /// Kept for source compatibility; nothing in this fork calls it (same as pristine, where it
    /// had no other caller once `set_state_object_from_gui()` is behind the `editor` feature).
    #[allow(dead_code)]
    fn set_state_inner(&self, state: &mut PluginState) -> bool {
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

    /// Immediately set the plugin state. Returns the plugin's refusal if it would not accept the
    /// state (see [`Plugin::ATOMIC_STATE_RESTORE`]), in which case nothing was written, or a
    /// refusal of its own if the plugin failed to reinitialize afterwards. In other wrappers
    /// state is set from a couple places, so this function is here to be consistent and to
    /// centralize all of this behavior. Without the atomic contract this includes
    /// `permit_alloc()`s around the deserialization and initialization for the use case where
    /// `set_state_object_from_gui()` was called while the plugin is process audio. `prepared` was
    /// already built off the audio thread by `Plugin::prepare_fields()` (T1 maintainer fix,
    /// section 8, S-28).
    ///
    /// Implicitly emits `Task::StateChanged` when committed on the calling thread; after an
    /// audio-thread commit the restoring thread posts it instead (T1 round-3 fix).
    ///
    /// # Notes
    ///
    /// `self.plugin` must _not_ be locked while calling this function or it will deadlock.
    fn commit_restore(
        &self,
        state: &mut PluginState,
        prepared: &dyn PreparedFields<P>,
        restore: RestoreContext,
    ) -> Result<(), StateRefusal> {
        let success = if P::ATOMIC_STATE_RESTORE {
            {
                // The plugin's hooks see the live instance, so this needs the plugin lock: free
                // between process calls and while not processing, taken only while the restoring
                // thread reinitializes after an earlier restore, in which case nothing is
                // written. Never waited for -- same contract, same primitive, as CLAP's and
                // VST3's `commit_restore` (T1 round-1 fix: this used to be a blocking `lock()`,
                // reachable from this audio callback whenever an atomic restore was pending).
                let Some(mut plugin) = self.plugin.try_lock() else {
                    return Err(StateRefusal::BUSY);
                };
                self.commit_restore_holding(&mut plugin, state, prepared, restore)?;
            }
            if restore.thread == RestoreThread::Audio {
                // `commit_restore_holding()` already fenced the callback off the plugin until the
                // restoring thread reinitializes it (T1 round-3 fix)
                return Ok(());
            }
            // Accepted either way: the state is applied (T1 round-4 fix)
            self.reinitialize_after_restore();
            true
        } else {
            let params_getter = |param_id: &str| self.param_id_to_ptr.get(param_id).copied();
            // FIXME: This is obviously not realtime-safe, but loading presets without doing this could
            //        lead to inconsistencies. It's the plugin's responsibility to not perform any
            //        realtime-unsafe work when the initialize function is called a second time if it
            //        supports runtime preset loading. `state::deserialize_object()` normally never
            //        allocates, but if the plugin has persistent non-parameter data then its
            //        `deserialize_fields()` implementation may still allocate.
            permit_alloc(|| unsafe {
                state::deserialize_object::<P>(
                    state,
                    None,
                    self.params.clone(),
                    params_getter,
                    Some(&self.buffer_config),
                    restore,
                )
            })?;

            // If the plugin was already initialized then it needs to be reinitialized (the
            // non-atomic path, as pinned)
            let success;
            {
                // NOTE: This needs to be dropped after the `plugin` lock to avoid deadlocks
                let mut activate_context = self.make_activate_context();
                let mut plugin = self.plugin.lock();

                // See above
                success = permit_alloc(|| {
                    plugin.activate(
                        &self.audio_io_layout,
                        &self.buffer_config,
                        &mut activate_context,
                    )
                });
                if success {
                    process_wrapper(|| plugin.reset());
                }
            }
            success
        };

        crate::nice_debug_assert!(
            success,
            "Plugin returned false when reinitializing after loading state"
        );

        // Under the atomic contract from the calling thread only: after an audio-thread commit the
        // restoring thread posts this (T1 round-3 fix), so nothing here touches the event loop
        // from the audio thread. Without the contract wherever the commit runs, as pinned.
        #[cfg(feature = "editor")]
        if restore.thread == RestoreThread::Caller || !P::ATOMIC_STATE_RESTORE {
            let task_posted = self.schedule_gui(Task::StateChanged);
            crate::nice_debug_assert!(task_posted, "The task queue is full, dropping task...");
        }

        if success { Ok(()) } else { Err(REINIT_FAILED) }
    }

    /// The atomic commit with the plugin already held by the caller (T1 round-4 fix; pool round-4
    /// terra): the audio callback commits a restore handed to it under its own plugin guard, so
    /// the audio thread takes no lock of its own for it. Committed on the audio thread, it leaves
    /// the reinitialization to the restoring thread and fences the callback off the plugin until
    /// then (`reinit_pending`, T1 round-3 fix: pinned, the reinitialization took a second,
    /// blocking `self.plugin.lock()` here and ran the plugin's `activate()` under a
    /// `permit_alloc`). Posts nothing.
    fn commit_restore_holding(
        &self,
        plugin: &mut P,
        state: &mut PluginState,
        prepared: &dyn PreparedFields<P>,
        restore: RestoreContext,
    ) -> Result<(), StateRefusal> {
        let params_getter = |param_id: &str| self.param_id_to_ptr.get(param_id).copied();
        unsafe {
            state::commit_atomic_restore::<P>(
                state,
                plugin,
                params_getter,
                Some(&self.buffer_config),
                prepared,
                restore,
            )?;
        }
        if restore.thread == RestoreThread::Audio {
            self.reinit_pending.store(true, Ordering::Release);
        }
        Ok(())
    }

    /// Reinitialize the plugin after an accepted atomic restore (`activate()` + `reset()`), on
    /// the restoring thread, never the audio thread (T1 round-3 fix). `reinit_pending` fences the
    /// audio callback off the plugin for the whole of it, so the blocking `lock()` here only ever
    /// waits for another non-audio holder, and the callback never waits for this one. The fence
    /// clears even if the plugin panics in here (T1 round-4 fix: a drop guard), so a panic never
    /// silences the callback for good. A failed `activate()` is logged here, on the restoring
    /// thread; it does not turn the already-applied restore into a refusal (T1 round-4 fix).
    fn reinitialize_after_restore(&self) {
        /// Clears the fence on every way out, the unwinding of a panic included.
        struct LiftFence<'a>(&'a AtomicBool);
        impl Drop for LiftFence<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Release);
            }
        }

        self.reinit_pending.store(true, Ordering::Release);
        let _lift = LiftFence(&self.reinit_pending);
        let success = {
            // NOTE: This needs to be dropped after the `plugin` lock to avoid deadlocks
            let mut activate_context = self.make_activate_context();
            let mut plugin = self.plugin.lock();
            let success = plugin.activate(
                &self.audio_io_layout,
                &self.buffer_config,
                &mut activate_context,
            );
            if success {
                process_wrapper(|| plugin.reset());
            }
            success
        };
        if !success {
            crate::nice_error!(
                "The state was restored, but the plugin returned false when reinitializing after \
                 it"
            );
        }
    }
}

/// T1 round-1 fix: `commit_restore`'s atomic branch used to take a blocking `Mutex::lock()`,
/// reachable from the audio callback (`run_audio_thread`'s end-of-callback restore drain)
/// whenever an atomic restore was pending -- a real audio-thread wait, unlike CLAP's and VST3's
/// `try_lock()`/[`StateRefusal::BUSY`] contract. Pinned here directly against a minimal plugin
/// built on the dummy backend, since the fork's own suite has no other standalone-wrapper
/// harness and standalone is out of DECKS's build.
#[cfg(test)]
mod tests {
    use super::super::backend::Dummy;
    use super::Wrapper;
    use super::WrapperConfig;
    use crate::prelude::*;
    use clap::Parser;
    use parking_lot::Mutex;
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    /// No parameters: `#[derive(Params)]` expands to paths rooted at the literal crate name
    /// `nice_plug`, which only resolves for an external consumer (as every other plugin in this
    /// fork is, including the integration tests under `tests/`) -- not from a unit test compiled
    /// as part of the `nice_plug` crate itself. A hand-written empty impl sidesteps that; nothing
    /// here needs an actual parameter.
    struct NoParams;

    // SAFETY: `param_map()` returns no pointers, so there is nothing for a caller to dereference.
    unsafe impl Params for NoParams {
        fn param_map(&self) -> Vec<(String, ParamPtr, String)> {
            Vec::new()
        }
    }

    struct Minimal {
        params: std::sync::Arc<NoParams>,
    }

    impl Default for Minimal {
        fn default() -> Self {
            Self {
                params: std::sync::Arc::new(NoParams),
            }
        }
    }

    impl Plugin for Minimal {
        const NAME: &'static str = "T1 Round-1 Minimal";
        const VENDOR: &'static str = "Cascadian Tech";
        const URL: &'static str = "";
        const EMAIL: &'static str = "";
        const VERSION: &'static str = "0.0.0";
        const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
            main_input_channels: None,
            main_output_channels: NonZeroU32::new(2),
            aux_input_ports: &[],
            aux_output_ports: &[],
            names: PortNames::const_default(),
        }];
        // The plugin under test does take the atomic restore contract: `commit_restore`'s
        // `if P::ATOMIC_STATE_RESTORE` branch is exactly the one this test exercises.
        const ATOMIC_STATE_RESTORE: bool = true;

        type Editor = ();
        type SysExMessage = ();
        type BackgroundTask = ();

        fn params(&self) -> std::sync::Arc<dyn Params> {
            self.params.clone()
        }

        fn activate(
            &mut self,
            _audio_io_layout: &AudioIOLayout,
            _buffer_config: &BufferConfig,
            _context: &mut impl ActivateContext<Self>,
        ) -> bool {
            true
        }

        fn process(
            &mut self,
            _buffer: &mut Buffer,
            _aux: &mut AuxiliaryBuffers,
            _context: &mut impl ProcessContext<Self>,
        ) -> ProcessStatus {
            ProcessStatus::Normal
        }
    }

    fn minimal_wrapper() -> std::sync::Arc<Wrapper<Minimal, Dummy>> {
        let config = WrapperConfig::parse_from(["t1-round1-test"]);
        let backend = Dummy::new::<Minimal>(config.clone());
        Wrapper::<Minimal, Dummy>::new(backend, config)
            .expect("the minimal plugin accepts its own (only) default IO layout")
    }

    /// The counterexample: an atomic restore drained on the audio callback while another thread
    /// (simulated here by simply holding the guard on this thread -- `parking_lot::Mutex` is not
    /// reentrant, so `commit_restore` blocking on `self.plugin.lock()` here would deadlock this
    /// test, not just run slow) holds the plugin mutex. The fixed, non-blocking contract refuses
    /// immediately with nothing written, exactly like CLAP's and VST3's `commit_restore`.
    #[test]
    fn standalone_commit_restore_refuses_busy_instead_of_blocking_when_contended() {
        let wrapper = minimal_wrapper();
        let _held = wrapper.plugin.lock();

        let mut state = PluginState {
            version: String::new(),
            params: BTreeMap::new(),
            fields: BTreeMap::new(),
        };
        let prepared: Box<dyn PreparedFields<Minimal>> = Box::new(NoFields);
        let restore = RestoreContext {
            origin: RestoreOrigin::Gui,
            thread: RestoreThread::Audio,
        };
        let outcome = wrapper.commit_restore(&mut state, &*prepared, restore);
        assert_eq!(
            outcome,
            Err(StateRefusal::BUSY),
            "a contended plugin lock refuses instead of waiting on the audio thread"
        );
    }
    /// The plugin of the contention test below: takes the atomic restore contract, and records
    /// when its restore commits, every activation (with the activating thread's name) and every
    /// processed callback.
    struct Contended {
        params: std::sync::Arc<NoParams>,
    }

    static CONTENDED_RESTORED: AtomicBool = AtomicBool::new(false);
    static CONTENDED_PROCESSED: AtomicU64 = AtomicU64::new(0);
    static CONTENDED_ACTIVATIONS: Mutex<Vec<String>> = Mutex::new(Vec::new());

    /// The audio thread of the contention test.
    const CONTENDED_AUDIO: &str = "t1-standalone-audio";

    impl Default for Contended {
        fn default() -> Self {
            Self {
                params: std::sync::Arc::new(NoParams),
            }
        }
    }

    impl Plugin for Contended {
        const NAME: &'static str = "T1 Round-3 Contended";
        const VENDOR: &'static str = "Cascadian Tech";
        const URL: &'static str = "";
        const EMAIL: &'static str = "";
        const VERSION: &'static str = "0.0.0";
        const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
            main_input_channels: None,
            main_output_channels: NonZeroU32::new(2),
            aux_input_ports: &[],
            aux_output_ports: &[],
            names: PortNames::const_default(),
        }];
        const ATOMIC_STATE_RESTORE: bool = true;

        type Editor = ();
        type SysExMessage = ();
        type BackgroundTask = ();

        fn params(&self) -> std::sync::Arc<dyn Params> {
            self.params.clone()
        }

        fn activate(
            &mut self,
            _audio_io_layout: &AudioIOLayout,
            _buffer_config: &BufferConfig,
            _context: &mut impl ActivateContext<Self>,
        ) -> bool {
            let thread = std::thread::current();
            CONTENDED_ACTIVATIONS
                .lock()
                .push(thread.name().unwrap_or("unnamed").to_string());
            true
        }

        fn state_restored(&mut self, _restore: RestoreContext) {
            CONTENDED_RESTORED.store(true, Ordering::SeqCst);
        }

        fn process(
            &mut self,
            _buffer: &mut Buffer,
            _aux: &mut AuxiliaryBuffers,
            _context: &mut impl ProcessContext<Self>,
        ) -> ProcessStatus {
            CONTENDED_PROCESSED.fetch_add(1, Ordering::SeqCst);
            ProcessStatus::Normal
        }
    }

    /// T1 round-3 (pool round-3 terra): an ACCEPTED atomic restore under plugin-lock contention.
    /// The pinned reinitialization after a successful commit (`activate()` + `reset()`) took a
    /// second, blocking `self.plugin.lock()` on the audio thread, so another holder of the plugin
    /// mutex stalled the audio callback -- there, or at the callback's own `process()` lock right
    /// after. Here the real audio callback runs on the dummy backend; the moment the restore
    /// commits, a contender takes the plugin mutex and holds it for a quarter of a second. The
    /// callback must keep returning on schedule throughout without processing the plugin, the
    /// restore must still report success, and the reinitialization must run once, off the audio
    /// thread, once the contender lets go.
    #[cfg(feature = "editor")]
    #[test]
    fn standalone_accepted_restore_under_lock_contention_never_blocks_the_audio_callback() {
        use std::time::{Duration, Instant};

        let config = WrapperConfig::parse_from(["t1-round3-test"]);
        let backend = Dummy::new::<Contended>(config.clone());
        let wrapper = Wrapper::<Contended, Dummy>::new(backend, config)
            .expect("the plugin accepts its only layout");
        assert_eq!(
            CONTENDED_ACTIVATIONS.lock().len(),
            1,
            "activated once at startup"
        );

        let terminate = Arc::new(AtomicBool::new(false));
        let audio = {
            let (wrapper, terminate) = (wrapper.clone(), terminate.clone());
            std::thread::Builder::new()
                .name(CONTENDED_AUDIO.to_string())
                .spawn(move || {
                    wrapper.run_audio_thread(terminate, Arc::new(AtomicBool::new(false)))
                })
                .unwrap()
        };
        let started = Instant::now();
        while CONTENDED_PROCESSED.load(Ordering::SeqCst) < 2 {
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "the callback never ran"
            );
            std::thread::sleep(Duration::from_millis(1));
        }

        // The contender: takes the plugin mutex as soon as the restore has committed and holds it.
        let contender = {
            let wrapper = wrapper.clone();
            std::thread::spawn(move || {
                let started = Instant::now();
                while !CONTENDED_RESTORED.load(Ordering::SeqCst) {
                    assert!(
                        started.elapsed() < Duration::from_secs(10),
                        "never committed"
                    );
                    std::thread::sleep(Duration::from_micros(100));
                }
                let held = wrapper.plugin.lock();
                let callbacks = wrapper.host_call_serial.load(Ordering::SeqCst);
                let processed = CONTENDED_PROCESSED.load(Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(250));
                let callbacks = wrapper.host_call_serial.load(Ordering::SeqCst) - callbacks;
                let processed = CONTENDED_PROCESSED.load(Ordering::SeqCst) - processed;
                drop(held);
                (callbacks, processed)
            })
        };

        let state = PluginState {
            version: String::new(),
            params: BTreeMap::new(),
            fields: BTreeMap::new(),
        };
        let outcome = wrapper.try_set_state_object_from_gui(state);
        let (callbacks, processed) = contender.join().unwrap();

        assert_eq!(outcome, Ok(()), "the accepted restore reports success");
        // About 23 dummy periods of 512 frames at 48 kHz fit in 250 ms; a blocked callback
        // manages at most the one already in flight.
        assert!(
            callbacks >= 3,
            "the audio callback kept running while the plugin mutex was held: {callbacks}"
        );
        assert_eq!(
            processed, 0,
            "nothing was processed while the plugin was held or its reinitialization pending"
        );
        let activations = CONTENDED_ACTIVATIONS.lock().clone();
        assert_eq!(
            activations.len(),
            2,
            "reinitialized exactly once: {activations:?}"
        );
        assert_ne!(
            activations[1], CONTENDED_AUDIO,
            "the reinitialization ran off the audio thread"
        );

        // Processing resumes once the reinitialization is done.
        let resumed = CONTENDED_PROCESSED.load(Ordering::SeqCst);
        let started = Instant::now();
        while CONTENDED_PROCESSED.load(Ordering::SeqCst) == resumed {
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "processing never resumed"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        terminate.store(true, Ordering::SeqCst);
        audio.join().unwrap();
    }
    /// What the two reinitialization tests below share: an atomic plugin whose first
    /// `activate()` (at startup) succeeds and whose second (the reinitialization after a restore)
    /// fails or panics, per `FAIL_WITH_PANIC`.
    struct Flaky<const FAIL_WITH_PANIC: bool> {
        params: std::sync::Arc<NoParams>,
    }

    static FLAKY_ACTIVATIONS: AtomicU64 = AtomicU64::new(0);
    static FLAKY_PANICKY_ACTIVATIONS: AtomicU64 = AtomicU64::new(0);
    static FLAKY_RESTORED: AtomicBool = AtomicBool::new(false);
    static FLAKY_PANICKY_RESTORED: AtomicBool = AtomicBool::new(false);

    impl<const FAIL_WITH_PANIC: bool> Default for Flaky<FAIL_WITH_PANIC> {
        fn default() -> Self {
            Self {
                params: std::sync::Arc::new(NoParams),
            }
        }
    }

    impl<const FAIL_WITH_PANIC: bool> Plugin for Flaky<FAIL_WITH_PANIC> {
        const NAME: &'static str = "T1 Round-4 Flaky";
        const VENDOR: &'static str = "Cascadian Tech";
        const URL: &'static str = "";
        const EMAIL: &'static str = "";
        const VERSION: &'static str = "0.0.0";
        const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
            main_input_channels: None,
            main_output_channels: NonZeroU32::new(2),
            aux_input_ports: &[],
            aux_output_ports: &[],
            names: PortNames::const_default(),
        }];
        const ATOMIC_STATE_RESTORE: bool = true;

        type Editor = ();
        type SysExMessage = ();
        type BackgroundTask = ();

        fn params(&self) -> std::sync::Arc<dyn Params> {
            self.params.clone()
        }

        fn activate(
            &mut self,
            _audio_io_layout: &AudioIOLayout,
            _buffer_config: &BufferConfig,
            _context: &mut impl ActivateContext<Self>,
        ) -> bool {
            let counter = if FAIL_WITH_PANIC {
                &FLAKY_PANICKY_ACTIVATIONS
            } else {
                &FLAKY_ACTIVATIONS
            };
            let first = counter.fetch_add(1, Ordering::SeqCst) == 0;
            if !first && FAIL_WITH_PANIC {
                panic!("the plugin panicked while reinitializing");
            }
            first
        }

        fn state_restored(&mut self, _restore: RestoreContext) {
            if FAIL_WITH_PANIC {
                FLAKY_PANICKY_RESTORED.store(true, Ordering::SeqCst);
            } else {
                FLAKY_RESTORED.store(true, Ordering::SeqCst);
            }
        }

        fn process(
            &mut self,
            _buffer: &mut Buffer,
            _aux: &mut AuxiliaryBuffers,
            _context: &mut impl ProcessContext<Self>,
        ) -> ProcessStatus {
            ProcessStatus::Normal
        }
    }

    /// Start the wrapper's own audio callback on the dummy backend; set the flag to stop it.
    fn start_audio<P: Plugin>(
        wrapper: &Arc<Wrapper<P, Dummy>>,
    ) -> (Arc<AtomicBool>, std::thread::JoinHandle<()>) {
        let terminate = Arc::new(AtomicBool::new(false));
        let handle = {
            let (wrapper, terminate) = (wrapper.clone(), terminate.clone());
            std::thread::spawn(move || {
                wrapper.run_audio_thread(terminate, Arc::new(AtomicBool::new(false)))
            })
        };
        (terminate, handle)
    }

    fn empty_state() -> PluginState {
        PluginState {
            version: String::new(),
            params: BTreeMap::new(),
            fields: BTreeMap::new(),
        }
    }

    /// T1 round-4 (pool round-4 luna; gemini's note): under the atomic contract a refusal means
    /// nothing was written. An accepted restore whose state committed and whose reinitialization
    /// then fails (the plugin's own `activate()` returns `false`) is reported accepted -- the
    /// failed reinitialization is logged on the restoring thread, never reported as a refusal of
    /// a state that was already applied.
    #[cfg(feature = "editor")]
    #[test]
    fn standalone_restore_with_a_failed_reinitialization_is_not_reported_refused() {
        let config = WrapperConfig::parse_from(["t1-round4-test"]);
        let backend = Dummy::new::<Flaky<false>>(config.clone());
        let wrapper = Wrapper::<Flaky<false>, Dummy>::new(backend, config).unwrap();
        let (terminate, audio) = start_audio(&wrapper);

        let outcome = wrapper.try_set_state_object_from_gui(empty_state());
        assert!(
            FLAKY_RESTORED.load(Ordering::SeqCst),
            "the state was committed"
        );
        assert_eq!(
            FLAKY_ACTIVATIONS.load(Ordering::SeqCst),
            2,
            "and reinitialized once"
        );
        assert_eq!(
            outcome,
            Ok(()),
            "a committed restore is not reported refused because its reinitialization failed"
        );
        assert!(!wrapper.reinit_pending.load(Ordering::SeqCst));

        terminate.store(true, Ordering::SeqCst);
        audio.join().unwrap();
    }

    /// T1 round-4 (pool round-4 qwenmax's note): a reinitialization that panics on the restoring
    /// thread does not leave the audio callback silenced for good: the pending flag clears as the
    /// panic unwinds.
    #[cfg(feature = "editor")]
    #[test]
    fn standalone_reinitialization_that_panics_does_not_silence_the_callback_for_good() {
        let config = WrapperConfig::parse_from(["t1-round4-test"]);
        let backend = Dummy::new::<Flaky<true>>(config.clone());
        let wrapper = Wrapper::<Flaky<true>, Dummy>::new(backend, config).unwrap();
        let (terminate, audio) = start_audio(&wrapper);

        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            wrapper.try_set_state_object_from_gui(empty_state())
        }));
        assert!(
            unwound.is_err(),
            "the plugin's panic reached the restoring thread"
        );
        assert!(FLAKY_PANICKY_RESTORED.load(Ordering::SeqCst));
        assert!(
            !wrapper.reinit_pending.load(Ordering::SeqCst),
            "the callback is not left silenced"
        );

        terminate.store(true, Ordering::SeqCst);
        audio.join().unwrap();
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
