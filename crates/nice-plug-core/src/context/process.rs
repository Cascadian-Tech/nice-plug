//! A context passed during the process function.

use crate::{
    midi::{MidiConfig, PluginNoteEvent},
    plugin::Plugin,
};

use super::PluginApi;

/// Contains both context data and callbacks the plugin can use during processing. Most notably this
/// is how a plugin sends and receives note events, gets transport information, and accesses
/// sidechain inputs and auxiliary outputs. This is passed to the plugin during as part of
/// [`Plugin::process()`][crate::plugin::Plugin::process()].
//
// # Safety
//
// The implementing wrapper needs to be able to handle concurrent requests, and it should perform
// the actual callback within [MainThreadQueue::schedule_gui].
pub trait ProcessContext<P: Plugin> {
    /// Get the current plugin API.
    fn plugin_api(&self) -> PluginApi;

    /// Execute a task on a background thread using `[Plugin::task_executor]`. This allows you to
    /// defer expensive tasks for later without blocking either the process function or the GUI
    /// thread. As long as creating the `task` is realtime-safe, this operation is too.
    ///
    /// # Note
    ///
    /// Scheduling the same task multiple times will cause those duplicate tasks to pile up. Try to
    /// either prevent this from happening, or check whether the task still needs to be completed in
    /// your task executor.
    fn execute_background(&self, task: P::BackgroundTask);

    /// Execute a task on a background thread using `[Plugin::task_executor]`. As long as creating
    /// the `task` is realtime-safe, this operation is too.
    ///
    /// # Note
    ///
    /// Scheduling the same task multiple times will cause those duplicate tasks to pile up. Try to
    /// either prevent this from happening, or check whether the task still needs to be completed in
    /// your task executor.
    fn execute_gui(&self, task: P::BackgroundTask);

    /// Get information about the current transport position and status.
    fn transport(&self) -> &Transport;

    /// Get the original host call around this slice: its serial, the host's frame count, this
    /// slice's offset within it and the raw transport segment current at the slice. Automation
    /// splits may call [`Plugin::process()`][crate::plugin::Plugin::process()] several times per
    /// host call; every slice of one call sees the same serial, frame count and segment anchor
    /// plus its own integer offset, so beat positions can be evaluated from the host's own anchor
    /// instead of being rederived from a slice start. Wrappers that do not supply this return
    /// [`HostCall::ABSENT`].
    fn host_call(&self) -> &HostCall {
        &HostCall::ABSENT
    }

    /// Returns the next note event, if there is one. Use
    /// [`NoteEvent::timing()`][crate::midi::NoteEvent::timing()] to get the event's timing
    /// within the buffer. Only available when [`Plugin::MIDI_INPUT`] is set.
    ///
    /// # Usage
    ///
    /// You will likely want to use this with a loop, since there may be zero, one, or more events
    /// for a sample:
    ///
    /// ```ignore
    /// let mut next_event = context.next_event();
    /// for (sample_id, channel_samples) in buffer.iter_samples().enumerate() {
    ///     while let Some(event) = next_event {
    ///         if event.timing() != sample_id as u32 {
    ///             break;
    ///         }
    ///
    ///         match event {
    ///             NoteEvent::NoteOn { note, velocity, .. } => { ... },
    ///             NoteEvent::NoteOff { note, .. } if note == 69 => { ... },
    ///             NoteEvent::PolyPressure { note, pressure, .. } { ... },
    ///             _ => (),
    ///         }
    ///
    ///         next_event = context.next_event();
    ///     }
    ///
    ///     // Do something with `channel_samples`...
    /// }
    ///
    /// ProcessStatus::Normal
    /// ```
    fn next_event(&mut self) -> Option<PluginNoteEvent<P>>;

    /// Try to send an event to the host's output event buffer. Only available when
    /// [`Plugin::MIDI_OUTPUT`] is set.
    fn try_send_event(
        &mut self,
        event: PluginNoteEvent<P>,
    ) -> Result<(), (PluginNoteEvent<P>, SendEventError)>;

    /// Update the current latency of the plugin. If the plugin is currently processing audio, then
    /// this may cause audio playback to be restarted.
    fn set_latency_samples(&self, samples: u32);

    /// Request the plugin to be restarted.
    fn request_restart(&self);

    /// Set the current voice **capacity** for this plugin (so not the number of currently active
    /// voices). This may only be called if `ClapPlugin::CLAP_POLY_MODULATION_CONFIG` is set.
    /// `capacity` must be between 1 and the configured maximum capacity. Changing this at runtime
    /// allows the host to better optimize polyphonic modulation, or to switch to strictly monophonic
    /// modulation when dropping the capacity down to 1.
    fn set_current_voice_capacity(&self, capacity: u32);

    // TODO: Add this, this works similar to [GuiContext::set_parameter] but it adds the parameter
    //       change to a queue (or directly to the VST3 plugin's parameter output queues) instead of
    //       using main thread host automation (and all the locks involved there).
    // fn set_parameter<P: Param>(&self, param: &P, value: P::Plain);
}

/// An error occurred while sending an event with [`ProcessContext::try_send_event()`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SendEventError {
    /// The host's output event buffer is full.
    #[error("Failed to send output event: Host output event buffer is full")]
    HostBufferFull,
    /// The host does not have an output event buffer.
    #[error("Failed to send output event: Host does not have output event buffer")]
    NoOutputBuffer,
    /// Invalid event type for the plugin's [`Plugin::MIDI_OUTPUT`] configuration.
    #[error(
        "Failed to send output event: Invalid event type for output config {midi_output_config:?}"
    )]
    InvalidEvent { midi_output_config: MidiConfig },
}

/// Read-only description of the original host call around the current process slice. See
/// [`ProcessContext::host_call()`]. Plain `Copy` data: nothing in it borrows from the host.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HostCall {
    /// Serial of the host's process call, rising by one per call to the wrapper's process entry
    /// for the lifetime of the plugin instance, never per slice. `0` only in [`Self::ABSENT`].
    pub serial: u64,
    /// The host's frame count for the whole call.
    pub frames: u32,
    /// The first frame of this slice within the call. Equal to `frames` only for an empty call.
    pub slice_offset: u32,
    /// Raw transport values current at this slice, anchored at the frame the host supplied them
    /// for.
    pub segment: HostSegment,
}

impl HostCall {
    /// No host call: what a wrapper without this information reports.
    pub const ABSENT: Self = Self {
        serial: 0,
        frames: 0,
        slice_offset: 0,
        segment: HostSegment::ABSENT,
    };
}

/// One transport segment of a host call: the raw values the host supplied, anchored at the frame
/// it supplied them for. In CLAP a transport event in the input stream opens a new segment at its
/// event sample; the call's own transport block and VST3's process context are anchored at frame
/// 0. The gated fields read zero where the host did not flag them valid (see
/// [`HostSegmentValidity`]); [`raw`][Self::raw] keeps what the host actually sent beside them,
/// whatever its flags say.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HostSegment {
    /// The frame within the call at which these values hold.
    pub start_frame: u32,
    /// Whether an in-stream transport event (CLAP) opened this segment; VST3 never sets it.
    pub from_event: bool,
    /// Whether the transport is running.
    pub playing: bool,
    /// Whether the host is within a pre-roll (CLAP only).
    pub preroll: bool,
    /// Which of the fields below the host supplied.
    pub valid: HostSegmentValidity,
    /// Tempo in beats per minute.
    pub tempo: f64,
    /// Tempo change per sample in beats per minute (CLAP `tempo_inc`); VST3 has none.
    pub tempo_increment: f64,
    /// Song position in quarter notes at `start_frame`.
    pub beats: f64,
    /// Song position in seconds at `start_frame` (CLAP seconds timeline).
    pub seconds: f64,
    /// Project position in samples at `start_frame` (VST3 `projectTimeSamples`).
    pub samples: i64,
    /// The current bar's start in quarter notes.
    pub bar_start: f64,
    /// Time signature numerator.
    pub time_sig_numerator: i32,
    /// Time signature denominator.
    pub time_sig_denominator: i32,
    /// The host's own payload for this segment, exactly as its transport struct carried it,
    /// beside the gated fields above (T1 round-3: an unflagged field's value is kept, not lost
    /// at the wrapper). Where a field is flagged valid it equals its gated twin.
    pub raw: HostSegmentRaw,
}

/// A transport segment's payload exactly as the host's own struct carried it, whatever the
/// host's validity flags say: CLAP's `clap_event_transport` or VST3's `ProcessContext`. In the
/// same units as [`HostSegment`]'s gated fields (CLAP's fixed-point positions divided by their
/// factor, exact for any position below 2^22 beats or seconds); zero where the format has no
/// such field at all (CLAP has no sample position; VST3 has no seconds timeline and no tempo
/// increment). The host's flags themselves are [`flags`][Self::flags], untranslated.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HostSegmentRaw {
    /// The host's own flag word: CLAP `clap_event_transport::flags`, VST3
    /// `ProcessContext::state`. `0` from a wrapper with no raw transport block.
    pub flags: u32,
    /// CLAP `tempo`; VST3 `tempo`.
    pub tempo: f64,
    /// CLAP `tempo_inc`; VST3 has none.
    pub tempo_increment: f64,
    /// CLAP `song_pos_beats`; VST3 `projectTimeMusic`.
    pub beats: f64,
    /// CLAP `song_pos_seconds`; VST3 has none.
    pub seconds: f64,
    /// VST3 `projectTimeSamples`; CLAP has none.
    pub samples: i64,
    /// CLAP `bar_start`; VST3 `barPositionMusic`.
    pub bar_start: f64,
    /// CLAP `tsig_num`; VST3 `timeSigNumerator`.
    pub time_sig_numerator: i32,
    /// CLAP `tsig_denom`; VST3 `timeSigDenominator`.
    pub time_sig_denominator: i32,
}

impl HostSegmentRaw {
    /// No payload at all.
    pub const ZERO: Self = Self {
        flags: 0,
        tempo: 0.0,
        tempo_increment: 0.0,
        beats: 0.0,
        seconds: 0.0,
        samples: 0,
        bar_start: 0.0,
        time_sig_numerator: 0,
        time_sig_denominator: 0,
    };
}

impl HostSegment {
    /// No transport information, anchored at the call start.
    pub const ABSENT: Self = Self {
        start_frame: 0,
        from_event: false,
        playing: false,
        preroll: false,
        valid: HostSegmentValidity::NONE,
        tempo: 0.0,
        tempo_increment: 0.0,
        beats: 0.0,
        seconds: 0.0,
        samples: 0,
        bar_start: 0.0,
        time_sig_numerator: 0,
        time_sig_denominator: 0,
        raw: HostSegmentRaw::ZERO,
    };

    /// A segment anchored at frame 0 from the public transport fields a wrapper already filled
    /// in, for wrappers without a raw transport block (the standalone backends). There is no host
    /// payload beyond those fields, so [`raw`][Self::raw] repeats them, with no flag word.
    pub fn from_transport(transport: &Transport) -> Self {
        let tempo = transport.tempo.unwrap_or(0.0);
        let beats = transport.pos_beats.unwrap_or(0.0);
        let seconds = transport.pos_seconds.unwrap_or(0.0);
        let samples = transport.pos_samples.unwrap_or(0);
        let bar_start = transport.bar_start_pos_beats.unwrap_or(0.0);
        let time_sig_numerator = transport.time_sig_numerator.unwrap_or(0);
        let time_sig_denominator = transport.time_sig_denominator.unwrap_or(0);
        Self {
            start_frame: 0,
            from_event: false,
            playing: transport.playing,
            preroll: transport.preroll_active.unwrap_or(false),
            valid: HostSegmentValidity {
                tempo: transport.tempo.is_some(),
                tempo_increment: false,
                beats: transport.pos_beats.is_some(),
                seconds: transport.pos_seconds.is_some(),
                samples: transport.pos_samples.is_some(),
                bar_start: transport.bar_start_pos_beats.is_some(),
                time_signature: transport.time_sig_numerator.is_some()
                    && transport.time_sig_denominator.is_some(),
            },
            tempo,
            tempo_increment: 0.0,
            beats,
            seconds,
            samples,
            bar_start,
            time_sig_numerator,
            time_sig_denominator,
            raw: HostSegmentRaw {
                flags: 0,
                tempo,
                tempo_increment: 0.0,
                beats,
                seconds,
                samples,
                bar_start,
                time_sig_numerator,
                time_sig_denominator,
            },
        }
    }
}

/// Which raw fields of a [`HostSegment`] the host supplied.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HostSegmentValidity {
    /// `tempo`.
    pub tempo: bool,
    /// `tempo_increment`: CLAP supplies it together with the tempo, VST3 never does.
    pub tempo_increment: bool,
    /// `beats`.
    pub beats: bool,
    /// `seconds`.
    pub seconds: bool,
    /// `samples`.
    pub samples: bool,
    /// `bar_start`.
    pub bar_start: bool,
    /// `time_sig_numerator` and `time_sig_denominator`.
    pub time_signature: bool,
}

impl HostSegmentValidity {
    /// Nothing supplied.
    pub const NONE: Self = Self {
        tempo: false,
        tempo_increment: false,
        beats: false,
        seconds: false,
        samples: false,
        bar_start: false,
        time_signature: false,
    };
}

/// Information about the plugin's transport. Depending on the plugin API and the host not all
/// fields may be available.
#[derive(Debug)]
pub struct Transport {
    /// Whether the transport is currently running.
    pub playing: bool,
    /// Whether recording is enabled in the project.
    pub recording: bool,
    /// Whether the pre-roll is currently active, if the plugin API reports this information.
    pub preroll_active: Option<bool>,

    /// The sample rate in Hertz. Also passed in
    /// [`Plugin::activate()`][crate::plugin::Plugin::activate()], so if you need this then you
    /// can also store that value.
    pub sample_rate: f32,
    /// The project's tempo in beats per minute.
    pub tempo: Option<f64>,
    /// The time signature's numerator.
    pub time_sig_numerator: Option<i32>,
    /// The time signature's denominator.
    pub time_sig_denominator: Option<i32>,

    // XXX: VST3 also has a continuous time in samples that ignores loops, but we can't reconstruct
    //      something similar in CLAP so it may be best to just ignore that so you can't rely on it
    /// The position in the song in samples. Can be used to calculate the time in seconds if needed.
    pub pos_samples: Option<i64>,
    /// The position in the song in seconds. Can be used to calculate the time in samples if needed.
    pub pos_seconds: Option<f64>,
    /// The position in the song in quarter notes. Can be calculated from the time in seconds and
    /// the tempo if needed.
    pub pos_beats: Option<f64>,
    /// The last bar's start position in beats. Can be calculated from the beat position and time
    /// signature if needed.
    pub bar_start_pos_beats: Option<f64>,
    /// The number of the bar at `bar_start_pos_beats`. This starts at 0 for the very first bar at
    /// the start of the song. Can be calculated from the beat position and time signature if
    /// needed.
    pub bar_number: Option<i32>,

    /// The loop range in samples, if the loop is active and this information is available. None of
    /// the plugin API docs mention whether this is exclusive or inclusive, but just assume that the
    /// end is exclusive. Can be calculated from the other loop range information if needed.
    pub loop_range_samples: Option<(i64, i64)>,
    /// The loop range in seconds, if the loop is active and this information is available. None of
    /// the plugin API docs mention whether this is exclusive or inclusive, but just assume that the
    /// end is exclusive. Can be calculated from the other loop range information if needed.
    pub loop_range_seconds: Option<(f64, f64)>,
    /// The loop range in quarter notes, if the loop is active and this information is available.
    /// None of the plugin API docs mention whether this is exclusive or inclusive, but just assume
    /// that the end is exclusive. Can be calculated from the other loop range information if
    /// needed.
    pub loop_range_beats: Option<(f64, f64)>,
}

impl Transport {
    /// Initialize the transport struct without any information.
    pub fn new(sample_rate: f32) -> Self {
        Self {
            playing: false,
            recording: false,
            preroll_active: None,

            sample_rate,
            tempo: None,
            time_sig_numerator: None,
            time_sig_denominator: None,

            pos_samples: None,
            pos_seconds: None,
            pos_beats: None,
            bar_start_pos_beats: None,
            bar_number: None,

            loop_range_samples: None,
            loop_range_seconds: None,
            loop_range_beats: None,
        }
    }

    /// The position in the song in samples. Will be calculated from other information if needed.
    pub fn pos_samples(&self) -> Option<i64> {
        match (
            self.pos_samples,
            self.pos_seconds,
            self.pos_beats,
            self.tempo,
        ) {
            (Some(pos_samples), _, _, _) => Some(pos_samples),
            (_, Some(pos_seconds), _, _) => {
                Some((pos_seconds * self.sample_rate as f64).round() as i64)
            }
            (_, _, Some(pos_beats), Some(tempo)) => {
                Some((pos_beats / tempo * 60.0 * self.sample_rate as f64).round() as i64)
            }
            (_, _, _, _) => None,
        }
    }

    /// The position in the song in seconds. Can be used to calculate the time in samples if needed.
    pub fn pos_seconds(&self) -> Option<f64> {
        match (
            self.pos_samples,
            self.pos_seconds,
            self.pos_beats,
            self.tempo,
        ) {
            (_, Some(pos_seconds), _, _) => Some(pos_seconds),
            (Some(pos_samples), _, _, _) => Some(pos_samples as f64 / self.sample_rate as f64),
            (_, _, Some(pos_beats), Some(tempo)) => Some(pos_beats / tempo * 60.0),
            (_, _, _, _) => None,
        }
    }

    /// The position in the song in quarter notes. Will be calculated from other information if
    /// needed.
    pub fn pos_beats(&self) -> Option<f64> {
        match (
            self.pos_samples,
            self.pos_seconds,
            self.pos_beats,
            self.tempo,
        ) {
            (_, _, Some(pos_beats), _) => Some(pos_beats),
            (_, Some(pos_seconds), _, Some(tempo)) => Some(pos_seconds / 60.0 * tempo),
            (Some(pos_samples), _, _, Some(tempo)) => {
                Some(pos_samples as f64 / self.sample_rate as f64 / 60.0 * tempo)
            }
            (_, _, _, _) => None,
        }
    }

    /// The last bar's start position in beats. Will be calculated from other information if needed.
    pub fn bar_start_pos_beats(&self) -> Option<f64> {
        if self.bar_start_pos_beats.is_some() {
            return self.bar_start_pos_beats;
        }

        match (
            self.time_sig_numerator,
            self.time_sig_denominator,
            self.pos_beats(),
        ) {
            (Some(time_sig_numerator), Some(time_sig_denominator), Some(pos_beats)) => {
                let quarter_note_bar_length =
                    time_sig_numerator as f64 / time_sig_denominator as f64 * 4.0;
                Some((pos_beats / quarter_note_bar_length).floor() * quarter_note_bar_length)
            }
            (_, _, _) => None,
        }
    }

    /// The number of the bar at `bar_start_pos_beats`. This starts at 0 for the very first bar at
    /// the start of the song. Will be calculated from other information if needed.
    pub fn bar_number(&self) -> Option<i32> {
        if self.bar_number.is_some() {
            return self.bar_number;
        }

        match (
            self.time_sig_numerator,
            self.time_sig_denominator,
            self.pos_beats(),
        ) {
            (Some(time_sig_numerator), Some(time_sig_denominator), Some(pos_beats)) => {
                let quarter_note_bar_length =
                    time_sig_numerator as f64 / time_sig_denominator as f64 * 4.0;
                Some((pos_beats / quarter_note_bar_length).floor() as i32)
            }
            (_, _, _) => None,
        }
    }

    /// The loop range in samples, if the loop is active and this information is available. None of
    /// the plugin API docs mention whether this is exclusive or inclusive, but just assume that the
    /// end is exclusive. Will be calculated from other information if needed.
    pub fn loop_range_samples(&self) -> Option<(i64, i64)> {
        match (
            self.loop_range_samples,
            self.loop_range_seconds,
            self.loop_range_beats,
            self.tempo,
        ) {
            (Some(loop_range_samples), _, _, _) => Some(loop_range_samples),
            (_, Some((start_seconds, end_seconds)), _, _) => Some((
                ((start_seconds * self.sample_rate as f64).round() as i64),
                ((end_seconds * self.sample_rate as f64).round() as i64),
            )),
            (_, _, Some((start_beats, end_beats)), Some(tempo)) => Some((
                (start_beats / tempo * 60.0 * self.sample_rate as f64).round() as i64,
                (end_beats / tempo * 60.0 * self.sample_rate as f64).round() as i64,
            )),
            (_, _, _, _) => None,
        }
    }

    /// The loop range in seconds, if the loop is active and this information is available. None of
    /// the plugin API docs mention whether this is exclusive or inclusive, but just assume that the
    /// end is exclusive. Will be calculated from other information if needed.
    pub fn loop_range_seconds(&self) -> Option<(f64, f64)> {
        match (
            self.loop_range_samples,
            self.loop_range_seconds,
            self.loop_range_beats,
            self.tempo,
        ) {
            (_, Some(loop_range_seconds), _, _) => Some(loop_range_seconds),
            (Some((start_samples, end_samples)), _, _, _) => Some((
                start_samples as f64 / self.sample_rate as f64,
                end_samples as f64 / self.sample_rate as f64,
            )),
            (_, _, Some((start_beats, end_beats)), Some(tempo)) => {
                Some((start_beats / tempo * 60.0, end_beats / tempo * 60.0))
            }
            (_, _, _, _) => None,
        }
    }

    /// The loop range in quarter notes, if the loop is active and this information is available.
    /// None of the plugin API docs mention whether this is exclusive or inclusive, but just assume
    /// that the end is exclusive. Will be calculated from other information if needed.
    pub fn loop_range_beats(&self) -> Option<(f64, f64)> {
        match (
            self.loop_range_samples,
            self.loop_range_seconds,
            self.loop_range_beats,
            self.tempo,
        ) {
            (_, _, Some(loop_range_beats), _) => Some(loop_range_beats),
            (_, Some((start_seconds, end_seconds)), _, Some(tempo)) => {
                Some((start_seconds / 60.0 * tempo, end_seconds / 60.0 * tempo))
            }
            (Some((start_samples, end_samples)), _, _, Some(tempo)) => Some((
                start_samples as f64 / self.sample_rate as f64 / 60.0 * tempo,
                end_samples as f64 / self.sample_rate as f64 / 60.0 * tempo,
            )),
            (_, _, _, _) => None,
        }
    }
}
