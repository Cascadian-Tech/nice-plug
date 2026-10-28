//! Protocol admission before the buffer manager creates any sample slice.
use clap_sys::{audio_buffer::clap_audio_buffer, process::clap_process};
use nice_plug_core::audio_setup::AudioIOLayout;
use std::num::NonZeroU32;
#[cfg(feature = "vst3")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "vst3")]
use vst3::Steinberg::Vst::{AudioBusBuffers, ProcessData};

fn bus_count(main: Option<NonZeroU32>, auxiliary: &[NonZeroU32]) -> usize {
    usize::from(main.is_some()) + auxiliary.len()
}
fn bus_channels(main: Option<NonZeroU32>, auxiliary: &[NonZeroU32], index: usize) -> usize {
    if let Some(main) = main
        && index == 0
    {
        return main.get() as usize;
    }
    auxiliary[index - usize::from(main.is_some())].get() as usize
}
unsafe fn samples_valid(pointers: *mut *mut f32, channels: usize) -> bool {
    if pointers.is_null() || !pointers.is_aligned() {
        return false;
    }
    (0..channels).all(|channel| {
        let pointer = unsafe { *pointers.add(channel) };
        !pointer.is_null() && pointer.is_aligned()
    })
}
unsafe fn clap_ports_valid(
    raw: *const clap_audio_buffer,
    count: u32,
    main: Option<NonZeroU32>,
    auxiliary: &[NonZeroU32],
) -> bool {
    if count as usize != bus_count(main, auxiliary) {
        return false;
    }
    if count > 0 && (raw.is_null() || !raw.is_aligned()) {
        return false;
    }
    (0..count as usize).all(|index| {
        let bus = unsafe { &*raw.add(index) };
        let channels = bus_channels(main, auxiliary, index);
        bus.channel_count as usize == channels && unsafe { samples_valid(bus.data32, channels) }
    })
}
// Callers retain the SDK pointer-array/sample-region allocation and lifetime contract.
pub(crate) unsafe fn clap_valid(
    layout: AudioIOLayout,
    process: &clap_process,
    min: u32,
    max: u32,
) -> bool {
    process.frames_count >= min
        && process.frames_count <= max
        && unsafe {
            clap_ports_valid(
                process.audio_inputs,
                process.audio_inputs_count,
                layout.main_input_channels,
                layout.aux_input_ports,
            )
        }
        && unsafe {
            clap_ports_valid(
                process.audio_outputs,
                process.audio_outputs_count,
                layout.main_output_channels,
                layout.aux_output_ports,
            )
        }
}
#[cfg(feature = "vst3")]
unsafe fn vst_ports_valid(
    raw: *const AudioBusBuffers,
    count: i32,
    main: Option<NonZeroU32>,
    auxiliary: &[NonZeroU32],
    active: &[AtomicBool],
) -> bool {
    let expected = bus_count(main, auxiliary);
    if count < 0 || count as usize > expected || active.len() < expected {
        return false;
    }
    if count > 0 && (raw.is_null() || !raw.is_aligned()) {
        return false;
    }
    (0..expected).all(|index| {
        let active = active[index].load(Ordering::Acquire);
        if index >= count as usize {
            return !active;
        }
        let bus = unsafe { &*raw.add(index) };
        let channels = bus_channels(main, auxiliary, index);
        let pointers = unsafe { bus.__field0.channelBuffers32 };
        bus.numChannels >= 0
            && bus.numChannels as usize == channels
            && !pointers.is_null()
            && pointers.is_aligned()
            && (!active || unsafe { samples_valid(pointers, channels) })
    })
}
#[cfg(feature = "vst3")]
pub(crate) unsafe fn vst3_valid(
    layout: AudioIOLayout,
    data: &ProcessData,
    inputs: &[AtomicBool],
    outputs: &[AtomicBool],
    max: u32,
) -> bool {
    data.numSamples > 0
        && data.numSamples as u32 <= max
        && data.symbolicSampleSize == vst3::Steinberg::Vst::SymbolicSampleSizes_::kSample32 as i32
        && unsafe {
            vst_ports_valid(
                data.inputs,
                data.numInputs,
                layout.main_input_channels,
                layout.aux_input_ports,
                inputs,
            )
        }
        && unsafe {
            vst_ports_valid(
                data.outputs,
                data.numOutputs,
                layout.main_output_channels,
                layout.aux_output_ports,
                outputs,
            )
        }
}
#[cfg(feature = "vst3")]
pub(crate) unsafe fn vst3_flush(layout: AudioIOLayout, data: &ProcessData) -> bool {
    if data.numSamples < 0 || data.numInputs < 0 || data.numOutputs < 0 {
        return false;
    }
    if data.numSamples == 0 {
        return true;
    }
    // Uniform zero-channel audio is a separate parameter-only shape. An absent
    // ordinary pointer array or a single malformed bus must never become flush.
    let expected = bus_count(layout.main_output_channels, layout.aux_output_ports);
    if expected == 0
        || data.numOutputs as usize != expected
        || data.outputs.is_null()
        || !data.outputs.is_aligned()
    {
        return false;
    }
    (0..expected).all(|index| unsafe { (*data.outputs.add(index)).numChannels == 0 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap_sys::audio_buffer::clap_audio_buffer;
    use nice_plug_core::audio_setup::{PortNames, new_nonzero_u32};
    use std::sync::atomic::Ordering;
    const LAYOUT: AudioIOLayout = AudioIOLayout {
        main_input_channels: None,
        main_output_channels: Some(new_nonzero_u32(2)),
        aux_input_ports: &[],
        aux_output_ports: &[new_nonzero_u32(2)],
        names: PortNames::const_default(),
    };
    const INPUT_PORTS: &[std::num::NonZeroU32] = &[new_nonzero_u32(2)];
    struct ClapAudio {
        process: clap_process,
        buffers: Box<[clap_audio_buffer; 2]>,
        _pointers: Box<[[*mut f32; 2]; 2]>,
        _samples: Box<[[f32; 8]; 4]>,
    }
    impl ClapAudio {
        fn new() -> Self {
            let mut samples = Box::new([[0.0; 8]; 4]);
            let mut pointers = Box::new([[std::ptr::null_mut(); 2]; 2]);
            let mut buffers: Box<[clap_audio_buffer; 2]> = Box::new(unsafe { std::mem::zeroed() });
            for bus in 0..2 {
                for channel in 0..2 {
                    pointers[bus][channel] = samples[bus * 2 + channel].as_mut_ptr();
                }
                buffers[bus].channel_count = 2;
                buffers[bus].data32 = pointers[bus].as_mut_ptr();
            }
            let mut process: clap_process = unsafe { std::mem::zeroed() };
            process.frames_count = 4;
            process.audio_outputs_count = 2;
            process.audio_outputs = buffers.as_mut_ptr();
            Self {
                process,
                buffers,
                _pointers: pointers,
                _samples: samples,
            }
        }
        fn valid(&self) -> bool {
            unsafe { clap_valid(LAYOUT, &self.process, 1, 8) }
        }
    }
    #[test]
    fn clap_rejects_short_array_before_unreported_read() {
        let mut audio = ClapAudio::new();
        assert!(audio.valid());
        audio.process.audio_outputs_count = 1;
        assert!(!audio.valid());
    }
    #[test]
    fn clap_rejects_wrong_channels_and_missing_pointer_array() {
        let mut audio = ClapAudio::new();
        audio.buffers[1].channel_count = 1;
        assert!(!audio.valid());
        audio.buffers[1].channel_count = 2;
        audio.buffers[1].data32 = std::ptr::null_mut();
        assert!(!audio.valid());
    }
    #[test]
    fn clap_rejects_zero_and_over_capacity_frames() {
        let mut audio = ClapAudio::new();
        audio.process.frames_count = 0;
        assert!(!audio.valid());
        audio.process.frames_count = 9;
        assert!(!audio.valid());
    }
    #[cfg(feature = "vst3")]
    struct VstAudio {
        data: ProcessData,
        buffers: Box<[vst3::Steinberg::Vst::AudioBusBuffers; 2]>,
        pointers: Box<[[*mut f32; 2]; 2]>,
        _samples: Box<[[f32; 8]; 4]>,
        active: [AtomicBool; 2],
    }
    #[cfg(feature = "vst3")]
    impl VstAudio {
        fn new() -> Self {
            let mut samples = Box::new([[0.0; 8]; 4]);
            let mut pointers = Box::new([[std::ptr::null_mut(); 2]; 2]);
            let mut buffers: Box<[vst3::Steinberg::Vst::AudioBusBuffers; 2]> =
                Box::new(unsafe { std::mem::zeroed() });
            for bus in 0..2 {
                for ch in 0..2 {
                    pointers[bus][ch] = samples[bus * 2 + ch].as_mut_ptr();
                }
                buffers[bus].numChannels = 2;
                buffers[bus].__field0.channelBuffers32 = pointers[bus].as_mut_ptr();
            }
            let mut data: ProcessData = unsafe { std::mem::zeroed() };
            data.numSamples = 4;
            data.numOutputs = 2;
            data.outputs = buffers.as_mut_ptr();
            Self {
                data,
                buffers,
                pointers,
                _samples: samples,
                active: [AtomicBool::new(true), AtomicBool::new(true)],
            }
        }
        fn valid(&self) -> bool {
            unsafe { vst3_valid(LAYOUT, &self.data, &[], &self.active, 8) }
        }
    }
    #[test]
    #[cfg(feature = "vst3")]
    fn vst_rejects_short_active_array_and_negative_counts() {
        let mut audio = VstAudio::new();
        assert!(audio.valid());
        audio.data.numOutputs = 1;
        assert!(!audio.valid());
        audio.data.numOutputs = -1;
        assert!(!audio.valid());
    }
    #[test]
    #[cfg(feature = "vst3")]
    fn vst_inactive_trailing_bus_can_be_omitted_but_not_active_one() {
        let mut audio = VstAudio::new();
        audio.data.numOutputs = 1;
        audio.active[1].store(false, Ordering::Relaxed);
        assert!(audio.valid());
        audio.active[1].store(true, Ordering::Relaxed);
        assert!(!audio.valid());
    }
    #[test]
    #[cfg(feature = "vst3")]
    fn vst_null_individual_samples_require_inactive_bus() {
        let mut audio = VstAudio::new();
        audio.pointers[1].fill(std::ptr::null_mut());
        audio.active[1].store(false, Ordering::Relaxed);
        assert!(audio.valid());
        audio.active[1].store(true, Ordering::Relaxed);
        assert!(!audio.valid());
    }
    #[test]
    #[cfg(feature = "vst3")]
    fn vst_missing_pointer_array_is_not_inactive_samples_or_flush() {
        let mut audio = VstAudio::new();
        audio.buffers[1].__field0.channelBuffers32 = std::ptr::null_mut();
        audio.active[1].store(false, Ordering::Relaxed);
        assert!(!audio.valid());
        assert!(!unsafe { vst3_flush(LAYOUT, &audio.data) });
    }
    #[test]
    #[cfg(feature = "vst3")]
    fn vst_input_count_is_independent_of_output_count() {
        let audio = VstAudio::new();
        let layout = AudioIOLayout {
            aux_input_ports: INPUT_PORTS,
            ..LAYOUT
        };
        assert!(!unsafe {
            vst3_valid(
                layout,
                &audio.data,
                &[AtomicBool::new(true)],
                &audio.active,
                8,
            )
        });
    }
    #[test]
    #[cfg(feature = "vst3")]
    fn vst_parameter_flush_is_distinct_from_regular_audio() {
        let mut audio = VstAudio::new();
        assert!(!unsafe { vst3_flush(LAYOUT, &audio.data) });
        audio.data.numSamples = 0;
        assert!(unsafe { vst3_flush(LAYOUT, &audio.data) });
        audio.data.numSamples = 4;
        for bus in audio.buffers.iter_mut() {
            bus.numChannels = 0;
        }
        assert!(unsafe { vst3_flush(LAYOUT, &audio.data) });
    }

    #[test]
    fn review_no_declared_clap_input_rejects_extra_before_access() {
        let mut audio = ClapAudio::new();
        assert!(audio.valid());
        // Zero declared input buses means an empty iteration. An extra reported
        // bus is invalid even when its pointer is NULL: reject before access.
        audio.process.audio_inputs_count = 1;
        assert!(!audio.valid());
        audio.process.audio_inputs_count = 0;
        assert!(audio.valid());
    }

    #[test]
    fn review_aux_only_clap_uses_each_declared_channel_count() {
        const AUX: &[NonZeroU32] = &[new_nonzero_u32(4), new_nonzero_u32(2)];
        const LAYOUT: AudioIOLayout = AudioIOLayout {
            main_input_channels: None,
            main_output_channels: None,
            aux_input_ports: &[],
            aux_output_ports: AUX,
            names: PortNames::const_default(),
        };
        let mut samples = Box::new([[0.0_f32; 8]; 6]);
        let mut pointers = [std::ptr::null_mut(); 6];
        for (p, samples) in pointers.iter_mut().zip(samples.iter_mut()) {
            *p = samples.as_mut_ptr();
        }
        let mut buses: [clap_audio_buffer; 2] = unsafe { std::mem::zeroed() };
        buses[0].channel_count = 4;
        buses[0].data32 = pointers.as_mut_ptr();
        buses[1].channel_count = 2;
        buses[1].data32 = unsafe { pointers.as_mut_ptr().add(4) };
        let mut process: clap_process = unsafe { std::mem::zeroed() };
        process.frames_count = 4;
        process.audio_outputs_count = 2;
        process.audio_outputs = buses.as_mut_ptr();
        assert!(unsafe { clap_valid(LAYOUT, &process, 1, 8) });
        buses[0].channel_count = 2;
        process.audio_outputs = buses.as_mut_ptr();
        assert!(!unsafe { clap_valid(LAYOUT, &process, 1, 8) });
        process.audio_outputs_count = 1;
        assert!(!unsafe { clap_valid(LAYOUT, &process, 1, 8) });
    }

    #[cfg(feature = "vst3")]
    #[test]
    fn review_aux_only_vst_input_count_is_independent() {
        const INPUT: &[NonZeroU32] = &[new_nonzero_u32(2)];
        const LAYOUT: AudioIOLayout = AudioIOLayout {
            main_input_channels: None,
            main_output_channels: None,
            aux_input_ports: INPUT,
            aux_output_ports: &[],
            names: PortNames::const_default(),
        };
        let mut audio = VstAudio::new();
        audio.data.inputs = audio.buffers.as_mut_ptr();
        audio.data.numInputs = 1;
        audio.data.outputs = std::ptr::null_mut();
        audio.data.numOutputs = 0;
        let inputs = [AtomicBool::new(true)];
        assert!(unsafe { vst3_valid(LAYOUT, &audio.data, &inputs, &[], 8) });
        let with_main = AudioIOLayout {
            main_output_channels: Some(new_nonzero_u32(2)),
            ..LAYOUT
        };
        let outputs = [AtomicBool::new(true)];
        assert!(!unsafe { vst3_valid(with_main, &audio.data, &inputs, &outputs, 8) });
        audio.data.numInputs = 0;
        assert!(!unsafe { vst3_valid(LAYOUT, &audio.data, &inputs, &[], 8) });
    }
}
