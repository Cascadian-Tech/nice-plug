//! Protocol admission before the buffer manager creates any sample slice.
use clap_sys::process::clap_process;
use nice_plug_core::audio_setup::AudioIOLayout;
#[cfg(feature = "vst3")]
use std::sync::atomic::AtomicBool;
#[cfg(feature = "vst3")]
use vst3::Steinberg::Vst::ProcessData;

// Test-first stubs: reject-shape regressions must fail before these are implemented.
pub(crate) unsafe fn clap_valid(
    _layout: AudioIOLayout,
    _process: &clap_process,
    _min: u32,
    _max: u32,
) -> bool {
    true
}
#[cfg(feature = "vst3")]
pub(crate) unsafe fn vst3_valid(
    _layout: AudioIOLayout,
    _data: &ProcessData,
    _inputs: &[AtomicBool],
    _outputs: &[AtomicBool],
    _max: u32,
) -> bool {
    true
}
#[cfg(feature = "vst3")]
pub(crate) unsafe fn vst3_flush(_layout: AudioIOLayout, _data: &ProcessData) -> bool {
    false
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
}
