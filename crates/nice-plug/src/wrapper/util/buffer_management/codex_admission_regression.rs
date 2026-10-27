use super::*;
use nice_plug_core::audio_setup::{PortNames, new_nonzero_u32};

const LAYOUT: AudioIOLayout = AudioIOLayout {
    main_input_channels: None,
    main_output_channels: Some(new_nonzero_u32(2)),
    aux_input_ports: &[],
    aux_output_ports: &[new_nonzero_u32(2), new_nonzero_u32(2)],
    names: PortNames::const_default(),
};

#[test]
fn absent_optional_outputs_have_independent_zeroed_channels() {
    let mut manager = BufferManager::for_audio_io_layout(8, LAYOUT);
    let buffers = unsafe { manager.create_buffers(2, 4, |_| {}) };
    assert_eq!(buffers.main_buffer.as_slice_immutable().len(), 2);
    for buffer in std::iter::once(&*buffers.main_buffer).chain(buffers.aux_outputs.iter()) {
        for channel in buffer.as_slice_immutable() {
            assert_eq!(
                channel.len(),
                4,
                "absent output must retain a valid bounded slice"
            );
            assert!(channel.iter().all(|s| *s == 0.0));
        }
    }
    buffers.main_buffer.as_slice()[0][0] = 0.25;
    assert_eq!(buffers.main_buffer.as_slice_immutable()[1][0], 0.0);
    for output in buffers.aux_outputs.iter() {
        assert!(
            output
                .as_slice_immutable()
                .iter()
                .all(|c| c.iter().all(|s| *s == 0.0))
        );
    }
    buffers.aux_outputs[0].as_slice()[0][0] = 0.5;
    assert_eq!(buffers.aux_outputs[0].as_slice_immutable()[1][0], 0.0);
    assert_eq!(buffers.aux_outputs[1].as_slice_immutable()[0][0], 0.0);
}

#[test]
fn discard_channels_clear_between_calls_and_offsets() {
    let mut manager = BufferManager::for_audio_io_layout(8, LAYOUT);
    {
        let buffers = unsafe { manager.create_buffers(0, 8, |_| {}) };
        for output in std::iter::once(buffers.main_buffer).chain(buffers.aux_outputs.iter_mut()) {
            for channel in output.as_slice() {
                assert_eq!(channel.len(), 8);
                channel.fill(0.75);
            }
        }
    }
    let buffers = unsafe { manager.create_buffers(3, 2, |_| {}) };
    for output in std::iter::once(&*buffers.main_buffer).chain(buffers.aux_outputs.iter()) {
        for channel in output.as_slice_immutable() {
            assert_eq!(channel.len(), 2);
            assert!(
                channel.iter().all(|s| *s == 0.0),
                "stale discard samples leaked"
            );
        }
    }
}

#[test]
fn inactive_main_keeps_active_input_for_other_outputs() {
    let layout = AudioIOLayout {
        main_input_channels: Some(new_nonzero_u32(2)),
        ..LAYOUT
    };
    let mut manager = BufferManager::for_audio_io_layout(8, layout);
    let mut left = [0.25_f32; 8];
    let mut right = [-0.5_f32; 8];
    let mut input = [left.as_mut_ptr(), right.as_mut_ptr()];
    let buffers = unsafe {
        manager.create_buffers(2, 3, |source| {
            *source.main_input_channel_pointers = Some(ChannelPointers {
                ptrs: NonNull::new(input.as_mut_ptr()).unwrap(),
                num_channels: 2,
            });
        })
    };
    assert_eq!(buffers.main_buffer.as_slice_immutable()[0], [0.25; 3]);
    assert_eq!(buffers.main_buffer.as_slice_immutable()[1], [-0.5; 3]);
}

#[test]
fn invalid_slice_window_is_rejected_before_any_mapping() {
    let mut manager = BufferManager::for_audio_io_layout(8, LAYOUT);
    assert!(unsafe { manager.try_create_buffers(7, 2, |_| {}) }.is_none());
    assert!(unsafe { manager.try_create_buffers(usize::MAX, 2, |_| {}) }.is_none());
    assert!(unsafe { manager.try_create_buffers(0, 9, |_| {}) }.is_none());
}

#[test]
fn valid_in_place_main_input_is_preserved() {
    let layout = AudioIOLayout {
        main_input_channels: Some(new_nonzero_u32(2)),
        ..LAYOUT
    };
    let mut manager = BufferManager::for_audio_io_layout(8, layout);
    let mut left = [0.125_f32; 8];
    let mut right = [-0.25_f32; 8];
    let mut pointers = [left.as_mut_ptr(), right.as_mut_ptr()];
    let buffers = unsafe {
        manager.create_buffers(1, 4, |source| {
            let ptrs = NonNull::new(pointers.as_mut_ptr()).unwrap();
            *source.main_input_channel_pointers = Some(ChannelPointers {
                ptrs,
                num_channels: 2,
            });
            *source.main_output_channel_pointers = Some(ChannelPointers {
                ptrs,
                num_channels: 2,
            });
        })
    };
    assert_eq!(buffers.main_buffer.as_slice_immutable()[0], [0.125; 4]);
    assert_eq!(buffers.main_buffer.as_slice_immutable()[1], [-0.25; 4]);
}
