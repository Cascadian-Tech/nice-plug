//! Helpers for safely constructing [`Buffer`]s from a plugin host's audio buffers.

pub(crate) mod admission;

use std::num::NonZeroU32;
use std::ptr::NonNull;

use nice_plug_core::audio_setup::AudioIOLayout;
use nice_plug_core::buffer::Buffer;

/// Buffers created using [`create_buffers`]. At some point the main `Plugin::process()` should
/// probably also take an argument like this instead of main+aux buffers if we also want to provide
/// access to overflowing input channels for e.g. stereo to mono plugins.
pub struct Buffers<'a, 'buffer: 'a> {
    pub main_buffer: &'a mut Buffer<'buffer>,

    // We can't use `AuxiliaryBuffers` here directly because we need different lifetimes for `'a`
    // and `'buffer` while `AuxiliaryBuffers` uses the same lifetime for both.
    pub aux_inputs: &'a mut [Buffer<'buffer>],
    pub aux_outputs: &'a mut [Buffer<'buffer>],
}

/// A helper for safely creating and initializing [`Buffer`]s based on the host's input and output
/// buffers.
pub struct BufferManager {
    max_buffer_size: usize,
    main_input_channels: usize,
    // Each channel owns a separate allocation, stable until the next activation.
    main_output_storage: Vec<Vec<f32>>,
    aux_output_storage: Vec<Vec<Vec<f32>>>,
    // These are the storage backing the fields in `BufferSource`. The wrapper needs to set these
    // values to match the channel pointers provided by the host. If audio buffers are not provided
    // for a bus, then they should be set to `None`. This helper will then copy data to the buffers
    // or fill them with zeroes if there is no data, while also accounting for in-place main IO
    // buffers.
    main_input_channel_pointers: Option<ChannelPointers>,
    main_output_channel_pointers: Option<ChannelPointers>,
    aux_input_channel_pointers: Vec<Option<ChannelPointers>>,
    aux_output_channel_pointers: Vec<Option<ChannelPointers>>,

    /// The backing buffers that will be filled during `create_buffers`. This `'static` lifetime
    /// will be shortened when returning a reference to these buffers in `create_buffers` to match
    /// the function's lifetime.
    main_buffer: Buffer<'static>,

    aux_input_buffers: Vec<Buffer<'static>>,
    /// Stores the data to back `aux_input_buffers`. We need to copy the host's auxiliary input
    /// buffers to our own first because the `Buffer` API is designed around mutable buffers, and
    /// the host may reuse its input buffers between plugins.
    aux_input_storage: Vec<Vec<Vec<f32>>>,

    aux_output_buffers: Vec<Buffer<'static>>,
}

// SAFETY: The raw pointers in the `ChannelPointers` fields/vectors are only used as scratch storage
//         inside of the `create_buffers()` function.
unsafe impl Send for BufferManager {}
unsafe impl Sync for BufferManager {}

/// Host data that the plugin's [`Buffer`]s should be created from. Leave these fields as `None`
/// values
#[derive(Debug)]
pub struct BufferSource<'a> {
    pub main_input_channel_pointers: &'a mut Option<ChannelPointers>,
    pub main_output_channel_pointers: &'a mut Option<ChannelPointers>,
    pub aux_input_channel_pointers: &'a mut [Option<ChannelPointers>],
    pub aux_output_channel_pointers: &'a mut [Option<ChannelPointers>],
}

/// Pointers to raw multichannel audio data for this port.
#[derive(Debug, Clone, Copy)]
pub struct ChannelPointers {
    /// A raw pointer to an array of f32 arrays, containing one array for each channel. `ptrs` must
    /// contain (at least) `num_channel` `*const f32`s, and each of those inner arrays must contain
    /// (at least) `num_samples` `f32` values.
    pub ptrs: NonNull<*mut f32>,
    /// The number of audio channels used for this port.
    pub num_channels: usize,
}

impl BufferManager {
    /// Initialize managed buffers for a specific audio IO layout. The actual buffers can be set up
    /// using channel pointer data using [`create_buffers()`][Self::create_buffers()].
    pub fn for_audio_io_layout(max_buffer_size: usize, audio_io_layout: AudioIOLayout) -> Self {
        crate::nice_debug_assert!(
            audio_io_layout
                .main_input_channels
                .map(NonZeroU32::get)
                .unwrap_or(0)
                <= audio_io_layout
                    .main_output_channels
                    .map(NonZeroU32::get)
                    .unwrap_or(0),
            "Stereo-to-mono and other many-to-few audio channel configurations are currently not \
             supported"
        );

        // The buffers are preallocated so that `create_buffers()` can be called without having to
        // allocate
        let mut main_buffer = Buffer::default();
        unsafe {
            main_buffer.set_slices(0, |output_slices| {
                output_slices.resize_with(
                    audio_io_layout
                        .main_output_channels
                        .map(NonZeroU32::get)
                        .unwrap_or(0) as usize,
                    || &mut [],
                );
            })
        };

        let mut aux_input_buffers = Vec::with_capacity(audio_io_layout.aux_input_ports.len());
        let mut aux_input_storage = Vec::with_capacity(audio_io_layout.aux_input_ports.len());
        for num_channels in audio_io_layout.aux_input_ports {
            let mut buffer = Buffer::default();
            unsafe {
                buffer.set_slices(0, |slices| {
                    slices.resize_with(num_channels.get() as usize, || &mut []);
                })
            };

            aux_input_buffers.push(buffer);
            aux_input_storage.push(vec![
                vec![0.0; max_buffer_size];
                num_channels.get() as usize
            ]);
        }

        let mut aux_output_buffers = Vec::with_capacity(audio_io_layout.aux_output_ports.len());
        for num_channels in audio_io_layout.aux_output_ports {
            let mut buffer = Buffer::default();
            unsafe {
                buffer.set_slices(0, |slices| {
                    slices.resize_with(num_channels.get() as usize, || &mut []);
                })
            };

            aux_output_buffers.push(buffer);
        }

        let main_output_storage = vec![
            vec![0.0; max_buffer_size];
            audio_io_layout
                .main_output_channels
                .map(NonZeroU32::get)
                .unwrap_or(0) as usize
        ];
        let aux_output_storage = audio_io_layout
            .aux_output_ports
            .iter()
            .map(|channels| vec![vec![0.0; max_buffer_size]; channels.get() as usize])
            .collect();

        Self {
            max_buffer_size,
            main_input_channels: audio_io_layout
                .main_input_channels
                .map(NonZeroU32::get)
                .unwrap_or(0) as usize,
            main_output_storage,
            aux_output_storage,
            main_input_channel_pointers: None,
            main_output_channel_pointers: None,
            aux_input_channel_pointers: vec![None; audio_io_layout.aux_input_ports.len()],
            aux_output_channel_pointers: vec![None; audio_io_layout.aux_output_ports.len()],

            main_buffer,

            aux_input_buffers,
            aux_input_storage,

            aux_output_buffers,
        }
    }

    /// Initialize the buffers using the host provided buffer pointers and return a reference to the
    /// created buffers that can be passed to `Plugin::process()`. This accounts for in-place main
    /// IO, missing channel pointers, null pointers, and mismatching channel counts. All
    /// uninitialized buffer data (aux outputs, and main output channels with no matching input
    /// channel) are filled with zeroes.
    ///
    /// `sample_offset` and `num_samples` can be used to slice a set of host channel pointers for
    /// sample accurate automation. Missing outputs use independently owned discard channels.
    /// Wrappers must separately distinguish legal inactive outputs from malformed host calls.
    ///
    /// # Panics
    ///
    /// May panic if one of the inner channel pointers is a null pointer.
    ///
    /// # Safety
    ///
    /// Any provided `ChannelPointers` must point to memory regions that remain valid to read from
    /// or write to for the lifetime of the returned [`Buffers`].
    #[cfg(any(feature = "standalone", test))]
    pub unsafe fn create_buffers<'a, 'buffer: 'a>(
        &'a mut self,
        sample_offset: usize,
        num_samples: usize,
        set_buffer_sources: impl FnOnce(&mut BufferSource),
    ) -> Buffers<'a, 'buffer> {
        unsafe { self.try_create_buffers(sample_offset, num_samples, set_buffer_sources) }
            .expect("Invalid buffer window or channel pointers")
    }

    /// Fallible wrapper adapter: reject invalid windows/counts/null samples before making slices.
    /// Missing sources are allowed here; protocol wrappers decide when absence is legal.
    ///
    /// # Safety
    /// Provided pointer arrays must contain the declared entries. Non-null sample regions must
    /// be valid for the requested window and returned borrow, with mutable outputs disjoint.
    /// The adapter cannot establish the allocation length or provenance of arbitrary host pointers.
    pub unsafe fn try_create_buffers<'a, 'buffer: 'a>(
        &'a mut self,
        sample_offset: usize,
        num_samples: usize,
        set_buffer_sources: impl FnOnce(&mut BufferSource),
    ) -> Option<Buffers<'a, 'buffer>> {
        if sample_offset.checked_add(num_samples)? > self.max_buffer_size {
            return None;
        }
        // Retire references from the previous borrow before touching owned backing again.
        unsafe {
            self.main_buffer.set_slices(0, |s| s.fill_with(|| &mut []));
            for buffer in self
                .aux_input_buffers
                .iter_mut()
                .chain(self.aux_output_buffers.iter_mut())
            {
                buffer.set_slices(0, |s| s.fill_with(|| &mut []));
            }
        }
        self.main_input_channel_pointers = None;
        self.main_output_channel_pointers = None;
        self.aux_input_channel_pointers.fill(None);
        self.aux_output_channel_pointers.fill(None);
        set_buffer_sources(&mut BufferSource {
            main_input_channel_pointers: &mut self.main_input_channel_pointers,
            main_output_channel_pointers: &mut self.main_output_channel_pointers,
            aux_input_channel_pointers: &mut self.aux_input_channel_pointers,
            aux_output_channel_pointers: &mut self.aux_output_channel_pointers,
        });
        if !unsafe {
            Self::pointers_match(self.main_input_channel_pointers, self.main_input_channels)
        } || !unsafe {
            Self::pointers_match(
                self.main_output_channel_pointers,
                self.main_output_storage.len(),
            )
        } {
            return None;
        }
        for (source, storage) in self
            .aux_input_channel_pointers
            .iter()
            .zip(&self.aux_input_storage)
        {
            if !unsafe { Self::pointers_match(*source, storage.len()) } {
                return None;
            }
        }
        for (source, storage) in self
            .aux_output_channel_pointers
            .iter()
            .zip(&self.aux_output_storage)
        {
            if !unsafe { Self::pointers_match(*source, storage.len()) } {
                return None;
            }
        }

        let output_source = self.main_output_channel_pointers;
        let input_source = self.main_input_channel_pointers;
        let storage = &mut self.main_output_storage;
        unsafe {
            self.main_buffer.set_slices(num_samples, |slices| {
                for (channel, slice) in slices.iter_mut().enumerate() {
                    let output = match output_source {
                        Some(source) => *source.ptrs.as_ptr().add(channel),
                        None => storage[channel].as_mut_ptr(),
                    }
                    .add(sample_offset);
                    if let Some(input) = input_source.filter(|p| channel < p.num_channels) {
                        let input = (*input.ptrs.as_ptr().add(channel)).add(sample_offset);
                        if input != output {
                            std::ptr::copy(input, output, num_samples);
                        }
                    } else {
                        std::ptr::write_bytes(output, 0, num_samples);
                    }
                    *slice = std::slice::from_raw_parts_mut(output, num_samples);
                }
            });
        }
        for (source, (storage, buffer)) in self.aux_input_channel_pointers.iter().zip(
            self.aux_input_storage
                .iter_mut()
                .zip(self.aux_input_buffers.iter_mut()),
        ) {
            for (channel, destination) in storage.iter_mut().enumerate() {
                match source {
                    Some(source) => unsafe {
                        let input = (*source.ptrs.as_ptr().add(channel)).add(sample_offset);
                        std::ptr::copy(input, destination.as_mut_ptr(), num_samples);
                    },
                    None => destination[..num_samples].fill(0.0),
                }
            }
            unsafe {
                buffer.set_slices(num_samples, |slices| {
                    for (slice, channel) in slices.iter_mut().zip(storage.iter_mut()) {
                        *slice = std::slice::from_raw_parts_mut(channel.as_mut_ptr(), num_samples);
                    }
                });
            }
        }
        for (source, (storage, buffer)) in self.aux_output_channel_pointers.iter().zip(
            self.aux_output_storage
                .iter_mut()
                .zip(self.aux_output_buffers.iter_mut()),
        ) {
            unsafe {
                buffer.set_slices(num_samples, |slices| {
                    for (channel, slice) in slices.iter_mut().enumerate() {
                        let output = match source {
                            Some(source) => *source.ptrs.as_ptr().add(channel),
                            None => storage[channel].as_mut_ptr(),
                        }
                        .add(sample_offset);
                        std::ptr::write_bytes(output, 0, num_samples);
                        *slice = std::slice::from_raw_parts_mut(output, num_samples);
                    }
                });
            }
        }
        Some(unsafe {
            std::mem::transmute::<Buffers<'a, 'static>, Buffers<'a, 'buffer>>(Buffers {
                main_buffer: &mut self.main_buffer,
                aux_inputs: &mut self.aux_input_buffers,
                aux_outputs: &mut self.aux_output_buffers,
            })
        })
    }

    // Pointer array extent and sample-region provenance remain the unsafe caller's contract.
    unsafe fn pointers_match(source: Option<ChannelPointers>, expected: usize) -> bool {
        let Some(source) = source else {
            return true;
        };
        if source.num_channels != expected || !source.ptrs.as_ptr().is_aligned() {
            return false;
        }
        (0..expected).all(|channel| {
            let pointer = unsafe { *source.ptrs.as_ptr().add(channel) };
            !pointer.is_null() && pointer.is_aligned()
        })
    }
}

#[cfg(any(miri, test))]
mod miri {
    use nice_plug_core::audio_setup::{PortNames, new_nonzero_u32};

    use super::*;

    const BUFFER_SIZE: usize = 512;
    const NUM_MAIN_INPUT_CHANNELS: usize = 1;
    const NUM_MAIN_OUTPUT_CHANNELS: usize = 2;

    const NUM_AUX_CHANNELS: usize = 2;
    const NUM_AUX_PORTS: usize = 2;

    const AUDIO_IO_LAYOUT: AudioIOLayout = AudioIOLayout {
        main_input_channels: Some(new_nonzero_u32(NUM_MAIN_INPUT_CHANNELS as u32)),
        main_output_channels: Some(new_nonzero_u32(NUM_MAIN_OUTPUT_CHANNELS as u32)),
        aux_input_ports: &[new_nonzero_u32(NUM_AUX_CHANNELS as u32); NUM_AUX_PORTS],
        aux_output_ports: &[new_nonzero_u32(NUM_AUX_CHANNELS as u32); NUM_AUX_PORTS],
        names: PortNames::const_default(),
    };

    #[test]
    fn buffer_io() {
        // This works very similarly to the standalone CPAL and dummy backends
        let mut main_io_storage = vec![vec![0.0f32; BUFFER_SIZE]; NUM_MAIN_OUTPUT_CHANNELS];
        let mut aux_input_storage =
            vec![vec![vec![0.0f32; BUFFER_SIZE]; NUM_AUX_CHANNELS]; NUM_AUX_PORTS];
        let mut aux_output_storage =
            vec![vec![vec![0.0f32; BUFFER_SIZE]; NUM_AUX_CHANNELS]; NUM_AUX_PORTS];

        let mut main_io_channel_pointers: Vec<*mut f32> = main_io_storage
            .iter_mut()
            .map(|channel_slice| channel_slice.as_mut_ptr())
            .collect();
        let mut aux_input_channel_pointers: Vec<Vec<*mut f32>> = aux_input_storage
            .iter_mut()
            .map(|aux_input_storage| {
                aux_input_storage
                    .iter_mut()
                    .map(|channel_slice| channel_slice.as_mut_ptr())
                    .collect()
            })
            .collect();
        let mut aux_output_channel_pointers: Vec<Vec<*mut f32>> = aux_output_storage
            .iter_mut()
            .map(|aux_output_storage| {
                aux_output_storage
                    .iter_mut()
                    .map(|channel_slice| channel_slice.as_mut_ptr())
                    .collect()
            })
            .collect();

        // The actual buffer management here works the same as in the JACK backend. See that
        // implementation for more information.
        let mut buffer_manager = BufferManager::for_audio_io_layout(BUFFER_SIZE, AUDIO_IO_LAYOUT);
        let buffers = unsafe {
            buffer_manager.create_buffers(0, BUFFER_SIZE, |buffer_sources| {
                *buffer_sources.main_output_channel_pointers = Some(ChannelPointers {
                    ptrs: NonNull::new(main_io_channel_pointers.as_mut_ptr()).unwrap(),
                    num_channels: main_io_channel_pointers.len(),
                });
                *buffer_sources.main_input_channel_pointers = Some(ChannelPointers {
                    ptrs: NonNull::new(main_io_channel_pointers.as_mut_ptr()).unwrap(),
                    num_channels: NUM_MAIN_INPUT_CHANNELS.min(main_io_channel_pointers.len()),
                });

                for (input_source_channel_pointers, input_channel_pointers) in buffer_sources
                    .aux_input_channel_pointers
                    .iter_mut()
                    .zip(aux_input_channel_pointers.iter_mut())
                {
                    *input_source_channel_pointers = Some(ChannelPointers {
                        ptrs: NonNull::new(input_channel_pointers.as_mut_ptr()).unwrap(),
                        num_channels: input_channel_pointers.len(),
                    });
                }

                for (output_source_channel_pointers, output_channel_pointers) in buffer_sources
                    .aux_output_channel_pointers
                    .iter_mut()
                    .zip(aux_output_channel_pointers.iter_mut())
                {
                    *output_source_channel_pointers = Some(ChannelPointers {
                        ptrs: NonNull::new(output_channel_pointers.as_mut_ptr()).unwrap(),
                        num_channels: output_channel_pointers.len(),
                    });
                }
            })
        };

        for channel_samples in buffers
            .main_buffer
            .iter_samples()
            .chain(
                buffers
                    .aux_inputs
                    .iter_mut()
                    .flat_map(|buffer| buffer.iter_samples()),
            )
            .chain(
                buffers
                    .aux_outputs
                    .iter_mut()
                    .flat_map(|buffer| buffer.iter_samples()),
            )
        {
            for sample in channel_samples {
                *sample += 1.0;
            }
        }

        // These checks are fine due to stacked borrows even without explicitly dropping `buffers`.
        // If we were to access `buffers` again after this miri would trigger an error.
        for channel in main_io_storage
            .iter()
            .chain(aux_output_storage.iter().flat_map(|storage| storage.iter()))
        {
            for sample in channel {
                assert!(*sample == 1.0);
            }
        }

        for channel in aux_input_storage.iter().flat_map(|storage| storage.iter()) {
            for sample in channel {
                assert!(*sample == 0.0);
            }
        }
    }
}

#[cfg(test)]
#[path = "buffer_management/codex_admission_regression.rs"]
mod codex_admission_regression;
