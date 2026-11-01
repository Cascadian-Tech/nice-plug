use nice_plug_core::params::{FloatParam, range::FloatRange};

#[test]
fn old_range_behaviors_unchanged() {
    let cases = [
        FloatRange::Linear {
            min: -1.0,
            max: 1.0,
        },
        FloatRange::Skewed {
            min: 0.0,
            max: 3.9810717,
            factor: 3.9810717f32.log2().recip(),
        },
        FloatRange::gain_range(-24.0, 12.0),
        FloatRange::Linear {
            min: -1.0,
            max: 1.0,
        },
        FloatRange::SymmetricalSkewed {
            min: -26.0,
            max: 6.0,
            factor: 1.0,
            center: 0.0,
        },
        FloatRange::Skewed {
            min: 0.0,
            max: 1.0,
            factor: 0.5,
        },
    ];
    let mut hash = 0xcbf29ce484222325u64;
    for range in cases {
        let reversed = FloatRange::Reversed(Box::leak(Box::new(range)));
        for range in [range, reversed] {
            for i in 0..=1000 {
                let n = i as f32 / 1000.0;
                let plain = range.unnormalize(n);
                for value in [
                    plain,
                    range.normalize(plain),
                    range.previous_step(plain, None, false),
                    range.next_step(plain, None, true),
                    range.previous_step(plain, Some(0.1), true),
                    range.next_step(plain, Some(0.1), false),
                    range.snap_to_step(plain, 0.25),
                ] {
                    hash ^= value.to_bits() as u64;
                    hash = hash.wrapping_mul(0x100000001b3);
                }
            }
            assert!(range.normalize(f32::NAN).is_nan());
            assert!(range.unnormalize(f32::NAN).is_nan());
        }
    }
    eprintln!(
        "OLD_RANGE_HASH={hash:016x}; range_bytes={}; param_bytes={}",
        std::mem::size_of::<FloatRange>(),
        std::mem::size_of::<FloatParam>()
    );
    assert_eq!(
        hash, 0xf10f50c76f39126c,
        "baseline mapping/stepping bits changed"
    );
}
