use nice_plug_core::params::{
    FloatParam, InternalParamMut, Param,
    range::{DecibelZeroRange, FloatRange, PositiveLogRange},
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    sync::{Arc, atomic::{AtomicUsize, Ordering}},
};

struct CountingAllocator;
thread_local! { static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) }; }
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|c| { if let Some(n) = c.get() { c.set(Some(n + 1)); } });
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) { unsafe { System.dealloc(ptr, layout) } }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATIONS.with(|c| { if let Some(n) = c.get() { c.set(Some(n + 1)); } });
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn log_range(low: f32, high: f32) -> FloatRange {
    FloatRange::Logarithmic(PositiveLogRange::new(low, high).unwrap())
}
fn db_range() -> FloatRange {
    FloatRange::DecibelsWithZero(DecibelZeroRange::new(-60.0, 6.0).unwrap())
}
fn reversed(range: FloatRange) -> FloatRange {
    FloatRange::Reversed(Box::leak(Box::new(range)))
}
fn close(actual: f32, expected: f32) {
    let ulp = (f32::from_bits(expected.to_bits() + 1) - expected).abs();
    let tolerance = (4.0 * ulp).max(expected.abs() * 1e-6);
    assert!(actual.is_finite() && (actual - expected).abs() <= tolerance,
        "actual {actual:?} ({:08x}), expected {expected:?} ({:08x}), tolerance {tolerance}", actual.to_bits(), expected.to_bits());
}

#[test]
fn exact_range_metadata_validation() {
    fn traits<T: Copy + Send + Sync>() {}
    traits::<PositiveLogRange>(); traits::<DecibelZeroRange>(); traits::<FloatRange>();
    for (lo, hi) in [(0.0, 1.0), (-1.0, 1.0), (1.0, 1.0), (2.0, 1.0),
        (f32::NAN, 1.0), (1.0, f32::NAN), (1.0, f32::INFINITY), (f32::NEG_INFINITY, 1.0)] {
        assert!(PositiveLogRange::new(lo, hi).is_none(), "{lo}..{hi}");
    }
    for (lo, hi) in [(0.0, 0.0), (1.0, 0.0), (f32::NAN, 6.0), (-60.0, f32::INFINITY),
        (-1000.0, 6.0), (-60.0, 1000.0), (0.0, f32::from_bits(1)),
        (f32::NEG_INFINITY, 6.0), (-60.0, f32::NAN)] {
        assert!(DecibelZeroRange::new(lo, hi).is_none(), "{lo}..{hi}");
    }
    let extremes = log_range(f32::from_bits(1), f32::MAX);
    assert_eq!(extremes.unnormalize(0.0).to_bits(), 1);
    assert_eq!(extremes.unnormalize(1.0).to_bits(), f32::MAX.to_bits());
    assert!(extremes.unnormalize(0.5).is_finite());
    let adjacent = log_range(1.0, f32::from_bits(1.0f32.to_bits() + 1));
    assert_eq!(adjacent.normalize(1.0), 0.0);
    assert_eq!(adjacent.normalize(adjacent.unnormalize(1.0)), 1.0);
    eprintln!("layout range={} float_param={} incremental_219={}",
        size_of::<FloatRange>(), size_of::<FloatParam>(),
        (size_of::<FloatParam>() as isize - 128) * 219);
}

fn check_oracle(kind: &str) {
    let mut rows = 0;
    let mut domain_count = 0;
    let mut previous = None;
    for line in include_str!("p3_tapers_oracle.csv").lines().filter(|l| !l.starts_with('#')) {
        let parts: Vec<_> = line.split(',').collect();
        if parts[0] != kind { continue; }
        let v: Vec<_> = parts[1..].iter().map(|v| f32::from_bits(u32::from_str_radix(v, 16).unwrap())).collect();
        let range = if kind == "log" { log_range(v[0], v[1]) } else { db_range() };
        close(range.unnormalize(v[2]), v[3]);
        assert!((range.normalize(v[3]) - v[4]).abs() <= 2e-6, "{line}");
        let domain = (v[0].to_bits(), v[1].to_bits());
        if previous != Some(domain) { domain_count += 1; previous = Some(domain); }
        rows += 1;
    }
    assert_eq!(domain_count, if kind == "log" { 8 } else { 1 });
    assert!(rows >= domain_count * 257);
    eprintln!("{kind}: {rows} independent Decimal vectors across {domain_count} domains");
}
#[test]
fn positive_log_range_matches_oracle() { check_oracle("log"); }
#[test]
fn silent_db_range_matches_approved_taper() {
    check_oracle("db");
    let range = db_range();
    assert_eq!(range.unnormalize(0.0).to_bits(), 0);
    assert_eq!(range.unnormalize(1.0).to_bits(), 0x3fff64c1);
    assert!((range.unnormalize(10.0 / 11.0) - 1.0).abs() <= 1e-6);
    assert_eq!(range.normalize(0.0005), 0.0);
    assert_eq!(range.normalize(0.0), 0.0);
    close(range.unnormalize(f32::from_bits(1)), 0.001);
}
#[test]
fn new_range_exception_guards_before_math() {
    for range in [log_range(20.0, 400.0), db_range()] {
        let lo = range.unnormalize(0.0); let hi = range.unnormalize(1.0);
        for n in [f32::NAN, f32::NEG_INFINITY, -1.0, -0.0] {
            assert_eq!(range.unnormalize(n).to_bits(), lo.to_bits());
        }
        for n in [1.0, 2.0, f32::INFINITY] { assert_eq!(range.unnormalize(n).to_bits(), hi.to_bits()); }
        for a in [f32::NAN, f32::NEG_INFINITY, -1.0, 0.0, lo] { assert_eq!(range.normalize(a), 0.0); }
        for a in [hi, f32::MAX, f32::INFINITY] { assert_eq!(range.normalize(a), 1.0); }
        let rev = reversed(range);
        assert_eq!(rev.normalize(f32::NAN), 1.0);
        assert_eq!(rev.normalize(f32::NEG_INFINITY), 1.0);
        assert_eq!(rev.normalize(f32::INFINITY), 0.0);
        assert_eq!(rev.unnormalize(f32::NAN), lo);
        assert_eq!(rev.unnormalize(f32::NEG_INFINITY), hi);
        assert_eq!(rev.unnormalize(f32::INFINITY), lo);
        assert_eq!(rev.unnormalize(0.0), hi);
        assert_eq!(rev.unnormalize(1.0), lo);
    }
}
#[test]
fn range_steps_and_text_share_mapping() {
    for range in [log_range(20.0, 400.0), log_range(0.3, 5.0), db_range()] {
        let lo = range.unnormalize(0.0); let hi = range.unnormalize(1.0);
        let plain = range.unnormalize(0.5);
        for (finer, delta) in [(false, 0.02), (true, 0.005)] {
            for direction in [-1.0, 1.0] {
                let expected = range.unnormalize(0.5 + direction * delta);
                let step = if direction < 0.0 { range.previous_step(plain, None, finer) } else { range.next_step(plain, None, finer) };
                close(step, expected);
                for plain_step in [0.0001, 0.1, 1000.0] {
                    let expected = if (expected - plain).abs() > plain_step {
                        ((expected / plain_step).round() * plain_step).clamp(lo, hi)
                    } else { (plain + direction * plain_step).clamp(lo, hi) };
                    let step = if direction < 0.0 { range.previous_step(plain, Some(plain_step), finer) } else { range.next_step(plain, Some(plain_step), finer) };
                    close(step, expected);
                }
            }
        }
        assert_eq!(range.previous_step(lo, None, false), lo);
        assert_eq!(range.next_step(hi, None, false), hi);
        let rev = reversed(range);
        assert_eq!(rev.previous_step(plain, None, false), range.next_step(plain, None, false));
        assert_eq!(rev.next_step(plain, None, true), range.previous_step(plain, None, true));
        assert_eq!(rev.snap_to_step(plain, 0.01), range.snap_to_step(plain, 0.01));
        let p = FloatParam::new("value", plain, range).with_unit(" units");
        let text = p.normalized_value_to_string(0.5, true);
        assert!((p.string_to_normalized_value(&text).unwrap() - 0.5).abs() <= 2e-6);
        assert_eq!(p.default_plain_value(), plain);
        unsafe { p._internal_set_normalized_value(0.8); p._internal_set_plain_value(p.default_plain_value()); }
        assert_eq!(p.unmodulated_plain_value(), plain);
    }
}
#[test]
fn plain_edit_survives_saturated_modulation() {
    for range in [FloatRange::Linear { min: 0.0, max: 1.0 }, db_range(), log_range(20.0, 400.0)] {
        for offset in [-1.0, 1.0] {
            let calls = Arc::new(AtomicUsize::new(0)); let observed = calls.clone();
            let p = FloatParam::new("base", range.unnormalize(0.5), range)
                .with_callback(Arc::new(move |_| { observed.fetch_add(1, Ordering::Relaxed); }));
            assert!(unsafe { p._internal_modulate_value(offset) });
            for n in [0.25, 0.75] {
                let plain = range.unnormalize(n);
                assert!(!unsafe { p._internal_set_plain_value(plain) });
                assert_eq!(p.unmodulated_plain_value(), plain);
                assert_eq!(p.unmodulated_normalized_value(), range.normalize(plain));
                assert_eq!(calls.load(Ordering::Relaxed), 1);
            }
            assert!(unsafe { p._internal_modulate_value(0.0) });
            assert_eq!(p.value(), range.unnormalize(0.75));
            assert_eq!(calls.load(Ordering::Relaxed), 2);
        }
    }
    let p = FloatParam::new("floor", 0.0005, db_range());
    assert!(unsafe { p._internal_modulate_value(-1.0) });
    assert!(!unsafe { p._internal_set_plain_value(0.0004) });
    assert_eq!(p.unmodulated_plain_value(), 0.0004);
    assert_eq!(p.unmodulated_normalized_value(), 0.0);
    assert!(unsafe { p._internal_modulate_value(0.0) });
    assert_eq!(p.value(), 0.0004);
    assert!(unsafe { p._internal_set_normalized_value(0.0) });
    assert_eq!(p.value().to_bits(), 0);
}
#[test]
fn taper_conversion_no_alloc() {
    let ranges = [log_range(20.0, 400.0), db_range()];
    let p = FloatParam::new("allocated before guard", 0.5, db_range());
    ALLOCATIONS.with(|c| c.set(Some(0)));
    for i in 0..10000 {
        let n = std::hint::black_box((i % 1001) as f32 / 1000.0);
        for r in ranges {
            let plain = std::hint::black_box(r.unnormalize(n));
            std::hint::black_box(r.normalize(plain));
            std::hint::black_box(r.previous_step(plain, None, false));
            std::hint::black_box(r.next_step(plain, Some(0.01), true));
            std::hint::black_box(r.snap_to_step(plain, 0.01));
        }
        unsafe { p._internal_set_normalized_value(n); p._internal_modulate_value(n - 0.5); }
    }
    let count = ALLOCATIONS.with(|c| c.replace(None)).unwrap();
    assert_eq!(count, 0);
}
