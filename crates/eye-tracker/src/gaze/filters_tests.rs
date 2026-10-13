// Tests for gaze::filters (One Euro filter), ported from tests/test_filters.py.
// Not ported: nothing. The random streams come from testing::gaze_synth::Rng, not numpy.

use approx::assert_relative_eq;

use super::*;
use crate::testing::gaze_synth::Rng;

fn run(filt: &mut OneEuroFilter, values: &[f64], fps: f64, t0: f64) -> Vec<f64> {
    values
        .iter()
        .enumerate()
        .map(|(i, &v)| filt.update(v, t0 + i as f64 / fps))
        .collect()
}

fn std(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt()
}

// --------------------------------------------------------------------------- OneEuroFilter
#[test]
fn test_first_sample_passes_through_and_constant_stays_constant() {
    let mut f = OneEuroFilter::new(1.0, 0.01, 1.0).unwrap();
    assert_eq!(f.value(), None);
    assert_eq!(f.update(42.0, 10.0), 42.0);
    let out = run(&mut f, &[42.0; 30], 30.0, 10.1);
    assert!(out.iter().all(|v| (v - 42.0).abs() < 1e-8));
    assert_relative_eq!(f.value().unwrap(), 42.0);
}

#[test]
fn test_jitter_is_reduced() {
    let mut rng = Rng::new(0);
    let noisy: Vec<f64> = rng
        .normal_vec(0.0, 20.0, 300)
        .iter()
        .map(|v| 500.0 + v)
        .collect();
    let mut f = OneEuroFilter::new(1.0, 0.0, 1.0).unwrap();
    let out = run(&mut f, &noisy, 30.0, 0.0);
    assert!(std(&out[30..]) < std(&noisy) / 3.0);
}

#[test]
fn test_beta_speeds_up_response_to_fast_moves() {
    let mut step = vec![0.0; 10];
    step.extend([1000.0; 20]);
    let slow = run(
        &mut OneEuroFilter::new(0.5, 0.0, 1.0).unwrap(),
        &step,
        15.0,
        0.0,
    );
    let fast = run(
        &mut OneEuroFilter::new(0.5, 0.01, 1.0).unwrap(),
        &step,
        15.0,
        0.0,
    );
    // Three frames after the jump the adaptive filter is much closer to the target.
    assert!(fast[12] > 900.0);
    assert!(slow[12] < 700.0);
}

#[test]
fn test_filter_is_time_aware() {
    // The same step sampled at 5 fps and 30 fps must be at a similar place after 1 s.
    fn after_one_second(fps: f64) -> f64 {
        let mut f = OneEuroFilter::new(1.0, 0.0, 1.0).unwrap();
        f.update(0.0, 0.0);
        let mut value = 0.0;
        for i in 1..=(fps as usize) {
            value = f.update(100.0, i as f64 / fps);
        }
        value
    }

    let (low, high) = (after_one_second(5.0), after_one_second(30.0));
    assert!((low - high).abs() < 10.0);
    assert!(50.0 < low && low < 100.0);
}

#[test]
fn test_non_advancing_timestamp_is_ignored() {
    let mut f = OneEuroFilter::new(1.0, 0.0, 1.0).unwrap();
    f.update(0.0, 1.0);
    let moved = f.update(100.0, 1.1);
    assert_eq!(f.update(5000.0, 1.1), moved); // duplicate timestamp
    assert_eq!(f.update(5000.0, 1.0), moved); // going back in time
    assert!(f.update(100.0, 1.2) > moved); // continues normally afterwards
}

#[test]
fn test_non_finite_input_holds_last_value() {
    let mut f = OneEuroFilter::default();
    assert!(f.update(f64::NAN, 0.0).is_nan()); // nothing to hold yet
    f.update(10.0, 0.1);
    assert_eq!(f.update(f64::NAN, 0.2), 10.0);
    assert_eq!(f.update(f64::INFINITY, 0.3), 10.0);
    assert_relative_eq!(f.update(10.0, 0.4), 10.0);
}

#[test]
fn test_reset_forgets_state() {
    let mut f = OneEuroFilter::default();
    f.update(0.0, 0.0);
    f.update(10.0, 0.1);
    f.reset();
    assert_eq!(f.value(), None);
    assert_eq!(f.update(77.0, 0.2), 77.0);
}

#[test]
fn test_invalid_parameters() {
    for (min_cutoff, beta, d_cutoff) in [(0.0, 0.0, 1.0), (1.0, 0.0, -1.0), (1.0, -0.1, 1.0)] {
        let err = OneEuroFilter::new(min_cutoff, beta, d_cutoff).unwrap_err();
        assert!(err.to_string().contains("min_cutoff and d_cutoff"));
    }
}

// --------------------------------------------------------------------------- PointFilter
#[test]
fn test_zero_smoothing_is_passthrough() {
    let mut pf = PointFilter::new(0.0);
    let mut rng = Rng::new(1);
    for i in 0..50 {
        let (x, y) = (rng.uniform(0.0, 3840.0), rng.uniform(0.0, 3840.0));
        assert_eq!(pf.update(x, y, i as f64 * 0.05), (x, y));
    }
}

#[test]
fn test_more_smoothing_means_less_jitter() {
    let mut rng = Rng::new(2);
    let noise: Vec<(f64, f64)> = (0..200)
        .map(|_| (rng.normal(960.0, 40.0), rng.normal(540.0, 40.0)))
        .collect();

    let jitter = |smoothing: f64| -> f64 {
        let mut pf = PointFilter::new(smoothing);
        let out: Vec<f64> = noise
            .iter()
            .enumerate()
            .map(|(i, &(x, y))| pf.update(x, y, i as f64 / 12.0).0)
            .collect();
        std(&out[20..])
    };

    let raw = std(&noise[20..].iter().map(|p| p.0).collect::<Vec<_>>());
    let (light, medium, heavy) = (jitter(0.1), jitter(0.5), jitter(1.0));
    assert!(raw > light && light > medium && medium > heavy);
    assert!(heavy < raw / 2.0);
}

#[test]
fn test_monitor_jump_is_followed_quickly_at_default_smoothing() {
    // A glance from the left to the right monitor at the balanced active rate (12 fps).
    let mut pf = PointFilter::new(0.5);
    let mut t = 0.0;
    for _ in 0..12 {
        pf.update(960.0, 540.0, t);
        t += 1.0 / 12.0;
    }
    let mut xs = Vec::new();
    for _ in 0..4 {
        xs.push(pf.update(2880.0, 540.0, t).0);
        t += 1.0 / 12.0;
    }
    assert!(xs[2] > 1920.0 + 0.8 * 960.0); // well inside the right monitor within 3 frames
}

#[test]
fn test_set_smoothing_clamps_and_keeps_state() {
    let mut pf = PointFilter::new(0.5);
    pf.update(100.0, 100.0, 0.0);
    pf.set_smoothing(7.0);
    assert_eq!(pf.smoothing(), 1.0);
    pf.set_smoothing(-3.0);
    assert_eq!(pf.smoothing(), 0.0);
    pf.set_smoothing(f64::NAN);
    assert_eq!(pf.smoothing(), 0.0);
    pf.set_smoothing(0.8);
    let (x, _) = pf.update(110.0, 100.0, 0.1);
    assert!(100.0 < x && x < 110.0); // continued from the earlier position, not from scratch
}

#[test]
fn test_point_filter_reset() {
    let mut pf = PointFilter::new(1.0);
    pf.update(0.0, 0.0, 0.0);
    pf.update(1000.0, 1000.0, 0.1);
    pf.reset();
    assert_eq!(pf.update(1000.0, 1000.0, 0.2), (1000.0, 1000.0));
}
