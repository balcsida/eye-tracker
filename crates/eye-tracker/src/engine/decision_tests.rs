// Scenario tests for the monitor switching decision (engine/decision.rs).
//
// The Python helpers took keyword arguments; here `step`/`run` are the variants without
// input activity and `step_act`/`run_act` take the mouse and key times.

use super::*;
use crate::types::Rect;

const NEVER: f64 = f64::NEG_INFINITY;

type Gaze = Option<(f64, f64)>;

fn mon(index: i32, x: i32, y: i32, w: i32, h: i32) -> Monitor {
    Monitor {
        primary: index == 0,
        ..Monitor::new(index, format!("M{index}"), Rect::new(x, y, w, h))
    }
}

fn hd(index: i32, x: i32, y: i32) -> Monitor {
    mon(index, x, y, 1920, 1080)
}

/// Two 1080p monitors side by side: 0 = left, 1 = right.
fn dual() -> Vec<Monitor> {
    vec![hd(0, 0, 0), hd(1, 1920, 0)]
}

/// Three in a row with the primary in the middle.
fn triple() -> Vec<Monitor> {
    vec![hd(0, -1920, 0), hd(1, 0, 0), hd(2, 1920, 0)]
}

/// 1 on top of 0.
fn stacked() -> Vec<Monitor> {
    vec![hd(0, 0, 0), hd(1, 0, -1080)]
}

/// A 1440p external monitor with a 1080p laptop panel centred below it.
fn laptop_below() -> Vec<Monitor> {
    vec![mon(0, 0, 0, 2560, 1440), hd(1, 320, 1440)]
}

const LEFT: (f64, f64) = (900.0, 500.0);
const RIGHT: (f64, f64) = (2900.0, 500.0);

fn decider() -> SwitchDecider {
    decider_with(&dual(), SwitchConfig::default())
}

fn decider_with(monitors: &[Monitor], config: SwitchConfig) -> SwitchDecider {
    SwitchDecider::new(monitors, config)
}

fn decider_cfg(config: SwitchConfig) -> SwitchDecider {
    decider_with(&dual(), config)
}

fn step_act(
    d: &mut SwitchDecider,
    now: f64,
    gaze: impl Into<Gaze>,
    current: Option<i32>,
    mouse: f64,
    key: f64,
) -> Decision {
    d.update(now, gaze.into(), current, mouse, key, true, false)
}

fn step(d: &mut SwitchDecider, now: f64, gaze: impl Into<Gaze>, current: Option<i32>) -> Decision {
    step_act(d, now, gaze, current, NEVER, NEVER)
}

fn run_act(
    d: &mut SwitchDecider,
    times: &[f64],
    gaze: (f64, f64),
    current: Option<i32>,
    mouse: f64,
    key: f64,
) -> Vec<Decision> {
    times
        .iter()
        .map(|&t| step_act(d, t, Some(gaze), current, mouse, key))
        .collect()
}

fn run(
    d: &mut SwitchDecider,
    times: &[f64],
    gaze: (f64, f64),
    current: Option<i32>,
) -> Vec<Decision> {
    run_act(d, times, gaze, current, NEVER, NEVER)
}

fn round6(x: f64) -> f64 {
    (x * 1e6).round_ties_even() / 1e6
}

fn ticks(start: f64, stop: f64, dt: f64) -> Vec<f64> {
    let n = ((stop - start) / dt).round_ties_even() as i64;
    (0..=n).map(|i| round6(start + i as f64 * dt)).collect()
}

fn ticks01(start: f64, stop: f64) -> Vec<f64> {
    ticks(start, stop, 0.1)
}

fn first_switch(decisions: &[Decision], times: &[f64]) -> Option<f64> {
    assert_eq!(decisions.len(), times.len());
    times
        .iter()
        .zip(decisions)
        .find(|(_, dec)| dec.fired())
        .map(|(t, _)| *t)
}

fn reasons(out: &[Decision]) -> Vec<&'static str> {
    out.iter().map(|o| o.reason).collect()
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-6,
        "{actual} != approx({expected})"
    );
}

// --------------------------------------------------------------------- basics
#[test]
fn test_looking_at_current_monitor_does_nothing() {
    let mut d = decider();
    for dec in run(&mut d, &ticks01(0.0, 2.0), LEFT, Some(0)) {
        assert_eq!(dec.reason, "same");
        assert_eq!(dec.target, None);
        assert_eq!(dec.candidate, Some(0));
        assert!(!dec.pending);
    }
}

#[test]
fn test_switch_fires_after_dwell_with_rising_progress() {
    let mut d = decider();
    let out = run(&mut d, &[0.0, 0.1, 0.2, 0.3], RIGHT, Some(0));
    assert_eq!(reasons(&out), ["dwell", "dwell", "dwell", "switch"]);
    let progress: Vec<f64> = out
        .iter()
        .map(|o| (o.progress * 1000.0).round_ties_even() / 1000.0)
        .collect();
    assert_eq!(progress, [0.0, 0.333, 0.667, 1.0]);
    assert!(out[..3].iter().all(|o| o.pending));
    assert_eq!(out[3].target, Some(1));
    assert_eq!(out[3].candidate, Some(1));
    assert!(!out[3].pending);
}

#[test]
fn test_dwell_zero_switches_immediately() {
    let mut d = decider_cfg(SwitchConfig {
        dwell_s: 0.0,
        ..Default::default()
    });
    let dec = step(&mut d, 5.0, RIGHT, Some(0));
    assert_eq!(dec.reason, "switch");
    assert_eq!(dec.target, Some(1));
    assert_eq!(dec.progress, 1.0);
}

#[test]
fn test_all_reasons_are_documented() {
    let mut reasons = REASONS.to_vec();
    reasons.sort_unstable();
    let mut expected = vec![
        "switch",
        "same",
        "dwell",
        "no_gaze",
        "off_screen",
        "mouse",
        "typing",
        "cooldown",
        "hysteresis",
        "disabled",
    ];
    expected.sort_unstable();
    assert_eq!(reasons, expected);
}

// ------------------------------------------------------------ dwell resetting
#[test]
fn test_missing_gaze_resets_dwell() {
    let mut d = decider();
    run(&mut d, &[0.0, 0.1, 0.2], RIGHT, Some(0));
    assert_eq!(step(&mut d, 0.25, None, Some(0)).reason, "no_gaze");
    // Dwell starts from scratch: 0.3 s after 0.3, not after 0.0.
    let out = run(&mut d, &[0.3, 0.4, 0.5, 0.6], RIGHT, Some(0));
    assert_eq!(reasons(&out), ["dwell", "dwell", "dwell", "switch"]);
}

#[test]
fn test_nan_gaze_is_treated_as_no_gaze() {
    let mut d = decider();
    let dec = step(&mut d, 0.0, Some((f64::NAN, 10.0)), Some(0));
    assert_eq!(dec.reason, "no_gaze");
    assert_eq!(dec.candidate, None);
}

#[test]
fn test_gap_between_updates_restarts_dwell() {
    let mut d = decider_cfg(SwitchConfig {
        max_gap_s: 0.75,
        ..Default::default()
    });
    step(&mut d, 0.0, RIGHT, Some(0));
    step(&mut d, 0.2, RIGHT, Some(0));
    // 1 s without any update: nothing is known about that time.
    let late = step(&mut d, 1.2, RIGHT, Some(0));
    assert_eq!(late.reason, "dwell");
    assert_eq!(late.progress, 0.0);
    assert_eq!(step(&mut d, 1.4, RIGHT, Some(0)).reason, "dwell");
    assert_eq!(step(&mut d, 1.5, RIGHT, Some(0)).reason, "switch");
}

#[test]
fn test_gap_below_limit_keeps_dwell() {
    let mut d = decider_cfg(SwitchConfig {
        dwell_s: 1.0,
        max_gap_s: 0.75,
        ..Default::default()
    });
    let out = run(&mut d, &[0.0, 0.7, 1.0], RIGHT, Some(0));
    assert_eq!(reasons(&out), ["dwell", "dwell", "switch"]);
}

#[test]
fn test_same_monitor_glance_resets_dwell() {
    let mut d = decider();
    run(&mut d, &[0.0, 0.1, 0.2], RIGHT, Some(0));
    assert_eq!(step(&mut d, 0.25, LEFT, Some(0)).reason, "same");
    assert_eq!(step(&mut d, 0.3, RIGHT, Some(0)).reason, "dwell");
    assert_eq!(step(&mut d, 0.5, RIGHT, Some(0)).reason, "dwell");
    assert!(step(&mut d, 0.6, RIGHT, Some(0)).fired());
}

// ------------------------------------------------------------------ hysteresis
#[test]
fn test_gaze_just_across_the_bezel_is_held_by_hysteresis() {
    let mut d = decider();
    // 20 px into the right monitor; threshold is 0.06 * 1080 = 64.8 px.
    let near_bezel = (1940.0, 500.0);
    let out = run(&mut d, &ticks01(0.0, 3.0), near_bezel, Some(0));
    assert!(out.iter().all(|o| o.reason == "hysteresis"));
    assert!(
        out.iter()
            .all(|o| o.candidate == Some(1) && !o.fired() && !o.pending)
    );
}

#[test]
fn test_gaze_clearly_across_the_bezel_passes_hysteresis() {
    let mut d = decider();
    let past = (1920.0 + 70.0, 500.0);
    let out = run(&mut d, &[0.0, 0.3], past, Some(0));
    assert_eq!(reasons(&out), ["dwell", "switch"]);
}

#[test]
fn test_hysteresis_flicker_at_bezel_resets_dwell() {
    let mut d = decider();
    step(&mut d, 0.0, RIGHT, Some(0));
    step(&mut d, 0.1, RIGHT, Some(0));
    assert_eq!(
        step(&mut d, 0.2, (1930.0, 500.0), Some(0)).reason,
        "hysteresis"
    );
    assert_eq!(step(&mut d, 0.3, RIGHT, Some(0)).reason, "dwell");
    assert!(step(&mut d, 0.6, RIGHT, Some(0)).fired());
}

#[test]
fn test_hysteresis_is_symmetric_for_switching_back() {
    let mut d = decider();
    let near_bezel_left = (1900.0, 500.0); // 20 px into the left monitor
    assert_eq!(
        step(&mut d, 0.0, near_bezel_left, Some(1)).reason,
        "hysteresis"
    );
    assert_eq!(step(&mut d, 0.1, (1800.0, 500.0), Some(1)).reason, "dwell");
}

#[test]
fn test_hysteresis_zero_switches_right_at_the_edge() {
    let mut d = decider_cfg(SwitchConfig {
        hysteresis: 0.0,
        ..Default::default()
    });
    let out = run(&mut d, &[0.0, 0.3], (1921.0, 500.0), Some(0));
    assert!(out.last().unwrap().fired());
}

#[test]
fn test_unknown_current_monitor_skips_hysteresis() {
    let mut d = decider();
    let near_bezel = (1925.0, 500.0);
    let out = run(&mut d, &[0.0, 0.3], near_bezel, None);
    assert_eq!(reasons(&out), ["dwell", "switch"]);
    assert_eq!(out.last().unwrap().target, Some(1));
}

#[test]
fn test_stale_current_index_is_treated_as_unknown() {
    let mut d = decider();
    let out = run(&mut d, &[0.0, 0.3], LEFT, Some(7));
    assert_eq!(out.last().unwrap().reason, "switch");
    assert_eq!(out.last().unwrap().target, Some(0));
}

// ------------------------------------------------------------ off-screen gaze
#[test]
fn test_glance_at_phone_below_monitors_is_ignored() {
    let mut d = decider();
    // Far below the right monitor: > 0.35 * 2203 px (its diagonal) outside.
    let phone = (2900.0, 1080.0 + 900.0);
    let out = run(&mut d, &ticks01(0.0, 2.0), phone, Some(0));
    assert!(
        out.iter()
            .all(|o| o.reason == "off_screen" && o.candidate.is_none())
    );
}

#[test]
fn test_glance_at_phone_mid_dwell_resets_dwell() {
    let mut d = decider();
    run(&mut d, &[0.0, 0.1, 0.2], RIGHT, Some(0));
    assert_eq!(
        step(&mut d, 0.25, (2900.0, 3000.0), Some(0)).reason,
        "off_screen"
    );
    let back = run(&mut d, &[0.3, 0.4, 0.5, 0.6], RIGHT, Some(0));
    assert_eq!(reasons(&back), ["dwell", "dwell", "dwell", "switch"]);
}

#[test]
fn test_gaze_slightly_outside_is_attributed_to_nearest_monitor() {
    let mut d = decider();
    let beyond_right_edge = (3840.0 + 150.0, 500.0);
    let out = run(&mut d, &[0.0, 0.3], beyond_right_edge, Some(0));
    assert_eq!(out[0].candidate, Some(1));
    assert_eq!(out.last().unwrap().target, Some(1));
}

#[test]
fn test_off_screen_near_current_monitor_is_same() {
    let mut d = decider();
    let above_left = (900.0, -200.0);
    assert_eq!(step(&mut d, 0.0, above_left, Some(0)).reason, "same");
}

#[test]
fn test_off_screen_margin_scales_with_diagonal() {
    let mut d = decider_cfg(SwitchConfig {
        off_screen_margin: 0.1,
        ..Default::default()
    });
    // 300 px below: beyond 0.1 * 2203 = 220 px, so off screen.
    assert_eq!(
        step(&mut d, 0.0, (2900.0, 1380.0), Some(0)).reason,
        "off_screen"
    );
    d.set_config(SwitchConfig {
        off_screen_margin: 0.35,
        ..Default::default()
    });
    assert_eq!(step(&mut d, 0.1, (2900.0, 1380.0), Some(0)).reason, "dwell");
}

#[test]
fn test_off_screen_gaze_just_across_the_seam_is_held_by_hysteresis() {
    // Looking down at the keyboard (or up) centred under the seam: 5 px past the
    // seam is no more a reason to switch than 5 px past the bezel on screen.
    for gy in [1300.0, -220.0] {
        let mut d = decider();
        let out = run(&mut d, &ticks01(0.0, 2.0), (1925.0, gy), Some(0));
        assert!(
            out.iter()
                .all(|o| o.reason == "hysteresis" && o.candidate == Some(1))
        );
        let back = run(&mut d, &ticks01(2.1, 4.0), (1915.0, gy), Some(1));
        assert!(
            back.iter()
                .all(|o| o.reason == "hysteresis" && o.candidate == Some(0))
        );
    }
}

#[test]
fn test_off_screen_hysteresis_grows_with_distance_below_the_screens() {
    let mut d = decider();
    // 220 px below the screens the gaze must favour the right monitor by 64.8 px:
    // hypot(dx, 220) - 220 >= 64.8 needs dx >= ~180 px past the seam.
    assert_eq!(
        step(&mut d, 0.0, (2000.0, 1300.0), Some(0)).reason,
        "hysteresis"
    );
    let out = run(&mut d, &[0.1, 0.4], (2150.0, 1300.0), Some(0));
    assert_eq!(reasons(&out), ["dwell", "switch"]);
}

#[test]
fn test_on_screen_hysteresis_is_unchanged_by_the_off_screen_rule() {
    let mut d = decider();
    assert_eq!(
        step(&mut d, 0.0, (1925.0, 540.0), Some(0)).reason,
        "hysteresis"
    );
    assert_eq!(step(&mut d, 0.1, (1990.0, 540.0), Some(0)).reason, "dwell");
}

/// Python's `random.Random(1234).gauss`: the Mersenne Twister stream cannot be reproduced
/// without a dependency, so a splitmix64 + Box-Muller generator with a fixed seed is used.
struct Gauss(u64);

impl Gauss {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn gauss(&mut self, mu: f64, sigma: f64) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        mu + sigma * (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

#[test]
fn test_jitter_around_the_seam_never_flaps() {
    for gy in [540.0, 1150.0, 1250.0, 1500.0, 1800.0] {
        let mut rng = Gauss(1234);
        let mut d = decider();
        let mut current = 0;
        let mut switches = 0;
        for i in 0..600 {
            // 60 s at 10 fps
            let t = f64::from(i) * 0.1;
            let gaze = (1920.0 + rng.gauss(0.0, 40.0), gy);
            let dec = step(&mut d, t, Some(gaze), Some(current));
            if dec.fired() {
                switches += 1;
                current = dec.target.unwrap();
                d.notify_switched(t, current);
            }
        }
        assert_eq!(switches, 0, "gy = {gy}");
    }
}

// ------------------------------------------------------------- looking away
#[test]
fn test_looking_away_is_off_screen_and_resets_dwell() {
    let mut d = decider();
    run(&mut d, &[0.0, 0.1, 0.2], RIGHT, Some(0));
    let away = d.update(0.25, Some(RIGHT), Some(0), NEVER, NEVER, true, true);
    assert_eq!(away.reason, "off_screen");
    assert_eq!(away.candidate, None);
    assert!(!away.fired());
    assert!(!away.pending);
    let back = run(&mut d, &[0.3, 0.4, 0.5, 0.6], RIGHT, Some(0));
    assert_eq!(reasons(&back), ["dwell", "dwell", "dwell", "switch"]);
}

#[test]
fn test_looking_away_without_gaze_is_off_screen() {
    let mut d = decider();
    assert_eq!(
        d.update(0.0, None, Some(0), NEVER, NEVER, true, true)
            .reason,
        "off_screen"
    );
}

#[test]
fn test_disabled_takes_precedence_over_looking_away() {
    let mut d = decider();
    let dec = d.update(0.0, Some(RIGHT), Some(0), NEVER, NEVER, false, true);
    assert_eq!(dec.reason, "disabled");
}

// ------------------------------------------------------------------- layouts
#[test]
fn test_three_monitor_row_switch_to_far_right() {
    let mut d = decider_with(&triple(), SwitchConfig::default());
    let out = run(&mut d, &[0.0, 0.1, 0.2, 0.3], (2900.0, 500.0), Some(0));
    assert_eq!(out.last().unwrap().target, Some(2));
}

#[test]
fn test_three_monitor_sweep_restarts_dwell_per_candidate() {
    let mut d = decider_with(&triple(), SwitchConfig::default());
    // The eyes sweep from the left monitor over the middle one to the right one.
    assert_eq!(
        step(&mut d, 0.0, (900.0, 500.0), Some(0)).candidate,
        Some(1)
    );
    assert_eq!(step(&mut d, 0.1, (900.0, 500.0), Some(0)).reason, "dwell");
    assert_eq!(step(&mut d, 0.2, (2900.0, 500.0), Some(0)).progress, 0.0);
    assert_eq!(step(&mut d, 0.3, (2900.0, 500.0), Some(0)).reason, "dwell");
    assert_eq!(step(&mut d, 0.4, (2900.0, 500.0), Some(0)).reason, "dwell");
    let fin = step(&mut d, 0.5, (2900.0, 500.0), Some(0));
    assert_eq!(fin.reason, "switch");
    assert_eq!(fin.target, Some(2));
}

#[test]
fn test_three_monitor_middle_to_left_and_back() {
    let mut d = decider_with(&triple(), SwitchConfig::default());
    let out = run(&mut d, &[0.0, 0.3], (-1000.0, 500.0), Some(1));
    assert_eq!(out.last().unwrap().target, Some(0));
    d.notify_switched(0.3, 0);
    let out = run(&mut d, &[1.0, 1.3], (1000.0, 500.0), Some(0));
    assert_eq!(out.last().unwrap().target, Some(1));
}

#[test]
fn test_stacked_monitors_switch_up_and_down() {
    let mut d = decider_with(&stacked(), SwitchConfig::default());
    let up = run(&mut d, &[0.0, 0.3], (960.0, -500.0), Some(0));
    assert_eq!(up.last().unwrap().target, Some(1));
    d.notify_switched(0.3, 1);
    let down = run(&mut d, &[1.0, 1.3], (960.0, 600.0), Some(1));
    assert_eq!(down.last().unwrap().target, Some(0));
}

#[test]
fn test_stacked_monitors_vertical_hysteresis() {
    let mut d = decider_with(&stacked(), SwitchConfig::default());
    // 30 px above the shared edge: less than 0.06 * 1080 = 64.8 px.
    assert_eq!(
        step(&mut d, 0.0, (960.0, -30.0), Some(0)).reason,
        "hysteresis"
    );
    assert_eq!(step(&mut d, 0.1, (960.0, -70.0), Some(0)).reason, "dwell");
}

#[test]
fn test_laptop_below_layout() {
    let mut d = decider_with(&laptop_below(), SwitchConfig::default());
    // Looking down at the laptop panel.
    assert_eq!(
        step(&mut d, 0.0, (1200.0, 1500.0), Some(0)).reason,
        "hysteresis"
    ); // 60 px < 64.8
    let out = run(&mut d, &[0.1, 0.4], (1200.0, 1900.0), Some(0));
    assert_eq!(out.last().unwrap().target, Some(1));
    d.notify_switched(0.4, 1);
    // Back up to the external monitor.
    let out = run(&mut d, &[2.0, 2.3], (1200.0, 700.0), Some(1));
    assert_eq!(out.last().unwrap().target, Some(0));
}

#[test]
fn test_laptop_below_empty_corner_maps_to_nearest_monitor() {
    let mut d = decider_with(&laptop_below(), SwitchConfig::default());
    // Below the external monitor but left of the laptop panel: nearer the external one.
    assert_eq!(step(&mut d, 0.0, (100.0, 1500.0), Some(0)).reason, "same");
    // Same point while the cursor is on the laptop: gaze favours the external monitor.
    let dec = step(&mut d, 0.1, (100.0, 1520.0), Some(1));
    assert_eq!(dec.candidate, Some(0));
}

#[test]
fn test_overlapping_monitors_prefer_current() {
    let mirrored = [hd(0, 0, 0), hd(1, 0, 0)];
    let mut d = decider_with(&mirrored, SwitchConfig::default());
    assert_eq!(step(&mut d, 0.0, (500.0, 500.0), Some(1)).reason, "same");
    assert_eq!(step(&mut d, 0.1, (500.0, 500.0), Some(0)).reason, "same");
}

// ---------------------------------------------------------------------- guards
#[test]
fn test_mouse_guard_blocks_then_fires_as_soon_as_it_expires() {
    let mut d = decider();
    let mouse = 0.1; // user moved the mouse; grace 1.5 s -> blocked until 1.6
    let times = ticks01(0.0, 2.0);
    let out = run_act(&mut d, &times, RIGHT, Some(0), mouse, NEVER);
    let r = reasons(&out);
    assert_eq!(r[..3], ["dwell"; 3]);
    assert!(r[3..16].iter().all(|&x| x == "mouse"));
    assert_close(first_switch(&out, &times).unwrap(), 1.6);
    // The dwell kept accumulating while guarded: progress stays at 1.
    assert!(out[3..16].iter().all(|o| o.progress == 1.0));
}

#[test]
fn test_typing_guard_blocks_then_fires_as_soon_as_it_expires() {
    let mut d = decider();
    let times = ticks01(0.0, 3.0);
    let out = run_act(&mut d, &times, RIGHT, Some(0), NEVER, 0.5);
    assert!(out[3..25].iter().all(|o| o.reason == "typing"));
    assert_close(first_switch(&out, &times).unwrap(), 2.5);
}

#[test]
fn test_guard_expiry_after_user_looked_away_does_not_fire() {
    let mut d = decider();
    run_act(&mut d, &ticks01(0.0, 1.0), RIGHT, Some(0), NEVER, 0.5);
    // The user looked back at their own monitor before the typing grace ended...
    run_act(&mut d, &ticks01(1.1, 2.4), LEFT, Some(0), NEVER, 0.5);
    // ...so a glance at the other monitor right at expiry needs a fresh dwell.
    let out = run_act(&mut d, &[2.5, 2.6, 2.8], RIGHT, Some(0), NEVER, 0.5);
    assert_eq!(reasons(&out), ["dwell", "dwell", "switch"]);
    assert_eq!(out[0].progress, 0.0);
}

#[test]
fn test_mouse_guard_reported_before_typing_guard() {
    let mut d = decider_cfg(SwitchConfig {
        dwell_s: 0.0,
        ..Default::default()
    });
    assert_eq!(
        step_act(&mut d, 1.0, RIGHT, Some(0), 0.9, 0.9).reason,
        "mouse"
    );
    assert_eq!(
        step_act(&mut d, 2.5, RIGHT, Some(0), 0.9, 0.9).reason,
        "typing"
    );
    assert!(step_act(&mut d, 3.0, RIGHT, Some(0), 0.9, 0.9).fired());
}

#[test]
fn test_pending_only_when_switch_could_fire_soon() {
    let mut d = decider();
    // Typing just now: blocked for 2 s, far beyond the dwell -> keep the slow rate.
    let typing = step_act(&mut d, 10.0, RIGHT, Some(0), NEVER, 10.0);
    assert_eq!(typing.reason, "dwell");
    assert!(!typing.pending);
    let blocked = run_act(&mut d, &[10.1, 10.2, 10.3], RIGHT, Some(0), NEVER, 10.0)
        .pop()
        .unwrap();
    assert_eq!(blocked.reason, "typing");
    assert!(!blocked.pending);
    // Close to expiry the scheduler should speed up.
    let later = run_act(&mut d, &ticks01(10.4, 11.4), RIGHT, Some(0), NEVER, 10.0);
    assert!(!later.iter().any(|o| o.pending));
    let soon = step_act(
        &mut d,
        12.0 - PENDING_HORIZON_S + 0.1,
        RIGHT,
        Some(0),
        NEVER,
        10.0,
    );
    assert_eq!(soon.reason, "typing");
    assert!(soon.pending);
}

#[test]
fn test_cooldown_prevents_ping_pong() {
    let mut d = decider_cfg(SwitchConfig {
        cooldown_s: 1.0,
        ..Default::default()
    });
    let out = run(&mut d, &[0.0, 0.3], RIGHT, Some(0));
    assert_eq!(out.last().unwrap().target, Some(1));
    d.notify_switched(0.3, 1);
    // Noisy gaze snaps back to the left monitor right after the switch.
    let back = run(&mut d, &ticks01(0.4, 1.2), LEFT, Some(1));
    let r = reasons(&back);
    assert!(!back.iter().any(|o| o.fired()));
    assert_eq!(r[..3], ["dwell"; 3]);
    assert!(r[3..].iter().all(|&x| x == "cooldown"));
    // Sampling speeds up only once the cooldown is about to end.
    assert!(!back[3].pending);
    assert!(back.last().unwrap().pending);
    assert!(step(&mut d, 1.3, LEFT, Some(1)).fired());
}

#[test]
fn test_single_frame_blip_never_switches() {
    let mut d = decider();
    for i in 0..20 {
        let t = f64::from(i) * 0.1;
        let gaze = if i % 3 == 0 { RIGHT } else { LEFT };
        assert!(!step(&mut d, t, Some(gaze), Some(0)).fired());
    }
}

#[test]
fn test_failed_switch_is_not_retried_every_frame() {
    let mut d = decider();
    let out = run(&mut d, &[0.0, 0.3], RIGHT, Some(0));
    assert!(out.last().unwrap().fired());
    // The caller could not move the cursor (no notify_switched, cursor still on 0).
    let retry = run(&mut d, &ticks01(0.4, 0.8), RIGHT, Some(0));
    assert!(!retry.iter().any(|o| o.fired()));
    assert_eq!(retry.last().unwrap().reason, "cooldown");
    assert!(step(&mut d, 0.9, RIGHT, Some(0)).fired());
}

#[test]
fn test_switch_from_no_monitor_after_dwell() {
    let mut d = decider();
    let out = run(&mut d, &[0.0, 0.1, 0.2, 0.3], LEFT, None);
    assert_eq!(reasons(&out), ["dwell", "dwell", "dwell", "switch"]);
    assert_eq!(out.last().unwrap().target, Some(0));
}

// ------------------------------------------------------ reading while typing
type KeyClock = Box<dyn Fn(f64) -> f64>;

/// Last keystroke time at `t` for key presses every 0.25 s during `bursts`.
fn keystrokes(bursts: &[(f64, f64)]) -> KeyClock {
    let every = 0.25;
    let bursts = bursts.to_vec();
    Box::new(move |t| {
        let mut best = NEVER;
        for &(start, stop) in &bursts {
            if t >= start {
                let n = ((t.min(stop) - start) / every + 1e-9).floor();
                best = best.max(start + n * every);
            }
        }
        best
    })
}

fn frames(start: f64, stop: f64, gaze: (f64, f64), dt: f64) -> Vec<(f64, Gaze)> {
    ticks(start, stop, dt)
        .into_iter()
        .map(|t| (t, Some(gaze)))
        .collect()
}

/// Run a script of `(time, gaze)` frames; the cursor follows every switch.
fn play(
    d: &mut SwitchDecider,
    script: &[(f64, Gaze)],
    mut current: i32,
    key: &KeyClock,
) -> Vec<(f64, Decision)> {
    let mut out = Vec::new();
    for &(t, gaze) in script {
        let dec = step_act(d, t, gaze, Some(current), NEVER, key(t));
        if let Some(target) = dec.target {
            d.notify_switched(t, target);
            current = target;
        }
        out.push((t, dec));
    }
    out
}

fn switch_times(out: &[(f64, Decision)]) -> Vec<f64> {
    out.iter()
        .filter(|(_, dec)| dec.fired())
        .map(|(t, _)| *t)
        .collect()
}

fn assert_times(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
    for (a, e) in actual.iter().zip(expected) {
        assert_close(*a, *e);
    }
}

#[test]
fn test_reading_one_monitor_while_typing_on_the_other_keeps_focus() {
    // Transcribing from a document on the left into an editor on the right: the
    // gaze stays on the left, keystrokes until t=8, then a reading pause.
    let mut d = decider();
    let key = keystrokes(&[(0.0, 8.0)]);
    let typing_phase = play(&mut d, &frames(0.0, 10.0, LEFT, 0.5), 1, &key);
    assert_times(&switch_times(&typing_phase), &[]);
    assert_eq!(d.reading_monitor(), Some(0));
    let pause = play(&mut d, &frames(10.25, 16.0, LEFT, 0.25), 1, &key);
    // The plain typing guard would have switched at t=10 (2 s after the last
    // key); a reading monitor needs reading_grace_s (6 s) of pause.
    assert_times(&switch_times(&pause), &[14.0]);
    let blocked: Vec<&Decision> = pause
        .iter()
        .filter(|(t, _)| *t < 14.0 - 1e-9)
        .map(|(_, dec)| dec)
        .collect();
    assert!(blocked.iter().all(|dec| dec.reason == "typing"));
    // The camera stays at the slow rate until the switch is about to happen.
    assert!(
        !pause
            .iter()
            .any(|(t, dec)| dec.pending && *t < 14.0 - PENDING_HORIZON_S - 1e-9)
    );
    assert!(
        pause
            .iter()
            .any(|(t, dec)| dec.pending && 14.0 - PENDING_HORIZON_S <= *t && *t < 14.0)
    );
}

#[test]
fn test_glancing_back_at_the_editor_keeps_the_reading_monitor() {
    let mut d = decider();
    let key = keystrokes(&[(0.0, 6.0)]);
    let script = [
        frames(0.0, 4.5, LEFT, 0.5),  // reading the source while typing
        frames(5.0, 6.0, RIGHT, 0.5), // looks at the editor to fix a typo
        frames(6.5, 13.0, LEFT, 0.5), // back to the source, reading pause
    ]
    .concat();
    let out = play(&mut d, &script, 1, &key);
    assert_times(&switch_times(&out), &[12.0]);
}

#[test]
fn test_finishing_a_word_while_looking_over_uses_the_typing_grace() {
    // The eyes move to the right monitor 0.2 s before the last keystroke: that is
    // a user about to work there, not someone reading while typing.
    let mut d = decider();
    let key = keystrokes(&[(0.0, 3.0)]);
    let script = [frames(0.0, 2.6, LEFT, 0.2), frames(2.8, 6.0, RIGHT, 0.2)].concat();
    let out = play(&mut d, &script, 0, &key);
    assert_eq!(d.reading_monitor(), None);
    assert_times(&switch_times(&out), &[5.0]);
}

#[test]
fn test_typing_resumed_right_after_a_look_makes_it_a_reading_monitor() {
    let mut d = decider();
    let key = keystrokes(&[(0.0, 2.0), (3.6, 6.0)]);
    let script = [
        frames(0.0, 2.0, LEFT, 0.5),
        frames(2.5, 3.0, RIGHT, 0.5), // reads something on the right, no switch yet
        frames(3.5, 6.0, LEFT, 0.5),  // back to the editor and typing again
    ]
    .concat();
    let out = play(&mut d, &script, 0, &key);
    assert_times(&switch_times(&out), &[]);
    assert_eq!(d.reading_monitor(), Some(1));
    let later = play(&mut d, &frames(6.5, 13.0, RIGHT, 0.25), 0, &key);
    assert_times(&switch_times(&later), &[12.0]);
}

#[test]
fn test_a_single_noisy_sample_does_not_make_a_reading_monitor() {
    let mut d = decider();
    let key = keystrokes(&[(0.0, 10.0)]);
    let script: Vec<(f64, Gaze)> = ticks(0.0, 10.0, 0.5)
        .into_iter()
        .map(|t| (t, Some(if t % 2.0 == 1.0 { RIGHT } else { LEFT })))
        .collect();
    let out = play(&mut d, &script, 0, &key);
    assert_times(&switch_times(&out), &[]);
    assert_eq!(d.reading_monitor(), None);
    let later = play(&mut d, &frames(10.5, 13.0, RIGHT, 0.25), 0, &key);
    assert_times(&switch_times(&later), &[12.0]);
}

#[test]
fn test_reading_is_recognised_at_the_slowest_typing_frame_rate() {
    // Eco profile: 1 fps while typing. Gaps above max_gap_s restart the dwell,
    // but must not hide that the user reads the other monitor.
    let mut d = decider();
    let key = keystrokes(&[(0.0, 8.0)]);
    let out = play(&mut d, &frames(0.0, 8.0, LEFT, 1.0), 1, &key);
    assert_times(&switch_times(&out), &[]);
    assert_eq!(d.reading_monitor(), Some(0));
    let pause = play(&mut d, &frames(8.5, 15.0, LEFT, 0.5), 1, &key);
    assert_times(&switch_times(&pause), &[14.0]);
}

#[test]
fn test_reading_monitor_is_forgotten_when_no_longer_looked_at() {
    let mut d = decider();
    let key = keystrokes(&[(0.0, 70.0)]);
    play(&mut d, &frames(0.0, 5.0, RIGHT, 0.5), 0, &key);
    assert_eq!(d.reading_monitor(), Some(1));
    // Keeps typing, now looking at the editor on the left only.
    play(
        &mut d,
        &frames(5.5, 5.0 + READING_FORGET_S - 0.5, LEFT, 0.5),
        0,
        &key,
    );
    assert_eq!(d.reading_monitor(), Some(1));
    play(
        &mut d,
        &frames(5.0 + READING_FORGET_S + 0.5, 70.0, LEFT, 0.5),
        0,
        &key,
    );
    assert_eq!(d.reading_monitor(), None);
    let out = play(&mut d, &frames(70.5, 73.0, RIGHT, 0.25), 0, &key);
    assert_times(&switch_times(&out), &[72.0]);
}

#[test]
fn test_reading_monitor_ends_with_a_long_typing_pause() {
    let mut d = decider();
    let key = keystrokes(&[(0.0, 5.0)]);
    play(&mut d, &frames(0.0, 5.0, LEFT, 0.5), 1, &key);
    assert_eq!(d.reading_monitor(), Some(0));
    play(&mut d, &frames(5.5, 11.5, RIGHT, 0.5), 1, &key); // reviews the text
    assert_eq!(d.reading_monitor(), None);
    let out = play(&mut d, &frames(12.0, 13.0, LEFT, 0.25), 1, &key);
    assert_times(&switch_times(&out), &[12.5]); // an ordinary dwell
}

#[test]
fn test_reading_monitor_is_dropped_when_the_cursor_changes_monitor() {
    let mut d = decider();
    let key = keystrokes(&[(0.0, 5.0)]);
    play(&mut d, &frames(0.0, 5.0, LEFT, 0.5), 1, &key);
    assert_eq!(d.reading_monitor(), Some(0));
    step_act(&mut d, 5.5, LEFT, Some(0), NEVER, key(5.5)); // the user took the cursor there
    assert_eq!(d.reading_monitor(), None);
    play(&mut d, &frames(0.0, 5.0, LEFT, 0.5), 1, &key);
    d.notify_switched(5.0, 0);
    assert_eq!(d.reading_monitor(), None);
}

#[test]
fn test_reading_rule_can_be_turned_off() {
    // not above typing_grace_s: plain typing guard
    let mut d = decider_cfg(SwitchConfig {
        reading_grace_s: 2.0,
        ..Default::default()
    });
    let key = keystrokes(&[(0.0, 8.0)]);
    let out = play(&mut d, &frames(0.0, 12.0, LEFT, 0.5), 1, &key);
    assert_eq!(d.reading_monitor(), None);
    assert_times(&switch_times(&out), &[10.0]);
}

#[test]
fn test_reading_monitor_does_not_delay_switches_to_other_monitors() {
    let mut d = decider_with(&triple(), SwitchConfig::default());
    let (left, right) = ((-1000.0, 500.0), (2900.0, 500.0));
    let key = keystrokes(&[(0.0, 5.0)]);
    play(&mut d, &frames(0.0, 5.0, left, 0.5), 1, &key);
    assert_eq!(d.reading_monitor(), Some(0));
    let out = play(&mut d, &frames(5.5, 8.0, right, 0.25), 1, &key);
    assert_times(&switch_times(&out), &[7.0]);
}

#[test]
fn test_clock_going_backwards_forgets_the_reading_monitor() {
    let mut d = decider();
    play(
        &mut d,
        &frames(0.0, 5.0, LEFT, 0.5),
        1,
        &keystrokes(&[(0.0, 5.0)]),
    );
    assert_eq!(d.reading_monitor(), Some(0));
    step_act(&mut d, 1.0, LEFT, Some(1), NEVER, 0.9);
    assert_eq!(d.reading_monitor(), None);
}

#[test]
fn test_gaze_held_at_the_bezel_is_no_look_at_the_other_monitor() {
    let mut d = decider();
    let near_bezel = (1900.0, 500.0); // 20 px into the left monitor, cursor on the right
    let out = play(
        &mut d,
        &frames(0.0, 5.0, near_bezel, 0.5),
        1,
        &keystrokes(&[(0.0, 5.0)]),
    );
    assert!(out.iter().all(|(_, dec)| dec.reason == "hysteresis"));
    assert_eq!(d.reading_monitor(), None);
}

#[test]
fn test_reset_forgets_the_reading_monitor() {
    let mut d = decider();
    play(
        &mut d,
        &frames(0.0, 5.0, LEFT, 0.5),
        1,
        &keystrokes(&[(0.0, 5.0)]),
    );
    assert_eq!(d.reading_monitor(), Some(0));
    d.reset();
    assert_eq!(d.reading_monitor(), None);
    play(
        &mut d,
        &frames(0.0, 5.0, LEFT, 0.5),
        1,
        &keystrokes(&[(0.0, 5.0)]),
    );
    d.set_monitors(&dual());
    assert_eq!(d.reading_monitor(), None);
}

// ------------------------------------------------------------ configuration
#[test]
fn test_disabled_resets_dwell() {
    let mut d = decider();
    run(&mut d, &[0.0, 0.1, 0.2], RIGHT, Some(0));
    let off = d.update(0.25, Some(RIGHT), Some(0), NEVER, NEVER, false, false);
    assert_eq!(off.reason, "disabled");
    assert_eq!(off.target, None);
    assert!(!off.pending);
    assert_eq!(step(&mut d, 0.3, RIGHT, Some(0)).progress, 0.0);
}

#[test]
fn test_no_monitors_is_disabled() {
    let mut d = decider_with(&[], SwitchConfig::default());
    assert_eq!(step(&mut d, 0.0, RIGHT, None).reason, "disabled");
}

#[test]
fn test_set_monitors_discards_dwell() {
    let mut d = decider();
    run(&mut d, &[0.0, 0.1, 0.2], RIGHT, Some(0));
    d.set_monitors(&dual());
    assert_eq!(step(&mut d, 0.3, RIGHT, Some(0)).reason, "dwell");
}

#[test]
fn test_reset_clears_cooldown() {
    let mut d = decider_cfg(SwitchConfig {
        cooldown_s: 5.0,
        ..Default::default()
    });
    run(&mut d, &[0.0, 0.3], RIGHT, Some(0));
    d.notify_switched(0.3, 1);
    assert_eq!(d.last_switch_target(), Some(1));
    d.reset();
    assert_eq!(d.last_switch_time(), f64::NEG_INFINITY);
    let out = run(&mut d, &[0.4, 0.7], LEFT, Some(1));
    assert!(out.last().unwrap().fired());
}

#[test]
fn test_set_config_takes_effect_immediately() {
    let mut d = decider_cfg(SwitchConfig {
        dwell_s: 2.0,
        ..Default::default()
    });
    step(&mut d, 0.0, RIGHT, Some(0));
    assert_eq!(step(&mut d, 0.5, RIGHT, Some(0)).reason, "dwell");
    d.set_config(SwitchConfig {
        dwell_s: 0.5,
        ..Default::default()
    });
    assert!(step(&mut d, 0.6, RIGHT, Some(0)).fired());
}

#[test]
fn test_config_from_settings_converts_milliseconds() {
    let s = SwitchingSettings {
        dwell_ms: 450,
        hysteresis: 0.1,
        off_screen_margin: 0.5,
        cooldown_ms: 800,
        mouse_grace_ms: 1000,
        typing_grace_ms: 2500,
        ..Default::default()
    };
    let cfg = SwitchConfig::from_settings(&s);
    assert_eq!(
        cfg,
        SwitchConfig {
            dwell_s: 0.45,
            hysteresis: 0.1,
            off_screen_margin: 0.5,
            cooldown_s: 0.8,
            mouse_grace_s: 1.0,
            typing_grace_s: 2.5,
            max_gap_s: SwitchConfig::default().max_gap_s,
            reading_grace_s: SwitchConfig::default().reading_grace_s,
        }
    );
}

#[test]
fn test_config_from_settings_reads_the_reading_grace() {
    let s = SwitchingSettings {
        reading_grace_ms: 9000,
        ..Default::default()
    };
    assert_close(SwitchConfig::from_settings(&s).reading_grace_s, 9.0);
}

#[test]
fn test_defaults_match_settings_defaults() {
    assert_eq!(
        SwitchConfig::from_settings(&SwitchingSettings::default()),
        SwitchConfig::default()
    );
}
