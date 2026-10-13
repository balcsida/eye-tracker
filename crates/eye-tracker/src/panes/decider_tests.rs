// Ported from tests/test_panes_decider.py (scenario tests for the split-pane focus decision).

use approx::assert_relative_eq;

use super::*;

const NEVER: f64 = f64::NEG_INFINITY;
const WINDOW: Rect = Rect::new(0, 0, 1920, 1080);
const LEFT: Rect = Rect::new(0, 0, 960, 1080);
const RIGHT: Rect = Rect::new(960, 0, 960, 1080);
const SIGMA: (f64, f64) = (100.0, 80.0);
const ON_LEFT: (f64, f64) = (400.0, 500.0);
const ON_RIGHT: (f64, f64) = (1500.0, 500.0);

fn snap_of(now: f64, rects: &[Rect], focused: usize, handle: isize) -> PaneSnapshot {
    let panes = rects
        .iter()
        .enumerate()
        .map(|(i, r)| Pane::new(format!("%{i}"), *r, i == focused, "tmux"))
        .collect();
    PaneSnapshot {
        window_handle: WindowHandle::Hwnd(handle),
        panes,
        taken_at: now,
    }
}

fn snap(now: f64) -> PaneSnapshot {
    snap_of(now, &[LEFT, RIGHT], 0, 7)
}

fn ticks(start: f64, stop: f64, dt: f64) -> Vec<f64> {
    let n = ((stop - start) / dt).round_ties_even() as usize;
    (0..=n)
        .map(|i| ((start + i as f64 * dt) * 1e6).round_ties_even() / 1e6)
        .collect()
}

fn ticks_tenth(start: f64, stop: f64) -> Vec<f64> {
    ticks(start, stop, 0.1)
}

/// The keyword arguments of the Python `run` helper.
struct Run {
    rects: Vec<Rect>,
    focused: usize,
    sigma: Option<(f64, f64)>,
    mouse: f64,
    key: f64,
    window: Option<Rect>,
    snapshot_age: f64,
}

impl Default for Run {
    fn default() -> Self {
        Self {
            rects: vec![LEFT, RIGHT],
            focused: 0,
            sigma: Some(SIGMA),
            mouse: NEVER,
            key: NEVER,
            window: Some(WINDOW),
            snapshot_age: 0.0,
        }
    }
}

fn run(
    d: &mut PaneDecider,
    times: &[f64],
    gaze: Option<(f64, f64)>,
    options: &Run,
) -> Vec<PaneDecision> {
    times
        .iter()
        .map(|&t| {
            let s = snap_of(t - options.snapshot_age, &options.rects, options.focused, 7);
            d.update(
                t,
                gaze,
                options.window,
                Some(&s),
                options.sigma,
                options.mouse,
                options.key,
                true,
            )
        })
        .collect()
}

fn run_default(d: &mut PaneDecider, times: &[f64], gaze: (f64, f64)) -> Vec<PaneDecision> {
    run(d, times, Some(gaze), &Run::default())
}

/// `d.update(t, gaze, WINDOW, snapshot, SIGMA, NEVER, NEVER)`.
fn update(
    d: &mut PaneDecider,
    t: f64,
    gaze: Option<(f64, f64)>,
    snapshot: Option<&PaneSnapshot>,
) -> PaneDecision {
    d.update(
        t,
        gaze,
        Some(WINDOW),
        snapshot,
        Some(SIGMA),
        NEVER,
        NEVER,
        true,
    )
}

fn first_fire(decisions: &[PaneDecision], times: &[f64]) -> Option<f64> {
    assert_eq!(decisions.len(), times.len());
    times
        .iter()
        .zip(decisions)
        .find(|(_, dec)| dec.fired())
        .map(|(t, _)| *t)
}

fn approx(actual: f64, expected: f64) {
    assert_relative_eq!(actual, expected, max_relative = 1e-6, epsilon = 1e-12);
}

fn id(text: &str) -> PaneId {
    PaneId::from(text)
}

// ----------------------------------------------------------------- geometry
#[test]
fn test_separation_axis() {
    assert_eq!(separation_axis(&LEFT, &RIGHT), 0);
    let top = Rect::new(0, 0, 1920, 540);
    let bottom = Rect::new(0, 540, 1920, 540);
    assert_eq!(separation_axis(&top, &bottom), 1);
    // A one-cell border between panes is still "side by side".
    assert_eq!(
        separation_axis(&Rect::new(0, 0, 952, 1080), &Rect::new(960, 0, 960, 1080)),
        0
    );
    // Diagonal neighbours: the larger centre offset decides.
    assert_eq!(
        separation_axis(&Rect::new(0, 0, 400, 400), &Rect::new(1400, 500, 400, 400)),
        0
    );
    assert_eq!(
        separation_axis(&Rect::new(0, 0, 400, 400), &Rect::new(500, 1400, 400, 400)),
        1
    );
}

#[test]
fn test_divider_margin() {
    approx(divider_margin((1010.0, 0.0), &LEFT, &RIGHT, 0), 50.0);
    approx(divider_margin((900.0, 0.0), &RIGHT, &LEFT, 0), 60.0);
    let gap_left = Rect::new(0, 0, 952, 100);
    let gap_right = Rect::new(960, 0, 960, 100);
    approx(
        divider_margin((1006.0, 0.0), &gap_left, &gap_right, 0),
        50.0,
    );
}

// ------------------------------------------------------------------- switching
#[test]
fn test_switch_after_the_dwell() {
    let mut d = PaneDecider::default();
    let times = ticks_tenth(0.0, 1.0);
    let decisions = run_default(&mut d, &times, ON_RIGHT);
    approx(first_fire(&decisions, &times).unwrap(), 0.6);
    let fired = decisions.iter().find(|dec| dec.fired()).unwrap();
    assert_eq!(fired.target.as_ref().unwrap().id, id("%1"));
    assert_eq!(fired.reason, "switch");
    assert_eq!(decisions[0].reason, "dwell");
    assert!(decisions[0].pending);
    assert!(
        decisions
            .iter()
            .all(|dec| PANE_REASONS.contains(&dec.reason))
    );
}

#[test]
fn test_dwell_is_a_majority_vote() {
    let mut d = PaneDecider::default();
    run_default(&mut d, &ticks_tenth(0.0, 2.0), ON_LEFT); // working in the focused pane
    let times = ticks_tenth(2.1, 3.0);
    let mut gazes = vec![ON_RIGHT; times.len()];
    gazes[3] = ON_LEFT; // one stray sample
    let decisions: Vec<PaneDecision> = times
        .iter()
        .zip(&gazes)
        .map(|(&t, &g)| update(&mut d, t, Some(g), Some(&snap(t))))
        .collect();
    let fired_at = first_fire(&decisions, &times).unwrap();
    assert!((2.6..=2.9).contains(&fired_at)); // not restarted by the stray sample
}

#[test]
fn test_scattered_gaze_never_switches() {
    let mut d = PaneDecider::default();
    let times = ticks_tenth(0.0, 5.0);
    let decisions: Vec<PaneDecision> = times
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            let gaze = if i % 2 == 1 { ON_RIGHT } else { ON_LEFT };
            update(&mut d, t, Some(gaze), Some(&snap(t)))
        })
        .collect();
    assert_eq!(first_fire(&decisions, &times), None);
}

fn reasons(decisions: &[PaneDecision]) -> std::collections::BTreeSet<&'static str> {
    decisions.iter().map(|dec| dec.reason).collect()
}

#[test]
fn test_small_pane_is_never_a_target() {
    let narrow = vec![Rect::new(0, 0, 1680, 1080), Rect::new(1680, 0, 240, 1080)];
    let mut d = PaneDecider::default();
    let decisions = run(
        &mut d,
        &ticks_tenth(0.0, 3.0),
        Some((1800.0, 500.0)),
        &Run {
            rects: narrow,
            ..Run::default()
        },
    );
    assert_eq!(reasons(&decisions), ["small"].into());
    // A worse calibration makes even half a screen too small.
    let mut d = PaneDecider::default();
    let decisions = run(
        &mut d,
        &ticks_tenth(0.0, 3.0),
        Some(ON_RIGHT),
        &Run {
            sigma: Some((400.0, 80.0)),
            ..Run::default()
        },
    );
    assert_eq!(reasons(&decisions), ["small"].into());
    // min_pane_px applies with a perfect calibration too.
    let mut d = PaneDecider::new(PaneConfig {
        min_pane_px: 1000.0,
        ..PaneConfig::default()
    });
    let decisions = run(
        &mut d,
        &[0.0],
        Some(ON_RIGHT),
        &Run {
            sigma: Some((1.0, 1.0)),
            ..Run::default()
        },
    );
    assert_eq!(decisions[0].reason, "small");
}

#[test]
fn test_stacked_panes_use_the_vertical_error() {
    let stacked = vec![Rect::new(0, 0, 1920, 540), Rect::new(0, 540, 1920, 540)];
    let below = Some((900.0, 800.0));
    let mut d = PaneDecider::default();
    let decisions = run(
        &mut d,
        &[0.0],
        below,
        &Run {
            rects: stacked.clone(),
            sigma: Some((10.0, 300.0)),
            ..Run::default()
        },
    );
    assert_eq!(decisions[0].reason, "small");
    let mut d = PaneDecider::default();
    let times = ticks_tenth(0.0, 1.0);
    let decisions = run(
        &mut d,
        &times,
        below,
        &Run {
            rects: stacked,
            sigma: Some((900.0, 80.0)),
            ..Run::default()
        },
    );
    assert!(first_fire(&decisions, &times).is_some());
}

#[test]
fn test_gaze_near_the_divider_is_held() {
    let mut d = PaneDecider::default();
    let times = ticks_tenth(0.0, 2.0);
    let decisions = run_default(&mut d, &times, (1000.0, 500.0)); // 40 px past it; σx 100 needs 50
    assert_eq!(reasons(&decisions), ["hysteresis"].into());
    let decisions = run_default(&mut PaneDecider::default(), &times, (1015.0, 500.0));
    assert!(first_fire(&decisions, &times).is_some());
}

#[test]
fn test_nothing_happens_without_a_fresh_snapshot_or_outside_the_window() {
    let mut d = PaneDecider::default();
    let stale = run(
        &mut d,
        &ticks_tenth(0.0, 1.0),
        Some(ON_RIGHT),
        &Run {
            snapshot_age: 2.0,
            ..Run::default()
        },
    );
    assert_eq!(stale.last().unwrap().reason, "stale");
    assert_eq!(update(&mut d, 0.0, Some(ON_RIGHT), None).reason, "no_panes");
    let mut d = PaneDecider::default();
    let outside = run(
        &mut d,
        &[0.0],
        Some((2500.0, 500.0)),
        &Run {
            window: Some(WINDOW),
            ..Run::default()
        },
    );
    assert_eq!(outside[0].reason, "outside");
    let gap = vec![Rect::new(0, 0, 950, 1080), Rect::new(970, 0, 950, 1080)];
    let on_gap = run(
        &mut d,
        &[0.1],
        Some((960.0, 500.0)),
        &Run {
            rects: gap,
            ..Run::default()
        },
    );
    assert_eq!(on_gap[0].reason, "no_pane");
    assert_eq!(run_default(&mut d, &[0.2], ON_LEFT)[0].reason, "same");
    assert_eq!(
        run(&mut d, &[0.3], None, &Run::default())[0].reason,
        "no_gaze"
    );
}

#[test]
fn test_disabled_clears_the_dwell() {
    let mut d = PaneDecider::default();
    run_default(&mut d, &ticks_tenth(0.0, 0.5), ON_RIGHT);
    let off = d.update(
        0.6,
        Some(ON_RIGHT),
        Some(WINDOW),
        Some(&snap(0.6)),
        Some(SIGMA),
        NEVER,
        NEVER,
        false,
    );
    assert_eq!(off.reason, "disabled");
    let times = ticks_tenth(0.7, 1.4);
    let decisions = run_default(&mut d, &times, ON_RIGHT);
    approx(first_fire(&decisions, &times).unwrap(), 1.3);
}

#[test]
fn test_a_long_gap_restarts_the_dwell() {
    let mut d = PaneDecider::default();
    run_default(&mut d, &ticks_tenth(0.0, 0.5), ON_RIGHT);
    let times = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1];
    let decisions = run_default(&mut d, &times, ON_RIGHT);
    approx(first_fire(&decisions, &times).unwrap(), 2.1);
}

#[test]
fn test_fallback_error_without_calibration_accuracy() {
    let mut d = PaneDecider::default();
    let times = ticks_tenth(0.0, 1.0);
    let decisions = run(
        &mut d,
        &times,
        Some(ON_RIGHT),
        &Run {
            sigma: None,
            ..Run::default()
        },
    );
    assert!(first_fire(&decisions, &times).is_some());
    assert!(FALLBACK_GAZE_ERROR_PX * PaneConfig::default().precision < 960.0);
}

// ----------------------------------------------------------------------- guards
#[test]
fn test_typing_mouse_and_cooldown_guards() {
    let mut d = PaneDecider::default();
    let times = ticks_tenth(0.0, 4.0);
    let decisions = run(
        &mut d,
        &times,
        Some(ON_RIGHT),
        &Run {
            key: 0.0, // typed at 0: 3 s of grace
            ..Run::default()
        },
    );
    approx(first_fire(&decisions, &times).unwrap(), 3.0);
    assert_eq!(decisions[10].reason, "typing");
    assert!(!decisions[10].pending); // far from firing: no fast sampling
    assert!(decisions[26].pending); // within PENDING_HORIZON_S of the end

    let mut d = PaneDecider::default();
    let decisions = run(
        &mut d,
        &times,
        Some(ON_RIGHT),
        &Run {
            mouse: 0.5,
            ..Run::default()
        },
    );
    approx(first_fire(&decisions, &times).unwrap(), 2.0);
    assert_eq!(decisions[10].reason, "mouse");

    // After a switch, the next one waits for the cooldown.
    let mut d = PaneDecider::default();
    run_default(&mut d, &ticks_tenth(0.0, 0.6), ON_RIGHT); // fires at 0.6: focus moves to %1
    let times = ticks_tenth(0.7, 2.0);
    let decisions = run(
        &mut d,
        &times,
        Some(ON_LEFT),
        &Run {
            focused: 1,
            ..Run::default()
        },
    );
    approx(first_fire(&decisions, &times).unwrap(), 1.6);
    assert!(!reasons(&decisions).contains("manual")); // that change was ours
}

#[test]
fn test_focus_changed_by_the_user_holds_switching() {
    let mut d = PaneDecider::default();
    run_default(&mut d, &ticks_tenth(0.0, 1.0), ON_LEFT); // focus on %0
    let times = ticks_tenth(1.1, 5.0);
    // The user focused %1 themselves while looking at %0.
    let decisions = run(
        &mut d,
        &times,
        Some(ON_LEFT),
        &Run {
            focused: 1,
            ..Run::default()
        },
    );
    let fired_at = first_fire(&decisions, &times).unwrap();
    assert!(fired_at >= 1.1 + PaneConfig::default().manual_grace_s - 1e-6);
    assert_eq!(decisions[10].reason, "manual");
}

#[test]
fn test_another_window_starts_from_scratch() {
    let mut d = PaneDecider::default();
    run_default(&mut d, &ticks_tenth(0.0, 0.5), ON_RIGHT);
    let other = snap_of(0.6, &[LEFT, RIGHT], 0, 8);
    assert_eq!(
        update(&mut d, 0.6, Some(ON_RIGHT), Some(&other)).reason,
        "dwell"
    );
    let times = ticks_tenth(0.7, 1.3);
    let decisions: Vec<PaneDecision> = times
        .iter()
        .map(|&t| {
            let s = snap_of(t, &[LEFT, RIGHT], 0, 8);
            update(&mut d, t, Some(ON_RIGHT), Some(&s))
        })
        .collect();
    approx(first_fire(&decisions, &times).unwrap(), 1.2);
}

#[test]
fn test_reading_another_pane_while_typing_keeps_the_focus() {
    // Typing in %0 while reading the output in %1: reading pauses do not move focus.
    let mut d = PaneDecider::default();
    for t in ticks(0.0, 3.0, 0.5) {
        d.update(
            t,
            Some(ON_RIGHT),
            Some(WINDOW),
            Some(&snap(t)),
            Some(SIGMA),
            NEVER,
            t, // typing all along
            true,
        );
    }
    assert_eq!(d.reading_pane(), Some(&id("%1")));
    // A reading pause: no keystroke since 3.0 s. The typing grace (3 s) would
    // end at 6.0; the reading grace (8 s) holds until 11.0.
    let times = ticks_tenth(3.1, 12.0);
    let decisions = run(
        &mut d,
        &times,
        Some(ON_RIGHT),
        &Run {
            key: 3.0,
            ..Run::default()
        },
    );
    approx(first_fire(&decisions, &times).unwrap(), 11.0);
}

#[test]
fn test_eligible_panes_and_config_from_settings() {
    let three = snap_of(
        0.0,
        &[
            LEFT,
            Rect::new(960, 0, 960, 540),
            Rect::new(960, 540, 960, 540),
        ],
        0,
        7,
    );
    let cfg = PaneConfig::default();
    assert_eq!(eligible_panes(Some(&three), SIGMA, &cfg), 2);
    assert_eq!(eligible_panes(Some(&three), (100.0, 300.0), &cfg), 2); // side by side: x decides
    assert_eq!(eligible_panes(Some(&three), (500.0, 80.0), &cfg), 0);
    assert_eq!(eligible_panes(None, SIGMA, &cfg), 0);

    let panes = PaneSettings {
        dwell_ms: 900,
        precision: 3.0,
        cooldown_ms: 200,
        typing_grace_ms: 1000,
        ..PaneSettings::default()
    };
    let switching = SwitchingSettings {
        mouse_grace_ms: 700,
        ..SwitchingSettings::default()
    };
    let cfg = PaneConfig::from_settings(&panes, &switching);
    assert_eq!(
        (cfg.dwell_s, cfg.precision, cfg.cooldown_s),
        (0.9, 3.0, 0.2)
    );
    assert_eq!(
        (cfg.mouse_grace_s, cfg.typing_grace_s, cfg.manual_grace_s),
        (0.7, 1.0, 1.0)
    );
    assert_eq!(
        (cfg.reading_grace_s, cfg.hysteresis, cfg.min_pane_px),
        (8.0, 0.5, 240.0)
    );
}

#[test]
fn test_reset_forgets_the_cooldown_and_the_window() {
    let mut d = PaneDecider::default();
    run_default(&mut d, &ticks_tenth(0.0, 0.6), ON_RIGHT);
    approx(d.last_switch_time(), 0.6);
    d.reset();
    assert_eq!(d.last_switch_time(), f64::NEG_INFINITY);
    assert_eq!(d.reading_pane(), None);
}
