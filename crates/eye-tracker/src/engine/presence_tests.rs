//! Tests for walk-away detection (engine/presence.rs).

use super::*;
use crate::config::PresenceSettings;

const IDLE: f64 = 999.0; // seconds since input: nobody touched anything

fn kinds(events: &[PresenceEvent]) -> Vec<PresenceEventKind> {
    events.iter().map(|e| e.kind).collect()
}

fn monitor_with(f: impl FnOnce(&mut PresenceConfig)) -> PresenceMonitor {
    let mut cfg = PresenceConfig {
        away_timeout_s: 45.0,
        warning_s: 10.0,
        ..PresenceConfig::default()
    };
    f(&mut cfg);
    PresenceMonitor::new(cfg, 0.0)
}

fn monitor() -> PresenceMonitor {
    monitor_with(|_| {})
}

/// Update every `dt` seconds in `[start, stop]` and collect (time, event).
fn feed(
    pm: &mut PresenceMonitor,
    start: f64,
    stop: f64,
    face: Option<bool>,
    idle: Option<f64>,
) -> Vec<(f64, PresenceEvent)> {
    let dt = 0.5;
    let mut out = Vec::new();
    let n = ((stop - start) / dt).round() as i64;
    for i in 0..=n {
        let t = start + i as f64 * dt;
        out.extend(pm.update(t, face, idle).into_iter().map(|e| (t, e)));
    }
    out
}

fn feed_absent(pm: &mut PresenceMonitor, start: f64, stop: f64) -> Vec<(f64, PresenceEvent)> {
    feed(pm, start, stop, Some(false), Some(IDLE))
}

fn timed_kinds(events: &[(f64, PresenceEvent)]) -> Vec<(f64, PresenceEventKind)> {
    events.iter().map(|(t, e)| (*t, e.kind)).collect()
}

fn approx(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-6, "{a} != {b}");
}

use PresenceEventKind::{Away, Cancel, Return, Warn};

#[test]
fn test_face_keeps_user_present() {
    let mut pm = monitor();
    assert!(feed(&mut pm, 0.0, 120.0, Some(true), Some(IDLE)).is_empty());
    assert_eq!(pm.state(), PresenceState::Present);
}

#[test]
fn test_full_cycle_warn_away_return() {
    let mut pm = monitor();
    let events = feed_absent(&mut pm, 0.5, 60.0);
    assert_eq!(timed_kinds(&events), [(35.0, Warn), (45.0, Away)]);
    approx(events[0].1.remaining_s, 10.0);
    assert_eq!(pm.state(), PresenceState::Away);
    assert_eq!(pm.remaining(60.0), 0.0);
    assert_eq!(kinds(&pm.update(61.0, Some(true), Some(IDLE))), [Return]);
    assert_eq!(pm.state(), PresenceState::Present);
}

#[test]
fn test_face_during_countdown_cancels() {
    let mut pm = monitor();
    feed_absent(&mut pm, 0.5, 40.0);
    assert_eq!(pm.state(), PresenceState::Warning);
    approx(pm.remaining(40.0), 5.0);
    assert_eq!(kinds(&pm.update(40.5, Some(true), Some(IDLE))), [Cancel]);
    assert_eq!(pm.state(), PresenceState::Present);
    // The absence timer restarts from the moment the face was seen.
    assert!(feed_absent(&mut pm, 41.0, 75.0).is_empty());
    assert_eq!(kinds(&pm.update(75.5, Some(false), Some(IDLE))), [Warn]);
}

#[test]
fn test_recent_input_counts_as_presence() {
    let mut pm = monitor();
    assert!(feed(&mut pm, 0.5, 120.0, Some(false), Some(0.2)).is_empty());
    assert_eq!(pm.state(), PresenceState::Present);
}

#[test]
fn test_input_cancels_countdown() {
    let mut pm = monitor();
    feed_absent(&mut pm, 0.5, 36.0);
    assert_eq!(pm.state(), PresenceState::Warning);
    let events = pm.update(36.5, Some(false), Some(INPUT_EVIDENCE_S - 0.1));
    assert_eq!(kinds(&events), [Cancel]);
}

#[test]
fn test_old_input_is_not_presence() {
    let mut pm = monitor();
    let events = feed(
        &mut pm,
        0.5,
        50.0,
        Some(false),
        Some(INPUT_EVIDENCE_S + 0.1),
    );
    assert_eq!(
        timed_kinds(&events).iter().map(|e| e.1).collect::<Vec<_>>(),
        [Warn, Away]
    );
}

#[test]
fn test_input_ignored_when_not_required() {
    let mut pm = monitor_with(|c| c.require_input_idle = false);
    let events = feed(&mut pm, 0.5, 50.0, Some(false), Some(0.0));
    assert_eq!(
        timed_kinds(&events).iter().map(|e| e.1).collect::<Vec<_>>(),
        [Warn, Away]
    );
}

#[test]
fn test_unknown_input_idle_relies_on_face_only() {
    let mut pm = monitor();
    let events = feed(&mut pm, 0.5, 50.0, Some(false), None);
    assert_eq!(
        timed_kinds(&events).iter().map(|e| e.1).collect::<Vec<_>>(),
        [Warn, Away]
    );
}

#[test]
fn test_unknown_face_freezes_timers() {
    let mut pm = monitor();
    feed_absent(&mut pm, 0.5, 20.0);
    // Camera off (privacy mode) for a long time: never counts as absence.
    assert!(feed(&mut pm, 20.5, 500.0, None, Some(IDLE)).is_empty());
    assert_eq!(pm.state(), PresenceState::Present);
    // Absence counts again from the end of the unknown period.
    let events = feed_absent(&mut pm, 500.5, 560.0);
    assert_eq!(timed_kinds(&events), [(535.0, Warn), (545.0, Away)]);
}

#[test]
fn test_unknown_face_during_countdown_cancels_it() {
    let mut pm = monitor();
    feed_absent(&mut pm, 0.5, 38.0);
    assert_eq!(pm.state(), PresenceState::Warning);
    assert_eq!(kinds(&pm.update(38.5, None, Some(IDLE))), [Cancel]);
    assert_eq!(pm.state(), PresenceState::Present);
}

#[test]
fn test_unknown_face_while_away_stays_away() {
    let mut pm = monitor();
    feed_absent(&mut pm, 0.5, 50.0);
    assert!(feed(&mut pm, 50.5, 100.0, None, Some(IDLE)).is_empty());
    assert_eq!(pm.state(), PresenceState::Away);
}

#[test]
fn test_no_countdown_goes_straight_to_away() {
    let mut pm = monitor_with(|c| c.warning_s = 0.0);
    let events = feed_absent(&mut pm, 0.5, 60.0);
    assert_eq!(timed_kinds(&events), [(45.0, Away)]);
}

#[test]
fn test_sparse_updates_never_skip_the_countdown() {
    let mut pm = monitor();
    pm.update(1.0, Some(true), Some(IDLE));
    // Next update only after the whole timeout (e.g. the process was suspended).
    let events = pm.update(100.0, Some(false), Some(IDLE));
    assert_eq!(kinds(&events), [Warn]);
    approx(events[0].remaining_s, 10.0);
    assert!(pm.update(105.0, Some(false), Some(IDLE)).is_empty());
    assert_eq!(kinds(&pm.update(110.0, Some(false), Some(IDLE))), [Away]);
}

#[test]
fn test_warning_longer_than_timeout_still_waits_briefly() {
    let mut pm = monitor_with(|c| {
        c.away_timeout_s = 5.0;
        c.warning_s = 30.0;
    });
    // A single missed detection must not flash the countdown.
    assert!(pm.update(0.5, Some(false), Some(IDLE)).is_empty());
    let events = pm.update(1.0, Some(false), Some(IDLE));
    assert_eq!(kinds(&events), [Warn]);
    approx(events[0].remaining_s, 4.0);
    assert_eq!(kinds(&pm.update(5.0, Some(false), Some(IDLE))), [Away]);
}

#[test]
fn test_disabled_is_always_present() {
    let mut pm = monitor_with(|c| c.enabled = false);
    assert!(feed_absent(&mut pm, 0.5, 200.0).is_empty());
    assert_eq!(pm.state(), PresenceState::Present);
    assert!(pm.remaining(200.0).is_infinite());
}

#[test]
fn test_disabling_during_countdown_cancels_it() {
    let mut pm = monitor();
    feed_absent(&mut pm, 0.5, 36.0);
    assert_eq!(pm.state(), PresenceState::Warning);
    pm.set_config(PresenceConfig {
        enabled: false,
        ..PresenceConfig::default()
    });
    assert_eq!(kinds(&pm.update(36.5, Some(false), Some(IDLE))), [Cancel]);
    assert!(pm.update(37.0, Some(false), Some(IDLE)).is_empty());
}

#[test]
fn test_disabling_while_away_reports_return() {
    let mut pm = monitor();
    feed_absent(&mut pm, 0.5, 50.0);
    pm.set_config(PresenceConfig {
        enabled: false,
        ..PresenceConfig::default()
    });
    assert_eq!(kinds(&pm.update(50.5, Some(false), Some(IDLE))), [Return]);
    assert_eq!(pm.state(), PresenceState::Present);
}

#[test]
fn test_reenabling_does_not_count_disabled_time_as_absence() {
    let mut pm = monitor_with(|c| c.enabled = false);
    pm.update(1.0, Some(false), Some(IDLE));
    pm.set_config(PresenceConfig::default());
    approx(pm.remaining(500.0), 45.0);
    assert!(pm.update(500.0, Some(false), Some(IDLE)).is_empty());
    let events = feed_absent(&mut pm, 500.5, 546.0);
    assert_eq!(timed_kinds(&events), [(535.0, Warn), (545.0, Away)]);
}

#[test]
fn test_longer_timeout_during_countdown_cancels_it() {
    let mut pm = monitor();
    feed_absent(&mut pm, 0.5, 36.0);
    pm.set_config(PresenceConfig {
        away_timeout_s: 300.0,
        warning_s: 10.0,
        ..PresenceConfig::default()
    });
    assert_eq!(kinds(&pm.update(36.5, Some(false), Some(IDLE))), [Cancel]);
    assert_eq!(pm.state(), PresenceState::Present);
}

#[test]
fn test_reset_returns_to_present_silently() {
    let mut pm = monitor();
    feed_absent(&mut pm, 0.5, 50.0);
    assert_eq!(pm.state(), PresenceState::Away);
    pm.reset(60.0);
    assert_eq!(pm.state(), PresenceState::Present);
    assert_eq!(pm.last_seen(), 60.0);
    assert!(pm.update(60.5, Some(false), Some(IDLE)).is_empty());
    approx(pm.remaining(60.5), 44.5);
}

#[test]
fn test_config_from_settings() {
    let s = PresenceSettings {
        enabled: false,
        away_timeout_s: 90,
        warning_s: 5,
        require_input_idle: false,
        ..PresenceSettings::default()
    };
    assert_eq!(
        PresenceConfig::from_settings(&s),
        PresenceConfig {
            enabled: false,
            away_timeout_s: 90.0,
            warning_s: 5.0,
            require_input_idle: false
        }
    );
    assert_eq!(
        PresenceConfig::from_settings(&PresenceSettings::default()),
        PresenceConfig::default()
    );
}
