// Tests for the shared reading rule (engine/reading.rs) with non-monitor keys.
//
// The monitor decider's use of it is covered by decision_tests.rs.

use super::*;
use crate::engine::decision;

const TYPING_GRACE_S: f64 = 3.0;
const READING_GRACE_S: f64 = 8.0;

fn update(tracker: &mut ReadingTracker<&'static str>, now: f64, last_key: f64) {
    tracker.update(now, last_key, TYPING_GRACE_S, READING_GRACE_S, &(), true);
}

fn look<K: Clone + PartialEq + Debug>(
    tracker: &mut ReadingTracker<K>,
    key: K,
    start: f64,
    stop: f64,
    last_key: f64,
) {
    let mut t = start;
    while t <= stop + 1e-9 {
        tracker.observe(key.clone(), t);
        tracker.update(t, last_key, TYPING_GRACE_S, READING_GRACE_S, &(), true);
        t = (t * 1e6 + 0.5e6).round() / 1e6;
    }
}

#[test]
fn test_typing_through_a_look_at_another_pane_makes_it_the_reading_pane() {
    let mut tracker = ReadingTracker::new();
    look(&mut tracker, "%2", 0.0, 2.0, READING_EVIDENCE_S + 0.5);
    assert_eq!(tracker.reading(), Some(&"%2"));
    tracker.end_look();
    update(&mut tracker, 3.0, 2.5);
    assert_eq!(tracker.reading(), Some(&"%2"));
}

#[test]
fn test_a_single_sample_or_a_short_overlap_is_no_evidence() {
    let mut tracker = ReadingTracker::new();
    tracker.observe("%2", 0.0);
    update(&mut tracker, 0.0, 1.5);
    assert_eq!(tracker.reading(), None);
    tracker.end_look();
    look(&mut tracker, "%3", 10.0, 11.0, 10.2); // typing stopped as the look began
    tracker.end_look();
    update(&mut tracker, 11.0 + READING_RETURN_S + 1.0, 10.2);
    assert_eq!(tracker.reading(), None);
}

#[test]
fn test_reading_ends_with_a_typing_pause_or_when_forgotten() {
    let mut tracker = ReadingTracker::new();
    for t in [0.0, 0.5, 1.0, 1.5] {
        tracker.observe(7, t);
        tracker.update(t, 1.5, TYPING_GRACE_S, READING_GRACE_S, &(), true);
    }
    assert_eq!(tracker.reading(), Some(&7));
    tracker.end_look();
    // typing paused for the reading grace
    tracker.update(1.5 + 8.0, 1.5, TYPING_GRACE_S, READING_GRACE_S, &(), true);
    assert_eq!(tracker.reading(), None);

    for t in [20.0, 20.5, 21.0, 21.5] {
        tracker.observe(7, t);
        tracker.update(t, t, TYPING_GRACE_S, READING_GRACE_S, &(), true);
    }
    tracker.end_look();
    let late = 21.5 + READING_FORGET_S + 1.0;
    // typing on, but never looked at again
    tracker.update(late, late, TYPING_GRACE_S, READING_GRACE_S, &(), true);
    assert_eq!(tracker.reading(), None);
}

#[test]
fn test_rule_is_off_when_the_reading_grace_is_not_longer() {
    let mut tracker = ReadingTracker::new();
    for t in [0.0, 0.5, 1.0, 1.5] {
        tracker.observe("a", t);
        tracker.update(t, 1.5, 3.0, 3.0, &(), true);
    }
    assert_eq!(tracker.reading(), None);
}

#[test]
fn test_inactive_context_recognises_nothing_and_clear_forgets() {
    let mut tracker = ReadingTracker::new();
    for t in [0.0, 0.5, 1.0, 1.5] {
        tracker.observe("a", t);
        tracker.update(t, 1.5, TYPING_GRACE_S, READING_GRACE_S, &(), false);
    }
    assert_eq!(tracker.reading(), None);
    update(&mut tracker, 1.5, 1.5);
    assert_eq!(tracker.reading(), Some(&"a"));
    tracker.clear();
    assert_eq!(tracker.reading(), None);
    update(&mut tracker, 2.0, f64::NEG_INFINITY);
    assert_eq!(tracker.reading(), None);
}

#[test]
fn test_monitor_decider_re_exports_the_constants() {
    assert_eq!(decision::READING_FORGET_S, READING_FORGET_S);
    assert_eq!(decision::READING_EVIDENCE_S, READING_EVIDENCE_S);
}
