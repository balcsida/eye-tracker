//! Tests for the shoulder-surfer guard (engine/guard.rs).

use super::*;
use crate::config::PrivacySettings;
use GuardEvent::{Clear, Trigger};

fn guard(delay_s: f64, clear_s: f64) -> ShoulderGuard {
    ShoulderGuard::new(GuardConfig {
        enabled: true,
        delay_s,
        clear_s,
        ..GuardConfig::default()
    })
}

fn upd(g: &mut ShoulderGuard, t: f64, faces: Option<u32>) -> Option<GuardEvent> {
    g.update(t, faces, None, f64::NEG_INFINITY)
}

fn feed(
    g: &mut ShoulderGuard,
    start: f64,
    stop: f64,
    faces: Option<u32>,
) -> Vec<(f64, GuardEvent)> {
    feed_boxes(g, start, stop, faces, None, 0.25, f64::NEG_INFINITY)
}

fn feed_boxes(
    g: &mut ShoulderGuard,
    start: f64,
    stop: f64,
    faces: Option<u32>,
    face_box: Option<FaceBox>,
    dt: f64,
    last_input: f64,
) -> Vec<(f64, GuardEvent)> {
    let mut out = Vec::new();
    for i in 0..=((stop - start) / dt).round() as i64 {
        let t = start + i as f64 * dt;
        if let Some(result) = g.update(t, faces, face_box, last_input) {
            out.push((t, result));
        }
    }
    out
}

/// `feed_boxes` with the default step and no input.
fn fb(
    g: &mut ShoulderGuard,
    start: f64,
    stop: f64,
    faces: u32,
    face_box: Option<FaceBox>,
) -> Vec<(f64, GuardEvent)> {
    feed_boxes(
        g,
        start,
        stop,
        Some(faces),
        face_box,
        0.25,
        f64::NEG_INFINITY,
    )
}

#[test]
fn test_default_config_is_disabled() {
    let mut g = ShoulderGuard::default();
    assert!(feed(&mut g, 0.0, 10.0, Some(3)).is_empty());
    assert!(!g.active());
}

#[test]
fn test_single_face_never_triggers() {
    let mut g = guard(2.0, 1.5);
    assert!(feed(&mut g, 0.0, 30.0, Some(1)).is_empty());
    assert!(feed(&mut g, 30.25, 40.0, Some(0)).is_empty());
}

#[test]
fn test_second_face_triggers_after_delay() {
    let mut g = guard(2.0, 1.5);
    assert_eq!(feed(&mut g, 0.0, 10.0, Some(2)), [(2.0, Trigger)]);
    assert!(g.active());
}

#[test]
fn test_passer_by_is_ignored() {
    let mut g = guard(2.0, 1.5);
    assert!(feed(&mut g, 0.0, 1.5, Some(2)).is_empty());
    // Walked out of view: the run of two faces restarts.
    assert!(feed(&mut g, 1.75, 2.5, Some(1)).is_empty());
    assert!(feed(&mut g, 2.75, 4.5, Some(2)).is_empty());
    assert_eq!(feed(&mut g, 4.75, 4.75, Some(2)), [(4.75, Trigger)]);
}

#[test]
fn test_clears_after_second_face_is_gone() {
    let mut g = guard(1.0, 1.5);
    feed(&mut g, 0.0, 1.0, Some(2));
    assert!(g.active());
    assert_eq!(feed(&mut g, 1.25, 5.0, Some(1)), [(2.75, Clear)]);
    assert!(!g.active());
}

#[test]
fn test_brief_dropout_does_not_clear() {
    let mut g = guard(1.0, 1.5);
    feed(&mut g, 0.0, 1.0, Some(2));
    assert!(feed(&mut g, 1.25, 2.25, Some(1)).is_empty());
    assert!(feed(&mut g, 2.5, 3.0, Some(2)).is_empty());
    assert!(g.active());
    // The clear timer restarted when the second face reappeared.
    assert_eq!(feed(&mut g, 3.25, 5.0, Some(1)), [(4.75, Clear)]);
}

#[test]
fn test_unknown_face_count_keeps_state() {
    let mut g = guard(1.0, 1.5);
    feed(&mut g, 0.0, 1.0, Some(2));
    assert!(feed(&mut g, 1.25, 20.0, None).is_empty());
    assert!(g.active());
}

#[test]
fn test_unknown_face_count_breaks_a_pending_run() {
    let mut g = guard(2.0, 1.5);
    feed(&mut g, 0.0, 1.5, Some(2));
    upd(&mut g, 1.75, None);
    assert_eq!(upd(&mut g, 2.0, Some(2)), None); // run restarted at 2.0
    assert_eq!(upd(&mut g, 3.75, Some(2)), None);
    assert_eq!(upd(&mut g, 4.0, Some(2)), Some(Trigger));
}

#[test]
fn test_zero_delay_triggers_immediately() {
    let mut g = guard(0.0, 1.5);
    assert_eq!(upd(&mut g, 5.0, Some(2)), Some(Trigger));
}

#[test]
fn test_disabling_while_active_clears_once() {
    let mut g = guard(0.0, 1.5);
    upd(&mut g, 0.0, Some(2));
    g.set_config(GuardConfig {
        enabled: false,
        ..GuardConfig::default()
    });
    assert_eq!(upd(&mut g, 0.5, Some(2)), Some(Clear));
    assert_eq!(upd(&mut g, 1.0, Some(2)), None);
    assert!(!g.active());
}

#[test]
fn test_reset_is_silent() {
    let mut g = guard(0.0, 1.5);
    upd(&mut g, 0.0, Some(3));
    g.reset();
    assert!(!g.active());
    assert_eq!(upd(&mut g, 0.5, Some(1)), None);
}

#[test]
fn test_brief_dropouts_of_the_second_face_do_not_restart_the_delay() {
    // A distant, partly hidden onlooker is missed in some frames (4 fps idle rate).
    let mut g = guard(2.0, 1.5);
    let mut out = Vec::new();
    for i in 0..20 {
        let t = i as f64 * 0.25;
        let faces = if i % 4 == 3 { 1 } else { 2 }; // every fourth detection misses the onlooker
        if let Some(result) = upd(&mut g, t, Some(faces)) {
            out.push((t, result));
        }
    }
    assert_eq!(out, [(2.0, Trigger)]);
}

#[test]
fn test_single_missed_frame_is_tolerated_at_slow_frame_rates() {
    // 1 fps (eco profile while typing): one miss is a 2 s gap between detections.
    let mut g = guard(2.0, 1.5);
    assert_eq!(upd(&mut g, 0.0, Some(2)), None);
    assert_eq!(upd(&mut g, 1.0, Some(1)), None);
    assert_eq!(upd(&mut g, 2.0, Some(2)), Some(Trigger));
}

#[test]
fn test_long_dropout_restarts_the_delay() {
    let mut g = guard(2.0, 1.5);
    assert!(feed(&mut g, 0.0, 1.5, Some(2)).is_empty());
    assert!(feed(&mut g, 1.75, 3.0, Some(1)).is_empty()); // gone for 1.5 s
    assert!(feed(&mut g, 3.25, 5.0, Some(2)).is_empty());
    assert_eq!(feed(&mut g, 5.25, 5.25, Some(2)), [(5.25, Trigger)]);
}

// The user close to the camera, and a small onlooker behind them.
const OWNER: FaceBox = (0.35, 0.25, 0.3, 0.4);
const OWNER_MOVED: FaceBox = (0.40, 0.28, 0.28, 0.38);
const ONLOOKER: FaceBox = (0.75, 0.10, 0.08, 0.1);
// r2-gaze-engine-01: a user about 65 cm from the camera, and a colleague leaning
// in beside them to about 45 cm (a face side ~1.5x the user's), not overlapping.
const SEATED_USER: FaceBox = (0.40, 0.30, 0.20, 0.27);
const LEANING_COLLEAGUE: FaceBox = (0.05, 0.15, 0.30, 0.40);

#[test]
fn test_guard_stays_up_when_the_user_leaves_and_the_onlooker_stays() {
    let mut g = guard(1.0, 1.5);
    assert_eq!(fb(&mut g, 0.0, 1.0, 2, Some(OWNER)), [(1.0, Trigger)]);
    // The user walks away; only the onlooker's small face is left in view.
    assert!(fb(&mut g, 1.25, 10.0, 1, Some(ONLOOKER)).is_empty());
    assert!(g.active());
    assert!(g.owner_missing());
    // No face at all: the onlooker may just be missed, so the guard stays up...
    assert!(fb(&mut g, 10.25, 20.0, 0, None).is_empty());
    assert!(g.owner_missing());
    // ...until the user is back at the desk.
    assert_eq!(
        fb(&mut g, 20.25, 22.0, 1, Some(OWNER_MOVED)),
        [(21.75, Clear)]
    );
    assert!(!g.owner_missing());
}

#[test]
fn test_onlooker_missed_for_a_moment_keeps_the_guard_up() {
    // r2-gaze-engine-02: detection gaps of a far face must not lift the curtain.
    let mut g = guard(2.0, 1.5);
    let ninf = f64::NEG_INFINITY;
    let run =
        |g: &mut ShoulderGuard, a, b, faces, bx| feed_boxes(g, a, b, Some(faces), bx, 0.5, ninf);
    assert_eq!(run(&mut g, 0.0, 5.0, 2, Some(OWNER)), [(2.0, Trigger)]);
    assert!(run(&mut g, 5.5, 30.0, 1, Some(ONLOOKER)).is_empty());
    // Four empty frames at the 2 fps "no face" rate, then the onlooker again.
    assert!(run(&mut g, 30.5, 32.0, 0, None).is_empty());
    assert!(run(&mut g, 32.5, 40.0, 1, Some(ONLOOKER)).is_empty());
    assert!(g.active());
    assert!(g.owner_missing());
}

#[test]
fn test_camera_gap_keeps_the_owner_missing() {
    let mut g = guard(1.0, 1.5);
    fb(&mut g, 0.0, 1.0, 2, Some(OWNER));
    fb(&mut g, 1.25, 3.0, 1, Some(ONLOOKER));
    assert!(feed_boxes(&mut g, 3.25, 10.0, None, None, 0.25, f64::NEG_INFINITY).is_empty());
    assert!(g.owner_missing());
    assert!(fb(&mut g, 10.25, 12.0, 0, None).is_empty());
    assert!(g.active());
}

#[test]
fn test_larger_colleague_leaning_in_never_becomes_the_owner() {
    // r2-gaze-engine-01 / r2-controller-01: the owner is found by continuity.
    let mut g = guard(2.0, 1.5);
    assert!(fb(&mut g, 0.0, 10.0, 1, Some(SEATED_USER)).is_empty());
    // The colleague is closer to the camera, so theirs is the primary face box.
    assert_eq!(
        fb(&mut g, 10.25, 14.0, 2, Some(LEANING_COLLEAGUE)),
        [(12.25, Trigger)]
    );
    // They straighten up and are gone from one frame to the next.
    assert_eq!(
        fb(&mut g, 14.25, 20.0, 1, Some(SEATED_USER)),
        [(15.75, Clear)]
    );
    assert!(!g.owner_missing());
    assert!(!g.active());
}

#[test]
fn test_lone_face_while_a_second_one_was_just_seen_is_not_taken_for_the_owner() {
    let mut g = guard(2.0, 1.5);
    fb(&mut g, 0.0, 5.0, 1, Some(SEATED_USER));
    fb(&mut g, 5.25, 6.0, 2, Some(LEANING_COLLEAGUE));
    // The user turns to the colleague and is missed for a frame: only the
    // colleague's face is found. That must not make the colleague the owner.
    fb(&mut g, 6.25, 6.25, 1, Some(LEANING_COLLEAGUE));
    assert_eq!(
        fb(&mut g, 6.5, 9.0, 2, Some(LEANING_COLLEAGUE)),
        [(7.25, Trigger)]
    );
    assert_eq!(
        fb(&mut g, 9.25, 12.0, 1, Some(SEATED_USER)),
        [(10.75, Clear)]
    );
    assert!(!g.owner_missing());
}

#[test]
fn test_user_moving_while_the_onlooker_is_there_stays_the_owner() {
    let mut g = guard(1.0, 1.5);
    fb(&mut g, 0.0, 2.0, 1, Some(OWNER));
    let shifted = (0.30, 0.22, 0.3, 0.4); // overlaps OWNER: followed
    fb(&mut g, 2.25, 4.0, 2, Some(shifted));
    assert!(g.active());
    assert_eq!(fb(&mut g, 4.25, 6.0, 1, Some(shifted)), [(5.75, Clear)]);
}

#[test]
fn test_largest_face_is_the_owner_only_while_none_is_known() {
    // Switched on with two faces in view, the larger one the colleague's: the
    // guard cannot know better, so the user alone then counts as a stranger...
    let mut g = guard(1.0, 1.5);
    fb(&mut g, 0.0, 2.0, 2, Some(LEANING_COLLEAGUE));
    assert!(g.active());
    assert!(fb(&mut g, 2.25, 10.0, 1, Some(SEATED_USER)).is_empty());
    assert!(g.owner_missing());
    // ...until they use the computer: then their face is the owner's.
    let typed = 10.0;
    assert_eq!(
        feed_boxes(&mut g, 10.25, 12.0, Some(1), Some(SEATED_USER), 0.25, typed),
        [(11.75, Clear)]
    );
    assert!(!g.owner_missing());
    // And stays so: the colleague leaning in again does not take over.
    assert_eq!(
        fb(&mut g, 12.25, 14.0, 2, Some(LEANING_COLLEAGUE)),
        [(13.25, Trigger)]
    );
    assert_eq!(
        fb(&mut g, 14.25, 16.0, 1, Some(SEATED_USER)),
        [(15.75, Clear)]
    );
}

#[test]
fn test_input_right_after_the_user_left_does_not_adopt_the_onlooker() {
    let mut g = guard(1.0, 1.5);
    fb(&mut g, 0.0, 1.0, 2, Some(OWNER));
    // The user's face has left the picture; their hand is still on the mouse.
    fb(&mut g, 1.25, 1.25, 1, Some(ONLOOKER));
    let last_touch = 1.25 + INPUT_ADOPT_DELAY_S - 0.5;
    assert!(feed_boxes(&mut g, 1.5, 20.0, Some(1), Some(ONLOOKER), 0.25, last_touch).is_empty());
    assert!(g.owner_missing());
}

#[test]
fn test_input_while_no_face_is_in_view_releases_the_guard() {
    let mut g = guard(1.0, 1.5);
    fb(&mut g, 0.0, 1.0, 2, Some(OWNER));
    fb(&mut g, 1.25, 5.0, 1, Some(ONLOOKER));
    assert!(fb(&mut g, 5.25, 10.0, 0, None).is_empty());
    // Someone works at the computer, but the camera sees no face (looking down).
    assert_eq!(
        feed_boxes(&mut g, 10.25, 12.0, Some(0), None, 0.25, 10.0),
        [(11.75, Clear)]
    );
    assert!(!g.owner_missing());
}

#[test]
fn test_guard_clears_when_the_onlooker_leaves_and_the_user_stays() {
    let mut g = guard(1.0, 1.5);
    fb(&mut g, 0.0, 1.0, 2, Some(OWNER));
    assert!(g.active());
    // The user shifts in the chair while the onlooker walks off.
    assert_eq!(fb(&mut g, 1.25, 5.0, 1, Some(OWNER_MOVED)), [(2.75, Clear)]);
    assert!(!g.owner_missing());
}

#[test]
fn test_user_returning_next_to_the_onlooker_is_not_owner_missing() {
    let mut g = guard(1.0, 1.5);
    fb(&mut g, 0.0, 1.0, 2, Some(OWNER));
    fb(&mut g, 1.25, 3.0, 1, Some(ONLOOKER));
    assert!(g.owner_missing());
    fb(&mut g, 3.25, 4.0, 2, Some(OWNER));
    assert!(g.active());
    assert!(!g.owner_missing());
}

#[test]
fn test_new_onlooker_is_noticed_while_the_user_is_wrongly_judged_gone() {
    // r3-logic-03: after a misjudged "the user left", every later onlooker was
    // ignored until the next keyboard or mouse input.
    let mut g = guard(2.0, 1.5);
    fb(&mut g, 0.0, 3.0, 1, Some(SEATED_USER));
    // A colleague leans in (the primary face) while the user makes room.
    assert_eq!(
        fb(&mut g, 3.25, 6.0, 2, Some(LEANING_COLLEAGUE)),
        [(5.25, Trigger)]
    );
    // The colleague leaves; the user now sits further back and aside, reading.
    let leaned_back = (0.55, 0.36, 0.13, 0.18);
    assert!(fb(&mut g, 6.25, 126.0, 1, Some(leaned_back)).is_empty());
    assert!(g.owner_missing());
    // A stranger looks over the user's shoulder: the guard reacts again, once.
    assert_eq!(
        fb(&mut g, 126.25, 136.0, 2, Some(leaned_back)),
        [(128.25, Trigger)]
    );
    assert!(g.active());
    // The stranger leaves: still judged by boxes, the guard stays up as before.
    assert!(fb(&mut g, 136.25, 140.0, 1, Some(leaned_back)).is_empty());
    assert!(g.owner_missing());
}

#[test]
fn test_user_back_next_to_the_onlooker_does_not_trigger_again() {
    let mut g = guard(1.0, 1.5);
    assert_eq!(fb(&mut g, 0.0, 1.0, 2, Some(OWNER)), [(1.0, Trigger)]);
    fb(&mut g, 1.25, 5.0, 1, Some(ONLOOKER));
    assert!(g.owner_missing());
    // The user sits down at their old place next to the onlooker: same visit.
    assert!(fb(&mut g, 5.25, 15.0, 2, Some(OWNER_MOVED)).is_empty());
    assert!(g.active());
    assert!(!g.owner_missing());
}

#[test]
fn test_without_face_boxes_the_guard_clears_as_before() {
    let mut g = guard(1.0, 1.5);
    fb(&mut g, 0.0, 1.0, 2, None);
    assert_eq!(fb(&mut g, 1.25, 5.0, 1, Some(ONLOOKER)), [(2.75, Clear)]);
}

#[test]
fn test_reset_forgets_the_owner() {
    let mut g = guard(0.0, 1.5);
    g.update(0.0, Some(2), Some(OWNER), f64::NEG_INFINITY);
    g.update(0.5, Some(1), Some(ONLOOKER), f64::NEG_INFINITY);
    assert!(g.owner_missing());
    g.reset();
    assert!(!g.owner_missing());
    assert!(!g.active());
}

#[test]
fn test_config_from_settings() {
    let s = PrivacySettings {
        shoulder_guard: true,
        guard_delay_s: 4.0,
        ..PrivacySettings::default()
    };
    let cfg = GuardConfig::from_settings(&s);
    assert!(cfg.enabled);
    assert_eq!(cfg.delay_s, 4.0);
    assert_eq!(cfg.clear_s, GuardConfig::default().clear_s);
    assert!(!GuardConfig::from_settings(&PrivacySettings::default()).enabled);
}
