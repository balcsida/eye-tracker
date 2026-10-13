//! Tests for hook-free mouse/keyboard activity tracking (engine/input_state.rs).
//!
//! `test_invalid_provider_values_are_ignored` loses its `"garbage"` reading: a
//! provider returns `Option<f64>`, so a non-numeric value cannot exist.

use std::cell::RefCell;
use std::rc::Rc;

use super::*;

const NEVER: f64 = f64::NEG_INFINITY;

fn approx(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-6, "{a} != {b}");
}

/// OS idle-time provider driven by explicit input timestamps.
struct FakeIdle {
    last_input: Option<f64>,
    now: f64,
    fail: bool,
}

type Idle = Rc<RefCell<FakeIdle>>;

fn fake_idle(last_input: Option<f64>) -> Idle {
    Rc::new(RefCell::new(FakeIdle {
        last_input,
        now: 0.0,
        fail: false,
    }))
}

fn provider(idle: &Idle) -> IdleProvider {
    let idle = Rc::clone(idle);
    Box::new(move || {
        let f = idle.borrow();
        if f.fail {
            return None; // the Python provider raised OSError
        }
        f.last_input.map(|t| (f.now - t).max(0.0))
    })
}

fn make(idle: Idle, key_idle: Idle) -> (InputTracker, Idle, Idle) {
    (
        InputTracker::new(provider(&idle), provider(&key_idle)),
        idle,
        key_idle,
    )
}

fn poll(tr: &mut InputTracker, idle: &Idle, key: &Idle, now: f64, pos: (i32, i32)) {
    idle.borrow_mut().now = now;
    key.borrow_mut().now = now;
    tr.poll(now, pos);
}

fn silent() -> InputTracker {
    InputTracker::new(Box::new(|| None), Box::new(|| None))
}

/// A tracker whose idle timer plays back `values`, then reports nothing.
fn scripted<const N: usize>(values: [Option<f64>; N]) -> InputTracker {
    let mut it = values.into_iter();
    InputTracker::new(Box::new(move || it.next().flatten()), Box::new(|| None))
}

// ----------------------------------------------------------------------- mouse
#[test]
fn test_initial_state_is_never() {
    let tr = silent();
    assert_eq!(tr.last_mouse_activity(), NEVER);
    assert_eq!(tr.last_key_activity(), NEVER);
    assert_eq!(tr.last_any_activity(), NEVER);
    assert_eq!(tr.seconds_since_any(10.0), None);
    assert!(!tr.manual_move());
}

#[test]
fn test_first_poll_only_sets_the_baseline() {
    let mut tr = silent();
    tr.poll(1.0, (500, 500));
    assert_eq!(tr.cursor(), (500, 500));
    assert_eq!(tr.last_mouse_activity(), NEVER);
    assert!(!tr.manual_move());
}

#[test]
fn test_cursor_move_is_mouse_activity() {
    let mut tr = silent();
    tr.poll(1.0, (500, 500));
    tr.poll(1.1, (520, 505));
    assert!(tr.manual_move());
    assert_eq!(tr.last_mouse_activity(), 1.1);
    assert_eq!(tr.last_any_activity(), 1.1);
    approx(tr.seconds_since_any(1.6).unwrap(), 0.5);
    tr.poll(1.2, (520, 505));
    assert!(!tr.manual_move());
    assert_eq!(tr.last_mouse_activity(), 1.1);
}

#[test]
fn test_tiny_jitter_is_not_mouse_activity() {
    let mut tr = silent();
    tr.poll(1.0, (500, 500));
    tr.poll(1.1, (502, 500));
    tr.poll(1.2, (503, 501));
    assert_eq!(tr.last_mouse_activity(), NEVER);
}

#[test]
fn test_slow_drag_is_mouse_activity() {
    // Dragging a slider at ~15 px/s: 1.5 px per 100 ms poll, below the per-poll
    // move threshold every time.
    let mut tr = silent();
    let mut moves = Vec::new();
    for i in 0..31 {
        let t = ((1.0 + 0.1 * f64::from(i)) * 1e6).round() / 1e6;
        tr.poll(
            t,
            ((500.0 + 1.5 * f64::from(i)).round_ties_even() as i32, 500),
        );
        moves.push(tr.manual_move());
    }
    assert!(moves.iter().any(|m| *m));
    assert!(tr.last_mouse_activity() >= 3.0); // detected again and again, not just once
    assert_eq!(tr.last_key_activity(), NEVER);
}

#[test]
fn test_one_pixel_jitter_never_adds_up_to_a_move() {
    let mut tr = silent();
    for i in 0..100 {
        tr.poll(1.0 + 0.1 * f64::from(i), (500 + i % 2, 500 - (i / 2) % 2));
    }
    assert_eq!(tr.last_mouse_activity(), NEVER);
}

#[test]
fn test_programmatic_move_is_not_user_activity() {
    let mut tr = silent();
    tr.poll(1.0, (500, 500));
    tr.note_programmatic_move((2880, 540), 1.05);
    tr.poll(1.1, (2881, 541));
    assert!(!tr.manual_move());
    assert_eq!(tr.last_mouse_activity(), NEVER);
    assert_eq!(tr.cursor(), (2881, 541));
}

#[test]
fn test_user_move_after_warp_is_detected() {
    let mut tr = silent();
    tr.poll(1.0, (500, 500));
    tr.note_programmatic_move((2880, 540), 1.05);
    tr.poll(1.1, (2880, 540));
    tr.poll(1.2, (2870, 540));
    assert!(tr.manual_move());
    assert_eq!(tr.last_mouse_activity(), 1.2);
}

#[test]
fn test_warp_landing_elsewhere_is_user_activity() {
    let mut tr = silent();
    tr.poll(1.0, (500, 500));
    tr.note_programmatic_move((2880, 540), 1.05);
    tr.poll(1.1, (2900, 540)); // the user moved the mouse during the warp
    assert!(tr.manual_move());
}

#[test]
fn test_warp_explains_only_one_jump() {
    let mut tr = silent();
    tr.poll(1.0, (500, 500));
    tr.note_programmatic_move((2880, 540), 1.0);
    tr.poll(1.1, (2880, 540));
    tr.poll(1.2, (500, 500)); // user drags it back
    tr.poll(1.3, (2880, 540)); // ...and to the warp target again
    assert!(tr.manual_move());
    assert_eq!(tr.last_mouse_activity(), 1.3);
}

#[test]
fn test_warp_seen_late_by_a_slow_poll_is_still_explained() {
    let mut tr = silent();
    tr.poll(1.0, (500, 500));
    tr.note_programmatic_move((2880, 540), 1.0);
    tr.poll(2.0, (2880, 540)); // first poll after the warp, 1 s later
    assert!(!tr.manual_move());
}

#[test]
fn test_warp_expires() {
    let mut tr = silent();
    tr.poll(1.0, (500, 500));
    tr.note_programmatic_move((2880, 540), 1.0);
    tr.poll(1.1, (500, 500)); // warp not visible yet (asynchronous warp)
    tr.poll(1.4, (2880, 540)); // arrives within the window
    assert!(!tr.manual_move());
    tr.note_programmatic_move((100, 100), 2.0);
    tr.poll(2.1, (2880, 540));
    tr.poll(3.0, (100, 100)); // far too late: a user move to the same place
    assert!(tr.manual_move());
}

// -------------------------------------------------------------------- keyboard
#[test]
fn test_key_idle_provider_sets_key_activity() {
    let (mut tr, idle, key) = make(fake_idle(Some(4.0)), fake_idle(Some(4.0)));
    poll(&mut tr, &idle, &key, 5.0, (0, 0));
    approx(tr.last_key_activity(), 4.0);
    approx(tr.last_any_activity(), 4.0);
    key.borrow_mut().last_input = Some(5.5);
    idle.borrow_mut().last_input = Some(5.5);
    poll(&mut tr, &idle, &key, 6.0, (0, 0));
    approx(tr.last_key_activity(), 5.5);
}

#[test]
fn test_idle_reset_without_cursor_motion_is_typing() {
    let (mut tr, idle, key) = make(fake_idle(Some(0.0)), fake_idle(None));
    poll(&mut tr, &idle, &key, 1.0, (100, 100));
    assert_eq!(tr.last_key_activity(), NEVER); // first poll: baseline only
    idle.borrow_mut().last_input = Some(1.07); // a key press between polls
    poll(&mut tr, &idle, &key, 1.1, (100, 100));
    approx(tr.last_key_activity(), 1.07);
    assert_eq!(tr.last_mouse_activity(), NEVER);
    approx(tr.seconds_since_any(1.1).unwrap(), 0.03);
}

#[test]
fn test_idle_reset_with_cursor_motion_is_mouse_not_typing() {
    let (mut tr, idle, key) = make(fake_idle(Some(0.0)), fake_idle(None));
    poll(&mut tr, &idle, &key, 1.0, (100, 100));
    idle.borrow_mut().last_input = Some(1.08);
    poll(&mut tr, &idle, &key, 1.1, (300, 100));
    assert_eq!(tr.last_mouse_activity(), 1.1);
    assert_eq!(tr.last_key_activity(), NEVER);
    // The mouse stopped; the OS input time does not move, so still no typing.
    poll(&mut tr, &idle, &key, 1.2, (300, 100));
    poll(&mut tr, &idle, &key, 1.3, (300, 100));
    assert_eq!(tr.last_key_activity(), NEVER);
}

#[test]
fn test_small_cursor_motion_blocks_typing_inference() {
    let (mut tr, idle, key) = make(fake_idle(Some(0.0)), fake_idle(None));
    poll(&mut tr, &idle, &key, 1.0, (100, 100));
    idle.borrow_mut().last_input = Some(1.08);
    poll(&mut tr, &idle, &key, 1.1, (101, 100)); // slow, precise mouse motion
    assert_eq!(tr.last_key_activity(), NEVER);
    assert_eq!(tr.last_mouse_activity(), NEVER);
    approx(tr.last_any_activity(), 1.08);
}

#[test]
fn test_idle_jitter_is_ignored() {
    let (mut tr, idle, key) = make(fake_idle(Some(0.0)), fake_idle(None));
    poll(&mut tr, &idle, &key, 1.0, (100, 100));
    idle.borrow_mut().last_input = Some(0.03); // clock skew between our clock and the OS idle timer
    poll(&mut tr, &idle, &key, 1.1, (100, 100));
    assert_eq!(tr.last_key_activity(), NEVER);
}

#[test]
fn test_stale_cached_idle_value_is_ignored() {
    let mut tr = scripted([Some(30.0), Some(30.0), Some(30.0)]);
    tr.poll(1.0, (0, 0));
    tr.poll(1.5, (0, 0));
    tr.poll(2.0, (0, 0));
    assert_eq!(tr.last_key_activity(), NEVER);
}

#[test]
fn test_quantised_idle_timer_during_mouse_motion_is_not_typing() {
    // Windows ticks the idle timer in 15.6 ms steps: two polls during continuous
    // mouse motion both read 0.0. The second reading must still be consumed.
    let mut tr = scripted([Some(5.0), Some(0.0), Some(0.0), Some(0.1), Some(0.2)]);
    tr.poll(1.0, (100, 100));
    tr.poll(1.1, (130, 100)); // moving
    tr.poll(1.2, (160, 100)); // still moving; same idle reading as before
    tr.poll(1.3, (160, 100)); // stopped: last input at 1.2 was the mouse
    tr.poll(1.4, (160, 100));
    approx(tr.last_mouse_activity(), 1.2);
    assert_eq!(tr.last_key_activity(), NEVER);
}

#[test]
fn test_failing_providers_are_tolerated() {
    let (mut tr, idle, key) = make(fake_idle(Some(0.0)), fake_idle(None));
    idle.borrow_mut().fail = true;
    key.borrow_mut().fail = true;
    poll(&mut tr, &idle, &key, 1.0, (100, 100));
    poll(&mut tr, &idle, &key, 1.1, (200, 100));
    assert_eq!(tr.last_mouse_activity(), 1.1);
    assert_eq!(tr.last_key_activity(), NEVER);
}

#[test]
fn test_invalid_provider_values_are_ignored() {
    let mut tr = scripted([Some(f64::NAN), Some(-1.0), None]);
    for i in 0..3 {
        tr.poll(1.0 + f64::from(i), (0, 0));
    }
    assert_eq!(tr.last_any_activity(), NEVER);
    assert_eq!(tr.seconds_since_any(5.0), None);
}

#[test]
fn test_os_idle_populates_any_activity() {
    let (mut tr, idle, key) = make(fake_idle(Some(2.0)), fake_idle(None));
    poll(&mut tr, &idle, &key, 10.0, (0, 0));
    approx(tr.seconds_since_any(10.0).unwrap(), 8.0);
}
