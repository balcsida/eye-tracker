// Tests for per-monitor cursor/window memory (engine/window_memory.rs).

use super::*;
use crate::testing;
use crate::types::WindowHandle;

fn left() -> Monitor {
    Monitor {
        primary: true,
        ..Monitor::new(0, "left", Rect::new(0, 0, 1920, 1080))
    }
}

fn right() -> Monitor {
    Monitor::new(1, "right", Rect::new(1920, 0, 2560, 1440))
}

fn window(handle: isize) -> WindowRef {
    WindowRef {
        handle: WindowHandle::Hwnd(handle),
        pid: None,
        rect: None,
    }
}

#[test]
fn test_cursor_memory_per_monitor() {
    let mut mem = WindowMemory::new();
    assert_eq!(mem.last_cursor(0), None);
    mem.record_cursor(0, (100, 200));
    mem.record_cursor(1, (2000, 300));
    mem.record_cursor(0, (150, 250));
    assert_eq!(mem.last_cursor(0), Some((150, 250)));
    assert_eq!(mem.last_cursor(1), Some((2000, 300)));
}

#[test]
fn test_window_memory_forget_and_clear() {
    let mut mem = WindowMemory::new();
    mem.record_window(0, window(1));
    mem.record_window(1, window(2));
    assert_eq!(
        mem.last_window(0).map(|w| &w.handle),
        Some(&WindowHandle::Hwnd(1))
    );
    mem.forget_window(0);
    mem.forget_window(5); // unknown index is fine
    assert!(mem.last_window(0).is_none());
    assert_eq!(
        mem.last_window(1).map(|w| &w.handle),
        Some(&WindowHandle::Hwnd(2))
    );
    mem.record_cursor(1, (2000, 10));
    mem.clear();
    assert!(mem.last_window(1).is_none());
    assert_eq!(mem.last_cursor(1), None);
}

#[test]
fn test_last_mode_returns_remembered_position() {
    let mut mem = WindowMemory::new();
    mem.record_cursor(1, (3000, 700));
    assert_eq!(
        choose_cursor_target("last", &right(), &mem, None, None),
        (3000, 700)
    );
}

#[test]
fn test_last_mode_ignores_position_outside_monitor() {
    let mut mem = WindowMemory::new();
    mem.record_cursor(1, (100, 100)); // stale: layout changed
    assert_eq!(
        choose_cursor_target("last", &right(), &mem, None, None),
        (3200, 720)
    );
}

#[test]
fn test_last_mode_falls_back_to_window_centre() {
    let mem = WindowMemory::new();
    let window = Rect::new(2000, 100, 800, 600);
    assert_eq!(
        choose_cursor_target("last", &right(), &mem, None, Some(window)),
        (2400, 400)
    );
}

#[test]
fn test_last_mode_uses_visible_part_of_straddling_window() {
    let mem = WindowMemory::new();
    let window = Rect::new(1520, 100, 800, 400); // half on the left monitor
    let target = choose_cursor_target("last", &right(), &mem, None, Some(window));
    assert_eq!(target, (2120, 300));
    assert!(
        right()
            .rect
            .contains(f64::from(target.0), f64::from(target.1))
    );
}

#[test]
fn test_last_mode_ignores_window_on_another_monitor() {
    let mem = WindowMemory::new();
    let window = Rect::new(100, 100, 500, 500);
    assert_eq!(
        choose_cursor_target("last", &right(), &mem, None, Some(window)),
        (3200, 720)
    );
}

#[test]
fn test_center_mode() {
    let mut mem = WindowMemory::new();
    mem.record_cursor(0, (5, 5));
    assert_eq!(
        choose_cursor_target("center", &left(), &mem, Some((10.0, 10.0)), None),
        (960, 540)
    );
}

#[test]
fn test_gaze_mode_uses_gaze_inside_monitor() {
    let mem = WindowMemory::new();
    assert_eq!(
        choose_cursor_target("gaze", &right(), &mem, Some((2500.4, 900.6)), None),
        (2500, 901)
    );
}

#[test]
fn test_gaze_mode_clamps_into_monitor_away_from_the_edges() {
    // Never onto the outermost row, column or corner (auto-hide taskbars, docks
    // and hot corners react to it): 2 % of 2560 x 1440 is a 51 x 29 px inset.
    let mem = WindowMemory::new();
    let target = choose_cursor_target("gaze", &right(), &mem, Some((5000.0, -40.0)), None);
    assert_eq!(target, (right().rect.right() - 1 - 51, 29));
    assert!(
        right()
            .rect
            .contains(f64::from(target.0), f64::from(target.1))
    );
}

#[test]
fn test_gaze_mode_keeps_the_minimum_inset_on_1080p() {
    let cases = [
        ((-150.0, -100.0), (38, 22)),        // hot corner at the top left
        ((960.0, 1200.0), (960, 1079 - 22)), // auto-hidden taskbar at the bottom
        ((1925.0, 500.0), (1919 - 38, 500)), // the edge next to the other monitor
    ];
    for (gaze, expected) in cases {
        assert_eq!(
            choose_cursor_target("gaze", &left(), &WindowMemory::new(), Some(gaze), None),
            expected
        );
    }
}

#[test]
fn test_gaze_mode_inset_on_a_small_monitor_uses_the_minimum() {
    let small = Monitor::new(3, "small", Rect::new(0, 0, 400, 300));
    assert_eq!(
        choose_cursor_target(
            "gaze",
            &small,
            &WindowMemory::new(),
            Some((-5.0, 999.0)),
            None
        ),
        (16, 283)
    );
}

#[test]
fn test_gaze_mode_without_gaze_uses_centre() {
    for gaze in [None, Some((f64::NAN, 3.0))] {
        assert_eq!(
            choose_cursor_target("gaze", &left(), &WindowMemory::new(), gaze, None),
            (960, 540)
        );
    }
}

#[test]
fn test_unknown_mode_uses_centre() {
    let logs = testing::capture_logs();
    let target = choose_cursor_target(
        "teleport",
        &left(),
        &WindowMemory::new(),
        Some((5.0, 5.0)),
        None,
    );
    assert_eq!(target, (960, 540));
    assert!(
        logs.messages_at_least(log::Level::Warn)
            .iter()
            .any(|m| m.contains("teleport"))
    );
}

#[test]
fn test_target_is_always_inside_monitor() {
    let odd = Monitor::new(2, "tiny", Rect::new(-7, -3, 1, 1));
    let mem = WindowMemory::new();
    for mode in CURSOR_MODES {
        assert_eq!(
            choose_cursor_target(
                mode,
                &odd,
                &mem,
                Some((1e6, -1e6)),
                Some(Rect::new(-100, -100, 50, 50))
            ),
            (-7, -3)
        );
    }
}
