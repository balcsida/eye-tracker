//! Tests of `overlay.rs` (the overlay tests of tests/test_ui_basic.py).
//!
//! Not ported as such: `grab()` of a live widget (replaced by rendering the dot into an image),
//! the real-time wait of `test_overlay_hides_stale_dot` (the clock is an argument now), and the
//! `ui_scale` monkeypatch of `test_overlay_dot_is_scaled_for_the_screen` (the window is built
//! with the scale directly, as `GazeOverlay` takes it from the monitor).

use super::*;
use crate::types::Rect;

fn left() -> Monitor {
    Monitor {
        primary: true,
        ..Monitor::new(0, "Left", Rect::new(0, 0, 1920, 1080))
    }
}

fn right() -> Monitor {
    Monitor::new(1, "Right", Rect::new(1920, 0, 1920, 1080))
}

fn monitors() -> Vec<Monitor> {
    vec![left(), right()]
}

fn gaze(x: f64, y: f64, timestamp: f64) -> Option<GazePoint> {
    Some(GazePoint { x, y, timestamp })
}

fn contains(window: &DotWindow, x: i32, y: i32) -> bool {
    let (left, top, w, h) = window.geometry();
    left <= x && x < left + w && top <= y && y < top + h
}

fn enabled_overlay() -> GazeOverlay {
    let mut overlay = GazeOverlay::new(false, 2500);
    overlay.set_enabled(true);
    overlay
}

#[test]
fn test_overlay_shows_dot_only_on_gaze_monitor() {
    let monitors = monitors();
    let mut overlay = enabled_overlay();
    assert!(overlay.windows().is_empty()); // created lazily
    overlay.set_gaze(gaze(2500.0, 400.0, 1.0), &monitors, 0.0);
    assert_eq!(overlay.visible_monitor(), Some(1));
    assert_eq!(overlay.windows().keys().copied().collect::<Vec<_>>(), [1]);
    let window = &overlay.windows()[&1];
    assert!(contains(window, 2500, 400));
    let (x, y, w, h) = window.geometry();
    assert!(right().rect.contains(f64::from(x), f64::from(y)));
    assert!(
        right()
            .rect
            .contains(f64::from(x + w - 1), f64::from(y + h - 1))
    );
    assert_eq!(window.viewport_builder().mouse_passthrough, Some(true)); // click-through
    assert!(window.render().is_some());

    overlay.set_gaze(gaze(300.0, 300.0, 2.0), &monitors, 0.0);
    assert_eq!(overlay.visible_monitor(), Some(0));
    assert_eq!(
        overlay.windows().keys().copied().collect::<Vec<_>>(),
        [0, 1]
    );

    overlay.set_gaze(None, &monitors, 0.0);
    assert_eq!(overlay.visible_monitor(), None);
}

#[test]
fn test_overlay_dot_is_scaled_for_the_screen() {
    let window = DotWindow::new(&left(), 2.0);
    assert!((window.radius - 28.0).abs() < 1e-9);
    assert!(window.side >= 2 * 28 * 2);
    let image = window.render().unwrap();
    assert_eq!(image.width, window.side as u32);
    // The glow fades out before the corner; the dot is opaque in the middle.
    assert_eq!(image.pixel(0, 0)[3], 0);
    let middle = image.width / 2;
    assert!(image.pixel(middle, middle)[3] > 200);
}

#[test]
fn test_overlay_keeps_windows_inside_their_monitor() {
    let monitors = monitors();
    let mut overlay = enabled_overlay();
    overlay.set_gaze(gaze(1919.0, 1079.0, 0.0), &monitors, 0.0); // bottom-right corner of LEFT
    let window = &overlay.windows()[&0];
    let (x, y, w, h) = window.geometry();
    assert!(
        left()
            .rect
            .contains(f64::from(x + w - 1), f64::from(y + h - 1))
    );
    assert_eq!(
        (f64::from(x) + window.dot.0, f64::from(y) + window.dot.1),
        (1919.0, 1079.0)
    );

    overlay.set_gaze(gaze(-400.0, 500.0, 1.0), &monitors, 0.0); // left of every monitor
    assert_eq!(overlay.visible_monitor(), Some(0));
    let window = &overlay.windows()[&0];
    assert!(window.off_screen);
    assert_eq!(window.geometry().0, 0);
    assert!(window.render().is_some());
    overlay.set_gaze(gaze(f64::NAN, 1.0, 2.0), &monitors, 0.0);
    assert_eq!(overlay.visible_monitor(), None);
}

#[test]
fn test_overlay_follows_settings_and_state() {
    let monitors = monitors();
    let mut overlay = GazeOverlay::new(true, 2500);
    let mut inputs = OverlayInputs {
        monitors: &monitors,
        show_gaze_overlay: false,
        state: TrackingState::Tracking,
        gaze: gaze(100.0, 100.0, 0.0),
        now: 0.0,
    };
    overlay.update(&inputs);
    assert!(!overlay.enabled()); // default setting is off
    assert!(overlay.windows().is_empty());

    inputs.show_gaze_overlay = true;
    overlay.update(&inputs);
    assert!(overlay.enabled());
    inputs.gaze = gaze(100.0, 100.0, 1.0);
    overlay.update(&inputs);
    assert_eq!(overlay.visible_monitor(), Some(0));

    inputs.state = TrackingState::Paused;
    overlay.update(&inputs);
    assert_eq!(overlay.visible_monitor(), None);

    inputs.show_gaze_overlay = false;
    overlay.update(&inputs);
    assert!(!overlay.enabled());
    assert!(overlay.windows().is_empty());
}

#[test]
fn test_overlay_starts_enabled_from_settings() {
    assert!(GazeOverlay::from_settings(true).enabled());
    assert!(!GazeOverlay::from_settings(false).enabled());
}

#[test]
fn test_overlay_hides_stale_dot() {
    let monitors = monitors();
    let mut overlay = GazeOverlay::new(false, 10);
    overlay.set_enabled(true);
    overlay.set_gaze(gaze(100.0, 100.0, 0.0), &monitors, 100.0);
    assert_eq!(overlay.visible_monitor(), Some(0));
    overlay.tick(100.005);
    assert_eq!(overlay.visible_monitor(), Some(0));
    overlay.set_gaze(gaze(100.0, 100.0, 1.0), &monitors, 100.008); // an update restarts the timer
    overlay.tick(100.015);
    assert_eq!(overlay.visible_monitor(), Some(0));
    overlay.tick(100.019);
    assert_eq!(overlay.visible_monitor(), None);
}

#[test]
fn test_overlay_follows_layout_changes() {
    let mut layout = monitors();
    let mut overlay = enabled_overlay();
    overlay.set_gaze(gaze(2500.0, 100.0, 0.0), &layout, 0.0);
    assert_eq!(overlay.visible_monitor(), Some(1));
    // The right monitor was unplugged; the controller's layout changes a moment later. The next
    // gaze update picks it up without any extra call.
    layout = vec![left()];
    overlay.set_gaze(gaze(2500.0, 100.0, 1.0), &layout, 0.0);
    assert_eq!(overlay.visible_monitor(), Some(0)); // off-screen marker on the nearest monitor
    assert_eq!(overlay.windows().keys().copied().collect::<Vec<_>>(), [0]);

    overlay.on_screens_changed(); // the system reported a screen change: windows dropped
    assert!(overlay.windows().is_empty());
    assert_eq!(overlay.visible_monitor(), None);
    overlay.set_gaze(gaze(10.0, 10.0, 2.0), &layout, 0.0);
    assert_eq!(overlay.visible_monitor(), Some(0));
}

#[test]
fn test_overlay_shows_a_viewport_for_the_visible_window() {
    // egui runs headless: without a backend the viewport is embedded in a window.
    let monitors = monitors();
    let mut overlay = enabled_overlay();
    let ctx = egui::Context::default();
    let inputs = OverlayInputs {
        monitors: &monitors,
        show_gaze_overlay: true,
        state: TrackingState::Tracking,
        gaze: gaze(2500.0, 400.0, 1.0),
        now: 0.0,
    };
    for _ in 0..2 {
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            overlay.show(ui.ctx(), &inputs)
        });
        output.textures_delta.clear(); // nothing renders the textures
    }
    assert_eq!(overlay.visible_monitor(), Some(1));
}
