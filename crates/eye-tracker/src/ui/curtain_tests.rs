//! Tests of `curtain.rs` (the curtain tests of tests/test_ui_basic.py).
//!
//! Not ported as such:
//! * `grab()` of a live widget (the window is drawn by `show`, which a headless smoke test runs);
//! * the Qt key events and the real button click of `test_curtain_escape_and_button_dismiss`: the
//!   key decision (`CurtainWindow::handles_key`) and `dismiss()` (what both the key and the
//!   button call) are tested instead;
//! * `test_curtain_falls_back_to_qt_screens` uses the screen list of the inputs, not `QScreen`s;
//! * the `winId` of the activated window (the owner maps the `ActivateWindow` viewport to its
//!   native handle).

use super::*;

const NOW: f64 = 10.0;

fn first() -> Monitor {
    Monitor {
        primary: true,
        ..Monitor::new(0, "A", Rect::new(0, 0, 1024, 768))
    }
}

fn second() -> Monitor {
    Monitor::new(1, "B", Rect::new(1024, 0, 1280, 720))
}

fn rects(curtain: &PrivacyCurtain) -> Vec<Rect> {
    curtain.windows().iter().map(|w| w.monitor.rect).collect()
}

/// A curtain that sees `monitors` and `screens`, with the guard configured as in `_guarded`.
fn curtain_on(monitors: &[Monitor], screens: &[Monitor]) -> PrivacyCurtain {
    let mut curtain = PrivacyCurtain::new();
    curtain.set_environment(monitors, screens, None);
    curtain
}

#[test]
fn test_curtain_covers_every_screen() {
    let monitors = [first(), second()];
    let mut curtain = curtain_on(&monitors, &monitors);
    curtain.on_settings_changed(true, "curtain");
    curtain.on_guard_changed(true, NOW);
    assert!(curtain.is_showing());
    let windows = curtain.windows().to_vec();
    assert_eq!(rects(&curtain), [monitors[0].rect, monitors[1].rect]);
    for window in &windows {
        let flags = window.viewport_builder();
        assert_eq!(flags.window_level, Some(egui::WindowLevel::AlwaysOnTop));
        assert_eq!(flags.decorations, Some(false));
        assert_eq!(flags.transparent, Some(false)); // fully opaque
        assert_eq!(flags.active, Some(true)); // may take the keyboard
    }
    curtain.on_guard_changed(true, NOW); // repeated trigger: same windows
    assert_eq!(curtain.windows(), windows);
    curtain.on_guard_changed(false, NOW);
    assert!(!curtain.is_showing());
}

#[test]
fn test_curtain_also_covers_screens_the_controller_has_not_seen() {
    // Right after a hot-plug the controller's layout is still the old one.
    let far = Monitor {
        primary: true,
        ..Monitor::new(0, "far", Rect::new(50_000, 0, 1280, 720))
    };
    let screens = [first(), second()];
    let mut curtain = curtain_on(std::slice::from_ref(&far), &screens);
    curtain.show_curtain(NOW);
    let mut expected = vec![far.rect, screens[0].rect, screens[1].rect];
    let mut found = rects(&curtain);
    expected.sort_by_key(|r| (r.x, r.y));
    found.sort_by_key(|r| (r.x, r.y));
    assert_eq!(found, expected);
    // The extra screens are numbered after the controller's monitors.
    let indices: Vec<i32> = curtain.windows().iter().map(|w| w.monitor.index).collect();
    assert_eq!(indices, [0, 1, 2]);
    // Degenerate screens are skipped.
    let empty = Monitor::new(3, "empty", Rect::new(0, 0, 0, 0));
    let mut curtain = curtain_on(&[], std::slice::from_ref(&empty));
    curtain.show_curtain(NOW);
    assert!(!curtain.is_showing());
}

#[test]
fn test_curtain_rebuilds_after_a_screen_change() {
    let (a, b) = (first(), second());
    let mut curtain = curtain_on(std::slice::from_ref(&a), std::slice::from_ref(&a));
    curtain.show_curtain(NOW);
    assert_eq!(curtain.windows().len(), 1);
    // A monitor was plugged in.
    curtain.set_environment(&[a.clone(), b.clone()], &[a.clone(), b.clone()], None);
    curtain.tick(NOW);
    let monitors: Vec<&Monitor> = curtain.windows().iter().map(|w| &w.monitor).collect();
    assert_eq!(monitors, [&a, &b]);
    curtain.hide_curtain();
    curtain.on_screens_changed(); // nothing shown: nothing rebuilt
    curtain.tick(NOW);
    assert!(!curtain.is_showing());
}

#[test]
fn test_curtain_escape_and_button_dismiss() {
    let monitors = [first(), second()];
    let mut curtain = curtain_on(&monitors, &monitors);
    curtain.on_settings_changed(true, "curtain");
    curtain.on_guard_changed(true, NOW);
    assert!(!CurtainWindow::handles_key(Key::A)); // other keys do nothing
    assert!(CurtainWindow::handles_key(Key::Escape));
    assert!(curtain.is_showing());
    curtain.dismiss(); // the Escape key
    assert!(!curtain.is_showing());
    assert_eq!(curtain.drain_actions(), [CurtainAction::Dismissed]);
    curtain.dismiss(); // nothing shown: no second signal
    assert!(curtain.drain_actions().is_empty());

    curtain.on_guard_changed(true, NOW);
    curtain.dismiss(); // the button
    assert!(!curtain.is_showing());
    assert_eq!(curtain.drain_actions(), [CurtainAction::Dismissed]);
}

#[test]
fn test_curtain_shows_whatever_the_controller_asks_for() {
    let monitors = [first(), second()];
    let mut curtain = curtain_on(&monitors, &monitors);
    // "lock" that could not lock: the controller falls back to the curtain.
    curtain.on_settings_changed(true, "lock");
    curtain.on_guard_changed(true, NOW);
    assert!(curtain.is_showing());

    curtain.on_state_changed(TrackingState::Privacy); // the guard can never clear now
    assert!(!curtain.is_showing());

    curtain.on_guard_changed(true, NOW);
    assert!(curtain.is_showing());
    curtain.on_settings_changed(true, "notify"); // switched to notifications only
    assert!(!curtain.is_showing());

    curtain.on_settings_changed(true, "curtain");
    curtain.on_guard_changed(true, NOW);
    assert!(curtain.is_showing());
    curtain.on_settings_changed(false, "curtain");
    assert!(!curtain.is_showing());
}

#[test]
fn test_curtain_follows_the_inputs_it_is_given() {
    let monitors = [first(), second()];
    let mut curtain = PrivacyCurtain::new();
    let mut inputs = CurtainInputs {
        monitors: &monitors,
        screens: &monitors,
        cursor: None,
        guard_active: false,
        state: TrackingState::Tracking,
        shoulder_guard: true,
        guard_action: "curtain",
        now: NOW,
    };
    curtain.update(&inputs);
    assert!(!curtain.is_showing());
    inputs.guard_active = true;
    curtain.update(&inputs);
    assert!(curtain.is_showing());
    inputs.state = TrackingState::Paused;
    curtain.update(&inputs);
    assert!(!curtain.is_showing());
    // Still guarded, but dismissed by the state: only a new trigger shows it again.
    curtain.update(&inputs);
    assert!(!curtain.is_showing());
    inputs.state = TrackingState::Tracking;
    inputs.guard_active = false;
    curtain.update(&inputs);
    inputs.guard_active = true;
    curtain.update(&inputs);
    assert!(curtain.is_showing());
    inputs.guard_action = "notify";
    curtain.update(&inputs);
    assert!(!curtain.is_showing());
}

#[test]
fn test_curtain_falls_back_to_the_system_screens() {
    let screens = [first(), second()];
    let mut curtain = curtain_on(&[], &screens);
    curtain.show_curtain(NOW);
    assert_eq!(curtain.windows().len(), screens.len());
    curtain.hide_curtain();
    assert!(!curtain.is_showing());
}

#[test]
fn test_curtain_promises_esc_only_with_keyboard_focus() {
    let monitors = [first(), second()];
    let mut curtain = curtain_on(&monitors, &monitors);
    curtain.on_settings_changed(true, "curtain");
    curtain.on_guard_changed(true, NOW);
    // Until the activation is confirmed nothing promises that Esc works.
    assert!(curtain.windows().iter().all(|w| w.hint == HINT_NO_KEYBOARD));
    curtain.set_window_focus(0, true); // the system grants the activation
    assert!(curtain.has_keyboard());
    assert!(curtain.windows().iter().all(|w| w.hint == HINT));
    assert!(
        curtain.windows()[0]
            .accessible_description()
            .contains("Press Esc")
    );
    curtain.ensure_keyboard(NOW + FOCUS_RECHECK_S); // the delayed check: focused, so no request
    assert!(curtain.drain_actions().is_empty());
}

#[test]
fn test_curtain_refused_focus_asks_the_platform_and_says_click() {
    // Windows' foreground lock refuses a camera-triggered activation (ui_app-04).
    let monitors = [first(), second()];
    let mut curtain = curtain_on(&monitors, &monitors);
    curtain.on_settings_changed(true, "curtain");
    curtain.on_guard_changed(true, NOW);
    curtain.tick(NOW + FOCUS_RECHECK_S); // what the timer runs FOCUS_RECHECK_MS later
    assert_eq!(
        curtain.drain_actions(),
        [CurtainAction::ActivateWindow {
            viewport: CurtainWindow::viewport_id(0),
            pid: std::process::id(),
        }]
    );
    for window in curtain.windows() {
        assert_eq!(window.hint, HINT_NO_KEYBOARD);
        assert!(window.accessible_description().contains("Click Dismiss"));
    }
    // Clicking the curtain gives it the keyboard: from then on Esc works.
    curtain.set_window_focus(1, true);
    assert!(curtain.windows().iter().all(|w| w.hint == HINT));
    curtain.hide_curtain();
    curtain.ensure_keyboard(NOW + 1.0); // a late timer after hiding does nothing
    assert!(curtain.drain_actions().is_empty());
}

#[test]
fn test_curtain_focuses_the_window_under_the_cursor() {
    let monitors = [first(), second()];
    let mut curtain = curtain_on(&monitors, &monitors);
    curtain.set_environment(&monitors, &monitors, Some((1500.0, 100.0)));
    curtain.show_curtain(NOW);
    assert_eq!(curtain.focus_command, Some(1));
    curtain.hide_curtain();
    curtain.set_environment(&monitors, &monitors, Some((-50.0, -50.0)));
    curtain.show_curtain(NOW);
    assert_eq!(curtain.focus_command, Some(0)); // not on any monitor: the first window
}

#[test]
fn test_curtain_content_is_centred() {
    let layout = content_layout(1000.0, 800.0, 1.0, 40.0, 20.0, 36.0);
    let block = 96.0 + 28.0 + 40.0 + 10.0 + 20.0 + 30.0 + 36.0;
    assert_eq!(layout.badge, (452.0, (800.0 - block) / 2.0, 96.0));
    assert_eq!(layout.title.1, 40.0);
    assert!((layout.button_top + 36.0 - (800.0 + block) / 2.0).abs() < 1e-9);
    // A window shorter than its content starts at the top.
    assert_eq!(
        content_layout(1000.0, 100.0, 1.0, 40.0, 20.0, 36.0).badge.1,
        0.0
    );
}

#[test]
fn test_curtain_draws_one_viewport_per_monitor() {
    // egui runs headless: without a backend the viewports are embedded in windows.
    let monitors = [first(), second()];
    let mut curtain = PrivacyCurtain::new();
    let inputs = CurtainInputs {
        monitors: &monitors,
        screens: &monitors,
        cursor: None,
        guard_active: true,
        state: TrackingState::Tracking,
        shoulder_guard: true,
        guard_action: "curtain",
        now: NOW,
    };
    let ctx = egui::Context::default();
    for _ in 0..2 {
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            assert!(curtain.show(ui.ctx(), &inputs).is_empty());
        });
        output.textures_delta.clear(); // nothing renders the textures
    }
    assert!(curtain.is_showing());
}
