//! Tests of `countdown.rs` (the countdown tests of tests/test_ui_basic.py).
//!
//! Not ported:
//! * `test_every_widget_uses_the_system_font`: Qt font families; egui draws every text with the
//!   fonts of its own style, so there is nothing per widget to check.
//! * the window flag checks of `test_countdown_never_takes_focus` read the Qt flags; the same
//!   promises are checked on the egui `ViewportBuilder`.
//! * `toast.grab()` (painting a live widget); the ring painting is tested on `render_ring` and
//!   the toast is drawn by a headless smoke test.
//! * the fake clock is the `now` argument.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;

const T0: f64 = 100.0;

/// Fixed-size text: every character is 10 points wide, lines are 20 and 14 points high.
struct FixedMeasure;

impl TextMeasure for FixedMeasure {
    fn title_width(&self, text: &str) -> f64 {
        10.0 * text.chars().count() as f64
    }

    fn hint_width(&self, text: &str) -> f64 {
        8.0 * text.chars().count() as f64
    }

    fn title_height(&self) -> f64 {
        20.0
    }

    fn hint_height(&self) -> f64 {
        14.0
    }
}

fn left() -> Monitor {
    Monitor {
        primary: true,
        ..Monitor::new(0, "Left", Rect::new(0, 0, 1920, 1080))
    }
}

fn right() -> Monitor {
    Monitor::new(1, "Right", Rect::new(1920, 0, 1920, 1080))
}

/// Settings as after the setup assistant: before it, a walk-away lock only notifies.
fn finished_setup(action: &str) -> CountdownSettings {
    let mut settings = Settings::default();
    settings.general.first_run_done = true;
    settings.presence.action = action.to_string();
    CountdownSettings::from_settings(&settings)
}

fn toast(action: &str) -> CountdownToast {
    CountdownToast::new(None, finished_setup(action))
}

#[test]
fn test_countdown_text_and_ticks() {
    let mut toast = toast("lock");
    toast.show_warning(8.0, T0);
    assert!(toast.visible());
    assert!(toast.active());
    assert_eq!(
        toast.text(T0),
        "Locking in 8 s — move the mouse or look at the camera to cancel"
    );
    let now = T0 + 1.2;
    assert_eq!(toast.title_text(now), "Locking in 7 s");
    assert!((toast.remaining(now) - 6.8).abs() < 1e-9);
    assert!(toast.text(now).starts_with("Locking in 7 s"));
    let now = now + 7.0;
    assert_eq!(toast.title_text(now), "Locking now…");
    assert_eq!(toast.seconds_left(now), 0);
    assert_eq!(toast.fraction(now), 0.0); // empty ring
    toast.cancel(); // away_cancelled
    assert!(!toast.visible());
    assert!(!toast.active());
    assert_eq!(toast.remaining(now), 0.0);
}

#[test]
fn test_countdown_text_per_presence_action() {
    for (action, title) in [
        ("lock", "Locking in 8 s"),
        ("lock_and_display_off", "Locking in 8 s"),
        ("display_off", "Turning displays off in 8 s"),
        ("notify", "Marking you as away in 8 s"),
    ] {
        let mut toast = toast(action);
        toast.show_warning(7.5, T0);
        assert!(toast.visible());
        assert_eq!(toast.title_text(T0), title);
        assert!(toast.text(T0).starts_with(&format!("{title} — ")));
    }
}

#[test]
fn test_countdown_announces_a_notification_until_the_setup_is_finished() {
    // Regression (r2-docs-02): the walk-away action only notifies before the setup assistant was
    // finished, yet the toast said "Locking in 10 s".
    for action in ["lock", "lock_and_display_off", "display_off"] {
        let mut settings = finished_setup(action);
        settings.first_run_done = false;
        let mut toast = CountdownToast::new(None, settings.clone());
        toast.show_warning(7.5, T0);
        assert_eq!(toast.action(), "notify");
        assert_eq!(toast.title_text(T0), "Marking you as away in 8 s");

        settings.first_run_done = true; // the assistant was finished meanwhile
        toast.on_settings_changed(&settings);
        assert_eq!(toast.action(), action);
        assert!(!toast.title_text(T0).starts_with("Marking you as away"));
    }
}

/// A platform that reports `input_idle` as given and counts the questions.
struct CapabilityPlatform {
    input_idle: bool,
    calls: AtomicUsize,
}

impl PlatformServices for CapabilityPlatform {
    fn capabilities(&self) -> HashMap<&'static str, bool> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        HashMap::from([("input_idle", self.input_idle)])
    }
}

#[test]
fn test_countdown_offers_only_what_can_cancel_it() {
    // Regression (r2-docs-04): on Wayland outside GNOME keyboard and mouse use is not observed,
    // so only looking at the camera cancels the countdown there.
    let platform = Arc::new(CapabilityPlatform {
        input_idle: false,
        calls: AtomicUsize::new(0),
    });
    let mut toast = CountdownToast::new(Some(platform.clone()), finished_setup("lock"));
    toast.show_warning(5.0, T0);
    assert_eq!(toast.hint_text(), "Look at the camera to cancel");
    assert!(toast.text(T0).ends_with(" — look at the camera to cancel"));
    assert_eq!(platform.calls.load(Ordering::SeqCst), 1); // asked once, not on every repaint

    let supported = Arc::new(CapabilityPlatform {
        input_idle: true,
        calls: AtomicUsize::new(0),
    });
    let toast = CountdownToast::new(Some(supported), finished_setup("lock"));
    assert_eq!(
        toast.hint_text(),
        "Move the mouse or look at the camera to cancel"
    );
}

#[test]
fn test_countdown_is_centred_on_a_monitor() {
    let monitors = [left(), right()];
    let mut toast = toast("lock");
    toast.show_warning(10.0, T0);
    for (cursor, expected) in [
        (Some((2500.0, 100.0)), right()),
        (Some((300.0, 100.0)), left()),
        (None, left()), // no cursor: the primary monitor
        (Some((-100.0, -100.0)), left()),
    ] {
        let env = ToastEnv {
            monitors: &monitors,
            screens: &monitors,
            cursor,
        };
        toast.place(&env, &FixedMeasure);
        let geometry = toast.geometry().unwrap();
        let centre = (
            f64::from(geometry.x) + f64::from(geometry.w) / 2.0,
            f64::from(geometry.y) + f64::from(geometry.h) / 2.0,
        );
        assert!(expected.rect.contains(centre.0, centre.1));
        assert!((centre.0 - expected.rect.center().0).abs() <= 1.0);
        assert!(geometry.w < expected.rect.w);
    }
}

#[test]
fn test_countdown_is_sized_around_its_text() {
    let monitors = [left()];
    let env = ToastEnv {
        monitors: &monitors,
        screens: &monitors,
        cursor: None,
    };
    let mut toast = toast("lock");
    toast.show_warning(3.0, T0);
    toast.place(&env, &FixedMeasure);
    let geometry = toast.geometry().unwrap();
    // The widest text is the hint (46 characters at 8) against "Locking in 10 s" at 10.
    let widest = 8.0 * toast.hint_text().chars().count() as f64;
    assert_eq!(
        f64::from(geometry.w),
        (22.0 + 52.0 + 18.0 + widest + 26.0).ceil() * ui_scale_of(&monitors[0])
    );
    assert_eq!(geometry.h, 92_i32.max((34.0 + 44.0) as i32));
}

fn ui_scale_of(monitor: &Monitor) -> f64 {
    util::ui_scale(Some(monitor.scale))
}

#[test]
fn test_countdown_without_monitors_uses_the_primary_screen() {
    let screens = [left(), right()];
    let env = ToastEnv {
        monitors: &[],
        screens: &screens,
        cursor: None,
    };
    let mut toast = toast("lock");
    toast.show_warning(3.0, T0);
    toast.place(&env, &FixedMeasure);
    let geometry = toast.geometry().unwrap();
    let centre = (
        f64::from(geometry.x) + f64::from(geometry.w) / 2.0,
        f64::from(geometry.y) + f64::from(geometry.h) / 2.0,
    );
    assert!(left().rect.contains(centre.0, centre.1));
    // Nothing known at all: nowhere to place it.
    let nowhere = ToastEnv {
        monitors: &[],
        screens: &[],
        cursor: None,
    };
    let mut lost = self::toast("lock");
    lost.show_warning(3.0, T0);
    lost.place(&nowhere, &FixedMeasure);
    assert_eq!(lost.geometry(), None);
}

#[test]
fn test_countdown_follows_presence_action() {
    let mut settings = finished_setup("lock");
    let mut toast = CountdownToast::new(None, settings.clone());
    settings.action = "display_off".to_string();
    toast.on_settings_changed(&settings);
    toast.show_warning(4.5, T0);
    assert_eq!(toast.title_text(T0), "Turning displays off in 5 s");

    settings.action = "notify".to_string();
    settings.require_input_idle = false;
    toast.on_settings_changed(&settings);
    assert!(toast.visible());
    assert_eq!(
        toast.text(T0),
        "Marking you as away in 5 s — look at the camera to cancel"
    );

    settings.action = "none".to_string(); // nothing will happen, so nothing to announce
    toast.on_settings_changed(&settings);
    assert!(!toast.visible());
    toast.show_warning(5.0, T0);
    assert!(!toast.visible());
}

#[test]
fn test_countdown_repeated_warnings_keep_the_ring_total() {
    let mut toast = toast("lock");
    toast.show_warning(10.0, T0);
    let now = T0 + 3.0;
    toast.show_warning(7.0, now); // the controller repeating its warning
    assert!((toast.total - 10.0).abs() < 1e-9);
    toast.show_warning(20.0, now); // a later deadline: a new countdown
    assert!((toast.total - 20.0).abs() < 1e-9);
    toast.show_warning(f64::NAN, now);
    assert_eq!(toast.remaining(now), 0.0);
}

#[test]
fn test_countdown_hides_when_state_changes() {
    for state in [
        TrackingState::Away,
        TrackingState::Paused,
        TrackingState::Privacy,
        TrackingState::Locked,
        TrackingState::Yielded,
        TrackingState::Calibrating,
    ] {
        let mut toast = toast("lock");
        toast.show_warning(5.0, T0);
        toast.on_state_changed(state);
        assert!(!toast.visible(), "{state:?}");
    }
    let mut toast = toast("lock");
    toast.show_warning(5.0, T0);
    toast.on_state_changed(TrackingState::Tracking);
    assert!(toast.visible());
}

#[test]
fn test_countdown_never_takes_focus() {
    let builder = toast("lock").viewport_builder();
    assert_eq!(builder.active, Some(false));
    assert_eq!(builder.window_level, Some(egui::WindowLevel::AlwaysOnTop));
    assert_eq!(builder.decorations, Some(false));
    assert_eq!(builder.taskbar, Some(false));
    assert_eq!(builder.transparent, Some(true));
}

#[test]
fn test_countdown_fades_in() {
    let mut toast = toast("lock");
    toast.show_warning(5.0, T0);
    assert_eq!(toast.opacity(T0), 0.0);
    let half = toast.opacity(T0 + FADE_S / 2.0);
    assert!(0.5 < half && half < 1.0, "{half}"); // ease-out
    assert_eq!(toast.opacity(T0 + 1.0), 1.0);
}

#[test]
fn test_countdown_ring_shrinks_clockwise_from_the_top() {
    let side = 104;
    let full = render_ring(side, 1.0).unwrap();
    let half = render_ring(side, 0.5).unwrap();
    let empty = render_ring(side, 0.0).unwrap();
    let top = (side / 2, 3);
    let right = (side - 4, side / 2);
    let left = (3, side / 2);
    // The faint track is all that is left of an empty ring.
    assert_eq!(empty.pixel(top.0, top.1)[3], 34);
    // Half the time left: the right half (clockwise from 12 o'clock) is still accent coloured.
    assert!(half.pixel(right.0, right.1)[3] > 200);
    assert_eq!(half.pixel(left.0, left.1)[3], 34);
    assert!(full.pixel(left.0, left.1)[3] > 200);
    assert_eq!(render_ring(0, 0.5), None);
}

#[test]
fn test_countdown_draws_a_viewport() {
    // egui runs headless: without a backend the viewport is embedded in a window.
    let monitors = [left(), right()];
    let settings = finished_setup("lock");
    let mut toast = CountdownToast::new(None, settings.clone());
    let mut inputs = CountdownInputs {
        env: ToastEnv {
            monitors: &monitors,
            screens: &monitors,
            cursor: Some((100.0, 100.0)),
        },
        state: TrackingState::Tracking,
        settings: &settings,
        warning: Some(8.0),
        cancelled: false,
        now: T0,
    };
    let ctx = egui::Context::default();
    for frame in 0..3 {
        inputs.now = T0 + f64::from(frame) * 0.3;
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            toast.show(ui.ctx(), &inputs)
        });
        output.textures_delta.clear(); // nothing renders the textures
        inputs.warning = None;
    }
    assert!(toast.visible());
    assert!(toast.geometry().is_some());
    inputs.cancelled = true;
    toast.update(&inputs);
    assert!(!toast.visible());
}
