//! Tests for `engine::controller::Controller` (tests/test_controller.py).
//!
//! Everything the controller touches is faked (`testing::controller_fakes`): the vision
//! worker (observations are pushed by the test and handled at once), the platform
//! (lock/display/window calls are recorded, never performed; it also plays the cursor), the
//! clock, the monitor layout and the hotkey manager. Config and calibration files live in a
//! temp dir.
//!
//! Not ported:
//!
//! * `test_qt_cursor_reads_position`: there is no Qt cursor any more; the cursor is
//!   `PlatformServices::cursor_position`, tested with each backend's fake (windows_tests,
//!   macos_tests, linux_tests).
//! * `test_qt_monitors_offscreen`: Qt's screens are replaced by `platform::monitors()`; its
//!   ordering and filtering are tested in `platform/mod.rs` (same test name) on fake displays,
//!   since a headless test has no display to read.
//! * `test_doctor_reads_the_hotkeys_from_the_running_app`: `diagnostics` is not ported yet
//!   (its `_hotkey_registration` does not exist); the `status()["hotkeys"]` it reads is
//!   covered by `test_status_says_which_hotkeys_the_system_took`.

use std::collections::HashSet;
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{Value, json};

use super::*;
use crate::config::Settings;
use crate::gaze::calibration::{CalibrationSample, samples_to_arrays};
use crate::gaze::learning::refit_model;
use crate::gaze::model::GazeModel;
use crate::gaze::store::{CalibrationLibrary, load_calibration, save_calibration};
use crate::panes::registry::PaneRegistry;
use crate::panes::types::{Pane, PaneError, PaneId, PaneProvider, PaneSnapshot};
use crate::panes::worker::{DetectResult, PaneEvent, PaneWorker, PaneWorkerOptions};
use crate::paths;
use crate::platform::hotkeys::{format_hotkey, parse_hotkey};
use crate::testing::app_dirs;
use crate::testing::controller_fakes::*;
use crate::testing::gaze_synth;
use crate::types::{
    AppIdentity, Monitor, Observation, Rect, TrackingState as S, WindowHandle, WindowRef,
    WorkerStats, layout_signature, virtual_bounds,
};
use crate::vision::Frame;

fn approx(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-6 * expected.abs() + 1e-12,
        "{actual} != approx {expected}"
    );
}

fn win(handle: isize, pid: u32, rect: Rect) -> WindowRef {
    WindowRef {
        handle: WindowHandle::Hwnd(handle),
        pid: Some(pid),
        rect: Some(rect),
    }
}

fn hwnd(handle: isize) -> WindowHandle {
    WindowHandle::Hwnd(handle)
}

fn opts(settings: Settings) -> Opts {
    Opts {
        settings: Some(settings),
        ..Opts::default()
    }
}

fn last_interval(h: &Harness) -> f64 {
    *h.worker().s().intervals.last().expect("an interval")
}

fn last_gate(h: &Harness) -> (bool, f64) {
    *h.worker().s().gates.last().expect("a gate")
}

/// Let every guard expire (mouse grace, typing grace, cooldown).
fn settle_mouse(h: &mut Harness) {
    h.tick(3.0);
}

fn obs(point: (f64, f64)) -> Observation {
    gaze_obs(point, 1)
}

fn cal() -> CalibrationData {
    make_calibration(&monitors(), None)
}

fn moves(h: &Harness) -> Vec<(i32, i32)> {
    h.p().cursor.moves.clone()
}

fn set_cursor(h: &Harness, pos: (i32, i32)) {
    h.p().cursor.position = pos;
}

fn calls(h: &Harness) -> Vec<Call> {
    h.p().calls.clone()
}

fn count(names: &[&str], name: &str) -> usize {
    names.iter().filter(|n| **n == name).count()
}

// ----------------------------------------------------------------------------- start
#[test]
fn test_start_configures_worker_and_hotkeys() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    assert_eq!(h.state(), S::Tracking);
    let w = h.worker();
    assert!(w.s().started);
    assert!(!w.s().active[0]); // camera kept closed until the state is known
    assert!(w.is_active());
    assert_eq!(w.s().max_faces, [1]);
    assert_eq!(last_gate(&h), (true, 2.0));
    approx(last_interval(&h), 1.0 / 4.0); // balanced "idle"
    assert_eq!(
        h.controller.backend_info(),
        (BACKEND.0.to_string(), BACKEND.1.to_string())
    );
    assert!(h.controller.is_calibrated());
    let expected: HashSet<String> = ["toggle_tracking", "toggle_privacy", "recalibrate"]
        .map(String::from)
        .into();
    assert_eq!(h.hotkeys.names(), expected);
    assert_eq!(h.state_changed(), [S::Tracking]);
}

#[test]
fn test_needs_calibration_without_calibration() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts {
        calibrated: false,
        ..Opts::default()
    });
    assert_eq!(h.state(), S::NeedsCalibration);
    assert!(h.worker().is_active()); // presence and the guard still use the camera
    assert!(h.calibration_required().is_empty());
    assert_eq!(h.controller.calibration_reason(), "not calibrated yet");
}

#[test]
fn test_start_paused_never_opens_camera() {
    let _dirs = app_dirs();
    let mut s = make_settings();
    s.general.start_paused = true;
    let h = make_controller(opts(s));
    assert_eq!(h.state(), S::Paused);
    assert!(!h.worker().s().active.contains(&true));
}

#[test]
fn test_locked_session_at_start_keeps_camera_closed() {
    let _dirs = app_dirs();
    let platform = Arc::new(FakePlatform::new());
    platform.p().locked = Some(true);
    let h = make_controller(Opts {
        platform: Some(platform),
        ..Opts::default()
    });
    assert_eq!(h.state(), S::Locked);
    assert!(!h.worker().s().active.contains(&true));
}

// ------------------------------------------------------------------------- switching
#[test]
fn test_switch_after_dwell_restores_cursor_and_window() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.p().foreground = Some(win(7, 42, Rect::new(2000, 100, 1200, 800)));
    h.tick(0.6); // first cursor poll; the focused window is remembered for monitor 1
    h.p().foreground = None;
    set_cursor(&h, (2600, 400)); // the user works on the right monitor ...
    h.tick(0.1);
    set_cursor(&h, (500, 500)); // ... and comes back to the left one
    h.tick(0.1);
    settle_mouse(&mut h);

    for _ in 0..3 {
        // 0.0, 0.1, 0.2 s of dwell
        h.push(obs(RIGHT_CENTRE), 0.1);
    }
    assert!(h.switched().is_empty());
    assert!(moves(&h).is_empty());

    h.push(obs(RIGHT_CENTRE), 0.1); // 0.3 s: dwell complete
    assert_eq!(h.switched(), [1]);
    assert_eq!(moves(&h), [(2600, 400)]); // the remembered position, not the centre
    assert!(calls(&h).contains(&Call::ActivateWindow(hwnd(7))));
    assert!(!h.platform.names().contains(&"window_at"));
    let gaze = h.gaze_changed().last().copied().flatten().expect("a gaze");
    approx(gaze.x, RIGHT_CENTRE.0);

    // The warp is not user activity, and the cursor now matches the gaze: no more switches.
    h.tick(0.1);
    h.feed(&obs(RIGHT_CENTRE), 1.0, 0.1);
    assert_eq!(h.switched(), [1]);
}

#[test]
fn test_switch_to_centre_focuses_window_under_cursor() {
    let _dirs = app_dirs();
    let mut s = make_settings();
    s.switching.cursor_target = "center".into();
    let mut h = make_controller(opts(s));
    h.p().window_under = Some(win(9, 1, Rect::new(1920, 0, 1920, 1080)));
    h.feed(&obs((2500.0, 300.0)), 0.5, 0.1);
    assert_eq!(h.switched(), [1]);
    assert_eq!(moves(&h), [(2880, 540)]);
    assert!(calls(&h).contains(&Call::WindowAt(2880, 540)));
    assert!(calls(&h).contains(&Call::ActivateWindow(hwnd(9))));
}

#[test]
fn test_switch_can_be_disabled() {
    let _dirs = app_dirs();
    let mut s = make_settings();
    s.switching.enabled = false;
    let mut h = make_controller(opts(s));
    h.feed(&obs(RIGHT_CENTRE), 1.0, 0.1);
    assert!(h.switched().is_empty());
    assert!(!h.gaze_changed().is_empty()); // the gaze is still estimated (overlay, preview)
}

#[test]
fn test_typing_guard_suppresses_switch_until_it_expires() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.p().key_idle = Some(0.1); // the user keeps typing on the left monitor
    h.feed(&obs(LEFT_CENTRE), 1.5, 0.1);
    h.p().key_idle = Some(1000.0); // typing stopped as the user looks right
    h.feed(&obs(RIGHT_CENTRE), 1.0, 0.1);
    assert!(h.switched().is_empty());
    h.feed(&obs(RIGHT_CENTRE), 1.2, 0.1); // the typing grace (2 s) is over
    assert_eq!(h.switched(), [1]);
}

#[test]
fn test_reading_pause_does_not_move_focus_to_the_reading_monitor() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.p().key_idle = Some(0.1); // typing on the left while reading the right monitor
    h.feed(&obs(RIGHT_CENTRE), 1.5, 0.1);
    h.p().key_idle = Some(1000.0); // a reading pause
    h.feed(&obs(RIGHT_CENTRE), 4.0, 0.1);
    assert!(h.switched().is_empty());
    h.feed(&obs(RIGHT_CENTRE), 2.5, 0.1); // reading grace (6 s) over
    assert_eq!(h.switched(), [1]);
}

#[test]
fn test_mouse_guard_suppresses_switch() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.tick(0.1);
    for i in 0..10 {
        // the user moves the mouse on the left monitor
        set_cursor(&h, (500 + 20 * i, 500));
        h.push(obs(RIGHT_CENTRE), 0.1);
        let now = h.clock.now();
        h.controller.tick(now);
    }
    assert!(h.switched().is_empty());
    h.feed(&obs(RIGHT_CENTRE), 1.6, 0.1); // mouse grace (1.5 s) over
    assert_eq!(h.switched(), [1]);
}

#[test]
fn test_refused_cursor_warp_notifies_once() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.p().cursor.allow = false;
    h.feed(&obs(RIGHT_CENTRE), 0.5, 0.1);
    h.feed(&obs(RIGHT_CENTRE), 2.0, 0.1);
    assert!(moves(&h).len() >= 2);
    assert_eq!(count_titles(&mut h, "Cannot move the cursor"), 1);
}

fn count_titles(h: &mut Harness, title: &str) -> usize {
    h.titles().iter().filter(|t| *t == title).count()
}

#[test]
fn test_wrong_switch_is_recorded_as_drift() {
    let _dirs = app_dirs();
    let mut s = make_settings();
    s.switching.cursor_target = "center".into();
    let mut h = make_controller(opts(s));
    h.tick(0.1);
    h.feed(&obs(RIGHT_CENTRE), 0.5, 0.1);
    assert_eq!(h.switched(), [1]);
    h.tick(0.1);
    set_cursor(&h, (400, 400)); // dragged straight back to the left monitor
    h.tick(0.1);
    assert_eq!(h.controller.drift.event_count(), 1);
}

// ------------------------------------------------------------------------- presence
#[test]
fn test_presence_warning_then_lock_then_return() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default()); // timeout 10 s, 5 s countdown, action "lock"
    h.feed(&no_face(), 4.5, 0.5);
    assert!(h.away_warning().is_empty());
    h.feed(&no_face(), 0.5, 0.5);
    let warnings = h.away_warning();
    assert_eq!(warnings.len(), 1);
    approx(warnings[0], 5.0);
    approx(last_interval(&h), 1.0 / 12.0); // sample fast to cancel quickly
    h.feed(&no_face(), 5.0, 0.5);
    assert_eq!(count(&h.platform.names(), "lock_screen"), 1);
    assert_eq!(h.state(), S::Away);
    assert_eq!(h.away_cancelled(), 1); // the countdown toast goes away
    approx(last_interval(&h), 1.0); // "away" rate

    h.push(obs((500.0, 500.0)), 0.1);
    assert_eq!(h.state(), S::Tracking);
    assert!(!h.platform.names().contains(&"wake_display"));
}

#[test]
fn test_display_off_and_wake_on_return() {
    let _dirs = app_dirs();
    let mut s = make_settings();
    s.presence.action = "display_off".into();
    let mut h = make_controller(opts(s));
    h.feed(&no_face(), 10.0, 0.5);
    assert_eq!(h.platform.names(), ["display_off"]);
    assert_eq!(h.state(), S::Away);
    h.push(obs((500.0, 500.0)), 0.1);
    assert_eq!(h.platform.names(), ["display_off", "wake_display"]);
    assert_eq!(h.state(), S::Tracking);
}

#[test]
fn test_input_cancels_countdown() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.tick(0.1);
    h.feed(&no_face(), 5.0, 0.5);
    assert_eq!(h.away_warning().len(), 1);
    set_cursor(&h, (800, 600)); // the user touches the mouse
    h.tick(0.1);
    assert_eq!(h.away_cancelled(), 1);
    h.feed(&no_face(), 5.0, 0.5);
    assert!(!h.platform.names().contains(&"lock_screen"));
}

#[test]
fn test_privacy_mode_releases_camera_and_freezes_presence() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.controller.set_privacy(true);
    assert_eq!(h.state(), S::Privacy);
    assert!(!h.worker().is_active());
    for _ in 0..120 {
        // a minute without a camera is never absence
        h.push(no_face(), 0.5); // late frames are ignored
        let now = h.clock.now();
        h.controller.tick(now);
    }
    assert!(h.away_warning().is_empty());
    assert!(calls(&h).is_empty());

    h.controller.set_privacy(false);
    assert_eq!(h.state(), S::Tracking);
    assert!(h.worker().is_active());
    h.feed(&no_face(), 4.0, 0.5);
    assert!(h.away_warning().is_empty()); // timers restarted, not resumed
}

#[test]
fn test_session_lock_pauses_and_unlock_resumes() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.p().locked = Some(true);
    h.tick(2.1);
    assert_eq!(h.state(), S::Locked);
    assert!(!h.worker().is_active());
    h.p().locked = Some(false);
    h.tick(2.1);
    assert_eq!(h.state(), S::Tracking);
    assert!(h.worker().is_active());
}

#[test]
fn test_session_lock_ignored_when_disabled() {
    let _dirs = app_dirs();
    let mut s = make_settings();
    s.privacy.pause_when_locked = false;
    let mut h = make_controller(opts(s));
    h.p().locked = Some(true);
    h.tick(2.1);
    assert_eq!(h.state(), S::Tracking);
}

// --------------------------------------------------------------------- camera yield
#[test]
fn test_yield_camera_to_other_app() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.p().camera_in_use = Some(true);
    h.tick(3.1);
    assert_eq!(h.state(), S::Yielded);
    assert_eq!(
        h.controller.yield_reason(),
        "Another app is using the camera"
    );
    assert!(!h.worker().is_active());
    assert_eq!(h.notify().last().expect("a notice").0, "Tracking paused");
    h.p().camera_in_use = Some(false);
    h.tick(3.1);
    assert_eq!(h.state(), S::Tracking);
    assert!(h.worker().is_active());
}

#[test]
fn test_pause_for_listed_app() {
    let _dirs = app_dirs();
    let mut s = make_settings();
    s.privacy.yield_camera = false;
    s.privacy.pause_for_apps = vec!["Zoom.exe".into()];
    let mut h = make_controller(opts(s));
    h.p().camera_in_use = Some(true); // ignored: yield_camera is off
    h.tick(3.1);
    assert_eq!(h.state(), S::Tracking);
    h.p().processes = ["zoom.exe", "explorer.exe"].map(String::from).into();
    h.tick(3.1);
    assert_eq!(h.state(), S::Yielded);
    assert_eq!(h.controller.yield_reason(), "Zoom.exe is running");
}

// ----------------------------------------------------------------------------- guard
#[test]
fn test_shoulder_guard_curtain() {
    let _dirs = app_dirs();
    let mut s = make_settings();
    s.privacy.shoulder_guard = true;
    s.privacy.guard_action = "curtain".into();
    let mut h = make_controller(opts(s));
    assert_eq!(h.worker().s().max_faces.last(), Some(&2));
    h.feed(&gaze_obs((500.0, 500.0), 2), 1.8, 0.1);
    assert!(h.guard_changed().is_empty());
    h.feed(&gaze_obs((500.0, 500.0), 2), 0.4, 0.1);
    assert_eq!(h.guard_changed(), [true]);
    assert!(h.controller.guard_active());
    h.feed(&gaze_obs((500.0, 500.0), 1), 1.7, 0.1);
    assert_eq!(h.guard_changed(), [true, false]);
    assert!(!h.platform.names().contains(&"lock_screen"));
}

#[test]
fn test_shoulder_guard_lock_action() {
    let _dirs = app_dirs();
    let mut s = make_settings();
    s.privacy.shoulder_guard = true;
    s.privacy.guard_action = "lock".into();
    let mut h = make_controller(opts(s));
    h.feed(&gaze_obs((500.0, 500.0), 2), 2.5, 0.1);
    assert_eq!(h.platform.names(), ["lock_screen"]);
    assert!(h.guard_changed().is_empty());
}

#[test]
fn test_privacy_mode_takes_the_curtain_down() {
    let _dirs = app_dirs();
    let mut s = make_settings();
    s.privacy.shoulder_guard = true;
    let mut h = make_controller(opts(s));
    h.feed(&gaze_obs((500.0, 500.0), 2), 2.5, 0.1);
    assert_eq!(h.guard_changed(), [true]);
    h.controller.set_privacy(true);
    assert_eq!(h.guard_changed(), [true, false]);
}

// --------------------------------------------------------------- calibration/layout
fn third() -> Monitor {
    Monitor::new(2, "third", Rect::new(3840, 0, 1920, 1080))
}

#[test]
fn test_layout_change_invalidates_calibration() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.monitors.lock().push(third());
    h.controller.refresh_monitors();
    assert_eq!(h.state(), S::NeedsCalibration);
    assert_eq!(h.calibration_required(), ["the monitor layout changed"]);
    assert_eq!(h.controller.monitors().len(), 3);

    h.monitors.lock().remove(2); // plugged back as it was: the calibration is valid again
    h.controller.refresh_monitors();
    assert_eq!(h.state(), S::Tracking);
    assert_eq!(h.calibration_required(), ["the monitor layout changed"]);
}

#[test]
fn test_begin_and_finish_calibration() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts {
        calibrated: false,
        ..Opts::default()
    });
    h.controller.begin_calibration();
    assert_eq!(h.state(), S::Calibrating);
    assert_eq!(last_gate(&h), (false, 2.0)); // every frame is a sample
    approx(last_interval(&h), 1.0 / 24.0);
    let face = no_face();
    for _ in 0..80 {
        // 40 s without a face: presence is frozen while calibrating
        h.push(face.clone(), 0.5);
        let now = h.clock.now();
        h.controller.tick(now);
    }
    assert_eq!(h.observations().last(), Some(&face));
    assert!(h.away_warning().is_empty());

    h.controller.finish_calibration(None); // cancelled
    assert_eq!(h.state(), S::NeedsCalibration);
    assert_eq!(last_gate(&h), (true, 2.0));

    h.controller.begin_calibration();
    h.controller.finish_calibration(Some(cal()));
    assert_eq!(h.state(), S::Tracking);
    let stored = load_calibration(&paths::calibration_file()).expect("stored");
    assert_eq!(stored.backend, BACKEND.0);
    assert!(h.controller.calibration().is_some());
}

#[test]
fn test_begin_calibration_turns_privacy_off() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.controller.set_privacy(true);
    h.controller.begin_calibration();
    assert!(!h.controller.privacy());
    assert_eq!(h.state(), S::Calibrating);
    assert!(h.worker().is_active());
}

#[test]
fn test_backend_change_revalidates_calibration() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    let mut new = h.controller.settings().clone();
    new.general.backend = "lite".into();
    h.controller.apply_settings(&new);
    let (source, backend) = *h.worker().s().reconfigures.last().expect("a reconfigure");
    assert!(!source);
    assert!(backend);
    // The worker still reports the old backend; the settings decide meanwhile.
    assert_eq!(h.controller.backend_info().0, "lite");
    assert_eq!(h.state(), S::NeedsCalibration);
    assert!(
        h.calibration_required()
            .last()
            .expect("a request")
            .contains("fake")
    );
}

#[test]
fn test_implicit_learning_refits_and_saves() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.tick(0.1);
    for i in 0..25 {
        // 25 "move the mouse, let it rest" moments
        let point = (300 + 40 * i, 400);
        set_cursor(&h, point);
        h.tick(0.1);
        h.push(obs((f64::from(point.0), f64::from(point.1))), 0.4);
    }
    let stored = load_calibration(&paths::calibration_file()).expect("stored");
    assert_eq!(stored.implicit_samples.len(), 25);
    assert!(stored.implicit_samples.iter().all(|s| s.point_id < 0));
    assert!(h.notify().is_empty()); // predictions agreed with the cursor: no drift alert
}

// ---------------------------------------------------------------------- settings/IPC
#[test]
fn test_apply_settings_persists_and_reconfigures() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    let mut new = h.controller.settings().clone();
    new.switching.dwell_ms = 800;
    new.privacy.shoulder_guard = true;
    new.camera.device = "1".into();
    new.performance.motion_threshold = 4.0;
    new.hotkeys.toggle_privacy = "ctrl+alt+x".into();
    h.controller.apply_settings(&new);

    let saved = Settings::load(&paths::settings_file());
    assert_eq!(saved.to_dict(), new.to_dict());
    assert_eq!(
        h.settings_changed()
            .last()
            .expect("settings")
            .switching
            .dwell_ms,
        800
    );
    assert_eq!(h.worker().s().max_faces.last(), Some(&2));
    assert_eq!(last_gate(&h), (true, 4.0));
    let (source, backend) = *h.worker().s().reconfigures.last().expect("a reconfigure");
    assert!(source);
    assert!(!backend);
    assert_eq!(
        h.hotkeys.combo("toggle_privacy").as_deref(),
        Some("ctrl+alt+x")
    );

    h.feed(&obs(RIGHT_CENTRE), 0.7, 0.1); // the longer dwell is in effect
    assert!(h.switched().is_empty());
    h.feed(&obs(RIGHT_CENTRE), 0.2, 0.1);
    assert_eq!(h.switched(), [1]);
}

#[test]
fn test_handle_command_covers_every_ipc_command() {
    let commands: HashSet<&str> = COMMANDS.into_iter().collect();
    let ipc: HashSet<&str> = crate::ipc::COMMANDS.iter().copied().collect();
    assert_eq!(commands, ipc);
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    let c = &mut h.controller;

    assert_eq!(c.handle_command("pause"), "ok");
    assert_eq!(c.state(), S::Paused);
    assert_eq!(c.handle_command("resume"), "ok");
    assert_eq!(c.state(), S::Tracking);
    assert_eq!(c.handle_command(" TOGGLE\n"), "ok");
    assert_eq!(c.state(), S::Paused);
    assert_eq!(c.handle_command("toggle"), "ok");
    assert_eq!(c.state(), S::Tracking);
    assert_eq!(c.handle_command("privacy-on"), "ok");
    assert_eq!(c.state(), S::Privacy);
    assert_eq!(c.handle_command("privacy-off"), "ok");
    assert_eq!(c.state(), S::Tracking);
    assert_eq!(c.handle_command("privacy-toggle"), "ok");
    assert!(c.privacy());
    assert_eq!(c.handle_command("privacy-toggle"), "ok");
    assert!(!c.privacy());
    assert_eq!(c.handle_command("calibrate"), "ok");
    assert_eq!(h.calibration_required(), ["ipc"]);

    let c = &mut h.controller;
    let status: Value = serde_json::from_str(&c.handle_command("status")).expect("JSON");
    assert_eq!(status["state"], "tracking");
    assert_eq!(status["calibrated"], true);
    assert_eq!(status["backend"], "fake");
    assert_eq!(status["monitors"], 2);

    for cmd in UI_COMMANDS {
        assert_eq!(c.handle_command(cmd), "ok");
    }
    assert_eq!(h.ui_requested(), UI_COMMANDS);

    let c = &mut h.controller;
    assert!(c.handle_command("explode").starts_with("error:"));
    assert!(c.handle_command("").starts_with("error:"));
}

/// r3-ux-docs-02: after a reboot or a silent upgrade the camera came back on.
#[test]
fn test_privacy_mode_survives_a_restart() {
    let _dirs = app_dirs();
    let mut first = make_controller(Opts::default());
    first.controller.set_privacy(true);
    first.controller.shutdown();
    let mut second = make_controller(Opts::default());
    assert!(second.controller.privacy());
    assert!(second.controller.privacy_restored());
    assert_eq!(second.state(), S::Privacy);
    assert!(!second.worker().s().active.contains(&true)); // the camera never opened
    let status: Value =
        serde_json::from_str(&second.controller.handle_command("status")).expect("JSON");
    assert_eq!(status["privacy"], true);
    second.controller.set_privacy(false);
    assert!(!second.controller.privacy_restored());
    second.controller.shutdown();
    let third = make_controller(Opts::default());
    assert!(!third.controller.privacy());
    assert_eq!(third.state(), S::Tracking);
}

#[test]
fn test_privacy_mode_is_forgotten_when_not_remembered() {
    let _dirs = app_dirs();
    let mut first = make_controller(Opts::default());
    first.controller.set_privacy(true);
    first.controller.shutdown();
    let mut forgetful = make_settings();
    forgetful.privacy.remember_privacy_mode = false;
    let mut second = make_controller(opts(forgetful));
    assert!(!second.controller.privacy());
    assert_eq!(second.state(), S::Tracking);
    second.controller.shutdown();
    // Remembering turned on later does not bring back the choice of a run that ignored it.
    let third = make_controller(Opts::default());
    assert!(!third.controller.privacy());
}

#[test]
fn test_a_damaged_privacy_state_file_counts_as_off() {
    let _dirs = app_dirs();
    std::fs::create_dir_all(paths::state_file().parent().expect("a parent")).expect("dir");
    std::fs::write(paths::state_file(), "{not json").expect("write");
    let h = make_controller(Opts::default());
    assert!(!h.controller.privacy());
    std::fs::write(paths::state_file(), r#"{"privacy_mode": "yes"}"#).expect("write");
    assert!(!make_controller(Opts::default()).controller.privacy());
}

/// r3-ux-docs-05: both are toggles, and pausing also stops walk-away detection.
#[test]
fn test_pause_and_privacy_hotkeys_say_what_they_did() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.press("toggle_tracking");
    let (title, message) = h.notify().last().cloned().expect("a notice");
    assert_eq!(title, "Tracking paused");
    assert!(message.contains("walk-away detection"));
    h.press("toggle_tracking");
    assert_eq!(
        h.notify().last().cloned().expect("a notice"),
        ("Tracking resumed".into(), "The camera is on again.".into())
    );
    h.press("toggle_privacy");
    let (title, message) = h.notify().last().cloned().expect("a notice");
    assert_eq!(title, "Privacy mode on");
    assert!(message.contains("The camera is off"));
    h.press("toggle_tracking"); // paused as well while private
    h.press("toggle_tracking");
    assert_eq!(
        h.notify().last().cloned().expect("a notice"),
        (
            "Tracking resumed".into(),
            "Privacy mode is still on: the camera stays off.".into()
        )
    );
    h.press("toggle_privacy");
    assert_eq!(
        h.notify().last().cloned().expect("a notice"),
        ("Privacy mode off".into(), "The camera is on again.".into())
    );
    // The same over the command socket (ctl); a command that changes nothing is silent.
    let before = h.notify().len();
    assert_eq!(h.controller.handle_command("resume"), "ok");
    assert_eq!(h.notify().len(), before);
    assert_eq!(h.controller.handle_command("privacy-on"), "ok");
    assert_eq!(h.notify().last().expect("a notice").0, "Privacy mode on");
    // Notifications turned off: none of these.
    let mut quiet = h.controller.settings().clone();
    quiet.general.notifications = false;
    h.controller.apply_settings(&quiet);
    let before = h.notify().len();
    h.press("toggle_privacy");
    assert_eq!(h.notify().len(), before);
}

#[test]
fn test_mode_confirmation_names_the_registered_hotkey() {
    let _dirs = app_dirs();
    // The fake manager reports parsed hotkeys, like a real one.
    let mut h = make_controller(Opts::default());
    assert_eq!(h.controller.handle_command("pause"), "ok");
    let label = format_hotkey(
        &parse_hotkey(&Settings::default().hotkeys.toggle_tracking).expect("a hotkey"),
        None,
    );
    assert!(
        h.notify()
            .last()
            .expect("a notice")
            .1
            .contains(&format!("until you resume ({label})."))
    );
}

// --------------------------------------------------------------------- rate/stats/etc
#[test]
fn test_rate_follows_activity() {
    let _dirs = app_dirs();
    let mut s = make_settings();
    s.switching.cursor_target = "center".into();
    let mut h = make_controller(opts(s));
    approx(last_interval(&h), 1.0 / 4.0); // idle
    h.push(obs(RIGHT_CENTRE), 0.1); // a switch is being considered
    approx(last_interval(&h), 1.0 / 12.0);
    h.feed(&obs(RIGHT_CENTRE), 1.5, 0.1);
    assert_eq!(h.switched(), [1]);
    approx(last_interval(&h), 1.0 / 4.0); // settled on the right monitor
    h.p().key_idle = Some(0.1);
    h.tick(0.1);
    approx(last_interval(&h), 1.0 / 2.0); // typing
    h.push(no_face(), 0.1);
    approx(last_interval(&h), 1.0 / 2.0); // no face: "noface" rate
    h.controller.begin_calibration();
    approx(last_interval(&h), 1.0 / 24.0);
}

#[test]
fn test_profile_change_updates_rate() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    let mut new = h.controller.settings().clone();
    new.performance.profile = "eco".into();
    h.controller.apply_settings(&new);
    approx(last_interval(&h), 1.0 / 2.0); // eco idle
}

fn failing_stats(message: &str) -> WorkerStats {
    WorkerStats {
        camera_open: false,
        last_error: Some(message.into()),
        ..WorkerStats::default()
    }
}

fn open_stats() -> WorkerStats {
    WorkerStats {
        camera_open: true,
        ..WorkerStats::default()
    }
}

#[test]
fn test_camera_error_from_worker_stats() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.on_stats(failing_stats("Camera 0 could not be opened"));
    assert_eq!(h.state(), S::CameraError);
    assert_eq!(
        h.notify().last().cloned().expect("a notice"),
        (
            "Camera unavailable".into(),
            "Camera 0 could not be opened".into()
        )
    );
    h.on_stats(open_stats());
    assert_eq!(h.state(), S::Tracking);
}

#[test]
fn test_stats_changed() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.tick(2.1);
    let all = h.stats_changed();
    assert!(!all.is_empty());
    let stats = all.last().expect("stats").clone();
    for key in [
        "fps",
        "target_fps",
        "inference_ms",
        "skip_ratio",
        "cpu_percent",
        "state",
    ] {
        assert!(stats.contains_key(key), "{key}");
    }
    assert_eq!(stats["backend"], "fake");
    assert_eq!(stats["state"], "tracking");
    approx(stats["target_fps"].as_f64().expect("a number"), 4.0);
    assert_eq!(h.controller.stats_snapshot(), stats);
}

fn frame() -> Frame {
    Frame::filled(4, 4, [0, 0, 0])
}

#[test]
fn test_preview_frames_are_forwarded() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.controller.set_preview(true, None);
    assert_eq!(h.worker().s().previews, [true]);
    approx(last_interval(&h), 1.0 / 24.0);
    h.on_preview(frame());
    assert_eq!(h.preview_frames(), [frame()]);
    h.controller.set_preview(false, None);
    h.on_preview(frame());
    assert_eq!(h.preview_frames().len(), 1);
}

#[test]
fn test_hotkeys_are_marshalled_to_the_main_thread() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    let toggle = h.hotkeys.callback("toggle_tracking");
    std::thread::spawn(move || toggle()).join().expect("thread");
    assert_eq!(h.state(), S::Tracking); // not handled on the hotkey thread
    h.process();
    assert_eq!(h.state(), S::Paused);

    h.press("toggle_privacy");
    assert_eq!(h.state(), S::Privacy);
    h.press("recalibrate");
    assert_eq!(h.calibration_required(), ["hotkey"]);
}

#[test]
fn test_observations_from_worker_thread() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    let o = obs((500.0, 500.0));
    let worker = h.worker();
    let sent = o.clone();
    std::thread::spawn(move || worker.push(sent))
        .join()
        .expect("thread");
    assert!(h.observations().is_empty());
    h.process();
    assert_eq!(h.observations(), [o]);
}

#[test]
fn test_hotkey_registration_failure_notifies() {
    let _dirs = app_dirs();
    let hotkeys = Arc::new(FakeHotkeys::new());
    hotkeys
        .s()
        .fail
        .insert(Settings::default().hotkeys.recalibrate); // this platform's default
    let mut h = make_controller(Opts {
        hotkeys: Some(hotkeys.clone()),
        ..Opts::default()
    });
    assert!(!hotkeys.names().contains("recalibrate"));
    let (title, message) = h.notify().last().cloned().expect("a notice");
    assert_eq!(title, "Hotkey unavailable");
    assert!(message.contains("used by another app"));
}

#[test]
fn test_hotkey_failure_notification_gives_the_managers_reason() {
    let _dirs = app_dirs();
    let hotkeys = Arc::new(FakeHotkeys::new());
    let mut s = make_settings();
    s.hotkeys.recalibrate = "ctrl+alt+t".into();
    hotkeys.s().fail.insert("ctrl+alt+t".into());
    hotkeys.s().errors.insert(
        "recalibrate".into(),
        "Ctrl+Alt+T is AltGr+T, which types '\u{20ba}' on the Turkish Q keyboard layout.".into(),
    );
    let mut h = make_controller(Opts {
        hotkeys: Some(hotkeys),
        ..opts(s)
    });
    assert_eq!(
        h.notify().last().cloned().expect("a notice"),
        (
            "Hotkey unavailable".into(),
            "Ctrl+Alt+T is AltGr+T, which types '\u{20ba}' on the Turkish Q keyboard layout. \
             Choose another in Settings \u{2192} Hotkeys."
                .into()
        )
    );
}

#[test]
fn test_hotkeys_disabled() {
    let _dirs = app_dirs();
    let mut s = make_settings();
    s.hotkeys.enabled = false;
    let mut h = make_controller(opts(s));
    assert!(h.hotkeys.names().is_empty());
    assert!(h.hotkeys.s().stopped);
    assert_eq!(
        Value::Object(h.controller.status())["hotkeys"],
        json!({"registered": [], "errors": {}})
    );
}

const IN_USE: &str = "⌃⌥⌘C is already in use by another application";

fn hotkey_notices(h: &mut Harness) -> Vec<String> {
    h.notify()
        .into_iter()
        .filter(|(title, _)| title == "Hotkey unavailable")
        .map(|(_, message)| message)
        .collect()
}

fn hotkey_status(h: &mut Harness) -> Value {
    Value::Object(h.controller.status())["hotkeys"].clone()
}

/// Regression (r2-docs-12): `doctor` can only learn from the running app why a hotkey
/// failed, e.g. because another application owns the combination.
#[test]
fn test_status_says_which_hotkeys_the_system_took() {
    let _dirs = app_dirs();
    let hotkeys = Arc::new(FakeHotkeys::new());
    hotkeys
        .s()
        .fail
        .insert(Settings::default().hotkeys.recalibrate);
    hotkeys
        .s()
        .errors
        .insert("recalibrate".into(), format!("{IN_USE}."));
    let mut h = make_controller(Opts {
        hotkeys: Some(hotkeys),
        ..Opts::default()
    });
    let expected = json!({
        "registered": ["toggle_privacy", "toggle_tracking"],
        "errors": {"recalibrate": IN_USE},
    });
    assert_eq!(hotkey_status(&mut h), expected);
    let status: Value = serde_json::from_str(&h.controller.handle_command("status")).expect("JSON");
    assert_eq!(status["hotkeys"], expected);

    // Released while Settings → Hotkeys records a shortcut: not a failure.
    h.controller.suspend_hotkeys(true);
    let suspended = hotkey_status(&mut h);
    assert_eq!(suspended["registered"], json!([]));
    let errors = suspended["errors"].as_object().expect("errors");
    let names: HashSet<&str> = errors.keys().map(String::as_str).collect();
    assert_eq!(
        names,
        HashSet::from(["toggle_tracking", "toggle_privacy", "recalibrate"])
    );
    assert!(errors.values().all(|reason| {
        reason
            .as_str()
            .expect("text")
            .contains("Settings → Hotkeys")
    }));
    h.controller.suspend_hotkeys(false);
    assert_eq!(hotkey_status(&mut h), expected);

    // A cleared hotkey is neither registered nor an error.
    let mut s = h.controller.settings().clone();
    s.hotkeys.recalibrate = String::new();
    h.controller.apply_settings(&s);
    assert_eq!(
        hotkey_status(&mut h),
        json!({"registered": ["toggle_privacy", "toggle_tracking"], "errors": {}})
    );
}

/// Regression (r2-ui-app-05): the settings dialog releases the hotkeys while a shortcut
/// field records and gives them back afterwards; an unchanged failure was announced again
/// every time.
#[test]
fn test_leaving_the_hotkey_fields_does_not_repeat_the_failure() {
    let _dirs = app_dirs();
    let hotkeys = Arc::new(FakeHotkeys::new());
    let default = Settings::default().hotkeys.recalibrate;
    hotkeys.s().fail.insert(default.clone());
    hotkeys
        .s()
        .errors
        .insert("recalibrate".into(), IN_USE.into());
    let mut h = make_controller(Opts {
        hotkeys: Some(hotkeys.clone()),
        ..Opts::default()
    });
    assert_eq!(
        hotkey_notices(&mut h),
        [format!("{IN_USE}. Choose another in Settings → Hotkeys.")]
    );

    for _ in 0..3 {
        h.controller.suspend_hotkeys(true);
        h.controller.suspend_hotkeys(false);
    }
    assert_eq!(hotkey_notices(&mut h).len(), 1);

    // Changing the hotkeys is a deliberate act: every failure is reported again.
    let mut s = h.controller.settings().clone();
    s.hotkeys.toggle_tracking = "ctrl+alt+meta+y".into();
    hotkeys.s().fail.insert("ctrl+alt+meta+y".into());
    hotkeys.s().errors.insert(
        "toggle_tracking".into(),
        "⌃⌥⌘Y is already in use by another application".into(),
    );
    h.controller.apply_settings(&s);
    let notices = hotkey_notices(&mut h);
    assert_eq!(notices.len(), 2);
    assert!(notices[1].contains(IN_USE));
    assert!(notices[1].contains("⌃⌥⌘Y"));

    // Once a combination works, a later failure of it is news again.
    hotkeys.s().fail.remove(&default);
    h.controller.suspend_hotkeys(true);
    h.controller.suspend_hotkeys(false);
    assert!(hotkeys.names().contains("recalibrate"));
    assert_eq!(hotkey_notices(&mut h).len(), 2); // ⌃⌥⌘Y: unchanged, not repeated
    hotkeys.s().fail.insert(default.clone());
    h.controller.suspend_hotkeys(true);
    h.controller.suspend_hotkeys(false);
    let notices = hotkey_notices(&mut h);
    assert_eq!(notices.len(), 3);
    assert!(notices[2].starts_with(IN_USE));

    // Switching the hotkeys off and on again announces the failures afresh.
    let mut s = h.controller.settings().clone();
    s.hotkeys.enabled = false;
    h.controller.apply_settings(&s);
    let mut s = s.clone();
    s.hotkeys.enabled = true;
    h.controller.apply_settings(&s);
    assert_eq!(hotkey_notices(&mut h).len(), 4);
}

#[test]
fn test_shutdown_stops_everything() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.controller.shutdown();
    h.controller.shutdown(); // idempotent
    assert!(h.worker().s().stopped);
    assert!(h.hotkeys.s().stopped);
    h.push(obs(RIGHT_CENTRE), 0.1); // late callbacks are ignored
    assert!(h.observations().is_empty());
}

#[test]
fn test_preview_requested_before_start() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts {
        start: false,
        ..Opts::default()
    });
    h.controller.set_preview(true, None);
    h.controller.start();
    assert_eq!(h.worker().s().previews, [true]);
    approx(last_interval(&h), 1.0 / 24.0);
}

#[test]
fn test_blink_holds_the_gaze() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.push(obs(RIGHT_CENTRE), 0.1);
    h.push(obs(RIGHT_CENTRE), 0.1);
    let mut blink = obs(RIGHT_CENTRE);
    blink.blink = true;
    h.push(blink, 0.1); // a blink must not restart the dwell
    h.push(obs(RIGHT_CENTRE), 0.1);
    assert_eq!(h.switched(), [1]);
}

// ------------------------------------------------------------------ default seams
#[test]
fn test_default_factories_build_but_do_not_open() {
    let mut s = Settings::default();
    assert_eq!(camera_fps(&s), 15);
    let mut factory = source_factory(&s);
    let source = factory().expect("created, not opened");
    assert!(!source.is_open());
    s.performance.profile = "responsive".into();
    assert_eq!(camera_fps(&s), 30);

    let (events, _receiver) = crossbeam_channel::unbounded();
    let worker = make_vision_worker(
        Box::new(|| Err("no camera".into())),
        Box::new(|| Err("no backend".into())),
        events,
        Arc::new(|| 0.0),
    );
    // Built, not started: nothing opened.
    assert!(!worker.stats().camera_open);
    assert_eq!(worker.backend_info(), None);
}

/// suspend_hotkeys releases the OS registrations and restores them afterwards.
#[test]
fn test_platform_property_and_hotkey_suspension() {
    let _dirs = app_dirs();
    let manager = Arc::new(FakeHotkeys::new());
    let platform: Arc<dyn PlatformServices> = Arc::new(crate::platform::GenericPlatform);
    let factory = manager.clone();
    let mut controller = Controller::new(
        &Settings::default(),
        platform.clone(),
        ControllerOptions {
            worker_factory: Some(Box::new(|source, backend, events| {
                Box::new(FakeWorker::new(source, backend, events))
            })),
            monitors_provider: Some(Box::new(Vec::new)),
            hotkey_manager_factory: Some(Box::new(move || {
                factory.clone() as Arc<dyn HotkeyManager>
            })),
            ..ControllerOptions::default()
        },
    );
    controller.start();
    assert!(Arc::ptr_eq(controller.platform(), &platform));
    let all: HashSet<String> = ["recalibrate", "toggle_privacy", "toggle_tracking"]
        .map(String::from)
        .into();
    assert_eq!(manager.names(), all);

    controller.suspend_hotkeys(true);
    assert!(manager.names().is_empty());
    let fired = |c: &mut Controller| -> Vec<String> {
        c.drain_events()
            .into_iter()
            .filter_map(|e| match e {
                ControllerEvent::CalibrationRequired(r) => Some(r),
                _ => None,
            })
            .collect()
    };
    fired(&mut controller);
    controller.handle_hotkey("recalibrate"); // a press racing the release is ignored
    assert!(fired(&mut controller).is_empty());

    controller.suspend_hotkeys(false);
    assert_eq!(manager.names(), all);
    controller.handle_hotkey("recalibrate");
    assert_eq!(fired(&mut controller), ["hotkey"]);
    controller.shutdown();
}

// ======================================================== review regressions
const USER_BOX: (f64, f64, f64, f64) = (0.35, 0.25, 0.3, 0.4);
const ONLOOKER_BOX: (f64, f64, f64, f64) = (0.8, 0.1, 0.1, 0.13);

/// A frame too dark or uniform to judge (lens covered, shutter closed).
fn blind_obs() -> Observation {
    Observation {
        blind: true,
        ..Observation::new(0.0, 0)
    }
}

fn with_size(obs: Observation, size: (u32, u32)) -> Observation {
    Observation {
        frame_size: size,
        ..obs
    }
}

fn boxed_obs(faces: u32, face_box: (f64, f64, f64, f64)) -> Observation {
    Observation {
        face_box: Some(face_box),
        ..gaze_obs((500.0, 500.0), faces)
    }
}

fn guard_settings(action: &str) -> Settings {
    let mut s = make_settings();
    s.privacy.shoulder_guard = true;
    s.privacy.guard_action = action.into();
    s
}

fn lock_session(h: &mut Harness) {
    h.p().locked = Some(true);
    h.tick(2.1);
    assert_eq!(h.state(), S::Locked);
}

fn unlock_session(h: &mut Harness) {
    h.p().locked = Some(false);
    h.tick(2.1);
}

fn ticks(h: &mut Harness, seconds: f64, step: f64) {
    for _ in 0..(seconds / step).round() as usize {
        h.tick(step);
    }
}

fn unreliable() -> Arc<FakePlatform> {
    let platform = Arc::new(FakePlatform::new());
    platform.p().cursor_reliable = false;
    platform
}

// ------------------------------------------ unreliable pointer / refused warps
#[test]
fn test_unreliable_pointer_position_follows_the_last_switch() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts {
        platform: Some(unreliable()),
        ..Opts::default()
    });
    h.p().cursor.track = false; // pos() keeps saying "left" (XWayland over a Wayland window)
    h.feed(&obs(RIGHT_CENTRE), 0.5, 0.1);
    assert_eq!(h.switched(), [1]);
    h.feed(&obs(RIGHT_CENTRE), 10.0, 0.1);
    // No re-warp every cooldown although the reported position never changed.
    assert_eq!(h.switched(), [1]);
    assert_eq!(moves(&h).len(), 1);
    h.feed(&obs(LEFT_CENTRE), 1.0, 0.1);
    assert_eq!(h.switched(), [1, 0]);
}

#[test]
fn test_unreliable_pointer_position_is_not_learned_from() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts {
        platform: Some(unreliable()),
        ..Opts::default()
    });
    h.p().cursor.track = false;
    h.feed(&obs(RIGHT_CENTRE), 0.5, 0.1);
    assert_eq!(h.switched(), [1]);
    set_cursor(&h, (400, 400)); // a stale position shows up
    h.tick(0.1);
    h.push(obs((400.0, 400.0)), 0.4);
    h.tick(3.0);
    assert!(h.controller.learner.samples().is_empty()); // not a learning label
    assert_eq!(h.controller.drift.event_count(), 0); // not an undone (or kept) switch
}

#[test]
fn test_refused_pointer_moves_are_not_switches_and_back_off() {
    let _dirs = app_dirs();
    let limit = WARP_REFUSAL_LIMIT as usize;
    let mut h = make_controller(Opts::default());
    h.p().window_under = Some(win(9, 1, Rect::new(1920, 0, 1920, 1080)));
    h.p().cursor.allow = false;
    h.feed(&obs(RIGHT_CENTRE), 3.0, 0.1);
    assert_eq!(moves(&h).len(), limit);
    assert!(h.switched().is_empty());
    assert_eq!(h.controller.status()["switches"], 0);
    assert!(!h.platform.names().contains(&"activate_window")); // nothing moved, nothing focused
    assert_eq!(h.titles(), ["Cannot move the cursor"]);

    h.feed(&obs(RIGHT_CENTRE), 55.0, 0.5);
    assert_eq!(moves(&h).len(), limit); // switching is paused
    h.feed(&obs(RIGHT_CENTRE), 5.0, 0.5); // the pause is over: one more try
    assert_eq!(moves(&h).len(), limit + 1);
    assert_eq!(h.titles(), ["Cannot move the cursor"]); // told once

    h.p().cursor.allow = true; // e.g. ydotoold was started
    h.feed(&obs(RIGHT_CENTRE), 2.0 * WARP_BACKOFF_S, 0.5);
    assert_eq!(h.switched(), [1]);
    assert!(calls(&h).contains(&Call::ActivateWindow(hwnd(9))));
}

/// r3-logic-04: a lock screen or UAC prompt that takes the input right after a lock poll
/// refused enough moves to suspend switching and warn the user.
#[test]
fn test_pointer_moves_refused_by_a_new_lock_screen_are_not_reported() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.tick(2.0); // the lock poll has just run
    h.p().locked = Some(true); // e.g. a UAC prompt raised by a background updater
    h.p().cursor.allow = false;
    h.feed(&obs(RIGHT_CENTRE), 1.9, 1.0 / 12.0);
    assert_eq!(moves(&h).len(), 1); // the first refusal revealed the lock
    assert_eq!(h.controller.warp_refusals, 0); // and did not count towards the backoff
    assert!(h.titles().is_empty());
    assert_eq!(h.state(), S::Locked);
    unlock_session(&mut h);
    h.p().cursor.allow = true;
    h.feed(&obs(RIGHT_CENTRE), 1.0, 0.1);
    assert_eq!(h.switched(), [1]);
}

// --------------------------------------------------------------- shoulder guard
#[test]
fn test_guard_does_not_lock_again_right_after_the_user_unlocked() {
    let _dirs = app_dirs();
    let mut h = make_controller(opts(guard_settings("lock")));
    let two = gaze_obs((500.0, 500.0), 2);
    h.feed(&two, 2.5, 0.1);
    assert_eq!(count(&h.platform.names(), "lock_screen"), 1);
    lock_session(&mut h);
    unlock_session(&mut h);
    assert_eq!(h.state(), S::Tracking);
    h.feed(&two, 5.0, 0.1); // the colleague is still there
    assert_eq!(count(&h.platform.names(), "lock_screen"), 1);
    assert_eq!(h.guard_changed(), [true]); // the curtain covers the screens instead
    h.feed(&obs((500.0, 500.0)), 2.0, 0.1); // out of view for a moment
    assert_eq!(h.guard_changed(), [true, false]);
    // A second face soon after may be the same colleague: covered, not locked.
    h.feed(&two, 2.5, 0.1);
    assert_eq!(count(&h.platform.names(), "lock_screen"), 1);
    assert_eq!(h.guard_changed(), [true, false, true]);
    // Once nobody has looked over the shoulder for the whole grace, it locks again.
    h.feed(&obs((500.0, 500.0)), 2.0, 0.1);
    h.feed(&obs((500.0, 500.0)), GUARD_RELOCK_GRACE_S, 1.0);
    h.feed(&two, 2.5, 0.1);
    assert_eq!(count(&h.platform.names(), "lock_screen"), 2);
}

/// r3-logic-01: every 1.5 s dropout (the colleague turns to the user or looks down at notes)
/// ended the relock grace, so the next look locked the user out.
#[test]
fn test_guard_grace_survives_a_colleague_who_drops_out_of_view() {
    let _dirs = app_dirs();
    let mut h = make_controller(opts(guard_settings("lock")));
    let two = gaze_obs((500.0, 500.0), 2);
    h.feed(&two, 2.5, 0.1);
    lock_session(&mut h);
    unlock_session(&mut h);
    h.feed(&two, 3.0, 0.1);
    assert_eq!(h.guard_changed(), [true]);
    h.controller.dismiss_curtain(); // Esc: the user carries on working
    for _ in 0..20 {
        // a long visit, well beyond GUARD_RELOCK_GRACE_S in total
        h.feed(&obs((500.0, 500.0)), 2.0, 0.1); // profile view: one face detected
        h.feed(&two, 30.0, 0.5); // looking at the screen again
    }
    assert_eq!(count(&h.platform.names(), "lock_screen"), 1);
    assert_eq!(h.state(), S::Tracking);
}

#[test]
fn test_no_switching_under_the_privacy_curtain() {
    let _dirs = app_dirs();
    let mut h = make_controller(opts(guard_settings("curtain")));
    h.p().window_under = Some(win(42, 1, Rect::new(1920, 0, 1920, 1080)));
    h.feed(&gaze_obs((500.0, 500.0), 2), 2.5, 0.1);
    assert_eq!(h.guard_changed(), [true]);
    h.feed(&gaze_obs(RIGHT_CENTRE, 2), 2.0, 0.1); // is the other screen covered too?
    assert!(h.switched().is_empty());
    assert!(moves(&h).is_empty());
    assert!(!h.platform.names().contains(&"window_at"));
    assert!(!h.platform.names().contains(&"activate_window"));

    h.controller.dismiss_curtain(); // Esc: the user carries on
    assert_eq!(h.guard_changed(), [true, false]);
    assert!(h.controller.guard_active());
    h.feed(&gaze_obs(RIGHT_CENTRE, 2), 1.0, 0.1);
    assert_eq!(h.switched(), [1]);
}

#[test]
fn test_guard_run_does_not_span_a_calibration() {
    let _dirs = app_dirs();
    let mut h = make_controller(opts(guard_settings("lock")));
    let two = gaze_obs((500.0, 500.0), 2);
    h.feed(&two, 1.8, 0.1);
    h.controller.begin_calibration();
    h.feed(&two, 1.0, 0.1);
    h.controller.finish_calibration(None);
    h.push(two.clone(), 0.1);
    assert!(!h.platform.names().contains(&"lock_screen"));
    h.feed(&two, 2.5, 0.1); // a fresh, continuous run
    assert_eq!(count(&h.platform.names(), "lock_screen"), 1);
}

#[test]
fn test_guard_run_does_not_span_a_camera_error() {
    let _dirs = app_dirs();
    let mut h = make_controller(opts(guard_settings("lock")));
    let two = gaze_obs((500.0, 500.0), 2);
    h.feed(&two, 1.8, 0.1);
    h.on_stats(failing_stats("Camera stopped"));
    assert_eq!(h.state(), S::CameraError);
    h.tick(1.0);
    h.on_stats(open_stats());
    h.push(two, 0.1);
    assert!(!h.platform.names().contains(&"lock_screen"));
}

#[test]
fn test_onlooker_left_alone_keeps_the_curtain_but_counts_as_a_face() {
    // Faces are not recognised, so walk-away detection cannot tell the onlooker from the
    // user: any face in view counts as someone at the computer (the guard's judgement "not
    // the user" must never lock a user out, r2-controller-01).
    let _dirs = app_dirs();
    let mut h = make_controller(opts(guard_settings("curtain")));
    h.feed(&boxed_obs(2, USER_BOX), 2.5, 0.1);
    assert_eq!(h.guard_changed(), [true]);
    h.feed(&boxed_obs(1, ONLOOKER_BOX), 30.0, 0.5); // the user left, the onlooker stayed
    assert_eq!(h.guard_changed(), [true]); // the curtain stays up in front of them
    assert!(h.controller.guard_active());
    assert!(h.away_warning().is_empty());
    assert!(!h.platform.names().contains(&"lock_screen"));
    h.feed(&no_face(), 3.0, 0.5); // missed for a moment: still covered
    assert_eq!(h.guard_changed(), [true]);
}

// r2-controller-01: the user ~65 cm from the camera; a colleague leans in beside them to
// ~45 cm, so the colleague's face is the larger (primary) one.
const SEATED_USER_BOX: (f64, f64, f64, f64) = (0.40, 0.30, 0.20, 0.27);
const LEANING_COLLEAGUE_BOX: (f64, f64, f64, f64) = (0.05, 0.15, 0.30, 0.40);

#[test]
fn test_colleague_leaning_in_closer_than_the_user_never_locks_the_user() {
    let _dirs = app_dirs();
    let mut h = make_controller(opts(guard_settings("curtain"))); // walk-away action "lock"
    h.feed(&boxed_obs(1, SEATED_USER_BOX), 2.0, 0.1);
    h.feed(&boxed_obs(2, LEANING_COLLEAGUE_BOX), 2.5, 0.1);
    assert_eq!(h.guard_changed(), [true]);
    // The colleague leaves; the user reads without touching anything.
    h.feed(&boxed_obs(1, SEATED_USER_BOX), 2.0, 0.1);
    assert_eq!(h.guard_changed(), [true, false]); // recognised as the user
    assert!(!h.controller.guard_active());
    h.feed(&boxed_obs(1, SEATED_USER_BOX), 30.0, 0.5);
    assert!(h.away_warning().is_empty());
    assert!(!h.platform.names().contains(&"lock_screen"));
}

#[test]
fn test_user_misjudged_as_the_onlooker_is_never_locked_and_input_lifts_the_curtain() {
    let _dirs = app_dirs();
    let mut h = make_controller(opts(guard_settings("curtain")));
    // The guard starts with both faces in view and the colleague's is the larger one: it
    // takes the colleague for the user and the user for the onlooker.
    h.feed(&boxed_obs(2, LEANING_COLLEAGUE_BOX), 2.5, 0.1);
    assert_eq!(h.guard_changed(), [true]);
    h.feed(&boxed_obs(1, SEATED_USER_BOX), 30.0, 0.5);
    assert_eq!(h.guard_changed(), [true]);
    assert!(h.away_warning().is_empty()); // the visible face keeps the user present
    assert!(!h.platform.names().contains(&"lock_screen"));
    set_cursor(&h, (800, 600)); // the user touches the mouse
    h.tick(0.1);
    h.feed(&boxed_obs(1, SEATED_USER_BOX), 2.0, 0.1);
    assert_eq!(h.guard_changed(), [true, false]);
    assert!(!h.controller.guard_active());
}

/// r3-logic-03: the guard stayed silent for every later onlooker until input.
#[test]
fn test_new_onlooker_after_a_misjudged_departure_is_reported() {
    let _dirs = app_dirs();
    let mut h = make_controller(opts(guard_settings("notify")));
    h.feed(&boxed_obs(1, SEATED_USER_BOX), 3.0, 0.1);
    h.feed(&boxed_obs(2, LEANING_COLLEAGUE_BOX), 3.0, 0.1);
    assert_eq!(h.titles(), ["Someone is looking at your screen"]);
    let leaned_back = (0.55, 0.36, 0.13, 0.18); // the user made room and stays there
    h.feed(&boxed_obs(1, leaned_back), 120.0, 0.5);
    assert!(h.controller.guard.owner_missing());
    h.feed(&boxed_obs(2, leaned_back), 10.0, 0.5); // a stranger looks on
    assert_eq!(h.titles(), ["Someone is looking at your screen"; 2]);
    assert!(!h.platform.names().contains(&"lock_screen"));
}

#[test]
fn test_user_left_alone_clears_the_guard() {
    let _dirs = app_dirs();
    let mut h = make_controller(opts(guard_settings("curtain")));
    h.feed(&boxed_obs(2, USER_BOX), 2.5, 0.1);
    h.feed(&boxed_obs(1, USER_BOX), 5.5, 0.5); // the onlooker left
    assert_eq!(h.guard_changed(), [true, false]);
    assert!(h.away_warning().is_empty());
}

// ----------------------------------------------------------------- camera yield
fn camera_in_use_is_checked_before_the_camera_reopens(leave: &str) {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    match leave {
        "resume" => h.controller.pause(),
        "privacy-off" => h.controller.set_privacy(true),
        _ => lock_session(&mut h),
    }
    assert!(!h.worker().is_active());
    h.p().camera_in_use = Some(true); // a video call starts meanwhile
    h.tick(5.0);
    let opened = h.worker().s().active.iter().filter(|a| **a).count();
    match leave {
        "resume" => h.controller.resume(),
        "privacy-off" => h.controller.set_privacy(false),
        _ => unlock_session(&mut h),
    }
    assert_eq!(h.state(), S::Yielded);
    // The call's camera was never grabbed.
    assert_eq!(h.worker().s().active.iter().filter(|a| **a).count(), opened);
}

#[test]
fn test_camera_in_use_is_checked_before_the_camera_reopens() {
    for leave in ["resume", "privacy-off", "unlock"] {
        camera_in_use_is_checked_before_the_camera_reopens(leave);
    }
}

// --------------------------------------------------------------- camera errors
#[test]
fn test_camera_failing_again_after_a_pause_is_noticed() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    let failing = failing_stats("Camera 0 could not be opened");
    h.worker().s().stats = failing.clone();
    h.on_stats(failing);
    assert_eq!(h.state(), S::CameraError);
    h.controller.pause();
    h.controller.resume();
    assert_eq!(h.state(), S::Tracking); // judged afresh
    // The worker fails with the same message again, so it publishes nothing new.
    h.tick(CAMERA_ERROR_RECHECK_S - 1.0);
    assert_eq!(h.state(), S::Tracking); // a slow camera gets time to open
    h.tick(1.5);
    assert_eq!(h.state(), S::CameraError);
    assert_eq!(count_titles(&mut h, "Camera unavailable"), 2);
}

#[test]
fn test_camera_working_after_a_pause_is_no_error() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.on_stats(failing_stats("Camera 0 could not be opened"));
    h.controller.pause();
    h.controller.resume();
    h.worker().s().stats = open_stats();
    h.tick(CAMERA_ERROR_RECHECK_S + 1.0);
    assert_eq!(h.state(), S::Tracking);
}

#[test]
fn test_failing_backend_is_a_camera_error() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    let mut stats = WorkerStats {
        camera_open: true,
        last_error: Some("Vision backend keeps failing: boom".into()),
        ..WorkerStats::default()
    };
    stats.extra.insert("backend_failing".into(), json!(true));
    h.on_stats(stats);
    assert_eq!(h.state(), S::CameraError);
    let mut fine = open_stats();
    fine.extra.insert("backend_failing".into(), json!(false));
    h.on_stats(fine);
    assert_eq!(h.state(), S::Tracking);
}

#[test]
fn test_blocked_camera_permission_is_explained() {
    let _dirs = app_dirs();
    let platform = Arc::new(FakePlatform::new());
    platform.p().camera_permission = Some(false);
    let mut h = make_controller(Opts {
        platform: Some(platform),
        ..Opts::default()
    });
    h.on_stats(failing_stats("Camera 0 could not be opened"));
    assert_eq!(h.state(), S::CameraError);
    assert_eq!(h.titles(), ["Camera access blocked"]);
    assert_eq!(h.permission_needed(), ["camera"]);
}

#[test]
fn test_input_ends_away_when_the_camera_failed_meanwhile() {
    let _dirs = app_dirs();
    let mut s = make_settings();
    s.presence.action = "display_off".into();
    let mut h = make_controller(opts(s));
    h.feed(&no_face(), 10.0, 0.5);
    assert_eq!(h.state(), S::Away);
    h.on_stats(failing_stats("Camera unplugged"));
    ticks(&mut h, 10.0, 0.5);
    assert_eq!(h.state(), S::Away); // nobody there
    h.p().idle = Some(0.2); // the user is back at the keyboard
    h.tick(0.1);
    assert_eq!(h.state(), S::CameraError);
    assert_eq!(h.platform.names(), ["display_off", "wake_display"]);
}

// ------------------------------------------------------------------ blind camera
#[test]
fn test_covered_camera_pauses_walk_away_detection() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.feed(&obs((500.0, 500.0)), 1.0, 0.1);
    h.feed(&blind_obs(), 60.0, 0.5); // shutter closed; the user keeps reading
    assert!(h.away_warning().is_empty());
    assert!(!h.platform.names().contains(&"lock_screen"));
    assert_eq!(h.titles(), ["Camera appears covered"]);
}

#[test]
fn test_darkness_after_the_user_left_still_locks() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.feed(&no_face(), 4.0, 0.5); // the user left, then the lights went out
    h.feed(&blind_obs(), 7.0, 0.5);
    assert_eq!(count(&h.platform.names(), "lock_screen"), 1);
    assert!(!h.titles().contains(&"Camera appears covered".to_string()));
}

#[test]
fn test_covered_camera_locks_after_a_long_time_without_input() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.feed(&obs((500.0, 500.0)), 1.0, 0.1);
    h.feed(&blind_obs(), BLIND_FREEZE_MAX_S - 1.0, 0.5);
    assert!(h.away_warning().is_empty());
    h.feed(&blind_obs(), 12.0, 0.5); // desk lamp off and gone: lock eventually
    assert_eq!(count(&h.platform.names(), "lock_screen"), 1);
}

#[test]
fn test_covered_camera_keeps_the_curtain_up() {
    let _dirs = app_dirs();
    let mut h = make_controller(opts(guard_settings("curtain")));
    h.feed(&gaze_obs((500.0, 500.0), 2), 2.5, 0.1);
    assert_eq!(h.guard_changed(), [true]);
    h.feed(&blind_obs(), 5.0, 0.5); // covering the camera does not uncover the screens
    assert_eq!(h.guard_changed(), [true]);
    h.feed(&no_face(), 2.0, 0.1);
    assert_eq!(h.guard_changed(), [true, false]);
}

// ---------------------------------------------------------------- look-away
fn save(data: &CalibrationData) {
    save_calibration(&paths::calibration_file(), data).expect("save");
}

fn uncalibrated() -> Opts {
    Opts {
        calibrated: false,
        ..Opts::default()
    }
}

#[test]
fn test_glance_below_the_screens_is_off_screen() {
    let _dirs = app_dirs();
    save(&make_calibration(&monitors(), Some(&[0, 1])));
    let mut h = make_controller(uncalibrated());
    assert_eq!(h.state(), S::Tracking);
    h.feed(&obs(LEFT_CENTRE), 0.5, 0.1);
    h.feed(&obs((2500.0, 4000.0)), 2.0, 0.1); // the phone on the desk, below the right monitor
    assert!(h.switched().is_empty());
    assert_eq!(
        h.controller
            .last_decision
            .as_ref()
            .expect("a decision")
            .reason,
        "off_screen"
    );
    assert_eq!(h.gaze_changed().last(), Some(&None));
    assert!(h.controller.learner.samples().is_empty());
}

#[test]
fn test_glance_below_the_screens_without_gaze_features_is_not_detected() {
    // The control: a model that does not know its gaze features, with a backend that
    // declares none, clips the estimate to just below the right monitor.
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.feed(&obs((2500.0, 4000.0)), 2.0, 0.1);
    assert_eq!(h.switched(), [1]);
}

#[test]
fn test_backend_gaze_features_detect_looking_away_with_an_old_model() {
    let _dirs = app_dirs();
    let (maker, _requested) = fake_backends();
    let mut h = make_controller(Opts {
        backend_factory: Some(maker),
        ..Opts::default()
    }); // its model predates `nonlinear`
    h.worker().build_backend(); // the worker builds the backend
    h.tick(0.1);
    assert_eq!(h.controller.gaze_feature_indices(), Some(vec![0, 1]));
    h.feed(&obs((2500.0, 4000.0)), 2.0, 0.1);
    assert!(h.switched().is_empty());
}

#[test]
fn test_phone_below_the_monitors_does_not_switch_with_realistic_features() {
    use gaze_synth::{GAZE, Rng, SampleOptions, SynthOptions, TWO, below_points};
    let _dirs = app_dirs();
    let mut rng = Rng::new(7);
    let samples = gaze_synth::calibration_samples(&TWO, &mut rng, 0.5, &SampleOptions::default());
    let (x, y, _) = samples_to_arrays(&samples).expect("arrays");
    // Degree 3 is what calibration selects for such data; clipped and bent, its estimates
    // for a glance at the phone land just below the right monitor, well within the
    // decider's off-screen margin.
    let model = GazeModel::new(3, 1.0, Some(&GAZE))
        .expect("model")
        .fit(&x, &y, None, virtual_bounds(TWO.iter()).ok(), None)
        .expect("fit");
    let mut cal = make_calibration(&TWO, None);
    cal.samples = samples;
    cal.model = model;
    save(&cal);
    let mut h = make_controller(uncalibrated());
    assert_eq!(h.state(), S::Tracking);
    for cm in [60.0, 80.0] {
        let right_only = [TWO[1].clone()];
        let phone = below_points(&right_only, &mut rng, 40, cm); // the cursor is on the left
        let features = gaze_synth::synth_features(&phone, &mut rng, 0.5, &SynthOptions::default());
        for r in 0..features.nrows() {
            let row: Vec<f64> = features.row(r).iter().copied().collect();
            let o = Observation {
                features: Some(row),
                quality: 1.0,
                ..Observation::new(0.0, 1)
            };
            h.push(o, 0.1);
            let now = h.clock.now();
            h.controller.tick(now);
        }
    }
    assert!(h.switched().is_empty());
}

/// r2-gaze-engine-03: looking away is judged from the combined gaze direction on the
/// calibrated monitors, not from each feature's calibrated range.
#[test]
fn test_gaze_estimated_just_below_a_monitor_still_switches() {
    let _dirs = app_dirs();
    save(&make_calibration(&monitors(), Some(&[0, 1])));
    let mut h = make_controller(uncalibrated());
    assert_eq!(h.state(), S::Tracking);
    let model = h.controller.model().expect("a model").clone();
    assert!(model.has_linear_estimate());
    let taskbar = (2880.0, 1200.0); // the estimate lands a little below the right monitor
    // What the per-feature test said.
    assert!(
        model
            .looks_away(
                &features_for(taskbar.0, taskbar.1),
                None,
                crate::gaze::model::LOOK_AWAY_EXCESS,
                None,
                crate::gaze::model::LOOK_AWAY_MARGIN
            )
            .expect("an answer")
    );
    h.feed(&obs(taskbar), 1.0, 0.1);
    assert_eq!(h.switched(), [1]);
    h.feed(&obs((960.0, 2600.0)), 2.0, 0.1); // papers on the desk below the left monitor
    assert_eq!(h.switched(), [1]);
}

// --------------------------------------------------------- backend identity
#[test]
fn test_new_backend_with_the_old_identity_is_accepted() {
    let _dirs = app_dirs();
    let (maker, requested) = fake_backends();
    let mut h = make_controller(Opts {
        backend_factory: Some(maker),
        ..Opts::default()
    });
    let backend = (BACKEND.0.to_string(), BACKEND.1.to_string());
    h.worker().build_backend();
    h.tick(0.1);
    assert_eq!(h.controller.backend_info(), backend);
    let mut new = h.controller.settings().clone();
    new.general.backend = "lite".into();
    h.controller.apply_settings(&new);
    assert_eq!(h.controller.backend_info().0, "lite"); // the settings decide meanwhile
    assert_eq!(h.state(), S::NeedsCalibration);
    h.worker().build_backend(); // the old backend recreated before the swap: not the new one
    h.tick(0.1);
    assert_eq!(h.controller.backend_info().0, "lite");
    h.worker().build_new_backend(); // "lite" was unavailable and the same backend came back
    h.tick(0.1);
    assert_eq!(*requested.lock(), ["auto", "auto", "lite"]);
    assert_eq!(h.controller.backend_info(), backend);
    assert_eq!(h.state(), S::Tracking);
}

#[test]
fn test_frames_of_the_previous_backend_do_not_invalidate_the_calibration() {
    let _dirs = app_dirs();
    let limit = FEATURE_MISMATCH_LIMIT;
    let mut h = make_controller(Opts::default());
    let stale = Observation {
        features: Some(vec![0.5; 3]),
        quality: 1.0,
        ..Observation::new(0.0, 1)
    };
    for _ in 0..limit - 1 {
        h.push(stale.clone(), 0.1);
    }
    h.push(obs((500.0, 500.0)), 0.1); // the new backend's features fit
    for _ in 0..limit - 1 {
        h.push(stale.clone(), 0.1);
    }
    assert_eq!(h.state(), S::Tracking);
    assert!(h.calibration_required().is_empty());
    h.push(stale, 0.1); // consistently wrong: the calibration really does not fit
    assert_eq!(h.state(), S::NeedsCalibration);
    assert_eq!(
        h.calibration_required(),
        ["the camera features no longer match the calibration"]
    );
}

// --------------------------------------------------- calibration announcements
#[test]
fn test_layout_change_while_locked_is_announced_after_unlock() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    lock_session(&mut h);
    h.monitors.lock().push(third());
    h.controller.refresh_monitors(); // docked while locked
    assert!(h.calibration_required().is_empty());
    unlock_session(&mut h);
    assert_eq!(h.state(), S::NeedsCalibration);
    h.tick(1.0);
    assert!(h.calibration_required().is_empty()); // the layout gets time to settle
    ticks(&mut h, 2.0, 0.5);
    assert_eq!(h.calibration_required(), ["the monitor layout changed"]);
    ticks(&mut h, 5.0, 0.5);
    assert_eq!(h.calibration_required(), ["the monitor layout changed"]);
}

#[test]
fn test_layout_restored_while_locked_is_not_announced() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    lock_session(&mut h);
    h.monitors.lock().push(third());
    h.controller.refresh_monitors();
    h.monitors.lock().remove(2);
    h.controller.refresh_monitors();
    unlock_session(&mut h);
    ticks(&mut h, 5.0, 0.5);
    assert_eq!(h.state(), S::Tracking);
    assert!(h.calibration_required().is_empty());
}

#[test]
fn test_layout_change_while_away_is_announced_on_return() {
    let _dirs = app_dirs();
    let mut s = make_settings();
    s.presence.action = "display_off".into();
    let mut h = make_controller(opts(s));
    h.feed(&no_face(), 10.0, 0.5);
    assert_eq!(h.state(), S::Away);
    h.monitors.lock().push(third());
    h.controller.refresh_monitors();
    assert!(h.calibration_required().is_empty());
    h.feed(&obs((500.0, 500.0)), 3.0, 0.5);
    assert_eq!(h.state(), S::NeedsCalibration);
    assert_eq!(h.calibration_required(), ["the monitor layout changed"]);
}

#[test]
fn test_unusable_calibration_is_recalled_when_the_user_comes_back() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.monitors.lock().push(third());
    h.controller.refresh_monitors();
    assert_eq!(h.calibration_required(), ["the monitor layout changed"]);
    lock_session(&mut h);
    unlock_session(&mut h);
    ticks(&mut h, 3.0, 0.5);
    assert_eq!(h.calibration_required().len(), 1); // said a moment ago
    lock_session(&mut h);
    h.tick(CALIBRATION_REMINDER_S); // the night passes
    unlock_session(&mut h);
    ticks(&mut h, 3.0, 0.5);
    assert_eq!(h.calibration_required(), ["the monitor layout changed"; 2]);
}

/// r3-ux-docs-01: a setup that was never calibrated (and starts at login) was never told
/// again why nothing switches.
#[test]
fn test_never_calibrated_is_recalled_when_the_user_comes_back() {
    let _dirs = app_dirs();
    let mut h = make_controller(uncalibrated());
    lock_session(&mut h);
    unlock_session(&mut h);
    ticks(&mut h, 3.0, 0.5);
    assert!(h.calibration_required().is_empty()); // the app offered it at start
    lock_session(&mut h);
    h.tick(CALIBRATION_REMINDER_S); // the night passes
    unlock_session(&mut h);
    ticks(&mut h, 3.0, 0.5);
    assert_eq!(h.calibration_required(), ["not calibrated yet"]);
}

/// r3-ux-docs-03: one monitor, or switching turned off, needs no calibration: no warning
/// state, no prompts.
#[test]
fn test_no_calibration_is_asked_for_where_it_changes_nothing() {
    let _dirs = app_dirs();
    let mut s = make_settings();
    s.switching.enabled = false;
    let mut h = make_controller(Opts {
        calibrated: false,
        ..opts(s)
    });
    assert_eq!(h.state(), S::Tracking);
    lock_session(&mut h);
    h.tick(CALIBRATION_REMINDER_S);
    unlock_session(&mut h);
    ticks(&mut h, 3.0, 0.5);
    assert!(h.calibration_required().is_empty());
    // Turning switching on: now a calibration is what is missing, and the user back from
    // the night is told so.
    let mut on = h.controller.settings().clone();
    on.switching.enabled = true;
    h.controller.apply_settings(&on);
    assert_eq!(h.state(), S::NeedsCalibration);
    ticks(&mut h, 3.0, 0.5);
    assert_eq!(h.calibration_required(), ["not calibrated yet"]);
    // One monitor left (a laptop undocked): nothing to calibrate for.
    h.monitors.lock().remove(1);
    h.controller.refresh_monitors();
    assert_eq!(h.state(), S::Tracking);
    assert_eq!(h.controller.status()["calibrated"], false);
}

// ------------------------------------------------------- calibration profiles
fn current_key(h: &Harness) -> crate::gaze::store::ProfileKey {
    h.controller.calibration().expect("a calibration").key()
}

#[test]
fn test_calibration_profiles_follow_the_desk() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    let office = current_key(&h);
    h.monitors.lock().push(third());
    h.controller.refresh_monitors();
    assert_eq!(h.state(), S::NeedsCalibration);
    h.controller.begin_calibration();
    let layout = h.monitors.lock().clone();
    h.controller
        .finish_calibration(Some(make_calibration(&layout, None)));
    let home = current_key(&h);
    assert_eq!(h.state(), S::Tracking);
    assert_eq!(h.controller.calibrations().len(), 2);

    h.monitors.lock().remove(2); // back at the office
    h.controller.refresh_monitors();
    assert_eq!(h.state(), S::Tracking);
    assert_eq!(current_key(&h), office);
    h.monitors.lock().push(third()); // and home again
    h.controller.refresh_monitors();
    assert_eq!(h.state(), S::Tracking);
    assert_eq!(current_key(&h), home);
    assert_eq!(h.calibration_required(), ["the monitor layout changed"]); // once, ever

    let stored = CalibrationLibrary::load(&paths::calibration_file());
    assert_eq!(stored.len(), 2);
    // Most recently used.
    assert_eq!(stored.latest().expect("latest").layout_signature, home.0);
    assert_eq!(h.controller.status()["calibration"]["profiles"], 2);
}

#[test]
fn test_learned_samples_stay_with_their_profile() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.tick(0.1);
    for i in 0..3 {
        // three "move the mouse, let it rest" moments at the office
        let point = (300 + 200 * i, 400);
        set_cursor(&h, point);
        h.tick(0.1);
        h.push(obs((f64::from(point.0), f64::from(point.1))), 0.4);
    }
    assert_eq!(h.controller.learner.samples().len(), 3);

    h.monitors.lock().push(third());
    h.controller.refresh_monitors();
    h.controller.begin_calibration();
    let layout = h.monitors.lock().clone();
    h.controller
        .finish_calibration(Some(make_calibration(&layout, None)));
    assert!(h.controller.learner.samples().is_empty()); // a new desk learns afresh

    h.monitors.lock().remove(2);
    h.controller.refresh_monitors();
    assert_eq!(h.controller.learner.samples().len(), 3);
    assert_eq!(stored_office().implicit_samples.len(), 3); // saved with the office profile
}

#[test]
fn test_start_uses_the_profile_for_the_current_layout() {
    let _dirs = app_dirs();
    save(&cal());
    save(&make_calibration(&[left(), right(), third()], None));
    let h = make_controller(uncalibrated()); // started at the two-monitor desk
    assert_eq!(h.state(), S::Tracking);
    let current = h.controller.calibration().expect("a calibration");
    assert_eq!(current.layout_signature, layout_signature(&monitors()));
    let stored = CalibrationLibrary::load(&paths::calibration_file());
    assert_eq!(
        stored.latest().expect("latest").layout_signature,
        layout_signature(&monitors())
    );
}

// ------------------------------------------------------------------- camera
#[test]
fn test_calibration_records_the_camera() {
    let _dirs = app_dirs();
    let mut h = make_controller(uncalibrated());
    h.controller.begin_calibration();
    for _ in 0..3 {
        h.push(with_size(obs((500.0, 500.0)), (640, 480)), 0.1);
    }
    assert_eq!(
        h.controller.camera_identity(),
        ("0".to_string(), (640, 480))
    );
    h.controller.finish_calibration(Some(cal()));
    let stored = load_calibration(&paths::calibration_file()).expect("stored");
    assert_eq!(stored.camera, "0");
    assert_eq!(stored.frame_size, (640, 480));
    assert_eq!(h.state(), S::Tracking);
}

#[test]
fn test_another_camera_needs_another_calibration() {
    let _dirs = app_dirs();
    let mut data = cal();
    data.camera = "0".into();
    save(&data);
    let mut h = make_controller(uncalibrated());
    assert_eq!(h.state(), S::Tracking);
    let mut new = h.controller.settings().clone();
    new.camera.device = "1".into();
    h.controller.apply_settings(&new);
    assert_eq!(h.state(), S::NeedsCalibration);
    assert_eq!(h.controller.calibration_reason(), "the camera changed");
    assert_eq!(h.calibration_required(), ["the camera changed"]);
    let mut back = h.controller.settings().clone();
    back.camera.device = "0".into();
    h.controller.apply_settings(&back);
    assert_eq!(h.state(), S::Tracking);
}

#[test]
fn test_each_camera_has_its_own_profile() {
    let _dirs = app_dirs();
    for device in ["0", "1"] {
        let mut data = cal();
        data.camera = device.into();
        save(&data);
    }
    let mut h = make_controller(uncalibrated());
    assert_eq!(h.state(), S::Tracking);
    assert_eq!(
        h.controller.calibration().expect("a calibration").camera,
        "0"
    );
    let mut new = h.controller.settings().clone();
    new.camera.device = "1".into();
    h.controller.apply_settings(&new);
    assert_eq!(h.state(), S::Tracking);
    assert_eq!(
        h.controller.calibration().expect("a calibration").camera,
        "1"
    );
}

#[test]
fn test_camera_aspect_change_needs_another_calibration() {
    let _dirs = app_dirs();
    let mut data = cal();
    data.camera = "0".into();
    data.frame_size = (640, 480);
    save(&data);
    let mut h = make_controller(uncalibrated());
    h.push(with_size(obs((500.0, 500.0)), (1280, 960)), 0.1); // same shape, more pixels
    assert_eq!(h.state(), S::Tracking);
    h.push(with_size(obs((500.0, 500.0)), (1280, 720)), 0.1); // 16:9: another field of view
    assert_eq!(h.state(), S::NeedsCalibration);
    assert!(
        h.controller
            .calibration_reason()
            .starts_with("the camera's aspect ratio changed")
    );
    assert_eq!(h.calibration_required().len(), 1);
    h.push(with_size(obs((500.0, 500.0)), (640, 480)), 0.1);
    assert_eq!(h.state(), S::Tracking);
}

// ------------------------------------------------------ calibration + privacy
#[test]
fn test_calibration_restores_privacy_mode() {
    for saved in [false, true] {
        let _dirs = app_dirs();
        let mut h = make_controller(Opts::default());
        h.controller.set_privacy(true);
        h.controller.begin_calibration();
        assert!(h.worker().is_active());
        h.controller
            .finish_calibration(if saved { Some(cal()) } else { None });
        assert!(h.controller.privacy());
        assert_eq!(h.state(), S::Privacy);
        assert!(!h.worker().is_active());
    }
}

#[test]
fn test_privacy_changed_during_calibration_wins() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.controller.set_privacy(true);
    h.controller.begin_calibration();
    h.controller.set_privacy(false); // e.g. `eye-tracker ctl privacy-off`
    h.controller.finish_calibration(None);
    assert!(!h.controller.privacy());
    assert_eq!(h.state(), S::Tracking);
}

// ------------------------------------------------------------------ learning
#[test]
fn test_implausible_learning_labels_are_ignored() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.tick(0.1);
    set_cursor(&h, (3700, 1000)); // the pointer parked on the right ...
    h.tick(0.1);
    h.push(obs((200.0, 100.0)), 0.4); // ... while the user reads on the left
    assert!(h.controller.learner.samples().is_empty());
    assert_eq!(h.controller.drift.event_count(), 1); // still evidence about accuracy
    set_cursor(&h, (3000, 600));
    h.tick(0.1);
    h.push(obs((3000.0, 600.0)), 0.4);
    assert_eq!(h.controller.learner.samples().len(), 1);
}

#[test]
fn test_kept_switch_counts_as_correct() {
    let _dirs = app_dirs();
    let mut s = make_settings();
    s.switching.cursor_target = "center".into();
    let mut h = make_controller(opts(s));
    h.feed(&obs(RIGHT_CENTRE), 0.5, 0.1);
    assert_eq!(h.switched(), [1]);
    h.feed(&obs(RIGHT_CENTRE), 2.5, 0.1);
    assert_eq!(h.controller.drift.event_count(), 1);
    assert_eq!(h.controller.drift.error_rate(), 0.0);
}

#[test]
fn test_smaller_learning_capacity_refits_the_model() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.tick(0.1);
    for i in 0..25 {
        // enough settles for one refit
        let point = (300 + 40 * i, 400);
        set_cursor(&h, point);
        h.tick(0.1);
        h.push(obs((f64::from(point.0), f64::from(point.1))), 0.4);
    }
    let current = h.controller.calibration().expect("a calibration");
    let refined = current.model.to_dict();
    assert_eq!(current.implicit_samples.len(), 25);
    let mut new = h.controller.settings().clone();
    new.learning.max_samples = 0;
    h.controller.apply_settings(&new);
    let current = h.controller.calibration().expect("a calibration");
    assert!(current.implicit_samples.is_empty());
    assert_ne!(current.model.to_dict(), refined);
    assert!(h.controller.model().is_some()); // the model in use is the profile's
    let stored = load_calibration(&paths::calibration_file()).expect("stored");
    assert!(stored.implicit_samples.is_empty());
}

// -------------------------------------------------------------------- preview
#[test]
fn test_preview_consumers_are_counted() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.controller.set_preview(true, Some("window"));
    h.controller.set_preview(true, Some("wizard"));
    h.controller.set_preview(false, Some("window")); // the preview window is hidden
    assert_eq!(h.worker().s().previews, [true]);
    assert!(h.controller.preview_enabled());
    h.on_preview(frame());
    assert_eq!(h.preview_frames(), [frame()]); // the wizard still gets frames
    h.controller.set_preview(false, Some("wizard"));
    assert_eq!(h.worker().s().previews, [true, false]);
    assert!(!h.controller.preview_enabled());
}

// ------------------------------------------------------------- macOS specifics
#[test]
fn test_missing_accessibility_is_reported_at_start() {
    for status in ["stale", "missing"] {
        let _dirs = app_dirs();
        let platform = Arc::new(FakePlatform::new());
        platform.p().accessibility = status;
        let mut h = make_controller(Opts {
            platform: Some(platform),
            start: false,
            ..Opts::default()
        });
        h.controller.start();
        assert_eq!(h.titles(), ["Accessibility access needed"]);
        assert_eq!(h.permission_needed(), ["accessibility"]);
        assert_eq!(
            h.notify()[0].1.contains("remove Eye Tracker"),
            status == "stale"
        );
    }
}

#[test]
fn test_accessibility_is_not_needed_without_window_focus() {
    let _dirs = app_dirs();
    let platform = Arc::new(FakePlatform::new());
    platform.p().accessibility = "missing";
    let mut s = make_settings();
    s.switching.focus_window = false;
    let mut h = make_controller(Opts {
        platform: Some(platform),
        ..opts(s)
    });
    assert!(h.notify().is_empty());
}

#[test]
fn test_failed_activations_report_missing_accessibility() {
    let _dirs = app_dirs();
    let platform = Arc::new(FakePlatform::new());
    platform.p().accessibility = "granted";
    let mut s = make_settings();
    s.switching.cursor_target = "center".into();
    let mut h = make_controller(Opts {
        platform: Some(platform),
        ..opts(s)
    });
    h.p().window_under = Some(win(9, 1, Rect::new(0, 0, 3840, 1080)));
    h.p().activate_ok = false;
    h.p().accessibility = "missing"; // revoked while running
    for point in [RIGHT_CENTRE, LEFT_CENTRE, RIGHT_CENTRE] {
        h.feed(&obs(point), 1.0, 0.1);
    }
    assert_eq!(h.switched(), [1, 0, 1]);
    assert_eq!(h.titles(), ["Accessibility access needed"]);
}

#[test]
fn test_app_nap_is_allowed_only_while_paused_or_private() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    assert_eq!(h.p().background, [true]);
    h.controller.pause();
    assert_eq!(h.p().background, [true, false]);
    h.controller.resume();
    assert_eq!(h.p().background, [true, false, true]);
    lock_session(&mut h); // unlocking must still be noticed promptly
    unlock_session(&mut h);
    h.controller.set_privacy(true);
    assert_eq!(h.p().background, [true, false, true, false]);
}

// ------------------------------------------------------------- screen lock thread
fn slow_platform() -> Arc<FakePlatform> {
    Arc::new(FakePlatform::slow_lock())
}

/// Let a slow lock call return and deliver its result to the main thread.
fn finish_lock(h: &mut Harness) {
    assert!(h.platform.slow);
    h.platform.release.set();
    let thread = h.controller.lock_thread.take().expect("a lock thread");
    thread.join().expect("the lock thread");
    h.process(); // the queued result
}

fn threaded(platform: Arc<FakePlatform>, settings: Option<Settings>) -> Opts {
    Opts {
        settings,
        platform: Some(platform),
        threaded_locks: true,
        ..Opts::default()
    }
}

#[test]
fn test_walk_away_lock_does_not_block_the_gui_thread() {
    let _dirs = app_dirs();
    let platform = slow_platform();
    let mut h = make_controller(threaded(platform.clone(), None));
    h.feed(&no_face(), 10.0, 0.5); // returns although lock_screen is still blocked
    assert!(platform.entered.wait(5.0));
    assert_ne!(h.p().lock_threads, [std::thread::current().id()]);
    assert_eq!(h.state(), S::Away);
    h.feed(&no_face(), 5.0, 0.5); // still away, still one lock request in flight
    assert_eq!(h.p().lock_threads.len(), 1);
    finish_lock(&mut h);
    assert_eq!(h.platform.names(), ["lock_screen"]);
    assert!(h.titles().is_empty());
    // The lock is noticed soon, so the camera is released promptly.
    h.p().locked = Some(true);
    h.tick(0.6);
    assert_eq!(h.state(), S::Locked);
}

#[test]
fn test_failed_walk_away_lock_notifies_from_the_thread_result() {
    let _dirs = app_dirs();
    let platform = slow_platform();
    platform.p().lock_ok = false;
    let mut h = make_controller(threaded(platform.clone(), None));
    h.feed(&no_face(), 10.0, 0.5);
    assert!(platform.entered.wait(5.0));
    assert!(h.titles().is_empty()); // nothing known yet
    finish_lock(&mut h);
    assert_eq!(h.titles(), ["Could not lock the screen"]);
}

#[test]
fn test_guard_lock_runs_on_a_thread_and_falls_back_to_the_curtain() {
    let _dirs = app_dirs();
    let platform = slow_platform();
    platform.p().lock_ok = false;
    let mut h = make_controller(threaded(platform.clone(), Some(guard_settings("lock"))));
    h.feed(&gaze_obs((500.0, 500.0), 2), 2.5, 0.1);
    assert!(platform.entered.wait(5.0));
    assert!(h.guard_changed().is_empty());
    finish_lock(&mut h);
    assert_eq!(h.guard_changed(), [true]); // the onlooker is still there
}

#[test]
fn test_failed_guard_lock_after_the_onlooker_left_shows_no_curtain() {
    let _dirs = app_dirs();
    let platform = slow_platform();
    platform.p().lock_ok = false;
    let mut h = make_controller(threaded(platform.clone(), Some(guard_settings("lock"))));
    h.feed(&gaze_obs((500.0, 500.0), 2), 2.5, 0.1);
    assert!(platform.entered.wait(5.0));
    h.feed(&obs((500.0, 500.0)), 2.0, 0.1); // gone before the lock call returned
    assert!(!h.controller.guard_active());
    finish_lock(&mut h);
    assert!(h.guard_changed().is_empty());
}

#[test]
fn test_successful_guard_lock_from_the_thread_arms_the_relock_grace() {
    let _dirs = app_dirs();
    let platform = slow_platform();
    let mut h = make_controller(threaded(platform.clone(), Some(guard_settings("lock"))));
    let two = gaze_obs((500.0, 500.0), 2);
    h.feed(&two, 2.5, 0.1);
    assert!(platform.entered.wait(5.0));
    finish_lock(&mut h);
    lock_session(&mut h);
    unlock_session(&mut h);
    h.feed(&two, 5.0, 0.1); // the colleague is still there: the curtain, not another lock
    assert_eq!(count(&h.platform.names(), "lock_screen"), 1);
    assert_eq!(h.guard_changed(), [true]);
}

#[test]
fn test_shutdown_waits_briefly_for_a_lock_in_progress() {
    let _dirs = app_dirs();
    let platform = slow_platform();
    let mut h = make_controller(threaded(platform.clone(), None));
    h.feed(&no_face(), 10.0, 0.5);
    assert!(platform.entered.wait(5.0));
    assert!(h.controller.lock_thread.is_some());
    let release = platform.clone();
    let timer = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(100));
        release.release.set();
    });
    h.controller.shutdown();
    assert!(h.controller.lock_thread.is_none()); // joined
    timer.join().expect("timer");
    assert!(h.titles().is_empty()); // no result is handled after shutdown
}

// ------------------------------------------------------------ first-run safety
/// journeys-05: never lock (or blank) a PC whose owner has not finished setup.
#[test]
fn test_walk_away_only_notifies_before_the_setup_is_finished() {
    for action in ["lock", "lock_and_display_off", "display_off"] {
        let _dirs = app_dirs();
        let mut s = make_settings();
        s.general.first_run_done = false;
        s.presence.action = action.into();
        let mut h = make_controller(opts(s));
        h.feed(&no_face(), 10.0, 0.5);
        assert_eq!(h.state(), S::Away);
        assert!(h.platform.names().is_empty()); // neither locked nor displays off
        assert_eq!(h.titles(), ["Are you still there?"]);
        assert!(h.notify()[0].1.contains("Finish the setup"));

        let mut finished = h.controller.settings().clone();
        finished.general.first_run_done = true;
        h.controller.apply_settings(&finished);
        h.push(obs((500.0, 500.0)), 0.1); // back ...
        h.feed(&no_face(), 10.0, 0.5); // ... and away again, after the setup
        let expected = if action.contains("display") {
            "display_off"
        } else {
            "lock_screen"
        };
        assert_eq!(h.platform.names()[0], expected);
    }
}

#[test]
fn test_no_action_stays_silent_before_the_setup_is_finished() {
    let _dirs = app_dirs();
    let mut s = make_settings();
    s.general.first_run_done = false;
    s.presence.action = "none".into();
    let mut h = make_controller(opts(s));
    h.feed(&no_face(), 10.0, 0.5);
    assert_eq!(h.state(), S::Away);
    assert!(h.notify().is_empty());
    assert!(h.platform.names().is_empty());
}

// ================================================== second review regressions
// ------------------------------------------------------------ r2-controller-02
#[test]
fn test_preview_frame_captured_before_privacy_mode_is_not_shown() {
    let _dirs = app_dirs();
    let mut h = make_controller(Opts::default());
    h.controller.set_preview(true, Some("window"));
    let from_worker = |h: &Harness| {
        let worker = h.worker();
        std::thread::spawn(move || worker.send_preview(frame()))
            .join()
            .expect("thread");
    };

    from_worker(&h); // analysed just before the privacy hotkey was handled
    h.controller.set_privacy(true);
    h.process();
    assert!(h.preview_frames().is_empty());
    from_worker(&h); // finished by the worker after the camera was switched off
    h.process();
    assert!(h.preview_frames().is_empty());

    h.controller.set_privacy(false);
    from_worker(&h);
    h.process();
    assert_eq!(h.preview_frames(), [frame()]);
}

// ------------------------------------------------------------ r2-controller-03
fn pause_when_locked_off() -> Settings {
    let mut s = make_settings();
    s.privacy.pause_when_locked = false;
    s
}

fn lock_session_keep_tracking(h: &mut Harness) {
    h.p().locked = Some(true);
    h.tick(2.1);
    assert_eq!(h.state(), S::Tracking); // pause_when_locked is off
}

#[test]
fn test_no_switching_on_the_lock_screen() {
    let _dirs = app_dirs();
    let mut h = make_controller(opts(pause_when_locked_off()));
    lock_session_keep_tracking(&mut h);
    h.p().cursor.allow = false; // the lock screen owns the input desktop
    for _ in 0..4 {
        // the user at the desk looks at both screens
        h.feed(&obs(RIGHT_CENTRE), 2.0, 0.1);
        h.feed(&obs(LEFT_CENTRE), 2.0, 0.1);
    }
    assert!(moves(&h).is_empty());
    assert!(!h.platform.names().contains(&"activate_window"));
    assert!(h.titles().is_empty()); // no "Cannot move the cursor" (with a Wayland hint) on Windows

    h.p().locked = Some(false);
    h.p().cursor.allow = true;
    h.tick(2.1);
    h.feed(&obs(RIGHT_CENTRE), 1.0, 0.1);
    assert_eq!(h.switched(), [1]);
}

#[test]
fn test_refused_moves_around_a_lock_do_not_suspend_switching_after_unlock() {
    let _dirs = app_dirs();
    let mut h = make_controller(opts(pause_when_locked_off()));
    // The session is locked but that is not noticed yet (polled every LOCK_POLL_S): the
    // system refuses the moves and switching is suspended.
    h.p().cursor.allow = false;
    h.feed(&obs(RIGHT_CENTRE), 3.0, 0.1);
    assert_eq!(moves(&h).len(), WARP_REFUSAL_LIMIT as usize);
    lock_session_keep_tracking(&mut h);
    h.p().locked = Some(false);
    h.p().cursor.allow = true;
    h.tick(2.1);
    h.feed(&obs(RIGHT_CENTRE), 1.0, 0.1);
    assert_eq!(h.switched(), [1]); // at once, not WARP_BACKOFF_S later
}

// ------------------------------------------------------------ r2-controller-04
/// The two-monitor profile after learning that the gaze lands 300 px further right than the
/// calibration says (40 samples).
fn learned_office_profile() -> CalibrationData {
    let mut data = cal();
    let mut learned = Vec::new();
    for i in 0..40 {
        let x = 200.0 + 3200.0 * f64::from(i) / 39.0;
        let target = x + 300.0;
        let monitor = if target < 1920.0 { 0 } else { 1 };
        learned.push(CalibrationSample::weighted(
            features_for(x, 540.0),
            target,
            540.0,
            monitor,
            -1 - i64::from(i),
            0.5,
        ));
    }
    let model = refit_model(&data.samples, &learned, &data.model, None, None).expect("refit");
    data.implicit_samples = learned;
    data.model = model;
    data
}

fn predicted_x(h: &Harness, x: f64) -> f64 {
    let model = h.controller.model().expect("a model");
    model
        .predict(&features_for(x, 540.0))
        .expect("a prediction")[0]
}

fn stored_office() -> CalibrationData {
    let office: Vec<CalibrationData> = CalibrationLibrary::load(&paths::calibration_file())
        .iter()
        .filter(|p| p.monitors.len() == 2)
        .cloned()
        .collect();
    assert_eq!(office.len(), 1);
    office.into_iter().next().expect("the office profile")
}

#[test]
fn test_lower_learning_capacity_applies_to_a_profile_used_later() {
    let _dirs = app_dirs();
    save(&learned_office_profile());
    save(&make_calibration(&[left(), right(), third()], None));
    let mut h = make_controller(Opts {
        calibrated: false,
        start: false,
        ..Opts::default()
    });
    h.monitors.lock().push(third()); // at the home desk
    h.controller.start();
    assert_eq!(h.state(), S::Tracking);
    let mut new = h.controller.settings().clone();
    new.learning.max_samples = 0; // "forget what was learned"
    h.controller.apply_settings(&new);

    h.monitors.lock().remove(2); // docked at the office
    h.controller.refresh_monitors();
    assert_eq!(h.state(), S::Tracking);
    let current = h.controller.calibration().expect("a calibration");
    assert_eq!(current.monitors.len(), 2);
    assert!(h.controller.learner.samples().is_empty());
    assert!(current.implicit_samples.is_empty());
    let x = predicted_x(&h, 960.0); // no learned shift left
    assert!((x - 960.0).abs() <= 1.0, "{x}");
    assert!(stored_office().implicit_samples.is_empty());
}

#[test]
fn test_lower_learning_capacity_in_the_settings_file_applies_at_start() {
    let _dirs = app_dirs();
    let office = learned_office_profile();
    save(&office);
    let mut s = make_settings();
    s.learning.max_samples = 10;
    let h = make_controller(Opts {
        calibrated: false,
        ..opts(s)
    });
    assert_eq!(h.state(), S::Tracking);
    let current = h.controller.calibration().expect("a calibration");
    let kept = h.controller.learner.samples().to_vec();
    assert_eq!(kept.len(), 10);
    assert_eq!(current.implicit_samples, kept);
    let expected = refit_model(&office.samples, &kept, &office.model, None, None).expect("refit");
    approx(
        predicted_x(&h, 960.0),
        expected
            .predict(&features_for(960.0, 540.0))
            .expect("a prediction")[0],
    );
    assert_eq!(stored_office().implicit_samples.len(), 10);
}

// ------------------------------------------------------------ r2-controller-05
#[test]
fn test_unreliable_pointer_follows_its_monitor_when_screens_are_renumbered() {
    let _dirs = app_dirs();
    let b_alone = Monitor {
        primary: true,
        ..Monitor::new(0, "right", right().rect)
    };
    let c_alone = Monitor::new(1, "third", third().rect);
    save(&make_calibration(&[left(), right(), third()], None));
    save(&make_calibration(&[b_alone.clone(), c_alone.clone()], None));
    let mut h = make_controller(Opts {
        calibrated: false,
        start: false,
        platform: Some(unreliable()),
        ..Opts::default()
    });
    h.monitors.lock().push(third());
    h.controller.start();
    h.p().cursor.track = false; // the reported position stays on the left monitor
    h.feed(&obs(RIGHT_CENTRE), 0.5, 0.1);
    assert_eq!(h.switched(), [1]); // the pointer is on B, monitor 1

    *h.monitors.lock() = vec![b_alone, c_alone]; // A unplugged: B is now 0, C is now 1
    h.controller.refresh_monitors();
    assert_eq!(h.state(), S::Tracking);
    assert_eq!(h.controller.assumed_monitor, Some(0)); // still B
    h.feed(&obs((4800.0, 540.0)), 1.0, 0.1); // look at C
    assert_eq!(h.switched(), [1, 1]);
    assert!(moves(&h).last().expect("a move").0 >= third().rect.x);
}

#[test]
fn test_unreliable_pointer_follows_its_monitor_by_name_when_the_desktop_moves() {
    // A unplugged and the desktop laid out anew from x = 0: B takes A's place and C takes
    // B's old rectangle, so only the screen names tell them apart.
    let _dirs = app_dirs();
    let b_moved = Monitor {
        primary: true,
        ..Monitor::new(0, "right", left().rect)
    };
    let c_moved = Monitor::new(1, "third", right().rect);
    save(&make_calibration(&[left(), right(), third()], None));
    save(&make_calibration(&[b_moved.clone(), c_moved.clone()], None));
    let mut h = make_controller(Opts {
        calibrated: false,
        start: false,
        platform: Some(unreliable()),
        ..Opts::default()
    });
    h.monitors.lock().push(third());
    h.controller.start();
    h.p().cursor.track = false;
    h.feed(&obs(RIGHT_CENTRE), 0.5, 0.1);
    assert_eq!(h.switched(), [1]); // the pointer is on B

    *h.monitors.lock() = vec![b_moved, c_moved];
    h.controller.refresh_monitors();
    assert_eq!(h.state(), S::Tracking);
    assert_eq!(h.controller.assumed_monitor, Some(0)); // still B, now on the left
    h.feed(&obs(RIGHT_CENTRE), 1.0, 0.1); // look at C, where B used to be
    assert_eq!(h.switched(), [1, 1]);
}

#[test]
fn test_same_monitor_prefers_names_then_places() {
    let same = same_monitor;
    let a = Monitor::new(0, "DP-1", left().rect);
    let b = Monitor::new(1, "DP-2", right().rect);
    let c = Monitor::new(2, "HDMI-1", third().rect);
    let abc = [a.clone(), b.clone(), c.clone()];
    // Unchanged, and renumbered with the coordinates kept.
    assert_eq!(same(&b, &abc, &abc), Some(&b));
    let renumbered_b = Monitor::new(0, "DP-2", right().rect);
    let after = [
        renumbered_b.clone(),
        Monitor::new(1, "HDMI-1", third().rect),
    ];
    assert_eq!(same(&b, &abc, &after), Some(&renumbered_b));
    // Laid out anew: the name follows the monitor.
    let moved = [
        Monitor::new(0, "DP-2", left().rect),
        Monitor::new(1, "HDMI-1", right().rect),
    ];
    assert_eq!(same(&b, &abc, &moved), Some(&moved[0]));
    // Names that cannot tell (duplicates, positional fallbacks): the same place.
    let twins = [
        Monitor::new(0, "U2419H", left().rect),
        Monitor::new(1, "U2419H", right().rect),
    ];
    assert_eq!(same(&twins[1], &twins, &twins[..1]), None); // the right one was unplugged
    let fallback: Vec<Monitor> = [left(), right(), third()]
        .iter()
        .enumerate()
        .map(|(i, m)| Monitor::new(i as i32, format!("Screen {}", i + 1), m.rect))
        .collect();
    let renumbered = [
        Monitor::new(0, "Screen 1", right().rect),
        Monitor::new(1, "Screen 2", third().rect),
    ];
    assert_eq!(
        same(&fallback[1], &fallback, &renumbered),
        Some(&renumbered[0])
    );
    // Gone: whatever covers its centre now, if anything.
    assert_eq!(same(&c, &abc, &abc[..2]), None);
    let wider = Monitor::new(0, "DP-9", Rect::new(0, 0, 5760, 1080));
    assert_eq!(same(&c, &abc, std::slice::from_ref(&wider)), Some(&wider));
}

// ------------------------------------------------------------ r2-controller-06
#[test]
fn test_failed_lock_after_the_displays_went_off_wakes_them_on_return() {
    let _dirs = app_dirs();
    let mut s = make_settings();
    s.presence.action = "lock_and_display_off".into();
    let platform = Arc::new(FakePlatform::new());
    platform.p().lock_ok = false;
    let mut h = make_controller(Opts {
        platform: Some(platform),
        ..opts(s)
    });
    h.feed(&no_face(), 10.0, 0.5);
    assert_eq!(h.platform.names(), ["display_off", "lock_screen"]);
    assert_eq!(h.titles(), ["Could not lock the screen"]);
    h.push(obs((500.0, 500.0)), 0.1); // back, face only (no input)
    assert_eq!(
        h.platform.names(),
        ["display_off", "lock_screen", "wake_display"]
    );
    assert_eq!(h.state(), S::Tracking);
}

#[test]
fn test_failed_lock_from_the_thread_after_the_displays_went_off_wakes_them() {
    let _dirs = app_dirs();
    let mut s = make_settings();
    s.presence.action = "lock_and_display_off".into();
    let platform = slow_platform();
    platform.p().lock_ok = false;
    let mut h = make_controller(threaded(platform.clone(), Some(s)));
    h.feed(&no_face(), 10.0, 0.5);
    assert!(platform.entered.wait(5.0));
    finish_lock(&mut h);
    h.push(obs((500.0, 500.0)), 0.1);
    assert_eq!(
        h.platform.names(),
        ["display_off", "lock_screen", "wake_display"]
    );
}

#[test]
fn test_successful_lock_and_display_off_does_not_wake_the_displays() {
    let _dirs = app_dirs();
    let mut s = make_settings();
    s.presence.action = "lock_and_display_off".into();
    let mut h = make_controller(opts(s));
    h.feed(&no_face(), 10.0, 0.5);
    assert_eq!(h.platform.names(), ["display_off", "lock_screen"]);
    h.push(obs((500.0, 500.0)), 0.1); // back before the lock was noticed
    assert!(!h.platform.names().contains(&"wake_display"));
}

// ----------------------------------------------------------------- split panes
fn terminal() -> WindowRef {
    win(0x7E, 777, Rect::new(0, 0, 1920, 1080))
}

fn terminal_app() -> AppIdentity {
    AppIdentity {
        process: "windowsterminal".into(),
        app_id: "CASCADIA_HOSTING_WINDOW_CLASS".into(),
    }
}

fn app(process: &str, app_id: &str) -> AppIdentity {
    AppIdentity {
        process: process.into(),
        app_id: app_id.into(),
    }
}

const LEFT_PANE: (f64, f64) = (480.0, 540.0);
const RIGHT_PANE: (f64, f64) = (1440.0, 540.0);
const RIGHT_PANE_CENTRE: (i32, i32) = (1440, 540);

fn pid(id: &str) -> PaneId {
    PaneId::Str(id.into())
}

/// What [`PaneProviderFake`] records and answers.
struct ProviderState {
    calls: Vec<(&'static str, String)>,
    focused: String,
    focus_ok: bool,
    /// The panes and where they are (changed by tests to re-split the window).
    rects: Vec<(String, Rect)>,
}

/// Two side-by-side panes on the left monitor; records every call.
#[derive(Clone)]
struct PaneProviderFake(Arc<Mutex<ProviderState>>, String);

impl PaneProviderFake {
    fn new(name: &str) -> Self {
        Self(
            Arc::new(Mutex::new(ProviderState {
                calls: Vec::new(),
                focused: "%0".into(),
                focus_ok: true,
                rects: vec![
                    ("%0".into(), Rect::new(0, 0, 960, 1080)),
                    ("%1".into(), Rect::new(960, 0, 960, 1080)),
                ],
            })),
            name.into(),
        )
    }

    fn calls(&self) -> Vec<(&'static str, String)> {
        self.0.lock().calls.clone()
    }

    fn names(&self) -> Vec<&'static str> {
        self.calls().into_iter().map(|(name, _)| name).collect()
    }

    fn called(&self, name: &str, arg: &str) -> bool {
        self.calls().iter().any(|(n, a)| *n == name && a == arg)
    }

    fn focus_calls(&self) -> Vec<String> {
        self.calls()
            .into_iter()
            .filter(|(n, _)| *n == "focus")
            .map(|(_, a)| a)
            .collect()
    }
}

fn handle_text(handle: &WindowHandle) -> String {
    format!("{handle:?}")
}

impl PaneProvider for PaneProviderFake {
    fn name(&self) -> &str {
        &self.1
    }

    fn applies(&mut self, app: &AppIdentity) -> bool {
        self.0.lock().calls.push(("applies", app.process.clone()));
        true
    }

    fn detect(
        &mut self,
        window: &WindowRef,
        _app: &AppIdentity,
    ) -> Result<Option<PaneSnapshot>, PaneError> {
        let mut state = self.0.lock();
        state.calls.push(("detect", handle_text(&window.handle)));
        let panes = state
            .rects
            .iter()
            .map(|(id, rect)| Pane::new(id.as_str(), *rect, state.focused == *id, self.1.clone()))
            .collect();
        Ok(Some(PaneSnapshot {
            window_handle: window.handle.clone(),
            panes,
            taken_at: 0.0,
        }))
    }

    fn focus(&mut self, _window: &WindowRef, pane: &Pane) -> Result<bool, PaneError> {
        let mut state = self.0.lock();
        state.calls.push(("focus", pane.id.to_string()));
        if state.focus_ok {
            state.focused = pane.id.to_string();
        }
        Ok(state.focus_ok)
    }
}

fn pane_settings() -> Settings {
    let mut s = make_settings();
    s.panes.enabled = true;
    s
}

/// Records how it was stopped (`stop` joins the thread, `request_stop` does not).
struct RecordingPaneWorker {
    inner: PaneWorker,
    stops: Arc<Mutex<Vec<&'static str>>>,
}

impl PaneWorkerLike for RecordingPaneWorker {
    fn start(&mut self) {
        self.inner.start();
    }

    fn stop(&mut self, timeout: f64) {
        self.stops.lock().push("stop");
        // As PaneWorker.stop: request the stop, then wait.
        self.request_stop();
        PaneWorkerLike::stop(&mut self.inner, timeout);
    }

    fn request_stop(&mut self) {
        self.stops.lock().push("request_stop");
        self.inner.request_stop();
    }

    fn set_registry(&self, registry: PaneRegistry) {
        self.inner.set_registry(registry);
    }

    fn disabled_providers(&self) -> HashSet<String> {
        self.inner.disabled_providers()
    }

    fn drain_events(&self) -> Vec<PaneEvent> {
        self.inner.drain_events()
    }

    fn request_detect(&self, window: &WindowRef, app: &AppIdentity) -> u64 {
        self.inner.request_detect(window, app)
    }

    fn request_focus(&self, window: &WindowRef, pane: &Pane) {
        self.inner.request_focus(window, pane);
    }
}

#[derive(Default)]
struct PaneOpts {
    settings: Option<Settings>,
    app: Option<AppIdentity>,
    /// The provider's name (`"fake"` by default).
    provider: Option<&'static str>,
    stops: Option<Arc<Mutex<Vec<&'static str>>>>,
    trace_path: Option<std::path::PathBuf>,
}

/// A controller whose pane worker is synchronous and whose provider is a fake.
fn make_pane_controller(o: PaneOpts) -> (Harness, PaneProviderFake) {
    let provider = PaneProviderFake::new(o.provider.unwrap_or("fake"));
    let platform = Arc::new(FakePlatform::new());
    platform
        .p()
        .apps
        .insert(terminal().handle, o.app.unwrap_or_else(terminal_app));
    platform.p().foreground = Some(terminal());
    let clock = FakeClock::new(100.0);
    let registry_provider = provider.clone();
    let worker_clock = clock.clone();
    let stops = o.stops;
    let mut h = make_controller(Opts {
        settings: Some(o.settings.unwrap_or_else(pane_settings)),
        platform: Some(platform),
        start: false,
        clock: Some(clock),
        trace_path: o.trace_path,
        pane_registry_factory: Some(Box::new(move |s: &Settings| {
            PaneRegistry::new(vec![Box::new(registry_provider.clone())])
                .with_desktop_apps(s.panes.desktop_apps)
        })),
        pane_worker_factory: Some(Box::new(move |registry| {
            // Created on the first window poll.
            let inner = PaneWorker::new(
                registry,
                PaneWorkerOptions {
                    clock: worker_clock.clock(),
                    synchronous: true,
                    ..PaneWorkerOptions::default()
                },
            );
            match &stops {
                Some(stops) => Box::new(RecordingPaneWorker {
                    inner,
                    stops: stops.clone(),
                }) as Box<dyn PaneWorkerLike>,
                None => Box::new(inner),
            }
        })),
        ..Opts::default()
    });
    h.controller.start();
    (h, provider)
}

fn pane_status(h: &mut Harness) -> Value {
    Value::Object(h.controller.status())["panes"].clone()
}

#[test]
fn test_split_panes_are_off_by_default() {
    let _dirs = app_dirs();
    let mut s = make_settings();
    s.switching.focus_window = false;
    let (mut h, provider) = make_pane_controller(PaneOpts {
        settings: Some(s),
        ..PaneOpts::default()
    });
    settle_mouse(&mut h);
    h.feed(&obs(RIGHT_PANE), 3.0, 0.1);
    assert!(provider.calls().is_empty());
    assert!(!h.platform.names().contains(&"window_app"));
    let status = pane_status(&mut h);
    assert_eq!(status["enabled"], false);
    assert_eq!(status["switches"], 0);
}

#[test]
fn test_gaze_on_another_pane_moves_only_the_keyboard_focus() {
    let _dirs = app_dirs();
    let mut s = pane_settings();
    s.panes.move_cursor = false; // the cursor stays too (see the tests below for the default)
    let (mut h, provider) = make_pane_controller(PaneOpts {
        settings: Some(s),
        ..PaneOpts::default()
    });
    settle_mouse(&mut h); // also the first window polls: the panes are known
    assert!(provider.called("detect", &handle_text(&terminal().handle)));
    h.feed(&obs(LEFT_PANE), 1.0, 0.1);
    assert!(!provider.names().contains(&"focus"));
    h.feed(&obs(RIGHT_PANE), 1.0, 0.1);
    assert_eq!(provider.focus_calls(), ["%1"]);
    // No cursor warp, no window activation, no monitor switch.
    assert!(moves(&h).is_empty());
    assert!(!h.platform.names().contains(&"activate_window"));
    assert!(h.switched().is_empty());
    let status = pane_status(&mut h);
    assert_eq!(status["switches"], 1);
    assert_eq!(status["provider"], "fake");
    assert_eq!(status["panes"], 2);
    assert_eq!(status["eligible"], 1);
    assert!(!status["sigma_px"].is_null());
    // The focus now is where the user looks: nothing more happens.
    h.feed(&obs(RIGHT_PANE), 2.0, 0.1);
    assert_eq!(provider.focus_calls().len(), 1);
}

/// Work in the left (focused) pane, then look at the right one long enough.
fn look_at_the_right_pane(h: &mut Harness) {
    h.feed(&obs(LEFT_PANE), 1.0, 0.1);
    h.feed(&obs(RIGHT_PANE), 1.0, 0.1);
}

#[test]
fn test_the_cursor_follows_into_the_pane_centre() {
    let _dirs = app_dirs();
    let (mut h, provider) = make_pane_controller(PaneOpts::default());
    settle_mouse(&mut h); // the cursor rests at (500, 500): in the left pane
    look_at_the_right_pane(&mut h);
    assert!(provider.called("focus", "%1"));
    assert_eq!(moves(&h), [RIGHT_PANE_CENTRE]);
    assert!(!h.platform.names().contains(&"activate_window"));
    assert!(h.switched().is_empty()); // no monitor switch
}

#[test]
fn test_the_cursor_returns_to_where_it_was_left_in_the_pane() {
    let _dirs = app_dirs();
    let (mut h, _provider) = make_pane_controller(PaneOpts::default());
    settle_mouse(&mut h);
    set_cursor(&h, (1700, 300)); // the user worked in the right pane ...
    h.tick(0.1);
    set_cursor(&h, (500, 500)); // ... and went back to the left one
    h.tick(0.1);
    settle_mouse(&mut h);
    look_at_the_right_pane(&mut h);
    assert_eq!(moves(&h), [(1700, 300)]);
}

#[test]
fn test_a_remembered_spot_outside_the_pane_now_is_ignored() {
    let _dirs = app_dirs();
    let (mut h, provider) = make_pane_controller(PaneOpts::default());
    settle_mouse(&mut h);
    set_cursor(&h, (1700, 300));
    h.tick(0.1);
    set_cursor(&h, (500, 500));
    h.tick(0.1);
    // The right pane shrinks to the lower half: (1700, 300) is no longer in it.
    provider.0.lock().rects[1].1 = Rect::new(960, 540, 960, 540);
    settle_mouse(&mut h); // the next refreshes see the new layout
    h.feed(&obs(LEFT_PANE), 1.5, 0.1);
    h.feed(&obs((1440.0, 810.0)), 1.0, 0.1);
    assert!(provider.called("focus", "%1"));
    assert_eq!(moves(&h), [(1440, 810)]); // the centre of the pane as it is now
}

#[test]
fn test_no_cursor_move_when_the_focus_failed() {
    let _dirs = app_dirs();
    let (mut h, provider) = make_pane_controller(PaneOpts::default());
    provider.0.lock().focus_ok = false;
    settle_mouse(&mut h);
    look_at_the_right_pane(&mut h);
    assert!(provider.called("focus", "%1"));
    assert!(moves(&h).is_empty());
}

#[test]
fn test_no_cursor_move_when_it_is_already_in_the_pane() {
    let _dirs = app_dirs();
    let (mut h, provider) = make_pane_controller(PaneOpts::default());
    set_cursor(&h, (1500, 900)); // resting in the right pane, focus in the left one
    settle_mouse(&mut h);
    look_at_the_right_pane(&mut h);
    assert!(provider.called("focus", "%1"));
    assert!(moves(&h).is_empty());
}

#[test]
fn test_our_own_cursor_move_is_not_mouse_use() {
    let _dirs = app_dirs();
    let (mut h, provider) = make_pane_controller(PaneOpts::default());
    settle_mouse(&mut h);
    look_at_the_right_pane(&mut h);
    assert_eq!(moves(&h), [RIGHT_PANE_CENTRE]);
    let mouse_before = h.controller.input.last_mouse_activity();
    h.feed(&obs(RIGHT_PANE), 1.0, 0.1); // polls see the cursor at its new place
    assert_eq!(h.controller.input.last_mouse_activity(), mouse_before);
    // So the mouse grace does not hold the next pane switch: back to the left pane after
    // the cooldown and the dwell, well within the mouse grace (1.5 s).
    h.feed(&obs(LEFT_PANE), 1.0, 0.1);
    assert_eq!(provider.focus_calls(), ["%1", "%0"]);
}

#[test]
fn test_refused_cursor_moves_count_towards_the_warp_backoff() {
    let _dirs = app_dirs();
    let (mut h, provider) = make_pane_controller(PaneOpts::default());
    settle_mouse(&mut h);
    h.p().cursor.allow = false;
    look_at_the_right_pane(&mut h);
    assert!(provider.called("focus", "%1")); // the focus moved anyway
    assert_eq!(moves(&h), [RIGHT_PANE_CENTRE]);
    assert_eq!(h.controller.warp_refusals, 1);
}

#[test]
fn test_remembered_spots_of_closed_panes_and_windows_are_forgotten() {
    let _dirs = app_dirs();
    let (mut h, provider) = make_pane_controller(PaneOpts::default());
    settle_mouse(&mut h);
    set_cursor(&h, (1700, 300));
    h.tick(0.1);
    assert_eq!(h.controller.pane_cursor.len(), 1);
    let window = h.controller.pane_cursor[0].1.clone();
    assert_eq!(h.controller.pane_cursor[0].2[&pid("%1")], (1700, 300));
    // The right pane closes (another one opens): its spot goes.
    provider.0.lock().rects = vec![
        ("%0".into(), Rect::new(0, 0, 960, 1080)),
        ("%2".into(), Rect::new(960, 0, 960, 1080)),
    ];
    h.feed(&obs(LEFT_PANE), 1.5, 0.1);
    assert!(!h.controller.pane_cursor[0].2.contains_key(&pid("%1")));
    // The window closes while another one has the focus: its spots go.
    h.p().invalid_windows.insert(window.handle.clone());
    h.p().foreground = Some(win(0x99, 5, Rect::new(0, 0, 800, 600)));
    h.tick(0.6);
    assert!(h.controller.pane_cursor.is_empty());
}

#[test]
fn test_status_never_carries_titles_or_paths() {
    let _dirs = app_dirs();
    let (mut h, _provider) = make_pane_controller(PaneOpts::default());
    settle_mouse(&mut h);
    let status = pane_status(&mut h);
    let keys: HashSet<&str> = status
        .as_object()
        .expect("an object")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        keys,
        HashSet::from([
            "enabled",
            "supported",
            "provider",
            "panes",
            "eligible",
            "switches",
            "sigma_px",
            "failed_providers",
        ])
    );
    serde_json::to_string(&status).expect("JSON-safe for `eye-tracker ctl status`");
}

#[test]
fn test_denied_apps_are_never_inspected() {
    let _dirs = app_dirs();
    let (mut h, provider) = make_pane_controller(PaneOpts {
        app: Some(app("code", "Chrome_WidgetWin_1")),
        ..PaneOpts::default()
    });
    settle_mouse(&mut h);
    h.feed(&obs(RIGHT_PANE), 3.0, 0.1);
    assert!(provider.calls().is_empty()); // not even applies()
    assert_eq!(pane_status(&mut h)["panes"], 0);
}

#[test]
fn test_desktop_apps_are_inspected_only_when_desktop_apps_is_on() {
    for process in ["claude", "chatgpt"] {
        let _dirs = app_dirs();
        let (mut h, provider) = make_pane_controller(PaneOpts {
            app: Some(app(process, "Chrome_WidgetWin_1")),
            provider: Some("desktop_apps"),
            ..PaneOpts::default()
        });
        settle_mouse(&mut h);
        h.feed(&obs(RIGHT_PANE), 3.0, 0.1);
        assert!(provider.calls().is_empty()); // denied: not even applies()
        let mut on = h.controller.settings().clone();
        on.panes.desktop_apps = true;
        h.controller.apply_settings(&on);
        h.feed(&obs(RIGHT_PANE), 3.0, 0.1);
        assert!(provider.called("detect", &handle_text(&terminal().handle)));
        assert!(provider.called("focus", "%1"));
    }
}

#[test]
fn test_desktop_apps_never_opens_other_chromium_apps() {
    let _dirs = app_dirs();
    let mut s = pane_settings();
    s.panes.desktop_apps = true;
    let (mut h, provider) = make_pane_controller(PaneOpts {
        settings: Some(s),
        app: Some(app("code", "Chrome_WidgetWin_1")),
        provider: Some("desktop_apps"),
        ..PaneOpts::default()
    });
    settle_mouse(&mut h);
    h.feed(&obs(RIGHT_PANE), 3.0, 0.1);
    assert!(provider.calls().is_empty());
}

#[test]
fn test_pane_dwell_raises_the_frame_rate() {
    let _dirs = app_dirs();
    let mut s = pane_settings();
    s.panes.dwell_ms = 2000;
    let (mut h, _provider) = make_pane_controller(PaneOpts {
        settings: Some(s),
        ..PaneOpts::default()
    });
    settle_mouse(&mut h);
    h.feed(&obs(LEFT_PANE), 1.0, 0.1);
    assert!(!h.controller.pending);
    h.feed(&obs(RIGHT_PANE), 0.5, 0.1);
    assert!(h.controller.pending);
}

#[test]
fn test_no_pane_switching_right_after_a_monitor_switch() {
    let _dirs = app_dirs();
    let (mut h, provider) = make_pane_controller(PaneOpts::default());
    settle_mouse(&mut h);
    set_cursor(&h, (2500, 500)); // the user works on the right monitor ...
    h.tick(0.1);
    settle_mouse(&mut h);
    h.feed(&obs(RIGHT_PANE), 0.5, 0.1); // ... and looks back at the terminal on the left
    assert_eq!(h.switched(), [0]);
    // The monitor switch armed the pause (1.5 s): the pane dwell (0.6 s) waits for it.
    h.feed(&obs(RIGHT_PANE), 1.0, 0.1);
    assert!(!provider.names().contains(&"focus"));
    h.feed(&obs(RIGHT_PANE), 1.5, 0.1);
    assert!(provider.called("focus", "%1"));
}

#[test]
fn test_typing_holds_pane_switching() {
    let _dirs = app_dirs();
    let (mut h, provider) = make_pane_controller(PaneOpts::default());
    settle_mouse(&mut h);
    h.p().key_idle = Some(0.0); // typing in the focused pane
    h.feed(&obs(LEFT_PANE), 1.0, 0.1);
    h.p().key_idle = Some(1000.0);
    h.feed(&obs(RIGHT_PANE), 2.0, 0.1);
    assert!(!provider.names().contains(&"focus")); // the pane typing grace is 3 s
    h.feed(&obs(RIGHT_PANE), 1.5, 0.1);
    assert!(provider.called("focus", "%1"));
}

#[test]
fn test_turning_split_panes_off_stops_the_worker() {
    let _dirs = app_dirs();
    let (mut h, provider) = make_pane_controller(PaneOpts::default());
    settle_mouse(&mut h);
    assert!(h.controller.pane_worker.is_some());
    let mut off = h.controller.settings().clone();
    off.panes.enabled = false;
    h.controller.apply_settings(&off);
    assert!(h.controller.pane_worker.is_none());
    provider.0.lock().calls.clear();
    h.feed(&obs(RIGHT_PANE), 3.0, 0.1);
    assert!(provider.calls().is_empty());
    let mut on = h.controller.settings().clone();
    on.panes.enabled = true;
    h.controller.apply_settings(&on);
    h.feed(&obs(RIGHT_PANE), 3.0, 0.1);
    assert!(provider.called("focus", "%1"));
}

#[test]
fn test_default_pane_providers_follow_the_settings() {
    let _dirs = app_dirs();
    let h = make_controller(opts(pane_settings()));
    let registry = default_pane_registry(h.controller.platform(), h.controller.settings());
    let names: Vec<&str> = registry.providers().iter().map(|p| p.name()).collect();
    // Windows Terminal (UI Automation) is offered on Windows only.
    let expected: &[&str] = if cfg!(windows) {
        &["wezterm", "windows_terminal", "tmux"]
    } else {
        &["wezterm", "tmux"]
    };
    assert_eq!(names, expected);
    // Not started: nothing asked for a window yet, so no thread.
    assert!(h.controller.pane_worker.is_none());
}

#[test]
fn test_unsupported_platform_never_starts_split_panes() {
    let _dirs = app_dirs();
    let platform = Arc::new(FakePlatform::new());
    platform.p().panes_capable = false;
    platform.p().foreground = Some(terminal());
    platform.p().apps.insert(terminal().handle, terminal_app());
    let mut h = make_controller(Opts {
        platform: Some(platform),
        ..opts(pane_settings())
    });
    settle_mouse(&mut h);
    h.feed(&obs(RIGHT_PANE), 2.0, 0.1);
    assert!(h.controller.pane_worker.is_none());
    assert_eq!(pane_status(&mut h)["supported"], false);
}

#[test]
fn test_results_for_a_window_left_meanwhile_are_ignored() {
    let _dirs = app_dirs();
    let (mut h, _provider) = make_pane_controller(PaneOpts::default());
    settle_mouse(&mut h);
    assert_eq!(pane_status(&mut h)["panes"], 2);
    h.p().foreground = Some(win(0x99, 5, Rect::new(0, 0, 800, 600)));
    h.tick(0.6); // another window (no known app): its panes are not asked for
    assert_eq!(pane_status(&mut h)["panes"], 0);
    let stale = DetectResult {
        window_handle: terminal().handle,
        snapshot: None,
        request_id: 1,
    };
    h.controller.pane_snapshot = None;
    h.controller.on_pane_detected(stale);
    assert!(h.controller.pane_snapshot.is_none());
}

#[test]
fn test_turning_split_panes_off_does_not_wait_for_the_worker() {
    let _dirs = app_dirs();
    let stops: Arc<Mutex<Vec<&'static str>>> = Arc::default();
    let (mut h, _provider) = make_pane_controller(PaneOpts {
        stops: Some(stops.clone()),
        ..PaneOpts::default()
    });
    settle_mouse(&mut h);
    let mut off = h.controller.settings().clone();
    off.panes.enabled = false;
    h.controller.apply_settings(&off); // on the GUI thread: no join
    assert_eq!(*stops.lock(), ["request_stop"]);
    let mut on = off.clone();
    on.panes.enabled = true;
    h.controller.apply_settings(&on);
    settle_mouse(&mut h);
    assert!(h.controller.pane_worker.is_some());
    h.controller.shutdown(); // quitting waits for a provider call in progress
    assert_eq!(stops.lock()[1..], ["stop", "request_stop"]);
}

#[test]
fn test_a_reused_window_handle_of_another_process_is_another_window() {
    let _dirs = app_dirs();
    let (mut h, _provider) = make_pane_controller(PaneOpts::default());
    settle_mouse(&mut h);
    assert_eq!(count(&h.platform.names(), "window_app"), 1);
    h.p().foreground = Some(WindowRef {
        pid: Some(31337),
        ..terminal()
    });
    h.tick(0.6);
    // Identified again (and with it the deny-list checked again), panes asked again.
    assert_eq!(count(&h.platform.names(), "window_app"), 2);
    assert_eq!(
        h.controller.pane_window.as_ref().expect("a window").pid,
        Some(31337)
    );
}

#[test]
fn test_detections_from_before_a_provider_swap_are_dropped() {
    let _dirs = app_dirs();
    let (mut h, _provider) = make_pane_controller(PaneOpts::default());
    settle_mouse(&mut h);
    let snapshot = h.controller.pane_snapshot.clone().expect("a snapshot");
    let before = h.controller.pane_last_request;
    let mut swapped = h.controller.settings().clone();
    swapped.panes.tmux = false; // another set of providers
    h.controller.apply_settings(&swapped);
    assert!(h.controller.pane_snapshot.is_none());
    // An answer to a request made with the old providers arrives late: ignored.
    h.controller.on_pane_detected(DetectResult {
        window_handle: terminal().handle,
        snapshot: Some(snapshot.clone()),
        request_id: before,
    });
    assert!(h.controller.pane_snapshot.is_none());
    // The next request's answer counts.
    h.controller.on_pane_detected(DetectResult {
        window_handle: terminal().handle,
        snapshot: Some(snapshot.clone()),
        request_id: before + 1,
    });
    assert_eq!(h.controller.pane_snapshot, Some(snapshot));
}

#[test]
fn test_trace_records_pane_decisions() {
    let _dirs = app_dirs();
    let tmp = tempfile::tempdir().expect("tmp");
    let path = tmp.path().join("trace.jsonl");
    let (mut h, _provider) = make_pane_controller(PaneOpts {
        trace_path: Some(path.clone()),
        ..PaneOpts::default()
    });
    settle_mouse(&mut h);
    h.feed(&obs(RIGHT_PANE), 1.0, 0.1);
    h.controller.shutdown();
    let text = std::fs::read_to_string(&path).expect("the trace");
    let rows: Vec<Value> = text
        .lines()
        .map(|line| serde_json::from_str(line).expect("JSON"))
        .collect();
    let reasons: Vec<&Value> = rows.iter().filter_map(|r| r.get("pane_reason")).collect();
    assert!(reasons.contains(&&json!("dwell")));
    assert!(reasons.contains(&&json!("switch")));
    let fired = rows
        .iter()
        .find(|r| r["pane_reason"] == "switch")
        .expect("a switch");
    assert_eq!(fired["pane_target"], "%1");
    assert_eq!(fired["n_panes"], 2);
}
