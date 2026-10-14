//! Tests for the settings dialog (the settings part of `tests/test_ui_dialogs.py`).
//!
//! The dialog's state runs against a fake controller that applies the
//! [`SettingsAction`]s as the app would. Start at login, camera probing and the
//! diagnostics report are fakes, so nothing here touches the OS, a camera or the desktop.
//!
//! Not ported:
//! * `test_recalibrate_button_requests_calibration`: it clicked the Qt button; the egui
//!   button pushes [`SettingsAction::CalibrationRequested`] in the drawing code.
//! * `test_a_linux_device_path_in_use_is_not_probed`: it monkeypatched
//!   `vision.camera.device_index`; `vision/camera_tests.rs` covers the link resolution and
//!   [`camera_in_use`] only passes the path on.
//! * `test_setup_assistant_runs_now_and_leaves_the_walk_away_lock_alone` is ported without
//!   its button click (the egui button pushes [`SettingsAction::SetupRequested`]).
//! * The calibration window tests belong to `calibration_window_tests.rs`, the wizard tests
//!   to `wizard_tests.rs`.
//! * `test_diagnostics_module_absence_is_handled`: the report module is always built in, so
//!   [`load_diagnostics_text`] has no fallback text for a missing one.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::json;

use super::*;
use crate::config::Settings;
use crate::platform::autostart::{AutostartBackend, AutostartError, Status};
use crate::platform::base::{GenericPlatform, PlatformServices};
use crate::platform::hotkeys::{
    Hotkey, HotkeyArg, HotkeyCallback, HotkeyError, HotkeyManager, UnsupportedHotkeyManager,
};
use crate::vision::camera::CameraInfo;

// =========================================================================== fakes
/// What the fake start-at-login entry did (shared with the test).
pub(crate) struct AutostartLog {
    pub enabled: bool,
    pub calls: Vec<String>,
    pub fail: Option<String>,
    /// What status() reports while not enabled.
    pub idle_status: Status,
}

/// Replaces the start-at-login functions (never touches the real OS entry).
pub(crate) struct FakeAutostart(pub Arc<Mutex<AutostartLog>>);

impl FakeAutostart {
    pub fn create() -> (Box<dyn AutostartBackend>, Arc<Mutex<AutostartLog>>) {
        let log = Arc::new(Mutex::new(AutostartLog {
            enabled: false,
            calls: Vec::new(),
            fail: None,
            idle_status: Status::Disabled,
        }));
        (Box::new(FakeAutostart(log.clone())), log)
    }
}

impl AutostartBackend for FakeAutostart {
    fn is_supported(&self) -> bool {
        true
    }
    fn is_enabled(&self) -> bool {
        self.0.lock().enabled
    }
    fn enable(&self, _background: bool) -> Result<(), Box<dyn std::error::Error>> {
        let mut log = self.0.lock();
        log.calls.push("enable".into());
        if let Some(fail) = &log.fail {
            return Err(Box::new(AutostartError(fail.clone())));
        }
        log.enabled = true;
        Ok(())
    }
    fn disable(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut log = self.0.lock();
        log.calls.push("disable".into());
        if let Some(fail) = &log.fail {
            return Err(Box::new(AutostartError(fail.clone())));
        }
        log.enabled = false;
        Ok(())
    }
    fn status(&self) -> Option<Status> {
        let log = self.0.lock();
        Some(if log.enabled {
            Status::Enabled
        } else {
            log.idle_status
        })
    }
}

/// Every capability claimed, as the Python FakePlatform did.
pub(crate) fn all_caps() -> HashMap<&'static str, bool> {
    GenericPlatform
        .capabilities()
        .into_keys()
        .map(|k| (k, true))
        .collect()
}

/// The `skip` set of every probe (the camera in use is never opened).
type Probes = Arc<Mutex<Vec<BTreeSet<u32>>>>;

pub(crate) fn fake_probe(found: Vec<CameraInfo>) -> (CameraProbe, Probes) {
    let probes: Probes = Arc::default();
    let seen = probes.clone();
    let probe = CameraProbe::new(Arc::new(move |_max, _api: &str, skip: &[u32]| {
        seen.lock().push(skip.iter().copied().collect());
        found.clone()
    }));
    (probe, probes)
}

pub(crate) fn two_cameras() -> Vec<CameraInfo> {
    vec![
        CameraInfo {
            index: 0,
            name: "Integrated Camera".into(),
            width: 640,
            height: 480,
        },
        CameraInfo {
            index: 2,
            name: "USB Cam".into(),
            width: 1280,
            height: 720,
        },
    ]
}

/// What the app does with the dialog's actions, against a fake controller.
struct FakeController {
    settings: Settings,
    applied: Vec<Settings>,
    suspended: Vec<bool>,
    calibration_requests: usize,
    closed: Vec<bool>,
    /// What `CollectDiagnostics` hands back; `None`: whether the hotkeys are suspended.
    report: Option<Result<String, String>>,
}

impl FakeController {
    fn new(settings: Settings) -> Self {
        Self {
            settings,
            applied: Vec::new(),
            suspended: Vec::new(),
            calibration_requests: 0,
            closed: Vec::new(),
            report: Some(Ok("FAKE DIAGNOSTICS REPORT".into())),
        }
    }

    fn apply_settings(&mut self, dialog: &mut SettingsDialog, settings: Settings) {
        self.applied.push(settings.clone());
        self.settings = settings;
        dialog.on_settings_changed(&self.settings);
    }

    /// Carry out the dialog's actions in order; returns them.
    fn run(&mut self, dialog: &mut SettingsDialog) -> Vec<SettingsAction> {
        let mut all = Vec::new();
        loop {
            let actions = dialog.take_actions();
            if actions.is_empty() {
                return all;
            }
            for action in actions {
                match &action {
                    SettingsAction::Apply(settings) => {
                        self.apply_settings(dialog, (**settings).clone());
                    }
                    SettingsAction::SuspendHotkeys(s) => self.suspended.push(*s),
                    SettingsAction::CollectDiagnostics => {
                        let report = self.report.clone().unwrap_or_else(|| {
                            Ok(format!(
                                "hotkeys suspended: {}",
                                self.suspended.last().copied().unwrap_or(false)
                            ))
                        });
                        dialog.set_diagnostics(report);
                    }
                    SettingsAction::CalibrationRequested => self.calibration_requests += 1,
                    SettingsAction::SetupRequested => {}
                    SettingsAction::Closed { accepted } => self.closed.push(*accepted),
                }
                all.push(action);
            }
        }
    }
}

struct Env {
    dialog: SettingsDialog,
    controller: FakeController,
    autostart: Arc<Mutex<AutostartLog>>,
    probes: Probes,
}

fn env_with(
    settings: Settings,
    caps: HashMap<&'static str, bool>,
    manager: Option<Arc<dyn HotkeyManager>>,
    found: Vec<CameraInfo>,
    configure: impl FnOnce(&mut AutostartLog),
) -> Env {
    let (backend, autostart) = FakeAutostart::create();
    configure(&mut autostart.lock());
    let (probe, probes) = fake_probe(found);
    let dialog = SettingsDialog::new(&settings, caps, manager, backend).with_camera_probe(probe);
    Env {
        dialog,
        controller: FakeController::new(settings),
        autostart,
        probes,
    }
}

fn env() -> Env {
    env_with(Settings::default(), all_caps(), None, two_cameras(), |_| {})
}

fn env_for(settings: Settings) -> Env {
    env_with(settings, all_caps(), None, two_cameras(), |_| {})
}

impl Env {
    fn apply(&mut self) -> bool {
        let ok = self.dialog.apply();
        self.controller.run(&mut self.dialog);
        ok
    }

    fn last_applied(&self) -> &Settings {
        self.controller
            .applied
            .last()
            .expect("settings were applied")
    }

    /// Wait for the camera probe thread (Qt: process events until the button is enabled).
    fn wait_for_cameras(&mut self) -> bool {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            self.dialog.poll_cameras();
            if self.dialog.detect_enabled() {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        false
    }
}

/// Every field changed from its default, with floats a spin box cannot show exactly.
fn non_default_settings() -> Settings {
    let mut s = Settings::default();
    s.general.backend = "lite".into();
    s.general.start_paused = true;
    s.general.first_run_done = true;
    s.general.notifications = false;
    s.general.log_level = "DEBUG".into();
    s.camera.device = "2".into();
    s.camera.width = 1280;
    s.camera.height = 720;
    s.camera.api = "dshow".into();
    s.performance.profile = "eco".into();
    s.performance.motion_gate = false;
    s.performance.motion_threshold = 3.33;
    s.switching.enabled = false;
    s.switching.dwell_ms = 450;
    s.switching.hysteresis = 0.073;
    s.switching.off_screen_margin = 0.421;
    s.switching.cooldown_ms = 900;
    s.switching.mouse_grace_ms = 1234;
    s.switching.typing_grace_ms = 2500;
    s.switching.reading_grace_ms = 7500;
    s.switching.cursor_target = "gaze".into();
    s.switching.focus_window = false;
    s.switching.smoothing = 0.377;
    s.panes.enabled = true;
    s.panes.move_cursor = false;
    s.panes.precision = 3.33;
    s.presence.enabled = false;
    s.presence.action = "display_off".into();
    s.presence.away_timeout_s = 120;
    s.presence.warning_s = 15;
    s.presence.require_input_idle = false;
    s.presence.wake_on_return = false;
    s.privacy.remember_privacy_mode = false;
    s.privacy.pause_when_locked = false;
    s.privacy.yield_camera = false;
    s.privacy.pause_for_apps = vec!["zoom.exe".into(), "obs64.exe".into()];
    s.privacy.shoulder_guard = true;
    s.privacy.guard_action = "lock".into();
    s.privacy.guard_delay_s = 3.25;
    s.hotkeys.enabled = false;
    s.hotkeys.toggle_tracking = "Ctrl+Shift+F9".into(); // not canonical: must survive untouched
    s.hotkeys.toggle_privacy = "alt+meta+p".into();
    s.hotkeys.recalibrate = "ctrl+alt+k".into();
    s.learning.adaptive = false;
    s.learning.max_samples = 800;
    s.learning.drift_alerts = false;
    s.ui.show_gaze_overlay = true;
    s
}

/// Settings without a widget: only the setup assistant sets the first-run flag
/// (while it is false the walk-away lock is only a notification). Split-pane
/// focus is experimental: the dialog offers the switch and the precision, the
/// finer tuning is in settings.json only (docs/configuration.md).
/// `panes.desktop_apps` has a checkbox: it reads another app's accessibility
/// tree, so it is a visible, deliberate choice (off by default). `general.coreml`
/// (macOS: Core ML or tract for the landmark model) is a settings.json switch.
fn unbound() -> BTreeSet<String> {
    let mut keys: BTreeSet<String> = [
        "dwell_ms",
        "typing_grace_ms",
        "reading_grace_ms",
        "cooldown_ms",
        "after_monitor_switch_ms",
        "hysteresis",
        "min_pane_px",
        "tmux",
        "wezterm",
        "windows_terminal",
    ]
    .iter()
    .map(|name| format!("panes.{name}"))
    .collect();
    keys.insert("general.first_run_done".into());
    keys.insert("general.coreml".into());
    keys
}

// ================================================================== settings dialog
#[test]
fn test_every_setting_has_a_widget() {
    let e = env();
    let keys: BTreeSet<String> = describe_settings().into_iter().map(|r| r.key).collect();
    let bound: BTreeSet<String> = e
        .dialog
        .bound_keys()
        .into_iter()
        .map(String::from)
        .collect();
    let expected: BTreeSet<String> = keys.difference(&unbound()).cloned().collect();
    assert_eq!(bound, expected);
}

#[test]
fn test_tooltips_come_from_the_settings_documentation() {
    let e = env();
    let unbound = unbound();
    for row in describe_settings() {
        if row.doc.is_empty() || unbound.contains(&row.key) {
            continue;
        }
        assert!(e.dialog.tooltip(&row.key).contains(row.doc), "{}", row.key);
    }
}

/// r3-ux-docs-08: on Wayland outside GNOME input cannot cancel the countdown.
#[test]
fn test_input_that_the_system_does_not_report_is_marked() {
    let mut caps = all_caps();
    caps.insert("input_idle", false);
    let e = env_with(Settings::default(), caps, None, two_cameras(), |_| {});
    assert!(
        e.dialog
            .text("presence.require_input_idle")
            .ends_with("(not supported on this system)")
    );
}

#[test]
fn test_split_pane_options_on_the_switching_page() {
    let mut e = env();
    assert!(!e.dialog.checked("panes.enabled")); // experimental: off by default
    assert!(e.dialog.checked("panes.move_cursor")); // but once on, the cursor follows like for monitors
    assert!(!e.dialog.is_enabled("panes.precision"));
    assert!(!e.dialog.is_enabled("panes.move_cursor"));
    e.dialog.set_checked("panes.enabled", true);
    assert!(e.dialog.is_enabled("panes.precision"));
    assert!(e.dialog.is_enabled("panes.move_cursor"));
    e.dialog.set_number("panes.precision", 4.0);
    e.dialog.set_checked("panes.move_cursor", false);
    assert!(e.apply());
    let new = e.last_applied();
    assert!(new.panes.enabled);
    assert!(!new.panes.move_cursor);
    approx::assert_abs_diff_eq!(new.panes.precision, 4.0, epsilon = 1e-9);
    assert_eq!(new.panes.dwell_ms, 400); // not in the dialog: kept as it was
    assert!(!new.panes.desktop_apps); // opt-in, not switched on with the rest
    assert!(e.dialog.is_enabled("panes.desktop_apps"));
    assert!(e.dialog.text("panes.desktop_apps").contains("Claude"));
    e.dialog.set_checked("panes.desktop_apps", true);
    assert!(e.apply());
    assert!(e.last_applied().panes.desktop_apps);
    e.dialog.set_checked("panes.enabled", false);
    assert!(!e.dialog.is_enabled("panes.desktop_apps"));

    let mut caps = all_caps();
    caps.insert("panes", false);
    let e = env_with(Settings::default(), caps, None, two_cameras(), |_| {});
    assert!(
        e.dialog
            .text("panes.enabled")
            .ends_with("(not supported on this system)")
    );
}

#[test]
fn test_pages_cover_the_six_sections() {
    let mut e = env();
    let keys: Vec<&str> = PAGES.iter().map(|(key, _, _)| *key).collect();
    assert_eq!(
        keys,
        [
            "general",
            "switching",
            "presence",
            "camera",
            "hotkeys",
            "diagnostics"
        ]
    );
    for key in keys {
        e.dialog.show_page(key);
        assert_eq!(e.dialog.current_page(), key);
    }
}

#[test]
fn test_round_trip_leaves_every_value_untouched() {
    let original = non_default_settings();
    let mut e = env_for(original.clone());
    let shown = e.dialog.settings();
    assert_eq!(shown, original);
    assert!(!e.dialog.is_modified());
    assert!(!e.dialog.apply_enabled());
    // OK without edits closes without reconfiguring anything.
    assert!(e.dialog.accept());
    e.controller.run(&mut e.dialog);
    assert!(e.controller.applied.is_empty());
    assert_eq!(e.controller.closed, [true]);
}

#[test]
fn test_apply_hands_an_edited_copy_to_the_controller() {
    let mut e = env();
    let before = e.controller.settings.clone();
    e.dialog.set_number("switching.dwell_ms", 700.0);
    e.dialog.select("presence.action", "display_off");
    e.dialog
        .set_text("privacy.pause_for_apps", "zoom.exe\n\n  teams.exe  \n");

    assert!(e.dialog.is_modified());
    assert!(e.dialog.apply_enabled());
    e.dialog.apply();
    let actions = e.controller.run(&mut e.dialog);

    assert_eq!(e.controller.applied.len(), 1);
    let new = e.controller.applied[0].clone();
    assert_eq!(new.switching.dwell_ms, 700);
    assert_eq!(new.presence.action, "display_off");
    assert_eq!(new.privacy.pause_for_apps, ["zoom.exe", "teams.exe"]);
    let mut expected = before.clone();
    expected.switching.dwell_ms = 700;
    expected.presence.action = "display_off".into();
    expected.privacy.pause_for_apps = vec!["zoom.exe".into(), "teams.exe".into()];
    assert_eq!(new, expected);
    assert_eq!(actions, [SettingsAction::Apply(Box::new(new))]);
    // The applied state is the new baseline.
    assert!(!e.dialog.is_modified());
    assert!(!e.dialog.apply_enabled());
}

#[test]
fn test_float_edits_are_converted_back_from_display_units() {
    let mut e = env();
    e.dialog.set_number("switching.hysteresis", 12.0);
    e.dialog.set_number("switching.smoothing", 80.0);
    assert!(e.apply());
    let new = e.last_applied();
    approx::assert_abs_diff_eq!(new.switching.hysteresis, 0.12, epsilon = 1e-9);
    approx::assert_abs_diff_eq!(new.switching.smoothing, 0.8, epsilon = 1e-9);
}

#[test]
fn test_dependent_widgets_follow_their_master_checkbox() {
    let mut e = env();
    e.dialog.set_checked("switching.enabled", true);
    assert!(e.dialog.is_enabled("switching.dwell_ms"));
    e.dialog.set_checked("switching.enabled", false);
    assert!(!e.dialog.is_enabled("switching.dwell_ms"));
}

#[test]
fn test_restore_defaults_keeps_the_first_run_flag() {
    let mut e = env_for(non_default_settings());
    e.dialog.restore_defaults();
    let mut expected = Settings::default();
    expected.general.first_run_done = true;
    assert_eq!(e.dialog.settings(), expected);
    assert!(e.dialog.is_modified());
}

#[test]
fn test_external_settings_change_refreshes_untouched_widgets_only() {
    let mut e = env();
    e.dialog.set_number("switching.dwell_ms", 1500.0);
    let mut other = e.controller.settings.clone();
    other.general.notifications = false; // e.g. changed from the tray
    other.switching.dwell_ms = 100;
    e.dialog.on_settings_changed(&other);
    let shown = e.dialog.settings();
    assert!(!shown.general.notifications);
    assert_eq!(shown.switching.dwell_ms, 1500); // the user's pending edit wins
}

#[test]
fn test_presence_page_promises_that_nothing_leaves_the_computer() {
    let e = env();
    assert!(
        e.dialog
            .texts()
            .iter()
            .any(|t| t.contains("Everything stays on this computer"))
    );
}

// ------------------------------------------------------------------------ hotkeys
#[test]
fn test_hotkey_edit_canonicalises_valid_text() {
    let mut e = env();
    e.dialog.set_hotkey("toggle_tracking", "Ctrl + Alt + X");
    assert_eq!(
        e.dialog.hotkey_edit("toggle_tracking").unwrap().hotkey(),
        "ctrl+alt+x"
    );
    assert!(e.dialog.is_valid());
}

#[test]
fn test_invalid_hotkeys_block_apply() {
    for (text, fragment) in [
        ("x", "modifier"),
        ("ctrl+alt", "needs a key"),
        ("ctrl+alt+nosuchkey", "unsupported key"),
        ("shift+a", "Shift alone"),
    ] {
        let mut e = env();
        e.dialog.set_hotkey("toggle_privacy", text);
        assert!(!e.dialog.is_valid());
        let errors = e.dialog.hotkey_errors();
        let error = &errors
            .iter()
            .find(|(a, _)| *a == "toggle_privacy")
            .unwrap()
            .1;
        assert!(error.contains(fragment), "{error}");
        assert!(!e.dialog.ok_enabled());
        assert!(!e.apply());
        assert!(e.controller.applied.is_empty());
        // Fixing it clears the error.
        e.dialog.set_hotkey("toggle_privacy", "ctrl+alt+j");
        assert!(e.dialog.is_valid());
        assert!(e.dialog.ok_enabled());
    }
}

#[test]
fn test_duplicate_hotkeys_are_rejected() {
    let mut e = env();
    e.dialog
        .set_hotkey("toggle_tracking", &Settings::default().hotkeys.recalibrate);
    let errors = e.dialog.hotkey_errors();
    assert_eq!(errors.len(), 1);
    assert!(errors[0].1.contains("same shortcut"));
}

#[test]
fn test_recorded_hotkey_is_applied() {
    let mut e = env();
    let held = HeldModifiers {
        ctrl: true,
        alt: true,
        ..Default::default()
    };
    // egui reports Command as `mac_cmd` (our "meta") and Control as `ctrl` on macOS too.
    e.dialog.hotkey_key_press("recalibrate", egui::Key::K, held);
    assert_eq!(
        e.dialog.hotkey_edit("recalibrate").unwrap().hotkey(),
        "ctrl+alt+k"
    );
    e.dialog.hotkey_key_press(
        "recalibrate",
        egui::Key::Backspace,
        HeldModifiers::default(),
    );
    assert_eq!(e.dialog.hotkey_edit("recalibrate").unwrap().hotkey(), "");
    e.dialog.set_hotkey("recalibrate", "ctrl+alt+k");
    assert!(e.apply());
    assert_eq!(e.last_applied().hotkeys.recalibrate, "ctrl+alt+k");
}

#[test]
fn test_hotkey_text_from_key_events() {
    let mods = |names: &[&str]| HeldModifiers {
        ctrl: names.contains(&"ctrl"),
        alt: names.contains(&"alt"),
        shift: names.contains(&"shift"),
        meta: names.contains(&"meta"),
    };
    assert_eq!(
        hotkey_text_from_key(egui::Key::ControlLeft, mods(&["ctrl"])),
        None
    );
    assert_eq!(
        hotkey_text_from_key(egui::Key::P, mods(&["ctrl", "alt"])).as_deref(),
        Some("ctrl+alt+p")
    );
    assert_eq!(
        hotkey_text_from_key(egui::Key::F5, mods(&["shift"])).as_deref(),
        Some("shift+f5")
    );
    // Shift+1 arrives as "!"; hotkeys bind the physical key.
    assert_eq!(
        hotkey_text_from_key(egui::Key::Exclamationmark, mods(&["ctrl", "shift"])).as_deref(),
        Some("ctrl+shift+1")
    );
    // macOS: egui reports Command as mac_cmd (meta) and Control as ctrl.
    let command_shift = egui::Modifiers {
        shift: true,
        mac_cmd: true,
        command: true,
        ..Default::default()
    };
    assert_eq!(
        hotkey_text_from_key(egui::Key::G, HeldModifiers::from_egui(command_shift, false))
            .as_deref(),
        Some("shift+meta+g")
    );
    let control = egui::Modifiers {
        ctrl: true,
        ..Default::default()
    };
    assert_eq!(
        hotkey_text_from_key(egui::Key::G, HeldModifiers::from_egui(control, false)).as_deref(),
        Some("ctrl+g")
    );
}

// ------------------------------------------------------------------ camera & stats
#[test]
fn test_detect_cameras_fills_the_device_list() {
    let mut e = env();
    e.dialog.detect_cameras();
    assert!(e.wait_for_cameras());
    let values: Vec<String> = e
        .dialog
        .items("camera.device")
        .into_iter()
        .map(|(v, _)| v)
        .collect();
    assert_eq!(values, ["0", "2"]);
    let (value, label) = e.dialog.current("camera.device");
    assert_eq!(value, "0"); // the configured device stays selected
    assert!(label.contains("Integrated Camera"));
    e.dialog.select("camera.device", "2");
    assert!(e.apply());
    assert_eq!(e.last_applied().camera.device, "2");
}

#[test]
fn test_live_stats_are_shown() {
    let mut e = env();
    e.dialog.on_stats(&json!({
        "fps": 11.94,
        "target_fps": 12.0,
        "inference_ms": 9.51,
        "skip_ratio": 0.4,
        "cpu_percent": 0.64,
        "state": "tracking",
        "state_label": "Tracking",
        "backend": "facemesh",
    }));
    let text = e.dialog.stats_text();
    for part in [
        "Tracking",
        "11.9 fps",
        "target 12",
        "9.5 ms",
        "40 % skipped",
        "CPU 0.6 %",
    ] {
        assert!(text.contains(part), "{part} in {text}");
    }
}

#[test]
fn test_format_stats_tolerates_junk() {
    assert_eq!(format_stats(&Value::Null), "");
    assert_eq!(
        format_stats(&json!({"fps": f64::NAN, "state": "nonsense"})),
        "nonsense"
    );
    assert_eq!(
        format_stats(&json!({"state": TrackingState::Paused.value(), "cpu_percent": 1})),
        "Paused · CPU 1.0 %"
    );
}

// --------------------------------------------------------------- start at login
#[test]
fn test_start_at_login_checkbox_uses_autostart() {
    let mut e = env();
    assert!(e.dialog.autostart_enabled());
    assert!(!e.dialog.autostart_checked());
    e.dialog.set_autostart_checked(true);
    assert!(e.dialog.is_modified());
    assert!(e.apply());
    assert_eq!(e.autostart.lock().calls, ["enable"]);
    e.autostart.lock().fail = Some("access denied".into());
    e.dialog.set_autostart_checked(false);
    assert!(!e.apply());
    assert!(e.dialog.status_text().contains("access denied"));
}

// ------------------------------------------------------------------ diagnostics
#[test]
fn test_diagnostics_page_copy_and_log_folder() {
    let mut e = env();
    e.dialog.show_page("diagnostics");
    e.controller.run(&mut e.dialog);
    assert_eq!(e.dialog.diagnostics_text(), "FAKE DIAGNOSTICS REPORT");

    e.dialog.copy_diagnostics();
    assert_eq!(
        e.dialog.clipboard.as_deref(),
        Some("FAKE DIAGNOSTICS REPORT")
    );

    assert_eq!(e.dialog.open_log_folder(), file_url(&paths::log_dir()));
}

#[test]
fn test_diagnostics_failure_is_shown_not_raised() {
    let mut e = env();
    e.controller.report = Some(Err("boom".into()));
    e.dialog.refresh_diagnostics();
    e.controller.run(&mut e.dialog);
    assert!(e.dialog.diagnostics_text().contains("boom"));
}

#[test]
fn test_unsupported_hotkeys_explain_the_ctl_alternative() {
    let manager: Arc<dyn HotkeyManager> = Arc::new(UnsupportedHotkeyManager::new(Some(
        "Wayland session".into(),
    )));
    let e = env_with(
        Settings::default(),
        all_caps(),
        Some(manager),
        two_cameras(),
        |_| {},
    );
    let texts = e.dialog.texts().join(" ");
    assert!(texts.contains("Wayland session"));
    assert!(texts.contains(&html_escape(&cli_command_text(&["ctl", "toggle"]))));
}

/// Regression (r2-docs-01): packages have no "eye-tracker" command on the PATH.
#[test]
fn test_ctl_alternative_names_the_command_of_this_copy() {
    // A note that names another command (the source install's) still gets the real ones.
    let note = "Wayland: bind 'eye-tracker-of-elsewhere ctl toggle'";
    let manager: Arc<dyn HotkeyManager> =
        Arc::new(UnsupportedHotkeyManager::new(Some(note.into())));
    let e = env_with(
        Settings::default(),
        all_caps(),
        Some(manager),
        two_cameras(),
        |_| {},
    );
    let texts = e.dialog.texts().join(" ");
    for action in ["toggle", "privacy-toggle", "calibrate"] {
        let command = html_escape(&cli_command_text(&["ctl", action]));
        assert!(
            texts.contains(&format!("<code>{command}</code>")),
            "{texts}"
        );
    }

    // A note that already names them this way is not repeated.
    let note = format!("Wayland: bind '{} toggle'", cli_command_text(&["ctl"]));
    let manager: Arc<dyn HotkeyManager> = Arc::new(UnsupportedHotkeyManager::new(Some(note)));
    let e = env_with(
        Settings::default(),
        all_caps(),
        Some(manager),
        two_cameras(),
        |_| {},
    );
    assert!(!e.dialog.texts().join(" ").contains("<code>"));
}

/// Regression (r2-ui-app-01): "Show the setup assistant at next start" saved
/// first_run_done=False, which turned the walk-away lock into a notification at
/// once, and for good if the assistant was then cancelled.
#[test]
fn test_setup_assistant_runs_now_and_leaves_the_walk_away_lock_alone() {
    let mut e = env();
    let keys = e.dialog.bound_keys();
    assert!(
        !keys
            .iter()
            .any(|k| e.dialog.text(k).contains("setup assistant"))
    );
    e.dialog.restore_defaults();
    assert!(e.apply());
    assert_eq!(
        e.controller.settings.general.first_run_done,
        Settings::default().general.first_run_done
    );
    let mut finished = Settings::default();
    finished.general.first_run_done = true;
    e.controller.apply_settings(&mut e.dialog, finished);
    e.dialog.restore_defaults();
    assert!(e.apply());
    assert!(e.controller.settings.general.first_run_done);
}

#[test]
fn test_modifier_preview_uses_the_platform_key_names() {
    let meta = if cfg!(windows) { "Win" } else { "Super" };
    assert_eq!(modifier_preview(&[], None), "");
    assert_eq!(
        modifier_preview(&["ctrl", "meta"], Some(false)),
        format!("Ctrl+{meta}+…")
    );
    assert_eq!(
        modifier_preview(&["meta", "alt", "ctrl"], Some(false)),
        format!("Ctrl+Alt+{meta}+…")
    );
    assert_eq!(modifier_preview(&["ctrl", "meta"], Some(true)), "⌃⌘…");
}

/// Registered combinations never reach the dialog, so they are released while recording.
#[test]
fn test_hotkeys_are_suspended_while_a_shortcut_is_recorded() {
    let mut e = env();
    e.dialog.set_recording("toggle_tracking", true);
    e.dialog.set_recording("toggle_tracking", true); // no duplicate notification
    e.dialog.set_recording("toggle_tracking", false);
    e.dialog.flush_deferred(); // given back once focus has settled
    e.controller.run(&mut e.dialog);
    assert_eq!(e.controller.suspended, [true, false]);
    e.dialog.set_recording("toggle_tracking", true);
    // The field is hidden mid-recording (it loses the focus): the hotkeys come back.
    e.dialog.set_recording("toggle_tracking", false);
    e.dialog.flush_deferred();
    e.controller.run(&mut e.dialog);
    assert_eq!(e.controller.suspended, [true, false, true, false]);
}

/// Regression (r2-ui-app-05): every focus change re-registered all hotkeys, and
/// the controller repeated "Hotkey unavailable" for one that cannot be registered.
#[test]
fn test_moving_between_shortcut_fields_does_not_re_register_the_hotkeys() {
    let mut e = env();
    let [first, second, third] = HOTKEY_ACTIONS.map(|(action, _)| action);
    e.dialog.set_recording(first, true);
    // Focus leaves the old field, then enters the new one.
    e.dialog.set_recording(first, false);
    e.dialog.set_recording(second, true);
    e.dialog.flush_deferred();
    e.dialog.set_recording(second, false);
    e.dialog.set_recording(third, true);
    e.dialog.flush_deferred();
    e.controller.run(&mut e.dialog);
    assert_eq!(e.controller.suspended, [true]);
    e.dialog.set_recording(third, false); // focus leaves the shortcut fields
    e.controller.run(&mut e.dialog);
    assert_eq!(e.controller.suspended, [true]);
    e.dialog.flush_deferred();
    e.controller.run(&mut e.dialog);
    assert_eq!(e.controller.suspended, [true, false]);
}

#[test]
fn test_closing_the_dialog_gives_the_hotkeys_back_at_once() {
    let mut e = env();
    e.dialog.set_recording("toggle_tracking", true);
    // The deferred resume would never run once the closed dialog is dropped.
    e.dialog.reject();
    e.controller.run(&mut e.dialog);
    assert_eq!(e.controller.suspended, [true, false]);
    e.dialog.set_recording("toggle_tracking", false);
    e.dialog.flush_deferred();
    e.controller.run(&mut e.dialog);
    assert_eq!(e.controller.suspended, [true, false]); // nothing left to give back
}

#[test]
fn test_diagnostics_see_the_hotkeys_registered() {
    let mut e = env();
    e.controller.report = None;
    e.dialog.set_recording("toggle_tracking", true);
    e.dialog.set_recording("toggle_tracking", false); // clicked from the field to the page
    e.dialog.refresh_diagnostics(); // before the deferred resume has run
    let actions = e.controller.run(&mut e.dialog);
    assert_eq!(e.dialog.diagnostics_text(), "hotkeys suspended: false");
    // The report is collected by the controller's side, after the hotkeys came back.
    assert_eq!(
        actions,
        [
            SettingsAction::SuspendHotkeys(true),
            SettingsAction::SuspendHotkeys(false),
            SettingsAction::CollectDiagnostics,
        ]
    );
}

// ======================================================== settings: review fixes
#[test]
fn test_backend_choices_are_the_opencv_backends() {
    let e = env();
    let items = e.dialog.items("general.backend");
    let values: Vec<&str> = items.iter().map(|(v, _)| v.as_str()).collect();
    #[cfg(not(target_os = "macos"))]
    assert_eq!(values, ["auto", "facemesh", "lite"]);
    #[cfg(target_os = "macos")]
    assert_eq!(values, ["auto", "facemesh", "lite", "apple"]);
    assert!(items[1].1.starts_with("Face mesh"));
    assert!(items[2].1.starts_with("Lite"));
    // Available, so not marked otherwise.
    assert!(
        items
            .iter()
            .all(|(_, label)| !label.contains("not available"))
    );
    let labels: Vec<&str> = items.iter().map(|(_, l)| l.as_str()).collect();
    let labels = labels.join(" ");
    assert!(!labels.contains("MediaPipe"));
    assert!(!labels.contains("OpenCV"));
}

/// Regression (ui_app-02): a second OK must retry, not accept the failed state.
#[test]
fn test_failed_start_at_login_change_stays_pending() {
    let mut e = env();
    e.autostart.lock().fail = Some("access denied".into());
    e.dialog.set_autostart_checked(true);
    assert!(!e.apply());
    assert!(e.dialog.status_text().contains("access denied"));
    assert!(e.dialog.autostart_checked());
    assert!(e.dialog.is_modified()); // still pending: Apply stays enabled
    assert!(e.dialog.apply_enabled());
    // OK tries again, fails again, and the dialog stays open.
    assert!(!e.dialog.accept());
    e.controller.run(&mut e.dialog);
    assert_eq!(e.autostart.lock().calls, ["enable", "enable"]);
    assert!(e.controller.closed.is_empty());
    assert!(!e.autostart.lock().enabled);
    e.autostart.lock().fail = None;
    assert!(e.dialog.accept());
    e.controller.run(&mut e.dialog);
    assert_eq!(e.autostart.lock().calls, ["enable", "enable", "enable"]);
    assert!(e.autostart.lock().enabled);
    assert_eq!(e.controller.closed, [true]);
    assert!(!e.dialog.is_modified());
}

#[test]
fn test_start_at_login_explains_a_stale_login_item() {
    let mut e = env_with(
        Settings::default(),
        all_caps(),
        None,
        two_cameras(),
        |log| {
            log.idle_status = Status::Stale;
        },
    );
    assert!(e.dialog.autostart_note_text().contains("no longer exists"));
    assert!(!e.dialog.autostart_checked());
    e.dialog.set_autostart_checked(true); // re-enabling points the item at this copy
    assert!(e.apply());
    assert_eq!(e.autostart.lock().calls, ["enable"]);
    assert!(e.dialog.autostart_note_text().is_empty());
}

#[test]
fn test_reading_grace_is_editable_next_to_the_typing_grace() {
    let mut e = env();
    assert_eq!(
        e.dialog.number("switching.reading_grace_ms"),
        Settings::default().switching.reading_grace_ms as f64
    );
    e.dialog.set_number("switching.reading_grace_ms", 9000.0);
    e.dialog.set_checked("switching.enabled", false);
    // Follows "Move the cursor to the monitor I look at".
    assert!(!e.dialog.is_enabled("switching.reading_grace_ms"));
    assert!(e.apply());
    assert_eq!(e.last_applied().switching.reading_grace_ms, 9000);
}

/// Regression (ui_app-10): a hand-edited bad shortcut must not lock the dialog.
#[test]
fn test_invalid_hotkey_does_not_block_other_changes_while_hotkeys_are_off() {
    let mut settings = Settings::default();
    settings.hotkeys.enabled = false;
    settings.hotkeys.toggle_tracking = "ctrl+alt+".into();
    let mut e = env_for(settings);
    assert!(e.dialog.is_valid());
    assert!(e.dialog.ok_enabled());
    e.dialog.set_checked("general.notifications", false);
    assert!(e.apply());
    let applied = e.last_applied();
    assert!(!applied.general.notifications);
    assert_eq!(applied.hotkeys.toggle_tracking, "ctrl+alt+"); // left as it was
    // Turning shortcuts back on brings the problem up, also in the footer,
    // which is visible from every page.
    e.dialog.show_page("general");
    e.dialog.set_checked("hotkeys.enabled", true);
    assert!(!e.dialog.is_valid());
    assert!(!e.dialog.ok_enabled());
    assert!(
        e.dialog
            .status_text()
            .starts_with("Pause / resume tracking:")
    );
    e.dialog.set_hotkey("toggle_tracking", "ctrl+alt+j");
    assert!(e.dialog.is_valid());
    assert!(e.dialog.ok_enabled());
    assert_eq!(e.dialog.status_text(), "");
}

/// A hotkey manager whose `layout_conflict` flags Ctrl+Alt+T (AltGr+T).
#[derive(Default)]
struct LayoutAwareHotkeys {
    asked: Mutex<Vec<String>>,
}

impl HotkeyManager for LayoutAwareHotkeys {
    fn supported(&self) -> bool {
        true
    }
    fn name(&self) -> &'static str {
        "fake"
    }
    fn note(&self) -> Option<String> {
        None
    }
    fn register(&self, _name: &str, _hotkey: HotkeyArg<'_>, _callback: HotkeyCallback) -> bool {
        false
    }
    fn unregister(&self, _name: &str) -> bool {
        false
    }
    fn unregister_all(&self) {}
    fn registered(&self) -> HashMap<String, Hotkey> {
        HashMap::new()
    }
    fn last_error(&self, _name: &str) -> Option<String> {
        None
    }
    fn layout_conflict(&self, hotkey: HotkeyArg<'_>) -> Result<Option<String>, HotkeyError> {
        let text = match hotkey {
            HotkeyArg::Hotkey(hk) => hk.to_string(),
            HotkeyArg::Text(text) => text.to_string(),
        };
        self.asked.lock().push(text.clone());
        Ok((text == "ctrl+alt+t").then(|| {
            "Ctrl+Alt+T is AltGr+T, which types '₺' on the Turkish Q keyboard layout".into()
        }))
    }
    fn start(&self) {}
    fn stop(&self) {}
}

#[test]
fn test_shortcuts_that_type_a_character_are_flagged() {
    let manager = Arc::new(LayoutAwareHotkeys::default());
    let mut e = env_with(
        Settings::default(),
        all_caps(),
        Some(manager.clone()),
        two_cameras(),
        |_| {},
    );
    e.dialog.set_hotkey("toggle_tracking", "ctrl+alt+t");
    // A warning, not an error: the OS refuses it, but other changes can be applied.
    assert!(e.dialog.is_valid());
    let warnings = e.dialog.hotkey_warnings();
    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].1.contains("Turkish Q"));
    assert!(warnings[0].1.starts_with("Pause / resume tracking:"));
    assert!(e.dialog.hotkey_edit("toggle_tracking").unwrap().invalid());
    e.dialog.set_checked("general.notifications", false);
    let asked = manager.asked.lock().clone();
    assert_eq!(asked.iter().filter(|h| *h == "ctrl+alt+t").count(), 1); // answers are cached
    e.dialog.set_hotkey("toggle_tracking", "ctrl+alt+meta+t");
    assert!(e.dialog.hotkey_warnings().is_empty());
}

/// Regression (vision-01): closing a DirectShow probe of the camera in use kills it.
#[test]
fn test_detect_cameras_never_probes_the_camera_in_use() {
    let mut e = env();
    e.dialog.detect_cameras();
    assert!(e.wait_for_cameras());
    // The applied device, not the edited one.
    assert_eq!(*e.probes.lock(), [BTreeSet::from([0])]);
    assert!(
        e.dialog
            .camera_note()
            .unwrap()
            .contains("The camera in use is not probed.")
    );
    let mut video = e.controller.settings.clone();
    video.camera.device = "clip.mp4".into(); // not a camera index: nothing to leave out
    e.controller.apply_settings(&mut e.dialog, video);
    e.dialog.detect_cameras();
    assert!(e.wait_for_cameras());
    assert_eq!(e.probes.lock().last(), Some(&BTreeSet::new()));
}

#[test]
fn test_the_camera_in_use_stays_selectable_after_detection() {
    let usb = vec![CameraInfo {
        index: 2,
        name: "USB Cam".into(),
        width: 1280,
        height: 720,
    }];
    let mut e = env_with(Settings::default(), all_caps(), None, usb, |_| {});
    e.dialog.detect_cameras();
    assert!(e.wait_for_cameras());
    let values: Vec<String> = e
        .dialog
        .items("camera.device")
        .into_iter()
        .map(|(v, _)| v)
        .collect();
    assert_eq!(values, ["2", "0"]);
    let (value, label) = e.dialog.current("camera.device");
    assert_eq!(value, "0");
    assert!(label.contains("in use"));
    assert!(!e.dialog.is_modified());
}

// ============================================================ egui (not in Python)
/// One egui frame without a display (an egui context without a window).
pub(crate) fn run_headless(ctx: &egui::Context, mut frame: impl FnMut(&mut egui::Ui)) {
    let mut output = ctx.run_ui(egui::RawInput::default(), |ui| frame(ui));
    // Nothing renders the textures here.
    output.textures_delta.clear();
}

/// Every page draws without a display (an egui context without a window).
#[test]
fn test_every_page_draws_headless() {
    let mut e = env();
    let ctx = egui::Context::default();
    for (key, _, _) in PAGES {
        e.dialog.show_page(key);
        for _ in 0..2 {
            run_headless(&ctx, |ui| {
                let actions = e.dialog.show(ui.ctx(), &SettingsInputs::default());
                assert!(
                    !actions
                        .iter()
                        .any(|a| matches!(a, SettingsAction::Closed { .. }))
                );
            });
        }
    }
}

#[test]
fn test_rich_text_keeps_the_words_and_drops_the_markup() {
    let ctx = egui::Context::default();
    run_headless(&ctx, |ui| {
        let job = rich_text(ui, "<b>Bold</b> &amp; <code>a &lt;b&gt;</code><br>end");
        assert_eq!(job.text, "Bold & a <b>\nend");
    });
}

#[test]
fn test_settings_window_is_opaque() {
    // Left unset, glow makes it as transparent as the root: no title bar background on macOS.
    assert_eq!(SettingsDialog::viewport_builder().transparent, Some(false));
}
