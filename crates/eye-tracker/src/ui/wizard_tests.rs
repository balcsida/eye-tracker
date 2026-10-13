//! Tests for the first-run wizard (the wizard part of `tests/test_ui_dialogs.py`).
//!
//! The wizard runs against a fake platform; start at login and camera probing are fakes,
//! so nothing here touches the OS or a camera. Time is passed in, as the Python tests'
//! `FakeClock` did.
//!
//! Every wizard test of the Python file is ported. `test_wizard_flow_applies_choices` no
//! longer checks `preview_owners`: the wizard's [`WizardAction::SetPreview`] is its own.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use parking_lot::Mutex;

use super::*;
use crate::config::Settings;
use crate::platform::base::{GenericPlatform, PlatformServices};
use crate::ui::settings_dialog::tests::{
    AutostartLog, FakeAutostart, fake_probe, run_headless, two_cameras,
};

/// Records permission requests; claims every capability.
struct FakePlatform {
    calls: Mutex<Vec<(String, String)>>,
    perms: Mutex<HashMap<&'static str, Option<bool>>>,
    caps: Mutex<HashMap<&'static str, bool>>,
    open_settings: bool,
    accessibility: &'static str,
}

impl FakePlatform {
    fn new() -> Self {
        Self {
            calls: Mutex::default(),
            perms: Mutex::new(HashMap::from([
                ("camera", Some(true)),
                ("accessibility", Some(false)),
            ])),
            caps: Mutex::new(
                GenericPlatform
                    .capabilities()
                    .into_keys()
                    .map(|k| (k, true))
                    .collect(),
            ),
            open_settings: true,
            accessibility: "unknown",
        }
    }
}

impl PlatformServices for FakePlatform {
    fn capabilities(&self) -> HashMap<&'static str, bool> {
        self.caps.lock().clone()
    }
    fn permissions(&self) -> HashMap<&'static str, Option<bool>> {
        self.perms.lock().clone()
    }
    fn request_permission(&self, name: &str) {
        self.calls.lock().push(("request".into(), name.into()));
    }
    fn open_permission_settings(&self, name: &str) -> bool {
        self.calls.lock().push(("open".into(), name.into()));
        self.open_settings
    }
    fn accessibility_status(&self) -> &'static str {
        self.accessibility
    }
}

struct Env {
    wizard: FirstRunWizard,
    platform: Arc<FakePlatform>,
    autostart: Arc<Mutex<AutostartLog>>,
    probes: Arc<Mutex<Vec<BTreeSet<u32>>>>,
    /// Every action so far, in order.
    actions: Vec<WizardAction>,
    /// The controller's settings (as the actions left them).
    settings: Settings,
}

struct Setup {
    settings: Settings,
    platform: FakePlatform,
    state: TrackingState,
    monitors: usize,
    show_permissions: bool,
    fail: Option<String>,
}

impl Default for Setup {
    fn default() -> Self {
        Self {
            settings: Settings::default(),
            platform: FakePlatform::new(),
            state: TrackingState::NeedsCalibration,
            monitors: 2,
            show_permissions: false,
            fail: None,
        }
    }
}

fn wizard(setup: Setup) -> Env {
    let (backend, autostart) = FakeAutostart::create();
    autostart.lock().fail = setup.fail;
    let (probe, probes) = fake_probe(two_cameras());
    let platform = Arc::new(setup.platform);
    let wizard = FirstRunWizard::new(
        &setup.settings,
        platform.clone(),
        setup.state,
        Some(setup.monitors),
        Some(setup.show_permissions),
        backend,
    )
    .with_camera_probe(probe);
    Env {
        wizard,
        platform,
        autostart,
        probes,
        actions: Vec::new(),
        settings: setup.settings,
    }
}

impl Env {
    /// Carry out the wizard's actions as the controller would.
    fn run(&mut self) {
        for action in self.wizard.take_actions() {
            if let WizardAction::ApplySettings(settings) = &action {
                self.settings = (**settings).clone();
                self.wizard.on_settings_changed(settings);
            }
            self.actions.push(action);
        }
    }

    fn applied(&self) -> Vec<&Settings> {
        self.actions
            .iter()
            .filter_map(|a| match a {
                WizardAction::ApplySettings(s) => Some(&**s),
                _ => None,
            })
            .collect()
    }

    fn previews(&self) -> Vec<bool> {
        self.actions
            .iter()
            .filter_map(|a| match a {
                WizardAction::SetPreview(on) => Some(*on),
                _ => None,
            })
            .collect()
    }

    fn calibration_requested(&self) -> bool {
        self.actions.contains(&WizardAction::CalibrationRequested)
    }

    fn wait_for_cameras(&mut self) -> bool {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            self.wizard.poll_cameras();
            if self.wizard.detect_enabled() {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        false
    }
}

fn face(now: f64, count: u32) -> Observation {
    let mut obs = Observation::new(now, count);
    obs.quality = if count > 0 { 1.0 } else { 0.0 };
    obs
}

#[test]
fn test_wizard_flow_applies_choices() {
    let mut e = wizard(Setup::default());
    let original = e.settings.clone();
    let mut now = 1000.0;
    assert_eq!(e.wizard.current_id(), PAGE_WELCOME);

    e.wizard.next();
    e.run();
    assert_eq!(e.wizard.current_id(), PAGE_CAMERA);
    assert_eq!(e.previews(), [true]);
    assert_eq!(e.wizard.face_status(now), FaceStatus::Waiting);
    e.wizard.on_observation(&face(now, 1), now);
    assert_eq!(e.wizard.face_status(now), FaceStatus::Face);
    assert!(e.wizard.face_status(now).text().0.starts_with('✓'));
    now += 1.5;
    e.wizard.on_observation(&face(now, 0), now);
    assert_eq!(e.wizard.face_status(now), FaceStatus::NoFace);
    assert!(e.wizard.face_status(now).text().0.starts_with('✗'));
    e.wizard.on_preview_frame(&Frame::filled(64, 48, [0, 0, 0]));
    assert!(e.wizard.has_image());

    e.wizard.next(); // the permissions page is skipped off macOS
    e.run();
    assert_eq!(e.wizard.current_id(), PAGE_PRESENCE);
    assert_eq!(e.previews(), [true, false]);
    e.wizard.set_presence_choice("display_off");
    e.wizard.set_timeout(90);

    e.wizard.next();
    assert_eq!(e.wizard.current_id(), PAGE_FINISH);
    assert!(e.wizard.calibrate()); // default on
    e.wizard.set_autostart_checked(true);
    e.wizard.accept();
    e.run();

    assert!(e.actions.contains(&WizardAction::Closed { accepted: true }));
    let applied = e.applied();
    assert_eq!(applied.len(), 1);
    let mut expected = original.clone();
    expected.general.first_run_done = true;
    expected.presence.enabled = true;
    expected.presence.action = "display_off".into();
    expected.presence.away_timeout_s = 90;
    assert_eq!(*applied[0], expected);
    assert_eq!(e.autostart.lock().calls, ["enable"]);
    assert!(e.wizard.wants_calibration());
    assert!(e.calibration_requested());
}

/// r3-ux-docs-03: with one monitor a calibration changes nothing.
#[test]
fn test_wizard_recommends_no_calibration_for_one_monitor() {
    let single = wizard(Setup {
        monitors: 1,
        ..Setup::default()
    });
    let double = wizard(Setup::default());
    assert!(!single.wizard.calibrate());
    assert!(single.wizard.one_monitor());
    assert!(double.wizard.calibrate());
    assert!(!double.wizard.one_monitor());
}

/// r3-ux-docs-08: where input is not counted, moving the mouse cancels nothing.
#[test]
fn test_wizard_countdown_names_only_what_cancels_it() {
    let mut texts = Vec::new();
    for (require_input, observable) in [(true, true), (false, true), (true, false)] {
        let mut settings = Settings::default();
        settings.presence.require_input_idle = require_input;
        let platform = FakePlatform::new();
        platform.caps.lock().insert("input_idle", observable);
        let e = wizard(Setup {
            settings,
            platform,
            ..Setup::default()
        });
        texts.push(e.wizard.countdown_text().to_string());
    }
    assert!(texts[0].contains("move the mouse or look at the camera to cancel it"));
    for text in &texts[1..] {
        assert!(text.contains("look at the camera to cancel it"));
        assert!(!text.contains("mouse"));
    }
}

/// r3-ux-docs-04: on Windows 11 the icon starts in the hidden overflow area.
#[test]
fn test_wizard_finish_page_says_how_to_reach_the_tray_icon() {
    assert!(finish_subtitle(Platform::Windows).contains("click ^ on the taskbar"));
    assert!(finish_subtitle(Platform::Other).contains("right-click the eye icon"));
    assert!(finish_subtitle(Platform::MacOs).contains("menu bar"));
}

#[test]
fn test_wizard_walk_away_off_and_no_calibration() {
    let mut e = wizard(Setup::default());
    e.wizard.set_presence_choice("off");
    assert!(!e.wizard.timeout_enabled());
    e.wizard.set_calibrate(false);
    e.wizard.accept();
    e.run();
    let applied = e.applied();
    let last = applied.last().unwrap();
    assert!(!last.presence.enabled);
    assert!(last.general.first_run_done);
    assert!(e.autostart.lock().calls.is_empty()); // unchanged: nothing to do
    assert!(!e.calibration_requested());
    assert!(!e.wizard.wants_calibration());
}

#[test]
fn test_wizard_autostart_failure_is_reported_not_raised() {
    let mut e = wizard(Setup {
        fail: Some("no permission".into()),
        ..Setup::default()
    });
    e.wizard.set_autostart_checked(true);
    e.wizard.accept();
    e.run();
    assert!(e.applied().last().unwrap().general.first_run_done);
    assert!(
        e.wizard
            .autostart_error()
            .unwrap()
            .contains("no permission")
    );
    let notified: Vec<&str> = e
        .actions
        .iter()
        .filter_map(|a| match a {
            WizardAction::Notify { message, .. } => Some(message.as_str()),
            _ => None,
        })
        .collect();
    assert!(notified.last().unwrap().contains("no permission"));
}

#[test]
fn test_wizard_permissions_page_on_macos() {
    let mut e = wizard(Setup {
        show_permissions: true,
        ..Setup::default()
    });
    e.wizard.next();
    e.wizard.next();
    assert_eq!(e.wizard.current_id(), PAGE_PERMISSIONS);
    let (camera, _, camera_allow) = e.wizard.permission_status("camera");
    let (accessibility, _, accessibility_allow) = e.wizard.permission_status("accessibility");
    assert_eq!(camera, "✓ Allowed");
    assert_eq!(accessibility, "✗ Not allowed yet");
    assert!(!camera_allow);
    assert!(accessibility_allow);
    e.wizard.request_permission("accessibility");
    e.wizard.open_permission_settings("camera");
    assert_eq!(
        *e.platform.calls.lock(),
        [
            ("request".to_string(), "accessibility".to_string()),
            ("open".to_string(), "camera".to_string())
        ]
    );
    e.wizard.next();
    assert_eq!(e.wizard.current_id(), PAGE_PRESENCE);
}

#[test]
fn test_wizard_camera_choice_is_live_and_reverted_on_cancel() {
    let mut e = wizard(Setup::default());
    e.wizard.next();
    e.wizard.detect_cameras();
    assert!(e.wait_for_cameras());
    let values: Vec<&str> = e.wizard.devices().iter().map(|(_, v)| v.as_str()).collect();
    assert_eq!(values, ["0", "2"]);
    e.wizard.select_device("2");
    e.run();
    assert_eq!(e.applied().last().unwrap().camera.device, "2");
    assert_eq!(e.wizard.device(), "2");
    e.wizard.reject();
    e.run();
    assert_eq!(e.applied().last().unwrap().camera.device, "0");
    assert!(!e.settings.general.first_run_done);
    assert_eq!(e.previews().last(), Some(&false));
}

#[test]
fn test_wizard_reports_a_camera_that_is_off() {
    let mut e = wizard(Setup {
        state: TrackingState::Paused,
        ..Setup::default()
    });
    let now = 1000.0;
    e.wizard.next();
    assert_eq!(e.wizard.face_status(now), FaceStatus::Off);
    e.wizard.on_state_changed(TrackingState::CameraError);
    assert_eq!(e.wizard.face_status(now), FaceStatus::Error);
    e.wizard.on_state_changed(TrackingState::NeedsCalibration);
    e.wizard.on_observation(&face(now, 1), now);
    assert_eq!(e.wizard.face_status(now), FaceStatus::Face);
}

#[test]
fn test_wizard_frame_conversion_rejects_junk() {
    // Four bytes per pixel (a float frame in the Python test) is no BGR8 image.
    let junk = Frame {
        width: 4,
        height: 4,
        data: vec![0; 4 * 4 * 4],
    };
    assert!(frame_to_image(&junk).is_none());
    let image = frame_to_image(&Frame::filled(6, 4, [0, 0, 0])).unwrap();
    assert_eq!(image.size, [6, 4]);
}

// =========================================================== wizard: review fixes
#[test]
fn test_wizard_names_a_camera_blocked_by_privacy_settings() {
    let platform = FakePlatform::new();
    platform.perms.lock().insert("camera", Some(false));
    let mut e = wizard(Setup {
        platform,
        ..Setup::default()
    });
    let now = 1000.0;
    e.wizard.next();
    assert!(!e.wizard.camera_help_visible(now));
    e.wizard.on_state_changed(TrackingState::CameraError);
    assert_eq!(e.wizard.face_status(now), FaceStatus::Blocked);
    let text = e.wizard.face_status(now).text().0;
    assert!(text.contains("privacy settings"));
    assert!(!text.contains("try another one"));
    assert!(e.wizard.camera_help_visible(now)); // the help and its settings button
    e.wizard.open_camera_privacy_settings();
    assert_eq!(
        e.platform.calls.lock().last(),
        Some(&("open".to_string(), "camera".to_string()))
    );
    // The camera works again: the hints go away.
    e.wizard.on_state_changed(TrackingState::NeedsCalibration);
    e.wizard.on_observation(&face(now, 1), now);
    assert_eq!(e.wizard.face_status(now), FaceStatus::Face);
    assert!(!e.wizard.camera_help_visible(now));
}

#[test]
fn test_wizard_offers_the_privacy_settings_for_any_camera_error() {
    let mut platform = FakePlatform::new();
    platform.perms.lock().insert("camera", None); // the OS cannot tell
    platform.open_settings = false;
    let mut e = wizard(Setup {
        platform,
        ..Setup::default()
    });
    let now = 1000.0;
    e.wizard.next();
    e.wizard.on_state_changed(TrackingState::CameraError);
    assert_eq!(e.wizard.face_status(now), FaceStatus::Error);
    assert!(
        e.wizard
            .face_status(now)
            .text()
            .0
            .contains("another app may be using it")
    );
    assert!(e.wizard.camera_help_visible(now));
    // No settings page here (Linux): the steps stand out.
    e.wizard.open_camera_privacy_settings();
    assert!(e.wizard.help_highlighted);
    assert_eq!(
        e.wizard.camera_help_text(),
        camera_privacy_help(Platform::CURRENT)
    );
}

#[test]
fn test_wizard_detect_cameras_skips_the_camera_in_use() {
    let mut e = wizard(Setup::default());
    e.wizard.next(); // the live preview holds camera 0 now
    e.wizard.detect_cameras();
    assert!(e.wait_for_cameras());
    assert_eq!(*e.probes.lock(), [BTreeSet::from([0])]);
}

#[test]
fn test_wizard_explains_a_stale_accessibility_grant() {
    let mut platform = FakePlatform::new();
    platform.accessibility = "stale";
    let mut e = wizard(Setup {
        platform,
        show_permissions: true,
        ..Setup::default()
    });
    e.wizard.next();
    e.wizard.next();
    assert_eq!(e.wizard.current_id(), PAGE_PERMISSIONS);
    let (text, _, allow) = e.wizard.permission_status("accessibility");
    assert!(text.starts_with("✗ Granted to an earlier version"));
    assert!(allow);
}

// ============================================================ egui (not in Python)
/// Every page draws without a display (an egui context without a window).
#[test]
fn test_every_page_draws_headless() {
    let mut e = wizard(Setup {
        show_permissions: true,
        ..Setup::default()
    });
    let ctx = egui::Context::default();
    for page in 0..5 {
        if page == PAGE_CAMERA {
            e.wizard.on_preview_frame(&Frame::filled(64, 48, [0, 0, 0]));
        }
        for _ in 0..2 {
            run_headless(&ctx, |ui| {
                let actions = e.wizard.show(ui.ctx(), &WizardInputs { now: 1000.0 });
                assert!(
                    !actions
                        .iter()
                        .any(|a| matches!(a, WizardAction::Closed { .. }))
                );
            });
        }
        e.wizard.next();
    }
}

/// The page fills the window between the title and the buttons, which sit at the bottom (a
/// right-to-left layout straight in the button panel grew it over the page, frame by frame).
#[test]
fn test_wizard_buttons_sit_below_the_page() {
    let mut e = wizard(Setup {
        show_permissions: true,
        ..Setup::default()
    });
    e.wizard.on_preview_frame(&Frame::filled(64, 48, [0, 0, 0]));
    let ctx = egui::Context::default();
    let window = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(680.0, 600.0));
    for page in 0..5 {
        let mut shapes = Vec::new();
        for _ in 0..50 {
            let input = egui::RawInput {
                screen_rect: Some(window),
                ..Default::default()
            };
            let mut output = ctx.run_ui(input, |ui| e.wizard.draw(ui, 1000.0));
            output.textures_delta.clear();
            shapes = output.shapes;
        }
        // Where a text shows: the part of it its clip rect lets through.
        let visible = |needle: &str| {
            shapes.iter().find_map(|clipped| match &clipped.shape {
                egui::Shape::Text(text) if text.galley.text().contains(needle) => {
                    Some(text.visual_bounding_rect().intersect(clipped.clip_rect))
                }
                _ => None,
            })
        };
        let cancel = visible("Cancel").expect("the Cancel button is drawn");
        assert!(
            cancel.min.y > window.max.y - 50.0,
            "page {page}: {cancel:?}"
        );
        if page == PAGE_WELCOME {
            let body = visible("Runs entirely on this computer").expect("the page is drawn");
            assert!(body.is_positive(), "the page is clipped away: {body:?}");
            assert!(body.max.y < cancel.min.y);
        }
        e.wizard.next();
    }
}
