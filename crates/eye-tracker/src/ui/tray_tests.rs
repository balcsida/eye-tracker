//! Tests for the tray (the `test_tray_*`, `test_autostart_status_helper_*` and
//! `test_taskbar_theme_*` tests of tests/test_ui_basic.py).
//!
//! They run against the model and a fake backend, driven by a fake controller that applies the
//! returned actions the way the app does, so nothing here needs a display or a real tray.
//!
//! Not ported:
//!
//! * `test_tray_images_render_for_every_state`, `test_tray_template_icons_are_monochrome_masks`
//!   and `test_tray_glyph_colour_follows_taskbar_theme`: they test the icon drawing of
//!   `ui/icons.rs`, which has its own tests.
//! * The `menuRole == NoRole` assertion of `test_tray_menu_layout`: `muda` has no menu roles, so
//!   there is nothing that moves "About" or "Quit" out of the tray menu.
//! * `test_tray_uses_the_platform_autostart_module`'s patching of the real
//!   `platform::autostart` functions: Rust cannot patch a module, and the real functions write
//!   login items. The notification rules it covers run against a fake backend, and
//!   `PlatformAutostart` is only checked to implement `AutostartBackend`.
//! * Anything that needs a real tray (`tray.show()` on a desktop, a popped-up menu, a shown
//!   notification, the `tray-icon`/`ksni`/`notify-rust` bindings): the native backends are
//!   exercised on a desktop only. `test_tray_notify_without_tray_is_quiet` runs against a fake
//!   without a tray area instead of Qt's offscreen platform.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use serde_json::json;

use super::*;
use crate::config::Settings;
use crate::platform::hotkeys::{format_hotkey, parse_hotkey};

use TrackingState as S;

const ALL_STATES: [TrackingState; 10] = [
    S::Starting,
    S::NeedsCalibration,
    S::Calibrating,
    S::Tracking,
    S::Paused,
    S::Privacy,
    S::Away,
    S::Locked,
    S::Yielded,
    S::CameraError,
];

// --------------------------------------------------------------------------------- fakes
/// What a [`FakeBackend`] was told, shared with the test.
#[derive(Default)]
struct BackendLog {
    available: bool,
    visible: bool,
    icons: Vec<TrayImage>,
    tooltip: String,
    menu: Option<TrayModel>,
    popups: usize,
    shown: Vec<(String, String, bool)>,
}

struct FakeBackend(Rc<RefCell<BackendLog>>);

impl TrayBackend for FakeBackend {
    fn available(&self) -> bool {
        self.0.borrow().available
    }
    fn is_visible(&self) -> bool {
        self.0.borrow().visible
    }
    fn show(&mut self) {
        self.0.borrow_mut().visible = true;
    }
    fn hide(&mut self) {
        self.0.borrow_mut().visible = false;
    }
    fn set_icon(&mut self, image: &TrayImage) {
        self.0.borrow_mut().icons.push(image.clone());
    }
    fn set_tooltip(&mut self, tooltip: &str) {
        self.0.borrow_mut().tooltip = tooltip.to_string();
    }
    fn set_menu(&mut self, model: &TrayModel) {
        self.0.borrow_mut().menu = Some(model.clone());
    }
    fn popup_menu(&mut self) {
        self.0.borrow_mut().popups += 1;
    }
    fn show_message(&mut self, title: &str, message: &str, critical: bool) {
        self.0
            .borrow_mut()
            .shown
            .push((title.to_string(), message.to_string(), critical));
    }
    fn poll_events(&mut self) -> Vec<BackendEvent> {
        Vec::new()
    }
}

/// What the tray did to a [`FakeAutostart`].
#[derive(Default)]
struct AutostartLog {
    supported: bool,
    enabled: bool,
    /// An `AutostartError` with this message, when set.
    fail: Option<String>,
    /// Any other error, when set.
    explode: bool,
    background: Option<bool>,
    calls: Vec<(&'static str, Option<bool>)>,
}

struct FakeAutostart(Rc<RefCell<AutostartLog>>);

fn fake_autostart() -> (FakeAutostart, Rc<RefCell<AutostartLog>>) {
    let log = Rc::new(RefCell::new(AutostartLog {
        supported: true,
        ..Default::default()
    }));
    (FakeAutostart(log.clone()), log)
}

impl AutostartBackend for FakeAutostart {
    fn is_supported(&self) -> bool {
        self.0.borrow().supported
    }
    fn is_enabled(&self) -> bool {
        self.0.borrow().enabled
    }
    fn enable(&self, background: bool) -> Result<(), Box<dyn Error>> {
        let mut log = self.0.borrow_mut();
        log.calls.push(("enable", Some(background)));
        if let Some(message) = &log.fail {
            return Err(Box::new(platform_autostart::AutostartError(
                message.clone(),
            )));
        }
        if log.explode {
            return Err("unexpected".into());
        }
        log.enabled = true;
        log.background = Some(background);
        Ok(())
    }
    fn disable(&self) -> Result<(), Box<dyn Error>> {
        let mut log = self.0.borrow_mut();
        log.calls.push(("disable", None));
        log.enabled = false;
        Ok(())
    }
}

/// An autostart backend that also reports `status()` (like the real module).
struct StatusAutostart(Rc<Cell<AutostartState>>);

impl AutostartBackend for StatusAutostart {
    fn is_supported(&self) -> bool {
        true
    }
    fn is_enabled(&self) -> bool {
        self.0.get() == AutostartState::Enabled
    }
    fn status(&self) -> Option<AutostartState> {
        Some(self.0.get())
    }
    fn enable(&self, _background: bool) -> Result<(), Box<dyn Error>> {
        self.0.set(AutostartState::Enabled);
        Ok(())
    }
    fn disable(&self) -> Result<(), Box<dyn Error>> {
        self.0.set(AutostartState::Disabled);
        Ok(())
    }
}

/// The Controller's state and the methods the UI calls, applying the tray's actions.
///
/// The state is derived from the `paused`/`privacy` flags with the real controller's priority
/// (privacy > paused).
struct FakeController {
    settings: Settings,
    state: TrackingState,
    paused: bool,
    privacy: bool,
    yield_reason: String,
    calibration_reason: String,
    hotkey_manager: Option<HashMap<String, Hotkey>>,
    stats: TrayStats,
    hotkeys_as_text: bool,
    /// `apply_settings` and `toggle_pause` raise.
    failing: bool,
    calls: Vec<(&'static str, Option<bool>)>,
    applied_settings: Vec<Settings>,
}

impl FakeController {
    fn new() -> FakeController {
        let mut settings = Settings::default();
        // As after the setup assistant: before it, a walk-away lock only notifies.
        settings.general.first_run_done = true;
        FakeController {
            settings,
            state: S::Tracking,
            paused: false,
            privacy: false,
            yield_reason: String::new(),
            calibration_reason: String::new(),
            hotkey_manager: None,
            stats: TrayStats::default(),
            hotkeys_as_text: HOTKEY_AS_TEXT,
            failing: false,
            calls: Vec::new(),
            applied_settings: Vec::new(),
        }
    }

    /// Test helper: jump to `state` (flags follow).
    fn set_state(&mut self, state: TrackingState) {
        self.paused = state == S::Paused;
        self.privacy = state == S::Privacy;
        self.state = state;
    }

    fn derive(&mut self) {
        self.state = if self.privacy {
            S::Privacy
        } else if self.paused {
            S::Paused
        } else {
            S::Tracking
        };
    }

    fn inputs(&self) -> TrayInputs {
        TrayInputs {
            state: self.state,
            paused: self.paused,
            privacy: self.privacy,
            calibration_reason: self.calibration_reason.clone(),
            yield_reason: self.yield_reason.clone(),
            settings: self.settings.clone(),
            registered_hotkeys: self.hotkey_manager.clone(),
            stats: self.stats.clone(),
            hotkeys_as_text: self.hotkeys_as_text,
            ..TrayInputs::default()
        }
    }

    /// Run the actions that call the controller; the others (windows, quit) are returned.
    fn apply(&mut self, actions: Vec<TrayAction>) -> Vec<TrayAction> {
        let mut windows = Vec::new();
        for action in actions {
            match action {
                TrayAction::TogglePause => {
                    self.calls.push(("toggle_pause", None));
                    if !self.failing {
                        self.paused = !self.paused;
                        self.derive();
                    }
                }
                TrayAction::SetPrivacy(enabled) => {
                    self.calls.push(("set_privacy", Some(enabled)));
                    self.privacy = enabled;
                    self.derive();
                }
                TrayAction::SetGazeDot(enabled) => {
                    let mut updated = self.settings.clone();
                    updated.ui.show_gaze_overlay = enabled;
                    self.apply_settings(updated);
                }
                TrayAction::SetSplitPanes(enabled) => {
                    let mut updated = self.settings.clone();
                    updated.panes.enabled = enabled;
                    self.apply_settings(updated);
                }
                other => windows.push(other),
            }
        }
        windows
    }

    fn apply_settings(&mut self, settings: Settings) {
        self.calls.push(("apply_settings", None));
        if self.failing {
            return;
        }
        self.applied_settings.push(settings.clone());
        self.settings = settings;
    }
}

/// A tray, its fake backend and a fake controller feeding it.
struct Harness {
    controller: FakeController,
    tray: Tray,
    backend: Rc<RefCell<BackendLog>>,
    clock: Rc<Cell<f64>>,
}

impl Harness {
    fn new(controller: FakeController) -> Harness {
        let (autostart, _log) = fake_autostart();
        Harness::with_autostart(controller, Box::new(autostart))
    }

    fn with_autostart(controller: FakeController, autostart: Box<dyn AutostartBackend>) -> Harness {
        let backend = Rc::new(RefCell::new(BackendLog::default()));
        let clock = Rc::new(Cell::new(100.0));
        let clock_for_tray = clock.clone();
        let tray = Tray::new(
            Box::new(FakeBackend(backend.clone())),
            Box::new(|_state, _dark| (2, 2, vec![255; 2 * 2 * 4])),
            autostart,
            Box::new(move || clock_for_tray.get()),
            controller.inputs(),
        );
        Harness {
            controller,
            tray,
            backend,
            clock,
        }
    }

    /// A tray area that shows messages (what `_capture_messages` patched in).
    fn with_tray_area(mut self) -> Harness {
        {
            let mut backend = self.backend.borrow_mut();
            backend.available = true;
            backend.visible = true;
        }
        self.tray.show();
        self
    }

    fn sync(&mut self) {
        self.tray.sync(self.controller.inputs());
    }

    fn set_state(&mut self, state: TrackingState) {
        self.controller.set_state(state);
        self.sync();
    }

    /// Click a menu item and let the controller answer, like the app's loop.
    fn click(&mut self, id: ItemId) -> Vec<TrayAction> {
        let actions = self.tray.activate(id);
        let windows = self.controller.apply(actions);
        self.sync();
        windows
    }

    fn icon_click(&mut self, click: IconClick) -> Vec<TrayAction> {
        let actions = self.tray.icon_click(click);
        let windows = self.controller.apply(actions);
        self.sync();
        windows
    }

    /// The menu as the user sees it after the controller registered its hotkeys.
    fn about_to_show(&mut self) {
        self.sync();
        self.tray.about_to_show();
    }

    fn item(&self, id: ItemId) -> MenuItem {
        self.tray.model().item(id).cloned().expect("menu item")
    }

    fn label(&self, id: ItemId) -> String {
        self.item(id).label().to_string()
    }

    fn shown(&self) -> Vec<(String, String, bool)> {
        self.backend.borrow().shown.clone()
    }
}

fn harness() -> Harness {
    Harness::new(FakeController::new())
}

fn quiet_settings(settings: &Settings) -> Settings {
    let mut quiet = settings.clone();
    quiet.general.notifications = false;
    quiet
}

// ------------------------------------------------------------------------------- the menu
#[test]
fn test_tray_menu_reflects_state() {
    for state in ALL_STATES {
        let mut h = harness();
        h.set_state(state);
        assert_eq!(h.tray.state(), state);
        let expected_pause = if state == S::Paused {
            "Resume tracking"
        } else {
            "Pause tracking"
        };
        assert_eq!(h.label(ItemId::Pause), expected_pause);
        assert!(h.item(ItemId::Privacy).checkable);
        assert_eq!(h.item(ItemId::Privacy).checked, state == S::Privacy);
        assert!(h.item(ItemId::Status).text.starts_with(state.label()));
        assert!(!h.item(ItemId::Status).enabled);
        assert!(
            h.backend
                .borrow()
                .icons
                .last()
                .is_some_and(|icon| !icon.rgba.is_empty())
        );
        assert!(
            h.backend
                .borrow()
                .tooltip
                .starts_with(&format!("Eye Tracker — {}", state.label()))
        );
        let needs = state == S::NeedsCalibration;
        assert_eq!(h.item(ItemId::Calibrate).bold, needs);
        assert_eq!(
            h.label(ItemId::Calibrate),
            if needs {
                "Calibrate now…"
            } else {
                "Calibrate…"
            }
        );
        let busy = state == S::Calibrating;
        assert_eq!(h.item(ItemId::Pause).enabled, !busy);
        assert_eq!(h.item(ItemId::Privacy).enabled, !busy);
        assert_eq!(h.item(ItemId::Calibrate).enabled, !busy);
    }
}

#[test]
fn test_tray_menu_layout() {
    let h = harness();
    let items: Vec<&MenuItem> = h
        .tray
        .model()
        .entries
        .iter()
        .filter_map(|entry| match entry {
            MenuEntry::Item(item) => Some(item),
            MenuEntry::Separator => None,
        })
        .collect();
    let labels: Vec<&str> = items.iter().map(|item| item.label()).collect();
    assert_eq!(
        labels[1..],
        [
            "Pause tracking",
            "Privacy mode",
            "Calibrate…",
            "Show gaze dot",
            "Follow split panes",
            "Camera preview…",
            "Settings…",
            "Start at login",
            "Check for updates…",
            "About Eye Tracker",
            "Quit Eye Tracker",
        ]
    );
    let checkable: Vec<&str> = items
        .iter()
        .filter(|item| item.checkable)
        .map(|item| item.label())
        .collect();
    assert_eq!(
        checkable,
        [
            "Privacy mode",
            "Show gaze dot",
            "Follow split panes",
            "Start at login"
        ]
    );
    // Hidden like the Python's QAction.setVisible(update_supported()).
    assert_eq!(h.item(ItemId::Update).visible, update_supported());
    let separators = h
        .tray
        .model()
        .entries
        .iter()
        .filter(|entry| matches!(entry, MenuEntry::Separator))
        .count();
    assert_eq!(separators, 4);
}

#[test]
fn test_tray_pause_text_follows_the_paused_flag() {
    // Paused *and* in privacy mode: the state shows PRIVACY, but leaving privacy
    // mode would still leave tracking paused, so the item must offer "Resume".
    let mut h = harness();
    h.click(ItemId::Pause);
    assert_eq!(h.controller.state, S::Paused);
    h.click(ItemId::Privacy);
    assert_eq!(h.controller.state, S::Privacy);
    assert!(h.item(ItemId::Privacy).checked);
    assert_eq!(h.label(ItemId::Pause), "Resume tracking");
    h.click(ItemId::Pause); // resumes while privacy mode stays on
    assert_eq!(h.controller.state, S::Privacy); // no state change, but the flag changed
    assert_eq!(h.label(ItemId::Pause), "Pause tracking");
    h.click(ItemId::Privacy);
    assert_eq!(h.controller.state, S::Tracking);
    assert!(!h.item(ItemId::Privacy).checked);
}

#[test]
fn test_tray_tooltip_format() {
    let mut h = harness();
    h.controller.stats = TrayStats::from_json(&json!({
        "fps": 4.2, "cpu_percent": 0.64, "state": "tracking"
    }));
    h.sync();
    assert_eq!(
        h.tray.tooltip_text(),
        "Eye Tracker — Tracking · 4 fps · CPU 0.6 %"
    );
    assert_eq!(h.backend.borrow().tooltip, h.tray.tooltip_text());
    h.set_state(S::Paused); // camera off: no frame rate
    assert_eq!(
        h.backend.borrow().tooltip,
        "Eye Tracker — Paused · CPU 0.6 %"
    );
    h.set_state(S::CameraError);
    h.controller.stats = TrayStats::from_json(&json!({
        "last_error": "Camera 0 could not be opened"
    }));
    h.sync();
    assert_eq!(
        h.backend.borrow().tooltip,
        "Eye Tracker — Camera unavailable"
    );
    assert_eq!(
        h.tray.status_text(),
        "Camera unavailable — Camera 0 could not be opened"
    );
    // Garbage is ignored: NaN (JSON null), a string, and something that is no mapping at all.
    h.controller.stats = TrayStats::from_json(&json!({
        "fps": f64::NAN, "cpu_percent": "x"
    }));
    assert_eq!(h.controller.stats, TrayStats::default());
    h.controller.stats = TrayStats::from_json(&json!("not a mapping"));
    assert_eq!(h.controller.stats, TrayStats::default());
    h.set_state(S::Tracking);
    assert_eq!(h.tray.tooltip_text(), "Eye Tracker — Tracking");
    h.controller.stats = TrayStats::from_json(&json!({"fps": 0.5, "cpu_percent": true}));
    h.sync();
    assert_eq!(h.tray.tooltip_text(), "Eye Tracker — Tracking · 0.5 fps");
    assert!(h.tray.tooltip_text().chars().count() <= 127);
}

#[test]
fn test_tray_status_line_explains_yield_and_calibration() {
    let mut h = harness();
    h.controller.yield_reason = "zoom.exe is running".to_string();
    h.set_state(S::Yielded);
    assert_eq!(h.item(ItemId::Status).text, "Paused — zoom.exe is running");
    assert!(
        h.backend
            .borrow()
            .tooltip
            .starts_with("Eye Tracker — Camera in use by another app")
    );
    h.controller.yield_reason = String::new();
    h.set_state(S::Tracking);
    h.set_state(S::Yielded);
    assert_eq!(h.item(ItemId::Status).text, S::Yielded.label());

    h.controller.calibration_reason = "the monitor layout changed".to_string();
    h.set_state(S::NeedsCalibration);
    assert!(
        h.item(ItemId::Calibrate)
            .tooltip
            .contains("the monitor layout changed")
    );
}

#[test]
fn test_tray_actions_drive_controller() {
    let mut h = harness();
    let mut emitted: Vec<TrayAction> = Vec::new();

    h.click(ItemId::Privacy);
    assert!(h.controller.calls.contains(&("set_privacy", Some(true))));
    assert!(h.item(ItemId::Privacy).checked);
    h.click(ItemId::Privacy);
    assert!(h.controller.calls.contains(&("set_privacy", Some(false))));
    assert!(!h.item(ItemId::Privacy).checked);

    h.click(ItemId::Pause);
    assert_eq!(h.label(ItemId::Pause), "Resume tracking");
    h.click(ItemId::Pause);
    assert_eq!(h.label(ItemId::Pause), "Pause tracking");
    h.icon_click(IconClick::Middle);
    assert!(h.controller.paused);

    h.click(ItemId::GazeDot);
    assert!(!h.controller.applied_settings.is_empty());
    assert!(
        h.controller
            .applied_settings
            .last()
            .is_some_and(|settings| settings.ui.show_gaze_overlay)
    );
    assert!(h.controller.settings.ui.show_gaze_overlay);
    assert!(h.item(ItemId::GazeDot).checked);
    h.click(ItemId::GazeDot);
    assert!(!h.controller.settings.ui.show_gaze_overlay);
    assert!(!h.item(ItemId::GazeDot).checked);

    for id in [
        ItemId::Settings,
        ItemId::Calibrate,
        ItemId::Preview,
        ItemId::About,
        ItemId::Quit,
    ] {
        emitted.extend(h.click(id));
    }
    emitted.extend(h.icon_click(IconClick::Double));
    emitted.extend(h.icon_click(IconClick::Single)); // single click: nothing
    assert_eq!(
        emitted,
        [
            TrayAction::OpenSettings,
            TrayAction::OpenCalibration,
            TrayAction::OpenPreview,
            TrayAction::OpenAbout,
            TrayAction::Quit,
            TrayAction::OpenSettings,
        ]
    );
}

#[test]
fn test_tray_update_entry_opens_the_update_flow() {
    let mut h = harness();
    assert_eq!(h.label(ItemId::Update), "Check for updates…");
    h.tray.set_update_available(Some("1.2.3"));
    assert_eq!(h.label(ItemId::Update), "Update to 1.2.3…");
    // The entry is hidden (and so not clickable) where updates are not supported.
    let expected = if update_supported() {
        vec![TrayAction::OpenUpdate]
    } else {
        Vec::new()
    };
    assert_eq!(h.click(ItemId::Update), expected);
    h.tray.set_update_available(None);
    assert_eq!(h.label(ItemId::Update), "Check for updates…");
}

#[test]
fn test_tray_single_click_opens_the_menu_on_windows() {
    // r3-ux-docs-04: a left click on the Windows tray icon did nothing at all.
    let mut h = harness();
    h.tray.set_menu_on_click(true);
    let mut settings: Vec<TrayAction> = Vec::new();
    h.icon_click(IconClick::Single);
    assert!(h.tray.click_pending()); // opens once it is no double-click
    // Not yet: the double-click window is still open.
    h.tray.tick(h.clock.get() + 0.1);
    assert_eq!(h.backend.borrow().popups, 0);
    // A double-click opens the settings instead, and its second click is no click.
    settings.extend(h.icon_click(IconClick::Double));
    assert!(!h.tray.click_pending());
    assert_eq!(settings, [TrayAction::OpenSettings]);
    h.icon_click(IconClick::Single);
    assert!(!h.tray.click_pending());
    h.clock.set(h.clock.get() + 1.0);
    h.icon_click(IconClick::Single);
    assert!(h.tray.click_pending());
    h.tray.tick(h.clock.get() + 1.0); // what the timer runs
    assert!(!h.tray.click_pending());
    assert_eq!(h.backend.borrow().popups, 1);
    // Elsewhere the desktop decides what a click does.
    h.tray.set_menu_on_click(false);
    h.clock.set(h.clock.get() + 1.0);
    h.icon_click(IconClick::Single);
    assert!(!h.tray.click_pending());
}

#[test]
fn test_tray_tooltips_say_what_pause_and_privacy_also_do() {
    // r3-ux-docs-02 / -05: pausing stops the walk-away lock; privacy mode is kept.
    let mut h = harness();
    assert!(h.item(ItemId::Pause).tooltip.contains("walk-away lock"));
    assert!(
        h.item(ItemId::Privacy)
            .tooltip
            .contains("stays on after a restart")
    );
    h.controller.settings.privacy.remember_privacy_mode = false;
    h.sync();
    assert!(!h.item(ItemId::Privacy).tooltip.contains("restart"));
}

#[test]
fn test_tray_gaze_dot_keeps_other_settings() {
    let mut h = harness();
    // Changed behind the tray's back (e.g. through IPC): must not be undone.
    h.controller.settings.presence.action = "display_off".to_string();
    h.click(ItemId::GazeDot);
    assert!(h.controller.settings.ui.show_gaze_overlay);
    assert_eq!(h.controller.settings.presence.action, "display_off");
}

#[test]
fn test_tray_follow_split_panes_toggles_the_setting() {
    let mut h = harness();
    assert!(!h.item(ItemId::SplitPanes).checked); // experimental: off by default
    assert!(h.item(ItemId::SplitPanes).tooltip.contains("Experimental"));
    h.controller.settings.presence.action = "display_off".to_string(); // changed elsewhere: kept
    h.click(ItemId::SplitPanes);
    assert!(h.controller.settings.panes.enabled);
    assert_eq!(h.controller.settings.presence.action, "display_off");
    assert!(h.item(ItemId::SplitPanes).checked);
    h.click(ItemId::SplitPanes);
    assert!(!h.controller.settings.panes.enabled);
    assert!(!h.item(ItemId::SplitPanes).checked);
    // Turned on in the settings dialog: the check mark follows.
    h.controller.settings.panes.enabled = true;
    h.sync();
    assert!(h.item(ItemId::SplitPanes).checked);
}

#[test]
fn test_tray_survives_failing_controller() {
    let mut controller = FakeController::new();
    controller.failing = true;
    let mut h = Harness::new(controller);
    h.click(ItemId::GazeDot);
    // Reverted: the setting did not change (and the native menu's own flip is undone).
    assert!(!h.item(ItemId::GazeDot).checked);
    assert_eq!(
        h.backend
            .borrow()
            .menu
            .as_ref()
            .map(|menu| menu.item(ItemId::GazeDot).map(|i| i.checked)),
        Some(Some(false))
    );
    h.click(ItemId::Pause); // logged, not raised
    assert_eq!(h.label(ItemId::Pause), "Pause tracking");
}

// ------------------------------------------------------------------------------ autostart
#[test]
fn test_tray_autostart_toggle() {
    let (backend, log) = fake_autostart();
    let mut h = Harness::with_autostart(FakeController::new(), Box::new(backend));
    assert!(h.item(ItemId::Autostart).visible);
    assert!(!h.item(ItemId::Autostart).checked);
    h.click(ItemId::Autostart);
    assert!(log.borrow().enabled);
    assert_eq!(log.borrow().background, Some(true)); // a login start runs in the background
    assert!(h.item(ItemId::Autostart).checked);
    h.click(ItemId::Autostart);
    assert!(!log.borrow().enabled);
    assert!(!h.item(ItemId::Autostart).checked);
    log.borrow_mut().enabled = true; // changed elsewhere, e.g. in Task Manager
    h.tray.about_to_show();
    assert!(h.item(ItemId::Autostart).checked);
}

#[test]
fn test_tray_autostart_failure_is_reported() {
    let (backend, log) = fake_autostart();
    log.borrow_mut().fail = Some("Could not enable start at login: access denied".to_string());
    let mut h = Harness::with_autostart(FakeController::new(), Box::new(backend)).with_tray_area();
    h.click(ItemId::Autostart);
    assert!(!h.item(ItemId::Autostart).checked);
    assert_eq!(
        h.shown(),
        [(
            "Start at login".to_string(),
            "Could not enable start at login: access denied".to_string(),
            true
        )]
    );
}

#[test]
fn test_tray_uses_the_platform_autostart_module() {
    // The real module writes login items, so it is only checked to fit the tray; the rules
    // for its errors run against a fake that fails like it.
    let _platform: Box<dyn AutostartBackend> = Box::new(PlatformAutostart);
    let (backend, log) = fake_autostart();
    log.borrow_mut().fail =
        Some("Could not enable start at login: registry is read-only".to_string());
    let mut h = Harness::with_autostart(FakeController::new(), Box::new(backend)).with_tray_area();
    // Errors from a user action are shown even with notifications off and muted.
    h.controller.settings = quiet_settings(&h.controller.settings);
    h.sync();
    h.tray.mute_notifications(60.0);

    h.click(ItemId::Autostart);
    assert_eq!(log.borrow().calls, [("enable", Some(true))]);
    assert!(!h.item(ItemId::Autostart).checked);
    assert_eq!(
        h.shown(),
        [(
            "Start at login".to_string(),
            "Could not enable start at login: registry is read-only".to_string(),
            true
        )]
    );

    log.borrow_mut().fail = None;
    log.borrow_mut().enabled = true;
    h.tray.about_to_show();
    assert!(h.item(ItemId::Autostart).checked);
    h.click(ItemId::Autostart);
    assert_eq!(log.borrow().calls.last(), Some(&("disable", None)));
    assert!(!h.item(ItemId::Autostart).checked);

    log.borrow_mut().explode = true;
    h.click(ItemId::Autostart);
    assert_eq!(
        h.shown().last().map(|shown| shown.1.clone()),
        Some("The login item could not be changed; see the log for details.".to_string())
    );
}

#[test]
fn test_tray_hides_autostart_when_unsupported() {
    let (backend, log) = fake_autostart();
    log.borrow_mut().supported = false;
    let h = Harness::with_autostart(FakeController::new(), Box::new(backend));
    assert!(!h.item(ItemId::Autostart).visible);
    assert!(
        h.tray
            .model()
            .visible_items()
            .all(|item| item.id != ItemId::Autostart)
    );
}

#[test]
fn test_tray_explains_a_login_item_that_does_not_start_this_copy() {
    let state = Rc::new(Cell::new(AutostartState::Stale)); // e.g. the app was moved
    let mut h = Harness::with_autostart(
        FakeController::new(),
        Box::new(StatusAutostart(state.clone())),
    );
    assert!(!h.item(ItemId::Autostart).checked);
    assert_eq!(
        h.item(ItemId::Autostart).text,
        "Start at login (needs repair)"
    );
    assert!(
        h.item(ItemId::Autostart)
            .tooltip
            .contains("no longer exists")
    );
    h.click(ItemId::Autostart); // re-enabling points the login item at this copy
    assert_eq!(state.get(), AutostartState::Enabled);
    assert!(h.item(ItemId::Autostart).checked);
    assert_eq!(h.item(ItemId::Autostart).text, "Start at login");

    state.set(AutostartState::OtherProfile);
    h.tray.about_to_show();
    assert!(!h.item(ItemId::Autostart).checked);
    assert_eq!(
        h.item(ItemId::Autostart).text,
        "Start at login (another profile)"
    );
    assert!(h.item(ItemId::Autostart).tooltip.contains("--config-dir"));
}

#[test]
fn test_autostart_status_helper_tolerates_old_backends() {
    let (plain, _log) = fake_autostart();
    assert_eq!(autostart_status(&plain), ""); // no status(): unknown

    /// A backend whose status cannot be read (Python: `status()` raised).
    struct Broken;
    impl AutostartBackend for Broken {
        fn is_supported(&self) -> bool {
            true
        }
        fn is_enabled(&self) -> bool {
            false
        }
        fn status(&self) -> Option<AutostartState> {
            None
        }
        fn enable(&self, _background: bool) -> Result<(), Box<dyn Error>> {
            Ok(())
        }
        fn disable(&self) -> Result<(), Box<dyn Error>> {
            Ok(())
        }
    }
    assert_eq!(autostart_status(&Broken), "");

    let real = StatusAutostart(Rc::new(Cell::new(AutostartState::Stale)));
    assert_eq!(autostart_status(&real), "stale");
}

#[test]
fn test_taskbar_theme_is_only_read_from_the_registry_on_windows() {
    // Regression (packaging-02): other systems stay away from the registry.
    assert_eq!(
        windows_personalize_value_on(false, "SystemUsesLightTheme"),
        None
    );
    let _dark: bool = system_tray_is_dark();
}

// ---------------------------------------------------------------------------- notifications
#[test]
fn test_tray_notifications_respect_settings() {
    let mut h = harness().with_tray_area();

    h.tray.notify_from_controller("Switched", "hello");
    assert!(h.tray.notify("t", "own message", false, false));
    h.tray.mute_notifications(5.0);
    assert!(!h.tray.notify("t", "muted", false, false));
    // A login start stays quiet.
    assert!(
        !h.tray
            .notify_from_controller("t", "muted controller message")
    );
    assert!(h.tray.notify("t", "forced while muted", true, true));
    h.clock.set(h.clock.get() + 6.0);
    assert!(h.tray.notify("t", "unmuted", false, false));

    h.controller.settings = quiet_settings(&h.controller.settings);
    h.sync();
    // The app's own informational message.
    assert!(!h.tray.notify("t", "suppressed", false, false));
    assert!(h.tray.notify("t", "forced", true, false));
    let shown = h.shown();
    let messages: Vec<&str> = shown.iter().map(|(_t, m, _c)| m.as_str()).collect();
    assert_eq!(
        messages,
        [
            "hello",
            "own message",
            "forced while muted",
            "unmuted",
            "forced"
        ]
    );
    assert!(shown[2].2); // critical → warning icon
}

#[test]
fn test_tray_does_not_filter_controller_notifications_twice() {
    // With notifications off the controller only sends what the user asked for
    // (walk-away "notify", shoulder guard "notify") and failures: always shown.
    let mut controller = FakeController::new();
    controller.settings = quiet_settings(&controller.settings);
    let mut h = Harness::new(controller).with_tray_area();
    h.tray.notify_from_controller(
        "Could not lock the screen",
        "This system does not allow it.",
    );
    assert_eq!(
        h.shown(),
        [(
            "Could not lock the screen".to_string(),
            "This system does not allow it.".to_string(),
            false
        )]
    );
}

#[test]
fn test_tray_notify_without_tray_is_quiet() {
    let mut h = harness();
    // No system tray (the backend says so): nothing is shown, nothing raises.
    h.tray.show();
    assert!(!h.tray.notify("title", "message", false, false));
    h.tray.hide();
    assert!(h.shown().is_empty());
}

#[test]
fn test_tray_notify_needs_a_tray_area_even_when_the_icon_is_shown() {
    // Desktops without a tray area drop messages silently.
    let mut h = harness();
    h.tray.show();
    assert!(h.backend.borrow().visible);
    assert!(!h.tray.available());
    // Not reported as shown: an unseen calibration hint must not count as given.
    assert!(!h.tray.notify("lost", "nobody sees this", false, false));
    assert!(!h.tray.notify("lost", "nor this", true, false));
    h.backend.borrow_mut().available = true;
    assert!(h.tray.notify("shown", "message", false, false));
    let titles: Vec<String> = h.shown().into_iter().map(|(title, _m, _c)| title).collect();
    assert_eq!(titles, ["shown"]);
    h.tray.hide();
}

#[test]
fn test_tray_dispose_is_final_and_idempotent() {
    let mut h = harness().with_tray_area();
    h.tray.dispose();
    h.tray.dispose();
    // Late inputs are ignored, not crashing.
    h.controller.set_state(S::Paused);
    h.controller.stats = TrayStats::from_json(&json!({"fps": 3.0}));
    h.controller.settings = Settings::default();
    h.sync();
    assert!(!h.tray.notify_from_controller("late", "message"));
    h.tray.show();
    assert!(!h.backend.borrow().visible);
    assert!(!h.tray.notify("t", "m", true, false));
    assert_eq!(h.tray.state(), S::Tracking);
    assert!(h.tray.activate(ItemId::Quit).is_empty());
    assert!(h.tray.icon_click(IconClick::Middle).is_empty());
}

// --------------------------------------------------------------------------------- hotkeys
fn controller_without_text_hotkeys() -> FakeController {
    let mut controller = FakeController::new();
    controller.hotkeys_as_text = false;
    controller
}

#[cfg(not(target_os = "macos"))] // macOS shows ⌃⌥ glyphs instead
#[test]
fn test_tray_shows_hotkeys_from_settings() {
    // The defaults differ per OS (Win key on Windows, Shift on Linux); set them.
    let mut controller = controller_without_text_hotkeys();
    controller.settings.hotkeys.toggle_tracking = "ctrl+alt+t".to_string();
    controller.settings.hotkeys.toggle_privacy = "ctrl+alt+p".to_string();
    controller.settings.hotkeys.recalibrate = "ctrl+alt+c".to_string();
    let mut h = Harness::new(controller);
    let shortcut = |h: &Harness, id| h.item(id).shortcut.as_ref().map(|hk| hk.to_string());
    assert_eq!(shortcut(&h, ItemId::Pause).as_deref(), Some("ctrl+alt+t"));
    assert_eq!(shortcut(&h, ItemId::Privacy).as_deref(), Some("ctrl+alt+p"));
    assert_eq!(
        shortcut(&h, ItemId::Calibrate).as_deref(),
        Some("ctrl+alt+c")
    );
    assert_eq!(h.item(ItemId::Pause).text, "Pause tracking");
    assert!(h.item(ItemId::Pause).tooltip.ends_with("· Ctrl+Alt+T"));
    h.controller.settings.hotkeys.enabled = false;
    h.sync();
    assert!(h.item(ItemId::Pause).shortcut.is_none());
    assert!(!h.item(ItemId::Pause).tooltip.contains("Ctrl"));
}

#[test]
fn test_tray_shows_only_registered_hotkeys() {
    let mut h = Harness::new(controller_without_text_hotkeys());
    // After controller.start(): only what the OS accepted is shown. Here the
    // privacy combination is owned by another app and recalibrate was remapped.
    h.controller.hotkey_manager = Some(HashMap::from([
        (
            "toggle_tracking".to_string(),
            parse_hotkey("ctrl+alt+t").expect("hotkey"),
        ),
        (
            "recalibrate".to_string(),
            parse_hotkey("ctrl+shift+k").expect("hotkey"),
        ),
    ]));
    h.about_to_show();
    let expected = parse_hotkey("ctrl+shift+k").expect("hotkey");
    assert_eq!(h.item(ItemId::Calibrate).shortcut, Some(expected.clone()));
    assert!(h.item(ItemId::Privacy).shortcut.is_none());
    let label = format_hotkey(&expected, None);
    assert!(
        h.item(ItemId::Calibrate)
            .tooltip
            .ends_with(&format!("· {label}"))
    );

    h.controller.hotkey_manager = Some(HashMap::new()); // e.g. unsupported on Wayland
    h.about_to_show();
    assert!(h.item(ItemId::Pause).shortcut.is_none());
    assert!(h.item(ItemId::Calibrate).shortcut.is_none());
}

#[test]
fn test_tray_names_hotkeys_like_the_tooltip_on_windows() {
    // Regression (r2-hotkeys-04): Qt drew the Windows default as "Meta+Ctrl+Alt+T".
    let mut controller = FakeController::new();
    controller.hotkeys_as_text = true;
    controller.settings.hotkeys.toggle_tracking = "ctrl+alt+meta+t".to_string();
    controller.settings.hotkeys.toggle_privacy = String::new();
    controller.settings.hotkeys.recalibrate = "ctrl+shift+alt+c".to_string();
    controller.set_state(S::NeedsCalibration);
    let mut h = Harness::new(controller);
    let pause_label = format_hotkey(&parse_hotkey("ctrl+alt+meta+t").expect("hotkey"), None);
    let calibrate_label = format_hotkey(&parse_hotkey("ctrl+shift+alt+c").expect("hotkey"), None);
    #[cfg(windows)]
    assert_eq!(pause_label, "Ctrl+Alt+Win+T");
    // QMenu draws the text after the tab in the shortcut column.
    assert_eq!(
        h.item(ItemId::Pause).text,
        format!("Pause tracking\t{pause_label}")
    );
    assert_eq!(
        h.item(ItemId::Calibrate).text,
        format!("Calibrate now…\t{calibrate_label}")
    );
    assert_eq!(h.item(ItemId::Privacy).text, "Privacy mode");
    // A native menu would render "Meta+Ctrl+Alt+T".
    assert!(h.item(ItemId::Pause).shortcut.is_none());
    assert!(
        h.item(ItemId::Pause)
            .tooltip
            .ends_with(&format!("· {pause_label}"))
    );
    // The hotkey survives the text changes of the actions...
    h.controller.paused = true;
    h.set_state(S::Paused);
    assert_eq!(
        h.item(ItemId::Pause).text,
        format!("Resume tracking\t{pause_label}")
    );
    assert_eq!(
        h.item(ItemId::Calibrate).text,
        format!("Calibrate…\t{calibrate_label}")
    );
    // ...and goes away with the hotkey (another app owns it).
    h.controller.hotkey_manager = Some(HashMap::from([(
        "recalibrate".to_string(),
        parse_hotkey("ctrl+shift+alt+c").expect("hotkey"),
    )]));
    h.about_to_show();
    assert_eq!(h.item(ItemId::Pause).text, "Resume tracking");
    assert_eq!(
        h.item(ItemId::Calibrate).text,
        format!("Calibrate…\t{calibrate_label}")
    );
    let texts: Vec<String> = h
        .tray
        .model()
        .visible_items()
        .map(|i| i.text.clone())
        .collect();
    assert!(!texts.join(" ").contains("Meta"));
}
