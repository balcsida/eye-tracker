//! First-run assistant: what the app does, camera check, permissions, walk-away action.
//!
//! Pages:
//!
//! ```text
//! Welcome → Camera → Permissions (macOS only) → Walk away → Finish
//! ```
//!
//! The camera choice is applied immediately so the live preview and the "face
//! detected" indicator reflect the selected device (and reverted on Cancel). All
//! other choices are applied when the user presses Finish, together with
//! `general.first_run_done = true`. If "Calibrate now" is ticked,
//! [`WizardAction::CalibrationRequested`] follows the wizard's closing.
//!
//! The state ([`FirstRunWizard`]) holds every rule; [`FirstRunWizard::show`] draws it with egui
//! and returns what the Python wizard would have called on the controller as
//! [`WizardAction`]s.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use egui::{Color32, RichText};

use super::icons;
use super::settings_dialog::{
    CameraProbe, camera_in_use, camera_label, hint_label, meta, rich_text,
};
use super::util::egui_colors::{ACCENT, DANGER, SUCCESS, WARNING};
use crate::APP_NAME;
use crate::config::{Platform, Settings};
use crate::platform::autostart::AutostartBackend;
use crate::platform::base::PlatformServices;
use crate::platform::hotkeys::{format_hotkey, parse_hotkey};
use crate::types::{Observation, TrackingState};
use crate::vision::Frame;
use crate::vision::camera::CameraInfo;

pub const PAGE_WELCOME: usize = 0;
pub const PAGE_CAMERA: usize = 1;
pub const PAGE_PERMISSIONS: usize = 2;
pub const PAGE_PRESENCE: usize = 3;
pub const PAGE_FINISH: usize = 4;

/// Walk-away choices: (value, title, description). `"off"` disables presence detection.
pub const PRESENCE_CHOICES: [(&str, &str, &str); 5] = [
    (
        "lock",
        "Lock the computer",
        "Recommended. Unlock as usual when you are back.",
    ),
    (
        "lock_and_display_off",
        "Lock and turn the displays off",
        "Also saves power while you are away.",
    ),
    (
        "display_off",
        "Turn the displays off",
        "They come back when you return. Does not lock.",
    ),
    (
        "notify",
        "Only show a notification",
        "Nothing else happens.",
    ),
    (
        "off",
        "Do nothing",
        "Walk-away detection stays off. You can turn it on later.",
    ),
];

fn capabilities_for(choice: &str) -> &'static [&'static str] {
    match choice {
        "lock" => &["lock"],
        "lock_and_display_off" => &["lock", "display_off"],
        "display_off" => &["display_off"],
        _ => &[],
    }
}

/// How long a face counts as "detected" after the last observation that had one.
const FACE_FRESH_S: f64 = 1.0;
const CAMERA_STALE_S: f64 = 2.5;

/// Where to allow camera access, by platform (Linux: every other Unix).
pub fn camera_privacy_help(platform: Platform) -> String {
    match platform {
        Platform::Windows => {
            "If Windows blocks the camera: Settings › Privacy & security › Camera, \
                              turn on “Camera access” and “Let desktop apps access your camera”."
                .into()
        }
        Platform::MacOs => format!(
            "If macOS blocks the camera: System Settings › Privacy & Security › Camera, allow \
             {APP_NAME}."
        ),
        Platform::Other => "If your account may not use the camera: add it to the “video” group \
                            (sudo usermod -aG video $USER), then log out and back in."
            .into(),
    }
}

/// An image that owns a copy of a camera frame; `None` if the frame is not a whole image.
pub fn frame_to_image(frame: &Frame) -> Option<egui::ColorImage> {
    if frame.width == 0 || frame.height == 0 || frame.data.len() != frame.width * frame.height * 3 {
        return None;
    }
    let rgb: Vec<u8> = frame
        .data
        .as_chunks::<3>()
        .0
        .iter()
        .flat_map(|bgr| [bgr[2], bgr[1], bgr[0]])
        .collect();
    Some(egui::ColorImage::from_rgb(
        [frame.width, frame.height],
        &rgb,
    ))
}

/// Where the app lives from now on, and how its icon is used there.
pub fn finish_subtitle(platform: Platform) -> String {
    match platform {
        Platform::MacOs => {
            format!("{APP_NAME} keeps running in the menu bar: click the eye icon for its menu.")
        }
        // Windows 11 puts new tray icons in the hidden overflow area.
        Platform::Windows => format!(
            "{APP_NAME} keeps running in the system tray: click the eye icon for its menu. If \
             you don't see it, click ^ on the taskbar."
        ),
        Platform::Other => format!(
            "{APP_NAME} keeps running in the system tray: right-click the eye icon for its menu, \
             double-click it for the settings."
        ),
    }
}

/// What the camera indicator shows (see [`FirstRunWizard::face_status`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceStatus {
    Face,
    NoFace,
    Waiting,
    Error,
    Blocked,
    Off,
    Busy,
}

impl FaceStatus {
    /// The indicator's text and colour.
    pub fn text(self) -> (&'static str, Option<Color32>) {
        match self {
            Self::Face => ("✓  Face detected", Some(SUCCESS)),
            Self::NoFace => (
                "✗  No face detected — sit in front of the camera",
                Some(WARNING),
            ),
            Self::Waiting => ("Starting the camera…", None),
            Self::Error => (
                "✗  The camera could not be opened — another app may be using it, or the \
                 system's privacy settings may block it",
                Some(DANGER),
            ),
            Self::Blocked => (
                "✗  Camera access is blocked in the system's privacy settings",
                Some(DANGER),
            ),
            Self::Off => (
                "The camera is off — tracking is paused or in privacy mode",
                Some(WARNING),
            ),
            Self::Busy => (
                "The camera is in use by another app — close it to continue",
                Some(WARNING),
            ),
        }
    }
}

/// What the wizard would have called on the controller or emitted.
#[derive(Debug, Clone, PartialEq)]
pub enum WizardAction {
    /// `Controller::set_preview(enabled, wizard)`: the wizard is a preview consumer of its own.
    SetPreview(bool),
    /// `Controller::apply_settings`.
    ApplySettings(Box<Settings>),
    /// `Controller.notify`: a tray notification.
    Notify { title: String, message: String },
    /// "Calibrate now" was ticked (comes after the wizard closed).
    CalibrationRequested,
    /// Finish (`accepted`) or Cancel / the window's close button.
    Closed { accepted: bool },
}

/// What the wizard reads while it is shown.
pub struct WizardInputs {
    /// Monotonic seconds (the face indicator's clock).
    pub now: f64,
}

/// First-run assistant.
pub struct FirstRunWizard {
    platform: Arc<dyn PlatformServices>,
    pub(crate) capabilities: HashMap<&'static str, bool>,
    show_permissions: bool,
    autostart: Box<dyn AutostartBackend>,
    /// The controller's current settings (kept up to date by the owner).
    settings: Settings,
    original_device: String,
    device: String,
    preview_on: bool,
    last_obs_at: Option<f64>,
    last_face_at: Option<f64>,
    tracking_state: TrackingState,
    wants_calibration: bool,
    autostart_error: Option<String>,
    current: usize,
    // Camera page.
    /// (label, value)
    devices: Vec<(String, String)>,
    probe: CameraProbe,
    detecting: bool,
    image: Option<egui::ColorImage>,
    texture: Option<egui::TextureHandle>,
    placeholder: String,
    /// The privacy help was asked for where no settings page could be opened.
    help_highlighted: bool,
    // Presence page.
    presence: &'static str,
    timeout: i64,
    countdown: String,
    // Finish page.
    calibrate: bool,
    one_monitor: bool,
    autostart_supported: bool,
    autostart_checked: bool,
    shortcuts: String,
    actions: Vec<WizardAction>,
}

impl FirstRunWizard {
    /// The assistant for the controller's `settings`.
    ///
    /// `state` is the controller's tracking state, `monitor_count` how many monitors it sees
    /// (`None` if it cannot say). `show_permissions` shows the macOS permissions page;
    /// `None` shows it on macOS only.
    pub fn new(
        settings: &Settings,
        platform: Arc<dyn PlatformServices>,
        state: TrackingState,
        monitor_count: Option<usize>,
        show_permissions: Option<bool>,
        autostart: Box<dyn AutostartBackend>,
    ) -> Self {
        let capabilities = platform.capabilities();
        let mut wizard = Self {
            platform,
            capabilities,
            show_permissions: show_permissions.unwrap_or(cfg!(target_os = "macos")),
            autostart,
            settings: settings.clone(),
            original_device: settings.camera.device.clone(),
            device: settings.camera.device.clone(),
            preview_on: false,
            last_obs_at: None,
            last_face_at: None,
            tracking_state: state,
            wants_calibration: false,
            autostart_error: None,
            current: PAGE_WELCOME,
            devices: Vec::new(),
            probe: CameraProbe::default(),
            detecting: false,
            image: None,
            texture: None,
            placeholder: "Camera preview".into(),
            help_highlighted: false,
            presence: "lock",
            timeout: 0,
            countdown: String::new(),
            calibrate: true,
            one_monitor: false,
            autostart_supported: false,
            autostart_checked: false,
            shortcuts: String::new(),
            actions: Vec::new(),
        };
        wizard.fill_devices(&[]);
        wizard.load_presence(settings);
        wizard.load_finish(settings, monitor_count);
        wizard
    }

    /// Probe cameras with `probe` instead of the real camera list (tests).
    pub fn with_camera_probe(mut self, probe: CameraProbe) -> Self {
        self.probe = probe;
        self
    }

    fn cap(&self, name: &str) -> bool {
        self.capabilities.get(name).copied().unwrap_or(true)
    }

    // ============================================================== public API
    pub fn current_id(&self) -> usize {
        self.current
    }

    /// The page after this one (the permissions page is skipped where it does not apply).
    fn next_id(&self) -> Option<usize> {
        match self.current {
            PAGE_CAMERA if !self.show_permissions => Some(PAGE_PRESENCE),
            PAGE_FINISH => None,
            page => Some(page + 1),
        }
    }

    fn back_id(&self) -> Option<usize> {
        match self.current {
            PAGE_WELCOME => None,
            PAGE_PRESENCE if !self.show_permissions => Some(PAGE_CAMERA),
            page => Some(page - 1),
        }
    }

    /// Next (does nothing on the last page).
    pub fn next(&mut self) {
        if let Some(page) = self.next_id() {
            self.on_page_changed(page);
        }
    }

    pub fn back(&mut self) {
        if let Some(page) = self.back_id() {
            self.on_page_changed(page);
        }
    }

    /// Whether the user asked to calibrate right after the wizard (set on Finish).
    pub fn wants_calibration(&self) -> bool {
        self.wants_calibration
    }

    pub fn autostart_error(&self) -> Option<&str> {
        self.autostart_error.as_deref()
    }

    /// The selected walk-away choice (a [`PRESENCE_CHOICES`] value).
    pub fn chosen_presence(&self) -> &'static str {
        self.presence
    }

    pub fn set_presence_choice(&mut self, value: &str) {
        if let Some((choice, _, _)) = PRESENCE_CHOICES.iter().find(|(v, _, _)| *v == value) {
            self.presence = choice;
        }
    }

    /// The text of a walk-away choice (marked where the system lacks what it needs).
    pub fn presence_label(&self, value: &str) -> String {
        let title = PRESENCE_CHOICES
            .iter()
            .find(|(v, _, _)| *v == value)
            .map_or("", |(_, t, _)| *t);
        if capabilities_for(value).iter().any(|c| !self.cap(c)) {
            format!("{title} (may not work on this system)")
        } else {
            title.to_string()
        }
    }

    /// The walk-away timeout in seconds.
    pub fn timeout(&self) -> i64 {
        self.timeout
    }

    pub fn set_timeout(&mut self, seconds: i64) {
        let (lo, hi) = timeout_range();
        self.timeout = seconds.clamp(lo, hi);
    }

    /// Whether the timeout can be edited (not for "Do nothing").
    pub fn timeout_enabled(&self) -> bool {
        self.presence != "off"
    }

    /// The countdown sentence of the walk-away page.
    pub fn countdown_text(&self) -> &str {
        &self.countdown
    }

    /// "Calibrate now" is ticked.
    pub fn calibrate(&self) -> bool {
        self.calibrate
    }

    pub fn set_calibrate(&mut self, checked: bool) {
        self.calibrate = checked;
    }

    /// The one-monitor explanation is shown instead of a recommendation.
    pub fn one_monitor(&self) -> bool {
        self.one_monitor
    }

    pub fn autostart_checked(&self) -> bool {
        self.autostart_checked
    }

    pub fn set_autostart_checked(&mut self, checked: bool) {
        self.autostart_checked = checked;
    }

    /// The shortcuts line of the finish page (rich text; empty: none).
    pub fn shortcuts_text(&self) -> &str {
        &self.shortcuts
    }

    /// The settings Finish applies (current controller settings plus the choices).
    pub fn final_settings(&self) -> Settings {
        let mut settings = self.settings.clone();
        settings.camera.device = self.device.clone();
        let choice = self.chosen_presence();
        if choice == "off" {
            settings.presence.enabled = false;
        } else {
            settings.presence.enabled = true;
            settings.presence.action = choice.to_string();
            settings.presence.away_timeout_s = self.timeout;
        }
        settings.general.first_run_done = true;
        settings
    }

    /// Probe for cameras in the background and refresh the device list.
    pub fn detect_cameras(&mut self) {
        // The camera behind the live preview is not probed (see camera_in_use).
        let skip: BTreeSet<u32> = camera_in_use(&self.settings);
        if self.probe.start(&self.settings.camera.api, 4, &skip) {
            self.detecting = true;
        }
    }

    /// Whether "Detect cameras" can be pressed (no detection running).
    pub fn detect_enabled(&self) -> bool {
        !self.detecting
    }

    /// Hand over a finished camera detection (called every frame).
    pub fn poll_cameras(&mut self) {
        if let Some(cameras) = self.probe.poll() {
            self.detecting = false;
            self.fill_devices(&cameras);
        }
    }

    /// (label, value) of the camera list.
    pub fn devices(&self) -> &[(String, String)] {
        &self.devices
    }

    /// The selected camera.
    pub fn device(&self) -> &str {
        &self.device
    }

    /// Switch to another camera (applied immediately).
    pub fn select_device(&mut self, device: &str) {
        let device = device.trim();
        if device.is_empty() || device == self.device {
            return;
        }
        self.device = device.to_string();
        self.ensure_device_listed();
        self.apply_device(&self.device.clone());
    }

    pub fn request_permission(&mut self, name: &str) {
        self.platform.request_permission(name);
    }

    /// Open the OS settings page for a permission. Returns whether one opened.
    pub fn open_permission_settings(&self, name: &str) -> bool {
        self.platform.open_permission_settings(name)
    }

    /// The camera page's button: the OS page, or the steps where there is none.
    pub fn open_camera_privacy_settings(&mut self) {
        if !self.open_permission_settings("camera") {
            // No settings page to open (Linux): the help text says what to do.
            self.help_highlighted = true;
        }
    }

    /// Finish.
    pub fn accept(&mut self) {
        let settings = self.final_settings();
        self.actions
            .push(WizardAction::ApplySettings(Box::new(settings)));
        self.apply_autostart(self.autostart_checked);
        self.wants_calibration = self.calibrate;
        self.stop_preview();
        self.actions.push(WizardAction::Closed { accepted: true });
        log::info!(
            "First-run setup finished (calibrate now: {})",
            self.wants_calibration
        );
        if self.wants_calibration {
            self.actions.push(WizardAction::CalibrationRequested);
        }
    }

    /// Cancel or the window's close button.
    pub fn reject(&mut self) {
        self.stop_preview();
        if self.device != self.original_device {
            // The camera choice was applied live; Cancel restores the previous one.
            self.device = self.original_device.clone();
            self.apply_device(&self.original_device.clone());
        }
        self.actions.push(WizardAction::Closed { accepted: false });
    }

    /// The actions collected since the last call.
    pub fn take_actions(&mut self) -> Vec<WizardAction> {
        std::mem::take(&mut self.actions)
    }

    // ============================================================== controller events
    /// `Controller.settings_changed`.
    pub fn on_settings_changed(&mut self, settings: &Settings) {
        self.settings = settings.clone();
    }

    /// `Controller.preview_frame`.
    pub fn on_preview_frame(&mut self, frame: &Frame) {
        if !self.preview_on {
            return;
        }
        if let Some(image) = frame_to_image(frame) {
            self.image = Some(image);
            self.texture = None;
        }
    }

    pub fn has_image(&self) -> bool {
        self.image.is_some()
    }

    /// `Controller.observation`, at monotonic time `now`.
    pub fn on_observation(&mut self, obs: &Observation, now: f64) {
        self.last_obs_at = Some(now);
        if self.tracking_state == TrackingState::CameraError {
            // Frames arrive again: the controller clears the error with its next
            // state change, but the indicator need not wait for it.
            self.tracking_state = TrackingState::Starting;
        }
        if obs.face_count > 0 {
            self.last_face_at = Some(now);
        }
    }

    /// `Controller.state_changed`.
    pub fn on_state_changed(&mut self, state: TrackingState) {
        self.tracking_state = state;
    }

    /// What the camera indicator shows at monotonic time `now`.
    ///
    /// `Face` (a face was seen within the last second), `NoFace`, `Waiting` (no frames yet),
    /// `Error` (the camera cannot be opened), `Blocked` (it cannot be opened and the OS
    /// reports that camera access is denied), `Off` (paused or privacy mode) or `Busy`
    /// (another app has the camera).
    pub fn face_status(&self, now: f64) -> FaceStatus {
        let state = self.tracking_state;
        if state == TrackingState::CameraError {
            return if self.camera_permission() == Some(false) {
                FaceStatus::Blocked
            } else {
                FaceStatus::Error
            };
        }
        if state == TrackingState::Yielded {
            return FaceStatus::Busy;
        }
        if !state.camera_active() {
            return FaceStatus::Off;
        }
        match self.last_obs_at {
            Some(at) if now - at <= CAMERA_STALE_S => {}
            _ => return FaceStatus::Waiting,
        }
        if self.last_face_at.is_some_and(|at| now - at <= FACE_FRESH_S) {
            FaceStatus::Face
        } else {
            FaceStatus::NoFace
        }
    }

    /// Whether the camera page shows the privacy help and its settings button: only when
    /// the camera cannot be opened (the OS privacy switch is the usual culprit, and not
    /// something "try another camera" would fix).
    pub fn camera_help_visible(&self, now: f64) -> bool {
        matches!(
            self.face_status(now),
            FaceStatus::Error | FaceStatus::Blocked
        )
    }

    /// The camera privacy help shown on the camera page.
    pub fn camera_help_text(&self) -> String {
        camera_privacy_help(Platform::CURRENT)
    }

    /// The OS camera permission (`None`: unknown or not applicable).
    fn camera_permission(&self) -> Option<bool> {
        self.platform.permissions().get("camera").copied().flatten()
    }

    // ============================================================== permissions
    /// The status line of a permission and whether "Allow…" can still be pressed.
    pub fn permission_status(&self, name: &str) -> (String, Option<Color32>, bool) {
        let value = self.platform.permissions().get(name).copied().flatten();
        let stale = self.platform.accessibility_status() == "stale";
        let (text, color) = match value {
            Some(true) => ("✓ Allowed", Some(SUCCESS)),
            // Granted to an earlier build: the toggle looks on but no longer applies.
            _ if name == "accessibility" && stale => (
                "✗ Granted to an earlier version — remove it with “−” and add it again",
                Some(DANGER),
            ),
            Some(false) => ("✗ Not allowed yet", Some(DANGER)),
            None => ("Not decided yet", None),
        };
        // Nothing left to ask for once granted.
        (text.to_string(), color, value != Some(true))
    }

    // ============================================================== pages
    fn load_presence(&mut self, settings: &Settings) {
        let choice = if settings.presence.enabled {
            settings.presence.action.as_str()
        } else {
            "off"
        };
        let choice = if choice == "none" { "off" } else { choice };
        self.presence = "lock";
        self.set_presence_choice(choice);
        self.set_timeout(settings.presence.away_timeout_s);
        let warning = settings.presence.warning_s;
        // Like the countdown toast's hint: input cancels the countdown only where the
        // settings count it and the system reports it (not on Wayland outside GNOME).
        let cancel = if settings.presence.require_input_idle && self.cap("input_idle") {
            "move the mouse or look at the camera"
        } else {
            "look at the camera"
        };
        self.countdown = if warning != 0 {
            format!("A {warning}-second countdown comes first: {cancel} to cancel it.")
        } else {
            "The action happens without a countdown.".into()
        };
    }

    fn load_finish(&mut self, settings: &Settings, monitor_count: Option<usize>) {
        self.autostart_supported = self.autostart.is_supported();
        self.autostart_checked = self.autostart_supported && self.autostart.is_enabled();
        // A calibration only serves switching: with one monitor it would change
        // nothing, so it is neither ticked nor recommended there.
        let single = monitor_count == Some(1);
        self.calibrate = !single && settings.switching.enabled;
        self.one_monitor = single;
        let hotkeys = &settings.hotkeys;
        let rows: Vec<String> = [
            ("Pause / resume", &hotkeys.toggle_tracking),
            ("Privacy mode", &hotkeys.toggle_privacy),
            ("Recalibrate", &hotkeys.recalibrate),
        ]
        .into_iter()
        .filter_map(|(label, text)| {
            let shown = format_hotkey(&parse_hotkey(text).ok()?, None);
            Some(format!("{label}: <b>{shown}</b>"))
        })
        .collect();
        self.shortcuts = if hotkeys.enabled && !rows.is_empty() && self.cap("hotkeys") {
            format!("Shortcuts — {}", rows.join(" · "))
        } else {
            String::new()
        };
    }

    fn on_page_changed(&mut self, page: usize) {
        self.current = page;
        if page == PAGE_CAMERA {
            self.start_preview();
        } else {
            self.stop_preview();
        }
    }

    // ============================================================== camera
    fn start_preview(&mut self) {
        if self.preview_on {
            return;
        }
        self.preview_on = true;
        // One owner among others (the preview window): see Controller::set_preview.
        self.actions.push(WizardAction::SetPreview(true));
    }

    fn stop_preview(&mut self) {
        if !self.preview_on {
            return;
        }
        self.preview_on = false;
        self.actions.push(WizardAction::SetPreview(false));
        self.clear_preview("Camera preview");
    }

    fn clear_preview(&mut self, placeholder: &str) {
        self.image = None;
        self.texture = None;
        self.placeholder = placeholder.to_string();
    }

    fn fill_devices(&mut self, found: &[CameraInfo]) {
        self.devices.clear();
        for info in found {
            let value = info.index.to_string();
            if !self.devices.iter().any(|(_, v)| *v == value) {
                self.devices.push((camera_label(info), value));
            }
        }
        self.ensure_device_listed();
    }

    fn ensure_device_listed(&mut self) {
        if !self.devices.iter().any(|(_, v)| *v == self.device) {
            let label =
                if !self.device.is_empty() && self.device.bytes().all(|b| b.is_ascii_digit()) {
                    format!("Camera {}", self.device)
                } else {
                    self.device.clone()
                };
            self.devices.push((label, self.device.clone()));
        }
    }

    fn apply_device(&mut self, device: &str) {
        if self.settings.camera.device == device {
            return;
        }
        self.settings.camera.device = device.to_string();
        self.last_obs_at = None;
        self.last_face_at = None;
        self.clear_preview("Switching camera…");
        self.actions
            .push(WizardAction::ApplySettings(Box::new(self.settings.clone())));
    }

    // ============================================================== misc
    fn apply_autostart(&mut self, wanted: bool) {
        if !self.autostart_supported {
            return; // unsupported here: nothing to change
        }
        if wanted == self.autostart.is_enabled() {
            return;
        }
        let result = if wanted {
            self.autostart.enable(true)
        } else {
            self.autostart.disable()
        };
        if let Err(err) = result {
            log::warn!("Changing start at login failed: {err}");
            self.autostart_error = Some(err.to_string());
            self.actions.push(WizardAction::Notify {
                title: "Start at login".into(),
                message: err.to_string(),
            });
        }
    }

    // ============================================================== drawing
    /// The wizard's window.
    pub fn viewport_id() -> egui::ViewportId {
        egui::ViewportId::from_hash_of("eye-tracker-setup")
    }

    /// The window: titled, opaque, with the app icon.
    pub fn viewport_builder() -> egui::ViewportBuilder {
        icons::with_app_icon(
            egui::ViewportBuilder::default()
                .with_title(format!("{APP_NAME} setup"))
                .with_inner_size([680.0, 600.0])
                .with_min_inner_size([640.0, 560.0])
                // Under glow a child viewport left unset inherits the transparent root's flag.
                .with_transparent(false),
        )
    }

    /// Draw the wizard in its own window; returns what it asks the controller to do.
    pub fn show(&mut self, ctx: &egui::Context, inputs: &WizardInputs) -> Vec<WizardAction> {
        self.poll_cameras();
        ctx.show_viewport_immediate(
            Self::viewport_id(),
            Self::viewport_builder(),
            |ui, _class| self.draw(ui, inputs.now),
        );
        if matches!(self.current, PAGE_CAMERA | PAGE_PERMISSIONS) {
            // The status is refreshed twice a second.
            ctx.request_repaint_after(std::time::Duration::from_millis(500));
        }
        self.take_actions()
    }

    fn draw(&mut self, ui: &mut egui::Ui, now: f64) {
        if ui.ctx().input(|i| i.viewport().close_requested()) {
            self.reject();
            return;
        }
        let (title, subtitle) = match self.current {
            PAGE_WELCOME => (
                format!("Welcome to {APP_NAME}"),
                "Look at a monitor and your cursor — and keyboard focus — follow.".to_string(),
            ),
            PAGE_CAMERA => (
                "Camera".into(),
                "Sit as you usually do and check that your face is detected.".into(),
            ),
            PAGE_PERMISSIONS => (
                "Permissions".into(),
                "macOS asks you to allow a few things once.".into(),
            ),
            PAGE_PRESENCE => (
                "When you walk away".into(),
                "Eye Tracker notices when you leave. What should happen?".into(),
            ),
            _ => ("You're all set".into(), finish_subtitle(Platform::CURRENT)),
        };
        egui::Panel::top("wizard-title").show(ui, |ui| {
            ui.add_space(8.0);
            ui.label(RichText::new(title).size(20.0).strong());
            ui.add(egui::Label::new(subtitle).wrap());
            ui.add_space(8.0);
        });
        egui::Panel::bottom("wizard-buttons").show(ui, |ui| {
            ui.add_space(6.0);
            // Straight in the panel, a right-to-left layout centres the buttons in all the height
            // left, and the panel grows to hold them frame by frame until it covers the page; a
            // row keeps it one line tall.
            ui.horizontal(|ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Cancel").clicked() {
                        self.reject();
                    }
                    if self.current == PAGE_FINISH {
                        if ui.button("Finish").clicked() {
                            self.accept();
                        }
                    } else if ui.button("Next >").clicked() {
                        self.next();
                    }
                    // No back button on the start page.
                    if self.back_id().is_some() && ui.button("< Back").clicked() {
                        self.back();
                    }
                });
            });
            ui.add_space(6.0);
        });
        egui::CentralPanel::default().show(ui, |ui| match self.current {
            PAGE_WELCOME => draw_welcome(ui),
            PAGE_CAMERA => self.draw_camera(ui, now),
            PAGE_PERMISSIONS => self.draw_permissions(ui),
            PAGE_PRESENCE => self.draw_presence(ui),
            _ => self.draw_finish(ui),
        });
    }

    fn draw_camera(&mut self, ui: &mut egui::Ui, now: f64) {
        ui.horizontal(|ui| {
            ui.label("Camera:");
            let selected = self
                .devices
                .iter()
                .find(|(_, v)| *v == self.device)
                .map_or_else(|| self.device.clone(), |(l, _)| l.clone());
            let mut chosen = None;
            egui::ComboBox::from_id_salt("wizard-camera")
                .selected_text(selected)
                .width(ui.available_width() - 140.0)
                .show_ui(ui, |ui| {
                    for (label, value) in &self.devices {
                        if ui.selectable_label(*value == self.device, label).clicked() {
                            chosen = Some(value.clone());
                        }
                    }
                });
            if let Some(value) = chosen {
                self.select_device(&value);
            }
            let label = if self.detecting {
                "Detecting…"
            } else {
                "Detect cameras"
            };
            if ui
                .add_enabled(!self.detecting, egui::Button::new(label))
                .on_hover_text("Look for connected cameras (each may briefly turn on).")
                .clicked()
            {
                self.detect_cameras();
            }
        });
        self.draw_preview(ui);
        let status = self.face_status(now);
        let (text, color) = status.text();
        let mut text = RichText::new(text).size(15.0).strong();
        if let Some(color) = color {
            text = text.color(color);
        }
        ui.vertical_centered(|ui| {
            ui.add(egui::Label::new(text).wrap());
            if self.camera_help_visible(now) {
                let help = RichText::new(self.camera_help_text());
                let help = if self.help_highlighted {
                    help.color(WARNING)
                } else {
                    help.weak()
                };
                ui.add(egui::Label::new(help).wrap());
                if ui.button("Open camera privacy settings").clicked() {
                    self.open_camera_privacy_settings();
                }
            }
        });
        ui.add(hint_label(
            "The preview is shown only here and is never saved. The camera light may stay on \
             while Eye Tracker is running; privacy mode turns it off.",
        ));
    }

    /// The camera preview with rounded corners; frames are kept in memory only.
    fn draw_preview(&mut self, ui: &mut egui::Ui) {
        let size = egui::vec2(
            ui.available_width(),
            (ui.available_height() - 110.0).max(200.0),
        );
        let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let radius = 12.0;
        painter.rect_filled(rect, radius, Color32::from_rgb(12, 14, 22));
        if let Some(image) = &self.image {
            let texture = self.texture.get_or_insert_with(|| {
                ui.ctx()
                    .load_texture("wizard-preview", image.clone(), Default::default())
            });
            let image_size = texture.size_vec2();
            let scale = (rect.width() / image_size.x).min(rect.height() / image_size.y);
            let target = egui::Rect::from_center_size(rect.center(), image_size * scale);
            egui::Image::new((texture.id(), target.size()))
                .corner_radius(radius)
                .paint_at(ui, target);
        } else {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                &self.placeholder,
                egui::FontId::proportional(14.0),
                Color32::from_rgb(156, 163, 175),
            );
        }
        painter.rect_stroke(
            rect,
            radius,
            egui::Stroke::new(1.0, Color32::from_white_alpha(30)),
            egui::StrokeKind::Inside,
        );
    }

    fn draw_permissions(&mut self, ui: &mut egui::Ui) {
        let rows = [
            ("camera", "Camera", "Needed to see where you are looking."),
            (
                "accessibility",
                "Accessibility",
                "Lets Eye Tracker give keyboard focus to the window on the screen you look at. \
                 Without it only the cursor moves.",
            ),
        ];
        for (name, title, text) in rows {
            let (status, color, allow) = self.permission_status(name);
            ui.horizontal(|ui| {
                ui.label(RichText::new(title).strong());
                let status = RichText::new(status);
                ui.label(match color {
                    Some(color) => status.color(color),
                    None => status,
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Open Settings").clicked() {
                        self.open_permission_settings(name);
                    }
                    if ui.add_enabled(allow, egui::Button::new("Allow…")).clicked() {
                        self.request_permission(name);
                    }
                });
            });
            ui.add(hint_label(text));
            ui.add_space(10.0);
        }
        ui.add(hint_label(
            "You can change these later in System Settings → Privacy & Security.",
        ));
    }

    fn draw_presence(&mut self, ui: &mut egui::Ui) {
        for (value, _, description) in PRESENCE_CHOICES {
            let label = self.presence_label(value);
            if ui.radio(self.presence == value, label).clicked() {
                self.set_presence_choice(value);
            }
            ui.indent(value, |ui| ui.add(hint_label(description)));
        }
        ui.add_space(10.0);
        let row = meta("presence.away_timeout_s");
        ui.horizontal(|ui| {
            ui.label("After");
            let (lo, hi) = timeout_range();
            let mut timeout = self.timeout;
            let response = ui
                .add_enabled(
                    self.timeout_enabled(),
                    egui::DragValue::new(&mut timeout)
                        .range(lo..=hi)
                        .speed(0.5)
                        .suffix(" seconds"),
                )
                .on_hover_text(row.map_or("", |r| r.doc));
            if response.changed() {
                self.set_timeout(timeout);
            }
            ui.label("without a face or any input");
        });
        ui.add(hint_label(&self.countdown));
    }

    fn draw_finish(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(
            &mut self.calibrate,
            "Calibrate now (recommended, about 30 seconds)",
        )
        .on_hover_text(
            "Switching needs a short calibration: you look at a few dots on each screen.",
        );
        // Shown instead of a recommendation when there is nothing to switch between.
        if self.one_monitor {
            ui.add(hint_label(
                "With one monitor there is nothing to switch between, so no calibration is \
                 needed: walk-away detection, privacy mode and the shoulder guard work as they \
                 are. Connect a second monitor and calibrate from the tray menu.",
            ));
        }
        ui.add_enabled(
            self.autostart_supported,
            egui::Checkbox::new(
                &mut self.autostart_checked,
                format!("Start {APP_NAME} when I log in"),
            ),
        );
        ui.add_space(10.0);
        if !self.shortcuts.is_empty() {
            let job = rich_text(ui, &self.shortcuts);
            ui.add(egui::Label::new(job).wrap());
        }
        ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
            ui.add(hint_label(
                "Everything can be changed later under Settings.",
            ));
        });
    }
}

/// The walk-away timeout's range in seconds (from the settings metadata).
fn timeout_range() -> (i64, i64) {
    let row = meta("presence.away_timeout_s");
    (
        row.and_then(|r| r.lo).map_or(5, |v| v as i64),
        row.and_then(|r| r.hi).map_or(3600, |v| v as i64),
    )
}

fn draw_welcome(ui: &mut egui::Ui) {
    ui.spacing_mut().item_spacing.y = 12.0;
    draw_illustration(ui);
    ui.add(
        egui::Label::new(format!(
            "{APP_NAME} uses your webcam to notice which screen you are looking at and moves \
             the mouse cursor there, so you can read on one monitor and start typing on the \
             other without reaching for the mouse. It can also lock the computer when you walk \
             away."
        ))
        .wrap(),
    );
    let accent = ACCENT;
    egui::Frame::new()
        .fill(accent.linear_multiply(0.08))
        .stroke(egui::Stroke::new(1.0, accent.linear_multiply(0.30)))
        .corner_radius(10.0)
        .inner_margin(egui::Margin::symmetric(16, 12))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.spacing_mut().item_spacing.y = 6.0;
            for text in [
                "Runs entirely on this computer — no account, no internet connection.",
                "Camera frames are analysed in memory and never saved or sent anywhere.",
                "Privacy mode switches the camera off completely, from the tray or a hotkey.",
            ] {
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new("✓").color(SUCCESS));
                    ui.label(text);
                });
            }
        });
}

/// Two monitors and a cursor hopping to the one being looked at.
fn draw_illustration(ui: &mut egui::Ui) {
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), 120.0),
        egui::Sense::hover(),
    );
    let painter = ui.painter_at(rect);
    let text = ui.visuals().text_color();
    let accent = ACCENT;
    let (mon_w, mon_h, gap) = (150.0, 88.0, 22.0);
    let x0 = rect.left() + (rect.width() - 2.0 * mon_w - gap) / 2.0;
    let y0 = rect.top() + 6.0;
    for n in 0..2 {
        let monitor = egui::Rect::from_min_size(
            egui::pos2(x0 + n as f32 * (mon_w + gap), y0),
            egui::vec2(mon_w, mon_h),
        );
        let fill = if n == 0 {
            text.gamma_multiply(0.05)
        } else {
            accent.gamma_multiply(0.12)
        };
        painter.rect(
            monitor,
            8.0,
            fill,
            egui::Stroke::new(2.0, text.gamma_multiply(0.55)),
            egui::StrokeKind::Inside,
        );
        let stand = egui::Rect::from_min_size(
            egui::pos2(monitor.center().x - 16.0, monitor.bottom() + 5.0),
            egui::vec2(32.0, 4.0),
        );
        painter.rect_filled(stand, 2.0, text.gamma_multiply(0.35));
    }
    // Gaze arc from the left monitor to the right one.
    let start = egui::pos2(x0 + mon_w * 0.55, y0 + mon_h * 0.45);
    let end = egui::pos2(x0 + mon_w + gap + mon_w * 0.45, y0 + mon_h * 0.45);
    let control = egui::pos2((start.x + end.x) / 2.0, y0 - 2.0);
    let points: Vec<egui::Pos2> = (0..=32)
        .map(|i| {
            let t = i as f32 / 32.0;
            let u = 1.0 - t;
            egui::pos2(
                u * u * start.x + 2.0 * u * t * control.x + t * t * end.x,
                u * u * start.y + 2.0 * u * t * control.y + t * t * end.y,
            )
        })
        .collect();
    painter.extend(egui::Shape::dashed_line(
        &points,
        egui::Stroke::new(3.0, accent),
        8.0,
        6.0,
    ));
    // Cursor arrow at the destination.
    let (cx, cy) = (end.x + 4.0, end.y + 4.0);
    let arrow: Vec<egui::Pos2> = [
        (0.0, 0.0),
        (0.0, 22.0),
        (6.0, 16.5),
        (10.5, 25.0),
        (14.0, 23.5),
        (9.5, 15.0),
        (17.0, 15.0),
    ]
    .iter()
    .map(|(dx, dy)| egui::pos2(cx + dx, cy + dy))
    .collect();
    // The arrow is concave: fill its head and its tail as two convex parts.
    let head = vec![arrow[0], arrow[1], arrow[6]];
    let tail = vec![arrow[2], arrow[3], arrow[4], arrow[5]];
    for part in [head, tail] {
        painter.add(egui::Shape::convex_polygon(
            part,
            Color32::WHITE,
            egui::Stroke::NONE,
        ));
    }
    painter.add(egui::Shape::closed_line(
        arrow,
        egui::Stroke::new(1.2, Color32::from_rgb(20, 20, 30)),
    ));
}

#[cfg(test)]
#[path = "wizard_tests.rs"]
mod tests;
