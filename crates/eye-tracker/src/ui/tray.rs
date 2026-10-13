//! System-tray icon, menu, tooltip and notifications.
//!
//! The tray is the app's main surface. It mirrors the controller's state (icon,
//! tooltip, menu texts and check marks) and turns menu clicks into controller calls
//! or into actions the application wires to windows (settings, calibration,
//! preview, about, quit). It holds no logic of its own.
//!
//! The Qt widget is split three ways, so that the logic runs without a display:
//!
//! * [`TrayInputs`] are the plain values the Python widget read from the controller and the
//!   settings; [`TrayModel::build`] turns them into the menu (texts, check marks, tooltips,
//!   hotkeys), the tooltip and the status line. Pure.
//! * [`Tray`] keeps what the Python widget kept between calls (the mute window, the click
//!   timer, the cached icon, the autostart state), pushes the model to a [`TrayBackend`] and
//!   turns clicks into [`TrayAction`]s for the application to run. It never holds the
//!   controller.
//! * The backend draws it: `tray-icon` with its `muda` menu on Windows and macOS, `ksni`
//!   (StatusNotifierItem) on Linux. Notifications go through `notify-rust`.
//!
//! The application feeds fresh inputs with [`Tray::sync`] whenever the controller's state, flags,
//! stats, settings or registered hotkeys change, calls [`Tray::poll`] every frame (the OS
//! backends deliver clicks through channels) and runs the returned actions.

use std::collections::HashMap;

use serde_json::Value;

use super::util::{
    ACCENT, ACCENT_LIGHT, DANGER, SUCCESS, WARNING, autostart_status, elide, format_cpu,
    format_fps, parse_hotkey_text, system_tray_is_dark,
};
use crate::cli::cli_command_text;
use crate::config::Settings;
use crate::panes::worker::monotonic;
use crate::platform::autostart::{self as platform_autostart, AutostartBackend, SystemAutostart};
use crate::platform::hotkeys::{Hotkey, format_hotkey};
use crate::types::TrackingState;
use crate::update::fetch::supported as update_supported;
use crate::{APP_NAME, APP_SLUG};

/// Windows truncates tray tooltips at 127 characters.
const TOOLTIP_LIMIT: usize = 127;
/// Longest detail (camera error, yield reason) appended to the menu's status line.
const STATUS_DETAIL_LIMIT: usize = 60;

const MUTED_GREY: &str = "#9CA3AF";

/// Controller hotkey actions (`settings.hotkeys` field names) per menu item.
const HOTKEY_ACTIONS: [&str; 3] = ["toggle_tracking", "toggle_privacy", "recalibrate"];

/// Show hotkeys as menu text after a tab instead of as the action's shortcut.
///
/// Windows has no native tray menu with shortcut columns we control, and renders a shortcut
/// with its own names and order: "Meta+Ctrl+Alt+T", although Windows keyboards have no Meta key
/// and the tooltip ([`format_hotkey`]) says "Ctrl+Alt+Win+T". Text after a tab is drawn in the
/// shortcut column, so the result looks the same. Elsewhere the shortcut stays: macOS renders it
/// with the right glyphs, and the DBusMenu of Linux trays sends it as key names ("Super") the
/// shell displays.
pub const HOTKEY_AS_TEXT: bool = cfg!(windows);

/// Open the menu on a plain (left) click too.
///
/// Windows shows a tray icon's menu only on a right-click, while users click tray apps with the
/// left button and then see nothing happen. macOS shows the menu on any click itself, and Linux
/// tray hosts differ (some already show it on a left click, a second menu must not appear
/// there), so only Windows needs it.
pub const MENU_ON_CLICK: bool = cfg!(windows);

/// Seconds a single click waits for a possible second one (the system's double-click interval).
const DOUBLE_CLICK_INTERVAL: f64 = 0.5;

/// How long a notification stays on screen.
pub const NOTIFICATION_MS: u32 = 6000;

const PAUSE_TIP: &str =
    "Stop moving the cursor; the camera is released and the walk-away lock pauses too";
const RESUME_TIP: &str = "Start following your gaze again";
const PRIVACY_TIP: &str = "Release the camera completely (its light turns off)";
/// Appended while privacy mode is remembered across restarts (the default).
const PRIVACY_REMEMBERED_TIP: &str = "; it stays on after a restart";
const CALIBRATE_TIP: &str = "Look at a few dots so the tracker learns your monitors";
const GAZE_DOT_TIP: &str = "Draw a dot where the tracker thinks you are looking";
const PANES_TIP: &str = "Experimental: keyboard focus also follows your gaze between large split panes \
of tmux, WezTerm and Windows Terminal";
const PREVIEW_TIP: &str = "See what the camera sees (never saved)";

/// The colour of the status line's dot for a tracking state.
pub fn status_color(state: TrackingState) -> &'static str {
    match state {
        TrackingState::Starting => ACCENT,
        TrackingState::NeedsCalibration => WARNING,
        TrackingState::Calibrating => ACCENT,
        TrackingState::Tracking => SUCCESS,
        TrackingState::Paused => MUTED_GREY,
        TrackingState::Privacy => ACCENT_LIGHT,
        TrackingState::Away => MUTED_GREY,
        TrackingState::Locked => MUTED_GREY,
        TrackingState::Yielded => MUTED_GREY,
        TrackingState::CameraError => DANGER,
    }
}

/// Menu text and tooltip of "Start at login" by [`autostart_status`] value.
fn autostart_texts(status: &str) -> (String, String) {
    match status {
        "stale" => (
            "Start at login (needs repair)".to_string(),
            format!(
                "The login item starts a copy of {APP_NAME} that no longer exists or will be \
                 gone after a restart. Click to point it at this copy."
            ),
        ),
        "other-profile" => (
            "Start at login (another profile)".to_string(),
            format!(
                "The login item starts {APP_NAME} with another settings folder (--config-dir). \
                 Click to start this one instead."
            ),
        ),
        _ => (
            "Start at login".to_string(),
            format!("Start {APP_NAME} quietly in the tray when you log in"),
        ),
    }
}

/// The login item as the menu shows it.
#[derive(Debug, Clone, PartialEq)]
pub struct AutostartInputs {
    pub supported: bool,
    pub enabled: bool,
    /// [`autostart_status`] value; `""` while enabled, unsupported or unknown.
    pub status: String,
}

impl Default for AutostartInputs {
    fn default() -> Self {
        Self {
            supported: true,
            enabled: false,
            status: String::new(),
        }
    }
}

// -------------------------------------------------------------------------------- inputs
/// The tracker's stats dict as far as the tray reads it (`fps`, `cpu_percent`, `last_error`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrayStats {
    pub fps: Option<f64>,
    pub cpu_percent: Option<f64>,
    pub last_error: Option<String>,
}

impl TrayStats {
    /// Take the controller's stats mapping; garbage (non-numbers, booleans, a value that is not
    /// a mapping at all) is ignored.
    pub fn from_json(stats: &Value) -> TrayStats {
        let number = |key: &str| match stats.get(key) {
            Some(Value::Number(n)) => n.as_f64().filter(|x| x.is_finite()),
            _ => None,
        };
        TrayStats {
            fps: number("fps"),
            cpu_percent: number("cpu_percent"),
            last_error: match stats.get("last_error") {
                Some(Value::String(error)) => Some(error.clone()),
                _ => None,
            },
        }
    }
}

/// The values the tray reads from the controller and the settings.
#[derive(Debug, Clone)]
pub struct TrayInputs {
    pub state: TrackingState,
    /// The controller's `paused` flag. The state shows only the strongest cause: a paused
    /// tracker in privacy mode reports `Privacy`, yet "Resume" is right.
    pub paused: bool,
    /// The controller's `privacy` flag.
    pub privacy: bool,
    /// Why switching is off while the state is `NeedsCalibration`.
    pub calibration_reason: String,
    /// Which app holds the camera while the state is `Yielded`.
    pub yield_reason: String,
    pub settings: Settings,
    /// The hotkeys the OS accepted, once the controller has a hotkey manager (none on Wayland,
    /// none that another app owns); `None` before that (the settings are shown then).
    pub registered_hotkeys: Option<HashMap<String, Hotkey>>,
    pub stats: TrayStats,
    /// Filled in by the [`Tray`] from its autostart backend; [`Tray::sync`] ignores this field.
    pub autostart: AutostartInputs,
    /// A newer version, named in the menu (`None`: "Check for updates…").
    pub update_version: Option<String>,
    /// Whether the "Check for updates…" entry exists at all (Windows only for now).
    pub update_supported: bool,
    /// See [`HOTKEY_AS_TEXT`].
    pub hotkeys_as_text: bool,
}

impl TrayInputs {
    /// Inputs for `state` with the flags a controller without them would derive.
    pub fn new(state: TrackingState, settings: Settings) -> TrayInputs {
        TrayInputs {
            state,
            paused: state == TrackingState::Paused,
            privacy: state == TrackingState::Privacy,
            calibration_reason: String::new(),
            yield_reason: String::new(),
            settings,
            registered_hotkeys: None,
            stats: TrayStats::default(),
            autostart: AutostartInputs::default(),
            update_version: None,
            update_supported: update_supported(),
            hotkeys_as_text: HOTKEY_AS_TEXT,
        }
    }

    /// Global hotkeys by controller action (`toggle_tracking`, …) to show.
    ///
    /// Once the controller has a hotkey manager, only the combinations the OS accepted are
    /// shown. Before that (the manager is created by `controller.start()`), the settings.
    pub fn hotkeys(&self) -> HashMap<String, Hotkey> {
        if let Some(registered) = &self.registered_hotkeys {
            return registered.clone();
        }
        let settings = &self.settings.hotkeys;
        if !settings.enabled {
            return HashMap::new();
        }
        let mut hotkeys = HashMap::new();
        for name in HOTKEY_ACTIONS {
            let text = match name {
                "toggle_tracking" => &settings.toggle_tracking,
                "toggle_privacy" => &settings.toggle_privacy,
                _ => &settings.recalibrate,
            };
            if let Some(hotkey) = parse_hotkey_text(Some(text)) {
                hotkeys.insert(name.to_string(), hotkey);
            }
        }
        hotkeys
    }

    /// E.g. `"Eye Tracker — Tracking · 4 fps · CPU 0.6 %"`.
    pub fn tooltip_text(&self) -> String {
        let mut parts = vec![format!("{APP_NAME} — {}", self.state.label())];
        if let Some(fps) = self.stats.fps
            && self.state.camera_active()
            && self.state != TrackingState::CameraError
        {
            parts.push(format_fps(fps));
        }
        if let Some(cpu) = self.stats.cpu_percent {
            parts.push(format_cpu(cpu));
        }
        elide(&parts.join(" · "), TOOLTIP_LIMIT)
    }

    /// The disabled first menu line, e.g. `"Tracking"` or the camera error.
    pub fn status_text(&self) -> String {
        let label = self.state.label();
        match self.state {
            TrackingState::CameraError => {
                if let Some(error) = &self.stats.last_error
                    && !error.trim().is_empty()
                {
                    return format!("{label} — {}", elide(error, STATUS_DETAIL_LIMIT));
                }
            }
            // "zoom.exe is running" / "Another app is using the camera".
            TrackingState::Yielded if !self.yield_reason.trim().is_empty() => {
                return format!(
                    "Paused — {}",
                    elide(&self.yield_reason, STATUS_DETAIL_LIMIT)
                );
            }
            _ => {}
        }
        label.to_string()
    }
}

impl Default for TrayInputs {
    fn default() -> Self {
        TrayInputs::new(TrackingState::Starting, Settings::default())
    }
}

// --------------------------------------------------------------------------------- model
/// The menu items, in menu order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ItemId {
    Status,
    Pause,
    Privacy,
    Calibrate,
    GazeDot,
    SplitPanes,
    Preview,
    Settings,
    Autostart,
    Update,
    About,
    Quit,
}

impl ItemId {
    const ALL: [ItemId; 12] = [
        ItemId::Status,
        ItemId::Pause,
        ItemId::Privacy,
        ItemId::Calibrate,
        ItemId::GazeDot,
        ItemId::SplitPanes,
        ItemId::Preview,
        ItemId::Settings,
        ItemId::Autostart,
        ItemId::Update,
        ItemId::About,
        ItemId::Quit,
    ];

    /// The id the OS menus carry for this item.
    pub fn key(self) -> &'static str {
        match self {
            ItemId::Status => "status",
            ItemId::Pause => "pause",
            ItemId::Privacy => "privacy",
            ItemId::Calibrate => "calibrate",
            ItemId::GazeDot => "gaze_dot",
            ItemId::SplitPanes => "split_panes",
            ItemId::Preview => "preview",
            ItemId::Settings => "settings",
            ItemId::Autostart => "autostart",
            ItemId::Update => "update",
            ItemId::About => "about",
            ItemId::Quit => "quit",
        }
    }

    pub fn from_key(key: &str) -> Option<ItemId> {
        ItemId::ALL.into_iter().find(|id| id.key() == key)
    }
}

/// One menu item.
#[derive(Debug, Clone, PartialEq)]
pub struct MenuItem {
    pub id: ItemId,
    /// With the hotkey after a tab where [`HOTKEY_AS_TEXT`] applies.
    pub text: String,
    pub enabled: bool,
    pub visible: bool,
    pub checkable: bool,
    pub checked: bool,
    pub bold: bool,
    pub tooltip: String,
    /// The hotkey as the tooltip names it (`"Ctrl+Alt+T"`, `"⌃⌥T"`).
    pub hotkey_label: Option<String>,
    /// The hotkey the OS menu displays next to the item (never set where it is menu text).
    pub shortcut: Option<Hotkey>,
}

impl MenuItem {
    fn new(id: ItemId, text: impl Into<String>) -> MenuItem {
        MenuItem {
            id,
            text: text.into(),
            enabled: true,
            visible: true,
            checkable: false,
            checked: false,
            bold: false,
            tooltip: String::new(),
            hotkey_label: None,
            shortcut: None,
        }
    }

    /// The text without the hotkey shown after a tab (Windows).
    pub fn label(&self) -> &str {
        self.text.split('\t').next().unwrap_or("")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MenuEntry {
    Item(MenuItem),
    Separator,
}

/// The tray's menu, tooltip and status line for one set of inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct TrayModel {
    pub entries: Vec<MenuEntry>,
    pub tooltip: String,
    /// The colour of the dot before the status line.
    pub status_color: &'static str,
}

impl TrayModel {
    pub fn build(inputs: &TrayInputs) -> TrayModel {
        let state = inputs.state;
        let busy = state == TrackingState::Calibrating;
        let needs = state == TrackingState::NeedsCalibration;
        let settings = &inputs.settings;

        let hotkeys = inputs.hotkeys();
        let labels: HashMap<&str, String> = HOTKEY_ACTIONS
            .iter()
            .filter_map(|name| {
                hotkeys
                    .get(*name)
                    .map(|hotkey| (*name, format_hotkey(hotkey, None)))
            })
            .collect();
        // A hotkey action: its text followed by the hotkey on Windows, the hotkey as the
        // shortcut elsewhere, and the hotkey named in the tooltip.
        let hotkey_item = |id: ItemId, name: &str, text: &str, tip: String| {
            let mut item = MenuItem::new(id, text);
            let label = labels.get(name);
            if let Some(label) = label {
                if inputs.hotkeys_as_text {
                    item.text = format!("{text}\t{label}");
                } else {
                    item.shortcut = hotkeys.get(name).cloned();
                }
            }
            // format_hotkey gives the platform's wording ("Ctrl+Alt+T", "⌃⌥T").
            item.tooltip = match label {
                Some(label) => format!("{tip} · {label}"),
                None => tip,
            };
            item.hotkey_label = label.cloned();
            item
        };

        let mut status = MenuItem::new(ItemId::Status, inputs.status_text());
        status.enabled = false;

        let mut pause = hotkey_item(
            ItemId::Pause,
            "toggle_tracking",
            if inputs.paused {
                "Resume tracking"
            } else {
                "Pause tracking"
            },
            (if inputs.paused { RESUME_TIP } else { PAUSE_TIP }).to_string(),
        );
        pause.enabled = !busy;

        let mut privacy_tip = PRIVACY_TIP.to_string();
        if settings.privacy.remember_privacy_mode {
            privacy_tip.push_str(PRIVACY_REMEMBERED_TIP);
        }
        let mut privacy = hotkey_item(
            ItemId::Privacy,
            "toggle_privacy",
            "Privacy mode",
            privacy_tip,
        );
        privacy.checkable = true;
        privacy.checked = inputs.privacy;
        privacy.enabled = !busy;

        let reason = inputs.calibration_reason.trim();
        let calibrate_tip = if needs && !reason.is_empty() {
            format!("Switching is off until you calibrate ({reason})")
        } else {
            CALIBRATE_TIP.to_string()
        };
        let mut calibrate = hotkey_item(
            ItemId::Calibrate,
            "recalibrate",
            if needs {
                "Calibrate now…"
            } else {
                "Calibrate…"
            },
            calibrate_tip,
        );
        calibrate.bold = needs;
        calibrate.enabled = !busy;

        let mut gaze_dot = MenuItem::new(ItemId::GazeDot, "Show gaze dot");
        gaze_dot.checkable = true;
        gaze_dot.checked = settings.ui.show_gaze_overlay;
        gaze_dot.tooltip = GAZE_DOT_TIP.to_string();

        let mut panes = MenuItem::new(ItemId::SplitPanes, "Follow split panes");
        panes.checkable = true;
        panes.checked = settings.panes.enabled;
        panes.tooltip = PANES_TIP.to_string();

        let mut preview = MenuItem::new(ItemId::Preview, "Camera preview…");
        preview.tooltip = PREVIEW_TIP.to_string();

        let login = &inputs.autostart;
        let (login_text, login_tip) = autostart_texts(&login.status);
        let mut autostart = MenuItem::new(ItemId::Autostart, login_text);
        autostart.tooltip = login_tip;
        autostart.checkable = true;
        autostart.checked = login.enabled;
        autostart.visible = login.supported;

        let mut update = MenuItem::new(
            ItemId::Update,
            match inputs.update_version.as_deref() {
                Some(version) if !version.is_empty() => format!("Update to {version}…"),
                _ => "Check for updates…".to_string(),
            },
        );
        update.visible = inputs.update_supported;

        let entries = vec![
            MenuEntry::Item(status),
            MenuEntry::Separator,
            MenuEntry::Item(pause),
            MenuEntry::Item(privacy),
            MenuEntry::Separator,
            MenuEntry::Item(calibrate),
            MenuEntry::Item(gaze_dot),
            MenuEntry::Item(panes),
            MenuEntry::Item(preview),
            MenuEntry::Separator,
            MenuEntry::Item(MenuItem::new(ItemId::Settings, "Settings…")),
            MenuEntry::Item(autostart),
            MenuEntry::Separator,
            MenuEntry::Item(update),
            MenuEntry::Item(MenuItem::new(ItemId::About, format!("About {APP_NAME}"))),
            MenuEntry::Item(MenuItem::new(ItemId::Quit, format!("Quit {APP_NAME}"))),
        ];
        TrayModel {
            entries,
            tooltip: inputs.tooltip_text(),
            status_color: status_color(state),
        }
    }

    /// The item `id` (visible or not).
    pub fn item(&self, id: ItemId) -> Option<&MenuItem> {
        self.entries.iter().find_map(|entry| match entry {
            MenuEntry::Item(item) if item.id == id => Some(item),
            _ => None,
        })
    }

    /// The items that are shown, in menu order.
    pub fn visible_items(&self) -> impl Iterator<Item = &MenuItem> {
        self.entries.iter().filter_map(|entry| match entry {
            MenuEntry::Item(item) if item.visible => Some(item),
            _ => None,
        })
    }
}

// -------------------------------------------------------------------------------- actions
/// What a tray click asks the application to do (the Python widget called the controller or
/// emitted a signal).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrayAction {
    /// `controller.toggle_pause()`.
    TogglePause,
    /// `controller.set_privacy(enabled)`.
    SetPrivacy(bool),
    /// Show the calibration window (`open_calibration`).
    OpenCalibration,
    /// Apply the controller's current settings with `ui.show_gaze_overlay` set (start from the
    /// controller's settings, not a cached copy, so that no other change is undone).
    SetGazeDot(bool),
    /// Likewise for `panes.enabled`.
    SetSplitPanes(bool),
    OpenPreview,
    OpenSettings,
    /// The update entry: install when a newer version is known, else check (`open_update`).
    OpenUpdate,
    OpenAbout,
    /// `quit_requested`.
    Quit,
}

/// A click on the tray icon itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IconClick {
    /// A left click (Qt's `Trigger`).
    Single,
    Double,
    Middle,
}

/// What a backend reports back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendEvent {
    Menu(ItemId),
    Icon(IconClick),
    /// The menu is about to open (Qt's `aboutToShow`).
    AboutToShow,
}

/// The tray icon as a backend draws it.
#[derive(Debug, Clone, PartialEq)]
pub struct TrayImage {
    pub width: u32,
    pub height: u32,
    /// RGBA8.
    pub rgba: Vec<u8>,
    /// A monochrome mask the system tints (the macOS menu bar).
    pub template: bool,
}

/// The OS side of the tray.
pub trait TrayBackend {
    /// Whether the desktop has a system tray to show the icon in.
    fn available(&self) -> bool;
    /// Whether the icon is shown.
    fn is_visible(&self) -> bool;
    fn show(&mut self);
    fn hide(&mut self);
    fn set_icon(&mut self, image: &TrayImage);
    fn set_tooltip(&mut self, tooltip: &str);
    fn set_menu(&mut self, model: &TrayModel);
    /// Show the menu at the pointer (what a single click on the icon does on Windows).
    fn popup_menu(&mut self);
    fn show_message(&mut self, title: &str, message: &str, critical: bool);
    /// Clicks and menu events since the last call.
    fn poll_events(&mut self) -> Vec<BackendEvent>;
}

/// Renders the tray icon of a state for a dark (`true`) or light taskbar as RGBA8
/// `(width, height, pixels)`; the application wires `icons::tray_icon` into it.
pub type IconRenderer = Box<dyn Fn(TrackingState, bool) -> (u32, u32, Vec<u8>)>;

// ----------------------------------------------------------------------------------- Tray
/// Tray icon with the app menu.
///
/// Menu: status line · Pause/Resume tracking · Privacy mode · Calibrate… ·
/// Show gaze dot · Camera preview… · Settings… · Start at login · About · Quit.
/// Global hotkeys are shown next to the actions they trigger. Double-clicking
/// the icon opens the settings; a middle click pauses or resumes tracking. On
/// Windows a single click opens the menu as well (see [`MENU_ON_CLICK`]).
pub struct Tray {
    backend: Box<dyn TrayBackend>,
    render_icon: IconRenderer,
    autostart: Box<dyn AutostartBackend>,
    clock: Box<dyn Fn() -> f64>,
    inputs: TrayInputs,
    model: TrayModel,
    pushed_model: Option<TrayModel>,
    pushed_tooltip: Option<String>,
    muted_until: f64,
    icon_key: Option<(TrackingState, bool)>,
    menu_on_click: bool,
    /// When the pending single click opens the menu (the Python's single-shot click timer).
    click_due: Option<f64>,
    ignore_clicks_until: f64,
    disposed: bool,
}

impl Tray {
    /// Args:
    ///
    /// * `backend`: draws the tray ([`native_backend`]).
    /// * `render_icon`: the icon of a state, see [`IconRenderer`].
    /// * `autostart`: the start-at-login backend ([`SystemAutostart`]).
    /// * `clock`: monotonic seconds; only used by [`Tray::mute_notifications`] and the clicks.
    pub fn new(
        backend: Box<dyn TrayBackend>,
        render_icon: IconRenderer,
        autostart: Box<dyn AutostartBackend>,
        clock: Box<dyn Fn() -> f64>,
        inputs: TrayInputs,
    ) -> Tray {
        let model = TrayModel::build(&inputs);
        let mut tray = Tray {
            backend,
            render_icon,
            autostart,
            clock,
            inputs,
            model,
            pushed_model: None,
            pushed_tooltip: None,
            muted_until: f64::NEG_INFINITY,
            icon_key: None,
            menu_on_click: MENU_ON_CLICK,
            click_due: None,
            ignore_clicks_until: f64::NEG_INFINITY,
            disposed: false,
        };
        tray.refresh_autostart();
        tray.apply();
        tray
    }

    /// [`Tray::new`] with the real autostart backend and clock.
    pub fn with_defaults(
        backend: Box<dyn TrayBackend>,
        render_icon: IconRenderer,
        inputs: TrayInputs,
    ) -> Tray {
        Tray::new(
            backend,
            render_icon,
            Box::new(SystemAutostart),
            Box::new(monotonic),
            inputs,
        )
    }

    // ------------------------------------------------------------------ public API
    /// Whether the desktop has a system tray to show the icon in.
    pub fn available(&self) -> bool {
        self.backend.available()
    }

    /// The tracking state the tray currently shows.
    pub fn state(&self) -> TrackingState {
        self.inputs.state
    }

    /// The menu, tooltip and status line as last built.
    pub fn model(&self) -> &TrayModel {
        &self.model
    }

    /// Whether a single click is waiting to open the menu (see [`MENU_ON_CLICK`]).
    pub fn click_pending(&self) -> bool {
        self.click_due.is_some()
    }

    /// Open the menu on a single click, as on Windows (see [`MENU_ON_CLICK`]).
    pub fn set_menu_on_click(&mut self, enabled: bool) {
        self.menu_on_click = enabled;
    }

    /// Show the icon (logs a hint when the desktop has no tray).
    pub fn show(&mut self) {
        if self.disposed {
            return;
        }
        self.refresh_icon(true);
        self.backend.show();
        if !self.available() {
            log::warn!(
                "No system tray is available; control the app with '{} …' \
                 or enable a tray/AppIndicator extension",
                cli_command_text(&["ctl"])
            );
        }
    }

    /// Hide the icon (it can be shown again).
    pub fn hide(&mut self) {
        if !self.disposed {
            self.backend.hide();
        }
    }

    /// Hide the icon and release the menu (call before quitting). Idempotent.
    ///
    /// Inputs that still arrive afterwards are ignored.
    pub fn dispose(&mut self) {
        if self.disposed {
            return;
        }
        self.disposed = true;
        self.click_due = None;
        self.backend.hide();
    }

    /// Reflect the controller's current values in the icon, tooltip and menu.
    ///
    /// The login item (`inputs.autostart`) is the tray's own business and kept as it is.
    pub fn sync(&mut self, mut inputs: TrayInputs) {
        if self.disposed {
            return;
        }
        inputs.autostart = self.inputs.autostart.clone();
        // Stats arrive every few seconds: a cheap moment to notice that the taskbar switched
        // between light and dark.
        let icon_moment = inputs.state != self.inputs.state || inputs.stats != self.inputs.stats;
        self.inputs = inputs;
        self.model = TrayModel::build(&self.inputs);
        self.push(false);
        if icon_moment {
            self.refresh_icon(false);
        }
    }

    /// The tooltip, e.g. `"Eye Tracker — Tracking · 4 fps · CPU 0.6 %"`.
    pub fn tooltip_text(&self) -> String {
        self.inputs.tooltip_text()
    }

    /// The disabled first menu line, e.g. `"Tracking"` or the camera error.
    pub fn status_text(&self) -> String {
        self.inputs.status_text()
    }

    /// Show a tray notification; returns whether one was shown.
    ///
    /// Honours `settings.general.notifications` and [`Tray::mute_notifications`] unless
    /// `force` (errors caused by a user action). `critical` shows a warning icon instead of
    /// the app icon.
    pub fn notify(&mut self, title: &str, message: &str, force: bool, critical: bool) -> bool {
        self.deliver(title, message, !force, critical, true)
    }

    /// A notification the controller sent.
    ///
    /// The controller has already applied `settings.general.notifications`: what it sends
    /// while they are off is something the user asked to be told (walk-away "notify", shoulder
    /// guard "notify") or a failed lock. Filtering again would swallow exactly those. The
    /// startup mute still applies.
    pub fn notify_from_controller(&mut self, title: &str, message: &str) -> bool {
        self.deliver(title, message, true, false, false)
    }

    /// Suppress notifications for `seconds` (a start at login stays quiet).
    ///
    /// Only [`Tray::notify`] calls with `force` get through meanwhile.
    pub fn mute_notifications(&mut self, seconds: f64) {
        self.muted_until = (self.clock)() + seconds.max(0.0);
    }

    /// Seconds until [`Tray::mute_notifications`] ends (0 when not muted).
    pub fn muted_for(&self) -> f64 {
        (self.muted_until - (self.clock)()).max(0.0)
    }

    /// Name a newer version in the menu (`None`: back to "Check for updates…").
    pub fn set_update_available(&mut self, version: Option<&str>) {
        if self.disposed {
            return;
        }
        self.inputs.update_version = version.map(str::to_string);
        self.apply();
    }

    /// The system switched between light and dark (see [`system_tray_is_dark`]).
    pub fn on_theme_changed(&mut self) {
        self.refresh_icon(false);
    }

    /// Run what the backend has collected (menu clicks, icon clicks, the menu opening) and the
    /// single-click timer; returns the actions for the application to run.
    pub fn poll(&mut self) -> Vec<TrayAction> {
        let mut actions = Vec::new();
        for event in self.backend.poll_events() {
            match event {
                BackendEvent::Menu(id) => actions.extend(self.activate(id)),
                BackendEvent::Icon(click) => actions.extend(self.icon_click(click)),
                BackendEvent::AboutToShow => self.about_to_show(),
            }
        }
        let now = (self.clock)();
        self.tick(now);
        actions
    }

    /// A click on a menu item (what a triggered `QAction` did).
    pub fn activate(&mut self, id: ItemId) -> Vec<TrayAction> {
        if self.disposed {
            return Vec::new();
        }
        let Some(item) = self.model.item(id) else {
            return Vec::new();
        };
        if !item.enabled || !item.visible {
            return Vec::new();
        }
        let checked = item.checked;
        let actions = match id {
            ItemId::Status => Vec::new(),
            ItemId::Pause => vec![TrayAction::TogglePause],
            ItemId::Privacy => vec![TrayAction::SetPrivacy(!checked)],
            ItemId::Calibrate => vec![TrayAction::OpenCalibration],
            ItemId::GazeDot => vec![TrayAction::SetGazeDot(!checked)],
            ItemId::SplitPanes => vec![TrayAction::SetSplitPanes(!checked)],
            ItemId::Preview => vec![TrayAction::OpenPreview],
            ItemId::Settings => vec![TrayAction::OpenSettings],
            ItemId::Autostart => {
                self.change_autostart(!checked);
                Vec::new()
            }
            ItemId::Update => vec![TrayAction::OpenUpdate],
            ItemId::About => vec![TrayAction::OpenAbout],
            ItemId::Quit => vec![TrayAction::Quit],
        };
        // The check marks always show the real state, whatever the controller call does: a
        // native menu flips a check mark by itself when it is clicked.
        self.push(true);
        actions
    }

    /// A click on the icon: double-click opens the settings, a middle click pauses or resumes
    /// tracking, and on Windows a single click opens the menu once it is clearly no
    /// double-click (the first click of a double-click is reported as a single click too).
    pub fn icon_click(&mut self, click: IconClick) -> Vec<TrayAction> {
        if self.disposed {
            return Vec::new();
        }
        let now = (self.clock)();
        match click {
            IconClick::Double => {
                self.click_due = None;
                // The second click of the pair may be reported as a single click too.
                self.ignore_clicks_until = now + DOUBLE_CLICK_INTERVAL;
                vec![TrayAction::OpenSettings]
            }
            IconClick::Middle => {
                self.push(true);
                vec![TrayAction::TogglePause]
            }
            IconClick::Single => {
                if self.menu_on_click && now >= self.ignore_clicks_until {
                    self.click_due = Some(now + DOUBLE_CLICK_INTERVAL);
                }
                Vec::new()
            }
        }
    }

    /// Open the menu of a pending single click once its time has come.
    pub fn tick(&mut self, now: f64) {
        if let Some(due) = self.click_due
            && now >= due
        {
            self.click_due = None;
            self.popup_menu();
        }
    }

    /// Show the menu at the pointer (what a single click on the icon does on Windows).
    pub fn popup_menu(&mut self) {
        if self.disposed {
            return;
        }
        self.about_to_show();
        self.backend.popup_menu();
    }

    /// The menu is about to open: the login item can be changed elsewhere (Task Manager, System
    /// Settings), so read it again. The application feeds fresh hotkeys and state with
    /// [`Tray::sync`] (hotkeys are registered only once the controller has started).
    pub fn about_to_show(&mut self) {
        if self.disposed {
            return;
        }
        self.refresh_autostart();
        self.apply();
    }

    // ------------------------------------------------------------------------ internals
    /// Rebuild the model from the inputs and show it.
    fn apply(&mut self) {
        self.model = TrayModel::build(&self.inputs);
        self.push(false);
        self.refresh_icon(false);
    }

    /// Hand the model to the backend; each change is a round trip to the shell, so unchanged
    /// parts are skipped unless `force`.
    fn push(&mut self, force: bool) {
        if force || self.pushed_model.as_ref() != Some(&self.model) {
            self.backend.set_menu(&self.model);
            self.pushed_model = Some(self.model.clone());
        }
        if force || self.pushed_tooltip.as_deref() != Some(&self.model.tooltip) {
            self.backend.set_tooltip(&self.model.tooltip);
            self.pushed_tooltip = Some(self.model.tooltip.clone());
        }
    }

    fn refresh_icon(&mut self, force: bool) {
        if self.disposed {
            return;
        }
        let mask = cfg!(target_os = "macos");
        let dark = if mask { true } else { system_tray_is_dark() };
        let key = (self.inputs.state, dark);
        if Some(key) == self.icon_key && !force {
            return;
        }
        self.icon_key = Some(key);
        let (width, height, rgba) = (self.render_icon)(self.inputs.state, dark);
        self.backend.set_icon(&TrayImage {
            width,
            height,
            rgba,
            template: mask,
        });
    }

    fn refresh_autostart(&mut self) {
        let backend = &*self.autostart;
        let supported = backend.is_supported();
        let enabled = supported && backend.is_enabled();
        let status = if supported && !enabled {
            autostart_status(backend)
        } else {
            String::new()
        };
        self.inputs.autostart = AutostartInputs {
            supported,
            enabled,
            status,
        };
    }

    fn change_autostart(&mut self, enable: bool) {
        let result = if enable {
            self.autostart.enable(true)
        } else {
            self.autostart.disable()
        };
        if let Err(error) = result {
            // AutostartError: the message is meant for the user.
            if let Some(error) = error.downcast_ref::<platform_autostart::AutostartError>() {
                log::warn!("Could not change start at login: {error}");
                let message = error.to_string();
                let message = if message.is_empty() {
                    "The login item could not be changed.".to_string()
                } else {
                    message
                };
                self.notify("Start at login", &message, true, true);
            } else {
                log::error!("Changing start at login failed: {error}");
                self.notify(
                    "Start at login",
                    "The login item could not be changed; see the log for details.",
                    true,
                    true,
                );
            }
        }
        self.refresh_autostart();
        self.apply();
    }

    /// Show a message unless `filtered` and muted (or off in the settings).
    fn deliver(
        &mut self,
        title: &str,
        message: &str,
        filtered: bool,
        critical: bool,
        check_settings: bool,
    ) -> bool {
        if self.disposed {
            return false;
        }
        if filtered && check_settings && !self.inputs.settings.general.notifications {
            log::debug!("Notification off in settings: {title}");
            return false;
        }
        if filtered && (self.clock)() < self.muted_until {
            log::debug!("Notification muted: {title}");
            return false;
        }
        if !self.messages_available() {
            log::info!("Notification (no tray to show it): {title} — {message}");
            return false;
        }
        self.backend.show_message(title, message, critical);
        true
    }

    // A desktop without a tray area (GNOME without an AppIndicator extension, a bare window
    // manager) drops notifications from tray apps silently. Callers must learn that nothing was
    // shown: an unseen calibration hint, for instance, must not count as given.
    fn messages_available(&self) -> bool {
        self.backend.is_visible() && self.backend.available()
    }
}

// ----------------------------------------------------------------------------- backends
/// A backend without a tray: nothing is shown and no notification is delivered.
#[derive(Debug, Default)]
pub struct NullBackend;

impl TrayBackend for NullBackend {
    fn available(&self) -> bool {
        false
    }
    fn is_visible(&self) -> bool {
        false
    }
    fn show(&mut self) {}
    fn hide(&mut self) {}
    fn set_icon(&mut self, _image: &TrayImage) {}
    fn set_tooltip(&mut self, _tooltip: &str) {}
    fn set_menu(&mut self, _model: &TrayModel) {}
    fn popup_menu(&mut self) {}
    fn show_message(&mut self, _title: &str, _message: &str, _critical: bool) {}
    fn poll_events(&mut self) -> Vec<BackendEvent> {
        Vec::new()
    }
}

/// The tray backend of this OS (or [`NullBackend`] where it cannot be created).
///
/// Create it on the UI thread once the event loop runs: the Windows and macOS trays need the
/// thread that pumps the OS messages.
pub fn native_backend() -> Box<dyn TrayBackend> {
    #[cfg(any(windows, target_os = "macos"))]
    {
        Box::new(native::NativeBackend::new())
    }
    #[cfg(target_os = "linux")]
    {
        Box::new(sni::SniBackend::new())
    }
    #[cfg(not(any(windows, target_os = "macos", target_os = "linux")))]
    {
        Box::new(NullBackend)
    }
}

/// Show a desktop notification (the app icon, or the system's warning icon when `critical`).
///
/// Sent from a thread of its own: on Linux the call talks to the notification daemon over
/// D-Bus and may block.
#[cfg(any(windows, target_os = "macos", target_os = "linux"))]
fn send_notification(title: &str, message: &str, critical: bool) {
    let mut notification = notify_rust::Notification::new();
    notification
        .appname(APP_NAME)
        .summary(title)
        .body(message)
        .icon(if critical { "dialog-warning" } else { APP_SLUG })
        .timeout(notify_rust::Timeout::Milliseconds(NOTIFICATION_MS));
    let spawned = std::thread::Builder::new()
        .name("notification".to_string())
        .spawn(move || {
            #[cfg(target_os = "macos")]
            set_notification_application();
            if let Err(error) = notification.show() {
                log::warn!("Could not show a notification: {error}");
            }
        });
    if let Err(error) = spawned {
        log::warn!("Could not start the notification thread: {error}");
    }
}

/// Tell notify-rust which app the notifications come from, once. Left alone, it asks AppleScript
/// for the app "use_default" and macOS shows a "Where is use_default?" chooser instead.
///
/// It cannot be retried (notify-rust sets it once, even on failure), so it is chosen up front:
/// the app's identifier inside its bundle, Terminal's (as notify-rust suggests) for a bare
/// binary, which has no identifier macOS knows.
#[cfg(target_os = "macos")]
fn set_notification_application() {
    static SET: std::sync::Once = std::sync::Once::new();
    SET.call_once(|| {
        let in_bundle = objc2_foundation::NSBundle::mainBundle()
            .bundleIdentifier()
            .is_some_and(|id| id.to_string() == crate::APP_ID);
        let bundle = if in_bundle {
            crate::APP_ID
        } else {
            "com.apple.Terminal"
        };
        match notify_rust::set_application(bundle) {
            Ok(()) => log::debug!("Notifications are sent as {bundle}"),
            Err(error) => log::warn!("Could not send notifications as {bundle}: {error}"),
        }
    });
}

/// The coloured dot in front of the status line (16 × 16, straight-alpha RGBA).
#[cfg(any(windows, target_os = "macos", target_os = "linux"))]
fn status_dot(color: &str) -> Option<super::icons::Rgba> {
    super::icons::status_dot_icon(color).best(16).cloned()
}

#[cfg(any(windows, target_os = "macos"))]
mod native {
    //! `tray-icon` with its `muda` menu (Windows and macOS).
    //!
    //! `muda` menus have no tooltips, no bold items and no `aboutToShow` hook, so those parts of
    //! the model are not drawn; the text of "Calibrate now…" carries the emphasis, and the
    //! application's regular [`Tray::sync`](super::Tray::sync) keeps the menu current.
    //! `tray-icon` reports a double-click on Windows only.

    use std::collections::HashMap;

    use tray_icon::menu::Icon as MenuIcon;
    use tray_icon::menu::accelerator::Accelerator;
    use tray_icon::menu::{
        CheckMenuItem, IconMenuItem, IsMenuItem, Menu, MenuEvent, MenuId,
        MenuItem as PlainMenuItem, PredefinedMenuItem,
    };
    use tray_icon::{
        Icon, MouseButton, MouseButtonState, TrayIcon, TrayIconBuilder, TrayIconEvent,
    };

    use super::{
        BackendEvent, IconClick, ItemId, MENU_ON_CLICK, MenuEntry, MenuItem, TrayBackend,
        TrayImage, TrayModel, send_notification, status_dot,
    };

    enum Widget {
        Plain(PlainMenuItem),
        Check(CheckMenuItem),
        Status(IconMenuItem),
    }

    impl Widget {
        fn create(item: &MenuItem, color: &str) -> Widget {
            let id = MenuId::new(item.id.key());
            if item.id == ItemId::Status {
                let icon = status_dot(color)
                    .and_then(|dot| MenuIcon::from_rgba(dot.data, dot.width, dot.height).ok());
                Widget::Status(IconMenuItem::with_id(
                    id,
                    &item.text,
                    item.enabled,
                    icon,
                    None,
                ))
            } else if item.checkable {
                Widget::Check(CheckMenuItem::with_id(
                    id,
                    &item.text,
                    item.enabled,
                    item.checked,
                    None,
                ))
            } else {
                Widget::Plain(PlainMenuItem::with_id(id, &item.text, item.enabled, None))
            }
        }

        fn as_menu_item(&self) -> &dyn IsMenuItem {
            match self {
                Widget::Plain(item) => item,
                Widget::Check(item) => item,
                Widget::Status(item) => item,
            }
        }

        fn update(&self, item: &MenuItem, color: &str) {
            let accelerator = item
                .shortcut
                .as_ref()
                .and_then(|hotkey| hotkey.to_string().parse::<Accelerator>().ok());
            match self {
                Widget::Plain(widget) => {
                    widget.set_text(&item.text);
                    widget.set_enabled(item.enabled);
                    let _ = widget.set_accelerator(accelerator);
                }
                Widget::Check(widget) => {
                    widget.set_text(&item.text);
                    widget.set_enabled(item.enabled);
                    widget.set_checked(item.checked);
                    let _ = widget.set_accelerator(accelerator);
                }
                Widget::Status(widget) => {
                    widget.set_text(&item.text);
                    widget.set_enabled(item.enabled);
                    let icon = status_dot(color)
                        .and_then(|dot| MenuIcon::from_rgba(dot.data, dot.width, dot.height).ok());
                    widget.set_icon(icon);
                }
            }
        }
    }

    pub struct NativeBackend {
        menu: Menu,
        widgets: HashMap<ItemId, Widget>,
        /// The shown items and separators (`None`) the menu was last built with.
        layout: Vec<Option<ItemId>>,
        tray: Option<TrayIcon>,
        image: Option<TrayImage>,
        tooltip: String,
        visible: bool,
    }

    impl NativeBackend {
        pub fn new() -> NativeBackend {
            NativeBackend {
                menu: Menu::new(),
                widgets: HashMap::new(),
                layout: Vec::new(),
                tray: None,
                image: None,
                tooltip: String::new(),
                visible: false,
            }
        }

        fn rebuild_menu(&mut self, model: &TrayModel, layout: Vec<Option<ItemId>>) {
            while self.menu.remove_at(0).is_some() {}
            self.widgets.clear();
            for entry in &model.entries {
                match entry {
                    MenuEntry::Item(item) if item.visible => {
                        let widget = Widget::create(item, model.status_color);
                        if let Err(error) = self.menu.append(widget.as_menu_item()) {
                            log::warn!("Could not add a tray menu item: {error}");
                        }
                        self.widgets.insert(item.id, widget);
                    }
                    MenuEntry::Item(_) => {}
                    MenuEntry::Separator => {
                        if let Err(error) = self.menu.append(&PredefinedMenuItem::separator()) {
                            log::warn!("Could not add a tray menu separator: {error}");
                        }
                    }
                }
            }
            self.layout = layout;
        }

        fn apply_image(tray: &TrayIcon, image: &TrayImage) {
            let icon = match Icon::from_rgba(image.rgba.clone(), image.width, image.height) {
                Ok(icon) => icon,
                Err(error) => {
                    log::warn!("Invalid tray icon: {error}");
                    return;
                }
            };
            #[cfg(target_os = "macos")]
            let result = if image.template {
                tray.set_icon_templated(Some(icon))
            } else {
                tray.set_icon(Some(icon))
            };
            #[cfg(not(target_os = "macos"))]
            let result = tray.set_icon(Some(icon));
            if let Err(error) = result {
                log::warn!("Could not set the tray icon: {error}");
            }
        }
    }

    impl TrayBackend for NativeBackend {
        fn available(&self) -> bool {
            true
        }

        fn is_visible(&self) -> bool {
            self.tray.is_some() && self.visible
        }

        fn show(&mut self) {
            if self.tray.is_none() {
                let mut builder = TrayIconBuilder::new()
                    .with_menu(Box::new(self.menu.clone()))
                    .with_tooltip(&self.tooltip)
                    // Windows: the single click opens the menu after the double-click window.
                    .with_menu_on_left_click(!MENU_ON_CLICK);
                if let Some(image) = &self.image
                    && let Ok(icon) = Icon::from_rgba(image.rgba.clone(), image.width, image.height)
                {
                    builder = builder.with_icon(icon);
                }
                match builder.build() {
                    Ok(tray) => {
                        if let Some(image) = &self.image {
                            Self::apply_image(&tray, image);
                        }
                        self.tray = Some(tray);
                    }
                    Err(error) => {
                        log::warn!("Could not create the tray icon: {error}");
                        return;
                    }
                }
            }
            if let Some(tray) = &self.tray
                && let Err(error) = tray.set_visible(true)
            {
                log::warn!("Could not show the tray icon: {error}");
            }
            self.visible = true;
        }

        fn hide(&mut self) {
            if let Some(tray) = &self.tray
                && let Err(error) = tray.set_visible(false)
            {
                log::warn!("Could not hide the tray icon: {error}");
            }
            self.visible = false;
        }

        fn set_icon(&mut self, image: &TrayImage) {
            self.image = Some(image.clone());
            if let Some(tray) = &self.tray {
                Self::apply_image(tray, image);
            }
        }

        fn set_tooltip(&mut self, tooltip: &str) {
            self.tooltip = tooltip.to_string();
            if let Some(tray) = &self.tray
                && let Err(error) = tray.set_tooltip(Some(tooltip))
            {
                log::debug!("Could not set the tray tooltip: {error}");
            }
        }

        fn set_menu(&mut self, model: &TrayModel) {
            let layout: Vec<Option<ItemId>> = model
                .entries
                .iter()
                .filter_map(|entry| match entry {
                    MenuEntry::Item(item) if item.visible => Some(Some(item.id)),
                    MenuEntry::Item(_) => None,
                    MenuEntry::Separator => Some(None),
                })
                .collect();
            if layout != self.layout {
                self.rebuild_menu(model, layout);
            }
            for item in model.visible_items() {
                if let Some(widget) = self.widgets.get(&item.id) {
                    widget.update(item, model.status_color);
                }
            }
        }

        fn popup_menu(&mut self) {
            if let Some(tray) = &self.tray {
                tray.show_menu();
            }
        }

        fn show_message(&mut self, title: &str, message: &str, critical: bool) {
            send_notification(title, message, critical);
        }

        fn poll_events(&mut self) -> Vec<BackendEvent> {
            let mut events = Vec::new();
            while let Ok(event) = MenuEvent::receiver().try_recv() {
                if let Some(id) = ItemId::from_key(event.id.as_ref()) {
                    events.push(BackendEvent::Menu(id));
                }
            }
            while let Ok(event) = TrayIconEvent::receiver().try_recv() {
                match event {
                    TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } => events.push(BackendEvent::Icon(IconClick::Single)),
                    TrayIconEvent::Click {
                        button: MouseButton::Middle,
                        button_state: MouseButtonState::Up,
                        ..
                    } => events.push(BackendEvent::Icon(IconClick::Middle)),
                    TrayIconEvent::DoubleClick {
                        button: MouseButton::Left,
                        ..
                    } => events.push(BackendEvent::Icon(IconClick::Double)),
                    _ => {}
                }
            }
            events
        }
    }
}

#[cfg(target_os = "linux")]
mod sni {
    //! `ksni` (StatusNotifierItem over D-Bus) for Linux.
    //!
    //! SNI hosts show the menu themselves and know no double-click, so the icon reports a left
    //! click (`Activate`, ignored unless the menu is opened on a click) and a middle click
    //! (`SecondaryActivate`); the settings open from the menu. Menu tooltips and bold items do
    //! not exist in DBusMenu.

    use crossbeam_channel::{Receiver, Sender, unbounded};
    use ksni::blocking::{Handle, TrayMethods};
    use ksni::menu::{CheckmarkItem, StandardItem};

    use super::{
        BackendEvent, IconClick, ItemId, MenuEntry, MenuItem, TrayBackend, TrayImage, TrayModel,
        send_notification, status_dot,
    };
    use crate::platform::hotkeys::Hotkey;
    use crate::{APP_NAME, APP_SLUG};

    struct SniTray {
        model: TrayModel,
        image: Option<TrayImage>,
        tooltip: String,
        events: Sender<BackendEvent>,
    }

    /// A DBusMenu shortcut: modifier names followed by the key.
    fn shortcut(hotkey: &Hotkey) -> Vec<Vec<String>> {
        let mut keys: Vec<String> = hotkey
            .ordered_modifiers()
            .into_iter()
            .map(|modifier| {
                match modifier {
                    "ctrl" => "Control",
                    "alt" => "Alt",
                    "shift" => "Shift",
                    _ => "Super",
                }
                .to_string()
            })
            .collect();
        keys.push(hotkey.key().to_uppercase());
        vec![keys]
    }

    /// The status dot as a PNG (what DBusMenu takes as an item icon).
    fn dot_png(dot: &crate::ui::icons::Rgba) -> Option<Vec<u8>> {
        let mut pixmap = tiny_skia::Pixmap::new(dot.width, dot.height)?;
        for (pixel, rgba) in pixmap.pixels_mut().iter_mut().zip(dot.data.chunks_exact(4)) {
            *pixel =
                tiny_skia::ColorU8::from_rgba(rgba[0], rgba[1], rgba[2], rgba[3]).premultiply();
        }
        pixmap.encode_png().ok()
    }

    impl SniTray {
        fn menu_item(&self, item: &MenuItem) -> ksni::MenuItem<SniTray> {
            let id = item.id;
            let activate: Box<dyn Fn(&mut SniTray) + Send> = Box::new(move |this: &mut SniTray| {
                let _ = this.events.send(BackendEvent::Menu(id));
            });
            let shortcut = item.shortcut.as_ref().map(shortcut).unwrap_or_default();
            if item.checkable {
                CheckmarkItem {
                    label: item.text.clone(),
                    enabled: item.enabled,
                    visible: item.visible,
                    checked: item.checked,
                    shortcut,
                    activate,
                    ..Default::default()
                }
                .into()
            } else {
                let icon_data = if item.id == ItemId::Status {
                    status_dot(self.model.status_color)
                        .and_then(|dot| dot_png(&dot))
                        .unwrap_or_default()
                } else {
                    Vec::new()
                };
                StandardItem {
                    label: item.text.clone(),
                    enabled: item.enabled,
                    visible: item.visible,
                    icon_data,
                    shortcut,
                    activate,
                    ..Default::default()
                }
                .into()
            }
        }
    }

    impl ksni::Tray for SniTray {
        fn id(&self) -> String {
            APP_SLUG.to_string()
        }

        fn title(&self) -> String {
            APP_NAME.to_string()
        }

        fn icon_name(&self) -> String {
            if self.image.is_some() {
                String::new()
            } else {
                APP_SLUG.to_string()
            }
        }

        fn icon_pixmap(&self) -> Vec<ksni::Icon> {
            let Some(image) = &self.image else {
                return Vec::new();
            };
            // ARGB32 in network byte order.
            let data = image
                .rgba
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|[r, g, b, a]| [*a, *r, *g, *b])
                .collect();
            vec![ksni::Icon {
                width: image.width as i32,
                height: image.height as i32,
                data,
            }]
        }

        fn tool_tip(&self) -> ksni::ToolTip {
            ksni::ToolTip {
                title: self.tooltip.clone(),
                ..Default::default()
            }
        }

        fn menu(&self) -> Vec<ksni::MenuItem<Self>> {
            self.model
                .entries
                .iter()
                .map(|entry| match entry {
                    MenuEntry::Item(item) => self.menu_item(item),
                    MenuEntry::Separator => ksni::MenuItem::Separator,
                })
                .collect()
        }

        fn activate(&mut self, _x: i32, _y: i32) {
            let _ = self.events.send(BackendEvent::Icon(IconClick::Single));
        }

        fn secondary_activate(&mut self, _x: i32, _y: i32) {
            let _ = self.events.send(BackendEvent::Icon(IconClick::Middle));
        }

        fn menu_about_to_show(&mut self) {
            let _ = self.events.send(BackendEvent::AboutToShow);
        }
    }

    pub struct SniBackend {
        handle: Option<Handle<SniTray>>,
        /// The latest state, for a tray that is shown (again) later.
        model: TrayModel,
        image: Option<TrayImage>,
        tooltip: String,
        events: Sender<BackendEvent>,
        received: Receiver<BackendEvent>,
    }

    impl SniBackend {
        pub fn new() -> SniBackend {
            let (events, received) = unbounded();
            SniBackend {
                handle: None,
                model: TrayModel {
                    entries: Vec::new(),
                    tooltip: String::new(),
                    status_color: "",
                },
                image: None,
                tooltip: String::new(),
                events,
                received,
            }
        }

        fn update(&self, change: impl FnOnce(&mut SniTray)) {
            if let Some(handle) = &self.handle {
                let _ = handle.update(change);
            }
        }
    }

    impl TrayBackend for SniBackend {
        fn available(&self) -> bool {
            self.handle
                .as_ref()
                .is_some_and(|handle| !handle.is_closed())
        }

        fn is_visible(&self) -> bool {
            self.available()
        }

        fn show(&mut self) {
            if self.handle.is_some() {
                return;
            }
            let tray = SniTray {
                model: self.model.clone(),
                image: self.image.clone(),
                tooltip: self.tooltip.clone(),
                events: self.events.clone(),
            };
            match tray.spawn() {
                Ok(handle) => self.handle = Some(handle),
                Err(error) => log::warn!("Could not create the tray icon: {error}"),
            }
        }

        fn hide(&mut self) {
            if let Some(handle) = self.handle.take() {
                handle.shutdown().wait();
            }
        }

        fn set_icon(&mut self, image: &TrayImage) {
            self.image = Some(image.clone());
            let image = image.clone();
            self.update(move |tray| tray.image = Some(image));
        }

        fn set_tooltip(&mut self, tooltip: &str) {
            self.tooltip = tooltip.to_string();
            let tooltip = tooltip.to_string();
            self.update(move |tray| tray.tooltip = tooltip);
        }

        fn set_menu(&mut self, model: &TrayModel) {
            self.model = model.clone();
            let model = model.clone();
            self.update(move |tray| tray.model = model);
        }

        // SNI hosts own the menu: there is no way to pop it up from the app.
        fn popup_menu(&mut self) {}

        fn show_message(&mut self, title: &str, message: &str, critical: bool) {
            send_notification(title, message, critical);
        }

        fn poll_events(&mut self) -> Vec<BackendEvent> {
            self.received.try_iter().collect()
        }
    }
}

#[cfg(test)]
#[path = "tray_tests.rs"]
mod tests;
