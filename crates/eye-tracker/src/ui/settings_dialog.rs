//! Settings dialog: every option of [`Settings`], grouped by task.
//!
//! The dialog edits a *copy* of the controller's settings. Nothing changes until
//! the user presses OK or Apply, which hands the copy over as
//! [`SettingsAction::Apply`] (the app passes it to `Controller::apply_settings`). Each
//! widget is bound to one settings key through a small binding object that remembers
//! what was loaded, so
//!
//! * a value the user did not touch is written back exactly as it was loaded (no
//!   float drift from spin-box rounding), and
//! * settings changed elsewhere while the dialog is open (tray menu, IPC) refresh
//!   the untouched widgets without discarding the user's pending edits.
//!
//! Tooltips come from the field documentation in [`describe_settings`].
//!
//! The state ([`SettingsDialog`]) holds the edited copy and every rule; [`SettingsDialog::show`]
//! only draws it with egui and returns what the Python dialog would have called on the
//! controller as [`SettingsAction`]s.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::{Arc, OnceLock};
use std::thread::JoinHandle;

use crossbeam_channel::Receiver;
use egui::text::LayoutJob;
use egui::{RichText, TextFormat};
use serde_json::Value;

use super::icons;
use super::util::egui_colors::{ACCENT, DANGER, WARNING};
use crate::cli::cli_command_text;
use crate::config::{SettingRow, Settings, describe_settings};
use crate::engine::scheduler::PROFILES;
use crate::gaze::store::CalibrationData;
use crate::platform::autostart::{AutostartBackend, Status};
use crate::platform::hotkeys::{Hotkey, HotkeyArg, HotkeyManager, format_hotkey, parse_hotkey};
use crate::types::TrackingState;
use crate::vision::camera::{CameraInfo, device_index, list_cameras};
use crate::{APP_NAME, paths};

/// Hotkey actions shown on the Hotkeys page: (settings field, label).
pub const HOTKEY_ACTIONS: [(&str, &str); 3] = [
    ("toggle_tracking", "Pause / resume tracking"),
    ("toggle_privacy", "Privacy mode (camera off)"),
    ("recalibrate", "Recalibrate"),
];

/// Explanation under "Start at login" by [`Status`] value.
fn autostart_note(status: &str) -> String {
    match status {
        "stale" => format!(
            "The login item starts a copy of {APP_NAME} that no longer exists or will be gone \
             after a restart. Tick the box and apply to point it at this copy."
        ),
        "other-profile" => format!(
            "The login item starts {APP_NAME} with another settings folder (--config-dir). \
             Tick the box and apply to start this one instead."
        ),
        _ => String::new(),
    }
}

/// Vision backends offered in the settings (`general.backend` values), best first.
pub const BACKEND_LABELS: &[(&str, &str)] = &[
    ("auto", "Automatic (recommended)"),
    (
        "facemesh",
        "Face mesh — head pose + eye direction (most accurate)",
    ),
    (
        "lite",
        "Lite — head pose from face geometry only (lightest)",
    ),
    #[cfg(target_os = "macos")]
    (
        "apple",
        "Apple Vision — head pose + pupils with Apple's face models",
    ),
];

/// The pages: (key, title, subtitle).
pub const PAGES: [(&str, &str, &str); 6] = [
    (
        "general",
        "General",
        "Startup, notifications and the vision backend.",
    ),
    (
        "switching",
        "Switching",
        "When and how the cursor follows your gaze.",
    ),
    (
        "presence",
        "Presence & privacy",
        "Walk-away lock, camera privacy and the shoulder guard.",
    ),
    (
        "camera",
        "Camera & performance",
        "Capture device, frame rate and CPU use.",
    ),
    (
        "hotkeys",
        "Hotkeys",
        "Global shortcuts that work in every application.",
    ),
    (
        "diagnostics",
        "Diagnostics",
        "System report for bug reports and troubleshooting.",
    ),
];

/// Settings metadata (range, choices, documentation) of one "section.field" key.
pub(crate) fn meta(key: &str) -> Option<&'static SettingRow> {
    static ROWS: OnceLock<Vec<SettingRow>> = OnceLock::new();
    ROWS.get_or_init(describe_settings)
        .iter()
        .find(|row| row.key == key)
}

fn doc(key: &str) -> &'static str {
    meta(key).map_or("", |row| row.doc)
}

/// `value[section][field]` of a settings dictionary.
fn get(settings: &Value, key: &str) -> Value {
    let (section, field) = key.split_once('.').unwrap_or((key, ""));
    settings[section][field].clone()
}

fn set(settings: &mut Value, key: &str, value: Value) {
    let (section, field) = key.split_once('.').unwrap_or((key, ""));
    if let Some(section) = settings.get_mut(section).and_then(Value::as_object_mut) {
        section.insert(field.to_string(), value);
    }
}

// ============================================================ controller data
const TRACKING_STATES: [TrackingState; 10] = [
    TrackingState::Starting,
    TrackingState::NeedsCalibration,
    TrackingState::Calibrating,
    TrackingState::Tracking,
    TrackingState::Paused,
    TrackingState::Privacy,
    TrackingState::Away,
    TrackingState::Locked,
    TrackingState::Yielded,
    TrackingState::CameraError,
];

/// One line summarising `Controller.stats_changed` data.
///
/// Example: `"Tracking · 4.0 fps (target 4) · 9.5 ms/frame · 38 % skipped · CPU 0.6 %"`.
/// Unknown or missing keys are left out, so this never fails.
pub fn format_stats(stats: &Value) -> String {
    let Some(stats) = stats.as_object() else {
        return String::new();
    };
    let mut parts: Vec<String> = Vec::new();
    let text = |key: &str| {
        stats
            .get(key)
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
    };
    if let Some(label) = text("state_label") {
        parts.push(label.to_string());
    } else if let Some(state) = text("state") {
        let known = TRACKING_STATES.iter().find(|s| s.value() == state);
        parts.push(known.map_or(state, |s| s.label()).to_string());
    }
    // `as_f64` is `None` for booleans and JSON has no NaN: junk is left out.
    let number = |key: &str| {
        stats
            .get(key)
            .and_then(Value::as_f64)
            .filter(|v| v.is_finite())
    };
    if let Some(fps) = number("fps") {
        let mut text = format!("{fps:.1} fps");
        if let Some(target) = number("target_fps").filter(|t| *t != 0.0) {
            text += &format!(" (target {target})");
        }
        parts.push(text);
    }
    if let Some(inference) = number("inference_ms").filter(|v| *v != 0.0) {
        parts.push(format!("{inference:.1} ms/frame"));
    }
    if let Some(skip) = number("skip_ratio") {
        parts.push(format!("{:.0} % skipped", skip * 100.0));
    }
    if let Some(cpu) = number("cpu_percent") {
        parts.push(format!("CPU {cpu:.1} %"));
    }
    if let Some(backend) = text("backend") {
        parts.push(backend.to_string());
    }
    parts.join(" · ")
}

/// List cameras that deliver frames (runs on a worker thread; tests replace it).
///
/// Indices in `skip` are never opened: the camera the tracker is using must
/// not be probed, since closing the probe would tear down the tracker's capture
/// of the same device on DirectShow.
pub fn probe_cameras(max_index: u32, api: &str, skip: &[u32]) -> Vec<CameraInfo> {
    list_cameras(max_index, api, skip).unwrap_or_else(|err| {
        log::warn!("Camera detection failed: {err}");
        Vec::new()
    })
}

/// The camera index the controller has applied, as a `skip` set for probing.
///
/// Takes the *applied* settings, not a value being edited in a dialog: that is
/// the device the vision worker may hold open. On Linux a `/dev/videoN` path
/// or a stable `/dev/v4l/by-id/…` link counts as the index it leads to.
/// Empty for a video file.
pub fn camera_in_use(applied: &Settings) -> BTreeSet<u32> {
    let device = applied.camera.device.trim();
    if !device.is_empty() && device.bytes().all(|b| b.is_ascii_digit()) {
        return device.parse().into_iter().collect();
    }
    if !device.starts_with("/dev/") {
        return BTreeSet::new();
    }
    device_index(device).into_iter().collect()
}

/// The `doctor` report as text, without a camera probe (so no camera light flashes).
///
/// The controller's hotkey manager, when there is one, tells the report which hotkeys the OS
/// accepted.
pub fn load_diagnostics_text(hotkey_manager: Option<&dyn HotkeyManager>) -> String {
    crate::diagnostics::format_report(&crate::diagnostics::collect_report(false, hotkey_manager))
}

// =================================================================== camera probe
/// The function a [`CameraProbe`] runs: `(max_index, api, skip)`.
pub type ProbeFn = Arc<dyn Fn(u32, &str, &[u32]) -> Vec<CameraInfo> + Send + Sync>;

/// Runs [`probe_cameras`] on a background thread.
///
/// Opening camera devices can take seconds per index on Windows, which must not
/// freeze the UI. [`CameraProbe::poll`] hands over the result on the UI thread.
pub struct CameraProbe {
    probe: ProbeFn,
    running: Option<(JoinHandle<()>, Receiver<Vec<CameraInfo>>)>,
}

impl Default for CameraProbe {
    fn default() -> Self {
        Self::new(Arc::new(probe_cameras))
    }
}

impl CameraProbe {
    pub fn new(probe: ProbeFn) -> Self {
        Self {
            probe,
            running: None,
        }
    }

    pub fn running(&self) -> bool {
        self.running.is_some()
    }

    /// Start probing (never opening the indices in `skip`).
    ///
    /// Returns `false` if a probe is already running or the thread cannot start.
    pub fn start(&mut self, api: &str, max_index: u32, skip: &BTreeSet<u32>) -> bool {
        if self.running() {
            return false;
        }
        let (tx, rx) = crossbeam_channel::bounded(1);
        let probe = self.probe.clone();
        let api = api.to_string();
        let skip: Vec<u32> = skip.iter().copied().collect();
        let spawned = std::thread::Builder::new()
            .name("eye-tracker-camera-probe".into())
            .spawn(move || {
                // The dialog may have been closed while probing: nobody listens then.
                let _ = tx.send(probe(max_index, &api, &skip));
            });
        match spawned {
            Ok(handle) => {
                self.running = Some((handle, rx));
                true
            }
            Err(err) => {
                log::warn!("Camera detection failed: {err}");
                false
            }
        }
    }

    /// The cameras found, once the probe has finished (then `running()` is `false` again).
    pub fn poll(&mut self) -> Option<Vec<CameraInfo>> {
        let (_, rx) = self.running.as_ref()?;
        let cameras = match rx.try_recv() {
            Ok(cameras) => cameras,
            Err(crossbeam_channel::TryRecvError::Empty) => return None,
            // The probe panicked: nothing found.
            Err(crossbeam_channel::TryRecvError::Disconnected) => Vec::new(),
        };
        if let Some((handle, _)) = self.running.take() {
            let _ = handle.join();
        }
        Some(cameras)
    }
}

/// `"Camera N"` for an index, the text itself for a file.
fn device_label(value: &str) -> String {
    if !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit()) {
        format!("Camera {value}")
    } else {
        value.to_string()
    }
}

/// The combo box label of a detected camera: `"Integrated Camera — 640×480"`.
pub(crate) fn camera_label(info: &CameraInfo) -> String {
    if info.width > 0 && info.height > 0 {
        format!("{} — {}×{}", info.name, info.width, info.height)
    } else {
        info.name.clone()
    }
}

// =================================================================== hotkey editor
/// The modifiers held with a key, by their settings names.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HeldModifiers {
    pub ctrl: bool,
    pub alt: bool,
    pub shift: bool,
    /// Win, Super or Command.
    pub meta: bool,
}

impl HeldModifiers {
    /// egui's modifiers plus the Win/Super key, which egui only reports on macOS (as
    /// `mac_cmd`); elsewhere the owner tracks the `SuperLeft`/`SuperRight` key events.
    pub fn from_egui(modifiers: egui::Modifiers, super_held: bool) -> Self {
        Self {
            ctrl: modifiers.ctrl,
            alt: modifiers.alt,
            shift: modifiers.shift,
            meta: modifiers.mac_cmd || super_held,
        }
    }

    /// Canonical modifier names in ctrl, alt, shift, meta order.
    pub fn names(self) -> Vec<&'static str> {
        [
            (self.ctrl, "ctrl"),
            (self.alt, "alt"),
            (self.shift, "shift"),
            (self.meta, "meta"),
        ]
        .into_iter()
        .filter_map(|(held, name)| held.then_some(name))
        .collect()
    }
}

fn is_modifier_key(key: egui::Key) -> bool {
    use egui::Key;
    matches!(
        key,
        Key::ShiftLeft
            | Key::ShiftRight
            | Key::ControlLeft
            | Key::ControlRight
            | Key::AltLeft
            | Key::AltRight
            | Key::SuperLeft
            | Key::SuperRight
    )
}

/// The settings name of a key, `None` for keys hotkeys do not name.
fn key_name(key: egui::Key) -> Option<String> {
    use egui::Key;
    let name = match key {
        Key::Space => "space",
        Key::Enter => "enter",
        Key::Tab => "tab",
        Key::Escape => "escape",
        Key::Backspace => "backspace",
        Key::Insert => "insert",
        Key::Delete => "delete",
        Key::Home => "home",
        Key::End => "end",
        Key::PageUp => "pageup",
        Key::PageDown => "pagedown",
        Key::ArrowLeft => "left",
        Key::ArrowRight => "right",
        Key::ArrowUp => "up",
        Key::ArrowDown => "down",
        Key::Minus => "minus",
        Key::Equals => "equal",
        Key::OpenBracket => "bracketleft",
        Key::CloseBracket => "bracketright",
        Key::Backslash => "backslash",
        Key::Semicolon => "semicolon",
        Key::Quote => "quote",
        Key::Backtick => "grave",
        Key::Comma => "comma",
        Key::Period => "period",
        Key::Slash => "slash",
        _ => {
            let name = key.name();
            let lower = name.to_lowercase();
            // Letters, digits ("Num1" is named "1") and F1-F24.
            let letter = name.len() == 1 && name.chars().all(|c| c.is_ascii_alphanumeric());
            let function = lower
                .strip_prefix('f')
                .and_then(|n| n.parse::<u32>().ok())
                .is_some_and(|n| (1..=24).contains(&n));
            return (letter || function).then_some(lower);
        }
    };
    Some(name.to_string())
}

/// With Shift held the logical key is the shifted symbol ("!" for Shift+1). Global hotkeys
/// are bound to physical keys, so map back to the unshifted key (US layout).
fn shifted_us(key: egui::Key) -> Option<&'static str> {
    use egui::Key;
    Some(match key {
        Key::Exclamationmark => "1",
        Key::Plus => "equal",
        Key::OpenCurlyBracket => "bracketleft",
        Key::CloseCurlyBracket => "bracketright",
        Key::Pipe => "backslash",
        Key::Colon => "semicolon",
        Key::Questionmark => "slash",
        _ => return None,
    })
}

/// Held modifiers in the notation of [`format_hotkey`], e.g. `"Ctrl+Win+…"`.
///
/// Shown while a shortcut is being recorded. Built with `format_hotkey` so
/// the preview and the finished shortcut name the keys alike (Win on Windows,
/// Super on Linux, ⌃⌥⇧⌘ on macOS). Empty when nothing is held.
pub fn modifier_preview(names: &[&str], macos: Option<bool>) -> String {
    if names.is_empty() {
        return String::new();
    }
    // format_hotkey needs a key: F1 is valid with any modifier; drop its label.
    let text = Hotkey::new(names, "f1")
        .map(|hk| format_hotkey(&hk, macos))
        .unwrap_or_default();
    let prefix = match text.strip_suffix("F1") {
        Some(prefix) => prefix.to_string(),
        // Unexpected label: fall back to the plain names.
        None => {
            let names: Vec<String> = names
                .iter()
                .map(|n| {
                    let mut chars = n.chars();
                    chars.next().map_or_else(String::new, |c| {
                        c.to_uppercase()
                            .chain(chars.flat_map(char::to_lowercase))
                            .collect()
                    })
                })
                .collect();
            names.join("+") + "+"
        }
    };
    prefix + "…"
}

/// Settings-style hotkey text (`"ctrl+alt+p"`) for a key press.
///
/// Returns `None` for a bare modifier press. The result is *not* validated: an
/// unsupported key or a missing modifier yields text that [`parse_hotkey`]
/// rejects with a helpful message.
pub fn hotkey_text_from_key(key: egui::Key, modifiers: HeldModifiers) -> Option<String> {
    if is_modifier_key(key) {
        return None;
    }
    let names = modifiers.names();
    let name = key_name(key)
        .or_else(|| {
            names
                .contains(&"shift")
                .then(|| shifted_us(key).map(String::from))
                .flatten()
        })
        .unwrap_or_else(|| key.name().to_lowercase());
    let mut parts: Vec<String> = names.into_iter().map(String::from).collect();
    parts.push(name);
    Some(parts.join("+"))
}

/// Records a global shortcut: click it, then press the key combination.
///
/// Esc (without modifiers) cancels recording and Backspace/Delete clears the
/// shortcut. [`HotkeyEdit::hotkey`] returns the canonical settings text (`"ctrl+alt+p"`)
/// when the combination is valid, otherwise the raw attempt, so the owner can
/// show why it was rejected.
#[derive(Debug, Clone, Default)]
pub struct HotkeyEdit {
    value: String,
    recording: bool,
    invalid: bool,
    /// The modifiers held so far while recording (`"Ctrl+Alt+…"`).
    preview: Option<String>,
}

impl HotkeyEdit {
    /// Canonical hotkey text, `""` when unset, or the raw text if invalid.
    pub fn hotkey(&self) -> &str {
        &self.value
    }

    /// Set the shortcut (canonicalised when valid). Returns whether the value changed.
    fn set_hotkey(&mut self, text: &str) -> bool {
        let text = text.trim();
        let value = if text.is_empty() {
            String::new()
        } else {
            parse_hotkey(text).map_or_else(|_| text.to_string(), |hk| hk.to_string())
        };
        self.preview = None;
        let changed = value != self.value;
        self.value = value;
        changed
    }

    /// Whether the field is marked as rejected (red frame).
    pub fn invalid(&self) -> bool {
        self.invalid
    }

    /// Whether the field waits for a key combination.
    pub fn recording(&self) -> bool {
        self.recording
    }

    /// What the field shows for the current value (platform notation).
    pub fn display_text(&self) -> String {
        if self.value.is_empty() {
            return String::new();
        }
        parse_hotkey(&self.value).map_or_else(|_| self.value.clone(), |hk| format_hotkey(&hk, None))
    }

    /// What the field shows now: the held modifiers while recording, else the shortcut.
    fn shown_text(&self) -> String {
        self.preview.clone().unwrap_or_else(|| self.display_text())
    }

    fn placeholder(&self) -> &'static str {
        if self.recording {
            "Press a shortcut…"
        } else {
            "Not set"
        }
    }
}

// ======================================================================= bindings
#[derive(Debug, Clone)]
enum Editor {
    Check {
        checked: bool,
    },
    Combo {
        /// (value, label)
        options: Vec<(String, String)>,
        index: usize,
    },
    Int {
        value: i64,
        lo: i64,
        hi: i64,
        step: i64,
        suffix: &'static str,
    },
    /// Shows `value * factor` with `decimals` decimals.
    Float {
        value: f64,
        lo: f64,
        hi: f64,
        factor: f64,
        decimals: usize,
        step: f64,
        suffix: &'static str,
    },
    /// 0..=100 %, for a 0..1 setting.
    Slider {
        value: i64,
        left: &'static str,
        right: &'static str,
    },
    Lines {
        text: String,
        placeholder: &'static str,
    },
    Hotkey(HotkeyEdit),
    /// An editable list: a camera index or a file path.
    Device {
        /// (label, value)
        items: Vec<(String, String)>,
        index: Option<usize>,
        text: String,
    },
}

fn round_to(value: f64, decimals: usize) -> f64 {
    let scale = 10f64.powi(decimals as i32);
    (value * scale).round() / scale
}

impl Editor {
    /// The raw widget state, compared with the state right after loading.
    fn snapshot(&self) -> Value {
        match self {
            Editor::Check { checked } => Value::from(*checked),
            Editor::Combo { index, .. } => Value::from(*index),
            Editor::Int { value, .. } | Editor::Slider { value, .. } => Value::from(*value),
            Editor::Float { value, .. } => Value::from(*value),
            Editor::Lines { text, .. } => Value::from(text.as_str()),
            Editor::Hotkey(edit) => Value::from(edit.hotkey()),
            Editor::Device { .. } => Value::from(self.device_value()),
        }
    }

    /// The settings value the widget shows.
    fn read(&self) -> Value {
        match self {
            Editor::Check { checked } => Value::from(*checked),
            Editor::Combo { options, index } => {
                Value::from(options.get(*index).map_or("", |(v, _)| v.as_str()))
            }
            Editor::Int { value, .. } => Value::from(*value),
            Editor::Float {
                value,
                factor,
                decimals,
                ..
            } => Value::from(round_to(value / factor, decimals + 4)),
            Editor::Slider { value, .. } => Value::from(*value as f64 / 100.0),
            Editor::Lines { text, .. } => Value::from(
                text.lines()
                    .map(str::trim)
                    .filter(|line| !line.is_empty())
                    .collect::<Vec<_>>(),
            ),
            Editor::Hotkey(edit) => Value::from(edit.hotkey()),
            Editor::Device { .. } => Value::from(self.device_value()),
        }
    }

    /// Show a settings value (clamped and rounded as the widget would).
    fn write(&mut self, value: &Value) {
        let text = || {
            value
                .as_str()
                .map_or_else(|| value.to_string(), String::from)
        };
        match self {
            Editor::Check { checked } => *checked = value.as_bool().unwrap_or(false),
            Editor::Combo { options, index } => {
                let wanted = text();
                *index = match options.iter().position(|(v, _)| *v == wanted) {
                    Some(found) => found,
                    // Settings may contain a value unknown to this UI version (or not offered on
                    // this system, like the apple backend); keep it selectable.
                    None => {
                        options.push((wanted.clone(), wanted));
                        options.len() - 1
                    }
                };
            }
            Editor::Int {
                value: shown,
                lo,
                hi,
                ..
            } => {
                let number = value
                    .as_i64()
                    .or_else(|| value.as_f64().map(|v| v.trunc() as i64))
                    .unwrap_or(0);
                *shown = number.clamp(*lo, *hi);
            }
            Editor::Float {
                value: shown,
                lo,
                hi,
                factor,
                decimals,
                ..
            } => {
                let number = value.as_f64().unwrap_or(0.0) * *factor;
                *shown = round_to(number.clamp(*lo, *hi), *decimals);
            }
            Editor::Slider { value: shown, .. } => {
                *shown = ((value.as_f64().unwrap_or(0.0) * 100.0).round() as i64).clamp(0, 100);
            }
            Editor::Lines { text, .. } => {
                let lines: Vec<String> = value
                    .as_array()
                    .map(|items| {
                        items
                            .iter()
                            .map(|v| v.as_str().map_or_else(|| v.to_string(), String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                *text = lines.join("\n");
            }
            Editor::Hotkey(edit) => {
                edit.set_hotkey(value.as_str().unwrap_or(""));
            }
            Editor::Device { .. } => self.select_device(&text()),
        }
    }

    /// The device as entered: the selected item's value, or the typed text.
    fn device_value(&self) -> String {
        let Editor::Device { items, index, text } = self else {
            return String::new();
        };
        match index.and_then(|i| items.get(i)) {
            Some((label, value)) if *label == *text && !value.is_empty() => value.clone(),
            _ => text.trim().to_string(),
        }
    }

    /// Select the item with `value`, adding one if needed.
    fn select_device(&mut self, value: &str) {
        let Editor::Device { items, index, text } = self else {
            return;
        };
        let found = match items.iter().position(|(_, v)| v == value) {
            Some(found) => found,
            None => {
                items.push((device_label(value), value.to_string()));
                items.len() - 1
            }
        };
        *index = Some(found);
        *text = items[found].0.clone();
    }
}

/// Connects one settings key to a widget.
///
/// The snapshot captures the raw widget state; while it equals the state right
/// after [`Binding::load`], the originally loaded value is returned unchanged.
#[derive(Debug, Clone)]
struct Binding {
    key: &'static str,
    /// The checkbox text (other widgets have a row label instead).
    text: String,
    tooltip: String,
    editor: Editor,
    loaded: Value,
    loaded_snapshot: Value,
}

impl Binding {
    fn load(&mut self, value: &Value) {
        self.editor.write(value);
        self.loaded = value.clone();
        self.loaded_snapshot = self.editor.snapshot();
    }

    fn modified(&self) -> bool {
        self.editor.snapshot() != self.loaded_snapshot
    }

    fn value(&self) -> Value {
        if self.modified() {
            self.editor.read()
        } else {
            self.loaded.clone()
        }
    }
}

// ======================================================================= the dialog
/// What the dialog would have called on the controller or emitted.
#[derive(Debug, Clone, PartialEq)]
pub enum SettingsAction {
    /// `Controller::apply_settings` with the edited copy (also the `applied` signal).
    Apply(Box<Settings>),
    /// `Controller::suspend_hotkeys`: released while a shortcut is being recorded.
    SuspendHotkeys(bool),
    /// Collect the diagnostics report ([`load_diagnostics_text`] with the controller's hotkey
    /// manager) and hand it back with [`SettingsDialog::set_diagnostics`]. Comes after any
    /// [`SettingsAction::SuspendHotkeys`] that gives the hotkeys back, so the report sees them.
    CollectDiagnostics,
    /// The user pressed "Recalibrate…".
    CalibrationRequested,
    /// The user pressed "Run setup assistant…" (the app opens the first-run assistant).
    SetupRequested,
    /// The dialog closed (OK: `accepted`; Cancel or the window's close button: not).
    Closed { accepted: bool },
}

/// What the dialog reads from the controller while it is shown.
#[derive(Default)]
pub struct SettingsInputs<'a> {
    /// `Controller::calibration()`.
    pub calibration: Option<&'a CalibrationData>,
    /// `Controller::is_calibrated`: a stored calibration may no longer fit.
    pub is_calibrated: bool,
    pub calibration_reason: &'a str,
    /// `Controller::backend_info()`: (name, version).
    pub backend_info: Option<(&'a str, &'a str)>,
}

/// A row of a settings group.
#[derive(Debug, Clone)]
enum Row {
    /// A bound widget; `label` is the row label (checkboxes carry their own text).
    Field {
        key: &'static str,
        label: Option<&'static str>,
    },
    /// Width × height.
    Resolution,
    /// A muted explanation, disabled with `follows` when given.
    Hint {
        text: String,
        follows: Option<&'static str>,
    },
    Autostart,
    AutostartNote,
    SetupButton,
    UpdatesHint,
    Calibration,
    ActionNote,
    CameraNote,
    Stats,
    BackendLabel,
    HotkeyError,
    HotkeyWarning,
}

#[derive(Debug, Clone)]
enum Item {
    Group {
        title: &'static str,
        rows: Vec<Row>,
    },
    /// Rich text (`<b>`, `<br>`, `<code>`) in a tinted frame.
    Banner(String),
}

/// All user settings, grouped into pages with a sidebar.
pub struct SettingsDialog {
    caps: HashMap<&'static str, bool>,
    hotkey_manager: Option<Arc<dyn HotkeyManager>>,
    autostart: Box<dyn AutostartBackend>,
    /// Updates can be checked on this system (`update::fetch::supported`).
    pub(crate) update_supported: bool,
    base: Settings,
    bindings: Vec<Binding>,
    /// Master checkbox key → the keys it enables.
    depends: Vec<(&'static str, Vec<&'static str>)>,
    pages: Vec<Vec<Item>>,
    page: usize,
    hotkey_errors: Vec<(&'static str, String)>,
    hotkey_warnings: Vec<(&'static str, String)>,
    /// `layout_conflict()` answers by canonical hotkey text.
    conflicts: HashMap<String, Option<String>>,
    /// The footer shows a hotkey error (cleared once the shortcuts are valid).
    status_shows_hotkey_error: bool,
    status: String,
    status_is_error: bool,
    diagnostics_loaded: bool,
    diagnostics: String,
    /// "Copy report" waits for the report it asked for.
    copy_pending: bool,
    /// Text for the clipboard, put there by the next frame.
    clipboard: Option<String>,
    loading: bool,
    /// This dialog asked the controller to release the global hotkeys.
    hotkeys_suspended: bool,
    /// A shortcut field stopped recording: give the hotkeys back once focus settled.
    resume_due: bool,
    /// The Win/Super key is held (egui reports it as a key, not a modifier, outside macOS).
    super_held: bool,
    autostart_supported: bool,
    autostart_checked: bool,
    /// What the OS has (read at start, updated only by a successful change).
    autostart_initial: bool,
    autostart_note: String,
    camera_probe: CameraProbe,
    /// Camera indices left out of the running probe (the one in use).
    probe_skip: BTreeSet<u32>,
    detecting: bool,
    camera_note: Option<String>,
    stats: String,
    hotkey_banner: Option<String>,
    file_dialog: Option<egui_file_dialog::FileDialog>,
    actions: Vec<SettingsAction>,
}

impl SettingsDialog {
    /// The dialog for the controller's current `settings`.
    ///
    /// `caps` are the platform's capabilities; `hotkey_manager` is the controller's (it knows
    /// best whether hotkeys work and which ones a keyboard layout would swallow).
    pub fn new(
        settings: &Settings,
        caps: HashMap<&'static str, bool>,
        hotkey_manager: Option<Arc<dyn HotkeyManager>>,
        autostart: Box<dyn AutostartBackend>,
    ) -> Self {
        let autostart_supported = autostart.is_supported();
        let mut dialog = Self {
            caps,
            hotkey_manager,
            autostart,
            update_supported: crate::update::fetch::supported(),
            base: settings.clone(),
            bindings: Vec::new(),
            depends: Vec::new(),
            pages: Vec::new(),
            page: 0,
            hotkey_errors: Vec::new(),
            hotkey_warnings: Vec::new(),
            conflicts: HashMap::new(),
            status_shows_hotkey_error: false,
            status: String::new(),
            status_is_error: false,
            diagnostics_loaded: false,
            diagnostics: String::new(),
            copy_pending: false,
            clipboard: None,
            loading: false,
            hotkeys_suspended: false,
            resume_due: false,
            super_held: false,
            autostart_supported,
            autostart_checked: false,
            autostart_initial: false,
            autostart_note: String::new(),
            camera_probe: CameraProbe::default(),
            probe_skip: BTreeSet::new(),
            detecting: false,
            camera_note: None,
            stats: "Waiting for statistics…".into(),
            hotkey_banner: None,
            file_dialog: None,
            actions: Vec::new(),
        };
        dialog.build();
        dialog.autostart_initial = dialog.read_autostart();
        dialog.autostart_checked = dialog.autostart_initial;
        dialog.refresh_autostart_note();
        dialog.load(settings);
        dialog
    }

    /// Probe cameras with `probe` instead of [`probe_cameras`] (tests).
    pub fn with_camera_probe(mut self, probe: CameraProbe) -> Self {
        self.camera_probe = probe;
        self
    }

    fn cap(&self, name: &str) -> bool {
        self.caps.get(name).copied().unwrap_or(true)
    }

    // ============================================================== public API
    /// The settings as currently shown (a new object; unexposed fields preserved).
    pub fn settings(&self) -> Settings {
        let mut out = self.base.to_dict();
        for binding in &self.bindings {
            set(&mut out, binding.key, binding.value());
        }
        Settings::from_dict(&out)
    }

    /// Whether anything differs from the last loaded/applied state.
    pub fn is_modified(&self) -> bool {
        self.autostart_checked != self.autostart_initial
            || self.bindings.iter().any(Binding::modified)
    }

    pub fn is_valid(&self) -> bool {
        self.hotkey_errors.is_empty()
    }

    /// Whether OK is enabled.
    pub fn ok_enabled(&self) -> bool {
        self.is_valid()
    }

    /// Whether Apply is enabled.
    pub fn apply_enabled(&self) -> bool {
        self.is_valid() && self.is_modified()
    }

    /// Every `section.field` key that has a widget.
    pub fn bound_keys(&self) -> Vec<&'static str> {
        self.bindings.iter().map(|b| b.key).collect()
    }

    fn binding(&self, key: &str) -> Option<&Binding> {
        self.bindings.iter().find(|b| b.key == key)
    }

    fn binding_mut(&mut self, key: &str) -> Option<&mut Binding> {
        self.bindings.iter_mut().find(|b| b.key == key)
    }

    /// The tooltip of the widget bound to `key`.
    pub fn tooltip(&self, key: &str) -> &str {
        self.binding(key).map_or("", |b| b.tooltip.as_str())
    }

    /// The checkbox text of the widget bound to `key`.
    pub fn text(&self, key: &str) -> &str {
        self.binding(key).map_or("", |b| b.text.as_str())
    }

    /// Whether the widget bound to `key` can be edited (its master checkboxes are ticked).
    pub fn is_enabled(&self, key: &str) -> bool {
        if key == "updates.check" && !self.update_supported {
            return false;
        }
        self.depends
            .iter()
            .filter(|(_, keys)| keys.contains(&key))
            .all(|(master, _)| self.checked(master))
    }

    /// Whether the checkbox bound to `key` is ticked.
    pub fn checked(&self, key: &str) -> bool {
        matches!(
            self.binding(key).map(|b| &b.editor),
            Some(Editor::Check { checked: true })
        )
    }

    /// Tick or clear the checkbox bound to `key`.
    pub fn set_checked(&mut self, key: &str, value: bool) {
        if let Some(Editor::Check { checked }) = self.binding_mut(key).map(|b| &mut b.editor) {
            *checked = value;
        }
        self.on_changed();
    }

    /// The number a spin box or slider bound to `key` shows (in display units).
    pub fn number(&self, key: &str) -> f64 {
        match self.binding(key).map(|b| &b.editor) {
            Some(Editor::Int { value, .. } | Editor::Slider { value, .. }) => *value as f64,
            Some(Editor::Float { value, .. }) => *value,
            _ => 0.0,
        }
    }

    /// Set the number a spin box or slider bound to `key` shows (in display units, clamped).
    pub fn set_number(&mut self, key: &str, number: f64) {
        match self.binding_mut(key).map(|b| &mut b.editor) {
            Some(Editor::Int { value, lo, hi, .. }) => *value = (number as i64).clamp(*lo, *hi),
            Some(Editor::Slider { value, .. }) => *value = (number as i64).clamp(0, 100),
            Some(Editor::Float {
                value,
                lo,
                hi,
                decimals,
                ..
            }) => *value = round_to(number.clamp(*lo, *hi), *decimals),
            _ => {}
        }
        self.on_changed();
    }

    /// Replace the text of the multi-line field bound to `key`.
    pub fn set_text(&mut self, key: &str, value: &str) {
        if let Some(Editor::Lines { text, .. }) = self.binding_mut(key).map(|b| &mut b.editor) {
            *text = value.to_string();
        }
        self.on_changed();
    }

    /// Select the item with `value` in the list bound to `key` (a combo box or the device).
    pub fn select(&mut self, key: &str, value: &str) {
        match self.binding_mut(key).map(|b| &mut b.editor) {
            Some(Editor::Combo { options, index }) => {
                if let Some(found) = options.iter().position(|(v, _)| v == value) {
                    *index = found;
                }
            }
            Some(editor @ Editor::Device { .. }) => {
                let Editor::Device { items, .. } = &*editor else {
                    return;
                };
                if items.iter().any(|(_, v)| v == value) {
                    editor.select_device(value);
                }
            }
            _ => {}
        }
        self.on_changed();
    }

    /// (value, label) of every item of the list bound to `key`.
    pub fn items(&self, key: &str) -> Vec<(String, String)> {
        match self.binding(key).map(|b| &b.editor) {
            Some(Editor::Combo { options, .. }) => options.clone(),
            Some(Editor::Device { items, .. }) => {
                items.iter().map(|(l, v)| (v.clone(), l.clone())).collect()
            }
            _ => Vec::new(),
        }
    }

    /// (value, label) of the selected item of the list bound to `key`.
    pub fn current(&self, key: &str) -> (String, String) {
        match self.binding(key).map(|b| &b.editor) {
            Some(Editor::Combo { options, index }) => {
                options.get(*index).cloned().unwrap_or_default()
            }
            Some(editor @ Editor::Device { text, .. }) => (editor.device_value(), text.clone()),
            _ => Default::default(),
        }
    }

    pub fn hotkey_edit(&self, action: &str) -> Option<&HotkeyEdit> {
        match self
            .binding(&format!("hotkeys.{action}"))
            .map(|b| &b.editor)
        {
            Some(Editor::Hotkey(edit)) => Some(edit),
            _ => None,
        }
    }

    fn hotkey_edit_mut(&mut self, action: &str) -> Option<&mut HotkeyEdit> {
        let key = format!("hotkeys.{action}");
        match self.binding_mut(&key).map(|b| &mut b.editor) {
            Some(Editor::Hotkey(edit)) => Some(edit),
            _ => None,
        }
    }

    /// Set a shortcut field (canonicalised when valid), as typing it would.
    pub fn set_hotkey(&mut self, action: &str, text: &str) {
        if self
            .hotkey_edit_mut(action)
            .is_some_and(|edit| edit.set_hotkey(text))
        {
            self.on_changed();
        }
    }

    /// The shortcut errors by action, in the page's order.
    pub fn hotkey_errors(&self) -> &[(&'static str, String)] {
        &self.hotkey_errors
    }

    /// The shortcuts the OS will refuse (they type a character), by action.
    pub fn hotkey_warnings(&self) -> &[(&'static str, String)] {
        &self.hotkey_warnings
    }

    pub fn stats_text(&self) -> &str {
        &self.stats
    }

    /// The footer's status line.
    pub fn status_text(&self) -> &str {
        &self.status
    }

    /// "Start at login" is ticked.
    pub fn autostart_checked(&self) -> bool {
        self.autostart_checked
    }

    pub fn set_autostart_checked(&mut self, checked: bool) {
        self.autostart_checked = checked;
        self.on_changed();
    }

    /// Whether "Start at login" can be changed here.
    pub fn autostart_enabled(&self) -> bool {
        self.autostart_supported
    }

    /// The explanation under "Start at login" (empty: hidden).
    pub fn autostart_note_text(&self) -> &str {
        &self.autostart_note
    }

    /// The result line of the last camera detection.
    pub fn camera_note(&self) -> Option<&str> {
        self.camera_note.as_deref()
    }

    /// Whether "Detect cameras" can be pressed (no detection running).
    pub fn detect_enabled(&self) -> bool {
        !self.detecting
    }

    /// The banner of the Hotkeys page (rich text), if hotkeys are limited here.
    pub fn hotkey_banner(&self) -> Option<&str> {
        self.hotkey_banner.as_deref()
    }

    /// Every label and banner text of every page (rich text as written).
    pub fn texts(&self) -> Vec<String> {
        let mut out = Vec::new();
        for item in self.pages.iter().flatten() {
            match item {
                Item::Banner(text) => out.push(text.clone()),
                Item::Group { rows, .. } => {
                    for row in rows {
                        if let Row::Hint { text, .. } = row {
                            out.push(text.clone());
                        }
                    }
                }
            }
        }
        out
    }

    /// Switch to a page by key (`"general"`, `"hotkeys"`, …).
    pub fn show_page(&mut self, key: &str) {
        if let Some(row) = PAGES.iter().position(|(k, _, _)| *k == key) {
            self.on_page_changed(row);
        }
    }

    pub fn current_page(&self) -> &'static str {
        PAGES[self.page].0
    }

    /// Validate and hand the settings to the controller. Returns `false` if invalid.
    pub fn apply(&mut self) -> bool {
        if !self.is_valid() {
            self.show_hotkey_error();
            return false;
        }
        let new = self.settings();
        if new.to_dict() != self.base.to_dict() {
            self.actions
                .push(SettingsAction::Apply(Box::new(new.clone())));
        }
        // The new state becomes the baseline for "modified"; values the controller adjusted
        // (clamped) arrive with its settings_changed and refresh the untouched widgets.
        let autostart_ok = self.apply_autostart();
        self.load(&new);
        if autostart_ok {
            self.set_status("", false);
        }
        autostart_ok
    }

    /// Show default values (still to be applied).
    ///
    /// Settings without a widget, such as the first-run flag, keep their value.
    pub fn restore_defaults(&mut self) {
        let defaults = Settings::default().to_dict();
        for binding in &mut self.bindings {
            binding.editor.write(&get(&defaults, binding.key));
        }
        self.on_changed();
    }

    /// OK: apply, and close unless that failed.
    pub fn accept(&mut self) -> bool {
        if !self.apply() {
            return false;
        }
        self.close(true);
        true
    }

    /// Cancel or the window's close button.
    pub fn reject(&mut self) {
        self.close(false);
    }

    fn close(&mut self, accepted: bool) {
        // Closed while a shortcut field had focus: give the hotkeys back now; the deferred
        // resume would never run once the dialog is dropped.
        for binding in &mut self.bindings {
            if let Editor::Hotkey(edit) = &mut binding.editor {
                edit.recording = false;
                edit.preview = None;
            }
        }
        self.resume_due = false;
        self.resume_hotkeys();
        self.actions.push(SettingsAction::Closed { accepted });
    }

    /// The actions collected since the last call.
    pub fn take_actions(&mut self) -> Vec<SettingsAction> {
        std::mem::take(&mut self.actions)
    }

    // ============================================================== controller events
    /// `Controller.stats_changed`.
    pub fn on_stats(&mut self, stats: &Value) {
        let text = format_stats(stats);
        if !text.is_empty() {
            self.stats = text;
        }
    }

    /// Settings changed elsewhere: refresh the widgets the user has not touched.
    pub fn on_settings_changed(&mut self, settings: &Settings) {
        self.loading = true;
        self.base = settings.clone();
        let dict = settings.to_dict();
        for binding in &mut self.bindings {
            if !binding.modified() {
                binding.load(&get(&dict, binding.key));
            }
        }
        self.loading = false;
        self.on_changed();
    }

    // ============================================================== building
    fn register(&mut self, key: &'static str, text: &str, tooltip: Option<String>, editor: Editor) {
        let mut binding = Binding {
            key,
            text: text.to_string(),
            tooltip: tooltip.unwrap_or_else(|| doc(key).to_string()),
            editor,
            loaded: Value::Null,
            loaded_snapshot: Value::Null,
        };
        binding.loaded_snapshot = binding.editor.snapshot();
        self.bindings.push(binding);
    }

    fn check(&mut self, rows: &mut Vec<Row>, key: &'static str, text: &str) -> &'static str {
        self.register(key, text, None, Editor::Check { checked: false });
        rows.push(Row::Field { key, label: None });
        key
    }

    /// Append `(not supported on this system)` to a checkbox the platform cannot serve.
    fn mark_unsupported(&mut self, key: &str, capability: &str) {
        if !self.cap(capability)
            && let Some(binding) = self.binding_mut(key)
        {
            binding.text += " (not supported on this system)";
        }
    }

    fn combo(
        &mut self,
        rows: &mut Vec<Row>,
        key: &'static str,
        label: &'static str,
        options: &[(&str, &str)],
    ) -> &'static str {
        let options: Vec<(String, String)> = options
            .iter()
            .map(|(v, t)| ((*v).to_string(), (*t).to_string()))
            .collect();
        self.register(key, "", None, Editor::Combo { options, index: 0 });
        rows.push(Row::Field {
            key,
            label: Some(label),
        });
        key
    }

    fn int(&mut self, key: &'static str, suffix: &'static str, step: i64) -> Editor {
        let row = meta(key);
        Editor::Int {
            value: 0,
            lo: row.and_then(|r| r.lo).map_or(0, |v| v as i64),
            hi: row.and_then(|r| r.hi).map_or(1_000_000, |v| v as i64),
            step,
            suffix,
        }
    }

    fn int_row(
        &mut self,
        rows: &mut Vec<Row>,
        key: &'static str,
        label: &'static str,
        suffix: &'static str,
        step: i64,
    ) -> &'static str {
        let editor = self.int(key, suffix, step);
        self.register(key, "", None, editor);
        rows.push(Row::Field {
            key,
            label: Some(label),
        });
        key
    }

    #[allow(clippy::too_many_arguments)]
    fn float(
        &mut self,
        rows: &mut Vec<Row>,
        key: &'static str,
        label: &'static str,
        factor: f64,
        decimals: usize,
        step: f64,
        suffix: &'static str,
    ) -> &'static str {
        let row = meta(key);
        let editor = Editor::Float {
            value: 0.0,
            lo: row.and_then(|r| r.lo).map_or(0.0, |v| v * factor),
            hi: row.and_then(|r| r.hi).map_or(1e9, |v| v * factor),
            factor,
            decimals,
            step,
            suffix,
        };
        self.register(key, "", None, editor);
        rows.push(Row::Field {
            key,
            label: Some(label),
        });
        key
    }

    fn slider(
        &mut self,
        rows: &mut Vec<Row>,
        key: &'static str,
        label: &'static str,
        left: &'static str,
        right: &'static str,
    ) -> &'static str {
        let editor = Editor::Slider {
            value: 0,
            left,
            right,
        };
        self.register(key, "", None, editor);
        rows.push(Row::Field {
            key,
            label: Some(label),
        });
        key
    }

    fn lines(
        &mut self,
        rows: &mut Vec<Row>,
        key: &'static str,
        label: &'static str,
        placeholder: &'static str,
    ) {
        let editor = Editor::Lines {
            text: String::new(),
            placeholder,
        };
        self.register(key, "", None, editor);
        rows.push(Row::Field {
            key,
            label: Some(label),
        });
    }

    fn depends(&mut self, master: &'static str, keys: Vec<&'static str>) {
        self.depends.push((master, keys));
    }

    fn hint(text: impl Into<String>) -> Row {
        Row::Hint {
            text: text.into(),
            follows: None,
        }
    }

    fn build(&mut self) {
        let general = self.build_general();
        let switching = self.build_switching();
        let presence = self.build_presence();
        let camera = self.build_camera();
        let hotkeys = self.build_hotkeys();
        self.pages = vec![general, switching, presence, camera, hotkeys, Vec::new()];
    }

    fn build_general(&mut self) -> Vec<Item> {
        let mut items = Vec::new();
        let mut rows = vec![Row::Autostart, Row::AutostartNote];
        self.check(
            &mut rows,
            "general.start_paused",
            "Start with tracking paused",
        );
        // Opens the assistant now. A "show it at next start" switch would have
        // to clear general.first_run_done, and until the assistant is finished
        // that flag also turns the walk-away lock into a mere notification.
        rows.push(Row::SetupButton);
        items.push(Item::Group {
            title: "Startup",
            rows,
        });

        let mut rows = Vec::new();
        self.check(&mut rows, "general.notifications", "Show notifications");
        self.check(
            &mut rows,
            "ui.show_gaze_overlay",
            "Show a dot where I am looking (testing aid)",
        );
        items.push(Item::Group {
            title: "Interface",
            rows,
        });

        let mut rows = Vec::new();
        self.check(
            &mut rows,
            "updates.check",
            "Look for a new version once a day",
        );
        rows.push(Row::UpdatesHint);
        items.push(Item::Group {
            title: "Updates",
            rows,
        });

        items.push(Item::Group {
            title: "Calibration",
            rows: vec![Row::Calibration],
        });

        let mut rows = Vec::new();
        let available = crate::vision::backends::available_backends();
        let options: Vec<(&str, String)> = BACKEND_LABELS
            .iter()
            .map(|&(value, label)| {
                let missing = value != "auto" && !available.contains(&value);
                let text = if missing {
                    format!("{label} — not available")
                } else {
                    label.to_string()
                };
                (value, text)
            })
            .collect();
        let options: Vec<(&str, &str)> = options.iter().map(|(v, t)| (*v, t.as_str())).collect();
        self.combo(&mut rows, "general.backend", "Vision backend", &options);
        rows.push(Self::hint(
            "Face mesh follows your head and your eyes; Lite follows only the head, \
             for slow computers. Changing the backend requires a new calibration.",
        ));
        self.combo(
            &mut rows,
            "general.log_level",
            "Log level",
            &[
                ("DEBUG", "Debug"),
                ("INFO", "Info"),
                ("WARNING", "Warning"),
                ("ERROR", "Error"),
            ],
        );
        items.push(Item::Group {
            title: "Advanced",
            rows,
        });
        items
    }

    fn build_switching(&mut self) -> Vec<Item> {
        let mut items = Vec::new();
        let mut rows = Vec::new();
        let enabled = self.check(
            &mut rows,
            "switching.enabled",
            "Move the cursor to the monitor I look at",
        );
        let target = self.combo(
            &mut rows,
            "switching.cursor_target",
            "Cursor lands",
            &[
                ("last", "Where it was last on that monitor"),
                ("center", "In the centre of the monitor"),
                ("gaze", "Where I am looking (estimated)"),
            ],
        );
        let focus = self.check(
            &mut rows,
            "switching.focus_window",
            "Also give keyboard focus to that monitor",
        );
        self.mark_unsupported(focus, "focus");
        items.push(Item::Group {
            title: "Switching",
            rows,
        });

        let mut rows = Vec::new();
        let dwell = self.int_row(
            &mut rows,
            "switching.dwell_ms",
            "Look for at least",
            " ms",
            50,
        );
        let cooldown = self.int_row(
            &mut rows,
            "switching.cooldown_ms",
            "Pause between switches",
            " ms",
            100,
        );
        let mouse = self.int_row(
            &mut rows,
            "switching.mouse_grace_ms",
            "Wait after mouse use",
            " ms",
            250,
        );
        let typing = self.int_row(
            &mut rows,
            "switching.typing_grace_ms",
            "Wait after typing",
            " ms",
            250,
        );
        let reading = self.int_row(
            &mut rows,
            "switching.reading_grace_ms",
            "… while reading another monitor",
            " ms",
            500,
        );
        items.push(Item::Group {
            title: "Timing",
            rows,
        });

        let mut rows = Vec::new();
        let smoothing = self.slider(
            &mut rows,
            "switching.smoothing",
            "Smoothing",
            "Responsive",
            "Steady",
        );
        let hysteresis = self.float(
            &mut rows,
            "switching.hysteresis",
            "Cross into a monitor by",
            100.0,
            0,
            1.0,
            " % of its size",
        );
        let margin = self.float(
            &mut rows,
            "switching.off_screen_margin",
            "Ignore gaze beyond",
            100.0,
            0,
            5.0,
            " % of the diagonal",
        );
        items.push(Item::Group {
            title: "Sensitivity",
            rows,
        });
        self.depends(
            enabled,
            vec![
                target, focus, dwell, cooldown, mouse, typing, reading, smoothing, hysteresis,
                margin,
            ],
        );

        let mut rows = Vec::new();
        let follow = self.check(
            &mut rows,
            "panes.enabled",
            "Also move keyboard focus to the split pane I look at",
        );
        self.mark_unsupported(follow, "panes");
        let move_cursor = self.check(
            &mut rows,
            "panes.move_cursor",
            "Also move the cursor into that pane, where I left it",
        );
        let precision = self.float(
            &mut rows,
            "panes.precision",
            "Only panes at least",
            1.0,
            1,
            0.5,
            " × my gaze error",
        );
        rows.push(Self::hint(
            "Works with tmux, WezTerm and Windows Terminal panes on the monitor you are on. \
             Small panes are left alone: how small depends on how accurate your calibration \
             is.",
        ));
        let desktop = self.check(
            &mut rows,
            "panes.desktop_apps",
            "Also switch between side-by-side sessions in the Claude and ChatGPT apps",
        );
        rows.push(Row::Hint {
            text: "Two Claude chats, or a ChatGPT or Codex conversation and its side chat: the \
                   one you look at gets the focus in its message box. Windows only. Asks the app \
                   for its accessibility tree, which can cost the app some CPU and memory."
                .into(),
            follows: Some(desktop),
        });
        items.push(Item::Group {
            title: "Split panes (experimental)",
            rows,
        });
        self.depends(follow, vec![move_cursor, precision, desktop]);

        let mut rows = Vec::new();
        let adaptive = self.check(
            &mut rows,
            "learning.adaptive",
            "Learn from how I use the mouse",
        );
        let max_samples = self.int_row(
            &mut rows,
            "learning.max_samples",
            "Keep at most",
            " samples",
            50,
        );
        self.depends(adaptive, vec![max_samples]);
        self.check(
            &mut rows,
            "learning.drift_alerts",
            "Suggest recalibrating when accuracy drops",
        );
        items.push(Item::Group {
            title: "Adaptive accuracy",
            rows,
        });
        items
    }

    fn build_presence(&mut self) -> Vec<Item> {
        let mut items = vec![Item::Banner(
            "<b>Everything stays on this computer.</b> Camera frames are analysed in memory \
             and never saved or sent anywhere. Privacy mode turns the camera off completely."
                .into(),
        )];
        let mut rows = Vec::new();
        let enabled = self.check(
            &mut rows,
            "presence.enabled",
            "React when I leave the computer",
        );
        let action = self.combo(
            &mut rows,
            "presence.action",
            "Then",
            &[
                ("lock", "Lock the computer"),
                ("lock_and_display_off", "Lock and turn the displays off"),
                ("display_off", "Turn the displays off"),
                ("notify", "Only show a notification"),
                ("none", "Do nothing"),
            ],
        );
        rows.push(Row::ActionNote);
        let timeout = self.int_row(&mut rows, "presence.away_timeout_s", "After", " s away", 5);
        let warning = self.int_row(&mut rows, "presence.warning_s", "Countdown", " s", 1);
        let idle = self.check(
            &mut rows,
            "presence.require_input_idle",
            "Keyboard or mouse use counts as being here",
        );
        // Wayland outside GNOME: only the camera can cancel the countdown.
        self.mark_unsupported(idle, "input_idle");
        let wake = self.check(
            &mut rows,
            "presence.wake_on_return",
            "Turn the displays back on when I return",
        );
        self.depends(enabled, vec![action, timeout, warning, idle, wake]);
        items.push(Item::Group {
            title: "When I walk away",
            rows,
        });

        let mut rows = Vec::new();
        self.check(
            &mut rows,
            "privacy.remember_privacy_mode",
            "Keep privacy mode on after a restart (the camera stays off)",
        );
        self.check(
            &mut rows,
            "privacy.pause_when_locked",
            "Release the camera while the screen is locked",
        );
        let yield_box = self.check(
            &mut rows,
            "privacy.yield_camera",
            "Release the camera while another app uses it",
        );
        self.mark_unsupported(yield_box, "camera_in_use");
        self.lines(
            &mut rows,
            "privacy.pause_for_apps",
            "Pause while these apps run",
            "One program per line, e.g.\nzoom.exe\nobs64.exe",
        );
        items.push(Item::Group {
            title: "Camera privacy",
            rows,
        });

        let mut rows = Vec::new();
        let guard = self.check(
            &mut rows,
            "privacy.shoulder_guard",
            "React when someone looks over my shoulder",
        );
        let guard_action = self.combo(
            &mut rows,
            "privacy.guard_action",
            "Then",
            &[
                ("curtain", "Cover the screens"),
                ("notify", "Only show a notification"),
                ("lock", "Lock the computer"),
            ],
        );
        let delay = self.float(
            &mut rows,
            "privacy.guard_delay_s",
            "After",
            1.0,
            1,
            0.5,
            " s",
        );
        self.depends(guard, vec![guard_action, delay]);
        items.push(Item::Group {
            title: "Shoulder guard",
            rows,
        });
        items
    }

    fn build_camera(&mut self) -> Vec<Item> {
        let mut items = Vec::new();
        let mut rows = Vec::new();
        self.register(
            "camera.device",
            "",
            None,
            Editor::Device {
                items: Vec::new(),
                index: None,
                text: String::new(),
            },
        );
        rows.push(Row::Field {
            key: "camera.device",
            label: Some("Device"),
        });
        let width = self.int("camera.width", " px", 160);
        self.register("camera.width", "", None, width);
        let height = self.int("camera.height", " px", 120);
        self.register("camera.height", "", None, height);
        rows.push(Row::Resolution);
        rows.push(Self::hint(
            "640 × 480 is plenty: frames are scaled down before analysis.",
        ));
        self.combo(
            &mut rows,
            "camera.api",
            "Capture API",
            &[
                ("auto", "Automatic"),
                ("dshow", "DirectShow (Windows)"),
                ("msmf", "Media Foundation (Windows)"),
                ("avfoundation", "AVFoundation (macOS)"),
                ("v4l2", "Video4Linux2 (Linux)"),
                ("any", "Any available"),
            ],
        );
        rows.push(Row::CameraNote);
        items.push(Item::Group {
            title: "Camera",
            rows,
        });

        let mut rows = Vec::new();
        let profile_text = [
            ("eco", "Eco — lowest CPU"),
            ("balanced", "Balanced (recommended)"),
            ("responsive", "Responsive — fastest switching"),
        ];
        let options: Vec<(&str, String)> = profile_text
            .iter()
            .map(|&(value, text)| {
                let fps = PROFILES
                    .iter()
                    .find(|(name, _)| *name == value)
                    .map(|(_, rates)| rates.active);
                match fps {
                    Some(fps) if fps != 0.0 => (value, format!("{text}, up to {fps} fps")),
                    _ => (value, text.to_string()),
                }
            })
            .collect();
        let options: Vec<(&str, &str)> = options.iter().map(|(v, t)| (*v, t.as_str())).collect();
        self.combo(&mut rows, "performance.profile", "Profile", &options);
        let gate = self.check(
            &mut rows,
            "performance.motion_gate",
            "Skip analysis while the picture is still",
        );
        let threshold = self.float(
            &mut rows,
            "performance.motion_threshold",
            "Motion threshold",
            1.0,
            1,
            0.5,
            "",
        );
        self.depends(gate, vec![threshold]);
        items.push(Item::Group {
            title: "Performance",
            rows,
        });

        items.push(Item::Group {
            title: "Live",
            rows: vec![Row::Stats, Row::BackendLabel],
        });
        items
    }

    fn build_hotkeys(&mut self) -> Vec<Item> {
        let mut items = Vec::new();
        let mut rows = Vec::new();
        let enabled = self.check(&mut rows, "hotkeys.enabled", "Enable global shortcuts");
        let mut edits = Vec::new();
        for (action, label) in HOTKEY_ACTIONS {
            let key = match action {
                "toggle_tracking" => "hotkeys.toggle_tracking",
                "toggle_privacy" => "hotkeys.toggle_privacy",
                _ => "hotkeys.recalibrate",
            };
            let tooltip = format!(
                "{}\nClick, then press the shortcut. Backspace clears it.",
                doc(key)
            );
            self.register(
                key,
                "",
                Some(tooltip),
                Editor::Hotkey(HotkeyEdit::default()),
            );
            rows.push(Row::Field {
                key,
                label: Some(label),
            });
            edits.push(key);
        }
        rows.push(Row::HotkeyError);
        // Valid shortcuts the OS will refuse (they type a character, e.g. AltGr).
        rows.push(Row::HotkeyWarning);
        self.depends(enabled, edits);
        let meta_key = if cfg!(target_os = "macos") {
            "Cmd"
        } else if cfg!(windows) {
            "Win"
        } else {
            "Super"
        };
        rows.push(Self::hint(format!(
            "Shortcuts need at least one of Ctrl, Alt or {meta_key}. Shift alone only works \
             with function keys."
        )));
        items.push(Item::Group {
            title: "Global shortcuts",
            rows,
        });
        // The controller's manager knows best (e.g. XWayland limits); without one fall back
        // to the platform's capability matrix.
        let supported = self
            .hotkey_manager
            .as_ref()
            .map_or_else(|| self.cap("hotkeys"), |m| m.supported());
        let note = self.hotkey_manager.as_ref().and_then(|m| m.note());
        self.hotkey_banner = hotkey_banner(supported, note);
        if let Some(banner) = &self.hotkey_banner {
            items.push(Item::Banner(banner.clone()));
        }
        items
    }

    // ============================================================== loading
    fn load(&mut self, settings: &Settings) {
        self.loading = true;
        self.base = settings.clone();
        let dict = settings.to_dict();
        for binding in &mut self.bindings {
            binding.load(&get(&dict, binding.key));
        }
        // The start-at-login baseline is not reset here: it is what the OS
        // has (read at start, updated only by a successful change), so a
        // change that failed stays pending and OK/Apply retry it.
        self.loading = false;
        self.on_changed();
    }

    // ============================================================== slots
    fn on_changed(&mut self) {
        if self.loading {
            return;
        }
        self.validate_hotkeys();
        // The error label is on the Hotkeys page; the footer says why OK is off
        // whichever page is shown.
        if !self.is_valid() {
            self.show_hotkey_error();
        } else if self.status_shows_hotkey_error {
            self.set_status("", false);
        }
    }

    /// Put the first shortcut error in the footer (cleared once all are valid).
    fn show_hotkey_error(&mut self) {
        let text = self.hotkey_errors.first().map_or_else(
            || "Fix the highlighted shortcut first.".to_string(),
            |(_, e)| e.clone(),
        );
        self.set_status(&text, true);
        self.status_shows_hotkey_error = true;
    }

    fn set_status(&mut self, text: &str, error: bool) {
        self.status_shows_hotkey_error = false;
        self.status = text.to_string();
        self.status_is_error = error;
    }

    fn on_page_changed(&mut self, row: usize) {
        self.page = row;
        if PAGES[row].0 == "diagnostics" && !self.diagnostics_loaded {
            self.refresh_diagnostics();
        }
    }

    // ============================================================== hotkeys
    /// Start or stop recording in a shortcut field (it gained or lost the focus).
    pub fn set_recording(&mut self, action: &str, recording: bool) {
        let Some(edit) = self.hotkey_edit_mut(action) else {
            return;
        };
        if edit.recording == recording {
            return;
        }
        edit.recording = recording;
        edit.preview = None;
        self.on_hotkey_recording(recording);
    }

    fn on_hotkey_recording(&mut self, recording: bool) {
        // Registered global hotkeys are swallowed by the OS before they reach this
        // dialog, so they are released while a shortcut is being recorded.
        if recording {
            self.suspend_hotkeys(true);
            return;
        }
        // Moving from one shortcut field to the next leaves the first (false)
        // just before entering the second (true). Resuming at once would
        // re-register every hotkey (repeating "Hotkey unavailable" for one that
        // fails) only to release them again, so it waits until focus settled.
        self.resume_due = true;
    }

    /// Run what waited for focus to settle (the next frame; Qt's zero-delay timer).
    pub fn flush_deferred(&mut self) {
        if std::mem::take(&mut self.resume_due) {
            self.resume_hotkeys_if_idle();
        }
    }

    /// Give the hotkeys back unless a shortcut field is recording again.
    fn resume_hotkeys_if_idle(&mut self) {
        let recording = self
            .bindings
            .iter()
            .any(|b| matches!(&b.editor, Editor::Hotkey(edit) if edit.recording));
        if !recording {
            self.resume_hotkeys();
        }
    }

    /// Give the hotkeys back now if this dialog released them.
    fn resume_hotkeys(&mut self) {
        self.suspend_hotkeys(false);
    }

    fn suspend_hotkeys(&mut self, suspended: bool) {
        if suspended == self.hotkeys_suspended {
            return;
        }
        self.hotkeys_suspended = suspended;
        self.actions.push(SettingsAction::SuspendHotkeys(suspended));
    }

    /// A key press in a recording shortcut field. Returns `false` when it ended recording
    /// (Esc), so the field gives up the focus.
    pub fn hotkey_key_press(
        &mut self,
        action: &str,
        key: egui::Key,
        modifiers: HeldModifiers,
    ) -> bool {
        let held = modifiers.names();
        let bare = held.is_empty();
        if bare && key == egui::Key::Escape {
            self.set_recording(action, false);
            return false;
        }
        if bare && matches!(key, egui::Key::Backspace | egui::Key::Delete) {
            self.set_hotkey(action, "");
            return true;
        }
        match hotkey_text_from_key(key, modifiers) {
            // Only modifiers so far: show them as a live preview.
            None => {
                if let Some(edit) = self.hotkey_edit_mut(action) {
                    edit.preview = Some(modifier_preview(&held, None)).filter(|p| !p.is_empty());
                }
            }
            Some(text) => {
                let changed = self
                    .hotkey_edit_mut(action)
                    .is_some_and(|edit| edit.set_hotkey(&text));
                if changed {
                    self.on_changed();
                }
            }
        }
        true
    }

    /// A modifier was released: with none held the field shows its shortcut again.
    fn hotkey_modifiers_released(&mut self, action: &str, modifiers: HeldModifiers) {
        if let Some(edit) = self.hotkey_edit_mut(action) {
            let held = modifiers.names();
            edit.preview = if held.is_empty() {
                None
            } else {
                Some(modifier_preview(&held, None))
            };
        }
    }

    fn validate_hotkeys(&mut self) {
        let mut errors: Vec<(&'static str, String)> = Vec::new();
        let mut warnings: Vec<(&'static str, String)> = Vec::new();
        let mut seen: Vec<(String, &'static str)> = Vec::new();
        let label = |action: &str| {
            HOTKEY_ACTIONS
                .iter()
                .find(|(a, _)| *a == action)
                .map_or("", |(_, l)| *l)
        };
        // With global shortcuts off nothing is registered, and the fields cannot
        // be edited: a bad stored value must not block every other change.
        let active = self.binding("hotkeys.enabled").is_none() || self.checked("hotkeys.enabled");
        for (action, _) in HOTKEY_ACTIONS {
            let text = self
                .hotkey_edit(action)
                .map(|e| e.hotkey().to_string())
                .unwrap_or_default();
            if text.is_empty() || !active {
                continue;
            }
            let canonical = match parse_hotkey(&text) {
                Ok(hk) => hk.to_string(),
                Err(err) => {
                    errors.push((action, format!("{}: {err}", label(action))));
                    continue;
                }
            };
            if let Some((_, first)) = seen.iter().find(|(c, _)| *c == canonical) {
                errors.push((
                    action,
                    format!("{}: same shortcut as {}", label(action), label(first)),
                ));
                continue;
            }
            seen.push((canonical.clone(), action));
            if let Some(conflict) = self.layout_conflict(&canonical) {
                warnings.push((
                    action,
                    format!("{}: {conflict}. Choose another shortcut.", label(action)),
                ));
            }
        }
        for (action, _) in HOTKEY_ACTIONS {
            let invalid = errors.iter().chain(&warnings).any(|(a, _)| *a == action);
            if let Some(edit) = self.hotkey_edit_mut(action) {
                edit.invalid = invalid;
            }
        }
        self.hotkey_errors = errors;
        self.hotkey_warnings = warnings;
    }

    /// Why `hotkey` would swallow a character on an installed keyboard layout.
    ///
    /// Asks the controller's hotkey manager (`layout_conflict`, read-only);
    /// answers are cached for the life of the dialog, since the check reads
    /// every installed layout.
    fn layout_conflict(&mut self, hotkey: &str) -> Option<String> {
        if let Some(cached) = self.conflicts.get(hotkey) {
            return cached.clone();
        }
        let result = self.hotkey_manager.as_ref().and_then(|manager| {
            manager
                .layout_conflict(HotkeyArg::Text(hotkey))
                .unwrap_or_else(|err| {
                    log::debug!("layout_conflict({hotkey}) failed: {err}");
                    None
                })
                .filter(|text| !text.is_empty())
        });
        self.conflicts.insert(hotkey.to_string(), result.clone());
        result
    }

    // ============================================================== camera
    /// Probe connected cameras in the background and fill the device list.
    pub fn detect_cameras(&mut self) {
        let api = self
            .binding("camera.api")
            .map(|b| b.value())
            .and_then(|v| v.as_str().map(String::from))
            .filter(|api| !api.is_empty())
            .unwrap_or_else(|| "auto".into());
        self.probe_skip = camera_in_use(&self.base);
        if self.camera_probe.start(&api, 4, &self.probe_skip) {
            self.detecting = true;
        }
    }

    /// Hand over a finished camera detection (called every frame).
    pub fn poll_cameras(&mut self) {
        if let Some(cameras) = self.camera_probe.poll() {
            self.on_cameras_detected(&cameras);
        }
    }

    fn on_cameras_detected(&mut self, found: &[CameraInfo]) {
        self.detecting = false;
        let skip = self.probe_skip.clone();
        if let Some(binding) = self.binding_mut("camera.device") {
            let current = binding.editor.device_value();
            if let Editor::Device { items, index, .. } = &mut binding.editor {
                items.clear();
                *index = None;
                for info in found {
                    items.push((camera_label(info), info.index.to_string()));
                }
                for index in &skip {
                    // Not probed, but it exists: keep it selectable.
                    let value = index.to_string();
                    if !items.iter().any(|(_, v)| *v == value) {
                        items.push((format!("Camera {index} (in use)"), value));
                    }
                }
            }
            binding.editor.select_device(&current);
        }
        let other = if skip.is_empty() { "" } else { "other " };
        let mut parts = vec![if found.is_empty() {
            format!("No {other}camera found.")
        } else {
            format!("Found {} {other}camera(s).", found.len())
        }];
        if !skip.is_empty() {
            parts.push("The camera in use is not probed.".into());
        }
        self.camera_note = Some(parts.join(" "));
        self.on_changed();
    }

    /// Use a video or image file (from the file picker) as the device.
    pub fn select_device_file(&mut self, path: &Path) {
        let value = path.to_string_lossy().into_owned();
        if let Some(binding) = self.binding_mut("camera.device") {
            binding.editor.select_device(&value);
        }
        self.on_changed();
    }

    // ============================================================== autostart
    fn read_autostart(&self) -> bool {
        self.autostart_supported && self.autostart.is_enabled()
    }

    fn apply_autostart(&mut self) -> bool {
        let wanted = self.autostart_checked;
        if wanted == self.autostart_initial {
            return true;
        }
        let result = if wanted {
            self.autostart.enable(true)
        } else {
            self.autostart.disable()
        };
        if let Err(err) = result {
            // autostart_initial keeps the OS state, so the change stays pending
            // (Apply enabled) and the next OK/Apply tries again.
            log::warn!("Changing start at login failed: {err}");
            self.set_status(&err.to_string(), true);
            return false;
        }
        self.autostart_initial = wanted;
        self.refresh_autostart_note();
        true
    }

    /// Explain an unchecked box whose login item exists but does not start this copy.
    fn refresh_autostart_note(&mut self) {
        let status = if self.autostart_supported {
            self.autostart.status().map_or("", Status::as_str)
        } else {
            ""
        };
        self.autostart_note = autostart_note(status);
    }

    // ============================================================== notes
    /// What the platform cannot do of the chosen walk-away action, if anything.
    pub fn action_note(&self) -> Option<String> {
        let action = self.binding("presence.action")?.editor.read();
        let needs: &[&str] = match action.as_str().unwrap_or("") {
            "lock" => &["lock"],
            "display_off" => &["display_off"],
            "lock_and_display_off" => &["lock", "display_off"],
            _ => &[],
        };
        let missing: Vec<&str> = needs
            .iter()
            .filter(|n| !self.cap(n))
            .map(|n| {
                if *n == "lock" {
                    "locking"
                } else {
                    "turning displays off"
                }
            })
            .collect();
        (!missing.is_empty()).then(|| {
            format!(
                "This system does not support {}; a notification is shown instead.",
                missing.join(" and ")
            )
        })
    }

    fn updates_hint(&self) -> &'static str {
        if self.update_supported {
            "Off by default: then the app never uses the network. It only tells you when there \
             is a new version and installs it when you agree. “Check for updates…” in the tray \
             menu works either way."
        } else {
            "Checking for updates is not available on this system yet: nothing uses the \
             network. Take new versions from the releases page."
        }
    }

    // ============================================================== diagnostics
    /// (Re)build the diagnostics report.
    pub fn refresh_diagnostics(&mut self) {
        self.diagnostics_loaded = true;
        // The report says which hotkeys are registered. Clicking from a shortcut
        // field to this page has only scheduled giving them back (see
        // on_hotkey_recording); do it first, or every hotkey would look refused.
        self.resume_due = false;
        self.resume_hotkeys_if_idle();
        self.actions.push(SettingsAction::CollectDiagnostics);
    }

    /// The report asked for with [`SettingsAction::CollectDiagnostics`], or why it failed.
    pub fn set_diagnostics(&mut self, report: Result<String, String>) {
        self.diagnostics = report.unwrap_or_else(|err| {
            log::warn!("Diagnostics report failed: {err}");
            format!("Could not collect the diagnostics report: {err}")
        });
        if std::mem::take(&mut self.copy_pending) {
            self.copy_diagnostics();
        }
    }

    pub fn diagnostics_text(&self) -> &str {
        &self.diagnostics
    }

    /// "Copy report": the report goes to the clipboard with the next frame.
    pub fn copy_diagnostics(&mut self) {
        if !self.diagnostics_loaded {
            self.copy_pending = true;
            self.refresh_diagnostics();
            return;
        }
        self.clipboard = Some(self.diagnostics.clone());
        self.set_status("Report copied to the clipboard.", false);
    }

    /// "Open log folder": the `file://` address of the log folder to open.
    pub fn open_log_folder(&self) -> String {
        file_url(&paths::log_dir())
    }

    // ============================================================== drawing
    /// The dialog's window.
    pub fn viewport_id() -> egui::ViewportId {
        egui::ViewportId::from_hash_of("eye-tracker-settings")
    }

    /// The window: titled, opaque, with the app icon.
    pub fn viewport_builder() -> egui::ViewportBuilder {
        icons::with_app_icon(
            egui::ViewportBuilder::default()
                .with_title(format!("{APP_NAME} Settings"))
                .with_inner_size([860.0, 640.0])
                .with_min_inner_size([760.0, 540.0])
                // Under glow a child viewport left unset inherits the transparent root's flag.
                .with_transparent(false),
        )
    }

    /// Draw the dialog in its own window; returns what it asks the controller to do.
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        inputs: &SettingsInputs<'_>,
    ) -> Vec<SettingsAction> {
        self.flush_deferred();
        self.poll_cameras();
        ctx.show_viewport_immediate(
            Self::viewport_id(),
            Self::viewport_builder(),
            |ui, _class| self.draw(ui, inputs),
        );
        if self.detecting {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        self.take_actions()
    }

    fn draw(&mut self, ui: &mut egui::Ui, inputs: &SettingsInputs<'_>) {
        let ctx = ui.ctx().clone();
        if let Some(text) = self.clipboard.take() {
            ctx.copy_text(text);
        }
        if ctx.input(|i| i.viewport().close_requested()) {
            self.reject();
            return;
        }
        // Read before drawing: Esc in a shortcut field ends its recording on the way.
        let recording = self
            .bindings
            .iter()
            .any(|b| matches!(&b.editor, Editor::Hotkey(e) if e.recording));
        if let Some(dialog) = &mut self.file_dialog {
            dialog.update(&ctx);
            if let Some(path) = dialog.take_picked() {
                self.file_dialog = None;
                self.select_device_file(&path);
            }
        }
        egui::Panel::bottom("settings-footer").show(ui, |ui| self.draw_footer(ui));
        egui::Panel::left("settings-nav")
            .exact_size(200.0)
            .resizable(false)
            .show(ui, |ui| {
                ui.add_space(10.0);
                for (row, (_, title, _)) in PAGES.iter().enumerate() {
                    let selected = row == self.page;
                    let text = RichText::new(*title).size(15.0);
                    let button = egui::Button::selectable(selected, text)
                        .min_size(egui::vec2(ui.available_width(), 32.0));
                    if ui.add(button).clicked() && !selected {
                        self.on_page_changed(row);
                    }
                }
            });
        egui::CentralPanel::default().show(ui, |ui| {
            let (_, title, subtitle) = PAGES[self.page];
            ui.label(RichText::new(title).size(22.0).strong());
            ui.label(RichText::new(subtitle).weak());
            ui.add_space(8.0);
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    if PAGES[self.page].0 == "diagnostics" {
                        self.draw_diagnostics(ui);
                    } else {
                        let items = std::mem::take(&mut self.pages[self.page]);
                        for item in &items {
                            self.draw_item(ui, item, inputs);
                        }
                        self.pages[self.page] = items;
                    }
                });
        });
        // Enter is OK and Esc is Cancel, unless a shortcut field took the keys.
        if !recording && self.file_dialog.is_none() {
            if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
                self.reject();
            } else if self.ok_enabled()
                && ctx.memory(|m| m.focused().is_none())
                && ctx.input(|i| i.key_pressed(egui::Key::Enter))
            {
                self.accept();
            }
        }
    }

    fn draw_footer(&mut self, ui: &mut egui::Ui) {
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui
                .button("Restore defaults")
                .on_hover_text("Show the default value of every setting (Apply to keep).")
                .clicked()
            {
                self.restore_defaults();
            }
            let status = RichText::new(&self.status);
            ui.label(if self.status_is_error {
                status.color(DANGER)
            } else {
                status
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add_enabled(self.apply_enabled(), egui::Button::new("Apply"))
                    .clicked()
                {
                    self.apply();
                }
                if ui.button("Cancel").clicked() {
                    self.reject();
                }
                if ui
                    .add_enabled(self.ok_enabled(), egui::Button::new("OK"))
                    .clicked()
                {
                    self.accept();
                }
            });
        });
        ui.add_space(6.0);
    }

    fn draw_item(&mut self, ui: &mut egui::Ui, item: &Item, inputs: &SettingsInputs<'_>) {
        match item {
            Item::Banner(text) => banner(ui, text),
            Item::Group { title, rows } => {
                ui.add_space(6.0);
                ui.label(RichText::new(*title).strong());
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    egui::Grid::new(*title)
                        .num_columns(2)
                        .spacing([16.0, 8.0])
                        .show(ui, |ui| {
                            for row in rows {
                                self.draw_row(ui, row, inputs);
                            }
                        });
                });
            }
        }
    }

    /// One grid row: a label (or nothing) and its widget.
    fn draw_row(&mut self, ui: &mut egui::Ui, row: &Row, inputs: &SettingsInputs<'_>) {
        match row {
            Row::Field { key, label } => {
                let enabled = self.is_enabled(key);
                match label {
                    Some(label) => {
                        ui.add_enabled(enabled, egui::Label::new(*label))
                            .on_hover_text(doc(key));
                    }
                    None => {
                        ui.label("");
                    }
                }
                ui.add_enabled_ui(enabled, |ui| self.draw_field(ui, key));
            }
            Row::Resolution => {
                ui.label("Resolution")
                    .on_hover_text("Requested capture size.");
                ui.horizontal(|ui| {
                    self.draw_field(ui, "camera.width");
                    ui.label("×");
                    self.draw_field(ui, "camera.height");
                });
            }
            Row::Hint { text, follows } => {
                let enabled = follows.is_none_or(|key| self.is_enabled(key));
                ui.label("");
                ui.add_enabled(enabled, hint_label(text));
            }
            Row::Autostart => {
                ui.label("");
                let tooltip = if self.autostart_supported {
                    "Runs quietly in the tray after you log in."
                } else {
                    "Start at login is not supported on this system."
                };
                let mut checked = self.autostart_checked;
                let text = format!("Start {APP_NAME} when I log in");
                let response = ui
                    .add_enabled(
                        self.autostart_supported,
                        egui::Checkbox::new(&mut checked, text),
                    )
                    .on_hover_text(tooltip);
                if response.changed() {
                    self.set_autostart_checked(checked);
                }
            }
            Row::AutostartNote => {
                if self.autostart_note.is_empty() {
                    return;
                }
                ui.label("");
                ui.add(hint_label(&self.autostart_note));
            }
            Row::SetupButton => {
                ui.label("");
                if ui
                    .button("Run setup assistant…")
                    .on_hover_text(
                        "Check the camera and choose what happens when you walk away, step by \
                         step. Cancelling it changes nothing.",
                    )
                    .clicked()
                {
                    self.actions.push(SettingsAction::SetupRequested);
                }
            }
            Row::UpdatesHint => {
                ui.label("");
                ui.add(hint_label(self.updates_hint()));
            }
            Row::Calibration => {
                ui.label("");
                ui.horizontal(|ui| {
                    ui.label(calibration_status_text(inputs));
                    if ui
                        .button("Recalibrate…")
                        .on_hover_text("Run the calibration again (about 30 seconds).")
                        .clicked()
                    {
                        self.actions.push(SettingsAction::CalibrationRequested);
                    }
                });
            }
            Row::ActionNote => {
                let Some(note) = self.action_note() else {
                    return;
                };
                ui.label("");
                ui.add(hint_label(&note));
            }
            Row::CameraNote => {
                let Some(note) = &self.camera_note else {
                    return;
                };
                ui.label("");
                ui.add(hint_label(note));
            }
            Row::Stats => {
                ui.label("");
                ui.add(egui::Label::new(self.stats.as_str()).selectable(true));
            }
            Row::BackendLabel => {
                let Some((name, version)) = inputs.backend_info else {
                    return;
                };
                ui.label("");
                ui.add(hint_label(&format!("Backend in use: {name} ({version})")));
            }
            Row::HotkeyError | Row::HotkeyWarning => {
                let (lines, color) = if matches!(row, Row::HotkeyError) {
                    (&self.hotkey_errors, DANGER)
                } else {
                    (&self.hotkey_warnings, WARNING)
                };
                if lines.is_empty() {
                    return;
                }
                let text: Vec<&str> = lines.iter().map(|(_, t)| t.as_str()).collect();
                ui.label("");
                ui.label(RichText::new(text.join("\n")).color(color));
            }
        }
        ui.end_row();
    }

    fn draw_field(&mut self, ui: &mut egui::Ui, key: &'static str) {
        let Some(index) = self.bindings.iter().position(|b| b.key == key) else {
            return;
        };
        if let Editor::Hotkey(_) = self.bindings[index].editor {
            self.draw_hotkey(ui, key);
            return;
        }
        if let Editor::Device { .. } = self.bindings[index].editor {
            self.draw_device(ui, index);
            return;
        }
        let binding = &mut self.bindings[index];
        let tooltip = binding.tooltip.clone();
        let response = match &mut binding.editor {
            Editor::Check { checked } => ui.checkbox(checked, binding.text.as_str()),
            Editor::Combo { options, index } => {
                let before = *index;
                let selected = options
                    .get(*index)
                    .map_or("", |(_, t)| t.as_str())
                    .to_string();
                let mut response = egui::ComboBox::from_id_salt(key)
                    .selected_text(selected)
                    .show_ui(ui, |ui| {
                        for (n, (_, text)) in options.iter().enumerate() {
                            ui.selectable_value(index, n, text.as_str());
                        }
                    })
                    .response;
                if *index != before {
                    response.mark_changed();
                }
                response
            }
            Editor::Int {
                value,
                lo,
                hi,
                step,
                suffix,
            } => ui.add(
                egui::DragValue::new(value)
                    .range(*lo..=*hi)
                    .speed(*step as f64 / 10.0)
                    .suffix(*suffix),
            ),
            Editor::Float {
                value,
                lo,
                hi,
                decimals,
                step,
                suffix,
                ..
            } => ui.add(
                egui::DragValue::new(value)
                    .range(*lo..=*hi)
                    .speed(*step / 10.0)
                    .fixed_decimals(*decimals)
                    .suffix(*suffix),
            ),
            Editor::Slider { value, left, right } => {
                ui.horizontal(|ui| {
                    ui.add(hint_label(left));
                    let response = ui.add(
                        egui::Slider::new(value, 0..=100)
                            .show_value(false)
                            .step_by(1.0),
                    );
                    ui.add(hint_label(right));
                    ui.label(format!("{value} %"));
                    response
                })
                .inner
            }
            Editor::Lines { text, placeholder } => ui.add(
                egui::TextEdit::multiline(text)
                    .hint_text(*placeholder)
                    .desired_rows(4)
                    .lock_focus(false),
            ),
            Editor::Hotkey(_) | Editor::Device { .. } => return,
        };
        let response = if tooltip.is_empty() {
            response
        } else {
            response.on_hover_text(tooltip)
        };
        if response.changed() {
            self.on_changed();
        }
    }

    fn draw_device(&mut self, ui: &mut egui::Ui, index: usize) {
        let mut changed = false;
        let mut detect = false;
        let mut browse = false;
        let detecting = self.detecting;
        let binding = &mut self.bindings[index];
        let tooltip = binding.tooltip.clone();
        let Editor::Device {
            items,
            index: selected,
            text,
        } = &mut binding.editor
        else {
            return;
        };
        ui.horizontal(|ui| {
            let edit = egui::TextEdit::singleline(text)
                .hint_text("Camera number or a video file")
                .desired_width(240.0);
            changed |= ui.add(edit).on_hover_text(tooltip).changed();
            egui::ComboBox::from_id_salt("camera.device.items")
                .selected_text("")
                .width(24.0)
                .show_ui(ui, |ui| {
                    for (n, (label, _)) in items.iter().enumerate() {
                        if ui
                            .selectable_label(*selected == Some(n), label.as_str())
                            .clicked()
                        {
                            *selected = Some(n);
                            *text = label.clone();
                            changed = true;
                        }
                    }
                });
            let label = if detecting {
                "Detecting…"
            } else {
                "Detect cameras"
            };
            detect = ui
                .add_enabled(!detecting, egui::Button::new(label))
                .on_hover_text("Look for connected cameras (each may briefly turn on).")
                .clicked();
            browse = ui
                .button("…")
                .on_hover_text("Use a video or image file instead of a camera (for testing).")
                .clicked();
        });
        if changed {
            self.on_changed();
        }
        if detect {
            self.detect_cameras();
        }
        if browse {
            let mut dialog = egui_file_dialog::FileDialog::new()
                .title("Choose a video or image")
                .add_file_filter_extensions(
                    "Video or image",
                    vec![
                        "mp4", "avi", "mkv", "mov", "webm", "png", "jpg", "jpeg", "bmp",
                    ],
                )
                .default_file_filter("Video or image");
            dialog.pick_file();
            self.file_dialog = Some(dialog);
        }
    }

    fn draw_hotkey(&mut self, ui: &mut egui::Ui, key: &'static str) {
        let action = key.trim_start_matches("hotkeys.");
        let Some(edit) = self.hotkey_edit(action) else {
            return;
        };
        let tooltip = self.tooltip(key).to_string();
        let shown = edit.shown_text();
        let text = if shown.is_empty() {
            RichText::new(edit.placeholder()).weak()
        } else {
            RichText::new(shown)
        };
        let mut button = egui::Button::new(text)
            .selected(edit.recording())
            .min_size(egui::vec2(220.0, 0.0));
        if edit.invalid() {
            button = button.stroke(egui::Stroke::new(1.0, DANGER));
        }
        let mut clear = false;
        let response = ui
            .horizontal(|ui| {
                let response = ui.add(button).on_hover_text(tooltip);
                clear = ui
                    .button("Clear")
                    .on_hover_text("Remove this shortcut.")
                    .clicked();
                response
            })
            .inner;
        if response.clicked() {
            response.request_focus();
        }
        if clear {
            self.set_hotkey(action, "");
        }
        let focused = response.has_focus();
        let recording = self.hotkey_edit(action).is_some_and(HotkeyEdit::recording);
        if focused != recording {
            self.set_recording(action, focused);
        }
        if !focused {
            return;
        }
        // While recording, claim every key combination (Tab, arrows, Esc included).
        ui.memory_mut(|m| {
            m.set_focus_lock_filter(
                response.id,
                egui::EventFilter {
                    tab: true,
                    horizontal_arrows: true,
                    vertical_arrows: true,
                    escape: true,
                },
            );
        });
        let events = ui.input(|i| i.events.clone());
        for event in events {
            let egui::Event::Key {
                key,
                physical_key,
                pressed,
                repeat,
                modifiers,
            } = event
            else {
                continue;
            };
            if matches!(
                physical_key.unwrap_or(key),
                egui::Key::SuperLeft | egui::Key::SuperRight
            ) {
                self.super_held = pressed;
            }
            let held = HeldModifiers::from_egui(modifiers, self.super_held);
            if !pressed {
                if is_modifier_key(key) {
                    self.hotkey_modifiers_released(action, held);
                }
                continue;
            }
            if repeat {
                continue;
            }
            if !self.hotkey_key_press(action, key, held) {
                response.surrender_focus();
                break;
            }
        }
    }

    fn draw_diagnostics(&mut self, ui: &mut egui::Ui) {
        let mut text = if self.diagnostics.is_empty() && self.diagnostics_loaded {
            "Collecting…".to_string()
        } else {
            self.diagnostics.clone()
        };
        ui.add(
            egui::TextEdit::multiline(&mut text)
                .font(egui::TextStyle::Monospace)
                .interactive(true)
                .desired_rows(18)
                .desired_width(f32::INFINITY)
                .code_editor(),
        );
        ui.horizontal(|ui| {
            if ui.button("Refresh").clicked() {
                self.refresh_diagnostics();
            }
            if ui
                .button("Copy report")
                .on_hover_text("Copy the report to paste into a bug report.")
                .clicked()
            {
                self.copy_diagnostics();
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Open log folder").clicked() {
                    ui.ctx()
                        .open_url(egui::OpenUrl::new_tab(self.open_log_folder()));
                }
            });
        });
        ui.add(hint_label(
            "The report contains no images and nothing is sent anywhere.",
        ));
    }
}

/// The calibration line of the General page.
pub fn calibration_status_text(inputs: &SettingsInputs<'_>) -> String {
    let Some(data) = inputs.calibration else {
        return "Not calibrated yet.".into();
    };
    let created: String = data.created_at.chars().take(10).collect();
    let mut text = String::from("Calibrated");
    if !created.is_empty() {
        text += &format!(" on {created}");
    }
    if let Some(grade) = data.grade().filter(|g| !g.is_empty()) {
        text += &format!(" · quality: {grade}");
    }
    text += ".";
    // A stored calibration may no longer fit (new monitor layout, other backend).
    if !inputs.is_calibrated {
        if inputs.calibration_reason.is_empty() {
            text += " Needs recalibration.";
        } else {
            text += &format!(" Needs recalibration: {}.", inputs.calibration_reason);
        }
    }
    text
}

/// The banner of the Hotkeys page where hotkeys are unavailable or limited (rich text).
fn hotkey_banner(supported: bool, note: Option<String>) -> Option<String> {
    let note = note.filter(|n| !n.trim().is_empty());
    if supported {
        return note.map(|n| html_escape(&n));
    }
    let note = note
        .unwrap_or_else(|| "Global shortcuts are not available in this desktop session.".into());
    let mut text = html_escape(&note);
    // Named as this copy is run (the .AppImage file, eye-tracker-cli, …):
    // the packages put no "eye-tracker" command on the PATH, and a
    // desktop shortcut bound to a missing command fails silently. Added
    // unless the note already names the commands that way.
    if !note.contains(&cli_command_text(&["ctl"])) {
        let commands: Vec<String> = ["toggle", "privacy-toggle", "calibrate"]
            .iter()
            .map(|action| {
                format!(
                    "<code>{}</code>",
                    html_escape(&cli_command_text(&["ctl", action]))
                )
            })
            .collect();
        text += "<br>Bind these commands in your desktop's keyboard settings instead: ";
        text += &commands.join(" · ");
    }
    Some(text)
}

/// Python's `html.escape`.
pub(crate) fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

/// `file://` address of a local folder (percent-encoded, as `QUrl::fromLocalFile`).
pub(crate) fn file_url(path: &Path) -> String {
    let text = path.to_string_lossy().replace('\\', "/");
    let mut out = String::from(if text.starts_with('/') {
        "file://"
    } else {
        "file:///"
    });
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric() || b"/-._~:".contains(&byte) {
            out.push(char::from(byte));
        } else {
            out += &format!("%{byte:02X}");
        }
    }
    out
}

/// A muted, wrapping, selectable explanation.
pub(crate) fn hint_label(text: &str) -> egui::Label {
    egui::Label::new(RichText::new(text).weak())
        .wrap()
        .selectable(true)
}

/// Rich text with `<b>`, `<code>` and `<br>` (the markup the Python labels used).
pub(crate) fn rich_text(ui: &egui::Ui, html: &str) -> LayoutJob {
    let mut job = LayoutJob::default();
    let body = egui::TextStyle::Body.resolve(ui.style());
    let mono = egui::TextStyle::Monospace.resolve(ui.style());
    let color = ui.visuals().text_color();
    let strong = ui.visuals().strong_text_color();
    let (mut bold, mut code) = (false, false);
    let mut rest = html;
    while !rest.is_empty() {
        if rest.starts_with('<') {
            let end = rest.find('>').map_or(rest.len(), |e| e + 1);
            match &rest[..end] {
                "<b>" => bold = true,
                "</b>" => bold = false,
                "<code>" => code = true,
                "</code>" => code = false,
                "<br>" => job.append("\n", 0.0, TextFormat::simple(body.clone(), color)),
                _ => {}
            }
            rest = &rest[end..];
            continue;
        }
        let end = rest.find('<').unwrap_or(rest.len());
        let text = rest[..end]
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&#x27;", "'")
            .replace("&nbsp;", "\u{a0}")
            .replace("&amp;", "&");
        let format = TextFormat {
            font_id: if code { mono.clone() } else { body.clone() },
            color: if bold { strong } else { color },
            ..Default::default()
        };
        job.append(&text, 0.0, format);
        rest = &rest[end..];
    }
    job
}

/// Rich text in a tinted, rounded frame.
pub(crate) fn banner(ui: &mut egui::Ui, html: &str) {
    egui::Frame::new()
        .fill(ACCENT.linear_multiply(0.10))
        .stroke(egui::Stroke::new(1.0, ACCENT.linear_multiply(0.35)))
        .corner_radius(8.0)
        .inner_margin(egui::Margin::symmetric(14, 10))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            let job = rich_text(ui, html);
            ui.add(egui::Label::new(job).wrap().selectable(true));
        });
}

#[cfg(test)]
#[path = "settings_dialog_tests.rs"]
pub(crate) mod tests;
