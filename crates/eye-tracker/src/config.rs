//! User settings: typed structs persisted as JSON.
//!
//! Loading is forgiving by design. Unknown keys are ignored, missing keys fall back to
//! defaults, wrongly typed values are replaced by the default and out-of-range numbers are
//! clamped. A hand-edited or older config file can therefore never prevent the app from
//! starting; problems are reported through the log.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use serde::Serialize;
use serde_json::{Map, Value};

/// Format version written into the settings file. Files of every earlier version keep
/// loading; [`migrate`] updates them:
///
/// 1. The first builds.
/// 2. Today's default hotkeys (see [`HOTKEY_MODIFIERS`]). A version 1 file whose three hotkeys
///    are still an earlier build's untouched defaults gets today's defaults. The next save
///    writes version 2, so this happens once: hotkeys chosen afterwards are kept even if they
///    equal an old default.
pub const CONFIG_VERSION: i64 = 2;

/// Vision backend names of earlier versions and what they are called now. The MediaPipe-runtime
/// backend became the self-contained "facemesh" one (same landmark model), the OpenCV/YuNet one
/// became "lite".
pub const LEGACY_BACKENDS: [(&str, &str); 2] = [("mediapipe", "facemesh"), ("opencv", "lite")];
/// The hotkey defaults of the first builds (`hotkeys.toggle_tracking`, `toggle_privacy`,
/// `recalibrate`) on every platform. Ctrl+Alt+letter is AltGr+letter on many Windows layouts,
/// opens a terminal on Linux desktops and is a Rectangle / Magnet shortcut on macOS.
pub const LEGACY_HOTKEYS: [&str; 3] = ["ctrl+alt+t", "ctrl+alt+p", "ctrl+alt+c"];
/// The Linux (and other X11 desktop) defaults of later version 1 builds. Alt+Shift and
/// Ctrl+Shift are common keyboard-layout switches there, which make this chord impossible to
/// press, and JetBrains IDEs use it.
pub const LEGACY_HOTKEYS_X11: [&str; 3] =
    ["ctrl+alt+shift+t", "ctrl+alt+shift+p", "ctrl+alt+shift+c"];
/// The first file version that may hold old default hotkeys on purpose.
const HOTKEY_DEFAULTS_VERSION: i64 = 2;
const HOTKEY_FIELDS: [&str; 3] = ["toggle_tracking", "toggle_privacy", "recalibrate"];

/// The platform groups the defaults differ by (Python's `sys.platform` prefixes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    Windows,
    MacOs,
    /// Linux/X11 and other Unix desktops.
    Other,
}

impl Platform {
    pub const CURRENT: Platform = if cfg!(target_os = "windows") {
        Platform::Windows
    } else if cfg!(target_os = "macos") {
        Platform::MacOs
    } else {
        Platform::Other
    };
}

/// Modifiers of the default hotkeys by target platform; the keys are T (pause /
/// resume tracking), P (privacy mode) and C (calibrate) everywhere. A global hotkey
/// takes its combination away from every other application, so the defaults must
/// type no character, stay pressable while a keyboard-layout switch is configured,
/// and be no default shortcut of the system or of the tools this app's users are
/// likely to run. Ctrl+Alt+Meta (Ctrl+Alt+Win, ⌃⌥⌘, Ctrl+Alt+Super) passes on
/// every platform; what was checked:
///
/// * Windows reports AltGr as Ctrl+Alt, and the layout tables of the stock Windows
///   layouts show Ctrl+Alt(+Shift)+T, P and C typing characters on 19-47 layouts each
///   (Ctrl+Alt+T is '₺' on Turkish Q, Ctrl+Alt+C 'ć' on Polish (Programmers),
///   Ctrl+Alt+P 'ö' on US-International). No Windows layout uses the Win key as a
///   character modifier, and the shell's own Win shortcuts (Game Bar Win+Alt+*,
///   Win+Ctrl+*, the Office key Ctrl+Alt+Shift+Win) leave Ctrl+Alt+Win+T/P/C free.
/// * Linux (X11): AltGr is a modifier of its own (Mod5), so Ctrl+Alt chords never
///   type. GNOME, KDE Plasma, Xfce, Cinnamon and MATE open a terminal on Ctrl+Alt+T,
///   switch virtual terminals on Ctrl+Alt+F1-F12 and bind Super+letter (Super+P:
///   displays), but none binds Ctrl+Alt+Super+T, P or C, and X grabs match the
///   modifiers exactly, so Super+P is not Ctrl+Alt+Super+P. Ctrl+Alt+Shift, the
///   default of earlier builds, failed twice. With the XKB layout switches that
///   bilingual users pick most, Alt+Shift (grp:alt_shift_toggle) and Ctrl+Shift
///   (grp:ctrl_shift_toggle), whichever of the pair is pressed second switches the
///   layout instead of adding its modifier (xkeyboard-config symbols/group), so the
///   chord cannot be formed and arrives as Ctrl+Alt+T (a terminal) or Ctrl+Shift+T.
///   And IDEs use it: the JetBrains keymap for Refactor This, Copy Reference and
///   Introduce Functional Parameter (Ctrl+Alt+Shift+T/C/P), VS Code for Copy Path
///   (Ctrl+Alt+Shift+C). Neither binds Ctrl+Alt+Super. The other usual switches
///   (Super+Space, Caps Lock, Shift+Caps Lock) leave the defaults alone; the rare
///   options that do (grp:ctrl_alt_toggle, the Win-key switches and selectors,
///   altwin:ctrl_win, ...) are detected by the hotkey manager, which then reports
///   the hotkey as unavailable instead of grabbing a chord that cannot be pressed.
///   One limit remains where a desktop binds Super on its own (Xubuntu's menu):
///   its grab takes the keyboard while Super is held, so the chord works only
///   with Ctrl or Alt pressed first. The hotkey manager detects that and says so.
/// * macOS: Control+Option+Command+letter types nothing. Rectangle and Magnet, the
///   window managers multi-monitor Mac users run most, take ⌃⌥ with arrows and
///   letters by default (⌃⌥T Last Two Thirds, ⌃⌥C Center) and ⌃⌥⌘ only with ←/→
///   (previous / next display); Rectangle's Spectacle scheme uses ⌥⌘ (⌥⌘C Center).
///   Moom and BetterTouchTool use only the shortcuts their user records. macOS binds
///   no ⌃⌥⌘+letter (VoiceOver, while it runs, reads every ⌃⌥ chord as a command of
///   its own). Hotkeys are registered exclusively, so a combination that another
///   app already holds is reported ("Hotkey unavailable: ⌃⌥⌘T is already in use by
///   another application") instead of firing in both apps.
///
/// The table stays per platform so that one platform's defaults can change without
/// touching the others; [`LEGACY_HOTKEYS`] and [`LEGACY_HOTKEYS_X11`] list
/// what earlier builds used, and [`migrate`] moves untouched old files over.
pub const HOTKEY_MODIFIERS: [(Platform, &str); 2] = [
    (Platform::Windows, "ctrl+alt+meta"),
    (Platform::MacOs, "ctrl+alt+meta"),
];
/// Linux/X11 and other Unix desktops.
pub const HOTKEY_MODIFIERS_OTHER: &str = "ctrl+alt+meta";

/// The default hotkey for `key` on `platform` (see [`HOTKEY_MODIFIERS`]).
pub fn default_hotkey(platform: Platform, key: &str) -> String {
    let modifiers = HOTKEY_MODIFIERS
        .iter()
        .find(|(p, _)| *p == platform)
        .map_or(HOTKEY_MODIFIERS_OTHER, |(_, m)| m);
    format!("{modifiers}+{key}")
}

/// Declarative description of one setting beyond its type and default.
#[derive(Clone, Copy)]
struct Meta {
    lo: Option<f64>,
    hi: Option<f64>,
    choices: &'static [&'static str],
    doc: &'static str,
}

impl Meta {
    const NONE: Meta = Meta {
        lo: None,
        hi: None,
        choices: &[],
        doc: "",
    };
}

/// One row of [`describe_settings`]: everything the docs and the settings UI know about a
/// setting.
#[derive(Debug, Clone, PartialEq)]
pub struct SettingRow {
    pub section: &'static str,
    pub field: &'static str,
    /// `section.field`, the key as written in the docs.
    pub key: String,
    /// `bool`, `int`, `float`, `str` or `list[str]`.
    pub kind: &'static str,
    pub default: Value,
    pub lo: Option<f64>,
    pub hi: Option<f64>,
    pub choices: &'static [&'static str],
    pub doc: &'static str,
}

/// A value type a setting can have, with the forgiving conversion from JSON.
trait Setting: Clone + Into<Value> {
    const KIND: &'static str;
    /// `value` converted to this type, or `default` (after a warning) when it does not fit.
    fn coerce(key: &str, value: &Value, default: &Self, meta: &Meta) -> Self;
}

impl Setting for bool {
    const KIND: &'static str = "bool";
    fn coerce(key: &str, value: &Value, default: &Self, _: &Meta) -> Self {
        value.as_bool().unwrap_or_else(|| {
            log::warn!("Setting {key} expects true/false; using default {default:?}");
            *default
        })
    }
}

/// The finite, clamped number in `value`, if it is one.
fn clamped_number(key: &str, value: &Value, meta: &Meta) -> Option<f64> {
    let mut num = value.as_f64().filter(|_| value.is_number())?;
    if !num.is_finite() {
        return None;
    }
    if let Some(lo) = meta.lo.filter(|lo| num < *lo) {
        log::warn!("Setting {key}={value} below minimum {lo}; clamped");
        num = lo;
    }
    if let Some(hi) = meta.hi.filter(|hi| num > *hi) {
        log::warn!("Setting {key}={value} above maximum {hi}; clamped");
        num = hi;
    }
    Some(num)
}

impl Setting for i64 {
    const KIND: &'static str = "int";
    fn coerce(key: &str, value: &Value, default: &Self, meta: &Meta) -> Self {
        match clamped_number(key, value, meta) {
            Some(num) => num.round_ties_even() as i64,
            None => {
                log::warn!("Setting {key} expects a finite number; using default {default}");
                *default
            }
        }
    }
}

impl Setting for f64 {
    const KIND: &'static str = "float";
    fn coerce(key: &str, value: &Value, default: &Self, meta: &Meta) -> Self {
        clamped_number(key, value, meta).unwrap_or_else(|| {
            log::warn!("Setting {key} expects a finite number; using default {default:?}");
            *default
        })
    }
}

impl Setting for String {
    const KIND: &'static str = "str";
    fn coerce(key: &str, value: &Value, default: &Self, meta: &Meta) -> Self {
        let Some(text) = value.as_str() else {
            log::warn!("Setting {key} expects a string; using default {default:?}");
            return default.clone();
        };
        if !meta.choices.is_empty() && !meta.choices.contains(&text) {
            log::warn!(
                "Setting {key}={text:?} not one of {:?}; using default",
                meta.choices
            );
            return default.clone();
        }
        text.to_string()
    }
}

impl Setting for Vec<String> {
    const KIND: &'static str = "list[str]";
    fn coerce(key: &str, value: &Value, default: &Self, _: &Meta) -> Self {
        let items: Option<Vec<&str>> = value
            .as_array()
            .and_then(|a| a.iter().map(Value::as_str).collect());
        match items {
            Some(items) => items
                .into_iter()
                .filter(|v| !v.trim().is_empty())
                .map(String::from)
                .collect(),
            None => {
                log::warn!("Setting {key} expects a list of strings; using default");
                default.clone()
            }
        }
    }
}

/// `Meta` from `lo = …, hi = …, choices = …, doc = …` (each optional, in this order).
macro_rules! meta {
    () => { Meta::NONE };
    (lo = $v:expr $(, $($rest:tt)*)?) => { Meta { lo: Some($v as f64), ..meta!($($($rest)*)?) } };
    (hi = $v:expr $(, $($rest:tt)*)?) => { Meta { hi: Some($v as f64), ..meta!($($($rest)*)?) } };
    (choices = $v:expr $(, $($rest:tt)*)?) => { Meta { choices: $v, ..meta!($($($rest)*)?) } };
    (doc = $v:expr $(, $($rest:tt)*)?) => { Meta { doc: $v, ..meta!($($($rest)*)?) } };
}

/// Defines the settings sections and `Settings` from one list, so a field's type, default,
/// limits and documentation live in one place and `describe_settings()` cannot drift from it.
/// `$p` names the `Platform` the default expressions may look at.
macro_rules! settings {
    ($p:ident; $($section:ident: $Name:ident {
        $($field:ident: $ty:ty = $default:expr => { $($meta:tt)* }),* $(,)?
    }),* $(,)?) => {
        $(
            #[derive(Debug, Clone, PartialEq, Serialize)]
            pub struct $Name { $(pub $field: $ty,)* }

            impl $Name {
                fn defaults($p: Platform) -> Self {
                    let _ = $p;
                    Self { $($field: $default,)* }
                }

                fn apply(&mut self, data: &Map<String, Value>) {
                    $(if let Some(value) = data.get(stringify!($field)) {
                        let key = concat!(stringify!($section), ".", stringify!($field));
                        self.$field = Setting::coerce(key, value, &self.$field, &meta!($($meta)*));
                    })*
                }

                fn rows(&self) -> Vec<SettingRow> {
                    vec![$({
                        let meta = meta!($($meta)*);
                        SettingRow {
                            section: stringify!($section),
                            field: stringify!($field),
                            key: concat!(stringify!($section), ".", stringify!($field)).to_string(),
                            kind: <$ty as Setting>::KIND,
                            default: Value::from(self.$field.clone()),
                            lo: meta.lo,
                            hi: meta.hi,
                            choices: meta.choices,
                            doc: meta.doc,
                        }
                    }),*]
                }
            }

            impl Default for $Name {
                fn default() -> Self {
                    Self::defaults(Platform::CURRENT)
                }
            }
        )*

        #[derive(Debug, Clone, PartialEq, Serialize)]
        pub struct Settings {
            pub version: i64,
            $(pub $section: $Name,)*
        }

        impl Settings {
            fn defaults(platform: Platform) -> Self {
                Self { version: CONFIG_VERSION, $($section: $Name::defaults(platform),)* }
            }

            fn apply(&mut self, data: &Map<String, Value>) {
                $(match data.get(stringify!($section)) {
                    Some(Value::Object(section)) => self.$section.apply(section),
                    Some(_) => log::warn!(
                        "Setting {} should be an object; ignored", stringify!($section)
                    ),
                    None => {}
                })*
            }
        }

        /// Flat description of every setting (used by docs and the settings UI), in file order.
        pub fn describe_settings() -> Vec<SettingRow> {
            let settings = Settings::default();
            let mut rows = Vec::new();
            $(rows.extend(settings.$section.rows());)*
            rows
        }
    };
}

settings! { platform;
    general: GeneralSettings {
        backend: String = "auto".into() => {
            choices = &["auto", "facemesh", "lite", "apple"],
            doc = "Vision backend. 'facemesh' tracks 478 face landmarks including the irises \
                (head pose + eye direction); 'lite' uses 5 landmarks (head pose only, lowest \
                CPU); 'apple' (macOS only) uses Apple's Vision framework for head pose and \
                pupils, on the Neural Engine where available. 'auto' picks facemesh."
        },
        coreml: bool = true => {
            doc = "macOS: run the facemesh landmark model through Core ML, on the Neural Engine \
                when available; off means tract on the CPU."
        },
        start_paused: bool = false => { doc = "Start with tracking paused." },
        first_run_done: bool = false => {
            doc = "Set after the first-run wizard completes. Until then a walk-away action that \
                locks or turns the displays off only shows a notification."
        },
        notifications: bool = true => { doc = "Show tray notifications." },
        log_level: String = "INFO".into() => {
            choices = &["DEBUG", "INFO", "WARNING", "ERROR"],
            doc = "Log file verbosity. DEBUG helps with bug reports; logs never contain images \
                or keys."
        },
    },
    camera: CameraSettings {
        device: String = "0".into() => {
            doc = "Camera index ('0', '1', …) or a local image file or folder of image frames \
                (useful for testing; video files are not supported). On Linux also /dev/videoN or a stable link such as \
                /dev/v4l/by-id/…, which keeps finding a camera that was re-plugged under \
                another number. URLs and other network sources are refused: Eye Tracker works \
                fully offline."
        },
        width: i64 = 640 => { lo = 160, hi = 1920, doc = "Requested capture width." },
        height: i64 = 480 => { lo = 120, hi = 1080, doc = "Requested capture height." },
        api: String = "auto".into() => {
            choices = &["auto", "dshow", "msmf", "avfoundation", "v4l2", "any"],
            doc = "Capture API hint. The platform's native camera API is always used; only dshow has \
                an effect (it keeps the DirectShow format policy)."
        },
    },
    performance: PerformanceSettings {
        profile: String = "balanced".into() => {
            choices = &["eco", "balanced", "responsive"],
            doc = "Frame-rate profile. Eco minimises CPU, responsive minimises latency."
        },
        motion_gate: bool = true => {
            doc = "Skip face analysis when the camera image has not changed."
        },
        motion_threshold: f64 = 2.0 => {
            lo = 0.1, hi = 20.0, doc = "Mean grey-level change that counts as motion."
        },
    },
    switching: SwitchingSettings {
        enabled: bool = true => { doc = "Move the cursor to the monitor you look at." },
        dwell_ms: i64 = 300 => {
            lo = 0, hi = 3000,
            doc = "How long you must look at another monitor before switching."
        },
        hysteresis: f64 = 0.06 => {
            lo = 0.0, hi = 0.5,
            doc = "How far (fraction of the monitor size) the gaze must cross into another \
                monitor."
        },
        off_screen_margin: f64 = 0.35 => {
            lo = 0.05, hi = 2.0,
            doc = "Gaze further than this (fraction of the monitor diagonal) outside every \
                monitor is treated as looking away (phone, desk) and ignored."
        },
        cooldown_ms: i64 = 600 => { lo = 0, hi = 5000, doc = "Minimum time between two switches." },
        mouse_grace_ms: i64 = 1500 => {
            lo = 0, hi = 10000, doc = "No switching for this long after you move the mouse."
        },
        typing_grace_ms: i64 = 2000 => {
            lo = 0, hi = 10000, doc = "No switching for this long after you type."
        },
        reading_grace_ms: i64 = 6000 => {
            lo = 0, hi = 60000,
            doc = "After you typed while looking at another monitor (e.g. copying from a \
                document there), switching to that monitor waits this long after your last \
                keystroke instead of the typing grace, so reading pauses do not move your \
                keyboard focus. 0 turns this off."
        },
        cursor_target: String = "last".into() => {
            choices = &["last", "center", "gaze"],
            doc = "Where the cursor lands: last position on that monitor, its centre, or the \
                estimated gaze point."
        },
        focus_window: bool = true => {
            doc = "Also give keyboard focus to the last-used window on the target monitor."
        },
        smoothing: f64 = 0.5 => {
            lo = 0.0, hi = 1.0, doc = "Gaze smoothing strength (0 = raw, 1 = heavy)."
        },
    },
    panes: PaneSettings {
        enabled: bool = false => {
            doc = "Experimental: also move keyboard focus between split panes of supported \
                terminals (tmux, WezTerm, Windows Terminal) on the monitor you are already on. \
                Only panes large enough for your calibration's accuracy take part."
        },
        dwell_ms: i64 = 400 => {
            lo = 100, hi = 5000,
            doc = "How long you must look at another pane before it gets focus."
        },
        typing_grace_ms: i64 = 3000 => {
            lo = 0, hi = 20000,
            doc = "No pane switching for this long after you type, or after you switched panes \
                yourself (a keyboard shortcut or a click in the terminal)."
        },
        reading_grace_ms: i64 = 8000 => {
            lo = 0, hi = 60000,
            doc = "After you typed while looking at another pane (e.g. reading its output), \
                switching to that pane waits this long after your last keystroke instead of \
                the typing grace. 0 turns this off."
        },
        cooldown_ms: i64 = 1000 => {
            lo = 0, hi = 10000, doc = "Minimum time between two pane switches."
        },
        after_monitor_switch_ms: i64 = 1500 => {
            lo = 0, hi = 10000,
            doc = "No pane switching for this long after the cursor moved to another monitor."
        },
        precision: f64 = 2.5 => {
            lo = 1.0, hi = 6.0,
            doc = "A pane takes part only if it is at least this many times your gaze error \
                (measured by the calibration) wide, for panes side by side, or tall, for \
                stacked panes. Higher is safer, lower allows smaller panes."
        },
        hysteresis: f64 = 0.5 => {
            lo = 0.0, hi = 3.0,
            doc = "How far past the divider the gaze must be, as a fraction of your gaze error."
        },
        min_pane_px: i64 = 240 => {
            lo = 0, hi = 4000,
            doc = "Panes narrower (or lower) than this many pixels never take part."
        },
        move_cursor: bool = true => {
            doc = "Also move the mouse cursor into the pane that gets the keyboard focus, back \
                to where you last left it in that pane."
        },
        tmux: bool = true => {
            doc = "Follow tmux panes (through the tmux command, also in WSL)."
        },
        wezterm: bool = true => { doc = "Follow WezTerm panes (through the wezterm command)." },
        windows_terminal: bool = true => {
            doc = "Follow Windows Terminal panes (Windows only, through UI Automation)."
        },
        desktop_apps: bool = false => {
            doc = "Also follow sessions side by side in the Claude desktop app (two chats) and \
                the ChatGPT desktop app, which hosts Codex (main conversation and side chat), \
                on Windows only: the session you look at gets the keyboard focus in its \
                message box. Reads the app's accessibility tree, which makes the app build \
                that tree; this may cost the app some CPU and memory while it is on."
        },
    },
    presence: PresenceSettings {
        enabled: bool = true => { doc = "React when you walk away from the computer." },
        action: String = "lock".into() => {
            choices = &["none", "notify", "display_off", "lock", "lock_and_display_off"],
            doc = "What to do when you are away."
        },
        away_timeout_s: i64 = 45 => {
            lo = 5, hi = 3600,
            doc = "Seconds without a face (and without input) before acting; the countdown \
                runs during the last seconds of this time."
        },
        warning_s: i64 = 10 => {
            lo = 0, hi = 120,
            doc = "Countdown shown before the action (0 = act without a countdown). Being seen \
                by the camera cancels it, and so does keyboard or mouse use while \
                require_input_idle is on."
        },
        require_input_idle: bool = true => {
            doc = "Keyboard or mouse activity counts as presence even without a face (not on \
                Wayland outside GNOME, where the system does not report it)."
        },
        wake_on_return: bool = true => {
            doc = "Turn the displays back on when you return (if they were only switched off)."
        },
    },
    privacy: PrivacySettings {
        remember_privacy_mode: bool = true => {
            doc = "Privacy mode stays on when Eye Tracker or the computer restarts (also after \
                an update), so the camera never comes back on by itself."
        },
        pause_when_locked: bool = true => {
            doc = "Release the camera while the session is locked."
        },
        yield_camera: bool = true => {
            doc = "Release the camera while another app is using it (Windows and Linux only). \
                Walk-away detection and the shoulder guard pause meanwhile."
        },
        pause_for_apps: Vec<String> = Vec::new() => {
            doc = "Process names that pause tracking while running (e.g. 'obs64.exe'). The \
                camera is released, so walk-away detection and the shoulder guard pause too."
        },
        shoulder_guard: bool = false => { doc = "React when a second face appears behind you." },
        guard_action: String = "curtain".into() => {
            choices = &["notify", "curtain", "lock"],
            doc = "Shoulder guard reaction. After you unlock a lock it caused, 'lock' covers \
                the screens instead of locking again until nobody has looked over your \
                shoulder for 5 minutes."
        },
        guard_delay_s: f64 = 2.0 => {
            lo = 0.5, hi = 30.0,
            doc = "Seconds a second face must be visible before reacting."
        },
    },
    hotkeys: HotkeySettings {
        enabled: bool = true => { doc = "Register global hotkeys." },
        // Looked up per platform (not plain defaults) so tests can check every platform.
        toggle_tracking: String = default_hotkey(platform, "t") => {
            doc = "Pause / resume tracking."
        },
        toggle_privacy: String = default_hotkey(platform, "p") => {
            doc = "Privacy mode (camera fully off)."
        },
        recalibrate: String = default_hotkey(platform, "c") => { doc = "Start calibration." },
    },
    learning: LearningSettings {
        adaptive: bool = true => {
            doc = "Refine the calibration from your natural mouse use."
        },
        max_samples: i64 = 400 => {
            lo = 0, hi = 5000, doc = "Maximum number of learned samples kept."
        },
        drift_alerts: bool = true => { doc = "Suggest recalibration when accuracy drops." },
    },
    updates: UpdateSettings {
        check: bool = false => {
            doc = "Look for a newer release once a day and say so in the tray menu. Off by \
                default: this is the only thing the app ever sends over the network, one HTTPS \
                request to api.github.com that carries no identifier (see docs/privacy.md). It \
                never installs anything without asking. Checking by hand (tray menu) works \
                with this off. Windows only for now."
        },
    },
    ui: UISettings {
        show_gaze_overlay: bool = false => {
            doc = "Draw a dot where you are looking (testing aid)."
        },
    },
}

impl Default for Settings {
    fn default() -> Self {
        Self::defaults(Platform::CURRENT)
    }
}

impl Settings {
    /// Settings from parsed JSON; anything that does not fit is replaced by the default.
    pub fn from_dict(data: &Value) -> Settings {
        Self::from_dict_for(data, Platform::CURRENT)
    }

    fn from_dict_for(data: &Value, platform: Platform) -> Settings {
        let mut settings = Settings::defaults(platform);
        let Some(map) = data.as_object() else {
            log::warn!("Settings root is not an object; using defaults");
            return settings;
        };
        let migrated = migrate(map, platform);
        settings.apply(&migrated);
        settings.version = CONFIG_VERSION;
        settings
    }

    /// Load settings, falling back to defaults on any error (never panics).
    pub fn load(path: &Path) -> Settings {
        Self::load_for(path, Platform::CURRENT)
    }

    fn load_for(path: &Path, platform: Platform) -> Settings {
        let raw = match fs::read_to_string(path) {
            Ok(raw) => raw,
            Err(err) if err.kind() == io::ErrorKind::NotFound => {
                return Settings::defaults(platform);
            }
            Err(err) => {
                log::warn!("Could not read {}: {err}; using defaults", path.display());
                return Settings::defaults(platform);
            }
        };
        match serde_json::from_str::<Value>(&raw) {
            Ok(data) => Self::from_dict_for(&data, platform),
            Err(err) => {
                let mut backup = path.as_os_str().to_owned();
                backup.push(".corrupt");
                let backup = PathBuf::from(backup);
                log::warn!(
                    "Invalid JSON in {} ({err}); moved to {}",
                    path.display(),
                    backup
                        .file_name()
                        .map_or_else(String::new, |n| n.to_string_lossy().into())
                );
                let _ = fs::rename(path, &backup);
                Settings::defaults(platform)
            }
        }
    }

    /// Atomically write the settings as pretty-printed JSON.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let json = serde_json::to_string_pretty(self).map_err(io::Error::other)?;
        atomic_write_text(path, &(json + "\n"))
    }

    /// The settings as a JSON value (the file's content).
    pub fn to_dict(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

static TEMP_COUNTER: AtomicU32 = AtomicU32::new(0);

/// Write `text` to `path` atomically (temp file + rename).
pub fn atomic_write_text(path: &Path, text: &str) -> io::Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let name = path
        .file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    let tmp = parent.join(format!(
        "{name}.{}-{}.tmp",
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let write = || -> io::Result<()> {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        let mut file = options.open(&tmp)?;
        file.write_all(text.as_bytes())?;
        file.sync_all()?;
        drop(file);
        fs::rename(&tmp, path)
    };
    write().inspect_err(|_| {
        let _ = fs::remove_file(&tmp);
    })
}

// ---------------------------------------------------------------------- internals

/// Rewrite values written by earlier versions (before validation). The caller's data is never
/// modified.
fn migrate(data: &Map<String, Value>, platform: Platform) -> Map<String, Value> {
    let mut data = data.clone();

    let backend = data
        .get("general")
        .and_then(|g| g.get("backend"))
        .and_then(Value::as_str);
    if let Some((old, renamed)) = LEGACY_BACKENDS
        .iter()
        .find(|(old, _)| Some(*old) == backend)
    {
        log::info!("Setting general.backend='{old}' is now called '{renamed}'");
        if let Some(Value::Object(general)) = data.get_mut("general") {
            general.insert("backend".into(), Value::from(*renamed));
        }
    }

    if file_version(&data) < HOTKEY_DEFAULTS_VERSION
        && let Some(Value::Object(hotkeys)) = data.get_mut("hotkeys")
    {
        {
            let stored: Option<Vec<String>> = HOTKEY_FIELDS
                .iter()
                .map(|name| hotkeys.get(*name).and_then(Value::as_str))
                .map(|v| v.map(|s| s.trim().to_lowercase()))
                .collect();
            let is_old_default = stored.is_some_and(|stored| {
                earlier_default_hotkeys(platform)
                    .iter()
                    .any(|trio| trio.iter().zip(&stored).all(|(a, b)| a == b))
            });
            if is_old_default {
                // Written before today's defaults and all three still at an earlier build's
                // defaults: the user never chose them. A newer file may hold the same values
                // on purpose, which is why only old files are looked at.
                let defaults = HotkeySettings::defaults(platform);
                let fresh = [
                    defaults.toggle_tracking,
                    defaults.toggle_privacy,
                    defaults.recalibrate,
                ];
                log::info!(
                    "Hotkeys at the defaults of an earlier version; using today's defaults ({})",
                    fresh.join(", ")
                );
                for (name, value) in HOTKEY_FIELDS.iter().zip(fresh) {
                    hotkeys.insert((*name).into(), Value::from(value));
                }
            }
        }
    }
    data
}

/// The format version a settings dict was written with.
///
/// Files have always carried one; a missing or malformed value (a hand-written file) counts as
/// the oldest format.
fn file_version(data: &Map<String, Value>) -> i64 {
    match data.get("version") {
        Some(v) if v.is_u64() => v.as_i64().unwrap_or(i64::MAX),
        Some(v) => v.as_i64().unwrap_or(1),
        None => 1,
    }
}

/// The hotkey trios that version 1 builds wrote as defaults on this platform.
fn earlier_default_hotkeys(platform: Platform) -> &'static [[&'static str; 3]] {
    match platform {
        Platform::Windows | Platform::MacOs => &[LEGACY_HOTKEYS],
        Platform::Other => &[LEGACY_HOTKEYS, LEGACY_HOTKEYS_X11],
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
