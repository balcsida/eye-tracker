//! `eye-tracker doctor` and `eye-tracker bench`.
//!
//! [`collect_report`] gathers everything useful for a bug report into a JSON-friendly value and
//! [`format_report`] renders it for humans. Collection never fails as a whole: a section that
//! cannot be gathered carries an `error` entry instead. The report is meant to be pasted into
//! public bug reports, so it must not reveal the user's account name: the home directory is shown
//! as `~` wherever it appears (paths, quoted paths, error messages), camera files by their name
//! only, camera device links without their serial number, and nothing derived from the user name
//! (such as the instance socket name) is included.
//!
//! Whether the global hotkeys could be registered is only known to the app that registered them:
//! the settings window passes its hotkey manager in, and the command-line `doctor` asks a running
//! instance (`status`).
//!
//! [`run_bench`] measures what the vision pipeline costs on this machine, once as fast as
//! possible and once at the rate the app uses while you sit still.
//!
//! Differences from the Python module, which reported on its Python environment:
//!
//! * `qt` became `ui` (the egui and eframe versions and the display backend winit uses), and
//!   `libraries` lists the Rust crates behind the vision, camera and UI stacks, with the
//!   versions Cargo.lock pinned for this build.
//! * The check for unwanted Python distributions (the MediaPipe runtime with its usage logger)
//!   is gone: there is no Python environment, and the models run in tract.
//! * The `app` section has no `python`/`implementation`; it names the build instead.
//! * The monitors have no `logical_dpi` and `refresh_hz` (`platform::monitors` does not
//!   report them), and there is no headless `virtual` screen.

use std::collections::{BTreeSet, HashMap};
use std::fmt::Display;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};
use thiserror::Error;

pub use crate::cli::format_command;

use crate::cli::{FROZEN_CLI_NAME, cli_command};
use crate::config::{Settings, describe_settings};
use crate::engine::scheduler::{DEFAULT_PROFILE, PROFILES};
use crate::gaze::calibration::CalibrationReport;
use crate::gaze::store::{CalibrationData, CalibrationLibrary, utc_now_iso};
use crate::platform::autostart::{self, Status};
use crate::platform::base::PlatformServices;
use crate::platform::get_platform;
use crate::platform::hotkeys::{HotkeyArg, HotkeyManager, create_hotkey_manager, parse_hotkey};
use crate::types::{Monitor, Rect, layout_signature, virtual_bounds};
use crate::vision::backends::{
    BACKEND_MODEL_FILES, BackendUnavailable, MODEL_FILES, VisionBackend, available_backends,
    backend_class, create_backend, model_bytes,
};
use crate::vision::camera::{CameraError, CameraInfo, FrameSource, list_cameras, open_source};
use crate::vision::motion::MotionGate;
use crate::{APP_NAME, VERSION, ipc, paths};

/// Things worth knowing on Linux that are not problems (shown under "OS integration").
const LINUX_CAMERA_NOTE: &str = "apps that open /dev/video* are noticed (refused attempts too, via fanotify on \
     Linux 5.13+); apps using the PipeWire camera portal are not: add them to \
     'Pause while these apps run' (e.g. zoom, teams-for-linux, skypeforlinux, obs)";
const WAYLAND_CURSOR_NOTE: &str = "moved exactly with swaymsg (sway) or hyprctl (Hyprland); elsewhere it needs \
     ydotool 1.x with ydotoold running and a flat pointer-acceleration profile for \
     ydotool's virtual device (ydotool 0.1 is not supported)";
/// `eye-tracker-cli` on macOS: the system judges the Accessibility permission of whatever started
/// the command (the terminal), so the app's own grant is unknown.
const MAC_CLI_ACCESSIBILITY_NOTE: &str = "unknown from the command line: macOS judges the Accessibility permission of the \
     program that started this command (the terminal), not the app's own. The app's \
     Settings > Diagnostics shows the app's state";
/// A lock tool being installed does not mean it locks this session: loginctl is on every systemd
/// system, but a bare i3/sway session has nothing listening.
const LINUX_LOCK_NOTE: &str = "only shows which lock tools are installed; whether one locks this session is \
     known when it is tried. If nothing locks, you get a 'Could not lock the screen' \
     notification: test it once with a short 'After ... s away'";
/// `hotkeys.registration` when no running app can say what the OS accepted.
const HOTKEYS_NOT_RUNNING: &str = "unknown: Eye Tracker is not running. When it starts, a hotkey the system refuses \
     (for example because another app uses it) is reported in a 'Hotkey unavailable' \
     notification and in the log";
const HOTKEYS_NOT_REPORTED: &str = "unknown: the running app does not report it; see its 'Hotkey unavailable' \
     notification or the log";
/// How long `doctor` waits for the running instance's `status`.
const STATUS_TIMEOUT_MS: u64 = 1000;
/// Camera indices probed by `doctor --probe-cameras`.
const PROBE_MAX_INDEX: u32 = 4;
/// Shown for empty values. Reports are pasted into bug reports and printed on consoles with
/// legacy code pages, so the text output is kept ASCII-only.
const EMPTY: &str = "-";
/// `settings.hotkeys` fields, in the order the settings window shows them.
const HOTKEY_NAMES: [&str; 3] = ["toggle_tracking", "toggle_privacy", "recalibrate"];
/// The order of `PlatformServices::capabilities` keys in the report (a HashMap has none).
const CAPABILITY_ORDER: [&str; 11] = [
    "lock",
    "display_off",
    "wake_display",
    "input_idle",
    "key_idle",
    "session_locked",
    "focus",
    "cursor",
    "camera_in_use",
    "hotkeys",
    "panes",
];
const PERMISSION_ORDER: [&str; 2] = ["camera", "accessibility"];
/// Whether commands in the report are for a POSIX shell (Windows shells give `~` no meaning).
const POSIX_SHELL: bool = !cfg!(windows);
/// The versions the lock file pins for this build (the report's "libraries").
const CARGO_LOCK: &str = include_str!("../../../Cargo.lock");

// ======================================================================= doctor
/// What the report asks the machine; [`Host::system`] is the real one, tests replace the parts.
struct Host {
    /// A `std::env::consts::OS` value.
    os: &'static str,
    /// The running executable.
    executable: PathBuf,
    services: Arc<dyn PlatformServices>,
    /// The screen-lock tools available here (only Linux knows them).
    lock_methods: Box<dyn Fn() -> Option<Vec<String>>>,
    /// The monitors, or why there are none to ask about.
    monitors: Box<dyn Fn() -> Result<Vec<Monitor>, String>>,
    /// `list_cameras(max_index, api)`.
    list_cameras: CameraLister,
    hotkey_manager: Box<dyn Fn() -> Box<dyn HotkeyManager>>,
    instance_running: Box<dyn Fn() -> bool>,
    /// The running instance's `status` reply (read-only).
    instance_status: Box<dyn Fn() -> Option<Value>>,
    autostart: Box<dyn Fn() -> AutostartInfo>,
}

type CameraLister = Box<dyn Fn(u32, &str) -> Result<Vec<CameraInfo>, CameraError>>;

/// What the start-at-login entry says.
struct AutostartInfo {
    supported: bool,
    status: Status,
    enabled: bool,
    location: Option<String>,
    registered: Option<Vec<String>>,
    command: Vec<String>,
}

impl Host {
    fn system() -> Host {
        Host {
            os: std::env::consts::OS,
            executable: std::env::current_exe().unwrap_or_default(),
            services: get_platform(),
            lock_methods: Box::new(system_lock_methods),
            monitors: Box::new(system_monitors),
            list_cameras: Box::new(|max_index, api| list_cameras(max_index, api, &[])),
            hotkey_manager: Box::new(create_hotkey_manager),
            instance_running: Box::new(|| ipc::is_running(ipc::PROBE_TIMEOUT_MS)),
            instance_status: Box::new(system_instance_status),
            autostart: Box::new(system_autostart),
        }
    }
}

/// Gather diagnostics.
///
/// `probe_cameras` briefly opens camera indices 0-3 (their lights may flash). `hotkey_manager`
/// is the app's own hotkey manager (the settings window passes it), which knows what the OS
/// accepted; without it, a running instance is asked for its `status`.
pub fn collect_report(probe_cameras: bool, hotkey_manager: Option<&dyn HotkeyManager>) -> Value {
    collect_report_with(&Host::system(), probe_cameras, hotkey_manager)
}

fn collect_report_with(
    host: &Host,
    probe_cameras: bool,
    hotkey_manager: Option<&dyn HotkeyManager>,
) -> Value {
    let (settings, settings_info) = settings_section();
    let mut report = json!({
        "generated_at": utc_now_iso(),
        "app": safe(|| app_section(host)),
        "system": safe(|| system_section(host)),
        "ui": safe(|| Ok(ui_section())),
        "monitors": safe(|| monitors_section(host)),
        "libraries": safe(|| Ok(libraries_section())),
        "backends": safe(|| backends_section(&settings)),
        "cameras": safe(|| cameras_section(host, &settings, probe_cameras)),
        "platform": safe(|| platform_section(host)),
    });
    let running = instance_running(&report);
    report["hotkeys"] = safe(|| hotkeys_section(host, &settings, hotkey_manager, running));
    report["autostart"] = safe(|| Ok(autostart_section(host)));
    report["paths"] = safe(|| Ok(paths_section()));
    report["settings"] = settings_info;
    report["calibration"] = safe(|| calibration_section(&report, &settings));
    report["problems"] = json!(problems(&report));
    // Every section redacts its paths; this catches what they cannot foresee (an OS error
    // quoting a file under the home directory, a note, ...).
    redact_tree(&report)
}

/// Render a [`collect_report`] value as aligned, human-readable text.
pub fn format_report(report: &Value) -> String {
    let version = report
        .pointer("/app/version")
        .map_or_else(|| VERSION.to_string(), text);
    let title = format!("{APP_NAME} {version} - diagnostics");
    let generated = report.get("generated_at").map_or_else(|| "?".into(), text);
    let mut lines = vec![
        title.clone(),
        "=".repeat(title.chars().count()),
        format!("Generated {generated}"),
        String::new(),
    ];

    lines.push("Problems".into());
    match report.get("problems").and_then(Value::as_array) {
        Some(problems) if !problems.is_empty() => {
            lines.extend(problems.iter().map(|p| format!("  ! {}", text(p))));
        }
        _ => lines.push("  none found".into()),
    }
    lines.push(String::new());

    for (key, heading) in [
        ("app", "App"),
        ("system", "System"),
        ("ui", "UI"),
        ("libraries", "Libraries"),
        ("backends", "Vision backends"),
        ("cameras", "Cameras"),
        ("monitors", "Monitors"),
        ("platform", "OS integration"),
        ("hotkeys", "Hotkeys"),
        ("autostart", "Start at login"),
        ("calibration", "Calibration"),
        ("settings", "Settings"),
        ("paths", "Paths"),
    ] {
        let Some(section) = report.get(key).filter(|s| !s.is_null()) else {
            continue;
        };
        lines.push(heading.into());
        lines.extend(format_section(key, section));
        lines.push(String::new());
    }
    let mut out = lines.join("\n").trim_end().to_string();
    out.push('\n');
    out
}

/// The monitors the OS reports.
pub fn current_monitors() -> Vec<Monitor> {
    crate::platform::monitors()
}

// ------------------------------------------------------------------ real host
fn system_monitors() -> Result<Vec<Monitor>, String> {
    let unset = |name: &str| std::env::var_os(name).is_none_or(|v| v.is_empty());
    if cfg!(target_os = "linux") && unset("DISPLAY") && unset("WAYLAND_DISPLAY") {
        return Err("no display available (DISPLAY and WAYLAND_DISPLAY are not set)".into());
    }
    Ok(current_monitors())
}

#[cfg(target_os = "linux")]
fn system_lock_methods() -> Option<Vec<String>> {
    use crate::platform::linux::{LinuxOptions, LinuxPlatform};
    // The platform singleton is a trait object; a second one without background threads answers.
    let platform = LinuxPlatform::with_options(LinuxOptions {
        background_threads: false,
        ..LinuxOptions::default()
    });
    Some(platform.lock_methods())
}

#[cfg(not(target_os = "linux"))]
fn system_lock_methods() -> Option<Vec<String>> {
    None
}

fn system_instance_status() -> Option<Value> {
    let reply = match ipc::send_command("status", STATUS_TIMEOUT_MS, 0) {
        Ok(Some(reply)) if !reply.is_empty() && !reply.starts_with("error") => reply,
        Ok(_) => return None,
        Err(err) => {
            log::debug!("Asking the running instance for its status failed: {err}");
            return None;
        }
    };
    serde_json::from_str::<Value>(&reply)
        .ok()
        .filter(Value::is_object)
}

fn system_autostart() -> AutostartInfo {
    AutostartInfo {
        supported: autostart::is_supported(),
        status: autostart::status(None),
        enabled: autostart::is_enabled(None),
        location: autostart::location(),
        registered: autostart::registered_command(),
        command: autostart::launch_command(true, None),
    }
}

// ------------------------------------------------------------------ sections
type Section = Result<Value, SectionError>;

/// Why a section could not be collected, as `"Kind: message"`.
#[derive(Debug)]
struct SectionError(String);

impl SectionError {
    fn new(kind: &str, message: impl Display) -> Self {
        SectionError(format!("{kind}: {message}"))
    }
}

fn safe(collect: impl FnOnce() -> Section) -> Value {
    let message = match catch_unwind(AssertUnwindSafe(collect)) {
        Ok(Ok(section)) => return section,
        Ok(Err(err)) => err.0,
        Err(payload) => {
            let reason = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_default();
            format!("Panic: {reason}")
        }
    };
    log::debug!("Diagnostics section failed: {message}");
    // OS errors quote paths.
    json!({ "error": redact(&message) })
}

fn app_section(host: &Host) -> Section {
    Ok(json!({
        "name": APP_NAME,
        "version": VERSION,
        "frozen": true,
        "executable": redact_path(&host.executable),
        // What "eye-tracker ..." means for this copy (release packages do not put an
        // eye-tracker command on the PATH).
        "command_line": redacted_command(&cli_command(&[]), POSIX_SHELL),
        "build": if cfg!(debug_assertions) { "debug" } else { "release" },
        "instance_running": (host.instance_running)(),
    }))
}

fn system_section(host: &Host) -> Section {
    let os = match host.os {
        "linux" => "Linux",
        "macos" => "Darwin",
        "windows" => "Windows",
        other => other,
    };
    let mut info = Map::new();
    info.insert("os".into(), json!(os));
    info.insert("release".into(), json!(System::kernel_version()));
    info.insert("version".into(), json!(System::kernel_long_version()));
    info.insert("machine".into(), json!(std::env::consts::ARCH));
    info.insert("platform".into(), json!(System::long_os_version()));
    info.insert(
        "cpu_count".into(),
        json!(std::thread::available_parallelism().map(usize::from).ok()),
    );
    if host.os == "macos" {
        info.insert("macos_version".into(), json!(System::os_version()));
    }
    if host.os == "linux" {
        info.insert("session_type".into(), json!(env_var("XDG_SESSION_TYPE")));
        info.insert("desktop".into(), json!(env_var("XDG_CURRENT_DESKTOP")));
        info.insert("display".into(), json!(env_is_set("DISPLAY")));
        info.insert(
            "wayland_display".into(),
            json!(env_is_set("WAYLAND_DISPLAY")),
        );
    }
    info.insert("wayland".into(), json!(host.services.is_wayland()));
    Ok(Value::Object(info))
}

fn ui_section() -> Value {
    json!({
        "egui": locked_version("egui"),
        "eframe": locked_version("eframe"),
        "display_backend": display_backend(std::env::consts::OS),
        "WINIT_UNIX_BACKEND": env_var("WINIT_UNIX_BACKEND").unwrap_or_else(|| "not set".into()),
    })
}

/// The windowing system winit connects to: on Linux Wayland when the session offers it (unless
/// `WINIT_UNIX_BACKEND` says otherwise), else X11.
fn display_backend(os: &str) -> &'static str {
    match os {
        "linux" => match env_var("WINIT_UNIX_BACKEND").as_deref() {
            Some("wayland") => "wayland",
            Some("x11") => "x11",
            _ if env_is_set("WAYLAND_DISPLAY") => "wayland",
            _ if env_is_set("DISPLAY") => "x11",
            _ => "none",
        },
        "macos" => "appkit",
        "windows" => "win32",
        _ => "unknown",
    }
}

fn monitors_section(host: &Host) -> Section {
    let monitors = match (host.monitors)() {
        Ok(monitors) => monitors,
        Err(reason) => return Ok(json!({ "error": reason })),
    };
    let items: Vec<Value> = monitors
        .iter()
        .map(|m| {
            json!({
                "index": m.index,
                "name": m.name,
                "rect": m.rect.to_list(),
                "primary": m.primary,
                "scale": m.scale,
            })
        })
        .collect();
    let mut out = Map::new();
    out.insert("count".into(), json!(items.len()));
    out.insert("items".into(), Value::Array(items));
    if let Ok(bounds) = virtual_bounds(&monitors) {
        out.insert("virtual_desktop".into(), json!(bounds.to_list()));
        out.insert(
            "layout_signature".into(),
            json!(layout_signature(&monitors)),
        );
    }
    Ok(Value::Object(out))
}

/// The version `Cargo.lock` pins for `name` in this build.
fn locked_version(name: &str) -> Option<String> {
    let header = format!("name = \"{name}\"");
    let mut lines = CARGO_LOCK.lines();
    lines.find(|line| *line == header)?;
    let version = lines.next()?.strip_prefix("version = \"")?;
    Some(version.trim_end_matches('"').to_string())
}

fn libraries_section() -> Value {
    json!({
        "tract": locked_version("tract-onnx"),
        "nokhwa": locked_version("nokhwa"),
        "egui": locked_version("egui"),
        "eframe": locked_version("eframe"),
        "image": locked_version("image"),
    })
}

fn backends_section(settings: &Settings) -> Section {
    let mut models = Map::new();
    for (filename, digest) in MODEL_FILES {
        let users: Vec<&str> = BACKEND_MODEL_FILES
            .iter()
            .filter(|(_, files)| files.contains(&filename))
            .map(|(name, _)| *name)
            .collect();
        let bytes = model_bytes(filename).filter(|b| !b.is_empty());
        let mut entry = Map::new();
        entry.insert("used_by".into(), json!(users));
        entry.insert("present".into(), json!(bytes.is_some()));
        if let Some(bytes) = bytes {
            entry.insert("size".into(), json!(bytes.len()));
            entry.insert(
                "sha256_ok".into(),
                json!(hex::encode(Sha256::digest(bytes)) == digest),
            );
        }
        models.insert(filename.into(), Value::Object(entry));
    }
    let mut out = Map::new();
    out.insert("configured".into(), json!(settings.general.backend));
    out.insert("available".into(), json!(available_backends()));
    out.insert("models".into(), Value::Object(models));
    match backend_class(&settings.general.backend) {
        Ok(class) => {
            out.insert("active".into(), json!(class.name()));
            out.insert("feature_version".into(), json!(class.feature_version()));
        }
        Err(BackendUnavailable(reason)) => {
            out.insert("active".into(), Value::Null);
            out.insert("active_error".into(), json!(redact(&reason)));
        }
    }
    Ok(Value::Object(out))
}

fn cameras_section(host: &Host, settings: &Settings, probe: bool) -> Section {
    let mut out = Map::new();
    out.insert(
        "configured".into(),
        json!(describe_device(&settings.camera.device)),
    );
    out.insert(
        "resolution".into(),
        json!([settings.camera.width, settings.camera.height]),
    );
    out.insert("api".into(), json!(settings.camera.api));
    out.insert("probed".into(), json!(probe));
    if probe {
        let cameras = (host.list_cameras)(PROBE_MAX_INDEX, &settings.camera.api)
            .map_err(|err| SectionError::new("CameraError", err))?;
        let devices: Vec<Value> = cameras
            .iter()
            .map(
                |c| json!({"index": c.index, "name": c.name, "width": c.width, "height": c.height}),
            )
            .collect();
        out.insert("devices".into(), Value::Array(devices));
    }
    Ok(Value::Object(out))
}

fn platform_section(host: &Host) -> Section {
    let services = &host.services;
    let permissions = services.permissions();
    let mut out = Map::new();
    out.insert("name".into(), json!(services.name()));
    out.insert("wayland".into(), json!(services.is_wayland()));
    out.insert(
        "cursor_position_reliable".into(),
        json!(services.cursor_position_reliable()),
    );
    out.insert(
        "capabilities".into(),
        ordered(services.capabilities(), &CAPABILITY_ORDER),
    );
    out.insert(
        "permissions".into(),
        ordered(permissions.clone(), &PERMISSION_ORDER),
    );
    let accessibility = services.accessibility_status();
    if accessibility != "unknown" {
        out.insert("accessibility".into(), json!(accessibility));
    }
    if let Some(methods) = (host.lock_methods)() {
        out.insert("lock_methods".into(), json!(methods));
    }
    let mut notes = Map::new();
    if host.os == "linux" {
        notes.insert("camera_release".into(), json!(LINUX_CAMERA_NOTE));
        if services.is_wayland() {
            notes.insert("cursor".into(), json!(WAYLAND_CURSOR_NOTE));
        }
        notes.insert("lock".into(), json!(LINUX_LOCK_NOTE));
    }
    if host.os == "macos" && permissions.get("accessibility") == Some(&None) && is_cli(host) {
        notes.insert("accessibility".into(), json!(MAC_CLI_ACCESSIBILITY_NOTE));
    }
    if !notes.is_empty() {
        out.insert("notes".into(), Value::Object(notes));
    }
    Ok(Value::Object(out))
}

/// Whether this is the console executable of a release package (`eye-tracker-cli`).
fn is_cli(host: &Host) -> bool {
    host.executable
        .file_stem()
        .is_some_and(|stem| stem.to_string_lossy().to_lowercase() == FROZEN_CLI_NAME)
}

/// `map` as a JSON object: the keys of `order` first, then the others alphabetically.
fn ordered<V: Into<Value>>(mut map: HashMap<&'static str, V>, order: &[&'static str]) -> Value {
    let mut out = Map::new();
    for key in order {
        if let Some(value) = map.remove(key) {
            out.insert((*key).into(), value.into());
        }
    }
    let mut rest: Vec<_> = map.into_iter().collect();
    rest.sort_by_key(|(key, _)| *key);
    out.extend(
        rest.into_iter()
            .map(|(key, value)| (key.into(), value.into())),
    );
    Value::Object(out)
}

/// What the running instance's `status` says about its hotkeys.
struct InstanceHotkeys {
    registered: BTreeSet<String>,
    /// name -> why it is not registered.
    errors: HashMap<String, String>,
    /// Its manager's note.
    note: String,
}

/// Configured hotkeys, what is wrong with them, and what the OS accepted.
///
/// `doctor` cannot register anything itself (a running app holds its hotkeys, and grabbing them
/// would change global state), so registration results come from `live_manager` (the app's own
/// manager) or a running instance; otherwise they are unknown.
fn hotkeys_section(
    host: &Host,
    settings: &Settings,
    live_manager: Option<&dyn HotkeyManager>,
    running: bool,
) -> Section {
    let manager = (host.hotkey_manager)();
    let mut configured = Map::new();
    let mut invalid = Map::new();
    let mut conflicts = Map::new();
    for name in HOTKEY_NAMES {
        let hotkey_text = match name {
            "toggle_tracking" => &settings.hotkeys.toggle_tracking,
            "toggle_privacy" => &settings.hotkeys.toggle_privacy,
            _ => &settings.hotkeys.recalibrate,
        };
        configured.insert(name.into(), json!(hotkey_text));
        if hotkey_text.is_empty() {
            continue;
        }
        let hotkey = match parse_hotkey(hotkey_text) {
            Ok(hotkey) => hotkey,
            Err(err) => {
                invalid.insert(name.into(), json!(err.to_string()));
                continue;
            }
        };
        // Read-only: e.g. Ctrl+Alt+T is AltGr+T on some Windows layouts, and the manager refuses
        // to register a combination that types a character.
        match manager.layout_conflict(HotkeyArg::from(&hotkey)) {
            Ok(Some(conflict)) if !conflict.is_empty() => {
                conflicts.insert(name.into(), json!(conflict));
            }
            Ok(_) => {}
            Err(err) => log::debug!("layout_conflict({hotkey_text}) failed: {err}"),
        }
    }
    // The manager that runs knows more than a fresh one (e.g. that another program grabs the
    // Super key on its own, which only shows once grabbing).
    let instance = (live_manager.is_none() && running)
        .then(|| instance_hotkeys(host))
        .flatten();
    let note = match (live_manager, &instance) {
        (Some(live), _) => live.note().unwrap_or_default(),
        (None, Some(instance)) if !instance.note.is_empty() => instance.note.clone(),
        _ => manager.note().unwrap_or_default(),
    };
    let wanted: Vec<&str> = HOTKEY_NAMES
        .into_iter()
        .filter(|name| {
            configured[*name].as_str().is_some_and(|t| !t.is_empty())
                && !invalid.contains_key(*name)
        })
        .collect();
    let mut out = Map::new();
    out.insert("enabled".into(), json!(settings.hotkeys.enabled));
    out.insert("backend".into(), json!(manager.name()));
    out.insert("supported".into(), json!(manager.supported()));
    out.insert("note".into(), json!(note));
    out.insert("configured".into(), Value::Object(configured));
    out.insert("invalid".into(), Value::Object(invalid));
    out.insert("layout_conflicts".into(), Value::Object(conflicts));
    if settings.hotkeys.enabled && !wanted.is_empty() && manager.supported() {
        out.insert(
            "registration".into(),
            hotkey_registration(&wanted, live_manager, running, instance.as_ref()),
        );
    }
    Ok(Value::Object(out))
}

/// `{name: "registered" | "not registered: <why>"}`, or why that is unknown.
///
/// `instance` is what [`instance_hotkeys`] returned, when the caller asked a running instance.
fn hotkey_registration(
    names: &[&str],
    live_manager: Option<&dyn HotkeyManager>,
    running: bool,
    instance: Option<&InstanceHotkeys>,
) -> Value {
    let (registered, errors): (BTreeSet<String>, HashMap<String, String>) =
        if let Some(live) = live_manager {
            let registered = live.registered().into_keys().collect();
            let errors = names
                .iter()
                .filter_map(|name| {
                    let reason = live.last_error(name).filter(|r| !r.is_empty())?;
                    Some(((*name).to_string(), reason))
                })
                .collect();
            (registered, errors)
        } else if !running {
            return json!(HOTKEYS_NOT_RUNNING);
        } else if let Some(instance) = instance {
            (instance.registered.clone(), instance.errors.clone())
        } else {
            return json!(HOTKEYS_NOT_REPORTED);
        };
    let result: Map<String, Value> = names
        .iter()
        .map(|name| {
            let state = if registered.contains(*name) {
                "registered".to_string()
            } else {
                let why = errors.get(*name).map_or("reason unknown", String::as_str);
                format!("not registered: {why}")
            };
            ((*name).to_string(), json!(state))
        })
        .collect();
    Value::Object(result)
}

/// What the running instance's `status` says about its hotkeys; `None` when the instance does
/// not answer or does not report its hotkeys (an older version).
fn instance_hotkeys(host: &Host) -> Option<InstanceHotkeys> {
    let status = (host.instance_status)()?;
    let hotkeys = status.get("hotkeys")?.as_object()?;
    // A list of names, or a {name: hotkey} mapping: iterating gives the names.
    let registered = match hotkeys.get("registered") {
        Some(Value::Array(names)) => names.iter().map(text).collect(),
        Some(Value::Object(names)) => names.keys().cloned().collect(),
        _ => BTreeSet::new(),
    };
    let errors = hotkeys
        .get("errors")
        .and_then(Value::as_object)
        .map(|errors| {
            errors
                .iter()
                .filter(|(_, why)| truthy(why))
                .map(|(name, why)| (name.clone(), text(why)))
                .collect()
        })
        .unwrap_or_default();
    let note = hotkeys
        .get("note")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    Some(InstanceHotkeys {
        registered,
        errors,
        note,
    })
}

fn autostart_section(host: &Host) -> Value {
    let info = (host.autostart)();
    json!({
        "supported": info.supported,
        "status": info.status.as_str(),
        "enabled": info.enabled,
        "location": info.location.as_deref().map(redact),
        "registered": info.registered.as_deref().map(|c| redacted_command(c, POSIX_SHELL)),
        "command": redacted_command(&info.command, POSIX_SHELL),
    })
}

/// [`format_command`] with the home directory shown as `~`.
///
/// A POSIX shell expands `~` only outside quotes, so there the `~/` that replaced the home
/// directory at the start of an argument stays in front of the quoted rest, and a command the
/// report suggests still runs when pasted: `~/'my apps/eye-tracker'`, not
/// `'~/my apps/eye-tracker'`. Windows shells give `~` no meaning (there it only shortens the
/// path), so `posix_shell` is `false` there and the command is rendered as [`format_command`]
/// renders it.
fn redacted_command(parts: &[String], posix_shell: bool) -> String {
    let shown: Vec<String> = parts.iter().map(|part| redact(part)).collect();
    if !posix_shell {
        return format_command(&shown);
    }
    let pattern = home_pattern();
    parts
        .iter()
        .zip(&shown)
        .map(|(part, shown)| {
            if pattern.as_ref().is_some_and(|p| p.matches_start(part)) {
                shell_home_path(shown)
            } else {
                shell_quote(shown)
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// `~` or `~/rest` for a POSIX shell: the tilde bare, the rest quoted as needed.
fn shell_home_path(shown: &str) -> String {
    let (tilde, rest) = shown.split_once('/').unwrap_or((shown, ""));
    if tilde != "~" {
        // The home directory followed by something else than a slash.
        return shell_quote(shown);
    }
    if !shown.contains('/') {
        return tilde.to_string();
    }
    if rest.is_empty() {
        return "~/".to_string();
    }
    format!("~/{}", shell_quote(rest))
}

/// `shlex.quote`.
fn shell_quote(text: &str) -> String {
    let safe = |c: char| c.is_ascii_alphanumeric() || "_@%+=:,./-".contains(c);
    if !text.is_empty() && text.chars().all(safe) {
        return text.to_string();
    }
    format!("'{}'", text.replace('\'', "'\"'\"'"))
}

fn paths_section() -> Value {
    // No socket name: it is derived from the user name and would reveal it.
    json!({
        "profile": if paths::base_override().is_some() { "custom (--config-dir)" } else { "default" },
        "config_dir": redact_path(&paths::config_dir()),
        "data_dir": redact_path(&paths::data_dir()),
        "log_dir": redact_path(&paths::log_dir()),
        "settings_file": redact_path(&paths::settings_file()),
        "calibration_file": redact_path(&paths::calibration_file()),
        "log_file": redact_path(&paths::log_file()),
    })
}

/// Settings as the app would load them, without the side effects of `Settings::load` (which
/// moves a corrupt file aside).
fn settings_section() -> (Settings, Value) {
    let path = paths::settings_file();
    let mut info = Map::new();
    info.insert("file_exists".into(), json!(path.is_file()));
    let mut settings = Settings::default();
    if path.is_file() {
        let data = std::fs::read_to_string(&path)
            .map_err(|err| format!("{:?}: {err}", err.kind()))
            .and_then(|raw| {
                serde_json::from_str::<Value>(&raw).map_err(|err| format!("JSONDecodeError: {err}"))
            });
        match data {
            Ok(data) => {
                settings = Settings::from_dict(&data);
                info.insert("valid".into(), json!(data.is_object()));
            }
            Err(err) => {
                info.insert("valid".into(), json!(false));
                info.insert("error".into(), json!(redact(&err)));
            }
        }
    }
    let (current, defaults) = (settings.to_dict(), Settings::default().to_dict());
    let non_default: Map<String, Value> = describe_settings()
        .into_iter()
        .filter_map(|row| {
            let value = get_setting(&current, &row.key);
            (value != get_setting(&defaults, &row.key))
                .then(|| (row.key.clone(), shareable_setting(&row.key, value)))
        })
        .collect();
    info.insert("non_default".into(), Value::Object(non_default));
    (settings, Value::Object(info))
}

fn get_setting<'a>(settings: &'a Value, dotted: &str) -> &'a Value {
    let (section, field) = dotted.split_once('.').unwrap_or((dotted, ""));
    settings
        .get(section)
        .and_then(|s| s.get(field))
        .unwrap_or(&Value::Null)
}

/// A setting value as it may appear in a pasted report (no account name).
fn shareable_setting(key: &str, value: &Value) -> Value {
    match value {
        Value::String(s) if key == "camera.device" => json!(describe_device(s)), // may be a file
        Value::String(s) => json!(redact(s)),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|v| v.as_str().map_or_else(|| v.clone(), |s| json!(redact(s))))
                .collect(),
        ),
        other => other.clone(),
    }
}

fn calibration_section(report: &Value, settings: &Settings) -> Section {
    let path = paths::calibration_file();
    if !path.is_file() {
        return Ok(json!({"exists": false}));
    }
    let library = CalibrationLibrary::load(&path);
    let Some(data) = library.latest() else {
        return Ok(json!({"exists": true, "valid": false}));
    };
    let mut out = Map::new();
    out.insert("exists".into(), json!(true));
    out.insert("valid".into(), json!(true));
    out.insert("profiles".into(), json!(library.len()));
    out.insert("created_at".into(), json!(data.created_at));
    out.insert("backend".into(), json!(data.backend));
    out.insert("feature_version".into(), json!(data.feature_version));
    out.insert(
        "camera".into(),
        json!((!data.camera.is_empty()).then(|| describe_device(&data.camera))),
    );
    out.insert(
        "frame_size".into(),
        json!((data.frame_size.0 != 0 && data.frame_size.1 != 0).then_some(data.frame_size)),
    );
    out.insert("grade".into(), json!(data.grade()));
    out.insert("monitors".into(), json!(data.monitors.len()));
    out.insert("samples".into(), json!(data.samples.len()));
    out.insert(
        "implicit_samples".into(),
        json!(data.implicit_samples.len()),
    );
    out.insert("model_degree".into(), json!(data.model.degree));
    if let Some(summary) = calibration_summary(&data.report) {
        out.insert("summary".into(), json!(summary));
    }
    if library.len() > 1 {
        // Most recently used first, one per desk / camera / backend.
        let profiles: Map<String, Value> = library
            .profiles()
            .iter()
            .enumerate()
            .map(|(n, profile)| ((n + 1).to_string(), json!(describe_profile(profile))))
            .collect();
        out.insert("profile_list".into(), Value::Object(profiles));
    }
    let backends = report.get("backends");
    let active = backends
        .and_then(|b| b.get("active"))
        .and_then(Value::as_str);
    let feature_version = backends
        .and_then(|b| b.get("feature_version"))
        .and_then(Value::as_str);
    let items = report.pointer("/monitors/items").and_then(Value::as_array);
    match (active, feature_version, items) {
        (Some(active), Some(feature_version), Some(items)) if !active.is_empty() => {
            let monitors = items
                .iter()
                .map(monitor_from_item)
                .collect::<Result<Vec<_>, _>>()?;
            // What the app does: the best profile for this setup, not just the latest.
            let (matched, reason) = library.match_profile(
                active,
                feature_version,
                &monitors,
                Some(&settings.camera.device),
                None,
            );
            out.insert("compatible".into(), json!(matched.is_some()));
            match matched {
                None => {
                    out.insert("reason".into(), json!(reason));
                }
                Some(profile) if !std::ptr::eq(profile, data) => {
                    out.insert("matching_profile".into(), json!(describe_profile(profile)));
                }
                Some(_) => {}
            }
        }
        _ => {
            out.insert("compatible".into(), Value::Null);
        }
    }
    Ok(Value::Object(out))
}

/// A monitor as the `monitors` section lists it.
fn monitor_from_item(item: &Value) -> Result<Monitor, SectionError> {
    let bad = || SectionError::new("KeyError", format!("bad monitor entry {item}"));
    let rect: Vec<i32> = item
        .get("rect")
        .and_then(Value::as_array)
        .ok_or_else(bad)?
        .iter()
        .map(|v| v.as_i64().and_then(|n| i32::try_from(n).ok()))
        .collect::<Option<_>>()
        .ok_or_else(bad)?;
    let rect: [i32; 4] = rect.try_into().map_err(|_| bad())?;
    Ok(Monitor {
        index: item
            .get("index")
            .and_then(Value::as_i64)
            .and_then(|n| i32::try_from(n).ok())
            .ok_or_else(bad)?,
        name: item
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(bad)?
            .to_string(),
        rect: Rect::from_list(rect),
        primary: item
            .get("primary")
            .and_then(Value::as_bool)
            .ok_or_else(bad)?,
        scale: item.get("scale").and_then(Value::as_f64).ok_or_else(bad)?,
    })
}

/// One line per stored calibration, e.g. `2026-09-30 facemesh, 2 monitor(s), camera 0, good`.
fn describe_profile(data: &CalibrationData) -> String {
    let created: String = data.created_at.chars().take(10).collect();
    let mut parts = vec![
        format!("{created} {}", data.backend),
        format!("{} monitor(s)", data.monitors.len()),
    ];
    if !data.camera.is_empty() {
        parts.push(describe_device(&data.camera));
    }
    if let Some(grade) = data.grade().filter(|g| !g.is_empty()) {
        parts.push(grade.to_string());
    }
    parts.join(", ")
}

fn calibration_summary(report: &Map<String, Value>) -> Option<String> {
    if report.is_empty() {
        return None;
    }
    CalibrationReport::from_dict(report)
        .ok()
        .map(|r| r.summary())
}

/// Plain-language list of things that stop features from working.
fn problems(report: &Value) -> Vec<String> {
    let mut problems = Vec::new();
    for (key, section) in report.as_object().into_iter().flatten() {
        if let Some(error) = section
            .as_object()
            .filter(|s| s.len() == 1)
            .and_then(|s| s.get("error"))
        {
            problems.push(format!(
                "Could not collect {key} information: {}",
                text(error)
            ));
        }
    }

    let backends = report.get("backends").unwrap_or(&Value::Null);
    if backends
        .get("available")
        .and_then(Value::as_array)
        .is_some_and(Vec::is_empty)
    {
        problems
            .push("No vision backend is available (it needs the bundled model files).".to_string());
    }
    for (filename, model) in backends
        .get("models")
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
    {
        let users: Vec<String> = model
            .get("used_by")
            .and_then(Value::as_array)
            .map(|users| users.iter().map(text).collect())
            .unwrap_or_default();
        let users = if users.is_empty() {
            "vision".to_string()
        } else {
            users.join(", ")
        };
        if !model.get("present").is_some_and(truthy) {
            problems.push(format!(
                "The model file {filename} ({users} backend) is missing."
            ));
        } else if model.get("sha256_ok") == Some(&Value::Bool(false)) {
            problems.push(format!(
                "The model file {filename} ({users} backend) is damaged (checksum)."
            ));
        }
    }
    if let Some(error) = backends.get("active_error").filter(|e| truthy(e)) {
        problems.push(text(error));
    }

    let monitors = report.get("monitors").unwrap_or(&Value::Null);
    let one_monitor = is_single_real_monitor(monitors);
    if one_monitor {
        problems.push(
            "Only one monitor detected: switching needs two or more \
             (walk-away and privacy features still work)."
                .to_string(),
        );
    }

    let cameras = report.get("cameras").unwrap_or(&Value::Null);
    if cameras.get("probed").is_some_and(truthy)
        && cameras
            .get("devices")
            .and_then(Value::as_array)
            .is_some_and(Vec::is_empty)
    {
        let hint = if instance_running(report) {
            " (it may be in use by Eye Tracker itself)"
        } else {
            ""
        };
        problems.push(format!("No camera delivered a picture{hint}."));
    }

    let platform = report.get("platform").unwrap_or(&Value::Null);
    let permissions = platform.get("permissions").unwrap_or(&Value::Null);
    if permissions.get("camera") == Some(&Value::Bool(false)) {
        problems
            .push("Camera access is blocked by the operating system's privacy settings.".into());
    }
    if platform.get("accessibility").and_then(Value::as_str) == Some("stale") {
        problems.push(
            "Accessibility permission was granted to an earlier version, so keyboard focus \
             cannot follow your gaze: in System Settings > Privacy & Security > \
             Accessibility remove Eye Tracker with '-' and add it again."
                .into(),
        );
    } else if permissions.get("accessibility") == Some(&Value::Bool(false)) {
        problems.push(
            "Accessibility permission is missing: keyboard focus cannot follow your gaze.".into(),
        );
    }
    let capabilities = platform.get("capabilities").unwrap_or(&Value::Null);
    let capability = |name: &str| capabilities.get(name).is_none_or(truthy);
    if truthy(capabilities) && !capability("cursor") {
        problems.push(
            "This session does not allow moving the cursor. On Wayland, sway and Hyprland \
             work directly; elsewhere install ydotool 1.x and run ydotoold."
                .into(),
        );
    }
    if truthy(capabilities) && !capability("lock") {
        problems.push("Locking the screen is not supported here.".into());
    }

    let hotkeys = report.get("hotkeys").unwrap_or(&Value::Null);
    let hotkeys_enabled = hotkeys.get("enabled").is_some_and(truthy);
    if hotkeys_enabled && let Some(note) = hotkeys.get("note").filter(|n| truthy(n)) {
        problems.push(format!("Global hotkeys: {}", text(note)));
    }
    for (name, error) in hotkeys
        .get("invalid")
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
    {
        problems.push(format!("Hotkey {name} is invalid: {}", text(error)));
    }
    if hotkeys_enabled {
        let conflicts = hotkeys.get("layout_conflicts").and_then(Value::as_object);
        for (name, conflict) in conflicts.into_iter().flatten() {
            problems.push(format!("Hotkey {name} cannot be used: {}.", text(conflict)));
        }
        let registration = hotkeys.get("registration").and_then(Value::as_object);
        for (name, result) in registration.into_iter().flatten() {
            let result = text(result);
            let reason = result.strip_prefix("not registered: ").unwrap_or(&result);
            if reason != result && !conflicts.is_some_and(|c| c.contains_key(name)) {
                problems.push(format!("Hotkey {name} could not be registered: {reason}."));
            }
        }
    }

    let autostart_info = report.get("autostart").unwrap_or(&Value::Null);
    if autostart_info.get("status").and_then(Value::as_str) == Some("stale") {
        problems.push(
            "Start at login points to a copy of Eye Tracker that no longer exists; \
             turn it off and on again."
                .into(),
        );
    }

    let settings = report.get("settings").unwrap_or(&Value::Null);
    if settings.get("valid") == Some(&Value::Bool(false)) {
        problems.push("The settings file is not valid JSON; defaults are used.".into());
    }

    let calibration = report.get("calibration").unwrap_or(&Value::Null);
    // A calibration only serves monitor switching: none is missing with switching turned off,
    // or with one real monitor (reported above already).
    let switching_wanted = settings
        .pointer("/non_default/switching.enabled")
        .is_none_or(|v| *v != Value::Bool(false));
    if calibration.get("exists") == Some(&Value::Bool(false)) {
        if switching_wanted && !one_monitor {
            // Spelled as this copy is run: release packages have no "eye-tracker" command.
            let command = redacted_command(&cli_command(&["calibrate"]), POSIX_SHELL);
            problems.push(format!(
                "Not calibrated yet: choose 'Calibrate...' in the tray menu or run '{command}'."
            ));
        }
    } else if calibration.get("valid") == Some(&Value::Bool(false)) {
        problems.push("The calibration file is damaged; please recalibrate.".into());
    } else if calibration.get("compatible") == Some(&Value::Bool(false)) {
        let reason = calibration
            .get("reason")
            .map_or_else(|| "None".into(), text);
        problems.push(format!("The calibration no longer applies ({reason})."));
    }
    problems
}

/// Exactly one monitor that is a real screen.
fn is_single_real_monitor(monitors: &Value) -> bool {
    monitors.get("count").and_then(Value::as_f64) == Some(1.0)
        && !monitors.get("virtual").is_some_and(truthy)
}

fn instance_running(report: &Value) -> bool {
    report.pointer("/app/instance_running").is_some_and(truthy)
}

// ------------------------------------------------------------------ formatting
fn format_section(key: &str, section: &Value) -> Vec<String> {
    let Some(map) = section.as_object() else {
        return vec![format!("  {}", text(section))];
    };
    if key == "monitors"
        && let Some(items) = map.get("items").and_then(Value::as_array)
    {
        let mut rows = Vec::new();
        for m in items {
            let rect = m.get("rect").and_then(Value::as_array);
            let at = |i: usize| rect.and_then(|r| r.get(i)).map_or_else(|| "?".into(), text);
            let flags = if m.get("primary").is_some_and(truthy) {
                " primary"
            } else {
                ""
            };
            let scale = m.get("scale").and_then(Value::as_f64).unwrap_or(1.0);
            let dpi = m
                .get("logical_dpi")
                .map_or_else(String::new, |dpi| format!(", {} dpi", text(dpi)));
            rows.push((
                format!("#{}", m.get("index").map_or_else(|| "?".into(), text)),
                format!(
                    "{}  {}x{} at ({}, {}){flags}, scale {}{dpi}",
                    m.get("name").map_or_else(|| "?".into(), text),
                    at(2),
                    at(3),
                    at(0),
                    at(1),
                    format_g(scale),
                ),
            ));
        }
        if let Some(signature) = map.get("layout_signature") {
            rows.push(("layout".into(), text(signature)));
        }
        if map.get("virtual").is_some_and(truthy) {
            rows.push((
                "note".into(),
                "virtual screen of a headless platform, not a real monitor".into(),
            ));
        }
        return format_rows(&rows);
    }
    if key == "cameras" && map.get("probed").is_some_and(truthy) {
        let devices = map.get("devices").and_then(Value::as_array);
        let base: Map<String, Value> = map
            .iter()
            .filter(|(k, _)| *k != "devices")
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let mut lines = format_rows(&flatten(&base, ""));
        if devices.is_none_or(Vec::is_empty) {
            lines.push("  (no camera found)".into());
        }
        for d in devices.into_iter().flatten() {
            let field = |name: &str| d.get(name).map_or_else(|| "?".into(), text);
            lines.push(format!(
                "  camera {:<12} {} ({}x{})",
                field("index"),
                field("name"),
                field("width"),
                field("height")
            ));
        }
        return lines;
    }
    format_rows(&flatten(map, ""))
}

fn flatten(section: &Map<String, Value>, prefix: &str) -> Vec<(String, String)> {
    let mut rows = Vec::new();
    for (key, value) in section {
        let label = format!("{prefix}{key}");
        match value {
            Value::Object(map) if map.is_empty() => rows.push((label, EMPTY.into())),
            Value::Object(map) => rows.extend(flatten(map, &format!("{label}."))),
            _ => rows.push((label, format_value(value))),
        }
    }
    rows
}

fn format_value(value: &Value) -> String {
    match value {
        Value::Null => "unknown".into(),
        Value::Bool(flag) => if *flag { "yes" } else { "no" }.into(),
        Value::Number(n) if n.is_f64() => format_g(n.as_f64().unwrap_or_default()),
        Value::Array(items) if items.is_empty() => EMPTY.into(),
        Value::Array(items) => items
            .iter()
            .map(format_value)
            .collect::<Vec<_>>()
            .join(", "),
        other => Some(text(other))
            .filter(|t| !t.is_empty())
            .unwrap_or_else(|| EMPTY.into()),
    }
}

fn format_rows(rows: &[(String, String)]) -> Vec<String> {
    if rows.is_empty() {
        return vec![format!("  {EMPTY}")];
    }
    let width = rows
        .iter()
        .map(|(k, _)| k.chars().count())
        .max()
        .unwrap_or(0)
        .min(34);
    rows.iter()
        .map(|(k, v)| format!("  {k:<width$}  {v}"))
        .collect()
}

/// Python's `str()` of a JSON value.
fn text(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Python's truth value of a JSON value.
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(n) => n.as_f64().is_some_and(|n| n != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

/// Python's `format(value, "g")`: six significant digits, no trailing zeros.
fn format_g(value: f64) -> String {
    if !value.is_finite() {
        return if value.is_nan() {
            "nan"
        } else if value > 0.0 {
            "inf"
        } else {
            "-inf"
        }
        .into();
    }
    if value == 0.0 {
        return "0".into();
    }
    let scientific = format!("{value:.5e}");
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let trim = |s: &str| {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s.to_string()
        }
    };
    if (-4..6).contains(&exponent) {
        let decimals = usize::try_from(5 - exponent).unwrap_or(0);
        trim(&format!("{value:.decimals$}"))
    } else {
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", trim(mantissa), exponent.abs())
    }
}

// ------------------------------------------------------------------ helpers
fn env_var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn env_is_set(name: &str) -> bool {
    env_var(name).is_some()
}

/// Every spelling of the home directory, as the parts between its separators.
struct HomePattern {
    /// Longest home first.
    alternatives: Vec<Vec<String>>,
    /// File systems that ignore case: `C:\USERS\ALICE` is the same directory.
    ignore_case: bool,
}

fn is_separator(c: char) -> bool {
    c == '/' || c == '\\'
}

impl HomePattern {
    fn new(homes: impl IntoIterator<Item = String>) -> Option<HomePattern> {
        let mut homes: Vec<String> = homes
            .into_iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        homes.sort_by_key(|home| std::cmp::Reverse(home.chars().count()));
        let alternatives: Vec<Vec<String>> = homes
            .iter()
            .map(|home| split_separators(home))
            // "/" or "C:\" alone would redact everything
            .filter(|parts| parts.iter().map(|p| p.chars().count()).sum::<usize>() >= 2)
            .collect();
        (!alternatives.is_empty()).then_some(HomePattern {
            alternatives,
            ignore_case: cfg!(windows) || cfg!(target_os = "macos"),
        })
    }

    fn same(&self, a: char, b: char) -> bool {
        a == b || (self.ignore_case && a.to_lowercase().eq(b.to_lowercase()))
    }

    /// The end of the home directory spelled at `start`, if one is: either slash matches either
    /// separator ("C:/Users/..." from a UI toolkit), and a longer name that merely starts with
    /// the home path (`/home/al` in `/home/alice`) is no match.
    fn match_at(&self, chars: &[char], start: usize) -> Option<usize> {
        'alternatives: for parts in &self.alternatives {
            let mut at = start;
            for (n, part) in parts.iter().enumerate() {
                if n > 0 {
                    let run = at;
                    while chars.get(at).copied().is_some_and(is_separator) {
                        at += 1;
                    }
                    if at == run {
                        continue 'alternatives;
                    }
                }
                for expected in part.chars() {
                    if !chars.get(at).is_some_and(|c| self.same(*c, expected)) {
                        continue 'alternatives;
                    }
                    at += 1;
                }
            }
            // Not followed by a character that would continue the last name (alice2, alice.old).
            if chars
                .get(at)
                .is_some_and(|c| c.is_alphanumeric() || matches!(c, '_' | '.' | '-'))
            {
                continue;
            }
            return Some(at);
        }
        None
    }

    fn matches_start(&self, text: &str) -> bool {
        let chars: Vec<char> = text.chars().collect();
        self.match_at(&chars, 0).is_some()
    }

    fn replace_all(&self, text: &str) -> String {
        let chars: Vec<char> = text.chars().collect();
        let mut out = String::with_capacity(text.len());
        let mut at = 0;
        while at < chars.len() {
            match self.match_at(&chars, at) {
                Some(end) => {
                    out.push('~');
                    at = end;
                }
                None => {
                    out.push(chars[at]);
                    at += 1;
                }
            }
        }
        out
    }
}

/// `re.split(r"[\\/]+", path)`: the parts between runs of separators (`/a` gives `["", "a"]`).
fn split_separators(path: &str) -> Vec<String> {
    let mut parts = vec![String::new()];
    let mut in_run = false;
    for c in path.chars() {
        if is_separator(c) {
            if !in_run {
                parts.push(String::new());
            }
            in_run = true;
        } else {
            in_run = false;
            if let Some(last) = parts.last_mut() {
                last.push(c);
            }
        }
    }
    parts
}

#[cfg(test)]
thread_local! {
    /// The home directories [`home_pattern`] uses in a test (`Path.home` was monkeypatched).
    static TEST_HOMES: std::cell::RefCell<Option<Vec<String>>> = const { std::cell::RefCell::new(None) };
}

fn home_dirs() -> Vec<String> {
    #[cfg(test)]
    if let Some(homes) = TEST_HOMES.with(|h| h.borrow().clone()) {
        return homes;
    }
    let Some(dirs) = directories::BaseDirs::new() else {
        return Vec::new();
    };
    let home = dirs.home_dir();
    let mut homes = vec![home.to_string_lossy().into_owned()];
    // e.g. /home -> /var/home
    if let Ok(resolved) = home.canonicalize() {
        let resolved = resolved.to_string_lossy().into_owned();
        homes.push(
            resolved
                .strip_prefix(r"\\?\")
                .map_or(resolved.clone(), String::from),
        );
    }
    homes
}

fn home_pattern() -> Option<HomePattern> {
    HomePattern::new(home_dirs())
}

/// Replace the home directory with `~` wherever it appears in `value`.
///
/// Not only as a prefix: paths also turn up quoted (Explorer's "Copy as path" in *Pause while
/// these apps run*), inside OS error messages and with either slash.
fn redact(value: &str) -> String {
    home_pattern().map_or_else(|| value.to_string(), |p| p.replace_all(value))
}

fn redact_path(path: &Path) -> String {
    redact(&path.to_string_lossy())
}

/// `value` with [`redact`] applied to every string in nested objects and arrays.
fn redact_tree(value: &Value) -> Value {
    let Some(pattern) = home_pattern() else {
        return value.clone();
    };
    fn walk(item: &Value, pattern: &HomePattern) -> Value {
        match item {
            Value::String(s) => json!(pattern.replace_all(s)),
            Value::Array(items) => Value::Array(items.iter().map(|i| walk(i, pattern)).collect()),
            Value::Object(map) => Value::Object(
                map.iter()
                    .map(|(k, v)| (k.clone(), walk(v, pattern)))
                    .collect(),
            ),
            other => other.clone(),
        }
    }
    walk(value, &pattern)
}

/// A camera setting as a report may show it: no directories, no serial number.
fn describe_device(device: &str) -> String {
    let spec = device.trim();
    if !spec.is_empty() && spec.chars().all(|c| c.is_ascii_digit()) {
        return format!("camera {spec}");
    }
    if spec.is_empty() {
        return "none".into();
    }
    let mut name = Path::new(spec)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    if spec.starts_with("/dev/") {
        // /dev/videoN or a udev link; by-id names end in the camera's serial.
        if spec.contains("/by-id/") {
            name = strip_serial(&name);
        }
        return format!("device {name}");
    }
    format!("file {name}")
}

/// `usb-<vendor>_<model>_<serial>-video-index0` -> `usb-<vendor>_<model>_*-video-index0`.
fn strip_serial(name: &str) -> String {
    let serial_end = name.rfind("-video-index").filter(|end| {
        let index = &name[end + "-video-index".len()..];
        !index.is_empty() && index.chars().all(|c| c.is_ascii_digit())
    });
    let Some(end) = serial_end else {
        return name.to_string();
    };
    match name[..end].rfind('_') {
        Some(underscore) if underscore + 1 < end => {
            format!("{}_*{}", &name[..underscore], &name[end..])
        }
        _ => name.to_string(),
    }
}

// ======================================================================== bench
/// The benchmark could not run.
#[derive(Debug, Error)]
pub enum BenchError {
    /// No frames, the camera is unavailable, ...
    #[error("{0}")]
    Failed(String),
    /// An argument is out of range.
    #[error("{0}")]
    Invalid(String),
    /// Invalid device.
    #[error(transparent)]
    Camera(#[from] CameraError),
    /// No usable backend.
    #[error(transparent)]
    Backend(#[from] BackendUnavailable),
}

/// What [`run_bench`] takes besides the duration, the device and the backend.
#[derive(Debug, Clone)]
pub struct BenchOptions {
    pub width: usize,
    pub height: usize,
    pub api: String,
    /// The idle mode's rate; the balanced profile's idle rate when `None`.
    pub idle_fps: Option<f64>,
}

impl Default for BenchOptions {
    fn default() -> Self {
        BenchOptions {
            width: 640,
            height: 480,
            api: "auto".into(),
            idle_fps: None,
        }
    }
}

/// Measure the capture + analysis loop without any UI.
///
/// Two modes run for `seconds` each:
///
/// * `max`: every frame is analysed as fast as possible (raw throughput);
/// * `idle`: the balanced profile's idle rate with the motion gate on, which is how the app
///   runs while you sit still in front of one monitor.
///
/// `device` is a camera index or an image/frame-directory path. Reports per mode: fps,
/// analysed fps, inference ms (p50/p95/mean), process CPU as a share of the whole machine and of
/// one core, and how often a face was seen.
pub fn run_bench(
    seconds: f64,
    device: &str,
    backend: &str,
    options: &BenchOptions,
) -> Result<Value, BenchError> {
    if seconds.is_nan() || seconds <= 0.0 {
        return Err(BenchError::Invalid("seconds must be positive".into()));
    }
    let balanced_idle = PROFILES
        .iter()
        .find(|(name, _)| *name == DEFAULT_PROFILE)
        .map_or(4.0, |(_, rates)| rates.idle);
    let rate = options
        .idle_fps
        .filter(|f| *f != 0.0)
        .unwrap_or(balanced_idle);
    let mut source = open_source(
        device,
        options.width,
        options.height,
        &options.api,
        15,
        true,
    )?;
    if !source.open() {
        return Err(BenchError::Failed(source.last_error().unwrap_or_else(
            || format!("Could not open {}", describe_device(device)),
        )));
    }
    let result = create_backend(backend, 1)
        .map_err(BenchError::from)
        .and_then(|mut engine| {
            let result = bench_with(&mut source, engine.as_mut(), seconds, rate);
            engine.close();
            result.map(|(frame_size, modes)| {
                json!({
                    "backend": engine.name(),
                    "feature_version": engine.feature_version(),
                    "device": describe_device(device),
                    "frame_size": frame_size,
                    "cpu_count": cpu_count(),
                    "seconds_per_mode": seconds,
                    "modes": modes,
                })
            })
        });
    source.release();
    result
}

fn cpu_count() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
}

/// The frame size and the `max` and `idle` measurements.
fn bench_with(
    source: &mut dyn FrameSource,
    engine: &mut dyn VisionBackend,
    seconds: f64,
    idle_rate: f64,
) -> Result<([usize; 2], Value), BenchError> {
    let mut stamp = Timestamps::default();
    let mut frame = None;
    // The first inferences are slow (lazy initialisation); keep them out.
    for _ in 0..3 {
        frame = source.read();
        if let Some(frame) = &frame {
            engine.process(frame, stamp.next());
        }
    }
    let Some(frame) = frame else {
        return Err(BenchError::Failed(
            source
                .last_error()
                .unwrap_or_else(|| "The video source delivered no frames".into()),
        ));
    };
    let frame_size = [frame.width, frame.height];
    let max = bench_mode(source, engine, seconds, None, None, &mut stamp)?;
    let idle = bench_mode(
        source,
        engine,
        seconds,
        Some(idle_rate),
        Some(MotionGate::default()),
        &mut stamp,
    )?;
    Ok((frame_size, json!({"max": max, "idle": idle})))
}

/// Strictly increasing timestamps for the backend (coarse clocks repeat values).
#[derive(Debug)]
struct Timestamps {
    last: f64,
}

impl Default for Timestamps {
    fn default() -> Self {
        Timestamps {
            last: f64::NEG_INFINITY,
        }
    }
}

impl Timestamps {
    fn next(&mut self) -> f64 {
        self.last = crate::panes::worker::monotonic().max(self.last + 1e-3);
        self.last
    }
}

/// CPU time this process has used so far, in seconds.
fn process_cpu_seconds(system: &mut System, pid: Pid) -> f64 {
    system.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::nothing().with_cpu(),
    );
    system
        .process(pid)
        .map_or(0.0, |p| p.accumulated_cpu_time() as f64 / 1000.0)
}

fn bench_mode(
    source: &mut dyn FrameSource,
    engine: &mut dyn VisionBackend,
    seconds: f64,
    target_fps: Option<f64>,
    mut gate: Option<MotionGate>,
    stamp: &mut Timestamps,
) -> Result<Value, BenchError> {
    let period = target_fps.filter(|f| *f != 0.0).map_or(0.0, |f| 1.0 / f);
    let mut inference_ms = Vec::new();
    let mut read_ms = Vec::new();
    let (mut frames, mut analysed, mut faces, mut failures) = (0_u64, 0_u64, 0_u64, 0_u32);
    let mut have_observation = false;

    let mut system = System::new();
    let pid = Pid::from_u32(std::process::id());
    let cpu_start = process_cpu_seconds(&mut system, pid);
    let start = Instant::now();
    let mut next_slot = 0.0;
    loop {
        let now = start.elapsed().as_secs_f64();
        if now >= seconds {
            break;
        }
        if period > 0.0 {
            if now < next_slot {
                std::thread::sleep(Duration::from_secs_f64((next_slot - now).min(0.05)));
                continue;
            }
            // A fixed schedule; after a stall, restart it rather than burst.
            next_slot = (next_slot + period).max(now);
        }
        let read_start = Instant::now();
        let frame = source.read();
        read_ms.push(read_start.elapsed().as_secs_f64() * 1000.0);
        let Some(frame) = frame else {
            failures += 1;
            if failures >= 5 {
                return Err(BenchError::Failed(source.last_error().unwrap_or_else(
                    || "The video source stopped delivering frames".into(),
                )));
            }
            continue;
        };
        failures = 0;
        frames += 1;
        if let Some(gate) = gate.as_mut()
            && have_observation
            && !gate.should_process(&frame, now)
        {
            continue;
        }
        let inference_start = Instant::now();
        let observation = engine.process(&frame, stamp.next());
        inference_ms.push(inference_start.elapsed().as_secs_f64() * 1000.0);
        analysed += 1;
        have_observation = true;
        if observation.face_count > 0 {
            faces += 1;
        }
        if let Some(gate) = gate.as_mut() {
            gate.mark_processed(&frame, now, observation.face_box);
        }
    }
    let elapsed = start.elapsed().as_secs_f64().max(1e-9);
    let cpu_seconds = process_cpu_seconds(&mut system, pid) - cpu_start;
    let cores = cpu_count() as f64;
    let ratio = |count: u64, of: u64| (of > 0).then(|| round(count as f64 / of as f64, 3));

    Ok(json!({
        "target_fps": target_fps,
        "seconds": round(elapsed, 3),
        "frames": frames,
        "analysed": analysed,
        "fps": round(frames as f64 / elapsed, 2),
        "analysed_fps": round(analysed as f64 / elapsed, 2),
        "skip_ratio": (frames > 0).then(|| round(1.0 - analysed as f64 / frames as f64, 3)),
        "inference_ms": stats(&inference_ms),
        "read_ms": stats(&read_ms),
        "cpu_percent": round(100.0 * cpu_seconds / elapsed / cores, 2),
        "cpu_percent_core": round(100.0 * cpu_seconds / elapsed, 2),
        "face_ratio": ratio(faces, analysed),
    }))
}

fn round(value: f64, digits: i32) -> f64 {
    let scale = 10_f64.powi(digits);
    (value * scale).round() / scale
}

/// `numpy.percentile(sorted, q)` (linear interpolation).
fn percentile(sorted: &[f64], q: f64) -> f64 {
    let position = (sorted.len() - 1) as f64 * q / 100.0;
    let (low, high) = (position.floor() as usize, position.ceil() as usize);
    sorted[low] + (sorted[high] - sorted[low]) * (position - low as f64)
}

fn stats(values: &[f64]) -> Value {
    if values.is_empty() {
        return json!({"p50": null, "p95": null, "mean": null});
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    json!({
        "p50": round(percentile(&sorted, 50.0), 3),
        "p95": round(percentile(&sorted, 95.0), 3),
        "mean": round(sorted.iter().sum::<f64>() / sorted.len() as f64, 3),
    })
}

/// Render a [`run_bench`] result as a small table.
pub fn format_bench(result: &Value) -> String {
    let modes = result.get("modes").unwrap_or(&Value::Null);
    let columns: Vec<&Value> = ["max", "idle"]
        .iter()
        .filter_map(|name| modes.get(name))
        .collect();
    let header = |mode: &Value| match mode.get("target_fps").and_then(Value::as_f64) {
        None => "max speed".to_string(),
        Some(target) => format!("idle ({} fps)", format_g(target)),
    };
    let number = |mode: &Value, key: &str| mode.get(key).and_then(Value::as_f64);
    let ms = |stats: Option<&Value>| match stats.and_then(|s| {
        s.get("p50")
            .and_then(Value::as_f64)
            .zip(s.get("p95").and_then(Value::as_f64))
    }) {
        Some((p50, p95)) => format!("{p50:.1} / {p95:.1} ms"),
        None => EMPTY.to_string(),
    };
    let pct = |value: Option<f64>| value.map_or_else(|| EMPTY.to_string(), |v| format!("{v:.1} %"));
    let per_second = |mode: &Value, key: &str| {
        number(mode, key).map_or_else(|| EMPTY.to_string(), |v| format!("{v:.1}"))
    };

    let rows: Vec<(&str, Vec<String>)> = vec![
        (
            "frames per second",
            columns.iter().map(|m| per_second(m, "fps")).collect(),
        ),
        (
            "analysed per second",
            columns
                .iter()
                .map(|m| per_second(m, "analysed_fps"))
                .collect(),
        ),
        (
            "inference p50 / p95",
            columns.iter().map(|m| ms(m.get("inference_ms"))).collect(),
        ),
        (
            "CPU, whole machine",
            columns
                .iter()
                .map(|m| pct(number(m, "cpu_percent")))
                .collect(),
        ),
        (
            "CPU, % of one core",
            columns
                .iter()
                .map(|m| pct(number(m, "cpu_percent_core")))
                .collect(),
        ),
        (
            "face seen",
            columns
                .iter()
                .map(|m| pct(number(m, "face_ratio").map(|r| 100.0 * r)))
                .collect(),
        ),
    ];
    let size = result.get("frame_size").and_then(Value::as_array);
    let dimension = |i: usize| size.and_then(|s| s.get(i)).map_or_else(|| "?".into(), text);
    let field = |key: &str| result.get(key).map_or_else(|| "None".into(), text);
    let mut lines = vec![
        format!(
            "{APP_NAME} benchmark - {} backend ({}), {} at {}x{}, {} logical CPUs",
            field("backend"),
            field("feature_version"),
            field("device"),
            dimension(0),
            dimension(1),
            field("cpu_count"),
        ),
        String::new(),
        format!(
            "{:<22}{}",
            "",
            columns
                .iter()
                .map(|m| format!("{:<18}", header(m)))
                .collect::<String>()
        ),
    ];
    for (label, values) in rows {
        let cells: String = values.iter().map(|v| format!("{v:<18}")).collect();
        lines.push(format!("{label:<22}{cells}"));
    }
    let mut out = lines
        .iter()
        .map(|l| l.trim_end())
        .collect::<Vec<_>>()
        .join("\n");
    out.push('\n');
    out
}

#[cfg(test)]
#[path = "diagnostics_tests.rs"]
mod tests;
