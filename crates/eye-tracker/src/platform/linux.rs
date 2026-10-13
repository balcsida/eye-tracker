//! Linux integration for X11 and Wayland sessions.
//!
//! # Design
//!
//! Linux has no single desktop API, so every feature is a short list of strategies tried in
//! order, gated by what the session actually offers:
//!
//! * **X11** gets the full feature set in-process: idle time from the MIT-SCREEN-SAVER
//!   extension, window focus through EWMH and display power through the DPMS extension (all
//!   through `x11rb`), cursor warping through the UI toolkit.
//! * **Wayland** forbids global window inspection and pointer warping by design. The app still
//!   runs (winit is steered to XWayland so calibration windows can be positioned, see
//!   [`LinuxPlatform::prepare_process`]), presence and screen locking work through logind and
//!   desktop-specific tools, the cursor moves through the compositor's own IPC (sway, Hyprland)
//!   or `ydotool`, and window focus is reported as unsupported.
//!
//! Queries that are polled (idle time, lock state) go through D-Bus in-process (`zbus`, blocking
//! API) and fall back to `gdbus`/`loginctl`. Desktop tools are run with `std::process` - never
//! through a shell - and a short timeout. Everything polled is cached so a busy tick never turns
//! into a stream of process launches, and the camera-in-use scan runs on a background thread.
//!
//! `prepare_process` sets two variables for winit: `WINIT_X11_SCALE_FACTOR=1` (native pixels,
//! nothing scaled) and, on Wayland with XWayland available, hides `WAYLAND_DISPLAY` (and
//! `WAYLAND_SOCKET`) from this process so winit picks X11. Desktop tools started by this module
//! get the session's original values back.
//!
//! # Lock detection
//!
//! logind's `LockedHint` is only set by lockers that report it (GNOME Shell, KDE, light-locker
//! ...). Tiling-WM lockers (swaylock, i3lock via xss-lock, hyprlock, slock ...) never do, so a
//! running locker process of this user's login counts as "locked" as well (setuid lockers such as
//! slock included, see [`ProcFs::locker_running`]), and `loginctl lock-session` only counts as a
//! successful lock when something reacted to it (see [`LinuxPlatform::lock_screen`]). Tiling
//! compositors that pose as GNOME or KDE through `XDG_CURRENT_DESKTOP` are recognised by their
//! IPC sockets.
//!
//! # Camera use by other apps
//!
//! A background thread looks for processes holding a physical `/dev/video*` node
//! (`/proc/<pid>/fd`). With fanotify (unprivileged since Linux 5.13) it is woken by every open and
//! close of a camera node and can tell our own from other processes'. V4L2 lets only one process
//! stream, so another app that opens the camera read-write while we stream and closes it again at
//! once (a browser joining a call gets EBUSY) was refused; the camera is then reported busy for a
//! grace period so that app can retry. With inotify only (no process attribution) holders are
//! still found as soon as they open the device; without either, the scan is polled. Apps that
//! capture through PipeWire's camera portal never open the node themselves and cannot be seen;
//! `pause_for_apps` covers those.
//!
//! # Cursor on Wayland
//!
//! `ydotool` (1.x, with `ydotoold` running) moves the pointer through a virtual uinput mouse. Its
//! "absolute" move is emulated with relative motion, so the compositor's pointer acceleration
//! distorts it unless the acceleration profile is flat. The injected motion also resets GNOME's
//! idle timer; such resets are recognised and not reported as user input.
//!
//! Every public method degrades to `None`/`false` instead of panicking.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::ffi::CString;
use std::io::{self, Read};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex, RwLock};
use x11rb::connection::{Connection as _, RequestConnection as _};
use x11rb::cookie::VoidCookie;
use x11rb::errors::{ConnectionError, ReplyError};
use x11rb::protocol::dpms::{ConnectionExt as _, DPMSMode};
use x11rb::protocol::screensaver::{self, ConnectionExt as _};
use x11rb::protocol::xproto::{
    AtomEnum, ClientMessageEvent, ConnectionExt as _, EventMask, MOTION_NOTIFY_EVENT,
};
use x11rb::protocol::xtest::ConnectionExt as _;
use x11rb::rust_connection::RustConnection;
use x11rb::wrapper::ConnectionExt as _;

#[cfg(test)]
use super::base::GenericPlatform;
use super::base::{PlatformServices, which};
use crate::panes::providers::command::CommandResult;
use crate::panes::worker::monotonic;
use crate::types::{AppIdentity, Rect, WindowHandle, WindowRef};

/// Filesystem roots (the tests point [`LinuxOptions`] at a fake tree).
pub const PROC_ROOT: &str = "/proc";
pub const DEV_ROOT: &str = "/dev";
pub const SYS_ROOT: &str = "/sys";

const ACTION_TIMEOUT_S: f64 = 5.0; // lock / display power: user-visible, may legitimately take a moment
const QUERY_TIMEOUT_S: f64 = 2.0; // polled queries must never stall a tick for long
const IDLE_QUERY_TIMEOUT_S: f64 = 1.0; // the idle time is read on every tick
const DBUS_TIMEOUT_MS: u64 = 500;
const LOCKED_HINT_TTL_S: f64 = 2.0;
const IDLE_TTL_S: f64 = 0.5;
const RETRY_AFTER_FAILURE_S: f64 = 60.0;
/// Retry delay after a failed idle query on a desktop where it worked before: a busy
/// gnome-shell must not hide the user's typing for a whole minute.
const IDLE_TRANSIENT_RETRY_S: f64 = 1.0;
const SESSION_RETRY_S: f64 = 30.0;
/// How long `loginctl lock-session` may take to make a locker appear.
const LOCK_CONFIRM_S: f64 = 2.0;
const LOCK_CONFIRM_POLL_S: f64 = 0.2;
/// An idle-timer reset this close to our own `ydotool` motion was caused by it.
const INJECTION_WINDOW_S: f64 = 0.5;
/// While displays are off with DPMS enabled only for that, how often and how long to watch for
/// them waking up (so DPMS can be switched back off).
const DPMS_WATCH_INTERVAL_S: f64 = 1.0;
const DPMS_WATCH_MAX_S: f64 = 12.0 * 3600.0;

// X11 protocol constants (<X11/X.h>).
const X_CURRENT_TIME: u32 = 0;
const X_IS_VIEWABLE: u8 = 2;
const X_SUBSTRUCTURE_NOTIFY_MASK: u32 = 1 << 19;
const X_SUBSTRUCTURE_REDIRECT_MASK: u32 = 1 << 20;
const DPMS_MODE_ON: u16 = 0;
const DPMS_MODE_OFF: u16 = 3;

/// `_NET_ACTIVE_WINDOW` source indication "pager": window managers apply no focus-stealing
/// prevention to requests that come from pagers and taskbars.
const EWMH_SOURCE_PAGER: u32 = 2;

/// Window types that are part of the desktop shell, not user windows.
const SKIPPED_WINDOW_TYPES: [&str; 4] = [
    "_NET_WM_WINDOW_TYPE_DESKTOP",
    "_NET_WM_WINDOW_TYPE_DOCK",
    "_NET_WM_WINDOW_TYPE_SPLASH",
    "_NET_WM_WINDOW_TYPE_NOTIFICATION",
];

/// Processes that keep camera nodes open to *monitor* them. They only count as "using the
/// camera" while they stream (see [`maps_video_device`]).
const CAMERA_BROKERS: [&str; 2] = ["pipewire", "wireplumber"];

/// Screen lockers that run exactly while the screen is locked and do not report it to logind.
/// `/proc/<pid>/comm` holds at most 15 characters (see [`is_locker_comm`]).
const LOCKER_COMMS: [&str; 13] = [
    "i3lock",
    "swaylock",
    "hyprlock",
    "xsecurelock",
    "slock",
    "gtklock",
    "waylock",
    "xlock",
    "alock",
    "sflock",
    "xtrlock",
    "kscreenlocker_greet",
    "xscreensaver-auth",
];

/// Desktops whose own locker reacts to logind's `Lock` signal, so `loginctl lock-session` is
/// trusted there without further evidence.
const LOGIND_LOCK_DESKTOPS: [&str; 9] = [
    "gnome", "kde", "plasma", "cinnamon", "mate", "xfce", "budgie", "unity", "deepin",
];

/// Distributions that brand GNOME Shell sessions with their own name in front (`ubuntu:GNOME`,
/// `pop:GNOME`). Other desktops that list GNOME second do so for compatibility only: Budgie
/// (`Budgie:GNOME`) locks with budgie-screensaver, a gnome-screensaver 3.6 fork that never
/// reports to logind's `LockedHint`.
const GNOME_SHELL_VENDORS: [&str; 4] = ["ubuntu", "pop", "zorin", "endless"];

/// Variables set by tiling window managers / compositors for their IPC (sway also sets
/// `I3SOCK`). See [`standalone_compositor`].
const COMPOSITOR_SOCKET_VARS: [&str; 5] = [
    "SWAYSOCK",
    "I3SOCK",
    "HYPRLAND_INSTANCE_SIGNATURE",
    "NIRI_SOCKET",
    "WAYFIRE_SOCKET",
];

/// D-Bus errors meaning "this service / method does not exist here" (as opposed to a timeout or
/// a busy service), in D-Bus error names and gdbus messages.
const DBUS_MISSING_MARKERS: [&str; 6] = [
    "ServiceUnknown",
    "UnknownMethod",
    "UnknownObject",
    "UnknownInterface",
    "NameHasNoOwner",
    "was not provided by any .service files",
];

const MUTTER_IDLE_SERVICE: &str = "org.gnome.Mutter.IdleMonitor";
const MUTTER_IDLE_PATH: &str = "/org/gnome/Mutter/IdleMonitor/Core";
const MUTTER_IDLE: [&str; 9] = [
    "gdbus",
    "call",
    "--session",
    "--dest",
    MUTTER_IDLE_SERVICE,
    "--object-path",
    MUTTER_IDLE_PATH,
    "--method",
    "org.gnome.Mutter.IdleMonitor.GetIdletime",
];
const LOGIN1: &str = "org.freedesktop.login1";
const LOGIN1_PATH: &str = "/org/freedesktop/login1";
const LOGIN1_MANAGER: &str = "org.freedesktop.login1.Manager";
const LOGIN1_SESSION: &str = "org.freedesktop.login1.Session";
const DBUS_PROPERTIES: &str = "org.freedesktop.DBus.Properties";

/// Where `ydotoold` listens when `YDOTOOL_SOCKET` is not set (besides
/// `$XDG_RUNTIME_DIR/.ydotool_socket`).
const YDOTOOL_DEFAULT_SOCKETS: &[&str] = &["/tmp/.ydotool_socket"];

/// `/proc/<pid>/loginuid` of a process that did not come from a login: `(uid_t)-1`.
const AUDIT_UID_UNSET: u64 = 0xFFFF_FFFF;

/// Seconds on a monotonic clock (injectable for tests).
pub(crate) type Clock = Arc<dyn Fn() -> f64 + Send + Sync>;
/// Sleeps for the given seconds (injectable for tests).
pub(crate) type Sleep = Arc<dyn Fn(f64) + Send + Sync>;
/// Environment variables by name.
pub(crate) type EnvMap = HashMap<String, String>;
/// Physical camera nodes (`/dev/videoN`); `None` when unknown.
type Devices = Option<HashSet<String>>;

// ---------------------------------------------------------------------------
// pure helpers
// ---------------------------------------------------------------------------
fn var<'a>(env: &'a EnvMap, key: &str) -> &'a str {
    env.get(key).map_or("", String::as_str)
}

fn command(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|part| (*part).to_string()).collect()
}

/// How a command is named in logs: its first two words.
fn command_name(argv: &[String]) -> String {
    argv.iter().take(2).cloned().collect::<Vec<_>>().join(" ")
}

/// The real user id.
fn getuid() -> Option<u32> {
    // SAFETY: getuid has no preconditions and cannot fail.
    Some(unsafe { libc::getuid() })
}

/// Whether `name` matches `video\d+`.
fn is_video_node(name: &str) -> bool {
    name.strip_prefix("video")
        .is_some_and(|digits| !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()))
}

/// Whether a session id matches `[A-Za-z0-9_.-]+`.
fn valid_session_id(text: &str) -> bool {
    !text.is_empty()
        && text
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-'))
}

/// Parse a single unsigned integer from `gdbus call` output, e.g. `(uint64 12345,)`.
///
/// The first match of `\(\s*(?:u?int(?:16|32|64)\s+)?(\d+)\s*,?\s*\)`.
fn parse_gdbus_uint(text: Option<&str>) -> Option<u64> {
    let text = text.unwrap_or("");
    text.match_indices('(')
        .find_map(|(start, _)| gdbus_uint_at(&text[start + 1..]))
}

fn gdbus_uint_at(rest: &str) -> Option<u64> {
    let mut rest = rest.trim_start();
    let unsigned = rest.strip_prefix('u').unwrap_or(rest);
    if let Some(after) = unsigned.strip_prefix("int") {
        for bits in ["16", "32", "64"] {
            if let Some(tail) = after.strip_prefix(bits) {
                let trimmed = tail.trim_start();
                if trimmed.len() < tail.len() {
                    rest = trimmed; // the type needs whitespace before the number
                }
                break;
            }
        }
    }
    let digits = rest.len() - rest.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    if digits == 0 {
        return None;
    }
    let value = rest[..digits].parse().ok()?;
    let tail = rest[digits..].trim_start();
    let tail = tail.strip_prefix(',').unwrap_or(tail).trim_start();
    tail.starts_with(')').then_some(value)
}

/// Parse `loginctl show-session -p LockedHint [--value]` output.
fn parse_locked_hint(text: Option<&str>) -> Option<bool> {
    let first = text.unwrap_or("").trim().lines().next()?;
    let mut value = first.trim();
    if let Some((_, after)) = value.split_once('=') {
        value = after.trim();
    }
    match value.to_lowercase().as_str() {
        "yes" => Some(true),
        "no" => Some(false),
        _ => None,
    }
}

/// `(capable, enabled)` from `xset q` output; `None` when it does not say.
fn parse_xset_dpms(text: Option<&str>) -> Option<(bool, bool)> {
    let lowered = text.unwrap_or("").to_lowercase();
    if lowered.contains("does not have the dpms extension")
        || lowered.contains("not capable of dpms")
    {
        return Some((false, false));
    }
    let start = lowered.find("dpms is ")? + "dpms is ".len();
    let state = &lowered[start..];
    if state.starts_with("enabled") {
        Some((true, true))
    } else if state.starts_with("disabled") {
        Some((true, false))
    } else {
        None
    }
}

/// Whether a D-Bus error says the service or method does not exist (vs. a hiccup).
fn is_missing_service(text: &str) -> bool {
    DBUS_MISSING_MARKERS
        .iter()
        .any(|marker| text.contains(marker))
}

/// Lower-cased desktop names from `XDG_CURRENT_DESKTOP` and friends.
fn desktop_tokens(env: &EnvMap) -> HashSet<String> {
    [
        "XDG_CURRENT_DESKTOP",
        "XDG_SESSION_DESKTOP",
        "DESKTOP_SESSION",
    ]
    .iter()
    .flat_map(|key| var(env, key).split([':', ';']))
    .map(|token| token.trim().to_lowercase())
    .filter(|token| !token.is_empty())
    .collect()
}

fn is_wayland_env(env: &EnvMap) -> bool {
    var(env, "XDG_SESSION_TYPE").trim().to_lowercase() == "wayland"
        || !var(env, "WAYLAND_DISPLAY").is_empty()
}

fn is_kde_env(env: &EnvMap) -> bool {
    !var(env, "KDE_FULL_SESSION").is_empty()
        || desktop_tokens(env)
            .iter()
            .any(|token| token.contains("kde") || token.contains("plasma"))
}

fn is_gnome_env(env: &EnvMap) -> bool {
    // Covers "GNOME", "ubuntu:GNOME", "pop:GNOME", "GNOME-Classic", "Budgie:GNOME" …
    desktop_tokens(env)
        .iter()
        .any(|token| token.contains("gnome"))
}

/// Whether a tiling window manager / compositor runs the session.
///
/// Such sessions often export `XDG_CURRENT_DESKTOP=GNOME` (or KDE) so that portals and settings
/// apps work; their own sockets tell the truth, and then no GNOME or KDE lock screen exists to
/// react to logind or report the lock.
fn standalone_compositor(env: &EnvMap) -> bool {
    COMPOSITOR_SOCKET_VARS
        .iter()
        .any(|key| !var(env, key).is_empty())
}

/// Whether the desktop's own locker reacts to `loginctl lock-session`.
fn desktop_handles_logind_lock(env: &EnvMap) -> bool {
    if standalone_compositor(env) {
        return false;
    }
    if !var(env, "KDE_FULL_SESSION").is_empty() {
        return true;
    }
    desktop_tokens(env)
        .iter()
        .any(|token| LOGIND_LOCK_DESKTOPS.iter().any(|name| token.contains(name)))
}

/// The session's own desktop: the first `XDG_CURRENT_DESKTOP` entry (later ones are
/// compatibility fallbacks), else the session name; lower-cased.
fn session_desktop(env: &EnvMap) -> String {
    [
        "XDG_CURRENT_DESKTOP",
        "XDG_SESSION_DESKTOP",
        "DESKTOP_SESSION",
    ]
    .iter()
    .flat_map(|key| var(env, key).split([':', ';']))
    .map(str::trim)
    .find(|token| !token.is_empty())
    .map(str::to_lowercase)
    .unwrap_or_default()
}

/// Whether GNOME Shell (or GNOME Flashback) runs the session, as opposed to a desktop that
/// merely lists GNOME as a fallback (see [`GNOME_SHELL_VENDORS`]).
fn runs_gnome_shell(env: &EnvMap) -> bool {
    let own = session_desktop(env);
    own.contains("gnome") || GNOME_SHELL_VENDORS.contains(&own.as_str())
}

/// Desktops whose lock screen always reports itself through logind's `LockedHint`.
///
/// Only these may have a locker's claimed success checked against the hint; elsewhere a lock
/// that the hint never shows would be taken for a failure, and the next locker would lock the
/// user a second time.
fn locked_hint_authoritative(env: &EnvMap) -> bool {
    if standalone_compositor(env) {
        return false;
    }
    is_kde_env(env) || runs_gnome_shell(env)
}

/// Screen-lock strategies, most universal first.
fn lock_commands(session_id: Option<&str>) -> Vec<Vec<String>> {
    let mut commands = vec![
        // logind asks whichever locker the desktop registered (GNOME, KDE, Cinnamon, xss-lock
        // for tiling WMs …). Without an id it targets the caller's session.
        match session_id {
            Some(id) => command(&["loginctl", "lock-session", id]),
            None => command(&["loginctl", "lock-session"]),
        },
        command(&["xdg-screensaver", "lock"]),
        command(&["gnome-screensaver-command", "-l"]),
        command(&["dm-tool", "lock"]),
        command(&["xflock4"]),
    ];
    // KDE's qdbus is packaged under several names depending on distro and Qt version.
    for tool in ["qdbus", "qdbus6", "qdbus-qt6", "qdbus-qt5"] {
        commands.push(command(&[
            tool,
            "org.freedesktop.ScreenSaver",
            "/ScreenSaver",
            "Lock",
        ]));
    }
    commands.push(command(&[
        "gdbus",
        "call",
        "--session",
        "--dest",
        "org.freedesktop.ScreenSaver",
        "--object-path",
        "/ScreenSaver",
        "--method",
        "org.freedesktop.ScreenSaver.Lock",
    ]));
    commands
}

/// How `doctor` names a [`lock_commands`] entry (ASCII, like the report).
///
/// The D-Bus clients are named with the call they make; "gdbus call" alone would not say what
/// it does.
fn lock_method_label(argv: &[String]) -> String {
    if argv[0] == "gdbus" || argv[0].starts_with("qdbus") {
        return format!("{} org.freedesktop.ScreenSaver.Lock", argv[0]);
    }
    command_name(argv)
}

/// Desktop-specific display power strategies, in the order they are tried.
///
/// X11 DPMS is handled separately ([`LinuxPlatform::x11_dpms`]): under XWayland it would only
/// affect a virtual output, and `xset` needs care.
fn display_power_commands(env: &EnvMap, on: bool) -> Vec<Vec<String>> {
    let wayland = is_wayland_env(env);
    let state = if on { "on" } else { "off" };
    let mut commands = Vec::new();
    if wayland && is_kde_env(env) {
        commands.push(command(&["kscreen-doctor", "--dpms", state]));
    }
    if is_gnome_env(env) {
        let mode = if on { "0" } else { "1" }; // Mutter PowerSaveMode: 0 = on, 1 = standby
        commands.push(command(&[
            "busctl",
            "--user",
            "set-property",
            "org.gnome.Mutter.DisplayConfig",
            "/org/gnome/Mutter/DisplayConfig",
            "org.gnome.Mutter.DisplayConfig",
            "PowerSaveMode",
            "i",
            mode,
        ]));
        commands.push(command(&[
            "gdbus",
            "call",
            "--session",
            "--dest",
            "org.gnome.Mutter.DisplayConfig",
            "--object-path",
            "/org/gnome/Mutter/DisplayConfig",
            "--method",
            "org.freedesktop.DBus.Properties.Set",
            "org.gnome.Mutter.DisplayConfig",
            "PowerSaveMode",
            &format!("<int32 {mode}>"),
        ]));
    }
    if wayland && !var(env, "SWAYSOCK").is_empty() {
        // "power" replaced "dpms" in sway 1.8; older versions only know "dpms".
        commands.push(command(&["swaymsg", "output", "*", "power", state]));
        commands.push(command(&["swaymsg", "output", "*", "dpms", state]));
    }
    if wayland && !var(env, "HYPRLAND_INSTANCE_SIGNATURE").is_empty() {
        commands.push(command(&["hyprctl", "dispatch", "dpms", state]));
    }
    commands
}

/// Whether `ydotoold` appears to be running (its socket exists).
///
/// ydotool 1.x cannot inject anything without the daemon; the client looks at
/// `$YDOTOOL_SOCKET`, else `$XDG_RUNTIME_DIR/.ydotool_socket` or `/tmp/.ydotool_socket`
/// depending on its version (`default_sockets`).
fn ydotool_socket_present(env: &EnvMap, default_sockets: &[&str]) -> bool {
    let explicit = var(env, "YDOTOOL_SOCKET").trim();
    let mut candidates: Vec<PathBuf> = Vec::new();
    if !explicit.is_empty() {
        candidates.push(PathBuf::from(explicit));
    } else {
        let runtime = var(env, "XDG_RUNTIME_DIR").trim();
        if !runtime.is_empty() {
            candidates.push(Path::new(runtime).join(".ydotool_socket"));
        }
        candidates.extend(default_sockets.iter().map(PathBuf::from));
    }
    candidates.iter().any(|path| path.exists())
}

fn pointer_command_ok(argv: &[String], result: &CommandResult) -> bool {
    if result.returncode != 0 {
        return false;
    }
    if argv[0] == "hyprctl" {
        // hyprctl exits with 0 even when the dispatcher failed; it prints "ok" on success.
        let out = result.stdout.trim().to_lowercase();
        return out.is_empty() || out == "ok";
    }
    true
}

fn read_text(path: &Path) -> String {
    std::fs::read(path)
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
        .unwrap_or_default()
}

/// The audit login UID of a process; `None` when unset or unreadable.
fn login_uid(proc_dir: &Path) -> Option<u32> {
    let text = read_text(&proc_dir.join("loginuid"));
    let text = text.trim();
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let value: u64 = text.parse().ok()?;
    if value == AUDIT_UID_UNSET {
        return None;
    }
    u32::try_from(value).ok()
}

/// Whether the process at `proc_dir` belongs to this user (see [`ProcFs::locker_running`]).
///
/// It does when this user owns its `/proc` entry, or when its audit login UID is ours
/// (`own_login`, or `uid` when this process has none): the latter survives the setuid tricks of
/// slock & co. A login UID that is unset on either side proves nothing, so then only the owner
/// counts.
fn belongs_to_user(proc_dir: &Path, own_login: Option<u32>, uid: Option<u32>) -> bool {
    let Some(uid) = uid else {
        return true; // no users to tell apart
    };
    if login_uid(proc_dir).is_some_and(|login| login == own_login.unwrap_or(uid)) {
        return true;
    }
    // An error: exited meanwhile.
    std::fs::symlink_metadata(proc_dir).is_ok_and(|meta| meta.uid() == uid)
}

/// Whether a `comm` value names a locker (comm holds the first 15 bytes of the name).
fn is_locker_comm(comm: &str) -> bool {
    LOCKER_COMMS
        .iter()
        .any(|name| &name[..name.len().min(15)] == comm)
}

fn is_camera_broker(proc_dir: &Path) -> bool {
    let comm = read_text(&proc_dir.join("comm"));
    let comm = comm.trim();
    CAMERA_BROKERS.iter().any(|broker| comm.starts_with(broker))
}

/// Streaming V4L2 clients mmap the device's buffers; idle monitors do not.
fn maps_video_device(proc_dir: &Path) -> bool {
    read_text(&proc_dir.join("maps")).contains("/dev/video")
}

fn video_target(target: &str, devices: Option<&HashSet<String>>) -> bool {
    let path = target.split(" (deleted)").next().unwrap_or("");
    match devices {
        None => path.starts_with("/dev/video"),
        Some(devices) => devices.contains(path),
    }
}

fn as_xid(handle: &WindowHandle) -> Option<u32> {
    match handle {
        WindowHandle::X11(id) if *id > 0 => Some(*id),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// environment and desktop tools
// ---------------------------------------------------------------------------
/// The process environment, or a private copy of it (tests).
pub(crate) struct Env {
    fake: Option<Mutex<EnvMap>>,
}

impl Env {
    fn process() -> Self {
        Self { fake: None }
    }

    #[cfg(test)]
    pub(crate) fn fake() -> Self {
        Self {
            fake: Some(Mutex::new(EnvMap::new())),
        }
    }

    pub(crate) fn vars(&self) -> EnvMap {
        match &self.fake {
            Some(vars) => vars.lock().clone(),
            None => std::env::vars_os()
                .filter_map(|(k, v)| Some((k.into_string().ok()?, v.into_string().ok()?)))
                .collect(),
        }
    }

    pub(crate) fn get(&self, key: &str) -> Option<String> {
        match &self.fake {
            Some(vars) => vars.lock().get(key).cloned(),
            None => std::env::var(key).ok(),
        }
    }

    pub(crate) fn set(&self, key: &str, value: &str) {
        match &self.fake {
            Some(vars) => {
                vars.lock().insert(key.to_string(), value.to_string());
            }
            // SAFETY: only `prepare_process` changes the environment, and it runs once at
            // start-up, before the UI and the worker threads exist (as its contract says).
            None => unsafe { std::env::set_var(key, value) },
        }
    }

    pub(crate) fn remove(&self, key: &str) {
        match &self.fake {
            Some(vars) => {
                vars.lock().remove(key);
            }
            // SAFETY: see `set`.
            None => unsafe { std::env::remove_var(key) },
        }
    }
}

/// Finds and runs desktop tools (faked by the tests).
pub(crate) trait Tools: Send + Sync {
    /// Full path of a program on `PATH` (`shutil.which`).
    fn which(&self, name: &str) -> Option<String>;

    /// Run `argv` (no shell, no standard input) with `env` as its whole environment; an error
    /// when it cannot start or does not finish within `timeout` seconds (it is killed then).
    fn run(&self, argv: &[String], timeout: f64, env: &EnvMap) -> Result<CommandResult, String>;
}

struct SystemTools;

impl Tools for SystemTools {
    fn which(&self, name: &str) -> Option<String> {
        which(name).map(|path| path.to_string_lossy().into_owned())
    }

    fn run(&self, argv: &[String], timeout: f64, env: &EnvMap) -> Result<CommandResult, String> {
        let (program, args) = argv.split_first().ok_or("no command to run")?;
        let mut child = Command::new(program)
            .args(args)
            .env_clear()
            .envs(env)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("failed to run: {e}"))?;
        // Read both pipes on their own threads so a chatty child cannot block on a full pipe.
        let readers = [
            child
                .stdout
                .take()
                .map(|s| Box::new(s) as Box<dyn Read + Send>),
            child
                .stderr
                .take()
                .map(|s| Box::new(s) as Box<dyn Read + Send>),
        ]
        .map(|stream| {
            thread::spawn(move || {
                let mut bytes = Vec::new();
                if let Some(mut stream) = stream {
                    let _ = stream.read_to_end(&mut bytes);
                }
                String::from_utf8_lossy(&bytes).into_owned()
            })
        });
        let deadline = Instant::now() + Duration::from_secs_f64(timeout.clamp(0.0, 3600.0));
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(2)),
                Ok(None) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!("timed out after {timeout:.1} s"));
                }
                Err(e) => return Err(format!("failed to run: {e}")),
            }
        };
        let [stdout, stderr] = readers.map(|reader| reader.join().unwrap_or_default());
        Ok(CommandResult::new(
            status.code().unwrap_or(-1),
            stdout,
            stderr,
        ))
    }
}

// ---------------------------------------------------------------------------
// /proc, /dev and /sys
// ---------------------------------------------------------------------------
type UidFn = Arc<dyn Fn() -> Option<u32> + Send + Sync>;
type ReadlinkFn = Arc<dyn Fn(&Path) -> io::Result<String> + Send + Sync>;

/// Processes and camera nodes, read from `/proc`, `/dev` and `/sys` (or a fake tree in tests).
///
/// Shared by the platform and the camera watcher thread.
pub(crate) struct ProcFs {
    proc_root: PathBuf,
    dev_root: PathBuf,
    sys_root: PathBuf,
    getuid: RwLock<UidFn>,
    /// Reads `/proc/<pid>/fd/<n>` symlinks (tests fake them with text files).
    readlink: RwLock<ReadlinkFn>,
}

impl ProcFs {
    fn new(proc_root: PathBuf, dev_root: PathBuf, sys_root: PathBuf) -> Self {
        Self {
            proc_root,
            dev_root,
            sys_root,
            getuid: RwLock::new(Arc::new(getuid)),
            readlink: RwLock::new(Arc::new(|path: &Path| {
                std::fs::read_link(path).map(|target| target.to_string_lossy().into_owned())
            })),
        }
    }

    fn uid(&self) -> Option<u32> {
        let getuid = self.getuid.read().clone();
        getuid()
    }

    fn read_link(&self, path: &Path) -> io::Result<String> {
        let readlink = self.readlink.read().clone();
        readlink(path)
    }

    /// Whether a screen locker of this user's login runs; `None` without `/proc`.
    ///
    /// Every process is looked at, not only those owned by this user: slock (and other lockers
    /// installed setuid root, e.g. swaylock built for shadow passwords, xlock) run as another
    /// user or have dropped privileges in a way that makes their `/proc` entry root's. `comm` is
    /// readable for every process, and so is `loginuid`, which the audit subsystem sets at login
    /// and which no setuid changes: a locker counts as ours when we own it or when its login UID
    /// is ours (see [`belongs_to_user`]). (Not the audit *session* id: an app started by the
    /// systemd user manager has a different one than the graphical session its locker runs in.)
    /// Without login UIDs (no audit support) only the owner tells, as before.
    fn locker_running(&self) -> Option<bool> {
        if !self.proc_root.is_dir() {
            return None;
        }
        let own_login = login_uid(&self.proc_root.join("self"));
        let uid = self.uid();
        let scan = || -> io::Result<bool> {
            for entry in std::fs::read_dir(&self.proc_root)? {
                let entry = entry?;
                if !is_pid(&entry.file_name()) {
                    continue;
                }
                let path = entry.path();
                let comm = read_text(&path.join("comm"));
                if is_locker_comm(comm.trim()) && belongs_to_user(&path, own_login, uid) {
                    return Ok(true);
                }
            }
            Ok(false)
        };
        scan().ok()
    }

    /// `(pid, /proc/<pid>)` of this user's processes.
    ///
    /// Other users' file descriptors are unreadable anyway; skipping them by owner is much
    /// cheaper than failing on each one.
    fn user_processes(&self) -> io::Result<Vec<(u32, PathBuf)>> {
        let uid = self.uid();
        let mut found = Vec::new();
        for entry in std::fs::read_dir(&self.proc_root)? {
            let entry = entry?;
            let name = entry.file_name();
            if !is_pid(&name) {
                continue;
            }
            let path = entry.path();
            if let Some(uid) = uid {
                match std::fs::symlink_metadata(&path) {
                    Ok(meta) if meta.uid() == uid => {}
                    _ => continue, // another user's, or exited meanwhile
                }
            }
            let Some(pid) = name.to_str().and_then(|n| n.parse().ok()) else {
                continue;
            };
            found.push((pid, path));
        }
        Ok(found)
    }

    fn scan_camera_users(&self, devices: Option<&HashSet<String>>) -> Option<bool> {
        if !self.proc_root.is_dir() {
            return None;
        }
        if devices.is_some_and(HashSet::is_empty) {
            return Some(false); // no camera to compete for
        }
        let own_pid = std::process::id();
        let processes = match self.user_processes() {
            Ok(processes) => processes,
            Err(e) => {
                log::debug!("camera scan failed: {e}");
                return None;
            }
        };
        for (pid, path) in processes {
            if pid != own_pid && self.holds_video_device(&path, devices) {
                log::debug!("Camera in use by pid {pid}");
                return Some(true);
            }
        }
        Some(false)
    }

    /// Whether this process has a physical camera node open.
    fn holds_camera_ourselves(&self, devices: Option<&HashSet<String>>) -> bool {
        if devices.is_some_and(HashSet::is_empty) {
            return false;
        }
        let fd_dir = self
            .proc_root
            .join(std::process::id().to_string())
            .join("fd");
        self.fd_targets(&fd_dir, |target| video_target(target, devices))
            .unwrap_or(false)
    }

    fn holds_video_device(&self, proc_dir: &Path, devices: Option<&HashSet<String>>) -> bool {
        // Other users' processes (permission denied) or a process that exited: an error.
        self.fd_targets(&proc_dir.join("fd"), |target| video_target(target, devices))
            .is_ok_and(|held| {
                // PipeWire keeps camera nodes open just to monitor them; it only competes for
                // the camera while it streams to some client.
                held && (!is_camera_broker(proc_dir) || maps_video_device(proc_dir))
            })
    }

    /// Whether a descriptor in `fd_dir` points at a target `matches` accepts.
    fn fd_targets(&self, fd_dir: &Path, matches: impl Fn(&str) -> bool) -> io::Result<bool> {
        for fd in std::fs::read_dir(fd_dir)? {
            let Ok(target) = self.read_link(&fd?.path()) else {
                continue;
            };
            if matches(&target) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// `/dev/videoN` paths backed by hardware; `None` when unknown.
    fn physical_video_devices(&self) -> Devices {
        let names = video_nodes(&self.dev_root).ok()?;
        let v4l = self.sys_root.join("class").join("video4linux");
        if !v4l.is_dir() {
            return Some(names.iter().map(|name| format!("/dev/{name}")).collect());
        }
        // Loopback devices (OBS virtual camera, v4l2loopback) have no parent "device" link;
        // apps reading them never block our physical camera.
        Some(
            names
                .iter()
                .filter(|name| !v4l.join(name).exists() || v4l.join(name).join("device").exists())
                .map(|name| format!("/dev/{name}"))
                .collect(),
        )
    }

    /// Whether this user may open a camera node (`video` group / logind ACL).
    fn camera_access(&self) -> Option<bool> {
        let nodes = video_nodes(&self.dev_root).ok()?;
        if nodes.is_empty() {
            return None;
        }
        Some(nodes.iter().any(|name| {
            let path = self.dev_root.join(name);
            CString::new(path.as_os_str().as_bytes()).is_ok_and(|path| {
                // SAFETY: `path` is a valid NUL-terminated string that outlives the call.
                unsafe { libc::access(path.as_ptr(), libc::R_OK | libc::W_OK) == 0 }
            })
        }))
    }
}

fn is_pid(name: &std::ffi::OsStr) -> bool {
    !name.is_empty() && name.as_bytes().iter().all(u8::is_ascii_digit)
}

/// Names of the `videoN` nodes in `dev_root`.
fn video_nodes(dev_root: &Path) -> io::Result<Vec<String>> {
    let mut names = Vec::new();
    for entry in std::fs::read_dir(dev_root)? {
        if let Ok(name) = entry?.file_name().into_string()
            && is_video_node(&name)
        {
            names.push(name);
        }
    }
    Ok(names)
}

// ---------------------------------------------------------------------------
// X11 idle time (MIT-SCREEN-SAVER)
// ---------------------------------------------------------------------------
/// Milliseconds since the last input (faked by the tests).
pub(crate) trait IdleSource: Send + Sync {
    fn idle_ms(&self) -> Option<u64>;
    fn available(&self) -> bool;
}

/// Milliseconds since the last input, from the MIT-SCREEN-SAVER extension.
///
/// One private connection is opened lazily and reused; the lock serialises access.
struct XssIdle {
    clock: Clock,
    state: Mutex<XssState>,
}

struct XssState {
    handles: Option<(RustConnection, u32)>,
    retry_at: f64,
}

impl XssIdle {
    fn new(clock: Clock) -> Self {
        Self {
            clock,
            state: Mutex::new(XssState {
                handles: None,
                retry_at: 0.0,
            }),
        }
    }

    fn ensure(&self, state: &mut XssState) -> bool {
        if state.handles.is_some() {
            return true;
        }
        let now = (self.clock)();
        if now < state.retry_at {
            return false;
        }
        state.handles = Self::open();
        if state.handles.is_none() {
            state.retry_at = now + RETRY_AFTER_FAILURE_S;
            return false;
        }
        true
    }

    fn open() -> Option<(RustConnection, u32)> {
        let (conn, screen) = match RustConnection::connect(None) {
            Ok(connected) => connected,
            Err(e) => {
                log::debug!("XOpenDisplay failed: {e}");
                return None;
            }
        };
        // Querying info on a server without the extension would fail on every tick.
        match conn.extension_information(screensaver::X11_EXTENSION_NAME) {
            Ok(Some(_)) => {}
            _ => {
                log::debug!("X server lacks the MIT-SCREEN-SAVER extension");
                return None;
            }
        }
        let root = conn.setup().roots.get(screen)?.root;
        Some((conn, root))
    }
}

impl IdleSource for XssIdle {
    fn idle_ms(&self) -> Option<u64> {
        let mut state = self.state.lock();
        if !self.ensure(&mut state) {
            return None;
        }
        let (conn, root) = state.handles.as_ref()?;
        match conn
            .screensaver_query_info(*root)
            .map_err(ReplyError::from)
            .and_then(|cookie| cookie.reply())
        {
            Ok(info) => Some(u64::from(info.ms_since_user_input)),
            Err(e) => {
                log::debug!("XScreenSaverQueryInfo failed: {e}");
                None
            }
        }
    }

    fn available(&self) -> bool {
        self.ensure(&mut self.state.lock())
    }
}

// ---------------------------------------------------------------------------
// X11 windows and DPMS
// ---------------------------------------------------------------------------
/// A failed X11 request.
#[derive(Debug)]
pub(crate) enum XError {
    /// The connection to the X server is gone or unusable.
    Connection(String),
    /// The server refused the request (a window that vanished, …).
    Request(String),
}

impl XError {
    fn is_connection(&self) -> bool {
        matches!(self, XError::Connection(_))
    }
}

impl std::fmt::Display for XError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            XError::Connection(e) | XError::Request(e) => f.write_str(e),
        }
    }
}

impl From<ConnectionError> for XError {
    fn from(e: ConnectionError) -> Self {
        XError::Connection(e.to_string())
    }
}

impl From<ReplyError> for XError {
    fn from(e: ReplyError) -> Self {
        match e {
            ReplyError::ConnectionError(e) => e.into(),
            ReplyError::X11Error(e) => XError::Request(format!("{e:?}")),
        }
    }
}

pub(crate) type XResult<T> = Result<T, XError>;

/// The X server requests [`EwmhClient`] needs (faked by the tests).
pub(crate) trait XServer: Send {
    fn root(&self) -> u32;
    fn intern_atom(&self, name: &str) -> XResult<u32>;
    /// A property of any type as integers; `None` when the window does not have it.
    fn property(&self, window: u32, atom: u32) -> XResult<Option<Vec<u32>>>;
    /// `(width, height)` of the window.
    fn geometry(&self, window: u32) -> XResult<(i32, i32)>;
    fn map_state(&self, window: u32) -> XResult<u8>;
    /// The window's origin in root (= global) coordinates.
    fn root_origin(&self, window: u32) -> XResult<(i32, i32)>;
    /// `WM_CLASS` as `(instance, class)`.
    fn wm_class(&self, window: u32) -> XResult<Option<(String, String)>>;
    /// Send a 32-bit client message about `window` to the root window.
    fn send_client_message(
        &self,
        window: u32,
        message_type: u32,
        data: [u32; 5],
        event_mask: u32,
    ) -> XResult<()>;
    fn flush(&self) -> XResult<()>;
    fn has_extension(&self, name: &str) -> XResult<bool>;
    /// An XTest relative pointer motion.
    fn fake_relative_motion(&self, dx: i16) -> XResult<()>;
    fn sync(&self) -> XResult<()>;
    fn dpms_capable(&self) -> XResult<bool>;
    /// `(power level, enabled)`.
    fn dpms_info(&self) -> XResult<(u16, bool)>;
    fn dpms_set_enabled(&self, enabled: bool) -> XResult<()>;
    fn dpms_force_level(&self, level: u16) -> XResult<()>;
}

/// Opens a connection to the X server.
pub(crate) type XFactory = Box<dyn Fn() -> XResult<Box<dyn XServer>> + Send + Sync>;

struct X11Server {
    conn: RustConnection,
    root: u32,
}

/// Errors of requests without a reply are logged, not raised (python-xlib's error handler).
fn checked(cookie: Result<VoidCookie<'_, RustConnection>, ConnectionError>) -> XResult<()> {
    match cookie?.check() {
        Ok(()) => Ok(()),
        Err(ReplyError::X11Error(e)) => {
            log::debug!("X11 error: {e:?}");
            Ok(())
        }
        Err(ReplyError::ConnectionError(e)) => Err(e.into()),
    }
}

fn open_x11() -> XResult<Box<dyn XServer>> {
    let (conn, screen) =
        RustConnection::connect(None).map_err(|e| XError::Connection(e.to_string()))?;
    let root = conn
        .setup()
        .roots
        .get(screen)
        .ok_or_else(|| XError::Connection(format!("no screen {screen}")))?
        .root;
    Ok(Box::new(X11Server { conn, root }))
}

impl XServer for X11Server {
    fn root(&self) -> u32 {
        self.root
    }

    fn intern_atom(&self, name: &str) -> XResult<u32> {
        Ok(self.conn.intern_atom(false, name.as_bytes())?.reply()?.atom)
    }

    fn property(&self, window: u32, atom: u32) -> XResult<Option<Vec<u32>>> {
        let reply = self
            .conn
            .get_property(false, window, atom, AtomEnum::ANY, 0, u32::MAX)?
            .reply()?;
        if reply.type_ == u32::from(AtomEnum::NONE) {
            return Ok(None);
        }
        let values: Option<Vec<u32>> = match reply.format {
            32 => reply.value32().map(Iterator::collect),
            16 => reply.value16().map(|v| v.map(u32::from).collect()),
            _ => reply.value8().map(|v| v.map(u32::from).collect()),
        };
        Ok(values)
    }

    fn geometry(&self, window: u32) -> XResult<(i32, i32)> {
        let geometry = self.conn.get_geometry(window)?.reply()?;
        Ok((i32::from(geometry.width), i32::from(geometry.height)))
    }

    fn map_state(&self, window: u32) -> XResult<u8> {
        let attributes = self.conn.get_window_attributes(window)?.reply()?;
        Ok(u8::from(attributes.map_state))
    }

    fn root_origin(&self, window: u32) -> XResult<(i32, i32)> {
        let origin = self
            .conn
            .translate_coordinates(window, self.root, 0, 0)?
            .reply()?;
        Ok((i32::from(origin.dst_x), i32::from(origin.dst_y)))
    }

    fn wm_class(&self, window: u32) -> XResult<Option<(String, String)>> {
        let class = x11rb::properties::WmClass::get(&self.conn, window)?.reply()?;
        Ok(class.map(|class| {
            (
                String::from_utf8_lossy(class.instance()).into_owned(),
                String::from_utf8_lossy(class.class()).into_owned(),
            )
        }))
    }

    fn send_client_message(
        &self,
        window: u32,
        message_type: u32,
        data: [u32; 5],
        event_mask: u32,
    ) -> XResult<()> {
        let event = ClientMessageEvent::new(32, window, message_type, data);
        self.conn
            .send_event(false, self.root, EventMask::from(event_mask), event)?
            .ignore_error();
        Ok(())
    }

    fn flush(&self) -> XResult<()> {
        Ok(self.conn.flush()?)
    }

    fn has_extension(&self, name: &str) -> XResult<bool> {
        Ok(self.conn.query_extension(name.as_bytes())?.reply()?.present)
    }

    fn fake_relative_motion(&self, dx: i16) -> XResult<()> {
        // detail 1: relative motion; root NONE: the pointer's screen.
        checked(self.conn.xtest_fake_input(
            MOTION_NOTIFY_EVENT,
            1,
            X_CURRENT_TIME,
            x11rb::NONE,
            dx,
            0,
            0,
        ))
    }

    fn sync(&self) -> XResult<()> {
        Ok(self.conn.sync()?)
    }

    fn dpms_capable(&self) -> XResult<bool> {
        Ok(self.conn.dpms_capable()?.reply()?.capable)
    }

    fn dpms_info(&self) -> XResult<(u16, bool)> {
        let info = self.conn.dpms_info()?.reply()?;
        Ok((u16::from(info.power_level), info.state))
    }

    fn dpms_set_enabled(&self, enabled: bool) -> XResult<()> {
        if enabled {
            checked(self.conn.dpms_enable())
        } else {
            checked(self.conn.dpms_disable())
        }
    }

    fn dpms_force_level(&self, level: u16) -> XResult<()> {
        checked(self.conn.dpms_force_level(DPMSMode::from(level)))
    }
}

/// Top-level windows (EWMH hints of the window manager) and DPMS of the X server.
///
/// Uses its own connection (never the UI toolkit's) guarded by a lock, so any thread may call
/// it. `factory` exists for tests.
pub(crate) struct EwmhClient {
    factory: XFactory,
    clock: Clock,
    own_pid: u32,
    state: Mutex<EwmhState>,
}

struct EwmhState {
    display: Option<Box<dyn XServer>>,
    atoms: HashMap<String, u32>,
    retry_at: f64,
}

/// One connected display and its atom cache, for the queries of [`EwmhClient`].
struct XSession<'a> {
    x: &'a dyn XServer,
    root: u32,
    atoms: &'a mut HashMap<String, u32>,
}

impl XSession<'_> {
    fn atom(&mut self, name: &str) -> XResult<u32> {
        if let Some(atom) = self.atoms.get(name) {
            return Ok(*atom);
        }
        let atom = self.x.intern_atom(name)?;
        self.atoms.insert(name.to_string(), atom);
        Ok(atom)
    }

    fn prop(&mut self, window: u32, name: &str) -> XResult<Option<Vec<u32>>> {
        let atom = self.atom(name)?;
        self.x.property(window, atom)
    }

    fn pid(&mut self, window: u32) -> XResult<Option<u32>> {
        Ok(self
            .prop(window, "_NET_WM_PID")?
            .and_then(|values| values.first().copied()))
    }

    /// Mapped on the current desktop, not minimised, not part of the shell.
    fn eligible(&mut self, window: u32) -> XResult<bool> {
        // Windows on other workspaces are unmapped by the WM (their client window is then
        // "unviewable"), so this also filters other desktops.
        if self.x.map_state(window)? != X_IS_VIEWABLE {
            return Ok(false);
        }
        let state = self.prop(window, "_NET_WM_STATE")?.unwrap_or_default();
        if state.contains(&self.atom("_NET_WM_STATE_HIDDEN")?) {
            return Ok(false);
        }
        let types = self
            .prop(window, "_NET_WM_WINDOW_TYPE")?
            .unwrap_or_default();
        let mut skipped = HashSet::new();
        for name in SKIPPED_WINDOW_TYPES {
            skipped.insert(self.atom(name)?);
        }
        Ok(!types.iter().any(|t| skipped.contains(t)))
    }

    /// Visible outer rectangle: client area + WM decorations - CSD shadows.
    fn frame_rect(&mut self, window: u32) -> XResult<Option<Rect>> {
        let (w, h) = self.x.geometry(window)?;
        // Asking the root window to translate the client's origin yields root (= global)
        // coordinates, whatever the reparenting depth.
        let (x, y) = self.x.root_origin(window)?;
        let mut rect = (x, y, w, h);
        let frame = self.prop(window, "_NET_FRAME_EXTENTS")?; // left, right, top, bottom
        if let Some([left, right, top, bottom]) = extents(frame.as_deref()) {
            rect = (
                rect.0 - left,
                rect.1 - top,
                rect.2 + left + right,
                rect.3 + top + bottom,
            );
        }
        self.without_shadow(window, rect)
    }

    /// The client window without WM decorations and without CSD shadows.
    fn client_rect(&mut self, window: u32) -> XResult<Option<Rect>> {
        let (w, h) = self.x.geometry(window)?;
        let (x, y) = self.x.root_origin(window)?;
        self.without_shadow(window, (x, y, w, h))
    }

    /// `rect` minus the invisible CSD shadow margins; `None` when nothing is left.
    fn without_shadow(
        &mut self,
        window: u32,
        (mut x, mut y, mut w, mut h): (i32, i32, i32, i32),
    ) -> XResult<Option<Rect>> {
        let shadow = self.prop(window, "_GTK_FRAME_EXTENTS")?;
        if let Some([left, right, top, bottom]) = extents(shadow.as_deref()) {
            (x, y, w, h) = (x + left, y + top, w - left - right, h - top - bottom);
        }
        Ok((w > 0 && h > 0).then(|| Rect::new(x, y, w, h)))
    }
}

/// The first four values of an extents property.
fn extents(values: Option<&[u32]>) -> Option<[i32; 4]> {
    match values? {
        [left, right, top, bottom, ..] => Some([*left, *right, *top, *bottom].map(|v| v as i32)),
        _ => None,
    }
}

impl EwmhClient {
    pub(crate) fn new(factory: Option<XFactory>, clock: Clock, own_pid: Option<u32>) -> Self {
        Self {
            factory: factory.unwrap_or_else(|| Box::new(open_x11)),
            clock,
            own_pid: own_pid.unwrap_or_else(std::process::id),
            state: Mutex::new(EwmhState {
                display: None,
                atoms: HashMap::new(),
                retry_at: 0.0,
            }),
        }
    }

    // ------------------------------------------------------------- public API
    fn foreground(&self) -> Option<WindowRef> {
        self.call(None, |s| {
            let active = s.prop(s.root, "_NET_ACTIVE_WINDOW")?;
            let Some(wid) = active.and_then(|v| v.first().copied()).filter(|w| *w > 0) else {
                return Ok(None);
            };
            if !s.eligible(wid)? {
                return Ok(None);
            }
            let pid = s.pid(wid)?;
            if pid == Some(self.own_pid) {
                return Ok(None);
            }
            Ok(Some(WindowRef {
                handle: WindowHandle::X11(wid),
                pid,
                rect: s.frame_rect(wid)?,
            }))
        })
    }

    fn window_at(&self, x: i32, y: i32) -> Option<WindowRef> {
        self.call(None, |s| {
            let mut stack = s
                .prop(s.root, "_NET_CLIENT_LIST_STACKING")?
                .unwrap_or_default();
            if stack.is_empty() {
                stack = s.prop(s.root, "_NET_CLIENT_LIST")?.unwrap_or_default();
            }
            for &wid in stack.iter().rev() {
                // the stacking list is bottom-to-top
                if wid == 0 {
                    continue;
                }
                let mut probe = || -> XResult<Option<(Option<u32>, Rect)>> {
                    let Some(rect) = s.frame_rect(wid)? else {
                        return Ok(None);
                    };
                    if !rect.contains(f64::from(x), f64::from(y)) || !s.eligible(wid)? {
                        return Ok(None);
                    }
                    Ok(Some((s.pid(wid)?, rect)))
                };
                let (pid, rect) = match probe() {
                    Ok(Some(found)) => found,
                    Ok(None) => continue,
                    Err(e) if e.is_connection() => return Err(e),
                    Err(_) => continue, // the window vanished between listing and querying it
                };
                if pid == Some(self.own_pid) {
                    continue; // our overlay / dialogs: look at what is underneath
                }
                return Ok(Some(WindowRef {
                    handle: WindowHandle::X11(wid),
                    pid,
                    rect: Some(rect),
                }));
            }
            Ok(None)
        })
    }

    fn rect(&self, wid: u32) -> Option<Rect> {
        self.call(None, |s| s.frame_rect(wid))
    }

    fn client_rect(&self, wid: u32) -> Option<Rect> {
        self.call(None, |s| s.client_rect(wid))
    }

    /// `(pid, WM_CLASS class)` of a window; `None` when X is unreachable.
    fn app_class(&self, wid: u32) -> Option<(Option<u32>, String)> {
        self.call(None, |s| {
            let class = s.x.wm_class(wid)?.map(|(_, class)| class);
            Ok(Some((s.pid(wid)?, class.unwrap_or_default())))
        })
    }

    fn is_valid(&self, wid: u32) -> bool {
        self.call(false, |s| s.eligible(wid))
    }

    fn activate(&self, wid: u32) -> bool {
        self.call(false, |s| {
            let atom = s.atom("_NET_ACTIVE_WINDOW")?;
            s.x.send_client_message(
                wid,
                atom,
                [EWMH_SOURCE_PAGER, X_CURRENT_TIME, 0, 0, 0],
                X_SUBSTRUCTURE_REDIRECT_MASK | X_SUBSTRUCTURE_NOTIFY_MASK,
            )?;
            s.x.flush()?;
            Ok(true)
        })
    }

    /// Wake the displays with a 1 px XTest pointer move there and back.
    fn nudge_pointer(&self) -> bool {
        self.call(false, |s| {
            if !s.x.has_extension("XTEST")? {
                return Ok(false);
            }
            for dx in [1, -1] {
                s.x.fake_relative_motion(dx)?;
            }
            s.x.sync()?;
            Ok(true)
        })
    }

    /// `(capable, enabled)` of the server's DPMS; `None` when X is unreachable.
    fn dpms_status(&self) -> Option<(bool, bool)> {
        self.call(None, |s| {
            if !s.x.has_extension("DPMS")? || !s.x.dpms_capable()? {
                return Ok(Some((false, false)));
            }
            Ok(Some((true, s.x.dpms_info()?.1)))
        })
    }

    /// Whether the monitors are powered on (DPMS power level "On").
    fn dpms_monitors_on(&self) -> Option<bool> {
        self.call(None, |s| Ok(Some(s.x.dpms_info()?.0 == DPMS_MODE_ON)))
    }

    /// Force the monitors off or on; DPMS must be enabled for this.
    fn dpms_force(&self, off: bool) -> bool {
        self.call(false, |s| {
            s.x.dpms_force_level(if off { DPMS_MODE_OFF } else { DPMS_MODE_ON })?;
            s.x.sync()?;
            Ok(true)
        })
    }

    fn dpms_set_enabled(&self, enabled: bool) -> bool {
        self.call(false, |s| {
            s.x.dpms_set_enabled(enabled)?;
            s.x.sync()?;
            Ok(true)
        })
    }

    // -------------------------------------------------------------- internals
    fn call<T>(&self, default: T, query: impl FnOnce(&mut XSession<'_>) -> XResult<T>) -> T {
        let mut guard = self.state.lock();
        if !self.connect(&mut guard) {
            return default;
        }
        let state = &mut *guard;
        let Some(display) = state.display.as_deref() else {
            return default;
        };
        let result = query(&mut XSession {
            x: display,
            root: display.root(),
            atoms: &mut state.atoms,
        });
        match result {
            Ok(value) => value,
            Err(e) => {
                log::debug!("X11 request failed: {e}");
                if e.is_connection() {
                    state.display = None;
                    state.retry_at = (self.clock)() + 1.0;
                }
                default
            }
        }
    }

    fn connect(&self, state: &mut EwmhState) -> bool {
        if state.display.is_some() {
            return true;
        }
        let now = (self.clock)();
        if now < state.retry_at {
            return false;
        }
        match (self.factory)() {
            Ok(display) => {
                state.display = Some(display);
                state.atoms.clear();
                true
            }
            Err(e) => {
                log::debug!("Cannot open an X11 connection: {e}");
                state.retry_at = now + RETRY_AFTER_FAILURE_S;
                false
            }
        }
    }
}

// ---------------------------------------------------------------------------
// D-Bus (zbus, in-process)
// ---------------------------------------------------------------------------
/// A value of a D-Bus reply, as far as this module needs to tell them apart.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum DBusValue {
    Bool(bool),
    Int(i64),
    /// A string or an object path.
    Str(String),
    Other,
}

/// Result of a D-Bus method call.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DBusReply {
    pub values: Vec<DBusValue>,
    /// D-Bus error name; empty on success.
    pub error: String,
}

impl DBusReply {
    pub(crate) fn values(values: Vec<DBusValue>) -> Self {
        Self {
            values,
            error: String::new(),
        }
    }

    pub(crate) fn failed(error: &str) -> Self {
        Self {
            values: Vec::new(),
            error: error.to_string(),
        }
    }

    fn ok(&self) -> bool {
        self.error.is_empty()
    }
}

/// One D-Bus method call with string arguments.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DBusCall<'a> {
    pub system: bool,
    pub service: &'a str,
    pub path: &'a str,
    pub interface: &'a str,
    pub method: &'a str,
    pub args: &'a [&'a str],
}

/// Blocking D-Bus method calls (faked by the tests).
pub(crate) trait DBus: Send + Sync {
    /// `None` when the bus is unavailable; callers then fall back to the command-line tools.
    fn call(&self, call: &DBusCall<'_>) -> Option<DBusReply>;
}

/// Blocking D-Bus method calls through zbus, without launching processes.
///
/// Polled queries (idle time, lock state) would otherwise start `gdbus` or `loginctl` several
/// times a second. Any thread may call it.
struct ZbusDBus {
    clock: Clock,
    state: Mutex<ZbusState>,
}

#[derive(Default)]
struct ZbusState {
    /// Session bus, system bus.
    connections: [Option<zbus::blocking::Connection>; 2],
    retry_at: [f64; 2],
}

impl ZbusDBus {
    fn new(clock: Clock) -> Self {
        Self {
            clock,
            state: Mutex::new(ZbusState::default()),
        }
    }

    fn connection(&self, system: bool) -> Option<zbus::blocking::Connection> {
        let bus = usize::from(system);
        let now = (self.clock)();
        let mut state = self.state.lock();
        if let Some(connection) = &state.connections[bus] {
            return Some(connection.clone());
        }
        if now < state.retry_at[bus] {
            return None;
        }
        let builder = if system {
            zbus::blocking::connection::Builder::system()
        } else {
            zbus::blocking::connection::Builder::session()
        };
        match builder.and_then(|b| {
            b.method_timeout(Duration::from_millis(DBUS_TIMEOUT_MS))
                .build()
        }) {
            Ok(connection) => {
                state.connections[bus] = Some(connection.clone());
                Some(connection)
            }
            Err(e) => {
                state.retry_at[bus] = now + RETRY_AFTER_FAILURE_S;
                log::debug!(
                    "D-Bus {} bus unavailable: {e}",
                    if system { "system" } else { "session" }
                );
                None
            }
        }
    }
}

/// Plain values from zvariant values (variants unwrapped, object paths as strings).
fn dbus_unwrap(value: zbus::zvariant::Value<'_>) -> DBusValue {
    use zbus::zvariant::Value;
    match value {
        Value::Bool(b) => DBusValue::Bool(b),
        Value::U8(v) => DBusValue::Int(i64::from(v)),
        Value::I16(v) => DBusValue::Int(i64::from(v)),
        Value::U16(v) => DBusValue::Int(i64::from(v)),
        Value::I32(v) => DBusValue::Int(i64::from(v)),
        Value::U32(v) => DBusValue::Int(i64::from(v)),
        Value::I64(v) => DBusValue::Int(v),
        Value::U64(v) => DBusValue::Int(i64::try_from(v).unwrap_or(i64::MAX)),
        Value::Str(s) => DBusValue::Str(s.to_string()),
        Value::ObjectPath(p) => DBusValue::Str(p.to_string()),
        Value::Value(inner) => dbus_unwrap(*inner),
        _ => DBusValue::Other,
    }
}

impl DBus for ZbusDBus {
    fn call(&self, call: &DBusCall<'_>) -> Option<DBusReply> {
        let connection = self.connection(call.system)?;
        let destination = Some(call.service);
        let interface = Some(call.interface);
        let result = match call.args {
            [] => connection.call_method(destination, call.path, interface, call.method, &()),
            [a] => connection.call_method(destination, call.path, interface, call.method, &(a,)),
            [a, b] => {
                connection.call_method(destination, call.path, interface, call.method, &(a, b))
            }
            _ => {
                log::debug!(
                    "D-Bus call {}.{}: too many arguments",
                    call.interface,
                    call.method
                );
                return None;
            }
        };
        match result {
            Ok(message) => {
                let body = message.body();
                if body.is_empty() {
                    return Some(DBusReply::values(Vec::new()));
                }
                match body.deserialize::<zbus::zvariant::Structure<'_>>() {
                    Ok(fields) => Some(DBusReply::values(
                        fields.into_fields().into_iter().map(dbus_unwrap).collect(),
                    )),
                    Err(e) => {
                        log::debug!("D-Bus reply of {}.{}: {e}", call.interface, call.method);
                        Some(DBusReply::values(vec![DBusValue::Other]))
                    }
                }
            }
            Err(zbus::Error::MethodError(name, _, _)) => Some(DBusReply::failed(name.as_str())),
            Err(zbus::Error::InputOutput(e)) if e.kind() == io::ErrorKind::TimedOut => {
                Some(DBusReply::failed("org.freedesktop.DBus.Error.NoReply"))
            }
            Err(e) => {
                log::debug!("D-Bus call {}.{} failed: {e}", call.interface, call.method);
                self.state.lock().connections[usize::from(call.system)] = None;
                None
            }
        }
    }
}

// ---------------------------------------------------------------------------
// camera-node notifications (fanotify / inotify)
// ---------------------------------------------------------------------------
/// An open and/or close of a watched camera node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CameraEvent {
    pub opened: bool,
    /// Whether another process did it; `None` when the notifier cannot tell.
    pub foreign: Option<bool>,
    /// A descriptor opened for writing was closed. Capture opens the node read-write, device
    /// listings (browsers enumerating cameras, `v4l2-ctl` queries) mostly read-only - so this
    /// marks a capture attempt ending.
    pub wrote: bool,
}

impl CameraEvent {
    pub(crate) const fn new(opened: bool, foreign: Option<bool>) -> Self {
        Self {
            opened,
            foreign,
            wrote: false,
        }
    }

    /// Forces a rescan (a notifier that failed, a queue overflow).
    const RESCAN: CameraEvent = CameraEvent::new(false, None);
}

/// Blocking source of [`CameraEvent`] for a set of device nodes.
pub(crate) trait CameraNotifier: Send {
    fn kind(&self) -> &'static str;
    /// False once the notifier failed for good (the watcher then polls).
    fn alive(&self) -> bool;
    fn wait(&mut self, timeout: f64) -> Vec<CameraEvent>;
    fn close(&mut self);
}

/// Opens the best available notifier for some device nodes, or `None` (poll).
pub(crate) type NotifierFactory =
    Arc<dyn Fn(&[String]) -> Option<Box<dyn CameraNotifier>> + Send + Sync>;

// <linux/fanotify.h>, <sys/inotify.h>
const FAN_CLOSE_WRITE: u64 = 0x0000_0008;
const FAN_CLOSE_NOWRITE: u64 = 0x0000_0010;
const FAN_OPEN: u64 = 0x0000_0020;
const FAN_Q_OVERFLOW: u64 = 0x0000_4000;
const FANOTIFY_METADATA_VERSION: u8 = 3;
const IN_CLOSE_WRITE: u32 = 0x0000_0008;
const IN_CLOSE_NOWRITE: u32 = 0x0000_0010;
const IN_OPEN: u32 = 0x0000_0020;
/// `struct fanotify_event_metadata`: event_len, vers, reserved, metadata_len, mask, fd, pid.
const FAN_EVENT_SIZE: usize = 24;
/// `struct inotify_event` without its trailing name: wd, mask, cookie, len.
const IN_EVENT_SIZE: usize = 16;

fn u32_at(data: &[u8], offset: usize) -> u32 {
    u32::from_ne_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

/// Events from a `read` of a fanotify descriptor.
///
/// For an unprivileged listener the kernel reports the pid only for events caused by the
/// listener's own process (0 otherwise), which is exactly the own-vs-foreign distinction needed
/// here.
fn parse_fanotify(data: &[u8], own_pid: u32) -> Vec<CameraEvent> {
    let mut events = Vec::new();
    let mut offset = 0;
    while offset + FAN_EVENT_SIZE <= data.len() {
        let length = u32_at(data, offset) as usize;
        let version = data[offset + 4];
        let mut mask_bytes = [0_u8; 8];
        mask_bytes.copy_from_slice(&data[offset + 8..offset + 16]);
        let mask = u64::from_ne_bytes(mask_bytes);
        let pid = u32_at(data, offset + 20);
        if length < FAN_EVENT_SIZE {
            break; // malformed: never loop forever
        }
        offset += length;
        // FAN_REPORT_FID groups never receive file descriptors (fd == FAN_NOFD). Quick
        // open/close pairs of one process may arrive merged in one event.
        if version != FANOTIFY_METADATA_VERSION || mask & FAN_Q_OVERFLOW != 0 {
            events.push(CameraEvent::RESCAN);
        } else if mask & (FAN_OPEN | FAN_CLOSE_WRITE | FAN_CLOSE_NOWRITE) != 0 {
            events.push(CameraEvent {
                opened: mask & FAN_OPEN != 0,
                foreign: Some(pid != own_pid),
                wrote: mask & FAN_CLOSE_WRITE != 0,
            });
        }
    }
    events
}

/// Events from a `read` of an inotify descriptor (no process information).
fn parse_inotify(data: &[u8]) -> Vec<CameraEvent> {
    let mut events = Vec::new();
    let mut offset = 0;
    while offset + IN_EVENT_SIZE <= data.len() {
        let mask = u32_at(data, offset + 4);
        let name_len = u32_at(data, offset + 12) as usize;
        offset += IN_EVENT_SIZE + name_len;
        events.push(CameraEvent {
            opened: mask & IN_OPEN != 0,
            foreign: None,
            wrote: mask & IN_CLOSE_WRITE != 0,
        });
    }
    events
}

type EventParser = Box<dyn Fn(&[u8]) -> Vec<CameraEvent> + Send>;

/// Reads events from a non-blocking notification descriptor.
pub(crate) struct FdNotifier {
    fd: i32,
    kind: &'static str,
    parse: EventParser,
}

impl FdNotifier {
    pub(crate) fn new(fd: i32, kind: &'static str, parse: EventParser) -> Self {
        Self { fd, kind, parse }
    }

    /// Wait up to `timeout` seconds for the descriptor to have data.
    ///
    /// `poll`, not `select`: `select` cannot watch descriptors numbered `FD_SETSIZE` (1024) or
    /// higher, which a busy process may hand out.
    fn readable(&self, timeout: f64) -> io::Result<bool> {
        let mut poller = libc::pollfd {
            fd: self.fd,
            events: libc::POLLIN,
            revents: 0,
        };
        let ms = (timeout * 1000.0).ceil().clamp(0.0, f64::from(i32::MAX)) as i32;
        // SAFETY: one valid pollfd, and the count says so.
        let ready = unsafe { libc::poll(&mut poller, 1, ms) };
        if ready < 0 {
            let err = io::Error::last_os_error();
            if err.kind() == io::ErrorKind::Interrupted {
                return Ok(false);
            }
            return Err(err);
        }
        Ok(ready > 0)
    }

    fn read(&self) -> io::Result<Vec<u8>> {
        let mut buffer = vec![0_u8; 16384];
        // SAFETY: the buffer is valid for writes of its whole length.
        let count = unsafe { libc::read(self.fd, buffer.as_mut_ptr().cast(), buffer.len()) };
        if count < 0 {
            return Err(io::Error::last_os_error());
        }
        buffer.truncate(count as usize);
        Ok(buffer)
    }
}

impl CameraNotifier for FdNotifier {
    fn kind(&self) -> &'static str {
        self.kind
    }

    /// False once the descriptor failed and was closed (see [`wait`](Self::wait)).
    fn alive(&self) -> bool {
        self.fd >= 0
    }

    /// Events that arrive within `timeout` seconds (possibly none).
    ///
    /// A failing descriptor is closed and reported once as a forced rescan; `alive` is then
    /// `false` so the watcher can fall back to polling.
    fn wait(&mut self, timeout: f64) -> Vec<CameraEvent> {
        if self.fd < 0 {
            thread::sleep(Duration::from_secs_f64(timeout.max(0.0)));
            return Vec::new();
        }
        let data = self.readable(timeout).and_then(|ready| {
            if ready {
                self.read().map(Some)
            } else {
                Ok(None)
            }
        });
        match data {
            Ok(None) => Vec::new(),
            Ok(Some(data)) => (self.parse)(&data),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => Vec::new(),
            Err(e) => {
                log::info!(
                    "Camera {} stopped working ({e}); polling the cameras instead",
                    self.kind
                );
                self.close();
                vec![CameraEvent::RESCAN]
            }
        }
    }

    fn close(&mut self) {
        let fd = std::mem::replace(&mut self.fd, -1);
        if fd >= 0 {
            // SAFETY: the descriptor is ours and closed exactly once.
            unsafe { libc::close(fd) };
        }
    }
}

impl Drop for FdNotifier {
    fn drop(&mut self) {
        self.close();
    }
}

fn c_path(path: &str) -> io::Result<CString> {
    CString::new(path).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))
}

/// Unprivileged fanotify (Linux >= 5.13): opens with own-vs-foreign attribution.
fn open_fanotify(paths: &[String]) -> io::Result<Box<dyn CameraNotifier>> {
    // Unprivileged groups must report file handles instead of descriptors.
    let flags = libc::FAN_CLOEXEC | libc::FAN_NONBLOCK | libc::FAN_REPORT_FID;
    // SAFETY: plain system call with flag arguments.
    let fd = unsafe { libc::fanotify_init(flags, libc::O_RDONLY as libc::c_uint) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    let notifier = FdNotifier::new(fd, "fanotify", {
        let own_pid = std::process::id();
        Box::new(move |data| parse_fanotify(data, own_pid))
    });
    let mut marked = 0;
    for path in paths {
        let mask = FAN_OPEN | FAN_CLOSE_WRITE | FAN_CLOSE_NOWRITE;
        let cpath = c_path(path)?;
        // SAFETY: `cpath` is a valid NUL-terminated path that outlives the call.
        let result = unsafe {
            libc::fanotify_mark(fd, libc::FAN_MARK_ADD, mask, libc::AT_FDCWD, cpath.as_ptr())
        };
        if result == 0 {
            marked += 1;
        } else {
            log::debug!(
                "fanotify cannot watch {path}: fanotify_mark: {}",
                io::Error::last_os_error()
            );
        }
    }
    if marked == 0 {
        return Err(io::Error::other("fanotify could not watch any camera node"));
    }
    Ok(Box::new(notifier))
}

/// inotify: opens and closes of the nodes, without knowing who did them.
fn open_inotify(paths: &[String]) -> io::Result<Box<dyn CameraNotifier>> {
    // SAFETY: plain system call with flag arguments.
    let fd = unsafe { libc::inotify_init1(libc::IN_NONBLOCK | libc::IN_CLOEXEC) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    let notifier = FdNotifier::new(fd, "inotify", Box::new(parse_inotify));
    let mut watched = 0;
    for path in paths {
        let cpath = c_path(path)?;
        let mask = IN_OPEN | IN_CLOSE_WRITE | IN_CLOSE_NOWRITE;
        // SAFETY: `cpath` is a valid NUL-terminated path that outlives the call.
        if unsafe { libc::inotify_add_watch(fd, cpath.as_ptr(), mask) } >= 0 {
            watched += 1;
        }
    }
    if watched == 0 {
        return Err(io::Error::other("inotify could not watch any camera node"));
    }
    Ok(Box::new(notifier))
}

/// Opens a notifier for some device nodes.
type NotifierOpener<'a> = &'a dyn Fn(&[String]) -> io::Result<Box<dyn CameraNotifier>>;

/// The best available notifier for these device nodes, or `None` (poll).
fn open_camera_notifier(paths: &[String]) -> Option<Box<dyn CameraNotifier>> {
    open_first_notifier(
        paths,
        &[("fanotify", &open_fanotify), ("inotify", &open_inotify)],
    )
}

/// The first of `openers` that works for `paths`.
fn open_first_notifier(
    paths: &[String],
    openers: &[(&str, NotifierOpener<'_>)],
) -> Option<Box<dyn CameraNotifier>> {
    if paths.is_empty() {
        return None;
    }
    for (kind, open) in openers {
        match open(paths) {
            Ok(notifier) => {
                log::debug!("Watching {} with {}", paths.join(", "), notifier.kind());
                return Some(notifier);
            }
            Err(e) => log::debug!("{kind} unavailable for camera nodes: {e}"),
        }
    }
    None
}

// ---------------------------------------------------------------------------
// camera watcher
// ---------------------------------------------------------------------------
/// A flag threads can wait for (`threading.Event`).
#[derive(Default)]
struct Flag {
    set: Mutex<bool>,
    changed: Condvar,
}

impl Flag {
    fn set(&self) {
        *self.set.lock() = true;
        self.changed.notify_all();
    }

    fn wait(&self, timeout: f64) -> bool {
        let mut set = self.set.lock();
        if !*set {
            let timeout = Duration::from_secs_f64(timeout.clamp(0.0, 1e9));
            self.changed.wait_while_for(&mut set, |set| !*set, timeout);
        }
        *set
    }
}

/// How the camera watcher paces itself (seconds).
#[derive(Debug, Clone, Copy)]
pub(crate) struct CameraTimings {
    pub poll_s: f64,
    pub safety_rescan_s: f64,
    pub idle_stop_s: f64,
    pub debounce_s: f64,
    pub grace_s: f64,
    pub cooldown_s: f64,
    pub max_cooldown_s: f64,
    /// Upper bound for one wait, so stop requests and idleness are noticed.
    pub max_wait_s: f64,
    /// How long the first query after a (re)start waits for a fresh scan. A `/proc` scan takes
    /// milliseconds; this only bounds a pathological one.
    pub first_scan_wait_s: f64,
}

impl Default for CameraTimings {
    fn default() -> Self {
        Self {
            poll_s: 3.0,
            safety_rescan_s: 30.0,
            idle_stop_s: 10.0,
            debounce_s: 0.1,
            grace_s: 30.0,
            cooldown_s: 60.0,
            max_cooldown_s: 1800.0,
            max_wait_s: 1.0,
            first_scan_wait_s: 0.5,
        }
    }
}

type DevicesFn = Arc<dyn Fn() -> Devices + Send + Sync>;
type ScanFn = Arc<dyn Fn(Option<&HashSet<String>>) -> Option<bool> + Send + Sync>;
type OwnUseFn = Arc<dyn Fn(Option<&HashSet<String>>) -> bool + Send + Sync>;

/// What the camera watcher asks the system.
#[derive(Clone)]
pub(crate) struct CameraProbes {
    /// The physical camera nodes.
    pub devices: DevicesFn,
    /// Whether another process holds one of them; `None` when unknown.
    pub scan: ScanFn,
    /// Whether this process holds one of them.
    pub own_use: OwnUseFn,
}

/// Whether another process uses - or just tried to open - a physical camera.
///
/// [`query`](Self::query) never runs the `/proc` scan on the caller's thread: a background thread
/// keeps the answer fresh while somebody keeps asking and ends itself `idle_stop_s` after the
/// last question. It rescans when notified of an open or close of a camera node (plus a slow
/// safety rescan) or, without notifications (or once the notifier failed), every `poll_s`. The
/// first question after the thread (re)started briefly waits for its first scan.
/// `threaded = false` scans inline at most every `poll_s` instead (tests).
///
/// **Refused opens.** When a notifier attributes to another process the close of a read-write
/// descriptor of a camera node while we hold a camera ourselves, that app just tried to start the
/// camera and failed (V4L2 has one streamer; capture opens read-write, device listings
/// read-only). The camera is then reported busy for `grace_s` so the controller releases it and
/// the app can retry, and for as long as the app then holds it. Some apps also probe devices
/// read-write, so when such a period ends without the app taking the camera it counts as a false
/// alarm; from the second consecutive false alarm on, further attempts are ignored for a cooldown
/// that doubles each time (`cooldown_s` .. `max_cooldown_s`), which bounds the cost of an app
/// that keeps probing the cameras in the background.
pub(crate) struct CameraWatcher {
    probes: Mutex<CameraProbes>,
    pub(crate) timings: Mutex<CameraTimings>,
    clock: Clock,
    threaded: bool,
    notifier_factory: NotifierFactory,
    state: Mutex<WatcherState>,
    wakeup: Flag,
}

struct WatcherState {
    /// Set by every finished scan; replaced by a fresh one when the thread starts.
    scanned: Arc<Flag>,
    /// Generation of the running thread.
    thread: Option<u64>,
    generation: u64,
    handle: Option<JoinHandle<()>>,
    stopping: bool,
    last_query: f64,
    next_scan_at: f64,
    holders: Option<bool>,
    contention_until: Option<f64>,
    contention_confirmed: bool,
    false_alarms: u32,
    cooldown_until: f64,
}

/// Clean-up of a watcher thread, also when it panics (the `finally` of its loop).
struct WatcherThread<'a> {
    watcher: &'a CameraWatcher,
    generation: u64,
    scanned: Arc<Flag>,
    notifier: Option<Box<dyn CameraNotifier>>,
}

impl Drop for WatcherThread<'_> {
    fn drop(&mut self) {
        if let Some(mut notifier) = self.notifier.take() {
            notifier.close();
        }
        {
            let mut state = self.watcher.state.lock();
            if state.thread == Some(self.generation) {
                state.thread = None;
            }
        }
        // A thread that died before its first scan must not keep a query waiting for it. (Its
        // own flag: a successor has a new one.)
        self.scanned.set();
    }
}

impl CameraWatcher {
    pub(crate) fn new(
        probes: CameraProbes,
        clock: Clock,
        threaded: bool,
        notifier_factory: NotifierFactory,
    ) -> Self {
        Self {
            probes: Mutex::new(probes),
            timings: Mutex::new(CameraTimings::default()),
            clock,
            threaded,
            notifier_factory,
            state: Mutex::new(WatcherState {
                scanned: Arc::new(Flag::default()),
                thread: None,
                generation: 0,
                handle: None,
                stopping: false,
                last_query: f64::NEG_INFINITY,
                next_scan_at: f64::NEG_INFINITY,
                holders: None,
                contention_until: None,
                contention_confirmed: false,
                false_alarms: 0,
                cooldown_until: f64::NEG_INFINITY,
            }),
            wakeup: Flag::default(),
        }
    }

    fn probes(&self) -> CameraProbes {
        self.probes.lock().clone()
    }

    fn timings(&self) -> CameraTimings {
        *self.timings.lock()
    }

    // ------------------------------------------------------------- public API
    /// Latest answer; `None` when no scan could tell yet.
    ///
    /// The first question after the thread ended (nobody asked for `idle_stop_s`, e.g. while
    /// tracking was paused or the screen locked) waits up to `first_scan_wait_s` for the new
    /// thread's first scan: the answer from before the pause may be hours old, and the
    /// controller asks exactly then to decide whether the camera may be reopened.
    pub(crate) fn query(self: &Arc<Self>) -> Option<bool> {
        let now = (self.clock)();
        let (started, scanned, due) = {
            let mut state = self.state.lock();
            state.last_query = now;
            let due = now >= state.next_scan_at;
            let started = self.threaded && self.ensure_thread_locked(&mut state);
            (started, state.scanned.clone(), due)
        };
        if started {
            scanned.wait(self.timings().first_scan_wait_s);
        } else if due && !self.threaded {
            self.rescan();
        }
        let now = (self.clock)();
        let mut state = self.state.lock();
        self.state_locked(&mut state, now)
    }

    /// Scan `/proc` for other holders now (on the calling thread).
    pub(crate) fn rescan(&self) {
        let devices = (self.probes().devices)();
        self.rescan_with(devices.as_ref());
    }

    /// React to opens/closes of camera nodes: rescan soon, spot refused opens.
    pub(crate) fn handle_events(&self, events: &[CameraEvent]) {
        if events.is_empty() {
            return;
        }
        // A refused capture attempt: another process opened the node read-write and closed it
        // again (after EBUSY). An app that keeps it open is found by the rescan instead;
        // read-only opens are device listings.
        if events.iter().any(|e| e.foreign == Some(true) && e.wrote) {
            let probes = self.probes();
            if (probes.own_use)((probes.devices)().as_ref()) {
                let now = (self.clock)();
                self.begin_contention_locked(&mut self.state.lock(), now);
            }
        }
        let soon = (self.clock)() + self.timings().debounce_s;
        let mut state = self.state.lock();
        state.next_scan_at = state.next_scan_at.min(soon);
    }

    /// End the background thread (tests; production threads end when idle).
    #[cfg(test)]
    pub(crate) fn stop(&self) {
        let handle = {
            let mut state = self.state.lock();
            state.stopping = true;
            state.handle.take()
        };
        self.wakeup.set();
        if let Some(handle) = handle {
            let _ = handle.join();
        }
    }

    // -------------------------------------------------------------- internals
    /// Start the watcher thread unless it runs; `true` when it was started.
    fn ensure_thread_locked(self: &Arc<Self>, state: &mut WatcherState) -> bool {
        if state.thread.is_some() || state.stopping {
            return false;
        }
        // Nothing watched the nodes since the previous thread ended: the new one must set up
        // its notifier and rescan at once, not at the old deadline, and the old answer no
        // longer counts (`None` until that scan is done).
        state.next_scan_at = f64::NEG_INFINITY;
        state.holders = None;
        state.scanned = Arc::new(Flag::default());
        state.generation += 1;
        let generation = state.generation;
        let scanned = state.scanned.clone();
        let watcher = Arc::clone(self);
        match thread::Builder::new()
            .name("camera-watch".into())
            .spawn(move || watcher.run(generation, scanned))
        {
            Ok(handle) => {
                state.thread = Some(generation);
                state.handle = Some(handle);
                true
            }
            Err(e) => {
                log::debug!("camera watcher could not start: {e}");
                false
            }
        }
    }

    /// The watcher thread; `scanned` is the flag its first scan sets.
    fn run(&self, generation: u64, scanned: Arc<Flag>) {
        let mut this = WatcherThread {
            watcher: self,
            generation,
            scanned,
            notifier: None,
        };
        let mut watched: Option<BTreeSet<String>> = None;
        loop {
            let now = (self.clock)();
            let timings = self.timings();
            let due = {
                let mut state = self.state.lock();
                if state.stopping || now - state.last_query > timings.idle_stop_s {
                    // Cleared under the lock so a concurrent query starts a new thread.
                    if state.thread == Some(generation) {
                        state.thread = None;
                    }
                    return;
                }
                state.next_scan_at
            };
            if now >= due {
                let devices = (self.probes().devices)();
                let key = devices
                    .as_ref()
                    .filter(|devices| !devices.is_empty())
                    .map(|devices| devices.iter().cloned().collect::<BTreeSet<_>>());
                if key != watched {
                    // first run, or cameras (un)plugged
                    if let Some(mut notifier) = this.notifier.take() {
                        notifier.close();
                    }
                    watched = key;
                    this.notifier = watched.as_ref().and_then(|paths| {
                        (self.notifier_factory)(&paths.iter().cloned().collect::<Vec<_>>())
                    });
                }
                self.rescan_with(devices.as_ref());
                let interval = if this.notifier.is_some() {
                    timings.safety_rescan_s
                } else {
                    timings.poll_s
                };
                let next = (self.clock)() + interval;
                self.state.lock().next_scan_at = next;
                continue;
            }
            let timeout = (due - now).min(timings.max_wait_s);
            let Some(notifier) = this.notifier.as_mut() else {
                self.wakeup.wait(timeout);
                continue;
            };
            let events = notifier.wait(timeout);
            self.handle_events(&events);
            if !notifier.alive() {
                // Poll from now on (poll_s, not the slow safety rescan). The device set stays
                // "watched", so a notifier that keeps failing is not re-created in a loop; the
                // next device change or thread restart tries again.
                notifier.close();
                this.notifier = None;
                let soon = (self.clock)() + timings.poll_s;
                let mut state = self.state.lock();
                state.next_scan_at = state.next_scan_at.min(soon);
            }
        }
    }

    fn rescan_with(&self, devices: Option<&HashSet<String>>) {
        let holders = (self.probes().scan)(devices);
        let now = (self.clock)();
        let poll_s = self.timings().poll_s;
        let mut state = self.state.lock();
        state.holders = holders;
        state.next_scan_at = state.next_scan_at.max(now + poll_s);
        if holders == Some(true) && state.contention_until.is_some() {
            state.contention_confirmed = true;
        }
        state.scanned.set();
    }

    fn state_locked(&self, state: &mut WatcherState, now: f64) -> Option<bool> {
        self.settle_contention_locked(state, now);
        if state.contention_until.is_some() {
            return Some(true);
        }
        state.holders
    }

    fn begin_contention_locked(&self, state: &mut WatcherState, now: f64) {
        if now < state.cooldown_until {
            log::debug!("Another app opened the camera; ignored (recent false alarms)");
            return;
        }
        let grace_s = self.timings().grace_s;
        if state.contention_until.is_none() {
            log::info!(
                "Another app tried to open the camera while tracking uses it; releasing it for \
                 {grace_s:.0} s so that app can start"
            );
            state.contention_confirmed = false;
        }
        state.contention_until = Some(now + grace_s);
    }

    fn settle_contention_locked(&self, state: &mut WatcherState, now: f64) {
        let Some(until) = state.contention_until else {
            return;
        };
        if now < until {
            return;
        }
        state.contention_until = None;
        if state.contention_confirmed || state.holders == Some(true) {
            state.false_alarms = 0;
            return;
        }
        state.false_alarms += 1;
        if state.false_alarms >= 2 {
            let timings = self.timings();
            let cooldown = (timings.cooldown_s * 2_f64.powi(state.false_alarms as i32 - 2))
                .min(timings.max_cooldown_s);
            state.cooldown_until = now + cooldown;
            log::info!(
                "No app took the camera after it was released; ignoring camera probes for \
                 {cooldown:.0} s"
            );
        } else {
            log::info!("No app took the camera after it was released; tracking resumes");
        }
    }
}

// ---------------------------------------------------------------------------
// platform
// ---------------------------------------------------------------------------
/// What [`LinuxPlatform::with_options`] can change (tests); production uses the defaults.
pub(crate) struct LinuxOptions {
    pub proc_root: PathBuf,
    pub dev_root: PathBuf,
    pub sys_root: PathBuf,
    pub clock: Clock,
    pub sleep: Sleep,
    /// `false` keeps all work on the calling thread.
    pub background_threads: bool,
    pub camera_notifier_factory: NotifierFactory,
}

impl Default for LinuxOptions {
    fn default() -> Self {
        Self {
            proc_root: PROC_ROOT.into(),
            dev_root: DEV_ROOT.into(),
            sys_root: SYS_ROOT.into(),
            clock: Arc::new(monotonic),
            sleep: Arc::new(|seconds| thread::sleep(Duration::from_secs_f64(seconds.max(0.0)))),
            background_threads: true,
            camera_notifier_factory: Arc::new(open_camera_notifier),
        }
    }
}

/// Who turned DPMS on just to switch the monitors off (it was off before).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DpmsRestore {
    Xlib,
    Xset,
}

#[derive(Default)]
struct SessionState {
    resolved: Option<String>,
    retry_at: f64,
    /// logind session id -> D-Bus object path.
    paths: HashMap<String, String>,
}

struct IdleState {
    /// Clock time of the last user input.
    last_input_at: Option<f64>,
    checked_at: f64,
    retry_at: f64,
    ever_ok: bool,
    /// Start and end of our last ydotool move.
    injected: Option<(f64, f64)>,
}

#[derive(Default)]
struct PointerState {
    /// Whether the ydotool CLI is 1.x (`--absolute`); `None` until a probe answered.
    ydotool_cli: Option<bool>,
    probed_at: Option<f64>,
    warned: bool,
}

/// Replacements for single steps of the lock logic (the Python tests' monkeypatching).
#[cfg(test)]
#[derive(Default)]
#[allow(clippy::type_complexity)]
pub(crate) struct TestHooks {
    pub lock_confirmed: Option<Box<dyn Fn(Option<&str>, bool) -> bool + Send + Sync>>,
    pub logind_lock_took_effect: Option<Box<dyn Fn(Option<&str>) -> bool + Send + Sync>>,
    pub locker_running: Option<Box<dyn Fn() -> Option<bool> + Send + Sync>>,
    pub app_process_name: Option<Box<dyn Fn(Option<i64>) -> Option<String> + Send + Sync>>,
}

/// Linux implementation of [`PlatformServices`] (X11 and Wayland).
pub struct LinuxPlatform {
    env: Env,
    tools: Box<dyn Tools>,
    fs: Arc<ProcFs>,
    clock: Clock,
    sleep: Sleep,
    background: bool,
    /// Variables set by `prepare_process` -> their original value (`None` = unset), restored
    /// for child processes (see `child_env`).
    env_overrides: Mutex<HashMap<String, Option<String>>>,
    xss: Box<dyn IdleSource>,
    ewmh: Arc<EwmhClient>,
    dbus: Box<dyn DBus>,
    ydotool_default_sockets: &'static [&'static str],
    session: Mutex<SessionState>,
    locked_cache: Mutex<Option<(f64, Option<bool>)>>,
    idle: Mutex<IdleState>,
    pointer: Mutex<PointerState>,
    dpms_restore: Arc<Mutex<Option<DpmsRestore>>>,
    dpms_watch: Mutex<Option<JoinHandle<()>>>,
    camera: Arc<CameraWatcher>,
    #[cfg(test)]
    hooks: TestHooks,
}

impl LinuxPlatform {
    pub fn new() -> Self {
        Self::with_options(LinuxOptions::default())
    }

    pub(crate) fn with_options(options: LinuxOptions) -> Self {
        let clock = options.clock;
        let fs = Arc::new(ProcFs::new(
            options.proc_root,
            options.dev_root,
            options.sys_root,
        ));
        let probes = CameraProbes {
            devices: {
                let fs = fs.clone();
                Arc::new(move || fs.physical_video_devices())
            },
            scan: {
                let fs = fs.clone();
                Arc::new(move |devices| fs.scan_camera_users(devices))
            },
            own_use: {
                let fs = fs.clone();
                Arc::new(move |devices| fs.holds_camera_ourselves(devices))
            },
        };
        Self {
            env: Env::process(),
            tools: Box::new(SystemTools),
            camera: Arc::new(CameraWatcher::new(
                probes,
                clock.clone(),
                options.background_threads,
                options.camera_notifier_factory,
            )),
            fs,
            sleep: options.sleep,
            background: options.background_threads,
            env_overrides: Mutex::new(HashMap::new()),
            xss: Box::new(XssIdle::new(clock.clone())),
            ewmh: Arc::new(EwmhClient::new(None, clock.clone(), None)),
            dbus: Box::new(ZbusDBus::new(clock.clone())),
            ydotool_default_sockets: YDOTOOL_DEFAULT_SOCKETS,
            session: Mutex::new(SessionState {
                retry_at: f64::NEG_INFINITY,
                ..SessionState::default()
            }),
            locked_cache: Mutex::new(None),
            idle: Mutex::new(IdleState {
                last_input_at: None,
                checked_at: f64::NEG_INFINITY,
                retry_at: f64::NEG_INFINITY,
                ever_ok: false,
                injected: None,
            }),
            pointer: Mutex::new(PointerState::default()),
            dpms_restore: Arc::new(Mutex::new(None)),
            dpms_watch: Mutex::new(None),
            clock,
            #[cfg(test)]
            hooks: TestHooks::default(),
        }
    }

    fn now(&self) -> f64 {
        (self.clock)()
    }

    // ---------------------------------------------------------------- session
    /// The session's environment: ours, with the changes of `prepare_process` undone.
    fn session_env(&self) -> EnvMap {
        let mut env = self.env.vars();
        for (key, original) in self.env_overrides.lock().iter() {
            match original {
                None => env.remove(key),
                Some(value) => env.insert(key.clone(), value.clone()),
            };
        }
        env
    }

    /// A real X11 session (XWayland does not count: it only sees X11 clients).
    fn x11_session(&self) -> bool {
        let env = self.session_env();
        !is_wayland_env(&env) && !var(&env, "DISPLAY").is_empty()
    }

    fn set_env(&self, key: &str, value: Option<&str>) {
        self.env_overrides
            .lock()
            .entry(key.to_string())
            .or_insert_with(|| self.env.get(key));
        match value {
            Some(value) => self.env.set(key, value),
            None => self.env.remove(key),
        }
    }

    /// Environment for desktop tools: as the user's session, not as ours.
    fn child_env(&self) -> EnvMap {
        // The session's own WAYLAND_DISPLAY: kscreen-doctor & co. hidden from Wayland would
        // run on XWayland, where e.g. DPMS control does not exist.
        let mut env = self.session_env();
        // AppImage runtimes point LD_LIBRARY_PATH at bundled libraries, which can break system
        // binaries linked against newer ones.
        if let Some(original) = env.remove("LD_LIBRARY_PATH_ORIG") {
            if original.is_empty() {
                env.remove("LD_LIBRARY_PATH");
            } else {
                env.insert("LD_LIBRARY_PATH".into(), original);
            }
        }
        env
    }

    // ----------------------------------------------------------- subprocesses
    /// Run a desktop tool; `None` when it is missing, hangs or cannot start.
    fn run(&self, argv: &[String], timeout: f64) -> Option<CommandResult> {
        let exe = self.tools.which(&argv[0])?;
        let mut full = vec![exe];
        full.extend_from_slice(&argv[1..]);
        match self.tools.run(&full, timeout, &self.child_env()) {
            Ok(result) => Some(result),
            Err(e) => {
                log::debug!("{} {e}", argv[0]);
                None
            }
        }
    }

    /// Run commands in order until one exits with 0; return its name.
    fn first_success(&self, commands: &[Vec<String>], timeout: f64) -> Option<String> {
        for argv in commands {
            let Some(result) = self.run(argv, timeout) else {
                continue;
            };
            if result.returncode == 0 {
                return Some(command_name(argv));
            }
            log::debug!(
                "{} exited with {}: {}",
                command_name(argv),
                result.returncode,
                truncate(result.stderr.trim(), 200)
            );
        }
        None
    }

    fn available(&self, argv: &[String]) -> bool {
        self.tools.which(&argv[0]).is_some()
    }

    // -------------------------------------------------------------- lock screen
    /// The screen-lock tools installed here, in the order [`lock_screen`](Self::lock_screen)
    /// tries them.
    ///
    /// Shown by `doctor`. Being installed is not proof that a tool locks this session:
    /// `loginctl lock-session` exists on every systemd system, but on a bare i3 or sway session
    /// nothing listens to logind's `Lock` signal. Whether one works is only known when it is
    /// tried; `lock_screen` checks that and reports failure.
    pub fn lock_methods(&self) -> Vec<String> {
        let handled = desktop_handles_logind_lock(&self.session_env());
        lock_commands(None)
            .iter()
            .filter(|argv| self.available(argv))
            .map(|argv| {
                let mut label = lock_method_label(argv);
                if argv[0] == "loginctl" && !handled {
                    label += " (works only if a locker such as xss-lock listens to logind)";
                }
                label
            })
            .collect()
    }

    fn lock_with_tools(&self) -> Option<String> {
        let env = self.session_env();
        let session = self.session_id();
        let session = session.as_deref();
        // Where LockedHint is authoritative, any locker's "success" can be checked.
        let verify_all = session.is_some() && locked_hint_authoritative(&env);
        // A desktop with a lock screen of its own: the display manager's lock (dm-tool switches
        // to the greeter) would come on top of it.
        let own_locker = desktop_handles_logind_lock(&env);
        let mut unconfirmed: Option<String> = None; // a locker ran, but the screen did not lock in time
        for argv in lock_commands(session) {
            if !self.available(&argv) {
                continue;
            }
            if unconfirmed.is_some() && self.lock_evident(session) {
                return unconfirmed; // a slow locker reacted after all: do not stack another
            }
            if unconfirmed.is_some() && own_locker && argv[0] == "dm-tool" {
                // The desktop's locker said it locked but nothing shows it (or it refused by
                // policy): a greeter on top would make the user unlock twice, or lock them again
                // right after they unlocked.
                log::debug!("Not stacking dm-tool lock on the desktop's own lock screen");
                continue;
            }
            let Some(result) = self.run(&argv, ACTION_TIMEOUT_S) else {
                continue;
            };
            let name = command_name(&argv);
            if result.returncode != 0 {
                log::debug!(
                    "{name} exited with {}: {}",
                    result.returncode,
                    truncate(result.stderr.trim(), 200)
                );
                continue;
            }
            let confirmed = if argv[0] == "loginctl" {
                self.logind_lock_took_effect(session)
            } else {
                !verify_all || self.lock_confirmed(session, true)
            };
            if !confirmed {
                log::info!(
                    "{name}: the screen did not lock within {LOCK_CONFIRM_S:.0} s; trying other \
                     lockers"
                );
                unconfirmed.get_or_insert(name);
                continue;
            }
            return Some(name);
        }
        // The last attempt may have been answered just after its wait.
        if unconfirmed.is_some() && self.lock_evident(session) {
            return unconfirmed;
        }
        None
    }

    /// Whether something handled logind's `Lock` signal (see [`lock_screen`](Self::lock_screen)).
    fn logind_lock_took_effect(&self, session: Option<&str>) -> bool {
        #[cfg(test)]
        if let Some(hook) = &self.hooks.logind_lock_took_effect {
            return hook(session);
        }
        let env = self.session_env();
        if !desktop_handles_logind_lock(&env) {
            return self.lock_confirmed(session, false);
        }
        if session.is_some() && locked_hint_authoritative(&env) {
            return self.lock_confirmed(session, true);
        }
        // Cinnamon, MATE, Xfce ... listen to logind, but not all of their lock screens report
        // the hint: nothing reliable to check.
        true
    }

    /// Wait up to `LOCK_CONFIRM_S` for the lock to show (`LockedHint` or a locker).
    ///
    /// `trust_unknown`: count it as locked when the hint could not be read at all (no evidence
    /// either way) and no locker process appeared.
    fn lock_confirmed(&self, session: Option<&str>, trust_unknown: bool) -> bool {
        #[cfg(test)]
        if let Some(hook) = &self.hooks.lock_confirmed {
            return hook(session, trust_unknown);
        }
        let deadline = self.now() + LOCK_CONFIRM_S;
        let mut hint_known = false;
        loop {
            if let Some(session) = session {
                let hint = self.logind_locked_hint(session);
                if hint == Some(true) {
                    return true;
                }
                hint_known = hint_known || hint.is_some();
            }
            if self.locker_running() == Some(true) {
                return true;
            }
            if self.now() >= deadline {
                return trust_unknown && !hint_known;
            }
            (self.sleep)(LOCK_CONFIRM_POLL_S);
        }
    }

    /// Whether the session shows as locked right now (locker process or hint).
    ///
    /// The process scan comes first: without D-Bus the hint costs a `loginctl` launch.
    fn lock_evident(&self, session: Option<&str>) -> bool {
        if self.locker_running() == Some(true) {
            return true;
        }
        session.is_some_and(|session| self.logind_locked_hint(session) == Some(true))
    }

    // ------------------------------------------------------------- lock state
    /// logind's `LockedHint`, or a running locker that does not report to it.
    fn query_locked_state(&self) -> Option<bool> {
        let session = self.session_id();
        let hint = session
            .as_deref()
            .and_then(|session| self.logind_locked_hint(session));
        if hint == Some(true) {
            return Some(true);
        }
        if hint == Some(false) && locked_hint_authoritative(&self.session_env()) {
            return Some(false); // GNOME / KDE always set the hint: no need to scan processes
        }
        let locker = self.locker_running();
        if locker == Some(true) {
            return Some(true);
        }
        if hint.is_some() {
            return hint;
        }
        // logind did not answer. Without a session to ask, a locker process is the only
        // evidence there is; with one, a failed query means "unknown" rather than "unlocked",
        // so a real lock screen is never mistaken for a return.
        if session.is_none() { locker } else { None }
    }

    fn logind_locked_hint(&self, session: &str) -> Option<bool> {
        let (answered, value) = self.locked_hint_dbus(session);
        if answered {
            return value;
        }
        let result = self.run(
            &command(&[
                "loginctl",
                "show-session",
                session,
                "-p",
                "LockedHint",
                "--value",
            ]),
            QUERY_TIMEOUT_S,
        )?;
        if result.returncode != 0 {
            return None;
        }
        parse_locked_hint(Some(&result.stdout))
    }

    /// `(answered, LockedHint)` from logind over D-Bus, without a process launch.
    fn locked_hint_dbus(&self, session: &str) -> (bool, Option<bool>) {
        let cached = self.session.lock().paths.get(session).cloned();
        let path = match cached {
            Some(path) => path,
            None => {
                let Some(reply) = self.dbus.call(&DBusCall {
                    system: true,
                    service: LOGIN1,
                    path: LOGIN1_PATH,
                    interface: LOGIN1_MANAGER,
                    method: "GetSession",
                    args: &[session],
                }) else {
                    return (false, None);
                };
                if !reply.ok() {
                    log::debug!("logind GetSession({session}) failed: {}", reply.error);
                    return (true, None);
                }
                match reply.values.first() {
                    Some(DBusValue::Str(path)) if path.starts_with('/') => {
                        self.session
                            .lock()
                            .paths
                            .insert(session.to_string(), path.clone());
                        path.clone()
                    }
                    _ => {
                        log::debug!(
                            "Unexpected GetSession reply {:?}; using loginctl",
                            reply.values
                        );
                        return (false, None);
                    }
                }
            }
        };
        let Some(reply) = self.dbus.call(&DBusCall {
            system: true,
            service: LOGIN1,
            path: &path,
            interface: DBUS_PROPERTIES,
            method: "Get",
            args: &[LOGIN1_SESSION, "LockedHint"],
        }) else {
            return (false, None);
        };
        if reply.ok() {
            if let Some(DBusValue::Bool(value)) = reply.values.first() {
                return (true, Some(*value));
            }
            log::debug!(
                "Unexpected LockedHint reply {:?}; using loginctl",
                reply.values
            );
            return (false, None);
        }
        log::debug!(
            "LockedHint of session {session} unavailable: {}",
            reply.error
        );
        self.session.lock().paths.remove(session); // the session may have ended
        (true, None)
    }

    fn locker_running(&self) -> Option<bool> {
        #[cfg(test)]
        if let Some(hook) = &self.hooks.locker_running {
            return hook();
        }
        self.fs.locker_running()
    }

    /// The logind session of this desktop.
    ///
    /// `XDG_SESSION_ID` is missing when the app is started by a systemd user unit; the user's
    /// graphical ("Display") session is used then. A failed lookup (logind still busy at login)
    /// is retried every 30 s.
    fn session_id(&self) -> Option<String> {
        let env_id = self.env.get("XDG_SESSION_ID").unwrap_or_default();
        let env_id = env_id.trim();
        if !env_id.is_empty() {
            return valid_session_id(env_id).then(|| env_id.to_string());
        }
        let mut state = self.session.lock();
        if let Some(resolved) = &state.resolved {
            return Some(resolved.clone());
        }
        let now = self.now();
        if now < state.retry_at {
            return None;
        }
        let found = self.lookup_display_session();
        if !found.is_empty() {
            state.resolved = Some(found.clone());
            return Some(found);
        }
        state.retry_at = now + SESSION_RETRY_S;
        None
    }

    fn lookup_display_session(&self) -> String {
        let Some(uid) = self.fs.uid() else {
            return String::new();
        };
        let Some(result) = self.run(
            &command(&[
                "loginctl",
                "show-user",
                &uid.to_string(),
                "-p",
                "Display",
                "--value",
            ]),
            QUERY_TIMEOUT_S,
        ) else {
            return String::new();
        };
        if result.returncode != 0 {
            return String::new();
        }
        let first = result.stdout.trim().lines().next().unwrap_or("").trim();
        let candidate = match first.split_once('=') {
            Some((_, value)) => value.trim(),
            None => first,
        };
        if valid_session_id(candidate) {
            candidate.to_string()
        } else {
            String::new()
        }
    }

    // ----------------------------------------------------------- display power
    /// Monitor power through the X server's DPMS extension.
    ///
    /// Forcing a power level requires DPMS to be enabled. A user who disabled it (`xset -dpms`:
    /// kiosks, presentations) gets it disabled again once the monitors are back on, and a server
    /// without DPMS (Xvnc, xrdp, many VMs) reports failure instead of pretending, so the caller
    /// can tell the user.
    fn x11_dpms(&self, off: bool) -> bool {
        let Some((capable, enabled)) = self.ewmh.dpms_status() else {
            return self.xset_dpms(off);
        };
        if !capable {
            log::debug!("The X server has no usable DPMS");
            return false;
        }
        let (ok, start_watch) = {
            let mut restore = self.dpms_restore.lock();
            if !off {
                let ok = self.ewmh.dpms_force(false);
                self.restore_dpms_locked(&mut restore);
                return ok;
            }
            if !enabled {
                if !self.ewmh.dpms_set_enabled(true) {
                    return false;
                }
                *restore = Some(DpmsRestore::Xlib);
            }
            let ok = self.ewmh.dpms_force(true);
            if !ok {
                self.restore_dpms_locked(&mut restore);
            }
            (ok, ok && *restore == Some(DpmsRestore::Xlib))
        };
        if start_watch && self.background {
            self.start_dpms_watch();
        }
        ok
    }

    /// `xset` fallback when the X server cannot be reached in-process.
    fn xset_dpms(&self, off: bool) -> bool {
        if self.tools.which("xset").is_none() {
            return false;
        }
        let state = self
            .run(&command(&["xset", "q"]), QUERY_TIMEOUT_S)
            .filter(|query| query.returncode == 0)
            .and_then(|query| parse_xset_dpms(Some(&query.stdout)));
        if state.is_some_and(|(capable, _)| !capable) {
            // xset would print a complaint and still exit with 0.
            log::debug!("The X server has no usable DPMS");
            return false;
        }
        let mut restore = self.dpms_restore.lock();
        if off && state.is_some_and(|(_, enabled)| !enabled) {
            *restore = Some(DpmsRestore::Xset); // "xset dpms force" enables DPMS as a side effect
        }
        let result = self.run(
            &command(&["xset", "dpms", "force", if off { "off" } else { "on" }]),
            ACTION_TIMEOUT_S,
        );
        let ok = result.is_some_and(|result| result.returncode == 0);
        if !off || !ok {
            self.restore_dpms_locked(&mut restore);
        }
        ok
    }

    fn restore_dpms_locked(&self, restore: &mut Option<DpmsRestore>) {
        match restore.take() {
            Some(DpmsRestore::Xlib) => {
                self.ewmh.dpms_set_enabled(false);
            }
            Some(DpmsRestore::Xset) => {
                self.run(&command(&["xset", "-dpms"]), ACTION_TIMEOUT_S);
            }
            None => {}
        }
    }

    /// Disable DPMS again once the monitors are on; `true` when nothing is left to do.
    #[cfg(test)]
    fn restore_dpms_if_awake(&self) -> bool {
        restore_dpms_if_awake(&self.dpms_restore, &self.ewmh)
    }

    fn start_dpms_watch(&self) {
        // Input wakes the monitors without us (e.g. wake_on_return is off), and DPMS must not
        // stay enabled with the server's blanking timeouts then.
        let mut watch = self.dpms_watch.lock();
        if watch.as_ref().is_some_and(|handle| !handle.is_finished()) {
            return;
        }
        let restore = self.dpms_restore.clone();
        let ewmh = self.ewmh.clone();
        let spawned = thread::Builder::new()
            .name("dpms-restore".into())
            .spawn(move || {
                let deadline = Instant::now() + Duration::from_secs_f64(DPMS_WATCH_MAX_S);
                while Instant::now() < deadline {
                    thread::sleep(Duration::from_secs_f64(DPMS_WATCH_INTERVAL_S));
                    if restore_dpms_if_awake(&restore, &ewmh) {
                        return;
                    }
                }
            });
        match spawned {
            Ok(handle) => *watch = Some(handle),
            Err(e) => log::debug!("DPMS watch failed: {e}"),
        }
    }

    // ------------------------------------------------------------------ input
    /// Idle time from GNOME's Mutter (the only source on GNOME Wayland).
    ///
    /// Results are cached for a short while and extrapolated: returning a stale value unchanged
    /// would move the implied "last input" timestamp forward and look like fresh (keyboard)
    /// activity. An idle reset caused by our own `ydotool` motion is not user input and keeps
    /// the previous timestamp.
    fn mutter_idle_s(&self) -> Option<f64> {
        let now = self.now();
        let mut idle = self.idle.lock();
        let last = idle.last_input_at;
        if let Some(last) = last
            && now - idle.checked_at < IDLE_TTL_S
        {
            return Some((now - last).max(0.0));
        }
        if now < idle.retry_at {
            return None;
        }
        let (ms, missing) = self.query_mutter_idle_ms();
        let Some(ms) = ms else {
            idle.last_input_at = None;
            // Back off for long only where the service does not exist (KDE, sway ...); a hiccup
            // of a working gnome-shell is retried soon.
            let permanent = missing || !idle.ever_ok;
            let delay = if permanent {
                RETRY_AFTER_FAILURE_S
            } else {
                IDLE_TRANSIENT_RETRY_S
            };
            idle.retry_at = now + delay;
            log::debug!("Mutter idle monitor unavailable; retrying in {delay:.0} s");
            return None;
        };
        idle.ever_ok = true;
        idle.checked_at = now;
        let observed = now - ms as f64 / 1000.0;
        let current = match last {
            Some(last) if injected_input_at(idle.injected, observed) => last,
            _ => observed,
        };
        idle.last_input_at = Some(current);
        Some((now - current).max(0.0))
    }

    /// `(idle ms, service missing)`: in-process D-Bus first, `gdbus` otherwise.
    fn query_mutter_idle_ms(&self) -> (Option<u64>, bool) {
        let reply = self.dbus.call(&DBusCall {
            system: false,
            service: MUTTER_IDLE_SERVICE,
            path: MUTTER_IDLE_PATH,
            interface: MUTTER_IDLE_SERVICE,
            method: "GetIdletime",
            args: &[],
        });
        if let Some(reply) = reply {
            if !reply.ok() {
                return (None, is_missing_service(&reply.error));
            }
            if let Some(DBusValue::Int(value)) = reply.values.first() {
                return (Some(u64::try_from(*value).unwrap_or(0)), false);
            }
            // An answer that did not convert as expected: never let that hide the idle time
            // for good - use gdbus instead.
            log::debug!(
                "Unexpected GetIdletime reply {:?}; using gdbus",
                reply.values
            );
        }
        let Some(result) = self.run(&command(&MUTTER_IDLE), IDLE_QUERY_TIMEOUT_S) else {
            // Missing gdbus is permanent; a timeout is a busy gnome-shell.
            return (None, self.tools.which("gdbus").is_none());
        };
        if result.returncode == 0
            && let Some(ms) = parse_gdbus_uint(Some(&result.stdout))
        {
            return (Some(ms), false);
        }
        (
            None,
            is_missing_service(&format!("{}\n{}", result.stdout, result.stderr)),
        )
    }

    // ----------------------------------------------------------------- cursor
    /// Pointer-warp commands usable in this Wayland session, most exact first.
    ///
    /// Lazy: ydotool is only probed when the compositor's IPC failed.
    fn wayland_pointer_commands(&self, x: i32, y: i32) -> impl Iterator<Item = Vec<String>> + '_ {
        let env = self.session_env();
        let (x, y) = (x.to_string(), y.to_string());
        let mut fixed = Vec::new();
        if !var(&env, "SWAYSOCK").is_empty() && self.tools.which("swaymsg").is_some() {
            // "-" is the current seat (sway >= 1.5); older versions need a name.
            for seat in ["-", "seat0"] {
                fixed.push(command(&["swaymsg", "seat", seat, "cursor", "set", &x, &y]));
            }
        }
        if !var(&env, "HYPRLAND_INSTANCE_SIGNATURE").is_empty()
            && self.tools.which("hyprctl").is_some()
        {
            fixed.push(command(&["hyprctl", "dispatch", "movecursor", &x, &y]));
        }
        let ydotool = std::iter::once_with(move || {
            self.ydotool_usable()
                .then(|| command(&["ydotool", "mousemove", "--absolute", "-x", &x, "-y", &y]))
        })
        .flatten();
        fixed.into_iter().chain(ydotool)
    }

    /// ydotool 1.x (`--absolute`) with its daemon running.
    fn ydotool_usable(&self) -> bool {
        if self.tools.which("ydotool").is_none() {
            return false;
        }
        let cli = {
            let mut pointer = self.pointer.lock();
            let now = self.now();
            let probe_due = pointer
                .probed_at
                .is_none_or(|at| now >= at + RETRY_AFTER_FAILURE_S);
            if pointer.ydotool_cli.is_none() && probe_due {
                pointer.probed_at = Some(now);
                let result = self.run(
                    &command(&["ydotool", "mousemove", "--help"]),
                    QUERY_TIMEOUT_S,
                );
                if let Some(result) = result {
                    let text = format!("{}\n{}", result.stdout, result.stderr);
                    pointer.ydotool_cli = Some(text.contains("--absolute"));
                }
            }
            if pointer.ydotool_cli == Some(false) && !pointer.warned {
                pointer.warned = true;
                log::warn!(
                    "ydotool is too old to move the pointer (0.1.x); install ydotool 1.x and run \
                     ydotoold"
                );
            }
            pointer.ydotool_cli
        };
        if cli != Some(true) {
            return false;
        }
        if !ydotool_socket_present(&self.session_env(), self.ydotool_default_sockets) {
            log::debug!("ydotool is installed but ydotoold is not running");
            return false;
        }
        true
    }
}

impl Default for LinuxPlatform {
    fn default() -> Self {
        Self::new()
    }
}

fn truncate(text: &str, max_chars: usize) -> &str {
    text.char_indices()
        .nth(max_chars)
        .map_or(text, |(end, _)| &text[..end])
}

fn injected_input_at(injected: Option<(f64, f64)>, when: f64) -> bool {
    injected.is_some_and(|(started, finished)| {
        started - 0.05 <= when && when <= finished + INJECTION_WINDOW_S
    })
}

/// Disable DPMS again once the monitors are on; `true` when nothing is left to do.
fn restore_dpms_if_awake(restore: &Mutex<Option<DpmsRestore>>, ewmh: &EwmhClient) -> bool {
    let mut restore = restore.lock();
    if *restore != Some(DpmsRestore::Xlib) {
        return true;
    }
    if ewmh.dpms_monitors_on() != Some(true) {
        return false;
    }
    log::debug!("Monitors woke up; disabling DPMS again as the user had it");
    *restore = None;
    ewmh.dpms_set_enabled(false);
    true
}

impl PlatformServices for LinuxPlatform {
    fn name(&self) -> &'static str {
        "linux"
    }

    fn is_wayland(&self) -> bool {
        is_wayland_env(&self.session_env())
    }

    // -------------------------------------------------------------- lifecycle
    /// Native pixels, and XWayland instead of native Wayland when possible.
    ///
    /// Native Wayland clients cannot position windows or read global coordinates, which
    /// calibration windows and the gaze overlay rely on. winit picks Wayland whenever
    /// `WAYLAND_DISPLAY` (or `WAYLAND_SOCKET`) is set, so they are hidden from this process.
    fn prepare_process(&self) {
        if self.env.get("WINIT_X11_SCALE_FACTOR").is_none() {
            self.set_env("WINIT_X11_SCALE_FACTOR", Some("1"));
        }
        if !(self.is_wayland() && self.env.get("DISPLAY").is_some_and(|d| !d.is_empty())) {
            return;
        }
        for key in ["WAYLAND_DISPLAY", "WAYLAND_SOCKET"] {
            if self.env.get(key).is_some() {
                self.set_env(key, None);
            }
        }
        log::info!("Wayland session: preferring XWayland (WAYLAND_DISPLAY hidden from the UI)");
    }

    fn capabilities(&self) -> HashMap<&'static str, bool> {
        let env = self.session_env();
        let x11 = self.x11_session();
        let has_display = !var(&env, "DISPLAY").is_empty();
        let gnome_idle = is_gnome_env(&env) && self.tools.which("gdbus").is_some();
        let x11_dpms = x11
            && match self.ewmh.dpms_status() {
                Some((capable, _)) => capable,
                None => self.tools.which("xset").is_some(),
            };
        let any_available = |commands: Vec<Vec<String>>| commands.iter().any(|c| self.available(c));
        HashMap::from([
            ("lock", any_available(lock_commands(None))),
            (
                "display_off",
                x11_dpms || any_available(display_power_commands(&env, false)),
            ),
            (
                "wake_display",
                x11_dpms || any_available(display_power_commands(&env, true)) || x11,
            ),
            ("input_idle", (x11 && self.xss.available()) || gnome_idle),
            ("key_idle", false),
            (
                "session_locked",
                self.tools.which("loginctl").is_some() || self.fs.proc_root.is_dir(),
            ),
            ("focus", x11),
            (
                "cursor",
                !self.is_wayland() || self.wayland_pointer_commands(0, 0).next().is_some(),
            ),
            ("camera_in_use", self.fs.proc_root.is_dir()),
            // Same rule as platform::hotkeys: X11 key grabs (also through XWayland).
            ("hotkeys", has_display),
            // Needs the focused window and its geometry, like "focus".
            ("panes", x11),
        ])
    }

    // ------------------------------------------------------------ session/power
    /// Lock the session with the first locker that demonstrably reacts.
    ///
    /// `loginctl lock-session` exits with 0 whether or not anything listens to logind's `Lock`
    /// signal (bare i3/sway without xss-lock or a swayidle `lock` hook). Outside desktops known
    /// to listen, it only counts once `LockedHint` is set or a locker process appeared;
    /// otherwise the other lockers are tried, and `false` tells the caller nothing locked.
    ///
    /// On GNOME and KDE, whose lock screens always set `LockedHint`, every locker's success is
    /// checked that way: GNOME Shell ignores lock requests when locking is disabled by policy
    /// (`disable-lock-screen`), while `loginctl` and the D-Bus calls still report success. Only a
    /// hint that cannot be read at all falls back to trusting the desktop.
    fn lock_screen(&self) -> bool {
        let Some(used) = self.lock_with_tools() else {
            log::warn!("Could not lock the screen: no screen locker responded");
            return false;
        };
        log::info!("Screen locked via {used}");
        *self.locked_cache.lock() = None; // the next poll must see the new state
        true
    }

    fn display_off(&self) -> bool {
        if self.x11_session() && self.x11_dpms(true) {
            log::debug!("Displays turned off via X11 DPMS");
            return true;
        }
        let commands = display_power_commands(&self.session_env(), false);
        let Some(used) = self.first_success(&commands, ACTION_TIMEOUT_S) else {
            log::debug!("No display power control available for this session");
            return false;
        };
        log::debug!("Displays turned off via {used}");
        true
    }

    fn wake_display(&self) -> bool {
        if self.x11_session() && self.x11_dpms(false) {
            return true;
        }
        let commands = display_power_commands(&self.session_env(), true);
        if self.first_success(&commands, ACTION_TIMEOUT_S).is_some() {
            return true;
        }
        // Any input wakes DPMS-blanked X11 screens; XTest synthesises some.
        self.x11_session() && self.ewmh.nudge_pointer()
    }

    fn is_session_locked(&self) -> Option<bool> {
        let now = self.now();
        let mut cache = self.locked_cache.lock();
        if let Some((at, value)) = *cache
            && now - at < LOCKED_HINT_TTL_S
        {
            return value;
        }
        let value = self.query_locked_state();
        *cache = Some((now, value));
        value
    }

    // -------------------------------------------------------------------- input
    fn seconds_since_input(&self) -> Option<f64> {
        if self.x11_session()
            && let Some(ms) = self.xss.idle_ms()
        {
            return Some(ms as f64 / 1000.0);
        }
        self.mutter_idle_s()
    }

    /// `false` on Wayland: XWayland only sees the pointer over X11 windows.
    fn cursor_position_reliable(&self) -> bool {
        !self.is_wayland()
    }

    /// X11: `None` (the UI toolkit warps the pointer). Wayland: compositor IPC or `ydotool`.
    ///
    /// sway and Hyprland place the pointer exactly through their IPC. `ydotool` emulates
    /// absolute moves with relative motion, which pointer acceleration distorts unless its
    /// virtual device uses a flat profile; that cannot be checked from here.
    fn move_cursor(&self, x: i32, y: i32) -> Option<bool> {
        if !self.is_wayland() {
            return None;
        }
        for argv in self.wayland_pointer_commands(x, y) {
            let started = self.now();
            let result = self.run(&argv, QUERY_TIMEOUT_S);
            if !result.is_some_and(|result| pointer_command_ok(&argv, &result)) {
                log::debug!("{} could not move the pointer", argv[0]);
                continue;
            }
            if argv[0] == "ydotool" {
                // uinput motion resets the compositor's idle timer like real input.
                let finished = self.now();
                self.idle.lock().injected = Some((started, finished));
            }
            return Some(true);
        }
        Some(false)
    }

    // ------------------------------------------------------------------ windows
    fn foreground_window(&self) -> Option<WindowRef> {
        if !self.x11_session() {
            return None;
        }
        self.ewmh.foreground()
    }

    fn window_at(&self, x: i32, y: i32) -> Option<WindowRef> {
        if !self.x11_session() {
            return None;
        }
        self.ewmh.window_at(x, y)
    }

    fn activate_window(&self, window: &WindowRef) -> bool {
        let Some(wid) = as_xid(&window.handle) else {
            return false;
        };
        if !self.x11_session() || !self.ewmh.is_valid(wid) {
            return false; // minimised or gone: never un-minimise behind the user's back
        }
        self.ewmh.activate(wid)
    }

    fn is_window_valid(&self, window: &WindowRef) -> bool {
        as_xid(&window.handle).is_some_and(|wid| self.x11_session() && self.ewmh.is_valid(wid))
    }

    fn window_rect(&self, window: &WindowRef) -> Option<Rect> {
        let wid = as_xid(&window.handle)?;
        if !self.x11_session() {
            return None;
        }
        self.ewmh.rect(wid)
    }

    fn window_client_rect(&self, window: &WindowRef) -> Option<Rect> {
        let wid = as_xid(&window.handle)?;
        if !self.x11_session() {
            return None;
        }
        self.ewmh.client_rect(wid)
    }

    fn window_app(&self, window: &WindowRef) -> Option<AppIdentity> {
        let wid = as_xid(&window.handle)?;
        if !self.x11_session() {
            return None;
        }
        let (pid, wm_class) = self.ewmh.app_class(wid)?;
        let pid = pid.or(window.pid);
        if window.pid.is_some() && pid != window.pid {
            return None; // the id now belongs to another client's window
        }
        let process = self.app_process_name(pid.map(i64::from))?;
        Some(AppIdentity {
            process,
            app_id: wm_class,
        })
    }

    #[cfg(test)]
    fn app_process_name(&self, pid: Option<i64>) -> Option<String> {
        match &self.hooks.app_process_name {
            Some(hook) => hook(pid),
            None => GenericPlatform.app_process_name(pid),
        }
    }

    // ------------------------------------------------------------------- camera
    /// Whether another process holds a physical camera, or was refused one just now.
    ///
    /// Never scans on the calling thread (see [`CameraWatcher`]); `None` until the first
    /// background scan finished.
    fn camera_in_use_by_other_app(&self) -> Option<bool> {
        self.camera.query()
    }

    // -------------------------------------------------------------- permissions
    fn permissions(&self) -> HashMap<&'static str, Option<bool>> {
        HashMap::from([("camera", self.fs.camera_access()), ("accessibility", None)])
    }
}

#[cfg(test)]
#[path = "linux_tests.rs"]
mod tests;
