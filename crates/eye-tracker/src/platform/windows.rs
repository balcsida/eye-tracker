//! Windows integration through the Win32 API.
//!
//! Design
//! ------
//! [`Win32Api`] is a thin binding layer. Each method answers one question ("which window is in
//! front?", "how long has the user been idle?") without any policy.
//!
//! [`WindowsPlatform`] holds the policy: which windows count as user windows, how keyboard focus
//! is acquired despite the foreground lock, what a failure means. The split keeps the policy
//! unit-testable with a fake binding on any OS.
//!
//! The module compiles on macOS and Linux too (the tests run everywhere): only the real binding
//! and the registry reader are Windows-only, the binding is created when a `WindowsPlatform`
//! method first needs it, and every public method degrades to `None`/`false` instead of
//! panicking.

use std::collections::{HashMap, HashSet};
use std::io;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, OnceLock};

use parking_lot::Mutex;

use super::base::{GenericPlatform, PlatformServices};
use crate::APP_ID;
use crate::types::{AppIdentity, Rect, WindowHandle, WindowRef};

// ---------------------------------------------------------------- constants
pub const GA_ROOT: u32 = 2;
pub const GW_HWNDNEXT: u32 = 2;
pub const GWL_EXSTYLE: i32 = -20;
pub const WS_POPUP: u32 = 0x8000_0000;
pub const WS_EX_TRANSPARENT: u32 = 0x0000_0020;
pub const WS_EX_TOOLWINDOW: u32 = 0x0000_0080;
pub const WS_EX_NOACTIVATE: u32 = 0x0800_0000;
pub const HWND_TOP: isize = 0;
pub const HWND_BROADCAST: isize = 0xFFFF;
pub const SWP_NOSIZE: u32 = 0x0001;
pub const SWP_NOMOVE: u32 = 0x0002;
pub const SWP_NOACTIVATE: u32 = 0x0010;
pub const SWP_ASYNCWINDOWPOS: u32 = 0x4000;
pub const WM_SYSCOMMAND: u32 = 0x0112;
pub const SC_MONITORPOWER: usize = 0xF170;
pub const MONITOR_POWER_OFF: isize = 2;
pub const ES_DISPLAY_REQUIRED: u32 = 0x0000_0002;
pub const ASFW_ANY: u32 = 0xFFFF_FFFF;
pub const INPUT_MOUSE: u32 = 0;
pub const INPUT_KEYBOARD: u32 = 1;
pub const MOUSEEVENTF_MOVE: u32 = 0x0001;
pub const KEYEVENTF_KEYUP: u32 = 0x0002;
pub const VK_MENU: u16 = 0x12;
/// An unassigned virtual key. Tapping it while Alt is held stops the Alt release from opening
/// the menu bar of the focused window (AutoHotkey's "menu mask key").
pub const VK_MENU_MASK: u16 = 0xE8;
pub const UOI_NAME: i32 = 2;
pub const DESKTOP_SWITCHDESKTOP: u32 = 0x0100;
pub const DWMWA_EXTENDED_FRAME_BOUNDS: u32 = 9;
pub const DWMWA_CLOAKED: u32 = 14;
pub const DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2: isize = -4;
pub const PROCESS_PER_MONITOR_DPI_AWARE: i32 = 2;
pub const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
pub const ERROR_SUCCESS: i32 = 0;
pub const ERROR_ACCESS_DENIED: u32 = 5;
pub const ERROR_INSUFFICIENT_BUFFER: i32 = 122;
pub const APPMODEL_ERROR_NO_PACKAGE: i32 = 15700;
/// HRESULT 0x80070005 as a signed 32-bit value.
pub const E_ACCESSDENIED: i32 = -2_147_024_891;
/// Seconds from the FILETIME epoch (1601-01-01) to the Unix epoch (1970-01-01).
pub const FILETIME_UNIX_OFFSET_S: u64 = 11_644_473_600;
/// FILETIME counts 100 ns intervals.
pub const FILETIME_TICKS_PER_S: u64 = 10_000_000;

/// Desktop and taskbar: "no window here" rather than a window to focus.
const SHELL_CLASSES: [&str; 4] = [
    "Progman",
    "WorkerW",
    "Shell_TrayWnd",
    "Shell_SecondaryTrayWnd",
];
/// Short-lived shell surfaces (menus, Alt+Tab, Start, tooltips...). Remembering one of these as
/// "the window on that monitor" and re-activating it later would pop up the Start menu or a
/// stale switcher, so they are skipped.
const TRANSIENT_CLASSES: [&str; 11] = [
    "#32768", // popup menus
    "tooltips_class32",
    "SysShadow",
    "ForegroundStaging",
    "MultitaskingViewFrame",
    "XamlExplorerHostIslandWindow",
    "TaskListThumbnailWnd",
    "NotifyIconOverflowWindow",
    "TopLevelWindowForOverflowXamlIsland",
    "Shell_InputSwitchTopLevelWindow",
    "Windows.UI.Core.CoreWindow", // Start, Search, Action Center as top-level windows
];
/// Upper bound for z-order walks (windows can be created/destroyed mid-walk).
const MAX_Z_WALK: usize = 2048;

const WEBCAM_KEY: &str =
    r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam";
const NON_PACKAGED: &str = "NonPackaged";
const SETTINGS_URIS: [(&str, &str); 1] = [("camera", "ms-settings:privacy-webcam")];
/// Slack when checking that a process was created before a camera session started. Both are
/// FILETIMEs from the same clock; this only absorbs rounding.
const SESSION_START_SLACK: u64 = 2 * FILETIME_TICKS_PER_S;
/// Slack when discarding camera sessions opened before the last boot.
const BOOT_TIME_SLACK: u64 = 600 * FILETIME_TICKS_PER_S;

// ---------------------------------------------------------------- helpers
/// Turn a panic into `None` (logged at DEBUG), per the platform contract.
fn best_effort<R>(what: &str, f: impl FnOnce() -> Option<R>) -> Option<R> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(_) => {
            log::debug!("{what} failed");
            None
        }
    }
}

/// Milliseconds from a 32-bit tick stamp to a 64-bit tick count.
///
/// `GetLastInputInfo` reports a 32-bit tick that wraps every 49.7 days; subtracting modulo
/// 2**32 keeps the result right across the wrap.
#[cfg_attr(not(windows), allow(dead_code))] // only the Win32 binding reads the tick count
fn elapsed_ms(now_ms: u64, last_ms: u64) -> u64 {
    ((now_ms & 0xFFFF_FFFF).wrapping_sub(last_ms & 0xFFFF_FFFF)) & 0xFFFF_FFFF
}

/// Windows-style case/separator normalisation (works on any host OS), like Python's
/// `ntpath.normcase(ntpath.normpath(path))`.
fn norm_path(path: &str) -> String {
    let path = path.replace('/', "\\");
    let (drive, rest) = match path.as_bytes() {
        [letter, b':', ..] if letter.is_ascii_alphabetic() => path.split_at(2),
        _ => ("", path.as_str()),
    };
    let root = if rest.starts_with("\\\\") && drive.is_empty() {
        "\\\\"
    } else if rest.starts_with('\\') {
        "\\"
    } else {
        ""
    };
    let mut parts: Vec<&str> = Vec::new();
    for part in rest.split('\\') {
        match part {
            "" | "." => {}
            ".." if parts.last().is_some_and(|last| *last != "..") => {
                parts.pop();
            }
            ".." if !root.is_empty() => {}
            _ => parts.push(part),
        }
    }
    let joined = format!("{drive}{root}{}", parts.join("\\"));
    let normal = if joined.is_empty() {
        ".".into()
    } else {
        joined
    };
    normal.to_lowercase()
}

/// `path` with links and packaged-app file redirection resolved; `None` on error.
fn real_path(host: &dyn Host, path: &str) -> Option<String> {
    let real = host.real_path(path)?;
    Some(real.strip_prefix("\\\\?\\").unwrap_or(&real).to_string())
}

/// Unix time (seconds) as a `FILETIME` tick count.
fn unix_to_filetime(seconds: f64) -> u64 {
    ((seconds + FILETIME_UNIX_OFFSET_S as f64) * FILETIME_TICKS_PER_S as f64) as u64
}

fn hwnd_of(window: Option<&WindowRef>) -> Option<isize> {
    match window?.handle {
        WindowHandle::Hwnd(handle) if handle > 0 => Some(handle),
        _ => None,
    }
}

/// The last path component (`ntpath.basename`).
fn nt_basename(path: &str) -> &str {
    path.rsplit(['\\', '/', ':']).next().unwrap_or("")
}

// ------------------------------------------------------------ Win32 binding
/// The Win32 calls the policy needs, one question each (a fake replaces it in tests).
trait Win32Api: Send + Sync {
    // windows
    fn foreground(&self) -> Option<isize>;
    fn root(&self, hwnd: isize) -> isize;
    /// `(thread id, process id)` owning the window (`(0, 0)` if it is gone).
    fn thread_process(&self, hwnd: isize) -> (u32, u32);
    fn class_name(&self, hwnd: isize) -> String;
    /// Visible frame in physical pixels.
    ///
    /// DWM's extended frame bounds exclude the invisible resize borders that `GetWindowRect`
    /// includes on Windows 10/11 (which make maximised windows appear to spill onto the
    /// neighbouring monitor).
    fn frame_rect(&self, hwnd: isize) -> Option<Rect>;
    /// Client area in physical pixels (the process is per-monitor DPI aware).
    fn client_rect(&self, hwnd: isize) -> Option<Rect>;
    fn is_window(&self, hwnd: isize) -> bool;
    fn is_visible(&self, hwnd: isize) -> bool;
    fn is_iconic(&self, hwnd: isize) -> bool;
    fn is_hung(&self, hwnd: isize) -> bool;
    /// Cloaked windows are "visible" but not shown (other virtual desktop, suspended UWP).
    fn is_cloaked(&self, hwnd: isize) -> bool;
    fn ex_style(&self, hwnd: isize) -> u32;
    fn window_from_point(&self, x: i32, y: i32) -> Option<isize>;
    fn next_window(&self, hwnd: isize) -> Option<isize>;
    fn set_foreground(&self, hwnd: isize) -> bool;
    /// Raise `hwnd` to the top of the z-order without waiting for its thread.
    ///
    /// `BringWindowToTop` (a synchronous `SetWindowPos`) blocks the caller until the window's
    /// thread processes the request, which is forever for a busy or debugger-paused app.
    /// `SWP_ASYNCWINDOWPOS` posts it instead.
    fn bring_to_top(&self, hwnd: isize) -> bool;
    fn allow_set_foreground_any(&self);
    fn current_thread_id(&self) -> u32;
    fn attach_thread_input(&self, thread: u32, to_thread: u32, attach: bool) -> bool;
    fn post_broadcast(&self, msg: u32, wparam: usize, lparam: isize) -> bool;
    /// A hidden, never-activated top-level window owned by the calling thread.
    ///
    /// Uses the predefined `STATIC` class, whose window procedure hands system commands to
    /// `DefWindowProc`, so no class has to be registered.
    fn create_hidden_window(&self) -> Option<isize>;
    /// `SendMessageW`; only used on windows of the calling thread (a direct call).
    fn send_message(&self, hwnd: isize, msg: u32, wparam: usize, lparam: isize) -> isize;
    fn destroy_window(&self, hwnd: isize) -> bool;
    // input
    /// Inject a no-op mouse event.
    ///
    /// Windows lets the process that injected the most recent input take the foreground (the
    /// same trick PowerToys uses), and an empty event has no visible effect.
    fn send_empty_mouse_input(&self) -> bool;
    /// Press and release Alt with the menu-mask key in between.
    ///
    /// An Alt press lifts the foreground lock; the masked release keeps the focused window from
    /// entering menu mode.
    fn tap_alt(&self) -> bool;
    /// Relative 1 px move there and back: real input that wakes sleeping displays.
    fn nudge_mouse(&self) -> bool;
    fn idle_ms(&self) -> Option<u64>;
    fn set_cursor_pos(&self, x: i32, y: i32) -> bool;
    fn cursor_pos(&self) -> Option<(i32, i32)>;
    // session
    fn lock_workstation(&self) -> bool;
    fn set_display_required(&self);
    /// Name of the desktop receiving input.
    ///
    /// `None` when it cannot be opened (the secure Winlogon desktop is not accessible to user
    /// processes), `""` when its name cannot be read.
    fn input_desktop_name(&self) -> Option<String>;
    // process
    /// Make the process Per-Monitor-V2 DPI aware; returns what was achieved.
    fn enable_per_monitor_dpi(&self) -> &'static str;
    fn set_app_user_model_id(&self, app_id: &str) -> bool;
    /// Full path of this process's executable.
    fn process_image_path(&self) -> Option<String>;
    /// Package family name of this process, `None` without package identity.
    ///
    /// A process installed as an MSIX package is recorded under this name in the camera
    /// consent store.
    fn current_package_family_name(&self) -> Option<String>;
    /// Package family name of process `pid`; `None` when it has none or is inaccessible.
    fn process_package_family_name(&self, pid: u32) -> Option<String>;
}

/// Creates the real binding; fails off Windows.
fn load_win32() -> Result<Arc<dyn Win32Api>, String> {
    #[cfg(windows)]
    {
        Ok(Arc::new(win32::Win32::new()))
    }
    #[cfg(not(windows))]
    {
        Err("the Win32 API is only available on Windows".into())
    }
}

// ------------------------------------------------------------------ registry
/// A registry value as read for the camera consent store.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(not(windows), allow(dead_code))] // only the Win32 reader (and the tests) build values
enum RegValue {
    /// `REG_DWORD` / `REG_QWORD`.
    Int(u64),
    /// `REG_SZ` / `REG_EXPAND_SZ`.
    Str(String),
    Other,
}

/// Read-only registry access (a fake replaces it in tests).
trait Registry: Send + Sync {
    fn subkeys(&self, hive: &str, path: &str) -> io::Result<Vec<String>>;
    fn values(&self, hive: &str, path: &str) -> io::Result<HashMap<String, RegValue>>;
}

/// Win32-backed reader. Missing keys read as empty (and everything does off Windows).
struct WinRegistry;

impl WinRegistry {
    #[cfg(windows)]
    const LIMIT: u32 = 4096;
}

impl Registry for WinRegistry {
    fn subkeys(&self, hive: &str, path: &str) -> io::Result<Vec<String>> {
        #[cfg(windows)]
        {
            Ok(win32::registry_subkeys(hive, path, Self::LIMIT))
        }
        #[cfg(not(windows))]
        {
            let _ = (hive, path);
            Ok(Vec::new())
        }
    }

    fn values(&self, hive: &str, path: &str) -> io::Result<HashMap<String, RegValue>> {
        #[cfg(windows)]
        {
            Ok(win32::registry_values(hive, path, Self::LIMIT))
        }
        #[cfg(not(windows))]
        {
            let _ = (hive, path);
            Ok(HashMap::new())
        }
    }
}

/// An app the camera consent store reports as currently streaming.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CameraUser {
    /// Package family name, or `#`-separated executable path for desktop apps.
    app: String,
    packaged: bool,
    /// `LastUsedTimeStart` of the open session (`FILETIME`, UTC); 0 if unknown.
    start: u64,
}

impl CameraUser {
    fn executable(&self) -> String {
        if self.packaged {
            self.app.clone()
        } else {
            self.app.replace('#', "\\")
        }
    }
}

/// A running process, as listed for the camera-session liveness check.
///
/// Only what every process yields cheaply. The creation time is read later, for the few
/// processes that match a session (see [`Host::process_created`]): for processes of other
/// accounts it can cost a full system snapshot each, which for a whole listing stalled the GUI
/// thread for about a second.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Process {
    pid: u32,
    /// Lower-case executable name.
    name: String,
}

/// `LastUsedTimeStart` of a session that is still open, else `None`.
fn session_start(values: &HashMap<String, RegValue>) -> Option<u64> {
    match (
        values.get("LastUsedTimeStart"),
        values.get("LastUsedTimeStop"),
    ) {
        (Some(RegValue::Int(start)), Some(RegValue::Int(0))) if *start > 0 => Some(*start),
        _ => None,
    }
}

/// Apps that started using the webcam and have not stopped yet.
///
/// Windows records camera sessions per app under the `CapabilityAccessManager` consent store
/// (the same data behind the camera privacy indicator): `LastUsedTimeStop` is 0 while a session
/// is open.
fn active_camera_users(registry: &dyn Registry) -> io::Result<Vec<CameraUser>> {
    let mut users = Vec::new();
    for name in registry.subkeys("HKCU", WEBCAM_KEY)? {
        let path = format!("{WEBCAM_KEY}\\{name}");
        if name.eq_ignore_ascii_case(NON_PACKAGED) {
            for app in registry.subkeys("HKCU", &path)? {
                if let Some(start) =
                    session_start(&registry.values("HKCU", &format!("{path}\\{app}"))?)
                {
                    users.push(CameraUser {
                        app,
                        packaged: false,
                        start,
                    });
                }
            }
        } else if let Some(start) = session_start(&registry.values("HKCU", &path)?) {
            users.push(CameraUser {
                app: name,
                packaged: true,
                start,
            });
        }
    }
    Ok(users)
}

fn consent_value(values: &HashMap<String, RegValue>) -> Option<String> {
    match values.get("Value") {
        Some(RegValue::Str(value)) if !value.trim().is_empty() => Some(value.trim().to_lowercase()),
        _ => None,
    }
}

// ---------------------------------------------------------------------- host
/// Process, file-system and shell queries outside Win32 (a fake replaces it in tests).
trait Host: Send + Sync {
    /// This program's executable path.
    fn executable(&self) -> Option<String>;
    /// `path` with links resolved, as the OS reports it; `None` on error.
    fn real_path(&self, path: &str) -> Option<String>;
    /// Running processes (pid and name); `None` if they cannot be listed.
    ///
    /// Only the name is requested: it is cheap for every process, whereas the creation time
    /// can cost a full system snapshot per process of another account (see [`Process`]).
    fn processes(&self) -> Option<Vec<Process>>;
    /// Creation time of `pid` as a `FILETIME` tick count; `None` when unreadable.
    fn process_created(&self, pid: u32) -> Option<u64>;
    /// Full executable path of `pid`; `None` when it is gone or not readable.
    fn process_exe(&self, pid: u32) -> Option<String>;
    /// System boot time as a `FILETIME` tick count.
    ///
    /// With Fast Startup this is the last full boot (resuming from hibernation keeps the
    /// uptime), so it is early rather than late: safe as a filter.
    fn boot_filetime(&self) -> Option<u64>;
    /// [`PlatformServices::app_process_name`] of the shared base implementation.
    fn app_process_name(&self, pid: u32) -> Option<String>;
    /// Open a URI with the shell (`ms-settings:` pages).
    fn open_uri(&self, uri: &str) -> bool;
}

/// The real host, through sysinfo and the shell.
struct SystemHost;

impl SystemHost {
    fn process(pid: u32, kind: sysinfo::ProcessRefreshKind) -> Option<sysinfo::System> {
        let mut system = sysinfo::System::new();
        let pid = sysinfo::Pid::from_u32(pid);
        system.refresh_processes_specifics(sysinfo::ProcessesToUpdate::Some(&[pid]), true, kind);
        system.process(pid).is_some().then_some(system)
    }
}

impl Host for SystemHost {
    fn executable(&self) -> Option<String> {
        std::env::current_exe()
            .ok()
            .map(|p| p.to_string_lossy().into_owned())
    }

    fn real_path(&self, path: &str) -> Option<String> {
        std::fs::canonicalize(path)
            .ok()
            .map(|p| p.to_string_lossy().into_owned())
    }

    fn processes(&self) -> Option<Vec<Process>> {
        let mut system = sysinfo::System::new();
        system.refresh_processes_specifics(
            sysinfo::ProcessesToUpdate::All,
            true,
            sysinfo::ProcessRefreshKind::nothing(),
        );
        let processes: Vec<Process> = system
            .processes()
            .iter()
            .map(|(pid, p)| Process {
                pid: pid.as_u32(),
                name: p.name().to_string_lossy().to_lowercase(),
            })
            .filter(|p| !p.name.is_empty())
            .collect();
        Some(processes)
    }

    fn process_created(&self, pid: u32) -> Option<u64> {
        let system = Self::process(pid, sysinfo::ProcessRefreshKind::nothing())?;
        let created = system.process(sysinfo::Pid::from_u32(pid))?.start_time();
        Some(unix_to_filetime(created as f64))
    }

    fn process_exe(&self, pid: u32) -> Option<String> {
        let kind = sysinfo::ProcessRefreshKind::nothing().with_exe(sysinfo::UpdateKind::Always);
        let system = Self::process(pid, kind)?;
        let exe = system.process(sysinfo::Pid::from_u32(pid))?.exe()?;
        Some(exe.to_string_lossy().into_owned()).filter(|e| !e.is_empty())
    }

    fn boot_filetime(&self) -> Option<u64> {
        Some(unix_to_filetime(sysinfo::System::boot_time() as f64))
    }

    fn app_process_name(&self, pid: u32) -> Option<String> {
        GenericPlatform.app_process_name(Some(i64::from(pid)))
    }

    fn open_uri(&self, uri: &str) -> bool {
        #[cfg(windows)]
        {
            win32::open_uri(uri)
        }
        #[cfg(not(windows))]
        {
            let _ = uri;
            false
        }
    }
}

// ------------------------------------------------------------------ platform
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Own,
    Shell,
    Transient,
    Normal,
}

enum ApiState {
    Unloaded,
    Ready(Arc<dyn Win32Api>),
    /// The failure is remembered: callers poll several times a second.
    Failed(String),
}

/// Windows 10/11 implementation of [`PlatformServices`].
pub struct WindowsPlatform {
    api: Mutex<ApiState>,
    api_factory: fn() -> Result<Arc<dyn Win32Api>, String>,
    registry: Box<dyn Registry>,
    host: Arc<dyn Host>,
    pid: u32,
    own_paths: OnceLock<HashSet<String>>,
    own_package_name: OnceLock<String>,
}

impl Default for WindowsPlatform {
    fn default() -> Self {
        Self::new()
    }
}

/// Destroys a window when dropped (Python's `try/finally`).
struct DestroyWindowOnDrop<'a>(&'a dyn Win32Api, isize);

impl Drop for DestroyWindowOnDrop<'_> {
    fn drop(&mut self) {
        self.0.destroy_window(self.1);
    }
}

/// Detaches the input queues again when dropped (Python's `try/finally`).
struct DetachOnDrop<'a>(&'a dyn Win32Api, u32, u32);

impl Drop for DetachOnDrop<'_> {
    fn drop(&mut self) {
        self.0.attach_thread_input(self.1, self.2, false);
    }
}

impl WindowsPlatform {
    /// Never touches Win32: the binding is created on first use.
    pub fn new() -> Self {
        Self::with_bindings(None, Box::new(WinRegistry), Arc::new(SystemHost))
    }

    fn with_bindings(
        api: Option<Arc<dyn Win32Api>>,
        registry: Box<dyn Registry>,
        host: Arc<dyn Host>,
    ) -> Self {
        Self {
            api: Mutex::new(api.map_or(ApiState::Unloaded, ApiState::Ready)),
            api_factory: load_win32,
            registry,
            host,
            pid: std::process::id(),
            own_paths: OnceLock::new(),
            own_package_name: OnceLock::new(),
        }
    }

    /// The binding, created on first use; `None` (logged) when it is unavailable.
    fn api(&self) -> Option<Arc<dyn Win32Api>> {
        let mut state = self.api.lock();
        if let ApiState::Unloaded = *state {
            *state = match (self.api_factory)() {
                Ok(api) => ApiState::Ready(api),
                Err(err) => {
                    log::warn!("Windows integration unavailable: {err}");
                    ApiState::Failed(err)
                }
            };
        }
        match &*state {
            ApiState::Ready(api) => Some(api.clone()),
            ApiState::Failed(err) => {
                log::debug!("Windows integration unavailable: {err}");
                None
            }
            ApiState::Unloaded => None,
        }
    }

    fn classify(&self, api: &dyn Win32Api, hwnd: isize) -> (Kind, u32) {
        let (_tid, pid) = api.thread_process(hwnd);
        if pid == self.pid {
            return (Kind::Own, pid);
        }
        let cls = api.class_name(hwnd);
        if SHELL_CLASSES.contains(&cls.as_str()) {
            return (Kind::Shell, pid);
        }
        if TRANSIENT_CLASSES.contains(&cls.as_str()) {
            return (Kind::Transient, pid);
        }
        (Kind::Normal, pid)
    }

    fn make_ref(api: &dyn Win32Api, hwnd: isize, pid: u32) -> WindowRef {
        WindowRef {
            handle: WindowHandle::Hwnd(hwnd),
            pid: (pid != 0).then_some(pid),
            rect: api.frame_rect(hwnd),
        }
    }

    fn window_below(&self, api: &dyn Win32Api, start: isize, x: i32, y: i32) -> Option<WindowRef> {
        let mut hwnd = api.next_window(start);
        for _step in 0..MAX_Z_WALK {
            let current = hwnd?;
            if Self::covers_point(api, current, x, y) {
                match self.classify(api, current) {
                    (Kind::Normal, pid) => return Some(Self::make_ref(api, current, pid)),
                    (Kind::Shell, _) => return None,
                    _ => {}
                }
            }
            hwnd = api.next_window(current);
        }
        None
    }

    fn covers_point(api: &dyn Win32Api, hwnd: isize, x: i32, y: i32) -> bool {
        if !api.is_visible(hwnd) || api.is_iconic(hwnd) || api.is_cloaked(hwnd) {
            return false;
        }
        if api.ex_style(hwnd) & (WS_EX_TRANSPARENT | WS_EX_NOACTIVATE) != 0 {
            return false; // click-through overlays and non-activatable tool windows
        }
        api.frame_rect(hwnd)
            .is_some_and(|rect| rect.contains(f64::from(x), f64::from(y)))
    }

    fn try_foreground(api: &dyn Win32Api, hwnd: isize) -> bool {
        // Raise only after a granted switch: raising a window that did not get focus would
        // show it on top while keystrokes still go elsewhere.
        if api.set_foreground(hwnd) {
            api.bring_to_top(hwnd);
        }
        api.foreground() == Some(hwnd)
    }

    /// Share the foreground thread's input state while switching.
    ///
    /// While attached, our thread counts as part of the foreground queue, which is what
    /// `SetForegroundWindow` checks. Skipped for hung windows: sharing input state with a
    /// thread that does not pump messages can stall ours.
    fn activate_attached(api: &dyn Win32Api, hwnd: isize) -> bool {
        let current = api.current_thread_id();
        let foreground = api.foreground();
        let fg_thread = foreground.map_or(0, |fg| api.thread_process(fg).0);
        if fg_thread == 0 || fg_thread == current {
            return Self::try_foreground(api, hwnd);
        }
        if foreground.is_some_and(|fg| api.is_hung(fg)) {
            return false;
        }
        if !api.attach_thread_input(current, fg_thread, true) {
            return false;
        }
        let _detach = DetachOnDrop(api, current, fg_thread);
        Self::try_foreground(api, hwnd)
    }

    // ------------------------------------------------------------------- camera
    fn own_executables(&self) -> &HashSet<String> {
        self.own_paths.get_or_init(|| {
            let mut candidates: Vec<Option<String>> = vec![self.host.executable()];
            // The image this process runs (the Python port needed it behind a venv launcher).
            match best_effort("process_image_path", || {
                self.api().map(|api| api.process_image_path())
            }) {
                Some(path) => candidates.push(path),
                None => log::debug!("Could not read the process image path"),
            }
            // Started from a packaged (MSIX) app, such as a terminal inside another app, this
            // process sees redirected paths (`AppData\Roaming`) while the consent store records
            // the real file under `AppData\Local\Packages\<family>\LocalCache`. Resolving the
            // path through the file system gives that real location.
            let real: Vec<Option<String>> = candidates
                .iter()
                .flatten()
                .map(|p| real_path(&*self.host, p))
                .collect();
            candidates.extend(real);
            candidates.iter().flatten().map(|p| norm_path(p)).collect()
        })
    }

    /// Lower-cased package family name of this process; `""` without package identity.
    fn own_package(&self) -> &str {
        self.own_package_name.get_or_init(|| {
            let name = best_effort("current_package_family_name", || {
                self.api().map(|api| api.current_package_family_name())
            });
            if name.is_none() {
                log::debug!("Could not read the package identity");
            }
            name.flatten().unwrap_or_default().to_lowercase()
        })
    }

    fn is_own_camera_user(&self, user: &CameraUser) -> bool {
        if user.packaged {
            // With package identity (e.g. an MSIX install) our own session is recorded under
            // the package family name, not a path.
            let own = self.own_package();
            return !own.is_empty() && user.app.to_lowercase() == own;
        }
        self.own_executables()
            .contains(&norm_path(&user.executable()))
    }

    /// Whether a process that can own `user`'s session is still running.
    ///
    /// The process must match the entry (full executable path, or package family) and must
    /// have been started before the session was: a copy of the app launched after a crash
    /// never touched the camera. The cheap identity checks come first, so the creation time is
    /// only read for the handful of processes that can be the app at all.
    fn session_is_live(
        &self,
        api: &dyn Win32Api,
        user: &CameraUser,
        processes: &[Process],
    ) -> bool {
        if user.packaged {
            let family = user.app.to_lowercase();
            return processes.iter().any(|p| {
                api.process_package_family_name(p.pid)
                    .unwrap_or_default()
                    .to_lowercase()
                    == family
                    && self.started_before(p.pid, user.start)
            });
        }
        let target = norm_path(&user.executable());
        let name = nt_basename(&target);
        for process in processes {
            if process.name != name || !self.started_before(process.pid, user.start) {
                continue;
            }
            // A path we may not read (an elevated process) gets the benefit of the doubt.
            match self.host.process_exe(process.pid) {
                None => return true,
                Some(exe) if norm_path(&exe) == target => return true,
                Some(_) => {}
            }
        }
        false
    }

    /// Whether `pid` may have opened a camera session that began at `session_start`.
    ///
    /// An unreadable creation time gets the benefit of the doubt.
    fn started_before(&self, pid: u32, session_start: u64) -> bool {
        self.host
            .process_created(pid)
            .is_none_or(|created| created <= session_start + SESSION_START_SLACK)
    }

    /// `false` when a Windows privacy switch blocks desktop apps from the camera.
    fn camera_permission(&self) -> Option<bool> {
        let read = || -> io::Result<_> {
            Ok((
                consent_value(&self.registry.values("HKLM", WEBCAM_KEY)?),
                consent_value(&self.registry.values("HKCU", WEBCAM_KEY)?),
                consent_value(
                    &self
                        .registry
                        .values("HKCU", &format!("{WEBCAM_KEY}\\{NON_PACKAGED}"))?,
                ),
            ))
        };
        let (device, user, desktop) = match read() {
            Ok(values) => values,
            Err(err) => {
                log::debug!("Could not read camera consent: {err}");
                return None;
            }
        };
        if [&device, &user, &desktop]
            .iter()
            .any(|v| v.as_deref() == Some("deny"))
        {
            return Some(false);
        }
        if user.as_deref() == Some("allow") {
            return Some(true);
        }
        None
    }
}

impl PlatformServices for WindowsPlatform {
    fn name(&self) -> &'static str {
        "windows"
    }

    // ---------------------------------------------------------------- lifecycle
    /// Per-Monitor-V2 DPI awareness, so global coordinates are physical pixels.
    fn prepare_process(&self) {
        best_effort("prepare_process", || {
            let api = self.api()?;
            let mode = api.enable_per_monitor_dpi();
            log::debug!("DPI awareness: {mode}");
            // Groups taskbar entries and notifications under the app.
            if !api.set_app_user_model_id(APP_ID) {
                log::debug!("Could not set the AppUserModelID");
            }
            Some(())
        });
    }

    fn capabilities(&self) -> HashMap<&'static str, bool> {
        let mut caps = GenericPlatform.capabilities();
        caps.extend([
            ("lock", true),
            ("display_off", true),
            ("wake_display", true),
            ("input_idle", true),
            ("key_idle", false),
            ("session_locked", true),
            ("focus", true),
            ("cursor", true),
            ("camera_in_use", true),
            ("hotkeys", true),
            ("panes", true),
        ]);
        caps
    }

    // ------------------------------------------------------------ session/power
    fn lock_screen(&self) -> bool {
        best_effort("lock_screen", || {
            if self.api()?.lock_workstation() {
                return Some(true);
            }
            log::warn!("LockWorkStation failed (locking may be disabled by policy)");
            Some(false)
        })
        .unwrap_or(false)
    }

    /// Power the monitors down through `DefWindowProc` of a window of our own.
    ///
    /// Any top-level window's `DefWindowProc` handles `SC_MONITORPOWER`, so a throw-away hidden
    /// window on the calling thread is enough, and sending to it is a plain function call.
    /// Broadcasting instead would queue the request in every window of the system: a suspended
    /// or debugger-paused app would run it whenever it next pumps messages, possibly long after
    /// the user came back, blacking the screens out again.
    fn display_off(&self) -> bool {
        best_effort("display_off", || {
            let api = self.api()?;
            let Some(hwnd) = api.create_hidden_window() else {
                log::debug!("Could not create a window for SC_MONITORPOWER; broadcasting instead");
                return Some(api.post_broadcast(WM_SYSCOMMAND, SC_MONITORPOWER, MONITOR_POWER_OFF));
            };
            let _destroy = DestroyWindowOnDrop(&*api, hwnd);
            api.send_message(hwnd, WM_SYSCOMMAND, SC_MONITORPOWER, MONITOR_POWER_OFF);
            Some(true)
        })
        .unwrap_or(false)
    }

    fn wake_display(&self) -> bool {
        best_effort("wake_display", || {
            let api = self.api()?;
            api.set_display_required();
            let before = api.cursor_pos();
            let ok = api.nudge_mouse();
            if let Some((x, y)) = before {
                // Pointer acceleration can turn +1/-1 into an uneven pair; put it back exactly.
                api.set_cursor_pos(x, y);
            }
            Some(ok)
        })
        .unwrap_or(false)
    }

    fn is_session_locked(&self) -> Option<bool> {
        best_effort("is_session_locked", || {
            let Some(name) = self.api()?.input_desktop_name() else {
                return Some(true); // the input desktop is the secure (Winlogon) desktop
            };
            if name.is_empty() {
                return None;
            }
            Some(name.to_lowercase() != "default")
        })
    }

    // -------------------------------------------------------------------- input
    fn seconds_since_input(&self) -> Option<f64> {
        best_effort("seconds_since_input", || {
            let ms = self.api()?.idle_ms()?;
            Some(ms as f64 / 1000.0)
        })
    }

    fn move_cursor(&self, x: i32, y: i32) -> Option<bool> {
        best_effort("move_cursor", || {
            let api = self.api()?;
            // A second SetCursorPos corrects the rare first move that lands off target when
            // crossing between monitors with different scale factors.
            for _attempt in 0..2 {
                if !api.set_cursor_pos(x, y) {
                    return Some(false);
                }
                if api.cursor_pos() == Some((x, y)) {
                    return Some(true);
                }
            }
            Some(true)
        })
    }

    fn cursor_position(&self) -> Option<(i32, i32)> {
        best_effort("cursor_position", || self.api()?.cursor_pos())
    }

    // ------------------------------------------------------------------ windows
    fn foreground_window(&self) -> Option<WindowRef> {
        best_effort("foreground_window", || {
            let api = self.api()?;
            let hwnd = api.foreground()?;
            let root = api.root(hwnd);
            match self.classify(&*api, root) {
                (Kind::Normal, pid) => Some(Self::make_ref(&*api, root, pid)),
                _ => None,
            }
        })
    }

    fn window_at(&self, x: i32, y: i32) -> Option<WindowRef> {
        best_effort("window_at", || {
            let api = self.api()?;
            let hit = api.window_from_point(x, y)?;
            let root = api.root(hit);
            match self.classify(&*api, root) {
                (Kind::Normal, pid) => Some(Self::make_ref(&*api, root, pid)),
                (Kind::Shell, _) => None, // desktop or taskbar: no window there
                // One of our own windows or a transient shell surface is on top; the user's
                // window is the next suitable one below it.
                _ => self.window_below(&*api, root, x, y),
            }
        })
    }

    /// Give `window` the foreground despite Windows' foreground lock.
    ///
    /// Strategies are tried from least to most intrusive; the first two inject no input at
    /// all, so the OS idle timer (which drives presence and the typing heuristic) is untouched
    /// in the common case.
    fn activate_window(&self, window: &WindowRef) -> bool {
        best_effort("activate_window", || {
            let api = self.api()?;
            let api = &*api;
            let hwnd = hwnd_of(Some(window))?;
            if !api.is_window(hwnd) || api.is_iconic(hwnd) {
                return None;
            }
            if api.foreground() == Some(hwnd) {
                return Some(true);
            }
            if api.is_hung(hwnd) {
                // An app that stopped pumping messages cannot take focus anyway, and waiting on
                // it would freeze our UI thread. (Windows only reports a window as hung after
                // ~5 s; bring_to_top never waits regardless.)
                log::debug!("Not activating window {hwnd:#x}: it is not responding");
                return None;
            }
            api.allow_set_foreground_any();

            if Self::activate_attached(api, hwnd) {
                return Some(true);
            }
            if api.send_empty_mouse_input() && Self::try_foreground(api, hwnd) {
                return Some(true);
            }
            if api.tap_alt() && Self::try_foreground(api, hwnd) {
                return Some(true);
            }
            Some(api.foreground() == Some(hwnd))
        })
        .unwrap_or(false)
    }

    fn is_window_valid(&self, window: &WindowRef) -> bool {
        best_effort("is_window_valid", || {
            let api = self.api()?;
            let hwnd = hwnd_of(Some(window))?;
            if !api.is_window(hwnd) {
                return None;
            }
            if window
                .pid
                .is_some_and(|pid| api.thread_process(hwnd).1 != pid)
            {
                return None; // the handle was recycled for another process's window
            }
            Some(api.is_visible(hwnd) && !api.is_iconic(hwnd) && !api.is_cloaked(hwnd))
        })
        .unwrap_or(false)
    }

    fn window_rect(&self, window: &WindowRef) -> Option<Rect> {
        best_effort("window_rect", || {
            let api = self.api()?;
            let hwnd = hwnd_of(Some(window)).filter(|h| api.is_window(*h))?;
            api.frame_rect(hwnd)
        })
    }

    fn window_client_rect(&self, window: &WindowRef) -> Option<Rect> {
        best_effort("window_client_rect", || {
            let api = self.api()?;
            let hwnd = hwnd_of(Some(window)).filter(|h| api.is_window(*h))?;
            api.client_rect(hwnd)
        })
    }

    fn window_app(&self, window: &WindowRef) -> Option<AppIdentity> {
        best_effort("window_app", || {
            let api = self.api()?;
            let hwnd = hwnd_of(Some(window)).filter(|h| api.is_window(*h))?;
            let pid = api.thread_process(hwnd).1;
            if pid == 0 || window.pid.is_some_and(|p| p != pid) {
                return None; // gone, or the handle was recycled for another process's window
            }
            let process = self.host.app_process_name(pid)?;
            Some(AppIdentity {
                process,
                app_id: api.class_name(hwnd),
            })
        })
    }

    // ------------------------------------------------------------------- camera
    fn camera_in_use_by_other_app(&self) -> Option<bool> {
        best_effort("camera_in_use_by_other_app", || {
            let users = match active_camera_users(&*self.registry) {
                Ok(users) => users,
                Err(err) => {
                    log::debug!("camera_in_use_by_other_app failed: {err}");
                    return None;
                }
            };
            let mut others: Vec<CameraUser> = users
                .into_iter()
                .filter(|user| !self.is_own_camera_user(user))
                .collect();
            if others.is_empty() {
                return Some(false);
            }
            // An app that crashed, or a machine that lost power, mid-session never records the
            // stop time, and the registry keeps that open session across reboots. Sessions
            // opened before this boot are certainly over.
            if let Some(boot) = self.host.boot_filetime() {
                // Generous slack: the boot time is derived from the current clock, which NTP
                // may have moved since. The process check below is exact.
                others.retain(|u| u.start >= boot.saturating_sub(BOOT_TIME_SLACK));
                if others.is_empty() {
                    return Some(false);
                }
            }
            let processes = self.host.processes().unwrap_or_default();
            if processes.is_empty() {
                return Some(true); // cannot tell live sessions from stale ones: trust the registry
            }
            let api = self.api()?;
            Some(
                others
                    .iter()
                    .any(|user| self.session_is_live(&*api, user, &processes)),
            )
        })
    }

    // -------------------------------------------------------------- permissions
    fn permissions(&self) -> HashMap<&'static str, Option<bool>> {
        HashMap::from([
            ("camera", self.camera_permission()),
            ("accessibility", None),
        ])
    }

    fn open_permission_settings(&self, name: &str) -> bool {
        SETTINGS_URIS
            .iter()
            .find(|(key, _)| *key == name)
            .is_some_and(|(_, uri)| self.host.open_uri(uri))
    }
}

// ------------------------------------------------------------ real binding
#[cfg(windows)]
mod win32 {
    //! Typed Win32 calls through the `windows` crate. Functions that older Windows versions
    //! lack are looked up at run time, so their absence degrades instead of failing to load.

    use std::collections::HashMap;
    use std::ffi::c_void;
    use std::mem::size_of;

    use windows::Win32::Foundation::{
        CloseHandle, ERROR_MORE_DATA, ERROR_SUCCESS as WIN32_SUCCESS, GetLastError, HANDLE, HWND,
        LPARAM, POINT, RECT, WPARAM,
    };
    use windows::Win32::Graphics::Gdi::ClientToScreen;
    use windows::Win32::System::LibraryLoader::{
        GetModuleFileNameW, GetModuleHandleW, GetProcAddress, LoadLibraryW,
    };
    use windows::Win32::System::Power::{EXECUTION_STATE, SetThreadExecutionState};
    use windows::Win32::System::Registry::{
        HKEY, HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE, KEY_READ, REG_DWORD, REG_EXPAND_SZ, REG_QWORD,
        REG_SZ, REG_VALUE_TYPE, RegCloseKey, RegEnumKeyExW, RegEnumValueW, RegOpenKeyExW,
    };
    use windows::Win32::System::Shutdown::LockWorkStation;
    use windows::Win32::System::StationsAndDesktops::{
        CloseDesktop, DESKTOP_ACCESS_FLAGS, DESKTOP_CONTROL_FLAGS, GetUserObjectInformationW,
        OpenInputDesktop, USER_OBJECT_INFORMATION_INDEX,
    };
    use windows::Win32::System::SystemInformation::GetTickCount64;
    use windows::Win32::System::Threading::{
        AttachThreadInput, GetCurrentThreadId, OpenProcess, PROCESS_ACCESS_RIGHTS,
    };
    use windows::Win32::UI::Input::KeyboardAndMouse::{
        GetLastInputInfo, INPUT, INPUT_0, INPUT_TYPE, KEYBD_EVENT_FLAGS, KEYBDINPUT, LASTINPUTINFO,
        MOUSE_EVENT_FLAGS, MOUSEINPUT, SendInput, VIRTUAL_KEY,
    };
    use windows::Win32::UI::Shell::ShellExecuteW;
    use windows::Win32::UI::WindowsAndMessaging::{
        AllowSetForegroundWindow, CreateWindowExW, DestroyWindow, GET_ANCESTOR_FLAGS,
        GET_WINDOW_CMD, GetAncestor, GetClassNameW, GetClientRect, GetCursorPos,
        GetForegroundWindow, GetWindow, GetWindowLongW, GetWindowRect, GetWindowThreadProcessId,
        IsIconic, IsWindow, IsWindowVisible, PostMessageW, SET_WINDOW_POS_FLAGS, SW_SHOWNORMAL,
        SendMessageW, SetCursorPos, SetForegroundWindow, SetWindowPos, WINDOW_EX_STYLE,
        WINDOW_LONG_PTR_INDEX, WINDOW_STYLE, WindowFromPoint,
    };
    use windows::core::{HSTRING, PCSTR, PCWSTR, PWSTR, w};

    use super::*;

    type IsHungAppWindowFn = unsafe extern "system" fn(HWND) -> i32;
    type SetProcessDpiAwarenessContextFn = unsafe extern "system" fn(isize) -> i32;
    type SetProcessDpiAwareFn = unsafe extern "system" fn() -> i32;
    type SetProcessDpiAwarenessFn = unsafe extern "system" fn(i32) -> i32;
    type SetAppUserModelIdFn = unsafe extern "system" fn(PCWSTR) -> i32;
    type GetCurrentPackageFamilyNameFn = unsafe extern "system" fn(*mut u32, PWSTR) -> i32;
    type GetPackageFamilyNameFn = unsafe extern "system" fn(HANDLE, *mut u32, PWSTR) -> i32;
    type DwmGetWindowAttributeFn = unsafe extern "system" fn(HWND, u32, *mut c_void, u32) -> i32;

    /// `name` from `dll`, or `None` when this Windows version lacks it.
    ///
    /// # Safety
    /// `F` must be the function's real prototype.
    unsafe fn optional<F: Copy>(dll: PCWSTR, name: &[u8]) -> Option<F> {
        // SAFETY: plain library and symbol lookups; `name` is NUL-terminated.
        let address = unsafe {
            let module = LoadLibraryW(dll).ok()?;
            GetProcAddress(module, PCSTR(name.as_ptr()))
        };
        if address.is_none() {
            log::debug!(
                "{} is not available on this Windows version",
                String::from_utf8_lossy(&name[..name.len() - 1])
            );
        }
        // SAFETY: the caller names the prototype; function pointers have the same size.
        address.map(|f| unsafe { std::mem::transmute_copy::<_, F>(&f) })
    }

    fn hwnd(handle: isize) -> HWND {
        HWND(handle as *mut c_void)
    }

    fn handle_of(window: HWND) -> Option<isize> {
        (!window.is_invalid()).then_some(window.0 as isize)
    }

    fn utf16_until_nul(buf: &[u16]) -> String {
        let end = buf.iter().position(|c| *c == 0).unwrap_or(buf.len());
        String::from_utf16_lossy(&buf[..end])
    }

    fn rect_of(rect: &RECT) -> Option<Rect> {
        let (width, height) = (rect.right - rect.left, rect.bottom - rect.top);
        (width > 0 && height > 0).then(|| Rect::new(rect.left, rect.top, width, height))
    }

    /// The Win32 binding. Statically linked functions exist on every supported Windows; the
    /// optional ones are resolved once here.
    pub(super) struct Win32 {
        is_hung_app_window: Option<IsHungAppWindowFn>,
        set_process_dpi_awareness_context: Option<SetProcessDpiAwarenessContextFn>,
        set_process_dpi_aware: Option<SetProcessDpiAwareFn>,
        set_process_dpi_awareness: Option<SetProcessDpiAwarenessFn>,
        set_app_user_model_id: Option<SetAppUserModelIdFn>,
        get_current_package_family_name: Option<GetCurrentPackageFamilyNameFn>,
        get_package_family_name: Option<GetPackageFamilyNameFn>,
        dwm_get_window_attribute: Option<DwmGetWindowAttributeFn>,
    }

    impl Win32 {
        pub(super) fn new() -> Self {
            // SAFETY: each type alias above is the documented prototype of its function.
            unsafe {
                Self {
                    is_hung_app_window: optional(w!("user32.dll"), b"IsHungAppWindow\0"),
                    set_process_dpi_awareness_context: optional(
                        w!("user32.dll"),
                        b"SetProcessDpiAwarenessContext\0",
                    ),
                    set_process_dpi_aware: optional(w!("user32.dll"), b"SetProcessDPIAware\0"),
                    set_process_dpi_awareness: optional(
                        w!("shcore.dll"),
                        b"SetProcessDpiAwareness\0",
                    ),
                    set_app_user_model_id: optional(
                        w!("shell32.dll"),
                        b"SetCurrentProcessExplicitAppUserModelID\0",
                    ),
                    get_current_package_family_name: optional(
                        w!("kernel32.dll"),
                        b"GetCurrentPackageFamilyName\0",
                    ),
                    get_package_family_name: optional(
                        w!("kernel32.dll"),
                        b"GetPackageFamilyName\0",
                    ),
                    dwm_get_window_attribute: optional(
                        w!("dwmapi.dll"),
                        b"DwmGetWindowAttribute\0",
                    ),
                }
            }
        }

        fn send(inputs: &[INPUT]) -> bool {
            // SAFETY: a valid INPUT slice with its element size.
            let sent = unsafe { SendInput(inputs, size_of::<INPUT>() as i32) };
            sent as usize == inputs.len()
        }

        fn mouse_input(dx: i32, flags: u32) -> INPUT {
            INPUT {
                r#type: INPUT_TYPE(INPUT_MOUSE),
                Anonymous: INPUT_0 {
                    mi: MOUSEINPUT {
                        dx,
                        dy: 0,
                        mouseData: 0,
                        dwFlags: MOUSE_EVENT_FLAGS(flags),
                        time: 0,
                        dwExtraInfo: 0,
                    },
                },
            }
        }

        fn key_input(vk: u16, flags: u32) -> INPUT {
            INPUT {
                r#type: INPUT_TYPE(INPUT_KEYBOARD),
                Anonymous: INPUT_0 {
                    ki: KEYBDINPUT {
                        wVk: VIRTUAL_KEY(vk),
                        wScan: 0,
                        dwFlags: KEYBD_EVENT_FLAGS(flags),
                        time: 0,
                        dwExtraInfo: 0,
                    },
                },
            }
        }

        /// Run a `Get*PackageFamilyName`-style call: ask for the size, then the name.
        fn read_package_name(call: impl Fn(*mut u32, PWSTR) -> i32) -> Option<String> {
            let mut length = 0u32;
            let status = call(&mut length, PWSTR::null());
            // APPMODEL_ERROR_NO_PACKAGE: no package identity. Anything but "buffer too small"
            // (with the size filled in) means there is no name to read.
            if status != ERROR_INSUFFICIENT_BUFFER || length == 0 {
                return None;
            }
            let mut buf = vec![0u16; length as usize];
            if call(&mut length, PWSTR(buf.as_mut_ptr())) != ERROR_SUCCESS {
                return None;
            }
            Some(utf16_until_nul(&buf)).filter(|name| !name.is_empty())
        }
    }

    // SAFETY (for every `unsafe` block below): Win32 calls with valid, live arguments; handles
    // that went stale make the calls fail, which is reported.
    impl Win32Api for Win32 {
        fn foreground(&self) -> Option<isize> {
            handle_of(unsafe { GetForegroundWindow() })
        }

        fn root(&self, window: isize) -> isize {
            handle_of(unsafe { GetAncestor(hwnd(window), GET_ANCESTOR_FLAGS(GA_ROOT)) })
                .unwrap_or(window)
        }

        fn thread_process(&self, window: isize) -> (u32, u32) {
            let mut pid = 0u32;
            let tid = unsafe { GetWindowThreadProcessId(hwnd(window), Some(&mut pid)) };
            (tid, pid)
        }

        fn class_name(&self, window: isize) -> String {
            let mut buf = [0u16; 257];
            let n = unsafe { GetClassNameW(hwnd(window), &mut buf) };
            if n > 0 {
                String::from_utf16_lossy(&buf[..n as usize])
            } else {
                String::new()
            }
        }

        fn frame_rect(&self, window: isize) -> Option<Rect> {
            let mut rect = RECT::default();
            let mut ok = false;
            if let Some(dwm_get_window_attribute) = self.dwm_get_window_attribute {
                let hr = unsafe {
                    dwm_get_window_attribute(
                        hwnd(window),
                        DWMWA_EXTENDED_FRAME_BOUNDS,
                        (&raw mut rect).cast(),
                        size_of::<RECT>() as u32,
                    )
                };
                ok = hr == 0;
            }
            if !ok {
                ok = unsafe { GetWindowRect(hwnd(window), &mut rect) }.is_ok();
            }
            if !ok {
                return None;
            }
            rect_of(&rect)
        }

        fn client_rect(&self, window: isize) -> Option<Rect> {
            let mut rect = RECT::default();
            unsafe { GetClientRect(hwnd(window), &mut rect) }.ok()?;
            let mut origin = POINT { x: 0, y: 0 };
            if !unsafe { ClientToScreen(hwnd(window), &mut origin) }.as_bool() {
                return None;
            }
            let size = rect_of(&rect)?;
            Some(Rect::new(origin.x, origin.y, size.w, size.h))
        }

        fn is_window(&self, window: isize) -> bool {
            unsafe { IsWindow(Some(hwnd(window))) }.as_bool()
        }

        fn is_visible(&self, window: isize) -> bool {
            unsafe { IsWindowVisible(hwnd(window)) }.as_bool()
        }

        fn is_iconic(&self, window: isize) -> bool {
            unsafe { IsIconic(hwnd(window)) }.as_bool()
        }

        fn is_hung(&self, window: isize) -> bool {
            self.is_hung_app_window
                .is_some_and(|is_hung| unsafe { is_hung(hwnd(window)) } != 0)
        }

        fn is_cloaked(&self, window: isize) -> bool {
            let Some(dwm_get_window_attribute) = self.dwm_get_window_attribute else {
                return false;
            };
            let mut value = 0u32;
            let hr = unsafe {
                dwm_get_window_attribute(
                    hwnd(window),
                    DWMWA_CLOAKED,
                    (&raw mut value).cast(),
                    size_of::<u32>() as u32,
                )
            };
            hr == 0 && value != 0
        }

        fn ex_style(&self, window: isize) -> u32 {
            (unsafe { GetWindowLongW(hwnd(window), WINDOW_LONG_PTR_INDEX(GWL_EXSTYLE)) }) as u32
        }

        fn window_from_point(&self, x: i32, y: i32) -> Option<isize> {
            handle_of(unsafe { WindowFromPoint(POINT { x, y }) })
        }

        fn next_window(&self, window: isize) -> Option<isize> {
            unsafe { GetWindow(hwnd(window), GET_WINDOW_CMD(GW_HWNDNEXT)) }
                .ok()
                .and_then(handle_of)
        }

        fn set_foreground(&self, window: isize) -> bool {
            unsafe { SetForegroundWindow(hwnd(window)) }.as_bool()
        }

        fn bring_to_top(&self, window: isize) -> bool {
            let flags = SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_ASYNCWINDOWPOS;
            unsafe {
                SetWindowPos(
                    hwnd(window),
                    Some(hwnd(HWND_TOP)),
                    0,
                    0,
                    0,
                    0,
                    SET_WINDOW_POS_FLAGS(flags),
                )
            }
            .is_ok()
        }

        fn allow_set_foreground_any(&self) {
            let _ = unsafe { AllowSetForegroundWindow(ASFW_ANY) };
        }

        fn current_thread_id(&self) -> u32 {
            unsafe { GetCurrentThreadId() }
        }

        fn attach_thread_input(&self, thread: u32, to_thread: u32, attach: bool) -> bool {
            unsafe { AttachThreadInput(thread, to_thread, attach) }.as_bool()
        }

        fn post_broadcast(&self, msg: u32, wparam: usize, lparam: isize) -> bool {
            unsafe {
                PostMessageW(
                    Some(hwnd(HWND_BROADCAST)),
                    msg,
                    WPARAM(wparam),
                    LPARAM(lparam),
                )
            }
            .is_ok()
        }

        fn create_hidden_window(&self) -> Option<isize> {
            let instance = unsafe { GetModuleHandleW(PCWSTR::null()) }.ok()?;
            let window = unsafe {
                CreateWindowExW(
                    WINDOW_EX_STYLE(WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE),
                    w!("STATIC"),
                    PCWSTR::null(),
                    WINDOW_STYLE(WS_POPUP),
                    0,
                    0,
                    0,
                    0,
                    None,
                    None,
                    Some(instance.into()),
                    None,
                )
            }
            .ok()?;
            handle_of(window)
        }

        fn send_message(&self, window: isize, msg: u32, wparam: usize, lparam: isize) -> isize {
            unsafe {
                SendMessageW(
                    hwnd(window),
                    msg,
                    Some(WPARAM(wparam)),
                    Some(LPARAM(lparam)),
                )
            }
            .0
        }

        fn destroy_window(&self, window: isize) -> bool {
            unsafe { DestroyWindow(hwnd(window)) }.is_ok()
        }

        fn send_empty_mouse_input(&self) -> bool {
            Self::send(&[Self::mouse_input(0, 0)])
        }

        fn tap_alt(&self) -> bool {
            Self::send(&[
                Self::key_input(VK_MENU, 0),
                Self::key_input(VK_MENU_MASK, 0),
                Self::key_input(VK_MENU_MASK, KEYEVENTF_KEYUP),
                Self::key_input(VK_MENU, KEYEVENTF_KEYUP),
            ])
        }

        fn nudge_mouse(&self) -> bool {
            Self::send(&[
                Self::mouse_input(1, MOUSEEVENTF_MOVE),
                Self::mouse_input(-1, MOUSEEVENTF_MOVE),
            ])
        }

        fn idle_ms(&self) -> Option<u64> {
            let mut info = LASTINPUTINFO {
                cbSize: size_of::<LASTINPUTINFO>() as u32,
                dwTime: 0,
            };
            if !unsafe { GetLastInputInfo(&mut info) }.as_bool() {
                return None;
            }
            Some(elapsed_ms(
                unsafe { GetTickCount64() },
                u64::from(info.dwTime),
            ))
        }

        fn set_cursor_pos(&self, x: i32, y: i32) -> bool {
            unsafe { SetCursorPos(x, y) }.is_ok()
        }

        fn cursor_pos(&self) -> Option<(i32, i32)> {
            let mut point = POINT::default();
            unsafe { GetCursorPos(&mut point) }.ok()?;
            Some((point.x, point.y))
        }

        fn lock_workstation(&self) -> bool {
            unsafe { LockWorkStation() }.is_ok()
        }

        fn set_display_required(&self) {
            // Without ES_CONTINUOUS this only resets the display idle timer once.
            unsafe { SetThreadExecutionState(EXECUTION_STATE(ES_DISPLAY_REQUIRED)) };
        }

        fn input_desktop_name(&self) -> Option<String> {
            let desktop = unsafe {
                OpenInputDesktop(
                    DESKTOP_CONTROL_FLAGS(0),
                    false,
                    DESKTOP_ACCESS_FLAGS(DESKTOP_SWITCHDESKTOP),
                )
            }
            .ok()?;
            let mut buf = [0u16; 256];
            let mut needed = 0u32;
            let ok = unsafe {
                GetUserObjectInformationW(
                    HANDLE(desktop.0),
                    USER_OBJECT_INFORMATION_INDEX(UOI_NAME),
                    Some(buf.as_mut_ptr().cast()),
                    size_of_val(&buf) as u32,
                    Some(&mut needed),
                )
            }
            .is_ok();
            let _ = unsafe { CloseDesktop(desktop) };
            Some(if ok {
                utf16_until_nul(&buf)
            } else {
                String::new()
            })
        }

        fn enable_per_monitor_dpi(&self) -> &'static str {
            if let Some(set_context) = self.set_process_dpi_awareness_context {
                if unsafe { set_context(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) } != 0 {
                    return "per-monitor-v2";
                }
                if unsafe { GetLastError() }.0 == ERROR_ACCESS_DENIED {
                    return "already set";
                }
            }
            if let Some(set_awareness) = self.set_process_dpi_awareness {
                let hr = unsafe { set_awareness(PROCESS_PER_MONITOR_DPI_AWARE) };
                if hr == 0 {
                    return "per-monitor";
                }
                if hr == E_ACCESSDENIED {
                    return "already set";
                }
            }
            if let Some(set_aware) = self.set_process_dpi_aware
                && unsafe { set_aware() } != 0
            {
                return "system";
            }
            "unaware"
        }

        fn set_app_user_model_id(&self, app_id: &str) -> bool {
            let Some(set_id) = self.set_app_user_model_id else {
                return false;
            };
            let app_id = HSTRING::from(app_id);
            unsafe { set_id(PCWSTR(app_id.as_ptr())) >= 0 }
        }

        fn process_image_path(&self) -> Option<String> {
            let mut buf = vec![0u16; 32768];
            let n = unsafe { GetModuleFileNameW(None, &mut buf) } as usize;
            (0 < n && n < buf.len()).then(|| String::from_utf16_lossy(&buf[..n]))
        }

        fn current_package_family_name(&self) -> Option<String> {
            let get_name = self.get_current_package_family_name?;
            Self::read_package_name(|length, buf| unsafe { get_name(length, buf) })
        }

        fn process_package_family_name(&self, pid: u32) -> Option<String> {
            let get_name = self.get_package_family_name?;
            let process = unsafe {
                OpenProcess(
                    PROCESS_ACCESS_RIGHTS(PROCESS_QUERY_LIMITED_INFORMATION),
                    false,
                    pid,
                )
            }
            .ok()?;
            let name =
                Self::read_package_name(|length, buf| unsafe { get_name(process, length, buf) });
            let _ = unsafe { CloseHandle(process) };
            name
        }
    }

    // ------------------------------------------------------------- registry
    fn open_key(hive: &str, path: &str) -> Option<HKEY> {
        let root = match hive {
            "HKCU" => HKEY_CURRENT_USER,
            "HKLM" => HKEY_LOCAL_MACHINE,
            _ => return None,
        };
        let mut key = HKEY::default();
        let path = HSTRING::from(path);
        let status = unsafe { RegOpenKeyExW(root, &path, Some(0), KEY_READ, &mut key) };
        (status == WIN32_SUCCESS).then_some(key)
    }

    pub(super) fn registry_subkeys(hive: &str, path: &str, limit: u32) -> Vec<String> {
        let Some(key) = open_key(hive, path) else {
            return Vec::new();
        };
        let mut names = Vec::new();
        let mut buf = vec![0u16; 256];
        for index in 0..limit {
            let mut length = buf.len() as u32;
            let status = unsafe {
                RegEnumKeyExW(
                    key,
                    index,
                    Some(PWSTR(buf.as_mut_ptr())),
                    &mut length,
                    None,
                    None,
                    None,
                    None,
                )
            };
            if status != WIN32_SUCCESS {
                break;
            }
            names.push(String::from_utf16_lossy(&buf[..length as usize]));
        }
        let _ = unsafe { RegCloseKey(key) };
        names
    }

    pub(super) fn registry_values(hive: &str, path: &str, limit: u32) -> HashMap<String, RegValue> {
        let Some(key) = open_key(hive, path) else {
            return HashMap::new();
        };
        let mut out = HashMap::new();
        let mut name = vec![0u16; 16384];
        let mut data = vec![0u8; 1024];
        let mut index = 0;
        while index < limit {
            let mut name_length = name.len() as u32;
            let mut data_length = data.len() as u32;
            let mut kind = REG_VALUE_TYPE::default();
            let status = unsafe {
                RegEnumValueW(
                    key,
                    index,
                    Some(PWSTR(name.as_mut_ptr())),
                    &mut name_length,
                    None,
                    Some(&mut kind.0),
                    Some(data.as_mut_ptr()),
                    Some(&mut data_length),
                )
            };
            if status == ERROR_MORE_DATA && data_length as usize > data.len() {
                data.resize(data_length as usize, 0);
                continue; // read the same value again with room for its data
            }
            if status != WIN32_SUCCESS {
                break;
            }
            let bytes = &data[..data_length as usize];
            let value = match kind {
                REG_DWORD if bytes.len() >= 4 => RegValue::Int(u64::from(u32::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3],
                ]))),
                REG_QWORD if bytes.len() >= 8 => {
                    let mut raw = [0u8; 8];
                    raw.copy_from_slice(&bytes[..8]);
                    RegValue::Int(u64::from_le_bytes(raw))
                }
                REG_SZ | REG_EXPAND_SZ => {
                    let wide: Vec<u16> = bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|pair| u16::from_le_bytes(*pair))
                        .collect();
                    RegValue::Str(utf16_until_nul(&wide))
                }
                _ => RegValue::Other,
            };
            out.insert(
                String::from_utf16_lossy(&name[..name_length as usize]),
                value,
            );
            index += 1;
        }
        let _ = unsafe { RegCloseKey(key) };
        out
    }

    /// Open a URI with the shell, as `os.startfile` did.
    pub(super) fn open_uri(uri: &str) -> bool {
        let target = HSTRING::from(uri);
        let result = unsafe {
            ShellExecuteW(
                None,
                w!("open"),
                &target,
                PCWSTR::null(),
                PCWSTR::null(),
                SW_SHOWNORMAL,
            )
        };
        // ShellExecuteW reports success with a value above 32.
        if result.0 as isize > 32 {
            return true;
        }
        log::warn!(
            "Could not open {uri}: ShellExecute error {}",
            result.0 as isize
        );
        false
    }
}

#[cfg(test)]
#[path = "windows_tests.rs"]
mod tests;
