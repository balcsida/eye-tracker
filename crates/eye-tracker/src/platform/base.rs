//! Operating-system integration contract.
//!
//! Every method is *best effort*: implementations must never panic for an unsupported
//! operation. They return `false`/`None` instead and log at DEBUG level, so the rest of the
//! application can degrade gracefully (for example on a Wayland session, where cursor warping is
//! not allowed).
//!
//! Thread-safety: the trait objects are `Send + Sync`. All methods may be called from the UI
//! thread. Methods marked *any thread* may additionally be called from worker threads.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use parking_lot::Mutex;
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

use crate::types::{AppIdentity, Rect, WindowRef};

/// Process names remembered by [`PlatformServices::app_process_name`].
const APP_NAME_CACHE_SIZE: usize = 64;
/// Executable extensions dropped from process names.
const EXECUTABLE_SUFFIXES: [&str; 3] = [".exe", ".app", ".bin"];

/// `"WindowsTerminal.exe"` → `"windowsterminal"`, `"/usr/bin/kitty"` → `"kitty"`: the
/// lower-cased file name without directory or executable extension.
pub fn process_basename(name: &str) -> String {
    let normalised = name.replace('/', "\\");
    let file = normalised.rsplit(['\\', ':']).next().unwrap_or("");
    let base = file.trim().to_lowercase();
    for suffix in EXECUTABLE_SUFFIXES {
        if let Some(stem) = base.strip_suffix(suffix) {
            return stem.to_string();
        }
    }
    base
}

/// The program called `name`, as `shutil.which` finds it: a path to an executable file, or
/// a name looked up in `PATH` (with the `PATHEXT` extensions on Windows).
pub fn which(name: &str) -> Option<PathBuf> {
    let has_dir = name.contains('/') || (cfg!(windows) && name.contains('\\'));
    if has_dir {
        return executable_candidate(Path::new(name));
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .filter(|dir| !dir.as_os_str().is_empty())
        .find_map(|dir| executable_candidate(&dir.join(name)))
}

fn executable_candidate(path: &Path) -> Option<PathBuf> {
    if cfg!(windows) {
        pathext_candidate(path)
    } else {
        unix_candidate(path)
    }
}

/// `path` as is, or with one of the `PATHEXT` extensions, if that is a file (Windows).
fn pathext_candidate(path: &Path) -> Option<PathBuf> {
    let extensions = std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".into());
    let extensions: Vec<&str> = extensions.split(';').filter(|e| !e.is_empty()).collect();
    let lower = path.to_string_lossy().to_lowercase();
    let candidates: Vec<PathBuf> = if extensions
        .iter()
        .any(|e| lower.ends_with(&e.to_lowercase()))
    {
        vec![path.to_path_buf()]
    } else {
        extensions
            .iter()
            .map(|e| {
                let mut full = path.as_os_str().to_owned();
                full.push(e);
                full.into()
            })
            .collect()
    };
    candidates.into_iter().find(|c| c.is_file())
}

/// `path` if it is an executable file (Unix).
fn unix_candidate(path: &Path) -> Option<PathBuf> {
    let meta = std::fs::metadata(path).ok()?;
    #[cfg(unix)]
    let executable = {
        use std::os::unix::fs::PermissionsExt;
        meta.permissions().mode() & 0o111 != 0
    };
    #[cfg(not(unix))]
    let executable = true;
    (meta.is_file() && executable).then(|| path.to_path_buf())
}

/// Names cached by `(pid, process creation time)`, oldest first.
///
/// The operating system reuses the pid of a process that quit, and a cache keyed by the pid
/// alone would give a new process (VS Code, a browser) the name of the old one (an app that may
/// be inspected). The creation time is read on every call; a process that has gone, or whose
/// creation time cannot be read, is not cached. The cache is bounded.
#[derive(Default)]
pub struct AppNameCache {
    entries: Mutex<Vec<((u32, u64), String)>>,
}

impl AppNameCache {
    /// Lower-cased executable name of `pid` without its extension.
    ///
    /// `created` and `name` read the process creation time and its name; `name` is only called
    /// on a cache miss.
    pub fn lookup(
        &self,
        pid: Option<i64>,
        created: impl Fn(u32) -> Option<u64>,
        name: impl Fn(u32) -> Option<String>,
    ) -> Option<String> {
        let pid = u32::try_from(pid.filter(|p| *p > 0)?).ok()?;
        let key = (pid, created(pid)?);
        let mut entries = self.entries.lock();
        if let Some((_, cached)) = entries.iter().find(|(k, _)| *k == key) {
            return Some(cached.clone());
        }
        let name = process_basename(&name(pid)?);
        if name.is_empty() {
            return None;
        }
        entries.retain(|(k, _)| k.0 != pid); // an earlier process with this pid
        if entries.len() >= APP_NAME_CACHE_SIZE {
            entries.remove(0);
        }
        entries.push((key, name.clone()));
        Some(name)
    }

    pub fn len(&self) -> usize {
        self.entries.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

static APP_NAMES: LazyLock<AppNameCache> = LazyLock::new(AppNameCache::default);

/// Reads one process's creation time and name through sysinfo.
fn system_process(pid: u32) -> Option<(u64, String)> {
    let pid = Pid::from_u32(pid);
    let mut system = System::new();
    system.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::nothing(),
    );
    let process = system.process(pid)?;
    Some((
        process.start_time(),
        process.name().to_string_lossy().into_owned(),
    ))
}

/// The OS integration contract. The default implementations do nothing: everything is
/// unsupported.
pub trait PlatformServices: Send + Sync {
    /// Implementation name, for logs.
    fn name(&self) -> &'static str {
        "generic"
    }

    // ---------------------------------------------------------------- lifecycle
    /// Called once, before the UI is created.
    ///
    /// Implementations set DPI awareness and UI environment variables so that global
    /// coordinates equal the native coordinates used by this trait (see `crate::types`).
    fn prepare_process(&self) {}

    /// Feature matrix shown by `eye-tracker doctor` and the settings UI.
    ///
    /// Keys: `lock`, `display_off`, `wake_display`, `input_idle`, `key_idle`,
    /// `session_locked`, `focus`, `cursor`, `camera_in_use`, `hotkeys`, `panes`
    /// ([`window_app`](Self::window_app) and [`window_client_rect`](Self::window_client_rect)
    /// work, so split panes of supported terminals can be followed).
    fn capabilities(&self) -> HashMap<&'static str, bool> {
        HashMap::from([
            ("lock", false),
            ("display_off", false),
            ("wake_display", false),
            ("input_idle", false),
            ("key_idle", false),
            ("session_locked", false),
            ("focus", false),
            ("cursor", true),
            ("camera_in_use", false),
            ("hotkeys", false),
            ("panes", false),
        ])
    }

    // ------------------------------------------------------------ session/power
    /// Lock the session. *Any thread.*
    fn lock_screen(&self) -> bool {
        false
    }

    /// Put all displays to sleep (without locking). *Any thread.*
    fn display_off(&self) -> bool {
        false
    }

    /// Wake the displays after `display_off`. *Any thread.*
    fn wake_display(&self) -> bool {
        false
    }

    /// Whether the session is locked / the lock screen is shown. *Any thread.*
    fn is_session_locked(&self) -> Option<bool> {
        None
    }

    // -------------------------------------------------------------------- input
    /// Seconds since the last keyboard or mouse input system-wide. *Any thread.*
    fn seconds_since_input(&self) -> Option<f64> {
        None
    }

    /// Seconds since the last *keyboard* input, where the OS can tell. *Any thread.*
    ///
    /// Only a timestamp is ever read; key contents are never observed.
    fn seconds_since_key_input(&self) -> Option<f64> {
        None
    }

    /// Whether the cursor position the UI reports tracks the pointer everywhere on the desktop.
    ///
    /// `false` on Wayland: an XWayland client only sees pointer motion over X11 windows, so the
    /// position it reports can be stale. The controller then stops deriving the current
    /// monitor, manual mouse use and learning labels from the cursor position.
    fn cursor_position_reliable(&self) -> bool {
        true
    }

    /// Move the pointer. `None` means this system cannot move it (counted as a refusal).
    fn move_cursor(&self, _x: i32, _y: i32) -> Option<bool> {
        None
    }

    /// The pointer position in global coordinates (Qt's `QCursor.pos()` in the Python app);
    /// `None` when it cannot be read.
    fn cursor_position(&self) -> Option<(i32, i32)> {
        None
    }

    // ------------------------------------------------------------------ windows
    /// The window that currently has keyboard focus (with its `rect`).
    fn foreground_window(&self) -> Option<WindowRef> {
        None
    }

    /// Top-level window under a screen point (excluding our own windows).
    fn window_at(&self, _x: i32, _y: i32) -> Option<WindowRef> {
        None
    }

    /// Bring a window to the front and give it keyboard focus (no synthetic click).
    fn activate_window(&self, _window: &WindowRef) -> bool {
        false
    }

    /// Window still exists, is visible, not minimised, and on the current virtual desktop /
    /// Space.
    ///
    /// Activating a window on another virtual desktop (Windows), workspace (X11) or Space
    /// (macOS) would make the system switch to it, so such windows are not valid switch
    /// targets.
    fn is_window_valid(&self, _window: &WindowRef) -> bool {
        false
    }

    fn window_rect(&self, _window: &WindowRef) -> Option<Rect> {
        None
    }

    /// The window's content area (without frame and title bar), in the same global coordinates
    /// as [`window_rect`](Self::window_rect). *Any thread.*
    ///
    /// Best effort: where the content area cannot be told apart it may be the frame rectangle.
    /// Split panes are mapped onto it.
    fn window_client_rect(&self, _window: &WindowRef) -> Option<Rect> {
        None
    }

    /// The application the window belongs to (process name and class / bundle id). *Any
    /// thread.*
    ///
    /// Never reads the window title. `None` when unknown.
    fn window_app(&self, _window: &WindowRef) -> Option<AppIdentity> {
        None
    }

    /// Lower-cased executable name of `pid` without its extension.
    ///
    /// Shared by the implementations of [`window_app`](Self::window_app). Names are cached by
    /// `(pid, process creation time)`; see [`AppNameCache`].
    fn app_process_name(&self, pid: Option<i64>) -> Option<String> {
        APP_NAMES.lookup(
            pid,
            |pid| system_process(pid).map(|(created, _)| created),
            |pid| system_process(pid).map(|(_, name)| name),
        )
    }

    fn same_window(&self, a: Option<&WindowRef>, b: Option<&WindowRef>) -> bool {
        match (a, b) {
            (Some(a), Some(b)) => a.handle == b.handle,
            _ => false,
        }
    }

    // ------------------------------------------------------------------- camera
    /// Whether another process is streaming from a camera, or was just refused one (Linux).
    /// *Any thread.*
    ///
    /// Implementations may cache the answer for a few seconds (the check can cost
    /// milliseconds). `None` means unknown.
    fn camera_in_use_by_other_app(&self) -> Option<bool> {
        None
    }

    /// Lower-cased executable names of running processes. *Any thread.*
    fn running_process_names(&self) -> HashSet<String> {
        let mut system = System::new();
        system.refresh_processes_specifics(
            ProcessesToUpdate::All,
            true,
            ProcessRefreshKind::nothing(),
        );
        system
            .processes()
            .values()
            .map(|p| p.name().to_string_lossy().to_lowercase())
            .filter(|name| !name.is_empty())
            .collect()
    }

    // -------------------------------------------------------------- permissions
    /// Permission status, e.g. `{"camera": Some(true), "accessibility": None}`.
    ///
    /// `None` means "not applicable / unknown".
    fn permissions(&self) -> HashMap<&'static str, Option<bool>> {
        HashMap::from([("camera", None), ("accessibility", None)])
    }

    /// Trigger the OS permission prompt, where one exists.
    fn request_permission(&self, _name: &str) {}

    /// Open the OS settings page for a permission.
    fn open_permission_settings(&self, _name: &str) -> bool {
        false
    }

    /// The Accessibility permission (macOS) in more detail than
    /// [`permissions`](Self::permissions).
    ///
    /// One of `"granted"`, `"missing"`, `"stale"` (granted to a previous build of the app,
    /// which no longer applies) or `"unknown"` (not applicable on this system, or cannot be
    /// determined).
    fn accessibility_status(&self) -> &'static str {
        "unknown"
    }

    /// Keep timers unthrottled while tracking (macOS App Nap); a no-op elsewhere.
    ///
    /// `true` asks the system not to throttle the process, `false` allows it again. Returns
    /// whether the requested state is in effect.
    fn set_background_activity(&self, _active: bool) -> bool {
        true
    }

    // ------------------------------------------------------------------- misc
    fn is_wayland(&self) -> bool {
        false
    }
}

/// Default implementation: everything unsupported.
#[derive(Debug, Default)]
pub struct GenericPlatform;

impl PlatformServices for GenericPlatform {}

#[cfg(test)]
#[path = "base_tests.rs"]
mod tests;
