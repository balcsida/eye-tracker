//! Tests for `platform::create` (the Windows arm) and `platform::windows`.
//!
//! The policy code in `WindowsPlatform` is exercised on every OS through a fake Win32 binding,
//! a fake registry and a fake host. A handful of live tests run only on Windows and only call
//! read-only APIs: nothing here locks the screen, turns a display off, synthesises input, moves
//! the cursor or activates a window.
//!
//! Not ported:
//! - `test_singleton`, `test_reset_for_tests`: covered by `get_platform_is_a_singleton_until_reset`
//!   in `platform/mod.rs` (a second test resetting the shared cache would race with it).
//! - `test_import_error_falls_back_with_warning`, `test_non_platform_class_is_rejected`: they
//!   patched Python's dynamic import table; Rust links the implementations statically.
//! - `test_prepare_process_keeps_user_override_but_warns` and the environment half of
//!   `test_prepare_process_sets_env_and_calls_api` / `test_prepare_process_survives_api_failure`:
//!   `QT_ENABLE_HIGHDPI_SCALING` only steered Qt; the egui app does not read it.
//! - `test_process_listing_asks_psutil_for_names_only`: it spied on psutil; the sysinfo listing
//!   requests `ProcessRefreshKind::nothing()` (names only) by construction.
//! - `test_every_binding_has_a_prototype`: ctypes `argtypes`/`restype` have no Rust equivalent
//!   (the `windows` crate's prototypes are typed at compile time).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::Mutex;

use super::*;
use crate::platform::{GenericPlatform, PlatformServices, create};

const OWN_PID: u32 = 4242;
const OUR_THREAD: u32 = 1;

// --------------------------------------------------------------------- fakes
#[derive(Debug, Clone)]
struct FakeWindow {
    hwnd: isize,
    pid: u32,
    tid: u32,
    cls: &'static str,
    rect: Option<Rect>,
    visible: bool,
    iconic: bool,
    cloaked: bool,
    hung: bool,
    ex_style: u32,
    /// Top-level ancestor for child windows.
    root: Option<isize>,
    /// Client area (`None`: like the frame).
    client: Option<Rect>,
}

fn win(hwnd: isize) -> FakeWindow {
    FakeWindow {
        hwnd,
        pid: 100,
        tid: 10,
        cls: "Notepad",
        rect: Some(Rect::new(0, 0, 800, 600)),
        visible: true,
        iconic: false,
        cloaked: false,
        hung: false,
        ex_style: 0,
        root: None,
        client: None,
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Call {
    SetForeground(isize),
    BringToTop(isize),
    AllowAny,
    Attach(u32, u32, bool),
    PostBroadcast(u32, usize, isize),
    CreateWindow,
    SendMessage(isize, u32, usize, isize),
    DestroyWindow(isize),
    EmptyInput,
    TapAlt,
    Nudge,
    SetCursor(i32, i32),
    Lock,
    DisplayRequired,
    Dpi,
    Aumid(String),
    PackageOf(u32),
}

impl Call {
    fn name(&self) -> &'static str {
        match self {
            Self::SetForeground(_) => "set_foreground",
            Self::BringToTop(_) => "bring_to_top",
            Self::AllowAny => "allow_any",
            Self::Attach(..) => "attach",
            Self::PostBroadcast(..) => "post_broadcast",
            Self::CreateWindow => "create_window",
            Self::SendMessage(..) => "send_message",
            Self::DestroyWindow(_) => "destroy_window",
            Self::EmptyInput => "empty_input",
            Self::TapAlt => "tap_alt",
            Self::Nudge => "nudge",
            Self::SetCursor(..) => "set_cursor",
            Self::Lock => "lock",
            Self::DisplayRequired => "display_required",
            Self::Dpi => "dpi",
            Self::Aumid(_) => "aumid",
            Self::PackageOf(_) => "package_of",
        }
    }
}

/// Which strategy lets `set_foreground` succeed.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Unlock {
    Any,
    Attach,
    Input,
    Alt,
    Never,
}

struct FakeState {
    windows: HashMap<isize, FakeWindow>,
    zorder: Vec<isize>,
    fg: Option<isize>,
    unlock: Unlock,
    hit: Option<isize>,
    calls: Vec<Call>,
    attached: HashSet<(u32, u32)>,
    injected: bool,
    alt: bool,
    idle: Option<u64>,
    desktop: Option<String>,
    cursor: (i32, i32),
    cursor_offsets: Vec<(i32, i32)>,
    cursor_ok: bool,
    image_path: Option<String>,
    can_create_window: bool,
    own_package: Option<String>,
    /// pid -> package family name.
    packages: HashMap<u32, String>,
    /// `send_message` panics (the Python test made it raise).
    send_fails: bool,
    /// `current_package_family_name` panics (the Python `NoPackageApi`).
    package_lookup_fails: bool,
}

/// Stand-in for the Win32 binding with a scriptable foreground lock.
struct FakeWin32 {
    state: Mutex<FakeState>,
}

impl FakeWin32 {
    fn new(windows: Vec<FakeWindow>) -> Self {
        Self::with(windows, None, Unlock::Any, None)
    }

    fn with(
        windows: Vec<FakeWindow>,
        foreground: Option<isize>,
        unlock: Unlock,
        hit: Option<isize>,
    ) -> Self {
        Self {
            state: Mutex::new(FakeState {
                zorder: windows.iter().map(|w| w.hwnd).collect(),
                windows: windows.into_iter().map(|w| (w.hwnd, w)).collect(),
                fg: foreground,
                unlock,
                hit,
                calls: Vec::new(),
                attached: HashSet::new(),
                injected: false,
                alt: false,
                idle: Some(1500),
                desktop: Some("Default".into()),
                cursor: (10, 20),
                cursor_offsets: Vec::new(),
                cursor_ok: true,
                image_path: Some(r"C:\Python\python.exe".into()),
                can_create_window: true,
                own_package: None,
                packages: HashMap::new(),
                send_fails: false,
                package_lookup_fails: false,
            }),
        }
    }

    fn s(&self) -> parking_lot::MutexGuard<'_, FakeState> {
        self.state.lock()
    }

    fn calls(&self) -> Vec<Call> {
        self.s().calls.clone()
    }

    fn names(&self) -> Vec<&'static str> {
        self.s().calls.iter().map(Call::name).collect()
    }

    fn record(&self, call: Call) {
        self.s().calls.push(call);
    }

    fn window(&self, hwnd: isize) -> FakeWindow {
        self.s().windows[&hwnd].clone()
    }

    fn allowed(&self) -> bool {
        let s = self.s();
        match s.unlock {
            Unlock::Any => true,
            Unlock::Attach => {
                let fg_tid = s.fg.and_then(|fg| s.windows.get(&fg)).map_or(0, |w| w.tid);
                s.attached.contains(&(OUR_THREAD, fg_tid))
            }
            Unlock::Input => s.injected,
            Unlock::Alt => s.alt,
            Unlock::Never => false,
        }
    }
}

impl Win32Api for FakeWin32 {
    fn foreground(&self) -> Option<isize> {
        self.s().fg
    }

    fn root(&self, hwnd: isize) -> isize {
        self.s()
            .windows
            .get(&hwnd)
            .and_then(|w| w.root)
            .unwrap_or(hwnd)
    }

    fn thread_process(&self, hwnd: isize) -> (u32, u32) {
        self.s()
            .windows
            .get(&hwnd)
            .map_or((0, 0), |w| (w.tid, w.pid))
    }

    fn class_name(&self, hwnd: isize) -> String {
        self.s()
            .windows
            .get(&hwnd)
            .map_or("", |w| w.cls)
            .to_string()
    }

    fn frame_rect(&self, hwnd: isize) -> Option<Rect> {
        self.s().windows.get(&hwnd)?.rect
    }

    fn client_rect(&self, hwnd: isize) -> Option<Rect> {
        let s = self.s();
        let w = s.windows.get(&hwnd)?;
        w.client.or(w.rect)
    }

    fn is_window(&self, hwnd: isize) -> bool {
        self.s().windows.contains_key(&hwnd)
    }

    fn is_visible(&self, hwnd: isize) -> bool {
        self.window(hwnd).visible
    }

    fn is_iconic(&self, hwnd: isize) -> bool {
        self.window(hwnd).iconic
    }

    fn is_hung(&self, hwnd: isize) -> bool {
        self.window(hwnd).hung
    }

    fn is_cloaked(&self, hwnd: isize) -> bool {
        self.window(hwnd).cloaked
    }

    fn ex_style(&self, hwnd: isize) -> u32 {
        self.window(hwnd).ex_style
    }

    fn window_from_point(&self, _x: i32, _y: i32) -> Option<isize> {
        self.s().hit
    }

    fn next_window(&self, hwnd: isize) -> Option<isize> {
        let s = self.s();
        let index = s.zorder.iter().position(|h| *h == hwnd)? + 1;
        s.zorder.get(index).copied()
    }

    fn set_foreground(&self, hwnd: isize) -> bool {
        self.record(Call::SetForeground(hwnd));
        if self.allowed() {
            self.s().fg = Some(hwnd);
            return true;
        }
        false
    }

    fn bring_to_top(&self, hwnd: isize) -> bool {
        self.record(Call::BringToTop(hwnd));
        true
    }

    fn allow_set_foreground_any(&self) {
        self.record(Call::AllowAny);
    }

    fn current_thread_id(&self) -> u32 {
        OUR_THREAD
    }

    fn attach_thread_input(&self, thread: u32, to_thread: u32, attach: bool) -> bool {
        self.record(Call::Attach(thread, to_thread, attach));
        let mut s = self.s();
        if attach {
            s.attached.insert((thread, to_thread));
        } else {
            s.attached.remove(&(thread, to_thread));
        }
        true
    }

    fn post_broadcast(&self, msg: u32, wparam: usize, lparam: isize) -> bool {
        self.record(Call::PostBroadcast(msg, wparam, lparam));
        true
    }

    fn create_hidden_window(&self) -> Option<isize> {
        self.record(Call::CreateWindow);
        self.s().can_create_window.then_some(0x5150)
    }

    fn send_message(&self, hwnd: isize, msg: u32, wparam: usize, lparam: isize) -> isize {
        assert!(!self.s().send_fails, "send failed");
        self.record(Call::SendMessage(hwnd, msg, wparam, lparam));
        0
    }

    fn destroy_window(&self, hwnd: isize) -> bool {
        self.record(Call::DestroyWindow(hwnd));
        true
    }

    fn send_empty_mouse_input(&self) -> bool {
        self.record(Call::EmptyInput);
        self.s().injected = true;
        true
    }

    fn tap_alt(&self) -> bool {
        self.record(Call::TapAlt);
        self.s().alt = true;
        true
    }

    fn nudge_mouse(&self) -> bool {
        self.record(Call::Nudge);
        let mut s = self.s();
        s.cursor = (s.cursor.0 + 1, s.cursor.1);
        true
    }

    fn idle_ms(&self) -> Option<u64> {
        self.s().idle
    }

    fn set_cursor_pos(&self, x: i32, y: i32) -> bool {
        self.record(Call::SetCursor(x, y));
        let mut s = self.s();
        if !s.cursor_ok {
            return false;
        }
        let (dx, dy) = if s.cursor_offsets.is_empty() {
            (0, 0)
        } else {
            s.cursor_offsets.remove(0)
        };
        s.cursor = (x + dx, y + dy);
        true
    }

    fn cursor_pos(&self) -> Option<(i32, i32)> {
        Some(self.s().cursor)
    }

    fn lock_workstation(&self) -> bool {
        self.record(Call::Lock);
        true
    }

    fn set_display_required(&self) {
        self.record(Call::DisplayRequired);
    }

    fn input_desktop_name(&self) -> Option<String> {
        self.s().desktop.clone()
    }

    fn enable_per_monitor_dpi(&self) -> &'static str {
        self.record(Call::Dpi);
        "per-monitor-v2"
    }

    fn set_app_user_model_id(&self, app_id: &str) -> bool {
        self.record(Call::Aumid(app_id.into()));
        true
    }

    fn process_image_path(&self) -> Option<String> {
        self.s().image_path.clone()
    }

    fn current_package_family_name(&self) -> Option<String> {
        assert!(!self.s().package_lookup_fails, "no appmodel");
        self.s().own_package.clone()
    }

    fn process_package_family_name(&self, pid: u32) -> Option<String> {
        self.record(Call::PackageOf(pid));
        self.s().packages.get(&pid).cloned()
    }
}

/// `(hive, path)` of a registry key.
type Key = (&'static str, String);

/// In-memory registry: `{(hive, path): {value_name: data}}`.
#[derive(Default)]
struct FakeRegistry {
    keys: Vec<(Key, HashMap<String, RegValue>)>,
}

impl FakeRegistry {
    fn new(keys: Vec<(Key, Values)>) -> Self {
        Self {
            keys: keys.into_iter().map(|(k, v)| (k, v.0)).collect(),
        }
    }
}

impl Registry for FakeRegistry {
    fn subkeys(&self, hive: &str, path: &str) -> io::Result<Vec<String>> {
        let prefix = format!("{path}\\");
        let mut children: Vec<String> = Vec::new();
        for ((key_hive, key_path), _) in &self.keys {
            if *key_hive == hive
                && let Some(rest) = key_path.strip_prefix(&prefix)
            {
                let child = rest.split('\\').next().unwrap_or("").to_string();
                if !children.contains(&child) {
                    children.push(child);
                }
            }
        }
        Ok(children)
    }

    fn values(&self, hive: &str, path: &str) -> io::Result<HashMap<String, RegValue>> {
        Ok(self
            .keys
            .iter()
            .find(|((h, p), _)| *h == hive && p == path)
            .map(|(_, v)| v.clone())
            .unwrap_or_default())
    }
}

struct RaisingRegistry;

impl Registry for RaisingRegistry {
    fn subkeys(&self, _hive: &str, _path: &str) -> io::Result<Vec<String>> {
        Err(io::Error::other("registry unavailable"))
    }

    fn values(&self, _hive: &str, _path: &str) -> io::Result<HashMap<String, RegValue>> {
        Err(io::Error::other("registry unavailable"))
    }
}

/// A fake running process: what `processes` lists plus its creation time.
#[derive(Debug, Clone)]
struct Proc {
    pid: u32,
    name: &'static str,
    /// FILETIME ticks; `None` = unreadable.
    created: Option<u64>,
}

/// A running process, by default started ten minutes before the camera session.
fn proc(pid: u32, name: &'static str) -> Proc {
    proc_created(pid, name, Some(SESSION_START - 600 * 10_000_000))
}

fn proc_created(pid: u32, name: &'static str, created: Option<u64>) -> Proc {
    Proc { pid, name, created }
}

/// Stand-in for the process, file-system and shell queries.
#[derive(Default)]
struct FakeHost {
    executable: Option<String>,
    /// path -> resolved path; paths not listed resolve to themselves, `None` = error.
    real_paths: HashMap<String, Option<String>>,
    processes: Option<Vec<Proc>>,
    exes: HashMap<u32, String>,
    boot: Option<u64>,
    app_names: Mutex<HashMap<u32, String>>,
    created_reads: Mutex<Vec<u32>>,
    processes_listed: AtomicUsize,
    opened: Mutex<Vec<String>>,
}

impl Host for FakeHost {
    fn executable(&self) -> Option<String> {
        self.executable.clone()
    }

    fn real_path(&self, path: &str) -> Option<String> {
        match self.real_paths.get(path) {
            Some(real) => real.clone(),
            None => Some(path.to_string()),
        }
    }

    fn processes(&self) -> Option<Vec<Process>> {
        self.processes_listed.fetch_add(1, Ordering::SeqCst);
        let listed = self.processes.as_ref()?;
        Some(
            listed
                .iter()
                .map(|p| Process {
                    pid: p.pid,
                    name: p.name.into(),
                })
                .collect(),
        )
    }

    fn process_created(&self, pid: u32) -> Option<u64> {
        self.created_reads.lock().push(pid);
        self.processes
            .iter()
            .flatten()
            .find(|p| p.pid == pid)
            .and_then(|p| p.created)
    }

    fn process_exe(&self, pid: u32) -> Option<String> {
        self.exes.get(&pid).cloned()
    }

    fn boot_filetime(&self) -> Option<u64> {
        self.boot
    }

    fn app_process_name(&self, pid: u32) -> Option<String> {
        self.app_names.lock().get(&pid).cloned()
    }

    fn open_uri(&self, uri: &str) -> bool {
        self.opened.lock().push(uri.into());
        true
    }
}

fn platform_with(
    api: &Arc<FakeWin32>,
    registry: impl Registry + 'static,
    host: &Arc<FakeHost>,
) -> WindowsPlatform {
    let mut plat =
        WindowsPlatform::with_bindings(Some(api.clone()), Box::new(registry), host.clone());
    plat.pid = OWN_PID;
    plat
}

fn make_platform(api: &Arc<FakeWin32>) -> WindowsPlatform {
    platform_with(api, FakeRegistry::default(), &Arc::new(FakeHost::default()))
}

/// A platform whose binding cannot be created (the Python `ExplodingApi`).
fn exploding_platform() -> WindowsPlatform {
    let mut plat = WindowsPlatform::with_bindings(
        None,
        Box::new(FakeRegistry::default()),
        Arc::new(FakeHost::default()),
    );
    plat.api_factory = || Err("boom".into());
    plat.pid = OWN_PID;
    plat
}

fn api(windows: Vec<FakeWindow>) -> Arc<FakeWin32> {
    Arc::new(FakeWin32::new(windows))
}

fn hwnd_ref(handle: isize) -> WindowRef {
    WindowRef {
        handle: WindowHandle::Hwnd(handle),
        pid: None,
        rect: None,
    }
}

fn hwnd_pid_ref(handle: isize, pid: u32) -> WindowRef {
    WindowRef {
        pid: Some(pid),
        ..hwnd_ref(handle)
    }
}

/// A handle that is not an HWND (Python's `WindowRef(handle=None)` / `"abc"`).
fn foreign_ref() -> WindowRef {
    WindowRef {
        handle: WindowHandle::X11(5),
        pid: None,
        rect: None,
    }
}

/// Registry values of one key.
struct Values(HashMap<String, RegValue>);

fn values(pairs: &[(&str, RegValue)]) -> Values {
    Values(
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect(),
    )
}

fn webcam() -> String {
    WEBCAM_KEY.to_string()
}

fn non_packaged() -> String {
    format!("{WEBCAM_KEY}\\NonPackaged")
}

/// FILETIME of the open camera session.
const SESSION_START: u64 = 133_000_000_000_000_000;
/// Booted an hour earlier.
const BOOT: u64 = SESSION_START - 3600 * FILETIME_TICKS_PER_S;
const ZOOM: &str = r"C:\Program Files\Zoom\Zoom.exe";
const ZOOM_KEY: &str = "C:#Program Files#Zoom#Zoom.exe";
const STORE_PYTHON: &str = "PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0";

fn in_use() -> Values {
    values(&[
        ("LastUsedTimeStart", RegValue::Int(SESSION_START)),
        ("LastUsedTimeStop", RegValue::Int(0)),
    ])
}

fn stopped() -> Values {
    values(&[
        ("LastUsedTimeStart", RegValue::Int(SESSION_START)),
        ("LastUsedTimeStop", RegValue::Int(133_000_000_100_000)),
    ])
}

fn consent(value: &str) -> Values {
    values(&[("Value", RegValue::Str(value.into()))])
}

// ------------------------------------------------------------ get_platform()
#[cfg(windows)]
#[test]
fn test_windows_gets_windows_platform() {
    assert_eq!(crate::platform::get_platform().name(), "windows");
}

#[test]
fn test_create_windows_does_not_touch_win32() {
    // Construction is lazy, so this works on every OS.
    assert_eq!(create("windows").name(), "windows");
}

#[test]
fn test_unknown_os_falls_back_to_base() {
    assert_eq!(create("plan9").name(), "generic");
}

// ------------------------------------------------------------ pure helpers
#[test]
fn test_elapsed_ms_handles_wrap() {
    let cases: [(u64, u64, u64); 4] = [
        (10_000, 9_000, 1_000),
        (0, 0, 0),
        // 64-bit tick past the 32-bit wrap, stamp taken just before it
        ((1 << 32) + 500, (1 << 32) - 500, 1_000),
        (5 * (1 << 32) + 42, 40, 2),
    ];
    for (now, last, expected) in cases {
        assert_eq!(elapsed_ms(now, last), expected, "{now} {last}");
    }
}

#[test]
fn test_norm_path_is_windows_style_everywhere() {
    assert_eq!(
        norm_path("C:/Program Files/App/APP.EXE"),
        r"c:\program files\app\app.exe"
    );
}

#[test]
fn test_hwnd_of() {
    let cases = [
        (WindowHandle::Hwnd(123), Some(123)),
        (WindowHandle::Hwnd(0), None),
        (WindowHandle::Hwnd(-5), None),
        (WindowHandle::X11(5), None),
        (WindowHandle::Ax { pid: 1, element: 2 }, None),
    ];
    for (handle, expected) in cases {
        let window = WindowRef {
            handle: handle.clone(),
            pid: None,
            rect: None,
        };
        assert_eq!(hwnd_of(Some(&window)), expected, "{handle:?}");
    }
}

#[test]
fn test_hwnd_of_none_ref() {
    assert_eq!(hwnd_of(None), None);
}

#[test]
fn test_camera_user_executable() {
    let user = CameraUser {
        app: "C:#Program Files#Zoom#Zoom.exe".into(),
        packaged: false,
        start: 0,
    };
    assert_eq!(user.executable(), r"C:\Program Files\Zoom\Zoom.exe");
    let packaged = CameraUser {
        app: "Microsoft.WindowsCamera_8wekyb3d8bbwe".into(),
        packaged: true,
        start: 0,
    };
    assert_eq!(
        packaged.executable(),
        "Microsoft.WindowsCamera_8wekyb3d8bbwe"
    );
}

#[test]
fn test_is_in_use() {
    let cases = [
        (in_use(), true),
        (stopped(), false),
        (
            values(&[
                ("LastUsedTimeStart", RegValue::Int(0)),
                ("LastUsedTimeStop", RegValue::Int(0)),
            ]),
            false,
        ),
        (values(&[("LastUsedTimeStart", RegValue::Int(5))]), false),
        (values(&[]), false),
        (
            values(&[
                ("LastUsedTimeStart", RegValue::Str("5".into())),
                ("LastUsedTimeStop", RegValue::Int(0)),
            ]),
            false,
        ),
    ];
    for (Values(values), expected) in cases {
        assert_eq!(session_start(&values).is_some(), expected, "{values:?}");
    }
}

#[test]
fn test_unix_to_filetime() {
    assert_eq!(unix_to_filetime(0.0), 116_444_736_000_000_000);
    assert_eq!(unix_to_filetime(1.5), 116_444_736_015_000_000);
}

#[test]
fn test_process_helpers_read_this_process() {
    // sysinfo-backed helpers (read-only, any OS).
    let host = SystemHost;
    let own_pid = std::process::id();
    let processes = host.processes().expect("process listing");
    assert!(!processes.is_empty());
    let own = processes
        .iter()
        .find(|p| p.pid == own_pid)
        .expect("this process is listed");
    assert_eq!(own.name, own.name.to_lowercase());
    let created = host.process_created(own_pid).expect("creation time");
    let boot = host.boot_filetime().expect("boot time");
    assert!(boot <= created);
    let exe = host.process_exe(own_pid).expect("executable");
    let exe = std::path::Path::new(&exe);
    assert!(exe.is_absolute());
    let file_name = exe.file_name().unwrap().to_string_lossy().to_lowercase();
    if cfg!(windows) {
        assert_eq!(file_name, own.name); // Windows names a process after its image
    } else {
        // The kernel's process name, which may be truncated (15 characters on Linux).
        assert!(file_name.starts_with(&own.name), "{file_name} {}", own.name);
    }
    assert_eq!(host.process_exe(u32::MAX), None);
    assert_eq!(host.process_created(u32::MAX), None);
}

// ------------------------------------------------------------- lifecycle
#[test]
fn test_capabilities_cover_base_keys() {
    let caps = make_platform(&api(vec![])).capabilities();
    let keys: HashSet<_> = caps.keys().collect();
    let base = GenericPlatform.capabilities();
    assert_eq!(keys, base.keys().collect());
    for key in [
        "lock",
        "display_off",
        "focus",
        "camera_in_use",
        "hotkeys",
        "input_idle",
    ] {
        assert!(caps[key], "{key}");
    }
    assert!(!caps["key_idle"]);
}

#[test]
fn test_prepare_process_sets_env_and_calls_api() {
    // The QT_ENABLE_HIGHDPI_SCALING half is not ported (see the header).
    let api = api(vec![]);
    make_platform(&api).prepare_process();
    assert!(api.calls().contains(&Call::Dpi));
    assert!(api.calls().contains(&Call::Aumid(crate::APP_ID.into())));
}

#[test]
fn test_prepare_process_survives_api_failure() {
    exploding_platform().prepare_process();
}

static LOADS: AtomicUsize = AtomicUsize::new(0);

#[test]
fn test_missing_win32_api_degrades() {
    // What happens on macOS/Linux (or a stripped-down Windows): the binding cannot load.
    #[cfg(not(windows))]
    {
        let err = load_win32().err().expect("no Win32 off Windows");
        assert!(err.contains("only available on Windows"), "{err}");
    }
    let mut plat = WindowsPlatform::with_bindings(
        None,
        Box::new(FakeRegistry::default()),
        Arc::new(FakeHost::default()),
    );
    plat.api_factory = || {
        LOADS.fetch_add(1, Ordering::SeqCst);
        Err("the Win32 API is only available on Windows".into())
    };
    assert_eq!(plat.seconds_since_input(), None);
    assert_eq!(plat.foreground_window().map(|w| w.handle), None);
    assert!(!plat.lock_screen());
    assert_eq!(plat.is_session_locked(), None);
    assert!(!plat.activate_window(&hwnd_ref(1)));
    assert_eq!(LOADS.load(Ordering::SeqCst), 1); // the failure is remembered, not retried per call
}

// ---------------------------------------------------------- session/power
#[test]
fn test_lock_display_off_and_wake_use_the_binding() {
    let api = api(vec![]);
    let plat = make_platform(&api);
    assert!(plat.lock_screen());
    assert!(plat.display_off());
    api.s().cursor = (300, 400);
    assert!(plat.wake_display());
    let names = api.names();
    assert_eq!(
        names[names.len() - 3..],
        ["display_required", "nudge", "set_cursor"]
    );
    assert_eq!(api.s().cursor, (300, 400)); // restored exactly
}

#[test]
fn test_display_off_is_sent_to_a_window_of_our_own() {
    // Nothing is queued in other apps' windows (a paused app would run it much later).
    let api = api(vec![]);
    assert!(make_platform(&api).display_off());
    assert_eq!(
        api.calls(),
        [
            Call::CreateWindow,
            Call::SendMessage(0x5150, WM_SYSCOMMAND, SC_MONITORPOWER, MONITOR_POWER_OFF),
            Call::DestroyWindow(0x5150),
        ]
    );
    assert!(!api.names().contains(&"post_broadcast"));
}

#[test]
fn test_display_off_falls_back_to_broadcast_without_a_window() {
    let api = api(vec![]);
    api.s().can_create_window = false;
    assert!(make_platform(&api).display_off());
    assert_eq!(
        api.calls().last(),
        Some(&Call::PostBroadcast(
            WM_SYSCOMMAND,
            SC_MONITORPOWER,
            MONITOR_POWER_OFF
        ))
    );
    assert!(!api.names().contains(&"send_message"));
}

#[test]
fn test_display_off_window_is_destroyed_when_sending_fails() {
    let api = api(vec![]);
    api.s().send_fails = true;
    assert!(!make_platform(&api).display_off());
    assert_eq!(api.calls().last(), Some(&Call::DestroyWindow(0x5150)));
}

#[test]
fn test_failures_are_reported_not_raised() {
    let plat = exploding_platform();
    assert!(!plat.lock_screen());
    assert!(!plat.display_off());
    assert!(!plat.wake_display());
    assert_eq!(plat.seconds_since_input(), None);
    assert_eq!(plat.is_session_locked(), None);
    assert_eq!(plat.move_cursor(1, 2), None);
    assert!(plat.foreground_window().is_none());
    assert!(plat.window_at(1, 2).is_none());
    assert!(!plat.activate_window(&hwnd_ref(5)));
    assert!(!plat.is_window_valid(&hwnd_ref(5)));
    assert_eq!(plat.window_rect(&hwnd_ref(5)), None);
}

#[test]
fn test_seconds_since_input() {
    let api = api(vec![]);
    let plat = make_platform(&api);
    api.s().idle = Some(2500);
    assert!((plat.seconds_since_input().unwrap() - 2.5).abs() < 1e-9);
    api.s().idle = None;
    assert_eq!(plat.seconds_since_input(), None);
    assert_eq!(plat.seconds_since_key_input(), None);
}

#[test]
fn test_is_session_locked() {
    let cases = [
        (Some("Default"), Some(false)),
        (Some("default"), Some(false)),
        (Some("Winlogon"), Some(true)),
        (None, Some(true)),
        (Some(""), None),
    ];
    for (desktop, expected) in cases {
        let api = api(vec![]);
        api.s().desktop = desktop.map(String::from);
        assert_eq!(
            make_platform(&api).is_session_locked(),
            expected,
            "{desktop:?}"
        );
    }
}

#[test]
fn test_move_cursor_retries_once_when_first_move_misses() {
    let api = api(vec![]);
    api.s().cursor_offsets = vec![(7, 0)];
    assert_eq!(make_platform(&api).move_cursor(100, 200), Some(true));
    let moves: Vec<Call> = api
        .calls()
        .into_iter()
        .filter(|c| c.name() == "set_cursor")
        .collect();
    assert_eq!(
        moves,
        [Call::SetCursor(100, 200), Call::SetCursor(100, 200)]
    );
    assert_eq!(api.s().cursor, (100, 200));
}

#[test]
fn test_move_cursor_single_call_when_on_target() {
    let api = api(vec![]);
    assert_eq!(make_platform(&api).move_cursor(5, 6), Some(true));
    assert_eq!(
        api.names().iter().filter(|n| **n == "set_cursor").count(),
        1
    );
}

#[test]
fn test_move_cursor_failure() {
    let api = api(vec![]);
    api.s().cursor_ok = false;
    assert_eq!(make_platform(&api).move_cursor(5, 6), Some(false));
}

// ------------------------------------------------------------------ windows
#[test]
fn test_foreground_resolves_root_window() {
    let top = FakeWindow {
        pid: 77,
        rect: Some(Rect::new(1920, 0, 1920, 1040)),
        ..win(1)
    };
    let child = FakeWindow {
        pid: 77,
        cls: "Edit",
        root: Some(1),
        ..win(2)
    };
    let api = Arc::new(FakeWin32::with(
        vec![top, child],
        Some(2),
        Unlock::Any,
        None,
    ));
    let window = make_platform(&api).foreground_window().unwrap();
    assert_eq!(window.handle, WindowHandle::Hwnd(1));
    assert_eq!(window.pid, Some(77));
    assert_eq!(window.rect, Some(Rect::new(1920, 0, 1920, 1040)));
}

#[test]
fn test_foreground_skips_shell_surfaces() {
    for cls in ["Progman", "WorkerW", "Shell_TrayWnd", "#32768"] {
        let window = FakeWindow { cls, ..win(1) };
        let api = Arc::new(FakeWin32::with(vec![window], Some(1), Unlock::Any, None));
        assert!(make_platform(&api).foreground_window().is_none(), "{cls}");
    }
}

#[test]
fn test_foreground_skips_own_windows() {
    let window = FakeWindow {
        pid: OWN_PID,
        ..win(1)
    };
    let api = Arc::new(FakeWin32::with(vec![window], Some(1), Unlock::Any, None));
    assert!(make_platform(&api).foreground_window().is_none());
}

#[test]
fn test_no_foreground() {
    assert!(make_platform(&api(vec![])).foreground_window().is_none());
}

#[test]
fn test_window_at_returns_root() {
    let top = FakeWindow { pid: 5, ..win(1) };
    let child = FakeWindow {
        pid: 5,
        cls: "Chrome_RenderWidgetHostHWND",
        root: Some(1),
        ..win(2)
    };
    let api = Arc::new(FakeWin32::with(
        vec![top, child],
        None,
        Unlock::Any,
        Some(2),
    ));
    let window = make_platform(&api).window_at(10, 10).unwrap();
    assert_eq!(window.handle, WindowHandle::Hwnd(1));
}

#[test]
fn test_window_at_nothing_or_desktop() {
    assert!(make_platform(&api(vec![])).window_at(1, 1).is_none());
    let desktop = FakeWindow {
        cls: "Progman",
        ..win(1)
    };
    let api = Arc::new(FakeWin32::with(vec![desktop], None, Unlock::Any, Some(1)));
    assert!(make_platform(&api).window_at(1, 1).is_none());
}

#[test]
fn test_window_at_looks_below_our_own_window() {
    let stack = vec![
        // our overlay/dialog
        FakeWindow {
            pid: OWN_PID,
            rect: Some(Rect::new(0, 0, 500, 500)),
            ..win(1)
        },
        FakeWindow {
            visible: false,
            ..win(2)
        },
        FakeWindow {
            ex_style: WS_EX_TRANSPARENT,
            ..win(3)
        },
        FakeWindow {
            cloaked: true,
            ..win(4)
        },
        FakeWindow {
            iconic: true,
            ..win(5)
        },
        // does not contain the point
        FakeWindow {
            rect: Some(Rect::new(900, 900, 50, 50)),
            ..win(6)
        },
        FakeWindow {
            ex_style: WS_EX_NOACTIVATE,
            ..win(7)
        },
        // the user's window
        FakeWindow {
            pid: 9,
            rect: Some(Rect::new(0, 0, 400, 400)),
            ..win(8)
        },
        FakeWindow {
            cls: "Progman",
            rect: Some(Rect::new(0, 0, 4000, 4000)),
            ..win(9)
        },
    ];
    let api = Arc::new(FakeWin32::with(stack, None, Unlock::Any, Some(1)));
    let window = make_platform(&api).window_at(100, 100).unwrap();
    assert_eq!(window.handle, WindowHandle::Hwnd(8));
    assert_eq!(window.pid, Some(9));
}

#[test]
fn test_window_at_below_transient_reaches_desktop() {
    let stack = vec![
        FakeWindow {
            cls: "tooltips_class32",
            rect: Some(Rect::new(0, 0, 50, 50)),
            ..win(1)
        },
        FakeWindow {
            cls: "Progman",
            rect: Some(Rect::new(0, 0, 4000, 4000)),
            ..win(2)
        },
        FakeWindow {
            rect: Some(Rect::new(0, 0, 400, 400)),
            ..win(3)
        },
    ];
    let api = Arc::new(FakeWin32::with(stack, None, Unlock::Any, Some(1)));
    assert!(make_platform(&api).window_at(10, 10).is_none());
}

#[test]
fn test_is_window_valid() {
    let api = api(vec![
        FakeWindow { pid: 3, ..win(1) },
        FakeWindow {
            iconic: true,
            ..win(2)
        },
        FakeWindow {
            visible: false,
            ..win(3)
        },
        FakeWindow {
            cloaked: true,
            ..win(4)
        },
    ]);
    let plat = make_platform(&api);
    assert!(plat.is_window_valid(&hwnd_pid_ref(1, 3)));
    assert!(plat.is_window_valid(&hwnd_ref(1))); // pid unknown
    assert!(!plat.is_window_valid(&hwnd_pid_ref(1, 99))); // recycled HWND
    assert!(!plat.is_window_valid(&hwnd_ref(2)));
    assert!(!plat.is_window_valid(&hwnd_ref(3)));
    assert!(!plat.is_window_valid(&hwnd_ref(4)));
    assert!(!plat.is_window_valid(&hwnd_ref(5))); // destroyed
    assert!(!plat.is_window_valid(&foreign_ref()));
}

#[test]
fn test_window_rect() {
    let api = api(vec![FakeWindow {
        rect: Some(Rect::new(-1920, 0, 1920, 1080)),
        ..win(1)
    }]);
    let plat = make_platform(&api);
    assert_eq!(
        plat.window_rect(&hwnd_ref(1)),
        Some(Rect::new(-1920, 0, 1920, 1080))
    );
    assert_eq!(plat.window_rect(&hwnd_ref(2)), None);
}

#[test]
fn test_window_client_rect() {
    let window = FakeWindow {
        rect: Some(Rect::new(0, 0, 1200, 800)),
        client: Some(Rect::new(8, 39, 1184, 753)),
        ..win(1)
    };
    let plat = make_platform(&api(vec![window]));
    assert_eq!(
        plat.window_client_rect(&hwnd_ref(1)),
        Some(Rect::new(8, 39, 1184, 753))
    );
    assert_eq!(plat.window_client_rect(&hwnd_ref(2)), None);
    assert_eq!(plat.window_client_rect(&foreign_ref()), None);
}

#[test]
fn test_window_app_names_process_and_class() {
    let api = api(vec![
        FakeWindow {
            pid: 100,
            cls: "CASCADIA_HOSTING_WINDOW_CLASS",
            ..win(1)
        },
        FakeWindow {
            pid: 200,
            cls: "Chrome_WidgetWin_1",
            ..win(2)
        },
    ]);
    let host = Arc::new(FakeHost::default());
    *host.app_names.lock() = HashMap::from([
        (100, "windowsterminal".to_string()),
        (200, "code".to_string()),
    ]);
    let plat = platform_with(&api, FakeRegistry::default(), &host);
    let identity = |process: &str, app_id: &str| AppIdentity {
        process: process.into(),
        app_id: app_id.into(),
    };
    assert_eq!(
        plat.window_app(&hwnd_pid_ref(1, 100)),
        Some(identity("windowsterminal", "CASCADIA_HOSTING_WINDOW_CLASS"))
    );
    assert_eq!(
        plat.window_app(&hwnd_ref(2)),
        Some(identity("code", "Chrome_WidgetWin_1"))
    );
    assert_eq!(plat.window_app(&hwnd_pid_ref(1, 999)), None); // recycled HWND
    assert_eq!(plat.window_app(&hwnd_ref(3)), None); // gone
    host.app_names.lock().clear();
    assert_eq!(plat.window_app(&hwnd_ref(1)), None); // process not readable
}

#[test]
fn test_panes_capability() {
    assert!(make_platform(&api(vec![])).capabilities()["panes"]);
}

#[test]
fn test_same_window_compares_handles() {
    let plat = make_platform(&api(vec![]));
    assert!(plat.same_window(Some(&hwnd_ref(5)), Some(&hwnd_pid_ref(5, 1))));
    assert!(!plat.same_window(Some(&hwnd_ref(5)), Some(&hwnd_ref(6))));
}

// ------------------------------------------------------------- activation
fn activation_setup(unlock: Unlock, hung: bool, fg_tid: u32, target_hung: bool) -> Arc<FakeWin32> {
    let current = FakeWindow {
        pid: 50,
        tid: fg_tid,
        hung,
        ..win(1)
    };
    let target = FakeWindow {
        pid: 60,
        tid: 30,
        hung: target_hung,
        ..win(2)
    };
    Arc::new(FakeWin32::with(
        vec![current, target],
        Some(1),
        unlock,
        None,
    ))
}

fn setup(unlock: Unlock) -> Arc<FakeWin32> {
    activation_setup(unlock, false, 20, false)
}

fn attach_calls(api: &FakeWin32) -> Vec<Call> {
    api.calls()
        .into_iter()
        .filter(|c| c.name() == "attach")
        .collect()
}

fn index_of(names: &[&str], name: &str) -> usize {
    names.iter().position(|n| *n == name).unwrap()
}

#[test]
fn test_already_foreground() {
    let api = Arc::new(FakeWin32::with(vec![win(1)], Some(1), Unlock::Any, None));
    assert!(make_platform(&api).activate_window(&hwnd_ref(1)));
    assert!(!api.names().contains(&"set_foreground"));
}

#[test]
fn test_minimised_or_invalid_windows_are_not_activated() {
    let minimised = FakeWindow {
        iconic: true,
        ..win(1)
    };
    let api = Arc::new(FakeWin32::with(
        vec![minimised, win(2)],
        Some(2),
        Unlock::Any,
        None,
    ));
    let plat = make_platform(&api);
    assert!(!plat.activate_window(&hwnd_ref(1)));
    assert!(!plat.activate_window(&hwnd_ref(99)));
    assert!(!plat.activate_window(&foreign_ref()));
    assert!(!api.names().contains(&"set_foreground"));
}

#[test]
fn test_attach_thread_input_first_and_no_synthetic_input() {
    let api = setup(Unlock::Attach);
    assert!(make_platform(&api).activate_window(&hwnd_ref(2)));
    assert_eq!(api.s().fg, Some(2));
    assert_eq!(
        attach_calls(&api),
        [
            Call::Attach(OUR_THREAD, 20, true),
            Call::Attach(OUR_THREAD, 20, false),
        ]
    );
    assert!(!api.names().contains(&"empty_input"));
    assert!(!api.names().contains(&"tap_alt"));
    assert!(api.s().attached.is_empty());
}

#[test]
fn test_falls_back_to_empty_input() {
    let api = setup(Unlock::Input);
    assert!(make_platform(&api).activate_window(&hwnd_ref(2)));
    assert!(api.names().contains(&"empty_input"));
    assert!(!api.names().contains(&"tap_alt"));
    assert!(api.s().attached.is_empty()); // always detached again
}

#[test]
fn test_falls_back_to_alt_tap() {
    let api = setup(Unlock::Alt);
    assert!(make_platform(&api).activate_window(&hwnd_ref(2)));
    let names = api.names();
    assert!(index_of(&names, "empty_input") < index_of(&names, "tap_alt"));
}

#[test]
fn test_gives_up_when_locked_out() {
    let api = setup(Unlock::Never);
    assert!(!make_platform(&api).activate_window(&hwnd_ref(2)));
    assert_eq!(api.s().fg, Some(1));
    assert!(api.s().attached.is_empty());
    assert!(!api.names().contains(&"bring_to_top")); // never raised without focus
}

#[test]
fn test_raises_window_after_granted_switch() {
    let api = setup(Unlock::Any);
    assert!(make_platform(&api).activate_window(&hwnd_ref(2)));
    let names = api.names();
    assert!(index_of(&names, "set_foreground") < index_of(&names, "bring_to_top"));
}

#[test]
fn test_hung_foreground_is_never_attached() {
    let api = activation_setup(Unlock::Input, true, 20, false);
    assert!(make_platform(&api).activate_window(&hwnd_ref(2)));
    assert_eq!(attach_calls(&api), []);
}

#[test]
fn test_own_thread_in_foreground_needs_no_attach() {
    let api = activation_setup(Unlock::Any, false, OUR_THREAD, false);
    assert!(make_platform(&api).activate_window(&hwnd_ref(2)));
    assert_eq!(attach_calls(&api), []);
}

#[test]
fn test_hung_target_is_left_alone() {
    // Raising a window whose thread does not pump would block our UI thread.
    let api = activation_setup(Unlock::Any, false, 20, true);
    assert!(!make_platform(&api).activate_window(&hwnd_ref(2)));
    assert_eq!(api.s().fg, Some(1));
    for name in [
        "set_foreground",
        "bring_to_top",
        "attach",
        "empty_input",
        "tap_alt",
    ] {
        assert!(!api.names().contains(&name), "{name}");
    }
}

// ------------------------------------------------------------------ camera
#[test]
fn test_active_camera_users() {
    let registry = FakeRegistry::new(vec![
        (("HKCU", webcam()), consent("Allow")),
        (
            (
                "HKCU",
                format!("{WEBCAM_KEY}\\Microsoft.WindowsCamera_8wekyb3d8bbwe"),
            ),
            in_use(),
        ),
        (("HKCU", format!("{WEBCAM_KEY}\\Other.App_123")), stopped()),
        (("HKCU", non_packaged()), consent("Allow")),
        (
            ("HKCU", format!("{}\\C:#Zoom#Zoom.exe", non_packaged())),
            in_use(),
        ),
        (
            ("HKCU", format!("{}\\C:#Old#old.exe", non_packaged())),
            stopped(),
        ),
        // HKLM is ignored
        (
            ("HKLM", format!("{}\\C:#Svc#svc.exe", non_packaged())),
            in_use(),
        ),
    ]);
    let mut users = active_camera_users(&registry).unwrap();
    users.sort_by(|a, b| a.app.cmp(&b.app));
    assert_eq!(
        users,
        [
            CameraUser {
                app: "C:#Zoom#Zoom.exe".into(),
                packaged: false,
                start: SESSION_START,
            },
            CameraUser {
                app: "Microsoft.WindowsCamera_8wekyb3d8bbwe".into(),
                packaged: true,
                start: SESSION_START,
            },
        ]
    );
}

/// Inputs of the camera tests' `_platform` helper.
struct CameraSetup {
    /// Non-packaged consent-store entries.
    entries: Vec<(&'static str, Values)>,
    processes: Option<Vec<Proc>>,
    /// pid -> executable path.
    exes: Vec<(u32, &'static str)>,
    packaged: Vec<(&'static str, Values)>,
    api: Option<Arc<FakeWin32>>,
    boot: Option<u64>,
}

impl Default for CameraSetup {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            processes: None,
            exes: Vec::new(),
            packaged: Vec::new(),
            api: None,
            boot: Some(BOOT),
        }
    }
}

/// Consent-store entries, running processes and boot time; the host records whose creation
/// times were read.
fn camera_platform(setup: CameraSetup) -> (WindowsPlatform, Arc<FakeHost>) {
    let mut keys: Vec<(Key, Values)> = setup
        .entries
        .into_iter()
        .map(|(app, v)| (("HKCU", format!("{}\\{app}", non_packaged())), v))
        .collect();
    keys.extend(
        setup
            .packaged
            .into_iter()
            .map(|(app, v)| (("HKCU", format!("{WEBCAM_KEY}\\{app}")), v)),
    );
    let host = Arc::new(FakeHost {
        processes: setup.processes,
        exes: setup
            .exes
            .into_iter()
            .map(|(pid, exe)| (pid, exe.to_string()))
            .collect(),
        boot: setup.boot,
        ..FakeHost::default()
    });
    let api = setup.api.unwrap_or_else(|| api(vec![]));
    let plat = platform_with(&api, FakeRegistry::new(keys), &host);
    plat.own_paths
        .set(HashSet::from([norm_path(r"C:\Python\python.exe")]))
        .unwrap();
    (plat, host)
}

fn zoom_in_use() -> Vec<(&'static str, Values)> {
    vec![(ZOOM_KEY, in_use())]
}

#[test]
fn test_nobody_using_the_camera() {
    let (plat, _) = camera_platform(CameraSetup {
        processes: Some(vec![proc(1, "zoom.exe")]),
        exes: vec![(1, ZOOM)],
        ..CameraSetup::default()
    });
    assert_eq!(plat.camera_in_use_by_other_app(), Some(false));
}

#[test]
fn test_own_process_is_ignored() {
    let (plat, _) = camera_platform(CameraSetup {
        entries: vec![("C:#Python#python.exe", in_use())],
        processes: Some(vec![proc(1, "python.exe")]),
        ..CameraSetup::default()
    });
    assert_eq!(plat.camera_in_use_by_other_app(), Some(false));
}

#[test]
fn test_other_running_app_counts() {
    let (plat, _) = camera_platform(CameraSetup {
        entries: zoom_in_use(),
        processes: Some(vec![proc(1, "zoom.exe")]),
        exes: vec![(1, ZOOM)],
        ..CameraSetup::default()
    });
    assert_eq!(plat.camera_in_use_by_other_app(), Some(true));
}

#[test]
fn test_stale_entry_of_exited_app_is_ignored() {
    let (plat, _) = camera_platform(CameraSetup {
        entries: zoom_in_use(),
        processes: Some(vec![proc(1, "explorer.exe")]),
        ..CameraSetup::default()
    });
    assert_eq!(plat.camera_in_use_by_other_app(), Some(false));
}

#[test]
fn test_same_name_at_another_path_does_not_count() {
    let (plat, _) = camera_platform(CameraSetup {
        entries: zoom_in_use(),
        processes: Some(vec![proc(1, "zoom.exe")]),
        exes: vec![(1, r"C:\Users\me\Downloads\Zoom.exe")],
        ..CameraSetup::default()
    });
    assert_eq!(plat.camera_in_use_by_other_app(), Some(false));
}

#[test]
fn test_app_started_after_the_session_does_not_count() {
    // Chrome crashed mid-call; the Chrome running now never opened the camera.
    let later = proc_created(
        1,
        "zoom.exe",
        Some(SESSION_START + 60 * FILETIME_TICKS_PER_S),
    );
    let (plat, _) = camera_platform(CameraSetup {
        entries: zoom_in_use(),
        processes: Some(vec![later]),
        exes: vec![(1, ZOOM)],
        ..CameraSetup::default()
    });
    assert_eq!(plat.camera_in_use_by_other_app(), Some(false));
}

#[test]
fn test_creation_time_rounding_is_tolerated() {
    let close = proc_created(1, "zoom.exe", Some(SESSION_START + FILETIME_TICKS_PER_S));
    let (plat, _) = camera_platform(CameraSetup {
        entries: zoom_in_use(),
        processes: Some(vec![close]),
        exes: vec![(1, ZOOM)],
        ..CameraSetup::default()
    });
    assert_eq!(plat.camera_in_use_by_other_app(), Some(true));
}

#[test]
fn test_unreadable_creation_time_or_path_is_trusted() {
    let (plat, _) = camera_platform(CameraSetup {
        entries: zoom_in_use(),
        processes: Some(vec![proc_created(1, "zoom.exe", None)]),
        ..CameraSetup::default()
    });
    assert_eq!(plat.camera_in_use_by_other_app(), Some(true)); // path unreadable (elevated app)
}

#[test]
fn test_session_from_before_the_last_boot_is_stale() {
    // Power loss mid-call leaves LastUsedTimeStop at 0 forever.
    let next_day = SESSION_START + 24 * 3600 * FILETIME_TICKS_PER_S;
    let (plat, host) = camera_platform(CameraSetup {
        entries: zoom_in_use(),
        processes: Some(vec![proc(1, "zoom.exe")]),
        exes: vec![(1, ZOOM)],
        boot: Some(next_day),
        ..CameraSetup::default()
    });
    assert_eq!(plat.camera_in_use_by_other_app(), Some(false));
    assert_eq!(host.processes_listed.load(Ordering::SeqCst), 0); // no process scan needed
}

#[test]
fn test_session_just_before_the_boot_time_goes_to_the_process_check() {
    // The boot time comes from the current clock, which NTP may have moved.
    let (plat, _) = camera_platform(CameraSetup {
        entries: zoom_in_use(),
        processes: Some(vec![proc(1, "zoom.exe")]),
        exes: vec![(1, ZOOM)],
        boot: Some(SESSION_START + 60 * FILETIME_TICKS_PER_S),
        ..CameraSetup::default()
    });
    assert_eq!(plat.camera_in_use_by_other_app(), Some(true));
}

#[test]
fn test_unknown_boot_time_skips_that_filter() {
    let (plat, _) = camera_platform(CameraSetup {
        entries: zoom_in_use(),
        processes: Some(vec![proc(1, "zoom.exe")]),
        exes: vec![(1, ZOOM)],
        boot: None,
        ..CameraSetup::default()
    });
    assert_eq!(plat.camera_in_use_by_other_app(), Some(true));
}

#[test]
fn test_unknown_process_list_trusts_the_registry() {
    for processes in [None, Some(vec![])] {
        let (plat, _) = camera_platform(CameraSetup {
            entries: zoom_in_use(),
            processes: processes.clone(),
            ..CameraSetup::default()
        });
        assert_eq!(
            plat.camera_in_use_by_other_app(),
            Some(true),
            "{processes:?}"
        );
    }
}

#[test]
fn test_creation_time_is_read_only_for_matching_processes() {
    // Reading it for every process stalled the GUI thread for about a second.
    let names: Vec<&'static str> = (10..60)
        .map(|pid| &*String::leak(format!("svchost{pid}.exe")))
        .collect();
    let mut processes: Vec<Proc> = (10..60)
        .zip(names)
        .map(|(pid, name)| proc(pid, name))
        .collect();
    processes.push(proc(7, "zoom.exe"));
    let (plat, host) = camera_platform(CameraSetup {
        entries: zoom_in_use(),
        processes: Some(processes),
        exes: vec![(7, ZOOM)],
        ..CameraSetup::default()
    });
    assert_eq!(plat.camera_in_use_by_other_app(), Some(true));
    assert_eq!(*host.created_reads.lock(), [7]);
}

fn api_with_packages(packages: &[(u32, &str)]) -> Arc<FakeWin32> {
    let api = api(vec![]);
    api.s().packages = packages
        .iter()
        .map(|(pid, p)| (*pid, p.to_string()))
        .collect();
    api
}

#[test]
fn test_creation_time_is_read_only_for_the_matching_package() {
    let api = api_with_packages(&[(7, "MSTeams_8wekyb3d8bbwe"), (8, "Some.Other_app")]);
    let (plat, host) = camera_platform(CameraSetup {
        processes: Some(vec![
            proc(5, "explorer.exe"),
            proc(7, "ms-teams.exe"),
            proc(8, "other.exe"),
        ]),
        packaged: vec![("MSTeams_8wekyb3d8bbwe", in_use())],
        api: Some(api),
        ..CameraSetup::default()
    });
    assert_eq!(plat.camera_in_use_by_other_app(), Some(true));
    assert_eq!(*host.created_reads.lock(), [7]);
}

#[test]
fn test_packaged_app_counts_while_its_package_runs() {
    let api = api_with_packages(&[(7, "Microsoft.WindowsCamera_8wekyb3d8bbwe")]);
    let (plat, _) = camera_platform(CameraSetup {
        processes: Some(vec![proc(5, "explorer.exe"), proc(7, "windowscamera.exe")]),
        packaged: vec![("microsoft.windowscamera_8wekyb3d8bbwe", in_use())],
        api: Some(api),
        ..CameraSetup::default()
    });
    assert_eq!(plat.camera_in_use_by_other_app(), Some(true));
}

#[test]
fn test_stale_packaged_entry_is_ignored() {
    // Teams lost power mid-call: its session stays open in the registry.
    let api = api_with_packages(&[(7, "Some.Other_app")]);
    let (plat, _) = camera_platform(CameraSetup {
        processes: Some(vec![proc(7, "ms-teams.exe")]),
        packaged: vec![("MSTeams_8wekyb3d8bbwe", in_use())],
        api: Some(api),
        ..CameraSetup::default()
    });
    assert_eq!(plat.camera_in_use_by_other_app(), Some(false));
}

#[test]
fn test_packaged_process_started_after_the_session_does_not_count() {
    let api = api_with_packages(&[(7, "MSTeams_8wekyb3d8bbwe")]);
    let late = proc_created(
        7,
        "ms-teams.exe",
        Some(SESSION_START + 60 * FILETIME_TICKS_PER_S),
    );
    let (plat, _) = camera_platform(CameraSetup {
        processes: Some(vec![late]),
        packaged: vec![("MSTeams_8wekyb3d8bbwe", in_use())],
        api: Some(api),
        ..CameraSetup::default()
    });
    assert_eq!(plat.camera_in_use_by_other_app(), Some(false));
}

#[test]
fn test_own_package_identity_is_ignored() {
    // Under an MSIX install our own session is recorded by package name.
    let api = api_with_packages(&[(OWN_PID, STORE_PYTHON)]);
    api.s().own_package = Some(STORE_PYTHON.into());
    let lower = STORE_PYTHON.to_lowercase();
    let (plat, _) = camera_platform(CameraSetup {
        processes: Some(vec![proc(OWN_PID, "python.exe")]),
        packaged: vec![(String::leak(lower.clone()), in_use())],
        api: Some(api),
        ..CameraSetup::default()
    });
    assert_eq!(plat.camera_in_use_by_other_app(), Some(false));
    assert_eq!(plat.own_package(), lower);
}

#[test]
fn test_other_package_is_not_mistaken_for_ours() {
    let api = api_with_packages(&[(9, "Microsoft.WindowsCamera_8wekyb3d8bbwe")]);
    api.s().own_package = Some(STORE_PYTHON.into());
    let (plat, _) = camera_platform(CameraSetup {
        processes: Some(vec![proc(9, "windowscamera.exe")]),
        packaged: vec![("Microsoft.WindowsCamera_8wekyb3d8bbwe", in_use())],
        api: Some(api),
        ..CameraSetup::default()
    });
    assert_eq!(plat.camera_in_use_by_other_app(), Some(true));
}

#[test]
fn test_without_package_identity_packaged_entries_are_others() {
    let (plat, _) = camera_platform(CameraSetup {
        processes: Some(vec![proc(9, "x.exe")]),
        ..CameraSetup::default()
    });
    assert_eq!(plat.own_package(), "");
    assert!(!plat.is_own_camera_user(&CameraUser {
        app: String::new(),
        packaged: true,
        start: 1,
    }));
}

#[test]
fn test_package_lookup_failure_means_no_package() {
    let api = api(vec![]);
    api.s().package_lookup_fails = true;
    assert_eq!(make_platform(&api).own_package(), "");
}

#[test]
fn test_registry_failure_is_unknown() {
    let plat = platform_with(
        &api(vec![]),
        RaisingRegistry,
        &Arc::new(FakeHost::default()),
    );
    plat.own_paths.set(HashSet::new()).unwrap();
    assert_eq!(plat.camera_in_use_by_other_app(), None);
}

#[test]
fn test_own_executables_include_process_image() {
    let api = api(vec![]);
    api.s().image_path = Some(r"C:\Base\Python312\python.exe".into());
    let host = Arc::new(FakeHost {
        executable: Some(r"D:\venv\Scripts\python.exe".into()),
        ..FakeHost::default()
    });
    let plat = platform_with(&api, FakeRegistry::default(), &host);
    let own = plat.own_executables();
    assert!(own.contains(&norm_path(r"D:\venv\Scripts\python.exe")));
    assert!(own.contains(&norm_path(r"C:\Base\Python312\python.exe")));
}

#[test]
fn test_redirected_interpreter_of_a_packaged_parent_is_own() {
    // Started from inside an MSIX app, the consent store records the real path.
    let virtual_path = r"C:\Users\me\AppData\Roaming\uv\python\cpython-3.12\python.exe";
    let real = concat!(
        r"C:\Users\me\AppData\Local\Packages\Host_abc\LocalCache",
        r"\Roaming\uv\python\cpython-3.12\python.exe"
    );
    let api = api(vec![]);
    api.s().image_path = Some(virtual_path.into());
    let host = Arc::new(FakeHost {
        executable: Some(virtual_path.into()),
        real_paths: HashMap::from([(virtual_path.to_string(), Some(format!("\\\\?\\{real}")))]),
        ..FakeHost::default()
    });
    let plat = platform_with(&api, FakeRegistry::default(), &host);
    let key = real.replace('\\', "#");
    assert!(plat.is_own_camera_user(&CameraUser {
        app: key,
        packaged: false,
        start: 1,
    }));
}

#[test]
fn test_unresolvable_path_is_skipped() {
    let exe = r"D:\venv\Scripts\python.exe";
    let api = api(vec![]);
    api.s().image_path = None;
    let host = Arc::new(FakeHost {
        executable: Some(exe.into()),
        real_paths: HashMap::from([(exe.to_string(), None)]),
        ..FakeHost::default()
    });
    let own = platform_with(&api, FakeRegistry::default(), &host)
        .own_executables()
        .clone();
    assert!(own.contains(&norm_path(exe)));
}

// ------------------------------------------------------------- permissions
#[test]
fn test_camera_permission() {
    let cases = [
        (vec![(("HKCU", webcam()), consent("Allow"))], Some(true)),
        (vec![(("HKCU", webcam()), consent("Deny"))], Some(false)),
        (
            vec![
                (("HKCU", webcam()), consent("Allow")),
                (("HKCU", non_packaged()), consent("Deny")),
            ],
            Some(false),
        ),
        (
            vec![
                (("HKLM", webcam()), consent("Deny")),
                (("HKCU", webcam()), consent("Allow")),
            ],
            Some(false),
        ),
        (vec![], None),
    ];
    for (index, (keys, expected)) in cases.into_iter().enumerate() {
        let plat = platform_with(
            &api(vec![]),
            FakeRegistry::new(keys),
            &Arc::new(FakeHost::default()),
        );
        assert_eq!(
            plat.permissions(),
            HashMap::from([("camera", expected), ("accessibility", None)]),
            "case {index}"
        );
    }
}

#[test]
fn test_camera_permission_unknown_on_registry_error() {
    let plat = platform_with(
        &api(vec![]),
        RaisingRegistry,
        &Arc::new(FakeHost::default()),
    );
    assert_eq!(plat.permissions()["camera"], None);
}

#[test]
fn test_open_permission_settings() {
    let host = Arc::new(FakeHost::default());
    let plat = platform_with(&api(vec![]), FakeRegistry::default(), &host);
    assert!(plat.open_permission_settings("camera"));
    assert!(!plat.open_permission_settings("accessibility"));
    assert_eq!(*host.opened.lock(), ["ms-settings:privacy-webcam"]);
}

// ------------------------------------------------- live, read-only (Windows)
/// Real Win32 calls, restricted to ones that only *read* state.
#[cfg(windows)]
mod live_read_only {
    use std::time::{Duration, Instant};

    use windows::Win32::UI::Input::KeyboardAndMouse::{INPUT, LASTINPUTINFO};

    use super::*;

    fn real_api() -> win32::Win32 {
        win32::Win32::new()
    }

    #[test]
    fn test_structure_sizes_match_the_sdk() {
        let pointer_size = size_of::<usize>();
        assert_eq!(size_of::<INPUT>(), if pointer_size == 8 { 40 } else { 28 });
        assert_eq!(size_of::<LASTINPUTINFO>(), 8);
    }

    #[test]
    fn test_input_idle_and_session() {
        let plat = WindowsPlatform::new();
        let idle = plat.seconds_since_input().expect("idle time");
        assert!(idle >= 0.0);
        // False on an interactive desktop; CI services may report True/None.
        let _ = plat.is_session_locked();
    }

    #[test]
    fn test_foreground_window_queries() {
        let plat = WindowsPlatform::new();
        let Some(window) = plat.foreground_window() else {
            return; // no user window in the foreground (headless session)
        };
        assert!(matches!(window.handle, WindowHandle::Hwnd(_)));
        let _ = plat.is_window_valid(&window);
        let rect = plat.window_rect(&window);
        assert!(rect.is_none_or(|r| r.w > 0 && r.h > 0));
        let client = plat.window_client_rect(&window);
        assert!(client.is_none_or(|r| r.w > 0 && r.h > 0));
        let app = plat.window_app(&window);
        assert!(app.is_none_or(|a| !a.process.is_empty() && a.process == a.process.to_lowercase()));
        if let Some(rect) = rect {
            let (cx, cy) = rect.center();
            let hit = plat.window_at(cx as i32, cy as i32);
            assert!(hit.is_none_or(|h| matches!(h.handle, WindowHandle::Hwnd(_))));
        }
    }

    #[test]
    fn test_camera_and_permissions() {
        let plat = WindowsPlatform::new();
        let _ = plat.camera_in_use_by_other_app();
        let perms = plat.permissions();
        assert_eq!(
            perms.keys().copied().collect::<HashSet<_>>(),
            HashSet::from(["camera", "accessibility"])
        );
        assert_eq!(perms["accessibility"], None);
    }

    #[test]
    fn test_registry_reader() {
        let reader = WinRegistry;
        assert!(!reader.subkeys("HKCU", "Software").unwrap().is_empty());
        let missing = r"Software\eye-tracker-tests\does-not-exist";
        assert!(reader.values("HKCU", missing).unwrap().is_empty());
        assert!(reader.subkeys("HKCU", missing).unwrap().is_empty());
    }

    #[test]
    fn test_process_image_path() {
        let path = real_api().process_image_path().expect("image path");
        assert!(std::path::Path::new(&path).is_file());
    }

    #[test]
    fn test_package_identity_queries() {
        let api = real_api();
        let own = api.current_package_family_name();
        assert!(own.as_ref().is_none_or(|name| name.contains('_')));
        assert_eq!(api.process_package_family_name(std::process::id()), own);
        assert_eq!(api.process_package_family_name(0), None); // the idle process cannot be opened
    }

    #[test]
    fn test_hidden_window_is_created_and_destroyed() {
        // Only our own invisible window is touched; no message is sent to it.
        let api = real_api();
        let hwnd = api.create_hidden_window().expect("hidden window");
        assert!(api.is_window(hwnd));
        assert!(!api.is_visible(hwnd));
        assert_eq!(
            api.thread_process(hwnd),
            (api.current_thread_id(), std::process::id())
        );
        assert!(api.destroy_window(hwnd));
        assert!(!api.is_window(hwnd));
    }

    #[test]
    fn test_bring_to_top_does_not_wait_for_a_busy_thread() {
        // A synchronous SetWindowPos would block until the owner pumps messages. The target is
        // a hidden window of our own, owned by a helper thread that stops pumping for a while,
        // like an app paused in a debugger.
        let (created_tx, created_rx) = crossbeam_channel::bounded(1);
        let (release_tx, release_rx) = crossbeam_channel::bounded::<()>(1);
        let owner = std::thread::spawn(move || {
            let api = real_api();
            let hwnd = api.create_hidden_window();
            created_tx.send(hwnd).unwrap();
            let _ = release_rx.recv_timeout(Duration::from_secs(5)); // not pumping meanwhile
            if let Some(hwnd) = hwnd {
                api.destroy_window(hwnd);
            }
        });
        let owned = created_rx
            .recv_timeout(Duration::from_secs(5))
            .unwrap()
            .expect("could not create the helper window");
        let started = Instant::now();
        real_api().bring_to_top(owned);
        let elapsed = started.elapsed();
        release_tx.send(()).unwrap();
        owner.join().unwrap();
        assert!(elapsed < Duration::from_millis(500));
    }
}
