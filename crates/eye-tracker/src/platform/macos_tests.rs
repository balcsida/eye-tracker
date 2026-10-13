//! Tests for the macOS platform layer.
//!
//! The frameworks are replaced by small fakes, so nothing here locks the screen, sleeps the
//! displays, moves the real cursor or starts a process: the tool lookup, the subprocess entry
//! points and the login framework are the [`FakeHost`]. The last tests only *read* the real
//! frameworks (idle time, window list, permission state, no prompts).
//!
//! Not ported:
//! * `test_real_import_failure_is_cached`: it tested pyobjc's lazy, cached `importlib` import,
//!   which has no counterpart (the frameworks are linked in).
//! * `test_err_value_and_handle_splitting` (the `_err_value` half and the `True`-as-pid case):
//!   they covered pyobjc's `(error, value)` out-parameter bridging and Python's `bool` being an
//!   `int`; the Rust traits return typed pairs and the pid is an `i32`.
//! * `test_background_activity_uses_the_named_option` and the `kAXTrustedCheckOptionPrompt`
//!   half of `test_accessibility_permission_and_prompt`: they covered the fallback constants used
//!   when a pyobjc build lacks a symbol; the Rust side links the symbols and passes the constants
//!   itself (`test_idle_uses_fallback_constants` still checks the idle-time constants).
//! * `test_camera_permission_via_objc_runtime`: it covered loading AVFoundation through the
//!   pyobjc bridge; the real camera backend (`AVCaptureDevice` by class name, loading the
//!   framework bundle first) is only exercised by `test_real_camera_authorization_is_a_status`.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};

use parking_lot::Mutex;

use super::*;
use crate::platform::base::GenericPlatform;
use crate::testing::{app_dirs, capture_logs};

const PID: i32 = 321;
const OTHER_PID: i32 = 654;
/// The pid of "our" process in the fakes.
const OWN_PID: i32 = 4242;
const AX_ERROR_ATTRIBUTE_UNSUPPORTED: i32 = -25205;

// ---------------------------------------------------------------------- fakes
#[derive(Clone)]
struct Clock(Arc<Mutex<f64>>);

impl Clock {
    fn new() -> Self {
        Self(Arc::new(Mutex::new(50.0)))
    }

    fn advance(&self, seconds: f64) {
        *self.0.lock() += seconds;
    }

    fn reader(&self) -> Box<dyn Fn() -> f64 + Send + Sync> {
        let now = self.0.clone();
        Box::new(move || *now.lock())
    }
}

/// Stands in for the tool lookup, `subprocess.run` / `Popen`, the login framework and the
/// process identity (the `recorder` fixture and the monkeypatched indirections).
#[derive(Default)]
struct FakeHost {
    tools: Mutex<HashMap<String, String>>,
    returncode: AtomicI32,
    runs: Mutex<Vec<Vec<String>>>,
    popens: Mutex<Vec<Vec<String>>>,
    /// `_IS_MACOS`: off unless a test turns it on.
    macos: AtomicBool,
    major_version: AtomicU32,
    process_names: Mutex<HashMap<i64, String>>,
    /// Status of `SACLockScreenImmediate`; `None` when the framework is not loadable.
    login_status: Mutex<Option<i32>>,
    loaded: Mutex<Vec<String>>,
    lock_calls: AtomicU32,
    own_pid: AtomicI32,
    /// The executable of a packaged build (`sys.frozen` / `sys.executable`).
    executable: Mutex<Option<PathBuf>>,
    /// Replaces the build fingerprint (`monkeypatch.setattr(macos, "_build_fingerprint", ...)`).
    fingerprint: Mutex<Option<String>>,
}

impl FakeHost {
    fn new() -> Arc<Self> {
        let host = Self::default();
        host.own_pid.store(OWN_PID, Ordering::SeqCst);
        Arc::new(host)
    }

    fn set_tools(&self, tools: &[(&str, &str)]) {
        *self.tools.lock() = tools
            .iter()
            .map(|(name, path)| (name.to_string(), path.to_string()))
            .collect();
    }

    fn runs(&self) -> Vec<Vec<String>> {
        self.runs.lock().clone()
    }

    fn popens(&self) -> Vec<Vec<String>> {
        self.popens.lock().clone()
    }

    fn on_macos(&self) {
        self.macos.store(true, Ordering::SeqCst);
    }

    fn set_major_version(&self, major: u32) {
        self.major_version.store(major, Ordering::SeqCst);
    }

    fn executables(&self) -> Option<(PathBuf, PathBuf)> {
        packaged_executables_for(self.executable.lock().as_deref()?)
    }
}

impl Host for FakeHost {
    fn is_macos(&self) -> bool {
        self.macos.load(Ordering::SeqCst)
    }

    fn tool(&self, name: &str) -> Option<String> {
        self.tools.lock().get(name).cloned()
    }

    fn run(&self, argv: &[String], timeout: f64) -> Result<i32, String> {
        assert!(timeout > 0.0, "tools must run with a timeout");
        self.runs.lock().push(argv.to_vec());
        Ok(self.returncode.load(Ordering::SeqCst))
    }

    fn spawn(&self, argv: &[String]) -> Result<(), String> {
        self.popens.lock().push(argv.to_vec());
        Ok(())
    }

    fn lock_screen_immediate(&self, framework: &str) -> Result<i32, String> {
        self.loaded.lock().push(framework.to_string());
        let status = (*self.login_status.lock())
            .ok_or_else(|| format!("{framework} is not loadable in tests"))?;
        self.lock_calls.fetch_add(1, Ordering::SeqCst);
        Ok(status)
    }

    fn macos_major_version(&self) -> u32 {
        self.major_version.load(Ordering::SeqCst)
    }

    fn own_pid(&self) -> i32 {
        self.own_pid.load(Ordering::SeqCst)
    }

    fn process_name(&self, pid: i64) -> Option<String> {
        self.process_names.lock().get(&pid).cloned()
    }

    fn build_fingerprint(&self) -> Option<String> {
        if let Some(build) = self.fingerprint.lock().clone() {
            return Some(build);
        }
        build_fingerprint_of(self.executables().as_ref())
    }

    fn runs_helper_executable(&self) -> bool {
        runs_helper_executable_of(self.executables().as_ref())
    }
}

struct FakeQuartz {
    idle: Mutex<HashMap<u32, f64>>,
    idle_calls: Mutex<Vec<(i32, u32)>>,
    session: Mutex<Option<SessionInfo>>,
    warp_error: AtomicI32,
    warps: Mutex<Vec<(f64, f64)>>,
    associations: Mutex<Vec<bool>>,
    windows: Mutex<Vec<CgWindow>>,
    main_height: Mutex<Option<f64>>,
}

impl FakeQuartz {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            idle: Mutex::new(HashMap::from([(0xFFFF_FFFF, 12.5), (10, 3.25)])),
            idle_calls: Mutex::new(Vec::new()),
            session: Mutex::new(Some(SessionInfo::default())),
            warp_error: AtomicI32::new(0),
            warps: Mutex::new(Vec::new()),
            associations: Mutex::new(Vec::new()),
            windows: Mutex::new(Vec::new()),
            main_height: Mutex::new(Some(1080.0)),
        })
    }

    fn set_windows(&self, windows: Vec<CgWindow>) {
        *self.windows.lock() = windows;
    }
}

impl Quartz for FakeQuartz {
    fn seconds_since_last_event(&self, state: i32, event_type: u32) -> Option<f64> {
        assert_eq!(state, 0);
        self.idle_calls.lock().push((state, event_type));
        self.idle.lock().get(&event_type).copied()
    }

    fn session_info(&self) -> Option<SessionInfo> {
        self.session.lock().clone()
    }

    fn warp_cursor(&self, x: f64, y: f64) -> i32 {
        self.warps.lock().push((x, y));
        self.warp_error.load(Ordering::SeqCst)
    }

    fn associate_mouse_and_cursor(&self, connected: bool) -> i32 {
        self.associations.lock().push(connected);
        0
    }

    fn main_display_height(&self) -> Option<f64> {
        *self.main_height.lock()
    }

    fn window_list(&self, options: u32, relative_to: u32) -> Option<Vec<CgWindow>> {
        assert_eq!(options, 1 | 16);
        assert_eq!(relative_to, 0);
        Some(self.windows.lock().clone())
    }
}

fn cg_window(pid: i32, x: f64, y: f64, w: f64, h: f64, layer: i64, alpha: f64) -> CgWindow {
    CgWindow {
        layer,
        alpha,
        owner_pid: i64::from(pid),
        bounds: Some((x, y, w, h)),
    }
}

/// A normal (layer 0), opaque window.
fn win(pid: i32, x: f64, y: f64, w: f64, h: f64) -> CgWindow {
    cg_window(pid, x, y, w, h, 0, 1.0)
}

/// An `AXValue` holding a CGPoint or CGSize, with the description pyobjc printed for it.
fn ax_point(x: f64, y: f64) -> AxAttr {
    AxAttr::Value(AxValue {
        unpacked: Some(AxStruct::Point(x, y)),
        description: format!("<AXValue 0x1> {{value = x:{x:.6} y:{y:.6} type = kAXValueCGPoint}}"),
    })
}

fn ax_size(w: f64, h: f64) -> AxAttr {
    AxAttr::Value(AxValue {
        unpacked: Some(AxStruct::Size(w, h)),
        description: format!("<AXValue 0x2> {{value = w:{w:.6} h:{h:.6} type = kAXValueCGSize}}"),
    })
}

struct AxState {
    /// Element names; the id of `names[i]` is `i + 1`.
    names: Vec<String>,
    trusted: bool,
    attrs: HashMap<(Element, String), AxAttr>,
    timeouts: Vec<(Element, f64)>,
    sets: Vec<(Element, String, bool)>,
    actions: Vec<(Element, String)>,
    prompts: u32,
    hit: Option<Element>,
    pids: HashMap<Element, i32>,
    unpack: bool,
    /// Elements whose app does not answer (every query times out).
    hung: HashSet<Element>,
    queries: Vec<(Element, String)>,
}

struct FakeAx(Mutex<AxState>);

impl FakeAx {
    fn new() -> Arc<Self> {
        Arc::new(Self(Mutex::new(AxState {
            names: Vec::new(),
            trusted: true,
            attrs: HashMap::new(),
            timeouts: Vec::new(),
            sets: Vec::new(),
            actions: Vec::new(),
            prompts: 0,
            hit: None,
            pids: HashMap::new(),
            unpack: true,
            hung: HashSet::new(),
            queries: Vec::new(),
        })))
    }

    /// The element called `name` (made up on first use).
    fn id(&self, name: &str) -> Element {
        let mut state = self.0.lock();
        let index = match state.names.iter().position(|n| n == name) {
            Some(index) => index,
            None => {
                state.names.push(name.to_string());
                state.names.len() - 1
            }
        };
        index + 1
    }

    fn set(&self, element: &str, key: &str, value: AxAttr) {
        let element = self.id(element);
        self.0.lock().attrs.insert((element, key.into()), value);
    }

    /// A window element with a frame; extra attributes with [`Self::set`].
    fn window(&self, name: &str, rect: (f64, f64, f64, f64), minimized: bool) -> Element {
        let (x, y, w, h) = rect;
        self.set(name, "AXPosition", ax_point(x, y));
        self.set(name, "AXSize", ax_size(w, h));
        self.set(name, "AXRole", AxAttr::Text("AXWindow".into()));
        self.set(name, "AXMinimized", AxAttr::Bool(minimized));
        self.id(name)
    }

    fn set_trusted(&self, trusted: bool) {
        self.0.lock().trusted = trusted;
    }

    fn hang(&self, element: Element) {
        self.0.lock().hung.insert(element);
    }

    fn queries(&self) -> Vec<(Element, String)> {
        self.0.lock().queries.clone()
    }

    fn timeouts(&self) -> Vec<(Element, f64)> {
        self.0.lock().timeouts.clone()
    }
}

impl Accessibility for FakeAx {
    fn create_application(&self, pid: i32) -> Element {
        self.id(&format!("app:{pid}"))
    }

    fn create_system_wide(&self) -> Element {
        self.id("system")
    }

    fn set_messaging_timeout(&self, element: Element, seconds: f64) -> i32 {
        self.0.lock().timeouts.push((element, seconds));
        0
    }

    fn copy_attribute(&self, element: Element, name: &str) -> (i32, Option<AxAttr>) {
        let mut state = self.0.lock();
        state.queries.push((element, name.to_string()));
        if state.hung.contains(&element) {
            return (AX_ERROR_CANNOT_COMPLETE, None);
        }
        match state.attrs.get(&(element, name.to_string())) {
            None => (AX_ERROR_ATTRIBUTE_UNSUPPORTED, None),
            Some(AxAttr::Value(value)) if !state.unpack => (
                0,
                Some(AxAttr::Value(AxValue {
                    unpacked: None,
                    ..value.clone()
                })),
            ),
            Some(value) => (0, Some(value.clone())),
        }
    }

    fn set_bool_attribute(&self, element: Element, name: &str, value: bool) -> i32 {
        self.0.lock().sets.push((element, name.to_string(), value));
        0
    }

    fn perform_action(&self, element: Element, action: &str) -> i32 {
        self.0.lock().actions.push((element, action.to_string()));
        0
    }

    fn element_at_position(&self, system: Element, _x: f64, _y: f64) -> (i32, Option<Element>) {
        assert_eq!(system, self.id("system"));
        match self.0.lock().hit {
            Some(hit) => (0, Some(hit)),
            None => (-25200, None),
        }
    }

    fn pid_of(&self, element: Element) -> (i32, Option<i32>) {
        (0, self.0.lock().pids.get(&element).copied())
    }

    fn is_process_trusted(&self) -> bool {
        self.0.lock().trusted
    }

    fn is_process_trusted_prompting(&self) -> bool {
        let mut state = self.0.lock();
        state.prompts += 1;
        state.trusted
    }
}

struct FakeRunningApp {
    terminated: bool,
    hidden: bool,
    bundle_id: String,
    activations: Vec<u64>,
    activate_result: bool,
}

struct FakeAppKit {
    apps: Mutex<HashMap<i32, FakeRunningApp>>,
    frontmost: AtomicI32,
    policies: Mutex<Vec<i64>>,
    mouse: Mutex<Option<(f64, f64)>>,
}

impl FakeAppKit {
    fn new() -> Arc<Self> {
        let kit = Self {
            apps: Mutex::new(HashMap::new()),
            frontmost: AtomicI32::new(PID),
            policies: Mutex::new(Vec::new()),
            mouse: Mutex::new(None),
        };
        kit.add(PID);
        Arc::new(kit)
    }

    fn add(&self, pid: i32) {
        self.apps.lock().insert(
            pid,
            FakeRunningApp {
                terminated: false,
                hidden: false,
                bundle_id: String::new(),
                activations: Vec::new(),
                activate_result: true,
            },
        );
    }

    fn with_app<R>(&self, pid: i32, f: impl FnOnce(&mut FakeRunningApp) -> R) -> R {
        f(self.apps.lock().get_mut(&pid).expect("app"))
    }

    fn activations(&self, pid: i32) -> Vec<u64> {
        self.with_app(pid, |app| app.activations.clone())
    }
}

impl AppKit for FakeAppKit {
    fn frontmost_pid(&self) -> Option<i32> {
        let frontmost = self.frontmost.load(Ordering::SeqCst);
        self.apps
            .lock()
            .contains_key(&frontmost)
            .then_some(frontmost)
    }

    fn running_application(&self, pid: i32) -> Option<RunningApp> {
        self.apps.lock().get(&pid).map(|app| RunningApp {
            terminated: app.terminated,
            hidden: app.hidden,
            bundle_id: app.bundle_id.clone(),
        })
    }

    fn activate(&self, pid: i32, options: u64) -> bool {
        match self.apps.lock().get_mut(&pid) {
            Some(app) => {
                app.activations.push(options);
                app.activate_result
            }
            None => false,
        }
    }

    fn set_activation_policy(&self, policy: i64) -> bool {
        self.policies.lock().push(policy);
        true
    }

    fn mouse_location(&self) -> Option<(f64, f64)> {
        *self.mouse.lock()
    }
}

#[derive(Default)]
struct FakeCoreFoundation {
    compared: Mutex<Vec<(Element, Element)>>,
}

impl CoreFoundation for FakeCoreFoundation {
    fn equal(&self, a: Element, b: Element) -> bool {
        self.compared.lock().push((a, b));
        a == b
    }
}

#[derive(Default)]
struct FakeFoundation {
    begun: Mutex<Vec<(u64, String)>>,
    ended: Mutex<Vec<String>>,
    fail: AtomicBool,
}

impl Foundation for FakeFoundation {
    fn begin_activity(&self, options: u64, reason: &str) -> Option<ActivityToken> {
        if self.fail.load(Ordering::SeqCst) {
            return None; // "no activities here"
        }
        let mut begun = self.begun.lock();
        begun.push((options, reason.to_string()));
        Some(Box::new(format!("token{}", begun.len())))
    }

    fn end_activity(&self, token: ActivityToken) {
        let token = token.downcast::<String>().expect("a fake token");
        self.ended.lock().push(*token);
    }
}

struct FakeCamera {
    status: Mutex<Option<i64>>,
    queried: Mutex<Vec<String>>,
    requested: Mutex<Vec<String>>,
}

impl FakeCamera {
    fn new(status: Option<i64>) -> Arc<Self> {
        Arc::new(Self {
            status: Mutex::new(status),
            queried: Mutex::new(Vec::new()),
            requested: Mutex::new(Vec::new()),
        })
    }
}

impl CameraAuthorization for FakeCamera {
    fn authorization_status(&self, media_type: &str) -> Option<i64> {
        self.queried.lock().push(media_type.to_string());
        *self.status.lock()
    }

    fn request_access(&self, media_type: &str) {
        self.requested.lock().push(media_type.to_string());
    }
}

// -------------------------------------------------------------------- fixtures
/// A platform whose frameworks are all "missing" until a test injects a fake (the `plat`
/// fixture), with its clock, host and state file.
struct Env {
    plat: MacPlatform,
    clock: Clock,
    host: Arc<FakeHost>,
    state_path: PathBuf,
    dir: tempfile::TempDir,
}

fn env() -> Env {
    let dir = tempfile::tempdir().expect("temporary directory");
    let state_path = dir.path().join("state").join(ACCESSIBILITY_STATE_FILE);
    let clock = Clock::new();
    let host = FakeHost::new();
    let plat = platform_on(&clock, &state_path, &host);
    Env {
        plat,
        clock,
        host,
        state_path,
        dir,
    }
}

fn platform_on(clock: &Clock, state_path: &Path, host: &Arc<FakeHost>) -> MacPlatform {
    MacPlatform::with_host(clock.reader(), Some(state_path.to_path_buf()), host.clone())
}

impl Env {
    fn quartz(&mut self) -> Arc<FakeQuartz> {
        let fake = FakeQuartz::new();
        self.plat.quartz = Some(fake.clone());
        fake
    }

    fn ax(&mut self) -> Arc<FakeAx> {
        let fake = FakeAx::new();
        self.plat.ax = Some(fake.clone());
        fake
    }

    fn appkit(&mut self) -> Arc<FakeAppKit> {
        let fake = FakeAppKit::new();
        self.plat.appkit = Some(fake.clone());
        fake
    }

    fn foundation(&mut self) -> Arc<FakeFoundation> {
        let fake = Arc::new(FakeFoundation::default());
        self.plat.foundation = Some(fake.clone());
        fake
    }

    fn camera(&mut self, status: Option<i64>) -> Arc<FakeCamera> {
        let fake = FakeCamera::new(status);
        self.plat.camera = Some(fake.clone());
        fake
    }
}

fn wref(pid: i32, element: Element) -> WindowRef {
    WindowRef {
        handle: WindowHandle::Ax { pid, element },
        pid: None,
        rect: None,
    }
}

fn token_of(plat: &MacPlatform) -> Option<String> {
    plat.activity.lock().as_ref().map(|token| {
        token
            .downcast_ref::<String>()
            .expect("a fake token")
            .clone()
    })
}

fn names<'a>(caps: &'a HashMap<&'static str, bool>) -> HashSet<&'a str> {
    caps.keys().copied().collect()
}

// ------------------------------------------------------------ graceful degradation
#[test]
fn test_everything_degrades_without_pyobjc() {
    let e = env();
    let plat = &e.plat;
    let reference = WindowRef {
        pid: Some(PID as u32),
        ..wref(PID, 0)
    };
    assert_eq!(plat.seconds_since_input(), None);
    assert_eq!(plat.seconds_since_key_input(), None);
    assert_eq!(plat.is_session_locked(), None);
    assert_eq!(plat.move_cursor(1, 2), None); // the pointer cannot be moved
    assert_eq!(plat.cursor_position(), None);
    assert!(plat.foreground_window().is_none());
    assert!(plat.window_at(1, 2).is_none());
    assert!(!plat.activate_window(&reference));
    assert!(!plat.is_window_valid(&reference));
    assert_eq!(plat.window_rect(&reference), None);
    assert!(!plat.same_window(Some(&reference), None));
    assert_eq!(plat.camera_in_use_by_other_app(), None);
    assert_eq!(
        plat.permissions(),
        HashMap::from([("camera", None), ("accessibility", None)])
    );
    assert!(!plat.set_accessory_app());
    plat.request_permission("accessibility");
    plat.request_permission("camera");
    plat.prepare_process();
    assert_eq!(
        names(&plat.capabilities()),
        names(&GenericPlatform.capabilities())
    );
    assert!(e.host.runs().is_empty());
    assert!(e.host.popens().is_empty());
}

#[test]
fn test_power_actions_are_macos_only() {
    let e = env();
    e.host.set_tools(&[
        ("pmset", "/usr/bin/pmset"),
        ("caffeinate", "/usr/bin/caffeinate"),
    ]);
    assert!(!e.plat.lock_screen());
    assert!(!e.plat.display_off());
    assert!(!e.plat.wake_display());
    assert!(!e.plat.open_permission_settings("camera"));
    assert!(e.host.runs().is_empty());
    assert!(e.host.popens().is_empty());
}

// ------------------------------------------------------------------ lock / power
#[test]
fn test_lock_uses_login_framework() {
    let e = env();
    e.host.on_macos();
    *e.host.login_status.lock() = Some(0);
    e.host.set_tools(&[("pmset", "/usr/bin/pmset")]);
    assert!(e.plat.lock_screen());
    assert_eq!(*e.host.loaded.lock(), [LOGIN_FRAMEWORK]);
    assert_eq!(e.host.lock_calls.load(Ordering::SeqCst), 1);
    assert!(e.host.runs().is_empty());
}

#[test]
fn test_lock_falls_back_to_display_sleep() {
    let e = env();
    e.host.on_macos();
    *e.host.login_status.lock() = Some(-1);
    e.host.set_tools(&[("pmset", "/usr/bin/pmset")]);
    assert!(e.plat.lock_screen());
    assert_eq!(
        e.host.runs(),
        [["/usr/bin/pmset", "displaysleepnow"].map(String::from)]
    );
}

#[test]
fn test_lock_fails_without_any_mechanism() {
    let e = env();
    e.host.on_macos();
    assert!(!e.plat.lock_screen());
    e.host.set_tools(&[("pmset", "/usr/bin/pmset")]);
    e.host.returncode.store(1, Ordering::SeqCst);
    assert!(!e.plat.lock_screen());
}

#[test]
fn test_display_off_and_wake() {
    let e = env();
    e.host.on_macos();
    e.host.set_tools(&[
        ("pmset", "/usr/bin/pmset"),
        ("caffeinate", "/usr/bin/caffeinate"),
    ]);
    assert!(e.plat.display_off());
    assert!(e.plat.wake_display());
    assert_eq!(
        e.host.runs(),
        [["/usr/bin/pmset", "displaysleepnow"].map(String::from)]
    );
    assert_eq!(
        e.host.popens(),
        [["/usr/bin/caffeinate", "-u", "-t", "2"].map(String::from)]
    );
    e.host.set_tools(&[]);
    assert!(!e.plat.wake_display());
}

// ------------------------------------------------------------------ input / session
#[test]
fn test_idle_times_from_quartz() {
    let mut e = env();
    e.quartz();
    approx::assert_abs_diff_eq!(e.plat.seconds_since_input().unwrap(), 12.5, epsilon = 1e-9);
    approx::assert_abs_diff_eq!(
        e.plat.seconds_since_key_input().unwrap(),
        3.25,
        epsilon = 1e-9
    );
}

#[test]
fn test_idle_uses_fallback_constants() {
    let mut e = env();
    let quartz = e.quartz();
    *quartz.idle.lock() = HashMap::from([(0xFFFF_FFFF, 1.0), (10, 1.0)]);
    assert_eq!(e.plat.seconds_since_input(), Some(1.0));
    assert_eq!(e.plat.seconds_since_key_input(), Some(1.0));
    assert_eq!(*quartz.idle_calls.lock(), [(0, 0xFFFF_FFFF), (0, 10)]);
}

#[test]
fn test_idle_rejects_nonsense() {
    for value in [-1.0, f64::NAN, f64::INFINITY] {
        let mut e = env();
        let quartz = e.quartz();
        quartz.idle.lock().insert(0xFFFF_FFFF, value);
        assert_eq!(e.plat.seconds_since_input(), None, "{value}");
    }
}

#[test]
fn test_session_locked() {
    let info = |locked, console| {
        Some(SessionInfo {
            screen_locked: locked,
            on_console: console,
        })
    };
    let cases = [
        (None, None),
        (Some(SessionInfo::default()), Some(false)),
        (info(true, None), Some(true)),
        (info(true, Some(true)), Some(true)),
        (info(false, Some(true)), Some(false)),
        (info(false, Some(false)), Some(true)), // fast user switching
    ];
    for (session, expected) in cases {
        let mut e = env();
        let quartz = e.quartz();
        *quartz.session.lock() = session.clone();
        assert_eq!(e.plat.is_session_locked(), expected, "{session:?}");
    }
}

#[test]
fn test_move_cursor_warps_and_reassociates() {
    let mut e = env();
    let quartz = e.quartz();
    assert_eq!(e.plat.move_cursor(2560, -300), Some(true));
    assert_eq!(*quartz.warps.lock(), [(2560.0, -300.0)]);
    assert_eq!(*quartz.associations.lock(), [true]);
    quartz.warp_error.store(1001, Ordering::SeqCst);
    assert_eq!(e.plat.move_cursor(0, 0), Some(false));
}

#[test]
fn test_cursor_position_is_in_top_left_points() {
    let mut e = env();
    let quartz = e.quartz();
    let appkit = e.appkit();
    *quartz.main_height.lock() = Some(1080.0);
    // Cocoa: 100 points above the main display's bottom edge, on a monitor to its left.
    *appkit.mouse.lock() = Some((-500.0, 100.0));
    assert_eq!(e.plat.cursor_position(), Some((-500, 980)));
    // Above the main display (a monitor stacked on top): negative y.
    *appkit.mouse.lock() = Some((10.5, 1380.0));
    assert_eq!(e.plat.cursor_position(), Some((10, -300)));
    *quartz.main_height.lock() = None;
    assert_eq!(e.plat.cursor_position(), None);
}

// ------------------------------------------------------------------ pure helpers
#[test]
fn test_err_value_and_handle_splitting() {
    let handle = |pid, element| WindowHandle::Ax { pid, element };
    assert_eq!(split_handle(&handle(PID, 7)), Some((PID, 7)));
    assert_eq!(split_handle(&handle(PID, 0)), Some((PID, 0)));
    assert_eq!(split_handle(&handle(0, 7)), None);
    assert_eq!(split_handle(&handle(-5, 7)), None);
    assert_eq!(split_handle(&WindowHandle::Hwnd(12345)), None);
}

#[test]
fn test_ax_value_repr_parsing() {
    let point = "<AXValue 0x6000> {value = x:24.000000 y:-38.500000 type = kAXValueCGPointType}";
    let size = "<AXValue 0x6001> {value = w:1200.000000 h:800.000000 type = kAXValueCGSizeType}";
    assert_eq!(pair_from_repr(point, false), Some((24.0, -38.5)));
    assert_eq!(pair_from_repr(size, true), Some((1200.0, 800.0)));
    assert_eq!(pair_from_repr(point, true), None);
    assert_eq!(pair_from_repr("garbage", false), None);
}

#[test]
fn test_rect_from_pairs() {
    assert_eq!(
        rect_from(Some((10.4, 20.6)), Some((800.0, 600.0))),
        Some(Rect::new(10, 21, 800, 600))
    );
    assert_eq!(rect_from(None, Some((1.0, 1.0))), None);
    assert_eq!(rect_from(Some((0.0, 0.0)), Some((0.0, 10.0))), None);
    assert_eq!(rect_from(Some((f64::NAN, 0.0)), Some((10.0, 10.0))), None);
}

#[test]
fn test_ax_rect_unpacks_values() {
    let ax = FakeAx::new();
    let window = ax.window("win", (-1440.0, 25.0, 1200.0, 800.0), false);
    assert_eq!(
        ax_rect(Some(&*ax), window),
        Some(Rect::new(-1440, 25, 1200, 800))
    );
    assert_eq!(ax_rect(Some(&*ax), ax.id("missing")), None);
    assert_eq!(ax_rect(None, window), None);
}

#[test]
fn test_ax_rect_falls_back_to_description() {
    let ax = FakeAx::new();
    let window = ax.window("win", (10.0, 20.0, 300.0, 200.0), false);
    ax.0.lock().unpack = false; // AXValueGetValue cannot unpack in this build
    assert_eq!(
        ax_rect(Some(&*ax), window),
        Some(Rect::new(10, 20, 300, 200))
    );
}

#[test]
fn test_ax_window_of_walks_up_the_hierarchy() {
    let ax = FakeAx::new();
    let window = ax.window("win", (0.0, 0.0, 10.0, 10.0), false);
    let text = |s: &str| AxAttr::Text(s.into());
    ax.set("button", "AXRole", text("AXButton"));
    ax.set("button", "AXWindow", AxAttr::Element(window));
    ax.set("cell", "AXRole", text("AXCell"));
    ax.set("cell", "AXParent", AxAttr::Element(ax.id("group")));
    ax.set("group", "AXRole", text("AXGroup"));
    ax.set("group", "AXParent", AxAttr::Element(window));
    ax.set("menu", "AXRole", text("AXMenuItem"));
    ax.set("menu", "AXParent", AxAttr::Element(ax.id("app")));
    ax.set("app", "AXRole", text("AXApplication"));
    assert_eq!(ax_window_of(&*ax, window), Some(window));
    assert_eq!(ax_window_of(&*ax, ax.id("button")), Some(window));
    assert_eq!(ax_window_of(&*ax, ax.id("cell")), Some(window));
    assert_eq!(ax_window_of(&*ax, ax.id("menu")), None);
}

#[test]
fn test_ax_window_of_is_bounded() {
    let ax = FakeAx::new();
    let looping = ax.id("loop");
    ax.set("loop", "AXRole", AxAttr::Text("AXGroup".into()));
    ax.set("loop", "AXParent", AxAttr::Element(looping)); // a broken, cyclic hierarchy
    assert_eq!(ax_window_of(&*ax, looping), None);
}

#[test]
fn test_cg_window_selection() {
    let windows = vec![
        cg_window(OTHER_PID, 0.0, 0.0, 3000.0, 25.0, 25, 1.0), // menu bar
        win(999, 0.0, 0.0, 3000.0, 2000.0),                    // our own window (pid 999 below)
        cg_window(OTHER_PID, 100.0, 100.0, 400.0, 300.0, 0, 0.0), // invisible
        win(PID, 50.0, 50.0, 800.0, 600.0),
        win(OTHER_PID, 0.0, 0.0, 1000.0, 1000.0),
        // The Python entry with a non-numeric owner pid: no usable owner.
        CgWindow {
            layer: 0,
            alpha: 1.0,
            owner_pid: 0,
            bounds: None,
        },
    ];
    assert_eq!(
        cg_window_at(&windows, 150.0, 150.0, 999),
        Some((PID, Rect::new(50, 50, 800, 600)))
    );
    assert_eq!(
        cg_window_at(&windows, 900.0, 900.0, 999),
        Some((OTHER_PID, Rect::new(0, 0, 1000, 1000)))
    );
    assert_eq!(cg_window_at(&windows, 5000.0, 5000.0, 999), None);
    assert_eq!(
        cg_front_rect(&windows, OTHER_PID, 999),
        Some(Rect::new(0, 0, 1000, 1000))
    );
    assert_eq!(cg_front_rect(&windows, 12, 999), None);
}

// ------------------------------------------------------------------ windows
#[test]
fn test_foreground_window_with_accessibility() {
    let mut e = env();
    let (ax, quartz) = (e.ax(), e.quartz());
    e.appkit();
    quartz.set_windows(vec![]);
    let window = ax.window("win", (100.0, 50.0, 800.0, 600.0), false);
    ax.set(
        &format!("app:{PID}"),
        "AXFocusedWindow",
        AxAttr::Element(window),
    );
    let reference = e.plat.foreground_window().expect("a window");
    assert_eq!(
        reference.handle,
        WindowHandle::Ax {
            pid: PID,
            element: window
        }
    );
    assert_eq!(reference.pid, Some(PID as u32));
    assert_eq!(reference.rect, Some(Rect::new(100, 50, 800, 600)));
    // The global AX timeout is set once; the polled app gets a tighter one.
    let (system, app) = (ax.id("system"), ax.id(&format!("app:{PID}")));
    assert_eq!(ax.timeouts(), [(system, 0.5), (app, 0.25)]);
    e.plat.foreground_window();
    let system_timeouts: Vec<_> = ax
        .timeouts()
        .into_iter()
        .filter(|(element, _)| *element == system)
        .collect();
    assert_eq!(system_timeouts, [(system, 0.5)]);
}

#[test]
fn test_foreground_window_without_accessibility() {
    let mut e = env();
    let (ax, quartz) = (e.ax(), e.quartz());
    e.appkit();
    ax.set_trusted(false);
    quartz.set_windows(vec![
        win(OTHER_PID, 0.0, 0.0, 10.0, 10.0),
        win(PID, 1920.0, 0.0, 1280.0, 720.0),
    ]);
    let reference = e.plat.foreground_window().expect("a window");
    assert_eq!(
        reference.handle,
        WindowHandle::Ax {
            pid: PID,
            element: 0
        }
    );
    assert_eq!(reference.rect, Some(Rect::new(1920, 0, 1280, 720)));
}

#[test]
fn test_foreground_window_skips_our_own_app() {
    let mut e = env();
    e.appkit();
    e.host.own_pid.store(PID, Ordering::SeqCst);
    assert!(e.plat.foreground_window().is_none());
}

#[test]
fn test_accessibility_trust_is_cached() {
    let mut e = env();
    let ax = e.ax();
    assert!(e.plat.ax_trusted());
    ax.set_trusted(false);
    e.clock.advance(1.0);
    assert!(e.plat.ax_trusted());
    e.clock.advance(5.0);
    assert!(!e.plat.ax_trusted());
}

#[test]
fn test_window_at_uses_window_list_then_ax() {
    let mut e = env();
    let (ax, quartz) = (e.ax(), e.quartz());
    quartz.set_windows(vec![win(PID, 0.0, 0.0, 800.0, 600.0)]);
    let back = ax.window("back", (0.0, 0.0, 1920.0, 1080.0), false);
    let small = ax.window("small", (0.0, 0.0, 200.0, 200.0), false);
    let mini = ax.window("mini", (0.0, 0.0, 800.0, 600.0), true);
    let exact = ax.window("exact", (0.0, 0.0, 800.0, 600.0), false);
    let windows = format!("app:{PID}");
    ax.set(
        &windows,
        "AXWindows",
        AxAttr::Elements(vec![mini, small, back, exact]),
    );
    let reference = e.plat.window_at(100, 100).expect("a window");
    // same frame as the window-list entry
    assert_eq!(
        reference.handle,
        WindowHandle::Ax {
            pid: PID,
            element: exact
        }
    );
    assert_eq!(reference.rect, Some(Rect::new(0, 0, 800, 600)));
    ax.set(
        &windows,
        "AXWindows",
        AxAttr::Elements(vec![mini, small, back]),
    );
    assert_eq!(
        e.plat.window_at(100, 100).expect("a window").handle,
        WindowHandle::Ax {
            pid: PID,
            element: small
        }
    );
    assert!(e.plat.window_at(5000, 100).is_none());
}

#[test]
fn test_window_at_without_accessibility() {
    let mut e = env();
    let (ax, quartz) = (e.ax(), e.quartz());
    ax.set_trusted(false);
    quartz.set_windows(vec![win(PID, 0.0, 0.0, 800.0, 600.0)]);
    let reference = e.plat.window_at(10, 10).expect("a window");
    assert_eq!(
        reference.handle,
        WindowHandle::Ax {
            pid: PID,
            element: 0
        }
    );
    assert_eq!(reference.rect, Some(Rect::new(0, 0, 800, 600)));
}

#[test]
fn test_window_at_hit_test_fallback() {
    let mut e = env();
    let ax = e.ax();
    let window = ax.window("win", (0.0, 0.0, 640.0, 480.0), false);
    ax.set("text", "AXRole", AxAttr::Text("AXTextArea".into()));
    ax.set("text", "AXWindow", AxAttr::Element(window));
    ax.0.lock().hit = Some(ax.id("text"));
    ax.0.lock().pids.insert(window, PID);
    let reference = e.plat.window_at(10, 10).expect("a window"); // Quartz missing: AX hit test
    assert_eq!(
        reference.handle,
        WindowHandle::Ax {
            pid: PID,
            element: window
        }
    );
    assert_eq!(reference.rect, Some(Rect::new(0, 0, 640, 480)));
    e.host.own_pid.store(PID, Ordering::SeqCst);
    assert!(e.plat.window_at(10, 10).is_none());
}

#[test]
fn test_activate_window_full_sequence() {
    let mut e = env();
    let (ax, appkit) = (e.ax(), e.appkit());
    e.quartz();
    let window = ax.window("win", (0.0, 0.0, 10.0, 10.0), false);
    assert!(e.plat.activate_window(&wref(PID, window)));
    let app = ax.id(&format!("app:{PID}"));
    let state = ax.0.lock();
    assert!(state.sets.contains(&(window, "AXMain".into(), true)));
    assert!(state.sets.contains(&(app, "AXFrontmost".into(), true)));
    assert_eq!(state.actions, [(window, "AXRaise".to_string())]);
    drop(state);
    assert_eq!(appkit.activations(PID), [2]);
}

#[test]
fn test_activate_window_refuses_minimised() {
    let mut e = env();
    let (ax, appkit) = (e.ax(), e.appkit());
    let window = ax.window("win", (0.0, 0.0, 10.0, 10.0), true);
    assert!(!e.plat.activate_window(&wref(PID, window)));
    assert!(appkit.activations(PID).is_empty());
    assert!(ax.0.lock().actions.is_empty());
}

#[test]
fn test_activate_app_without_accessibility() {
    let mut e = env();
    let (ax, appkit) = (e.ax(), e.appkit());
    e.host.set_major_version(13);
    ax.set_trusted(false);
    assert!(e.plat.activate_window(&wref(PID, 0)));
    assert_eq!(appkit.activations(PID), [2]);
    assert!(ax.0.lock().sets.is_empty());
    appkit.with_app(PID, |app| app.activate_result = false);
    assert!(!e.plat.activate_window(&wref(PID, 0)));
    // not running
    assert!(!e.plat.activate_window(&wref(OTHER_PID, 0)));
    let bogus = WindowRef {
        handle: WindowHandle::Hwnd(1),
        pid: None,
        rect: None,
    };
    assert!(!e.plat.activate_window(&bogus));
    // macOS 14+ ignores activation requests from an inactive (menu-bar) app while still
    // reporting success: asked anyway, but reported as failed.
    e.host.set_major_version(14);
    appkit.with_app(PID, |app| app.activate_result = true);
    assert!(!e.plat.activate_window(&wref(PID, 0)));
    assert_eq!(appkit.activations(PID), [2, 2, 2]);
}

#[test]
fn test_is_window_valid() {
    let mut e = env();
    let (ax, appkit) = (e.ax(), e.appkit());
    let window = ax.window("win", (0.0, 0.0, 10.0, 10.0), false);
    let mini = ax.window("mini", (0.0, 0.0, 10.0, 10.0), true);
    let closed = ax.id("closed");
    assert!(e.plat.is_window_valid(&wref(PID, window)));
    assert!(e.plat.is_window_valid(&wref(PID, 0)));
    assert!(!e.plat.is_window_valid(&wref(PID, mini)));
    assert!(!e.plat.is_window_valid(&wref(PID, closed)));
    assert!(!e.plat.is_window_valid(&wref(OTHER_PID, 0)));
    appkit.with_app(PID, |app| app.hidden = true);
    assert!(!e.plat.is_window_valid(&wref(PID, 0)));
    appkit.with_app(PID, |app| {
        app.hidden = false;
        app.terminated = true;
    });
    assert!(!e.plat.is_window_valid(&wref(PID, window)));
}

#[test]
fn test_window_rect() {
    let mut e = env();
    let (ax, quartz) = (e.ax(), e.quartz());
    let window = ax.window("win", (5.0, 6.0, 70.0, 80.0), false);
    quartz.set_windows(vec![win(PID, 1.0, 2.0, 30.0, 40.0)]);
    assert_eq!(
        e.plat.window_rect(&wref(PID, window)),
        Some(Rect::new(5, 6, 70, 80))
    );
    assert_eq!(
        e.plat.window_rect(&wref(PID, 0)),
        Some(Rect::new(1, 2, 30, 40))
    );
    let bogus = WindowRef {
        handle: WindowHandle::Hwnd(1),
        pid: None,
        rect: None,
    };
    assert_eq!(e.plat.window_rect(&bogus), None);
    // The content area is approximated by the frame.
    assert_eq!(
        e.plat.window_client_rect(&wref(PID, window)),
        Some(Rect::new(5, 6, 70, 80))
    );
}

#[test]
fn test_window_app_names_process_and_bundle() {
    let mut e = env();
    let appkit = e.appkit();
    appkit.with_app(PID, |app| app.bundle_id = "com.github.wez.wezterm".into());
    *e.host.process_names.lock() = HashMap::from([
        (i64::from(PID), "wezterm-gui".to_string()),
        (i64::from(OTHER_PID), "kitty".to_string()),
    ]);
    assert_eq!(
        e.plat.window_app(&wref(PID, 5)),
        Some(AppIdentity {
            process: "wezterm-gui".into(),
            app_id: "com.github.wez.wezterm".into()
        })
    );
    // Not a running application AppKit knows: the process name alone.
    assert_eq!(
        e.plat.window_app(&wref(OTHER_PID, 0)),
        Some(AppIdentity {
            process: "kitty".into(),
            app_id: String::new()
        })
    );
    let bogus = WindowRef {
        handle: WindowHandle::Hwnd(1),
        pid: None,
        rect: None,
    };
    assert_eq!(e.plat.window_app(&bogus), None);
}

#[test]
fn test_cursor_position_is_reliable() {
    assert!(env().plat.cursor_position_reliable());
}

// ------------------------------------------------------------------ windows: Spaces
#[test]
fn test_window_on_another_space_is_not_valid() {
    // AX keeps answering for a window on another Space (or behind a full-screen app) with a
    // frame inside the display; only the on-screen list tells.
    let mut e = env();
    let (ax, quartz) = (e.ax(), e.quartz());
    e.appkit();
    let window = ax.window("win", (100.0, 50.0, 800.0, 600.0), false);
    // a full-screen app in front
    quartz.set_windows(vec![win(OTHER_PID, 0.0, 0.0, 1920.0, 1080.0)]);
    assert!(!e.plat.is_window_valid(&wref(PID, window)));
    assert!(!e.plat.is_window_valid(&wref(PID, 0)));
    // back on the visible Space
    quartz
        .windows
        .lock()
        .push(win(PID, 101.0, 49.0, 799.0, 601.0));
    assert!(e.plat.is_window_valid(&wref(PID, window)));
    assert!(e.plat.is_window_valid(&wref(PID, 0)));
}

#[test]
fn test_other_window_of_the_same_app_does_not_count() {
    let mut e = env();
    let (ax, quartz) = (e.ax(), e.quartz());
    e.appkit();
    let window = ax.window("win", (100.0, 50.0, 800.0, 600.0), false);
    // a different window of the app
    quartz.set_windows(vec![win(PID, 1000.0, 50.0, 400.0, 300.0)]);
    assert!(!e.plat.is_window_valid(&wref(PID, window)));
    assert!(e.plat.is_window_valid(&wref(PID, 0)));
}

#[test]
fn test_frames_match_tolerance() {
    let frame = Rect::new(0, 0, 100, 100);
    assert!(frames_match(
        &frame,
        &Rect::new(2, -2, 98, 102),
        FRAME_TOLERANCE
    ));
    assert!(!frames_match(
        &frame,
        &Rect::new(3, 0, 100, 100),
        FRAME_TOLERANCE
    ));
}

// ------------------------------------------------------------------ windows: hung apps
#[test]
fn test_hung_frontmost_app_is_left_alone_for_a_while() {
    let mut e = env();
    let (ax, quartz) = (e.ax(), e.quartz());
    e.appkit();
    quartz.set_windows(vec![win(PID, 1920.0, 0.0, 1280.0, 720.0)]);
    let app = ax.id(&format!("app:{PID}"));
    ax.hang(app);
    let reference = e.plat.foreground_window().expect("a window");
    // application level, from the window list
    assert_eq!(
        reference.handle,
        WindowHandle::Ax {
            pid: PID,
            element: 0
        }
    );
    assert_eq!(reference.rect, Some(Rect::new(1920, 0, 1280, 720)));
    let asked = ax.queries().len();
    e.clock.advance(1.0);
    let again = e.plat.foreground_window().expect("a window");
    assert_eq!(
        (again.handle, again.rect),
        (reference.handle, reference.rect)
    );
    // no further AX request to the hung app
    assert_eq!(ax.queries().len(), asked);
    ax.0.lock().hung.clear();
    let window = ax.window("win", (1920.0, 0.0, 1280.0, 720.0), false);
    ax.set(
        &format!("app:{PID}"),
        "AXFocusedWindow",
        AxAttr::Element(window),
    );
    e.clock.advance(AX_UNRESPONSIVE_S);
    assert_eq!(
        e.plat.foreground_window().expect("a window").handle,
        WindowHandle::Ax {
            pid: PID,
            element: window
        }
    );
}

#[test]
fn test_ax_rect_stops_after_a_failed_position() {
    let ax = FakeAx::new();
    let window = ax.window("win", (0.0, 0.0, 10.0, 10.0), false);
    ax.hang(window);
    assert_eq!(
        ax_rect_checked(Some(&*ax), window),
        (AX_ERROR_CANNOT_COMPLETE, None)
    );
    // AXSize would time out again
    assert_eq!(ax.queries(), [(window, "AXPosition".to_string())]);
}

#[test]
fn test_hung_window_is_not_a_target() {
    let mut e = env();
    let (ax, quartz) = (e.ax(), e.quartz());
    e.appkit();
    let window = ax.window("win", (5.0, 6.0, 70.0, 80.0), false);
    quartz.set_windows(vec![win(PID, 5.0, 6.0, 70.0, 80.0)]);
    ax.hang(window);
    let reference = wref(PID, window);
    assert!(!e.plat.is_window_valid(&reference));
    let asked = ax.queries().len();
    // from the window list
    assert_eq!(
        e.plat.window_rect(&reference),
        Some(Rect::new(5, 6, 70, 80))
    );
    assert!(!e.plat.is_window_valid(&reference));
    assert_eq!(ax.queries().len(), asked);
}

#[test]
fn test_activating_a_hung_app_skips_accessibility() {
    let mut e = env();
    let (ax, appkit) = (e.ax(), e.appkit());
    let window = ax.window("win", (0.0, 0.0, 10.0, 10.0), false);
    ax.hang(window);
    assert!(e.plat.activate_window(&wref(PID, window)));
    assert_eq!(appkit.activations(PID), [2]);
    assert!(ax.0.lock().sets.is_empty());
    assert!(ax.0.lock().actions.is_empty());
    assert_eq!(ax.queries(), [(window, "AXMinimized".to_string())]);
}

#[test]
fn test_same_window() {
    let mut e = env();
    let cf = Arc::new(FakeCoreFoundation::default());
    e.plat.core_foundation = Some(cf.clone());
    let (w1, w2) = (11, 12);
    let a = wref(PID, w1);
    assert!(e.plat.same_window(Some(&a), Some(&wref(PID, w1))));
    assert!(!e.plat.same_window(Some(&a), Some(&wref(PID, w2))));
    assert!(!e.plat.same_window(Some(&a), Some(&wref(OTHER_PID, w1))));
    assert!(!e.plat.same_window(Some(&a), Some(&wref(PID, 0))));
    assert!(e.plat.same_window(Some(&wref(PID, 0)), Some(&wref(PID, 0))));
    assert!(!e.plat.same_window(Some(&a), None));
    assert_eq!(*cf.compared.lock(), [(w1, w1), (w1, w2)]);
}

// ------------------------------------------------------------------ permissions
#[test]
fn test_accessibility_permission_and_prompt() {
    let mut e = env();
    let ax = e.ax();
    assert_eq!(e.plat.permissions()["accessibility"], Some(true));
    ax.set_trusted(false);
    assert_eq!(e.plat.permissions()["accessibility"], Some(false));
    e.plat.request_permission("accessibility");
    assert_eq!(ax.0.lock().prompts, 1);
    // the trust answer is asked again after a prompt
    assert!(e.plat.trust_cache.lock().is_none());
}

#[test]
fn test_camera_permission_via_avfoundation() {
    for (status, expected) in [
        (0, None),
        (1, Some(false)),
        (2, Some(false)),
        (3, Some(true)),
    ] {
        let mut e = env();
        let camera = e.camera(Some(status));
        assert_eq!(e.plat.permissions()["camera"], expected, "status {status}");
        assert_eq!(*camera.queried.lock(), ["vide"]);
    }
}

#[test]
fn test_camera_permission_request() {
    let mut e = env();
    let camera = e.camera(Some(0));
    e.plat.request_permission("camera");
    assert_eq!(*camera.requested.lock(), ["vide"]);
}

#[test]
fn test_open_permission_settings() {
    let e = env();
    e.host.on_macos();
    e.host.set_tools(&[("open", "/usr/bin/open")]);
    assert!(e.plat.open_permission_settings("accessibility"));
    assert!(e.plat.open_permission_settings("camera"));
    assert!(!e.plat.open_permission_settings("bluetooth"));
    let url = |name: &str| {
        PERMISSION_URLS
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, url)| url.to_string())
            .unwrap()
    };
    assert_eq!(
        e.host.runs(),
        [
            vec!["/usr/bin/open".to_string(), url("accessibility")],
            vec!["/usr/bin/open".to_string(), url("camera")],
        ]
    );
    assert!(e.host.runs()[0][1].ends_with("Privacy_Accessibility"));
}

#[test]
fn test_set_accessory_app() {
    let mut e = env();
    let appkit = e.appkit();
    assert!(e.plat.set_accessory_app());
    assert_eq!(*appkit.policies.lock(), [1]);
}

// ------------------------------------------------------------------ App Nap
#[test]
fn test_accessory_app_opts_out_of_app_nap() {
    let mut e = env();
    e.appkit();
    let info = e.foundation();
    assert!(e.plat.set_accessory_app());
    // User initiated (no App Nap, no timer coalescing) but idle sleep still allowed.
    assert_eq!(
        *info.begun.lock(),
        [(0x00EF_FFFF, ACTIVITY_REASON.to_string())]
    );
    // the token must stay referenced
    assert_eq!(token_of(&e.plat).as_deref(), Some("token1"));
}

#[test]
fn test_background_activity_begins_and_ends_once() {
    let mut e = env();
    let info = e.foundation();
    assert!(e.plat.set_background_activity(true));
    assert!(e.plat.set_background_activity(true));
    assert_eq!(info.begun.lock().len(), 1);
    assert!(e.plat.set_background_activity(false));
    assert!(e.plat.set_background_activity(false));
    assert_eq!(*info.ended.lock(), ["token1"]);
    assert_eq!(token_of(&e.plat), None);
    // e.g. tracking resumed
    assert!(e.plat.set_background_activity(true));
    assert_eq!(token_of(&e.plat).as_deref(), Some("token2"));
}

#[test]
fn test_background_activity_degrades() {
    let mut e = env();
    let info = e.foundation();
    info.fail.store(true, Ordering::SeqCst);
    assert!(!e.plat.set_background_activity(true));
    assert_eq!(token_of(&e.plat), None);
    e.plat.foundation = None;
    assert!(!e.plat.set_background_activity(true));
    // nothing to end
    assert!(e.plat.set_background_activity(false));
}

// ------------------------------------------------------------------ Accessibility after updates
#[test]
fn test_accessibility_status_granted_and_missing() {
    let mut e = env();
    let ax = e.ax();
    assert_eq!(e.plat.accessibility_status(), "granted");
    ax.set_trusted(false);
    assert_eq!(e.plat.accessibility_status(), "missing");
    // an unidentified build has nothing to remember
    assert!(!e.state_path.exists());
}

#[test]
fn test_accessibility_status_unknown_without_the_api() {
    assert_eq!(env().plat.accessibility_status(), "unknown");
}

fn recorded_build(state_path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(state_path).expect("state file"))
        .expect("state file is JSON")
}

#[test]
fn test_accessibility_granted_to_an_earlier_build_is_stale() {
    let e = env();
    let fake = FakeAx::new();
    *e.host.fingerprint.lock() = Some("1.0:100:1".into());
    let first = platform_on(&e.clock, &e.state_path, &e.host);
    let mut first = first;
    first.ax = Some(fake.clone());
    assert_eq!(first.accessibility_status(), "granted");
    assert_eq!(
        recorded_build(&e.state_path),
        serde_json::json!({"trusted_build": "1.0:100:1"})
    );

    // An update: new ad-hoc signature, System Settings still shows the old grant.
    fake.set_trusted(false);
    *e.host.fingerprint.lock() = Some("1.1:120:2".into());
    let mut second = platform_on(&e.clock, &e.state_path, &e.host);
    second.ax = Some(fake.clone());
    assert_eq!(second.accessibility_status(), "stale");
    assert_eq!(second.permissions()["accessibility"], Some(false));

    // Re-granted: the record follows the new build.
    fake.set_trusted(true);
    assert_eq!(second.accessibility_status(), "granted");
    assert_eq!(
        recorded_build(&e.state_path),
        serde_json::json!({"trusted_build": "1.1:120:2"})
    );
}

#[test]
fn test_accessibility_never_granted_is_missing_for_packaged_builds() {
    let mut e = env();
    let ax = e.ax();
    *e.host.fingerprint.lock() = Some("1.0:100:1".into());
    ax.set_trusted(false);
    assert_eq!(e.plat.accessibility_status(), "missing");
    std::fs::create_dir_all(e.state_path.parent().unwrap()).unwrap();
    // a damaged record
    std::fs::write(&e.state_path, "not json").unwrap();
    let mut fresh = platform_on(&e.clock, &e.state_path, &e.host);
    fresh.ax = Some(ax.clone());
    assert_eq!(fresh.accessibility_status(), "missing");
}

const INFO_PLIST_HEAD: &str = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE plist PUBLIC \
    \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n\
    <plist version=\"1.0\">";

fn info_plist_dict(executable: &str) -> Vec<u8> {
    format!(
        "{INFO_PLIST_HEAD}<dict><key>CFBundleExecutable</key><string>{executable}</string></dict></plist>"
    )
    .into_bytes()
}

/// `Eye Tracker.app` as the spec builds it: `(app executable, eye-tracker-cli)`.
fn app_bundle(root: &Path, plist: Option<Vec<u8>>) -> (PathBuf, PathBuf) {
    let macos_dir = root.join("Eye Tracker.app").join("Contents").join("MacOS");
    std::fs::create_dir_all(&macos_dir).unwrap();
    let app = macos_dir.join("Eye Tracker");
    let cli = macos_dir.join("eye-tracker-cli");
    // written and signed separately: sizes and mtimes differ
    std::fs::write(&app, vec![0u8; 300]).unwrap();
    std::fs::write(&cli, vec![0u8; 200]).unwrap();
    let plist = plist.unwrap_or_else(|| info_plist_dict("Eye Tracker"));
    std::fs::write(macos_dir.parent().unwrap().join("Info.plist"), plist).unwrap();
    (app, cli)
}

fn mtime_ns(path: &Path) -> i128 {
    use std::os::unix::fs::MetadataExt;
    let stat = std::fs::metadata(path).unwrap();
    i128::from(stat.mtime()) * 1_000_000_000 + i128::from(stat.mtime_nsec())
}

#[test]
fn test_build_fingerprint_is_the_apps_in_both_executables() {
    let dir = tempfile::tempdir().unwrap();
    let (app, cli) = app_bundle(dir.path(), None);
    let from_app_executables = packaged_executables_for(&app);
    let from_app = build_fingerprint_of(from_app_executables.as_ref()).expect("a fingerprint");
    assert!(
        from_app.ends_with(&format!(":300:{}", mtime_ns(&app))),
        "{from_app}"
    );
    assert!(!runs_helper_executable_of(from_app_executables.as_ref()));
    let from_cli_executables = packaged_executables_for(&cli);
    assert_eq!(
        build_fingerprint_of(from_cli_executables.as_ref()),
        Some(from_app)
    );
    assert!(runs_helper_executable_of(from_cli_executables.as_ref()));
}

#[test]
fn test_build_fingerprint_without_a_usable_info_plist() {
    let array = format!("{INFO_PLIST_HEAD}<array/></plist>").into_bytes();
    let plists = [b"not a plist".to_vec(), info_plist_dict("../evil"), array];
    for plist in plists {
        let dir = tempfile::tempdir().unwrap();
        let (_app, cli) = app_bundle(dir.path(), Some(plist));
        let executables = packaged_executables_for(&cli);
        let fingerprint = build_fingerprint_of(executables.as_ref()).expect("a fingerprint");
        // its own file, as before
        assert!(fingerprint.contains(":200:"), "{fingerprint}");
        // still known by its name
        assert!(runs_helper_executable_of(executables.as_ref()));
    }
}

#[test]
fn test_cli_does_not_judge_or_record_the_apps_accessibility_grant() {
    // `eye-tracker-cli doctor` from Terminal: macOS asks about Terminal, not the app.
    let dir = tempfile::tempdir().unwrap();
    let (app, cli) = app_bundle(dir.path(), None);
    let e = env();
    *e.host.executable.lock() = Some(app);
    let fake = FakeAx::new();
    let mut gui = platform_on(&e.clock, &e.state_path, &e.host);
    gui.ax = Some(fake.clone());
    assert_eq!(gui.accessibility_status(), "granted");
    let recorded = std::fs::read_to_string(&e.state_path).unwrap();

    *e.host.executable.lock() = Some(cli);
    fake.set_trusted(false); // Terminal has no Accessibility access
    let mut doctor = platform_on(&e.clock, &e.state_path, &e.host);
    doctor.ax = Some(fake.clone());
    // not "stale", not "missing"
    assert_eq!(doctor.accessibility_status(), "unknown");
    assert_eq!(doctor.permissions()["accessibility"], None);
    assert_eq!(std::fs::read_to_string(&e.state_path).unwrap(), recorded);

    // Terminal has Accessibility access: not a grant of this (or any) app build, and not a
    // sign that the app has one (r3-platform-hotkeys-05).
    std::fs::write(&e.state_path, r#"{"trusted_build": "0.9:1:1"}"#).unwrap();
    fake.set_trusted(true);
    let mut trusted_terminal = platform_on(&e.clock, &e.state_path, &e.host);
    trusted_terminal.ax = Some(fake.clone());
    assert_eq!(trusted_terminal.accessibility_status(), "unknown");
    assert_eq!(trusted_terminal.permissions()["accessibility"], None);
    assert_eq!(
        recorded_build(&e.state_path),
        serde_json::json!({"trusted_build": "0.9:1:1"})
    );
}

#[test]
fn test_activation_without_accessibility_fails_on_sonoma() {
    // macOS 14+ ignores activation requests from an inactive (menu-bar) app.
    let mut e = env();
    let (ax, appkit) = (e.ax(), e.appkit());
    e.host.set_major_version(14);
    *e.host.fingerprint.lock() = Some("1.1:120:2".into());
    std::fs::create_dir_all(e.state_path.parent().unwrap()).unwrap();
    std::fs::write(&e.state_path, r#"{"trusted_build": "1.0:100:1"}"#).unwrap();
    ax.set_trusted(false);
    let reference = wref(PID, 0);
    let logs = capture_logs();
    assert!(!e.plat.activate_window(&reference));
    assert!(!e.plat.activate_window(&reference));
    // still asked, in case it works
    assert_eq!(appkit.activations(PID), [2, 2]);
    let warnings = logs.messages_at_least(log::Level::Warn);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("earlier version"));
    e.host.set_major_version(13);
    assert!(e.plat.activate_window(&reference));
}

#[test]
fn test_capabilities_with_frameworks() {
    let mut e = env();
    e.host.on_macos();
    e.quartz();
    e.appkit();
    e.ax();
    e.host.set_tools(&[
        ("pmset", "/usr/bin/pmset"),
        ("caffeinate", "/usr/bin/caffeinate"),
    ]);
    let caps = e.plat.capabilities();
    assert_eq!(names(&caps), names(&GenericPlatform.capabilities()));
    for key in ["lock", "display_off", "wake_display", "input_idle"] {
        assert!(caps[key], "{key}");
    }
    for key in [
        "key_idle",
        "session_locked",
        "focus",
        "cursor",
        "hotkeys",
        "panes",
    ] {
        assert!(caps[key], "{key}");
    }
    assert!(!caps["camera_in_use"]);
}

#[test]
fn test_module_contract() {
    fn assert_platform_services<T: PlatformServices>(_: &T) {}
    let e = env();
    assert_eq!(e.plat.name(), "macos");
    assert_platform_services(&e.plat);
    drop(e.dir);
}

// ------------------------------------------------------------------ the real frameworks
// These only read: idle time, the window list, permission state. Nothing is prompted, locked,
// moved or activated.

#[test]
fn test_real_quartz_answers() {
    let quartz = system::SystemQuartz;
    let idle = quartz
        .seconds_since_last_event(CG_EVENT_SOURCE_STATE_COMBINED, CG_ANY_INPUT_EVENT_TYPE)
        .expect("an answer");
    assert!(idle.is_finite() && idle >= 0.0, "{idle}");
    // Some(..) in a login session, None over ssh: either way no crash.
    let _ = quartz.session_info();
    let windows = quartz
        .window_list(
            CG_WINDOW_LIST_ON_SCREEN_ONLY | CG_WINDOW_LIST_EXCLUDE_DESKTOP,
            CG_NULL_WINDOW_ID,
        )
        .unwrap_or_default();
    for window in &windows {
        assert!(window.owner_pid >= 0, "{window:?}");
        assert!(window.alpha >= 0.0, "{window:?}");
    }
}

#[test]
fn test_real_appkit_knows_our_own_process() {
    let appkit = system::SystemAppKit;
    let own = i32::try_from(std::process::id()).unwrap();
    // A test binary is not an application: it may or may not be listed, but a listed app is
    // consistent with itself.
    if let Some(app) = appkit.running_application(own) {
        assert!(!app.terminated);
    }
    assert_eq!(appkit.running_application(i32::MAX), None);
    let _ = appkit.frontmost_pid();
}

#[test]
fn test_real_accessibility_interns_equal_elements() {
    let ax = system::SystemAccessibility::new();
    let first = ax.create_application(i32::MAX);
    let second = ax.create_application(i32::MAX);
    assert_ne!(first, 0);
    assert_eq!(first, second);
    assert_ne!(first, ax.create_application(i32::MAX - 1));
    assert_ne!(first, ax.create_system_wide());
    // An address that was never handed out is refused, not dereferenced.
    assert_eq!(ax.copy_attribute(12345, "AXRole").0, -25202);
    // No such process: asking for an attribute returns an error code at once.
    ax.set_messaging_timeout(first, 0.5);
    let (err, value) = ax.copy_attribute(first, "AXRole");
    assert_eq!(err == AX_SUCCESS, value.is_some());
    let _ = ax.is_process_trusted();
}

#[test]
fn test_real_camera_authorization_is_a_status() {
    let camera = system::SystemCamera;
    let status = camera
        .authorization_status(AV_MEDIA_TYPE_VIDEO)
        .expect("AVCaptureDevice");
    assert!((0..=3).contains(&status), "{status}");
}

#[test]
fn test_real_camera_access_block_logs_the_answer() {
    let logs = capture_logs();
    let block = &system::CAMERA_ACCESS_BLOCK;
    // SAFETY: the static block's invoke function takes the block and the answer.
    unsafe { (block.invoke)(block, objc2::runtime::Bool::YES) };
    assert!(
        logs.messages_at_least(log::Level::Info)
            .contains(&"Camera access granted: true".to_string())
    );
    // The layout of a global block: isa, flags, reserved, invoke, descriptor.
    assert_eq!(std::mem::size_of::<system::GlobalBlock>(), 32);
    assert_eq!(block.flags, 1 << 28);
}

#[test]
fn test_real_host_helpers() {
    let host = system::SystemHost;
    assert!(host.is_macos());
    assert!(host.macos_major_version() >= 12);
    assert_eq!(host.own_pid(), i32::try_from(std::process::id()).unwrap());
    assert_eq!(
        host.tool("sh").as_deref().map(|p| p.ends_with("/sh")),
        Some(true)
    );
    assert_eq!(host.tool("no-such-tool-anywhere"), None);
    assert_eq!(host.run(&["/usr/bin/true".into()], 5.0), Ok(0));
    assert_eq!(host.run(&["/usr/bin/false".into()], 5.0), Ok(1));
    assert!(host.run(&["/no/such/program".into()], 5.0).is_err());
    assert!(
        host.lock_screen_immediate("/no/such/framework").is_err(),
        "an unloadable framework is an error, not a lock"
    );
    assert!(host.spawn(&["/no/such/program".into()]).is_err());
}

#[test]
fn test_real_info_plist_formats() {
    // Calculator ships an XML Info.plist; the binary format is made from it with plutil.
    let calculator = Path::new("/System/Applications/Calculator.app/Contents/Info.plist");
    if !calculator.exists() {
        return;
    }
    assert_eq!(
        system::bundle_executable_name(calculator).as_deref(),
        Ok("Calculator")
    );
    let dir = tempfile::tempdir().unwrap();
    let binary = dir.path().join("Info.plist");
    std::fs::copy(calculator, &binary).unwrap();
    let converted = std::process::Command::new("plutil")
        .args(["-convert", "binary1"])
        .arg(&binary)
        .status();
    if converted.is_ok_and(|status| status.success()) {
        assert!(std::fs::read(&binary).unwrap().starts_with(b"bplist"));
        assert_eq!(
            system::bundle_executable_name(&binary).as_deref(),
            Ok("Calculator")
        );
    }
    assert!(system::bundle_executable_name(&dir.path().join("missing.plist")).is_err());
}

#[test]
fn test_real_platform_reads_without_side_effects() {
    // The real MacPlatform with its state file in a temporary directory: only queries.
    let dirs = app_dirs();
    let plat = MacPlatform::new();
    assert_eq!(plat.name(), "macos");
    let idle = plat.seconds_since_input().expect("an answer");
    assert!(idle >= 0.0);
    assert!(plat.seconds_since_key_input().is_some());
    let _ = plat.is_session_locked();
    let permissions = plat.permissions();
    assert!(permissions.contains_key("camera"));
    assert!(["granted", "missing", "stale", "unknown"].contains(&plat.accessibility_status()));
    assert!(plat.cursor_position_reliable());
    let caps = plat.capabilities();
    assert!(caps["lock"] && caps["hotkeys"] && caps["cursor"]);
    drop(dirs);
}
