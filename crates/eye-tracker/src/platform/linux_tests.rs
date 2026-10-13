//! Tests for the Linux platform layer (port of tests/test_platform_linux.py).
//!
//! Desktop tools, D-Bus, `/proc` and the X server are faked, so nothing here locks the screen,
//! touches displays or moves the real cursor. The environment is a private copy per platform
//! ([`Env::fake`]), so tests running in parallel never see each other's variables.
//!
//! Not ported:
//! - `test_prepare_keeps_native_wayland_without_xcb_cursor`: the libxcb-cursor probe was Qt's;
//!   winit has no such plugin (the integrator dropped it).
//! - `test_child_env_drops_bundle_library_path_when_frozen`: no PyInstaller `sys.frozen` in
//!   Rust; `LD_LIBRARY_PATH_ORIG` (AppImage) is still restored and tested.
//! - `test_doctor_shows_the_linux_lock_methods`: `diagnostics` is not ported yet.
//!
//! Adapted:
//! - The `prepare_process` tests check the winit variables (`WINIT_X11_SCALE_FACTOR`, hidden
//!   `WAYLAND_DISPLAY`) instead of `QT_ENABLE_HIGHDPI_SCALING` / `QT_QPA_PLATFORM`;
//!   `test_prepare_respects_explicit_platform_and_scaling` keeps only the scaling half (winit has
//!   no backend variable to respect).
//! - `test_tools_run_with_child_env_and_timeout` cannot see `stdin=DEVNULL` through the fake
//!   runner; `SystemTools::run` always sets it.
//! - `test_wake_x11_falls_back_to_xtest` / `test_wake_wayland_never_uses_xtest` observe the
//!   XTest motions on a fake X server without DPMS instead of monkeypatching `nudge_pointer`.
//! - Monkeypatched private methods (`_lock_confirmed`, `_logind_lock_took_effect`,
//!   `_locker_running`, `_app_process_name`) are [`TestHooks`].

use std::collections::{HashMap, HashSet, VecDeque};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use tempfile::TempDir;

use super::*;

const SESSION_ID_HINT: [&str; 6] = [
    "loginctl",
    "show-session",
    "2",
    "-p",
    "LockedHint",
    "--value",
];

fn argv(parts: &[&str]) -> Vec<String> {
    command(parts)
}

fn env_of(pairs: &[(&str, &str)]) -> EnvMap {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect()
}

/// `pytest.approx` with its default tolerances.
fn approx(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() <= (1e-6 * expected.abs()).max(1e-12)
}

macro_rules! assert_approx {
    ($actual:expr, $expected:expr) => {{
        let (actual, expected): (f64, f64) = ($actual, $expected);
        assert!(approx(actual, expected), "{actual} != approx {expected}");
    }};
}

// --------------------------------------------------------------------- clock
#[derive(Clone)]
struct TestClock(Arc<Mutex<f64>>);

impl TestClock {
    fn new() -> Self {
        Self(Arc::new(Mutex::new(1000.0)))
    }

    fn now(&self) -> f64 {
        *self.0.lock()
    }

    fn advance(&self, seconds: f64) {
        *self.0.lock() += seconds;
    }

    fn clock(&self) -> Clock {
        let me = self.clone();
        Arc::new(move || me.now())
    }

    fn sleep(&self) -> Sleep {
        let me = self.clone();
        Arc::new(move |seconds| me.advance(seconds))
    }
}

// --------------------------------------------------------------------- tools
type Response = Box<dyn FnMut(&[String]) -> Result<(i32, String), String> + Send>;

fn out(rc: i32, text: &str) -> Response {
    let text = text.to_string();
    Box::new(move |_| Ok((rc, text.clone())))
}

fn timeout() -> Response {
    Box::new(|_| Err("timed out".into()))
}

#[derive(Default)]
struct ToolsState {
    available: HashSet<String>,
    responses: Vec<(Vec<String>, Response)>,
    calls: Vec<Vec<String>>,
    kwargs: Vec<(f64, EnvMap)>,
}

/// Stands in for `which` and running programs.
#[derive(Clone, Default)]
struct FakeTools(Arc<Mutex<ToolsState>>);

impl FakeTools {
    fn install(&self, names: &[&str]) {
        self.0
            .lock()
            .available
            .extend(names.iter().map(|n| (*n).to_string()));
    }

    fn respond(&self, prefix: &[&str], response: Response) {
        let prefix = argv(prefix);
        let mut state = self.0.lock();
        state.responses.retain(|(p, _)| *p != prefix);
        state.responses.push((prefix, response));
    }

    fn calls(&self) -> Vec<Vec<String>> {
        self.0.lock().calls.clone()
    }

    fn clear_calls(&self) {
        self.0.lock().calls.clear();
    }

    fn names(&self) -> Vec<String> {
        self.calls().into_iter().map(|c| c[0].clone()).collect()
    }

    fn kwargs(&self) -> Vec<(f64, EnvMap)> {
        self.0.lock().kwargs.clone()
    }
}

impl Tools for FakeTools {
    fn which(&self, name: &str) -> Option<String> {
        self.0
            .lock()
            .available
            .contains(name)
            .then(|| name.to_string())
    }

    fn run(&self, argv: &[String], timeout: f64, env: &EnvMap) -> Result<CommandResult, String> {
        let mut state = self.0.lock();
        state.calls.push(argv.to_vec());
        state.kwargs.push((timeout, env.clone()));
        let best = state
            .responses
            .iter_mut()
            .filter(|(prefix, _)| argv.starts_with(prefix))
            .max_by_key(|(prefix, _)| prefix.len());
        let (rc, stdout) = match best {
            Some((_, response)) => response(argv)?,
            None => (1, String::new()),
        };
        Ok(CommandResult::new(rc, stdout, ""))
    }
}

// --------------------------------------------------------------------- idle, D-Bus
struct FakeXss(Option<u64>);

impl IdleSource for FakeXss {
    fn idle_ms(&self) -> Option<u64> {
        self.0
    }

    fn available(&self) -> bool {
        self.0.is_some()
    }
}

#[derive(Debug, Clone)]
struct RecordedCall {
    system: bool,
    service: String,
    path: String,
    method: String,
    args: Vec<String>,
}

type Answer = Box<dyn FnMut(&RecordedCall) -> Option<DBusReply> + Send>;

#[derive(Default)]
struct DBusState {
    answers: HashMap<String, Answer>,
    calls: Vec<RecordedCall>,
}

/// Stands in for zbus: "unavailable" (`None`) unless a test scripts an answer.
#[derive(Clone, Default)]
struct FakeDBus(Arc<Mutex<DBusState>>);

impl FakeDBus {
    fn answer(&self, method: &str, reply: DBusReply) {
        self.answer_with(method, Box::new(move |_| Some(reply.clone())));
    }

    fn answer_with(&self, method: &str, answer: Answer) {
        self.0.lock().answers.insert(method.to_string(), answer);
    }

    fn calls(&self) -> Vec<RecordedCall> {
        self.0.lock().calls.clone()
    }

    fn methods(&self) -> Vec<String> {
        self.calls().into_iter().map(|c| c.method).collect()
    }
}

impl DBus for FakeDBus {
    fn call(&self, call: &DBusCall<'_>) -> Option<DBusReply> {
        let recorded = RecordedCall {
            system: call.system,
            service: call.service.into(),
            path: call.path.into(),
            method: call.method.into(),
            args: call.args.iter().map(|a| (*a).to_string()).collect(),
        };
        let mut state = self.0.lock();
        state.calls.push(recorded.clone());
        state
            .answers
            .get_mut(call.method)
            .and_then(|answer| answer(&recorded))
    }
}

fn str_value(text: &str) -> DBusValue {
    DBusValue::Str(text.into())
}

fn no_x_server() -> XFactory {
    Box::new(|| Err(XError::Connection("no X server in tests".into())))
}

// --------------------------------------------------------------------- /proc
/// A fake `/proc` + `/dev` + `/sys` tree; fd "symlinks" are text files.
#[derive(Clone)]
struct ProcTree {
    proc: PathBuf,
    dev: PathBuf,
    sys: PathBuf,
}

impl ProcTree {
    fn new(root: &Path) -> Self {
        let tree = Self {
            proc: root.join("proc"),
            dev: root.join("dev"),
            sys: root.join("sys"),
        };
        std::fs::create_dir_all(&tree.proc).unwrap();
        std::fs::create_dir_all(&tree.dev).unwrap();
        std::fs::create_dir_all(tree.sys.join("class").join("video4linux")).unwrap();
        std::fs::create_dir_all(tree.proc.join("self")).unwrap(); // non-numeric entries are ignored
        tree
    }

    fn device(&self, name: &str, physical: bool) {
        std::fs::write(self.dev.join(name), "").unwrap();
        let entry = self.sys.join("class").join("video4linux").join(name);
        std::fs::create_dir_all(&entry).unwrap();
        if physical {
            std::fs::create_dir_all(entry.join("device")).unwrap();
        }
    }

    fn process(&self, pid: u32, targets: &[&str], comm: &str) {
        self.process_full(pid, targets, comm, "", None);
    }

    fn process_full(
        &self,
        pid: u32,
        targets: &[&str],
        comm: &str,
        maps: &str,
        loginuid: Option<u64>,
    ) {
        let base = self.proc.join(pid.to_string());
        std::fs::create_dir_all(base.join("fd")).unwrap();
        for (fd, target) in targets.iter().enumerate() {
            std::fs::write(base.join("fd").join(fd.to_string()), target).unwrap();
        }
        std::fs::write(base.join("comm"), format!("{comm}\n")).unwrap();
        std::fs::write(base.join("maps"), maps).unwrap();
        if let Some(loginuid) = loginuid {
            std::fs::write(base.join("loginuid"), loginuid.to_string()).unwrap();
        }
    }

    /// This process's audit login UID (`/proc/self/loginuid`).
    fn login(&self, loginuid: u64) {
        std::fs::write(
            self.proc.join("self").join("loginuid"),
            loginuid.to_string(),
        )
        .unwrap();
    }

    fn remove(&self, pid: u32) {
        std::fs::remove_dir_all(self.proc.join(pid.to_string())).unwrap();
    }

    fn owner(&self, pid: u32) -> u32 {
        std::fs::metadata(self.proc.join(pid.to_string()))
            .unwrap()
            .uid()
    }
}

fn read_fake_link(path: &Path) -> io::Result<String> {
    std::fs::read_to_string(path)
}

fn set_readlink(
    plat: &LinuxPlatform,
    readlink: impl Fn(&Path) -> io::Result<String> + Send + Sync + 'static,
) {
    *plat.fs.readlink.write() = Arc::new(readlink);
}

fn set_uid(plat: &LinuxPlatform, uid: Option<u32>) {
    *plat.fs.getuid.write() = Arc::new(move || uid);
}

// --------------------------------------------------------------------- platform
/// A platform whose tools, D-Bus, X server and environment are fakes.
fn build(
    options: LinuxOptions,
    tools: &FakeTools,
    dbus: &FakeDBus,
    clock: &TestClock,
) -> LinuxPlatform {
    let mut plat = LinuxPlatform::with_options(options);
    plat.env = Env::fake();
    plat.tools = Box::new(tools.clone());
    plat.ydotool_default_sockets = &[];
    plat.xss = Box::new(FakeXss(None));
    plat.ewmh = Arc::new(EwmhClient::new(Some(no_x_server()), clock.clock(), None));
    plat.dbus = Box::new(dbus.clone());
    set_readlink(&plat, read_fake_link);
    plat
}

struct Fx {
    tmp: TempDir,
    clock: TestClock,
    tools: FakeTools,
    dbus: FakeDBus,
    plat: LinuxPlatform,
}

/// Everything on the calling thread; no real X server, D-Bus or /proc is touched.
fn fx() -> Fx {
    let tmp = tempfile::tempdir().unwrap();
    let clock = TestClock::new();
    let tools = FakeTools::default();
    let dbus = FakeDBus::default();
    let plat = build(options_for(tmp.path(), &clock), &tools, &dbus, &clock);
    Fx {
        tmp,
        clock,
        tools,
        dbus,
        plat,
    }
}

fn options_for(root: &Path, clock: &TestClock) -> LinuxOptions {
    LinuxOptions {
        proc_root: root.join("proc"),
        dev_root: root.join("dev"),
        sys_root: root.join("sys"),
        clock: clock.clock(),
        sleep: clock.sleep(),
        background_threads: false,
        ..LinuxOptions::default()
    }
}

impl Fx {
    fn setenv(&self, key: &str, value: &str) {
        self.plat.env.set(key, value);
    }

    fn delenv(&self, key: &str) {
        self.plat.env.remove(key);
    }

    fn x11(&self, desktop: Option<&str>) {
        self.setenv("XDG_SESSION_TYPE", "x11");
        self.setenv("DISPLAY", ":0");
        if let Some(desktop) = desktop {
            self.setenv("XDG_CURRENT_DESKTOP", desktop);
        }
    }

    fn wayland(&self, desktop: Option<&str>, display: bool) {
        self.setenv("XDG_SESSION_TYPE", "wayland");
        self.setenv("WAYLAND_DISPLAY", "wayland-0");
        if display {
            self.setenv("DISPLAY", ":0");
        }
        if let Some(desktop) = desktop {
            self.setenv("XDG_CURRENT_DESKTOP", desktop);
        }
    }

    fn proc_tree(&self) -> ProcTree {
        ProcTree::new(self.tmp.path())
    }

    /// Another platform on the same tree, built with other options.
    fn platform(&self, options: LinuxOptions) -> LinuxPlatform {
        build(options, &self.tools, &FakeDBus::default(), &self.clock)
    }

    /// ydotool 1.x installed and ydotoold listening.
    fn ydotool_ready(&self) {
        let socket = self.tmp.path().join("ydotool.sock");
        std::fs::write(&socket, "").unwrap();
        self.setenv("YDOTOOL_SOCKET", socket.to_str().unwrap());
        self.tools.install(&["ydotool"]);
        self.tools
            .respond(&["ydotool", "mousemove", "--help"], out(0, YDOTOOL_1_HELP));
        self.tools
            .respond(&["ydotool", "mousemove", "--absolute"], out(0, ""));
    }
}

// ------------------------------------------------------------------ session type
#[test]
fn test_wayland_detection() {
    let cases: [(&[(&str, &str)], bool); 7] = [
        (&[], false),
        (&[("XDG_SESSION_TYPE", "x11")], false),
        (&[("XDG_SESSION_TYPE", "wayland")], true),
        (&[("XDG_SESSION_TYPE", "Wayland")], true),
        (&[("WAYLAND_DISPLAY", "wayland-0")], true),
        (
            &[
                ("XDG_SESSION_TYPE", "x11"),
                ("WAYLAND_DISPLAY", "wayland-1"),
            ],
            true,
        ),
        (&[("XDG_SESSION_TYPE", "tty")], false),
    ];
    for (env, expected) in cases {
        let fx = fx();
        for (key, value) in env {
            fx.setenv(key, value);
        }
        assert_eq!(fx.plat.is_wayland(), expected, "{env:?}");
    }
}

#[test]
fn test_desktop_detection() {
    assert!(is_gnome_env(&env_of(&[(
        "XDG_CURRENT_DESKTOP",
        "ubuntu:GNOME"
    )])));
    assert!(is_gnome_env(&env_of(&[(
        "XDG_SESSION_DESKTOP",
        "gnome-classic"
    )])));
    assert!(!is_gnome_env(&env_of(&[("XDG_CURRENT_DESKTOP", "KDE")])));
    assert!(is_kde_env(&env_of(&[("XDG_CURRENT_DESKTOP", "KDE")])));
    assert!(is_kde_env(&env_of(&[("DESKTOP_SESSION", "plasmawayland")])));
    assert!(is_kde_env(&env_of(&[("KDE_FULL_SESSION", "true")])));
    assert!(!is_kde_env(&env_of(&[("XDG_CURRENT_DESKTOP", "XFCE")])));
}

#[test]
fn test_locked_hint_is_authoritative_only_for_gnome_shell_and_kde() {
    let cases: [(&[(&str, &str)], bool); 14] = [
        (&[("XDG_CURRENT_DESKTOP", "GNOME")], true),
        (&[("XDG_CURRENT_DESKTOP", "ubuntu:GNOME")], true),
        (&[("XDG_CURRENT_DESKTOP", "pop:GNOME")], true),
        (&[("XDG_CURRENT_DESKTOP", "GNOME-Classic:GNOME")], true),
        (&[("XDG_CURRENT_DESKTOP", "GNOME-Flashback:GNOME")], true),
        (&[("XDG_SESSION_DESKTOP", "gnome")], true),
        (&[("XDG_CURRENT_DESKTOP", "KDE")], true),
        (&[("KDE_FULL_SESSION", "true")], true),
        // r3-platform-hotkeys-01: GNOME listed as a fallback, budgie-screensaver locks.
        (&[("XDG_CURRENT_DESKTOP", "Budgie:GNOME")], false),
        (&[("XDG_SESSION_DESKTOP", "budgie-desktop")], false),
        (&[("XDG_CURRENT_DESKTOP", "X-Cinnamon")], false),
        (&[("XDG_CURRENT_DESKTOP", "Unity:Unity7:ubuntu")], false),
        (
            &[
                ("XDG_CURRENT_DESKTOP", "GNOME"),
                ("SWAYSOCK", "/run/sway.sock"),
            ],
            false,
        ),
        (&[], false),
    ];
    for (env, expected) in cases {
        assert_eq!(locked_hint_authoritative(&env_of(env)), expected, "{env:?}");
    }
}

// ------------------------------------------------------------------ prepare_process
#[test]
fn test_prepare_forces_xcb_on_wayland_with_xwayland() {
    let fx = fx();
    fx.wayland(Some("GNOME"), true);
    fx.plat.prepare_process();
    // winit picks Wayland whenever WAYLAND_DISPLAY is set: hidden, so it uses XWayland.
    assert_eq!(fx.plat.env.get("WAYLAND_DISPLAY"), None);
    assert_eq!(
        fx.plat.env.get("WINIT_X11_SCALE_FACTOR").as_deref(),
        Some("1")
    );
    assert!(fx.plat.is_wayland()); // still a Wayland session
}

#[test]
fn test_prepare_keeps_native_wayland_without_xwayland() {
    let fx = fx();
    fx.wayland(Some("GNOME"), false);
    fx.plat.prepare_process();
    assert_eq!(
        fx.plat.env.get("WAYLAND_DISPLAY").as_deref(),
        Some("wayland-0")
    );
    assert_eq!(
        fx.plat.env.get("WINIT_X11_SCALE_FACTOR").as_deref(),
        Some("1")
    );
}

#[test]
fn test_prepare_respects_explicit_platform_and_scaling() {
    let fx = fx();
    fx.wayland(Some("GNOME"), true);
    fx.setenv("WINIT_X11_SCALE_FACTOR", "2");
    fx.plat.prepare_process();
    assert_eq!(
        fx.plat.env.get("WINIT_X11_SCALE_FACTOR").as_deref(),
        Some("2")
    );
}

#[test]
fn test_prepare_on_x11_does_not_touch_platform() {
    let fx = fx();
    fx.x11(None);
    fx.plat.prepare_process();
    assert_eq!(fx.plat.env.get("WAYLAND_DISPLAY"), None);
    assert_eq!(fx.plat.env.get("DISPLAY").as_deref(), Some(":0"));
    assert_eq!(
        fx.plat.env.get("WINIT_X11_SCALE_FACTOR").as_deref(),
        Some("1")
    );
}

#[test]
fn test_child_env_undoes_our_overrides() {
    let fx = fx();
    fx.wayland(Some("KDE"), true);
    fx.plat.prepare_process();
    let env = fx.plat.child_env();
    assert_eq!(
        env.get("WAYLAND_DISPLAY").map(String::as_str),
        Some("wayland-0")
    );
    assert!(!env.contains_key("WINIT_X11_SCALE_FACTOR"));
    assert_eq!(fx.plat.env.get("WAYLAND_DISPLAY"), None); // our own process keeps them
}

#[test]
fn test_child_env_restores_library_path() {
    let fx = fx();
    fx.setenv("LD_LIBRARY_PATH", "/tmp/.mount_EyeTr123/usr/lib");
    fx.setenv("LD_LIBRARY_PATH_ORIG", "/opt/lib");
    let env = fx.plat.child_env();
    assert_eq!(env["LD_LIBRARY_PATH"], "/opt/lib");
    assert!(!env.contains_key("LD_LIBRARY_PATH_ORIG"));
}

#[test]
fn test_tools_run_with_child_env_and_timeout() {
    let fx = fx();
    fx.wayland(Some("KDE"), true);
    fx.plat.prepare_process();
    fx.tools.install(&["kscreen-doctor"]);
    fx.tools.respond(&["kscreen-doctor"], out(0, ""));
    assert!(fx.plat.display_off());
    let (timeout, env) = &fx.tools.kwargs()[0];
    assert!(!env.contains_key("WINIT_X11_SCALE_FACTOR"));
    assert_eq!(
        env.get("WAYLAND_DISPLAY").map(String::as_str),
        Some("wayland-0")
    );
    assert_approx!(*timeout, 5.0);
}

// ------------------------------------------------------------------ lock screen
const ALL_LOCKERS: [&str; 10] = [
    "loginctl",
    "xdg-screensaver",
    "gnome-screensaver-command",
    "dm-tool",
    "xflock4",
    "qdbus",
    "qdbus6",
    "qdbus-qt6",
    "qdbus-qt5",
    "gdbus",
];

#[test]
fn test_lock_tries_every_locker_in_order() {
    let fx = fx();
    fx.setenv("XDG_SESSION_ID", "c2");
    fx.tools.install(&ALL_LOCKERS);
    assert!(!fx.plat.lock_screen());
    assert_eq!(fx.tools.names(), ALL_LOCKERS.map(String::from).to_vec());
    let calls = fx.tools.calls();
    assert_eq!(calls[0], argv(&["loginctl", "lock-session", "c2"]));
    assert_eq!(
        calls.last().unwrap().last().unwrap(),
        "org.freedesktop.ScreenSaver.Lock"
    );
}

#[test]
fn test_lock_stops_at_first_success() {
    let fx = fx();
    fx.setenv("XDG_SESSION_ID", "7");
    fx.tools.install(&ALL_LOCKERS);
    fx.tools.respond(&["loginctl"], out(1, ""));
    fx.tools.respond(&["xdg-screensaver"], out(0, ""));
    assert!(fx.plat.lock_screen());
    assert_eq!(
        fx.tools.calls(),
        vec![
            argv(&["loginctl", "lock-session", "7"]),
            argv(&["xdg-screensaver", "lock"])
        ]
    );
}

#[test]
fn test_lock_skips_missing_tools_and_timeouts() {
    let fx = fx();
    fx.setenv("XDG_SESSION_ID", "7");
    fx.tools.install(&["loginctl", "qdbus6"]);
    fx.tools.respond(&["loginctl"], timeout());
    fx.tools.respond(&["qdbus6"], out(0, ""));
    assert!(fx.plat.lock_screen());
    assert_eq!(
        fx.tools.calls(),
        vec![
            argv(&["loginctl", "lock-session", "7"]),
            argv(&[
                "qdbus6",
                "org.freedesktop.ScreenSaver",
                "/ScreenSaver",
                "Lock"
            ]),
        ]
    );
}

#[test]
fn test_lock_without_any_locker() {
    let fx = fx();
    assert!(!fx.plat.lock_screen());
    assert!(fx.tools.calls().is_empty());
}

/// Regression (r2-docs-13): doctor showed "lock yes" whenever loginctl was installed.
#[test]
fn test_lock_methods_lists_installed_tools_in_the_order_they_are_tried() {
    let fx = fx();
    fx.setenv("XDG_SESSION_ID", "2");
    fx.setenv("XDG_CURRENT_DESKTOP", "GNOME");
    fx.tools
        .install(&["gdbus", "dm-tool", "loginctl", "qdbus6"]);
    assert_eq!(
        fx.plat.lock_methods(),
        [
            "loginctl lock-session",
            "dm-tool lock",
            "qdbus6 org.freedesktop.ScreenSaver.Lock",
            "gdbus org.freedesktop.ScreenSaver.Lock",
        ]
    );
    assert!(fx.tools.calls().is_empty()); // listing runs nothing, not even a session lookup
    assert!(
        fx.plat
            .lock_methods()
            .iter()
            .all(|method| method.is_ascii())
    );
}

#[test]
fn test_lock_methods_flags_loginctl_where_nothing_may_listen() {
    let cases: [&[(&str, &str)]; 2] = [
        &[("XDG_CURRENT_DESKTOP", "i3")],
        &[
            ("XDG_CURRENT_DESKTOP", "GNOME"),
            ("SWAYSOCK", "/run/user/1000/sway-ipc.sock"),
        ],
    ];
    for env in cases {
        let fx = fx();
        for (key, value) in env {
            fx.setenv(key, value);
        }
        fx.tools.install(&["loginctl", "xdg-screensaver"]);
        let methods = fx.plat.lock_methods();
        let [loginctl, xdg] = methods.as_slice() else {
            panic!("{methods:?}");
        };
        assert!(
            loginctl.starts_with("loginctl lock-session ("),
            "{loginctl}"
        );
        assert!(loginctl.contains("listens to logind"));
        assert_eq!(xdg, "xdg-screensaver lock");
    }
}

#[test]
fn test_lock_methods_without_any_tool() {
    assert!(fx().plat.lock_methods().is_empty());
}

#[test]
fn test_lock_resolves_display_session_without_session_id() {
    let fx = fx();
    fx.setenv("XDG_CURRENT_DESKTOP", "GNOME");
    set_uid(&fx.plat, Some(1000));
    fx.tools.install(&["loginctl"]);
    fx.tools.respond(&["loginctl", "show-user"], out(0, "3\n"));
    fx.tools.respond(&["loginctl", "lock-session"], out(0, ""));
    fx.tools
        .respond(&["loginctl", "show-session"], out(0, "yes\n"));
    assert!(fx.plat.lock_screen());
    assert_eq!(
        fx.tools.calls(),
        vec![
            argv(&["loginctl", "show-user", "1000", "-p", "Display", "--value"]),
            argv(&["loginctl", "lock-session", "3"]),
            // GNOME confirmed it
            argv(&[
                "loginctl",
                "show-session",
                "3",
                "-p",
                "LockedHint",
                "--value"
            ]),
        ]
    );
}

#[test]
fn test_lock_falls_back_to_callers_session() {
    let fx = fx();
    fx.setenv("XDG_CURRENT_DESKTOP", "KDE");
    set_uid(&fx.plat, Some(1000));
    fx.tools.install(&["loginctl"]);
    fx.tools.respond(&["loginctl", "show-user"], out(0, "\n"));
    fx.tools.respond(&["loginctl", "lock-session"], out(0, ""));
    assert!(fx.plat.lock_screen());
    assert_eq!(
        fx.tools.calls().last().unwrap(),
        &argv(&["loginctl", "lock-session"])
    );
}

#[test]
fn test_lock_trusts_loginctl_on_desktops_that_handle_it() {
    for desktop in ["X-Cinnamon", "MATE", "XFCE", "Budgie:GNOME"] {
        // Their lock screens listen to logind but do not all report LockedHint.
        let fx = fx();
        fx.setenv("XDG_SESSION_ID", "2");
        fx.setenv("XDG_CURRENT_DESKTOP", desktop);
        fx.tools.install(&ALL_LOCKERS);
        fx.tools.respond(&["loginctl", "lock-session"], out(0, ""));
        // budgie-screensaver locks but never sets the hint: it must not be asked.
        fx.tools
            .respond(&["loginctl", "show-session"], out(0, "no\n"));
        assert!(fx.plat.lock_screen(), "{desktop}");
        assert_eq!(
            fx.tools.calls(),
            vec![argv(&["loginctl", "lock-session", "2"])],
            "{desktop}"
        );
    }
}

/// r3-platform-hotkeys-01: a greeter (dm-tool lock) on top of a lock screen that did not show
/// in time locks the user twice, or again right after they unlocked.
#[test]
fn test_lock_policy_refusal_on_gnome_is_not_overridden_by_the_display_manager() {
    let fx = fx();
    fx.setenv("XDG_SESSION_ID", "2");
    fx.setenv("XDG_CURRENT_DESKTOP", "ubuntu:GNOME");
    fx.tools
        .install(&["loginctl", "xdg-screensaver", "dm-tool"]);
    fx.tools.respond(&["loginctl", "lock-session"], out(0, ""));
    fx.tools.respond(&["xdg-screensaver"], out(0, ""));
    fx.tools.respond(&["dm-tool"], out(0, ""));
    fx.tools
        .respond(&["loginctl", "show-session"], out(0, "no\n"));
    assert!(!fx.plat.lock_screen()); // reported, so the guard shows its curtain
    assert!(!fx.tools.calls().contains(&argv(&["dm-tool", "lock"])));
}

#[test]
fn test_lock_on_gnome_and_kde_is_confirmed_by_locked_hint() {
    for desktop in ["GNOME", "KDE", "pop:GNOME"] {
        let fx = fx();
        fx.setenv("XDG_SESSION_ID", "2");
        fx.setenv("XDG_CURRENT_DESKTOP", desktop);
        fx.tools.install(&ALL_LOCKERS);
        fx.tools.respond(&["loginctl", "lock-session"], out(0, ""));
        let mut hints = VecDeque::from(["no\n"]);
        fx.tools.respond(
            &["loginctl", "show-session"],
            Box::new(move |_| Ok((0, hints.pop_front().unwrap_or("yes").to_string()))),
        );
        assert!(fx.plat.lock_screen(), "{desktop}");
        assert_eq!(
            fx.tools.calls(),
            vec![
                argv(&["loginctl", "lock-session", "2"]),
                argv(&SESSION_ID_HINT),
                argv(&SESSION_ID_HINT)
            ],
            "{desktop}"
        );
    }
}

/// disable-lock-screen: gnome-shell ignores every request, yet the tools exit with 0.
#[test]
fn test_lock_refused_by_gnome_policy_is_reported() {
    let fx = fx();
    fx.setenv("XDG_SESSION_ID", "2");
    fx.setenv("XDG_CURRENT_DESKTOP", "ubuntu:GNOME");
    fx.tools.install(&["loginctl", "xdg-screensaver", "gdbus"]);
    fx.tools.respond(&["loginctl", "lock-session"], out(0, ""));
    fx.tools.respond(&["xdg-screensaver"], out(0, ""));
    fx.tools.respond(&["gdbus"], out(0, "()"));
    fx.tools
        .respond(&["loginctl", "show-session"], out(0, "no\n"));
    let started = fx.clock.now();
    // False: the controller warns, and the shoulder guard falls back to its curtain.
    assert!(!fx.plat.lock_screen());
    let locks: Vec<String> = fx
        .tools
        .calls()
        .into_iter()
        .filter(|call| call[..2] != argv(&["loginctl", "show-session"]))
        .map(|call| call[0].clone())
        .collect();
    assert_eq!(locks, ["loginctl", "xdg-screensaver", "gdbus"]);
    // Each claimed success was checked for the bounded confirmation time.
    assert_approx!(fx.clock.now() - started, 3.0 * LOCK_CONFIRM_S);
}

#[test]
fn test_lock_on_gnome_confirmed_after_another_locker() {
    let fx = fx();
    fx.setenv("XDG_SESSION_ID", "2");
    fx.setenv("XDG_CURRENT_DESKTOP", "GNOME");
    fx.tools
        .install(&["loginctl", "xdg-screensaver", "dm-tool"]);
    fx.tools.respond(&["loginctl", "lock-session"], out(0, ""));
    let locked = Arc::new(AtomicBool::new(false));
    let flag = locked.clone();
    fx.tools.respond(
        &["xdg-screensaver"],
        Box::new(move |_| {
            flag.store(true, Ordering::SeqCst);
            Ok((0, String::new()))
        }),
    );
    fx.tools.respond(
        &["loginctl", "show-session"],
        Box::new(move |_| {
            Ok((
                0,
                if locked.load(Ordering::SeqCst) {
                    "yes"
                } else {
                    "no"
                }
                .into(),
            ))
        }),
    );
    assert!(fx.plat.lock_screen());
    assert!(!fx.tools.calls().contains(&argv(&["dm-tool", "lock"]))); // no second lock screen on top
}

#[test]
fn test_lock_on_gnome_seen_late_is_not_a_failure() {
    let mut fx = fx();
    fx.setenv("XDG_SESSION_ID", "2");
    fx.setenv("XDG_CURRENT_DESKTOP", "KDE");
    fx.tools.install(&["loginctl"]);
    fx.tools.respond(&["loginctl", "lock-session"], out(0, ""));
    let hint = Arc::new(Mutex::new("no"));
    let shown = hint.clone();
    fx.tools.respond(
        &["loginctl", "show-session"],
        Box::new(move |_| Ok((0, shown.lock().to_string()))),
    );
    fx.plat.hooks.lock_confirmed = Some(Box::new(move |_session, _trust_unknown| {
        *hint.lock() = "yes"; // the lock screen comes up just after the wait
        false
    }));
    assert!(fx.plat.lock_screen());
}

#[test]
fn test_lock_on_gnome_without_a_readable_hint_trusts_the_desktop() {
    let fx = fx();
    fx.setenv("XDG_SESSION_ID", "2");
    fx.setenv("XDG_CURRENT_DESKTOP", "GNOME");
    fx.tools.install(&["loginctl", "xdg-screensaver"]);
    fx.tools.respond(&["loginctl", "lock-session"], out(0, ""));
    fx.tools.respond(
        &["loginctl", "show-session"],
        out(1, "Failed to get session\n"),
    );
    assert!(fx.plat.lock_screen());
    assert!(
        !fx.tools
            .calls()
            .contains(&argv(&["xdg-screensaver", "lock"]))
    );
}

/// sway with XDG_CURRENT_DESKTOP=GNOME (for portals) and no swayidle lock hook.
#[test]
fn test_compositor_posing_as_gnome_is_not_trusted() {
    for socket_var in [
        "SWAYSOCK",
        "I3SOCK",
        "HYPRLAND_INSTANCE_SIGNATURE",
        "NIRI_SOCKET",
    ] {
        let fx = fx();
        let tree = fx.proc_tree();
        fx.setenv("XDG_SESSION_ID", "2");
        fx.wayland(Some("GNOME"), true);
        fx.setenv(socket_var, "/run/user/1000/wm.sock");
        let env = fx.plat.env.vars();
        assert!(!desktop_handles_logind_lock(&env));
        assert!(!locked_hint_authoritative(&env));
        fx.tools.install(&["loginctl"]);
        fx.tools.respond(&["loginctl", "lock-session"], out(0, ""));
        fx.tools
            .respond(&["loginctl", "show-session"], out(0, "no\n"));
        assert!(!fx.plat.lock_screen(), "{socket_var}"); // nothing locked: say so
        // A locker started by hand is noticed, although the hint says "unlocked".
        tree.process(200, &[], "swaylock");
        *fx.plat.locked_cache.lock() = None;
        assert_eq!(fx.plat.is_session_locked(), Some(true), "{socket_var}");
    }
}

#[test]
fn test_lock_unconfirmed_loginctl_tries_other_lockers() {
    // i3 without xss-lock: logind emits Lock, nobody listens, loginctl still exits 0.
    let fx = fx();
    let tree = fx.proc_tree();
    fx.setenv("XDG_SESSION_ID", "2");
    fx.setenv("XDG_CURRENT_DESKTOP", "i3");
    tree.process(100, &[], "i3");
    fx.tools.install(&["loginctl", "dm-tool"]);
    fx.tools.respond(&["loginctl", "lock-session"], out(0, ""));
    fx.tools
        .respond(&["loginctl", "show-session"], out(0, "no\n"));
    fx.tools.respond(&["dm-tool"], out(0, ""));
    let started = fx.clock.now();
    assert!(fx.plat.lock_screen());
    let calls = fx.tools.calls();
    assert_eq!(calls[0], argv(&["loginctl", "lock-session", "2"]));
    assert_eq!(calls.last().unwrap(), &argv(&["dm-tool", "lock"]));
    assert!(calls.contains(&argv(&SESSION_ID_HINT)));
    assert_approx!(fx.clock.now() - started, LOCK_CONFIRM_S); // bounded wait
}

#[test]
fn test_lock_unconfirmed_loginctl_alone_reports_failure() {
    let fx = fx();
    let _tree = fx.proc_tree();
    fx.setenv("XDG_SESSION_ID", "2");
    fx.setenv("XDG_CURRENT_DESKTOP", "sway");
    fx.tools.install(&["loginctl"]);
    fx.tools.respond(&["loginctl", "lock-session"], out(0, ""));
    fx.tools
        .respond(&["loginctl", "show-session"], out(0, "no\n"));
    // False lets the controller warn, and the shoulder guard fall back to its curtain.
    assert!(!fx.plat.lock_screen());
}

#[test]
fn test_lock_confirmed_by_locker_process() {
    let fx = fx();
    let tree = fx.proc_tree();
    fx.tools.install(&["loginctl", "dm-tool"]);
    fx.tools.respond(&["loginctl", "lock-session"], out(0, ""));
    // xss-lock + i3lock never set it
    fx.tools
        .respond(&["loginctl", "show-session"], out(0, "no\n"));
    let sleeps = Arc::new(Mutex::new(Vec::new()));
    let (recorded, clock, locker_tree) = (sleeps.clone(), fx.clock.clone(), tree.clone());
    let platform = fx.platform(LinuxOptions {
        sleep: Arc::new(move |seconds| {
            recorded.lock().push(seconds);
            clock.advance(seconds);
            locker_tree.process(4321, &[], "i3lock"); // xss-lock started the locker
        }),
        ..options_for(fx.tmp.path(), &fx.clock)
    });
    platform.env.set("XDG_SESSION_ID", "2");
    platform.env.set("XDG_CURRENT_DESKTOP", "i3");
    assert!(platform.lock_screen());
    assert_eq!(sleeps.lock().len(), 1);
    assert!(!fx.tools.calls().contains(&argv(&["dm-tool", "lock"]))); // no second lock screen on top
}

#[test]
fn test_lock_slow_locker_is_not_stacked_with_another() {
    let mut fx = fx();
    let tree = fx.proc_tree();
    fx.setenv("XDG_SESSION_ID", "2");
    fx.setenv("XDG_CURRENT_DESKTOP", "i3");
    fx.tools
        .install(&["loginctl", "xdg-screensaver", "dm-tool"]);
    fx.tools.respond(&["loginctl", "lock-session"], out(0, ""));
    let waited = Arc::new(Mutex::new(Vec::new()));
    let recorded = waited.clone();
    fx.plat.hooks.logind_lock_took_effect = Some(Box::new(move |session| {
        recorded.lock().push(session.map(String::from));
        tree.process(4321, &[], "i3lock"); // shows up just after the wait
        false
    }));
    assert!(fx.plat.lock_screen());
    assert_eq!(*waited.lock(), [Some("2".to_string())]);
    // no second lock screen
    assert_eq!(
        fx.tools.calls(),
        vec![argv(&["loginctl", "lock-session", "2"])]
    );
}

#[test]
fn test_lock_confirmed_by_locked_hint() {
    let fx = fx();
    fx.setenv("XDG_SESSION_ID", "2");
    fx.setenv("XDG_CURRENT_DESKTOP", "LXQt");
    fx.tools.install(&["loginctl", "dm-tool"]);
    fx.tools.respond(&["loginctl", "lock-session"], out(0, ""));
    let mut hints = VecDeque::from(["no\n", "no\n", "yes\n"]);
    fx.tools.respond(
        &["loginctl", "show-session"],
        Box::new(move |_| Ok((0, hints.pop_front().unwrap_or("yes").to_string()))),
    );
    assert!(fx.plat.lock_screen());
    assert!(!fx.tools.calls().contains(&argv(&["dm-tool", "lock"])));
}

// ------------------------------------------------------------------ display power
const XSET_Q_ENABLED: &str =
    "DPMS (Energy Star):\n  Standby: 600    Suspend: 600    Off: 600\n  DPMS is Enabled\n";

fn xset_q_disabled() -> String {
    XSET_Q_ENABLED.replace("Enabled", "Disabled")
}

#[test]
fn test_display_off_x11_uses_xset_without_python_xlib() {
    let fx = fx();
    fx.x11(Some("XFCE"));
    fx.tools.install(&["xset", "kscreen-doctor", "busctl"]);
    fx.tools.respond(&["xset"], out(0, ""));
    fx.tools.respond(&["xset", "q"], out(0, XSET_Q_ENABLED));
    assert!(fx.plat.display_off());
    assert!(fx.plat.wake_display());
    assert_eq!(
        fx.tools.calls(),
        vec![
            argv(&["xset", "q"]),
            argv(&["xset", "dpms", "force", "off"]),
            argv(&["xset", "q"]),
            argv(&["xset", "dpms", "force", "on"]),
        ]
    );
}

#[test]
fn test_xset_restores_disabled_dpms_after_waking() {
    // "xset -dpms" (kiosk, presentation): "dpms force" would enable DPMS for good.
    let fx = fx();
    fx.x11(Some("i3"));
    fx.tools.install(&["xset"]);
    fx.tools.respond(&["xset"], out(0, ""));
    fx.tools.respond(&["xset", "q"], out(0, &xset_q_disabled()));
    assert!(fx.plat.display_off());
    assert_eq!(
        fx.tools.calls().last().unwrap(),
        &argv(&["xset", "dpms", "force", "off"])
    );
    fx.tools.respond(&["xset", "q"], out(0, XSET_Q_ENABLED)); // "force" enabled it meanwhile
    assert!(fx.plat.wake_display());
    let calls = fx.tools.calls();
    assert_eq!(
        calls[calls.len() - 2..],
        [
            argv(&["xset", "dpms", "force", "on"]),
            argv(&["xset", "-dpms"])
        ]
    );
}

#[test]
fn test_xset_without_dpms_reports_failure() {
    for output in [
        "Server does not have the DPMS Extension\n",
        "Display is not capable of DPMS\n",
    ] {
        // Xvnc, xrdp, many VMs: xset complains on stderr and still exits with 0.
        let fx = fx();
        fx.x11(Some("XFCE"));
        fx.tools.install(&["xset"]);
        fx.tools.respond(&["xset"], out(0, ""));
        fx.tools.respond(&["xset", "q"], out(0, output));
        assert!(!fx.plat.display_off(), "{output}");
        assert_eq!(fx.tools.calls(), vec![argv(&["xset", "q"])]);
    }
}

#[test]
fn test_parse_xset_dpms() {
    let disabled = xset_q_disabled();
    let cases = [
        (XSET_Q_ENABLED, Some((true, true))),
        (disabled.as_str(), Some((true, false))),
        (
            "Server does not have the DPMS Extension",
            Some((false, false)),
        ),
        ("Keyboard Control:\n  auto repeat: on", None),
        ("", None),
    ];
    for (text, expected) in cases {
        assert_eq!(parse_xset_dpms(Some(text)), expected, "{text}");
    }
}

#[test]
fn test_x11_dpms_in_process_restores_disabled_state() {
    let (fx, xdisplay) = x11_fx();
    fx.tools.install(&["xset"]);
    xdisplay.0.lock().dpms_enabled = false;
    assert!(fx.plat.display_off());
    assert_eq!(xdisplay.0.lock().dpms_log, ["enable", "force 3"]);
    assert!(fx.plat.wake_display());
    assert_eq!(
        xdisplay.0.lock().dpms_log,
        ["enable", "force 3", "force 0", "disable"]
    );
    assert!(!xdisplay.0.lock().dpms_enabled);
    assert!(fx.tools.calls().is_empty()); // the X server did it all
}

#[test]
fn test_x11_dpms_in_process_keeps_enabled_state() {
    let (fx, xdisplay) = x11_fx();
    assert!(fx.plat.display_off());
    assert!(fx.plat.wake_display());
    assert_eq!(xdisplay.0.lock().dpms_log, ["force 3", "force 0"]);
    assert!(xdisplay.0.lock().dpms_enabled);
}

#[test]
fn test_x11_without_dpms_reports_failure() {
    let (fx, xdisplay) = x11_fx();
    fx.tools.install(&["xset"]);
    fx.tools.respond(&["xset"], out(0, ""));
    xdisplay.0.lock().extensions.remove("DPMS");
    assert!(!fx.plat.display_off());
    xdisplay.0.lock().extensions.insert("DPMS".into());
    xdisplay.0.lock().dpms_capable = false;
    assert!(!fx.plat.display_off());
    assert!(xdisplay.0.lock().dpms_log.is_empty());
    assert!(fx.tools.calls().is_empty()); // xset would only pretend
    assert!(!fx.plat.capabilities()["display_off"]);
}

#[test]
fn test_x11_dpms_disabled_again_when_monitors_wake_by_themselves() {
    // wake_on_return off: the user's input wakes the monitors, not wake_display().
    let (fx, xdisplay) = x11_fx();
    xdisplay.0.lock().dpms_enabled = false;
    assert!(fx.plat.display_off());
    assert!(!fx.plat.restore_dpms_if_awake()); // still off: keep waiting
    xdisplay.0.lock().power_level = 0;
    assert!(fx.plat.restore_dpms_if_awake());
    assert!(!xdisplay.0.lock().dpms_enabled);
    assert!(fx.plat.restore_dpms_if_awake()); // nothing left to do
    assert_eq!(xdisplay.0.lock().dpms_log, ["enable", "force 3", "disable"]);
}

#[test]
fn test_display_off_kde_wayland_uses_kscreen_doctor() {
    let fx = fx();
    fx.wayland(Some("KDE"), true);
    fx.tools.install(&["xset", "kscreen-doctor"]);
    fx.tools.respond(&["xset"], out(0, ""));
    fx.tools.respond(&["kscreen-doctor"], out(0, ""));
    assert!(fx.plat.display_off());
    assert!(fx.plat.wake_display());
    assert_eq!(
        fx.tools.calls(),
        vec![
            argv(&["kscreen-doctor", "--dpms", "off"]),
            argv(&["kscreen-doctor", "--dpms", "on"])
        ]
    );
}

#[test]
fn test_display_off_gnome_wayland_uses_mutter() {
    let fx = fx();
    fx.wayland(Some("ubuntu:GNOME"), true);
    fx.tools.install(&["xset", "busctl", "gdbus"]);
    fx.tools.respond(&["busctl"], out(0, ""));
    assert!(fx.plat.display_off());
    assert!(fx.plat.wake_display());
    let prefix = [
        "busctl",
        "--user",
        "set-property",
        "org.gnome.Mutter.DisplayConfig",
        "/org/gnome/Mutter/DisplayConfig",
        "org.gnome.Mutter.DisplayConfig",
        "PowerSaveMode",
        "i",
    ];
    let with = |mode: &str| {
        let mut call = argv(&prefix);
        call.push(mode.into());
        call
    };
    assert_eq!(fx.tools.calls(), vec![with("1"), with("0")]);
}

#[test]
fn test_display_off_gnome_falls_back_to_gdbus() {
    let fx = fx();
    fx.wayland(Some("GNOME"), true);
    fx.tools.install(&["gdbus"]);
    fx.tools.respond(&["gdbus"], out(0, "()\n"));
    assert!(fx.plat.display_off());
    let call = &fx.tools.calls()[0];
    assert_eq!(call[..3], argv(&["gdbus", "call", "--session"]));
    assert!(call.contains(&"org.freedesktop.DBus.Properties.Set".to_string()));
    assert_eq!(
        call[call.len() - 2..],
        argv(&["PowerSaveMode", "<int32 1>"])
    );
}

#[test]
fn test_display_off_sway_prefers_power_then_dpms() {
    let fx = fx();
    fx.wayland(Some("sway"), true);
    fx.setenv("SWAYSOCK", "/run/user/1000/sway-ipc.sock");
    fx.tools.install(&["swaymsg"]);
    fx.tools
        .respond(&["swaymsg", "output", "*", "dpms"], out(0, ""));
    assert!(fx.plat.display_off());
    assert_eq!(
        fx.tools.calls(),
        vec![
            argv(&["swaymsg", "output", "*", "power", "off"]),
            argv(&["swaymsg", "output", "*", "dpms", "off"]),
        ]
    );
}

#[test]
fn test_display_off_unknown_wayland_is_unsupported() {
    let fx = fx();
    fx.wayland(Some("Weston"), true);
    fx.tools.install(&["xset"]);
    fx.tools.respond(&["xset"], out(0, ""));
    assert!(!fx.plat.display_off());
    assert!(!fx.plat.wake_display());
    assert!(fx.tools.calls().is_empty()); // xset would only blank XWayland's virtual output
}

#[test]
fn test_wake_x11_falls_back_to_xtest() {
    let mut fx = fx();
    fx.x11(None);
    let xdisplay = FakeDisplay::new();
    xdisplay.0.lock().extensions = HashSet::from(["XTEST".to_string()]); // no DPMS to use
    fx.plat.ewmh = Arc::new(xdisplay.client(&fx.clock));
    assert!(fx.plat.wake_display());
    assert_eq!(xdisplay.0.lock().motions, [1, -1]);
}

#[test]
fn test_wake_wayland_never_uses_xtest() {
    let mut fx = fx();
    fx.wayland(Some("GNOME"), true);
    let xdisplay = FakeDisplay::new();
    xdisplay.0.lock().extensions.insert("XTEST".into());
    fx.plat.ewmh = Arc::new(xdisplay.client(&fx.clock));
    assert!(!fx.plat.wake_display());
    assert!(
        xdisplay.0.lock().motions.is_empty(),
        "XTest used on Wayland"
    );
}

// ------------------------------------------------------------------ session lock state
#[test]
fn test_parse_locked_hint() {
    let cases = [
        (Some("yes\n"), Some(true)),
        (Some("no\n"), Some(false)),
        (Some("LockedHint=yes\n"), Some(true)),
        (Some("LockedHint=no"), Some(false)),
        (Some("  YES  "), Some(true)),
        (Some(""), None),
        (None, None),
        (Some("maybe"), None),
    ];
    for (text, expected) in cases {
        assert_eq!(parse_locked_hint(text), expected, "{text:?}");
    }
}

#[test]
fn test_session_locked_queries_logind_and_caches() {
    let fx = fx();
    fx.setenv("XDG_SESSION_ID", "5");
    fx.tools.install(&["loginctl"]);
    let mut answers = VecDeque::from(["yes\n", "no\n"]);
    fx.tools.respond(
        &["loginctl", "show-session"],
        Box::new(move |_| Ok((0, answers.pop_front().unwrap().to_string()))),
    );
    assert_eq!(fx.plat.is_session_locked(), Some(true));
    assert_eq!(
        fx.tools.calls(),
        vec![argv(&[
            "loginctl",
            "show-session",
            "5",
            "-p",
            "LockedHint",
            "--value"
        ])]
    );
    fx.clock.advance(1.5);
    assert_eq!(fx.plat.is_session_locked(), Some(true)); // cached
    assert_eq!(fx.tools.calls().len(), 1);
    fx.clock.advance(1.0);
    assert_eq!(fx.plat.is_session_locked(), Some(false));
    assert_eq!(fx.tools.calls().len(), 2);
}

#[test]
fn test_session_locked_unknown_without_loginctl() {
    let fx = fx();
    fx.setenv("XDG_SESSION_ID", "5");
    assert_eq!(fx.plat.is_session_locked(), None);
    assert!(fx.tools.calls().is_empty());
}

#[test]
fn test_session_locked_unknown_on_error() {
    let fx = fx();
    fx.setenv("XDG_SESSION_ID", "5");
    fx.tools.install(&["loginctl"]);
    fx.tools.respond(
        &["loginctl"],
        out(1, "Failed to get session: No session '5' known\n"),
    );
    assert_eq!(fx.plat.is_session_locked(), None);
}

#[test]
fn test_session_locked_rejects_odd_session_id() {
    let fx = fx();
    fx.setenv("XDG_SESSION_ID", "5 --help");
    fx.tools.install(&["loginctl"]);
    assert_eq!(fx.plat.is_session_locked(), None);
    assert!(fx.tools.calls().is_empty());
}

#[test]
fn test_lock_invalidates_locked_cache() {
    let fx = fx();
    fx.setenv("XDG_SESSION_ID", "5");
    fx.setenv("XDG_CURRENT_DESKTOP", "GNOME");
    fx.tools.install(&["loginctl"]);
    let mut hints = VecDeque::from(["no\n", "yes\n"]);
    fx.tools.respond(
        &["loginctl", "show-session"],
        Box::new(move |_| Ok((0, hints.pop_front().unwrap_or("yes").to_string()))),
    );
    fx.tools.respond(&["loginctl", "lock-session"], out(0, ""));
    assert_eq!(fx.plat.is_session_locked(), Some(false));
    assert!(fx.plat.lock_screen());
    assert_eq!(fx.plat.is_session_locked(), Some(true));
}

#[test]
fn test_session_locked_by_locker_without_locked_hint() {
    // sway + swaylock: logind's LockedHint stays "no" the whole time.
    let fx = fx();
    let tree = fx.proc_tree();
    fx.setenv("XDG_SESSION_ID", "5");
    fx.wayland(Some("sway"), true);
    fx.tools.install(&["loginctl"]);
    fx.tools
        .respond(&["loginctl", "show-session"], out(0, "no\n"));
    tree.process(100, &[], "sway");
    assert_eq!(fx.plat.is_session_locked(), Some(false));
    tree.process(200, &[], "swaylock");
    fx.clock.advance(2.5);
    assert_eq!(fx.plat.is_session_locked(), Some(true));
    tree.remove(200);
    fx.clock.advance(2.5);
    assert_eq!(fx.plat.is_session_locked(), Some(false)); // the unlock is seen too
}

#[test]
fn test_session_locked_locker_names_are_truncated_like_comm() {
    let fx = fx();
    let tree = fx.proc_tree();
    fx.setenv("XDG_CURRENT_DESKTOP", "i3");
    tree.process(300, &[], "kscreenlocker_g"); // kscreenlocker_greet, 15 chars
    assert_eq!(fx.plat.is_session_locked(), Some(true));
}

#[test]
fn test_session_locked_ignores_other_users_lockers() {
    let fx = fx();
    let tree = fx.proc_tree();
    fx.setenv("XDG_CURRENT_DESKTOP", "i3");
    tree.login(1000);
    tree.process_full(300, &[], "i3lock", "", Some(1001)); // another user's login
    tree.process_full(301, &[], "slock", "", Some(AUDIT_UID_UNSET)); // no login
    let owner = tree.owner(300);
    set_uid(&fx.plat, Some(owner + 1)); // none of the entries is ours
    assert_eq!(fx.plat.is_session_locked(), Some(false));
}

/// A locker started by a service (no login UID) still counts when we own it.
#[test]
fn test_session_locked_by_own_locker_outside_a_login() {
    let fx = fx();
    let tree = fx.proc_tree();
    fx.setenv("XDG_CURRENT_DESKTOP", "i3");
    tree.login(1000);
    tree.process_full(300, &[], "swaylock", "", Some(AUDIT_UID_UNSET));
    set_uid(&fx.plat, Some(tree.owner(300)));
    assert_eq!(fx.plat.is_session_locked(), Some(true));
}

/// Started outside a login ourselves: the locker's login UID is compared with our UID.
#[test]
fn test_session_locked_by_a_setuid_locker_when_we_have_no_login_uid() {
    let fx = fx();
    let tree = fx.proc_tree();
    fx.setenv("XDG_CURRENT_DESKTOP", "i3");
    tree.login(AUDIT_UID_UNSET);
    tree.process(400, &[], "slock");
    let owner = tree.owner(400);
    set_uid(&fx.plat, Some(owner + 1)); // root's entry, not ours
    let loginuid = tree.proc.join("400").join("loginuid");
    std::fs::write(&loginuid, (owner + 2).to_string()).unwrap(); // another user's
    assert_eq!(fx.plat.is_session_locked(), Some(false));
    std::fs::write(&loginuid, (owner + 1).to_string()).unwrap(); // ours
    *fx.plat.locked_cache.lock() = None;
    assert_eq!(fx.plat.is_session_locked(), Some(true));
}

/// xss-lock -- slock: slock runs as 'nobody' and its /proc entry belongs to root.
#[test]
fn test_session_locked_by_a_setuid_locker() {
    let fx = fx();
    let tree = fx.proc_tree();
    fx.setenv("XDG_CURRENT_DESKTOP", "i3");
    tree.login(1000);
    tree.process_full(100, &[], "i3", "", Some(1000));
    tree.process_full(400, &[], "slock", "", Some(1000));
    set_uid(&fx.plat, Some(tree.owner(400) + 1)); // the entry is not ours
    assert_eq!(fx.plat.is_session_locked(), Some(true));
}

#[test]
fn test_session_locked_without_login_uids_falls_back_to_the_owner() {
    // A kernel without audit support: no loginuid files at all.
    let fx = fx();
    let tree = fx.proc_tree();
    fx.setenv("XDG_CURRENT_DESKTOP", "i3");
    tree.process(300, &[], "i3lock");
    let owner = tree.owner(300);
    set_uid(&fx.plat, Some(owner + 1));
    assert_eq!(fx.plat.is_session_locked(), Some(false));
    set_uid(&fx.plat, Some(owner));
    *fx.plat.locked_cache.lock() = None;
    assert_eq!(fx.plat.is_session_locked(), Some(true));
}

/// No second locker (dm-tool: a second unlock) on top of slock.
#[test]
fn test_lock_confirmed_by_a_setuid_locker() {
    let fx = fx();
    let tree = fx.proc_tree();
    tree.login(1000);
    fx.tools.install(&["loginctl", "dm-tool"]);
    fx.tools.respond(&["loginctl", "lock-session"], out(0, ""));
    fx.tools
        .respond(&["loginctl", "show-session"], out(0, "no\n")); // xss-lock never sets it
    let (clock, locker_tree) = (fx.clock.clone(), tree.clone());
    let platform = fx.platform(LinuxOptions {
        sleep: Arc::new(move |seconds| {
            clock.advance(seconds);
            locker_tree.process_full(4321, &[], "slock", "", Some(1000));
        }),
        ..options_for(fx.tmp.path(), &fx.clock)
    });
    platform.env.set("XDG_SESSION_ID", "2");
    platform.env.set("XDG_CURRENT_DESKTOP", "i3");
    set_uid(&platform, Some(12345)); // slock's entry belongs to root
    assert!(platform.lock_screen());
    assert!(!fx.tools.calls().contains(&argv(&["dm-tool", "lock"])));
}

#[test]
fn test_login_uid_parsing() {
    let cases = [
        ("1000\n", Some(1000)),
        ("0", Some(0)),
        ("4294967295", None),
        ("", None),
        ("junk", None),
    ];
    for (text, expected) in cases {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("loginuid"), text).unwrap();
        assert_eq!(login_uid(tmp.path()), expected, "{text:?}");
        assert_eq!(login_uid(&tmp.path().join("missing")), None);
    }
}

#[test]
fn test_session_locked_without_logind_session_relies_on_lockers() {
    let fx = fx();
    let tree = fx.proc_tree();
    fx.setenv("XDG_CURRENT_DESKTOP", "i3");
    set_uid(&fx.plat, None); // no session can be looked up
    tree.process(100, &[], "i3");
    assert_eq!(fx.plat.is_session_locked(), Some(false));
    tree.process(101, &[], "slock");
    *fx.plat.locked_cache.lock() = None;
    assert_eq!(fx.plat.is_session_locked(), Some(true));
}

#[test]
fn test_session_locked_failed_query_is_unknown_not_unlocked() {
    // A lock screen that is up must not be mistaken for the user's return.
    let fx = fx();
    let _tree = fx.proc_tree();
    fx.setenv("XDG_SESSION_ID", "5");
    fx.setenv("XDG_CURRENT_DESKTOP", "GNOME");
    fx.tools.install(&["loginctl"]);
    fx.tools.respond(&["loginctl", "show-session"], timeout());
    assert_eq!(fx.plat.is_session_locked(), None);
}

#[test]
fn test_session_locked_gnome_trusts_locked_hint_without_scanning() {
    let mut fx = fx();
    let _tree = fx.proc_tree();
    fx.setenv("XDG_SESSION_ID", "5");
    fx.setenv("XDG_CURRENT_DESKTOP", "ubuntu:GNOME");
    fx.tools.install(&["loginctl"]);
    fx.tools
        .respond(&["loginctl", "show-session"], out(0, "no\n"));
    fx.plat.hooks.locker_running = Some(Box::new(|| panic!("scanned /proc")));
    assert_eq!(fx.plat.is_session_locked(), Some(false));
}

#[test]
fn test_session_lookup_failure_is_retried() {
    // Autostarted by a systemd user unit (no XDG_SESSION_ID) while logind is slow.
    let fx = fx();
    set_uid(&fx.plat, Some(1000));
    fx.tools.install(&["loginctl"]);
    let mut first = true;
    fx.tools.respond(
        &["loginctl", "show-user"],
        Box::new(move |_| {
            if std::mem::take(&mut first) {
                return Err("timed out".into());
            }
            Ok((0, "3\n".into()))
        }),
    );
    fx.tools
        .respond(&["loginctl", "show-session"], out(0, "yes\n"));
    assert_eq!(fx.plat.is_session_locked(), None);
    fx.clock.advance(5.0);
    assert_eq!(fx.plat.is_session_locked(), None); // not retried on every poll
    let verbs: Vec<String> = fx.tools.calls().into_iter().map(|c| c[1].clone()).collect();
    assert_eq!(verbs, ["show-user"]);
    fx.clock.advance(30.0);
    assert_eq!(fx.plat.is_session_locked(), Some(true));
    assert_eq!(
        fx.tools.calls().last().unwrap(),
        &argv(&[
            "loginctl",
            "show-session",
            "3",
            "-p",
            "LockedHint",
            "--value"
        ])
    );
}

#[test]
fn test_locked_hint_through_dbus_in_process() {
    let fx = fx();
    fx.setenv("XDG_SESSION_ID", "5");
    fx.setenv("XDG_CURRENT_DESKTOP", "GNOME");
    fx.tools.install(&["loginctl"]);
    let path = "/org/freedesktop/login1/session/_35";
    fx.dbus
        .answer("GetSession", DBusReply::values(vec![str_value(path)]));
    let mut hints = VecDeque::from([true, false]);
    fx.dbus.answer_with(
        "Get",
        Box::new(move |_| {
            Some(DBusReply::values(vec![DBusValue::Bool(
                hints.pop_front().unwrap(),
            )]))
        }),
    );
    assert_eq!(fx.plat.is_session_locked(), Some(true));
    fx.clock.advance(2.5);
    assert_eq!(fx.plat.is_session_locked(), Some(false));
    assert!(fx.tools.calls().is_empty()); // no loginctl process per poll
    assert_eq!(fx.dbus.methods(), ["GetSession", "Get", "Get"]); // the session path is cached
    let get = &fx.dbus.calls()[1];
    assert!(get.system);
    assert_eq!(get.path, path);
    assert_eq!(get.args, ["org.freedesktop.login1.Session", "LockedHint"]);
}

#[test]
fn test_locked_hint_dbus_error_forgets_session_path() {
    let fx = fx();
    fx.setenv("XDG_SESSION_ID", "5");
    fx.setenv("XDG_CURRENT_DESKTOP", "GNOME");
    fx.tools.install(&["loginctl"]);
    fx.dbus.answer(
        "GetSession",
        DBusReply::values(vec![str_value("/org/freedesktop/login1/session/_35")]),
    );
    fx.dbus.answer(
        "Get",
        DBusReply::failed("org.freedesktop.DBus.Error.UnknownObject"),
    );
    assert_eq!(fx.plat.is_session_locked(), None);
    fx.clock.advance(2.5);
    assert_eq!(fx.plat.is_session_locked(), None);
    assert_eq!(
        fx.dbus.methods(),
        ["GetSession", "Get", "GetSession", "Get"]
    );
    assert!(fx.tools.calls().is_empty()); // D-Bus answered: no pointless loginctl fallback
}

#[test]
fn test_locked_hint_unexpected_dbus_reply_falls_back_to_loginctl() {
    let cases = [
        // a variant left unconverted
        (
            str_value("/org/freedesktop/login1/session/_35"),
            str_value("yes"),
        ),
        // an object path that could not be converted
        (DBusValue::Other, DBusValue::Bool(true)),
    ];
    for (session_reply, hint_reply) in cases {
        let fx = fx();
        fx.setenv("XDG_SESSION_ID", "5");
        fx.setenv("XDG_CURRENT_DESKTOP", "GNOME");
        fx.tools.install(&["loginctl"]);
        fx.tools
            .respond(&["loginctl", "show-session"], out(0, "yes\n"));
        fx.dbus
            .answer("GetSession", DBusReply::values(vec![session_reply.clone()]));
        fx.dbus
            .answer("Get", DBusReply::values(vec![hint_reply.clone()]));
        assert_eq!(fx.plat.is_session_locked(), Some(true), "{session_reply:?}");
        assert_eq!(fx.tools.names(), ["loginctl"]);
    }
}

// ------------------------------------------------------------------ idle time
#[test]
fn test_parse_gdbus_uint() {
    let cases = [
        (Some("(uint64 12345,)\n"), Some(12345)),
        (Some("(uint64 0,)"), Some(0)),
        (Some("(uint32 7,)"), Some(7)),
        (Some("(42,)"), Some(42)),
        (
            Some("Error: GDBus.Error:org.freedesktop.DBus.Error.ServiceUnknown"),
            None,
        ),
        (Some(""), None),
        (None, None),
    ];
    for (text, expected) in cases {
        assert_eq!(parse_gdbus_uint(text), expected, "{text:?}");
    }
}

#[test]
fn test_idle_x11_uses_screensaver_extension() {
    let mut fx = fx();
    fx.x11(Some("GNOME"));
    fx.tools.install(&["gdbus"]);
    fx.plat.xss = Box::new(FakeXss(Some(1500)));
    assert_approx!(fx.plat.seconds_since_input().unwrap(), 1.5);
    assert!(fx.tools.calls().is_empty());
    assert_eq!(fx.plat.seconds_since_key_input(), None);
}

#[test]
fn test_idle_gnome_wayland_uses_mutter_with_extrapolated_cache() {
    let mut fx = fx();
    fx.wayland(Some("GNOME"), true);
    fx.plat.xss = Box::new(FakeXss(Some(999_999))); // XWayland idle must be ignored
    fx.tools.install(&["gdbus"]);
    fx.tools.respond(&["gdbus"], out(0, "(uint64 12345,)\n"));
    assert_approx!(fx.plat.seconds_since_input().unwrap(), 12.345);
    assert_eq!(
        fx.tools.calls()[0].last().unwrap(),
        "org.gnome.Mutter.IdleMonitor.GetIdletime"
    );
    fx.clock.advance(0.3);
    // Cached, and advanced with the clock so the implied last-input time stays put.
    assert_approx!(fx.plat.seconds_since_input().unwrap(), 12.645);
    assert_eq!(fx.tools.calls().len(), 1);
    fx.clock.advance(0.3);
    assert_approx!(fx.plat.seconds_since_input().unwrap(), 12.345);
    assert_eq!(fx.tools.calls().len(), 2);
}

#[test]
fn test_idle_backs_off_after_failure() {
    let fx = fx();
    fx.wayland(Some("KDE"), true);
    fx.tools.install(&["gdbus"]);
    fx.tools.respond(
        &["gdbus"],
        out(
            1,
            "Error: GDBus.Error:org.freedesktop.DBus.Error.ServiceUnknown",
        ),
    );
    assert_eq!(fx.plat.seconds_since_input(), None);
    fx.clock.advance(10.0);
    assert_eq!(fx.plat.seconds_since_input(), None);
    assert_eq!(fx.tools.calls().len(), 1);
    fx.clock.advance(55.0);
    assert_eq!(fx.plat.seconds_since_input(), None);
    assert_eq!(fx.tools.calls().len(), 2);
}

#[test]
fn test_idle_x11_without_libxss_falls_back_to_mutter() {
    let fx = fx();
    fx.x11(Some("GNOME"));
    fx.tools.install(&["gdbus"]);
    fx.tools.respond(&["gdbus"], out(0, "(uint64 2000,)"));
    assert_approx!(fx.plat.seconds_since_input().unwrap(), 2.0);
}

#[test]
fn test_idle_transient_failure_is_retried_soon() {
    // gnome-shell stalls once (monitor hotplug, GPU hang): typing must not vanish for 60 s.
    let fx = fx();
    fx.wayland(Some("GNOME"), true);
    fx.tools.install(&["gdbus"]);
    let mut answers: VecDeque<Result<(i32, String), String>> = VecDeque::from([
        Ok((0, "(uint64 100,)".to_string())),
        Err("timed out".to_string()),
    ]);
    fx.tools.respond(
        &["gdbus"],
        Box::new(move |_| {
            answers
                .pop_front()
                .unwrap_or(Ok((0, "(uint64 200,)".into())))
        }),
    );
    assert_approx!(fx.plat.seconds_since_input().unwrap(), 0.1);
    fx.clock.advance(1.0);
    assert_eq!(fx.plat.seconds_since_input(), None);
    fx.clock.advance(0.5);
    assert_eq!(fx.plat.seconds_since_input(), None); // short back-off
    fx.clock.advance(0.6);
    assert_approx!(fx.plat.seconds_since_input().unwrap(), 0.2);
    assert_eq!(fx.tools.calls().len(), 3);
    assert_approx!(fx.tools.kwargs()[0].0, IDLE_QUERY_TIMEOUT_S);
}

#[test]
fn test_idle_through_dbus_in_process() {
    let fx = fx();
    fx.wayland(Some("GNOME"), true);
    fx.tools.install(&["gdbus"]);
    fx.dbus.answer(
        "GetIdletime",
        DBusReply::values(vec![DBusValue::Int(12345)]),
    );
    assert_approx!(fx.plat.seconds_since_input().unwrap(), 12.345);
    fx.clock.advance(0.6);
    assert_approx!(fx.plat.seconds_since_input().unwrap(), 12.345);
    assert!(fx.tools.calls().is_empty()); // no gdbus process twice a second
    let call = &fx.dbus.calls()[0];
    assert!(!call.system);
    assert_eq!(call.service, "org.gnome.Mutter.IdleMonitor");
    assert_eq!(call.path, "/org/gnome/Mutter/IdleMonitor/Core");
}

#[test]
fn test_idle_dbus_missing_service_backs_off() {
    let fx = fx();
    fx.wayland(Some("KDE"), true);
    fx.dbus.answer(
        "GetIdletime",
        DBusReply::failed("org.freedesktop.DBus.Error.ServiceUnknown"),
    );
    assert_eq!(fx.plat.seconds_since_input(), None);
    fx.clock.advance(30.0);
    assert_eq!(fx.plat.seconds_since_input(), None);
    assert_eq!(fx.dbus.calls().len(), 1);
}

#[test]
fn test_idle_dbus_hiccup_after_success_retries_soon() {
    let fx = fx();
    fx.wayland(Some("GNOME"), true);
    let mut replies = VecDeque::from([
        DBusReply::values(vec![DBusValue::Int(500)]),
        DBusReply::failed("org.freedesktop.DBus.Error.NoReply"),
        DBusReply::values(vec![DBusValue::Int(700)]),
    ]);
    fx.dbus
        .answer_with("GetIdletime", Box::new(move |_| replies.pop_front()));
    assert_approx!(fx.plat.seconds_since_input().unwrap(), 0.5);
    fx.clock.advance(0.6);
    assert_eq!(fx.plat.seconds_since_input(), None);
    fx.clock.advance(1.1);
    assert_approx!(fx.plat.seconds_since_input().unwrap(), 0.7);
}

#[test]
fn test_idle_unexpected_dbus_reply_falls_back_to_gdbus() {
    let fx = fx();
    fx.wayland(Some("GNOME"), true);
    fx.tools.install(&["gdbus"]);
    fx.tools.respond(&["gdbus"], out(0, "(uint64 1500,)"));
    fx.dbus
        .answer("GetIdletime", DBusReply::values(vec![str_value("1500")]));
    assert_approx!(fx.plat.seconds_since_input().unwrap(), 1.5);
    assert_eq!(fx.tools.names(), ["gdbus"]);
}

#[test]
fn test_idle_ignores_reset_caused_by_our_ydotool_move() {
    let fx = fx();
    fx.wayland(Some("GNOME"), true);
    fx.ydotool_ready();
    fx.tools.install(&["gdbus"]);
    let mut idle_ms = VecDeque::from([5000, 600, 100]);
    fx.tools.respond(
        &["gdbus"],
        Box::new(move |_| Ok((0, format!("(uint64 {},)", idle_ms.pop_front().unwrap())))),
    );
    assert_approx!(fx.plat.seconds_since_input().unwrap(), 5.0); // last input at 995
    // uinput motion resets Mutter's idle timer
    assert_eq!(fx.plat.move_cursor(960, 540), Some(true));
    fx.clock.advance(0.6);
    // Mutter now says 0.6 s, but that reset was ours: the user has been idle for 5.6 s.
    assert_approx!(fx.plat.seconds_since_input().unwrap(), 5.6);
    fx.clock.advance(2.4);
    assert_approx!(fx.plat.seconds_since_input().unwrap(), 0.1); // real input later is seen
}

// ------------------------------------------------------------------ cursor
#[test]
fn test_cursor_x11_warps_through_x11() {
    // The Python app left X11 to Qt's QCursor.setPos; there is no toolkit fallback now.
    let (fx, xdisplay) = x11_fx();
    fx.tools.install(&["ydotool"]);
    assert_eq!(fx.plat.move_cursor(10, -20), Some(true));
    assert_eq!(xdisplay.0.lock().warps, [(10, -20)]);
    assert!(fx.tools.calls().is_empty());
}

#[test]
fn test_cursor_position_x11_queries_the_pointer() {
    let (fx, xdisplay) = x11_fx();
    xdisplay.0.lock().pointer = (2600, 400);
    assert_eq!(fx.plat.cursor_position(), Some((2600, 400)));
    fx.wayland(Some("GNOME"), true); // XWayland only sees X11 clients: never trust it
    assert_eq!(fx.plat.cursor_position(), None);
}

const YDOTOOL_1_HELP: &str = "Usage: mousemove [OPTION]... [-x <xpos> -y <ypos>] [-- <xpos> <ypos>]\n  \
     -a, --absolute             Use absolute position, not applied to wheel\n";

#[test]
fn test_cursor_wayland_uses_ydotool() {
    let fx = fx();
    fx.wayland(Some("GNOME"), true);
    fx.ydotool_ready();
    assert_eq!(fx.plat.move_cursor(2560, -20), Some(true));
    assert_eq!(fx.plat.move_cursor(10, 20), Some(true));
    assert_eq!(
        fx.tools.calls(),
        vec![
            argv(&["ydotool", "mousemove", "--help"]), // probed once
            argv(&[
                "ydotool",
                "mousemove",
                "--absolute",
                "-x",
                "2560",
                "-y",
                "-20"
            ]),
            argv(&["ydotool", "mousemove", "--absolute", "-x", "10", "-y", "20"]),
        ]
    );
}

#[test]
fn test_cursor_wayland_without_ydotool() {
    let fx = fx();
    fx.wayland(Some("GNOME"), true);
    assert_eq!(fx.plat.move_cursor(1, 2), Some(false));
    fx.ydotool_ready();
    fx.tools.respond(
        &["ydotool", "mousemove", "--absolute"],
        out(1, "failed to connect socket"),
    );
    assert_eq!(fx.plat.move_cursor(1, 2), Some(false));
}

#[test]
fn test_cursor_old_ydotool_is_unsupported() {
    // Debian/Ubuntu still ship 0.1.x, whose CLI has no --absolute.
    let fx = fx();
    fx.wayland(Some("GNOME"), true);
    fx.ydotool_ready();
    fx.tools.respond(
        &["ydotool", "mousemove", "--help"],
        out(1, "Usage: mousemove [--delay <ms>] <x> <y>"),
    );
    assert_eq!(fx.plat.move_cursor(1, 2), Some(false));
    assert!(!fx.plat.capabilities()["cursor"]);
    assert_eq!(
        fx.tools.calls(),
        vec![argv(&["ydotool", "mousemove", "--help"])]
    );
}

#[test]
fn test_cursor_ydotool_needs_its_daemon() {
    let fx = fx();
    fx.wayland(Some("GNOME"), true);
    fx.ydotool_ready();
    let missing = fx.tmp.path().join("missing.sock");
    fx.setenv("YDOTOOL_SOCKET", missing.to_str().unwrap());
    assert_eq!(fx.plat.move_cursor(1, 2), Some(false));
    assert!(!fx.plat.capabilities()["cursor"]);
    let runtime = fx.tmp.path().join("run");
    std::fs::create_dir(&runtime).unwrap();
    std::fs::write(runtime.join(".ydotool_socket"), "").unwrap();
    fx.delenv("YDOTOOL_SOCKET");
    fx.setenv("XDG_RUNTIME_DIR", runtime.to_str().unwrap());
    assert_eq!(fx.plat.move_cursor(1, 2), Some(true));
}

#[test]
fn test_cursor_sway_moves_through_its_ipc() {
    let fx = fx();
    fx.wayland(Some("sway"), true);
    fx.setenv("SWAYSOCK", "/run/user/1000/sway-ipc.sock");
    fx.ydotool_ready();
    fx.tools.install(&["swaymsg"]);
    fx.tools.respond(&["swaymsg"], out(0, ""));
    assert_eq!(fx.plat.move_cursor(2880, 540), Some(true));
    assert_eq!(
        fx.tools.calls(),
        vec![argv(&[
            "swaymsg", "seat", "-", "cursor", "set", "2880", "540"
        ])]
    );
    // Older sway without the "-" seat alias: the default seat name.
    fx.tools.clear_calls();
    fx.tools
        .respond(&["swaymsg", "seat", "-"], out(2, "Error: seat - not found"));
    assert_eq!(fx.plat.move_cursor(1, 2), Some(true));
    assert_eq!(
        fx.tools.calls().last().unwrap(),
        &argv(&["swaymsg", "seat", "seat0", "cursor", "set", "1", "2"])
    );
}

#[test]
fn test_cursor_hyprland_moves_through_its_ipc() {
    let fx = fx();
    fx.wayland(Some("Hyprland"), true);
    fx.setenv("HYPRLAND_INSTANCE_SIGNATURE", "abc");
    fx.tools.install(&["hyprctl"]);
    fx.tools.respond(&["hyprctl"], out(0, "ok\n"));
    assert_eq!(fx.plat.move_cursor(100, 200), Some(true));
    assert_eq!(
        fx.tools.calls(),
        vec![argv(&["hyprctl", "dispatch", "movecursor", "100", "200"])]
    );
    fx.tools
        .respond(&["hyprctl"], out(0, "Invalid dispatcher\n")); // hyprctl exits 0 regardless
    assert_eq!(fx.plat.move_cursor(100, 200), Some(false));
}

#[test]
fn test_cursor_position_reliability() {
    let fx = fx();
    fx.x11(None);
    assert!(fx.plat.cursor_position_reliable());
    fx.wayland(Some("GNOME"), true); // XWayland only sees the pointer over X11 windows
    assert!(!fx.plat.cursor_position_reliable());
}

// ------------------------------------------------------------------ camera in use
fn own_pid() -> u32 {
    std::process::id()
}

#[test]
fn test_camera_used_by_another_process() {
    let fx = fx();
    let tree = fx.proc_tree();
    tree.device("video0", true);
    tree.process(100, &["/dev/null", "socket:[123]"], "app");
    tree.process(200, &["/dev/null", "/dev/video0"], "app");
    assert_eq!(fx.plat.camera_in_use_by_other_app(), Some(true));
}

#[test]
fn test_camera_held_only_by_us() {
    let fx = fx();
    let tree = fx.proc_tree();
    tree.device("video0", true);
    tree.process(own_pid(), &["/dev/video0"], "app");
    tree.process(100, &["/dev/null"], "app");
    assert_eq!(fx.plat.camera_in_use_by_other_app(), Some(false));
}

#[test]
fn test_camera_scan_ignores_unreadable_entries() {
    let fx = fx();
    let tree = fx.proc_tree();
    tree.device("video0", true);
    tree.process(100, &["/dev/video0"], "app");
    std::fs::create_dir(tree.proc.join("300")).unwrap(); // no fd directory: exited / not ours
    set_readlink(&fx.plat, |path| {
        let pid = path
            .parent()
            .and_then(Path::parent)
            .and_then(Path::file_name);
        if pid == Some("100".as_ref()) {
            return Err(io::Error::from(io::ErrorKind::PermissionDenied));
        }
        read_fake_link(path)
    });
    assert_eq!(fx.plat.camera_in_use_by_other_app(), Some(false));
}

#[test]
fn test_camera_deleted_suffix() {
    let fx = fx();
    let tree = fx.proc_tree();
    tree.device("video0", true);
    tree.process(100, &["/dev/video0 (deleted)"], "app");
    assert_eq!(fx.plat.camera_in_use_by_other_app(), Some(true));
}

#[test]
fn test_camera_pipewire_monitoring_is_not_use() {
    let fx = fx();
    let tree = fx.proc_tree();
    tree.device("video0", true);
    tree.process(100, &["/dev/video0"], "wireplumber");
    tree.process(101, &["/dev/video0"], "pipewire");
    assert_eq!(fx.plat.camera_in_use_by_other_app(), Some(false));
}

#[test]
fn test_camera_pipewire_streaming_is_use() {
    let fx = fx();
    let tree = fx.proc_tree();
    tree.device("video0", true);
    let maps = "7f00-7f10 rw-s 00000000 00:05 42   /dev/video0\n";
    tree.process_full(101, &["/dev/video0"], "pipewire", maps, None);
    assert_eq!(fx.plat.camera_in_use_by_other_app(), Some(true));
}

#[test]
fn test_camera_virtual_loopback_is_ignored() {
    let fx = fx();
    let tree = fx.proc_tree();
    tree.device("video0", true);
    tree.device("video10", false); // OBS virtual camera
    tree.process(100, &["/dev/video10"], "obs");
    assert_eq!(fx.plat.camera_in_use_by_other_app(), Some(false));
}

#[test]
fn test_camera_no_devices_means_not_in_use() {
    let fx = fx();
    let tree = fx.proc_tree();
    tree.process(100, &["/dev/video0"], "app"); // stale: no device node exists
    assert_eq!(fx.plat.camera_in_use_by_other_app(), Some(false));
}

#[test]
fn test_camera_without_dev_info_matches_any_video_node() {
    let fx = fx();
    let tree = fx.proc_tree();
    tree.process(100, &["/dev/video3"], "app");
    let platform = fx.platform(LinuxOptions {
        dev_root: fx.tmp.path().join("missing"),
        ..options_for(fx.tmp.path(), &fx.clock)
    });
    assert_eq!(platform.camera_in_use_by_other_app(), Some(true));
}

#[test]
fn test_camera_unknown_without_proc() {
    let fx = fx();
    let platform = fx.platform(LinuxOptions {
        proc_root: fx.tmp.path().join("nope"),
        ..options_for(fx.tmp.path(), &fx.clock)
    });
    assert_eq!(platform.camera_in_use_by_other_app(), None);
    assert!(!platform.capabilities()["camera_in_use"]);
}

#[test]
fn test_camera_scan_skips_other_users_processes() {
    let fx = fx();
    let tree = fx.proc_tree();
    tree.device("video0", true);
    tree.process(100, &["/dev/video0"], "app");
    let owner = tree.owner(100);
    set_readlink(&fx.plat, |_| panic!("read another user's fds"));
    set_uid(&fx.plat, Some(owner + 1));
    assert_eq!(fx.plat.camera_in_use_by_other_app(), Some(false));
}

#[test]
fn test_camera_result_is_cached() {
    let fx = fx();
    let tree = fx.proc_tree();
    tree.device("video0", true);
    assert_eq!(fx.plat.camera_in_use_by_other_app(), Some(false));
    tree.process(100, &["/dev/video0"], "app");
    fx.clock.advance(2.0);
    assert_eq!(fx.plat.camera_in_use_by_other_app(), Some(false));
    fx.clock.advance(1.5);
    assert_eq!(fx.plat.camera_in_use_by_other_app(), Some(true));
}

#[test]
fn test_camera_permission() {
    let fx = fx();
    let tree = fx.proc_tree();
    assert_eq!(
        fx.plat.permissions(),
        HashMap::from([("camera", None), ("accessibility", None)])
    );
    tree.device("video0", true);
    assert_eq!(fx.plat.permissions()["camera"], Some(true));
}

// ------------------------------------------------------------------ camera: refused opens
/// A capture attempt that failed at once: read-write open, then close (merged).
fn opened(foreign: Option<bool>) -> [CameraEvent; 1] {
    [CameraEvent {
        opened: true,
        foreign,
        wrote: true,
    }]
}

/// A physical camera that this process streams from.
fn streaming(fx: &Fx) -> ProcTree {
    let tree = fx.proc_tree();
    tree.device("video0", true);
    tree.process(own_pid(), &["/dev/video0"], "eye-tracker");
    tree
}

fn grace_s(plat: &LinuxPlatform) -> f64 {
    plat.camera.timings.lock().grace_s
}

#[test]
fn test_refused_open_releases_the_camera_for_a_grace_period() {
    // A browser joining a call opens the node, gets EBUSY and closes it again at once.
    let fx = fx();
    let _tree = streaming(&fx);
    assert_eq!(fx.plat.camera_in_use_by_other_app(), Some(false));
    fx.plat.camera.handle_events(&opened(Some(true)));
    assert_eq!(fx.plat.camera_in_use_by_other_app(), Some(true)); // no holder, but an attempt
    fx.clock.advance(grace_s(&fx.plat) - 1.0);
    assert_eq!(fx.plat.camera_in_use_by_other_app(), Some(true));
    fx.clock.advance(2.0);
    assert_eq!(fx.plat.camera_in_use_by_other_app(), Some(false)); // nobody took it: resume
}

#[test]
fn test_refused_open_then_app_takes_the_camera() {
    let fx = fx();
    let tree = streaming(&fx);
    fx.plat.camera.handle_events(&opened(Some(true)));
    assert_eq!(fx.plat.camera_in_use_by_other_app(), Some(true));
    // We released the camera; the app retried and now streams.
    tree.process(own_pid(), &[], "eye-tracker");
    std::fs::remove_file(tree.proc.join(own_pid().to_string()).join("fd").join("0")).unwrap();
    tree.process(700, &["/dev/video0"], "chrome");
    fx.clock.advance(3.5);
    assert_eq!(fx.plat.camera_in_use_by_other_app(), Some(true));
    fx.clock.advance(grace_s(&fx.plat));
    assert_eq!(fx.plat.camera_in_use_by_other_app(), Some(true)); // held: stays released
    tree.remove(700);
    fx.clock.advance(3.5);
    assert_eq!(fx.plat.camera_in_use_by_other_app(), Some(false));
    assert_eq!(fx.plat.camera.state.lock().false_alarms, 0);
}

#[test]
fn test_own_or_unattributed_opens_are_not_contention() {
    // Our own probe/reopen (foreign=False), or inotify that cannot tell (None).
    for foreign in [Some(false), None] {
        let fx = fx();
        let _tree = streaming(&fx);
        fx.plat.camera.handle_events(&opened(foreign));
        assert_eq!(
            fx.plat.camera_in_use_by_other_app(),
            Some(false),
            "{foreign:?}"
        );
    }
}

#[test]
fn test_device_listing_is_not_contention() {
    // A browser enumerating cameras opens each node read-only and closes it.
    let fx = fx();
    let _tree = streaming(&fx);
    fx.plat.camera.handle_events(&[
        CameraEvent::new(true, Some(true)),
        CameraEvent::new(false, Some(true)),
    ]);
    assert_eq!(fx.plat.camera_in_use_by_other_app(), Some(false));
}

#[test]
fn test_app_keeping_the_node_open_is_found_by_the_rescan() {
    let fx = fx();
    let tree = streaming(&fx);
    assert_eq!(fx.plat.camera_in_use_by_other_app(), Some(false));
    tree.process(700, &["/dev/video0"], "zoom"); // opened, retrying, not closed yet
    fx.plat
        .camera
        .handle_events(&[CameraEvent::new(true, Some(true))]);
    fx.clock.advance(fx.plat.camera.timings.lock().debounce_s);
    assert_eq!(fx.plat.camera_in_use_by_other_app(), Some(true));
}

#[test]
fn test_foreign_open_while_we_do_not_stream_is_not_contention() {
    let fx = fx();
    let tree = fx.proc_tree();
    tree.device("video0", true);
    fx.plat.camera.handle_events(&opened(Some(true))); // it will succeed: no conflict
    assert_eq!(fx.plat.camera_in_use_by_other_app(), Some(false));
}

#[test]
fn test_repeated_false_alarms_back_off() {
    let fx = fx();
    let _tree = streaming(&fx);
    let watcher = &fx.plat.camera;
    let timings = *watcher.timings.lock();
    let false_alarm = || {
        watcher.handle_events(&opened(Some(true)));
        assert_eq!(fx.plat.camera_in_use_by_other_app(), Some(true));
        fx.clock.advance(timings.grace_s + 0.1);
        assert_eq!(fx.plat.camera_in_use_by_other_app(), Some(false));
    };
    false_alarm(); // the first one is free: the user may simply have been slow to retry
    false_alarm();
    // An app keeps probing in the background: ignored for a while ...
    watcher.handle_events(&opened(Some(true)));
    assert_eq!(fx.plat.camera_in_use_by_other_app(), Some(false));
    fx.clock.advance(timings.cooldown_s);
    false_alarm(); // ... then honoured again, with a longer cooldown afterwards
    fx.clock.advance(timings.cooldown_s);
    watcher.handle_events(&opened(Some(true)));
    assert_eq!(fx.plat.camera_in_use_by_other_app(), Some(false));
    fx.clock.advance(timings.cooldown_s);
    watcher.handle_events(&opened(Some(true)));
    assert_eq!(fx.plat.camera_in_use_by_other_app(), Some(true));
}

fn fan_event(mask: u64, pid: i32, length: u32) -> Vec<u8> {
    let mut event = Vec::new();
    event.extend_from_slice(&length.to_ne_bytes());
    event.extend_from_slice(&[3, 0]);
    event.extend_from_slice(&24_u16.to_ne_bytes());
    event.extend_from_slice(&mask.to_ne_bytes());
    event.extend_from_slice(&(-1_i32).to_ne_bytes());
    event.extend_from_slice(&pid.to_ne_bytes());
    event.resize(event.len().max(length as usize), 0);
    event
}

#[test]
fn test_parse_fanotify_attributes_opens() {
    let own = 4242;
    let data = [
        fan_event(0x20 | 0x10, own as i32, 56), // our read-only open+close, merged, FID record
        fan_event(0x20, 0, 24), // another process (pid hidden from unprivileged listeners)
        fan_event(0x08, 0, 24), // ... closing a read-write descriptor
        fan_event(0x20 | 0x08, 0, 24), // a refused capture attempt, merged
        fan_event(0x4000, 0, 24), // queue overflow: unknown
        fan_event(0x01, 0, 24), // an access event: not asked for, ignored
    ]
    .concat();
    assert_eq!(
        parse_fanotify(&data, own),
        [
            CameraEvent::new(true, Some(false)),
            CameraEvent::new(true, Some(true)),
            CameraEvent {
                opened: false,
                foreign: Some(true),
                wrote: true
            },
            CameraEvent {
                opened: true,
                foreign: Some(true),
                wrote: true
            },
            CameraEvent::new(false, None),
        ]
    );
    let malformed = fan_event(0x20, 0, 0).repeat(3);
    assert!(parse_fanotify(&malformed, own).is_empty());
}

#[test]
fn test_parse_inotify_has_no_attribution() {
    let event = |mask: u32, name_len: u32| {
        [
            1_i32.to_ne_bytes(),
            mask.to_ne_bytes(),
            0_u32.to_ne_bytes(),
            name_len.to_ne_bytes(),
        ]
        .concat()
    };
    let data = [
        event(0x20, 0),
        event(0x10, 4),
        b"x\0\0\0".to_vec(),
        event(0x08, 0),
    ]
    .concat();
    assert_eq!(
        parse_inotify(&data),
        [
            CameraEvent::new(true, None),
            CameraEvent::new(false, None),
            CameraEvent {
                opened: false,
                foreign: None,
                wrote: true
            },
        ]
    );
}

#[test]
fn test_no_notifier_without_nodes() {
    assert!(open_camera_notifier(&[]).is_none());
}

/// No fanotify (kernel < 5.13, a sandbox): inotify; neither: `None`, the watcher polls.
#[test]
fn test_notifier_falls_back_when_the_kernel_refuses() {
    let opened: Mutex<Vec<(&str, Vec<String>)>> = Mutex::new(Vec::new());
    let refused = |error: io::Error| {
        let error = Mutex::new(Some(error));
        move |_: &[String]| -> io::Result<Box<dyn CameraNotifier>> {
            Err(error
                .lock()
                .take()
                .unwrap_or_else(|| io::Error::other("refused")))
        }
    };
    let inotify = |paths: &[String]| -> io::Result<Box<dyn CameraNotifier>> {
        opened.lock().push(("inotify", paths.to_vec()));
        Ok(Box::new(FakeNotifier::default()))
    };
    let paths = [String::from("/dev/video0")];
    let fanotify = refused(io::Error::from_raw_os_error(libc::EPERM));
    let found = open_first_notifier(&paths, &[("fanotify", &fanotify), ("inotify", &inotify)]);
    assert_eq!(found.map(|n| n.kind()), Some("fake"));
    assert_eq!(*opened.lock(), [("inotify", paths.to_vec())]);
    // A C library without the wrapper is no reason to fail either.
    let fanotify = refused(io::Error::from(io::ErrorKind::Unsupported));
    let found = open_first_notifier(&paths, &[("fanotify", &fanotify), ("inotify", &inotify)]);
    assert!(found.is_some());
    let inotify = refused(io::Error::from_raw_os_error(libc::EMFILE));
    assert!(
        open_first_notifier(&paths, &[("fanotify", &fanotify), ("inotify", &inotify)]).is_none()
    );
}

// ------------------------------------------------------------------ camera: background thread
#[derive(Default)]
struct NotifierState {
    events: Mutex<VecDeque<Vec<CameraEvent>>>,
    ready: Flag,
    closed: AtomicBool,
    broken: AtomicBool,
    waits: AtomicUsize,
}

#[derive(Clone, Default)]
struct FakeNotifier(Arc<NotifierState>);

impl FakeNotifier {
    fn push(&self, events: Vec<CameraEvent>) {
        self.0.events.lock().push_back(events);
        self.0.ready.set();
    }

    /// Fail like a descriptor that cannot be read any more (see `FdNotifier::wait`).
    fn break_down(&self) {
        self.0.broken.store(true, Ordering::SeqCst);
        self.push(vec![CameraEvent::new(false, None)]);
    }

    fn closed(&self) -> bool {
        self.0.closed.load(Ordering::SeqCst)
    }

    fn waits(&self) -> usize {
        self.0.waits.load(Ordering::SeqCst)
    }
}

impl CameraNotifier for FakeNotifier {
    fn kind(&self) -> &'static str {
        "fake"
    }

    fn alive(&self) -> bool {
        !self.0.broken.load(Ordering::SeqCst) && !self.closed()
    }

    fn wait(&mut self, timeout: f64) -> Vec<CameraEvent> {
        self.0.waits.fetch_add(1, Ordering::SeqCst);
        if self.0.ready.wait(timeout) {
            *self.0.ready.set.lock() = false;
            return self.0.events.lock().pop_front().unwrap_or_default();
        }
        Vec::new()
    }

    fn close(&mut self) {
        self.0.closed.store(true, Ordering::SeqCst);
    }
}

fn eventually(predicate: impl Fn() -> bool) -> bool {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if predicate() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    predicate()
}

/// A platform with the camera watcher thread and fake notifiers; stopped when dropped.
struct Threaded {
    _fx: Fx,
    tree: ProcTree,
    plat: LinuxPlatform,
    notifiers: Arc<Mutex<Vec<FakeNotifier>>>,
}

impl Drop for Threaded {
    fn drop(&mut self) {
        self.plat.camera.stop();
    }
}

fn threaded() -> Threaded {
    let fx = fx();
    let tree = fx.proc_tree();
    let notifiers: Arc<Mutex<Vec<FakeNotifier>>> = Arc::default();
    let made = notifiers.clone();
    let plat = fx.platform(LinuxOptions {
        proc_root: tree.proc.clone(),
        dev_root: tree.dev.clone(),
        sys_root: tree.sys.clone(),
        camera_notifier_factory: Arc::new(move |_| {
            let notifier = FakeNotifier::default();
            made.lock().push(notifier.clone());
            Some(Box::new(notifier) as Box<dyn CameraNotifier>)
        }),
        ..LinuxOptions::default()
    });
    plat.camera.timings.lock().max_wait_s = 0.02;
    Threaded {
        _fx: fx,
        tree,
        plat,
        notifiers,
    }
}

impl Threaded {
    fn notifier(&self, index: usize) -> Option<FakeNotifier> {
        self.notifiers.lock().get(index).cloned()
    }
}

#[test]
fn test_camera_scan_runs_off_the_calling_thread() {
    let t = threaded();
    t.tree.device("video0", true);
    t.tree.process(100, &["/dev/video0"], "app");
    let scanners: Arc<Mutex<Vec<ThreadId>>> = Arc::default();
    {
        let mut probes = t.plat.camera.probes.lock();
        let scan = probes.scan.clone();
        let recorded = scanners.clone();
        probes.scan = Arc::new(move |devices| {
            recorded.lock().push(std::thread::current().id());
            scan(devices)
        });
    }
    assert!(eventually(
        || t.plat.camera_in_use_by_other_app() == Some(true)
    ));
    assert!(!scanners.lock().is_empty());
    assert!(!scanners.lock().contains(&std::thread::current().id()));
    assert_eq!(t.notifiers.lock().len(), 1); // event-driven: the node is watched
}

#[test]
fn test_camera_thread_rescans_on_open_events() {
    let t = threaded();
    t.tree.device("video0", true);
    assert!(eventually(
        || t.plat.camera_in_use_by_other_app() == Some(false)
    ));
    assert!(eventually(|| t.notifier(0).is_some()));
    t.tree.process(100, &["/dev/video0"], "app"); // an app starts streaming ...
    t.notifier(0)
        .unwrap()
        .push(vec![CameraEvent::new(true, None)]); // ... and is noticed
    assert!(eventually(
        || t.plat.camera_in_use_by_other_app() == Some(true)
    ));
}

#[test]
fn test_camera_thread_stops_when_nobody_asks() {
    let t = threaded();
    t.tree.device("video0", true);
    t.plat.camera.timings.lock().idle_stop_s = 0.1;
    t.plat.camera_in_use_by_other_app();
    assert!(eventually(|| t.plat.camera.state.lock().thread.is_none()));
    assert!(eventually(|| t.notifier(0).is_some_and(|n| n.closed())));
    t.plat.camera_in_use_by_other_app(); // asking again restarts it
    assert!(eventually(|| t.notifiers.lock().len() == 2));
}

/// Resuming after a pause: the forced check must not read the pre-pause answer.
#[test]
fn test_camera_answer_after_an_idle_stop_is_fresh() {
    let t = threaded();
    let watcher = &t.plat.camera;
    watcher.timings.lock().idle_stop_s = 0.1;
    // generous for slow CI; it returns once the scan is done
    watcher.timings.lock().first_scan_wait_s = 10.0;
    t.tree.device("video0", true);
    assert!(eventually(
        || t.plat.camera_in_use_by_other_app() == Some(false)
    ));
    assert!(eventually(|| watcher.state.lock().thread.is_none())); // paused: nobody asked
    t.tree.process(100, &["/dev/video0"], "app"); // a video call started meanwhile
    assert_eq!(t.plat.camera_in_use_by_other_app(), Some(true)); // at once, not one poll later
    t.tree.remove(100);
    assert!(eventually(|| watcher.state.lock().thread.is_none()));
    assert_eq!(t.plat.camera_in_use_by_other_app(), Some(false)); // and the reverse
}

#[test]
fn test_camera_thread_that_dies_early_does_not_stall_the_query() {
    let t = threaded();
    let watcher = &t.plat.camera;
    watcher.timings.lock().first_scan_wait_s = 30.0;
    watcher.probes.lock().devices = Arc::new(|| panic!("sysfs vanished"));
    let started = Instant::now();
    assert_eq!(t.plat.camera_in_use_by_other_app(), None); // unknown, not stale
    assert!(started.elapsed() < Duration::from_secs(5));
}

/// Not the 30 s safety rescan of a working notifier, and no new notifier per failure.
#[test]
fn test_camera_watcher_polls_once_its_notifier_broke() {
    let t = threaded();
    let watcher = &t.plat.camera;
    watcher.timings.lock().poll_s = 0.05;
    watcher.timings.lock().safety_rescan_s = 3600.0;
    t.tree.device("video0", true);
    assert!(eventually(
        || t.plat.camera_in_use_by_other_app() == Some(false)
    ));
    assert!(eventually(|| t.notifier(0).is_some()));
    let notifier = t.notifier(0).unwrap();
    notifier.break_down();
    assert!(eventually(|| notifier.closed()));
    let waits = notifier.waits();
    t.tree.process(100, &["/dev/video0"], "app"); // no event will tell: only polling finds it
    assert!(eventually(
        || t.plat.camera_in_use_by_other_app() == Some(true)
    ));
    assert_eq!(t.notifiers.lock().len(), 1);
    assert_eq!(notifier.waits(), waits); // the dead notifier is not waited on any more
}

#[test]
fn test_fd_notifier_reports_a_failing_descriptor_once() {
    use std::os::fd::IntoRawFd;
    let tmp = tempfile::tempdir().unwrap();
    // A write-only file: reading it fails.
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(tmp.path().join("node"))
        .unwrap();
    let mut notifier = FdNotifier::new(
        file.into_raw_fd(),
        "notifier",
        Box::new(|_| panic!("nothing can be read")),
    );
    assert!(notifier.alive());
    assert_eq!(notifier.wait(0.01), [CameraEvent::new(false, None)]);
    assert!(!notifier.alive()); // closed: the watcher polls from now on
    assert_eq!(notifier.fd, -1);
    assert!(notifier.wait(0.0).is_empty());
}

#[test]
fn test_real_notifier_reports_opens() {
    for kind in ["fanotify", "inotify"] {
        let tmp = tempfile::tempdir().unwrap();
        let node = tmp.path().join("video0");
        std::fs::write(&node, "").unwrap();
        let path = node.to_str().unwrap().to_string();
        let open = if kind == "fanotify" {
            open_fanotify
        } else {
            open_inotify
        };
        let mut notifier = match open(std::slice::from_ref(&path)) {
            Ok(notifier) => notifier,
            Err(e) => {
                eprintln!("skipped: {kind} unavailable here: {e}");
                continue;
            }
        };
        drop(std::fs::File::open(&node).unwrap());
        // The child opens read-write, like a capture attempt.
        let status = std::process::Command::new("sh")
            .args(["-c", "exec 3<>\"$1\"", "sh", &path])
            .status()
            .unwrap();
        assert!(status.success());
        let mut events = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline && !events.iter().any(|e: &CameraEvent| e.wrote) {
            events.extend(notifier.wait(0.2));
        }
        notifier.close();
        let opens: Vec<_> = events
            .iter()
            .filter(|e| e.opened)
            .map(|e| e.foreign)
            .collect();
        let writes: Vec<_> = events
            .iter()
            .filter(|e| e.wrote)
            .map(|e| e.foreign)
            .collect();
        if kind == "fanotify" {
            assert_eq!(opens, [Some(false), Some(true)]); // ours, then the child process's
            assert_eq!(writes, [Some(true)]);
        } else {
            assert_eq!(opens, [None, None]);
            assert_eq!(writes, [None]);
        }
    }
}

// ------------------------------------------------------------------ EWMH windows
type SentEvent = ((&'static str, u32, u32, [u32; 5]), u32);

struct FakeWindow {
    props: HashMap<String, Vec<u32>>,
    geometry: (i32, i32, i32, i32),
    map_state: u8,
    wm_class: Option<(String, String)>,
}

struct DisplayState {
    atoms: HashMap<String, u32>,
    atom_names: HashMap<u32, String>,
    windows: HashMap<u32, FakeWindow>,
    sent: Vec<SentEvent>,
    flushed: u32,
    /// The root window's properties fail like a connection that went away.
    broken: bool,
    extensions: HashSet<String>,
    dpms_capable: bool,
    dpms_enabled: bool,
    power_level: u16,
    dpms_log: Vec<String>,
    motions: Vec<i16>,
    pointer: (i32, i32),
    warps: Vec<(i32, i32)>,
}

const ROOT: u32 = 1;
const OWN_PID: u32 = 4242;

#[derive(Clone)]
struct FakeDisplay(Arc<Mutex<DisplayState>>);

impl FakeDisplay {
    fn new() -> Self {
        let root = FakeWindow {
            props: HashMap::new(),
            geometry: (0, 0, 100, 100),
            map_state: X_IS_VIEWABLE,
            wm_class: None,
        };
        Self(Arc::new(Mutex::new(DisplayState {
            atoms: HashMap::new(),
            atom_names: HashMap::new(),
            windows: HashMap::from([(ROOT, root)]),
            sent: Vec::new(),
            flushed: 0,
            broken: false,
            extensions: HashSet::from(["DPMS".to_string()]),
            dpms_capable: true,
            dpms_enabled: true,
            power_level: 0,
            dpms_log: Vec::new(),
            motions: Vec::new(),
            pointer: (0, 0),
            warps: Vec::new(),
        })))
    }

    fn client(&self, clock: &TestClock) -> EwmhClient {
        let display = self.clone();
        EwmhClient::new(
            Some(Box::new(move || {
                Ok(Box::new(display.clone()) as Box<dyn XServer>)
            })),
            clock.clock(),
            Some(OWN_PID),
        )
    }

    fn intern(&self, name: &str) -> u32 {
        let mut state = self.0.lock();
        intern(&mut state, name)
    }

    fn add(&self, wid: u32, rect: (i32, i32, i32, i32), pid: u32) {
        let window = FakeWindow {
            props: HashMap::from([("_NET_WM_PID".to_string(), vec![pid])]),
            geometry: rect,
            map_state: X_IS_VIEWABLE,
            wm_class: None,
        };
        self.0.lock().windows.insert(wid, window);
    }

    fn set_prop(&self, wid: u32, name: &str, values: &[u32]) {
        self.0
            .lock()
            .windows
            .get_mut(&wid)
            .unwrap()
            .props
            .insert(name.into(), values.to_vec());
    }

    fn set_atoms(&self, wid: u32, name: &str, atoms: &[&str]) {
        let values: Vec<u32> = atoms.iter().map(|atom| self.intern(atom)).collect();
        self.set_prop(wid, name, &values);
    }

    fn stack(&self, wids: &[u32]) {
        self.set_prop(ROOT, "_NET_CLIENT_LIST_STACKING", wids);
    }

    fn window<T>(&self, wid: u32, f: impl FnOnce(&mut FakeWindow) -> T) -> T {
        f(self.0.lock().windows.get_mut(&wid).unwrap())
    }
}

fn intern(state: &mut DisplayState, name: &str) -> u32 {
    if let Some(atom) = state.atoms.get(name) {
        return *atom;
    }
    let atom = state.atoms.len() as u32 + 100;
    state.atoms.insert(name.into(), atom);
    state.atom_names.insert(atom, name.into());
    atom
}

fn bad_window(wid: u32) -> XError {
    XError::Request(format!("BadWindow {wid}"))
}

impl XServer for FakeDisplay {
    fn root(&self) -> u32 {
        ROOT
    }

    fn intern_atom(&self, name: &str) -> XResult<u32> {
        Ok(intern(&mut self.0.lock(), name))
    }

    fn property(&self, window: u32, atom: u32) -> XResult<Option<Vec<u32>>> {
        let state = self.0.lock();
        if window == ROOT && state.broken {
            return Err(XError::Connection("X server went away".into()));
        }
        let win = state
            .windows
            .get(&window)
            .ok_or_else(|| bad_window(window))?;
        let name = &state.atom_names[&atom];
        Ok(win.props.get(name).cloned())
    }

    fn geometry(&self, window: u32) -> XResult<(i32, i32)> {
        let state = self.0.lock();
        let win = state
            .windows
            .get(&window)
            .ok_or_else(|| bad_window(window))?;
        Ok((win.geometry.2, win.geometry.3))
    }

    fn map_state(&self, window: u32) -> XResult<u8> {
        let state = self.0.lock();
        Ok(state
            .windows
            .get(&window)
            .ok_or_else(|| bad_window(window))?
            .map_state)
    }

    fn root_origin(&self, window: u32) -> XResult<(i32, i32)> {
        let state = self.0.lock();
        let win = state
            .windows
            .get(&window)
            .ok_or_else(|| bad_window(window))?;
        Ok((win.geometry.0, win.geometry.1))
    }

    fn wm_class(&self, window: u32) -> XResult<Option<(String, String)>> {
        let state = self.0.lock();
        Ok(state
            .windows
            .get(&window)
            .ok_or_else(|| bad_window(window))?
            .wm_class
            .clone())
    }

    fn send_client_message(
        &self,
        window: u32,
        message_type: u32,
        data: [u32; 5],
        event_mask: u32,
    ) -> XResult<()> {
        let event = (("ClientMessage", window, message_type, data), event_mask);
        self.0.lock().sent.push(event);
        Ok(())
    }

    fn flush(&self) -> XResult<()> {
        self.0.lock().flushed += 1;
        Ok(())
    }

    fn has_extension(&self, name: &str) -> XResult<bool> {
        Ok(self.0.lock().extensions.contains(name))
    }

    fn fake_relative_motion(&self, dx: i16) -> XResult<()> {
        self.0.lock().motions.push(dx);
        Ok(())
    }

    fn query_pointer(&self) -> XResult<(i32, i32)> {
        Ok(self.0.lock().pointer)
    }

    fn warp_pointer(&self, x: i32, y: i32) -> XResult<()> {
        let mut state = self.0.lock();
        state.warps.push((x, y));
        state.pointer = (x, y);
        Ok(())
    }

    fn sync(&self) -> XResult<()> {
        Ok(())
    }

    fn dpms_capable(&self) -> XResult<bool> {
        Ok(self.0.lock().dpms_capable)
    }

    fn dpms_info(&self) -> XResult<(u16, bool)> {
        let state = self.0.lock();
        Ok((state.power_level, state.dpms_enabled))
    }

    fn dpms_set_enabled(&self, enabled: bool) -> XResult<()> {
        let mut state = self.0.lock();
        state
            .dpms_log
            .push(if enabled { "enable" } else { "disable" }.into());
        state.dpms_enabled = enabled;
        if !enabled {
            state.power_level = 0; // disabling DPMS turns the monitors back on
        }
        Ok(())
    }

    fn dpms_force_level(&self, level: u16) -> XResult<()> {
        let mut state = self.0.lock();
        if !state.dpms_enabled {
            return Err(XError::Request("BadMatch: DPMS is disabled".into()));
        }
        state.dpms_log.push(format!("force {level}"));
        state.power_level = level;
        Ok(())
    }
}

/// The `x11_plat` fixture: an X11 session on a fake X server.
fn x11_fx() -> (Fx, FakeDisplay) {
    let mut fx = fx();
    fx.x11(None);
    let xdisplay = FakeDisplay::new();
    fx.plat.ewmh = Arc::new(xdisplay.client(&fx.clock));
    (fx, xdisplay)
}

fn x11_window(wid: u32) -> WindowRef {
    WindowRef {
        handle: WindowHandle::X11(wid),
        pid: None,
        rect: None,
    }
}

/// A handle of another platform: never an X11 window.
fn foreign_window() -> WindowRef {
    WindowRef {
        handle: WindowHandle::Hwnd(7),
        pid: None,
        rect: None,
    }
}

#[test]
fn test_window_at_prefers_topmost_window() {
    let (fx, xdisplay) = x11_fx();
    xdisplay.add(10, (0, 0, 1920, 1080), 500);
    xdisplay.add(20, (100, 100, 800, 600), 501);
    xdisplay.add(30, (2000, 0, 800, 600), 502); // other monitor
    xdisplay.stack(&[10, 20, 30]); // bottom → top
    let found = fx.plat.window_at(200, 200).unwrap();
    assert_eq!(found.handle, WindowHandle::X11(20));
    assert_eq!(found.pid, Some(501));
    assert_eq!(found.rect, Some(Rect::new(100, 100, 800, 600)));
    assert_eq!(
        fx.plat.window_at(2100, 50).unwrap().handle,
        WindowHandle::X11(30)
    );
    assert!(fx.plat.window_at(5000, 5000).is_none());
}

#[test]
fn test_window_at_skips_hidden_unmapped_shell_and_own_windows() {
    let (fx, xdisplay) = x11_fx();
    xdisplay.add(5, (0, 0, 1920, 1080), 500);
    xdisplay.set_atoms(5, "_NET_WM_WINDOW_TYPE", &["_NET_WM_WINDOW_TYPE_DESKTOP"]);
    xdisplay.add(10, (0, 0, 1000, 1000), 501);
    xdisplay.add(20, (0, 0, 1000, 1000), 500);
    xdisplay.set_atoms(20, "_NET_WM_STATE", &["_NET_WM_STATE_HIDDEN"]);
    xdisplay.add(30, (0, 0, 1000, 1000), 500);
    xdisplay.window(30, |w| w.map_state = 1); // IsUnviewable: other workspace
    xdisplay.add(40, (0, 0, 1000, 1000), OWN_PID); // our overlay
    xdisplay.add(50, (0, 0, 1920, 40), 500);
    xdisplay.set_atoms(50, "_NET_WM_WINDOW_TYPE", &["_NET_WM_WINDOW_TYPE_DOCK"]);
    xdisplay.stack(&[5, 10, 20, 30, 40, 50, 99]); // 99 vanished between listing and query
    let found = fx.plat.window_at(10, 10).unwrap();
    assert_eq!(found.handle, WindowHandle::X11(10));
    xdisplay.0.lock().windows.remove(&10);
    assert!(fx.plat.window_at(10, 10).is_none()); // only the desktop is left
}

#[test]
fn test_window_rect_includes_decorations_and_drops_csd_shadow() {
    let (fx, xdisplay) = x11_fx();
    xdisplay.add(10, (100, 130, 800, 600), 500);
    xdisplay.set_prop(10, "_NET_FRAME_EXTENTS", &[2, 2, 30, 2]);
    xdisplay.add(20, (0, 0, 848, 648), 500);
    xdisplay.set_prop(20, "_GTK_FRAME_EXTENTS", &[24, 24, 24, 24]);
    assert_eq!(
        fx.plat.window_rect(&x11_window(10)),
        Some(Rect::new(98, 100, 804, 632))
    );
    assert_eq!(
        fx.plat.window_rect(&x11_window(20)),
        Some(Rect::new(24, 24, 800, 600))
    );
}

#[test]
fn test_window_client_rect_excludes_decorations_and_csd_shadow() {
    let (fx, xdisplay) = x11_fx();
    xdisplay.add(10, (100, 130, 800, 600), 500);
    xdisplay.set_prop(10, "_NET_FRAME_EXTENTS", &[2, 2, 30, 2]);
    xdisplay.add(20, (0, 0, 848, 648), 500);
    xdisplay.set_prop(20, "_GTK_FRAME_EXTENTS", &[24, 24, 24, 24]);
    assert_eq!(
        fx.plat.window_client_rect(&x11_window(10)),
        Some(Rect::new(100, 130, 800, 600))
    );
    assert_eq!(
        fx.plat.window_client_rect(&x11_window(20)),
        Some(Rect::new(24, 24, 800, 600))
    );
    assert_eq!(fx.plat.window_client_rect(&x11_window(99)), None); // gone
    assert_eq!(fx.plat.window_client_rect(&foreign_window()), None);
}

#[test]
fn test_window_app_reads_wm_class_and_process() {
    let (mut fx, xdisplay) = x11_fx();
    xdisplay.add(10, (0, 0, 800, 600), 501);
    xdisplay.window(10, |w| w.wm_class = Some(("kitty".into(), "kitty".into())));
    xdisplay.add(20, (0, 0, 800, 600), 502); // no WM_CLASS
    let names = HashMap::from([(501_i64, "kitty"), (502, "xterm")]);
    fx.plat.hooks.app_process_name = Some(Box::new(move |pid| {
        pid.and_then(|pid| names.get(&pid))
            .map(|name| name.to_string())
    }));
    let identity = |process: &str, app_id: &str| AppIdentity {
        process: process.into(),
        app_id: app_id.into(),
    };
    let with_pid = |wid: u32, pid: u32| WindowRef {
        pid: Some(pid),
        ..x11_window(wid)
    };
    assert_eq!(
        fx.plat.window_app(&with_pid(10, 501)),
        Some(identity("kitty", "kitty"))
    );
    assert_eq!(
        fx.plat.window_app(&x11_window(20)),
        Some(identity("xterm", ""))
    );
    // another client's id now
    assert_eq!(fx.plat.window_app(&with_pid(10, 777)), None);
    assert_eq!(fx.plat.window_app(&x11_window(99)), None);
}

#[test]
fn test_pane_queries_need_x11() {
    let fx = fx();
    fx.wayland(Some("GNOME"), false);
    assert_eq!(fx.plat.window_client_rect(&x11_window(10)), None);
    assert_eq!(fx.plat.window_app(&x11_window(10)), None);
}

#[test]
fn test_foreground_window() {
    let (fx, xdisplay) = x11_fx();
    xdisplay.add(20, (100, 100, 800, 600), 501);
    xdisplay.add(40, (0, 0, 10, 10), OWN_PID);
    xdisplay.set_prop(ROOT, "_NET_ACTIVE_WINDOW", &[20]);
    let found = fx.plat.foreground_window().unwrap();
    assert_eq!(
        (found.handle, found.pid, found.rect),
        (
            WindowHandle::X11(20),
            Some(501),
            Some(Rect::new(100, 100, 800, 600))
        )
    );
    xdisplay.set_prop(ROOT, "_NET_ACTIVE_WINDOW", &[40]);
    assert!(fx.plat.foreground_window().is_none()); // our own window
    xdisplay.set_prop(ROOT, "_NET_ACTIVE_WINDOW", &[0]);
    assert!(fx.plat.foreground_window().is_none());
}

#[test]
fn test_activate_sends_net_active_window() {
    let (fx, xdisplay) = x11_fx();
    xdisplay.add(20, (100, 100, 800, 600), 500);
    assert!(fx.plat.activate_window(&x11_window(20)));
    let atom = xdisplay.intern("_NET_ACTIVE_WINDOW");
    let state = xdisplay.0.lock();
    assert_eq!(
        state.sent,
        [(
            ("ClientMessage", 20, atom, [2, 0, 0, 0, 0]),
            (1 << 20) | (1 << 19)
        )]
    );
    assert_eq!(state.flushed, 1);
}

#[test]
fn test_activate_refuses_minimised_or_missing_windows() {
    let (fx, xdisplay) = x11_fx();
    xdisplay.add(20, (0, 0, 10, 10), 500);
    xdisplay.set_atoms(20, "_NET_WM_STATE", &["_NET_WM_STATE_HIDDEN"]);
    assert!(!fx.plat.activate_window(&x11_window(20)));
    assert!(!fx.plat.activate_window(&x11_window(77)));
    assert!(!fx.plat.activate_window(&foreign_window()));
    assert!(xdisplay.0.lock().sent.is_empty());
    assert!(!fx.plat.is_window_valid(&x11_window(20)));
}

#[test]
fn test_window_functions_unsupported_on_wayland() {
    let (fx, xdisplay) = x11_fx();
    xdisplay.add(20, (0, 0, 800, 600), 500);
    xdisplay.stack(&[20]);
    xdisplay.set_prop(ROOT, "_NET_ACTIVE_WINDOW", &[20]);
    fx.wayland(Some("GNOME"), true); // XWayland only sees X11 clients: never trust it
    assert!(fx.plat.foreground_window().is_none());
    assert!(fx.plat.window_at(10, 10).is_none());
    assert!(!fx.plat.activate_window(&x11_window(20)));
    assert_eq!(fx.plat.window_rect(&x11_window(20)), None);
    assert!(xdisplay.0.lock().sent.is_empty());
}

#[test]
fn test_x11_connection_failure_backs_off() {
    let mut fx = fx();
    fx.x11(None);
    let attempts = Arc::new(AtomicUsize::new(0));
    let counted = attempts.clone();
    fx.plat.ewmh = Arc::new(EwmhClient::new(
        Some(Box::new(move || {
            counted.fetch_add(1, Ordering::SeqCst);
            Err(XError::Connection("Can't connect to display".into()))
        })),
        fx.clock.clock(),
        None,
    ));
    assert!(fx.plat.foreground_window().is_none());
    assert!(fx.plat.window_at(0, 0).is_none());
    assert_eq!(attempts.load(Ordering::SeqCst), 1);
    fx.clock.advance(61.0);
    assert!(fx.plat.foreground_window().is_none());
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
}

#[test]
fn test_x11_connection_loss_reconnects() {
    let (fx, xdisplay) = x11_fx();
    xdisplay.add(20, (0, 0, 800, 600), 500);
    xdisplay.set_prop(ROOT, "_NET_ACTIVE_WINDOW", &[20]);
    assert!(fx.plat.foreground_window().is_some());
    xdisplay.0.lock().broken = true;
    assert!(fx.plat.foreground_window().is_none());
    assert!(fx.plat.ewmh.state.lock().display.is_none());
    xdisplay.0.lock().broken = false;
    fx.clock.advance(2.0);
    assert!(fx.plat.foreground_window().is_some());
}

// ------------------------------------------------------------------ capabilities
fn capability_keys() -> HashSet<&'static str> {
    GenericPlatform.capabilities().into_keys().collect()
}

#[test]
fn test_capabilities_x11() {
    let mut fx = fx();
    fx.x11(None);
    fx.tools.install(&["loginctl", "xset"]);
    fx.plat.xss = Box::new(FakeXss(Some(0)));
    let caps = fx.plat.capabilities();
    assert_eq!(
        caps.keys().copied().collect::<HashSet<_>>(),
        capability_keys()
    );
    for key in [
        "lock",
        "display_off",
        "input_idle",
        "session_locked",
        "focus",
        "cursor",
        "hotkeys",
        "panes",
    ] {
        assert!(caps[key], "{key}");
    }
    assert!(!caps["key_idle"]);
}

#[test]
fn test_capabilities_wayland() {
    let fx = fx();
    fx.wayland(Some("GNOME"), false);
    fx.tools.install(&["busctl", "gdbus"]);
    let caps = fx.plat.capabilities();
    assert_eq!(
        caps.keys().copied().collect::<HashSet<_>>(),
        capability_keys()
    );
    assert!(caps["display_off"]);
    assert!(caps["input_idle"]);
    for key in ["focus", "cursor", "hotkeys", "panes"] {
        assert!(!caps[key], "{key}");
    }
    fx.ydotool_ready();
    assert!(fx.plat.capabilities()["cursor"]);
}

#[test]
fn test_module_imports_cleanly_everywhere() {
    let fx = fx();
    assert_eq!(fx.plat.name(), "linux");
}
