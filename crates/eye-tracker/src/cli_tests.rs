//! Tests of the command-line interface (tests/test_cli.py).
//!
//! The tray app itself is never started here (`Env::run_app` is replaced and records the
//! [`RunArgs`] it gets); the other subcommands run for real against a temporary config
//! directory. Autostart is faked so the real login entry is never touched, and so is the start of
//! the detached process. Output goes to buffers instead of `capsys`.
//!
//! `sys.executable` / `sys.frozen` / `sys.platform` / `shutil.which` are replaced by the
//! [`Context`] the commands are resolved from (and, for the detached start, by `Env::exe` and
//! `Env::os`).
//!
//! Not ported:
//! * `test_shown_commands_from_source`: there is no source install, so no `python -m eye_tracker`
//!   form.
//! * `test_python_dash_m_entry_point`: there is no `python -m`; the binaries are thin wrappers
//!   around `cli::main` / `cli::gui_main`.
//! * `test_keyboard_interrupt_exit_code`: Rust has no `KeyboardInterrupt` to raise from the
//!   fake `run_app` (a Ctrl+C ends the process); `EXIT_INTERRUPTED` is only the constant
//!   `run_app` may return.
//! * `test_help_examples_use_the_program_name` as a whole-command test: `program_name()` is that
//!   of the test executable. The examples are checked on `epilog` with the Windows console name.
//! * The `first.command is None` / `third.command == "run"` assertions of `test_run_is_the_default`:
//!   `RunArgs` has no `command` field.
//! * `stdout is subprocess.DEVNULL` of the detached start: the redirection happens inside
//!   `spawn_detached`, which is not run here (it would start a program).
//! * `test_ctl_with_an_instance_that_does_not_answer` replaces `server._handler` and connects a
//!   slot that never replies; here the event receiver is held and never answered.
//! * `test_doctor_json` checks the `ui` section where the Python report had `qt`.

use std::cell::{Cell, RefCell};
use std::fs;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::json;

use crate::config::Settings;
use crate::ipc::{Event, InstanceLock, InstanceServer};
use crate::platform::autostart::AutostartError;
use crate::testing::{AppDirs, app_dirs};

use super::*;

/// A release package whose folder holds the given executables; the first is the running one.
struct Package {
    _dir: tempfile::TempDir,
    folder: PathBuf,
    exe: PathBuf,
}

fn package(running: &str, others: &[&str], folder_name: &str) -> Package {
    let dir = tempfile::tempdir().unwrap();
    let folder = dir.path().join(folder_name);
    fs::create_dir(&folder).unwrap();
    for name in std::iter::once(&running).chain(others) {
        fs::write(folder.join(name), b"").unwrap();
    }
    let exe = folder.join(running);
    Package {
        _dir: dir,
        folder,
        exe,
    }
}

fn path_of(p: &Package, name: &str) -> String {
    p.folder.join(name).to_string_lossy().into_owned()
}

fn no_path(_: &str) -> Option<PathBuf> {
    None
}

fn context<'a>(
    exe: &'a Path,
    os: &'a str,
    which: &'a dyn Fn(&str) -> Option<PathBuf>,
) -> Context<'a> {
    Context {
        exe,
        os,
        appimage: "",
        which,
        profile: None,
    }
}

fn strings(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|s| (*s).to_string()).collect()
}

#[test]
fn test_shown_commands_name_the_console_executable_of_the_windows_package() {
    let p = package("EyeTracker.exe", &["eye-tracker-cli.exe"], "Eye Tracker");
    let (cli_exe, gui_exe) = (
        path_of(&p, "eye-tracker-cli.exe"),
        path_of(&p, "EyeTracker.exe"),
    );
    let ctx = context(&p.exe, "linux", &no_path);
    // The windowed app has no console: commands it shows use the console twin.
    assert_eq!(
        cli_command_in(&ctx, &["ctl", "toggle"]),
        [cli_exe.as_str(), "ctl", "toggle"]
    );
    assert_eq!(app_command_in(&ctx), [gui_exe.as_str()]);
    assert_eq!(program_name_in(&ctx), "eye-tracker-cli.exe");

    // Run as the console executable: the same commands, and the app is started with the
    // windowed one (not tied to the terminal).
    let p = package("eye-tracker-cli.exe", &["EyeTracker.exe"], "Eye Tracker");
    let (cli_exe, gui_exe) = (
        path_of(&p, "eye-tracker-cli.exe"),
        path_of(&p, "EyeTracker.exe"),
    );
    let ctx = context(&p.exe, "linux", &no_path);
    assert_eq!(
        cli_command_in(&ctx, &["ctl", "quit"]),
        [cli_exe.as_str(), "ctl", "quit"]
    );
    assert_eq!(app_command_in(&ctx), [gui_exe.as_str()]);
}

#[test]
fn test_shown_commands_use_the_installers_command_on_the_path() {
    // The Windows installer adds eye-tracker.exe (a copy of eye-tracker-cli.exe) and puts its
    // folder on the PATH: hints then show the short command the docs use.
    let p = package(
        "EyeTracker.exe",
        &["eye-tracker-cli.exe", "eye-tracker.exe"],
        "Eye Tracker",
    );
    let gui_exe = path_of(&p, "EyeTracker.exe");
    let on_path = std::cell::RefCell::new(Some(p.folder.join("eye-tracker.exe")));
    let which = |_: &str| on_path.borrow().clone();
    let ctx = context(&p.exe, "windows", &which);
    assert_eq!(
        cli_command_in(&ctx, &["ctl", "toggle"]),
        ["eye-tracker", "ctl", "toggle"]
    );
    assert_eq!(program_name_in(&ctx), "eye-tracker");
    // The app is still started windowed, never through the console command.
    assert_eq!(app_command_in(&ctx), [gui_exe.as_str()]);

    let q = package("eye-tracker.exe", &["EyeTracker.exe"], "Eye Tracker");
    let gui_exe = path_of(&q, "EyeTracker.exe");
    *on_path.borrow_mut() = Some(q.exe.clone());
    let ctx = context(&q.exe, "windows", &which);
    assert_eq!(cli_command_in(&ctx, &["doctor"]), ["eye-tracker", "doctor"]);
    assert_eq!(app_command_in(&ctx), [gui_exe.as_str()]);

    // PATH option unticked (or not seen yet by this process): the full path.
    let ctx = context(&p.exe, "windows", &no_path);
    assert_eq!(
        cli_command_in(&ctx, &[]),
        [path_of(&p, "eye-tracker-cli.exe")]
    );
    // Another copy's eye-tracker on the PATH would control the wrong installation.
    let other = tempfile::tempdir().unwrap();
    let other_exe = other.path().join("eye-tracker.exe");
    fs::write(&other_exe, b"").unwrap();
    let elsewhere = |_: &str| Some(other_exe.clone());
    let ctx = context(&p.exe, "windows", &elsewhere);
    assert_eq!(
        cli_command_in(&ctx, &[]),
        [path_of(&p, "eye-tracker-cli.exe")]
    );
}

#[test]
fn test_shown_commands_in_the_macos_and_linux_packages() {
    // Eye Tracker.app/Contents/MacOS
    let p = package("eye-tracker-cli", &["Eye Tracker"], "Eye Tracker");
    let ctx = context(&p.exe, "macos", &no_path);
    assert_eq!(cli_command_in(&ctx, &[]), [path_of(&p, "eye-tracker-cli")]);
    assert_eq!(app_command_in(&ctx), [path_of(&p, "Eye Tracker")]);

    // one executable for both
    let tarball = package("eye-tracker", &[], "eye-tracker");
    let ctx = context(&tarball.exe, "linux", &no_path);
    assert_eq!(
        cli_command_in(&ctx, &["doctor"]),
        [path_of(&tarball, "eye-tracker").as_str(), "doctor"]
    );
    assert_eq!(app_command_in(&ctx), [path_of(&tarball, "eye-tracker")]);
}

#[test]
fn test_shown_commands_inside_an_appimage() {
    // the bundle inside the image, mounted at a random path
    let p = package("eye-tracker", &[], "Eye Tracker");
    let image_dir = tempfile::tempdir().unwrap();
    let image = image_dir.path().join("Eye_Tracker-x86_64.AppImage");
    fs::write(&image, b"").unwrap();
    let image_text = image.to_string_lossy().into_owned();
    let mut ctx = context(&p.exe, "linux", &no_path);
    ctx.appimage = &image_text;
    assert_eq!(
        cli_command_in(&ctx, &["ctl", "toggle"]),
        [image_text.as_str(), "ctl", "toggle"]
    );
    assert_eq!(app_command_in(&ctx), [image_text.as_str()]);
    assert_eq!(program_name_in(&ctx), "Eye_Tracker-x86_64.AppImage");
    // A stale variable (inherited from another AppImage) naming no file is ignored.
    let gone = image_dir.path().join("gone.AppImage");
    let gone_text = gone.to_string_lossy().into_owned();
    ctx.appimage = &gone_text;
    assert!(cli_command_in(&ctx, &[])[0].ends_with("eye-tracker"));
}

#[test]
fn test_shown_commands_keep_the_profile() {
    // Each --config-dir profile is its own instance: ctl must address this one.
    let p = package("eye-tracker", &[], "Eye Tracker");
    let exe = path_of(&p, "eye-tracker");
    let profile = tempfile::tempdir().unwrap();
    let profile_text = profile.path().to_string_lossy().into_owned();
    let mut ctx = context(&p.exe, "linux", &no_path);
    ctx.profile = Some(profile.path());
    assert_eq!(
        cli_command_in(&ctx, &["ctl", "toggle"]),
        [exe.as_str(), "--config-dir", &profile_text, "ctl", "toggle"]
    );
    assert_eq!(
        app_command_in(&ctx),
        [exe.as_str(), "--config-dir", &profile_text]
    );
}

#[test]
fn test_ctl_says_how_to_start_this_copy() {
    // The command text of the "not running" message: the windowed executable and the profile.
    let p = package("eye-tracker-cli.exe", &["EyeTracker.exe"], "Eye Tracker");
    let profile = tempfile::tempdir().unwrap();
    let mut ctx = context(&p.exe, "windows", &no_path);
    ctx.profile = Some(profile.path());
    let gui_exe = path_of(&p, "EyeTracker.exe");
    let profile_text = profile.path().to_string_lossy().into_owned();
    assert_eq!(
        app_command_in(&ctx),
        [gui_exe.as_str(), "--config-dir", &profile_text]
    );

    // And the message itself, for this copy of the test executable.
    let fixture = Fixture::new();
    let outcome = fixture.run(&["ctl", "toggle"]);
    assert_eq!(outcome.code, EXIT_NOT_RUNNING);
    let expected = format_command(&app_command());
    assert!(
        outcome
            .err
            .contains(&format!("Start it with '{expected}'.")),
        "{}",
        outcome.err
    );
    // The profile as the app resolved it: on Windows that also expands 8.3 names such as
    // RUNNER~1 in the temporary directory.
    let profile = paths::base_override().expect("profile");
    assert!(expected.contains(&profile.display().to_string()));
}

#[test]
fn test_format_command_quotes_for_the_shell() {
    let parts = strings(&["/opt/Eye Tracker/eye-tracker", "it's", "", "plain-1.0"]);
    assert_eq!(
        format_command_for(&parts, false),
        "'/opt/Eye Tracker/eye-tracker' 'it'\"'\"'s' '' plain-1.0"
    );
    let parts = strings(&["C:\\Eye Tracker\\EyeTracker.exe", "a\"b", "", "dir\\ "]);
    assert_eq!(
        format_command_for(&parts, true),
        "\"C:\\Eye Tracker\\EyeTracker.exe\" a\\\"b \"\" \"dir\\ \""
    );
}

// ---------------------------------------------------------------------- fixtures
struct Outcome {
    code: i32,
    out: String,
    err: String,
}

/// A copy of a [`RunArgs`] (the real struct need not be `Clone`).
fn copy(args: &RunArgs) -> RunArgs {
    RunArgs {
        config_dir: args.config_dir.clone(),
        log_level: args.log_level.clone(),
        camera: args.camera.clone(),
        backend: args.backend.clone(),
        background: args.background,
        calibrate: args.calibrate,
        trace: args.trace.clone(),
    }
}

/// What the fake login entry holds and was asked (`FakeAutostart`).
struct FakeState {
    enabled: Cell<bool>,
    calls: RefCell<Vec<String>>,
    supported: Cell<bool>,
    fail: RefCell<Option<String>>,
    /// What `status` reports while not enabled.
    idle_status: Cell<Status>,
    /// The `config_dir` of every status/enable/disable call.
    profiles: RefCell<Vec<Option<PathBuf>>>,
    /// What `registered_command` reports (`None`: this copy when enabled).
    registered: RefCell<Option<Vec<String>>>,
}

const FAKE_EXE: &str = "C:\\Program Files\\Eye Tracker\\EyeTracker.exe";

struct FakeAutostart(Rc<FakeState>);

impl AutostartBackend for FakeAutostart {
    fn is_supported(&self) -> bool {
        self.0.supported.get()
    }

    fn is_enabled(&self) -> bool {
        self.0.enabled.get()
    }

    fn enable(&self, background: bool) -> Result<(), Box<dyn std::error::Error>> {
        self.enable_for_profile(background, None)
            .map_err(Into::into)
    }

    fn disable(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.disable_for_profile(None).map_err(Into::into)
    }

    fn status_for_profile(&self, config_dir: Option<&Path>) -> Status {
        self.0
            .profiles
            .borrow_mut()
            .push(config_dir.map(Path::to_path_buf));
        if self.0.enabled.get() {
            Status::Enabled
        } else {
            self.0.idle_status.get()
        }
    }

    fn enable_for_profile(
        &self,
        background: bool,
        config_dir: Option<&Path>,
    ) -> Result<(), AutostartError> {
        self.0
            .calls
            .borrow_mut()
            .push(format!("enable(background={background})"));
        self.0
            .profiles
            .borrow_mut()
            .push(config_dir.map(Path::to_path_buf));
        if let Some(message) = self.0.fail.borrow().clone() {
            return Err(AutostartError(message));
        }
        self.0.enabled.set(true);
        Ok(())
    }

    fn disable_for_profile(&self, config_dir: Option<&Path>) -> Result<(), AutostartError> {
        self.0.calls.borrow_mut().push("disable".to_string());
        self.0
            .profiles
            .borrow_mut()
            .push(config_dir.map(Path::to_path_buf));
        self.0.enabled.set(false);
        Ok(())
    }

    fn location(&self) -> Option<String> {
        Some("HKCU\\Run\\EyeTracker".to_string())
    }

    fn registered_command(&self) -> Option<Vec<String>> {
        if let Some(registered) = self.0.registered.borrow().clone() {
            return Some(registered);
        }
        self.0
            .enabled
            .get()
            .then(|| self.launch_command(true, None))
    }

    fn launch_command(&self, background: bool, config_dir: Option<&Path>) -> Vec<String> {
        let mut command = vec![FAKE_EXE.to_string()];
        if let Some(dir) = config_dir {
            command.extend(["--config-dir".to_string(), dir.display().to_string()]);
        }
        if background {
            command.push("--background".to_string());
        }
        command
    }
}

thread_local! {
    /// The `(level, console, log_to_file)` of every `setup_logging` call of this test.
    static LOGGING: RefCell<Vec<(String, bool, bool)>> = const { RefCell::new(Vec::new()) };
}

fn record_logging(level: &str, console: bool, log_to_file: bool) {
    LOGGING.with_borrow_mut(|calls| calls.push((level.to_string(), console, log_to_file)));
}

/// The `env` fixture (a temporary config directory) with everything the commands could start
/// or touch replaced. The process-wide logger is not started (it would capture the records of
/// other tests, see `logging_setup_tests.rs`). The instance lock file, which can live in the shared temporary
/// directory, is removed afterwards.
struct Fixture {
    dirs: AppDirs,
    run_calls: Rc<RefCell<Vec<RunArgs>>>,
    started: Rc<RefCell<Vec<Vec<String>>>>,
    fake: Rc<FakeState>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_file(InstanceLock::new(None).path());
    }
}

impl Fixture {
    fn new() -> Self {
        Fixture {
            dirs: app_dirs(),
            run_calls: Rc::default(),
            started: Rc::default(),
            fake: Rc::new(FakeState {
                enabled: Cell::new(false),
                calls: RefCell::default(),
                supported: Cell::new(true),
                fail: RefCell::default(),
                idle_status: Cell::new(Status::Disabled),
                profiles: RefCell::default(),
                registered: RefCell::default(),
            }),
        }
    }

    fn run(&self, argv: &[&str]) -> Outcome {
        self.run_with(argv, |_| {})
    }

    /// Runs the command line; `setup` changes the environment before it does.
    fn run_with(&self, argv: &[&str], setup: impl FnOnce(&mut Env)) -> Outcome {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let code = {
            let mut env = Env::process(&mut out, &mut err);
            let calls = Rc::clone(&self.run_calls);
            env.run_app = Box::new(move |args| {
                calls.borrow_mut().push(copy(args));
                0
            });
            env.autostart = Box::new(FakeAutostart(Rc::clone(&self.fake)));
            env.setup_logging = record_logging;
            let started = Rc::clone(&self.started);
            env.spawn = Box::new(move |command| {
                started.borrow_mut().push(command.to_vec());
                Ok(())
            });
            setup(&mut env);
            run_cli(
                std::iter::once(APP_SLUG).chain(argv.iter().copied()),
                &mut env,
            )
        };
        Outcome {
            code,
            out: String::from_utf8_lossy(&out).into_owned(),
            err: String::from_utf8_lossy(&err).into_owned(),
        }
    }

    fn profile(&self) -> String {
        paths::base_override().unwrap().display().to_string()
    }
}

/// An instance whose owner answers every command with `handler(command)`.
fn serve(handler: impl Fn(&str) -> String + Send + 'static) -> InstanceServer {
    let (events, received) = crossbeam_channel::unbounded();
    std::thread::spawn(move || {
        for event in received {
            let Event::CommandReceived { command, reply } = event;
            reply.send(handler(&command));
        }
    });
    let mut server = InstanceServer::new(events);
    assert!(server.listen());
    server
}

// ------------------------------------------------------------------------- basics
#[test]
fn test_version() {
    let fixture = Fixture::new();
    let outcome = fixture.run(&["--version"]);
    assert_eq!(outcome.code, EXIT_OK);
    assert_eq!(outcome.out.trim(), format!("Eye Tracker {VERSION}"));
}

#[test]
fn test_help_lists_every_subcommand() {
    let fixture = Fixture::new();
    let outcome = fixture.run(&["--help"]);
    assert_eq!(outcome.code, EXIT_OK);
    for command in [
        "run",
        "calibrate",
        "doctor",
        "bench",
        "ctl",
        "autostart",
        "reset",
    ] {
        assert!(outcome.out.contains(command), "{command}");
    }
    for option in [
        "--config-dir",
        "--log-level",
        "--camera",
        "--backend",
        "--background",
    ] {
        assert!(outcome.out.contains(option), "{option}");
    }
    assert!(outcome.out.contains("exit codes: 0 ok, 1 error"));
}

#[test]
fn test_usage_errors() {
    let fixture = Fixture::new();
    for argv in [
        &["frobnicate"][..],
        &["ctl", "format-disk"],
        &["bench", "--seconds", "-1"],
        &["bench", "--seconds", "nan"],
        &["--log-level", "loud"],
    ] {
        let outcome = fixture.run(argv);
        assert_eq!(outcome.code, EXIT_USAGE, "{argv:?}");
        assert!(outcome.err.contains("error:"), "{argv:?}: {}", outcome.err);
    }
    // The run options belong to `run` (and the main parser) only.
    assert_eq!(
        fixture.run(&["ctl", "status", "--trace", "x"]).code,
        EXIT_USAGE
    );
}

#[test]
fn test_ctl_commands_mirror_ipc() {
    let mut mirrored = CTL_COMMANDS.to_vec();
    let mut commands = ipc::COMMANDS.to_vec();
    mirrored.sort_unstable();
    commands.sort_unstable();
    assert_eq!(mirrored, commands);
}

#[test]
fn test_gui_main_survives_missing_std_streams() {
    // A windowed process has no console: every print fails, and nothing may panic.
    struct Closed;
    impl Write for Closed {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::from(io::ErrorKind::BrokenPipe))
        }
        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::from(io::ErrorKind::BrokenPipe))
        }
    }
    let _fixture = Fixture::new();
    let (mut out, mut err) = (Closed, Closed);
    let mut env = Env::process(&mut out, &mut err);
    assert_eq!(run_cli([APP_SLUG, "--version"], &mut env), EXIT_OK);
    assert_eq!(run_cli([APP_SLUG, "--help"], &mut env), EXIT_OK);
    assert_eq!(run_cli([APP_SLUG, "--bogus"], &mut env), EXIT_USAGE);
}

#[test]
fn test_help_examples_use_the_program_name() {
    let text = epilog("eye-tracker-cli.exe");
    assert!(text.contains("  eye-tracker-cli.exe ctl privacy-toggle  "));
    assert!(text.contains("\n  eye-tracker-cli.exe  "));
    assert!(!text.contains("\n  eye-tracker "));
}

#[test]
fn test_ctl_help_lists_the_commands() {
    assert_eq!(CTL_COMMANDS_HELP, CTL_COMMANDS.join(", "));
}

// ---------------------------------------------------------------------------- run
#[test]
fn test_run_is_the_default() {
    let fixture = Fixture::new();
    assert_eq!(fixture.run(&[]).code, 0);
    assert_eq!(
        fixture
            .run(&["--background", "--camera", "2", "--backend", "lite"])
            .code,
        0
    );
    assert_eq!(
        fixture
            .run(&["run", "--calibrate", "--log-level", "debug"])
            .code,
        0
    );
    let calls = fixture.run_calls.borrow();
    let [first, second, third] = &calls[..] else {
        panic!("expected three runs, got {}", calls.len());
    };
    assert!(!first.background);
    assert!(!first.calibrate);
    assert_eq!(first.camera, None);
    assert_eq!(first.log_level, None);
    assert!(second.background);
    assert_eq!(second.camera.as_deref(), Some("2"));
    assert_eq!(second.backend.as_deref(), Some("lite"));
    assert!(third.calibrate);
    assert_eq!(third.log_level.as_deref(), Some("DEBUG"));
}

#[test]
fn test_run_returns_the_exit_code_of_the_app() {
    let fixture = Fixture::new();
    let outcome = fixture.run_with(&[], |env| env.run_app = Box::new(|_| EXIT_INTERRUPTED));
    assert_eq!(outcome.code, EXIT_INTERRUPTED);
}

#[test]
fn test_global_options_work_before_and_after_the_subcommand() {
    let fixture = Fixture::new();
    assert_eq!(fixture.run(&["--camera", "1", "run"]).code, 0);
    assert_eq!(fixture.run(&["run", "--camera", "3"]).code, 0);
    // A value after the subcommand wins; an absent one does not reset the earlier one.
    assert_eq!(
        fixture.run(&["--camera", "1", "run", "--camera", "4"]).code,
        0
    );
    assert_eq!(
        fixture
            .run(&["--log-level", "error", "--trace", "a.jsonl", "run"])
            .code,
        0
    );
    assert_eq!(
        fixture
            .run(&["--trace", "a.jsonl", "run", "--trace", "b.jsonl"])
            .code,
        0
    );
    let calls = fixture.run_calls.borrow();
    let cameras: Vec<_> = calls.iter().map(|a| a.camera.as_deref()).collect();
    assert_eq!(cameras, [Some("1"), Some("3"), Some("4"), None, None]);
    assert_eq!(calls[3].log_level.as_deref(), Some("ERROR"));
    assert_eq!(calls[3].trace, Some(PathBuf::from("a.jsonl")));
    assert_eq!(calls[4].trace, Some(PathBuf::from("b.jsonl")));
}

#[test]
fn test_calibrate_subcommand_runs_the_app_in_calibration_mode() {
    let fixture = Fixture::new();
    assert_eq!(fixture.run(&["calibrate", "--background"]).code, 0);
    let calls = fixture.run_calls.borrow();
    let [args] = &calls[..] else {
        panic!("expected one run");
    };
    assert!(args.calibrate);
    assert!(args.background);
}

/// Pretend to run as `running` from a package folder with the given executables on `os`.
fn run_as(fixture: &Fixture, os: &str, package: &Package, argv: &[&str], running: bool) -> Outcome {
    let (exe, os) = (package.exe.clone(), os.to_string());
    fixture.run_with(argv, move |env| {
        env.exe = exe;
        env.os = os;
        env.instance_running = Box::new(move || running);
    })
}

#[test]
fn test_the_windows_console_command_starts_the_app_on_its_own() {
    // r3-ux-docs-06: run in the terminal, the app died with the terminal window.
    let fixture = Fixture::new();
    let p = package(
        "eye-tracker.exe",
        &["eye-tracker-cli.exe", "EyeTracker.exe"],
        "Eye Tracker",
    );
    let trace = fixture
        .dirs
        .path()
        .canonicalize()
        .unwrap()
        .join("trace.jsonl");
    let trace_text = trace.to_string_lossy().into_owned();
    let given = fixture.dirs.path().join("trace.jsonl");
    let first = run_as(
        &fixture,
        "windows",
        &p,
        &[
            "--camera",
            "1",
            "--log-level",
            "debug",
            "--trace",
            &given.to_string_lossy(),
        ],
        false,
    );
    let second = run_as(
        &fixture,
        "windows",
        &p,
        &["calibrate", "--background"],
        false,
    );
    assert_eq!((first.code, second.code), (0, 0));
    assert!(fixture.run_calls.borrow().is_empty()); // nothing ran inside the terminal
    let gui = path_of(&p, "EyeTracker.exe");
    let profile = fixture.profile();
    let started = fixture.started.borrow();
    let [one, two] = &started[..] else {
        panic!("expected two starts, got {started:?}");
    };
    assert_eq!(
        one,
        &strings(&[
            &gui,
            "--config-dir",
            &profile,
            "--log-level",
            "DEBUG",
            "--camera",
            "1",
            "--trace",
            &trace_text,
        ])
    );
    assert_eq!(
        two,
        &strings(&[
            &gui,
            "--config-dir",
            &profile,
            "--background",
            "--calibrate"
        ])
    );
    assert_eq!(DETACHED_FLAGS, 0x208);
    assert_eq!(DETACHED_FLAGS, DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP);
    assert!(first.out.contains("starting in the system tray"));
}

#[test]
fn test_the_console_command_hands_over_to_a_running_app_itself() {
    let fixture = Fixture::new();
    let p = package("eye-tracker-cli.exe", &["EyeTracker.exe"], "Eye Tracker");
    let outcome = run_as(&fixture, "windows", &p, &["--camera", "1"], true);
    assert_eq!(outcome.code, 0);
    assert!(fixture.started.borrow().is_empty());
    // Prints what the running instance answers.
    assert_eq!(fixture.run_calls.borrow().len(), 1);
}

#[test]
fn test_other_executables_run_the_app_in_place() {
    for (os, running) in [
        ("windows", "EyeTracker.exe"),
        ("linux", "eye-tracker-cli"),
        ("macos", "eye-tracker-cli"),
    ] {
        let fixture = Fixture::new();
        let p = package(running, &["EyeTracker.exe", "Eye Tracker"], "Eye Tracker");
        assert_eq!(run_as(&fixture, os, &p, &[], false).code, 0, "{os}");
        assert!(fixture.started.borrow().is_empty(), "{os}");
        assert_eq!(fixture.run_calls.borrow().len(), 1, "{os}");
    }
}

#[test]
fn test_a_failed_detached_start_runs_the_app_in_place() {
    let fixture = Fixture::new();
    let p = package("eye-tracker-cli.exe", &["EyeTracker.exe"], "Eye Tracker");
    let (exe, refuse) = (p.exe.clone(), |_: &[String]| {
        Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "blocked by policy",
        ))
    });
    let outcome = fixture.run_with(&[], move |env| {
        env.exe = exe;
        env.os = "windows".to_string();
        env.instance_running = Box::new(|| false);
        env.spawn = Box::new(refuse);
    });
    assert_eq!(outcome.code, 0);
    assert_eq!(fixture.run_calls.borrow().len(), 1);
}

#[test]
fn test_config_dir_option() {
    let fixture = Fixture::new();
    let portable = fixture.dirs.path().canonicalize().unwrap().join("portable");
    let outcome = fixture.run(&["--config-dir", &portable.to_string_lossy()]);
    assert_eq!(outcome.code, 0);
    assert_eq!(paths::config_dir(), portable);
}

// ------------------------------------------------------------------------- doctor
#[test]
fn test_doctor_json() {
    let fixture = Fixture::new();
    let outcome = fixture.run(&["doctor", "--json"]);
    assert_eq!(outcome.code, EXIT_OK);
    let report: Value = serde_json::from_str(&outcome.out).unwrap();
    for key in [
        "app", "system", "ui", "monitors", "backends", "cameras", "paths", "problems",
    ] {
        assert!(report.get(key).is_some(), "{key}");
    }
    assert_eq!(report["app"]["version"], VERSION);
    assert_eq!(report["cameras"]["probed"], json!(false));
    assert!(report["problems"].is_array());
}

#[test]
fn test_doctor_text() {
    let fixture = Fixture::new();
    let outcome = fixture.run(&["doctor", "--log-level", "error"]);
    assert_eq!(outcome.code, EXIT_OK);
    assert!(
        outcome
            .out
            .starts_with(&format!("Eye Tracker {VERSION} - diagnostics")),
        "{}",
        outcome.out
    );
    assert!(outcome.out.contains("Vision backends"));
}

// -------------------------------------------------------------------------- bench
fn write_image(path: &Path) {
    let mut image = image::RgbImage::new(160, 120);
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        *pixel = if (60..100).contains(&x) && (40..80).contains(&y) {
            image::Rgb([220, 160, 40])
        } else {
            image::Rgb([0, 0, 0])
        };
    }
    image.save(path).unwrap();
}

#[test]
fn test_bench_json_on_an_image() {
    let fixture = Fixture::new();
    let image = fixture.dirs.path().join("still.png");
    write_image(&image);
    let outcome = fixture.run(&[
        "bench",
        "--seconds",
        "0.2",
        "--camera",
        &image.to_string_lossy(),
        "--backend",
        "lite",
        "--json",
    ]);
    assert_eq!(outcome.code, EXIT_OK, "{}", outcome.err);
    let result: Value = serde_json::from_str(&outcome.out).unwrap();
    assert_eq!(result["backend"], "lite");
    assert_eq!(result["device"], "file still.png");
    let modes: Vec<&String> = result["modes"].as_object().unwrap().keys().collect();
    assert_eq!(modes.len(), 2);
    assert!(result["modes"].get("max").is_some() && result["modes"].get("idle").is_some());
    assert!(outcome.err.contains("Benchmarking"));
}

#[test]
fn test_bench_text_uses_the_settings() {
    let fixture = Fixture::new();
    let image = fixture.dirs.path().join("configured.png");
    write_image(&image);
    let mut settings = Settings::default();
    settings.camera.device = image.to_string_lossy().into_owned();
    settings.general.backend = "lite".to_string();
    settings.save(&paths::settings_file()).unwrap();
    let outcome = fixture.run(&["bench", "--seconds", "0.2"]);
    assert_eq!(outcome.code, EXIT_OK, "{}", outcome.err);
    assert!(
        outcome
            .out
            .starts_with("Eye Tracker benchmark - lite backend"),
        "{}",
        outcome.out
    );
    assert!(outcome.out.contains("file configured.png"));
}

#[test]
fn test_bench_reports_a_missing_source() {
    let fixture = Fixture::new();
    let missing = fixture.dirs.path().join("missing.mp4");
    let outcome = fixture.run(&[
        "bench",
        "--seconds",
        "0.2",
        "--camera",
        &missing.to_string_lossy(),
    ]);
    assert_eq!(outcome.code, EXIT_ERROR);
    assert!(outcome.err.contains("error: "), "{}", outcome.err);
    assert!(outcome.err.contains("not found"), "{}", outcome.err);
}

// ---------------------------------------------------------------------------- ctl
#[test]
fn test_ctl_without_an_instance() {
    let fixture = Fixture::new();
    let outcome = fixture.run(&["ctl", "status"]);
    assert_eq!(outcome.code, EXIT_NOT_RUNNING);
    assert_eq!(outcome.out, "");
    assert!(outcome.err.contains("Eye Tracker is not running"));
}

#[test]
fn test_ctl_with_an_instance() {
    let fixture = Fixture::new();
    let received = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&received);
    let mut server = serve(move |command| {
        log.lock().push(command.to_string());
        match command {
            "status" => json!({"state": "tracking"}).to_string(),
            "calibrate" => "error: calibration unavailable".to_string(),
            _ => "ok".to_string(),
        }
    });
    let status = fixture.run(&["ctl", "status"]);
    assert_eq!(status.code, EXIT_OK);
    assert_eq!(
        serde_json::from_str::<Value>(&status.out).unwrap(),
        json!({"state": "tracking"})
    );
    let toggle = fixture.run(&["ctl", "privacy-toggle"]);
    assert_eq!(toggle.code, EXIT_OK);
    assert_eq!(toggle.out.trim(), "ok");
    let calibrate = fixture.run(&["ctl", "calibrate"]);
    assert_eq!(calibrate.code, EXIT_ERROR);
    assert!(calibrate.err.contains("calibration unavailable"));
    assert_eq!(*received.lock(), ["status", "privacy-toggle", "calibrate"]);
    server.close();
}

#[test]
fn test_ctl_with_an_instance_that_does_not_answer() {
    let fixture = Fixture::new();
    // A server whose owner holds the request and never replies: the client times out.
    let (events, received) = crossbeam_channel::unbounded();
    let mut server = InstanceServer::new(events);
    assert!(server.listen());
    let outcome = fixture.run(&["ctl", "pause", "--timeout", "150"]);
    assert_eq!(outcome.code, EXIT_ERROR);
    assert!(
        outcome.err.contains("did not answer within 150 ms"),
        "{}",
        outcome.err
    );
    drop(received);
    server.close();
}

// ---------------------------------------------------------------------- autostart
#[test]
fn test_autostart_status_enable_disable() {
    let fixture = Fixture::new();
    let status = fixture.run(&["autostart"]);
    assert_eq!(status.code, EXIT_OK);
    assert!(status.out.contains("Start at login: disabled"));
    assert!(status.out.contains("HKCU\\Run\\EyeTracker"));
    assert!(status.out.contains("EyeTracker.exe"));
    assert!(status.out.contains("--background"));

    let enable = fixture.run(&["autostart", "enable"]);
    assert_eq!(enable.code, EXIT_OK);
    assert!(fixture.fake.enabled.get());
    assert!(enable.out.contains("Start at login enabled"));
    let status = fixture.run(&["autostart", "status"]);
    assert!(status.out.contains("Start at login: enabled"));

    let disable = fixture.run(&["autostart", "disable"]);
    assert_eq!(disable.code, EXIT_OK);
    assert!(!fixture.fake.enabled.get());
    assert!(disable.out.contains("Start at login disabled."));
    assert_eq!(
        *fixture.fake.calls.borrow(),
        ["enable(background=true)", "disable"]
    );
}

#[test]
fn test_autostart_failure_and_unsupported() {
    let fixture = Fixture::new();
    *fixture.fake.fail.borrow_mut() = Some("access denied".to_string());
    let failed = fixture.run(&["autostart", "enable"]);
    assert_eq!(failed.code, EXIT_ERROR);
    assert!(failed.err.contains("error: access denied"));

    fixture.fake.supported.set(false);
    let status = fixture.run(&["autostart", "status"]);
    assert_eq!(status.code, EXIT_OK);
    assert!(status.out.contains("not supported"));
    let enable = fixture.run(&["autostart", "enable"]);
    assert_eq!(enable.code, EXIT_ERROR);
    assert!(enable.err.contains("not supported"));
    assert_eq!(*fixture.fake.calls.borrow(), ["enable(background=true)"]);
}

#[test]
fn test_autostart_follows_the_config_dir() {
    // Regression (ui_app-11): a portable instance registered the default profile.
    let fixture = Fixture::new();
    let portable = fixture.dirs.path().join("portable");
    let text = portable.to_string_lossy().into_owned();
    assert_eq!(
        fixture
            .run(&["--config-dir", &text, "autostart", "enable"])
            .code,
        EXIT_OK
    );
    assert_eq!(*fixture.fake.profiles.borrow(), [Some(portable.clone())]);
    let status = fixture.run(&["autostart", "status", "--config-dir", &text]);
    assert_eq!(status.code, EXIT_OK);
    assert!(status.out.contains("Start at login: enabled"));
    assert!(
        status
            .out
            .replace(['"', '\''], "")
            .contains(&format!("--config-dir {text}"))
    );
    let disable = fixture.run(&["autostart", "disable", "--config-dir", &text]);
    assert_eq!(disable.code, EXIT_OK);
    assert_eq!(
        fixture.fake.profiles.borrow().last().unwrap(),
        &Some(portable)
    );
    // Without --config-dir the running process's profile applies (None).
    fixture.fake.profiles.borrow_mut().clear();
    paths::set_base_override(None);
    let enable = fixture.run(&["autostart", "enable"]);
    paths::set_base_override(Some(fixture.dirs.path()));
    assert_eq!(enable.code, EXIT_OK);
    assert_eq!(*fixture.fake.profiles.borrow(), [None]);
}

#[test]
fn test_autostart_status_explains_entries_for_other_copies() {
    let fixture = Fixture::new();
    fixture.fake.idle_status.set(Status::Stale);
    *fixture.fake.registered.borrow_mut() =
        Some(strings(&["/opt/old/eye-tracker", "--background"]));
    let stale = fixture.run(&["autostart"]);
    assert_eq!(stale.code, EXIT_OK);
    assert!(stale.out.contains("Start at login: stale\n")); // the status value itself
    assert!(stale.out.contains("Broken"));
    assert!(
        stale
            .out
            .contains("Registered: /opt/old/eye-tracker --background")
    );
    fixture.fake.idle_status.set(Status::OtherProfile);
    let other = fixture.run(&["autostart"]);
    assert!(other.out.contains("Start at login: other-profile\n"));
    assert!(other.out.contains("another profile"));
    let disable = fixture.run(&["autostart", "disable"]);
    assert_eq!(disable.code, EXIT_OK);
    assert!(disable.out.contains("left unchanged"));
}

#[test]
fn test_backend_option_takes_current_and_legacy_names() {
    assert_eq!(BACKEND_CHOICES, ["auto", "facemesh", "lite", "apple"]);
    let fixture = Fixture::new();
    for (given, expected) in [
        ("facemesh", "facemesh"),
        ("LITE", "lite"),
        ("apple", "apple"),
        ("mediapipe", "facemesh"), // Eye Tracker 0.1 names in old shortcuts
        ("opencv", "lite"),
    ] {
        assert_eq!(fixture.run(&["--backend", given]).code, 0, "{given}");
        assert_eq!(
            fixture
                .run_calls
                .borrow()
                .last()
                .unwrap()
                .backend
                .as_deref(),
            Some(expected),
            "{given}"
        );
    }
    let bad = fixture.run(&["--backend", "tflite"]);
    assert_eq!(bad.code, EXIT_USAGE);
    assert!(bad.err.contains("invalid choice"));
    assert_eq!(backend_name(" OpenCV "), "lite");
}

// -------------------------------------------------------------------------- reset
fn write_state() -> (PathBuf, PathBuf) {
    let settings_file = paths::settings_file();
    let calibration_file = paths::calibration_file();
    Settings::default().save(&settings_file).unwrap();
    fs::write(&calibration_file, "{}").unwrap();
    (settings_file, calibration_file)
}

#[test]
fn test_reset_calibration_only() {
    let fixture = Fixture::new();
    let (settings_file, calibration_file) = write_state();
    let outcome = fixture.run(&["reset", "--calibration"]);
    assert_eq!(outcome.code, EXIT_OK);
    assert!(settings_file.exists());
    assert!(!calibration_file.exists());
    assert!(outcome.out.contains("Deleted"));
}

#[test]
fn test_reset_all() {
    let fixture = Fixture::new();
    let (settings_file, calibration_file) = write_state();
    let corrupt = settings_file.with_extension("json.corrupt");
    fs::write(&corrupt, "{").unwrap();
    assert_eq!(fixture.run(&["reset", "--all"]).code, EXIT_OK);
    assert!(!settings_file.exists());
    assert!(!corrupt.exists());
    assert!(!calibration_file.exists());
    let again = fixture.run(&["reset", "--settings"]);
    assert_eq!(again.code, EXIT_OK);
    assert!(again.out.contains("Nothing to reset."));
}

#[test]
fn test_reset_needs_a_target() {
    let fixture = Fixture::new();
    let outcome = fixture.run(&["reset"]);
    assert_eq!(outcome.code, EXIT_USAGE);
    assert!(outcome.err.contains("--calibration, --settings or --all"));
}

#[test]
fn test_reset_refuses_while_the_app_runs() {
    let fixture = Fixture::new();
    let (_settings_file, calibration_file) = write_state();
    let mut server = serve(|_| "ok".to_string());
    let outcome = fixture.run(&["reset", "--all"]);
    assert_eq!(outcome.code, EXIT_ERROR);
    assert!(outcome.err.contains("Quit it first"));
    assert!(calibration_file.exists());
    server.close();
}

#[test]
fn test_command_line_tools_log_to_the_terminal_only() {
    let fixture = Fixture::new();
    fixture.run(&["reset", "--calibration"]);
    fixture.run(&["reset", "--calibration", "--log-level", "debug"]);
    fixture.run(&["ctl", "status", "--log-level", "error"]);
    LOGGING.with_borrow(|calls| {
        let expected = |level: &str| (level.to_string(), true, false);
        assert_eq!(
            *calls,
            [expected("WARNING"), expected("DEBUG"), expected("ERROR")]
        );
    });
    // The tray app sets its own logging up (run_app).
    LOGGING.with_borrow_mut(Vec::clear);
    fixture.run(&[]);
    LOGGING.with_borrow(|calls| assert!(calls.is_empty()));
}

// ---------------------------------------------------------------------- parser values
#[test]
fn test_positive_float_and_log_level_values() {
    assert_eq!(positive_float("0.2"), Ok(0.2));
    for bad in ["0", "-1", "nan", "inf", "x"] {
        assert!(positive_float(bad).is_err(), "{bad}");
    }
    assert_eq!(log_level("debug").as_deref(), Ok("DEBUG"));
    assert!(log_level("loud").is_err());
}
