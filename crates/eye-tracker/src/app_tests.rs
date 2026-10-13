//! Tests of the tray app (tests/test_app.py): the whole app built and started without a display.
//!
//! [`build_app`] creates the real controller and UI, but the vision worker, the platform, the
//! hotkey manager, the tray, the login item, the clock and logging are fakes: no camera is opened,
//! the pointer never moves, no global hotkey is registered and nothing is locked. The app's
//! wiring runs in [`EyeTrackerApp::step`], so the tests call it where the Python tests processed
//! Qt events, and advance the fake clock where they waited. A controller signal the Python tests
//! emitted by hand is handed to [`EyeTrackerApp::dispatch`]. Commands go through the real command
//! socket; a client blocks until it is answered, so it runs on a thread while the test steps the
//! app (a second launch, too).
//!
//! Not ported:
//! * The Qt application checks of `test_build_app_creates_and_wires_everything`
//!   (`applicationName`, `quitOnLastWindowClosed`): there is no `QApplication`.
//! * The `run_app(...) == 0` line of `test_second_instance_hands_over_and_exits`: `run_app` uses
//!   the real platform, logging and event loop.
//! * `test_window_requests_during_a_calibration_raise_it_instead` checks that no other window
//!   opens, but not how often the calibration was raised (`CalibrationWindow::start` cannot be
//!   replaced).
//! * The root logger level and log file assertions of
//!   `test_log_level_follows_settings_unless_given`: logging is faked (starting the real logger
//!   would replace the log capture of every other test); the levels handed to it are checked.
//! * `test_a_failing_login_item_check_does_not_stop_the_app`: `autostart::refresh` cannot fail.
//! * The `create_app` and `server` cases of
//!   `test_the_instance_lock_is_released_when_startup_raises`: creating the app and the server
//!   cannot fail; a panic in the hand-over is checked.
//! * `test_exec_quits_on_ctrl_c` and the other event-loop tests run [`EyeTrackerApp::step`] and
//!   [`AppContext::shutdown`] (what the event loop does) instead of eframe, so the exit code of a
//!   Ctrl+C (0, as in Python; only the event loop returns it) is not asserted. The signal tests
//!   are Unix only.
//! * `test_automatic_calibration_reasons_only_notify` replaced `tray.notify`, which ignored the
//!   startup mute; here the login start's quiet period is 0 instead.

use std::cell::RefCell;
use std::rc::Rc;
use std::thread;
use std::time::Duration;

use parking_lot::Mutex;

use super::*;
use crate::platform::autostart::Status;
use crate::platform::hotkeys::{HotkeyManager, UnsupportedHotkeyManager};
use crate::testing::controller_fakes::{
    FakeClock, FakePlatform, FakeWorker, fake_backends, make_calibration, run_now,
};
use crate::testing::{AppDirs, app_dirs, capture_logs};
use crate::types::Rect;
use crate::ui::tray::{BackendEvent, TrayImage, TrayModel};

fn two_monitors() -> Vec<Monitor> {
    vec![
        Monitor {
            primary: true,
            ..Monitor::new(0, "left", Rect::new(0, 0, 800, 600))
        },
        Monitor::new(1, "right", Rect::new(800, 0, 800, 600)),
    ]
}

fn one_monitor() -> Vec<Monitor> {
    two_monitors()[..1].to_vec()
}

// -------------------------------------------------------------------------- fakes
/// What the fake tray was asked to show.
#[derive(Default)]
struct TrayLog {
    /// The desktop has a tray area (offscreen Qt had none).
    area: bool,
    visible: bool,
    messages: Vec<(String, String)>,
    popups: usize,
}

struct FakeTray(Rc<RefCell<TrayLog>>);

impl TrayBackend for FakeTray {
    fn available(&self) -> bool {
        self.0.borrow().area
    }
    fn is_visible(&self) -> bool {
        self.0.borrow().visible
    }
    fn show(&mut self) {
        self.0.borrow_mut().visible = true;
    }
    fn hide(&mut self) {
        self.0.borrow_mut().visible = false;
    }
    fn set_icon(&mut self, _image: &TrayImage) {}
    fn set_tooltip(&mut self, _tooltip: &str) {}
    fn set_menu(&mut self, _model: &TrayModel) {}
    fn popup_menu(&mut self) {
        self.0.borrow_mut().popups += 1;
    }
    fn show_message(&mut self, title: &str, message: &str, _critical: bool) {
        self.0
            .borrow_mut()
            .messages
            .push((title.to_string(), message.to_string()));
    }
    fn poll_events(&mut self) -> Vec<BackendEvent> {
        Vec::new()
    }
}

/// The `fake_autostart` fixture: the app, tray, wizard and settings read (and could change) the
/// login item.
#[derive(Default)]
struct AutostartLog {
    unsupported: bool,
    calls: Vec<&'static str>,
}

struct FakeAutostart(Rc<RefCell<AutostartLog>>);

impl AutostartBackend for FakeAutostart {
    fn is_supported(&self) -> bool {
        !self.0.borrow().unsupported
    }
    fn is_enabled(&self) -> bool {
        false
    }
    fn status(&self) -> Option<Status> {
        Some(Status::Disabled)
    }
    fn enable(&self, _background: bool) -> Result<(), Box<dyn std::error::Error>> {
        self.0.borrow_mut().calls.push("enable");
        Ok(())
    }
    fn disable(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.0.borrow_mut().calls.push("disable");
        Ok(())
    }
    fn refresh(&self) -> bool {
        self.0.borrow_mut().calls.push("refresh");
        false
    }
}

/// Records which privacy settings pages would be opened (none is).
#[derive(Default)]
struct PermissionPlatform {
    opened: Mutex<Vec<String>>,
    /// macOS-like: Accessibility belongs to an earlier build until the test grants it.
    accessibility: Mutex<&'static str>,
}

impl PermissionPlatform {
    fn new(accessibility: &'static str) -> Arc<Self> {
        Arc::new(Self {
            opened: Mutex::new(Vec::new()),
            accessibility: Mutex::new(accessibility),
        })
    }

    fn opened(&self) -> Vec<String> {
        self.opened.lock().clone()
    }
}

impl PlatformServices for PermissionPlatform {
    fn open_permission_settings(&self, name: &str) -> bool {
        self.opened.lock().push(name.to_string());
        true
    }

    fn accessibility_status(&self) -> &'static str {
        *self.accessibility.lock()
    }
}

/// The levels given to the (fake) logger: `set_level` calls and `setup_logging` calls with
/// whether they log to the file.
static LEVELS: Mutex<Vec<String>> = Mutex::new(Vec::new());
static LOG_SETUPS: Mutex<Vec<(String, bool)>> = Mutex::new(Vec::new());

fn record_level(level: &str) {
    LEVELS.lock().push(level.to_string());
}

fn record_setup(level: &str, _console: bool, to_file: bool) {
    LOG_SETUPS.lock().push((level.to_string(), to_file));
}

fn no_error_box(_text: &str) {}

// ------------------------------------------------------------------------ harness
/// The `app_dirs` fixture; also removes the instance lock file, which can live in the shared
/// temporary directory.
struct Env {
    _dirs: AppDirs,
}

impl Drop for Env {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(InstanceLock::new(None).path());
    }
}

fn env() -> Env {
    Env { _dirs: app_dirs() }
}

/// The keyword arguments of the Python `build` fixture.
struct Setup {
    settings: Option<Settings>,
    monitors: Vec<Monitor>,
    platform: Option<Arc<dyn PlatformServices>>,
    args: RunArgs,
    quiet_s: f64,
    autostart_supported: bool,
    startup_wait_ms: u64,
    handover_wait_ms: u64,
    send_command: Option<SendCommand>,
    wait_ms: Option<Box<dyn FnMut(u64)>>,
    /// The vision worker cannot be created (a broken controller).
    broken: bool,
}

impl Default for Setup {
    fn default() -> Self {
        Self {
            settings: None,
            monitors: one_monitor(),
            platform: None,
            args: RunArgs {
                background: true,
                ..RunArgs::default()
            },
            quiet_s: STARTUP_QUIET_S,
            autostart_supported: true,
            startup_wait_ms: ipc::STARTUP_WAIT_MS,
            handover_wait_ms: HANDOVER_WAIT_MS,
            send_command: None,
            wait_ms: None,
            broken: false,
        }
    }
}

fn foreground() -> Setup {
    Setup {
        args: RunArgs::default(),
        ..Setup::default()
    }
}

struct Harness {
    ctx: AppContext,
    clock: FakeClock,
    /// What the controller's monitor provider returns.
    layout: Arc<Mutex<Vec<Monitor>>>,
    platform: Arc<FakePlatform>,
    workers: Arc<Mutex<Vec<FakeWorker>>>,
    tray: Rc<RefCell<TrayLog>>,
    autostart: Rc<RefCell<AutostartLog>>,
}

/// Build and start the app (call [`env`] first).
fn build(setup: Setup) -> Harness {
    let settings = setup.settings.unwrap_or_else(|| {
        let mut settings = Settings::default();
        settings.general.first_run_done = true;
        settings
    });
    settings.save(&paths::settings_file()).unwrap();
    let clock = FakeClock::new(100.0);
    let workers: Arc<Mutex<Vec<FakeWorker>>> = Arc::default();
    let made = workers.clone();
    let broken = setup.broken;
    let layout = Arc::new(Mutex::new(setup.monitors));
    let monitors = layout.clone();
    let tray = Rc::new(RefCell::new(TrayLog::default()));
    let autostart = Rc::new(RefCell::new(AutostartLog {
        unsupported: !setup.autostart_supported,
        calls: Vec::new(),
    }));
    let login_item = autostart.clone();
    let fake_platform = Arc::new(FakePlatform::new());
    let defaults = BuildOptions::default();
    let options = BuildOptions {
        platform: Some(
            setup
                .platform
                .unwrap_or_else(|| fake_platform.clone() as Arc<dyn PlatformServices>),
        ),
        controller: ControllerOptions {
            worker_factory: Some(Box::new(move |source, backend, events| {
                assert!(!broken, "no controller today");
                let worker = FakeWorker::new(source, backend, events);
                made.lock().push(worker.clone());
                Box::new(worker)
            })),
            monitors_provider: Some(Box::new(move || monitors.lock().clone())),
            hotkey_manager_factory: Some(Box::new(|| {
                Arc::new(UnsupportedHotkeyManager::new(Some("tests".into())))
                    as Arc<dyn HotkeyManager>
            })),
            lock_runner: Some(run_now()),
            backend_factory: Some(fake_backends().0),
            ..ControllerOptions::default()
        },
        tray_backend: Some(Box::new(FakeTray(tray.clone()))),
        autostart: Some(Rc::new(move || Box::new(FakeAutostart(login_item.clone())))),
        clock: Some(clock.clock()),
        startup_quiet_s: setup.quiet_s,
        startup_wait_ms: setup.startup_wait_ms,
        handover_wait_ms: setup.handover_wait_ms,
        send_command: setup.send_command.unwrap_or(defaults.send_command),
        wait_ms: setup.wait_ms.unwrap_or(defaults.wait_ms),
        setup_logging: record_setup,
        set_level: record_level,
        show_error: no_error_box,
        ..defaults
    };
    let mut ctx = build_app(&setup.args, options);
    if ctx.exit_code.is_none() {
        ctx.start();
    }
    Harness {
        ctx,
        clock,
        layout,
        platform: fake_platform,
        workers,
        tray,
        autostart,
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        self.ctx.shutdown();
    }
}

impl Harness {
    fn app(&mut self) -> &mut EyeTrackerApp {
        self.ctx.app.as_mut().expect("the app")
    }

    fn controller(&mut self) -> &mut Controller {
        self.ctx.controller().expect("the controller")
    }

    fn worker(&self) -> FakeWorker {
        let workers = self.workers.lock();
        assert_eq!(workers.len(), 1);
        workers[0].clone()
    }

    fn step(&mut self) {
        self.app().step();
    }

    /// Run pending zero-delay timers (Python's `settle`).
    fn settle(&mut self) {
        for _ in 0..3 {
            self.step();
        }
    }

    fn emit(&mut self, event: ControllerEvent) {
        self.app().dispatch(event);
    }

    /// Advance the clock in small steps until `condition` holds (Python's `_wait_for`).
    fn wait_for(&mut self, condition: impl Fn(&mut Harness) -> bool, timeout: f64) -> bool {
        let deadline = self.clock.now() + timeout;
        while !condition(self) && self.clock.now() < deadline {
            self.clock.advance(0.05);
            self.step();
        }
        condition(self)
    }

    /// Send a command through the command socket while the app answers it.
    fn send(&mut self, command: &str) -> Option<String> {
        let command = command.to_string();
        let client = thread::spawn(move || ipc::send_command(&command, ipc::DEFAULT_TIMEOUT_MS, 0));
        while !client.is_finished() {
            self.step();
            thread::sleep(Duration::from_millis(2));
        }
        client.join().unwrap().unwrap()
    }

    /// Titles the tray shows from now on, with its real settings and mute checks (Python's
    /// `_tray_messages`: offscreen there was no tray area).
    fn tray_messages(&mut self) {
        self.tray.borrow_mut().area = true;
    }

    fn titles(&self) -> Vec<String> {
        self.tray
            .borrow()
            .messages
            .iter()
            .map(|(title, _)| title.clone())
            .collect()
    }
}

/// What a second launch ended with.
struct Launched {
    exit_code: Option<i32>,
    app_built: bool,
    workers: usize,
    warnings: Vec<String>,
}

/// Launch a second instance on another thread while `first` answers it.
fn launch(first: &mut Harness, setup: impl FnOnce() -> Setup + Send + 'static) -> Launched {
    let second = thread::spawn(move || {
        let logs = capture_logs();
        let h = build(setup());
        let launched = Launched {
            exit_code: h.ctx.exit_code,
            app_built: h.ctx.app.is_some(),
            workers: h.workers.lock().len(),
            warnings: logs.messages_at_least(log::Level::Warn),
        };
        drop(h);
        launched
    });
    while !second.is_finished() {
        first.step();
        thread::sleep(Duration::from_millis(2));
    }
    second.join().unwrap()
}

// ----------------------------------------------------------------------- building
#[test]
fn test_build_app_creates_and_wires_everything() {
    let _env = env();
    let mut h = build(Setup {
        monitors: two_monitors(),
        ..Setup::default()
    });
    assert_eq!(h.ctx.exit_code, None);
    assert!(h.ctx.server.as_ref().unwrap().is_listening());
    assert!(h.app().tray().is_some());
    assert!(h.app().overlay().is_some());
    assert!(h.app().countdown().is_some());
    assert!(h.app().curtain().is_some());
    assert!(h.worker().s().started);
    assert!(!h.worker().s().stopped);

    // The command socket reaches the controller.
    let reply = h.send("status").unwrap();
    let status: Value = serde_json::from_str(&reply).unwrap();
    assert_eq!(status["state"], TrackingState::NeedsCalibration.value());
    assert_eq!(status["calibrated"], false);
    assert_eq!(h.send("pause").as_deref(), Some("ok"));
    assert_eq!(h.controller().state(), TrackingState::Paused);
    h.step();
    assert_eq!(h.app().tray().unwrap().state(), TrackingState::Paused);

    h.settle();
    // --background: no first-run wizard, no startup windows.
    assert!(h.app().wizard().is_none());
    assert!(h.app().calibration_window().is_none());
    assert!(h.app().settings_dialog().is_none());
    assert!(h.platform.p().cursor.moves.is_empty());
}

#[test]
fn test_ui_requests_open_single_windows() {
    let _env = env();
    let mut h = build(Setup::default());
    h.app().on_ui_requested("settings");
    assert!(h.app().settings_dialog().is_some());
    // A page change marks this dialog: the same one stays open.
    h.app().settings_dialog_mut().unwrap().show_page("hotkeys");
    assert_eq!(h.send("settings").as_deref(), Some("ok"));
    assert_eq!(h.app().settings_dialog().unwrap().current_page(), "hotkeys");

    // "show" raises an open window instead of opening another one.
    h.app().on_ui_requested("show");
    assert_eq!(h.app().settings_dialog().unwrap().current_page(), "hotkeys");
    h.app().settings_dialog_mut().unwrap().reject();
    h.settle();
    assert!(h.app().settings_dialog().is_none());

    // Without a system tray, "show" falls back to the settings.
    h.app().on_ui_requested("show");
    assert!(h.app().settings_dialog().is_some());
    h.app().settings_dialog_mut().unwrap().reject();
    h.settle();

    // With a tray, "show" pops up the tray menu instead.
    h.tray.borrow_mut().area = true;
    h.app().on_ui_requested("show");
    assert_eq!(h.tray.borrow().popups, 1);
    assert!(h.app().settings_dialog().is_none());

    h.app().on_tray_action(TrayAction::OpenAbout);
    assert!(h.app().about().unwrap().is_open());
    h.app().on_tray_action(TrayAction::OpenPreview);
    assert!(h.app().preview().unwrap().is_shown());
    assert_eq!(h.worker().s().previews.last(), Some(&true));
    h.app().on_tray_action(TrayAction::OpenPreview);
    assert!(h.app().preview().unwrap().is_shown());
    h.app().preview_mut().unwrap().close();
    h.step();
    assert_eq!(h.worker().s().previews.last(), Some(&false));
}

#[test]
fn test_calibration_opens_once() {
    let _env = env();
    let logs = capture_logs();
    let opened = || {
        logs.records()
            .iter()
            .filter(|(_, m)| m.starts_with("Opening the calibration"))
            .count()
    };
    let mut h = build(Setup {
        monitors: two_monitors(),
        ..Setup::default()
    });
    h.emit(ControllerEvent::CalibrationRequired("hotkey".into()));
    assert!(h.app().calibration_window().unwrap().is_active());
    assert_eq!(h.controller().state(), TrackingState::Calibrating);

    h.emit(ControllerEvent::CalibrationRequired("ipc".into()));
    h.app().on_tray_action(TrayAction::OpenCalibration);
    assert_eq!(h.send("calibrate").as_deref(), Some("ok"));
    assert!(h.app().calibration_window().unwrap().is_active());
    assert_eq!(opened(), 1);

    h.app().calibration_window_mut().unwrap().cancel();
    h.settle();
    assert!(h.app().calibration_window().is_none());
    assert_eq!(h.controller().state(), TrackingState::NeedsCalibration);

    h.app().on_tray_action(TrayAction::OpenCalibration);
    assert!(h.app().calibration_window().unwrap().is_active());
    assert_eq!(opened(), 2);
}

#[test]
fn test_automatic_calibration_reasons_only_notify() {
    let _env = env();
    let mut h = build(Setup {
        monitors: two_monitors(),
        quiet_s: 0.0,
        ..Setup::default()
    });
    h.tray_messages();
    let reason = "the monitor layout changed";
    h.emit(ControllerEvent::CalibrationRequired(reason.into()));
    h.emit(ControllerEvent::CalibrationRequired(reason.into()));
    assert!(h.app().calibration_window().is_none());
    let shown = h.tray.borrow().messages.clone();
    assert_eq!(shown.len(), 1);
    assert_eq!(shown[0].0, "Calibration needed");
    assert!(shown[0].1.contains(reason));

    // Another notification replaced ours: clicking it must not calibrate.
    h.emit(ControllerEvent::CalibrationRequired(
        "another reason".into(),
    ));
    h.emit(ControllerEvent::Notify {
        title: "Camera".into(),
        message: "Camera unavailable".into(),
    });
    h.app().on_message_clicked();
    assert!(h.app().calibration_window().is_none());

    h.emit(ControllerEvent::CalibrationRequired(
        "a third reason".into(),
    ));
    h.app().on_message_clicked();
    assert!(h.app().calibration_window().unwrap().is_active());
}

#[test]
fn test_single_monitor_gets_no_calibration_nag() {
    let _env = env();
    let mut h = build(Setup {
        quiet_s: 0.0,
        ..Setup::default()
    });
    h.tray_messages();
    h.emit(ControllerEvent::CalibrationRequired(
        "the monitor layout changed".into(),
    ));
    assert!(h.titles().is_empty());
}

#[test]
fn test_helpers_follow_the_controller() {
    let _env = env();
    let mut h = build(Setup::default());

    h.emit(ControllerEvent::GuardChanged(true));
    assert!(h.app().curtain().unwrap().is_showing());
    h.emit(ControllerEvent::GuardChanged(false));
    assert!(!h.app().curtain().unwrap().is_showing());

    h.emit(ControllerEvent::AwayWarning(8.0));
    assert!(h.app().countdown().unwrap().active());
    assert!(h.app().countdown().unwrap().visible());
    h.emit(ControllerEvent::AwayCancelled);
    assert!(!h.app().countdown().unwrap().active());

    assert!(!h.app().overlay().unwrap().enabled());
    let mut updated = h.controller().settings().clone();
    updated.ui.show_gaze_overlay = true;
    h.controller().apply_settings(&updated);
    h.step();
    assert!(h.app().overlay().unwrap().enabled());
    assert!(Settings::load(&paths::settings_file()).ui.show_gaze_overlay);
}

#[test]
fn test_first_run_wizard_then_calibration() {
    let _env = env();
    let mut h = build(Setup {
        settings: Some(Settings::default()),
        monitors: two_monitors(),
        ..foreground()
    });
    h.settle();
    assert!(h.app().wizard().is_some());

    // While the wizard is open, "settings" raises the wizard.
    h.app().on_ui_requested("settings");
    assert!(h.app().settings_dialog().is_none());

    h.app().wizard_mut().unwrap().accept(); // "Calibrate now" is ticked by default
    h.settle();
    assert!(h.app().wizard().is_none());
    assert!(h.controller().settings().general.first_run_done);
    assert!(
        Settings::load(&paths::settings_file())
            .general
            .first_run_done
    );
    assert!(h.app().calibration_window().unwrap().is_active());
    // "Start at login" was left unticked.
    assert_eq!(h.autostart.borrow().calls, ["refresh"]);
}

#[test]
fn test_putting_off_the_offered_calibration_says_what_that_means() {
    // r3-ux-docs-01: Esc on the calibration the setup assistant opened left no hint.
    let _env = env();
    let mut h = build(Setup {
        settings: Some(Settings::default()),
        monitors: two_monitors(),
        ..foreground()
    });
    h.tray_messages();
    h.settle();
    h.app().wizard_mut().unwrap().accept(); // "Calibrate now" is ticked with two monitors
    h.settle();
    h.app().calibration_window_mut().unwrap().cancel();
    h.settle();
    assert_eq!(h.titles(), ["Calibration needed"]);
    // One the user asked for and cancelled needs no such hint (even unannounced).
    h.app().announced.clear();
    h.app().on_tray_action(TrayAction::OpenCalibration);
    h.app().calibration_window_mut().unwrap().cancel();
    h.settle();
    assert_eq!(h.titles(), ["Calibration needed"]);
}

#[test]
fn test_background_start_skips_the_wizard() {
    let _env = env();
    let mut h = build(Setup {
        settings: Some(Settings::default()),
        ..Setup::default()
    });
    h.settle();
    assert!(h.app().wizard().is_none());
}

#[test]
fn test_calibrate_option_opens_the_calibration() {
    let _env = env();
    let mut h = build(Setup {
        args: RunArgs {
            background: true,
            calibrate: true,
            ..RunArgs::default()
        },
        ..Setup::default()
    });
    assert!(h.app().calibration_window().is_none()); // opens once the event loop runs
    h.settle();
    assert!(h.app().calibration_window().unwrap().is_active());
}

#[test]
fn test_second_instance_hands_over_and_exits() {
    let _env = env();
    let mut first = build(Setup::default());
    let second = launch(&mut first, foreground);
    assert_eq!(second.exit_code, Some(0));
    assert!(!second.app_built);
    assert_eq!(second.workers, 0);
    // The first instance was asked to show itself (no tray area: settings).
    assert!(first.app().settings_dialog().is_some());

    let third = launch(&mut first, || Setup {
        args: RunArgs {
            background: true,
            calibrate: true,
            ..RunArgs::default()
        },
        ..Setup::default()
    });
    assert_eq!(third.exit_code, Some(0));
    assert!(first.app().calibration_window().is_some());
}

#[test]
fn test_a_login_start_does_not_pop_up_the_running_instance() {
    // Regression (r2-ui-app-04): a --background launch asked the instance to show itself, which
    // popped up its tray menu, the setup assistant or the settings.
    let _env = env();
    let mut first = build(Setup::default());
    let second = launch(&mut first, Setup::default);
    assert_eq!(second.exit_code, Some(0));
    assert!(!second.app_built);
    first.settle();
    assert!(first.app().settings_dialog().is_none());
    assert!(first.app().wizard().is_none());
}

#[test]
fn test_a_launch_waits_for_an_instance_that_is_shutting_down() {
    // Regression (r2-ui-app-02): quitting and starting again right away left nothing running:
    // the quitting instance still held the lock while it released the camera, and the new
    // launch gave up at once.
    let _env = env();
    let holder = InstanceLock::new(None); // the quitting instance
    assert!(holder.acquire());
    let waits = Rc::new(RefCell::new(Vec::new()));
    let (seen, quitting) = (waits.clone(), holder.clone());
    let mut h = build(Setup {
        startup_wait_ms: 0,
        wait_ms: Some(Box::new(move |ms| {
            seen.borrow_mut().push(ms);
            if seen.borrow().len() == 2 {
                quitting.release(); // it has stopped the camera and exits
            }
        })),
        ..foreground()
    });
    holder.release();
    // This launch took over as the instance.
    assert_eq!(h.ctx.exit_code, None);
    assert!(h.ctx.app.is_some());
    assert!(h.ctx.server.as_ref().unwrap().lock().unwrap().is_held());
    assert_eq!(*waits.borrow(), [HANDOVER_RETRY_MS; 2]);
    h.ctx.shutdown();
}

#[test]
fn test_a_launch_hands_over_once_a_busy_instance_answers() {
    let _env = env();
    let busy = "error: Eye Tracker is starting or shutting down; try again";
    let replies = RefCell::new(vec![busy, busy, "ok"].into_iter());
    let sent = Rc::new(RefCell::new(Vec::new()));
    let log = sent.clone();
    let holder = InstanceLock::new(None); // an instance that is still starting up
    assert!(holder.acquire());
    let h = build(Setup {
        startup_wait_ms: 0,
        send_command: Some(Box::new(move |command, _timeout, _wait| {
            log.borrow_mut().push(command.to_string());
            replies.borrow_mut().next().map(str::to_string)
        })),
        wait_ms: Some(Box::new(|_ms| {})),
        ..foreground()
    });
    holder.release();
    assert_eq!(h.ctx.exit_code, Some(0));
    assert!(h.ctx.app.is_none());
    assert_eq!(*sent.borrow(), ["show", "show", "show"]);
}

#[test]
fn test_options_for_a_new_instance_are_reported_as_not_applied() {
    // Regression (r2-docs-15): --trace (and --camera, --backend) were silently dropped when the
    // app was already running, and no trace file appeared.
    let env = env();
    let mut first = build(Setup::default());
    let trace = env._dirs.path().join("trace.jsonl");
    let path = trace.clone();
    let second = launch(&mut first, move || Setup {
        args: RunArgs {
            background: true,
            trace: Some(path),
            camera: Some("clip.mp4".into()),
            ..RunArgs::default()
        },
        ..Setup::default()
    });
    assert_eq!(second.exit_code, Some(0));
    assert!(!trace.exists());
    let messages: Vec<&String> = second
        .warnings
        .iter()
        .filter(|m| m.contains("not applied"))
        .collect();
    assert_eq!(messages.len(), 1);
    assert!(messages[0].contains("--trace, --camera were not applied"));
    assert!(!messages[0].contains("--backend"));
    assert!(messages[0].contains(&cli_command_text(&["ctl", "quit"])));

    let third = launch(&mut first, || Setup {
        args: RunArgs {
            background: true,
            calibrate: true,
            ..RunArgs::default()
        },
        ..Setup::default()
    });
    assert_eq!(third.exit_code, Some(0)); // nothing ignored
    assert!(!third.warnings.iter().any(|m| m.contains("not applied")));
}

#[test]
fn test_the_setup_assistant_from_the_settings_keeps_the_walk_away_lock() {
    // Regression (r2-ui-app-01): asking for the assistant "at next start" stored
    // first_run_done=false, which replaced the walk-away lock by a notification at once and for
    // good when the assistant was cancelled.
    let _env = env();
    let mut h = build(Setup::default());
    h.app().open_settings();
    assert!(h.app().settings_dialog().is_some());
    // "Run setup assistant…"
    h.app()
        .process_settings_actions(vec![SettingsAction::SetupRequested]);
    assert!(h.app().wizard().is_some());
    assert!(h.controller().settings().general.first_run_done);
    h.app().wizard_mut().unwrap().reject();
    h.settle();
    assert!(h.app().wizard().is_none());
    assert!(h.controller().settings().general.first_run_done);
    assert!(
        Settings::load(&paths::settings_file())
            .general
            .first_run_done
    );
}

#[test]
fn test_session_overrides_stay_out_of_the_settings_file() {
    let _env = env();
    let mut h = build(Setup {
        args: RunArgs {
            background: true,
            camera: Some("clip.mp4".into()),
            backend: Some("lite".into()),
            ..RunArgs::default()
        },
        ..Setup::default()
    });
    assert_eq!(h.controller().settings().camera.device, "clip.mp4");
    assert_eq!(h.controller().settings().general.backend, "lite");
    let mut updated = h.controller().settings().clone();
    updated.general.notifications = false;
    h.controller().apply_settings(&updated);
    h.step();
    let on_disk = Settings::load(&paths::settings_file());
    assert!(!on_disk.general.notifications);
    assert_eq!(on_disk.camera.device, "0");
    assert_eq!(on_disk.general.backend, "auto");

    // A value the user picks explicitly is saved.
    let mut updated = h.controller().settings().clone();
    updated.camera.device = "1".into();
    h.controller().apply_settings(&updated);
    h.step();
    assert_eq!(Settings::load(&paths::settings_file()).camera.device, "1");
}

#[test]
fn test_log_level_follows_settings_unless_given() {
    let _env = env();
    LEVELS.lock().clear();
    LOG_SETUPS.lock().clear();
    let mut settings = Settings::default();
    settings.general.first_run_done = true;
    settings.general.log_level = "DEBUG".into();
    let mut h = build(Setup {
        settings: Some(settings.clone()),
        ..Setup::default()
    });
    assert_eq!(LEVELS.lock().last().map(String::as_str), Some("DEBUG"));
    let mut updated = h.controller().settings().clone();
    updated.general.log_level = "WARNING".into();
    h.controller().apply_settings(&updated);
    h.step();
    assert_eq!(LEVELS.lock().last().map(String::as_str), Some("WARNING"));
    // Console only until this process is known to be the instance, then the file too.
    assert_eq!(
        *LOG_SETUPS.lock(),
        [("INFO".to_string(), false), ("INFO".to_string(), true)]
    );
    drop(h);

    LEVELS.lock().clear();
    let mut locked = build(Setup {
        settings: Some(settings),
        args: RunArgs {
            background: true,
            log_level: Some("ERROR".into()),
            ..RunArgs::default()
        },
        ..Setup::default()
    });
    assert_eq!(LOG_SETUPS.lock().last(), Some(&("ERROR".to_string(), true)));
    let mut updated = locked.controller().settings().clone();
    updated.general.log_level = "INFO".into();
    locked.controller().apply_settings(&updated);
    locked.step();
    assert!(LEVELS.lock().is_empty());
}

#[test]
fn test_shutdown_is_clean_and_idempotent() {
    let _env = env();
    let mut h = build(Setup::default());
    h.app().open_settings();
    h.app().open_about();
    h.app().open_preview();
    h.app().open_calibration("user");
    h.emit(ControllerEvent::GuardChanged(true));
    h.emit(ControllerEvent::AwayWarning(5.0));
    assert_eq!(h.controller().state(), TrackingState::Calibrating);

    h.ctx.shutdown();
    assert!(h.app().closed());
    assert!(h.worker().s().stopped);
    assert!(!h.ctx.server.as_ref().unwrap().is_listening());
    assert!(!ipc::is_running(ipc::PROBE_TIMEOUT_MS));
    assert!(h.app().settings_dialog().is_none());
    assert!(h.app().calibration_window().is_none());
    assert!(h.app().preview().is_none());
    assert!(!h.app().curtain().unwrap().is_showing());
    assert!(!h.tray.borrow().visible);
    // The open calibration was cancelled before the controller stopped.
    assert_ne!(h.controller().state(), TrackingState::Calibrating);
    assert!(h.app().handle_command("status").starts_with("error:"));

    h.ctx.shutdown(); // idempotent
    h.app().open_settings(); // ignored after shutdown
    assert!(h.app().settings_dialog().is_none());
    h.settle();
}

#[test]
fn test_startup_failure_returns_an_error() {
    let _env = env();
    let h = build(Setup {
        broken: true,
        ..Setup::default()
    });
    assert_eq!(h.ctx.exit_code, Some(1));
    assert!(h.ctx.app.is_none());
    assert!(!ipc::is_running(ipc::PROBE_TIMEOUT_MS));
    // The instance lock was released too: the next launch may become the instance.
    let lock = InstanceLock::new(None);
    assert!(lock.acquire());
    assert!(lock.is_held());
    lock.release();
}

// -------------------------------------------------------------------- event loop
#[test]
fn test_exec_quits_on_ipc_quit() {
    let _env = env();
    let mut h = build(Setup::default());
    assert_eq!(h.send("quit").as_deref(), Some("ok"));
    assert!(!h.app().quit_requested()); // the reply goes out first
    h.clock.advance(QUIT_DELAY_MS as f64 / 1000.0);
    h.step();
    assert!(h.app().quit_requested());
    h.ctx.shutdown(); // what the event loop does once it ends
    assert!(h.app().closed());
    assert!(h.worker().s().stopped);
    assert!(!ipc::is_running(ipc::PROBE_TIMEOUT_MS));
}

#[test]
fn test_the_instance_lock_is_held_until_the_controller_has_stopped() {
    // Regression: the server (and with it the lock) was closed before the camera was released,
    // so a relaunch right after a quit started a second instance.
    let _env = env();
    let mut h = build(Setup::default());
    let held_while_stopping = Arc::new(Mutex::new(None));
    let seen = held_while_stopping.clone();
    h.worker().s().on_stop = Some(Box::new(move || {
        let other = InstanceLock::new(None);
        let acquired = other.acquire();
        other.release();
        *seen.lock() = Some(!acquired);
    }));
    h.ctx.shutdown();
    assert!(h.worker().s().stopped);
    assert_eq!(*held_while_stopping.lock(), Some(true));
    let next = InstanceLock::new(None);
    assert!(next.acquire()); // released once the app has stopped
    next.release();
}

#[test]
fn test_exec_quits_on_tray_quit() {
    let _env = env();
    let mut h = build(Setup::default());
    h.app().on_tray_action(TrayAction::Quit);
    assert!(h.app().quit_requested());
    h.ctx.shutdown();
    assert!(h.app().closed());
}

#[cfg(unix)]
fn sigint_handler() -> usize {
    // SAFETY: reads the current action into a zeroed struct.
    unsafe {
        let mut action: libc::sigaction = std::mem::zeroed();
        libc::sigaction(libc::SIGINT, std::ptr::null(), &mut action);
        action.sa_sigaction
    }
}

#[cfg(unix)]
#[test]
fn test_exec_quits_on_ctrl_c() {
    let _env = env();
    if sigint_handler() == libc::SIG_IGN {
        return; // SIGINT is ignored in this process (started in the background)
    }
    let mut h = build(Setup::default());
    let before = sigint_handler();
    let signals = SignalGuard::install();
    // SAFETY: our handler is installed; it only sets a flag.
    unsafe {
        libc::raise(libc::SIGINT);
    }
    h.step();
    drop(signals);
    assert!(h.app().quit_requested());
    assert!(h.app().interrupted());
    h.ctx.shutdown();
    assert!(h.app().closed());
    assert_eq!(sigint_handler(), before);
}

#[cfg(unix)]
#[test]
fn test_signal_handler_is_installed_only_inside_the_block() {
    let _env = env();
    if sigint_handler() == libc::SIG_IGN {
        return;
    }
    let before = sigint_handler();
    {
        let _signals = SignalGuard::install();
        assert_eq!(
            sigint_handler(),
            on_signal as extern "C" fn(libc::c_int) as usize
        );
        // SAFETY: our handler is installed; it only sets a flag.
        unsafe {
            libc::raise(libc::SIGINT);
        }
        // Deferred to the event loop.
        assert!(INTERRUPTED.swap(false, Ordering::SeqCst));
    }
    assert_eq!(sigint_handler(), before);
}

#[cfg(unix)]
#[test]
fn test_ignored_signals_stay_ignored() {
    let _env = env();
    // SAFETY: sets SIGINT to ignored and restores the previous action afterwards.
    unsafe {
        let mut ignore: libc::sigaction = std::mem::zeroed();
        ignore.sa_sigaction = libc::SIG_IGN;
        let mut previous: libc::sigaction = std::mem::zeroed();
        libc::sigaction(libc::SIGINT, &ignore, &mut previous);
        {
            let _signals = SignalGuard::install();
            assert_eq!(sigint_handler(), libc::SIG_IGN);
        }
        assert_eq!(sigint_handler(), libc::SIG_IGN);
        libc::sigaction(libc::SIGINT, &previous, std::ptr::null_mut());
    }
}

// -------------------------------------------------------------------- review fixes
#[test]
fn test_calibration_prompt_repeats_after_the_calibration_was_usable() {
    // Regression (ui_app-08, journeys-19): each reason was announced once per process.
    let _env = env();
    let mut h = build(Setup {
        monitors: two_monitors(),
        ..foreground()
    });
    h.tray_messages();
    let reason = "the monitor layout changed";
    h.emit(ControllerEvent::CalibrationRequired(reason.into()));
    h.emit(ControllerEvent::CalibrationRequired(reason.into()));
    assert_eq!(h.titles(), ["Calibration needed"]); // while still uncalibrated: once
    // The user recalibrates (or returns to a known desk): usable again.
    h.controller()
        .finish_calibration(Some(make_calibration(&two_monitors(), None)));
    assert!(h.controller().is_calibrated());
    h.step();
    assert!(h.app().announced.is_empty());
    // Days later, another dock: the same reason text deserves a new notification.
    let mut dock = two_monitors();
    dock.push(Monitor::new(2, "dock", Rect::new(1600, 0, 800, 600)));
    *h.layout.lock() = dock;
    h.controller().refresh_monitors();
    assert!(!h.controller().is_calibrated());
    h.emit(ControllerEvent::CalibrationRequired(reason.into()));
    assert_eq!(h.titles(), ["Calibration needed", "Calibration needed"]);
}

#[test]
fn test_a_calibration_prompt_that_was_not_shown_is_not_counted() {
    let _env = env();
    let mut h = build(Setup {
        monitors: two_monitors(),
        ..foreground()
    });
    h.tray_messages();
    let mut quiet = h.controller().settings().clone();
    quiet.general.notifications = false;
    h.controller().apply_settings(&quiet);
    h.step();
    h.emit(ControllerEvent::CalibrationRequired(
        "the monitor layout changed".into(),
    ));
    assert!(h.titles().is_empty());
    let mut loud = h.controller().settings().clone();
    loud.general.notifications = true;
    h.controller().apply_settings(&loud);
    h.step();
    h.emit(ControllerEvent::CalibrationRequired(
        "the monitor layout changed".into(),
    ));
    assert_eq!(h.titles(), ["Calibration needed"]);
}

#[test]
fn test_a_prompt_muted_after_a_login_start_is_shown_later() {
    let _env = env();
    let mut h = build(Setup {
        monitors: two_monitors(),
        quiet_s: 0.5,
        ..Setup::default()
    }); // --background: notifications muted at first
    h.tray_messages();
    assert!(h.app().tray().unwrap().muted_for() > 0.0);
    // A dock brings a monitor up while the login start is still quiet.
    h.emit(ControllerEvent::CalibrationRequired(
        "the monitor layout changed".into(),
    ));
    assert!(h.titles().is_empty());
    assert!(h.wait_for(|h| !h.titles().is_empty(), 3.0));
    assert_eq!(h.titles(), ["Calibration needed"]);
    h.app().on_message_clicked();
    assert!(h.app().calibration_window().unwrap().is_active());
}

#[test]
fn test_an_uncalibrated_login_start_still_says_why_nothing_switches() {
    // r3-ux-docs-01: a --background start dropped the calibration prompt for good.
    let _env = env();
    let mut h = build(Setup {
        monitors: two_monitors(),
        quiet_s: 0.3,
        ..Setup::default()
    }); // --background, set up, never calibrated
    h.tray_messages();
    h.settle();
    assert!(h.titles().is_empty()); // a login start stays quiet at first
    assert!(h.wait_for(|h| !h.titles().is_empty(), 3.0));
    assert_eq!(h.titles(), ["Calibration needed"]);
}

#[test]
fn test_no_calibration_prompt_where_it_changes_nothing() {
    // r3-ux-docs-03: with switching turned off a calibration achieves nothing.
    let _env = env();
    let mut settings = Settings::default();
    settings.general.first_run_done = true;
    settings.switching.enabled = false;
    let mut h = build(Setup {
        settings: Some(settings),
        monitors: two_monitors(),
        ..foreground()
    });
    h.tray_messages();
    h.settle();
    h.emit(ControllerEvent::CalibrationRequired(
        "not calibrated yet".into(),
    ));
    assert!(h.titles().is_empty());
    assert_eq!(h.controller().state(), TrackingState::Tracking);
}

#[test]
fn test_privacy_mode_kept_from_the_last_run_is_announced() {
    // r3-ux-docs-02: privacy mode survives a restart; the user is told once.
    let _env = env();
    std::fs::create_dir_all(paths::state_file().parent().unwrap()).unwrap();
    std::fs::write(paths::state_file(), "{\"privacy_mode\": true}\n").unwrap();
    let mut h = build(Setup {
        monitors: two_monitors(),
        quiet_s: 0.3,
        ..Setup::default()
    }); // a login start, e.g. after an overnight update
    h.tray_messages();
    assert!(h.controller().privacy());
    assert_eq!(h.controller().state(), TrackingState::Privacy);
    assert!(!h.worker().s().active.contains(&true)); // the camera never came on
    assert!(h.wait_for(|h| !h.titles().is_empty(), 3.0));
    // No calibration prompt while the camera is off on purpose.
    assert_eq!(h.titles(), ["Privacy mode is still on"]);
}

#[test]
fn test_window_requests_during_a_calibration_raise_it_instead() {
    // Regression (ui_app-12): a dialog hidden under the calibration took its keyboard.
    let _env = env();
    let mut h = build(Setup::default());
    h.app().open_calibration("user");
    assert!(h.app().calibration_window().unwrap().is_active());
    h.app().on_ui_requested("settings"); // e.g. 'eye-tracker ctl settings'
    h.app().on_tray_action(TrayAction::OpenPreview);
    h.app().on_tray_action(TrayAction::OpenAbout);
    h.app().open_wizard();
    assert!(h.app().settings_dialog().is_none());
    assert!(h.app().preview().is_none());
    assert!(h.app().about().is_none());
    assert!(h.app().wizard().is_none());
    assert!(h.app().calibration_window().unwrap().is_active());
}

#[test]
fn test_unfinished_setup_is_offered_after_a_login_start() {
    // Regression (journeys-05): starting at login skipped the setup assistant forever.
    let _env = env();
    let mut h = build(Setup {
        settings: Some(Settings::default()), // first_run_done is false
        quiet_s: 0.3,
        ..Setup::default()
    });
    h.tray_messages();
    h.settle();
    assert!(h.app().wizard().is_none()); // a login start stays quiet
    assert!(h.wait_for(|h| !h.titles().is_empty(), 3.0));
    assert_eq!(h.titles(), ["Finish setting up Eye Tracker"]);
    h.app().on_message_clicked();
    assert!(h.app().wizard().is_some());
    h.app().wizard_mut().unwrap().reject();
    h.settle();
    assert!(h.app().wizard().is_none());
}

#[test]
fn test_a_second_launch_opens_the_unfinished_setup() {
    let _env = env();
    let mut h = build(Setup {
        settings: Some(Settings::default()),
        ..Setup::default()
    });
    h.settle();
    assert!(h.app().wizard().is_none());
    h.app().on_ui_requested("show"); // launched again from the Start menu
    assert!(h.app().wizard().is_some());
    assert!(h.app().settings_dialog().is_none());
}

#[test]
fn test_startup_points_the_login_item_at_this_copy() {
    let _env = env();
    let mut h = build(Setup::default());
    assert!(h.autostart.borrow().calls.is_empty()); // once the event loop runs
    h.settle();
    assert_eq!(h.autostart.borrow().calls, ["refresh"]);
}

#[test]
fn test_a_lock_holder_that_does_not_answer_keeps_other_launches_out() {
    // windows-09: the instance lock decides who runs, not a racy socket probe.
    let _env = env();
    let holder = InstanceLock::new(None); // e.g. an instance that is still starting up, or hangs
    assert!(holder.acquire());
    let blocked = build(Setup {
        startup_wait_ms: 300,
        handover_wait_ms: 300, // then it gives up
        ..Setup::default()
    });
    assert_eq!(blocked.ctx.exit_code, Some(1));
    assert!(blocked.ctx.app.is_none());
    assert!(blocked.workers.lock().is_empty());
    holder.release();
    let mut h = build(Setup::default());
    assert_eq!(h.ctx.exit_code, None);
    let lock = h.ctx.server.as_ref().unwrap().lock().unwrap().clone();
    assert!(lock.is_held());
    h.ctx.shutdown();
    assert!(!lock.is_held()); // released for the next launch
}

#[test]
fn test_a_calibration_reminder_after_being_away_is_shown_again() {
    // The controller reminds the user on their return, with the same reason.
    let _env = env();
    let mut h = build(Setup {
        monitors: two_monitors(),
        ..foreground()
    });
    h.tray_messages();
    let reason = "the monitor layout changed";
    h.emit(ControllerEvent::CalibrationRequired(reason.into()));
    assert_eq!(h.titles(), ["Calibration needed"]);
    for state in [
        TrackingState::Away,
        TrackingState::Locked,
        TrackingState::Privacy,
    ] {
        h.emit(ControllerEvent::StateChanged(state)); // gone for a while ...
        h.emit(ControllerEvent::StateChanged(
            TrackingState::NeedsCalibration,
        )); // ... and back
        h.emit(ControllerEvent::CalibrationRequired(reason.into())); // the reminder
    }
    assert_eq!(h.titles(), ["Calibration needed"; 4]);
    // Other states do not repeat it.
    h.emit(ControllerEvent::StateChanged(TrackingState::Paused));
    h.emit(ControllerEvent::CalibrationRequired(reason.into()));
    assert_eq!(h.titles().len(), 4);
}

#[test]
fn test_dismissing_the_curtain_tells_the_controller() {
    let _env = env();
    let mut h = build(Setup::default());
    h.controller().show_curtain(true); // what the shoulder guard does
    h.step();
    assert!(h.app().curtain().unwrap().is_showing());
    h.app().curtain_mut().unwrap().dismiss(); // Esc or the button
    assert!(!h.app().curtain().unwrap().is_showing());
    h.step();
    // The controller took the curtain down (switching resumes).
    assert!(!h.app().guard_active);
}

#[test]
fn test_clicking_a_permission_notification_opens_the_privacy_settings() {
    let _env = env();
    let platform = PermissionPlatform::new("unknown");
    let mut h = build(Setup {
        platform: Some(platform.clone()),
        ..foreground()
    });
    h.tray_messages();
    h.emit(ControllerEvent::Notify {
        title: "Camera access blocked".into(),
        message: "Allow camera access …".into(),
    });
    h.emit(ControllerEvent::PermissionNeeded("camera".into()));
    assert_eq!(h.titles(), ["Camera access blocked"]);
    h.app().on_message_clicked();
    assert_eq!(platform.opened(), ["camera"]);
    h.app().on_message_clicked(); // a click is used once
    assert_eq!(platform.opened(), ["camera"]);

    // Another notification replaced it: its click does not open the settings.
    h.emit(ControllerEvent::Notify {
        title: "Accessibility access needed".into(),
        message: "…".into(),
    });
    h.emit(ControllerEvent::PermissionNeeded("accessibility".into()));
    h.emit(ControllerEvent::Notify {
        title: "Tracking paused".into(),
        message: "…".into(),
    });
    h.app().on_message_clicked();
    assert_eq!(platform.opened(), ["camera"]);
}

#[test]
fn test_a_permission_without_a_shown_notification_is_not_clickable() {
    let _env = env();
    let platform = PermissionPlatform::new("unknown");
    let mut h = build(Setup {
        monitors: two_monitors(),
        platform: Some(platform.clone()),
        ..foreground()
    });
    h.tray_messages();
    h.emit(ControllerEvent::CalibrationRequired(
        "the monitor layout changed".into(),
    ));
    // Notifications off: the controller emits permission_needed alone.
    h.emit(ControllerEvent::PermissionNeeded("camera".into()));
    h.app().on_message_clicked(); // the calibration notification was clicked
    assert!(platform.opened().is_empty());
    assert!(h.app().calibration_window().is_some());
}

#[test]
fn test_permission_notifications_muted_after_a_login_start_are_not_clickable() {
    let _env = env();
    let platform = PermissionPlatform::new("unknown");
    let mut h = build(Setup {
        platform: Some(platform.clone()),
        ..Setup::default()
    }); // --background: notifications muted at first
    assert!(h.app().tray().unwrap().muted_for() > 0.0);
    h.emit(ControllerEvent::Notify {
        title: "Camera access blocked".into(),
        message: "…".into(),
    });
    h.emit(ControllerEvent::PermissionNeeded("camera".into()));
    h.app().on_message_clicked();
    assert!(platform.opened().is_empty());
}

fn permission_notice_muted_after_a_login_start(granted_meanwhile: bool) {
    // journeys-15: the controller says it once, at startup, inside the quiet period.
    let _env = env();
    let platform = PermissionPlatform::new("stale");
    let mut h = build(Setup {
        platform: Some(platform.clone()),
        quiet_s: 0.3,
        ..Setup::default()
    }); // --background: notifications muted at first
    h.tray_messages();
    assert!(h.app().tray().unwrap().muted_for() > 0.0);
    h.emit(ControllerEvent::Notify {
        title: "Accessibility access needed".into(),
        message: "Remove it and add it again.".into(),
    });
    h.emit(ControllerEvent::PermissionNeeded("accessibility".into()));
    assert!(h.titles().is_empty());
    if granted_meanwhile {
        *platform.accessibility.lock() = "granted"; // fixed before the quiet period ended
    }
    assert!(h.wait_for(|h| h.app().deferred_permission.is_none(), 3.0));
    if granted_meanwhile {
        assert!(h.titles().is_empty());
        return;
    }
    assert_eq!(h.titles(), ["Accessibility access needed"]);
    h.app().on_message_clicked();
    assert_eq!(platform.opened(), ["accessibility"]);
}

#[test]
fn test_a_permission_notice_muted_after_a_login_start_is_shown_later() {
    permission_notice_muted_after_a_login_start(false);
    permission_notice_muted_after_a_login_start(true);
}

#[test]
fn test_a_muted_notice_is_not_repeated_when_notifications_were_off() {
    // Without a notify just before it, permission_needed has nothing to repeat.
    let _env = env();
    let platform = PermissionPlatform::new("stale");
    let mut h = build(Setup {
        platform: Some(platform),
        quiet_s: 0.3,
        ..Setup::default()
    });
    h.tray_messages();
    h.emit(ControllerEvent::PermissionNeeded("accessibility".into()));
    assert!(h.app().deferred_permission.is_none());
    assert!(!h.wait_for(|h| !h.titles().is_empty(), 0.8));
}

#[test]
fn test_the_login_item_is_left_alone_where_unsupported() {
    let _env = env();
    let mut h = build(Setup {
        autostart_supported: false,
        ..Setup::default()
    });
    h.settle();
    assert!(h.autostart.borrow().calls.is_empty());
}

#[test]
fn test_the_instance_lock_is_released_when_startup_raises() {
    // windows-09: an unexpected error must not keep every later launch out.
    let _env = env();
    let failed = std::panic::catch_unwind(|| {
        build(Setup {
            send_command: Some(Box::new(|_, _, _| panic!("send_command failed"))),
            ..Setup::default()
        })
    });
    assert!(failed.is_err());
    let lock = InstanceLock::new(None);
    assert!(lock.acquire());
    assert!(lock.is_held());
    lock.release();
}

#[test]
fn test_handover_requests_and_busy_replies() {
    let request = |background, calibrate| {
        handover_request(&AppOptions {
            background,
            calibrate,
            log_level_locked: false,
        })
    };
    assert_eq!(request(false, false), "show");
    assert_eq!(request(true, false), "status");
    assert_eq!(request(true, true), "calibrate");
    assert!(is_busy(
        "error: Eye Tracker is starting or shutting down; try again"
    ));
    assert!(!is_busy("error: unknown command 'x'"));
    assert!(!is_busy("ok"));
}

#[test]
fn test_overrides_are_applied_for_the_session() {
    let mut settings = Settings::default();
    let args = RunArgs {
        camera: Some(" clip.mp4 ".into()),
        backend: Some(String::new()),
        ..RunArgs::default()
    };
    let overrides = apply_overrides(&mut settings, &args);
    assert_eq!(settings.camera.device, "clip.mp4");
    assert_eq!(
        overrides,
        [("camera.device", ("0".to_string(), "clip.mp4".to_string()))]
    );
    assert_eq!(setting(&settings, "general.backend"), "auto");
}
