//! The tray application: startup sequence, single instance and window management.
//!
//! [`run_app`] is what `eye-tracker` (`run`) executes. It is split into [`build_app`], which
//! creates everything, [`AppContext::start`], which starts tracking and the UI, and the eframe
//! event loop, so tests can build and start the whole app without entering the loop. The startup
//! order matters:
//!
//! 1. logging (console only for now), so native libraries are quietened before they load;
//! 2. `PlatformServices::prepare_process` before any window or thread exists (DPI awareness);
//! 3. the single-instance check: whoever takes the [`InstanceLock`] is the instance. Otherwise
//!    (or if an older version answers) the running app is asked to show itself (or to
//!    calibrate; a login start only checks that it answers) and this process exits without
//!    touching the log file or the settings. An instance that is shutting down no longer answers
//!    but still holds the lock for a moment: the launch waits for it to exit and then becomes
//!    the instance;
//! 4. file logging, settings, the command socket, then (once the event loop runs, which the
//!    tray needs) the [`Controller`] and the UI; the login item is pointed at this copy if it
//!    moved.
//!
//! A process started at login (`--background`) stays quiet for [`STARTUP_QUIET_S`]. If the
//! first-run setup was never finished, it then offers the setup assistant with a notification,
//! and a second launch (or `ctl show`) opens the assistant.
//!
//! [`EyeTrackerApp`] owns the controller and every window. Its wiring runs in
//! [`EyeTrackerApp::step`] (commands, controller ticks and events, tray clicks, timers, the
//! update service), which needs no display; [`EyeTrackerApp::draw`] only draws the windows that
//! are open. It opens windows on request (tray menu, IPC, hotkeys) and keeps at most one of each.
//!
//! # The event loop
//!
//! The root viewport is a 1×1 transparent, click-through window, hidden while no window is
//! open (see [`Runner`]).

use std::any::Any;
use std::collections::HashSet;
use std::io::IsTerminal;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, TryRecvError};
use egui::{ViewportCommand, ViewportId};
use parking_lot::Mutex;
use serde_json::Value;

use crate::cli::{EXIT_ERROR, EXIT_OK, cli_command_text};
use crate::config::Settings;
use crate::engine::controller::{Controller, ControllerEvent, ControllerOptions};
use crate::ipc::{self, InstanceLock, InstanceServer};
use crate::logging_setup;
use crate::platform::autostart::{AutostartBackend, SystemAutostart};
use crate::platform::{PlatformServices, get_platform};
use crate::types::{GazePoint, Monitor, TrackingState};
use crate::ui::about::AboutDialog;
use crate::ui::calibration_window::{
    CalibrationAction, CalibrationOptions, CalibrationWindow, Inputs as CalibrationInputs,
};
use crate::ui::countdown::{CountdownInputs, CountdownSettings, CountdownToast, ToastEnv};
use crate::ui::curtain::{CurtainAction, CurtainInputs, CurtainWindow, PrivacyCurtain};
use crate::ui::icons;
use crate::ui::overlay::{GazeOverlay, OverlayInputs};
use crate::ui::preview::{PreviewAction, PreviewInputs, PreviewStats, PreviewWindow};
use crate::ui::settings_dialog::{
    SettingsAction, SettingsDialog, SettingsInputs, load_diagnostics_text,
};
use crate::ui::tray::{Tray, TrayAction, TrayBackend, TrayInputs, TrayStats, native_backend};
use crate::ui::update_dialog::{UpdateAction, UpdateDialog, UpdateInputs};
use crate::ui::wizard::{FirstRunWizard, WizardAction, WizardInputs};
use crate::update::service::{Event as UpdateEvent, Phase, UpdateOptions, UpdateService};
use crate::vision::worker::Clock;
use crate::{APP_NAME, APP_SLUG, VERSION, paths};

/// `calibration_required` reasons that are explicit user requests (the recalibrate hotkey,
/// `eye-tracker ctl calibrate`): the calibration opens at once. Any other reason (a changed
/// monitor layout, another vision backend) only produces a notification that opens it when
/// clicked: covering every screen unannounced could hide a dialog the user is working with, such
/// as the operating system's "keep these display settings?" countdown.
pub const EXPLICIT_CALIBRATION_REASONS: [&str; 3] = ["hotkey", "ipc", "user"];
/// Calibrations the app offered rather than the user asked for (`open_calibration` reasons):
/// closing one without saving brings the "Calibration needed" notice.
const OFFERED_CALIBRATIONS: [&str; 2] = ["wizard", "cli"];
/// With `--background` (login start) notifications stay quiet this long.
pub const STARTUP_QUIET_S: f64 = 20.0;
/// Clicking a "calibration needed" or "finish setup" notification later than this does nothing.
pub const PROMPT_CLICK_WINDOW_S: f64 = 600.0;
/// A `permission_needed` belongs to the controller notification emitted this recently (the
/// controller emits both back to back).
const PERMISSION_NOTICE_S: f64 = 1.0;
/// How often the event loop wakes up at the least, so Ctrl+C is handled.
pub const SIGNAL_POLL_MS: u64 = 500;
/// Delay before quitting on an IPC `quit`, so the reply reaches the client.
const QUIT_DELAY_MS: u64 = 100;
/// Margin after the startup mute ends before a deferred notification is shown.
const AFTER_QUIET_MS: u64 = 250;
/// Options that only a new instance applies.
const STARTUP_ONLY_OPTIONS: [&str; 3] = ["--trace", "--camera", "--backend"];
/// How long a launch keeps waiting for an instance that holds the lock but does not answer,
/// after [`ipc::STARTUP_WAIT_MS`]: one that is shutting down (its event loop has stopped while
/// it releases the camera, which can take a few seconds) exits within this time, and the launch
/// then takes its place.
pub const HANDOVER_WAIT_MS: u64 = 10_000;
/// Reply timeout for each retry during [`HANDOVER_WAIT_MS`].
const HANDOVER_REPLY_MS: u64 = 500;
/// Pause between two retries during [`HANDOVER_WAIT_MS`].
const HANDOVER_RETRY_MS: u64 = 250;
/// Part of the reply of an instance that cannot take commands yet or any more.
const BUSY_MARKER: &str = "is starting or shutting down";
/// `Controller::set_preview` owners: the preview window and the setup assistant.
const PREVIEW_OWNER: &str = "preview";
const WIZARD_OWNER: &str = "wizard";
/// The system's screens (for the curtain and the countdown toast) are read at most this often.
const SCREENS_POLL_S: f64 = 1.0;
/// The tray icon's size in pixels (the macOS menu bar is 22 points high, drawn at 2×).
const TRAY_ICON_PX: u32 = if cfg!(target_os = "macos") { 44 } else { 32 };

/// Command-line switches that change how the app starts (see `cli::run_args`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RunArgs {
    pub config_dir: Option<PathBuf>,
    pub log_level: Option<String>,
    pub camera: Option<String>,
    pub backend: Option<String>,
    pub background: bool,
    pub calibrate: bool,
    pub trace: Option<PathBuf>,
}

/// The switches of [`RunArgs`] the running app looks at.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AppOptions {
    /// Started at login: no first-run wizard, no startup notifications, and an instance that is
    /// already running is not asked to show itself.
    pub background: bool,
    /// Open the calibration right after start.
    pub calibrate: bool,
    /// `--log-level` was given, so the settings' level is ignored.
    pub log_level_locked: bool,
}

/// The kinds of notification that do something when clicked: open the calibration, the setup
/// assistant, the OS privacy settings of a missing permission, or the update window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prompt {
    Calibrate,
    Setup,
    Permission,
    Update,
}

/// What a timer of the app runs (the Python app's single-shot `QTimer`s).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Timer {
    AfterStart,
    PromptSetup,
    AnnounceDeferred,
    AnnounceDeferredPermission,
    RemindPrivacyMode,
    Quit,
}

/// `(command, timeout_ms, wait_ms) -> reply` ([`ipc::send_command`]).
pub type SendCommand = Box<dyn FnMut(&str, u64, u64) -> Option<String>>;
/// A start-at-login backend for each window that shows the login item.
pub type AutostartFactory = Rc<dyn Fn() -> Box<dyn AutostartBackend>>;

/// What [`build_app`] uses from the outside world; the defaults are the real ones and tests
/// replace them (the Python tests passed `platform`, `controller_kwargs` and monkeypatched the
/// rest).
pub struct BuildOptions {
    /// OS integration; defaults to [`get_platform`].
    pub platform: Option<Arc<dyn PlatformServices>>,
    /// The controller's seams (`trace_path` is set from `--trace`).
    pub controller: ControllerOptions,
    /// Defaults to the OS tray ([`native_backend`]), created once the event loop runs.
    pub tray_backend: Option<Box<dyn TrayBackend>>,
    /// Defaults to [`SystemAutostart`].
    pub autostart: Option<AutostartFactory>,
    /// Monotonic seconds of the app's timers (and the tray's); defaults to the process clock.
    pub clock: Option<Clock>,
    /// Defaults to the real update service.
    pub updates: Option<UpdateOptions>,
    pub startup_quiet_s: f64,
    /// See [`ipc::STARTUP_WAIT_MS`].
    pub startup_wait_ms: u64,
    pub handover_wait_ms: u64,
    pub send_command: SendCommand,
    /// Sleeps between two hand-over attempts.
    pub wait_ms: Box<dyn FnMut(u64)>,
    /// `logging_setup::setup_logging` and `set_level`.
    pub setup_logging: fn(&str, bool, bool),
    pub set_level: fn(&str),
    /// Tells a user without a console that startup failed.
    pub show_error: fn(&str),
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            platform: None,
            controller: ControllerOptions::default(),
            tray_backend: None,
            autostart: None,
            clock: None,
            updates: None,
            startup_quiet_s: STARTUP_QUIET_S,
            startup_wait_ms: ipc::STARTUP_WAIT_MS,
            handover_wait_ms: HANDOVER_WAIT_MS,
            send_command: Box::new(|command, timeout_ms, wait_ms| {
                ipc::send_command(command, timeout_ms, wait_ms)
                    .ok()
                    .flatten()
            }),
            wait_ms: Box::new(|ms| std::thread::sleep(Duration::from_millis(ms))),
            setup_logging: logging_setup::setup_logging,
            set_level: logging_setup::set_level,
            show_error,
        }
    }
}

// ============================================================================ run
/// Start the tray app and run the event loop. Returns the exit code.
pub fn run_app(args: &RunArgs) -> i32 {
    let mut context = build_app(args, BuildOptions::default());
    let code = match context.exit_code {
        Some(code) => code,
        None => run_event_loop(&mut context),
    };
    context.shutdown();
    logging_setup::shutdown_logging();
    code
}

/// What [`build_app`] created. [`AppContext::start`] starts it.
///
/// `exit_code` is set when there is nothing to run: the request was handed to an instance that
/// is already running, or startup failed.
pub struct AppContext {
    pub app: Option<EyeTrackerApp>,
    pub server: Option<InstanceServer>,
    pub exit_code: Option<i32>,
    /// The event loop's context, woken by every command that arrives so it is answered at once.
    waker: Arc<Mutex<Option<egui::Context>>>,
    show_error: fn(&str),
}

impl AppContext {
    fn exit(code: i32, show_error: fn(&str)) -> Self {
        Self {
            app: None,
            server: None,
            exit_code: Some(code),
            waker: Arc::default(),
            show_error,
        }
    }

    pub fn controller(&mut self) -> Option<&mut Controller> {
        self.app.as_mut()?.controller.as_mut()
    }

    /// Create the controller and the UI and start tracking (on the UI thread, once the event loop
    /// runs: the tray needs it). Returns the exit code when that failed.
    pub fn start(&mut self) -> Option<i32> {
        let app = self.app.as_mut()?;
        let Err(panic) = catch_unwind(AssertUnwindSafe(|| app.start())) else {
            return None;
        };
        let reason = panic_text(panic.as_ref());
        log::error!("{APP_NAME} could not start: {reason}");
        if let Some(server) = &mut self.server {
            server.close();
        }
        if let Some(mut app) = self.app.take() {
            let _ = catch_unwind(AssertUnwindSafe(|| app.shutdown()));
        }
        (self.show_error)(&format!(
            "{APP_NAME} could not start:\n\n{reason}\n\nDetails are in the log file:\n{}",
            paths::log_file().display()
        ));
        self.exit_code = Some(EXIT_ERROR);
        self.exit_code
    }

    /// Stop accepting commands, close every window and release the camera. Idempotent.
    pub fn shutdown(&mut self) {
        // The app first: until it has released the camera, hotkeys and files, the server keeps
        // the instance lock so a relaunch waits instead of starting a second instance.
        if let Some(app) = &mut self.app {
            app.shutdown();
        }
        if let Some(server) = &mut self.server {
            server.close();
        }
    }
}

/// Releases the instance lock when a panic unwinds through the startup (never keep a later
/// launch out after failing there).
struct ReleaseOnPanic<'a>(&'a InstanceLock);

impl Drop for ReleaseOnPanic<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.release();
        }
    }
}

/// Check for a running instance, then create the app with its settings and the command socket.
/// Nothing is started yet (see [`AppContext::start`]).
pub fn build_app(args: &RunArgs, mut options: BuildOptions) -> AppContext {
    let text = |value: &Option<String>| value.clone().filter(|v| !v.trim().is_empty());
    let app_options = AppOptions {
        background: args.background,
        calibrate: args.calibrate,
        log_level_locked: text(&args.log_level).is_some(),
    };
    if let Some(dir) = args
        .config_dir
        .as_deref()
        .filter(|d| !d.as_os_str().is_empty())
    {
        paths::set_base_override(Some(dir));
    }
    let level = text(&args.log_level).unwrap_or_else(|| "INFO".to_string());
    let console = std::io::stderr().is_terminal();
    // The log file belongs to the running instance; a process that only hands a request over to
    // it must not write (or rotate) it, so file logging starts after the single-instance check.
    (options.setup_logging)(&level, console, false);

    let services = options.platform.take().unwrap_or_else(get_platform);
    // Before any window or thread exists (DPI awareness).
    services.prepare_process();

    // Single instance. Taking the lock is atomic, so of two launches racing each other exactly
    // one becomes the instance; the other hands its request over (waiting for the winner to
    // start listening) and leaves.
    let request = handover_request(&app_options);
    let lock = InstanceLock::new(None);
    let _release = ReleaseOnPanic(&lock);
    let mut owner = lock.acquire();
    // The owner asks too: an instance of a version without the lock answers. A launch that lost
    // the lock waits for the winner to start listening.
    let wait = if owner { 0 } else { options.startup_wait_ms };
    let mut reply = (options.send_command)(request, ipc::DEFAULT_TIMEOUT_MS, wait);
    if !owner && reply.as_deref().is_none_or(is_busy) {
        (owner, reply) = wait_for_handover(&lock, request, &mut options);
    }
    if let Some(reply) = reply.as_deref().filter(|r| !is_busy(r)) {
        lock.release();
        if reply.starts_with("error") {
            log::warn!("The running instance refused {request:?}: {reply}");
            return AppContext::exit(EXIT_ERROR, options.show_error);
        }
        log::info!("{APP_NAME} is already running; asked it to {request}");
        warn_startup_options_ignored(args);
        return AppContext::exit(EXIT_OK, options.show_error);
    }
    if !owner || reply.is_some() {
        // Nobody answered in time, or only "starting or shutting down" (from an older version
        // without the lock, when this process holds it).
        lock.release();
        log::error!(
            "Another instance (lock {}) does not answer; exiting",
            lock.path().display()
        );
        (options.show_error)(&format!(
            "{APP_NAME} is already running but does not respond."
        ));
        return AppContext::exit(EXIT_ERROR, options.show_error);
    }

    (options.setup_logging)(&level, console, true);
    let mut settings = Settings::load(&paths::settings_file());
    if !app_options.log_level_locked {
        (options.set_level)(&settings.general.log_level);
    }
    let overrides = apply_overrides(&mut settings, args);
    log::info!(
        "{APP_NAME} {VERSION} starting ({}, {})",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    if let Some(trace) = args.trace.clone().filter(|t| !t.as_os_str().is_empty()) {
        options.controller.trace_path = Some(trace);
    }

    let waker: Arc<Mutex<Option<egui::Context>>> = Arc::default();
    let (server_tx, server_rx) = crossbeam_channel::unbounded();
    let (app_tx, app_rx) = crossbeam_channel::unbounded();
    let wake = waker.clone();
    // Commands arrive on the server's threads; wake the event loop for each one.
    let forwarder = std::thread::Builder::new()
        .name("ipc-wake".into())
        .spawn(move || {
            for event in server_rx {
                if app_tx.send(event).is_err() {
                    break;
                }
                if let Some(ctx) = wake.lock().as_ref() {
                    ctx.request_repaint();
                }
            }
        });
    if let Err(err) = forwarder {
        log::warn!("Cannot start the command forwarder: {err}");
    }
    let show_error = options.show_error;
    let mut app = EyeTrackerApp::new(settings, services, app_options, overrides, app_rx, options);
    // From here on the server owns the lock and releases it in close().
    let mut server = InstanceServer::new(server_tx).with_lock(lock.clone());
    if !server.listen() {
        if server.another_instance_running() {
            // Only possible where files cannot be locked: another instance claimed the socket
            // after our check, or it hangs.
            log::error!("Another instance is running but does not answer; exiting");
            show_error(&format!(
                "{APP_NAME} is already running but does not respond."
            ));
            server.close();
            app.shutdown();
            return AppContext::exit(EXIT_ERROR, show_error);
        }
        log::warn!(
            "Command socket unavailable; '{}' will not work",
            cli_command_text(&["ctl"])
        );
    }
    AppContext {
        app: Some(app),
        server: Some(server),
        exit_code: None,
        waker,
        show_error,
    }
}

// ========================================================================= the app
/// The calibration window and why it was opened.
struct OpenCalibration {
    window: CalibrationWindow,
    reason: String,
}

/// Owns the controller, the tray icon and the windows of the running app.
pub struct EyeTrackerApp {
    settings: Settings,
    services: Arc<dyn PlatformServices>,
    options: AppOptions,
    /// `(key, (saved value, session value))` for settings overridden on the command line. The
    /// session value stays out of the settings file when other settings are saved.
    overrides: Vec<(&'static str, (String, String))>,
    clock: Clock,
    autostart: AutostartFactory,
    set_level: fn(&str),
    quiet_s: f64,
    ipc_events: Receiver<ipc::Event>,
    // Consumed by start().
    controller_options: Option<ControllerOptions>,
    tray_backend: Option<Box<dyn TrayBackend>>,
    update_options: Option<UpdateOptions>,

    controller: Option<Controller>,
    started: bool,
    closed: bool,
    quit_requested: bool,
    interrupted: bool,
    next_tick: f64,
    /// The egui context, once the event loop runs (focusing windows).
    egui: Option<egui::Context>,

    tray: Option<Tray>,
    overlay: Option<GazeOverlay>,
    countdown: Option<CountdownToast>,
    curtain: Option<PrivacyCurtain>,
    settings_dialog: Option<SettingsDialog>,
    calibration: Option<OpenCalibration>,
    preview: Option<PreviewWindow>,
    about: Option<AboutDialog>,
    wizard: Option<FirstRunWizard>,
    updates: Option<UpdateService>,
    update_dialog: Option<UpdateDialog>,
    /// The newer version the update service found (the tray names it).
    update_version: Option<String>,
    /// The diagnostics report the settings window asked for, being collected.
    diagnostics: Option<Receiver<String>>,

    /// The latest gaze estimate and shoulder-guard flag the controller reported.
    gaze: Option<GazePoint>,
    guard_active: bool,
    /// The system's screens and when they were read.
    screens: Vec<Monitor>,
    screens_at: f64,
    /// Windows to bring to the front with the next frame.
    raise: Vec<ViewportId>,
    timers: Vec<(f64, Timer)>,

    /// Reasons announced with a "calibration needed" notification since the calibration was
    /// last usable (cleared when it is usable again).
    announced: HashSet<String>,
    /// The clickable notification on screen: (kind, when shown).
    prompt: Option<(Prompt, f64)>,
    /// The permission a [`Prompt::Permission`] notification is about.
    prompt_permission: Option<String>,
    /// When the controller's latest notification was shown (see `on_permission_needed`);
    /// `None` once used or when it was not shown.
    last_notice_at: Option<f64>,
    /// The controller's latest notification, shown or not: (title, message, when).
    last_notice: Option<(String, String, f64)>,
    /// A reason that could not be announced while notifications were muted.
    deferred_reason: Option<String>,
    /// A permission notice swallowed by the startup mute: (name, title, message).
    deferred_permission: Option<(String, String, String)>,
}

impl EyeTrackerApp {
    fn new(
        settings: Settings,
        services: Arc<dyn PlatformServices>,
        options: AppOptions,
        overrides: Vec<(&'static str, (String, String))>,
        ipc_events: Receiver<ipc::Event>,
        build: BuildOptions,
    ) -> Self {
        let clock = build
            .clock
            .unwrap_or_else(|| Arc::new(crate::panes::worker::monotonic));
        let mut controller_options = build.controller;
        if controller_options.clock.is_none() {
            controller_options.clock = Some(clock.clone());
        }
        Self {
            settings,
            services,
            options,
            overrides,
            clock,
            autostart: build
                .autostart
                .unwrap_or_else(|| Rc::new(|| Box::new(SystemAutostart))),
            set_level: build.set_level,
            quiet_s: build.startup_quiet_s,
            ipc_events,
            controller_options: Some(controller_options),
            tray_backend: build.tray_backend,
            update_options: build.updates,
            controller: None,
            started: false,
            closed: false,
            quit_requested: false,
            interrupted: false,
            next_tick: f64::NEG_INFINITY,
            egui: None,
            tray: None,
            overlay: None,
            countdown: None,
            curtain: None,
            settings_dialog: None,
            calibration: None,
            preview: None,
            about: None,
            wizard: None,
            updates: None,
            update_dialog: None,
            update_version: None,
            diagnostics: None,
            gaze: None,
            guard_active: false,
            screens: Vec::new(),
            screens_at: f64::NEG_INFINITY,
            raise: Vec::new(),
            timers: Vec::new(),
            announced: HashSet::new(),
            prompt: None,
            prompt_permission: None,
            last_notice_at: None,
            last_notice: None,
            deferred_reason: None,
            deferred_permission: None,
        }
    }

    // ---------------------------------------------------------------- accessors
    pub fn controller(&self) -> Option<&Controller> {
        self.controller.as_ref()
    }

    pub fn controller_mut(&mut self) -> Option<&mut Controller> {
        self.controller.as_mut()
    }

    pub fn tray(&self) -> Option<&Tray> {
        self.tray.as_ref()
    }

    pub fn tray_mut(&mut self) -> Option<&mut Tray> {
        self.tray.as_mut()
    }

    pub fn overlay(&self) -> Option<&GazeOverlay> {
        self.overlay.as_ref()
    }

    pub fn countdown(&self) -> Option<&CountdownToast> {
        self.countdown.as_ref()
    }

    pub fn curtain(&self) -> Option<&PrivacyCurtain> {
        self.curtain.as_ref()
    }

    pub fn curtain_mut(&mut self) -> Option<&mut PrivacyCurtain> {
        self.curtain.as_mut()
    }

    pub fn settings_dialog(&self) -> Option<&SettingsDialog> {
        self.settings_dialog.as_ref()
    }

    pub fn settings_dialog_mut(&mut self) -> Option<&mut SettingsDialog> {
        self.settings_dialog.as_mut()
    }

    pub fn calibration_window(&self) -> Option<&CalibrationWindow> {
        self.calibration.as_ref().map(|c| &c.window)
    }

    pub fn calibration_window_mut(&mut self) -> Option<&mut CalibrationWindow> {
        self.calibration.as_mut().map(|c| &mut c.window)
    }

    pub fn preview(&self) -> Option<&PreviewWindow> {
        self.preview.as_ref()
    }

    pub fn preview_mut(&mut self) -> Option<&mut PreviewWindow> {
        self.preview.as_mut()
    }

    pub fn about(&self) -> Option<&AboutDialog> {
        self.about.as_ref()
    }

    pub fn wizard(&self) -> Option<&FirstRunWizard> {
        self.wizard.as_ref()
    }

    pub fn wizard_mut(&mut self) -> Option<&mut FirstRunWizard> {
        self.wizard.as_mut()
    }

    pub fn closed(&self) -> bool {
        self.closed
    }

    /// The event loop should end (quit from the tray, IPC or Ctrl+C).
    pub fn quit_requested(&self) -> bool {
        self.quit_requested
    }

    /// Ctrl+C (or SIGTERM) asked the app to quit.
    pub fn interrupted(&self) -> bool {
        self.interrupted
    }

    fn now(&self) -> f64 {
        (self.clock)()
    }

    // ---------------------------------------------------------------- lifecycle
    /// Create the controller and the UI, start tracking and open startup windows.
    pub fn start(&mut self) {
        if self.started || self.closed {
            return;
        }
        self.started = true;
        let options = self.controller_options.take().unwrap_or_default();
        let mut controller = Controller::new(&self.settings, self.services.clone(), options);
        let backend = self.tray_backend.take().unwrap_or_else(native_backend);
        let clock = self.clock.clone();
        let mut tray = Tray::new(
            backend,
            Box::new(|state, dark| {
                let icon = icons::tray_icon(state, Some(dark), None);
                icon.best(TRAY_ICON_PX)
                    .map(|image| (image.width, image.height, image.data.clone()))
                    .unwrap_or_default()
            }),
            (self.autostart)(),
            Box::new(move || clock()),
            self.tray_inputs(&controller),
        );
        // These follow the controller by themselves: the toast the away warnings, the curtain
        // the guard flag, the overlay the gaze and `ui.show_gaze_overlay`.
        let settings = controller.settings();
        self.overlay = Some(GazeOverlay::from_settings(settings.ui.show_gaze_overlay));
        self.countdown = Some(CountdownToast::new(
            Some(self.services.clone()),
            CountdownSettings::from_settings(settings),
        ));
        self.curtain = Some(PrivacyCurtain::new());
        if self.options.background {
            tray.mute_notifications(self.quiet_s);
        }
        tray.show();
        self.tray = Some(tray);

        controller.start();
        self.controller = Some(controller);
        self.start_updates();
        // Windows open once the event loop runs, so the tray is already in place.
        let now = self.now();
        self.schedule(now, Timer::AfterStart);
        self.dispatch_controller_events();
    }

    /// Close every window, then stop tracking (releases the camera). Idempotent.
    pub fn shutdown(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        // Windows first, while the controller still runs: an open calibration reports its
        // cancellation, the preview turns the preview stream off.
        if let Some(mut calibration) = self.calibration.take() {
            calibration.window.cancel();
            let cancelled = calibration
                .window
                .take_actions()
                .into_iter()
                .any(|action| matches!(action, CalibrationAction::Cancelled));
            if cancelled && let Some(controller) = &mut self.controller {
                controller.finish_calibration(None);
            }
        }
        if let Some(updates) = &mut self.updates {
            updates.shutdown();
        }
        if let Some(mut wizard) = self.wizard.take() {
            wizard.reject();
            for action in wizard.take_actions() {
                if let WizardAction::SetPreview(enabled) = action {
                    self.set_preview(enabled, WIZARD_OWNER);
                }
            }
        }
        if let Some(mut dialog) = self.settings_dialog.take() {
            dialog.reject();
            for action in dialog.take_actions() {
                if let (SettingsAction::SuspendHotkeys(suspended), Some(controller)) =
                    (action, &mut self.controller)
                {
                    controller.suspend_hotkeys(suspended);
                }
            }
        }
        if let Some(mut preview) = self.preview.take() {
            preview.close();
            for PreviewAction::SetPreview(enabled) in preview.drain_actions() {
                self.set_preview(enabled, PREVIEW_OWNER);
            }
        }
        self.about = None;
        self.countdown = None;
        self.update_dialog = None;
        if let Some(overlay) = &mut self.overlay {
            overlay.close();
        }
        if let Some(curtain) = &mut self.curtain {
            curtain.close();
        }
        if let Some(controller) = &mut self.controller
            && catch_unwind(AssertUnwindSafe(|| controller.shutdown())).is_err()
        {
            log::error!("Controller shutdown failed");
        }
        if let Some(tray) = &mut self.tray {
            tray.dispose();
        }
    }

    /// Leave the event loop (shutdown follows).
    pub fn quit(&mut self) {
        log::info!("Quit requested");
        self.quit_requested = true;
    }

    // ----------------------------------------------------------------- the loop
    /// One round of the app's work: commands, the controller's tick and events, timers, tray
    /// clicks, the update service and the windows' state. Needs no display.
    pub fn step(&mut self) {
        if self.closed || !self.started {
            return;
        }
        if INTERRUPTED.swap(false, Ordering::SeqCst) {
            log::info!("Received an interrupt signal; quitting");
            self.interrupted = true;
            self.quit();
        }
        let now = self.now();
        self.handle_ipc();
        if let Some(controller) = &mut self.controller {
            if now >= self.next_tick {
                controller.tick(now);
                self.next_tick = now + controller.next_tick_in().as_secs_f64();
            } else {
                controller.process_queued();
            }
        }
        self.dispatch_controller_events();
        self.run_due_timers(now);
        self.poll_tray();
        self.poll_updates(now);
        self.poll_diagnostics();
        self.update_windows(now);
        self.dispatch_controller_events();
        self.sync_tray();
    }

    /// How long the event loop may sleep before the next [`step`](Self::step).
    pub fn next_wakeup(&self) -> Duration {
        let mut wait = Duration::from_millis(SIGNAL_POLL_MS);
        let now = self.now();
        if let Some(controller) = &self.controller {
            wait = wait.min(controller.next_tick_in());
        }
        for (due, _) in &self.timers {
            wait = wait.min(Duration::from_secs_f64((due - now).max(0.0)));
        }
        wait
    }

    /// Whether a window is (or is about to be) shown, so a ui pass must run.
    pub fn wants_ui(&self) -> bool {
        !self.raise.is_empty()
            || self
                .overlay
                .as_ref()
                .is_some_and(|o| o.visible_monitor().is_some())
            || self.countdown.as_ref().is_some_and(CountdownToast::visible)
            || self
                .curtain
                .as_ref()
                .is_some_and(PrivacyCurtain::is_showing)
            || self.preview.as_ref().is_some_and(PreviewWindow::is_shown)
            || self.calibration.is_some()
            || self.settings_dialog.is_some()
            || self.wizard.is_some()
            || self.about.as_ref().is_some_and(AboutDialog::is_open)
            || self
                .update_dialog
                .as_ref()
                .is_some_and(UpdateDialog::is_visible)
    }

    fn handle_ipc(&mut self) {
        while let Ok(ipc::Event::CommandReceived { command, reply }) = self.ipc_events.try_recv() {
            reply.send(self.handle_command(&command));
            self.dispatch_controller_events();
        }
    }

    /// IPC entry point: [`Controller::handle_command`] executes it (UI commands come back as
    /// `UiRequested`).
    pub fn handle_command(&mut self, command: &str) -> String {
        match &mut self.controller {
            Some(controller) if !self.closed => controller.handle_command(command),
            // A second launch that gets this waits for the instance (is_busy).
            _ => format!("error: {APP_NAME} {BUSY_MARKER}; try again"),
        }
    }

    fn dispatch_controller_events(&mut self) {
        loop {
            let events = match &mut self.controller {
                Some(controller) => controller.drain_events(),
                None => return,
            };
            if events.is_empty() {
                return;
            }
            for event in events {
                self.dispatch(event);
            }
        }
    }

    /// React to one controller event (what the Python app and its windows connected to the
    /// controller's signals).
    pub fn dispatch(&mut self, event: ControllerEvent) {
        let now = self.now();
        match event {
            ControllerEvent::StateChanged(state) => {
                if let Some(preview) = &mut self.preview {
                    preview.set_state(state);
                }
                if let Some(wizard) = &mut self.wizard {
                    wizard.on_state_changed(state);
                }
                self.on_state_changed(state);
            }
            ControllerEvent::GazeChanged(point) => {
                self.gaze = point;
                let monitors = self.monitors();
                if let Some(preview) = &mut self.preview {
                    preview.set_gaze(point, &monitors);
                }
            }
            ControllerEvent::Observation(obs) => {
                if let Some(calibration) = &mut self.calibration {
                    calibration.window.on_observation(now, &obs);
                }
                if let Some(wizard) = &mut self.wizard {
                    wizard.on_observation(&obs, now);
                }
                if let Some(preview) = &mut self.preview {
                    preview.set_observation(&obs);
                }
            }
            ControllerEvent::StatsChanged(stats) => {
                let stats = Value::Object(stats);
                if let Some(dialog) = &mut self.settings_dialog {
                    dialog.on_stats(&stats);
                }
                if let Some(preview) = &mut self.preview {
                    preview.set_stats(&preview_stats(&stats));
                }
            }
            ControllerEvent::Notify { title, message } => {
                self.controller_notice(&title, &message);
            }
            ControllerEvent::Switched(_) => {}
            ControllerEvent::AwayWarning(remaining) => {
                self.refresh_screens(now, false);
                if let Some(countdown) = &mut self.countdown {
                    countdown.show_warning(remaining, now);
                }
            }
            ControllerEvent::AwayCancelled => {
                if let Some(countdown) = &mut self.countdown {
                    countdown.cancel();
                }
            }
            ControllerEvent::GuardChanged(active) => {
                self.guard_active = active;
                self.refresh_screens(now, active);
                let monitors = self.monitors();
                let cursor = self.cursor();
                if let Some(curtain) = &mut self.curtain {
                    curtain.set_environment(&monitors, &self.screens, cursor);
                    curtain.on_guard_changed(active, now);
                }
            }
            ControllerEvent::PreviewFrame(frame) => {
                if let Some(preview) = &mut self.preview {
                    preview.set_frame(&frame);
                }
                if let Some(wizard) = &mut self.wizard {
                    wizard.on_preview_frame(&frame);
                }
            }
            ControllerEvent::CalibrationRequired(reason) => self.on_calibration_required(&reason),
            ControllerEvent::SettingsChanged(settings) => self.on_settings_changed(*settings),
            ControllerEvent::UiRequested(command) => self.on_ui_requested(&command),
            ControllerEvent::PermissionNeeded(name) => self.on_permission_needed(&name),
        }
    }

    /// A notification from the controller (or a window speaking for it): the tray shows it.
    fn controller_notice(&mut self, title: &str, message: &str) {
        if let Some(tray) = &mut self.tray {
            tray.notify_from_controller(title, message);
        }
        self.on_controller_notify(title, message);
    }

    fn schedule(&mut self, due: f64, what: Timer) {
        self.timers.push((due, what));
    }

    fn run_due_timers(&mut self, now: f64) {
        while let Some(at) = self
            .timers
            .iter()
            .enumerate()
            .filter(|(_, (due, _))| *due <= now)
            .min_by(|a, b| a.1.0.total_cmp(&b.1.0))
            .map(|(at, _)| at)
        {
            let (_, what) = self.timers.remove(at);
            match what {
                Timer::AfterStart => self.after_start(),
                Timer::PromptSetup => self.prompt_setup(),
                Timer::AnnounceDeferred => self.announce_deferred(),
                Timer::AnnounceDeferredPermission => self.announce_deferred_permission(),
                Timer::RemindPrivacyMode => self.remind_privacy_mode(),
                Timer::Quit => self.quit(),
            }
            self.dispatch_controller_events();
        }
    }

    fn poll_tray(&mut self) {
        let actions = match &mut self.tray {
            Some(tray) => tray.poll(),
            None => return,
        };
        for action in actions {
            self.on_tray_action(action);
            self.dispatch_controller_events();
        }
    }

    /// Run what a tray click asked for.
    pub fn on_tray_action(&mut self, action: TrayAction) {
        match action {
            TrayAction::TogglePause => {
                if let Some(controller) = &mut self.controller {
                    controller.toggle_pause();
                }
            }
            TrayAction::SetPrivacy(enabled) => {
                if let Some(controller) = &mut self.controller {
                    controller.set_privacy(enabled);
                }
            }
            TrayAction::OpenCalibration => self.open_calibration("tray"),
            TrayAction::SetGazeDot(enabled) => {
                if let Some(controller) = &mut self.controller {
                    let mut settings = controller.settings().clone();
                    settings.ui.show_gaze_overlay = enabled;
                    controller.apply_settings(&settings);
                }
            }
            TrayAction::SetSplitPanes(enabled) => {
                if let Some(controller) = &mut self.controller {
                    let mut settings = controller.settings().clone();
                    settings.panes.enabled = enabled;
                    controller.apply_settings(&settings);
                }
            }
            TrayAction::OpenPreview => self.open_preview(),
            TrayAction::OpenSettings => self.open_settings(),
            TrayAction::OpenUpdate => self.open_update(),
            TrayAction::OpenAbout => self.open_about(),
            TrayAction::Quit => self.quit(),
        }
    }

    fn sync_tray(&mut self) {
        let inputs = match &self.controller {
            Some(controller) => self.tray_inputs(controller),
            None => return,
        };
        if let Some(tray) = &mut self.tray {
            tray.sync(inputs);
        }
    }

    fn tray_inputs(&self, controller: &Controller) -> TrayInputs {
        let settings = controller.settings().clone();
        TrayInputs {
            paused: controller.paused(),
            privacy: controller.privacy(),
            calibration_reason: controller.calibration_reason().to_string(),
            yield_reason: controller.yield_reason().to_string(),
            registered_hotkeys: controller.hotkey_manager().map(|m| m.registered()),
            stats: TrayStats::from_json(&Value::Object(controller.stats_snapshot())),
            update_version: self.update_version.clone(),
            ..TrayInputs::new(controller.state(), settings)
        }
    }

    fn monitors(&mut self) -> Vec<Monitor> {
        self.controller
            .as_mut()
            .map(Controller::monitors)
            .unwrap_or_default()
    }

    fn cursor(&self) -> Option<(f64, f64)> {
        self.services
            .cursor_position()
            .map(|(x, y)| (f64::from(x), f64::from(y)))
    }

    /// Re-read the system's screens (at most once a second unless `force`).
    fn refresh_screens(&mut self, now: f64, force: bool) {
        if force || now - self.screens_at >= SCREENS_POLL_S {
            self.screens_at = now;
            self.screens = crate::platform::monitors();
        }
    }

    // ------------------------------------------------------------------ windows
    /// Answer a second launch or `ctl show`: bring up whatever needs attention.
    ///
    /// An open calibration, wizard or settings window is raised. If the first-run setup was
    /// never finished (e.g. the app was only ever started at login), the setup assistant opens.
    /// Otherwise the tray menu pops up next to the tray icon; without a system tray (e.g. GNOME
    /// without an AppIndicator extension) the settings window opens instead.
    pub fn present(&mut self) {
        if self.closed || self.raise_calibration() {
            return;
        }
        if self.wizard.is_some() {
            self.raise.push(FirstRunWizard::viewport_id());
            return;
        }
        if self.settings_dialog.is_some() {
            self.raise.push(SettingsDialog::viewport_id());
            return;
        }
        if self.setup_pending() {
            self.open_wizard();
            return;
        }
        if !self.popup_tray_menu() {
            self.open_settings();
        }
    }

    /// Show the settings window (a fresh one unless it is already open).
    pub fn open_settings(&mut self) {
        if self.closed || self.controller.is_none() || self.raise_calibration() {
            return;
        }
        if self.wizard.is_some() {
            self.raise.push(FirstRunWizard::viewport_id());
            return;
        }
        if self.settings_dialog.is_none()
            && let Some(controller) = &self.controller
        {
            self.settings_dialog = Some(SettingsDialog::new(
                controller.settings(),
                controller.platform().capabilities(),
                controller.hotkey_manager(),
                (self.autostart)(),
            ));
        }
        self.raise.push(SettingsDialog::viewport_id());
    }

    /// Open the calibration on every monitor, or raise it if it is already open.
    pub fn open_calibration(&mut self, reason: &str) {
        if self.closed || self.controller.is_none() {
            return;
        }
        let inputs = self.calibration_inputs();
        if let Some(open) = &mut self.calibration
            && open.window.is_active()
        {
            open.window.start(&inputs); // raises the existing windows
            return;
        }
        log::info!("Opening the calibration ({reason})");
        self.prompt = None;
        let mut window = CalibrationWindow::new(CalibrationOptions::default());
        // Kept before its actions run: without a monitor it reports `NotStarted` at once.
        window.start(&inputs);
        self.calibration = Some(OpenCalibration {
            window,
            reason: reason.to_string(),
        });
        self.process_calibration_actions(Vec::new());
    }

    /// Show the camera preview (one instance; closing only hides it).
    pub fn open_preview(&mut self) {
        if self.closed || self.controller.is_none() || self.raise_calibration() {
            return;
        }
        let monitors = self.monitors();
        let Some(controller) = &self.controller else {
            return;
        };
        let (name, version) = controller.backend_info();
        let info = (!name.is_empty()).then_some((name.as_str(), version.as_str()));
        let preview = self
            .preview
            .get_or_insert_with(|| PreviewWindow::new(controller.state(), info));
        if !preview.is_shown() {
            preview.open(&monitors, info);
        }
        self.raise.push(PreviewWindow::viewport_id());
        self.process_preview_actions();
    }

    /// Show the About dialog.
    pub fn open_about(&mut self) {
        if self.closed || self.raise_calibration() {
            return;
        }
        let about = self.about.get_or_insert_with(AboutDialog::new);
        if !about.is_open() {
            *about = AboutDialog::new();
            about.open();
        }
        self.raise.push(AboutDialog::viewport_id());
    }

    /// Show the update window (it looks for an update unless one is known).
    pub fn open_update(&mut self) {
        if self.closed || self.updates.is_none() || self.raise_calibration() {
            return;
        }
        let (Some(updates), dialog) = (&self.updates, &mut self.update_dialog) else {
            return;
        };
        let actions = dialog
            .get_or_insert_with(UpdateDialog::new)
            .present(updates.state());
        self.process_update_actions(actions);
    }

    /// Show the first-run wizard.
    pub fn open_wizard(&mut self) {
        if self.closed || self.controller.is_none() || self.raise_calibration() {
            return;
        }
        if self.wizard.is_none() {
            let monitors = self.monitors().len();
            let Some(controller) = &self.controller else {
                return;
            };
            self.wizard = Some(FirstRunWizard::new(
                controller.settings(),
                self.services.clone(),
                controller.state(),
                Some(monitors),
                None,
                (self.autostart)(),
            ));
        }
        self.raise.push(FirstRunWizard::viewport_id());
    }

    // -------------------------------------------------------------------- slots
    fn after_start(&mut self) {
        if self.closed || self.controller.is_none() {
            return;
        }
        self.refresh_autostart();
        self.remind_privacy_mode();
        let mut setup_offered = false;
        if self.setup_pending() {
            if self.options.background {
                // A login start stays quiet; offer the setup once that is over.
                self.after_quiet_period(Timer::PromptSetup);
                setup_offered = true; // the setup offers the calibration itself
            } else {
                self.open_wizard();
                return; // the wizard offers the calibration itself
            }
        }
        if self.options.calibrate {
            self.open_calibration("cli");
        } else if !setup_offered {
            // A login start too: the notice waits for the end of its quiet period (see
            // suggest_calibration). Skipping it there meant a setup that was never calibrated
            // and starts at login was never told why nothing switches.
            let reason = self.calibration_reason();
            self.suggest_calibration(&reason);
        }
    }

    /// `calibration_required` from the controller.
    pub fn on_calibration_required(&mut self, reason: &str) {
        if EXPLICIT_CALIBRATION_REASONS.contains(&reason) {
            self.open_calibration(reason);
        } else {
            self.suggest_calibration(reason);
        }
    }

    fn on_calibration_finished(&mut self, reason: &str, saved: bool) {
        self.calibration = None;
        log::info!(
            "Calibration {}",
            if saved {
                "saved"
            } else {
                "closed without saving"
            }
        );
        if !saved && OFFERED_CALIBRATIONS.contains(&reason) && !self.closed {
            // Offered rather than asked for (setup assistant, --calibrate): whoever put it off
            // learns what that means, and that a click brings it back.
            let reason = self.calibration_reason();
            self.suggest_calibration(&reason);
        }
    }

    fn on_wizard_finished(&mut self, wants_calibration: bool) {
        if self.closed || self.controller.is_none() || wants_calibration {
            return;
        }
        // Finished (or cancelled) without calibrating: say why switching is off.
        let reason = self.calibration_reason();
        self.suggest_calibration(&reason);
    }

    fn on_settings_changed(&mut self, settings: Settings) {
        if let Some(dialog) = &mut self.settings_dialog {
            dialog.on_settings_changed(&settings);
        }
        if let Some(wizard) = &mut self.wizard {
            wizard.on_settings_changed(&settings);
        }
        if !self.options.log_level_locked {
            (self.set_level)(&settings.general.log_level);
        }
        let now = self.now();
        if let Some(updates) = &mut self.updates {
            updates.set_auto(settings.updates.check, now);
        }
        self.save_without_overrides(&settings);
        self.settings = settings;
        // The tray applies `general.notifications` to what is said next.
        self.sync_tray();
    }

    /// `ui_requested` from the controller (IPC commands only the UI can carry out).
    pub fn on_ui_requested(&mut self, command: &str) {
        match command {
            "show" => self.present(),
            "settings" => self.open_settings(),
            "quit" => {
                let due = self.now() + QUIT_DELAY_MS as f64 / 1000.0;
                self.schedule(due, Timer::Quit);
            }
            _ => log::debug!("Ignoring unknown UI request {command:?}"),
        }
    }

    fn on_state_changed(&mut self, state: TrackingState) {
        let Some(controller) = &self.controller else {
            return;
        };
        if self.announced.is_empty() {
            return;
        }
        let absent = matches!(
            state,
            TrackingState::Away | TrackingState::Locked | TrackingState::Privacy
        );
        if controller.is_calibrated() || absent {
            // Usable again (recalibrated, or back at a known desk): the next time it stops being
            // usable deserves a notification, even for a reason announced before (docking at a
            // new desk is "the monitor layout changed" every time). And after being away, locked
            // or private, the controller reminds the user of a calibration that is still
            // unusable, with the same reason as before: that must be shown too.
            self.announced.clear();
        }
    }

    fn on_controller_notify(&mut self, title: &str, message: &str) {
        // The tray shows one message at a time; ours has been replaced.
        self.prompt = None;
        self.prompt_permission = None;
        let now = self.now();
        let shown = self.tray.as_ref().is_some_and(|t| t.muted_for() <= 0.0);
        self.last_notice_at = shown.then_some(now);
        self.last_notice = Some((title.to_string(), message.to_string(), now));
    }

    /// Make the notification that explains a missing permission clickable.
    ///
    /// The controller emits `PermissionNeeded` right after that `Notify` (or without one when
    /// notifications are off, and then there is nothing to click). Clicking opens the OS privacy
    /// settings for `name`.
    ///
    /// The controller says it only once, so a notice swallowed by the quiet period of a login
    /// start (typically the Accessibility check at startup) is shown once that period is over,
    /// if the permission is still missing.
    pub fn on_permission_needed(&mut self, name: &str) {
        let shown_at = self.last_notice_at.take();
        let notice = self.last_notice.take();
        let now = self.now();
        if shown_at.is_some_and(|at| now - at <= PERMISSION_NOTICE_S) {
            self.prompt = Some((Prompt::Permission, now));
            self.prompt_permission = Some(name.to_string());
            return;
        }
        let muted = self.tray.as_ref().is_some_and(|t| t.muted_for() > 0.0);
        let Some((title, message, at)) = notice else {
            return; // nothing was said (notifications off), so nothing to repeat
        };
        if now - at > PERMISSION_NOTICE_S || !muted {
            return;
        }
        if self.deferred_permission.is_none() {
            self.after_quiet_period(Timer::AnnounceDeferredPermission);
        }
        // One slot: a click can only open the settings of the notice shown last.
        self.deferred_permission = Some((name.to_string(), title, message));
    }

    fn announce_deferred_permission(&mut self) {
        let Some((name, title, message)) = self.deferred_permission.take() else {
            return;
        };
        if self.closed || self.tray.is_none() || !self.permission_still_needed(&name) {
            return;
        }
        // The controller applied the notifications setting when it sent it.
        if let Some(tray) = &mut self.tray
            && tray.notify(&title, &message, true, false)
        {
            self.prompt = Some((Prompt::Permission, self.now()));
            self.prompt_permission = Some(name);
        }
    }

    /// Whether permission `name` still blocks something (see `on_permission_needed`).
    fn permission_still_needed(&self, name: &str) -> bool {
        let Some(controller) = &self.controller else {
            return false;
        };
        match name {
            "camera" => controller.state() == TrackingState::CameraError,
            "accessibility" => matches!(self.services.accessibility_status(), "missing" | "stale"),
            _ => true,
        }
    }

    /// The tray notification was clicked.
    pub fn on_message_clicked(&mut self) {
        let prompt = self.prompt.take();
        let permission = self.prompt_permission.take();
        let Some((kind, shown_at)) = prompt else {
            return;
        };
        if self.controller.is_none() || self.now() - shown_at > PROMPT_CLICK_WINDOW_S {
            return;
        }
        let calibrated = self
            .controller
            .as_ref()
            .is_some_and(Controller::is_calibrated);
        match kind {
            Prompt::Setup if self.setup_pending() => self.open_wizard(),
            Prompt::Calibrate if !calibrated => self.open_calibration("notification"),
            Prompt::Permission => {
                if let Some(name) = permission.filter(|n| !n.is_empty()) {
                    self.open_permission_settings(&name);
                }
            }
            Prompt::Update => self.open_update(),
            _ => {}
        }
    }

    /// Create the update service (it looks for updates only if the user turned that on, or
    /// asks).
    fn start_updates(&mut self) {
        let mut updates = UpdateService::new(self.update_options.take().unwrap_or_default());
        updates.start(self.settings.updates.check, self.now());
        self.updates = Some(updates);
    }

    fn poll_updates(&mut self, now: f64) {
        let Some(updates) = &mut self.updates else {
            return;
        };
        updates.tick(now);
        for event in updates.drain_events() {
            match event {
                UpdateEvent::StateChanged(state) => {
                    let version = state.release.as_ref().map(|r| r.version.clone());
                    if state.phase == Phase::UpToDate {
                        self.set_update_available(None);
                    } else if version.is_some()
                        && matches!(
                            state.phase,
                            Phase::Available
                                | Phase::Downloading
                                | Phase::Installing
                                | Phase::Failed
                        )
                    {
                        self.set_update_available(version);
                    }
                }
                UpdateEvent::Announce(release) => self.on_update_announced(&release.version),
            }
        }
    }

    fn set_update_available(&mut self, version: Option<String>) {
        if let Some(tray) = &mut self.tray {
            tray.set_update_available(version.as_deref());
        }
        self.update_version = version;
    }

    /// A newer version was found by the daily check: say so once.
    fn on_update_announced(&mut self, version: &str) {
        let Some(tray) = &mut self.tray else {
            return;
        };
        if version.is_empty() {
            return;
        }
        let shown = tray.notify(
            "Update available",
            &format!("{APP_NAME} {version} is available. Click here to update."),
            false,
            false,
        );
        if shown {
            self.prompt = Some((Prompt::Update, self.now()));
        }
    }

    // ------------------------------------------------------------- the windows
    /// Inputs of the calibration window from the controller.
    fn calibration_inputs(&mut self) -> CalibrationInputs {
        let now = self.now();
        let monitors = self.monitors();
        let Some(controller) = &self.controller else {
            return CalibrationInputs::default();
        };
        CalibrationInputs {
            now,
            monitors,
            backend_info: controller.backend_info(),
            gaze_feature_indices: controller.gaze_feature_indices(),
            camera_device: controller.settings().camera.device.clone(),
            observations: Vec::new(),
        }
    }

    /// Apply what the controller says to the windows and run their timers (no drawing).
    fn update_windows(&mut self, now: f64) {
        let Some(controller) = &mut self.controller else {
            return;
        };
        let monitors = controller.monitors();
        let state = controller.state();
        let settings = controller.settings().clone();
        let follows_screens = self.guard_active
            || self
                .curtain
                .as_ref()
                .is_some_and(PrivacyCurtain::is_showing)
            || self.countdown.as_ref().is_some_and(CountdownToast::visible);
        if follows_screens {
            self.refresh_screens(now, false);
        }
        let cursor = if follows_screens { self.cursor() } else { None };
        if let Some(overlay) = &mut self.overlay {
            overlay.update(&OverlayInputs {
                monitors: &monitors,
                show_gaze_overlay: settings.ui.show_gaze_overlay,
                state,
                gaze: self.gaze,
                now,
            });
        }
        let countdown_settings = CountdownSettings::from_settings(&settings);
        if let Some(countdown) = &mut self.countdown {
            countdown.update(&CountdownInputs {
                env: ToastEnv {
                    monitors: &monitors,
                    screens: &self.screens,
                    cursor,
                },
                state,
                settings: &countdown_settings,
                warning: None,
                cancelled: false,
                now,
            });
        }
        let curtain_actions = match &mut self.curtain {
            Some(curtain) => {
                curtain.update(&CurtainInputs {
                    monitors: &monitors,
                    screens: &self.screens,
                    cursor,
                    guard_active: self.guard_active,
                    state,
                    shoulder_guard: settings.privacy.shoulder_guard,
                    guard_action: &settings.privacy.guard_action,
                    now,
                });
                curtain.drain_actions()
            }
            None => Vec::new(),
        };
        self.process_curtain_actions(curtain_actions);
        if self.calibration.is_some() {
            let inputs = self.calibration_inputs();
            let actions = match &mut self.calibration {
                Some(open) => {
                    open.window.update(&inputs);
                    open.window.take_actions()
                }
                None => Vec::new(),
            };
            self.process_calibration_actions(actions);
        }
        self.process_preview_actions();
        let actions = self
            .settings_dialog
            .as_mut()
            .map(SettingsDialog::take_actions)
            .unwrap_or_default();
        self.process_settings_actions(actions);
        let actions = self
            .wizard
            .as_mut()
            .map(FirstRunWizard::take_actions)
            .unwrap_or_default();
        self.process_wizard_actions(actions);
    }

    /// Draw the windows that are open and run what they ask for.
    pub fn draw(&mut self, ctx: &egui::Context) {
        if self.closed || self.controller.is_none() {
            return;
        }
        if self.egui.is_none() {
            self.egui = Some(ctx.clone());
        }
        let now = self.now();
        for id in std::mem::take(&mut self.raise) {
            ctx.send_viewport_cmd_to(id, ViewportCommand::Minimized(false));
            ctx.send_viewport_cmd_to(id, ViewportCommand::Focus);
        }
        let cursor = self.cursor();
        let Some(controller) = &mut self.controller else {
            return;
        };
        let monitors = controller.monitors();
        let state = controller.state();
        let settings = controller.settings().clone();
        if let Some(overlay) = &mut self.overlay {
            overlay.show(
                ctx,
                &OverlayInputs {
                    monitors: &monitors,
                    show_gaze_overlay: settings.ui.show_gaze_overlay,
                    state,
                    gaze: self.gaze,
                    now,
                },
            );
        }
        let curtain_actions = match &mut self.curtain {
            Some(curtain) => curtain.show(
                ctx,
                &CurtainInputs {
                    monitors: &monitors,
                    screens: &self.screens,
                    cursor,
                    guard_active: self.guard_active,
                    state,
                    shoulder_guard: settings.privacy.shoulder_guard,
                    guard_action: &settings.privacy.guard_action,
                    now,
                },
            ),
            None => Vec::new(),
        };
        let countdown_settings = CountdownSettings::from_settings(&settings);
        if let Some(countdown) = &mut self.countdown {
            countdown.show(
                ctx,
                &CountdownInputs {
                    env: ToastEnv {
                        monitors: &monitors,
                        screens: &self.screens,
                        cursor,
                    },
                    state,
                    settings: &countdown_settings,
                    warning: None,
                    cancelled: false,
                    now,
                },
            );
        }
        let preview_actions = match &mut self.preview {
            Some(preview) => preview.show(
                ctx,
                &PreviewInputs {
                    monitors: &monitors,
                    ..PreviewInputs::default()
                },
            ),
            None => Vec::new(),
        };
        let (name, version) = controller.backend_info();
        let settings_actions = match &mut self.settings_dialog {
            Some(dialog) => dialog.show(
                ctx,
                &SettingsInputs {
                    calibration: controller.calibration(),
                    is_calibrated: controller.is_calibrated(),
                    calibration_reason: controller.calibration_reason(),
                    backend_info: (!name.is_empty()).then_some((name.as_str(), version.as_str())),
                },
            ),
            None => Vec::new(),
        };
        let wizard_actions = match &mut self.wizard {
            Some(wizard) => wizard.show(ctx, &WizardInputs { now }),
            None => Vec::new(),
        };
        if let Some(about) = &mut self.about {
            about.show(ctx);
        }
        let update_actions = match (&mut self.update_dialog, &self.updates) {
            (Some(dialog), Some(updates)) => dialog.show(
                ctx,
                &UpdateInputs {
                    state: updates.state(),
                    can_install: updates.can_install(),
                },
            ),
            _ => Vec::new(),
        };
        let calibration_actions = if self.calibration.is_some() {
            let inputs = self.calibration_inputs();
            match &mut self.calibration {
                Some(open) => open.window.show(ctx, &inputs),
                None => Vec::new(),
            }
        } else {
            Vec::new()
        };
        self.process_curtain_actions(curtain_actions);
        for PreviewAction::SetPreview(enabled) in preview_actions {
            self.set_preview(enabled, PREVIEW_OWNER);
        }
        self.process_settings_actions(settings_actions);
        self.process_wizard_actions(wizard_actions);
        self.process_update_actions(update_actions);
        self.process_calibration_actions(calibration_actions);
        self.dispatch_controller_events();
    }

    fn set_preview(&mut self, enabled: bool, owner: &str) {
        if let Some(controller) = &mut self.controller {
            controller.set_preview(enabled, Some(owner));
        }
    }

    fn process_preview_actions(&mut self) {
        let actions = self
            .preview
            .as_mut()
            .map(PreviewWindow::drain_actions)
            .unwrap_or_default();
        for PreviewAction::SetPreview(enabled) in actions {
            self.set_preview(enabled, PREVIEW_OWNER);
        }
    }

    fn process_curtain_actions(&mut self, actions: Vec<CurtainAction>) {
        for action in actions {
            match action {
                // Esc or the button on the curtain: switching resumes (the guard stays on).
                CurtainAction::Dismissed => {
                    if let Some(controller) = &mut self.controller {
                        controller.dismiss_curtain();
                    }
                }
                CurtainAction::ActivateWindow { viewport, pid } => {
                    self.activate_curtain(viewport, pid);
                }
            }
        }
    }

    /// Activate a curtain window the toolkit could not bring to the front.
    fn activate_curtain(&mut self, viewport: ViewportId, pid: u32) {
        if let Some(ctx) = &self.egui {
            ctx.send_viewport_cmd_to(viewport, ViewportCommand::Focus);
        }
        // eframe offers no native handle of an immediate viewport, so the platform is given the
        // foreground window if it is ours, else our window at the centre of the curtain's
        // monitor (it covers the whole monitor and stays on top).
        let centre = self.curtain.as_ref().and_then(|curtain| {
            curtain
                .windows()
                .iter()
                .enumerate()
                .find(|(index, _)| CurtainWindow::viewport_id(*index) == viewport)
                .map(|(_, window)| window.monitor.rect.center())
        });
        let ours = |window: &crate::types::WindowRef| window.pid == Some(pid);
        let window = self.services.foreground_window().filter(ours).or_else(|| {
            let (x, y) = centre?;
            self.services
                .window_at(x.round() as i32, y.round() as i32)
                .filter(ours)
        });
        match window {
            Some(window) => {
                if !self.services.activate_window(&window) {
                    log::debug!("The privacy curtain could not be activated");
                }
            }
            None => log::debug!("The privacy curtain's window was not found"),
        }
    }

    fn process_calibration_actions(&mut self, mut actions: Vec<CalibrationAction>) {
        if let Some(open) = &mut self.calibration {
            actions.extend(open.window.take_actions());
        }
        let reason = match &self.calibration {
            Some(open) => open.reason.clone(),
            None => return,
        };
        for action in actions {
            match action {
                CalibrationAction::Started => {
                    if let Some(controller) = &mut self.controller {
                        controller.begin_calibration();
                    }
                }
                CalibrationAction::Finished { data, .. } => {
                    if let Some(controller) = &mut self.controller {
                        controller.finish_calibration(Some(*data));
                    }
                    self.on_calibration_finished(&reason, true);
                }
                CalibrationAction::Cancelled => {
                    if let Some(controller) = &mut self.controller {
                        controller.finish_calibration(None);
                    }
                    self.on_calibration_finished(&reason, false);
                }
                CalibrationAction::NotStarted => self.on_calibration_finished(&reason, false),
                CalibrationAction::Notify { title, message } => {
                    self.controller_notice(&title, &message);
                }
                CalibrationAction::StateChanged(_) => {}
            }
        }
    }

    fn process_settings_actions(&mut self, actions: Vec<SettingsAction>) {
        for action in actions {
            match action {
                SettingsAction::Apply(settings) => {
                    if let Some(controller) = &mut self.controller {
                        controller.apply_settings(&settings);
                    }
                }
                SettingsAction::SuspendHotkeys(suspended) => {
                    if let Some(controller) = &mut self.controller {
                        controller.suspend_hotkeys(suspended);
                    }
                }
                SettingsAction::CollectDiagnostics => self.collect_diagnostics(),
                SettingsAction::CalibrationRequested => self.open_calibration("settings"),
                // Opened right away rather than "at next start": waiting meant storing
                // first_run_done=false, which also switches the walk-away lock to a
                // notification until the assistant is finished (see the controller).
                SettingsAction::SetupRequested => self.open_wizard(),
                SettingsAction::Closed { .. } => self.settings_dialog = None,
            }
            self.dispatch_controller_events();
        }
    }

    /// Collect the diagnostics report off the UI thread: it asks the running instance (this
    /// one) for its status, which only the UI thread answers.
    fn collect_diagnostics(&mut self) {
        let manager = self
            .controller
            .as_ref()
            .and_then(Controller::hotkey_manager);
        let (tx, rx) = crossbeam_channel::bounded(1);
        let spawned = std::thread::Builder::new()
            .name("diagnostics".into())
            .spawn(move || {
                let _ = tx.send(load_diagnostics_text(manager.as_deref()));
            });
        match spawned {
            Ok(_) => self.diagnostics = Some(rx),
            Err(err) => {
                if let Some(dialog) = &mut self.settings_dialog {
                    dialog.set_diagnostics(Err(err.to_string()));
                }
            }
        }
    }

    fn poll_diagnostics(&mut self) {
        let Some(rx) = &self.diagnostics else {
            return;
        };
        let report = match rx.try_recv() {
            Ok(text) => Ok(text),
            Err(TryRecvError::Empty) => return,
            Err(TryRecvError::Disconnected) => Err("the report could not be collected".into()),
        };
        self.diagnostics = None;
        if let Some(dialog) = &mut self.settings_dialog {
            dialog.set_diagnostics(report);
        }
    }

    fn process_wizard_actions(&mut self, actions: Vec<WizardAction>) {
        for action in actions {
            match action {
                WizardAction::SetPreview(enabled) => self.set_preview(enabled, WIZARD_OWNER),
                WizardAction::ApplySettings(settings) => {
                    if let Some(controller) = &mut self.controller {
                        controller.apply_settings(&settings);
                    }
                }
                WizardAction::Notify { title, message } => {
                    self.controller_notice(&title, &message);
                }
                WizardAction::CalibrationRequested => self.open_calibration("wizard"),
                WizardAction::Closed { .. } => {
                    let wants = self
                        .wizard
                        .take()
                        .is_some_and(|wizard| wizard.wants_calibration());
                    self.on_wizard_finished(wants);
                }
            }
            self.dispatch_controller_events();
        }
    }

    fn process_update_actions(&mut self, actions: Vec<UpdateAction>) {
        let Some(updates) = &mut self.updates else {
            return;
        };
        for action in actions {
            match action {
                UpdateAction::CheckNow => {
                    updates.check_now();
                }
                UpdateAction::Install => {
                    updates.install();
                }
                UpdateAction::Cancel => updates.cancel(),
            }
        }
    }

    // ------------------------------------------------------------------ helpers
    fn calibration_reason(&self) -> String {
        self.controller
            .as_ref()
            .map(|c| c.calibration_reason().to_string())
            .unwrap_or_default()
    }

    fn popup_tray_menu(&mut self) -> bool {
        match &mut self.tray {
            Some(tray) if tray.available() => {
                tray.popup_menu();
                true
            }
            _ => false,
        }
    }

    /// Tell the user that switching needs a calibration.
    ///
    /// Once per reason until the calibration is usable again. A notification that could not be
    /// shown does not count: one suppressed by the quiet period after a login start is shown
    /// when that period ends.
    fn suggest_calibration(&mut self, reason: &str) {
        if self.tray.is_none() {
            return;
        }
        let monitors = self.monitors().len();
        let Some(controller) = &self.controller else {
            return;
        };
        if controller.is_calibrated() {
            return;
        }
        if self
            .calibration
            .as_ref()
            .is_some_and(|c| c.window.is_active())
        {
            return;
        }
        if monitors < 2 || self.announced.contains(reason) {
            return; // with one monitor there is nothing to switch between
        }
        if !controller.settings().switching.enabled {
            return; // switching is off on purpose: a calibration changes nothing
        }
        if controller.privacy() {
            // The camera is off on purpose; the controller reminds the user once privacy mode
            // ends.
            return;
        }
        let detail = if reason.is_empty() {
            String::new()
        } else {
            format!(" ({reason})")
        };
        let Some(tray) = &mut self.tray else {
            return;
        };
        let now = (self.clock)();
        let shown = tray.notify(
            "Calibration needed",
            &format!(
                "Switching monitors is off until you calibrate{detail}. \
                 Click here or choose “Calibrate now…” in the tray menu."
            ),
            false,
            false,
        );
        if shown {
            self.announced.insert(reason.to_string());
            self.prompt = Some((Prompt::Calibrate, now));
        } else if tray.muted_for() > 0.0 {
            // Swallowed by the quiet period of a login start: say it afterwards.
            if self.deferred_reason.is_none() {
                self.after_quiet_period(Timer::AnnounceDeferred);
            }
            self.deferred_reason = Some(reason.to_string());
        }
    }

    fn announce_deferred(&mut self) {
        let Some(reason) = self.deferred_reason.take() else {
            return;
        };
        if self.closed || self.controller.is_none() {
            return;
        }
        // The reason may be stale by now; the current one is what matters.
        let current = self.calibration_reason();
        let reason = if current.is_empty() { reason } else { current };
        self.suggest_calibration(&reason);
    }

    /// Say that privacy mode is still on from the previous run.
    ///
    /// It is remembered across restarts so that the camera never comes back on by itself
    /// (`privacy.remember_privacy_mode`); this tells the user why nothing switches and walk-away
    /// detection is off. A notice swallowed by the quiet period of a login start is shown once
    /// that period is over.
    fn remind_privacy_mode(&mut self) {
        if self.closed {
            return;
        }
        let (Some(controller), Some(tray)) = (&self.controller, &mut self.tray) else {
            return;
        };
        if !controller.privacy_restored() {
            return;
        }
        let shown = tray.notify(
            "Privacy mode is still on",
            "The camera stays off, as you left it, so monitor switching and walk-away \
             detection are paused. Turn privacy mode off in the tray menu.",
            false,
            false,
        );
        if !shown && tray.muted_for() > 0.0 {
            self.after_quiet_period(Timer::RemindPrivacyMode);
        }
    }

    /// After a quiet login start: offer the first-run setup that never happened.
    fn prompt_setup(&mut self) {
        if self.closed || self.tray.is_none() || !self.setup_pending() {
            return;
        }
        if self.wizard.is_some()
            || self
                .calibration
                .as_ref()
                .is_some_and(|c| c.window.is_active())
        {
            return;
        }
        let Some(tray) = &mut self.tray else {
            return;
        };
        let shown = tray.notify(
            &format!("Finish setting up {APP_NAME}"),
            "Check the camera and choose what happens when you walk away. \
             Click here to open the setup assistant.",
            false,
            false,
        );
        if shown {
            self.prompt = Some((Prompt::Setup, self.now()));
        }
    }

    /// Run `what` once the startup notification mute is over.
    fn after_quiet_period(&mut self, what: Timer) {
        let delay = self.tray.as_ref().map_or(0.0, Tray::muted_for);
        let due = self.now() + delay + AFTER_QUIET_MS as f64 / 1000.0;
        self.schedule(due, what);
    }

    /// The first-run setup was never finished.
    ///
    /// Nothing in the UI sets `first_run_done` back to `false`: running the assistant again from
    /// the settings opens it at once, so the walk-away lock stays in force.
    fn setup_pending(&self) -> bool {
        self.controller
            .as_ref()
            .is_some_and(|c| !c.settings().general.first_run_done)
    }

    /// Raise an open calibration instead of opening another window. Returns whether one was
    /// open.
    ///
    /// Every other window would open underneath its topmost full-screen surfaces and take the
    /// keyboard from it (Space, R and Esc would stop working with no visible reason).
    fn raise_calibration(&mut self) -> bool {
        if !self
            .calibration
            .as_ref()
            .is_some_and(|c| c.window.is_active())
        {
            return false;
        }
        let inputs = self.calibration_inputs();
        if let Some(open) = &mut self.calibration {
            open.window.start(&inputs); // only raises and refocuses the existing surfaces
        }
        true
    }

    /// Open the OS privacy settings page for permission `name`.
    fn open_permission_settings(&self, name: &str) {
        if !self.services.open_permission_settings(name) {
            log::info!("No privacy settings page to open for {name} here");
        }
    }

    /// Point the login item at this copy of the app if it moved (e.g. an update to another
    /// folder, a replaced AppImage).
    fn refresh_autostart(&self) {
        let autostart = (self.autostart)();
        if autostart.is_supported() {
            autostart.refresh();
        }
    }

    /// The controller saved `settings`; keep command-line overrides out of the file.
    fn save_without_overrides(&self, settings: &Settings) {
        let restore: Vec<(&str, &str)> = self
            .overrides
            .iter()
            .filter(|(key, (saved, session))| setting(settings, key) == session && saved != session)
            .map(|(key, (saved, _))| (*key, saved.as_str()))
            .collect();
        if restore.is_empty() {
            return;
        }
        let mut on_disk = settings.clone();
        for (key, value) in restore {
            set_setting(&mut on_disk, key, value);
        }
        if let Err(err) = on_disk.save(&paths::settings_file()) {
            log::warn!("Could not save the settings: {err}");
        }
    }
}

impl Drop for EyeTrackerApp {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// The preview's statistics from the controller's stats mapping.
fn preview_stats(stats: &Value) -> PreviewStats {
    let number = |key: &str| stats.get(key).and_then(Value::as_f64);
    PreviewStats {
        fps: number("fps"),
        target_fps: number("target_fps"),
        inference_ms: number("inference_ms"),
        skip_ratio: number("skip_ratio"),
        cpu_percent: number("cpu_percent"),
        backend: stats
            .get("backend")
            .and_then(Value::as_str)
            .map(str::to_string),
    }
}

// ======================================================================= helpers
/// What a launch asks an instance that is already running to do.
///
/// A login start (`--background`) must not pop anything up there (the tray menu, the setup
/// assistant or the settings take the keyboard from whatever the user is typing in), so it only
/// asks for `status`, which just proves that the instance answers.
fn handover_request(options: &AppOptions) -> &'static str {
    if options.calibrate {
        "calibrate"
    } else if options.background {
        "status"
    } else {
        "show"
    }
}

/// Whether `reply` says the instance cannot take commands yet or any more.
fn is_busy(reply: &str) -> bool {
    reply.starts_with("error") && reply.contains(BUSY_MARKER)
}

/// Wait for an instance that holds `lock` but did not answer `request`.
///
/// Such an instance is usually shutting down: it stops the camera with the event loop already
/// stopped, while the lock (and the socket, which still accepts connections) are released only
/// afterwards. Giving up at once would leave nothing running once it has exited: the classic
/// "quit from the tray and start again right away". So until the hand-over wait has passed, this
/// takes the lock as soon as it is free, or hands `request` over if the instance answers after
/// all.
///
/// Returns `(owner, reply)`: `(true, None)` when this process now holds the lock (it becomes the
/// instance); otherwise the last reply (`None`, or a "starting or shutting down" error) or the
/// first real answer.
fn wait_for_handover(
    lock: &InstanceLock,
    request: &str,
    options: &mut BuildOptions,
) -> (bool, Option<String>) {
    log::info!("The running instance does not answer; waiting for it (it may be shutting down)");
    let deadline = Instant::now() + Duration::from_millis(options.handover_wait_ms);
    loop {
        if lock.acquire() {
            log::info!("The previous instance has exited; starting");
            return (true, None);
        }
        let reply = (options.send_command)(request, HANDOVER_REPLY_MS, 0);
        if reply.as_deref().is_some_and(|r| !is_busy(r)) {
            return (false, reply);
        }
        let remaining = deadline
            .saturating_duration_since(Instant::now())
            .as_millis();
        if remaining == 0 {
            return (false, reply);
        }
        (options.wait_ms)(HANDOVER_RETRY_MS.min(u64::try_from(remaining).unwrap_or(u64::MAX)));
    }
}

/// Say which options had no effect because the request went to a running instance.
///
/// `--trace`, `--camera` and `--backend` are applied when an instance starts; a launch that only
/// hands a request over cannot change the running one, and silently dropping them left users
/// waiting for a trace file that never appeared.
fn warn_startup_options_ignored(args: &RunArgs) {
    let given = [
        args.trace
            .as_ref()
            .is_some_and(|t| !t.as_os_str().is_empty()),
        args.camera.as_ref().is_some_and(|c| !c.is_empty()),
        args.backend.as_ref().is_some_and(|b| !b.is_empty()),
    ];
    let ignored: Vec<&str> = STARTUP_ONLY_OPTIONS
        .iter()
        .zip(given)
        .filter(|(_, given)| *given)
        .map(|(option, _)| *option)
        .collect();
    if ignored.is_empty() {
        return;
    }
    let one = ignored.len() == 1;
    log::warn!(
        "{APP_NAME} is already running, so {} {} not applied: {} only when {APP_NAME} starts. \
         Quit it first (tray menu > Quit {APP_NAME}, or '{}'), then run the command again.",
        ignored.join(", "),
        if one { "was" } else { "were" },
        if one { "it applies" } else { "they apply" },
        cli_command_text(&["ctl", "quit"]),
    );
}

/// Apply `--camera` / `--backend` for this session.
///
/// Returns `(key, (saved value, session value))` per overridden setting.
fn apply_overrides(
    settings: &mut Settings,
    args: &RunArgs,
) -> Vec<(&'static str, (String, String))> {
    let mut overrides = Vec::new();
    for (option, key, value) in [
        ("camera", "camera.device", &args.camera),
        ("backend", "general.backend", &args.backend),
    ] {
        let Some(value) = value.as_deref().map(str::trim).filter(|v| !v.is_empty()) else {
            continue;
        };
        let saved = setting(settings, key).to_string();
        set_setting(settings, key, value);
        overrides.push((key, (saved, value.to_string())));
        // Camera values can be file paths; keep them out of the log.
        let shown = if option == "backend" {
            value
        } else {
            "(from the command line)"
        };
        log::info!("Setting {key} overridden for this session: {shown}");
    }
    overrides
}

/// The value of an overridable setting (`"camera.device"` or `"general.backend"`).
fn setting<'a>(settings: &'a Settings, key: &str) -> &'a str {
    match key {
        "camera.device" => &settings.camera.device,
        "general.backend" => &settings.general.backend,
        _ => "",
    }
}

fn set_setting(settings: &mut Settings, key: &str, value: &str) {
    match key {
        "camera.device" => settings.camera.device = value.to_string(),
        "general.backend" => settings.general.backend = value.to_string(),
        _ => {}
    }
}

fn panic_text(payload: &(dyn Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown error".to_string())
}

/// Tell a user without a console that startup failed (the log has details).
fn show_error(text: &str) {
    #[cfg(any(windows, target_os = "macos"))]
    {
        rfd::MessageDialog::new()
            .set_level(rfd::MessageLevel::Error)
            .set_title(APP_NAME)
            .set_description(text)
            .set_buttons(rfd::MessageButtons::Ok)
            .show();
    }
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        eprintln!("{APP_NAME}: {text}");
        let shown = notify_rust::Notification::new()
            .appname(APP_NAME)
            .summary(APP_NAME)
            .body(text)
            .icon("dialog-error")
            .show();
        if let Err(err) = shown {
            log::debug!("Could not show the error notification: {err}");
        }
    }
}

// ==================================================================== event loop
/// Set by the Ctrl+C / SIGTERM handler; [`EyeTrackerApp::step`] quits when it sees it.
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

/// Quit cleanly on Ctrl+C (and SIGTERM outside Windows) while it lives: the handler only sets a
/// flag the event loop polls (at least every [`SIGNAL_POLL_MS`]). Signals that are ignored (a
/// process started in the background by a shell, or with nohup) stay ignored. The previous
/// handlers come back on drop.
pub struct SignalGuard {
    #[cfg(unix)]
    previous: Vec<(libc::c_int, libc::sigaction)>,
    #[cfg(windows)]
    installed: bool,
}

#[cfg(unix)]
extern "C" fn on_signal(_signal: libc::c_int) {
    INTERRUPTED.store(true, Ordering::SeqCst);
}

#[cfg(windows)]
unsafe extern "system" fn on_console_event(event: u32) -> i32 {
    // CTRL_C_EVENT and CTRL_BREAK_EVENT; closing the console is left to the system.
    if event <= 1 {
        INTERRUPTED.store(true, Ordering::SeqCst);
        1
    } else {
        0
    }
}

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    // The `windows` crate's Win32_System_Console feature is not enabled for this one call.
    fn SetConsoleCtrlHandler(
        handler: Option<unsafe extern "system" fn(u32) -> i32>,
        add: i32,
    ) -> i32;
}

impl SignalGuard {
    pub fn install() -> Self {
        INTERRUPTED.store(false, Ordering::SeqCst);
        #[cfg(unix)]
        {
            let mut previous = Vec::new();
            for signal in [libc::SIGINT, libc::SIGTERM] {
                // SAFETY: sigaction is called with valid pointers to initialised structs; the
                // handler only stores to an atomic, which is async-signal-safe.
                unsafe {
                    let mut old: libc::sigaction = std::mem::zeroed();
                    if libc::sigaction(signal, std::ptr::null(), &mut old) != 0 {
                        log::debug!("Cannot read the handler of signal {signal}");
                        continue;
                    }
                    if old.sa_sigaction == libc::SIG_IGN {
                        continue;
                    }
                    let mut new: libc::sigaction = std::mem::zeroed();
                    new.sa_sigaction = on_signal as extern "C" fn(libc::c_int) as usize;
                    libc::sigemptyset(&mut new.sa_mask);
                    new.sa_flags = libc::SA_RESTART;
                    if libc::sigaction(signal, &new, std::ptr::null_mut()) == 0 {
                        previous.push((signal, old));
                    } else {
                        log::debug!("Cannot handle signal {signal}");
                    }
                }
            }
            Self { previous }
        }
        #[cfg(windows)]
        {
            // SAFETY: the handler is a plain function that lives for the whole process.
            let installed = unsafe { SetConsoleCtrlHandler(Some(on_console_event), 1) } != 0;
            Self { installed }
        }
        #[cfg(not(any(unix, windows)))]
        {
            Self {}
        }
    }
}

impl Drop for SignalGuard {
    fn drop(&mut self) {
        #[cfg(unix)]
        for (signal, old) in self.previous.drain(..) {
            // SAFETY: restores the handler read in `install`.
            unsafe {
                libc::sigaction(signal, &old, std::ptr::null_mut());
            }
        }
        #[cfg(windows)]
        if self.installed {
            // SAFETY: removes the handler added in `install`.
            unsafe {
                SetConsoleCtrlHandler(Some(on_console_event), 0);
            }
        }
    }
}

/// The eframe side of the app.
///
/// eframe runs [`eframe::App::logic`] for every wake-up, and the app's work happens there. A
/// hidden root viewport is painted from a timer where eframe cannot create windows (opening one
/// then panics), so while any window is open the root is shown, as a 1×1 transparent,
/// click-through window, and windows are drawn from the ui pass after it was shown. With nothing
/// open it is hidden again.
struct Runner<'a> {
    context: &'a mut AppContext,
    /// ui passes so far.
    passes: u64,
    /// What the root viewport was last told (eframe shows it after the first frame).
    root_visible: bool,
    /// Windows are drawn in the ui passes after this one.
    shown_in: u64,
}

impl eframe::App for Runner<'_> {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let Some(app) = self.context.app.as_mut() else {
            ctx.send_viewport_cmd(ViewportCommand::Close);
            return;
        };
        app.step();
        if app.quit_requested() {
            ctx.send_viewport_cmd(ViewportCommand::Close);
            return;
        }
        let wanted = app.wants_ui();
        if wanted != self.root_visible {
            ctx.send_viewport_cmd(ViewportCommand::Visible(wanted));
            self.root_visible = wanted;
            // The command takes effect at the end of this pass.
            self.shown_in = self.passes + 1;
        }
        ctx.request_repaint_after(app.next_wakeup());
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.passes += 1;
        if !self.root_visible {
            return;
        }
        if self.passes <= self.shown_in {
            ui.ctx().request_repaint();
            return;
        }
        if let Some(app) = self.context.app.as_mut() {
            app.draw(ui.ctx());
        }
    }

    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        // The gaze dot and the countdown toast are see-through windows, the root is invisible.
        [0.0; 4]
    }

    fn on_exit(&mut self) {
        self.context.shutdown();
    }
}

/// Start the app inside the event loop and run it until it quits. Returns the exit code.
fn run_event_loop(context: &mut AppContext) -> i32 {
    let viewport = egui::ViewportBuilder::default()
        .with_title(APP_NAME)
        .with_app_id(APP_SLUG)
        .with_visible(false)
        .with_decorations(false)
        .with_transparent(true)
        .with_inner_size([1.0, 1.0])
        .with_taskbar(false)
        .with_mouse_passthrough(true)
        .with_active(false);
    let native = eframe::NativeOptions {
        viewport,
        ..eframe::NativeOptions::default()
    };
    let signals = SignalGuard::install();
    let mut failed = None;
    let failed_code = &mut failed;
    let started = &mut *context;
    let result = eframe::run_native(
        APP_NAME,
        native,
        Box::new(move |creation| {
            let context = started;
            *context.waker.lock() = Some(creation.egui_ctx.clone());
            if let Some(app) = &context.app {
                // No Dock icon or application menu for a tray app. The .app bundle's
                // LSUIElement does this too; this covers running the bare binary.
                app.services.set_accessory_app();
            }
            if let Some(code) = context.start() {
                *failed_code = Some(code);
                return Err("startup failed".into());
            }
            Ok(Box::new(Runner {
                context,
                passes: 0,
                root_visible: true,
                shown_in: 1,
            }))
        }),
    );
    drop(signals);
    if let Some(code) = failed {
        return code;
    }
    if let Err(err) = result {
        log::error!("The event loop failed: {err}");
        show_error(&format!("{APP_NAME} could not start:\n\n{err}"));
        return EXIT_ERROR;
    }
    log::info!("{APP_NAME} exited (code {EXIT_OK})");
    EXIT_OK
}

#[cfg(test)]
#[path = "app_tests.rs"]
mod tests;
