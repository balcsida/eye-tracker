//! The fakes of tests/test_controller.py, shared with the app tests.
//!
//! Everything the controller touches is faked: the vision worker (observations are pushed by
//! the test and handled at once), the platform (lock/display/window calls are recorded, never
//! performed; it also plays the cursor), the clock, the monitor layout and the hotkey manager.
//! Config and calibration files live in a temp dir ([`super::app_dirs`]).

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use crossbeam_channel::Sender;
use parking_lot::{Condvar, Mutex, MutexGuard};

use crate::config::Settings;
use crate::engine::controller::{
    BackendFactoryMaker, Controller, ControllerEvent, ControllerOptions, JobRunner, LockJob,
    PaneRegistryFactory, PaneWorkerFactory, WorkerLike,
};
use crate::gaze::calibration::{CalibrationSample, samples_to_arrays};
use crate::gaze::model::GazeModel;
use crate::gaze::store::{CalibrationData, save_calibration};
use crate::paths;
use crate::platform::base::PlatformServices;
use crate::platform::hotkeys::{
    Hotkey, HotkeyArg, HotkeyCallback, HotkeyError, HotkeyManager, parse_hotkey,
};
use crate::types::{
    AppIdentity, GazePoint, Monitor, Observation, Rect, TrackingState, WindowHandle, WindowRef,
    WorkerStats, layout_signature, virtual_bounds,
};
use crate::vision::Frame;
use crate::vision::backends::base::VisionBackend;
use crate::vision::worker::{BackendFactory, Clock, SourceFactory, WorkerEvent};

pub(crate) const BACKEND: (&str, &str) = ("fake", "fake-1");
pub(crate) const RIGHT_CENTRE: (f64, f64) = (2880.0, 540.0);
pub(crate) const LEFT_CENTRE: (f64, f64) = (960.0, 540.0);
/// Synthetic features: the gaze point in kilo-pixels, so a linear model is exact.
pub(crate) const FEATURE_SCALE: f64 = 1000.0;

pub(crate) fn left() -> Monitor {
    Monitor {
        primary: true,
        ..Monitor::new(0, "left", Rect::new(0, 0, 1920, 1080))
    }
}

pub(crate) fn right() -> Monitor {
    Monitor::new(1, "right", Rect::new(1920, 0, 1920, 1080))
}

pub(crate) fn monitors() -> Vec<Monitor> {
    vec![left(), right()]
}

// ----------------------------------------------------------------------------- fakes
#[derive(Clone)]
pub(crate) struct FakeClock(Arc<Mutex<f64>>);

impl FakeClock {
    pub(crate) fn new(t: f64) -> Self {
        Self(Arc::new(Mutex::new(t)))
    }

    pub(crate) fn now(&self) -> f64 {
        *self.0.lock()
    }

    pub(crate) fn advance(&self, dt: f64) -> f64 {
        let mut t = self.0.lock();
        *t += dt;
        *t
    }

    pub(crate) fn clock(&self) -> Clock {
        let inner = self.0.clone();
        Arc::new(move || *inner.lock())
    }
}

/// The pointer (the Python `FakeCursor`), played by [`FakePlatform`].
pub(crate) struct FakeCursor {
    pub(crate) position: (i32, i32),
    pub(crate) moves: Vec<(i32, i32)>,
    pub(crate) allow: bool,
    /// False: moves succeed but the position stays the old one, like an XWayland client whose
    /// pointer went over a native Wayland window.
    pub(crate) track: bool,
}

/// An OS action [`FakePlatform`] recorded.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Call {
    WindowApp(WindowHandle),
    LockScreen,
    DisplayOff,
    WakeDisplay,
    WindowAt(i32, i32),
    ActivateWindow(WindowHandle),
}

impl Call {
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Call::WindowApp(_) => "window_app",
            Call::LockScreen => "lock_screen",
            Call::DisplayOff => "display_off",
            Call::WakeDisplay => "wake_display",
            Call::WindowAt(..) => "window_at",
            Call::ActivateWindow(_) => "activate_window",
        }
    }
}

pub(crate) struct FakePlatformState {
    pub(crate) calls: Vec<Call>,
    pub(crate) locked: Option<bool>,
    pub(crate) camera_in_use: Option<bool>,
    pub(crate) processes: HashSet<String>,
    pub(crate) idle: Option<f64>,
    pub(crate) key_idle: Option<f64>,
    pub(crate) foreground: Option<WindowRef>,
    pub(crate) window_under: Option<WindowRef>,
    pub(crate) lock_ok: bool,
    pub(crate) activate_ok: bool,
    pub(crate) cursor_reliable: bool,
    /// `"unknown"`: not applicable (not macOS).
    pub(crate) accessibility: &'static str,
    pub(crate) camera_permission: Option<bool>,
    pub(crate) background: Vec<bool>,
    /// Split panes: the application of each window handle, and the capability.
    pub(crate) apps: HashMap<WindowHandle, AppIdentity>,
    pub(crate) panes_capable: bool,
    /// Windows `is_window_valid` says are gone.
    pub(crate) invalid_windows: HashSet<WindowHandle>,
    pub(crate) cursor: FakeCursor,
    /// The threads `lock_screen` ran on.
    pub(crate) lock_threads: Vec<ThreadId>,
}

impl FakePlatformState {
    pub(crate) fn names(&self) -> Vec<&'static str> {
        self.calls.iter().map(Call::name).collect()
    }
}

/// A gate a slow `lock_screen` waits at (the Python `SlowLockPlatform`'s events).
#[derive(Default)]
pub(crate) struct Gate {
    open: Mutex<bool>,
    cond: Condvar,
}

impl Gate {
    pub(crate) fn set(&self) {
        *self.open.lock() = true;
        self.cond.notify_all();
    }

    /// Wait up to `timeout` seconds; whether the gate opened.
    pub(crate) fn wait(&self, timeout: f64) -> bool {
        let deadline = Instant::now() + Duration::from_secs_f64(timeout);
        let mut open = self.open.lock();
        while !*open {
            if self.cond.wait_until(&mut open, deadline).timed_out() {
                break;
            }
        }
        *open
    }
}

/// Records OS actions instead of performing them (never locks this machine).
pub(crate) struct FakePlatform {
    state: Mutex<FakePlatformState>,
    /// `lock_screen` blocks until the test releases it (a hung screensaver).
    pub(crate) slow: bool,
    pub(crate) entered: Gate,
    pub(crate) release: Gate,
}

impl FakePlatform {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(FakePlatformState {
                calls: Vec::new(),
                locked: Some(false),
                camera_in_use: Some(false),
                processes: HashSet::new(),
                idle: None,
                key_idle: None,
                foreground: None,
                window_under: None,
                lock_ok: true,
                activate_ok: true,
                cursor_reliable: true,
                accessibility: "unknown",
                camera_permission: None,
                background: Vec::new(),
                apps: HashMap::new(),
                panes_capable: true,
                invalid_windows: HashSet::new(),
                cursor: FakeCursor {
                    position: (500, 500),
                    moves: Vec::new(),
                    allow: true,
                    track: true,
                },
                lock_threads: Vec::new(),
            }),
            slow: false,
            entered: Gate::default(),
            release: Gate::default(),
        }
    }

    /// The Python `SlowLockPlatform`.
    pub(crate) fn slow_lock() -> Self {
        Self {
            slow: true,
            ..Self::new()
        }
    }

    /// The recorded state (never hold it across a controller call).
    pub(crate) fn p(&self) -> MutexGuard<'_, FakePlatformState> {
        self.state.lock()
    }

    pub(crate) fn names(&self) -> Vec<&'static str> {
        self.p().names()
    }
}

impl PlatformServices for FakePlatform {
    fn name(&self) -> &'static str {
        "fake"
    }

    fn capabilities(&self) -> HashMap<&'static str, bool> {
        let mut caps = crate::platform::GenericPlatform.capabilities();
        caps.insert("panes", self.p().panes_capable);
        caps
    }

    fn window_app(&self, window: &WindowRef) -> Option<AppIdentity> {
        let mut state = self.p();
        state.calls.push(Call::WindowApp(window.handle.clone()));
        state.apps.get(&window.handle).cloned()
    }

    fn window_client_rect(&self, window: &WindowRef) -> Option<Rect> {
        window.rect
    }

    fn lock_screen(&self) -> bool {
        if self.slow {
            self.p().lock_threads.push(std::thread::current().id());
            self.entered.set();
            assert!(
                self.release.wait(5.0),
                "the test never released the lock call"
            );
        } else {
            self.p().lock_threads.push(std::thread::current().id());
        }
        let mut state = self.p();
        state.calls.push(Call::LockScreen);
        state.lock_ok
    }

    fn display_off(&self) -> bool {
        self.p().calls.push(Call::DisplayOff);
        true
    }

    fn wake_display(&self) -> bool {
        self.p().calls.push(Call::WakeDisplay);
        true
    }

    fn is_session_locked(&self) -> Option<bool> {
        self.p().locked
    }

    fn seconds_since_input(&self) -> Option<f64> {
        self.p().idle
    }

    fn seconds_since_key_input(&self) -> Option<f64> {
        self.p().key_idle
    }

    fn cursor_position_reliable(&self) -> bool {
        self.p().cursor_reliable
    }

    fn move_cursor(&self, x: i32, y: i32) -> Option<bool> {
        let mut state = self.p();
        let cursor = &mut state.cursor;
        cursor.moves.push((x, y));
        if cursor.allow && cursor.track {
            cursor.position = (x, y);
        }
        Some(cursor.allow)
    }

    fn cursor_position(&self) -> Option<(i32, i32)> {
        Some(self.p().cursor.position)
    }

    fn foreground_window(&self) -> Option<WindowRef> {
        self.p().foreground.clone()
    }

    fn window_at(&self, x: i32, y: i32) -> Option<WindowRef> {
        let mut state = self.p();
        state.calls.push(Call::WindowAt(x, y));
        state.window_under.clone()
    }

    fn activate_window(&self, window: &WindowRef) -> bool {
        let mut state = self.p();
        state
            .calls
            .push(Call::ActivateWindow(window.handle.clone()));
        state.activate_ok
    }

    fn is_window_valid(&self, window: &WindowRef) -> bool {
        !self.p().invalid_windows.contains(&window.handle)
    }

    fn window_rect(&self, window: &WindowRef) -> Option<Rect> {
        window.rect
    }

    fn camera_in_use_by_other_app(&self) -> Option<bool> {
        self.p().camera_in_use
    }

    fn running_process_names(&self) -> HashSet<String> {
        self.p().processes.clone()
    }

    fn permissions(&self) -> HashMap<&'static str, Option<bool>> {
        HashMap::from([
            ("camera", self.p().camera_permission),
            ("accessibility", None),
        ])
    }

    fn accessibility_status(&self) -> &'static str {
        self.p().accessibility
    }

    fn set_background_activity(&self, active: bool) -> bool {
        self.p().background.push(active);
        true
    }
}

/// What [`FakeWorker`] was told, and what the test makes it report.
pub(crate) struct FakeWorkerState {
    pub(crate) intervals: Vec<f64>,
    pub(crate) active: Vec<bool>,
    pub(crate) max_faces: Vec<usize>,
    pub(crate) gates: Vec<(bool, f64)>,
    pub(crate) previews: Vec<bool>,
    /// `(source factory given, backend factory given)` of each `reconfigure`.
    pub(crate) reconfigures: Vec<(bool, bool)>,
    pub(crate) started: bool,
    pub(crate) stopped: bool,
    pub(crate) backend_info: Option<(String, String)>,
    pub(crate) stats: WorkerStats,
    pub(crate) backend_factory: Option<BackendFactory>,
    /// The backend factory of the latest `reconfigure` that had one.
    pub(crate) new_backend_factory: Option<BackendFactory>,
}

/// Records every control call; observations/stats/previews are pushed by the test.
#[derive(Clone)]
pub(crate) struct FakeWorker {
    state: Arc<Mutex<FakeWorkerState>>,
    events: Sender<WorkerEvent>,
}

impl FakeWorker {
    pub(crate) fn new(
        _source_factory: SourceFactory,
        backend_factory: BackendFactory,
        events: Sender<WorkerEvent>,
    ) -> Self {
        Self {
            state: Arc::new(Mutex::new(FakeWorkerState {
                intervals: Vec::new(),
                active: Vec::new(),
                max_faces: Vec::new(),
                gates: Vec::new(),
                previews: Vec::new(),
                reconfigures: Vec::new(),
                started: false,
                stopped: false,
                backend_info: Some((BACKEND.0.into(), BACKEND.1.into())),
                stats: WorkerStats {
                    fps: 4.0,
                    camera_open: true,
                    ..WorkerStats::default()
                },
                backend_factory: Some(backend_factory),
                new_backend_factory: None,
            })),
            events,
        }
    }

    pub(crate) fn s(&self) -> MutexGuard<'_, FakeWorkerState> {
        self.state.lock()
    }

    pub(crate) fn is_active(&self) -> bool {
        self.s().active.last() == Some(&true)
    }

    /// Send an observation as the worker thread would (handled at the next
    /// `process_queued`).
    pub(crate) fn push(&self, obs: Observation) {
        let _ = self.events.send(WorkerEvent::Observation(obs));
    }

    pub(crate) fn send_stats(&self, stats: WorkerStats) {
        let _ = self.events.send(WorkerEvent::Stats(stats));
    }

    pub(crate) fn send_preview(&self, frame: Frame) {
        let _ = self.events.send(WorkerEvent::Preview(frame));
    }

    /// The worker builds its backend through the controller's factory.
    pub(crate) fn build_backend(&self) {
        let factory = self.s().backend_factory.take();
        if let Some(mut factory) = factory {
            let _ = factory();
            self.s().backend_factory = Some(factory);
        }
    }

    /// The worker builds the backend of the latest `reconfigure`.
    pub(crate) fn build_new_backend(&self) {
        let factory = self.s().new_backend_factory.take();
        if let Some(mut factory) = factory {
            let _ = factory();
            self.s().new_backend_factory = Some(factory);
        }
    }
}

impl WorkerLike for FakeWorker {
    fn start(&self) {
        self.s().started = true;
    }

    fn stop(&self, _timeout: f64) {
        self.s().stopped = true;
    }

    fn set_interval(&self, seconds: f64) {
        self.s().intervals.push(seconds);
    }

    fn set_active(&self, active: bool) {
        self.s().active.push(active);
    }

    fn set_max_faces(&self, n: usize) {
        self.s().max_faces.push(n);
    }

    fn set_motion_gate(&self, enabled: bool, threshold: f64) {
        self.s().gates.push((enabled, threshold));
    }

    fn set_preview(&self, enabled: bool) {
        self.s().previews.push(enabled);
    }

    fn reconfigure(
        &self,
        source_factory: Option<SourceFactory>,
        backend_factory: Option<BackendFactory>,
    ) {
        let mut state = self.s();
        state
            .reconfigures
            .push((source_factory.is_some(), backend_factory.is_some()));
        if backend_factory.is_some() {
            state.new_backend_factory = backend_factory;
        }
    }

    fn stats(&self) -> WorkerStats {
        self.s().stats.clone()
    }

    fn backend_info(&self) -> Option<(String, String)> {
        self.s().backend_info.clone()
    }
}

pub(crate) struct FakeHotkeysState {
    pub(crate) bindings: HashMap<String, (String, HotkeyCallback)>,
    pub(crate) fail: HashSet<String>,
    pub(crate) errors: HashMap<String, String>,
    pub(crate) stopped: bool,
}

pub(crate) struct FakeHotkeys {
    state: Mutex<FakeHotkeysState>,
}

impl FakeHotkeys {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(FakeHotkeysState {
                bindings: HashMap::new(),
                fail: HashSet::new(),
                errors: HashMap::new(),
                stopped: false,
            }),
        }
    }

    pub(crate) fn s(&self) -> MutexGuard<'_, FakeHotkeysState> {
        self.state.lock()
    }

    /// The combination bound to `name`.
    pub(crate) fn combo(&self, name: &str) -> Option<String> {
        self.s().bindings.get(name).map(|(combo, _)| combo.clone())
    }

    pub(crate) fn names(&self) -> HashSet<String> {
        self.s().bindings.keys().cloned().collect()
    }

    /// The callback registered for `name` (call it without holding the lock).
    pub(crate) fn callback(&self, name: &str) -> HotkeyCallback {
        self.s().bindings[name].1.clone()
    }
}

impl HotkeyManager for FakeHotkeys {
    fn supported(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "fake"
    }

    fn note(&self) -> Option<String> {
        None
    }

    fn register(&self, name: &str, hotkey: HotkeyArg<'_>, callback: HotkeyCallback) -> bool {
        let combo = match hotkey {
            HotkeyArg::Text(text) => text.to_string(),
            HotkeyArg::Hotkey(hotkey) => hotkey.to_string(),
        };
        let mut state = self.s();
        if state.fail.contains(&combo) {
            return false;
        }
        state.bindings.insert(name.to_string(), (combo, callback));
        true
    }

    fn unregister(&self, name: &str) -> bool {
        self.s().bindings.remove(name).is_some()
    }

    fn unregister_all(&self) {
        self.s().bindings.clear();
    }

    fn registered(&self) -> HashMap<String, Hotkey> {
        self.s()
            .bindings
            .iter()
            .filter_map(|(name, (combo, _))| parse_hotkey(combo).ok().map(|hk| (name.clone(), hk)))
            .collect()
    }

    fn last_error(&self, name: &str) -> Option<String> {
        self.s().errors.get(name).cloned()
    }

    fn layout_conflict(&self, _hotkey: HotkeyArg<'_>) -> Result<Option<String>, HotkeyError> {
        Ok(None)
    }

    fn start(&self) {}

    fn stop(&self) {
        let mut state = self.s();
        state.stopped = true;
        state.bindings.clear();
    }
}

/// What the worker builds through the controller's backend factory.
pub(crate) struct FakeBackend;

impl VisionBackend for FakeBackend {
    fn name(&self) -> &'static str {
        BACKEND.0
    }

    fn feature_names(&self) -> &'static [&'static str] {
        &["gaze_x", "gaze_y"]
    }

    fn feature_version(&self) -> &'static str {
        BACKEND.1
    }

    fn gaze_features(&self) -> &'static [&'static str] {
        &["gaze_x", "gaze_y"]
    }

    fn process(&mut self, _frame_bgr: &Frame, timestamp: f64) -> Observation {
        Observation::new(timestamp, 0)
    }
}

/// Backend factories that build a [`FakeBackend`] whatever the setting says (like `auto`
/// falling back); the backend names requested are recorded (the `fake_backends` fixture).
pub(crate) fn fake_backends() -> (BackendFactoryMaker, Arc<Mutex<Vec<String>>>) {
    let requested: Arc<Mutex<Vec<String>>> = Arc::default();
    let record = requested.clone();
    let maker: BackendFactoryMaker = Arc::new(move |settings: &Settings, _max_faces| {
        let name = settings.general.backend.clone();
        let record = record.clone();
        Box::new(move || {
            record.lock().push(name.clone());
            Ok(Box::new(FakeBackend) as Box<dyn VisionBackend>)
        })
    });
    (maker, requested)
}

// --------------------------------------------------------------------------- helpers
/// Synchronous lock runner.
pub(crate) fn run_now() -> JobRunner {
    Box::new(|job: LockJob| job())
}

pub(crate) fn features_for(x: f64, y: f64) -> Vec<f64> {
    vec![x / FEATURE_SCALE, y / FEATURE_SCALE]
}

pub(crate) fn gaze_obs(point: (f64, f64), faces: u32) -> Observation {
    Observation {
        features: Some(features_for(point.0, point.1)),
        quality: 1.0,
        ..Observation::new(0.0, faces)
    }
}

pub(crate) fn no_face() -> Observation {
    Observation::new(0.0, 0)
}

pub(crate) fn make_calibration(
    monitors: &[Monitor],
    nonlinear: Option<&[usize]>,
) -> CalibrationData {
    let mut samples = Vec::new();
    let mut point = 0;
    for m in monitors {
        for nx in [0.1, 0.5, 0.9] {
            for ny in [0.1, 0.5, 0.9] {
                let (x, y) = m.rect.denormalize(nx, ny);
                for _ in 0..2 {
                    samples.push(CalibrationSample::new(
                        features_for(x, y),
                        x,
                        y,
                        m.index,
                        point,
                    ));
                }
                point += 1;
            }
        }
    }
    let (x, y, _) = samples_to_arrays(&samples).expect("samples");
    let model = GazeModel::new(1, 0.0, nonlinear)
        .expect("model")
        .fit(&x, &y, None, virtual_bounds(monitors).ok(), None)
        .expect("fit");
    let mut data = CalibrationData::new(
        BACKEND.0,
        BACKEND.1,
        layout_signature(monitors),
        monitors.to_vec(),
        samples,
        Vec::new(),
        model,
    );
    data.report.insert("grade".into(), "excellent".into());
    data
}

pub(crate) fn make_settings() -> Settings {
    let mut s = Settings::default();
    s.general.first_run_done = true; // walk-away actions only notify before that
    s.switching.smoothing = 0.0; // gaze == model output, exactly
    s.presence.away_timeout_s = 10;
    s.presence.warning_s = 5;
    s
}

/// The keyword arguments of the Python `make_controller` fixture.
pub(crate) struct Opts {
    pub(crate) settings: Option<Settings>,
    pub(crate) calibrated: bool,
    pub(crate) platform: Option<Arc<FakePlatform>>,
    pub(crate) hotkeys: Option<Arc<FakeHotkeys>>,
    pub(crate) start: bool,
    pub(crate) threaded_locks: bool,
    pub(crate) clock: Option<FakeClock>,
    pub(crate) trace_path: Option<PathBuf>,
    pub(crate) pane_registry_factory: Option<PaneRegistryFactory>,
    pub(crate) pane_worker_factory: Option<PaneWorkerFactory>,
    pub(crate) backend_factory: Option<BackendFactoryMaker>,
}

impl Default for Opts {
    fn default() -> Self {
        Self {
            settings: None,
            calibrated: true,
            platform: None,
            hotkeys: None,
            start: true,
            threaded_locks: false,
            clock: None,
            trace_path: None,
            pane_registry_factory: None,
            pane_worker_factory: None,
            backend_factory: None,
        }
    }
}

/// A controller with every seam faked, and the events it emitted.
pub(crate) struct Harness {
    pub(crate) controller: Controller,
    pub(crate) platform: Arc<FakePlatform>,
    pub(crate) clock: FakeClock,
    pub(crate) hotkeys: Arc<FakeHotkeys>,
    pub(crate) monitors: Arc<Mutex<Vec<Monitor>>>,
    pub(crate) workers: Arc<Mutex<Vec<FakeWorker>>>,
    events: Vec<ControllerEvent>,
}

/// Build (and by default start) a controller; call [`super::app_dirs`] first.
pub(crate) fn make_controller(opts: Opts) -> Harness {
    if opts.calibrated {
        save_calibration(
            &paths::calibration_file(),
            &make_calibration(&monitors(), None),
        )
        .expect("save the calibration");
    }
    let platform = opts
        .platform
        .unwrap_or_else(|| Arc::new(FakePlatform::new()));
    let hotkeys = opts.hotkeys.unwrap_or_else(|| Arc::new(FakeHotkeys::new()));
    let clock = opts.clock.unwrap_or_else(|| FakeClock::new(100.0));
    let layout = Arc::new(Mutex::new(monitors()));
    let workers: Arc<Mutex<Vec<FakeWorker>>> = Arc::default();
    let made = workers.clone();
    let provided = layout.clone();
    let manager = hotkeys.clone();
    let controller = Controller::new(
        &opts.settings.unwrap_or_else(make_settings),
        platform.clone(),
        ControllerOptions {
            worker_factory: Some(Box::new(move |source, backend, events| {
                let worker = FakeWorker::new(source, backend, events);
                made.lock().push(worker.clone());
                Box::new(worker)
            })),
            monitors_provider: Some(Box::new(move || provided.lock().clone())),
            clock: Some(clock.clock()),
            hotkey_manager_factory: Some(Box::new(move || {
                manager.clone() as Arc<dyn HotkeyManager>
            })),
            trace_path: opts.trace_path,
            // Screen locks run on a thread in the app; synchronously here unless a test is
            // about that thread.
            lock_runner: if opts.threaded_locks {
                None
            } else {
                Some(run_now())
            },
            pane_registry_factory: opts.pane_registry_factory,
            pane_worker_factory: opts.pane_worker_factory,
            backend_factory: opts.backend_factory,
        },
    );
    let mut h = Harness {
        controller,
        platform,
        clock,
        hotkeys,
        monitors: layout,
        workers,
        events: Vec::new(),
    };
    if opts.start {
        h.controller.start();
    }
    h
}

impl Harness {
    pub(crate) fn worker(&self) -> FakeWorker {
        self.workers.lock().last().expect("a worker").clone()
    }

    pub(crate) fn state(&self) -> TrackingState {
        self.controller.state()
    }

    /// The fake platform's recorded state (never hold it across a controller call).
    pub(crate) fn p(&self) -> MutexGuard<'_, FakePlatformState> {
        self.platform.p()
    }

    /// Handle what the threads delivered (Qt's `processEvents`).
    pub(crate) fn process(&mut self) {
        self.controller.process_queued();
    }

    pub(crate) fn push(&mut self, obs: Observation, dt: f64) {
        self.clock.advance(dt);
        self.worker().push(obs);
        self.process();
    }

    pub(crate) fn on_stats(&mut self, stats: WorkerStats) {
        self.worker().send_stats(stats);
        self.process();
    }

    pub(crate) fn on_preview(&mut self, frame: Frame) {
        self.worker().send_preview(frame);
        self.process();
    }

    /// Press the hotkey `name` (its callback, on this thread).
    pub(crate) fn press(&mut self, name: &str) {
        (self.hotkeys.callback(name))();
        self.process();
    }

    pub(crate) fn tick(&mut self, dt: f64) {
        let now = self.clock.advance(dt);
        self.controller.tick(now);
    }

    /// Push `obs` every `step` seconds (with a housekeeping tick each time).
    pub(crate) fn feed(&mut self, obs: &Observation, seconds: f64, step: f64) {
        for _ in 0..(seconds / step).round() as usize {
            self.push(obs.clone(), step);
            let now = self.clock.now();
            self.controller.tick(now);
        }
    }

    /// Every event so far.
    pub(crate) fn events(&mut self) -> &[ControllerEvent] {
        let fresh = self.controller.drain_events();
        self.events.extend(fresh);
        &self.events
    }

    fn select<T>(&mut self, pick: impl Fn(&ControllerEvent) -> Option<T>) -> Vec<T> {
        self.events().iter().filter_map(pick).collect()
    }

    pub(crate) fn state_changed(&mut self) -> Vec<TrackingState> {
        self.select(|e| match e {
            ControllerEvent::StateChanged(s) => Some(*s),
            _ => None,
        })
    }

    pub(crate) fn gaze_changed(&mut self) -> Vec<Option<GazePoint>> {
        self.select(|e| match e {
            ControllerEvent::GazeChanged(g) => Some(*g),
            _ => None,
        })
    }

    pub(crate) fn observations(&mut self) -> Vec<Observation> {
        self.select(|e| match e {
            ControllerEvent::Observation(o) => Some((**o).clone()),
            _ => None,
        })
    }

    pub(crate) fn stats_changed(&mut self) -> Vec<serde_json::Map<String, serde_json::Value>> {
        self.select(|e| match e {
            ControllerEvent::StatsChanged(s) => Some(s.clone()),
            _ => None,
        })
    }

    pub(crate) fn notify(&mut self) -> Vec<(String, String)> {
        self.select(|e| match e {
            ControllerEvent::Notify { title, message } => Some((title.clone(), message.clone())),
            _ => None,
        })
    }

    pub(crate) fn titles(&mut self) -> Vec<String> {
        self.notify().into_iter().map(|(title, _)| title).collect()
    }

    pub(crate) fn switched(&mut self) -> Vec<i32> {
        self.select(|e| match e {
            ControllerEvent::Switched(i) => Some(*i),
            _ => None,
        })
    }

    pub(crate) fn away_warning(&mut self) -> Vec<f64> {
        self.select(|e| match e {
            ControllerEvent::AwayWarning(s) => Some(*s),
            _ => None,
        })
    }

    pub(crate) fn away_cancelled(&mut self) -> usize {
        self.select(|e| matches!(e, ControllerEvent::AwayCancelled).then_some(()))
            .len()
    }

    pub(crate) fn guard_changed(&mut self) -> Vec<bool> {
        self.select(|e| match e {
            ControllerEvent::GuardChanged(b) => Some(*b),
            _ => None,
        })
    }

    pub(crate) fn preview_frames(&mut self) -> Vec<Frame> {
        self.select(|e| match e {
            ControllerEvent::PreviewFrame(f) => Some(f.clone()),
            _ => None,
        })
    }

    pub(crate) fn calibration_required(&mut self) -> Vec<String> {
        self.select(|e| match e {
            ControllerEvent::CalibrationRequired(r) => Some(r.clone()),
            _ => None,
        })
    }

    pub(crate) fn settings_changed(&mut self) -> Vec<Settings> {
        self.select(|e| match e {
            ControllerEvent::SettingsChanged(s) => Some((**s).clone()),
            _ => None,
        })
    }

    pub(crate) fn ui_requested(&mut self) -> Vec<String> {
        self.select(|e| match e {
            ControllerEvent::UiRequested(c) => Some(c.clone()),
            _ => None,
        })
    }

    pub(crate) fn permission_needed(&mut self) -> Vec<String> {
        self.select(|e| match e {
            ControllerEvent::PermissionNeeded(n) => Some(n.clone()),
            _ => None,
        })
    }
}
