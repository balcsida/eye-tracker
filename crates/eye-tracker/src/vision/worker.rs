//! Background thread that captures frames and turns them into observations.
//!
//! The worker owns the camera and the vision backend. Both are created, used and released on
//! the worker thread (backends need not be thread-safe, and camera drivers are happiest when one
//! thread does all the talking). The rest of the application steers it through thread-safe
//! setters and receives results as [`WorkerEvent`]s on the channel it passed to
//! [`VisionWorker::new`]; the controller drains that channel on the UI thread.
//!
//! Low CPU use comes from four mechanisms:
//!
//! * the loop sleeps between frames for the interval chosen by the rate scheduler,
//! * a [`MotionGate`] skips inference while the picture is unchanged (a copy of the last
//!   observation is emitted instead; never of a blink, nor of an observation the backend has not
//!   settled on, see [`VisionBackend::settled`]),
//! * the image helpers run on the worker thread alone (see [`super::threads`]), and
//! * the camera is released entirely whenever the worker is inactive.
//!
//! Analysed frames without a face are also checked for being *blind* (lens covered, shutter
//! closed, unlit room); such observations carry `Observation::blind` so that presence detection
//! can tell "cannot see" from "nobody here".

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_channel::Sender;
use parking_lot::{Condvar, Mutex};
use serde_json::{Value, json};
use thiserror::Error;

use super::Frame;
use super::backends::base::VisionBackend;
use super::camera::FrameSource;
use super::motion::{MotionGate, THUMB_SIZE, is_blind, thumbnail};
use super::threads::{MAX_OPENCV_THREADS, limit_opencv_threads};
use crate::panes::registry::panic_message;
use crate::panes::worker::monotonic;
use crate::types::{Observation, WorkerStats};

/// What the worker reports (the Python callbacks `on_observation`, `on_stats`, `on_preview`).
#[derive(Debug, Clone)]
pub enum WorkerEvent {
    /// Every observation, including `skipped` copies emitted while the motion gate holds.
    Observation(Observation),
    /// A [`WorkerStats`] snapshot at most once per second, plus immediately when the camera
    /// opens or closes or an error appears or clears. Besides the counters, `extra` carries
    /// `"backend"`, `"feature_version"`, `"device"` (the frame source's settings string, once
    /// opened), `"frame_size"` (`[width, height]` of the analysed frames) and
    /// `"backend_failing"` (the backend keeps failing; `last_error` says why).
    Stats(WorkerStats),
    /// An annotated copy of each analysed frame while the preview is enabled.
    Preview(Frame),
}

/// Why a factory could not create its source or backend (for example [`BackendUnavailable`](super::backends::base::BackendUnavailable)).
pub type FactoryError = Box<dyn std::error::Error + Send + Sync>;
/// Creates the frame source. Called on the worker thread each time the camera has to be
/// (re)opened; may fail.
pub type SourceFactory = Box<dyn FnMut() -> Result<Box<dyn FrameSource>, FactoryError> + Send>;
/// Creates the vision backend on the worker thread; may fail (e.g. [`BackendUnavailable`](super::backends::base::BackendUnavailable)).
pub type BackendFactory = Box<dyn FnMut() -> Result<Box<dyn VisionBackend>, FactoryError> + Send>;
/// Time source for observation timestamps and frame pacing (monotonic seconds).
pub type Clock = Arc<dyn Fn() -> f64 + Send + Sync>;

/// Interval used until the controller sets one (4 fps).
pub const DEFAULT_INTERVAL_S: f64 = 0.25;
const MAX_INTERVAL_S: f64 = 60.0;
const EWMA_ALPHA: f64 = 0.2;
const SKIP_ALPHA: f64 = 0.1;
// Repeated identical problems (a missing camera, a buggy backend) are logged at most this often
// at WARNING level; repeats in between go to DEBUG.
const LOG_REPEAT_S: f64 = 60.0;
// Timestamps handed to the backend must strictly increase even with a coarse clock.
const MIN_TIMESTAMP_STEP: f64 = 1e-4;

/// Delays between attempts to open the camera or create the backend.
pub const BACKOFF_S: [f64; 4] = [1.0, 2.0, 5.0, 10.0];
/// Consecutive `process()` failures after which the backend is recreated.
pub const BACKEND_FAILURE_LIMIT: u32 = 3;
/// Minimum seconds between routine [`WorkerEvent::Stats`].
pub const STATS_PERIOD_S: f64 = 1.0;
/// The backend is closed (freeing its memory) after this long inactive.
pub const IDLE_BACKEND_CLOSE_S: f64 = 300.0;
pub const THREAD_NAME: &str = "eye-tracker-vision";
/// How long [`VisionWorker::stop`] waits for the thread.
pub const STOP_TIMEOUT_S: f64 = 3.0;

/// [`VisionWorker::set_interval`] was given NaN.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("interval must be a number")]
pub struct InvalidInterval;

/// The settings of [`VisionWorker::new`].
pub struct WorkerOptions {
    /// Time source for observation timestamps and frame pacing.
    pub clock: Clock,
    /// Delays between attempts to open the camera or create the backend; the last one repeats.
    pub backoff_s: Vec<f64>,
    /// The backend is closed after the worker was inactive this long.
    pub idle_backend_close_s: f64,
}

impl Default for WorkerOptions {
    fn default() -> Self {
        Self {
            clock: Arc::new(monotonic),
            backoff_s: BACKOFF_S.to_vec(),
            idle_backend_close_s: IDLE_BACKEND_CLOSE_S,
        }
    }
}

/// Snapshot of the settings the worker thread acts on.
#[derive(Debug, Clone, Copy)]
struct Config {
    interval: f64,
    active: bool,
    max_faces: usize,
    gate_enabled: bool,
    gate_threshold: f64,
    preview: bool,
}

fn skipped_copy(obs: &Observation, now: f64) -> Observation {
    Observation {
        timestamp: now,
        skipped: true,
        inference_ms: 0.0,
        ..obs.clone()
    }
}

/// Run `f`, turning a panic into its message (the Python code caught the exception).
fn guarded<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|payload| panic_message(payload.as_ref()))
}

fn seconds(value: f64) -> Duration {
    Duration::from_secs_f64(value.clamp(0.0, 3600.0))
}

/// Shared with the controlling thread; guarded by `Shared::control`.
struct Control {
    interval: f64,
    active: bool,
    max_faces: usize,
    gate_enabled: bool,
    gate_threshold: f64,
    preview: bool,
    pending_source: Option<SourceFactory>,
    pending_backend: Option<BackendFactory>,
    stop_requested: bool,
    /// Bumped by changes that must cut a back-off wait short.
    control_gen: u64,
    backend_info: Option<(String, String)>,
    /// The thread was started and has not finished.
    alive: bool,
    thread: Option<JoinHandle<()>>,
}

impl Control {
    fn has_pending(&self) -> bool {
        self.pending_source.is_some() || self.pending_backend.is_some()
    }

    fn snapshot(&self) -> Config {
        Config {
            interval: self.interval,
            active: self.active,
            max_faces: self.max_faces,
            gate_enabled: self.gate_enabled,
            gate_threshold: self.gate_threshold,
            preview: self.preview,
        }
    }

    /// Stop, deactivation or a reconfiguration: whatever the thread was about to do is stale.
    fn must_yield(&self) -> bool {
        self.stop_requested || !self.active || self.has_pending()
    }
}

#[derive(Default)]
struct Counters {
    frames: u64,
    processed: u64,
    skipped: u64,
    camera_opens: u64,
    backend_errors: u64,
}

/// Statistics; guarded by `Shared::stats`. Lock order: `control`, then `stats`.
struct StatsState {
    stats: WorkerStats,
    counters: Counters,
}

/// What survives a stop, for the next start.
struct Carry {
    source_factory: SourceFactory,
    backend_factory: BackendFactory,
    last_ts: f64,
    log_times: HashMap<String, f64>,
}

struct Shared {
    control: Mutex<Control>,
    cond: Condvar,
    /// Signalled when the thread has finished.
    exited: Condvar,
    stats: Mutex<StatsState>,
    carry: Mutex<Option<Carry>>,
    backoff_step: AtomicUsize,
    events: Sender<WorkerEvent>,
    clock: Clock,
    backoff_s: Vec<f64>,
    idle_backend_close_s: f64,
}

impl Shared {
    fn stats_snapshot(&self) -> WorkerStats {
        let state = self.stats.lock();
        // The counters go in every snapshot, also those published before any frame was
        // analysed (a camera that opens but whose frames all fail).
        let mut stats = state.stats.clone();
        let c = &state.counters;
        for (key, value) in [
            ("frames", c.frames),
            ("processed", c.processed),
            ("skipped", c.skipped),
            ("camera_opens", c.camera_opens),
            ("backend_errors", c.backend_errors),
        ] {
            stats.extra.insert(key.into(), json!(value));
        }
        stats
    }
}

/// Runs capture and analysis on a background thread.
///
/// All results are [`WorkerEvent`]s sent over the channel given to [`VisionWorker::new`]; a
/// receiver that went away is ignored. The setters can be called from any thread and wake the
/// loop.
pub struct VisionWorker {
    shared: Arc<Shared>,
}

impl VisionWorker {
    pub fn new(
        source_factory: SourceFactory,
        backend_factory: BackendFactory,
        events: Sender<WorkerEvent>,
        options: WorkerOptions,
    ) -> Self {
        let mut stats = WorkerStats {
            target_fps: 1.0 / DEFAULT_INTERVAL_S,
            ..WorkerStats::default()
        };
        stats.extra.insert("backend_failing".into(), json!(false));
        let shared = Shared {
            control: Mutex::new(Control {
                interval: DEFAULT_INTERVAL_S,
                active: true,
                max_faces: 1,
                gate_enabled: true,
                gate_threshold: 2.0,
                preview: false,
                pending_source: None,
                pending_backend: None,
                stop_requested: false,
                control_gen: 0,
                backend_info: None,
                alive: false,
                thread: None,
            }),
            cond: Condvar::new(),
            exited: Condvar::new(),
            stats: Mutex::new(StatsState {
                stats,
                counters: Counters::default(),
            }),
            carry: Mutex::new(Some(Carry {
                source_factory,
                backend_factory,
                last_ts: f64::NEG_INFINITY,
                log_times: HashMap::new(),
            })),
            backoff_step: AtomicUsize::new(0),
            events,
            clock: options.clock,
            backoff_s: options.backoff_s,
            idle_backend_close_s: options.idle_backend_close_s,
        };
        Self {
            shared: Arc::new(shared),
        }
    }

    // ================================================================ control API
    /// Start the worker thread (no-op if it is already running).
    pub fn start(&self) {
        let mut control = self.shared.control.lock();
        if control.alive {
            if control.stop_requested {
                log::warn!("Vision worker is still stopping; start() ignored");
            }
            return;
        }
        control.thread = None; // the previous thread has finished
        control.stop_requested = false;
        control.alive = true;
        let shared = self.shared.clone();
        let handle = thread::Builder::new()
            .name(THREAD_NAME.into())
            .spawn(move || run(shared))
            .expect("spawn the vision worker thread");
        control.thread = Some(handle);
    }

    /// Stop the thread and wait up to [`STOP_TIMEOUT_S`] seconds for it to finish.
    pub fn stop(&self) {
        self.stop_within(STOP_TIMEOUT_S);
    }

    /// Stop the thread and wait up to `timeout` seconds for it to finish.
    ///
    /// The camera is released and the backend closed on the worker thread before it exits. Safe
    /// to call repeatedly, before [`start`](Self::start) and from the worker thread itself (then
    /// it does not wait).
    pub fn stop_within(&self, timeout: f64) {
        let mut control = self.shared.control.lock();
        control.stop_requested = true;
        control.control_gen += 1;
        self.shared.cond.notify_all();
        let Some(id) = control.thread.as_ref().map(|handle| handle.thread().id()) else {
            return;
        };
        if id == thread::current().id() {
            return;
        }
        let deadline = Instant::now() + seconds(timeout);
        while control.alive {
            if self
                .shared
                .exited
                .wait_until(&mut control, deadline)
                .timed_out()
            {
                break;
            }
        }
        if control.alive {
            // Typically a camera driver blocking in open(); the thread releases everything as
            // soon as that call returns.
            log::warn!("Vision worker did not stop within {timeout:.1} s");
        } else if let Some(handle) = control.thread.take() {
            drop(control);
            let _ = handle.join();
        }
    }

    pub fn is_running(&self) -> bool {
        let control = self.shared.control.lock();
        control.alive && !control.stop_requested
    }

    /// Target period between analysed frames (0 = as fast as possible, at most 60 s).
    ///
    /// A shorter interval takes effect immediately, even mid-wait.
    pub fn set_interval(&self, seconds: f64) -> Result<(), InvalidInterval> {
        if seconds.is_nan() {
            return Err(InvalidInterval);
        }
        let value = seconds.clamp(0.0, MAX_INTERVAL_S);
        {
            let mut control = self.shared.control.lock();
            if value == control.interval {
                return Ok(());
            }
            control.interval = value;
            self.shared.cond.notify_all();
        }
        self.shared.stats.lock().stats.target_fps = if value > 0.0 { 1.0 / value } else { 0.0 };
        Ok(())
    }

    /// `false` releases the camera and idles; `true` (re)opens it.
    ///
    /// Switching on forgets the last error (without publishing anything): it described the
    /// camera before the pause, and errors are only published when they change, so a camera
    /// that fails the same way again must count as a new failure.
    pub fn set_active(&self, active: bool) {
        let mut control = self.shared.control.lock();
        if control.active == active {
            return;
        }
        control.active = active;
        if active {
            // Forgotten before the worker thread can wake (it needs this lock): a failure it
            // reports at once, with the old message, must be published, not taken for the old
            // news. Lock order: control, then stats (never the other way round).
            self.shared.stats.lock().stats.last_error = None;
        }
        control.control_gen += 1;
        self.shared.cond.notify_all();
    }

    /// Faces the backend should look for (2 for the shoulder guard).
    pub fn set_max_faces(&self, n: usize) {
        self.shared.control.lock().max_faces = n.max(1);
    }

    /// Enable/disable the motion gate and set its threshold (grey levels).
    pub fn set_motion_gate(&self, enabled: bool, threshold: f64) {
        let mut control = self.shared.control.lock();
        control.gate_enabled = enabled;
        control.gate_threshold = threshold.max(0.0);
    }

    /// Deliver annotated frames as [`WorkerEvent::Preview`].
    ///
    /// While the preview is on the motion gate is bypassed so the picture stays live.
    pub fn set_preview(&self, enabled: bool) {
        let mut control = self.shared.control.lock();
        control.preview = enabled;
        self.shared.cond.notify_all();
    }

    /// Swap the camera and/or backend; applied on the worker thread.
    ///
    /// The old source is released and the old backend closed before the new ones are created.
    /// Any pending back-off is cancelled so the change is tried at once.
    pub fn reconfigure(
        &self,
        source_factory: Option<SourceFactory>,
        backend_factory: Option<BackendFactory>,
    ) {
        if source_factory.is_none() && backend_factory.is_none() {
            return;
        }
        let mut control = self.shared.control.lock();
        if let Some(factory) = source_factory {
            control.pending_source = Some(factory);
        }
        if let Some(factory) = backend_factory {
            control.pending_backend = Some(factory);
            // Unknown until the new backend exists: the old identity must not be taken for the
            // new one's (see backend_info).
            control.backend_info = None;
        }
        control.control_gen += 1;
        self.shared.cond.notify_all();
    }

    /// A snapshot of the current statistics (safe to keep and mutate).
    pub fn stats(&self) -> WorkerStats {
        self.shared.stats_snapshot()
    }

    /// `(name, feature_version)` of the most recently created backend.
    ///
    /// `None` before the first backend exists, and after [`reconfigure`](Self::reconfigure)
    /// asked for another one until that one is created.
    pub fn backend_info(&self) -> Option<(String, String)> {
        self.shared.control.lock().backend_info.clone()
    }
}

impl Drop for VisionWorker {
    fn drop(&mut self) {
        self.stop();
    }
}

// ================================================================ thread body
fn run(shared: Arc<Shared>) {
    let carry = shared
        .carry
        .lock()
        .take()
        .expect("the previous worker run returned its state");
    let mut runner = Runner::new(shared.clone(), carry);
    runner.run();
    *shared.carry.lock() = Some(runner.into_carry());
    let mut control = shared.control.lock();
    control.alive = false;
    shared.exited.notify_all();
}

/// The worker-thread state.
struct Runner {
    shared: Arc<Shared>,
    source_factory: SourceFactory,
    backend_factory: BackendFactory,
    source: Option<Box<dyn FrameSource>>,
    backend: Option<Box<dyn VisionBackend>>,
    applied_max_faces: Option<usize>,
    gate: MotionGate,
    gate_in_use: bool,
    last_obs: Option<Observation>,
    last_ts: f64,
    /// Scheduled start of the current frame.
    slot: Option<f64>,
    backend_failures: u32,
    backend_recreations: u32,
    log_times: HashMap<String, f64>,
    dt_ewma: Option<f64>,
    last_frame_t: Option<f64>,
    last_publish: f64,
}

impl Runner {
    fn new(shared: Arc<Shared>, carry: Carry) -> Self {
        let threshold = shared.control.lock().gate_threshold;
        let mut gate = MotionGate::default();
        gate.threshold = threshold;
        Self {
            shared,
            source_factory: carry.source_factory,
            backend_factory: carry.backend_factory,
            source: None,
            backend: None,
            applied_max_faces: None,
            gate,
            gate_in_use: false,
            last_obs: None,
            last_ts: carry.last_ts,
            slot: None,
            backend_failures: 0,
            backend_recreations: 0,
            log_times: carry.log_times,
            dt_ewma: None,
            last_frame_t: None,
            last_publish: f64::NEG_INFINITY,
        }
    }

    fn into_carry(self) -> Carry {
        Carry {
            source_factory: self.source_factory,
            backend_factory: self.backend_factory,
            last_ts: self.last_ts,
            log_times: self.log_times,
        }
    }

    fn now(&self) -> f64 {
        (self.shared.clock)()
    }

    fn set_backoff_step(&self, step: usize) {
        self.shared.backoff_step.store(step, Ordering::Relaxed);
    }

    fn run(&mut self) {
        log::debug!("Vision worker started");
        // The image helpers run on this thread alone (see `super::threads`).
        limit_opencv_threads(MAX_OPENCV_THREADS);
        self.slot = None;
        self.set_backoff_step(0);
        self.backend_failures = 0;
        self.backend_recreations = 0;
        self.shared
            .stats
            .lock()
            .stats
            .extra
            .insert("backend_failing".into(), json!(false));
        loop {
            let (cfg, new_source, new_backend) = {
                let mut control = self.shared.control.lock();
                if control.stop_requested {
                    break;
                }
                (
                    control.snapshot(),
                    control.pending_source.take(),
                    control.pending_backend.take(),
                )
            };
            let outcome = guarded(|| {
                if new_source.is_some() || new_backend.is_some() {
                    self.apply_reconfigure(new_source, new_backend);
                }
                self.iteration(&cfg);
            });
            if let Err(message) = outcome {
                // A bug must not kill the thread (the camera would stay on with nobody reading
                // it); report it and retry after a pause.
                self.log_throttled(
                    "loop",
                    &format!("Unexpected error in the vision worker: {message}"),
                );
                self.set_error("Internal error in the vision worker (see the log)");
                self.release_source();
                self.backoff_wait();
            }
        }
        self.release_source();
        self.close_backend();
        self.reset_tracking();
        self.publish_stats(true);
        log::debug!("Vision worker stopped");
    }

    fn iteration(&mut self, cfg: &Config) {
        if !cfg.active {
            self.go_idle();
            return;
        }
        self.apply_config(cfg);
        if self.backend.is_none() && !self.create_backend(cfg) {
            self.backoff_wait();
            return;
        }
        // Creating a backend can take a second; privacy mode, a lock or stop() may have
        // arrived meanwhile, and the camera must then stay off.
        if self.shared.control.lock().must_yield() {
            return;
        }
        if !self.ensure_source() {
            self.backoff_wait();
            return;
        }
        // Opening a camera can take seconds; if the worker was deactivated, stopped or
        // reconfigured meanwhile, act on that before using it.
        if self.shared.control.lock().must_yield() {
            return;
        }
        // Frames are scheduled on a grid (slot, slot + interval, ...) so that a timer overshoot
        // in one wait (~16 ms on Windows) is recovered in the next and the average rate matches
        // the target. Falling more than one interval off the grid (a slow frame, a changed
        // interval) restarts it from now.
        let now = self.now();
        let slot = match self.slot {
            Some(slot) if (now - slot).abs() <= cfg.interval => slot,
            _ => now,
        };
        if self.cycle(cfg) {
            self.slot = Some(self.wait_next(slot));
        } else {
            self.slot = None;
            self.backoff_wait();
        }
    }

    /// Read and analyse one frame. Returns `false` to back off before the next.
    fn cycle(&mut self, cfg: &Config) -> bool {
        let read_started = Instant::now();
        let (frame, is_open, last_error, read_panic) = {
            let source = self
                .source
                .as_mut()
                .expect("cycle runs with an open source");
            let (frame, read_panic) = match guarded(|| source.read()) {
                Ok(frame) => (frame, None),
                Err(message) => {
                    log::debug!("Frame source read panicked: {message}");
                    (None, Some(message))
                }
            };
            (frame, source.is_open(), source.last_error(), read_panic)
        };
        if let Some(message) = read_panic
            && is_open
        {
            self.set_error(&format!("Camera read failed: {message}"));
        }
        let read_ms = read_started.elapsed().as_secs_f64() * 1e3;
        let Some(frame) = frame else {
            if is_open {
                return true; // transient; the source gives up by itself if it persists
            }
            let message = last_error.unwrap_or_else(|| "Camera stopped delivering frames".into());
            self.log_throttled("read", &message);
            self.set_error(&message);
            self.release_source();
            self.reset_tracking();
            return false;
        };

        let mut now = self.now();
        if now <= self.last_ts {
            now = self.last_ts + MIN_TIMESTAMP_STEP;
        }
        self.last_ts = now;
        self.set_backoff_step(0);
        if self.backend_recreations == 0 {
            // A failing backend's error stays until a frame is analysed again.
            self.clear_error();
        }

        let use_gate = cfg.gate_enabled && !cfg.preview;
        if use_gate != self.gate_in_use {
            self.gate.reset();
            self.gate_in_use = use_gate;
        }
        // Never hold a closed-eyes observation: the reopening can be too subtle for the gate,
        // and a repeated blink would suppress gaze for seconds. Nor one the backend has not
        // settled on (its tracking crop was still catching up with a moved face): the picture
        // stops changing as soon as the user sits still again, and the gate would repeat the
        // biased features for seconds.
        let settled = self.backend.as_ref().is_some_and(|b| b.settled());
        if use_gate
            && let Some(last) = &self.last_obs
            && !last.blink
            && settled
            && !self.gate.should_process(&frame, now)
        {
            let copy = skipped_copy(last, now);
            self.record_frame(now, read_ms, true, 0.0);
            self.send(WorkerEvent::Observation(copy));
            return true;
        }

        let process_started = Instant::now();
        let outcome = {
            let backend = self.backend.as_mut().expect("cycle runs with a backend");
            guarded(|| backend.process(&frame, now))
        };
        let mut obs = match outcome {
            Ok(obs) => obs,
            Err(message) => return self.backend_failed(&message),
        };
        self.backend_failures = 0;
        if self.backend_recreations > 0 {
            self.backend_recreations = 0;
            self.shared
                .stats
                .lock()
                .stats
                .extra
                .insert("backend_failing".into(), json!(false));
            self.clear_error();
        }
        let elapsed_ms = process_started.elapsed().as_secs_f64() * 1e3;
        if use_gate {
            self.gate.mark_processed(&frame, now, obs.face_box);
        }
        if obs.face_count == 0 && !obs.blind {
            let reference = if use_gate {
                self.gate.reference()
            } else {
                None
            };
            let blind = match reference {
                Some(plane) => is_blind(plane),
                None => is_blind(&thumbnail(&frame, THUMB_SIZE)),
            };
            if blind {
                obs.blind = true;
            }
        }
        {
            let size = json!([obs.frame_size.0, obs.frame_size.1]);
            let mut state = self.shared.stats.lock();
            if state.stats.extra.get("frame_size") != Some(&size) {
                state.stats.extra.insert("frame_size".into(), size);
            }
        }
        let inference_ms = if obs.inference_ms != 0.0 {
            obs.inference_ms
        } else {
            elapsed_ms
        };
        self.last_obs = Some(obs.clone());
        self.record_frame(now, read_ms, false, inference_ms);
        self.send(WorkerEvent::Observation(obs.clone()));

        if cfg.preview {
            let backend = self.backend.as_ref().expect("cycle runs with a backend");
            match guarded(|| backend.annotate(&frame, &obs)) {
                Ok(annotated) => self.send(WorkerEvent::Preview(annotated)),
                Err(message) => self.log_throttled(
                    "annotate",
                    &format!("Could not annotate the preview frame: {message}"),
                ),
            }
        }
        true
    }

    /// `process()` failed. Returns whether to go on with the next frame at once.
    fn backend_failed(&mut self, message: &str) -> bool {
        self.backend_failures += 1;
        self.shared.stats.lock().counters.backend_errors += 1;
        self.log_throttled(
            "process",
            &format!("Vision backend failed to process a frame: {message}"),
        );
        if self.backend_failures < BACKEND_FAILURE_LIMIT {
            return true;
        }
        self.backend_recreations += 1;
        self.log_throttled(
            "recreate",
            &format!(
                "Vision backend failed {} times in a row; recreating it",
                self.backend_failures
            ),
        );
        // Make the failure visible (the camera would otherwise look healthy while nothing is
        // analysed) and keep the camera off while backing off.
        self.shared
            .stats
            .lock()
            .stats
            .extra
            .insert("backend_failing".into(), json!(true));
        let detail = if message.is_empty() { "error" } else { message };
        self.set_error(&format!("Vision backend keeps failing: {detail}"));
        self.close_backend();
        self.release_source();
        // Back off before recreating, longer each time it keeps failing, so a persistently
        // broken backend cannot burn CPU being rebuilt.
        self.set_backoff_step(self.backend_recreations as usize - 1);
        false
    }

    // ----------------------------------------------------------- resources
    fn create_backend(&mut self, cfg: &Config) -> bool {
        let backend = match (self.backend_factory)() {
            Ok(backend) => backend,
            Err(error) => {
                // A missing runtime or model (`BackendUnavailable`) is expected on some systems.
                let message = format!("Vision backend unavailable: {error}");
                self.log_throttled("backend", &message);
                self.set_error(&message);
                // Nothing can be analysed, so keep the camera (and its light) off.
                self.release_source();
                return false;
            }
        };
        let name = backend.name();
        let feature_version = backend.feature_version();
        self.backend = Some(backend);
        self.backend_failures = 0;
        {
            let mut control = self.shared.control.lock();
            // Built by a factory that reconfigure() replaced meanwhile: it is closed before
            // use, and its identity must not pass for the new one's.
            if control.pending_backend.is_none() {
                control.backend_info = Some((name.to_string(), feature_version.to_string()));
            }
        }
        self.applied_max_faces = None;
        self.apply_config(cfg);
        {
            let mut state = self.shared.stats.lock();
            state.stats.extra.insert("backend".into(), json!(name));
            state
                .stats
                .extra
                .insert("feature_version".into(), json!(feature_version));
        }
        log::debug!("Vision backend {name} created");
        true
    }

    fn close_backend(&mut self) {
        let backend = self.backend.take();
        self.applied_max_faces = None;
        let Some(mut backend) = backend else {
            return;
        };
        if let Err(message) = guarded(|| backend.close()) {
            log::debug!("Closing the vision backend failed: {message}");
        }
        // A new backend may produce different features; never reuse old results.
        self.reset_tracking();
    }

    fn ensure_source(&mut self) -> bool {
        if self.source.as_ref().is_some_and(|source| source.is_open()) {
            return true;
        }
        self.release_source();
        let mut source = match (self.source_factory)() {
            Ok(source) => source,
            Err(error) => {
                let message = error.to_string();
                let message = if message.is_empty() {
                    "error".into()
                } else {
                    message
                };
                self.log_throttled("open", &format!("Camera unavailable: {message}"));
                self.set_error(&message);
                return false;
            }
        };
        let (opened, error) = match guarded(|| (source.open(), source.last_error())) {
            Ok(result) => result,
            Err(message) => {
                log::debug!("Frame source open() panicked: {message}");
                (
                    false,
                    Some(format!("Camera could not be opened: {message}")),
                )
            }
        };
        let device = source.device();
        self.source = Some(source);
        if !opened {
            let message = error.unwrap_or_else(|| "Camera could not be opened".into());
            self.log_throttled("open", &message);
            self.set_error(&message);
            self.release_source();
            return false;
        }
        {
            let mut state = self.shared.stats.lock();
            state.counters.camera_opens += 1;
            state.stats.camera_open = true;
            state
                .stats
                .extra
                .insert("device".into(), device.map_or(Value::Null, Value::String));
        }
        if self.backend_recreations == 0 {
            self.clear_error();
        }
        self.publish_stats(true);
        true
    }

    fn release_source(&mut self) {
        let Some(mut source) = self.source.take() else {
            return;
        };
        if let Err(message) = guarded(|| source.release()) {
            log::debug!("Releasing the frame source failed: {message}");
        }
        let was_open = {
            let mut state = self.shared.stats.lock();
            let was_open = state.stats.camera_open;
            state.stats.camera_open = false;
            state.stats.fps = 0.0;
            was_open
        };
        self.dt_ewma = None;
        self.last_frame_t = None;
        if was_open {
            self.publish_stats(true);
        }
    }

    fn apply_reconfigure(
        &mut self,
        source_factory: Option<SourceFactory>,
        backend_factory: Option<BackendFactory>,
    ) {
        if let Some(factory) = source_factory {
            self.release_source();
            self.source_factory = factory;
        }
        if let Some(factory) = backend_factory {
            self.close_backend();
            self.backend_factory = factory;
            // A different backend gets a clean slate.
            self.backend_recreations = 0;
            self.shared
                .stats
                .lock()
                .stats
                .extra
                .insert("backend_failing".into(), json!(false));
        }
        self.reset_tracking();
        self.set_backoff_step(0);
        self.log_times.clear();
        self.clear_error();
    }

    fn apply_config(&mut self, cfg: &Config) {
        self.gate.threshold = cfg.gate_threshold;
        if let Some(backend) = self.backend.as_mut()
            && Some(cfg.max_faces) != self.applied_max_faces
        {
            if let Err(message) = guarded(|| backend.set_max_faces(cfg.max_faces)) {
                log::warn!(
                    "Vision backend rejected max_faces={}: {message}",
                    cfg.max_faces
                );
            }
            self.applied_max_faces = Some(cfg.max_faces);
        }
    }

    /// Forget the last frame: the next one is analysed afresh by gate and backend.
    ///
    /// Called whenever the camera was released or replaced. The backend (kept for up to
    /// [`IDLE_BACKEND_CLOSE_S`]) must forget the face it tracked too: after a pause the picture
    /// can be a different one, and a crop left over from before would latch onto whatever face
    /// is there now.
    fn reset_tracking(&mut self) {
        self.gate.reset();
        self.last_obs = None;
        let failed = self
            .backend
            .as_mut()
            .and_then(|backend| guarded(|| backend.reset()).err());
        if let Some(message) = failed {
            self.log_throttled(
                "reset",
                &format!("Vision backend failed to reset: {message}"),
            );
        }
    }

    // -------------------------------------------------------------- waiting
    /// Release the camera and block until activated, reconfigured or stopped.
    fn go_idle(&mut self) {
        self.release_source();
        self.reset_tracking();
        self.set_backoff_step(0);
        self.slot = None;
        let close_at = monotonic() + self.shared.idle_backend_close_s;
        let shared = self.shared.clone();
        let mut control = shared.control.lock();
        while !control.stop_requested && !control.active && !control.has_pending() {
            if self.backend.is_none() {
                shared.cond.wait(&mut control);
                continue;
            }
            let remaining = close_at - monotonic();
            if remaining > 0.0 {
                shared.cond.wait_for(&mut control, seconds(remaining));
                continue;
            }
            // Inactive for a long time (privacy mode, locked session): free the backend's
            // memory. Closing it under the lock is fine, it is fast.
            log::debug!("Vision worker idle; closing the backend to free memory");
            self.close_backend();
        }
    }

    /// Sleep until `slot + interval` and return that time.
    ///
    /// Setters wake the wait early so a shortened interval applies at once; `stop()`,
    /// `set_active(false)` and `reconfigure()` end it.
    fn wait_next(&self, slot: f64) -> f64 {
        let shared = &self.shared;
        let mut control = shared.control.lock();
        let mut due = slot + control.interval;
        while !control.stop_requested && control.active && !control.has_pending() {
            due = slot + control.interval;
            let remaining = due - self.now();
            if remaining <= 0.0 {
                break;
            }
            if shared
                .cond
                .wait_for(&mut control, seconds(remaining.min(MAX_INTERVAL_S)))
                .timed_out()
            {
                // Not re-checking the clock keeps the loop moving even with a clock that does
                // not advance.
                break;
            }
        }
        due
    }

    fn backoff_wait(&self) {
        let shared = &self.shared;
        let step = shared.backoff_step.fetch_add(1, Ordering::Relaxed);
        let delay = shared.backoff_s[step.min(shared.backoff_s.len() - 1)];
        let deadline = monotonic() + delay;
        let mut control = shared.control.lock();
        let generation = control.control_gen;
        while !control.stop_requested && control.control_gen == generation {
            let remaining = deadline - monotonic();
            if remaining <= 0.0 {
                return;
            }
            shared.cond.wait_for(&mut control, seconds(remaining));
        }
    }

    // ----------------------------------------------------------- statistics
    fn record_frame(&mut self, now: f64, read_ms: f64, skipped: bool, inference_ms: f64) {
        if let Some(last) = self.last_frame_t {
            let dt = now - last;
            if dt > 0.0 {
                self.dt_ewma = Some(match self.dt_ewma {
                    None => dt,
                    Some(ewma) => ewma + EWMA_ALPHA * (dt - ewma),
                });
            }
        }
        self.last_frame_t = Some(now);
        {
            let mut state = self.shared.stats.lock();
            let StatsState { stats, counters } = &mut *state;
            stats.fps = self.dt_ewma.map_or(0.0, |ewma| 1.0 / ewma);
            stats.skip_ratio += SKIP_ALPHA * ((if skipped { 1.0 } else { 0.0 }) - stats.skip_ratio);
            counters.frames += 1;
            if skipped {
                counters.skipped += 1;
            } else {
                counters.processed += 1;
                if stats.inference_ms <= 0.0 {
                    stats.inference_ms = inference_ms;
                } else {
                    stats.inference_ms += EWMA_ALPHA * (inference_ms - stats.inference_ms);
                }
            }
            let previous_read = stats
                .extra
                .get("read_ms")
                .and_then(Value::as_f64)
                .unwrap_or(read_ms);
            stats.extra.insert(
                "read_ms".into(),
                json!(previous_read + EWMA_ALPHA * (read_ms - previous_read)),
            );
            stats
                .extra
                .insert("motion".into(), json!(self.gate.last_motion()));
        }
        self.publish_stats(false);
    }

    fn set_error(&mut self, message: &str) {
        let changed = {
            let mut state = self.shared.stats.lock();
            let changed = state.stats.last_error.as_deref() != Some(message);
            state.stats.last_error = Some(message.to_string());
            changed
        };
        if changed {
            self.publish_stats(true);
        }
    }

    fn clear_error(&mut self) {
        let changed = self.shared.stats.lock().stats.last_error.take().is_some();
        if changed {
            self.publish_stats(true);
        }
    }

    fn publish_stats(&mut self, force: bool) {
        let now = monotonic();
        if !force && now - self.last_publish < STATS_PERIOD_S {
            return;
        }
        self.last_publish = now;
        let snapshot = self.shared.stats_snapshot();
        self.send(WorkerEvent::Stats(snapshot));
    }

    // -------------------------------------------------------------- helpers
    /// A consumer that went away is not the worker's problem.
    fn send(&self, event: WorkerEvent) {
        let _ = self.shared.events.send(event);
    }

    fn log_throttled(&mut self, key: &str, message: &str) {
        let now = monotonic();
        let last = self
            .log_times
            .get(key)
            .copied()
            .unwrap_or(f64::NEG_INFINITY);
        if now - last >= LOG_REPEAT_S {
            self.log_times.insert(key.to_string(), now);
            log::warn!("{message}");
        } else {
            log::debug!("{message}");
        }
    }
}

#[cfg(test)]
#[path = "worker_tests.rs"]
mod tests;
