// Ported from tests/test_worker.py (VisionWorker with fake sources and backends).
//
// The worker reports through a channel instead of callbacks, which changes some tests:
// - test_callback_exceptions_do_not_kill_the_thread, test_preview_callback_exception_is_contained:
//   no callbacks exist; test_a_dropped_receiver_does_not_kill_the_thread covers the equivalent
//   (a consumer that went away).
// - test_stop_from_callback_does_not_deadlock: becomes test_stop_from_the_worker_thread_does_not_
//   deadlock (stop() called from inside the backend).
// - test_worker_caps_opencv_threads: there is no thread pool to cap, nothing to mock
//   (see threads_tests.rs).
// - test_the_worker_wakes_with_the_last_error_already_forgotten: ported without the delayed
//   statistics lock (a Python lock stand-in); it checks the same ordering without widening the
//   race window.
// Fake backends report failures by panicking; the worker catches panics where the Python code
// caught exceptions.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{OnceLock, Weak};
use std::thread::ThreadId;

use approx::assert_relative_eq;
use crossbeam_channel::Receiver;

use super::*;
use crate::vision::backends::base::BackendUnavailable;
use crate::vision::camera::CameraError;

const FAST_BACKOFF: [f64; 3] = [0.01, 0.02, 0.04];
const TIMEOUT: f64 = 3.0;

// --------------------------------------------------------------------------- fakes
#[derive(Default)]
struct SourceState {
    open_calls: usize,
    release_calls: usize,
    reads: usize,
    threads: HashSet<ThreadId>,
    open: bool,
    error: Option<String>,
}

/// How a fake source behaves (the keyword arguments of the Python `FakeSource`).
#[derive(Clone)]
struct SourceSpec {
    can_open: Arc<dyn Fn() -> bool + Send + Sync>,
    frame_fn: Arc<dyn Fn(usize) -> Frame + Send + Sync>,
    die_after: Option<usize>,
    device: Option<String>,
}

impl Default for SourceSpec {
    fn default() -> Self {
        Self {
            can_open: Arc::new(|| true),
            frame_fn: Arc::new(|_| Frame::filled(64, 48, [100; 3])),
            die_after: None,
            device: None,
        }
    }
}

struct SourceInner {
    spec: SourceSpec,
    state: Mutex<SourceState>,
}

/// In-memory frame source with scriptable behaviour; clones share state.
#[derive(Clone)]
struct FakeSource(Arc<SourceInner>);

impl FakeSource {
    fn new(spec: SourceSpec) -> Self {
        Self(Arc::new(SourceInner {
            spec,
            state: Mutex::default(),
        }))
    }

    fn with_frames(frame_fn: impl Fn(usize) -> Frame + Send + Sync + 'static) -> Self {
        Self::new(SourceSpec {
            frame_fn: Arc::new(frame_fn),
            ..SourceSpec::default()
        })
    }

    fn is_open(&self) -> bool {
        self.0.state.lock().open
    }

    fn release_calls(&self) -> usize {
        self.0.state.lock().release_calls
    }

    fn reads(&self) -> usize {
        self.0.state.lock().reads
    }

    fn threads(&self) -> HashSet<ThreadId> {
        self.0.state.lock().threads.clone()
    }
}

impl FrameSource for FakeSource {
    fn open(&mut self) -> bool {
        let allowed = (self.0.spec.can_open)();
        let mut state = self.0.state.lock();
        state.threads.insert(thread::current().id());
        state.open_calls += 1;
        if !allowed {
            state.error = Some("fake camera is busy".into());
            return false;
        }
        state.open = true;
        state.error = None;
        true
    }

    fn read(&mut self) -> Option<Frame> {
        let reads = {
            let mut state = self.0.state.lock();
            if !state.open {
                return None;
            }
            state.reads += 1;
            if self.0.spec.die_after.is_some_and(|n| state.reads > n) {
                state.open = false;
                state.error = Some("fake camera unplugged".into());
                return None;
            }
            state.reads
        };
        Some((self.0.spec.frame_fn)(reads))
    }

    fn release(&mut self) {
        let mut state = self.0.state.lock();
        state.release_calls += 1;
        state.open = false;
    }

    fn is_open(&self) -> bool {
        self.0.state.lock().open
    }

    fn last_error(&self) -> Option<String> {
        self.0.state.lock().error.clone()
    }

    fn device(&self) -> Option<String> {
        self.0.spec.device.clone()
    }
}

#[derive(Default)]
struct BackendState {
    fail_times: usize,
    process_calls: usize,
    close_calls: usize,
    reset_calls: usize,
    max_faces_calls: Vec<usize>,
    timestamps: Vec<f64>,
    threads: HashSet<ThreadId>,
}

/// How a fake backend behaves. The Python `ScriptedBackend` (`blinks`, `face_count`),
/// `UnsettledBackend` (`unsettled`) and the broken-reset backend are options here.
#[derive(Clone)]
struct BackendSpec {
    name: &'static str,
    feature_version: &'static str,
    fail_times: usize,
    /// Report a blink for the first `blinks` analysed frames.
    blinks: usize,
    face_count: u32,
    /// `settled` is `false` after each of the first `unsettled` frames.
    unsettled: Option<usize>,
    broken_reset: bool,
    on_process: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl Default for BackendSpec {
    fn default() -> Self {
        Self {
            name: "fake",
            feature_version: "fake-1",
            fail_times: 0,
            blinks: 0,
            face_count: 1,
            unsettled: None,
            broken_reset: false,
            on_process: None,
        }
    }
}

struct BackendInner {
    spec: BackendSpec,
    state: Mutex<BackendState>,
}

/// Fake vision backend; clones share state.
#[derive(Clone)]
struct FakeBackend(Arc<BackendInner>);

impl FakeBackend {
    fn new() -> Self {
        Self::with(BackendSpec::default())
    }

    fn with(spec: BackendSpec) -> Self {
        let state = BackendState {
            fail_times: spec.fail_times,
            ..BackendState::default()
        };
        Self(Arc::new(BackendInner {
            spec,
            state: Mutex::new(state),
        }))
    }

    fn failing() -> Self {
        Self::with(BackendSpec {
            fail_times: 1_000_000,
            ..BackendSpec::default()
        })
    }

    fn process_calls(&self) -> usize {
        self.0.state.lock().process_calls
    }

    fn close_calls(&self) -> usize {
        self.0.state.lock().close_calls
    }

    fn reset_calls(&self) -> usize {
        self.0.state.lock().reset_calls
    }

    fn max_faces_calls(&self) -> Vec<usize> {
        self.0.state.lock().max_faces_calls.clone()
    }

    fn timestamps(&self) -> Vec<f64> {
        self.0.state.lock().timestamps.clone()
    }

    fn threads(&self) -> HashSet<ThreadId> {
        self.0.state.lock().threads.clone()
    }

    fn touch(&self) -> parking_lot::MutexGuard<'_, BackendState> {
        let mut state = self.0.state.lock();
        state.threads.insert(thread::current().id());
        state
    }
}

impl VisionBackend for FakeBackend {
    fn name(&self) -> &'static str {
        self.0.spec.name
    }

    fn feature_names(&self) -> &'static [&'static str] {
        &["a", "b"]
    }

    fn feature_version(&self) -> &'static str {
        self.0.spec.feature_version
    }

    fn process(&mut self, frame_bgr: &Frame, timestamp: f64) -> Observation {
        let calls = {
            let mut state = self.touch();
            state.process_calls += 1;
            state.timestamps.push(timestamp);
            if state.fail_times > 0 {
                state.fail_times -= 1;
                drop(state);
                panic!("simulated inference failure");
            }
            state.process_calls
        };
        if let Some(hook) = &self.0.spec.on_process {
            hook();
        }
        let frame_size = (frame_bgr.width as u32, frame_bgr.height as u32);
        if self.0.spec.face_count == 0 {
            return Observation {
                frame_size,
                ..Observation::new(timestamp, 0)
            };
        }
        Observation {
            features: Some(vec![1.0, calls as f64]),
            quality: 1.0,
            face_box: Some((0.25, 0.2, 0.5, 0.6)),
            inference_ms: 1.5,
            frame_size,
            blink: calls <= self.0.spec.blinks,
            ..Observation::new(timestamp, self.0.spec.face_count)
        }
    }

    fn set_max_faces(&mut self, n: usize) {
        self.touch().max_faces_calls.push(n);
    }

    fn settled(&self) -> bool {
        match self.0.spec.unsettled {
            Some(unsettled) => self.0.state.lock().process_calls > unsettled,
            None => true,
        }
    }

    fn reset(&mut self) {
        self.touch().reset_calls += 1;
        if self.0.spec.broken_reset {
            panic!("simulated reset failure");
        }
    }

    fn annotate(&self, frame_bgr: &Frame, _observation: &Observation) -> Frame {
        let mut out = frame_bgr.clone();
        out.set_pixel(0, 0, [1, 2, 3]);
        out
    }

    fn close(&mut self) {
        self.touch().close_calls += 1;
    }
}

/// A one-shot signal (`threading.Event`).
#[derive(Clone, Default)]
struct Event(Arc<(Mutex<bool>, Condvar)>);

impl Event {
    fn set(&self) {
        *self.0.0.lock() = true;
        self.0.1.notify_all();
    }

    fn wait(&self, seconds_: f64) -> bool {
        let deadline = Instant::now() + seconds(seconds_);
        let mut set = self.0.0.lock();
        while !*set {
            if self.0.1.wait_until(&mut set, deadline).timed_out() {
                break;
            }
        }
        *set
    }
}

#[derive(Default)]
struct Seen {
    observations: Vec<Observation>,
    stats: Vec<WorkerStats>,
    previews: Vec<Frame>,
}

/// Collects what the worker reports.
struct Recorder {
    rx: Receiver<WorkerEvent>,
    seen: Mutex<Seen>,
}

impl Recorder {
    fn new() -> (Sender<WorkerEvent>, Self) {
        let (tx, rx) = crossbeam_channel::unbounded();
        (
            tx,
            Self {
                rx,
                seen: Mutex::default(),
            },
        )
    }

    fn drain(&self) -> parking_lot::MutexGuard<'_, Seen> {
        let mut seen = self.seen.lock();
        while let Ok(event) = self.rx.try_recv() {
            match event {
                WorkerEvent::Observation(obs) => seen.observations.push(obs),
                WorkerEvent::Stats(stats) => seen.stats.push(stats),
                WorkerEvent::Preview(frame) => seen.previews.push(frame),
            }
        }
        seen
    }

    fn observations(&self) -> Vec<Observation> {
        self.drain().observations.clone()
    }

    fn stats(&self) -> Vec<WorkerStats> {
        self.drain().stats.clone()
    }

    fn previews(&self) -> usize {
        self.drain().previews.len()
    }

    fn preview(&self, n: usize) -> Frame {
        self.drain().previews[n].clone()
    }

    fn count(&self) -> usize {
        self.drain().observations.len()
    }

    fn wait_for_within(&self, timeout: f64, predicate: impl Fn(&Recorder) -> bool) -> bool {
        let deadline = Instant::now() + seconds(timeout);
        while Instant::now() < deadline {
            if predicate(self) {
                return true;
            }
            thread::sleep(Duration::from_millis(1));
        }
        predicate(self)
    }

    fn wait_for(&self, predicate: impl Fn(&Recorder) -> bool) -> bool {
        self.wait_for_within(TIMEOUT, predicate)
    }

    fn wait_for_observations(&self, n: usize) -> bool {
        self.wait_for(|r| r.count() >= n)
    }
}

fn strictly_increasing(values: &[f64]) -> bool {
    values.windows(2).all(|pair| pair[1] > pair[0])
}

fn wait_until(predicate: impl Fn() -> bool) -> bool {
    let deadline = Instant::now() + seconds(TIMEOUT);
    while Instant::now() < deadline {
        if predicate() {
            return true;
        }
        thread::sleep(Duration::from_millis(2));
    }
    predicate()
}

fn backend_factory(backend: &FakeBackend) -> BackendFactory {
    let backend = backend.clone();
    Box::new(move || Ok(Box::new(backend.clone()) as Box<dyn VisionBackend>))
}

fn source_factory(source: &FakeSource) -> SourceFactory {
    let source = source.clone();
    Box::new(move || Ok(Box::new(source.clone()) as Box<dyn FrameSource>))
}

fn failing_source_factory(message: &'static str) -> SourceFactory {
    Box::new(move || Err(CameraError(message.into()).into()))
}

fn fast_options() -> WorkerOptions {
    WorkerOptions {
        backoff_s: FAST_BACKOFF.to_vec(),
        ..WorkerOptions::default()
    }
}

/// Settings of a [`Harness`] (the attributes the Python tests assign before `start()`).
struct HarnessConfig {
    source: SourceSpec,
    /// The next backend fails this many times in a row (only the first one is faulty).
    backend_fail_times: usize,
    /// Every backend fails (instead of `backend_fail_times`).
    backends_always_fail: bool,
    options: WorkerOptions,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            source: SourceSpec::default(),
            backend_fail_times: 0,
            backends_always_fail: false,
            options: fast_options(),
        }
    }
}

/// What the harness's factories record.
struct Created {
    sources: Mutex<Vec<FakeSource>>,
    backends: Mutex<Vec<FakeBackend>>,
    factory_threads: Mutex<HashSet<ThreadId>>,
    source_spec: SourceSpec,
    backend_fail_times: Mutex<usize>,
    backends_always_fail: bool,
}

impl Created {
    fn make_source(&self) -> FakeSource {
        self.factory_threads.lock().insert(thread::current().id());
        let source = FakeSource::new(self.source_spec.clone());
        self.sources.lock().push(source.clone());
        source
    }

    fn make_backend(&self) -> FakeBackend {
        self.factory_threads.lock().insert(thread::current().id());
        let mut fail_times = self.backend_fail_times.lock();
        let backend = FakeBackend::with(BackendSpec {
            fail_times: if self.backends_always_fail {
                1_000_000
            } else {
                *fail_times
            },
            ..BackendSpec::default()
        });
        *fail_times = 0; // only the next backend is faulty
        self.backends.lock().push(backend.clone());
        backend
    }
}

/// A worker wired to fakes; every source/backend it creates is recorded.
struct Harness {
    recorder: Recorder,
    created: Arc<Created>,
    worker: VisionWorker,
}

impl Harness {
    fn new() -> Self {
        Self::with(HarnessConfig::default())
    }

    fn with(config: HarnessConfig) -> Self {
        let (tx, recorder) = Recorder::new();
        let created = Arc::new(Created {
            sources: Mutex::default(),
            backends: Mutex::default(),
            factory_threads: Mutex::default(),
            source_spec: config.source,
            backend_fail_times: Mutex::new(config.backend_fail_times),
            backends_always_fail: config.backends_always_fail,
        });
        let make_source = {
            let created = created.clone();
            Box::new(move || Ok(Box::new(created.make_source()) as Box<dyn FrameSource>))
        };
        let make_backend = {
            let created = created.clone();
            Box::new(move || Ok(Box::new(created.make_backend()) as Box<dyn VisionBackend>))
        };
        let worker = VisionWorker::new(make_source, make_backend, tx, config.options);
        worker.set_interval(0.005).unwrap();
        Self {
            recorder,
            created,
            worker,
        }
    }

    fn make_source(&self) -> SourceFactory {
        let created = self.created.clone();
        Box::new(move || Ok(Box::new(created.make_source()) as Box<dyn FrameSource>))
    }

    fn sources(&self) -> Vec<FakeSource> {
        self.created.sources.lock().clone()
    }

    fn backends(&self) -> Vec<FakeBackend> {
        self.created.backends.lock().clone()
    }

    fn obs(&self) -> Vec<Observation> {
        self.recorder.observations()
    }
}

fn changing_frames(n: usize) -> Frame {
    Frame::filled(64, 48, [if n % 2 == 1 { 40 } else { 200 }; 3])
}

// ------------------------------------------------------------------ lifecycle
#[test]
fn test_emits_observations_on_worker_thread() {
    let h = Harness::new();
    h.worker.set_motion_gate(false, 2.0);
    h.worker.start();
    assert!(h.recorder.wait_for_observations(5));
    let main = thread::current().id();
    let backends = h.backends();
    assert_eq!(backends.len(), 1);
    let backend_threads = backends[0].threads();
    assert!(!backend_threads.contains(&main));
    assert!(!h.created.factory_threads.lock().contains(&main));
    assert_eq!(*h.created.factory_threads.lock(), backend_threads);
    assert_eq!(h.sources()[0].threads(), backend_threads);
    let stamps: Vec<f64> = h.obs().iter().map(|o| o.timestamp).collect();
    assert!(strictly_increasing(&stamps));
    assert_eq!(
        h.worker.backend_info(),
        Some(("fake".into(), "fake-1".into()))
    );
    assert!(h.worker.is_running());
}

#[test]
fn test_stop_joins_releases_and_closes() {
    let h = Harness::new();
    h.worker.start();
    assert!(h.recorder.wait_for_observations(2));
    h.worker.stop_within(TIMEOUT);
    assert!(!h.worker.is_running());
    let backend = &h.backends()[0];
    assert_eq!(backend.close_calls(), 1);
    assert!(!backend.threads().contains(&thread::current().id())); // closed on the worker thread
    let source = &h.sources()[0];
    assert!(!source.is_open());
    assert!(source.release_calls() >= 1);
    let count = h.recorder.count();
    h.worker.stop(); // idempotent
    thread::sleep(Duration::from_millis(20));
    assert_eq!(h.recorder.count(), count);
    assert!(!h.worker.stats().camera_open);
}

#[test]
fn test_stop_before_start_is_harmless() {
    let h = Harness::new();
    h.worker.stop();
    assert!(!h.worker.is_running());
    assert!(h.sources().is_empty());
}

#[test]
fn test_restart_after_stop() {
    let h = Harness::new();
    h.worker.start();
    h.worker.start(); // second start is a no-op
    assert!(h.recorder.wait_for_observations(1));
    h.worker.stop();
    h.worker.start();
    let n = h.recorder.count();
    assert!(h.recorder.wait_for(|r| r.count() > n));
    assert_eq!(h.backends().len(), 2);
}

#[test]
fn test_stop_from_the_worker_thread_does_not_deadlock() {
    let slot: Arc<OnceLock<Arc<VisionWorker>>> = Arc::default();
    let hook_slot = slot.clone();
    let backend = FakeBackend::with(BackendSpec {
        on_process: Some(Arc::new(move || {
            if let Some(worker) = hook_slot.get() {
                worker.stop();
            }
        })),
        ..BackendSpec::default()
    });
    let (tx, _recorder) = Recorder::new();
    let worker = Arc::new(VisionWorker::new(
        source_factory(&FakeSource::new(SourceSpec::default())),
        backend_factory(&backend),
        tx,
        fast_options(),
    ));
    worker.set_interval(0.005).unwrap();
    slot.set(worker.clone()).ok();
    worker.start();
    assert!(wait_until(|| !worker.is_running()));
    assert!(wait_until(|| backend.close_calls() == 1));
}

// -------------------------------------------------------------------- pacing
#[test]
fn test_interval_limits_frame_rate() {
    let h = Harness::new();
    h.worker.set_motion_gate(false, 2.0);
    h.worker.set_interval(0.04).unwrap();
    h.worker.start();
    let n = 6;
    assert!(h.recorder.wait_for_observations(n));
    let stamps: Vec<f64> = h.obs()[..n].iter().map(|o| o.timestamp).collect();
    // Frames follow a grid (slot, slot + interval, ...): a frame that starts late (a timer
    // overshoot) is followed by an earlier one, so a single gap can be short (Windows CI saw 63,
    // 16 and 31 ms), but n frames never take less than n - 1 intervals. The clock may tick in
    // ~16 ms steps on Windows: allow a tick at either end.
    assert!(stamps[n - 1] - stamps[0] >= (n - 1) as f64 * 0.04 - 2.0 * 0.016);
    assert_relative_eq!(h.worker.stats().target_fps, 25.0, max_relative = 1e-6);
}

#[test]
fn test_shorter_interval_wakes_the_loop() {
    let h = Harness::new();
    h.worker.set_interval(30.0).unwrap();
    h.worker.start();
    assert!(h.recorder.wait_for_observations(1));
    let started = Instant::now();
    h.worker.set_interval(0.005).unwrap();
    assert!(h.recorder.wait_for_within(2.0, |r| r.count() >= 2));
    assert!(started.elapsed().as_secs_f64() < 2.0);
}

#[test]
fn test_invalid_interval() {
    let (tx, _recorder) = Recorder::new();
    let worker = VisionWorker::new(
        source_factory(&FakeSource::new(SourceSpec::default())),
        backend_factory(&FakeBackend::new()),
        tx,
        WorkerOptions::default(),
    );
    let error = worker.set_interval(f64::NAN).unwrap_err();
    assert!(error.to_string().contains("interval"));
    worker.set_interval(-1.0).unwrap();
    assert_eq!(worker.stats().target_fps, 0.0); // clamped to 0 = unlimited
}

#[test]
fn test_frozen_clock_still_progresses() {
    let (tx, recorder) = Recorder::new();
    let backend = FakeBackend::new();
    let worker = VisionWorker::new(
        source_factory(&FakeSource::new(SourceSpec::default())),
        backend_factory(&backend),
        tx,
        WorkerOptions {
            clock: Arc::new(|| 42.0),
            ..WorkerOptions::default()
        },
    );
    worker.set_interval(0.005).unwrap();
    worker.set_motion_gate(false, 2.0);
    worker.start();
    assert!(recorder.wait_for_observations(4));
    worker.stop();
    let stamps = backend.timestamps();
    assert_eq!(stamps[0], 42.0);
    assert!(strictly_increasing(&stamps));
}

// ----------------------------------------------------------------- activity
#[test]
fn test_set_active_false_releases_camera() {
    let h = Harness::new();
    h.worker.start();
    assert!(h.recorder.wait_for_observations(2));
    h.worker.set_active(false);
    let first = h.sources()[0].clone();
    assert!(wait_until(
        || !first.is_open() && !h.worker.stats().camera_open
    ));
    assert!(first.release_calls() >= 1);
    let count = h.recorder.count();
    thread::sleep(Duration::from_millis(30));
    assert_eq!(h.recorder.count(), count); // idle: nothing emitted
    assert!(
        h.recorder
            .wait_for(|r| r.stats().iter().any(|s| !s.camera_open))
    );

    h.worker.set_active(true);
    assert!(h.recorder.wait_for(|r| r.count() > count));
    assert_eq!(h.sources().len(), 2); // a fresh source was opened
    assert!(h.sources()[1].is_open());
    assert_eq!(h.backends().len(), 1); // backend survives short pauses
}

#[test]
fn test_inactive_before_start_never_opens_camera() {
    let h = Harness::new();
    h.worker.set_active(false);
    h.worker.start();
    thread::sleep(Duration::from_millis(30));
    assert!(h.sources().is_empty());
    assert_eq!(h.recorder.count(), 0);
}

#[test]
fn test_long_idle_closes_backend() {
    let h = Harness::with(HarnessConfig {
        options: WorkerOptions {
            idle_backend_close_s: 0.02,
            ..fast_options()
        },
        ..HarnessConfig::default()
    });
    h.worker.start();
    assert!(h.recorder.wait_for_observations(1));
    h.worker.set_active(false);
    assert!(wait_until(|| h.backends()[0].close_calls() == 1));
    h.worker.set_active(true);
    assert!(wait_until(|| h.backends().len() == 2));
    let n = h.recorder.count();
    assert!(h.recorder.wait_for(|r| r.count() > n));
}

// --------------------------------------------------------------- motion gate
#[test]
fn test_motion_gate_emits_skipped_copies() {
    let h = Harness::new();
    h.worker.set_motion_gate(true, 2.0);
    h.worker.start();
    assert!(h.recorder.wait_for_observations(6));
    let obs = h.obs();
    let backend = &h.backends()[0];
    assert_eq!(backend.process_calls(), 1); // identical frames: analysed once (max_skip_s = 2 s)
    assert!(!obs[0].skipped);
    for copy in &obs[1..6] {
        assert!(copy.skipped);
        assert!(copy.timestamp > obs[0].timestamp);
        assert_eq!(copy.inference_ms, 0.0);
        assert_eq!(copy.face_count, obs[0].face_count);
        assert!(copy.features.is_some());
        assert_eq!(copy.features, obs[0].features);
    }
    let stamps: Vec<f64> = obs.iter().map(|o| o.timestamp).collect();
    assert!(strictly_increasing(&stamps));
    assert!(h.worker.stats().skip_ratio > 0.0);
}

#[test]
fn test_motion_gate_processes_changing_frames() {
    let h = Harness::with(HarnessConfig {
        source: SourceSpec {
            frame_fn: Arc::new(changing_frames),
            ..SourceSpec::default()
        },
        ..HarnessConfig::default()
    });
    h.worker.set_motion_gate(true, 2.0);
    h.worker.start();
    assert!(h.recorder.wait_for_observations(5));
    h.worker.stop();
    assert!(!h.obs().iter().any(|o| o.skipped));
}

#[test]
fn test_motion_gate_disabled_processes_everything() {
    let h = Harness::new();
    h.worker.set_motion_gate(false, 2.0);
    h.worker.start();
    assert!(h.recorder.wait_for_observations(5));
    h.worker.stop();
    let obs = h.obs();
    assert!(!obs.iter().any(|o| o.skipped));
    assert_eq!(h.backends()[0].process_calls(), obs.len());
}

#[test]
fn test_preview_bypasses_gate_and_delivers_frames() {
    let h = Harness::new();
    h.worker.set_motion_gate(true, 2.0);
    h.worker.set_preview(true);
    h.worker.start();
    assert!(h.recorder.wait_for(|r| r.previews() >= 3));
    h.worker.stop();
    assert!(!h.obs().iter().any(|o| o.skipped));
    let preview = h.recorder.preview(0);
    assert_eq!((preview.height, preview.width), (48, 64));
    assert_eq!(preview.pixel(0, 0), [1, 2, 3]); // annotated by the backend
}

#[test]
fn test_preview_off_stops_frames() {
    let h = Harness::new();
    h.worker.set_preview(true);
    h.worker.start();
    assert!(h.recorder.wait_for(|r| r.previews() >= 1));
    h.worker.set_preview(false);
    let n = h.recorder.count();
    assert!(h.recorder.wait_for_observations(n + 3));
    let previews = h.recorder.previews();
    thread::sleep(Duration::from_millis(20));
    assert_eq!(h.recorder.previews(), previews);
}

// ------------------------------------------------------------ failure handling
#[test]
fn test_open_failure_backs_off_and_recovers() {
    let allowed = Arc::new(AtomicBool::new(false));
    let attempts: Arc<Mutex<Vec<Instant>>> = Arc::default();
    let h = Harness::with(HarnessConfig {
        source: SourceSpec {
            can_open: {
                let (allowed, attempts) = (allowed.clone(), attempts.clone());
                Arc::new(move || {
                    attempts.lock().push(Instant::now());
                    allowed.load(Ordering::SeqCst)
                })
            },
            ..SourceSpec::default()
        },
        ..HarnessConfig::default()
    });
    h.worker.start();
    assert!(wait_until(|| attempts.lock().len() >= 4));
    let times = attempts.lock().clone();
    let gap = |i: usize| (times[i + 1] - times[i]).as_secs_f64();
    assert!(gap(0) >= FAST_BACKOFF[0] * 0.8);
    assert!(gap(2) >= FAST_BACKOFF[2] * 0.8); // delays grow instead of spinning
    assert_eq!(h.recorder.count(), 0);
    assert!(h.recorder.wait_for(|r| {
        r.stats()
            .iter()
            .any(|s| s.last_error.as_deref() == Some("fake camera is busy"))
    }));
    let stats = h.worker.stats();
    assert!(!stats.camera_open);
    assert_eq!(stats.last_error.as_deref(), Some("fake camera is busy"));
    assert!(h.sources().iter().all(|s| !s.is_open()));

    allowed.store(true, Ordering::SeqCst);
    h.worker.reconfigure(Some(h.make_source()), None); // cancels the back-off
    assert!(h.recorder.wait_for_observations(1));
    assert!(wait_until(|| h.worker.stats().last_error.is_none()));
    assert!(h.worker.stats().camera_open);
}

#[test]
fn test_keeps_retrying_at_the_longest_backoff() {
    let h = Harness::with(HarnessConfig {
        source: SourceSpec {
            can_open: Arc::new(|| false),
            ..SourceSpec::default()
        },
        options: WorkerOptions {
            backoff_s: vec![0.001, 0.002, 0.004],
            ..WorkerOptions::default()
        },
        ..HarnessConfig::default()
    });
    h.worker.start();
    // More attempts than back-off steps: the last delay repeats forever.
    assert!(wait_until(|| h.sources().len() >= 8));
    assert!(h.worker.is_running());
    assert!(h.worker.shared.backoff_step.load(Ordering::Relaxed) >= 7);
}

#[test]
fn test_source_factory_exception_is_reported() {
    let (tx, _recorder) = Recorder::new();
    let worker = VisionWorker::new(
        failing_source_factory("Video source not found: missing.mp4"),
        backend_factory(&FakeBackend::new()),
        tx,
        fast_options(),
    );
    worker.start();
    assert!(wait_until(|| worker.stats().last_error.is_some()));
    assert!(
        worker
            .stats()
            .last_error
            .unwrap_or_default()
            .contains("missing.mp4")
    );
    assert!(worker.is_running());
    worker.stop();
}

#[test]
fn test_backend_unavailable_keeps_camera_closed() {
    let (tx, recorder) = Recorder::new();
    let sources: Arc<Mutex<Vec<FakeSource>>> = Arc::default();
    let attempts = Arc::new(AtomicUsize::new(0));
    let counted = attempts.clone();
    let created = sources.clone();
    let worker = VisionWorker::new(
        Box::new(move || {
            let source = FakeSource::new(SourceSpec::default());
            created.lock().push(source.clone());
            Ok(Box::new(source) as Box<dyn FrameSource>)
        }),
        Box::new(move || {
            counted.fetch_add(1, Ordering::SeqCst);
            Err(BackendUnavailable("no model".into()).into())
        }),
        tx,
        fast_options(),
    );
    worker.start();
    assert!(wait_until(|| attempts.load(Ordering::SeqCst) >= 3));
    assert!(
        worker
            .stats()
            .last_error
            .unwrap_or_default()
            .contains("no model")
    );
    assert!(sources.lock().is_empty()); // never opened the camera without a backend
    assert_eq!(recorder.count(), 0);
    worker.stop();
}

#[test]
fn test_backend_recreated_after_repeated_failures() {
    let h = Harness::with(HarnessConfig {
        backend_fail_times: 3, // the first backend fails three times in a row
        ..HarnessConfig::default()
    });
    h.worker.set_motion_gate(false, 2.0);
    h.worker.start();
    assert!(h.recorder.wait_for_observations(2));
    let backends = h.backends();
    assert_eq!(backends.len(), 2);
    assert_eq!(backends[0].close_calls(), 1);
    assert_eq!(backends[0].process_calls(), 3);
    assert_eq!(h.worker.stats().extra["backend_errors"], json!(3));
}

#[test]
fn test_persistently_failing_backend_backs_off() {
    let backends: Arc<Mutex<Vec<FakeBackend>>> = Arc::default();
    let created = backends.clone();
    let (tx, _recorder) = Recorder::new();
    let worker = VisionWorker::new(
        source_factory(&FakeSource::new(SourceSpec::default())),
        Box::new(move || {
            let backend = FakeBackend::failing();
            created.lock().push(backend.clone());
            Ok(Box::new(backend) as Box<dyn VisionBackend>)
        }),
        tx,
        WorkerOptions {
            backoff_s: vec![0.03, 0.06, 0.12],
            ..WorkerOptions::default()
        },
    );
    worker.set_interval(0.0).unwrap();
    worker.set_motion_gate(false, 2.0);
    worker.start();
    thread::sleep(Duration::from_millis(150));
    worker.stop();
    // Without back-off this would recreate the backend hundreds of times.
    let backends = backends.lock();
    assert!((1..=5).contains(&backends.len()), "{}", backends.len());
    assert!(backends.iter().all(|b| b.close_calls() == 1));
}

#[test]
fn test_single_backend_failure_does_not_recreate() {
    let h = Harness::with(HarnessConfig {
        backend_fail_times: 1,
        ..HarnessConfig::default()
    });
    h.worker.set_motion_gate(false, 2.0);
    h.worker.start();
    assert!(h.recorder.wait_for_observations(3));
    assert_eq!(h.backends().len(), 1);
}

#[test]
fn test_source_death_triggers_reopen() {
    let h = Harness::with(HarnessConfig {
        source: SourceSpec {
            die_after: Some(3),
            ..SourceSpec::default()
        },
        ..HarnessConfig::default()
    });
    h.worker.set_motion_gate(false, 2.0);
    h.worker.start();
    assert!(wait_until(|| h.sources().len() >= 2));
    assert!(h.recorder.wait_for(|r| {
        r.stats()
            .iter()
            .any(|s| s.last_error.as_deref() == Some("fake camera unplugged"))
    }));
    assert!(h.sources()[0].release_calls() >= 1);
    assert!(h.recorder.wait_for_observations(5));
}

#[test]
fn test_a_dropped_receiver_does_not_kill_the_thread() {
    let (tx, recorder) = Recorder::new();
    drop(recorder);
    let backend = FakeBackend::new();
    let worker = VisionWorker::new(
        source_factory(&FakeSource::new(SourceSpec::default())),
        backend_factory(&backend),
        tx,
        fast_options(),
    );
    worker.set_interval(0.002).unwrap();
    worker.set_motion_gate(false, 2.0);
    worker.start();
    assert!(wait_until(|| backend.process_calls() >= 10));
    assert!(worker.is_running());
    worker.stop();
}

// ----------------------------------------------------------- configuration
#[test]
fn test_set_max_faces_applies_on_worker_thread() {
    let h = Harness::new();
    h.worker.start();
    assert!(h.recorder.wait_for_observations(1));
    let backend = h.backends()[0].clone();
    assert_eq!(backend.max_faces_calls(), [1]);
    h.worker.set_max_faces(2);
    assert!(wait_until(|| backend.max_faces_calls() == [1, 2]));
    assert!(!backend.threads().contains(&thread::current().id()));
}

#[test]
fn test_new_backend_gets_current_max_faces() {
    let h = Harness::new();
    h.worker.set_max_faces(2);
    h.worker.start();
    assert!(h.recorder.wait_for_observations(1));
    assert_eq!(h.backends()[0].max_faces_calls(), [2]);
}

#[test]
fn test_reconfigure_swaps_source_and_backend() {
    let h = Harness::new();
    h.worker.set_motion_gate(false, 2.0);
    h.worker.start();
    assert!(h.recorder.wait_for_observations(2));
    let (old_source, old_backend) = (h.sources()[0].clone(), h.backends()[0].clone());

    let new_source = FakeSource::with_frames(|_| Frame::filled(32, 24, [0; 3]));
    let new_backend = FakeBackend::new();
    h.worker
        .reconfigure(Some(source_factory(&new_source)), None);
    assert!(wait_until(|| new_source.reads() > 0));
    assert!(!old_source.is_open());
    assert_eq!(old_backend.close_calls(), 0); // only the camera changed

    h.worker
        .reconfigure(None, Some(backend_factory(&new_backend)));
    assert!(wait_until(|| new_backend.process_calls() > 0));
    assert_eq!(old_backend.close_calls(), 1);
    assert_eq!(h.obs().last().unwrap().frame_size, (32, 24));
    h.worker.reconfigure(None, None); // nothing to do
}

#[test]
fn test_reconfigure_while_inactive() {
    let h = Harness::new();
    h.worker.start();
    assert!(h.recorder.wait_for_observations(1));
    h.worker.set_active(false);
    let new_backend = FakeBackend::new();
    h.worker
        .reconfigure(None, Some(backend_factory(&new_backend)));
    assert!(wait_until(|| h.backends()[0].close_calls() == 1));
    assert_eq!(new_backend.process_calls(), 0);
    h.worker.set_active(true);
    assert!(wait_until(|| new_backend.process_calls() > 0));
}

#[test]
fn test_stats_snapshot() {
    let h = Harness::new();
    h.worker.set_interval(0.01).unwrap();
    h.worker.set_motion_gate(false, 2.0);
    h.worker.start();
    assert!(h.recorder.wait_for_observations(6));
    let mut stats = h.worker.stats();
    assert!(stats.camera_open);
    assert!(stats.last_error.is_none());
    assert!(stats.fps > 0.0);
    assert_relative_eq!(stats.target_fps, 100.0, max_relative = 1e-6);
    assert_relative_eq!(stats.inference_ms, 1.5, max_relative = 1e-6);
    assert_eq!(stats.extra["backend"], json!("fake"));
    assert_eq!(stats.extra["feature_version"], json!("fake-1"));
    assert!(stats.extra["frames"].as_u64().unwrap() >= 6);
    assert_eq!(stats.extra["camera_opens"], json!(1));
    stats.extra.insert("frames".into(), json!(-1)); // snapshots are independent copies
    assert_ne!(h.worker.stats().extra["frames"], json!(-1));
    // The camera opening is published immediately.
    assert!(h.recorder.stats().iter().any(|s| s.camera_open));
}

#[test]
fn test_stats_report_device_and_frame_size() {
    let h = Harness::with(HarnessConfig {
        source: SourceSpec {
            device: Some("/dev/v4l/by-id/usb-cam-video-index0".into()),
            ..SourceSpec::default()
        },
        ..HarnessConfig::default()
    });
    h.worker.set_motion_gate(false, 2.0);
    h.worker.start();
    assert!(h.recorder.wait_for_observations(2));
    let extra = h.worker.stats().extra;
    h.worker.stop();
    assert_eq!(
        extra["device"],
        json!("/dev/v4l/by-id/usb-cam-video-index0")
    );
    assert_eq!(extra["frame_size"], json!([64, 48]));
    assert_eq!(extra["backend_failing"], json!(false));
}

// ------------------------------------------------- deactivation while starting
#[test]
fn test_camera_stays_off_when_deactivated_during_backend_creation() {
    // Privacy mode pressed while the backend loads: the camera must never open.
    let (creating, release) = (Event::default(), Event::default());
    let sources: Arc<Mutex<Vec<FakeSource>>> = Arc::default();
    let (tx, _recorder) = Recorder::new();
    let slow_backend = {
        let (creating, release) = (creating.clone(), release.clone());
        Box::new(move || {
            creating.set();
            release.wait(TIMEOUT);
            Ok(Box::new(FakeBackend::new()) as Box<dyn VisionBackend>)
        })
    };
    let created = sources.clone();
    let worker = VisionWorker::new(
        Box::new(move || {
            let source = FakeSource::new(SourceSpec::default());
            created.lock().push(source.clone());
            Ok(Box::new(source) as Box<dyn FrameSource>)
        }),
        slow_backend,
        tx,
        WorkerOptions::default(),
    );
    worker.start();
    assert!(creating.wait(TIMEOUT));
    worker.set_active(false);
    release.set();
    thread::sleep(Duration::from_millis(50));
    assert!(sources.lock().is_empty());
    worker.set_active(true);
    assert!(wait_until(|| sources
        .lock()
        .first()
        .is_some_and(FakeSource::is_open)));
    release.set();
    worker.stop();
}

#[test]
fn test_stop_during_backend_creation_never_opens_the_camera() {
    let (creating, release) = (Event::default(), Event::default());
    let sources: Arc<Mutex<Vec<FakeSource>>> = Arc::default();
    let (tx, _recorder) = Recorder::new();
    let slow_backend = {
        let (creating, release) = (creating.clone(), release.clone());
        Box::new(move || {
            creating.set();
            release.wait(TIMEOUT);
            Ok(Box::new(FakeBackend::new()) as Box<dyn VisionBackend>)
        })
    };
    let created = sources.clone();
    let worker = Arc::new(VisionWorker::new(
        Box::new(move || {
            let source = FakeSource::new(SourceSpec::default());
            created.lock().push(source.clone());
            Ok(Box::new(source) as Box<dyn FrameSource>)
        }),
        slow_backend,
        tx,
        WorkerOptions::default(),
    ));
    worker.start();
    assert!(creating.wait(TIMEOUT));
    let stopper = {
        let worker = worker.clone();
        thread::spawn(move || worker.stop_within(TIMEOUT))
    };
    thread::sleep(Duration::from_millis(20));
    release.set();
    stopper.join().unwrap();
    assert!(!worker.is_running());
    assert!(sources.lock().is_empty());
}

// ------------------------------------------------------- failing backend
#[test]
fn test_backend_that_keeps_failing_is_reported_and_releases_the_camera() {
    // A backend that fails on every frame must not look like healthy tracking.
    let h = Harness::with(HarnessConfig {
        backends_always_fail: true,
        options: WorkerOptions {
            backoff_s: vec![0.05],
            ..WorkerOptions::default()
        },
        ..HarnessConfig::default()
    });
    h.worker.set_motion_gate(false, 2.0);
    h.worker.start();
    assert!(wait_until(
        || h.worker.stats().extra.get("backend_failing") == Some(&json!(true))
    ));
    assert!(h.recorder.wait_for(|r| r.stats().iter().any(|s| {
        let error = s.last_error.clone().unwrap_or_default();
        error.starts_with("Vision backend keeps failing")
            && error.contains("simulated inference failure")
    })));
    // The camera is released while the worker backs off.
    assert!(h.recorder.wait_for(|r| {
        r.stats()
            .iter()
            .any(|s| !s.camera_open && s.extra.get("backend_failing") == Some(&json!(true)))
    }));
    assert!(h.sources()[0].release_calls() >= 1);
    // The error is sticky: reopening the camera does not clear it.
    assert!(wait_until(|| h.sources().len() >= 2));
    let reopened: Vec<WorkerStats> = h
        .recorder
        .stats()
        .into_iter()
        .filter(|s| s.camera_open && s.extra["camera_opens"].as_u64().unwrap_or(0) > 1)
        .collect();
    assert!(!reopened.is_empty());
    assert!(reopened.iter().all(|s| s.last_error.is_some()));
    assert_eq!(h.recorder.count(), 0);
}

#[test]
fn test_backend_failure_clears_once_frames_are_analysed_again() {
    let h = Harness::with(HarnessConfig {
        backend_fail_times: 3, // the first backend fails three times in a row
        options: WorkerOptions {
            backoff_s: vec![0.01],
            ..WorkerOptions::default()
        },
        ..HarnessConfig::default()
    });
    h.worker.set_motion_gate(false, 2.0);
    h.worker.start();
    assert!(h.recorder.wait_for_observations(2));
    assert!(wait_until(|| h.worker.stats().last_error.is_none()));
    assert_eq!(h.worker.stats().extra["backend_failing"], json!(false));
    assert!(
        h.recorder
            .stats()
            .iter()
            .any(|s| s.extra.get("backend_failing") == Some(&json!(true)))
    );
}

// ------------------------------------------------------------- blinks and gate
fn run_worker(
    spec: BackendSpec,
    frame_fn: impl Fn(usize) -> Frame + Send + Sync + 'static,
    n: usize,
    gate: bool,
) -> (FakeBackend, Vec<Observation>) {
    let (tx, recorder) = Recorder::new();
    let backend = FakeBackend::with(spec);
    let worker = VisionWorker::new(
        source_factory(&FakeSource::with_frames(frame_fn)),
        backend_factory(&backend),
        tx,
        WorkerOptions::default(),
    );
    worker.set_interval(0.002).unwrap();
    worker.set_motion_gate(gate, 2.0);
    worker.start();
    assert!(recorder.wait_for_observations(n));
    worker.stop();
    (backend, recorder.observations())
}

fn flat(_n: usize) -> Frame {
    Frame::filled(64, 48, [100; 3])
}

#[test]
fn test_gate_never_holds_a_blink() {
    // After an analysed blink the next frame is analysed even if it looks the same.
    let spec = BackendSpec {
        blinks: 3,
        ..BackendSpec::default()
    };
    let (_, obs) = run_worker(spec, flat, 8, true);
    assert_eq!(
        obs[..4].iter().map(|o| o.skipped).collect::<Vec<_>>(),
        [false; 4]
    );
    assert!(!obs[3].blink);
    assert!(obs[4..8].iter().all(|o| o.skipped)); // eyes open: the gate holds again
}

#[test]
fn test_gate_never_holds_an_unsettled_observation() {
    // Features from a crop still catching up must not be repeated. The user sits still right
    // after a posture shift, so the picture stops changing while the backend's tracking crop is
    // still off; the gate used to repeat that biased observation for max_skip_s.
    let spec = BackendSpec {
        unsettled: Some(3),
        ..BackendSpec::default()
    };
    let (backend, obs) = run_worker(spec, flat, 8, true);
    assert_eq!(
        obs[..4].iter().map(|o| o.skipped).collect::<Vec<_>>(),
        [false; 4]
    );
    assert!(obs[4..8].iter().all(|o| o.skipped)); // settled: the gate holds again
    assert_eq!(backend.process_calls(), 4);
}

#[test]
fn test_backend_is_reset_whenever_the_camera_is_released() {
    // A tracked face must not outlive a pause: the next picture may differ.
    let h = Harness::new();
    h.worker.set_motion_gate(false, 2.0);
    h.worker.start();
    assert!(h.recorder.wait_for_observations(2));
    let backend = h.backends()[0].clone();
    let resets = backend.reset_calls();
    h.worker.set_active(false);
    assert!(wait_until(|| backend.reset_calls() > resets));
    assert!(wait_until(|| !h.worker.stats().camera_open));
    h.worker.set_active(true);
    let n = h.recorder.count();
    assert!(h.recorder.wait_for_observations(n + 2));
    assert_eq!(h.backends().len(), 1); // the same backend, but forgetful
    assert!(!backend.threads().contains(&thread::current().id()));
    let resets = backend.reset_calls();
    h.worker.reconfigure(Some(h.make_source()), None); // another camera
    assert!(wait_until(|| backend.reset_calls() > resets));
}

#[test]
fn test_backend_is_reset_when_the_camera_stops_delivering() {
    let h = Harness::with(HarnessConfig {
        source: SourceSpec {
            die_after: Some(2),
            ..SourceSpec::default()
        },
        ..HarnessConfig::default()
    });
    h.worker.set_motion_gate(false, 2.0);
    h.worker.start();
    assert!(wait_until(|| h.sources().len() >= 2)); // died, then reopened
    assert!(h.backends()[0].reset_calls() >= 1);
}

#[test]
fn test_a_failing_backend_reset_does_not_stop_the_worker() {
    let (tx, recorder) = Recorder::new();
    let backend = FakeBackend::with(BackendSpec {
        broken_reset: true,
        ..BackendSpec::default()
    });
    let worker = VisionWorker::new(
        source_factory(&FakeSource::new(SourceSpec::default())),
        backend_factory(&backend),
        tx,
        WorkerOptions::default(),
    );
    worker.set_interval(0.002).unwrap();
    worker.set_motion_gate(false, 2.0);
    worker.start();
    assert!(recorder.wait_for_observations(2));
    worker.set_active(false);
    assert!(wait_until(|| backend.reset_calls() >= 1));
    worker.set_active(true);
    let n = recorder.count();
    assert!(recorder.wait_for_observations(n + 2));
    assert!(worker.is_running());
    worker.stop();
}

// ------------------------------------------------------------------ blindness
fn dark(_n: usize) -> Frame {
    Frame::filled(64, 48, [3; 3])
}

/// A random 8x6 pattern blown up to 64x48 (the seeded `rng` of the Python test).
fn textured(_n: usize) -> Frame {
    let mut state = 5u64;
    let mut cells = [[0u8; 3]; 48];
    for cell in &mut cells {
        for channel in cell.iter_mut() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *channel = (state >> 56) as u8;
        }
    }
    let mut frame = Frame::filled(64, 48, [0; 3]);
    for y in 0..48 {
        for x in 0..64 {
            frame.set_pixel(x, y, cells[(y / 8) * 8 + x / 8]);
        }
    }
    frame
}

fn no_face() -> BackendSpec {
    BackendSpec {
        face_count: 0,
        ..BackendSpec::default()
    }
}

#[test]
fn test_dark_featureless_frames_without_a_face_are_blind() {
    for gate in [true, false] {
        let (_, obs) = run_worker(no_face(), dark, 4, gate);
        assert!(obs.iter().all(|o| o.blind), "gate {gate}");
        assert!(!obs.iter().any(Observation::face_present), "gate {gate}");
        if gate {
            assert!(obs.iter().any(|o| o.skipped)); // skipped copies keep the flag
        }
    }
}

#[test]
fn test_ordinary_empty_scene_is_not_blind() {
    let (_, obs) = run_worker(no_face(), textured, 3, true);
    assert!(!obs.iter().any(|o| o.blind));
}

#[test]
fn test_a_face_in_the_dark_is_not_blind() {
    let (_, obs) = run_worker(BackendSpec::default(), dark, 3, true);
    assert!(!obs.iter().any(|o| o.blind));
    assert!(obs.iter().all(Observation::face_present));
}

#[test]
fn test_stats_published_at_most_once_per_second() {
    let h = Harness::new();
    h.worker.set_interval(0.002).unwrap();
    h.worker.set_motion_gate(false, 2.0);
    h.worker.start();
    assert!(h.recorder.wait_for_observations(12));
    // One forced publish when the camera opened; routine ones are 1 s apart.
    assert!(h.recorder.stats().len() <= 2);
}

// ------------------------------------------------------------ review hardening
#[test]
fn test_switching_on_forgets_the_last_error_without_publishing() {
    // A camera that fails the same way after a pause is reported again.
    let (tx, recorder) = Recorder::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let release = Event::default();
    let message = "Camera 0 could not be opened";
    let (counted, held) = (calls.clone(), release.clone());
    let worker = VisionWorker::new(
        Box::new(move || {
            if counted.fetch_add(1, Ordering::SeqCst) > 0 {
                held.wait(TIMEOUT); // hold the next attempt until the test is ready
            }
            Err(CameraError(message.into()).into())
        }),
        backend_factory(&FakeBackend::new()),
        tx,
        fast_options(),
    );
    worker.start();
    assert!(wait_until(
        || worker.stats().last_error.as_deref() == Some(message)
    ));
    worker.set_active(false);
    let published = recorder.stats().len();
    worker.set_active(true);
    assert!(worker.stats().last_error.is_none()); // forgotten at once ...
    assert_eq!(recorder.stats().len(), published); // ... without a "healthy" report
    release.set();
    // The same failure again is news again.
    assert!(recorder.wait_for(|r| {
        r.stats()
            .iter()
            .filter(|s| s.last_error.as_deref() == Some(message))
            .count()
            >= 2
    }));
    release.set();
    worker.stop();
}

#[test]
fn test_reconfigured_backend_identity_is_unknown_until_it_exists() {
    let h = Harness::new();
    h.worker.start();
    let fake = Some(("fake".to_string(), "fake-1".to_string()));
    assert!(wait_until(|| h.worker.backend_info() == fake));
    let release = Event::default();
    let held = release.clone();
    h.worker.reconfigure(
        None,
        Some(Box::new(move || {
            held.wait(TIMEOUT);
            Ok(Box::new(FakeBackend::with(BackendSpec {
                name: "new",
                feature_version: "new-1",
                ..BackendSpec::default()
            })) as Box<dyn VisionBackend>)
        })),
    );
    // The old backend's identity must not pass for the new one's meanwhile.
    assert_eq!(h.worker.backend_info(), None);
    release.set();
    let new = Some(("new".to_string(), "new-1".to_string()));
    assert!(wait_until(|| h.worker.backend_info() == new));
    h.worker.reconfigure(Some(h.make_source()), None); // camera only
    assert_eq!(h.worker.backend_info(), new);
}

#[test]
fn test_the_worker_wakes_with_the_last_error_already_forgotten() {
    // Regression: set_active(true) forgot the error only after waking the thread. A camera
    // failing at once with the old message was then compared with the old error, found not to
    // be news (nothing published), and erased right after: the failure was lost until the next
    // retry.
    let (tx, recorder) = Recorder::new();
    let message = "Camera 0 could not be opened";
    // The error the worker compares with, per attempt.
    let seen: Arc<Mutex<Vec<Option<String>>>> = Arc::default();
    let holder: Arc<OnceLock<Weak<Shared>>> = Arc::default();
    let (recorded, shared) = (seen.clone(), holder.clone());
    let worker = VisionWorker::new(
        Box::new(move || {
            let shared = shared.get().and_then(Weak::upgrade).expect("the worker");
            recorded
                .lock()
                .push(shared.stats.lock().stats.last_error.clone());
            Err(CameraError(message.into()).into())
        }),
        backend_factory(&FakeBackend::new()),
        tx,
        WorkerOptions {
            backoff_s: vec![TIMEOUT * 10.0], // one attempt per activation
            ..WorkerOptions::default()
        },
    );
    holder.set(Arc::downgrade(&worker.shared)).unwrap();
    worker.start();
    assert!(wait_until(
        || worker.stats().last_error.as_deref() == Some(message)
    ));
    worker.set_active(false);
    worker.set_active(true);
    assert!(wait_until(|| seen.lock().len() >= 2));
    assert_eq!(seen.lock()[1], None);
    assert!(recorder.wait_for(|r| {
        r.stats()
            .iter()
            .filter(|s| s.last_error.as_deref() == Some(message))
            .count()
            >= 2
    }));
    worker.stop();
}

#[test]
fn test_a_backend_built_by_a_replaced_factory_does_not_report_its_identity() {
    // A creation still running when reconfigure() replaced its factory.
    let h = Harness::new();
    let (old_entered, old_release) = (Event::default(), Event::default());
    let (new_entered, new_release) = (Event::default(), Event::default());
    let slow = |entered: &Event, release: &Event, name, version| -> BackendFactory {
        let (entered, release) = (entered.clone(), release.clone());
        Box::new(move || {
            entered.set();
            release.wait(TIMEOUT);
            Ok(Box::new(FakeBackend::with(BackendSpec {
                name,
                feature_version: version,
                ..BackendSpec::default()
            })) as Box<dyn VisionBackend>)
        })
    };
    h.worker.start();
    let fake = Some(("fake".to_string(), "fake-1".to_string()));
    assert!(wait_until(|| h.worker.backend_info() == fake));
    h.worker
        .reconfigure(None, Some(slow(&old_entered, &old_release, "old", "old-1")));
    assert!(old_entered.wait(TIMEOUT));
    // While "old" is built:
    h.worker
        .reconfigure(None, Some(slow(&new_entered, &new_release, "new", "new-1")));
    old_release.set();
    assert!(new_entered.wait(TIMEOUT));
    // "old" was built after the second reconfigure: it is not what comes next.
    assert_eq!(h.worker.backend_info(), None);
    new_release.set();
    let new = Some(("new".to_string(), "new-1".to_string()));
    assert!(wait_until(|| h.worker.backend_info() == new));
}
