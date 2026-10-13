//! Talk to pane providers off the GUI thread.
//!
//! Providers run command-line tools that can take tens of milliseconds, or hang
//! until their timeout, so [`PaneWorker`] runs them on one background thread
//! (like the controller's screen-lock thread) and hands the results back as
//! [`PaneEvent`]s, which the owner collects with [`PaneWorker::drain_events`].
//!
//! Requests do not pile up: a new detection request replaces one that has not
//! started yet, and a focus request goes before any detection. A provider whose
//! calls fail (return an error, or take longer than the call timeout) [`FAILURE_LIMIT`]
//! times in a row is switched off for the rest of the session.
//!
//! Providers may hold system resources (UI Automation COM objects, which belong
//! to the thread that made them). Their `close()`, where they have one, runs on
//! the worker's thread: for the old providers when [`PaneWorker::set_registry`]
//! replaces them, and for the current ones when the thread ends.
//!
//! `PaneWorkerOptions::synchronous` runs every request on the calling thread instead, for tests.

use std::collections::{HashMap, HashSet};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};
use parking_lot::{Condvar, Mutex};

use crate::types::{AppIdentity, WindowHandle, WindowRef};

use super::registry::{PaneRegistry, panic_message};
use super::types::{Pane, PaneError, PaneId, PaneProvider, PaneSnapshot};

/// Consecutive failures after which a provider is off for the session.
pub const FAILURE_LIMIT: u32 = 3;
/// A provider call taking longer than this counts as failed (its result is dropped).
pub const CALL_TIMEOUT_S: f64 = 2.5;
/// How long [`PaneWorker::stop`] waits for a call in progress.
pub const JOIN_TIMEOUT_S: f64 = 3.0;

/// Seconds on a monotonic clock (Python's `time.monotonic`).
pub fn monotonic() -> f64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64()
}

/// Answer to [`PaneWorker::request_detect`].
#[derive(Debug, Clone, PartialEq)]
pub struct DetectResult {
    pub window_handle: WindowHandle,
    /// The panes found (`taken_at` stamped with the worker's clock), or `None`.
    pub snapshot: Option<PaneSnapshot>,
    pub request_id: u64,
}

/// Answer to [`PaneWorker::request_focus`].
#[derive(Debug, Clone, PartialEq)]
pub struct FocusResult {
    pub window_handle: WindowHandle,
    pub pane_id: PaneId,
    pub ok: bool,
}

/// What the worker reports (the Python signals `detected` and `focused`).
#[derive(Debug, Clone, PartialEq)]
pub enum PaneEvent {
    /// A detection finished.
    Detected(DetectResult),
    /// A focus request finished.
    Focused(FocusResult),
}

struct DetectJob {
    window: WindowRef,
    app: AppIdentity,
    request_id: u64,
}

struct FocusJob {
    window: WindowRef,
    pane: Pane,
}

/// The settings of [`PaneWorker::new`].
pub struct PaneWorkerOptions {
    /// Stamps the snapshots (`taken_at`); [`monotonic`] by default.
    pub clock: Arc<dyn Fn() -> f64 + Send + Sync>,
    /// Run every request on the calling thread (tests).
    pub synchronous: bool,
    pub call_timeout_s: f64,
}

impl Default for PaneWorkerOptions {
    fn default() -> Self {
        Self {
            clock: Arc::new(monotonic),
            synchronous: false,
            call_timeout_s: CALL_TIMEOUT_S,
        }
    }
}

#[derive(Default)]
struct State {
    detect: Option<DetectJob>,
    focus: Option<FocusJob>,
    stopping: bool,
    /// Registries given to `set_registry` that the thread has not installed yet, oldest first.
    incoming: Vec<PaneRegistry>,
    failures: HashMap<String, u32>,
    disabled: HashSet<String>,
    next_id: u64,
}

struct Shared {
    state: Mutex<State>,
    cond: Condvar,
    /// The registry in use. The thread holds the lock for as long as it runs; in synchronous
    /// mode it is locked for the duration of a request.
    registry: Mutex<PaneRegistry>,
    events: Sender<PaneEvent>,
    clock: Arc<dyn Fn() -> f64 + Send + Sync>,
    synchronous: bool,
    call_timeout_s: f64,
}

/// Single background thread for pane detection and focus requests.
pub struct PaneWorker {
    shared: Arc<Shared>,
    events: Receiver<PaneEvent>,
    thread: Option<JoinHandle<()>>,
}

impl PaneWorker {
    pub fn new(registry: PaneRegistry, options: PaneWorkerOptions) -> Self {
        let (events_tx, events) = crossbeam_channel::unbounded();
        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            cond: Condvar::new(),
            registry: Mutex::new(registry),
            events: events_tx,
            clock: options.clock,
            synchronous: options.synchronous,
            call_timeout_s: options.call_timeout_s,
        });
        Self {
            shared,
            events,
            thread: None,
        }
    }

    // ------------------------------------------------------------ lifecycle
    pub fn running(&self) -> bool {
        self.thread.as_ref().is_some_and(|t| !t.is_finished())
    }

    /// Start the thread (a no-op when synchronous or already running).
    pub fn start(&mut self) {
        if self.shared.synchronous || self.running() {
            return;
        }
        {
            let mut state = self.shared.state.lock();
            state.stopping = false;
            state.detect = None; // asked while stopped: stale by now
            state.focus = None;
        }
        let shared = self.shared.clone();
        let spawned = thread::Builder::new()
            .name("eye-tracker-panes".into())
            .spawn(move || shared.run());
        match spawned {
            Ok(handle) => self.thread = Some(handle),
            Err(e) => log::warn!("The pane worker thread could not be started: {e}"),
        }
    }

    /// Drop pending requests and tell the thread to end, without waiting.
    ///
    /// Nothing is delivered after this. The thread finishes a provider call in progress, closes
    /// the providers and ends; without a running thread (or when synchronous) the providers are
    /// closed here.
    pub fn request_stop(&mut self) {
        // Asked before the thread is told to end: a thread seen alive closes the providers itself.
        let alive = self.running();
        {
            let mut state = self.shared.state.lock();
            state.stopping = true;
            state.detect = None;
            state.focus = None;
            self.shared.cond.notify_all();
        }
        if !alive {
            // Locked only if no thread (a hung one that `stop` gave up on) still uses it.
            if let Some(mut registry) = self.shared.registry.try_lock() {
                self.shared.install_incoming(&mut registry);
                registry.close();
            }
        }
    }

    /// [`request_stop`](Self::request_stop), then wait up to `timeout` seconds for the thread to end.
    pub fn stop(&mut self, timeout: f64) {
        self.request_stop();
        let Some(handle) = self.thread.take() else {
            return;
        };
        if handle.thread().id() == thread::current().id() {
            return;
        }
        let deadline = Instant::now() + Duration::from_secs_f64(timeout.clamp(0.0, 3600.0));
        while !handle.is_finished() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(2));
        }
        if handle.is_finished() {
            let _ = handle.join();
        } else {
            log::warn!("A pane provider call did not finish while stopping");
        }
    }

    /// Use other providers from the next request on (settings changed).
    ///
    /// The replaced providers are closed on the worker's thread before its next request (at once
    /// when synchronous).
    pub fn set_registry(&self, registry: PaneRegistry) {
        {
            let mut state = self.shared.state.lock();
            state.detect = None;
            state.incoming.push(registry);
            self.shared.cond.notify_all();
        }
        if self.shared.synchronous {
            let mut registry = self.shared.registry.lock();
            self.shared.install_incoming(&mut registry);
        }
    }

    pub fn disabled_providers(&self) -> HashSet<String> {
        self.shared.state.lock().disabled.clone()
    }

    /// What the worker has reported since the last call.
    pub fn drain_events(&self) -> Vec<PaneEvent> {
        self.events.try_iter().collect()
    }

    // ------------------------------------------------------------- requests
    /// Find the panes of `window`; the answer comes as [`PaneEvent::Detected`].
    ///
    /// Replaces a detection request that has not started yet. Returns the request's id (echoed
    /// in [`DetectResult`]).
    pub fn request_detect(&self, window: &WindowRef, app: &AppIdentity) -> u64 {
        let job = {
            let mut state = self.shared.state.lock();
            state.next_id += 1;
            let job = DetectJob {
                window: window.clone(),
                app: app.clone(),
                request_id: state.next_id,
            };
            if !self.shared.synchronous {
                let id = job.request_id;
                state.detect = Some(job);
                self.shared.cond.notify_all();
                return id;
            }
            job
        };
        let id = job.request_id;
        let mut registry = self.shared.registry.lock();
        self.shared.do_detect(&mut registry, job);
        id
    }

    /// Focus `pane`; the answer comes as [`PaneEvent::Focused`].
    pub fn request_focus(&self, window: &WindowRef, pane: &Pane) {
        let job = FocusJob {
            window: window.clone(),
            pane: pane.clone(),
        };
        if self.shared.synchronous {
            let mut registry = self.shared.registry.lock();
            self.shared.do_focus(&mut registry, job);
            return;
        }
        self.shared.state.lock().focus = Some(job);
        self.shared.cond.notify_all();
    }
}

impl Drop for PaneWorker {
    fn drop(&mut self) {
        if self.running() {
            self.stop(JOIN_TIMEOUT_S);
        }
    }
}

enum Job {
    Detect(DetectJob),
    Focus(FocusJob),
}

impl Shared {
    // --------------------------------------------------------------- thread
    fn run(&self) {
        let mut registry = self.registry.lock();
        self.serve(&mut registry);
        // On this thread: UI Automation objects belong to the thread that made them.
        self.install_incoming(&mut registry);
        registry.close();
    }

    fn serve(&self, registry: &mut PaneRegistry) {
        loop {
            self.install_incoming(registry);
            let job = {
                let mut state = self.state.lock();
                while !state.stopping
                    && state.detect.is_none()
                    && state.focus.is_none()
                    && state.incoming.is_empty()
                {
                    self.cond.wait(&mut state);
                }
                if state.stopping {
                    return;
                }
                if !state.incoming.is_empty() {
                    continue; // install them first
                }
                if let Some(focus) = state.focus.take() {
                    Job::Focus(focus)
                } else if let Some(detect) = state.detect.take() {
                    Job::Detect(detect)
                } else {
                    continue;
                }
            };
            // Never let the thread die.
            let outcome = catch_unwind(AssertUnwindSafe(|| match job {
                Job::Focus(job) => self.do_focus(registry, job),
                Job::Detect(job) => self.do_detect(registry, job),
            }));
            if let Err(payload) = outcome {
                log::warn!("Pane request failed: {}", panic_message(payload.as_ref()));
            }
        }
    }

    /// Switch to the newest registry given to `set_registry` and close the replaced ones.
    fn install_incoming(&self, registry: &mut PaneRegistry) {
        let incoming = std::mem::take(&mut self.state.lock().incoming);
        for new in incoming {
            let mut old = std::mem::replace(registry, new);
            old.close();
        }
    }

    fn do_detect(&self, registry: &mut PaneRegistry, job: DetectJob) {
        let mut snapshot = None;
        for provider in registry.providers_for(Some(&job.app)) {
            if self.is_disabled(provider.name()) {
                continue;
            }
            let found = self.call(provider, |p| p.detect(&job.window, &job.app));
            if let Some(Some(mut found)) = found
                && found.panes.len() >= 2
            {
                found.window_handle = job.window.handle.clone();
                found.taken_at = (self.clock)();
                snapshot = Some(found);
                break;
            }
        }
        self.emit(PaneEvent::Detected(DetectResult {
            window_handle: job.window.handle,
            snapshot,
            request_id: job.request_id,
        }));
    }

    fn do_focus(&self, registry: &mut PaneRegistry, job: FocusJob) {
        let mut ok = false;
        if let Some(provider) = registry.provider(&job.pane.provider)
            && !self.is_disabled(provider.name())
        {
            ok = self.call(provider, |p| p.focus(&job.window, &job.pane)) == Some(true);
        }
        self.emit(PaneEvent::Focused(FocusResult {
            window_handle: job.window.handle,
            pane_id: job.pane.id,
            ok,
        }));
    }

    /// Run one provider call, counting failures; `None` when it failed.
    fn call<T>(
        &self,
        provider: &mut dyn PaneProvider,
        call: impl FnOnce(&mut dyn PaneProvider) -> Result<T, PaneError>,
    ) -> Option<T> {
        let name = provider.name().to_string();
        let started = Instant::now();
        let result = match catch_unwind(AssertUnwindSafe(|| call(provider))) {
            Ok(Ok(result)) => result,
            Ok(Err(error)) => {
                self.failed(&name, &error.0);
                return None;
            }
            Err(payload) => {
                self.failed(&name, &panic_message(payload.as_ref()));
                return None;
            }
        };
        let elapsed = started.elapsed().as_secs_f64();
        if elapsed > self.call_timeout_s {
            self.failed(&name, &format!("took {elapsed:.1} s"));
            return None;
        }
        self.state.lock().failures.remove(&name);
        Some(result)
    }

    fn failed(&self, name: &str, why: &str) {
        let count = {
            let mut state = self.state.lock();
            let count = state.failures.get(name).copied().unwrap_or(0) + 1;
            state.failures.insert(name.to_string(), count);
            if count >= FAILURE_LIMIT {
                state.disabled.insert(name.to_string());
            }
            count
        };
        log::debug!("Pane provider {name} failed ({count} in a row): {why}");
        if count == FAILURE_LIMIT {
            log::info!("Pane provider {name} keeps failing; off until the next start");
        }
    }

    fn is_disabled(&self, name: &str) -> bool {
        self.state.lock().disabled.contains(name)
    }

    fn emit(&self, event: PaneEvent) {
        if self.state.lock().stopping && !self.synchronous {
            return;
        }
        let _ = self.events.send(event); // the worker was dropped: nobody listens
    }
}

#[cfg(test)]
#[path = "worker_tests.rs"]
mod tests;
