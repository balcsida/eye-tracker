// Ported from tests/test_panes_worker.py (the pane worker thread; providers are fakes).
//
// Changed: there is no Qt event loop. Results are collected with `drain_events()`, which the
// `wait` helper polls where the Python tests ran `processEvents()`. test_thread_answers_and_stops_cleanly
// no longer checks that results are delivered on the main thread (they are only ever drained by the
// caller). The slow-call test sleeps instead of patching `time.monotonic`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use super::*;
use crate::testing::panes_fakes::{
    Call, FakeRunner, RecordingProvider, current_thread_name, two_panes,
};
use crate::types::{Rect, WindowHandle};

use super::super::providers::command::CommandRunner;

fn app() -> AppIdentity {
    AppIdentity {
        process: "kitty".into(),
        app_id: "kitty".into(),
    }
}

fn window() -> WindowRef {
    WindowRef {
        handle: WindowHandle::Hwnd(7),
        pid: Some(70),
        rect: None,
    }
}

fn handle() -> WindowHandle {
    WindowHandle::Hwnd(7)
}

fn pane() -> Pane {
    Pane::new("%1", Rect::new(960, 0, 960, 1080), false, "fake")
}

fn registry(providers: &[&RecordingProvider]) -> PaneRegistry {
    PaneRegistry::new(
        providers
            .iter()
            .map(|p| Box::new((*p).clone()) as Box<dyn PaneProvider>)
            .collect(),
    )
}

fn sync_options(clock: f64) -> PaneWorkerOptions {
    PaneWorkerOptions {
        clock: Arc::new(move || clock),
        synchronous: true,
        ..PaneWorkerOptions::default()
    }
}

fn sync_worker(providers: &[&RecordingProvider], clock: f64) -> PaneWorker {
    PaneWorker::new(registry(providers), sync_options(clock))
}

fn detections(events: Vec<PaneEvent>) -> Vec<DetectResult> {
    events
        .into_iter()
        .filter_map(|e| match e {
            PaneEvent::Detected(result) => Some(result),
            PaneEvent::Focused(_) => None,
        })
        .collect()
}

fn focuses(events: Vec<PaneEvent>) -> Vec<FocusResult> {
    events
        .into_iter()
        .filter_map(|e| match e {
            PaneEvent::Focused(result) => Some(result),
            PaneEvent::Detected(_) => None,
        })
        .collect()
}

fn detect_calls(provider: &RecordingProvider) -> usize {
    provider
        .calls()
        .iter()
        .filter(|c| matches!(c, Call::Detect(_)))
        .count()
}

fn ids(snapshot: &PaneSnapshot) -> Vec<PaneId> {
    snapshot.panes.iter().map(|p| p.id.clone()).collect()
}

#[test]
fn test_detect_stamps_the_snapshot_with_the_worker_clock() {
    let provider = RecordingProvider::fake().snapshot(|window| {
        let mut snapshot = two_panes(window.handle.clone());
        snapshot.taken_at = -1.0;
        Some(snapshot)
    });
    let worker = sync_worker(&[&provider], 123.0);
    let request = worker.request_detect(&window(), &app());
    let results = detections(worker.drain_events());
    let [result] = &results[..] else {
        panic!("{results:?}")
    };
    assert_eq!(result.request_id, request);
    assert_eq!(result.window_handle, handle());
    let snapshot = result.snapshot.as_ref().unwrap();
    assert_eq!(snapshot.taken_at, 123.0);
    assert_eq!(ids(snapshot), vec!["%0".into(), "%1".into()]);
}

#[test]
fn test_first_provider_with_panes_wins() {
    let empty = RecordingProvider::new("empty");
    let found = RecordingProvider::new("found").snapshot(|window| {
        let mut snapshot = two_panes(window.handle.clone());
        for p in &mut snapshot.panes {
            p.provider = "found".into();
        }
        Some(snapshot)
    });
    let later = RecordingProvider::new("later").snapshot(|w| Some(two_panes(w.handle.clone())));
    let worker = sync_worker(&[&empty, &found, &later], 50.0);
    worker.request_detect(&window(), &app());
    let results = detections(worker.drain_events());
    assert_eq!(
        results[0].snapshot.as_ref().unwrap().provider(),
        Some("found")
    );
    assert!(empty.calls().contains(&Call::Detect(handle())));
    assert!(!later.calls().contains(&Call::Detect(handle())));
}

#[test]
fn test_focus_goes_to_the_panes_provider() {
    let fake = RecordingProvider::new("fake");
    let other = RecordingProvider::new("other");
    let worker = sync_worker(&[&other, &fake], 50.0);
    worker.request_focus(&window(), &pane());
    assert_eq!(
        focuses(worker.drain_events()),
        vec![FocusResult {
            window_handle: handle(),
            pane_id: "%1".into(),
            ok: true
        }]
    );
    assert_eq!(fake.calls(), vec![Call::Focus("%1".into())]);
    assert_eq!(other.calls(), vec![]);
    worker.request_focus(&window(), &Pane::new("%2", pane().rect, false, "gone"));
    assert!(!focuses(worker.drain_events())[0].ok);
}

#[test]
fn test_a_provider_failing_three_times_in_a_row_is_switched_off() {
    let flaky = RecordingProvider::new("flaky");
    flaky.set_fail(Some(PaneError::new("tmux did not answer within 1 s")));
    let backup = RecordingProvider::new("backup").snapshot(|w| Some(two_panes(w.handle.clone())));
    let worker = sync_worker(&[&flaky, &backup], 50.0);
    for _ in 0..FAILURE_LIMIT + 2 {
        worker.request_detect(&window(), &app());
    }
    assert_eq!(detect_calls(&flaky), FAILURE_LIMIT as usize);
    assert_eq!(
        worker.disabled_providers(),
        HashSet::from(["flaky".to_string()])
    );
    // the next provider answered
    assert!(
        detections(worker.drain_events())
            .iter()
            .all(|r| r.snapshot.is_some())
    );
    worker.request_focus(&window(), &Pane::new("%1", pane().rect, false, "flaky"));
    assert!(!flaky.calls().contains(&Call::Focus("%1".into())));
}

#[test]
fn test_a_success_resets_the_failure_count() {
    let provider = RecordingProvider::new("p").snapshot(|w| Some(two_panes(w.handle.clone())));
    let worker = sync_worker(&[&provider], 50.0);
    for _ in 0..5 {
        provider.set_fail(Some(PaneError::new("busy")));
        worker.request_detect(&window(), &app());
        worker.request_detect(&window(), &app());
        provider.set_fail(None);
        worker.request_detect(&window(), &app());
    }
    assert_eq!(worker.disabled_providers(), HashSet::new());
}

#[test]
fn test_slow_calls_count_as_failures() {
    let provider = RecordingProvider::new("slow").snapshot(|w| {
        std::thread::sleep(Duration::from_millis(30));
        Some(two_panes(w.handle.clone()))
    });
    let worker = PaneWorker::new(
        registry(&[&provider]),
        PaneWorkerOptions {
            synchronous: true,
            call_timeout_s: 0.01,
            ..PaneWorkerOptions::default()
        },
    );
    for _ in 0..FAILURE_LIMIT {
        worker.request_detect(&window(), &app());
    }
    // late answers are dropped
    assert!(
        detections(worker.drain_events())
            .iter()
            .all(|r| r.snapshot.is_none())
    );
    assert_eq!(
        worker.disabled_providers(),
        HashSet::from(["slow".to_string()])
    );
}

#[test]
fn test_denied_app_gets_an_empty_answer() {
    let provider = RecordingProvider::fake().snapshot(|w| Some(two_panes(w.handle.clone())));
    let worker = sync_worker(&[&provider], 50.0);
    worker.request_detect(
        &window(),
        &AppIdentity {
            process: "code".into(),
            app_id: "Chrome_WidgetWin_1".into(),
        },
    );
    assert_eq!(detections(worker.drain_events())[0].snapshot, None);
    assert_eq!(provider.calls(), vec![]);
}

// ----------------------------------------------------------------- the thread
/// Collect what the worker reported until `predicate` holds (the Python `_wait`).
fn wait(
    worker: &PaneWorker,
    sink: &mut Vec<PaneEvent>,
    timeout: f64,
    mut predicate: impl FnMut(&[PaneEvent]) -> bool,
) -> bool {
    let deadline = Instant::now() + Duration::from_secs_f64(timeout);
    while Instant::now() < deadline {
        sink.extend(worker.drain_events());
        if predicate(sink) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    false
}

fn wait_for(timeout: f64, mut predicate: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + Duration::from_secs_f64(timeout);
    while Instant::now() < deadline {
        if predicate() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    false
}

/// A `threading.Event`.
#[derive(Clone, Default)]
struct Flag(Arc<(Mutex<bool>, Condvar)>);

impl Flag {
    fn set(&self) {
        *self.0.0.lock() = true;
        self.0.1.notify_all();
    }

    fn wait(&self, seconds: f64) {
        let mut set = self.0.0.lock();
        if !*set {
            self.0
                .1
                .wait_for(&mut set, Duration::from_secs_f64(seconds));
        }
    }
}

#[test]
fn test_thread_answers_and_stops_cleanly() {
    let release = Flag::default();
    let threads: Arc<Mutex<Vec<String>>> = Arc::default();

    let blocking = {
        let release = release.clone();
        let threads = threads.clone();
        move |window: &WindowRef| {
            threads.lock().push(current_thread_name());
            release.wait(2.0);
            Some(two_panes(window.handle.clone()))
        }
    };
    let provider = RecordingProvider::fake().snapshot(blocking);
    let mut worker = PaneWorker::new(registry(&[&provider]), PaneWorkerOptions::default());
    let mut events = Vec::new();
    worker.start();
    let first = worker.request_detect(&window(), &app());
    assert!(wait_for(3.0, || threads.lock().len() == 1)); // running, blocked in the provider
    worker.request_detect(&window(), &app()); // replaced by the next one before it starts
    let last = worker.request_detect(&window(), &app());
    worker.request_focus(&window(), &pane()); // goes before the waiting detection
    release.set();
    assert!(wait(&worker, &mut events, 3.0, |e| { e.len() == 3 }));
    let results = detections(events.clone());
    assert_eq!(
        results.iter().map(|r| r.request_id).collect::<Vec<_>>(),
        vec![first, last]
    );
    assert_eq!(focuses(events.clone()).len(), 1);
    let calls = provider.calls();
    let focus_at = calls
        .iter()
        .position(|c| *c == Call::Focus("%1".into()))
        .unwrap();
    assert!(focus_at < calls.len() - 1);
    assert_eq!(threads.lock()[0], "eye-tracker-panes");
    worker.stop(JOIN_TIMEOUT_S);
    assert!(!worker.running());
    worker.request_detect(&window(), &app()); // after stop: dropped, no thread
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(detections(worker.drain_events()).len(), 0);
    assert_eq!(detections(events).len(), 2);
}

#[test]
fn test_stop_does_not_wait_forever_for_a_hung_provider() {
    let hang = Flag::default();
    let provider = RecordingProvider::fake().snapshot({
        let hang = hang.clone();
        move |_| {
            hang.wait(5.0);
            None
        }
    });
    let mut worker = PaneWorker::new(registry(&[&provider]), PaneWorkerOptions::default());
    worker.start();
    worker.request_detect(&window(), &app());
    assert!(wait_for(3.0, || !provider.calls().is_empty()));
    let started = Instant::now();
    worker.stop(0.2);
    assert!(started.elapsed() < Duration::from_secs(1));
    hang.set();
}

// ------------------------------------------------------------ closing providers
#[test]
fn test_replaced_and_final_providers_are_closed_on_the_worker_thread() {
    let old = RecordingProvider::new("old").snapshot(|w| Some(two_panes(w.handle.clone())));
    let new = RecordingProvider::new("new").snapshot(|w| Some(two_panes(w.handle.clone())));
    let mut worker = PaneWorker::new(registry(&[&old]), PaneWorkerOptions::default());
    worker.start();
    worker.request_detect(&window(), &app());
    assert!(wait_for(3.0, || !old.calls().is_empty()));
    worker.set_registry(registry(&[&new]));
    assert!(wait_for(3.0, || old.closed_on() == ["eye-tracker-panes"]));
    assert_eq!(new.closed_on(), Vec::<String>::new());
    worker.stop(JOIN_TIMEOUT_S);
    // before the thread ended
    assert_eq!(new.closed_on(), ["eye-tracker-panes"]);
    // once
    assert_eq!(old.closed_on(), ["eye-tracker-panes"]);
}

#[test]
fn test_request_stop_returns_at_once_and_the_thread_closes_up() {
    let release = Flag::default();
    let provider = RecordingProvider::fake().snapshot({
        let release = release.clone();
        move |_| {
            release.wait(5.0);
            None
        }
    });
    let mut worker = PaneWorker::new(registry(&[&provider]), PaneWorkerOptions::default());
    worker.start();
    worker.request_detect(&window(), &app());
    assert!(wait_for(3.0, || !provider.calls().is_empty())); // busy in the provider
    let started = Instant::now();
    worker.request_stop();
    assert!(started.elapsed() < Duration::from_millis(100)); // no join
    assert_eq!(provider.closed_on(), Vec::<String>::new()); // still in use on the thread
    release.set();
    assert!(wait_for(3.0, || provider.closed_on() == ["eye-tracker-panes"]));
    assert!(wait_for(3.0, || !worker.running()));
    assert_eq!(worker.drain_events(), vec![]); // nothing is delivered after a stop
}

#[test]
fn test_synchronous_worker_closes_on_the_calling_thread() {
    let old = RecordingProvider::new("old");
    let new = RecordingProvider::new("new");
    let mut worker = sync_worker(&[&old], 50.0);
    worker.set_registry(registry(&[&new]));
    assert_eq!(old.closed_on(), [current_thread_name()]);
    worker.stop(JOIN_TIMEOUT_S);
    assert_eq!(new.closed_on(), [current_thread_name()]);
}

/// A provider whose `close` panics (the Python test raised `OSError`).
struct BrokenClose;

impl PaneProvider for BrokenClose {
    fn name(&self) -> &str {
        "broken"
    }

    fn applies(&mut self, _app: &AppIdentity) -> bool {
        true
    }

    fn detect(
        &mut self,
        _window: &WindowRef,
        _app: &AppIdentity,
    ) -> Result<Option<PaneSnapshot>, PaneError> {
        Ok(None)
    }

    fn focus(&mut self, _window: &WindowRef, _pane: &Pane) -> Result<bool, PaneError> {
        Ok(false)
    }

    fn close(&mut self) {
        panic!("CoUninitialize said no");
    }
}

#[test]
fn test_a_failing_close_is_contained() {
    let after = RecordingProvider::new("after");
    let mut registry = PaneRegistry::new(vec![Box::new(BrokenClose), Box::new(after.clone())]);
    registry.close();
    assert_eq!(after.closed_on(), [current_thread_name()]);
}

#[test]
fn test_a_runner_is_not_needed_by_the_worker() {
    // New: the worker never touches a command runner; this keeps the import honest.
    let _: Option<Arc<dyn CommandRunner>> = Some(Arc::new(FakeRunner::default()));
}
