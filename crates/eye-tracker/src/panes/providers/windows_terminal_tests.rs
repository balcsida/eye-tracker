// Ported from tests/test_panes_windows_terminal.py (the Windows Terminal pane provider with a fake
// UI Automation client); the tests of its UI Automation binding are in
// platform/win_uia_tests.rs.
//
// Changed: the fake client implements the `UiaClient` trait and is shared (a clone goes to the
// provider) instead of being a plain object both sides hold.
//
// Not ported:
// - the `p.focus(...)` check of `pane.id is None` in the test of foreign panes: `Pane::id` is not
//   optional in Rust, so no such pane exists.
// - `worker._failures` in the worker tests: the worker's counters are private; the switch-off is
//   observed through `disabled_providers` and the calls the fake receives.

use std::collections::VecDeque;
use std::sync::Arc;

use parking_lot::Mutex;

use super::*;
use crate::panes::worker::{FAILURE_LIMIT, PaneEvent, PaneWorker, PaneWorkerOptions};
use crate::platform::win_uia::ComError;
use crate::types::WindowHandle;

use super::super::super::registry::PaneRegistry;

const HWND: isize = 0x40A2C;
const HWND_PART: i32 = 0x40A2C;
const AREA: Rect = Rect::new(0, 0, 1920, 1040);

fn window() -> WindowRef {
    WindowRef {
        handle: WindowHandle::Hwnd(HWND),
        pid: Some(9000),
        rect: Some(Rect::new(-8, -8, 1936, 1056)),
    }
}

fn app() -> AppIdentity {
    AppIdentity {
        process: "windowsterminal".into(),
        app_id: WINDOW_CLASS.into(),
    }
}

fn identity(process: &str, app_id: &str) -> AppIdentity {
    AppIdentity {
        process: process.into(),
        app_id: app_id.into(),
    }
}

fn element(
    runtime_id: Option<&[i32]>,
    rect: Option<Rect>,
    has_keyboard_focus: bool,
    offscreen: bool,
) -> UiaElement {
    UiaElement {
        runtime_id: runtime_id.map(<[i32]>::to_vec),
        rect,
        has_keyboard_focus,
        offscreen,
    }
}

fn left() -> UiaElement {
    element(
        Some(&[42, HWND_PART, 4, 101]),
        Some(Rect::new(8, 80, 948, 952)),
        true,
        false,
    )
}

fn right() -> UiaElement {
    element(
        Some(&[42, HWND_PART, 4, 102]),
        Some(Rect::new(964, 80, 948, 952)),
        false,
        false,
    )
}

fn runtime(ids: &[i32]) -> PaneId {
    PaneId::Runtime(ids.to_vec())
}

fn left_id() -> PaneId {
    runtime(&[42, HWND_PART, 4, 101])
}

fn right_id() -> PaneId {
    runtime(&[42, HWND_PART, 4, 102])
}

#[derive(Debug, Clone, PartialEq)]
enum Call {
    Foreground,
    Find(isize, String),
    Focus(isize, String),
}

struct FakeState {
    elements: Vec<UiaElement>,
    foreground: Option<isize>,
    fail: Option<UiaError>,
    calls: Vec<Call>,
    focused: Vec<UiaElement>,
    /// Foreground windows answered first, one per call (then `foreground`).
    foreground_sequence: VecDeque<Option<isize>>,
    closed: u32,
}

/// Answers like UI Automation would; records every call. Clones share the state.
#[derive(Clone)]
struct FakeUia(Arc<Mutex<FakeState>>);

impl FakeUia {
    fn new() -> Self {
        Self::with(vec![left(), right()])
    }

    fn with(elements: Vec<UiaElement>) -> Self {
        Self(Arc::new(Mutex::new(FakeState {
            elements,
            foreground: Some(HWND),
            fail: None,
            calls: Vec::new(),
            focused: Vec::new(),
            foreground_sequence: VecDeque::new(),
            closed: 0,
        })))
    }

    fn foreground(self, foreground: isize) -> Self {
        self.0.lock().foreground = Some(foreground);
        self
    }

    fn failing(error: UiaError) -> Self {
        let client = Self::new();
        client.0.lock().fail = Some(error);
        client
    }

    fn calls(&self) -> Vec<Call> {
        self.0.lock().calls.clone()
    }

    fn focused(&self) -> Vec<UiaElement> {
        self.0.lock().focused.clone()
    }

    fn closed(&self) -> u32 {
        self.0.lock().closed
    }

    fn set_elements(&self, elements: Vec<UiaElement>) {
        self.0.lock().elements = elements;
    }

    fn set_foreground_sequence(&self, sequence: &[Option<isize>]) {
        self.0.lock().foreground_sequence = sequence.iter().copied().collect();
    }
}

impl UiaClient for FakeUia {
    fn find(&self, hwnd: isize, class_name: &str) -> Result<Vec<UiaElement>, UiaError> {
        let mut state = self.0.lock();
        state.calls.push(Call::Find(hwnd, class_name.to_string()));
        match &state.fail {
            Some(error) => Err(error.clone()),
            None => Ok(state.elements.clone()),
        }
    }

    fn focus(
        &self,
        hwnd: isize,
        class_name: &str,
        matches: &dyn Fn(&UiaElement) -> bool,
        guard: Option<&dyn Fn() -> Result<bool, UiaError>>,
    ) -> Result<bool, UiaError> {
        let elements = {
            let mut state = self.0.lock();
            state.calls.push(Call::Focus(hwnd, class_name.to_string()));
            if let Some(error) = &state.fail {
                return Err(error.clone());
            }
            state.elements.clone()
        };
        for element in elements {
            if matches(&element) {
                // The guard asks for the foreground window: the state is not locked here.
                if let Some(guard) = guard
                    && !guard()?
                {
                    return Ok(false);
                }
                self.0.lock().focused.push(element);
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn foreground_window(&self) -> Result<Option<isize>, UiaError> {
        let mut state = self.0.lock();
        state.calls.push(Call::Foreground);
        Ok(match state.foreground_sequence.pop_front() {
            Some(answer) => answer,
            None => state.foreground,
        })
    }

    fn close(&self) {
        self.0.lock().closed += 1;
    }
}

fn provider(client: &FakeUia) -> WindowsTerminalProvider {
    WindowsTerminalProvider::new(Arc::new(|window: &WindowRef| {
        (window.handle == WindowHandle::Hwnd(HWND)).then_some(AREA)
    }))
    .with_client(Box::new(client.clone()))
}

fn detect(p: &mut WindowsTerminalProvider) -> Option<PaneSnapshot> {
    p.detect(&window(), &app()).expect("no UI Automation error")
}

fn right_pane() -> Pane {
    Pane::new(right_id(), right().rect.unwrap(), false, "windows_terminal")
}

fn ids(snapshot: &PaneSnapshot) -> Vec<PaneId> {
    snapshot.panes.iter().map(|p| p.id.clone()).collect()
}

// ------------------------------------------------------------------ applies
#[test]
fn test_creating_the_provider_touches_nothing() {
    let mut p = WindowsTerminalProvider::new(Arc::new(|_| None));
    assert_eq!(p.name(), "windows_terminal");
    assert!(p.applies(&app()));
    assert!(p.client.is_none()); // UI Automation only on the worker thread, when asked
}

#[test]
fn test_applies_to_windows_terminal_windows_only() {
    let mut p = provider(&FakeUia::new());
    assert!(p.applies(&app()));
    assert!(p.applies(&identity("WindowsTerminal", WINDOW_CLASS)));
    assert!(!p.applies(&identity("windowsterminal", "PseudoConsoleWindow")));
    assert!(!p.applies(&identity("openconsole", WINDOW_CLASS)));
    assert!(!p.applies(&identity("wezterm-gui", "org.wezfurlong.wezterm")));
    assert!(!p.applies(&identity("code", "Chrome_WidgetWin_1")));
}

// ------------------------------------------------------------------- detect
#[test]
fn test_detects_two_panes_with_the_focused_one() {
    let client = FakeUia::new();
    let snapshot = detect(&mut provider(&client)).expect("a snapshot");
    assert_eq!(snapshot.window_handle, WindowHandle::Hwnd(HWND));
    let rects: Vec<Rect> = snapshot.panes.iter().map(|p| p.rect).collect();
    assert_eq!(rects, vec![left().rect.unwrap(), right().rect.unwrap()]);
    assert_eq!(ids(&snapshot), vec![left_id(), right_id()]);
    assert!(
        snapshot
            .panes
            .iter()
            .all(|p| p.provider == "windows_terminal")
    );
    assert_eq!(snapshot.focused().map(|p| p.id.clone()), Some(left_id()));
    assert_eq!(
        client.calls(),
        vec![Call::Foreground, Call::Find(HWND, PANE_CLASS.into())]
    );
}

#[test]
fn test_three_panes_are_ordered_top_to_bottom_then_left_to_right() {
    let top_right = element(
        Some(&[42, 1, 3]),
        Some(Rect::new(964, 80, 948, 470)),
        false,
        false,
    );
    let bottom_right = element(
        Some(&[42, 1, 4]),
        Some(Rect::new(964, 556, 948, 476)),
        true,
        false,
    );
    let left = element(
        Some(&[42, 1, 2]),
        Some(Rect::new(8, 80, 948, 952)),
        false,
        false,
    );
    let client = FakeUia::with(vec![bottom_right, left, top_right]);
    let snapshot = detect(&mut provider(&client)).expect("a snapshot");
    assert_eq!(
        ids(&snapshot),
        vec![
            runtime(&[42, 1, 2]),
            runtime(&[42, 1, 3]),
            runtime(&[42, 1, 4])
        ]
    );
    assert_eq!(
        snapshot.focused().map(|p| p.id.clone()),
        Some(runtime(&[42, 1, 4]))
    );
}

#[test]
fn test_offscreen_and_empty_panes_are_ignored() {
    let hidden = element(
        Some(&[42, 1, 9]),
        Some(Rect::new(8, 80, 948, 952)),
        false,
        true,
    );
    let empty = element(Some(&[42, 1, 8]), None, false, false);
    let outside = element(
        Some(&[42, 1, 7]),
        Some(Rect::new(3000, 80, 500, 500)),
        false,
        false,
    );
    let client = FakeUia::with(vec![
        left(),
        hidden.clone(),
        empty.clone(),
        outside.clone(),
        right(),
    ]);
    let snapshot = detect(&mut provider(&client)).expect("a snapshot");
    assert_eq!(ids(&snapshot), vec![left_id(), right_id()]);
    // Only one visible pane left: no snapshot (tmux in it may then be asked).
    let client = FakeUia::with(vec![left(), hidden, empty, outside]);
    assert_eq!(detect(&mut provider(&client)), None);
    assert_eq!(detect(&mut provider(&FakeUia::with(vec![]))), None);
}

#[test]
fn test_pane_rectangles_are_cut_to_the_window() {
    let wide = element(
        Some(&[42, 1, 1]),
        Some(Rect::new(-20, 80, 990, 952)),
        true,
        false,
    );
    let client = FakeUia::with(vec![wide, right()]);
    let snapshot = detect(&mut provider(&client)).expect("a snapshot");
    assert_eq!(snapshot.panes[0].rect, Rect::new(0, 80, 970, 952));
}

#[test]
fn test_no_focus_flag_when_no_pane_has_the_keyboard_focus() {
    let unfocused = UiaElement {
        has_keyboard_focus: false,
        ..left()
    };
    let client = FakeUia::with(vec![unfocused, right()]);
    let snapshot = detect(&mut provider(&client)).expect("a snapshot");
    assert!(snapshot.focused().is_none());
}

#[test]
fn test_ids_are_stable_across_snapshots() {
    let client = FakeUia::new();
    let mut p = provider(&client);
    let first = detect(&mut p).expect("a snapshot");
    // The user focused the other pane and resized the split: same ids.
    client.set_elements(vec![
        UiaElement {
            rect: Some(Rect::new(8, 80, 600, 952)),
            has_keyboard_focus: false,
            ..left()
        },
        UiaElement {
            rect: Some(Rect::new(616, 80, 1296, 952)),
            has_keyboard_focus: true,
            ..right()
        },
    ]);
    let second = detect(&mut p).expect("a snapshot");
    assert_eq!(ids(&first), ids(&second));
    assert_eq!(second.focused().map(|p| p.id.clone()), Some(right_id()));
}

#[test]
fn test_without_a_runtime_id_the_rectangle_is_the_id() {
    let left = UiaElement {
        runtime_id: None,
        ..left()
    };
    let right = UiaElement {
        runtime_id: Some(vec![]),
        ..right()
    };
    let client = FakeUia::with(vec![left, right]);
    let mut p = provider(&client);
    let snapshot = detect(&mut p).expect("a snapshot");
    assert_eq!(
        ids(&snapshot),
        vec![
            PaneId::Rect(Rect::new(8, 80, 948, 952)),
            PaneId::Rect(Rect::new(964, 80, 948, 952)),
        ]
    );
    assert_eq!(detect(&mut p), Some(snapshot)); // stable while the layout is
    assert_eq!(pane_id(&element(None, None, false, false)), None);
}

#[test]
fn test_duplicate_ids_count_once() {
    let client = FakeUia::with(vec![left(), left()]);
    assert_eq!(detect(&mut provider(&client)), None);
}

#[test]
fn test_nothing_is_asked_unless_the_window_is_in_the_foreground() {
    let client = FakeUia::new().foreground(0x1234);
    let mut p = provider(&client);
    assert_eq!(detect(&mut p), None);
    assert_eq!(p.focus(&window(), &right_pane()), Ok(false));
    assert_eq!(client.calls(), vec![Call::Foreground, Call::Foreground]);
    assert!(client.focused().is_empty());
    let no_handle = WindowRef {
        handle: WindowHandle::X11(0),
        pid: None,
        rect: None,
    };
    assert_eq!(p.detect(&no_handle, &app()), Ok(None));
}

// -------------------------------------------------------------------- focus
#[test]
fn test_focus_sets_focus_on_the_matching_pane() {
    let client = FakeUia::new();
    let mut p = provider(&client);
    let snapshot = detect(&mut p).expect("a snapshot");
    let target = &snapshot.panes[1];
    assert_eq!(p.focus(&window(), target), Ok(true));
    assert_eq!(client.focused(), vec![right()]);
    // The foreground is checked again inside the walk, right before SetFocus.
    let calls = client.calls();
    assert_eq!(
        calls[calls.len() - 2..],
        [Call::Focus(HWND, PANE_CLASS.into()), Call::Foreground]
    );
}

#[test]
fn test_focus_of_a_pane_that_is_gone_fails() {
    let client = FakeUia::new();
    let mut p = provider(&client);
    let gone = Pane::new(
        runtime(&[42, HWND_PART, 4, 999]),
        right().rect.unwrap(),
        false,
        "windows_terminal",
    );
    assert_eq!(p.focus(&window(), &gone), Ok(false));
    assert!(client.focused().is_empty());
    // An off-screen element is never focused, even with the right id.
    client.set_elements(vec![
        left(),
        UiaElement {
            offscreen: true,
            ..right()
        },
    ]);
    assert_eq!(p.focus(&window(), &right_pane()), Ok(false));
    // Another provider's pane is not ours to focus.
    let tmux = Pane::new(right_id(), right().rect.unwrap(), false, "tmux");
    assert_eq!(p.focus(&window(), &tmux), Ok(false));
}

#[test]
fn test_no_focus_when_another_window_came_to_the_front_during_the_walk() {
    let client = FakeUia::new();
    let mut p = provider(&client);
    let target = right_pane();
    // In front before the walk, not after it.
    client.set_foreground_sequence(&[Some(HWND), Some(0x1234)]);
    assert_eq!(p.focus(&window(), &target), Ok(false));
    assert!(client.focused().is_empty());
    // The second check runs inside the walk, right before SetFocus.
    assert_eq!(
        client.calls(),
        vec![
            Call::Foreground,
            Call::Focus(HWND, PANE_CLASS.into()),
            Call::Foreground
        ]
    );
    assert_eq!(p.focus(&window(), &target), Ok(true)); // still in front: focused
}

#[test]
fn test_close_releases_the_automation_client() {
    let client = FakeUia::new();
    let mut p = provider(&client);
    p.close();
    assert_eq!(client.closed(), 1);
    p.close(); // nothing left to release
    assert_eq!(client.closed(), 1);
}

#[test]
fn test_com_errors_become_pane_errors() {
    let error = ComError::new("IUIAutomationElement::FindAll", 0x8013_1505);
    let client = FakeUia::failing(error.into());
    let mut p = provider(&client);
    let failure = p.detect(&window(), &app()).expect_err("a pane error");
    assert!(failure.0.contains("0x80131505"), "{failure}");
    assert!(p.focus(&window(), &right_pane()).is_err());
}

// -------------------------------------------------------- with the worker
fn worker(client: &FakeUia) -> PaneWorker {
    PaneWorker::new(
        PaneRegistry::new(vec![Box::new(provider(client))]),
        PaneWorkerOptions {
            synchronous: true,
            ..PaneWorkerOptions::default()
        },
    )
}

fn detected(worker: &PaneWorker) -> Vec<Option<PaneSnapshot>> {
    worker
        .drain_events()
        .into_iter()
        .filter_map(|event| match event {
            PaneEvent::Detected(result) => Some(result.snapshot),
            PaneEvent::Focused(_) => None,
        })
        .collect()
}

fn focus_results(worker: &PaneWorker) -> Vec<bool> {
    worker
        .drain_events()
        .into_iter()
        .filter_map(|event| match event {
            PaneEvent::Focused(result) => Some(result.ok),
            PaneEvent::Detected(_) => None,
        })
        .collect()
}

#[test]
fn test_worker_switches_the_provider_off_after_three_com_errors() {
    let error = ComError::new("IUIAutomation::ElementFromHandle", 0x8004_0201);
    let client = FakeUia::failing(error.into());
    let worker = worker(&client);
    for _ in 0..FAILURE_LIMIT + 2 {
        worker.request_detect(&window(), &app());
    }
    let finds = client
        .calls()
        .iter()
        .filter(|c| matches!(c, Call::Find(..)))
        .count();
    assert_eq!(finds, FAILURE_LIMIT as usize);
    assert_eq!(
        worker.disabled_providers(),
        ["windows_terminal".to_string()].into()
    );
    assert!(detected(&worker).iter().all(Option::is_none));
    let calls = client.calls().len();
    worker.request_focus(&window(), &right_pane());
    assert_eq!(focus_results(&worker).last(), Some(&false));
    assert_eq!(client.calls().len(), calls); // not asked any more
}

#[test]
fn test_worker_detects_and_focuses() {
    let client = FakeUia::new();
    let worker = worker(&client);
    worker.request_detect(&window(), &app());
    let results = detected(&worker);
    let [Some(snapshot)] = &results[..] else {
        panic!("expected one snapshot, got {results:?}");
    };
    assert_eq!(snapshot.provider(), Some("windows_terminal"));
    worker.request_focus(&window(), &snapshot.panes[1]);
    assert_eq!(focus_results(&worker).last(), Some(&true));
    assert_eq!(client.focused(), vec![right()]);
}

#[test]
fn test_denied_apps_never_reach_ui_automation() {
    let denied = [
        identity("code", "Chrome_WidgetWin_1"),
        identity("cursor", "Chrome_WidgetWin_1"),
        identity("claude", "Chrome_WidgetWin_1"),
        identity("Code", "Chrome_WidgetWin_1"),
        identity("someapp", "Chrome_WidgetWin_1"),
        // Even a Windows Terminal name on a Chromium window class is not asked.
        identity("windowsterminal", "Chrome_WidgetWin_1"),
    ];
    for app in denied {
        let client = FakeUia::new();
        let worker = worker(&client);
        worker.request_detect(&window(), &app);
        assert_eq!(client.calls(), vec![], "{}|{}", app.process, app.app_id);
        assert_eq!(detected(&worker), vec![None]);
    }
}
