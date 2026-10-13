//! Fakes shared by the split-pane tests (ported from tests/panes_fakes.py).
//!
//! Nothing here starts a process: command lines are answered from a table.
//!
//! Complete on purpose: the Windows Terminal and desktop-app provider tests reuse it later.
#![allow(dead_code)]

use std::sync::Arc;
use std::thread;

use parking_lot::Mutex;

use crate::panes::providers::command::{CommandResult, CommandRunner};
use crate::panes::types::{Pane, PaneError, PaneId, PaneProvider, PaneSnapshot};
use crate::types::{AppIdentity, Rect, WindowHandle, WindowRef};

/// What a [`FakeRunner`] answers with.
#[derive(Debug, Clone)]
pub(crate) enum Answer {
    Result(CommandResult),
    Error(PaneError),
}

impl From<&str> for Answer {
    /// Stdout of a successful run.
    fn from(stdout: &str) -> Self {
        Self::Result(CommandResult::new(0, stdout, ""))
    }
}

impl From<String> for Answer {
    fn from(stdout: String) -> Self {
        Self::Result(CommandResult::new(0, stdout, ""))
    }
}

impl From<CommandResult> for Answer {
    fn from(result: CommandResult) -> Self {
        Self::Result(result)
    }
}

impl From<PaneError> for Answer {
    fn from(error: PaneError) -> Self {
        Self::Error(error)
    }
}

#[derive(Default)]
struct RunnerState {
    answers: Vec<(Vec<String>, Answer)>,
    calls: Vec<Vec<String>>,
}

/// Answers command lines by their arguments after the program (`argv[1..]`).
///
/// A key is matched as a prefix of the arguments, so `["list-panes"]` answers every `list-panes`
/// call. Clones share their answers and recorded calls, so a test keeps one and hands the other to
/// the provider.
#[derive(Clone, Default)]
pub(crate) struct FakeRunner {
    state: Arc<Mutex<RunnerState>>,
}

impl FakeRunner {
    pub(crate) fn new(answers: &[(&[&str], Answer)]) -> Self {
        let runner = Self::default();
        for (key, answer) in answers {
            runner.answer(key, answer.clone());
        }
        runner
    }

    /// Set (or replace) the answer to the commands starting with `key`.
    pub(crate) fn answer(&self, key: &[&str], answer: impl Into<Answer>) {
        let key: Vec<String> = key.iter().map(|s| s.to_string()).collect();
        let answer = answer.into();
        let mut state = self.state.lock();
        match state.answers.iter_mut().find(|(k, _)| *k == key) {
            Some(entry) => entry.1 = answer,
            None => state.answers.push((key, answer)),
        }
    }

    /// Every argv run so far.
    pub(crate) fn calls(&self) -> Vec<Vec<String>> {
        self.state.lock().calls.clone()
    }

    /// The arguments after the program of every call.
    pub(crate) fn commands(&self) -> Vec<Vec<String>> {
        self.calls().into_iter().map(|c| c[1..].to_vec()).collect()
    }
}

impl CommandRunner for FakeRunner {
    fn run(&self, argv: &[String], _timeout: f64) -> Result<CommandResult, PaneError> {
        let mut state = self.state.lock();
        state.calls.push(argv.to_vec());
        let rest = &argv[1..];
        let best = state
            .answers
            .iter()
            .filter(|(key, _)| rest.starts_with(key))
            .max_by_key(|(key, _)| key.len());
        match best {
            None => Ok(CommandResult::new(1, "", "unknown command")),
            Some((_, Answer::Error(error))) => Err(error.clone()),
            Some((_, Answer::Result(result))) => Ok(result.clone()),
        }
    }
}

/// One call a [`RecordingProvider`] received.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Call {
    Applies(AppIdentity),
    Detect(WindowHandle),
    Focus(PaneId),
}

type SnapshotFn = Arc<dyn Fn(&WindowRef) -> Option<PaneSnapshot> + Send + Sync>;

#[derive(Default)]
struct ProviderState {
    calls: Vec<Call>,
    closed_on: Vec<String>,
    fail: Option<PaneError>,
}

/// A provider that records every call (to prove what is never asked).
///
/// Clones share their recordings, so a test keeps one and gives the other to the registry.
#[derive(Clone)]
pub(crate) struct RecordingProvider {
    name: String,
    applies: bool,
    snapshot: Option<SnapshotFn>,
    focus_ok: bool,
    state: Arc<Mutex<ProviderState>>,
}

impl RecordingProvider {
    pub(crate) fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            applies: true,
            snapshot: None,
            focus_ok: true,
            state: Arc::default(),
        }
    }

    /// A provider named "fake".
    pub(crate) fn fake() -> Self {
        Self::new("fake")
    }

    pub(crate) fn applies(mut self, applies: bool) -> Self {
        self.applies = applies;
        self
    }

    pub(crate) fn snapshot(
        mut self,
        snapshot: impl Fn(&WindowRef) -> Option<PaneSnapshot> + Send + Sync + 'static,
    ) -> Self {
        self.snapshot = Some(Arc::new(snapshot));
        self
    }

    pub(crate) fn focus_ok(mut self, focus_ok: bool) -> Self {
        self.focus_ok = focus_ok;
        self
    }

    /// Make `detect` and `focus` fail with `error` (`None`: work again).
    pub(crate) fn set_fail(&self, error: Option<PaneError>) {
        self.state.lock().fail = error;
    }

    pub(crate) fn calls(&self) -> Vec<Call> {
        self.state.lock().calls.clone()
    }

    /// Names of the threads `close` was called on.
    pub(crate) fn closed_on(&self) -> Vec<String> {
        self.state.lock().closed_on.clone()
    }
}

pub(crate) fn current_thread_name() -> String {
    thread::current().name().unwrap_or("").to_string()
}

impl PaneProvider for RecordingProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn applies(&mut self, app: &AppIdentity) -> bool {
        self.state.lock().calls.push(Call::Applies(app.clone()));
        self.applies
    }

    fn detect(
        &mut self,
        window: &WindowRef,
        _app: &AppIdentity,
    ) -> Result<Option<PaneSnapshot>, PaneError> {
        let fail = {
            let mut state = self.state.lock();
            state.calls.push(Call::Detect(window.handle.clone()));
            state.fail.clone()
        };
        if let Some(error) = fail {
            return Err(error);
        }
        Ok(self.snapshot.as_ref().and_then(|snapshot| snapshot(window)))
    }

    fn focus(&mut self, _window: &WindowRef, pane: &Pane) -> Result<bool, PaneError> {
        let fail = {
            let mut state = self.state.lock();
            state.calls.push(Call::Focus(pane.id.clone()));
            state.fail.clone()
        };
        match fail {
            Some(error) => Err(error),
            None => Ok(self.focus_ok),
        }
    }

    /// Records the name of the thread it was closed on.
    fn close(&mut self) {
        self.state.lock().closed_on.push(current_thread_name());
    }
}

/// Two side-by-side panes `%0` and `%1` of the window `handle`, `focused` having the focus.
pub(crate) struct TwoPanes {
    pub(crate) handle: WindowHandle,
    pub(crate) provider: &'static str,
    pub(crate) left: Rect,
    pub(crate) right: Rect,
    pub(crate) focused: &'static str,
    pub(crate) taken_at: f64,
}

impl Default for TwoPanes {
    fn default() -> Self {
        Self {
            handle: WindowHandle::Hwnd(7),
            provider: "fake",
            left: Rect::new(0, 0, 960, 1080),
            right: Rect::new(960, 0, 960, 1080),
            focused: "%0",
            taken_at: 0.0,
        }
    }
}

impl TwoPanes {
    pub(crate) fn build(self) -> PaneSnapshot {
        PaneSnapshot {
            window_handle: self.handle,
            panes: vec![
                Pane::new("%0", self.left, self.focused == "%0", self.provider),
                Pane::new("%1", self.right, self.focused == "%1", self.provider),
            ],
            taken_at: self.taken_at,
        }
    }
}

/// `TwoPanes::default()` for the window `handle`.
pub(crate) fn two_panes(handle: WindowHandle) -> PaneSnapshot {
    TwoPanes {
        handle,
        ..TwoPanes::default()
    }
    .build()
}
