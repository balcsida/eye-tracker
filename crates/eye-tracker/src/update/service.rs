//! The update service of the running app: when to look, what to tell, what to install.
//!
//! It lives next to the controller, not inside it: the engine that watches the
//! camera never touches the network. The service
//!
//! * looks for a newer release **only** when the user turned the check on (once a
//!   day, the first time a minute and a half after the app started) or asked for it
//!   (tray menu, About);
//! * tells about a version once (a tray notification), then only keeps the menu
//!   entry "Update to X.Y.Z…";
//! * installs nothing by itself: [`UpdateService::install`] is called after the
//!   user agreed in the update window.
//!
//! Work runs on a background thread; its results are handed over by
//! [`UpdateService::drain_events`], which the owner calls on the UI thread and which returns
//! [`Event`]s (the Qt signals `state_changed` and `announce`). The owner also calls
//! [`UpdateService::tick`] regularly (the Qt timer). What the service remembers between runs (the
//! time of the last check and the version it already told about) is kept in `updates.json` in
//! the data folder.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crossbeam_channel::{Receiver, Sender};
use serde_json::Value;

use super::fetch::{Fetcher, NetworkError, supported};
use super::installer::{self, InstallError};
use super::release::{MAX_API_BYTES, ReleaseInfo, is_newer, parse_release};
use crate::config::atomic_write_text;
use crate::{VERSION, paths};

pub use super::release::latest_release_url;

/// First automatic check, after the app started (the start itself stays quiet).
pub const FIRST_CHECK_DELAY_S: f64 = 90.0;
/// The automatic check is due again after this long.
pub const CHECK_INTERVAL_S: f64 = 24.0 * 3600.0;
/// How often the service looks whether a check is due (a laptop may sleep for days).
pub const TICK_INTERVAL_S: f64 = 3.0 * 3600.0;
/// Progress is passed on at most this often.
pub const PROGRESS_INTERVAL_S: f64 = 0.2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Phase {
    #[default]
    Idle,
    Checking,
    UpToDate,
    Available,
    Downloading,
    Installing,
    Failed,
}

impl Phase {
    /// The name the Python app used for the phase.
    pub fn value(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Checking => "checking",
            Self::UpToDate => "up_to_date",
            Self::Available => "available",
            Self::Downloading => "downloading",
            Self::Installing => "installing",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct UpdateState {
    pub phase: Phase,
    /// The newest release found (set from `Available` on, also while installing).
    pub release: Option<ReleaseInfo>,
    pub message: String,
    pub done: u64,
    pub total: u64,
    /// The user asked for the check that produced this state (a result is always shown).
    pub manual: bool,
    pub extra: BTreeMap<String, String>,
}

impl UpdateState {
    fn new(phase: Phase, release: Option<ReleaseInfo>, manual: bool) -> Self {
        Self {
            phase,
            release,
            manual,
            ..Self::default()
        }
    }

    fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = message.into();
        self
    }
}

/// What the Qt signals of the service were.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// Every change of the state.
    StateChanged(UpdateState),
    /// A version the user was not told about yet was found by an automatic check: show a
    /// notification once.
    Announce(ReleaseInfo),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckResult {
    pub release: Option<ReleaseInfo>,
    pub newer: bool,
}

/// Ask GitHub for the latest release and compare it with `running`.
pub fn check_for_update(fetcher: &mut Fetcher, running: &str) -> Result<CheckResult, NetworkError> {
    let body = fetcher.get(
        &latest_release_url(),
        MAX_API_BYTES,
        Some("application/vnd.github+json"),
    )?;
    let payload: Value = serde_json::from_slice(&body)
        .map_err(|_| NetworkError::new("GitHub's answer could not be read."))?;
    let release = parse_release(&payload).ok_or_else(|| {
        NetworkError::new("GitHub's answer is not a release this app understands.")
    })?;
    let newer = is_newer(&release, running);
    Ok(CheckResult {
        release: Some(release),
        newer,
    })
}

/// The small file that keeps the last check and the version told about.
struct Memory {
    path: PathBuf,
    last_check: f64,
    notified: String,
}

impl Memory {
    fn load(path: PathBuf) -> Self {
        let mut memory = Self {
            path,
            last_check: 0.0,
            notified: String::new(),
        };
        let data = std::fs::read_to_string(&memory.path)
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok());
        if let Some(Value::Object(data)) = data {
            if let Some(Value::Number(last)) = data.get("last_check") {
                memory.last_check = last.as_f64().unwrap_or(0.0);
            }
            if let Some(Value::String(notified)) = data.get("notified") {
                memory.notified = notified.clone();
            }
        }
        memory
    }

    fn save(&self) {
        let text = serde_json::json!({"last_check": self.last_check, "notified": self.notified});
        if atomic_write_text(&self.path, &format!("{text}\n")).is_err() {
            log::debug!("Could not save the update memory");
        }
    }
}

type Launcher = Box<dyn FnMut(&Path) -> Result<(), InstallError>>;
type FetcherFactory = Arc<dyn Fn() -> Fetcher + Send + Sync>;

/// What an [`UpdateService`] is made of; the defaults are the real ones, tests replace them.
pub struct UpdateOptions {
    pub fetcher_factory: FetcherFactory,
    /// Wall-clock seconds (the time of the last check is kept across runs).
    pub clock: Box<dyn Fn() -> f64>,
    pub running_version: String,
    pub memory_path: Option<PathBuf>,
    pub kind: Option<String>,
    pub launcher: Launcher,
    /// Whether this platform can look for updates at all.
    pub supported: Box<dyn Fn() -> bool>,
}

impl Default for UpdateOptions {
    fn default() -> Self {
        Self {
            fetcher_factory: Arc::new(Fetcher::new),
            clock: Box::new(|| {
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_or(0.0, |elapsed| elapsed.as_secs_f64())
            }),
            running_version: VERSION.to_string(),
            memory_path: None,
            kind: None,
            launcher: Box::new(installer::launch_installer),
            supported: Box::new(supported),
        }
    }
}

/// See the module docstring. Create it on the UI thread.
pub struct UpdateService {
    fetcher_factory: FetcherFactory,
    clock: Box<dyn Fn() -> f64>,
    running: String,
    launcher: Launcher,
    supported: Box<dyn Fn() -> bool>,
    kind: String,
    memory: Memory,
    auto: bool,
    state: UpdateState,
    thread: Option<JoinHandle<()>>,
    cancel: Arc<AtomicBool>,
    /// When the (single-shot) timer fires, on the clock of `tick(now)`.
    due: Option<f64>,
    finished_tx: Sender<UpdateState>,
    finished_rx: Receiver<UpdateState>,
    events: Vec<Event>,
}

impl UpdateService {
    pub fn new(options: UpdateOptions) -> Self {
        let UpdateOptions {
            fetcher_factory,
            clock,
            running_version,
            memory_path,
            kind,
            launcher,
            supported,
        } = options;
        let (finished_tx, finished_rx) = crossbeam_channel::unbounded();
        Self {
            fetcher_factory,
            clock,
            running: running_version,
            launcher,
            supported,
            kind: kind.unwrap_or_else(|| installer::install_kind().to_string()),
            memory: Memory::load(
                memory_path.unwrap_or_else(|| paths::data_dir().join("updates.json")),
            ),
            auto: false,
            state: UpdateState::default(),
            thread: None,
            cancel: Arc::new(AtomicBool::new(false)),
            due: None,
            finished_tx,
            finished_rx,
            events: Vec::new(),
        }
    }

    // ---------------------------------------------------------------- queries
    pub fn state(&self) -> &UpdateState {
        &self.state
    }

    /// Whether this platform can look for updates at all.
    pub fn available(&self) -> bool {
        (self.supported)()
    }

    /// `installer::KIND_SETUP` (the app can install it), `KIND_PORTABLE` or `KIND_UNSUPPORTED`.
    pub fn kind(&self) -> &str {
        &self.kind
    }

    pub fn busy(&self) -> bool {
        self.thread
            .as_ref()
            .is_some_and(|thread| !thread.is_finished())
    }

    pub fn can_install(&self) -> bool {
        self.kind == installer::KIND_SETUP
            && self
                .state
                .release
                .as_ref()
                .is_some_and(ReleaseInfo::installable)
            && matches!(self.state.phase, Phase::Available | Phase::Failed)
    }

    // --------------------------------------------------------------- lifecycle
    /// Begin: clean up old downloads and, when `auto`, schedule the first check.
    pub fn start(&mut self, auto: bool, now: f64) {
        installer::remove_downloads();
        self.set_auto(auto, now);
    }

    /// Turn the daily check on or off (the setting `updates.check`).
    pub fn set_auto(&mut self, enabled: bool, now: f64) {
        self.auto = enabled && self.available();
        if !self.auto {
            self.due = None;
            return;
        }
        if self.due.is_none() {
            self.due = Some(now + FIRST_CHECK_DELAY_S);
        }
    }

    pub fn shutdown(&mut self) {
        self.due = None;
        self.cancel.store(true, Ordering::SeqCst);
    }

    /// Run what is due; the owner calls this regularly with a monotonic `now` in seconds.
    pub fn tick(&mut self, now: f64) {
        if self.due.is_some_and(|due| now >= due) {
            self.due = None;
            self.scheduled_check(now);
        }
    }

    /// When the timer fires next, if it is running.
    pub fn next_tick(&self) -> Option<f64> {
        self.due
    }

    /// Hand over what happened since the last call (results of the background thread first).
    pub fn drain_events(&mut self) -> Vec<Event> {
        while let Ok(state) = self.finished_rx.try_recv() {
            self.on_finished(state);
        }
        std::mem::take(&mut self.events)
    }

    // ----------------------------------------------------------------- actions
    /// Look for an update because the user asked. Returns `false` when busy
    /// or unavailable on this platform.
    pub fn check_now(&mut self) -> bool {
        self.begin_check(true)
    }

    /// Download, verify and start the installer of the release found. Returns
    /// `false` when nothing can be installed right now.
    pub fn install(&mut self) -> bool {
        let Some(release) = self.state.release.clone() else {
            return false;
        };
        if self.busy() || !self.can_install() {
            return false;
        }
        self.cancel.store(false, Ordering::SeqCst);
        self.set(
            UpdateState::new(Phase::Downloading, Some(release.clone()), true)
                .with_message("Downloading…"),
        );
        self.start_thread(move |worker| worker.download(&release));
        true
    }

    /// Stop a running download.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }

    // --------------------------------------------------------------- internals
    /// The timer fired (`_tick`).
    fn scheduled_check(&mut self, now: f64) {
        if !self.auto {
            return;
        }
        let wall = (self.clock)();
        let due =
            wall - self.memory.last_check >= CHECK_INTERVAL_S || self.memory.last_check > wall;
        if due && !self.busy() {
            self.begin_check(false);
        }
        self.due = Some(now + TICK_INTERVAL_S);
    }

    fn begin_check(&mut self, manual: bool) -> bool {
        if self.busy() || !self.available() {
            return false;
        }
        if matches!(
            self.state.phase,
            Phase::Checking | Phase::Downloading | Phase::Installing
        ) {
            return false; // the result of the one before is still on its way
        }
        self.memory.last_check = (self.clock)();
        self.memory.save();
        let previous = self.state.release.clone();
        self.set(UpdateState::new(Phase::Checking, previous, manual).with_message("Checking…"));
        self.start_thread(move |worker| worker.check(manual));
        true
    }

    fn start_thread(&mut self, work: impl FnOnce(Worker) + Send + 'static) {
        let worker = Worker {
            fetcher_factory: Arc::clone(&self.fetcher_factory),
            running: self.running.clone(),
            cancel: Arc::clone(&self.cancel),
            finished: self.finished_tx.clone(),
        };
        self.thread = std::thread::Builder::new()
            .name("eye-tracker-update".into())
            .spawn(move || work(worker))
            .map_err(|error| log::warn!("Could not start the update thread: {error}"))
            .ok();
    }

    fn set(&mut self, state: UpdateState) {
        self.state = state.clone();
        self.events.push(Event::StateChanged(state));
    }

    /// A result from the worker thread, on the UI thread.
    fn on_finished(&mut self, state: UpdateState) {
        if state.phase == Phase::Downloading {
            if self.state.phase == Phase::Downloading {
                self.set(state); // progress; a cancelled download is no longer shown
            }
            return;
        }
        if state.phase == Phase::Installing {
            self.set(state.clone());
            self.start_installer(&state);
            return;
        }
        self.set(state.clone());
        let Some(release) = state.release else {
            return;
        };
        let told = self.memory.notified == release.version;
        if state.phase == Phase::Available && !told {
            self.memory.notified = release.version.clone();
            self.memory.save();
            if !state.manual {
                // a manual check shows its result in the window
                self.events.push(Event::Announce(release));
            }
        }
    }

    fn start_installer(&mut self, state: &UpdateState) {
        let path = PathBuf::from(state.extra.get("path").map_or("", String::as_str));
        if let Err(error) = (self.launcher)(&path) {
            log::warn!("Could not start the installer: {error}");
            self.set(
                UpdateState::new(Phase::Failed, state.release.clone(), true)
                    .with_message(reason(&error)),
            );
        }
    }
}

/// What the background thread needs of the service.
struct Worker {
    fetcher_factory: FetcherFactory,
    running: String,
    cancel: Arc<AtomicBool>,
    finished: Sender<UpdateState>,
}

impl Worker {
    fn send(&self, state: UpdateState) {
        // The service is gone: nobody is waiting for the result.
        let _ = self.finished.send(state);
    }

    fn check(&self, manual: bool) {
        let result = check_for_update(&mut (self.fetcher_factory)(), &self.running);
        match result {
            Err(error) => {
                log::info!("Update check failed: {error}");
                self.send(
                    UpdateState::new(Phase::Failed, None, manual).with_message(reason(&error)),
                );
            }
            Ok(result) => {
                let phase = if result.newer {
                    Phase::Available
                } else {
                    Phase::UpToDate
                };
                self.send(UpdateState::new(phase, result.release, manual));
            }
        }
    }

    fn download(&self, release: &ReleaseInfo) {
        let mut last_progress: Option<Instant> = None;
        let mut progress = |done: u64, total: u64| {
            let now = Instant::now();
            if last_progress
                .is_none_or(|last| now.duration_since(last).as_secs_f64() >= PROGRESS_INTERVAL_S)
            {
                last_progress = Some(now);
                self.send(UpdateState {
                    done,
                    total,
                    ..UpdateState::new(Phase::Downloading, Some(release.clone()), true)
                        .with_message("Downloading…")
                });
            }
        };
        let cancelled = || self.cancel.load(Ordering::SeqCst);
        let result = installer::download_installer(
            release,
            &mut (self.fetcher_factory)(),
            Some(&mut progress),
            Some(&cancelled),
            None,
        );
        match result {
            Ok(path) => {
                let mut state = UpdateState::new(Phase::Installing, Some(release.clone()), true)
                    .with_message("Installing…");
                state
                    .extra
                    .insert("path".into(), path.to_string_lossy().into_owned());
                self.send(state);
            }
            Err(InstallError::Cancelled) => {
                self.send(UpdateState::new(
                    Phase::Available,
                    Some(release.clone()),
                    true,
                ));
            }
            Err(error) => {
                log::warn!("Update download failed: {error}");
                self.send(
                    UpdateState::new(Phase::Failed, Some(release.clone()), true)
                        .with_message(reason(&error)),
                );
            }
        }
    }
}

fn reason(error: &dyn std::fmt::Display) -> String {
    let text = error.to_string();
    let text = text.trim();
    if text.is_empty() {
        "Something went wrong.".to_string()
    } else {
        text.to_string()
    }
}

#[cfg(test)]
#[path = "service_tests.rs"]
mod tests;
