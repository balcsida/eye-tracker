//! The orchestrator: camera, gaze model, decision engine and OS actions wired together.
//!
//! [`Controller`] lives on the UI thread and owns every moving part:
//!
//! * a vision worker (camera + face analysis on a background thread),
//! * the gaze model from the calibration, smoothed by a One Euro filter,
//! * the pure decision engine ([`super::decision`], [`super::presence`], [`super::guard`],
//!   [`super::scheduler`], [`super::input_state`], [`super::window_memory`],
//!   [`super::camera_yield`]),
//! * the platform services that move the cursor, focus windows and lock the screen,
//! * global hotkeys.
//!
//! # Threading
//!
//! Worker and hotkey callbacks arrive on foreign threads. They are handed to the UI thread
//! through channels that [`Controller::process_queued`] drains (at the start of every
//! [`Controller::tick`]). Locking the screen can block for seconds, so it runs on a
//! short-lived thread whose result comes back the same way; so do the split-pane providers,
//! on the pane worker's thread ([`crate::panes`]). Everything else runs on the UI thread, so
//! no locking is needed beyond the record of the backend the worker built.
//!
//! What the Python app emitted as Qt signals is collected as [`ControllerEvent`]s, which the
//! owner takes with [`Controller::drain_events`].
//!
//! # State
//!
//! The public [`TrackingState`] is *derived* from a small set of independent flags (user
//! pause, privacy mode, session locked, camera yielded, user away, camera error, calibrating,
//! calibration valid), in this priority order:
//!
//! ```text
//! PRIVACY > LOCKED > CALIBRATING > PAUSED > YIELDED > AWAY > CAMERA_ERROR
//!         > NEEDS_CALIBRATION > TRACKING
//! ```
//!
//! Deriving instead of transitioning means overlapping causes (paused *and* the session
//! locked, say) resolve naturally when one of them ends. Whether the camera is open follows
//! directly from the state (`TrackingState::camera_active`). Before the camera reopens after a
//! pause, privacy mode or a locked session, the "another app uses the camera" check runs
//! first, so a call that started meanwhile never has its camera grabbed.
//!
//! # Calibrations
//!
//! The calibration file holds one profile per setup (monitor layout, vision backend, camera;
//! see [`CalibrationLibrary`]). The controller uses the profile that fits the current setup
//! and switches profiles by itself when the setup changes (a laptop moving between docks), so
//! a recalibration is only needed for a setup that was never calibrated. When the calibration
//! becomes unusable `calibration_required` is emitted; if the user cannot be interrupted at
//! that moment (away, locked, privacy mode, displays off) the announcement is delivered once
//! they are back and the monitor layout has settled.
//!
//! # Split panes (experimental)
//!
//! With `panes.enabled` the focused window is followed as well: its application is
//! identified, its panes are asked for (once when it gets focus, then about every
//! [`PANE_REFRESH_S`] while the gaze is inside it) and, while the monitor decider finds the
//! gaze on the cursor's monitor, [`PaneDecider`] decides when the keyboard focus moves to
//! another pane. That asks the terminal to focus the pane; no window is activated. Once it
//! reports success the cursor follows into the pane (`panes.move_cursor`), through the same
//! announced warp as a monitor switch. Pane focus waits `panes.after_monitor_switch_ms` after a
//! monitor switch.
//!
//! # Unreliable pointer position
//!
//! On Wayland the pointer position can be stale (an XWayland client only sees the pointer
//! over X11 windows). There the monitor the pointer is on is taken to be the target of the
//! last successful switch, and nothing is learned from the pointer position (manual moves,
//! implicit labels, undone switches).
//!
//! # Housekeeping
//!
//! The owner calls [`Controller::tick`] every [`Controller::next_tick_in`]: 100 ms while
//! tracking (cursor polling resolution for the mouse/typing guards) and 500 ms otherwise.
//! Each tick polls input, records the focused window per monitor, checks for a locked session
//! and for other apps wanting the camera (on slower schedules), advances the walk-away timers,
//! adapts the camera frame rate, delivers deferred calibration announcements, publishes
//! statistics and, once a second, looks for a changed monitor layout.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};
use parking_lot::Mutex;
use serde_json::{Map, Value, json};

use super::camera_yield::{YieldInputs, should_yield};
use super::decision::{Decision, SwitchConfig, SwitchDecider};
use super::guard::{FaceBox, GuardConfig, GuardEvent, ShoulderGuard};
use super::input_state::InputTracker;
use super::presence::{
    PresenceConfig, PresenceEvent, PresenceEventKind, PresenceMonitor, PresenceState,
};
use super::scheduler::{PROFILES, RateContext, RatePolicy};
use super::window_memory::{WindowMemory, choose_cursor_target};
use crate::config::{Settings, atomic_write_text};
use crate::gaze::filters::PointFilter;
use crate::gaze::learning::{
    DriftMonitor, IMPLICIT_MAX_ERROR, ImplicitLearner, LearnerOptions, plausible_label, refit_model,
};
use crate::gaze::model::{GazeModel, LOOK_AWAY_EXCESS, LOOK_AWAY_MARGIN, gaze_feature_indices};
use crate::gaze::store::{CalibrationData, CalibrationLibrary, axis_error};
use crate::panes::decider::{
    FALLBACK_GAZE_ERROR_PX, PaneConfig, PaneDecider, PaneDecision, eligible_panes,
};
use crate::panes::providers::command::{WindowKey, window_key};
use crate::panes::registry::{PaneRegistry, current_system, default_providers, is_denied};
use crate::panes::types::{ClientRect, Pane, PaneId, PaneSnapshot};
use crate::panes::worker::{
    DetectResult, FocusResult, PaneEvent, PaneWorker, PaneWorkerOptions, monotonic,
};
use crate::platform::base::PlatformServices;
use crate::platform::fallback_screen_name;
use crate::platform::hotkeys::{
    HotkeyArg, HotkeyCallback, HotkeyManager, create_hotkey_manager, format_hotkey,
};
use crate::trace::TraceWriter;
use crate::types::{
    AppIdentity, GazePoint, Monitor, Observation, Rect, TrackingState, WindowRef, WorkerStats,
    layout_signature, monitor_at, nearest_monitor,
};
use crate::vision::Frame;
use crate::vision::backends::base::VisionBackend;
use crate::vision::camera::FrameSource;
use crate::vision::worker::{
    BackendFactory, Clock, FactoryError, SourceFactory, VisionWorker, WorkerEvent, WorkerOptions,
};
use crate::{VERSION, paths};

/// Housekeeping tick while tracking: the resolution of mouse/typing detection.
pub const TICK_ACTIVE_MS: u64 = 100;
/// Housekeeping tick in every other state.
pub const TICK_IDLE_MS: u64 = 500;
/// How often the focused window is sampled (remembered per monitor).
pub const WINDOW_POLL_S: f64 = 0.5;
/// How often the session lock state is checked.
pub const LOCK_POLL_S: f64 = 2.0;
/// How often other apps' camera use and the pause-for-apps list are checked.
pub const YIELD_POLL_S: f64 = 3.0;
/// How often `stats_changed` is emitted.
pub const STATS_PERIOD_S: f64 = 2.0;
/// Screen hot-plug events come in bursts (and displays drop out during sleep);
/// the layout is re-read once they have settled.
pub const SCREEN_DEBOUNCE_MS: u64 = 1000;
/// How often the monitor layout is read to notice hot-plugging (Qt's screen signals in the
/// Python app).
pub const MONITOR_POLL_S: f64 = 1.0;
/// The worker interval is only updated when it changes by more than this fraction.
pub const RATE_TOLERANCE: f64 = 0.05;
/// Moving the mouse back to the previous monitor this soon after an automatic
/// switch marks that switch as wrong (drift evidence); a switch left alone this
/// long counts as correct.
pub const WRONG_SWITCH_S: f64 = 2.0;
/// During a blink the last gaze point is held for this long, so a blink does not
/// restart the dwell toward another monitor.
pub const BLINK_HOLD_S: f64 = 0.5;
/// A longer gap without a usable gaze restarts the smoothing filter, so the next
/// estimate does not glide in from a stale position.
pub const FILTER_RESET_S: f64 = 1.0;
/// A filtered gaze jump larger than this fraction of the smallest monitor side
/// counts as "gaze moving" (the scheduler samples faster for a moment).
pub const GAZE_MOVE_FRACTION: f64 = 0.1;
pub const GAZE_MOVING_HOLD_S: f64 = 0.5;
/// Without an observation for this long (or 2.5 frame intervals, if longer) the
/// camera verdict is unknown and the walk-away timers freeze.
pub const STALE_OBSERVATION_S: f64 = 2.5;
/// Keyboard activity this recent selects the scheduler's "typing" rate.
pub const TYPING_RATE_WINDOW_S: f64 = 2.0;
/// Minimum time between two "accuracy dropped" notifications.
pub const DRIFT_ALERT_COOLDOWN_S: f64 = 1800.0;
/// Consecutive observations the gaze model must reject (wrong feature vector)
/// before the calibration is declared unusable. After a backend change a few
/// frames analysed by the old backend are still on their way; they must not
/// invalidate a calibration that fits the new one.
pub const FEATURE_MISMATCH_LIMIT: u32 = 5;
/// After the user unlocks a session the shoulder guard locked, a second face in
/// view shows the privacy curtain instead of locking again: otherwise a colleague
/// sitting down next to the user would lock the screen again two seconds after
/// every unlock. The grace lasts this long after the unlock and after every later
/// trigger it absorbs, so it holds while the colleague keeps coming back into
/// view (turning to the user, looking down at notes) and ends once nobody has
/// looked over the user's shoulder for this long.
pub const GUARD_RELOCK_GRACE_S: f64 = 300.0;
/// Consecutive refused pointer moves after which switching pauses (backing off
/// from [`WARP_BACKOFF_S`], doubling up to [`WARP_BACKOFF_MAX_S`]) instead of
/// retrying after every cooldown.
pub const WARP_REFUSAL_LIMIT: u32 = 3;
pub const WARP_BACKOFF_S: f64 = 60.0;
pub const WARP_BACKOFF_MAX_S: f64 = 1800.0;
/// A deferred "calibration needed" announcement is delivered once the user could
/// be interrupted for this long and no monitor change is pending: displays that
/// drop out during sleep come back unchanged and must not prompt.
pub const ANNOUNCE_SETTLE_S: f64 = 2.0 * SCREEN_DEBOUNCE_MS as f64 / 1000.0;
/// When the user comes back (from away, a locked session or privacy mode) to a
/// calibration that is still unusable, they are reminded, at most this often.
pub const CALIBRATION_REMINDER_S: f64 = 1800.0;
/// Blindness (lens covered, shutter closed, dark room) that begins within this
/// many seconds of seeing the user's face or their input means the user covered
/// the camera: walk-away detection then pauses instead of counting "no face".
pub const BLIND_ONSET_S: f64 = 3.0;
/// ...but for at most this long without any keyboard or mouse input: someone who
/// switched the desk lamp off and left must still be locked out eventually.
pub const BLIND_FREEZE_MAX_S: f64 = 300.0;
/// Minimum time between two "camera appears covered" notifications.
pub const BLIND_NOTICE_COOLDOWN_S: f64 = 600.0;
/// The worker reports a camera failure only when its message changes, so a camera
/// that still fails the same way after a pause would go unnoticed. With no frame
/// this long after the camera was switched on, the worker's error is read back.
pub const CAMERA_ERROR_RECHECK_S: f64 = 8.0;
/// Consecutive failed window activations after which a missing Accessibility
/// permission (macOS) is reported.
pub const ACTIVATION_FAILURE_LIMIT: u32 = 3;
/// How long [`Controller::shutdown`] waits for a screen lock still in progress.
pub const LOCK_JOIN_TIMEOUT_S: f64 = 2.0;
/// While the gaze is inside the focused terminal window its panes are asked for
/// again this often (at the next window poll), so the layout the pane decider
/// sees is never much older than its freshness limit (1.5 s).
pub const PANE_REFRESH_S: f64 = 1.0;
/// How long [`Controller::shutdown`] waits for a pane provider call in progress.
pub const PANE_JOIN_TIMEOUT_S: f64 = 2.0;
/// Windows whose pane cursor positions are remembered (least recently used dropped).
pub const PANE_CURSOR_WINDOWS: usize = 16;
/// Walk-away actions that lock the session or blank the displays. Until the
/// first-run setup is finished they only notify: the owner of a PC who never
/// saw the setup assistant must not be locked out by an app they just installed.
const DISRUPTIVE_AWAY_ACTIONS: [&str; 3] = ["lock", "lock_and_display_off", "display_off"];

/// The walk-away action that really runs: `settings.presence.action`, except
/// that a lock or blanked displays only notify while the first-run setup is not
/// finished (see `DISRUPTIVE_AWAY_ACTIONS`). The countdown toast announces this
/// one, so it never says "Locking" before a mere notification.
pub fn effective_away_action(settings: &Settings) -> &str {
    let action = settings.presence.action.as_str();
    if !settings.general.first_run_done && DISRUPTIVE_AWAY_ACTIONS.contains(&action) {
        return "notify";
    }
    action
}

/// IPC commands handled by [`Controller::handle_command`].
pub const COMMANDS: [&str; 11] = [
    "show",
    "settings",
    "pause",
    "resume",
    "toggle",
    "privacy-on",
    "privacy-off",
    "privacy-toggle",
    "calibrate",
    "status",
    "quit",
];
/// Commands that only the UI can carry out; forwarded through `ui_requested`.
pub const UI_COMMANDS: [&str; 3] = ["show", "settings", "quit"];

/// Hotkey setting names (`Settings::hotkeys.<name>`) in registration order.
pub const HOTKEY_ACTIONS: [&str; 3] = ["toggle_tracking", "toggle_privacy", "recalibrate"];

fn camera_off(state: TrackingState) -> bool {
    matches!(
        state,
        TrackingState::Paused
            | TrackingState::Privacy
            | TrackingState::Locked
            | TrackingState::Yielded
    )
}

/// Camera-off states in which the camera-in-use check is skipped; leaving one
/// re-runs it before the camera opens.
fn unchecked_camera_off(state: TrackingState) -> bool {
    matches!(
        state,
        TrackingState::Paused | TrackingState::Privacy | TrackingState::Locked
    )
}

/// States the user "comes back" from; a still unusable calibration is then
/// announced again (see [`CALIBRATION_REMINDER_S`]).
fn absent_state(state: TrackingState) -> bool {
    matches!(
        state,
        TrackingState::Away | TrackingState::Locked | TrackingState::Privacy
    )
}

/// States in which the app may be napped by macOS: the user stopped tracking,
/// and only their own action (hotkey, tray, IPC) resumes it. A locked session is
/// not one: its end must be noticed promptly.
fn nappable_state(state: TrackingState) -> bool {
    matches!(state, TrackingState::Paused | TrackingState::Privacy)
}

// ---------------------------------------------------------------------------- events
/// What the Python controller emitted as Qt signals, one variant per signal.
#[derive(Debug, Clone)]
pub enum ControllerEvent {
    /// The tracking state changed.
    StateChanged(TrackingState),
    /// New smoothed gaze estimate, or `None` when lost.
    GazeChanged(Option<GazePoint>),
    /// Every observation from the camera while it is on (calibration UI, preview, wizard).
    Observation(Box<Observation>),
    /// Periodic statistics (see [`Controller::stats_snapshot`]).
    StatsChanged(Map<String, Value>),
    /// A notification for the tray.
    Notify { title: String, message: String },
    /// The cursor was moved to the monitor with this index.
    Switched(i32),
    /// Walk-away countdown started: seconds until the away action.
    AwayWarning(f64),
    /// The countdown is over (the user came back, or it elapsed); hide the toast.
    AwayCancelled,
    /// Shoulder-guard curtain on/off (for the "curtain" action, and as the fallback
    /// of the "lock" action).
    GuardChanged(bool),
    /// Annotated camera frame while the preview is enabled.
    PreviewFrame(Frame),
    /// A (re)calibration is needed or was requested; the argument is the reason.
    ///
    /// Emitted for explicit requests (`"hotkey"`, `"ipc"`), when a usable
    /// calibration becomes unusable (deferred until the user can be interrupted),
    /// and as a reminder, at most every [`CALIBRATION_REMINDER_S`], when the
    /// user comes back from away, a locked session or privacy mode to a
    /// calibration that is still unusable. The same reason can therefore arrive
    /// more than once; each emission is meant to be shown.
    CalibrationRequired(String),
    /// Settings were applied (the new settings).
    SettingsChanged(Box<Settings>),
    /// An IPC command only the UI can perform: `"show"`, `"settings"` or `"quit"`.
    UiRequested(String),
    /// A feature is blocked by an OS permission (`"camera"` or `"accessibility"`).
    /// Emitted right after the `Notify` that explains it (also when informational
    /// notifications are turned off and that `Notify` was not emitted), so the UI can
    /// open `platform.open_permission_settings(name)` when the notification is clicked.
    PermissionNeeded(String),
}

// ---------------------------------------------------------------------------- seams
/// The part of [`VisionWorker`] the controller uses.
pub trait WorkerLike {
    fn start(&self);
    fn stop(&self, timeout: f64);
    fn set_interval(&self, seconds: f64);
    fn set_active(&self, active: bool);
    fn set_max_faces(&self, n: usize);
    fn set_motion_gate(&self, enabled: bool, threshold: f64);
    fn set_preview(&self, enabled: bool);
    fn reconfigure(
        &self,
        source_factory: Option<SourceFactory>,
        backend_factory: Option<BackendFactory>,
    );
    fn stats(&self) -> WorkerStats;
    fn backend_info(&self) -> Option<(String, String)>;
}

impl WorkerLike for VisionWorker {
    fn start(&self) {
        VisionWorker::start(self);
    }

    fn stop(&self, timeout: f64) {
        self.stop_within(timeout);
    }

    fn set_interval(&self, seconds: f64) {
        if VisionWorker::set_interval(self, seconds).is_err() {
            log::debug!("Ignored an invalid worker interval");
        }
    }

    fn set_active(&self, active: bool) {
        VisionWorker::set_active(self, active);
    }

    fn set_max_faces(&self, n: usize) {
        VisionWorker::set_max_faces(self, n);
    }

    fn set_motion_gate(&self, enabled: bool, threshold: f64) {
        VisionWorker::set_motion_gate(self, enabled, threshold);
    }

    fn set_preview(&self, enabled: bool) {
        VisionWorker::set_preview(self, enabled);
    }

    fn reconfigure(
        &self,
        source_factory: Option<SourceFactory>,
        backend_factory: Option<BackendFactory>,
    ) {
        VisionWorker::reconfigure(self, source_factory, backend_factory);
    }

    fn stats(&self) -> WorkerStats {
        VisionWorker::stats(self)
    }

    fn backend_info(&self) -> Option<(String, String)> {
        VisionWorker::backend_info(self)
    }
}

/// The part of [`PaneWorker`] the controller uses.
pub trait PaneWorkerLike {
    fn start(&mut self);
    fn stop(&mut self, timeout: f64);
    fn request_stop(&mut self);
    fn set_registry(&self, registry: PaneRegistry);
    fn disabled_providers(&self) -> HashSet<String>;
    fn drain_events(&self) -> Vec<PaneEvent>;
    fn request_detect(&self, window: &WindowRef, app: &AppIdentity) -> u64;
    fn request_focus(&self, window: &WindowRef, pane: &Pane);
}

impl PaneWorkerLike for PaneWorker {
    fn start(&mut self) {
        PaneWorker::start(self);
    }

    fn stop(&mut self, timeout: f64) {
        PaneWorker::stop(self, timeout);
    }

    fn request_stop(&mut self) {
        PaneWorker::request_stop(self);
    }

    fn set_registry(&self, registry: PaneRegistry) {
        PaneWorker::set_registry(self, registry);
    }

    fn disabled_providers(&self) -> HashSet<String> {
        PaneWorker::disabled_providers(self)
    }

    fn drain_events(&self) -> Vec<PaneEvent> {
        PaneWorker::drain_events(self)
    }

    fn request_detect(&self, window: &WindowRef, app: &AppIdentity) -> u64 {
        PaneWorker::request_detect(self, window, app)
    }

    fn request_focus(&self, window: &WindowRef, pane: &Pane) {
        PaneWorker::request_focus(self, window, pane);
    }
}

/// `(source_factory, backend_factory, events) -> worker`; the worker reports through `events`.
pub type WorkerFactory =
    Box<dyn FnMut(SourceFactory, BackendFactory, Sender<WorkerEvent>) -> Box<dyn WorkerLike>>;
/// `() -> monitors`.
pub type MonitorsProvider = Box<dyn FnMut() -> Vec<Monitor>>;
/// `() -> hotkey manager`.
pub type HotkeyManagerFactory = Box<dyn FnMut() -> Arc<dyn HotkeyManager>>;
/// A screen lock request, run off the UI thread by a [`JobRunner`].
pub type LockJob = Box<dyn FnOnce() + Send>;
/// Runs a job off the GUI thread (see [`ControllerOptions::lock_runner`]).
pub type JobRunner = Box<dyn FnMut(LockJob)>;
/// Builds the pane providers for settings (see [`ControllerOptions::pane_registry_factory`]).
pub type PaneRegistryFactory = Box<dyn FnMut(&Settings) -> PaneRegistry>;
/// Builds the pane worker for a registry.
pub type PaneWorkerFactory = Box<dyn FnMut(PaneRegistry) -> Box<dyn PaneWorkerLike>>;
/// `(settings, max_faces) -> backend factory` (the Python module's `_backend_factory`).
pub type BackendFactoryMaker = Arc<dyn Fn(&Settings, usize) -> BackendFactory + Send + Sync>;

/// The injectable parts of a [`Controller`]; every `None` takes the real one.
#[derive(Default)]
pub struct ControllerOptions {
    /// Defaults to a [`VisionWorker`].
    pub worker_factory: Option<WorkerFactory>,
    /// Defaults to [`crate::platform::monitors`].
    pub monitors_provider: Option<MonitorsProvider>,
    /// Monotonic time source in seconds.
    pub clock: Option<Clock>,
    /// Defaults to [`create_hotkey_manager`].
    pub hotkey_manager_factory: Option<HotkeyManagerFactory>,
    /// Write a per-frame trace (JSON lines) to this file.
    pub trace_path: Option<PathBuf>,
    /// Runs the lock job off the GUI thread, where the screen is locked (`lock_screen` can
    /// block for seconds); defaults to a short-lived thread. Tests pass a synchronous runner.
    pub lock_runner: Option<JobRunner>,
    /// `(settings) -> PaneRegistry` with the split-pane providers; defaults to
    /// `panes::registry::default_providers`.
    pub pane_registry_factory: Option<PaneRegistryFactory>,
    /// `(registry) -> worker`; defaults to a [`PaneWorker`] with its own thread. Tests pass a
    /// synchronous one.
    pub pane_worker_factory: Option<PaneWorkerFactory>,
    /// Builds the vision backend factories; defaults to [`backend_factory`]. Tests replace it
    /// (the Python tests monkeypatched `_backend_factory`).
    pub backend_factory: Option<BackendFactoryMaker>,
}

// --------------------------------------------------------------------------- helpers
/// The monitor of the layout `after` that is most likely the physical monitor `old` of the
/// layout `before`.
///
/// Indices follow the platform's screen order and say nothing once a screen was added or
/// removed. A screen's name (its connector or device name) follows the monitor even when the
/// desktop is laid out anew, so a name that is unique in both layouts decides first - unless
/// it is merely a positional fallback name. Then the same rectangle, then whichever monitor
/// now covers the old centre.
pub(crate) fn same_monitor<'a>(
    old: &Monitor,
    before: &[Monitor],
    after: &'a [Monitor],
) -> Option<&'a Monitor> {
    if let Some(exact) = after
        .iter()
        .find(|m| m.name == old.name && m.rect == old.rect)
    {
        return Some(exact);
    }
    let named: Vec<&Monitor> = after.iter().filter(|m| m.name == old.name).collect();
    if named.len() == 1
        && before.iter().filter(|m| m.name == old.name).count() == 1
        && old.name != fallback_screen_name(old.index)
    {
        return Some(named[0]);
    }
    if let Some(same_rect) = after.iter().find(|m| m.rect == old.rect) {
        return Some(same_rect);
    }
    let (cx, cy) = old.rect.center();
    monitor_at(after, cx, cy)
}

/// The default worker: a [`VisionWorker`] on the given clock.
pub fn make_vision_worker(
    source_factory: SourceFactory,
    backend_factory: BackendFactory,
    events: Sender<WorkerEvent>,
    clock: Clock,
) -> Box<dyn WorkerLike> {
    Box::new(VisionWorker::new(
        source_factory,
        backend_factory,
        events,
        WorkerOptions {
            clock,
            ..WorkerOptions::default()
        },
    ))
}

/// Capture rate to request: 15 fps unless the profile analyses faster than that.
///
/// A lower capture rate lets the camera expose longer in dim light (less noise,
/// steadier landmarks), and 15 fps covers every mode of eco and balanced except
/// calibration, which works fine with 15 samples per second.
pub fn camera_fps(settings: &Settings) -> u32 {
    let rates = PROFILES
        .iter()
        .find(|(name, _)| *name == settings.performance.profile)
        .or_else(|| PROFILES.iter().find(|(name, _)| *name == "balanced"))
        .map(|(_, rates)| rates.active)
        .unwrap_or(0.0);
    if rates > 15.0 { 30 } else { 15 }
}

/// The camera the settings select, created (not opened) by the worker.
pub fn source_factory(settings: &Settings) -> SourceFactory {
    let cam = &settings.camera;
    let device = cam.device.clone();
    let width = usize::try_from(cam.width).unwrap_or(0);
    let height = usize::try_from(cam.height).unwrap_or(0);
    let api = cam.api.clone();
    let fps = camera_fps(settings);
    Box::new(move || {
        let source = crate::vision::camera::open_source(&device, width, height, &api, fps, true)?;
        Ok(Box::new(source) as Box<dyn FrameSource>)
    })
}

/// The vision backend the settings select, created by the worker.
pub fn backend_factory(settings: &Settings, max_faces: usize) -> BackendFactory {
    let name = settings.general.backend.clone();
    Box::new(move || {
        crate::vision::backends::create_backend(&name, max_faces).map_err(FactoryError::from)
    })
}

/// Indices of a backend's gaze-direction features.
fn backend_gaze_indices(feature_names: &[&str], gaze_features: &[&str]) -> Option<Vec<usize>> {
    gaze_feature_indices(feature_names, gaze_features)
}

/// [`backend_gaze_indices`] for a backend known only by name (`None` if unknown).
fn named_backend_gaze_indices(name: &str) -> Option<Vec<usize>> {
    let class = crate::vision::backends::backend_class(name).ok()?;
    backend_gaze_indices(class.feature_names(), class.gaze_features())
}

/// A backend the worker built through one of our factories.
#[derive(Debug, Clone)]
struct BuiltBackend {
    /// Which factory built it (see `Controller::make_backend_factory`).
    generation: u64,
    /// `(name, feature_version)`.
    info: (String, String),
    /// Indices of its gaze-direction features (`None`: not declared).
    gaze_indices: Option<Vec<usize>>,
}

/// Whether privacy mode was on when the app last ran (`paths::state_file()`).
///
/// An unreadable file counts as "off": it is only ever written by
/// [`save_privacy_mode`], so damage means it never held a choice.
fn saved_privacy_mode() -> bool {
    let text = match std::fs::read_to_string(paths::state_file()) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return false,
        Err(e) => {
            log::warn!("Could not read the remembered privacy mode: {e}");
            return false;
        }
    };
    match serde_json::from_str::<Value>(&text) {
        Ok(data) => data.get("privacy_mode") == Some(&Value::Bool(true)),
        Err(e) => {
            log::warn!("Could not read the remembered privacy mode: {e}");
            false
        }
    }
}

/// Remember privacy mode for the next start (never fails).
fn save_privacy_mode(enabled: bool) {
    let text = format!("{}\n", json!({ "privacy_mode": enabled }));
    if let Err(e) = atomic_write_text(&paths::state_file(), &text) {
        log::warn!("Could not remember privacy mode for the next start: {e}");
    }
}

fn layout_key(monitors: &[Monitor]) -> Vec<(i32, i32, i32, i32, i32)> {
    monitors
        .iter()
        .map(|m| (m.index, m.rect.x, m.rect.y, m.rect.w, m.rect.h))
        .collect()
}

/// The provider switches of the split-pane settings.
fn pane_providers(settings: &Settings) -> (bool, bool, bool, bool) {
    let p = &settings.panes;
    (p.tmux, p.wezterm, p.windows_terminal, p.desktop_apps)
}

/// A pane id as the trace stores it.
fn trace_id(value: &PaneId) -> Value {
    match value {
        PaneId::Int(i) => json!(i),
        PaneId::Str(s) => json!(s),
        other => json!(other.to_string()),
    }
}

/// Python's `round(value, digits)`.
fn round_to(value: f64, digits: i32) -> f64 {
    let scale = 10f64.powi(digits);
    (value * scale).round_ties_even() / scale
}

fn finite(value: f64, digits: i32) -> Option<f64> {
    value.is_finite().then(|| round_to(value, digits))
}

/// `size` if both sides are positive, else `None` (unknown).
fn known_size(size: (u32, u32)) -> Option<(u32, u32)> {
    (size.0 > 0 && size.1 > 0).then_some(size)
}

/// The combination configured for a hotkey action (`""` for an unknown name).
fn hotkey_setting<'a>(settings: &'a Settings, name: &str) -> &'a str {
    let hk = &settings.hotkeys;
    match name {
        "toggle_tracking" => &hk.toggle_tracking,
        "toggle_privacy" => &hk.toggle_privacy,
        "recalibrate" => &hk.recalibrate,
        _ => "",
    }
}

/// `(name, feature_version)` of a vision backend.
type BackendIdentity = (String, String);
/// The cached gaze error of a pane's monitor: `(calibration serial, monitor index, monitor
/// rect, (x, y) error)`.
type PaneSigma = (u64, i32, Rect, Option<(f64, f64)>);
/// A window's remembered pane cursor positions: `(key, window, pane id -> position)`.
type PaneCursor = (WindowKey, WindowRef, HashMap<PaneId, (i32, i32)>);

/// Which saved calibration fits the current setup (see `Controller::match_profile`).
enum ProfileMatch {
    /// The profile in use.
    Current,
    /// Another saved profile.
    Other(Box<CalibrationData>),
    /// None; why.
    Unusable(String),
}

// ------------------------------------------------------------------------ controller
/// UI-thread orchestrator of tracking, switching, presence and privacy features.
///
/// Built with [`Controller::new`]; call [`start`](Self::start) once the owner is ready to take
/// events, [`tick`](Self::tick) every [`next_tick_in`](Self::next_tick_in), and
/// [`shutdown`](Self::shutdown) before exit.
pub struct Controller {
    trace: Option<TraceWriter>,
    last_decision: Option<Decision>,
    last_pane_decision: Option<PaneDecision>,
    settings: Settings,
    platform: Arc<dyn PlatformServices>,
    worker_factory: WorkerFactory,
    monitors_provider: MonitorsProvider,
    clock: Clock,
    hotkey_factory: HotkeyManagerFactory,
    backend_factory_maker: BackendFactoryMaker,
    events: Vec<ControllerEvent>,

    monitors: Vec<Monitor>,
    min_side: f64,
    state: TrackingState,
    started: bool,
    closed: bool,

    // Independent causes from which the state is derived (see derive_state).
    paused: bool,
    privacy: bool,
    // Privacy mode was switched back on at start (remember_privacy_mode).
    privacy_restored: bool,
    calibrating: bool,
    session_locked: bool,
    yield_reason: String,
    away: bool,
    camera_error: bool,
    // Privacy mode as it was when a calibration began (restored afterwards).
    privacy_before_calibration: bool,
    // Consumers of preview frames (see set_preview).
    preview_owners: HashSet<String>,

    // Decision engine.
    decider: SwitchDecider,
    presence: PresenceMonitor,
    guard: ShoulderGuard,
    policy: RatePolicy,
    input: InputTracker,
    memory: WindowMemory,
    filter: PointFilter,
    learner: ImplicitLearner,
    drift: DriftMonitor,

    // Split-pane focus (experimental, see crate::panes). The worker and its thread exist only
    // while the feature is on.
    pane_decider: PaneDecider,
    pane_registry_factory: PaneRegistryFactory,
    pane_worker_factory: PaneWorkerFactory,
    pane_worker: Option<Box<dyn PaneWorkerLike>>,
    pane_capable: Option<bool>,     // the platform's "panes" capability
    pane_window: Option<WindowRef>, // the focused window being followed
    pane_app: Option<AppIdentity>,
    pane_snapshot: Option<PaneSnapshot>,
    pane_requested_at: f64,
    // Detection results are accepted only for requests made after the last window change or
    // provider swap (ids come from the current worker).
    pane_last_request: u64,
    pane_valid_from: u64,
    pane_sigma_cache: Option<PaneSigma>,
    pane_sigma_used: Option<(f64, f64)>,
    pane_switches: u64,
    // Where the cursor was last seen in each pane (panes.move_cursor): window key -> (the
    // window, pane id -> position), most recently used last.
    pane_cursor: Vec<PaneCursor>,

    // Calibration: every saved profile, and the one in use (a copy of its library entry).
    library: CalibrationLibrary,
    calibration: Option<CalibrationData>,
    // Bumped whenever another calibration is taken into use (the Python object identity).
    calibration_serial: u64,
    // Set only while the calibration is usable (the Python `_model is not None`).
    model_usable: bool,
    calibration_reason: String,
    implicit_dirty: bool,
    feature_mismatches: u32,
    // "Calibration needed" announcements (see announce_calibration).
    announce_pending: Option<String>,
    interruptible_since: Option<f64>,
    last_announce: f64,

    // What we know about the vision backend (for calibration compatibility).
    reported_backend: Option<(String, String)>,
    expected_cache: RefCell<Option<(String, Option<BackendIdentity>)>>,
    // Every backend factory handed to the worker gets a generation number; the factory
    // records what it built (on the worker thread, hence the lock).
    backend_generation: u64,
    built_backend: Arc<Mutex<Option<BuiltBackend>>>,

    // Worker.
    worker: Option<Box<dyn WorkerLike>>,
    worker_tx: Sender<WorkerEvent>,
    worker_rx: Receiver<WorkerEvent>,
    // Worker events drained early (see discard_queued_previews), handled first.
    held_worker_events: Vec<WorkerEvent>,
    interval: Option<f64>,
    mode: &'static str,
    worker_stats: WorkerStats,
    camera_error_message: Option<String>,
    camera_on_since: Option<f64>,
    frame_size: Option<(u32, u32)>, // of the latest analysed frame

    // Tracking transients.
    last_obs_time: Option<f64>,
    last_face: Option<bool>,
    face_seen_at: Option<f64>,
    last_gaze: Option<(f64, f64)>,
    last_gaze_time: Option<f64>,
    looking_away: bool,
    gaze_none_sent: bool,
    gaze_moving_until: f64,
    pending: bool,
    last_switch: Option<(f64, Option<i32>, i32)>,
    switch_count: u64,

    // Pointer: whether its reported position can be trusted (not on Wayland), and otherwise
    // the monitor we last moved it to.
    cursor_reliable: bool,
    assumed_monitor: Option<i32>,
    // Refused pointer moves (see move_cursor).
    warp_refusals: u32,
    warp_suspensions: u32,
    switching_suspended_until: f64,
    cursor_warning_shown: bool,
    // Window activation failures (macOS Accessibility, see check_accessibility).
    activation_failures: u32,
    accessibility_notice_shown: bool,
    background_active: Option<bool>,

    // Blind camera (see face_verdict).
    blind_since: Option<f64>,
    blind_freeze: bool,
    blind_notice_at: f64,

    // Presence / guard side effects.
    warning_shown: bool,
    // The walk-away action left the displays off and the session unlocked, so the displays
    // are to be woken when the user returns (wake_on_return).
    displays_off_only: bool,
    // The walk-away action turned the displays off (whether or not it also locked); a lock
    // that fails afterwards leaves them "off only".
    away_displays_off: bool,
    curtain: bool,
    guard_locked: bool,
    guard_relock_until: f64,
    // Screen lock requests in flight ("away", "guard"; see request_lock).
    lock_runner: Option<JobRunner>,
    lock_purposes: HashSet<&'static str>,
    lock_thread: Option<JoinHandle<()>>,
    lock_tx: Sender<bool>,
    lock_rx: Receiver<bool>,

    // Housekeeping schedule (set in start()).
    next_window_poll: f64,
    next_lock_check: f64,
    next_yield_check: f64,
    yield_checked_at: f64,
    next_stats: f64,
    process: Option<sysinfo::System>,
    cpu_count: usize,
    cpu_percent: Option<f64>,
    last_stats: Map<String, Value>,

    hotkeys: Option<Arc<dyn HotkeyManager>>,
    hotkeys_suspended: bool,
    /// (hotkey name, combination) pairs whose failure was announced. The hotkeys are
    /// registered again after every stay in Settings → Hotkeys (see suspend_hotkeys); a
    /// failure that has not changed is not repeated.
    announced_hotkey_failures: HashSet<(String, String)>,
    hotkey_tx: Sender<String>,
    hotkey_rx: Receiver<String>,

    // Monitor hot-plug: the layout as last read by the poll, when the next poll is due, and
    // when the debounced refresh runs (the Python `_screen_timer`).
    polled_monitors: Vec<Monitor>,
    next_monitor_poll: f64,
    screen_refresh_due: Option<f64>,
}

impl Controller {
    /// A controller for `settings` (copied; read them back through [`settings`](Self::settings))
    /// and the OS integration `platform` (`crate::platform::get_platform()`).
    pub fn new(
        settings: &Settings,
        platform: Arc<dyn PlatformServices>,
        options: ControllerOptions,
    ) -> Self {
        let clock = options.clock.unwrap_or_else(|| Arc::new(monotonic));
        let s = settings.clone();
        let now = clock();
        let worker_factory = options.worker_factory.unwrap_or_else(|| {
            let clock = clock.clone();
            Box::new(move |source, backend, events| {
                make_vision_worker(source, backend, events, clock.clone())
            })
        });
        let monitors_provider = options
            .monitors_provider
            .unwrap_or_else(|| Box::new(crate::platform::monitors));
        let hotkey_factory = options
            .hotkey_manager_factory
            .unwrap_or_else(|| Box::new(|| Arc::from(create_hotkey_manager())));
        let pane_registry_factory = options.pane_registry_factory.unwrap_or_else(|| {
            let platform = platform.clone();
            Box::new(move |settings: &Settings| default_pane_registry(&platform, settings))
        });
        let pane_worker_factory = options.pane_worker_factory.unwrap_or_else(|| {
            let clock = clock.clone();
            Box::new(move |registry| {
                Box::new(PaneWorker::new(
                    registry,
                    PaneWorkerOptions {
                        clock: clock.clone(),
                        ..PaneWorkerOptions::default()
                    },
                )) as Box<dyn PaneWorkerLike>
            })
        });
        let idle = platform.clone();
        let key_idle = platform.clone();
        let input = InputTracker::new(
            Box::new(move || idle.seconds_since_input()),
            Box::new(move || key_idle.seconds_since_key_input()),
        );
        let learner = ImplicitLearner::new(LearnerOptions {
            max_samples: usize::try_from(s.learning.max_samples).unwrap_or(0),
            ..LearnerOptions::default()
        })
        .unwrap_or_default();
        let (worker_tx, worker_rx) = crossbeam_channel::unbounded();
        let (lock_tx, lock_rx) = crossbeam_channel::unbounded();
        let (hotkey_tx, hotkey_rx) = crossbeam_channel::unbounded();
        Self {
            trace: options.trace_path.map(TraceWriter::new),
            last_decision: None,
            last_pane_decision: None,
            platform,
            worker_factory,
            monitors_provider,
            hotkey_factory,
            backend_factory_maker: options
                .backend_factory
                .unwrap_or_else(|| Arc::new(backend_factory)),
            events: Vec::new(),

            monitors: Vec::new(),
            min_side: 1000.0,
            state: TrackingState::Starting,
            started: false,
            closed: false,

            paused: false,
            privacy: false,
            privacy_restored: false,
            calibrating: false,
            session_locked: false,
            yield_reason: String::new(),
            away: false,
            camera_error: false,
            privacy_before_calibration: false,
            preview_owners: HashSet::new(),

            decider: SwitchDecider::new(&[], SwitchConfig::from_settings(&s.switching)),
            presence: PresenceMonitor::new(PresenceConfig::from_settings(&s.presence), now),
            guard: ShoulderGuard::new(GuardConfig::from_settings(&s.privacy)),
            policy: RatePolicy::new(&s.performance.profile),
            input,
            memory: WindowMemory::new(),
            filter: PointFilter::new(s.switching.smoothing),
            learner,
            drift: DriftMonitor::default(),

            pane_decider: PaneDecider::new(PaneConfig::from_settings(&s.panes, &s.switching)),
            pane_registry_factory,
            pane_worker_factory,
            pane_worker: None,
            pane_capable: None,
            pane_window: None,
            pane_app: None,
            pane_snapshot: None,
            pane_requested_at: f64::NEG_INFINITY,
            pane_last_request: 0,
            pane_valid_from: 0,
            pane_sigma_cache: None,
            pane_sigma_used: None,
            pane_switches: 0,
            pane_cursor: Vec::new(),

            library: CalibrationLibrary::default(),
            calibration: None,
            calibration_serial: 0,
            model_usable: false,
            calibration_reason: "not calibrated yet".into(),
            implicit_dirty: false,
            feature_mismatches: 0,
            announce_pending: None,
            interruptible_since: None,
            last_announce: f64::NEG_INFINITY,

            reported_backend: None,
            expected_cache: RefCell::new(None),
            backend_generation: 0,
            built_backend: Arc::new(Mutex::new(None)),

            worker: None,
            worker_tx,
            worker_rx,
            held_worker_events: Vec::new(),
            interval: None,
            mode: "",
            worker_stats: WorkerStats::default(),
            camera_error_message: None,
            camera_on_since: None,
            frame_size: None,

            last_obs_time: None,
            last_face: None,
            face_seen_at: None,
            last_gaze: None,
            last_gaze_time: None,
            looking_away: false,
            gaze_none_sent: true,
            gaze_moving_until: f64::NEG_INFINITY,
            pending: false,
            last_switch: None,
            switch_count: 0,

            cursor_reliable: true,
            assumed_monitor: None,
            warp_refusals: 0,
            warp_suspensions: 0,
            switching_suspended_until: f64::NEG_INFINITY,
            cursor_warning_shown: false,
            activation_failures: 0,
            accessibility_notice_shown: false,
            background_active: None,

            blind_since: None,
            blind_freeze: false,
            blind_notice_at: f64::NEG_INFINITY,

            warning_shown: false,
            displays_off_only: false,
            away_displays_off: false,
            curtain: false,
            guard_locked: false,
            guard_relock_until: f64::NEG_INFINITY,
            lock_runner: options.lock_runner,
            lock_purposes: HashSet::new(),
            lock_thread: None,
            lock_tx,
            lock_rx,

            next_window_poll: f64::INFINITY,
            next_lock_check: f64::INFINITY,
            next_yield_check: f64::INFINITY,
            yield_checked_at: f64::NEG_INFINITY,
            next_stats: f64::INFINITY,
            process: None,
            cpu_count: 1,
            cpu_percent: None,
            last_stats: Map::new(),

            hotkeys: None,
            hotkeys_suspended: false,
            announced_hotkey_failures: HashSet::new(),
            hotkey_tx,
            hotkey_rx,

            polled_monitors: Vec::new(),
            next_monitor_poll: f64::INFINITY,
            screen_refresh_due: None,

            clock,
            settings: s,
        }
    }

    fn now(&self) -> f64 {
        (self.clock)()
    }

    fn emit(&mut self, event: ControllerEvent) {
        self.events.push(event);
    }

    /// What happened since the last call (the Python signals, in emission order).
    pub fn drain_events(&mut self) -> Vec<ControllerEvent> {
        std::mem::take(&mut self.events)
    }

    // ================================================================== lifecycle
    /// Load the calibration, start the camera worker, hotkeys and housekeeping.
    pub fn start(&mut self) {
        if self.started || self.closed {
            return;
        }
        let now = self.now();
        self.cursor_reliable = self.platform.cursor_position_reliable();
        if !self.cursor_reliable {
            log::info!(
                "The pointer position cannot be read reliably here (Wayland); the current \
                 monitor is taken from the last switch"
            );
        }
        let monitors = self.read_monitors();
        self.polled_monitors = monitors.clone();
        self.set_monitors(monitors);
        self.presence.reset(now);
        self.paused = self.paused || self.settings.general.start_paused;
        if saved_privacy_mode() && !self.privacy {
            if self.settings.privacy.remember_privacy_mode {
                // Turned on before the last quit, reboot or update: the camera must not come
                // back on by itself. Applied before the first state, so the worker never
                // opens it.
                self.privacy = true;
                self.privacy_restored = true;
                log::info!("Privacy mode is still on from the previous run");
            } else {
                // Not remembered: turning remembering on later must not bring back a choice
                // from a run that did not apply it.
                save_privacy_mode(false);
            }
        }
        self.load_calibration();

        let source = source_factory(&self.settings);
        let settings = self.settings.clone();
        let backend = self.make_backend_factory(&settings);
        let worker = (self.worker_factory)(source, backend, self.worker_tx.clone());
        // Keep the camera closed until the state says otherwise (start paused, locked
        // session, privacy mode set before start()).
        worker.set_active(false);
        worker.set_max_faces(self.max_faces());
        if !self.preview_owners.is_empty() {
            worker.set_preview(true);
        }
        self.worker = Some(worker);
        self.started = true;
        self.apply_motion_gate();
        self.sync_backend_info();
        self.validate_calibration(false);
        if !self.model_usable && self.calibration_useful() && !self.privacy {
            // The app offers the calibration at start (also after a login start, once its
            // quiet period is over); the reminders on the user's return
            // (remind_calibration) follow no sooner than CALIBRATION_REMINDER_S. In privacy
            // mode it does not, and leaving it brings the reminder.
            self.last_announce = now;
        }

        // Environment checks run before the camera may open.
        self.check_session_lock(now);
        self.check_camera_yield(now, false);
        self.next_window_poll = now + WINDOW_POLL_S;
        self.next_lock_check = now + LOCK_POLL_S;
        self.next_yield_check = now + YIELD_POLL_S;
        self.next_stats = now + STATS_PERIOD_S;
        self.next_monitor_poll = now + MONITOR_POLL_S;
        self.measure_cpu(); // baseline for the first reading

        self.update_state(); // STARTING -> real state: sets active flag and interval
        if let Some(worker) = &self.worker {
            worker.start();
        }
        self.setup_hotkeys();
        self.check_accessibility();
        log::info!(
            "Controller started: {}, {} monitor(s), calibration {}",
            self.state.value(),
            self.monitors.len(),
            if self.model_usable {
                "usable".to_string()
            } else {
                format!("unusable ({})", self.calibration_reason)
            }
        );
    }

    /// Stop the worker (releasing the camera), hotkeys and timers. Idempotent.
    pub fn shutdown(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        self.screen_refresh_due = None;
        if let Some(hotkeys) = &self.hotkeys {
            hotkeys.stop();
        }
        if let Some(worker) = &self.worker {
            worker.stop(3.0);
        }
        self.stop_pane_worker(true);
        if let Some(thread) = self.lock_thread.take() {
            // Let a lock in progress finish before this object may be dropped.
            let deadline = Instant::now() + Duration::from_secs_f64(LOCK_JOIN_TIMEOUT_S);
            while !thread.is_finished() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(5));
            }
            if thread.is_finished() {
                let _ = thread.join();
            }
        }
        if self.implicit_dirty && self.calibration.is_some() {
            self.save_calibration(true);
        }
        if let Some(trace) = &self.trace {
            trace.close();
        }
        log::info!("Controller stopped");
    }

    // ================================================================ properties
    /// Current tracking state.
    pub fn state(&self) -> TrackingState {
        self.state
    }

    /// Settings in effect. Change them with [`apply_settings`](Self::apply_settings).
    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    /// The user paused tracking.
    pub fn paused(&self) -> bool {
        self.paused
    }

    /// Privacy mode (camera fully off) is on.
    pub fn privacy(&self) -> bool {
        self.privacy
    }

    /// Privacy mode is on because it was on when the app last ran
    /// (`privacy.remember_privacy_mode`) and has not been changed since.
    pub fn privacy_restored(&self) -> bool {
        self.privacy_restored && self.privacy
    }

    /// At least one consumer wants preview frames (see [`set_preview`](Self::set_preview)).
    pub fn preview_enabled(&self) -> bool {
        !self.preview_owners.is_empty()
    }

    /// A calibration usable with the current camera backend and monitors exists.
    pub fn is_calibrated(&self) -> bool {
        self.model_usable
    }

    /// Why the calibration is unusable (`""` when it is usable).
    pub fn calibration_reason(&self) -> &str {
        if self.model_usable {
            ""
        } else {
            &self.calibration_reason
        }
    }

    /// The shoulder guard currently sees a second face.
    pub fn guard_active(&self) -> bool {
        self.guard.active()
    }

    pub fn presence_state(&self) -> PresenceState {
        self.presence.state()
    }

    /// Why the camera was released for another app (`""` if it was not).
    pub fn yield_reason(&self) -> &str {
        &self.yield_reason
    }

    /// The global hotkey manager (`None` before [`start`](Self::start)).
    pub fn hotkey_manager(&self) -> Option<Arc<dyn HotkeyManager>> {
        self.hotkeys.clone()
    }

    /// The OS integration layer this controller was built with.
    pub fn platform(&self) -> &Arc<dyn PlatformServices> {
        &self.platform
    }

    /// Release (`true`) or restore (`false`) the global hotkeys.
    ///
    /// The settings dialog suspends them while it records a new shortcut: the OS delivers a
    /// registered combination to us instead of the focused window, so without this the user
    /// could not re-record a shortcut that is in use.
    pub fn suspend_hotkeys(&mut self, suspended: bool) {
        if suspended == self.hotkeys_suspended {
            return;
        }
        self.hotkeys_suspended = suspended;
        if self.closed {
            return;
        }
        let Some(hotkeys) = self.hotkeys.clone() else {
            return;
        };
        if suspended {
            hotkeys.unregister_all();
        } else {
            self.register_hotkeys();
        }
    }

    /// The current monitor layout.
    pub fn monitors(&mut self) -> Vec<Monitor> {
        if self.monitors.is_empty() && !self.started {
            return self.read_monitors();
        }
        self.monitors.clone()
    }

    /// The calibration in use, or the most recent one when none fits the current setup (then
    /// it is not usable; see [`calibration_reason`](Self::calibration_reason)).
    pub fn calibration(&self) -> Option<&CalibrationData> {
        self.calibration.as_ref()
    }

    /// Every saved calibration profile, most recently used first.
    pub fn calibrations(&self) -> &[CalibrationData] {
        self.library.profiles()
    }

    /// The gaze model in use (set only while the calibration is usable).
    fn model(&self) -> Option<&GazeModel> {
        if self.model_usable {
            self.calibration.as_ref().map(|c| &c.model)
        } else {
            None
        }
    }

    /// `(name, feature_version)` of the vision backend in use (or expected).
    ///
    /// `("", "")` when no backend is available.
    pub fn backend_info(&self) -> (String, String) {
        self.current_backend().unwrap_or_default()
    }

    /// Indices of the gaze-direction features of the backend in use (or expected).
    ///
    /// What `calibration::evaluate(..., nonlinear)` takes, so that a new calibration applies
    /// nonlinear terms only to where the user looks, not to where their head is. `None` when
    /// the backend declares none (or is unknown).
    pub fn gaze_feature_indices(&self) -> Option<Vec<usize>> {
        let built = self.built_backend.lock().clone();
        if let Some(built) = built
            && built.generation == self.backend_generation
        {
            return built.gaze_indices;
        }
        let (name, _) = self.current_backend()?;
        named_backend_gaze_indices(&name)
    }

    /// `(device, (width, height))` of the camera in use.
    ///
    /// `device` is the camera device setting and the size is that of the frames analysed most
    /// recently (`(0, 0)` before the first). A new calibration records both
    /// (`CalibrationData::camera` and `frame_size`) so that it is only used with this camera;
    /// [`finish_calibration`](Self::finish_calibration) fills them in when they are missing.
    pub fn camera_identity(&self) -> (String, (u32, u32)) {
        (
            self.settings.camera.device.clone(),
            self.frame_size.unwrap_or((0, 0)),
        )
    }

    /// Seconds until the walk-away action (meaningful during a countdown).
    pub fn away_remaining(&self) -> f64 {
        self.presence.remaining(self.now())
    }

    /// The most recent `stats_changed` payload (empty before the first one).
    pub fn stats_snapshot(&self) -> Map<String, Value> {
        self.last_stats.clone()
    }

    // ============================================================ user commands
    /// Pause tracking; the camera is released.
    pub fn pause(&mut self) {
        if !self.paused {
            self.paused = true;
            log::info!("Tracking paused");
            self.update_state();
        }
    }

    /// Resume after [`pause`](Self::pause) (privacy mode, if on, stays on).
    pub fn resume(&mut self) {
        if self.paused {
            self.paused = false;
            log::info!("Tracking resumed");
            self.update_state();
        }
    }

    pub fn toggle_pause(&mut self) {
        if self.paused {
            self.resume();
        } else {
            self.pause();
        }
    }

    /// Privacy mode: the camera is fully released (its light goes off).
    ///
    /// The choice is remembered for the next start (`privacy.remember_privacy_mode`).
    pub fn set_privacy(&mut self, enabled: bool) {
        // An explicit choice made during a calibration is what applies afterwards.
        self.privacy_before_calibration = false;
        if enabled == self.privacy {
            return;
        }
        self.privacy = enabled;
        self.privacy_restored = false;
        log::info!("Privacy mode {}", if enabled { "on" } else { "off" });
        save_privacy_mode(enabled);
        self.update_state();
    }

    pub fn toggle_privacy(&mut self) {
        self.set_privacy(!self.privacy);
    }

    /// Deliver annotated camera frames through `PreviewFrame` (and sample faster).
    ///
    /// Several consumers (the preview window, the first-run wizard) can want frames at once;
    /// each passes its own `owner` name and frames flow while at least one of them has
    /// enabled the preview. Calls without an owner share one anonymous owner.
    pub fn set_preview(&mut self, enabled: bool, owner: Option<&str>) {
        let before = !self.preview_owners.is_empty();
        let owner = owner.unwrap_or("").to_string();
        if enabled {
            self.preview_owners.insert(owner);
        } else {
            self.preview_owners.remove(&owner);
        }
        let active = !self.preview_owners.is_empty();
        if active == before {
            return;
        }
        if let Some(worker) = &self.worker {
            worker.set_preview(active);
        }
        if !active {
            self.discard_queued_previews();
        }
        let now = self.now();
        self.update_rate(now, false);
    }

    /// The user dismissed the privacy curtain: monitor switching resumes.
    ///
    /// The shoulder guard stays active; it shows the curtain again only after the second face
    /// has left and come back.
    pub fn dismiss_curtain(&mut self) {
        if self.curtain {
            log::info!("Privacy curtain dismissed by the user");
            self.show_curtain(false);
        }
    }

    /// Enter `CALIBRATING`: the camera runs at the calibration rate and every observation is
    /// forwarded through `Observation`.
    ///
    /// An explicit calibration request needs the camera, so privacy mode is turned off
    /// meanwhile; like a user pause, it applies again once the calibration is finished or
    /// cancelled (unless it was changed during the calibration).
    pub fn begin_calibration(&mut self) {
        if self.calibrating {
            return;
        }
        self.privacy_before_calibration = self.privacy;
        if self.privacy {
            log::info!("Privacy mode turned off to calibrate");
            self.privacy = false;
        }
        self.calibrating = true;
        log::info!("Calibration started");
        self.update_state();
    }

    /// Leave `CALIBRATING`. `data` is saved and used; `None` means cancelled.
    ///
    /// `data.camera` and `data.frame_size` are filled in from
    /// [`camera_identity`](Self::camera_identity) when they are unknown. The calibration is
    /// stored as the profile of the current setup; profiles of other setups are kept.
    pub fn finish_calibration(&mut self, data: Option<CalibrationData>) {
        if let Some(data) = data {
            let data = self.with_camera_identity(data);
            let grade = data.grade().unwrap_or("ungraded").to_string();
            self.use_profile(Some(data), false);
            self.save_calibration(false);
            self.model_usable = false; // force a fresh validation below
            self.validate_calibration(false);
            self.reset_tracking();
            if self.model_usable {
                log::info!("Calibration saved ({grade})");
            } else {
                log::warn!(
                    "The new calibration is not usable: {}",
                    self.calibration_reason
                );
            }
        } else if self.calibrating {
            log::info!("Calibration cancelled");
            // The camera may have changed its frame size meanwhile (not judged during the
            // calibration).
            self.validate_calibration(true);
        }
        if self.privacy_before_calibration && !self.privacy {
            log::info!("Privacy mode on again after the calibration");
            self.privacy = true;
        }
        self.privacy_before_calibration = false;
        self.calibrating = false;
        self.update_state();
        self.apply_motion_gate();
    }

    /// Adopt new settings: persist them and reconfigure every component.
    pub fn apply_settings(&mut self, settings: &Settings) {
        let new = settings.clone();
        let old = std::mem::replace(&mut self.settings, new.clone());
        if let Err(e) = new.save(&paths::settings_file()) {
            log::error!("Could not save settings: {e}");
            self.notify("Settings not saved", &e.to_string(), true);
        }

        self.decider
            .set_config(SwitchConfig::from_settings(&new.switching));
        self.pane_decider
            .set_config(PaneConfig::from_settings(&new.panes, &new.switching));
        if old.panes != new.panes || old.switching.enabled != new.switching.enabled {
            self.reconfigure_panes(pane_providers(&old) != pane_providers(&new));
        }
        self.filter.set_smoothing(new.switching.smoothing);
        self.presence
            .set_config(PresenceConfig::from_settings(&new.presence));
        self.guard
            .set_config(GuardConfig::from_settings(&new.privacy));
        self.policy.set_profile(&new.performance.profile);
        if self
            .learner
            .set_max_samples(usize::try_from(new.learning.max_samples).unwrap_or(0))
        {
            // The model still carries the influence of the dropped samples.
            self.refit_after_trim();
        }

        let backend_changed = old.general.backend != new.general.backend;
        let source_changed = old.camera != new.camera || camera_fps(&old) != camera_fps(&new);
        if backend_changed {
            *self.expected_cache.borrow_mut() = None;
            // Unknown until the worker has built the new backend (which may even have the
            // same identity as the old one); the settings decide meanwhile.
            self.reported_backend = None;
        }
        if source_changed {
            self.frame_size = None; // known again with the first frame of the new source
        }
        if self.worker.is_some() {
            let max_faces = self.max_faces();
            if let Some(worker) = &self.worker {
                worker.set_max_faces(max_faces);
            }
            self.apply_motion_gate();
            if source_changed || backend_changed {
                let source = source_changed.then(|| source_factory(&new));
                let backend = if backend_changed {
                    Some(self.make_backend_factory(&new))
                } else {
                    None
                };
                if let Some(worker) = &self.worker {
                    worker.reconfigure(source, backend);
                }
            }
        }
        if old.hotkeys != new.hotkeys && self.hotkeys.is_some() {
            // A deliberate change: report every hotkey that does not work now, also one that
            // was announced before.
            self.announced_hotkey_failures.clear();
            self.register_hotkeys();
        }

        if self.started && !self.closed {
            let now = self.now();
            if !new.privacy.shoulder_guard {
                self.update_guard(now, None, None); // a disabled guard clears at once
            }
            if old.privacy != new.privacy {
                self.next_yield_check = now;
            }
            if backend_changed || source_changed {
                // Another backend or camera may need another calibration profile.
                self.validate_calibration(true);
            }
            self.update_presence(now);
            self.update_state();
            self.update_rate(now, true);
            self.check_accessibility();
        }
        log::info!("Settings applied");
        self.emit(ControllerEvent::SettingsChanged(Box::new(new)));
    }

    /// Re-read the monitor layout (called automatically after screen changes).
    pub fn refresh_monitors(&mut self) {
        if self.closed {
            return;
        }
        let monitors = self.read_monitors();
        if monitors.is_empty() {
            log::debug!("No monitors reported; keeping the previous layout");
            return;
        }
        self.polled_monitors = monitors.clone();
        if layout_key(&monitors) == layout_key(&self.monitors) {
            self.monitors = monitors; // names or scale factors may have changed
            return;
        }
        log::info!(
            "Monitor layout changed: {}",
            monitors
                .iter()
                .map(|m| format!(
                    "{}:{}x{}@{},{}",
                    m.index, m.rect.w, m.rect.h, m.rect.x, m.rect.y
                ))
                .collect::<Vec<_>>()
                .join(", ")
        );
        self.set_monitors(monitors);
        self.memory.clear();
        self.reset_tracking();
        if self.started {
            self.validate_calibration(true);
            self.update_state();
        }
    }

    /// Execute an IPC command (see [`COMMANDS`]) and return the reply line.
    ///
    /// `status` returns a one-line JSON object; UI commands are forwarded through
    /// `UiRequested`; everything else returns `"ok"` or `"error: …"`.
    pub fn handle_command(&mut self, command: &str) -> String {
        let cmd = command.trim().to_lowercase();
        let before = (self.paused, self.privacy);
        match cmd.as_str() {
            "pause" => self.pause(),
            "resume" => self.resume(),
            "toggle" => self.toggle_pause(),
            "privacy-on" => self.set_privacy(true),
            "privacy-off" => self.set_privacy(false),
            "privacy-toggle" => self.toggle_privacy(),
            "calibrate" => {
                self.emit(ControllerEvent::CalibrationRequired("ipc".into()));
                return "ok".into();
            }
            "status" => return Value::Object(self.status()).to_string(),
            _ if UI_COMMANDS.contains(&cmd.as_str()) => {
                self.emit(ControllerEvent::UiRequested(cmd));
                return "ok".into();
            }
            _ => return format!("error: unknown command {}", py_repr(command.trim())),
        }
        self.confirm_mode_change(before);
        "ok".into()
    }

    /// JSON-safe summary of the running instance (`eye-tracker ctl status`).
    pub fn status(&mut self) -> Map<String, Value> {
        let stats = &self.last_stats;
        let calibration = self.calibration.as_ref().map(|cal| {
            json!({
                "grade": cal.grade(),
                "created_at": cal.created_at,
                "backend": cal.backend,
                "usable": self.model_usable,
                "reason": self.calibration_reason(),
                "profiles": self.library.len(),
            })
        });
        let mut status = Map::new();
        status.insert("version".into(), json!(VERSION));
        status.insert("state".into(), json!(self.state.value()));
        status.insert("label".into(), json!(self.state.label()));
        status.insert("paused".into(), json!(self.paused));
        status.insert("privacy".into(), json!(self.privacy));
        status.insert("calibrated".into(), json!(self.model_usable));
        status.insert("calibration".into(), calibration.unwrap_or(Value::Null));
        status.insert("backend".into(), json!(self.backend_info().0));
        status.insert("monitors".into(), json!(self.monitors.len()));
        status.insert("presence".into(), json!(self.presence.state().value()));
        status.insert("guard_active".into(), json!(self.guard.active()));
        status.insert(
            "yield_reason".into(),
            if self.yield_reason.is_empty() {
                Value::Null
            } else {
                json!(self.yield_reason)
            },
        );
        for key in ["fps", "target_fps", "cpu_percent"] {
            status.insert(key.into(), stats.get(key).cloned().unwrap_or(Value::Null));
        }
        status.insert("switches".into(), json!(self.switch_count));
        status.insert("hotkeys".into(), Value::Object(self.hotkey_status()));
        status.insert("panes".into(), Value::Object(self.pane_status()));
        status
    }

    /// Split-pane focus at a glance: never titles, commands or paths.
    fn pane_status(&mut self) -> Map<String, Value> {
        let mut sigma = self.pane_sigma_used;
        if sigma.is_none()
            && let Some(rect) = self.pane_window.as_ref().and_then(|w| w.rect)
        {
            sigma = self.pane_sigma(rect);
        }
        let fallback = (FALLBACK_GAZE_ERROR_PX, FALLBACK_GAZE_ERROR_PX);
        let supported = self.panes_supported();
        let snapshot = self.pane_snapshot.as_ref();
        let mut failed: Vec<String> = self
            .pane_worker
            .as_ref()
            .map(|w| w.disabled_providers().into_iter().collect())
            .unwrap_or_default();
        failed.sort();
        let mut status = Map::new();
        status.insert("enabled".into(), json!(self.settings.panes.enabled));
        status.insert("supported".into(), json!(supported));
        status.insert(
            "provider".into(),
            json!(self.pane_snapshot.as_ref().and_then(|s| s.provider())),
        );
        status.insert(
            "panes".into(),
            json!(self.pane_snapshot.as_ref().map_or(0, |s| s.panes.len())),
        );
        status.insert(
            "eligible".into(),
            json!(eligible_panes(
                snapshot,
                sigma.unwrap_or(fallback),
                self.pane_decider.config()
            )),
        );
        status.insert("switches".into(), json!(self.pane_switches));
        status.insert(
            "sigma_px".into(),
            match sigma {
                None => Value::Null,
                Some((x, y)) => json!([round_to(x, 1), round_to(y, 1)]),
            },
        );
        status.insert("failed_providers".into(), json!(failed));
        status
    }

    /// The global hotkeys as the OS took them: `{"registered": [names], "errors": {name: why
    /// it is not registered}}`.
    ///
    /// Only this process knows: `eye-tracker doctor` reads it through `status` (it cannot
    /// register anything itself without taking the combinations away from the running app).
    /// Empty while hotkeys are off or unsupported here, where nothing is registered on
    /// purpose.
    fn hotkey_status(&self) -> Map<String, Value> {
        let empty = || {
            let mut status = Map::new();
            status.insert("registered".into(), json!([]));
            status.insert("errors".into(), json!({}));
            status
        };
        let Some(manager) = &self.hotkeys else {
            return empty();
        };
        if !self.settings.hotkeys.enabled || !manager.supported() {
            return empty();
        }
        let configured: Vec<&str> = HOTKEY_ACTIONS
            .iter()
            .copied()
            .filter(|name| !hotkey_setting(&self.settings, name).trim().is_empty())
            .collect();
        let mut status = Map::new();
        if self.hotkeys_suspended {
            // Released on purpose; they come back when the settings dialog is done.
            let released = "released while a shortcut is being recorded in Settings → Hotkeys";
            let errors: Map<String, Value> = configured
                .iter()
                .map(|name| ((*name).to_string(), json!(released)))
                .collect();
            status.insert("registered".into(), json!([]));
            status.insert("errors".into(), Value::Object(errors));
            return status;
        }
        let mut registered: Vec<String> = manager.registered().into_keys().collect();
        registered.sort();
        let mut errors = Map::new();
        for name in configured {
            if registered.iter().any(|r| r == name) {
                continue;
            }
            if let Some(reason) = Self::hotkey_error(manager.as_ref(), name) {
                errors.insert(name.to_string(), json!(reason));
            }
        }
        status.insert("registered".into(), json!(registered));
        status.insert("errors".into(), Value::Object(errors));
        if let Some(note) = manager.note()
            && !note.trim().is_empty()
        {
            // A limitation the running manager found (another program grabbing the Super key
            // on its own): doctor shows it.
            status.insert("note".into(), json!(note.trim()));
        }
        status
    }

    // ================================================================ housekeeping
    /// How long the owner should wait before the next [`tick`](Self::tick): 100 ms while
    /// tracking (or counting down to the walk-away action), 500 ms otherwise, and no longer
    /// than until a pending monitor refresh is due.
    pub fn next_tick_in(&self) -> Duration {
        let mut wait = Duration::from_millis(self.tick_interval());
        if let Some(due) = self.screen_refresh_due {
            let left = (due - self.now()).max(0.0);
            wait = wait.min(Duration::from_secs_f64(left));
        }
        wait
    }

    /// Handle what the worker threads, the hotkeys and the lock thread delivered (the Python
    /// queued signals). [`tick`](Self::tick) starts with this; call it alone to react sooner.
    pub fn process_queued(&mut self) {
        if self.closed {
            return;
        }
        let mut preview = None;
        let held = std::mem::take(&mut self.held_worker_events);
        let fresh: Vec<WorkerEvent> = self.worker_rx.try_iter().collect();
        for event in held.into_iter().chain(fresh) {
            match event {
                WorkerEvent::Observation(obs) => self.handle_observation(&obs),
                WorkerEvent::Stats(stats) => self.handle_worker_stats(stats),
                // Only the newest frame matters; never let frames pile up.
                WorkerEvent::Preview(frame) => preview = Some(frame),
            }
        }
        if let Some(frame) = preview {
            self.deliver_preview(frame);
        }
        let names: Vec<String> = self.hotkey_rx.try_iter().collect();
        for name in names {
            self.handle_hotkey(&name);
        }
        let results: Vec<bool> = self.lock_rx.try_iter().collect();
        for ok in results {
            self.finish_lock(ok);
        }
        self.process_pane_events();
    }

    /// One housekeeping step (the Python app's timer).
    pub fn tick(&mut self, now: f64) {
        if !self.started || self.closed {
            return;
        }
        self.process_queued();
        self.poll_monitors(now);
        self.poll_input(now);
        self.settle_last_switch(now);
        if now >= self.next_window_poll {
            self.next_window_poll = now + WINDOW_POLL_S;
            self.record_foreground_window(now);
        }
        if now >= self.next_lock_check {
            self.next_lock_check = now + LOCK_POLL_S;
            self.check_session_lock(now);
        }
        if now >= self.next_yield_check {
            self.next_yield_check = now + YIELD_POLL_S;
            self.check_camera_yield(now, false);
        }
        self.sync_backend_info();
        self.recheck_camera_error(now);
        self.update_presence(now);
        self.update_state();
        self.deliver_calibration_notice(now);
        self.update_rate(now, false);
        if now >= self.next_stats {
            self.next_stats = now + STATS_PERIOD_S;
            self.emit_stats();
        }
    }

    /// Notice a changed monitor layout (Qt's screen signals) and refresh once it settled.
    fn poll_monitors(&mut self, now: f64) {
        if now >= self.next_monitor_poll {
            self.next_monitor_poll = now + MONITOR_POLL_S;
            let monitors = self.read_monitors();
            if monitors != self.polled_monitors {
                self.polled_monitors = monitors;
                // Hot-plug events come in bursts: every change restarts the debounce.
                self.screen_refresh_due = Some(now + SCREEN_DEBOUNCE_MS as f64 / 1000.0);
            }
        }
        if self.screen_refresh_due.is_some_and(|due| now >= due) {
            self.screen_refresh_due = None;
            self.refresh_monitors();
        }
    }

    fn poll_input(&mut self, now: f64) {
        // On Wayland there is no position at all (Qt reported a stale one); the last one
        // keeps the input idle times polled.
        let pos = self
            .cursor_pos()
            .or_else(|| (!self.cursor_reliable).then(|| self.input.cursor()));
        let Some(pos) = pos else {
            return;
        };
        self.input.poll(now, pos);
        // A stale position (Wayland) says nothing about where the user put the pointer: no
        // remembered positions, learning labels or undone switches.
        if self.input.manual_move() && self.cursor_reliable {
            self.on_manual_move(pos, now);
        }
        if self.cursor_reliable {
            self.remember_pane_cursor(pos);
        }
    }

    fn on_manual_move(&mut self, pos: (i32, i32), now: f64) {
        let (px, py) = (f64::from(pos.0), f64::from(pos.1));
        let monitor = monitor_at(&self.monitors, px, py).map(|m| m.index);
        if let Some(index) = monitor {
            self.memory.record_cursor(index, pos);
        }
        if self.model_usable && self.settings.learning.adaptive {
            self.learner.on_manual_cursor(px, py, now);
        }
        let Some((switched_at, source, _target)) = self.last_switch else {
            return;
        };
        if now - switched_at > WRONG_SWITCH_S {
            self.last_switch = None;
            self.drift.record_switch(true);
        } else if monitor.is_some() && source.is_some() && monitor == source {
            // The user dragged the pointer straight back: that switch was wrong.
            self.last_switch = None;
            self.drift.record_wrong_switch();
            log::debug!("Automatic switch undone by the user");
            self.check_drift(now);
        }
    }

    /// A switch the user left alone for WRONG_SWITCH_S was a correct one.
    ///
    /// Recording those as well keeps the drift statistics honest: with adaptive learning off,
    /// undone switches would otherwise be the only events.
    fn settle_last_switch(&mut self, now: f64) {
        if let Some((at, _, _)) = self.last_switch
            && now - at > WRONG_SWITCH_S
        {
            self.last_switch = None;
            self.drift.record_switch(true);
        }
    }

    fn record_foreground_window(&mut self, now: f64) {
        if self.state != TrackingState::Tracking {
            return;
        }
        let remember = self.settings.switching.focus_window;
        let panes = self.panes_wanted();
        if !(remember || panes) {
            return;
        }
        let window = self.platform.foreground_window();
        let Some((window, rect)) = window
            .and_then(|w| w.rect.map(|r| (w, r)))
            .filter(|_| !self.monitors.is_empty())
        else {
            if panes {
                self.forget_pane_window();
            }
            return;
        };
        if remember {
            let (cx, cy) = rect.center();
            let monitor = monitor_at(&self.monitors, cx, cy)
                .or_else(|| nearest_monitor(&self.monitors, cx, cy).ok().map(|(m, _)| m))
                .map(|m| m.index);
            if let Some(index) = monitor {
                self.memory.record_window(index, window.clone());
            }
        }
        if panes {
            self.follow_pane_window(window, now);
        }
    }

    // ---------------------------------------------------------- split panes
    /// The platform's `panes` capability (asked once).
    fn panes_supported(&mut self) -> bool {
        if self.pane_capable.is_none() {
            let caps = self.platform.capabilities();
            self.pane_capable = Some(caps.get("panes").copied().unwrap_or(false));
        }
        self.pane_capable.unwrap_or(false)
    }

    fn panes_wanted(&mut self) -> bool {
        self.settings.panes.enabled && self.settings.switching.enabled && self.panes_supported()
    }

    fn ensure_pane_worker(&mut self) -> bool {
        if self.pane_worker.is_none() && self.panes_wanted() && !self.closed {
            let registry = (self.pane_registry_factory)(&self.settings);
            let mut worker = (self.pane_worker_factory)(registry);
            worker.start();
            self.pane_worker = Some(worker);
            // A new worker numbers its requests from 1 again.
            self.pane_last_request = 0;
            self.pane_valid_from = 0;
            log::info!("Split-pane focus is on (experimental)");
        }
        self.pane_worker.is_some()
    }

    /// Stop the pane worker. Only [`shutdown`](Self::shutdown) waits for a provider call in
    /// progress; a settings change must not block the GUI thread, and a stopped worker
    /// delivers nothing more.
    fn stop_pane_worker(&mut self, wait: bool) {
        let Some(mut worker) = self.pane_worker.take() else {
            return;
        };
        if wait {
            worker.stop(PANE_JOIN_TIMEOUT_S);
        } else {
            worker.request_stop();
        }
    }

    /// Split-pane settings changed: start, stop or re-provision the worker.
    fn reconfigure_panes(&mut self, providers_changed: bool) {
        if !self.panes_wanted() {
            if self.pane_worker.is_some() {
                log::info!("Split-pane focus is off");
            }
            self.stop_pane_worker(false);
            self.forget_pane_window();
            self.pane_decider.reset();
            return;
        }
        if self.pane_worker.is_some() && providers_changed {
            let registry = (self.pane_registry_factory)(&self.settings);
            if let Some(worker) = &self.pane_worker {
                worker.set_registry(registry);
            }
            self.pane_snapshot = None;
            self.pane_requested_at = f64::NEG_INFINITY;
            self.pane_valid_from = self.pane_last_request + 1;
        }
    }

    fn forget_pane_window(&mut self) {
        self.pane_window = None;
        self.pane_app = None;
        self.pane_snapshot = None;
        self.pane_requested_at = f64::NEG_INFINITY;
        self.pane_valid_from = self.pane_last_request + 1;
    }

    /// Track the focused window and ask for its panes when useful.
    ///
    /// A new window is asked about at once; the window being followed again every
    /// [`PANE_REFRESH_S`] while the gaze is inside it. Windows of denied applications (see
    /// `panes::registry`) are never asked about.
    fn follow_pane_window(&mut self, window: WindowRef, now: f64) {
        let rect = window.rect;
        let fresh = match &self.pane_window {
            Some(current)
                if current.pid == window.pid // a handle reused by another process's window
                    && self.platform.same_window(Some(current), Some(&window)) =>
            {
                // Keep the handle object the snapshot belongs to; only the frame moves.
                if window.rect != current.rect {
                    let mut moved = current.clone();
                    moved.rect = window.rect;
                    self.pane_window = Some(moved);
                }
                false
            }
            _ => {
                self.pane_snapshot = None;
                self.pane_requested_at = f64::NEG_INFINITY;
                self.pane_valid_from = self.pane_last_request + 1;
                self.pane_app = self.platform.window_app(&window);
                self.pane_window = Some(window);
                self.prune_pane_cursors();
                true
            }
        };
        let Some(app) = self.pane_app.clone() else {
            return;
        };
        if is_denied(&app, self.settings.panes.desktop_apps) {
            return;
        }
        if !fresh {
            let inside =
                matches!((self.last_gaze, rect), (Some((gx, gy)), Some(r)) if r.contains(gx, gy));
            if !inside || now - self.pane_requested_at < PANE_REFRESH_S {
                return;
            }
        }
        if !self.ensure_pane_worker() {
            return;
        }
        let (Some(worker), Some(window)) = (&self.pane_worker, &self.pane_window) else {
            return;
        };
        self.pane_requested_at = now;
        self.pane_last_request = worker.request_detect(window, &app);
        self.process_pane_events();
    }

    fn process_pane_events(&mut self) {
        let events = self
            .pane_worker
            .as_ref()
            .map(|w| w.drain_events())
            .unwrap_or_default();
        for event in events {
            match event {
                PaneEvent::Detected(result) => self.on_pane_detected(result),
                PaneEvent::Focused(result) => self.on_pane_focused(result),
            }
        }
    }

    fn on_pane_detected(&mut self, result: DetectResult) {
        let Some(window) = &self.pane_window else {
            return;
        };
        if self.closed || result.window_handle != window.handle {
            return; // the user moved on to another window meanwhile
        }
        if result.request_id < self.pane_valid_from {
            return; // asked before a window change or with the providers since replaced
        }
        let key = window_key(window);
        if let Some(snapshot) = &result.snapshot {
            // Panes that closed take their remembered cursor position with them.
            if let Some((_, _, spots)) = self.pane_cursor.iter_mut().find(|(k, _, _)| *k == key) {
                spots.retain(|id, _| snapshot.panes.iter().any(|p| p.id == *id));
            }
        }
        self.pane_snapshot = result.snapshot;
    }

    fn on_pane_focused(&mut self, result: FocusResult) {
        if self.closed {
            return;
        }
        if !result.ok {
            log::debug!("Could not focus pane {}", result.pane_id);
            return;
        }
        self.pane_switches += 1;
        if let Some(snapshot) = &self.pane_snapshot
            && snapshot.window_handle == result.window_handle
        {
            let pane = snapshot.pane(&result.pane_id).cloned();
            self.pane_snapshot = Some(snapshot.with_focus(&result.pane_id));
            if let Some(pane) = pane
                && self.settings.panes.move_cursor
            {
                self.move_cursor_into_pane(&pane);
            }
        }
        self.pane_requested_at = f64::NEG_INFINITY; // confirm the new layout at the next poll
    }

    // ------------------------------------------------- cursor in split panes
    /// After a pane got the keyboard focus: the cursor follows, to where the user last left it
    /// in that pane (if still inside it), else its centre.
    ///
    /// The same path as a monitor switch: the warp is announced to the input tracker first (so
    /// it is not taken for mouse use) and refusals count towards the warp backoff. Nothing
    /// happens if the cursor is already in the pane.
    fn move_cursor_into_pane(&mut self, pane: &Pane) {
        let Some(window) = &self.pane_window else {
            return;
        };
        let rect = pane.rect;
        let cursor = self.cursor_pos();
        if self.cursor_reliable
            && cursor.is_some_and(|(x, y)| rect.contains(f64::from(x), f64::from(y)))
        {
            return;
        }
        let key = window_key(window);
        let spot = self
            .pane_cursor
            .iter()
            .find(|(k, _, _)| *k == key)
            .and_then(|(_, _, spots)| spots.get(&pane.id).copied())
            .filter(|(x, y)| rect.contains(f64::from(*x), f64::from(*y)));
        let spot = spot.unwrap_or_else(|| {
            let (cx, cy) = rect.center();
            rect.clamp(cx, cy)
        });
        let now = self.now();
        self.input.note_programmatic_move(spot, now);
        if self.move_cursor(spot.0, spot.1, now) {
            log::debug!("Cursor moved into pane {}", pane.id);
        }
    }

    /// Record `pos` for the pane of the followed window that contains it.
    fn remember_pane_cursor(&mut self, pos: (i32, i32)) {
        let (Some(window), Some(snapshot)) = (&self.pane_window, &self.pane_snapshot) else {
            return;
        };
        if !self.settings.panes.move_cursor || snapshot.window_handle != window.handle {
            return;
        }
        let Some(pane) = snapshot
            .panes
            .iter()
            .find(|p| p.rect.contains(f64::from(pos.0), f64::from(pos.1)))
        else {
            return;
        };
        let key = window_key(window);
        let window = window.clone();
        let pane_id = pane.id.clone();
        // Re-inserted: most recently used last.
        let mut spots = match self.pane_cursor.iter().position(|(k, _, _)| *k == key) {
            Some(i) => self.pane_cursor.remove(i).2,
            None => HashMap::new(),
        };
        spots.insert(pane_id, pos);
        self.pane_cursor.push((key, window, spots));
        while self.pane_cursor.len() > PANE_CURSOR_WINDOWS {
            self.pane_cursor.remove(0);
        }
    }

    /// Forget the cursor positions of windows that are gone.
    fn prune_pane_cursors(&mut self) {
        let platform = self.platform.clone();
        self.pane_cursor
            .retain(|(_, window, _)| platform.is_window_valid(window));
    }

    /// The calibration's gaze error on the monitor showing `rect` (cached).
    fn pane_sigma(&mut self, rect: Rect) -> Option<(f64, f64)> {
        if self.calibration.is_none() || self.monitors.is_empty() {
            return None;
        }
        let (cx, cy) = rect.center();
        let monitor = monitor_at(&self.monitors, cx, cy)
            .or_else(|| nearest_monitor(&self.monitors, cx, cy).ok().map(|(m, _)| m))?
            .clone();
        if let Some((serial, index, mrect, sigma)) = self.pane_sigma_cache
            && serial == self.calibration_serial
            && index == monitor.index
            && mrect == monitor.rect
        {
            return sigma;
        }
        let cal = self.calibration.as_mut()?;
        let sigma = axis_error(cal, &monitor);
        self.pane_sigma_cache = Some((self.calibration_serial, monitor.index, monitor.rect, sigma));
        sigma
    }

    /// Move the keyboard focus between split panes (only while the monitor decider is
    /// content). Never activates a window; the cursor follows only once the focus call
    /// succeeded (`on_pane_focused`).
    fn pane_step(
        &mut self,
        now: f64,
        gaze: Option<(f64, f64)>,
        decision: &Decision,
        enabled: bool,
    ) -> PaneDecision {
        let s = &self.settings.panes;
        let active = enabled
            && s.enabled
            && self.pane_worker.is_some()
            && self.pane_window.is_some()
            && decision.target.is_none()
            && decision.reason == "same"
            && now - self.decider.last_switch_time() >= s.after_monitor_switch_ms as f64 / 1000.0;
        let rect = self.pane_window.as_ref().and_then(|w| w.rect);
        let sigma = match rect {
            Some(rect) if active => self.pane_sigma(rect),
            _ => None,
        };
        if active {
            self.pane_sigma_used = sigma;
        }
        let pane = self.pane_decider.update(
            now,
            gaze,
            rect,
            self.pane_snapshot.as_ref(),
            sigma,
            self.input.last_mouse_activity(),
            self.input.last_key_activity(),
            active,
        );
        if let (Some(target), Some(worker), Some(window)) =
            (&pane.target, &self.pane_worker, &self.pane_window)
        {
            worker.request_focus(window, target);
            self.process_pane_events();
        }
        pane
    }

    fn check_session_lock(&mut self, now: f64) {
        let Some(locked) = self.platform.is_session_locked() else {
            return;
        };
        if locked == self.session_locked {
            return;
        }
        self.session_locked = locked;
        if locked {
            log::info!("Session locked");
            return;
        }
        log::info!("Session unlocked");
        // Whoever unlocked is present; start the walk-away timers afresh.
        self.presence.reset(now);
        self.away = false;
        self.displays_off_only = false;
        self.away_displays_off = false;
        self.hide_countdown();
        // Pointer moves refused around the lock (the lock screen owns the input desktop until
        // it is noticed, at most LOCK_POLL_S) say nothing about the unlocked desktop: a
        // backoff armed then must not outlive the lock.
        self.warp_refusals = 0;
        self.warp_suspensions = 0;
        self.switching_suspended_until = f64::NEG_INFINITY;
        if self.guard_locked {
            // The user just unlocked a lock the shoulder guard caused. If the onlooker is
            // still there, cover the screens rather than lock again.
            self.guard_locked = false;
            self.guard_relock_until = now + GUARD_RELOCK_GRACE_S;
        }
    }

    /// Decide whether another app needs the camera (`yield_reason`).
    ///
    /// Skipped while the camera is off anyway or explicitly wanted, unless `force` (the camera
    /// is about to be switched back on).
    fn check_camera_yield(&mut self, now: f64, force: bool) {
        if !force
            && matches!(
                self.state,
                TrackingState::Privacy
                    | TrackingState::Locked
                    | TrackingState::Paused
                    | TrackingState::Calibrating
            )
        {
            return;
        }
        self.yield_checked_at = now;
        let p = &self.settings.privacy;
        let apps: Vec<String> = p
            .pause_for_apps
            .iter()
            .filter(|a| !a.trim().is_empty())
            .cloned()
            .collect();
        let mut reason = String::new();
        if !apps.is_empty() || p.yield_camera {
            // Listing processes costs a few milliseconds; only when it matters.
            let running = if apps.is_empty() {
                HashSet::new()
            } else {
                self.platform.running_process_names()
            };
            let in_use = if p.yield_camera {
                self.platform.camera_in_use_by_other_app()
            } else {
                None
            };
            let (yes, why) = should_yield(
                &YieldInputs {
                    camera_in_use_by_other: in_use,
                    running,
                    pause_for_apps: apps,
                },
                p.yield_camera,
            );
            if yes {
                reason = why;
            }
        }
        if reason == self.yield_reason {
            return;
        }
        let previous = std::mem::replace(&mut self.yield_reason, reason.clone());
        if !reason.is_empty() {
            log::info!("Releasing the camera: {reason}");
            if previous.is_empty() {
                self.notify(
                    "Tracking paused",
                    &format!("{reason} — the camera was released."),
                    false,
                );
            }
        } else {
            log::info!("The camera is free again; tracking resumes");
        }
    }

    fn emit_stats(&mut self) {
        let stats = self.safe_worker_stats();
        let active = self.state.camera_active();
        let target_fps = match self.interval {
            Some(interval) if active && interval > 0.0 => json!(finite(1.0 / interval, 2)),
            _ => json!(0.0),
        };
        let cpu = self.measure_cpu();
        let mut data = Map::new();
        data.insert(
            "fps".into(),
            if active {
                json!(finite(stats.fps, 2))
            } else {
                json!(0.0)
            },
        );
        data.insert("target_fps".into(), target_fps);
        data.insert("inference_ms".into(), json!(finite(stats.inference_ms, 2)));
        data.insert("skip_ratio".into(), json!(finite(stats.skip_ratio, 3)));
        data.insert("cpu_percent".into(), json!(cpu));
        data.insert("state".into(), json!(self.state.value()));
        data.insert("state_label".into(), json!(self.state.label()));
        data.insert("backend".into(), json!(self.backend_info().0));
        data.insert("camera_open".into(), json!(stats.camera_open));
        data.insert("last_error".into(), json!(stats.last_error));
        data.insert("mode".into(), json!(self.mode));
        self.last_stats = data.clone();
        self.emit(ControllerEvent::StatsChanged(data));
    }

    /// This process's CPU use as a percentage of the whole machine.
    fn measure_cpu(&mut self) -> Option<f64> {
        let pid = sysinfo::get_current_pid().ok()?;
        if self.process.is_none() {
            self.process = Some(sysinfo::System::new());
            self.cpu_count = std::thread::available_parallelism().map_or(1, usize::from);
        }
        let system = self.process.as_mut()?;
        system.refresh_processes_specifics(
            sysinfo::ProcessesToUpdate::Some(&[pid]),
            false,
            sysinfo::ProcessRefreshKind::nothing().with_cpu(),
        );
        let Some(process) = system.process(pid) else {
            log::debug!("CPU measurement failed");
            return None;
        };
        let value = f64::from(process.cpu_usage()) / self.cpu_count as f64;
        self.cpu_percent = Some(round_to(value, 2));
        self.cpu_percent
    }

    // ========================================================== worker events
    /// Drop preview frames still queued: they were captured while the camera was on (or the
    /// preview wanted), and would stay on screen under "camera off". Other events are kept.
    fn discard_queued_previews(&mut self) {
        let fresh: Vec<WorkerEvent> = self.worker_rx.try_iter().collect();
        self.held_worker_events.extend(fresh);
        self.held_worker_events
            .retain(|e| !matches!(e, WorkerEvent::Preview(_)));
    }

    fn deliver_preview(&mut self, frame: Frame) {
        // A frame the worker captured just before the camera was switched off (privacy mode,
        // pause, lock) arrives after the state change; shown then, it would stay on screen
        // under "camera off". Dropped, like the late observations in handle_observation.
        if !self.preview_owners.is_empty() && !self.closed && self.state.camera_active() {
            self.emit(ControllerEvent::PreviewFrame(frame));
        }
    }

    fn handle_worker_stats(&mut self, stats: WorkerStats) {
        if self.closed {
            return;
        }
        self.worker_stats = stats.clone();
        self.sync_backend_info();
        if self.state.camera_active() {
            self.judge_camera(&stats);
        }
        self.update_state();
    }

    /// Set the camera error from the worker's statistics.
    ///
    /// A backend that keeps failing counts too: the camera would look healthy while nothing is
    /// analysed.
    fn judge_camera(&mut self, stats: &WorkerStats) {
        let has_error = stats.last_error.as_deref().is_some_and(|e| !e.is_empty());
        let failing = stats
            .extra
            .get("backend_failing")
            .is_some_and(|v| v.as_bool().unwrap_or(false));
        let error = has_error && (!stats.camera_open || failing);
        self.set_camera_error(error, stats.last_error.clone());
    }

    /// Read the worker's error back when the camera stays silent after switching on.
    ///
    /// The worker publishes an error only when its message changes, so a camera that fails
    /// the same way as before a pause, lock or yield would otherwise leave the state at
    /// TRACKING with no frames and walk-away detection frozen.
    fn recheck_camera_error(&mut self, now: f64) {
        let Some(since) = self.camera_on_since else {
            return;
        };
        if self.camera_error
            || self.last_obs_time.is_some()
            || !self.state.camera_active()
            || now - since < CAMERA_ERROR_RECHECK_S
        {
            return;
        }
        let stats = self.safe_worker_stats();
        self.judge_camera(&stats);
    }

    fn set_camera_error(&mut self, error: bool, message: Option<String>) {
        if error == self.camera_error {
            if error {
                self.camera_error_message = message;
            }
            return;
        }
        self.camera_error = error;
        self.camera_error_message = if error { message.clone() } else { None };
        if !error {
            log::info!("Camera working again");
            return;
        }
        log::warn!("Camera problem: {}", message.as_deref().unwrap_or("None"));
        let permissions = self.platform.permissions();
        if permissions.get("camera") == Some(&Some(false)) {
            self.notify(
                "Camera access blocked",
                "The system privacy settings do not allow Eye Tracker to use the camera. Allow \
                 camera access there; tracking starts by itself afterwards.",
                false,
            );
            self.emit(ControllerEvent::PermissionNeeded("camera".into()));
        } else {
            let message = message
                .filter(|m| !m.is_empty())
                .unwrap_or_else(|| "The camera could not be opened.".into());
            self.notify("Camera unavailable", &message, false);
        }
    }

    // ============================================================== observations
    fn handle_observation(&mut self, obs: &Observation) {
        if self.closed || !self.started || !self.state.camera_active() {
            return; // a frame analysed just before the camera was switched off
        }
        let now = self.now();
        let previous = self.last_obs_time;
        self.last_obs_time = Some(now);
        if self.camera_error {
            self.set_camera_error(false, None); // frames arrive, so the camera works
            self.update_state();
        }
        let face = self.face_verdict(obs, now);
        self.note_frame_size(obs);
        self.emit(ControllerEvent::Observation(Box::new(obs.clone())));
        if self.state == TrackingState::Calibrating {
            self.last_face = face;
            // The guard sees nothing meanwhile: no multi-face run may span the calibration
            // and trigger on the first frame after it.
            self.update_guard(now, None, None);
            self.update_rate(now, false);
            return;
        }
        let stale = STALE_OBSERVATION_S.max(2.5 * self.interval.unwrap_or(0.0));
        if previous.is_none_or(|p| now - p > stale) {
            // The stream was interrupted: nothing is "continuous".
            self.update_guard(now, None, None);
        }
        let faces = if obs.blind {
            None
        } else {
            Some(obs.face_count)
        };
        self.update_guard(now, faces, obs.face_box);
        // Every detected face counts as the user for walk-away detection, also one the
        // shoulder guard judges to be the onlooker's: faces are compared by their boxes, not
        // recognised, and a wrong "that is not the user" would lock out a user sitting at
        // their desk (see engine/guard.rs).
        self.last_face = face;
        self.update_presence(now);
        self.update_state();
        self.last_decision = None;
        self.last_pane_decision = None;
        if self.state == TrackingState::Tracking {
            self.track(obs, now);
            self.update_state();
        }
        self.update_rate(now, false);
        if self.trace.is_some() {
            self.write_trace(obs, now);
        }
    }

    /// The observation's verdict for walk-away detection: face, no face, or `None` when the
    /// camera cannot tell.
    ///
    /// A blind frame (covered lens, closed shutter, dark room) cannot tell. If the blindness
    /// began while the user was demonstrably there (face just seen, or recent input), they
    /// covered the camera: presence pauses, for at most BLIND_FREEZE_MAX_S without input.
    /// Blindness that begins after the face was already gone (the user left, then the lights
    /// went out) keeps counting as "nobody here", so the walk-away lock still happens.
    fn face_verdict(&mut self, obs: &Observation, now: f64) -> Option<bool> {
        if !obs.blind {
            if self.blind_since.is_some() {
                log::info!("The camera sees again");
                self.end_blind_episode();
            }
            let present = obs.face_present();
            if present {
                self.face_seen_at = Some(now);
            }
            return Some(present);
        }
        let blind_since = match self.blind_since {
            Some(since) => since,
            None => {
                self.blind_since = Some(now);
                let face_recent = self.face_seen_at.is_some_and(|s| now - s <= BLIND_ONSET_S);
                let input_recent = now - self.input.last_any_activity() <= BLIND_ONSET_S;
                self.blind_freeze = face_recent || input_recent;
                log::info!(
                    "The camera sees nothing (covered or dark){}",
                    if self.blind_freeze {
                        "; walk-away detection paused"
                    } else {
                        ""
                    }
                );
                if self.blind_freeze && now - self.blind_notice_at >= BLIND_NOTICE_COOLDOWN_S {
                    self.blind_notice_at = now;
                    self.notify(
                        "Camera appears covered",
                        "The camera sees nothing, so walk-away detection is paused. Uncover \
                         it, or turn on privacy mode to switch the camera off.",
                        false,
                    );
                }
                now
            }
        };
        if !self.blind_freeze {
            return Some(false);
        }
        let evidence = blind_since.max(self.input.last_any_activity());
        if now - evidence > BLIND_FREEZE_MAX_S {
            return Some(false);
        }
        None
    }

    fn end_blind_episode(&mut self) {
        self.blind_since = None;
        self.blind_freeze = false;
    }

    /// Remember the camera's frame size; a new one can need another calibration.
    fn note_frame_size(&mut self, obs: &Observation) {
        let Some(size) = known_size(obs.frame_size) else {
            return;
        };
        if Some(size) == self.frame_size {
            return;
        }
        self.frame_size = Some(size);
        log::debug!("Camera frames are {}x{}", size.0, size.1);
        // During a calibration the new profile is judged when it is finished.
        if self.calibration.is_some() && self.state != TrackingState::Calibrating {
            self.validate_calibration(true);
        }
    }

    fn write_trace(&mut self, obs: &Observation, now: f64) {
        let cursor = self.cursor_pos();
        let decision = self.last_decision.as_ref();
        let pane = self.last_pane_decision.as_ref();
        let snapshot = self.pane_snapshot.as_ref();
        let gaze = if self.last_gaze_time == Some(now) {
            self.last_gaze.map(|(x, y)| vec![x, y])
        } else {
            None
        };
        let Some(trace) = &self.trace else {
            return;
        };
        let record = json!({
            "t": TraceWriter::number(Some(now), 3),
            "state": self.state.value(),
            "faces": obs.face_count,
            "usable": obs.usable(),
            "skipped": obs.skipped,
            "blink": obs.blink,
            "q": TraceWriter::number(Some(obs.quality), 3),
            "yaw": TraceWriter::number(obs.head_yaw, 2),
            "pitch": TraceWriter::number(obs.head_pitch, 2),
            "ms": TraceWriter::number(Some(obs.inference_ms), 2),
            "f": TraceWriter::vector(obs.features.as_deref(), 4),
            "gaze": TraceWriter::vector(gaze.as_deref(), 1),
            "cursor": cursor.map(|(x, y)| vec![x, y]),
            "cand": decision.and_then(|d| d.candidate),
            "reason": decision.map(|d| d.reason),
            "prog": decision.and_then(|d| TraceWriter::number(Some(d.progress), 2)),
            "target": decision.and_then(|d| d.target),
            // Split panes: ids only (tmux "%3", WezTerm numbers), never titles.
            "pane_cand": pane.and_then(|p| p.candidate.as_ref()).map(trace_id),
            "pane_reason": pane.map(|p| p.reason),
            "pane_prog": pane.and_then(|p| TraceWriter::number(Some(p.progress), 2)),
            "pane_target": pane.and_then(|p| p.target.as_ref()).map(|t| trace_id(&t.id)),
            "n_panes": snapshot.map_or(0, |s| s.panes.len()),
        });
        trace.write(&record);
    }

    fn track(&mut self, obs: &Observation, now: f64) {
        let gaze = self.estimate_gaze(obs, now);
        if !self.model_usable {
            return; // the calibration turned out to be unusable
        }
        self.publish_gaze(gaze, now);
        if self.settings.learning.adaptive && !self.looking_away {
            self.learn(obs, now);
        }
        let cursor = self.cursor_pos();
        let current = self.current_monitor(cursor);
        // No switching (and so no focus change) under the privacy curtain: focus would move
        // to a hidden window, and Esc would no longer reach the curtain. Nor on the lock
        // screen (tracking goes on there when pause_when_locked is off): the desktop is not
        // what the user sees, and the system refuses the pointer moves, which would count
        // towards the refused-warp backoff.
        let enabled = self.settings.switching.enabled
            && !self.curtain
            && !self.session_locked
            && now >= self.switching_suspended_until;
        let decision = self.decider.update(
            now,
            gaze,
            current,
            self.input.last_mouse_activity(),
            self.input.last_key_activity(),
            enabled,
            self.looking_away,
        );
        self.last_decision = Some(decision.clone());
        if let Some(target) = decision.target {
            self.switch_to(target, gaze, cursor, current, now);
        }
        let pane = self.pane_step(now, gaze, &decision, enabled);
        self.pending = decision.pending || pane.pending;
        self.last_pane_decision = Some(pane);
    }

    /// Index of the monitor the pointer is on (`None` if unknown).
    fn current_monitor(&self, cursor: Option<(i32, i32)>) -> Option<i32> {
        if !self.cursor_reliable {
            return self.assumed_monitor;
        }
        let (x, y) = cursor?;
        monitor_at(&self.monitors, f64::from(x), f64::from(y)).map(|m| m.index)
    }

    fn estimate_gaze(&mut self, obs: &Observation, now: f64) -> Option<(f64, f64)> {
        self.looking_away = false;
        self.model()?;
        if let (true, Some(features)) = (obs.usable(), obs.features.as_deref()) {
            let indices = self.gaze_feature_indices();
            let model = self.model()?;
            let result = model
                .looks_away(
                    features,
                    indices.as_deref(),
                    LOOK_AWAY_EXCESS,
                    Some(&self.monitors),
                    LOOK_AWAY_MARGIN,
                )
                .and_then(|away| {
                    if away {
                        Ok(None)
                    } else {
                        model.predict(features).map(Some)
                    }
                });
            let raw = match result {
                Ok(raw) => raw,
                Err(e) => {
                    self.feature_mismatches += 1;
                    if self.feature_mismatches < FEATURE_MISMATCH_LIMIT {
                        // Most likely a frame the previous backend analysed.
                        log::debug!("Gaze model rejected the camera features: {e}");
                        return None;
                    }
                    log::warn!("Gaze model rejected the camera features: {e}");
                    self.invalidate_calibration(
                        "the camera features no longer match the calibration",
                    );
                    return None;
                }
            };
            self.feature_mismatches = 0;
            let Some([x, y]) = raw else {
                // The gaze direction points far off every monitor (phone, desk, a person
                // beside the screens). Forget the last point so a blink right after does not
                // bring it back, and restart the filter.
                self.looking_away = true;
                self.last_gaze = None;
                self.last_gaze_time = None;
                return None;
            };
            if !(x.is_finite() && y.is_finite()) {
                return None;
            }
            if self.last_gaze_time.is_none_or(|t| now - t > FILTER_RESET_S) {
                self.filter.reset();
            }
            let (fx, fy) = self.filter.update(x, y, now);
            if let Some((lx, ly)) = self.last_gaze
                && (fx - lx).hypot(fy - ly) > GAZE_MOVE_FRACTION * self.min_side
            {
                self.gaze_moving_until = now + GAZE_MOVING_HOLD_S;
            }
            self.last_gaze = Some((fx, fy));
            self.last_gaze_time = Some(now);
            return Some((fx, fy));
        }
        if obs.blink
            && let (Some(gaze), Some(at)) = (self.last_gaze, self.last_gaze_time)
            && now - at <= BLINK_HOLD_S
        {
            return Some(gaze);
        }
        None
    }

    fn publish_gaze(&mut self, gaze: Option<(f64, f64)>, now: f64) {
        let Some((x, y)) = gaze else {
            if !self.gaze_none_sent {
                self.gaze_none_sent = true;
                self.emit(ControllerEvent::GazeChanged(None));
            }
            return;
        };
        self.gaze_none_sent = false;
        self.emit(ControllerEvent::GazeChanged(Some(GazePoint {
            x,
            y,
            timestamp: now,
        })));
    }

    fn switch_to(
        &mut self,
        target: i32,
        gaze: Option<(f64, f64)>,
        cursor: Option<(i32, i32)>,
        current: Option<i32>,
        now: f64,
    ) {
        let Some(monitor) = self.monitors.iter().find(|m| m.index == target).cloned() else {
            return;
        };
        if self.cursor_reliable
            && let (Some(current), Some(cursor)) = (current, cursor)
        {
            self.memory.record_cursor(current, cursor);
        }
        let focus_window = self.settings.switching.focus_window;
        let mut window = if focus_window {
            self.remembered_window(&monitor)
        } else {
            None
        };
        let (x, y) = choose_cursor_target(
            &self.settings.switching.cursor_target,
            &monitor,
            &self.memory,
            gaze,
            window.as_ref().and_then(|w| w.rect),
        );
        // Warp first, then activate: input injected by activation fallbacks must not be
        // mistaken for the user typing (see InputTracker).
        self.input.note_programmatic_move((x, y), now);
        if !self.move_cursor(x, y, now) {
            // Nothing moved, so this was no switch: no focus change, no count. The decider
            // armed its cooldown when it fired, so it retries after that.
            return;
        }
        if focus_window {
            if window.is_none() {
                window = self.platform.window_at(x, y);
            }
            if let Some(window) = window {
                self.activate(&monitor, window);
            }
        }

        self.decider.notify_switched(now, target);
        self.pane_decider.reset(); // another monitor: another pane context
        self.assumed_monitor = Some(target);
        if self.last_switch.is_some() {
            self.drift.record_switch(true); // the previous switch was kept until now
        }
        // Without a reliable pointer position an undone switch cannot be seen.
        self.last_switch = self.cursor_reliable.then_some((now, current, target));
        self.switch_count += 1;
        log::debug!("Switched to monitor {target} (cursor {x}, {y})");
        self.emit(ControllerEvent::Switched(target));
    }

    fn activate(&mut self, monitor: &Monitor, window: WindowRef) {
        if self.platform.activate_window(&window) {
            self.memory.record_window(monitor.index, window);
            self.activation_failures = 0;
            return;
        }
        log::debug!("Could not activate the window on monitor {}", monitor.index);
        self.activation_failures += 1;
        if self.activation_failures >= ACTIVATION_FAILURE_LIMIT {
            // Activation also fails for ordinary reasons (Windows' foreground lock, a window
            // that closed); only a missing permission is reported.
            self.activation_failures = 0;
            self.check_accessibility();
        }
    }

    /// The last window used on `monitor` if it still exists and is still there.
    fn remembered_window(&mut self, monitor: &Monitor) -> Option<WindowRef> {
        let window = self.memory.last_window(monitor.index)?.clone();
        if !self.platform.is_window_valid(&window) {
            self.memory.forget_window(monitor.index);
            return None;
        }
        let Some(rect) = self.platform.window_rect(&window).or(window.rect) else {
            return Some(window);
        };
        let (cx, cy) = rect.center();
        if !monitor.rect.contains(cx, cy) {
            self.memory.forget_window(monitor.index); // moved to another monitor
            return None;
        }
        if Some(rect) == window.rect {
            return Some(window);
        }
        Some(WindowRef {
            rect: Some(rect),
            ..window
        })
    }

    /// Warp the pointer; false if the system refused.
    ///
    /// Refusals in a row pause switching (backing off) instead of retrying after every
    /// cooldown, and the user is told once.
    fn move_cursor(&mut self, x: i32, y: i32, now: f64) -> bool {
        // `None`: the platform cannot move the pointer, which is a refusal too.
        if self.platform.move_cursor(x, y) == Some(true) {
            self.warp_refusals = 0;
            self.warp_suspensions = 0;
            self.switching_suspended_until = f64::NEG_INFINITY;
            return true;
        }
        // Windows refuses pointer moves as soon as the secure desktop (lock screen, UAC
        // prompt, Ctrl+Alt+Del) has the input, up to LOCK_POLL_S before the regular poll
        // notices. Ask now: a refusal caused by a lock says nothing about the desktop, so it
        // neither counts towards the backoff nor tells the user that the cursor cannot be
        // moved.
        self.next_lock_check = now + LOCK_POLL_S;
        self.check_session_lock(now);
        if self.session_locked {
            log::info!("The system refused to move the pointer: the session is locked");
            return false;
        }
        self.warp_refusals += 1;
        log::info!(
            "The system refused to move the pointer ({} in a row)",
            self.warp_refusals
        );
        if self.warp_refusals >= WARP_REFUSAL_LIMIT {
            self.warp_suspensions += 1;
            let exponent = (self.warp_suspensions - 1).min(16);
            let delay = WARP_BACKOFF_MAX_S.min(WARP_BACKOFF_S * 2f64.powi(exponent as i32));
            self.switching_suspended_until = now + delay;
            log::warn!("Pointer moves keep failing; switching paused for {delay:.0} s");
            if !self.cursor_warning_shown {
                self.cursor_warning_shown = true;
                self.notify(
                    "Cannot move the cursor",
                    "The system keeps refusing to move the mouse pointer, so switching \
                     monitors is paused for now. On Wayland, install ydotool (see the \
                     documentation).",
                    false,
                );
            }
        }
        false
    }

    // ------------------------------------------------------------------ learning
    fn learn(&mut self, obs: &Observation, now: f64) {
        if !self.model_usable || self.calibration.is_none() || self.learner.max_samples == 0 {
            return;
        }
        let monitors = &self.monitors;
        let lookup = |x: f64, y: f64| monitor_at(monitors, x, y).map(|m| m.index);
        let Some(sample) = self.learner.on_observation(obs, now, &lookup) else {
            return;
        };
        let predicted = self
            .model()
            .and_then(|model| Self::predict_point(model, &sample.features));
        let predicted_monitor = match predicted {
            Some((px, py)) => nearest_monitor(&self.monitors, px, py)
                .ok()
                .map(|(m, _)| m.index),
            None => None,
        };
        // The comparison is drift evidence either way, also for a rejected label.
        self.drift.record(predicted_monitor, sample.monitor_index);
        self.check_drift(now);
        if let Some((px, py)) = predicted
            && !plausible_label(&[px, py], &sample, &self.monitors, IMPLICIT_MAX_ERROR)
        {
            // The pointer was parked while the user read elsewhere: not a label.
            self.learner.discard_last();
            log::debug!(
                "Ignored an implausible learning sample at {:.0}, {:.0}",
                sample.x,
                sample.y
            );
            return;
        }
        self.implicit_dirty = true;
        if self.learner.should_refit() {
            self.refit();
        }
    }

    fn refit(&mut self) {
        self.learner.mark_refit();
        let learned = self.learner.samples().to_vec();
        let Some(cal) = self.calibration.as_mut() else {
            return;
        };
        let refined = match refit_model(
            &cal.samples,
            &learned,
            &cal.model,
            None,
            Some(&cal.monitors),
        ) {
            Ok(refined) => refined,
            Err(e) => {
                log::warn!("Could not refine the gaze model: {e}");
                return;
            }
        };
        cal.model = refined;
        cal.implicit_samples = learned;
        let count = cal.implicit_samples.len();
        self.save_calibration(true);
        log::info!("Gaze model refined with {count} learned samples");
    }

    /// Refit the calibration after learned samples were dropped (smaller capacity).
    fn refit_after_trim(&mut self) {
        let learned = self.learner.samples().to_vec();
        let Some(cal) = self.calibration.as_mut() else {
            return;
        };
        if !cal.model.is_fitted() {
            return;
        }
        let refined = match refit_model(
            &cal.samples,
            &learned,
            &cal.model,
            None,
            Some(&cal.monitors),
        ) {
            Ok(refined) => refined,
            Err(e) => {
                log::warn!("Could not refit the gaze model: {e}");
                return;
            }
        };
        cal.model = refined;
        cal.implicit_samples = learned;
        let count = cal.implicit_samples.len();
        self.save_calibration(true);
        log::info!("Gaze model refitted with the {count} learned samples kept");
    }

    fn check_drift(&mut self, now: f64) {
        if self.settings.learning.drift_alerts
            && self.drift.should_alert(now, DRIFT_ALERT_COOLDOWN_S)
        {
            self.notify(
                "Accuracy dropped",
                "Accuracy dropped — recalibrate? Choose Calibrate… in the tray menu.",
                false,
            );
        }
    }

    fn predict_point(model: &GazeModel, features: &[f64]) -> Option<(f64, f64)> {
        let [px, py] = model.predict(features).ok()?;
        (px.is_finite() && py.is_finite()).then_some((px, py))
    }

    // ================================================================== presence
    fn update_presence(&mut self, now: f64) {
        let mut face = self.presence_face(now);
        if face.is_none()
            && self.presence.state() == PresenceState::Away
            && self.state.camera_active()
            && self.state != TrackingState::Calibrating
        {
            // The camera should be watching but cannot tell (it failed or stalled while the
            // user was away). Input must still be able to prove their return; without input
            // AWAY simply stays, as with "no face".
            face = Some(false);
        }
        let idle = self.input.seconds_since_any(now);
        let events = self.presence.update(now, face, idle);
        self.handle_presence_events(events);
    }

    /// The camera's verdict for presence, `None` when it cannot tell.
    fn presence_face(&self, now: f64) -> Option<bool> {
        let state = self.state;
        if !state.camera_active()
            || matches!(
                state,
                TrackingState::Calibrating | TrackingState::CameraError
            )
        {
            return None;
        }
        let last = self.last_obs_time?;
        let face = self.last_face?;
        let stale = STALE_OBSERVATION_S.max(2.5 * self.interval.unwrap_or(0.0));
        if now - last > stale {
            return None;
        }
        Some(face)
    }

    fn handle_presence_events(&mut self, events: Vec<PresenceEvent>) {
        for event in events {
            match event.kind {
                PresenceEventKind::Warn => {
                    self.warning_shown = true;
                    self.emit(ControllerEvent::AwayWarning(event.remaining_s));
                }
                PresenceEventKind::Cancel => self.hide_countdown(),
                PresenceEventKind::Away => {
                    self.hide_countdown();
                    self.away = true;
                    self.perform_away_action();
                }
                PresenceEventKind::Return => {
                    self.away = false;
                    self.hide_countdown();
                    if self.displays_off_only && self.settings.presence.wake_on_return {
                        log::info!("User returned; waking the displays");
                        self.platform.wake_display();
                    }
                    self.displays_off_only = false;
                    self.away_displays_off = false;
                }
            }
        }
    }

    fn perform_away_action(&mut self) {
        let configured = self.settings.presence.action.clone();
        log::info!("User away; action: {configured}");
        if configured == "none" {
            return;
        }
        let setup_pending = !self.settings.general.first_run_done;
        let mut action = configured.as_str();
        let effective = effective_away_action(&self.settings).to_string();
        if effective != action {
            log::info!("First-run setup not finished: notifying instead of {action}");
            action = &effective;
        }
        if action == "notify" {
            let mut message = String::from("Nobody has been at the computer for a while.");
            if setup_pending {
                message += " Finish the setup to choose what happens when you walk away.";
            }
            self.notify("Are you still there?", &message, true);
            return;
        }
        let lock = matches!(action, "lock" | "lock_and_display_off") && !self.session_locked;
        let display_off = matches!(action, "display_off" | "lock_and_display_off");
        let mut displays_ok = false;
        if display_off {
            // Displays first: a lock screen could otherwise swallow the request.
            displays_ok = self.platform.display_off();
            if !displays_ok {
                log::warn!("Turning the displays off is not supported here");
                if !lock {
                    self.notify(
                        "Could not turn off the displays",
                        "This system does not allow it; choose another walk-away action.",
                        true,
                    );
                }
            }
        }
        // Set before the lock is requested: finish_lock may run synchronously inside
        // request_lock, and a failed lock corrects displays_off_only.
        self.away_displays_off = display_off && displays_ok;
        self.displays_off_only = self.away_displays_off && !lock;
        if lock {
            self.request_lock("away"); // the result arrives in finish_lock
        }
    }

    fn hide_countdown(&mut self) {
        if self.warning_shown {
            self.warning_shown = false;
            self.emit(ControllerEvent::AwayCancelled);
        }
    }

    // ===================================================================== guard
    fn update_guard(&mut self, now: f64, face_count: Option<u32>, face_box: Option<FaceBox>) {
        // Input tells the guard that someone works at the computer: while it judges the user
        // gone, that person's face becomes the user's again.
        let last_input = self.input.last_any_activity();
        match self.guard.update(now, face_count, face_box, last_input) {
            Some(GuardEvent::Trigger) => self.on_guard_trigger(now),
            // Only the curtain comes down. "Clear" means fewer than two faces for a moment
            // (GuardConfig::clear_s), which a colleague who turns to the user or looks down at
            // notes already causes; ending the relock grace here would lock the user out
            // again as soon as they look back.
            Some(GuardEvent::Clear) => self.show_curtain(false),
            None => {}
        }
    }

    fn on_guard_trigger(&mut self, now: f64) {
        let action = self.settings.privacy.guard_action.clone();
        log::info!("Shoulder guard: second face detected; action {action}");
        match action.as_str() {
            "lock" => {
                if self.session_locked {
                    return;
                }
                if now < self.guard_relock_until {
                    log::info!("Shoulder guard: showing the curtain instead of locking again");
                    // Most likely the same visit that followed the unlock: keep the grace
                    // going while the second face keeps coming back.
                    self.guard_relock_until = now + GUARD_RELOCK_GRACE_S;
                    self.show_curtain(true);
                    return;
                }
                // If locking fails, finish_lock shows the privacy curtain instead.
                self.request_lock("guard");
            }
            "curtain" => self.show_curtain(true),
            _ => self.notify(
                "Someone is looking at your screen",
                "A second face has been in view for a few seconds.",
                true,
            ),
        }
    }

    fn show_curtain(&mut self, visible: bool) {
        if visible == self.curtain {
            return;
        }
        self.curtain = visible;
        self.emit(ControllerEvent::GuardChanged(visible));
    }

    fn clear_guard(&mut self) {
        self.guard.reset();
        self.show_curtain(false);
    }

    // ============================================================== screen lock
    /// Lock the session without blocking the GUI thread.
    ///
    /// `lock_screen` may take seconds (a D-Bus call to a busy screensaver, `loginctl`,
    /// AppleScript) and is safe on any thread, so it runs through the lock runner and the
    /// result comes back to [`finish_lock`](Self::finish_lock) on the UI thread. `purpose`
    /// (`"away"` or `"guard"`) decides what a failure does; a request while another is in
    /// flight joins it.
    fn request_lock(&mut self, purpose: &'static str) {
        let first = self.lock_purposes.is_empty();
        self.lock_purposes.insert(purpose);
        if !first {
            return;
        }
        log::info!("Locking the screen ({purpose})");
        let platform = self.platform.clone();
        let results = self.lock_tx.clone();
        // lock thread (or the UI thread, see lock_runner)
        let job: LockJob = Box::new(move || {
            let ok =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| platform.lock_screen()))
                    .unwrap_or_else(|_| {
                        log::debug!("lock_screen failed");
                        false
                    });
            let _ = results.send(ok);
        });
        match &mut self.lock_runner {
            Some(runner) => runner(job),
            None => {
                // Default: a short-lived thread.
                let spawned = std::thread::Builder::new()
                    .name("eye-tracker-lock".into())
                    .spawn(job);
                match spawned {
                    Ok(thread) => self.lock_thread = Some(thread),
                    Err(e) => {
                        log::warn!("Could not start locking the screen: {e}");
                        self.finish_lock(false);
                        return;
                    }
                }
            }
        }
        // A synchronous runner has delivered the result already.
        if let Ok(ok) = self.lock_rx.try_recv() {
            self.finish_lock(ok);
        }
    }

    /// The screen lock requested by [`request_lock`](Self::request_lock) has finished.
    fn finish_lock(&mut self, ok: bool) {
        let purposes = std::mem::take(&mut self.lock_purposes);
        if self.closed || purposes.is_empty() {
            return;
        }
        if ok {
            if purposes.contains("guard") {
                self.guard_locked = true;
            }
            // Notice the lock soon so the camera is released promptly.
            self.next_lock_check = self.next_lock_check.min(self.now() + 0.5);
            return;
        }
        log::warn!("Locking the screen failed or is not supported here");
        if purposes.contains("away") && self.away {
            // The displays may already be off; with the session unlocked they are "off only"
            // and are woken when the user comes back.
            self.displays_off_only = self.away_displays_off;
        }
        if purposes.contains("away") {
            self.notify(
                "Could not lock the screen",
                "This system does not allow it; choose another walk-away action.",
                true,
            );
        }
        if purposes.contains("guard") && self.guard.active() {
            // The onlooker is still there: cover the screens instead.
            log::warn!("Showing the privacy curtain instead of the lock");
            self.show_curtain(true);
        }
    }

    // ===================================================================== state
    fn derive_state(&self) -> TrackingState {
        if !self.started {
            return TrackingState::Starting;
        }
        if self.privacy {
            return TrackingState::Privacy;
        }
        if self.session_locked && self.settings.privacy.pause_when_locked {
            return TrackingState::Locked;
        }
        if self.calibrating {
            return TrackingState::Calibrating;
        }
        if self.paused {
            return TrackingState::Paused;
        }
        if !self.yield_reason.is_empty() {
            return TrackingState::Yielded;
        }
        if self.away {
            return TrackingState::Away;
        }
        if self.camera_error {
            return TrackingState::CameraError;
        }
        if !self.model_usable && self.calibration_useful() {
            return TrackingState::NeedsCalibration;
        }
        TrackingState::Tracking
    }

    /// Whether a calibration would change anything: it only serves monitor switching, so not
    /// with one monitor or with switching turned off. Then presence, privacy mode and the
    /// shoulder guard are all there is, and the state must not ask for a calibration (warning
    /// badge, prompts).
    fn calibration_useful(&self) -> bool {
        self.monitors.len() >= 2 && self.settings.switching.enabled
    }

    fn update_state(&mut self) {
        let mut new = self.derive_state();
        if new == self.state {
            return;
        }
        if self.started
            && unchecked_camera_off(self.state)
            && new.camera_active()
            && new != TrackingState::Calibrating
        {
            // The camera is about to reopen, but whether another app uses it was not checked
            // while it was off: a call may have started meanwhile.
            let now = self.now();
            self.check_camera_yield(now, true);
            self.next_yield_check = now + YIELD_POLL_S;
            new = self.derive_state();
            if new == self.state {
                return;
            }
        }
        let old = std::mem::replace(&mut self.state, new);
        log::info!("State: {} -> {}", old.value(), new.value());
        let now = self.now();
        self.enter_state(old, new, now);
        self.emit(ControllerEvent::StateChanged(new));
    }

    fn enter_state(&mut self, old: TrackingState, new: TrackingState, now: f64) {
        if let Some(worker) = &self.worker {
            worker.set_active(new.camera_active());
        }
        if new.camera_active() && (old == TrackingState::Starting || !old.camera_active()) {
            self.camera_on_since = Some(now);
        }
        if !new.camera_active() {
            // Judged afresh once the camera is back on.
            self.camera_on_since = None;
            self.camera_error = false;
            self.last_obs_time = None;
            self.last_face = None;
            self.end_blind_episode();
            self.clear_guard();
            self.discard_queued_previews(); // captured while the camera was on
        } else if matches!(new, TrackingState::Calibrating | TrackingState::CameraError) {
            // No observations reach the guard in these states; a run in progress must not
            // survive them.
            self.update_guard(now, None, None);
        }
        if !new.camera_active() || new == TrackingState::Calibrating {
            // Time without a camera verdict never counts as absence.
            let events = self.presence.update(now, None, None);
            self.handle_presence_events(events);
        }
        if new != TrackingState::Tracking {
            self.reset_tracking();
        }
        if old == TrackingState::Calibrating || new == TrackingState::Calibrating {
            self.apply_motion_gate();
        }
        if (camera_off(old) || old == TrackingState::Calibrating) && self.yield_checked_at != now {
            self.next_yield_check = self.next_yield_check.min(now);
        }
        if absent_state(old) && !absent_state(new) && new != TrackingState::Calibrating {
            self.remind_calibration(now);
        }
        self.set_background_activity(!nappable_state(new));
        self.update_rate(now, false);
    }

    fn reset_tracking(&mut self) {
        self.decider.reset();
        self.pane_decider.reset();
        self.filter.reset();
        self.pending = false;
        self.last_gaze = None;
        self.last_gaze_time = None;
        self.looking_away = false;
        self.feature_mismatches = 0;
        self.gaze_moving_until = f64::NEG_INFINITY;
        self.publish_gaze(None, 0.0);
    }

    fn update_rate(&mut self, now: f64, force: bool) {
        if self.worker.is_none() {
            return;
        }
        let sw = &self.settings.switching;
        let typing_window = TYPING_RATE_WINDOW_S.max(sw.typing_grace_ms as f64 / 1000.0);
        let ctx = RateContext {
            state: self.state,
            pending_switch: self.pending,
            gaze_moving: now < self.gaze_moving_until,
            typing: now - self.input.last_key_activity() < typing_window,
            face_present: self.last_face != Some(false),
            presence_warning: self.presence.state() == PresenceState::Warning,
            preview: !self.preview_owners.is_empty(),
        };
        let interval = self.policy.interval(&ctx);
        self.mode = self.policy.mode(&ctx);
        if let Some(current) = self.interval
            && !force
            && (interval - current).abs() <= RATE_TOLERANCE * current
        {
            return;
        }
        self.interval = Some(interval);
        if let Some(worker) = &self.worker {
            worker.set_interval(interval);
        }
    }

    fn tick_interval(&self) -> u64 {
        let fast = self.state == TrackingState::Tracking
            || self.presence.state() == PresenceState::Warning;
        if fast { TICK_ACTIVE_MS } else { TICK_IDLE_MS }
    }

    fn apply_motion_gate(&self) {
        let Some(worker) = &self.worker else {
            return;
        };
        let perf = &self.settings.performance;
        // During calibration every frame is a sample; skipped copies would not be.
        let enabled = perf.motion_gate && self.state != TrackingState::Calibrating;
        worker.set_motion_gate(enabled, perf.motion_threshold);
    }

    fn max_faces(&self) -> usize {
        if self.settings.privacy.shoulder_guard {
            2
        } else {
            1
        }
    }

    /// Keep timers unthrottled while tracking (macOS App Nap); a no-op elsewhere.
    fn set_background_activity(&mut self, active: bool) {
        if Some(active) == self.background_active {
            return;
        }
        self.background_active = Some(active);
        self.platform.set_background_activity(active);
    }

    // =============================================================== calibration
    fn load_calibration(&mut self) {
        self.library = CalibrationLibrary::load(&paths::calibration_file());
        let latest = self.library.latest().cloned();
        self.use_profile(latest, false);
        if self.calibration.is_none() {
            self.calibration_reason = "not calibrated yet".into();
        }
    }

    /// Make `data` the calibration in use, with its own learned samples.
    ///
    /// The samples learned for the previous profile are kept in that profile. `save` also
    /// marks `data` as the most recently used and writes the file.
    fn use_profile(&mut self, data: Option<CalibrationData>, save: bool) {
        if data.is_none() && self.calibration.is_none() {
            return;
        }
        self.flush_learned();
        self.calibration = data;
        self.calibration_serial += 1;
        self.model_usable = false;
        self.learner.clear();
        if let Some(data) = &self.calibration
            && !data.implicit_samples.is_empty()
        {
            self.learner.load(data.implicit_samples.clone());
        }
        self.implicit_dirty = false;
        self.drift.reset();
        self.feature_mismatches = 0;
        let mut changed = false;
        if self
            .calibration
            .as_ref()
            .is_some_and(|d| self.learner.samples().len() < d.implicit_samples.len())
        {
            // The learning capacity was lowered after this profile learned (while another
            // profile was in use, or in the settings file): load() kept only what fits, but
            // the stored model was fitted with all of them.
            changed = self.refit_trimmed_profile();
        }
        if save
            && let Some(key) = self.calibration.as_ref().map(CalibrationData::key)
            && self.library.mark_used(&key)
        {
            changed = true;
        }
        if changed {
            self.write_library(true);
        }
    }

    /// Refit the calibration in use with the learned samples the learner kept.
    ///
    /// Otherwise the dropped samples would keep steering the gaze (with a capacity of 0, "no
    /// learned influence" would not hold) and stay in the file. Returns whether the profile
    /// changed (and should be saved); on failure the profile is left as it was.
    fn refit_trimmed_profile(&mut self) -> bool {
        let kept = self.learner.samples().to_vec();
        let max_samples = self.learner.max_samples;
        let Some(data) = self.calibration.as_mut() else {
            return false;
        };
        if data.model.is_fitted() {
            match refit_model(
                &data.samples,
                &kept,
                &data.model,
                None,
                Some(&data.monitors),
            ) {
                Ok(model) => data.model = model,
                Err(e) => {
                    log::warn!(
                        "Could not refit the gaze model after trimming learned samples: {e}"
                    );
                    return false;
                }
            }
        }
        data.implicit_samples = kept;
        log::info!(
            "Learned samples trimmed to the capacity of {max_samples}; gaze model refitted with {}",
            data.implicit_samples.len()
        );
        self.sync_library_entry();
        true
    }

    /// Copy the model and learned samples of the calibration in use into its library entry
    /// (the Python library held the very same object).
    fn sync_library_entry(&mut self) {
        let Some(cal) = &self.calibration else {
            return;
        };
        if let Some(entry) = self.library.get_mut(&cal.key()) {
            entry.model = cal.model.clone();
            entry.implicit_samples = cal.implicit_samples.clone();
        }
    }

    /// Copy unsaved learned samples into the calibration they belong to.
    fn flush_learned(&mut self) {
        if !self.implicit_dirty {
            return;
        }
        if let Some(cal) = self.calibration.as_mut() {
            cal.implicit_samples = self.learner.samples().to_vec();
            self.sync_library_entry();
        }
    }

    /// `data` with the camera device and frame size filled in where unknown.
    fn with_camera_identity(&self, mut data: CalibrationData) -> CalibrationData {
        if data.camera.trim().is_empty() {
            data.camera = self.settings.camera.device.clone();
        }
        if known_size(data.frame_size).is_none()
            && let Some(size) = self.frame_size
        {
            data.frame_size = size;
        }
        data
    }

    /// Store the calibration in use as the most recently used profile and write the file.
    fn save_calibration(&mut self, quiet: bool) -> bool {
        self.flush_learned();
        let Some(cal) = self.calibration.clone() else {
            return false;
        };
        self.library.put(cal);
        self.write_library(quiet)
    }

    fn write_library(&mut self, quiet: bool) -> bool {
        if let Err(e) = self.library.save(&paths::calibration_file()) {
            log::error!("Could not save the calibration: {e}");
            if !quiet {
                self.notify("Calibration not saved", &e.to_string(), true);
            }
            return false;
        }
        self.implicit_dirty = false;
        true
    }

    /// Decide which calibration fits the backend, monitors and camera in use.
    ///
    /// The profile in use is kept while it fits; otherwise the best saved profile for the
    /// current setup is taken (and learned samples follow it). Without one the calibration is
    /// unusable and, if it was usable and `announce`, `calibration_required` is emitted (now or
    /// once the user is back).
    fn validate_calibration(&mut self, announce: bool) {
        let was_usable = self.model_usable;
        if self.monitors.is_empty() && self.calibration.is_some() {
            return; // nothing to compare with; keep the current verdict
        }
        let reason = match self.match_profile() {
            ProfileMatch::Unusable(reason) => reason,
            found => {
                let switched = matches!(found, ProfileMatch::Other(_));
                if let ProfileMatch::Other(data) = found {
                    log::info!(
                        "Using the calibration of {} for this setup",
                        if data.created_at.is_empty() {
                            "an earlier session"
                        } else {
                            &data.created_at
                        }
                    );
                    self.use_profile(Some(*data), true);
                }
                if !was_usable || switched {
                    log::info!("Calibration is usable");
                    self.reset_tracking();
                }
                self.model_usable = true;
                self.calibration_reason = String::new();
                self.announce_pending = None;
                self.interruptible_since = None;
                return;
            }
        };
        self.model_usable = false;
        self.calibration_reason = reason.clone();
        if was_usable {
            log::warn!("Calibration no longer usable: {reason}");
            if announce {
                self.announce_calibration(&reason);
            }
        }
    }

    /// The profile to use for the current setup.
    fn match_profile(&self) -> ProfileMatch {
        let current = self.calibration.as_ref();
        let camera = self.settings.camera.device.as_str();
        let frame_size = self.frame_size;
        if let Some((name, version)) = self.current_backend() {
            let mut reason = String::from("not calibrated yet");
            if let Some(current) = current {
                let (ok, why) = current.is_compatible(
                    &name,
                    &version,
                    &self.monitors,
                    Some(camera),
                    frame_size,
                );
                if ok {
                    return ProfileMatch::Current;
                }
                reason = why;
            }
            let (data, why) = self.library.match_profile(
                &name,
                &version,
                &self.monitors,
                Some(camera),
                frame_size,
            );
            if let Some(data) = data {
                return ProfileMatch::Other(Box::new(data.clone()));
            }
            return ProfileMatch::Unusable(if current.is_some() { reason } else { why });
        }
        // Backend unknown (none installed?): the layout is all that can be checked.
        let signature = layout_signature(&self.monitors);
        let fits = |p: &CalibrationData| p.model.is_fitted() && p.layout_signature == signature;
        if current.is_some_and(fits) {
            return ProfileMatch::Current;
        }
        if let Some(other) = self.library.iter().find(|p| fits(p)) {
            return ProfileMatch::Other(Box::new(other.clone()));
        }
        match current {
            None => ProfileMatch::Unusable("not calibrated yet".into()),
            Some(current) if !current.model.is_fitted() => {
                ProfileMatch::Unusable("the calibration has no fitted model".into())
            }
            Some(_) => ProfileMatch::Unusable("the monitor layout changed".into()),
        }
    }

    fn invalidate_calibration(&mut self, reason: &str) {
        let was_usable = self.model_usable;
        self.model_usable = false;
        self.calibration_reason = reason.to_string();
        if was_usable {
            self.announce_calibration(reason);
        }
    }

    /// Emit `calibration_required` now, or once the user can be interrupted.
    fn announce_calibration(&mut self, reason: &str) {
        if self.may_interrupt() {
            self.emit_calibration_required(reason.to_string());
        } else {
            log::debug!("Calibration announcement deferred: {reason}");
            self.announce_pending = Some(reason.to_string());
            self.interruptible_since = None;
        }
    }

    /// The user is back (from away, a lock or privacy mode): remind them of a calibration
    /// that is still unusable, unless that was said recently.
    ///
    /// Also when there never was one: otherwise a setup that starts at login and was never
    /// calibrated would never be told why nothing switches.
    fn remind_calibration(&mut self, now: f64) {
        if self.model_usable
            || self.announce_pending.is_some()
            || now - self.last_announce < CALIBRATION_REMINDER_S
        {
            return;
        }
        self.announce_pending = Some(self.calibration_reason.clone());
        self.interruptible_since = None;
    }

    /// Deliver a deferred announcement once the user has been interruptible for
    /// ANNOUNCE_SETTLE_S with no monitor change pending.
    fn deliver_calibration_notice(&mut self, now: f64) {
        let Some(pending) = self.announce_pending.clone() else {
            return;
        };
        if self.model_usable {
            self.announce_pending = None; // usable again: nothing to say
            self.interruptible_since = None;
            return;
        }
        if !self.may_interrupt() || self.screen_refresh_due.is_some() {
            self.interruptible_since = None;
            return;
        }
        let Some(since) = self.interruptible_since else {
            self.interruptible_since = Some(now);
            return;
        };
        if now - since >= ANNOUNCE_SETTLE_S {
            let reason = if self.calibration_reason.is_empty() {
                pending
            } else {
                self.calibration_reason.clone()
            };
            self.emit_calibration_required(reason);
        }
    }

    fn emit_calibration_required(&mut self, reason: String) {
        self.announce_pending = None;
        self.interruptible_since = None;
        self.last_announce = self.now();
        self.emit(ControllerEvent::CalibrationRequired(reason));
    }

    /// Whether a calibration prompt is appropriate right now.
    ///
    /// Not while the user is away or displays are off (monitors drop out during display
    /// sleep and come back unchanged), and not where a calibration would not change anything
    /// (one monitor, or switching turned off; a pending announcement is delivered once that
    /// changes).
    fn may_interrupt(&self) -> bool {
        self.calibration_useful()
            && !self.calibrating
            && !self.displays_off_only
            && !matches!(
                self.state,
                TrackingState::Away | TrackingState::Locked | TrackingState::Privacy
            )
    }

    // =================================================================== backend
    /// A backend factory for the worker that records what it built.
    ///
    /// Every factory gets a new generation number. The worker keeps reporting its previous
    /// backend until it has built the new one, and the new backend can even have the same
    /// identity (`auto` falling back to `lite` again), so identities alone cannot tell a stale
    /// report from a fresh one.
    fn make_backend_factory(&mut self, settings: &Settings) -> BackendFactory {
        self.backend_generation += 1;
        let generation = self.backend_generation;
        let mut create = (self.backend_factory_maker)(settings, self.max_faces());
        let record = self.built_backend.clone();
        Box::new(move || -> Result<Box<dyn VisionBackend>, FactoryError> {
            // worker thread
            let backend = create()?;
            let built = BuiltBackend {
                generation,
                info: (
                    backend.name().to_string(),
                    backend.feature_version().to_string(),
                ),
                gaze_indices: backend_gaze_indices(
                    backend.feature_names(),
                    backend.gaze_features(),
                ),
            };
            *record.lock() = Some(built);
            Ok(backend)
        })
    }

    fn safe_backend_info(&self) -> Option<(String, String)> {
        let worker = self.worker.as_ref()?;
        let built = self.built_backend.lock().clone();
        if let Some(built) = built {
            // Only the backend built by the latest factory counts.
            return (built.generation == self.backend_generation).then_some(built.info);
        }
        if self.backend_generation > 1 {
            return None; // a new backend was requested and is not built yet
        }
        // The worker never used our factory (a test double): trust what it says.
        worker.backend_info()
    }

    fn sync_backend_info(&mut self) {
        let Some(info) = self.safe_backend_info() else {
            return;
        };
        if Some(&info) == self.reported_backend.as_ref() {
            return;
        }
        log::debug!("Vision backend in use: {} ({})", info.0, info.1);
        self.reported_backend = Some(info);
        self.feature_mismatches = 0;
        self.validate_calibration(true);
    }

    fn current_backend(&self) -> Option<(String, String)> {
        self.reported_backend
            .clone()
            .or_else(|| self.expected_backend())
    }

    /// The backend the settings select, before the worker has created it.
    fn expected_backend(&self) -> Option<(String, String)> {
        let name = &self.settings.general.backend;
        if let Some((cached, info)) = self.expected_cache.borrow().as_ref()
            && cached == name
        {
            return info.clone();
        }
        let info = match crate::vision::backends::backend_class(name) {
            Ok(class) => Some((
                class.name().to_string(),
                class.feature_version().to_string(),
            )),
            Err(e) => {
                log::debug!("Cannot determine the vision backend: {e}");
                None
            }
        };
        *self.expected_cache.borrow_mut() = Some((name.clone(), info.clone()));
        info
    }

    fn safe_worker_stats(&self) -> WorkerStats {
        match &self.worker {
            Some(worker) => worker.stats(),
            None => self.worker_stats.clone(),
        }
    }

    // ============================================================ permissions
    /// Tell the user (once) when focus cannot follow the gaze for lack of the macOS
    /// Accessibility permission, or because it belongs to an older build.
    fn check_accessibility(&mut self) {
        let sw = &self.settings.switching;
        if self.accessibility_notice_shown
            || !(sw.enabled && sw.focus_window)
            || self.monitors.len() < 2
        {
            return;
        }
        let status = self.platform.accessibility_status();
        if status != "missing" && status != "stale" {
            return;
        }
        self.accessibility_notice_shown = true;
        let message = if status == "stale" {
            "Keyboard focus cannot follow your gaze: the Accessibility permission belongs to a \
             previous version of Eye Tracker. In System Settings › Privacy & Security › \
             Accessibility, remove Eye Tracker with “−” and add it again."
        } else {
            "Keyboard focus cannot follow your gaze until Eye Tracker may control your \
             computer: System Settings › Privacy & Security › Accessibility."
        };
        self.notify("Accessibility access needed", message, true);
        self.emit(ControllerEvent::PermissionNeeded("accessibility".into()));
    }

    // =================================================================== hotkeys
    fn setup_hotkeys(&mut self) {
        if self.hotkeys.is_none() {
            self.hotkeys = Some((self.hotkey_factory)());
        }
        self.register_hotkeys();
    }

    fn register_hotkeys(&mut self) {
        let Some(manager) = self.hotkeys.clone() else {
            return;
        };
        if self.hotkeys_suspended {
            return;
        }
        manager.unregister_all();
        if !self.settings.hotkeys.enabled {
            // Turning them on again announces every failure afresh.
            self.announced_hotkey_failures.clear();
            manager.stop();
            return;
        }
        if !manager.supported() {
            if let Some(note) = manager.note() {
                log::info!("Global hotkeys unavailable: {note}");
            }
            return;
        }
        let mut failed: Vec<String> = Vec::new();
        let mut failing: HashSet<(String, String)> = HashSet::new();
        for name in HOTKEY_ACTIONS {
            let combo = hotkey_setting(&self.settings, name).trim().to_string();
            if combo.is_empty() {
                continue;
            }
            let sender = self.hotkey_tx.clone();
            let callback: HotkeyCallback = Arc::new(move || {
                // hotkey thread (or the main thread on macOS)
                let _ = sender.send(name.to_string());
            });
            if manager.register(name, HotkeyArg::Text(&combo), callback) {
                continue;
            }
            let key = (name.to_string(), combo.to_lowercase());
            failing.insert(key.clone());
            if self.announced_hotkey_failures.contains(&key) {
                log::debug!("Hotkey {combo} for {name} is still unavailable (already announced)");
                continue;
            }
            let reason = Self::hotkey_error(manager.as_ref(), name);
            failed.push(reason.unwrap_or_else(|| {
                format!("{combo} could not be registered (invalid, or used by another app)")
            }));
        }
        // Pairs that register now (or are no longer configured) are forgotten, so a
        // combination that fails again later is announced again.
        self.announced_hotkey_failures = failing;
        if !failed.is_empty() {
            self.notify(
                "Hotkey unavailable",
                &format!(
                    "{}. Choose another in Settings → Hotkeys.",
                    failed.join("; ")
                ),
                false,
            );
        }
    }

    /// The manager's explanation of why `name` is not registered, if it has one.
    fn hotkey_error(manager: &dyn HotkeyManager, name: &str) -> Option<String> {
        let reason = manager.last_error(name)?;
        // Joined into one sentence with the others; the caller adds the full stop.
        let reason = reason.trim().trim_end_matches('.').to_string();
        (!reason.is_empty()).then_some(reason)
    }

    fn handle_hotkey(&mut self, name: &str) {
        if self.closed {
            return;
        }
        if self.hotkeys_suspended {
            log::debug!("Hotkey {name} ignored while hotkeys are suspended");
            return;
        }
        log::info!("Hotkey: {name}");
        let before = (self.paused, self.privacy);
        match name {
            "toggle_tracking" => {
                self.toggle_pause();
                self.confirm_mode_change(before);
            }
            "toggle_privacy" => {
                self.toggle_privacy();
                self.confirm_mode_change(before);
            }
            "recalibrate" => self.emit(ControllerEvent::CalibrationRequired("hotkey".into())),
            _ => {}
        }
    }

    /// Confirm a pause or privacy change made by a hotkey or `ctl` command.
    ///
    /// Both are toggles, and the tray icon often sits in a hidden overflow area, so without a
    /// word the user cannot tell which way a press went, and pausing also stops walk-away
    /// detection. `before` is `(paused, privacy)` before the change. The tray menu needs no
    /// confirmation: its check marks show the result.
    fn confirm_mode_change(&mut self, before: (bool, bool)) {
        let (paused, privacy) = (self.paused, self.privacy);
        let (was_paused, was_private) = before;
        let stopped = if self.settings.presence.enabled {
            "Switching monitors and walk-away detection pause"
        } else {
            "Switching monitors pauses"
        };
        let back_on = if self.state.camera_active() {
            "The camera is on again.".to_string()
        } else {
            // e.g. the session is locked, or another app has the camera
            format!("The camera stays off for now ({}).", self.state.label())
        };
        let (title, message) = if privacy != was_private {
            if privacy {
                (
                    "Privacy mode on",
                    format!(
                        "The camera is off. {stopped} until you turn privacy mode off{}.",
                        self.hotkey_hint("toggle_privacy")
                    ),
                )
            } else if paused {
                (
                    "Privacy mode off",
                    "Tracking is still paused: the camera stays off.".to_string(),
                )
            } else {
                ("Privacy mode off", back_on)
            }
        } else if paused != was_paused {
            if paused {
                (
                    "Tracking paused",
                    format!(
                        "The camera is released. {stopped} until you resume{}.",
                        self.hotkey_hint("toggle_tracking")
                    ),
                )
            } else if privacy {
                (
                    "Tracking resumed",
                    "Privacy mode is still on: the camera stays off.".to_string(),
                )
            } else {
                ("Tracking resumed", back_on)
            }
        } else {
            return;
        };
        self.notify(title, &message, false);
    }

    /// `" (Ctrl+Alt+Win+T)"` for a registered hotkey `name`, else `""`.
    fn hotkey_hint(&self, name: &str) -> String {
        self.hotkeys
            .as_ref()
            .and_then(|manager| manager.registered().remove(name))
            .map(|hotkey| format!(" ({})", format_hotkey(&hotkey, None)))
            .unwrap_or_default()
    }

    // =================================================================== screens
    fn read_monitors(&mut self) -> Vec<Monitor> {
        (self.monitors_provider)()
    }

    fn set_monitors(&mut self, monitors: Vec<Monitor>) {
        let before = std::mem::replace(&mut self.monitors, monitors);
        let assumed = before
            .iter()
            .find(|m| Some(m.index) == self.assumed_monitor)
            .cloned();
        self.decider.set_monitors(&self.monitors);
        self.pane_decider.reset();
        self.pane_sigma_cache = None;
        let sides: Vec<i32> = self
            .monitors
            .iter()
            .filter(|m| m.rect.w > 0 && m.rect.h > 0)
            .map(|m| m.rect.w.min(m.rect.h))
            .collect();
        self.min_side = sides.iter().min().map_or(1000.0, |s| f64::from(*s));
        if self.cursor_reliable {
            return;
        }
        if let Some(assumed) = assumed {
            // Monitors are numbered in the order the platform lists the screens, so removing
            // a screen (or changing the primary) renumbers the others: follow the physical
            // monitor the pointer was moved to, not its old number.
            self.assumed_monitor = same_monitor(&assumed, &before, &self.monitors).map(|m| m.index);
        }
        if !self
            .monitors
            .iter()
            .any(|m| Some(m.index) == self.assumed_monitor)
        {
            // A stale position is still the best guess until the first switch.
            let cursor = self.cursor_pos();
            self.assumed_monitor = cursor.and_then(|(x, y)| {
                monitor_at(&self.monitors, f64::from(x), f64::from(y)).map(|m| m.index)
            });
        }
    }

    // ==================================================================== helpers
    fn cursor_pos(&self) -> Option<(i32, i32)> {
        self.platform.cursor_position()
    }

    /// Emit `Notify`. Informational messages respect the notifications setting; `force` is
    /// for actions the user chose to be notified about.
    fn notify(&mut self, title: &str, message: &str, force: bool) {
        log::info!("Notification: {title} — {message}");
        if force || self.settings.general.notifications {
            self.emit(ControllerEvent::Notify {
                title: title.to_string(),
                message: message.to_string(),
            });
        }
    }
}

impl Drop for Controller {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// The split-pane providers the settings enable (the Python `_default_pane_registry`).
fn default_pane_registry(
    platform: &Arc<dyn PlatformServices>,
    settings: &Settings,
) -> PaneRegistry {
    let platform = platform.clone();
    // pane worker thread
    let client_rect: ClientRect =
        Arc::new(move |window: &WindowRef| platform.window_client_rect(window));
    PaneRegistry::new(default_providers(
        &settings.panes,
        client_rect,
        None,
        current_system(),
    ))
    .with_desktop_apps(settings.panes.desktop_apps)
}

/// Python's `repr()` of a string, as the unknown-command error quoted it.
fn py_repr(text: &str) -> String {
    if text.contains('\'') && !text.contains('"') {
        format!("\"{text}\"")
    } else {
        format!("'{}'", text.replace('\'', "\\'"))
    }
}

#[cfg(test)]
#[path = "controller_tests.rs"]
mod tests;
