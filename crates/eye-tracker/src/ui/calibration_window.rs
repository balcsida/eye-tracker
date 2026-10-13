//! Full-screen calibration: one surface per monitor, an animated dot and a quality report.
//!
//! Flow (keyboard driven, mouse optional):
//!
//! ```text
//! intro ──Space──▶ running ──(all dots done)──▶ fitting ──▶ result ──Enter──▶ saved
//!   │                │  Space pauses                          │  R retries
//!   └──── Esc cancels at any point ◀──────────────────────────┘
//! ```
//!
//! Each monitor gets its own frameless, full-screen, stay-on-top surface (an immediate viewport
//! at the monitor's rectangle, see [`surface_viewport_builder`]). The
//! [`CalibrationCollector`] decides which dot is shown; the surface of that dot's monitor draws
//! it while the other surfaces point towards it. Observations arrive through
//! [`Inputs::observations`].
//!
//! The state machine ([`CalibrationWindow`], with the per-monitor view models [`Surface`] and
//! [`Card`]) is pure: it takes the time as `Inputs::now` and reports what the Python window
//! called on the controller as [`CalibrationAction`]s, so the whole sequence can be driven
//! deterministically in tests by advancing a fake clock and calling [`CalibrationWindow::tick`].
//! [`CalibrationWindow::show`] draws the surfaces with egui and feeds keys, button clicks and
//! window closes back into the machine.
//!
//! When the user's face disappears for more than [`NO_FACE_PAUSE_S`] the sequence pauses by
//! itself (the collector runs on a clock that stands still while paused) and resumes as soon as
//! the face is back, so a user who glances at the keyboard does not lose dots.
//!
//! While the window is open the controller is in `CALIBRATING`: walk-away detection and the
//! shoulder guard are suspended and the camera runs at the calibration rate. An abandoned window
//! must not keep it that way, so the calibration closes itself after [`IDLE_TIMEOUT_S`] without a
//! key press on the instructions, result or error screen (or while paused with Space), and after
//! [`NO_FACE_TIMEOUT_S`] of dots paused because nobody is in front of the camera.
//!
//! A monitor that is added, removed, resized or moved, or a new primary monitor, cancels the
//! calibration (with a notification): dots recorded on the old geometry could never be saved. The
//! controller polls the monitors and the window compares them with the ones it started on.
//!
//! Fitting the model ([`evaluate`]) takes about 0.1 s, longer on slow machines. It runs on a
//! short-lived background thread whose result the UI tick polls, so the windows stay responsive
//! (Esc works, the spinner turns) meanwhile. `fit_in_thread: false` fits synchronously on the tick
//! after the "Calculating" screen was shown, which keeps tests single-threaded.

use std::collections::BTreeMap;
use std::time::Duration;

use crossbeam_channel::{Receiver, TryRecvError};
use egui::{Color32, Pos2, Rect as UiRect, Stroke, Vec2};

use super::{icons, util};
use crate::gaze::calibration::{
    CalibrationCollector, CalibrationError, CalibrationReport, CalibrationSample,
    CalibrationTarget, CollectorOptions, Event, Phase, evaluate, make_plan,
};
use crate::gaze::model::{GazeModel, gaze_feature_indices};
use crate::gaze::store::CalibrationData;
use crate::types::{Monitor, Observation, layout_signature};
use crate::vision::backends::backend_class;

/// Pause automatically when no face was seen for this long.
pub const NO_FACE_PAUSE_S: f64 = 1.5;
/// Without any observation for this long the camera is reported as missing.
pub const NO_CAMERA_S: f64 = 2.0;
/// Close the calibration after this long without a key press while it waits for the user
/// (instructions, result, error, paused with Space). Walk-away locking is suspended while the
/// window is open.
pub const IDLE_TIMEOUT_S: f64 = 120.0;
/// Close the calibration after the dots were paused this long for lack of a face.
pub const NO_FACE_TIMEOUT_S: f64 = 60.0;

/// Animation / collector tick while running.
pub const TICK_MS: u64 = 30;
/// Intro: only the face indicator needs refreshing.
pub const IDLE_TICK_MS: u64 = 250;
/// The inactivity check also runs on screens that stop the animation timer.
pub const WATCHDOG_MS: u64 = 1000;

/// Share of the settle phase used to glide from the previous dot (same monitor) or to pop in
/// (new monitor). Moving targets are easier to follow than jumps.
pub const GLIDE_SHARE: f64 = 0.45;
pub const POP_SHARE: f64 = 0.3;

const ACCENT: Color32 = Color32::from_rgb(0x63, 0x66, 0xF1);
const ACCENT_2: Color32 = Color32::from_rgb(0x22, 0xD3, 0xEE);
const TEXT: Color32 = Color32::from_rgb(0xF5, 0xF7, 0xFA);
const MUTED: Color32 = Color32::from_rgb(0xA0, 0xA8, 0xB8);
const SUCCESS: Color32 = Color32::from_rgb(0x34, 0xD3, 0x99);
const WARNING: Color32 = Color32::from_rgb(0xF5, 0x9E, 0x0B);
const DANGER: Color32 = Color32::from_rgb(0xEF, 0x44, 0x44);
/// 92 % opacity.
const BG_INNER: Color32 = Color32::from_rgba_premultiplied(20, 24, 42, 235);
const BG_OUTER: Color32 = Color32::from_rgba_premultiplied(6, 8, 15, 235);

/// The color of a grade's badge.
fn grade_badge_color(grade: &str) -> Color32 {
    match grade {
        "excellent" => SUCCESS,
        "good" => ACCENT_2,
        "fair" => WARNING,
        "poor" => DANGER,
        _ => TEXT,
    }
}

// ============================================================ small helpers
fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

fn ease_out_cubic(t: f64) -> f64 {
    1.0 - (1.0 - clamp01(t)).powi(3)
}

fn ease_in_out(t: f64) -> f64 {
    let t = clamp01(t);
    if t < 0.5 {
        4.0 * t.powi(3)
    } else {
        1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
    }
}

fn ease_out_back(t: f64) -> f64 {
    let t = clamp01(t);
    let c1 = 1.70158;
    1.0 + (c1 + 1.0) * (t - 1.0).powi(3) + c1 * (t - 1.0).powi(2)
}

fn with_alpha(color: Color32, alpha: f64) -> Color32 {
    let [r, g, b, a] = color.to_srgba_unmultiplied();
    Color32::from_rgba_unmultiplied(r, g, b, (clamp01(alpha) * f64::from(a)).round() as u8)
}

fn grade_color(value: f64) -> Color32 {
    if value >= 0.97 {
        SUCCESS
    } else if value >= 0.9 {
        ACCENT_2
    } else if value >= 0.75 {
        WARNING
    } else {
        DANGER
    }
}

/// `floor(value * 100 + 1e-9)`: the percentage shown, never rounded up.
fn percent(value: f64) -> i64 {
    (value * 100.0 + 1e-9).floor() as i64
}

/// Monitors left to right, then top to bottom (the order the dots visit them).
fn spatial_order(monitors: &[Monitor]) -> Vec<&Monitor> {
    let mut ordered: Vec<&Monitor> = monitors.iter().collect();
    ordered.sort_by_key(|m| (m.rect.x, m.rect.y, m.index));
    ordered
}

/// Human labels numbered left to right ("Screen 1 · main").
fn monitor_labels(monitors: &[Monitor]) -> BTreeMap<i32, String> {
    spatial_order(monitors)
        .into_iter()
        .enumerate()
        .map(|(n, monitor)| {
            let mut text = format!("Screen {}", n + 1);
            if monitor.primary {
                text.push_str(" · main");
            }
            (monitor.index, text)
        })
        .collect()
}

/// The primary monitor, or the first one when none is flagged.
fn primary_of(monitors: &[Monitor]) -> Option<&Monitor> {
    monitors.iter().find(|m| m.primary).or(monitors.first())
}

// ================================================================ interface
/// Everything the window reads from the controller and the settings, one frame at a time.
///
/// `observations` are the ones that arrived since the previous frame, oldest first.
#[derive(Debug, Clone, Default)]
pub struct Inputs {
    /// Monotonic clock in seconds.
    pub now: f64,
    /// `controller.monitors()`; empty when they could not be listed.
    pub monitors: Vec<Monitor>,
    /// `controller.backend_info()`: backend name and feature version (`""` when not ready).
    pub backend_info: (String, String),
    /// `controller.gaze_feature_indices()`; `None` derives them from the backend class.
    pub gaze_feature_indices: Option<Vec<usize>>,
    /// `settings.camera.device`: the camera the calibration belongs to.
    pub camera_device: String,
    pub observations: Vec<Observation>,
}

/// What the window would have called on the controller or emitted as a signal.
#[derive(Debug, Clone)]
pub enum CalibrationAction {
    /// `controller.begin_calibration()`: the surfaces are open and the controller enters
    /// `CALIBRATING`.
    Started,
    /// The `state_changed` signal.
    StateChanged(CalibrationState),
    /// `controller.notify` (shown by the tray).
    Notify { title: String, message: String },
    /// `controller.finish_calibration(data)` after a save; `grade` is the report's grade.
    Finished {
        data: Box<CalibrationData>,
        grade: String,
    },
    /// `controller.finish_calibration(None)`: closed without saving.
    Cancelled,
    /// `finished(False)` without a calibration ever having begun: there is no monitor.
    NotStarted,
}

/// A key the window reacts to (everything else only proves someone is at the keyboard).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Key {
    Escape,
    Enter,
    Space,
    R,
    Other,
}

/// States of the calibration window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationState {
    /// Constructed, not started.
    Idle,
    /// Instructions shown, waiting for Space.
    Intro,
    /// Dots are being shown.
    Running,
    /// Computing the model.
    Fitting,
    /// Report shown, waiting for Save / Retry / Cancel.
    Result,
    /// Not enough data (or another problem); Retry / Cancel.
    Error,
    /// Finished (saved or cancelled).
    Closed,
}

impl CalibrationState {
    /// The Python `STATE_*` string.
    pub fn value(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Intro => "intro",
            Self::Running => "running",
            Self::Fitting => "fitting",
            Self::Result => "result",
            Self::Error => "error",
            Self::Closed => "closed",
        }
    }
}

/// The settings of a [`CalibrationWindow`] (its Python keyword arguments).
///
/// `points_per_monitor` is the number of dots per monitor (1, 5 or a square number, see
/// [`make_plan`]); `settle_s`, `collect_s`, `min_samples` and `max_retries` are the collector
/// timing; `margin` is the distance of the outer dots from the screen edges (fraction);
/// `fit_in_thread` fits the model on a background thread instead of synchronously on the next
/// tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibrationOptions {
    pub points_per_monitor: usize,
    pub settle_s: f64,
    pub collect_s: f64,
    pub min_samples: usize,
    pub max_retries: usize,
    pub margin: f64,
    pub fit_in_thread: bool,
}

impl Default for CalibrationOptions {
    fn default() -> Self {
        CalibrationOptions {
            points_per_monitor: 9,
            settle_s: 0.8,
            collect_s: 1.0,
            min_samples: 5,
            max_retries: 1,
            margin: 0.1,
            fit_in_thread: true,
        }
    }
}

type FitOutput = (GazeModel, CalibrationReport);

/// How a fit ended badly.
enum FitFailure {
    /// `evaluate` refused the data (the Python `ValueError`): not enough usable samples.
    NotEnoughData(CalibrationError),
    /// The fitting thread died.
    Crashed(String),
}

/// Indices of the vision backend's gaze-direction features (head rotation, eyes).
///
/// They are what [`evaluate`] takes as `nonlinear`: the model gets polynomial terms only for
/// them, so a changed posture (head position) does not bend the fit. Uses the controller's
/// `gaze_feature_indices` when it gave them and otherwise derives them from the class of the
/// backend `backend_info` names. `None` when neither is known, which means every feature is
/// treated alike (the behaviour of older calibrations).
pub fn controller_gaze_features(inputs: &Inputs) -> Option<Vec<usize>> {
    if let Some(indices) = &inputs.gaze_feature_indices {
        return Some(indices.clone());
    }
    let name = inputs.backend_info.0.as_str();
    if name.is_empty() {
        return None;
    }
    let class = match backend_class(name) {
        Ok(class) => class,
        Err(error) => {
            log::debug!("No backend class for the calibration: {error}");
            return None;
        }
    };
    gaze_feature_indices(class.feature_names(), class.gaze_features())
}

/// `indices`, or `None` when they do not fit the recorded feature vectors.
///
/// The backend may have changed between the start of the calibration and the fit; a stale index
/// must degrade the model (every feature nonlinear), not fail it.
fn fit_features(indices: Option<Vec<usize>>, samples: &[CalibrationSample]) -> Option<Vec<usize>> {
    let indices = indices.filter(|i| !i.is_empty())?;
    let Some(first) = samples.first() else {
        return Some(indices);
    };
    let n_features = first.features.len();
    if indices.iter().max().is_some_and(|&max| max >= n_features) {
        log::warn!(
            "Gaze feature indices {indices:?} do not fit {n_features} features; fitting without them"
        );
        return None;
    }
    Some(indices)
}

/// Run [`evaluate`] on a daemon-like thread; the returned receiver is polled by the UI.
///
/// Polling means the thread never touches the UI, so closing the calibration while it runs is
/// always safe: the receiver is dropped and the result is simply lost.
fn start_fit(
    samples: Vec<CalibrationSample>,
    monitors: Vec<Monitor>,
    nonlinear: Option<Vec<usize>>,
) -> Receiver<Result<FitOutput, CalibrationError>> {
    let (sender, receiver) = crossbeam_channel::bounded(1);
    let spawned = std::thread::Builder::new()
        .name("eye-tracker-calibration-fit".into())
        .spawn(move || {
            // The receiver may be gone (calibration closed); nobody wants the result then.
            let _ = sender.send(evaluate(&samples, &monitors, None, nonlinear.as_deref()));
        });
    if let Err(error) = spawned {
        // The sender went down with the failed closure: the poll reports a crashed fit.
        log::error!("Could not start the calibration fit thread: {error}");
    }
    receiver
}

// ================================================================== view models
/// What the active surface draws for the current target.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DotState {
    /// Global coordinates of the dot.
    pub x: f64,
    pub y: f64,
    /// Global coordinates of the target (the dot's destination).
    pub target_x: f64,
    pub target_y: f64,
    /// 0..1 appear animation.
    pub pop: f64,
    /// 0..1 settle progress (shrinking ring).
    pub ring: Option<f64>,
    /// 0..1 collect progress (filling arc).
    pub arc: Option<f64>,
    pub retry: bool,
    pub paused: bool,
}

/// A text of a [`Card`] (its `add_label` arguments); `name` makes it findable.
#[derive(Debug, Clone, PartialEq)]
pub struct CardLabel {
    pub text: String,
    pub name: Option<&'static str>,
    /// Point size.
    pub size: f32,
    pub bold: bool,
    pub muted: bool,
    pub color: Option<Color32>,
}

impl CardLabel {
    fn new(text: impl Into<String>) -> Self {
        CardLabel {
            text: text.into(),
            name: None,
            size: 11.0,
            bold: false,
            muted: false,
            color: None,
        }
    }

    fn named(mut self, name: &'static str) -> Self {
        self.name = Some(name);
        self
    }

    fn size(mut self, size: f32) -> Self {
        self.size = size;
        self
    }

    fn bold(mut self) -> Self {
        self.bold = true;
        self
    }

    fn muted(mut self) -> Self {
        self.muted = true;
        self
    }

    fn color(mut self, color: Color32) -> Self {
        self.color = Some(color);
        self
    }
}

/// A button of a [`Card`]: `name` identifies it for [`CalibrationWindow::press_button`].
#[derive(Debug, Clone, PartialEq)]
pub struct CardButton {
    pub name: &'static str,
    pub text: &'static str,
    pub primary: bool,
}

/// One row of a [`Card`].
#[derive(Debug, Clone, PartialEq)]
pub enum CardRow {
    Label(CardLabel),
    /// An upper-case pill, findable as the label `"badge"`.
    Badge {
        text: String,
        color: Color32,
    },
    /// Key caps with what they do: `(key, action)`.
    Keys(Vec<(&'static str, &'static str)>),
    /// A progress bar per row: `(label, value 0..1)`.
    Bars(Vec<(String, f64)>),
    Buttons(Vec<CardButton>),
    Spinner,
    /// Vertical space in points.
    Spacing(f32),
}

/// Centred content panel (instructions, results) built from simple parts.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Card {
    pub rows: Vec<CardRow>,
}

impl Card {
    fn clear(&mut self) {
        self.rows.clear();
    }

    fn add_label(&mut self, label: CardLabel) {
        self.rows.push(CardRow::Label(label));
    }

    fn add_badge(&mut self, text: &str, color: Color32) {
        self.rows.push(CardRow::Badge {
            text: text.to_uppercase(),
            color,
        });
    }

    fn add_keys(&mut self, items: &[(&'static str, &'static str)]) {
        self.rows.push(CardRow::Keys(items.to_vec()));
    }

    fn add_bars(&mut self, rows: Vec<(String, f64)>) {
        self.rows.push(CardRow::Bars(rows));
    }

    fn add_buttons(&mut self, buttons: &[CardButton]) {
        self.rows.push(CardRow::Buttons(buttons.to_vec()));
    }

    fn add_spinner(&mut self) {
        self.rows.push(CardRow::Spinner);
    }

    fn add_spacing(&mut self, points: f32) {
        self.rows.push(CardRow::Spacing(points));
    }

    /// The text of the label (or badge) called `name`.
    pub fn label(&self, name: &str) -> Option<&str> {
        self.rows.iter().find_map(|row| match row {
            CardRow::Label(label) if label.name == Some(name) => Some(label.text.as_str()),
            CardRow::Badge { text, .. } if name == "badge" => Some(text.as_str()),
            _ => None,
        })
    }

    /// The names of the buttons, left to right.
    pub fn button_names(&self) -> Vec<&'static str> {
        self.rows
            .iter()
            .filter_map(|row| match row {
                CardRow::Buttons(buttons) => Some(buttons.iter().map(|b| b.name)),
                _ => None,
            })
            .flatten()
            .collect()
    }

    /// Whether the card shows the "calculating" spinner.
    pub fn has_spinner(&self) -> bool {
        self.rows.iter().any(|row| matches!(row, CardRow::Spinner))
    }

    fn set_label(&mut self, name: &str, text: &str, color: Color32) {
        for row in &mut self.rows {
            if let CardRow::Label(label) = row
                && label.name == Some(name)
                && label.text != text
            {
                label.text = text.to_string();
                label.color = Some(color);
            }
        }
    }
}

/// What a surface shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceMode {
    Card,
    Running,
}

/// One monitor's full-screen calibration window (the view model [`CalibrationWindow::show`]
/// draws).
#[derive(Debug, Clone)]
pub struct Surface {
    pub monitor: Monitor,
    pub mode: SurfaceMode,
    /// Whether this monitor shows the dot (the others point towards it).
    pub active: bool,
    pub dot: Option<DotState>,
    pub progress: f64,
    pub progress_text: String,
    pub hint: String,
    pub hint_color: Color32,
    pub keys_text: String,
    pub card: Card,
    /// Whether the pointer is hidden over this surface (while the dots run).
    pub cursor_hidden: bool,
}

impl Surface {
    fn new(monitor: Monitor) -> Self {
        Surface {
            monitor,
            mode: SurfaceMode::Card,
            active: false,
            dot: None,
            progress: 0.0,
            progress_text: String::new(),
            hint: String::new(),
            hint_color: WARNING,
            keys_text: String::new(),
            card: Card::default(),
            cursor_hidden: false,
        }
    }

    /// Switch to the card and hand it over for rebuilding.
    fn show_card(&mut self) -> &mut Card {
        self.mode = SurfaceMode::Card;
        self.cursor_hidden = false;
        self.card.clear();
        &mut self.card
    }
}

/// The viewport of one surface: frameless, always on top, at the monitor's origin and as large
/// as it.
pub fn surface_viewport_builder(monitor: &Monitor) -> egui::ViewportBuilder {
    let rect = monitor.rect;
    icons::with_app_icon(
        egui::ViewportBuilder::default()
            .with_title("Eye Tracker — calibration")
            .with_decorations(false)
            .with_always_on_top()
            .with_transparent(true)
            .with_resizable(false)
            .with_position(Pos2::new(rect.x as f32, rect.y as f32))
            .with_inner_size(Vec2::new(rect.w as f32, rect.h as f32)),
    )
}

// ================================================================== window
/// Runs a calibration on every monitor and hands the result to the controller.
///
/// Keys: Space starts / pauses, Enter saves, R retries, Esc cancels.
pub struct CalibrationWindow {
    points_per_monitor: usize,
    timing: CollectorOptions,
    margin: f64,
    fit_in_thread: bool,
    fit: Option<Receiver<Result<FitOutput, CalibrationError>>>,
    fit_started: f64,

    state: CalibrationState,
    monitors: Vec<Monitor>,
    labels: BTreeMap<i32, String>,
    surfaces: Vec<Surface>,
    collector: Option<CalibrationCollector>,
    plan: Vec<CalibrationTarget>,
    model: Option<GazeModel>,
    report: Option<CalibrationReport>,
    data: Option<CalibrationData>,
    error: String,

    prev_target: Option<CalibrationTarget>,
    last_target: Option<CalibrationTarget>,
    retrying: bool,
    manual_pause: bool,
    auto_pause: bool,
    pause_started: Option<f64>,
    paused_total: f64,
    last_obs_at: Option<f64>,
    last_face_at: Option<f64>,
    face_grace_until: f64,
    /// When the user last pressed a key here; drives the idle timeout.
    last_input_at: f64,
    /// When the dots were paused for lack of a face (`None` while not).
    no_face_since: Option<f64>,
    /// Gaze-direction feature indices passed to the fit (see [`controller_gaze_features`]).
    nonlinear: Option<Vec<usize>>,
    /// The animation timer's interval; `None` while it is stopped.
    timer_ms: Option<u64>,
    /// The inactivity check runs from `start` until the calibration closes.
    watchdog_active: bool,
    /// The primary surface should be raised and focused on the next frame.
    focus_requested: bool,
    actions: Vec<CalibrationAction>,
}

impl CalibrationWindow {
    pub fn new(options: CalibrationOptions) -> Self {
        CalibrationWindow {
            points_per_monitor: options.points_per_monitor,
            timing: CollectorOptions {
                settle_s: options.settle_s,
                collect_s: options.collect_s,
                min_samples: options.min_samples,
                max_retries: options.max_retries,
            },
            margin: options.margin,
            fit_in_thread: options.fit_in_thread,
            fit: None,
            fit_started: 0.0,
            state: CalibrationState::Idle,
            monitors: Vec::new(),
            labels: BTreeMap::new(),
            surfaces: Vec::new(),
            collector: None,
            plan: Vec::new(),
            model: None,
            report: None,
            data: None,
            error: String::new(),
            prev_target: None,
            last_target: None,
            retrying: false,
            manual_pause: false,
            auto_pause: false,
            pause_started: None,
            paused_total: 0.0,
            last_obs_at: None,
            last_face_at: None,
            face_grace_until: f64::NEG_INFINITY,
            last_input_at: 0.0,
            no_face_since: None,
            nonlinear: None,
            timer_ms: None,
            watchdog_active: false,
            focus_requested: false,
            actions: Vec::new(),
        }
    }

    // ============================================================== properties
    pub fn state(&self) -> CalibrationState {
        self.state
    }

    /// True from [`start`](Self::start) until the calibration is saved or cancelled.
    pub fn is_active(&self) -> bool {
        !matches!(
            self.state,
            CalibrationState::Idle | CalibrationState::Closed
        )
    }

    pub fn report(&self) -> Option<&CalibrationReport> {
        self.report.as_ref()
    }

    pub fn model(&self) -> Option<&GazeModel> {
        self.model.as_ref()
    }

    /// The saved calibration (after [`save`](Self::save)).
    pub fn data(&self) -> Option<&CalibrationData> {
        self.data.as_ref()
    }

    pub fn error(&self) -> &str {
        &self.error
    }

    pub fn paused(&self) -> bool {
        self.manual_pause || self.auto_pause
    }

    pub fn current_target(&self) -> Option<CalibrationTarget> {
        self.collector.as_ref().and_then(|c| c.current())
    }

    pub fn progress(&self) -> f64 {
        self.collector.as_ref().map_or(0.0, |c| c.progress())
    }

    pub fn plan(&self) -> &[CalibrationTarget] {
        &self.plan
    }

    pub fn monitors(&self) -> &[Monitor] {
        &self.monitors
    }

    /// The per-monitor windows (primary first).
    pub fn surfaces(&self) -> &[Surface] {
        &self.surfaces
    }

    /// The interval of the animation timer, `None` while it is stopped.
    pub fn timer_ms(&self) -> Option<u64> {
        self.timer_ms
    }

    /// Whether the inactivity check runs (it also does on screens without an animation timer).
    pub fn watchdog_active(&self) -> bool {
        self.watchdog_active
    }

    /// The hint currently shown under the dot (`""` when none).
    pub fn hint(&self, now: f64) -> String {
        self.hint_with_color(now).0
    }

    /// What the window did that the controller must hear about, oldest first.
    pub fn take_actions(&mut self) -> Vec<CalibrationAction> {
        std::mem::take(&mut self.actions)
    }

    // ============================================================== lifecycle
    /// Open the surfaces on every monitor and show the instructions.
    ///
    /// Returns `false` (and queues [`CalibrationAction::NotStarted`]) when there is no monitor.
    /// Calling it while a calibration is open just raises the windows.
    pub fn start(&mut self, inputs: &Inputs) -> bool {
        if self.is_active() {
            // Asked for again (tray, hotkey, second launch): the user is here.
            self.last_input_at = inputs.now;
            self.focus_primary();
            return true;
        }
        let monitors = inputs.monitors.clone();
        let Some(primary) = primary_of(&monitors).cloned() else {
            log::warn!("Calibration needs at least one monitor");
            self.state = CalibrationState::Closed;
            self.actions.push(CalibrationAction::NotStarted);
            return false;
        };
        self.labels = monitor_labels(&monitors);
        self.surfaces = std::iter::once(&primary)
            .chain(monitors.iter().filter(|m| m.index != primary.index))
            .cloned()
            .map(Surface::new)
            .collect();
        self.monitors = monitors;
        self.actions.push(CalibrationAction::Started);

        log::info!("Calibration opened on {} monitor(s)", self.monitors.len());
        self.last_input_at = inputs.now;
        self.set_state(CalibrationState::Intro);
        self.show_intro(inputs.now);
        self.focus_primary();
        self.timer_ms = Some(IDLE_TICK_MS);
        self.watchdog_active = true;
        true
    }

    /// Start (or restart) showing the dots.
    pub fn begin(&mut self, inputs: &Inputs) {
        if !self.is_active() {
            return;
        }
        // The controller reports a new layout up to a second after the system does (it
        // debounces hot-plugs), so a change may have slipped past the polling: dots on the old
        // geometry could never be saved.
        if self.layout_changed(inputs) {
            self.on_screens_changed();
            return;
        }
        let plan = match make_plan(&self.monitors, self.points_per_monitor, self.margin) {
            Ok(plan) => plan,
            Err(error) => {
                return self.show_error(&error.to_string(), "Not enough data", inputs.now);
            }
        };
        let mut collector = match CalibrationCollector::new(&plan, self.timing) {
            Ok(collector) => collector,
            Err(error) => {
                return self.show_error(&error.to_string(), "Not enough data", inputs.now);
            }
        };
        self.plan = plan;
        let now = inputs.now;
        self.last_input_at = now;
        self.no_face_since = None;
        self.prev_target = None;
        self.last_target = None;
        self.retrying = false;
        self.manual_pause = false;
        self.auto_pause = false;
        self.pause_started = None;
        self.paused_total = 0.0;
        self.face_grace_until = now + NO_FACE_PAUSE_S;
        self.model = None;
        self.report = None;
        self.error.clear();
        self.drop_fit();
        collector.start(now);
        self.collector = Some(collector);
        self.set_state(CalibrationState::Running);
        for surface in &mut self.surfaces {
            surface.cursor_hidden = true;
        }
        self.timer_ms = Some(TICK_MS);
        self.tick(inputs);
    }

    /// Discard the collected dots and run the sequence again.
    pub fn retry(&mut self, inputs: &Inputs) {
        log::info!("Calibration restarted");
        self.begin(inputs);
    }

    /// Pause or resume the dots (Space while running).
    pub fn toggle_pause(&mut self, now: f64) {
        if self.state != CalibrationState::Running {
            return;
        }
        self.last_input_at = now;
        self.manual_pause = !self.manual_pause;
        self.update_pause(now);
        self.render(now);
    }

    /// Hand the calibration to the controller (Enter on the result screen).
    pub fn save(&mut self, inputs: &Inputs) -> bool {
        if self.state != CalibrationState::Result {
            return false;
        }
        let (Some(model), Some(report), Some(collector)) = (
            self.model.clone(),
            self.report.clone(),
            self.collector.as_ref(),
        ) else {
            return false;
        };
        if self.layout_changed(inputs) {
            // Retrying would plan on the same stale layout: close, and let the user start over
            // on the new one.
            self.on_screens_changed();
            return false;
        }
        let (backend, feature_version) = &inputs.backend_info;
        if backend.is_empty() || feature_version.is_empty() {
            // Without them the calibration could never be matched to the backend later.
            self.show_error(
                "The vision backend is not ready. Please try again.",
                "Not enough data",
                inputs.now,
            );
            return false;
        }
        let mut data = CalibrationData::new(
            backend.clone(),
            feature_version.clone(),
            layout_signature(&self.monitors),
            self.monitors.clone(),
            collector.samples().to_vec(),
            Vec::new(),
            model,
        );
        data.report = report.to_dict();
        // The calibration only fits the camera it was recorded with (its position and field of
        // view shape every feature).
        data.camera = inputs.camera_device.trim().to_string();
        data.frame_size = collector.frame_size();
        log::info!("Calibration saved: {}", report.summary());
        self.data = Some(data.clone());
        self.finish(Some((Box::new(data), report.grade)));
        true
    }

    /// Close without saving (Esc, or a surface closed by the window manager).
    pub fn cancel(&mut self) {
        if !self.is_active() {
            return;
        }
        log::info!("Calibration cancelled");
        self.finish(None);
    }

    // ============================================================== input
    /// Keyboard control shared by every surface. Returns true if handled.
    pub fn handle_key(&mut self, inputs: &Inputs, key: Key) -> bool {
        // Any key proves someone is at the keyboard (restarts the idle timeout).
        self.last_input_at = inputs.now;
        if key == Key::Escape {
            self.cancel();
            return true;
        }
        match (self.state, key) {
            (CalibrationState::Intro, Key::Enter | Key::Space) => self.begin(inputs),
            (CalibrationState::Running, Key::Space) => self.toggle_pause(inputs.now),
            (CalibrationState::Running, Key::R) => self.retry(inputs),
            (CalibrationState::Result, Key::Enter) => {
                self.save(inputs);
            }
            (CalibrationState::Result, Key::R) => self.retry(inputs),
            (CalibrationState::Error, Key::Enter | Key::R | Key::Space) => self.retry(inputs),
            _ => return false,
        }
        true
    }

    /// A card button was clicked (`"save"`, `"retry"` or `"cancel"`).
    pub fn press_button(&mut self, inputs: &Inputs, name: &str) {
        match name {
            "save" => {
                self.save(inputs);
            }
            "retry" => self.retry(inputs),
            "cancel" => self.cancel(),
            _ => {}
        }
    }

    /// A surface was closed by the window manager (Alt+F4, …): treat as cancel.
    pub fn surface_closed(&mut self) {
        self.cancel();
    }

    /// Feed one observation (`inputs.observations` of each frame, in order).
    pub fn on_observation(&mut self, now: f64, obs: &Observation) {
        if !self.is_active() {
            return;
        }
        // Any observation proves the camera works; a motion-gate copy (skipped) repeats the
        // last analysed frame, whose face is still in an unchanged image.
        self.last_obs_at = Some(now);
        if obs.face_count > 0 {
            self.last_face_at = Some(now);
        }
        match self.state {
            CalibrationState::Running => {
                // The controller turns the motion gate off while calibrating, but a skipped
                // copy may still predate the current dot: never record one.
                if !obs.skipped
                    && !self.paused()
                    && let Some(collector) = self.collector.as_mut()
                {
                    collector.add(obs);
                }
            }
            CalibrationState::Intro => self.refresh_face_chip(now),
            _ => {}
        }
    }

    /// A screen was added, removed, resized or moved, or the primary changed.
    pub fn on_screens_changed(&mut self) {
        if !self.is_active() {
            return;
        }
        log::info!("Monitor layout changed; calibration cancelled");
        self.notify(
            "Calibration cancelled",
            "The monitor layout changed. Please calibrate again.",
        );
        self.cancel();
    }

    /// Cancel when the monitors differ from the ones being calibrated: their geometry or the
    /// primary changed.
    pub fn check_screens(&mut self, inputs: &Inputs) {
        let primary = |monitors: &[Monitor]| primary_of(monitors).map(|m| m.index);
        if self.is_active()
            && !inputs.monitors.is_empty()
            && (self.layout_changed(inputs) || primary(&inputs.monitors) != primary(&self.monitors))
        {
            self.on_screens_changed();
        }
    }

    // ============================================================== ticking
    /// One frame: the monitors are checked, the observations consumed and the sequence
    /// advanced to the current time.
    pub fn update(&mut self, inputs: &Inputs) {
        self.check_screens(inputs);
        for obs in &inputs.observations {
            self.on_observation(inputs.now, obs);
        }
        self.tick(inputs);
    }

    /// Advance the sequence to the current time and redraw.
    ///
    /// Called every frame (every [`TICK_MS`] while dots are shown); tests call it directly after
    /// advancing a fake clock.
    pub fn tick(&mut self, inputs: &Inputs) {
        if self.check_idle(inputs.now) {
            return;
        }
        match self.state {
            CalibrationState::Intro => self.refresh_face_chip(inputs.now),
            CalibrationState::Running => self.tick_running(inputs),
            CalibrationState::Fitting => self.tick_fitting(inputs),
            _ => {}
        }
    }

    /// Close an abandoned calibration. Returns `true` if it was closed.
    ///
    /// Runs every [`WATCHDOG_MS`] (and on every [`tick`](Self::tick)). A calibration waiting for
    /// a key for [`IDLE_TIMEOUT_S`], or with its dots paused for lack of a face for
    /// [`NO_FACE_TIMEOUT_S`], is cancelled so that walk-away detection and the normal camera
    /// rate resume.
    pub fn check_idle(&mut self, now: f64) -> bool {
        if !self.is_active() {
            return false;
        }
        let waiting = matches!(
            self.state,
            CalibrationState::Intro | CalibrationState::Result | CalibrationState::Error
        ) || (self.state == CalibrationState::Running && self.manual_pause);
        let reason = if waiting && now - self.last_input_at >= IDLE_TIMEOUT_S {
            format!("Nothing was pressed for {} minutes.", IDLE_TIMEOUT_S / 60.0)
        } else if self.state == CalibrationState::Running
            && self.auto_pause
            && self
                .no_face_since
                .is_some_and(|since| now - since >= NO_FACE_TIMEOUT_S)
        {
            format!("No face was seen for {} seconds.", NO_FACE_TIMEOUT_S)
        } else {
            return false;
        };
        log::info!(
            "Calibration closed after inactivity ({})",
            self.state.value()
        );
        self.notify(
            "Calibration closed",
            &format!("{reason} Choose “Calibrate now…” in the tray menu to start again."),
        );
        self.cancel();
        true
    }

    fn tick_running(&mut self, inputs: &Inputs) {
        let now = inputs.now;
        let face_missing = now >= self.face_grace_until
            && self
                .last_face_at
                .is_none_or(|seen| now - seen > NO_FACE_PAUSE_S);
        if face_missing != self.auto_pause {
            self.auto_pause = face_missing;
            self.no_face_since = face_missing.then_some(now);
            log::debug!(
                "Calibration {}",
                if face_missing {
                    "paused: no face"
                } else {
                    "resumed"
                }
            );
            self.update_pause(now);
        }
        let collector_time = self.collector_time(now);
        let Some(collector) = self.collector.as_mut() else {
            return;
        };
        let events = collector.update(collector_time);
        let current = collector.current();
        for event in events {
            match event {
                Event::Target => {
                    self.prev_target = self.last_target;
                    self.last_target = current;
                    self.retrying = false;
                }
                Event::Retry => self.retrying = true,
                Event::Finished => {
                    self.enter_fitting(inputs);
                    return;
                }
            }
        }
        self.render(now);
    }

    /// Wall time minus paused time: the collector's clock stands still while paused.
    fn collector_time(&self, now: f64) -> f64 {
        let mut paused = self.paused_total;
        if let Some(started) = self.pause_started {
            paused += now - started;
        }
        now - paused
    }

    fn update_pause(&mut self, now: f64) {
        match self.pause_started {
            None if self.paused() => self.pause_started = Some(now),
            Some(started) if !self.paused() => {
                self.paused_total += now - started;
                self.pause_started = None;
            }
            _ => {}
        }
    }

    fn hint_with_color(&self, now: f64) -> (String, Color32) {
        if self.state != CalibrationState::Running {
            return (String::new(), MUTED);
        }
        if self.manual_pause {
            return ("Paused — press Space to continue".into(), TEXT);
        }
        if self.auto_pause {
            if self.last_obs_at.is_none_or(|seen| now - seen > NO_CAMERA_S) {
                return ("Waiting for the camera…".into(), WARNING);
            }
            return ("Can't see your face — check the camera".into(), WARNING);
        }
        if self.retrying {
            return ("Keep looking at the dot…".into(), WARNING);
        }
        (String::new(), MUTED)
    }

    fn dot_state(&self) -> Option<DotState> {
        let collector = self.collector.as_ref()?;
        let target = collector.current()?;
        let t = collector.phase_progress();
        let (mut x, mut y, mut pop) = (target.x, target.y, 1.0);
        let (mut ring, mut arc) = (None, None);
        if collector.phase() == Phase::Settle {
            match self.prev_target {
                Some(prev) if prev.monitor_index == target.monitor_index => {
                    let g = ease_in_out(t / GLIDE_SHARE);
                    x = prev.x + (target.x - prev.x) * g;
                    y = prev.y + (target.y - prev.y) * g;
                }
                _ => pop = ease_out_back(t / POP_SHARE),
            }
            ring = Some(t);
        } else {
            arc = Some(t);
        }
        Some(DotState {
            x,
            y,
            target_x: target.x,
            target_y: target.y,
            pop,
            ring,
            arc,
            retry: self.retrying,
            paused: self.paused(),
        })
    }

    fn render(&mut self, now: f64) {
        let Some(collector) = self.collector.as_ref() else {
            return;
        };
        if self.state != CalibrationState::Running {
            return;
        }
        let dot = self.dot_state();
        let target = collector.current();
        let progress = collector.progress();
        let (hint, hint_color) = self.hint_with_color(now);
        let total = self.plan.len();
        let index = (collector.current_index() + 1).min(total);
        let text = format!("Dot {index} of {total}");
        let keys = "Space  pause    ·    R  restart    ·    Esc  cancel";
        for surface in &mut self.surfaces {
            surface.mode = SurfaceMode::Running;
            surface.active = target.is_some_and(|t| surface.monitor.index == t.monitor_index);
            surface.dot = dot;
            surface.progress = progress;
            surface.progress_text.clone_from(&text);
            surface.hint.clone_from(&hint);
            surface.hint_color = hint_color;
            surface.keys_text = keys.to_string();
        }
    }

    // ============================================================== fitting
    fn enter_fitting(&mut self, inputs: &Inputs) {
        self.set_state(CalibrationState::Fitting);
        self.fit_started = inputs.now;
        for (n, surface) in self.surfaces.iter_mut().enumerate() {
            let card = surface.show_card();
            if n == 0 {
                card.add_spinner();
                card.add_label(
                    CardLabel::new("Calculating your calibration…")
                        .size(16.0)
                        .bold(),
                );
                card.add_label(CardLabel::new("This takes a moment.").muted());
            } else {
                card.add_label(CardLabel::new("Almost done…").muted());
            }
        }
        let samples = self.samples();
        self.nonlinear = fit_features(controller_gaze_features(inputs), &samples);
        if self.fit_in_thread {
            self.fit = Some(start_fit(
                samples,
                self.monitors.clone(),
                self.nonlinear.clone(),
            ));
        }
        // Without a thread the fit runs on the next tick, so the "Calculating" screen is
        // painted first; with one, ticks animate the spinner and poll.
        self.timer_ms = Some(TICK_MS);
    }

    fn samples(&self) -> Vec<CalibrationSample> {
        self.collector
            .as_ref()
            .map_or_else(Vec::new, |c| c.samples().to_vec())
    }

    fn tick_fitting(&mut self, inputs: &Inputs) {
        let outcome = match &self.fit {
            None => {
                let samples = self.samples();
                evaluate(&samples, &self.monitors, None, self.nonlinear.as_deref())
                    .map_err(FitFailure::NotEnoughData)
            }
            Some(receiver) => match receiver.try_recv() {
                Ok(result) => result.map_err(FitFailure::NotEnoughData),
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => Err(FitFailure::Crashed(
                    "the fitting thread stopped unexpectedly".into(),
                )),
            },
        };
        self.fit = None;
        match outcome {
            Ok((model, report)) => self.fit_succeeded(model, report, inputs.now),
            Err(failure) => self.fit_failed(failure, inputs.now),
        }
    }

    /// How long the "calculating" spinner has been turning (its angle follows from it).
    pub fn fitting_seconds(&self, now: f64) -> f64 {
        now - self.fit_started
    }

    fn fit_succeeded(&mut self, model: GazeModel, report: CalibrationReport, now: f64) {
        self.timer_ms = None;
        self.model = Some(model);
        self.last_input_at = now; // the idle timeout counts from the result
        self.set_state(CalibrationState::Result);
        self.show_result(&report);
        self.report = Some(report);
    }

    fn fit_failed(&mut self, failure: FitFailure, now: f64) {
        match failure {
            FitFailure::NotEnoughData(error) => {
                // `evaluate` fails when there is not enough data.
                log::info!("Calibration could not be fitted: {error}");
                let skipped = self
                    .collector
                    .as_ref()
                    .map_or(0, |c| c.skipped_points().len());
                let message = if skipped > 0 {
                    format!(
                        "Your face was not clearly visible for {skipped} of {} dots.",
                        self.plan.len()
                    )
                } else {
                    "Not enough usable data was collected.".to_string()
                };
                self.show_error(&message, "Not enough data", now);
            }
            FitFailure::Crashed(error) => {
                // A numerical failure must not leave full-screen windows without a way out.
                log::error!("Calibration fitting failed: {error}");
                self.show_error(
                    &format!("Something went wrong while fitting the calibration: {error}"),
                    "Calibration failed",
                    now,
                );
            }
        }
    }

    /// Forget a running fit; its thread finishes on its own and the result is ignored.
    fn drop_fit(&mut self) {
        self.fit = None;
    }

    // ============================================================== screens
    fn show_intro(&mut self, now: f64) {
        let n_points = self.points_per_monitor * self.monitors.len();
        let seconds = 5.0_f64.max(
            (n_points as f64 * (self.timing.settle_s + self.timing.collect_s) / 5.0)
                .round_ties_even()
                * 5.0,
        );
        let screens = if self.monitors.len() == 1 {
            "screen".to_string()
        } else {
            format!("{} screens", self.monitors.len())
        };
        let whose = if self.monitors.len() == 1 {
            "one"
        } else {
            "your"
        };
        for surface in &mut self.surfaces {
            let card = surface.show_card();
            card.add_label(CardLabel::new("Let's calibrate").size(22.0).bold());
            card.add_label(
                CardLabel::new(
                    "Look at each dot as it appears. Turn your head naturally — don't hold it still.",
                )
                .size(13.0),
            );
            card.add_label(
                CardLabel::new(format!(
                    "{n_points} dots on {whose} {screens} · about {seconds} seconds"
                ))
                .muted(),
            );
            card.add_spacing(6.0);
            card.add_label(CardLabel::new("").named("face").size(10.5));
            card.add_spacing(4.0);
            card.add_keys(&[("Space", "start"), ("Esc", "cancel")]);
        }
        self.refresh_face_chip(now);
    }

    fn refresh_face_chip(&mut self, now: f64) {
        let (text, color) = if self.last_obs_at.is_none_or(|seen| now - seen > NO_CAMERA_S) {
            ("●  Waiting for the camera…", MUTED)
        } else if self.last_face_at.is_some_and(|seen| now - seen <= 1.0) {
            ("●  Face detected — ready", SUCCESS)
        } else {
            ("●  Looking for your face…", WARNING)
        };
        for surface in &mut self.surfaces {
            surface.card.set_label("face", text, color);
        }
    }

    /// Labels of the monitors the report has no dots for, left to right.
    fn uncovered_labels(&self, report: &CalibrationReport) -> Vec<String> {
        spatial_order(&self.monitors)
            .into_iter()
            .filter(|m| report.uncovered_monitors.contains(&m.index))
            .map(|m| self.screen_label(m.index))
            .collect()
    }

    fn screen_label(&self, index: i32) -> String {
        self.labels
            .get(&index)
            .cloned()
            .unwrap_or_else(|| format!("Screen {}", index + 1))
    }

    fn show_result(&mut self, report: &CalibrationReport) {
        let skipped = self
            .collector
            .as_ref()
            .map_or(0, |c| c.skipped_points().len());
        let multi = self.monitors.len() > 1;
        let uncovered = self.uncovered_labels(report);
        let order: BTreeMap<i32, usize> = spatial_order(&self.monitors)
            .into_iter()
            .enumerate()
            .map(|(n, m)| (m.index, n))
            .collect();
        let mut rows: Vec<(i32, f64)> = report
            .per_monitor_accuracy
            .iter()
            .map(|(&index, &value)| (index, value))
            .collect();
        // Left to right, like the dots and the "Screen n" labels.
        rows.sort_by_key(|&(index, _)| (order.get(&index).copied().unwrap_or(order.len()), index));
        let rows: Vec<(String, f64)> = rows
            .into_iter()
            .map(|(index, value)| (self.screen_label(index), value))
            .collect();

        for (n, surface) in self.surfaces.iter_mut().enumerate() {
            let card = surface.show_card();
            if n > 0 {
                card.add_label(
                    CardLabel::new("Your results are on the main screen.")
                        .muted()
                        .size(12.0),
                );
                continue;
            }
            card.add_badge(&report.grade, grade_badge_color(&report.grade));
            if multi {
                let pct = percent(report.monitor_accuracy);
                card.add_label(
                    CardLabel::new(format!("{pct}%"))
                        .named("headline")
                        .size(40.0)
                        .bold(),
                );
                card.add_label(CardLabel::new("of glances landed on the right screen").muted());
            } else {
                let error = report.mean_error_px;
                let value = if error.is_finite() {
                    format!("{error:.0} px")
                } else {
                    "—".to_string()
                };
                card.add_label(CardLabel::new(value).named("headline").size(40.0).bold());
                card.add_label(CardLabel::new("average distance from the dots").muted());
            }
            card.add_spacing(6.0);
            if multi && !report.per_monitor_accuracy.is_empty() {
                card.add_bars(rows.clone());
                card.add_spacing(4.0);
            }
            let mut details = format!("{} samples from {} dots", report.n_samples, report.n_points);
            if report.mean_error_px.is_finite() && multi {
                details = format!("Mean error {:.0} px · {details}", report.mean_error_px);
            }
            if skipped > 0 {
                details.push_str(&format!(" · {skipped} skipped"));
            }
            card.add_label(CardLabel::new(details).named("details").muted().size(10.0));
            if !uncovered.is_empty() {
                // Every dot of these screens was skipped: switching to them cannot work.
                let verb = if uncovered.len() == 1 { "was" } else { "were" };
                card.add_label(
                    CardLabel::new(format!(
                        "{} {verb} not calibrated — make sure the camera sees your face when \
                         you look there, then press R to try again.",
                        uncovered.join(" and ")
                    ))
                    .named("tip")
                    .size(10.5)
                    .color(WARNING),
                );
            } else if report.grade == "fair" || report.grade == "poor" {
                card.add_label(
                    CardLabel::new(
                        "Tip: sit where you usually sit, make sure your face is evenly lit and \
                         turn your head a little towards each dot. Press R to try again.",
                    )
                    .named("tip")
                    .size(10.5)
                    .color(WARNING),
                );
            } else {
                card.add_label(
                    CardLabel::new(
                        "Stored only on this computer. It keeps improving as you use the mouse.",
                    )
                    .named("tip")
                    .size(10.0)
                    .muted(),
                );
            }
            card.add_spacing(8.0);
            card.add_buttons(&[
                CardButton {
                    name: "save",
                    text: "Save   ⏎",
                    primary: true,
                },
                CardButton {
                    name: "retry",
                    text: "Retry   R",
                    primary: false,
                },
                CardButton {
                    name: "cancel",
                    text: "Cancel   Esc",
                    primary: false,
                },
            ]);
        }
        self.focus_primary();
    }

    fn show_error(&mut self, message: &str, badge: &str, now: f64) {
        self.timer_ms = None;
        self.error = message.to_string();
        self.last_input_at = now; // the idle timeout counts from here
        self.set_state(CalibrationState::Error);
        for (n, surface) in self.surfaces.iter_mut().enumerate() {
            let card = surface.show_card();
            if n > 0 {
                card.add_label(CardLabel::new("See the main screen.").muted().size(12.0));
                continue;
            }
            card.add_badge(badge, DANGER);
            card.add_label(CardLabel::new("Let's try that again").size(20.0).bold());
            card.add_label(CardLabel::new(message).named("message").size(12.0));
            card.add_label(
                CardLabel::new(
                    "Check that the camera sees your whole face, that the room is not too dark \
                     and that nothing else is using the camera.",
                )
                .muted()
                .size(10.5),
            );
            card.add_spacing(8.0);
            card.add_buttons(&[
                CardButton {
                    name: "retry",
                    text: "Try again   R",
                    primary: true,
                },
                CardButton {
                    name: "cancel",
                    text: "Cancel   Esc",
                    primary: false,
                },
            ]);
        }
        self.focus_primary();
    }

    // ============================================================== internals
    fn set_state(&mut self, state: CalibrationState) {
        if state != self.state {
            self.state = state;
            self.actions.push(CalibrationAction::StateChanged(state));
        }
    }

    fn finish(&mut self, saved: Option<(Box<CalibrationData>, String)>) {
        if self.state == CalibrationState::Closed {
            return;
        }
        self.set_state(CalibrationState::Closed);
        self.timer_ms = None;
        self.watchdog_active = false;
        self.drop_fit();
        self.surfaces.clear();
        self.actions.push(match saved {
            Some((data, grade)) => CalibrationAction::Finished { data, grade },
            None => CalibrationAction::Cancelled,
        });
    }

    /// Whether the controller's monitor layout differs from the one being calibrated.
    fn layout_changed(&self, inputs: &Inputs) -> bool {
        !inputs.monitors.is_empty()
            && layout_signature(&inputs.monitors) != layout_signature(&self.monitors)
    }

    /// Tell the user through the controller's notifications (shown by the tray).
    fn notify(&mut self, title: &str, message: &str) {
        self.actions.push(CalibrationAction::Notify {
            title: title.to_string(),
            message: message.to_string(),
        });
    }

    fn focus_primary(&mut self) {
        self.focus_requested = !self.surfaces.is_empty();
    }

    // ================================================================ drawing
    /// Draw a frame: advance the sequence, show a viewport per surface, and feed their keys,
    /// button clicks and closes back. Returns what the controller must hear about.
    pub fn show(&mut self, ctx: &egui::Context, inputs: &Inputs) -> Vec<CalibrationAction> {
        self.update(inputs);
        if self.is_active() {
            let mut input = SurfaceInput::default();
            for (n, surface) in self.surfaces.iter().enumerate() {
                let id = surface_viewport_id(&surface.monitor);
                if n == 0 && self.focus_requested {
                    ctx.send_viewport_cmd_to(id, egui::ViewportCommand::Focus);
                }
                let builder = surface_viewport_builder(&surface.monitor);
                ctx.show_viewport_immediate(id, builder.clone(), |ui, _class| {
                    util::keep_viewport_placed(ui.ctx(), &builder);
                    draw_surface(ui, self, surface, inputs.now, &mut input);
                });
            }
            self.focus_requested = false;
            ctx.request_repaint_after(Duration::from_millis(
                self.timer_ms.map_or(WATCHDOG_MS, |ms| ms.min(WATCHDOG_MS)),
            ));
            if input.closed {
                self.surface_closed();
            }
            for key in input.keys {
                self.handle_key(inputs, key);
            }
            for name in input.buttons {
                self.press_button(inputs, name);
            }
        }
        self.take_actions()
    }
}

fn surface_viewport_id(monitor: &Monitor) -> egui::ViewportId {
    egui::ViewportId::from_hash_of(("eye-tracker-calibration", monitor.index))
}

/// What the surfaces' viewports saw in a frame.
#[derive(Default)]
struct SurfaceInput {
    keys: Vec<Key>,
    buttons: Vec<&'static str>,
    closed: bool,
}

fn key_of(key: egui::Key) -> Key {
    match key {
        egui::Key::Escape => Key::Escape,
        egui::Key::Enter => Key::Enter,
        egui::Key::Space => Key::Space,
        egui::Key::R => Key::R,
        _ => Key::Other,
    }
}

/// Points of a Qt point size (the Python sizes are in points at 96 dpi).
fn pt(size: f32) -> f32 {
    size * 4.0 / 3.0
}

fn draw_surface(
    ui: &mut egui::Ui,
    window: &CalibrationWindow,
    surface: &Surface,
    now: f64,
    input: &mut SurfaceInput,
) {
    let ctx = ui.ctx().clone();
    ctx.input(|i| {
        input.closed |= i.viewport().close_requested();
        for event in &i.events {
            if let egui::Event::Key {
                key,
                pressed: true,
                repeat: false,
                ..
            } = event
            {
                input.keys.push(key_of(*key));
            }
        }
    });
    ctx.send_viewport_cmd(egui::ViewportCommand::CursorVisible(!surface.cursor_hidden));

    egui::CentralPanel::no_frame().show(ui, |ui| {
        let rect = ui.max_rect();
        let painter = ui.painter_at(rect);
        paint_background(&painter, rect);
        match surface.mode {
            SurfaceMode::Running => {
                match surface.dot {
                    Some(dot) if surface.active => paint_dot(&painter, rect, surface, &dot),
                    Some(dot) => paint_pointer(&painter, rect, surface, &dot),
                    None => {}
                }
                paint_footer(&painter, rect, surface);
            }
            SurfaceMode::Card => draw_card(&ctx, window, surface, rect, now, input),
        }
    });
}

/// The surface's translucent radial background.
fn paint_background(painter: &egui::Painter, rect: UiRect) {
    let mut mesh = egui::Mesh::default();
    mesh.colored_vertex(rect.center(), BG_INNER);
    for corner in [
        rect.left_top(),
        rect.right_top(),
        rect.right_bottom(),
        rect.left_bottom(),
    ] {
        mesh.colored_vertex(corner, BG_OUTER);
    }
    for n in 0..4u32 {
        mesh.add_triangle(0, 1 + n, 1 + (n + 1) % 4);
    }
    painter.add(egui::Shape::mesh(mesh));
}

fn local(rect: UiRect, surface: &Surface, x: f64, y: f64) -> Pos2 {
    rect.min
        + Vec2::new(
            (x - f64::from(surface.monitor.rect.x)) as f32,
            (y - f64::from(surface.monitor.rect.y)) as f32,
        )
}

/// Points of an arc of the circle around `center`, clockwise from `start` radians.
fn arc_points(center: Pos2, radius: f32, start: f32, sweep: f32) -> Vec<Pos2> {
    let steps = ((sweep.abs() / std::f32::consts::TAU) * 96.0)
        .ceil()
        .max(2.0) as usize;
    (0..=steps)
        .map(|n| {
            let angle = start + sweep * n as f32 / steps as f32;
            center + Vec2::angled(angle) * radius
        })
        .collect()
}

fn paint_dot(painter: &egui::Painter, rect: UiRect, surface: &Surface, dot: &DotState) {
    let c = local(rect, surface, dot.x, dot.y);
    let r = 11.0 * dot.pop.max(0.0) as f32;
    let base_a = if dot.paused {
        Color32::from_rgb(107, 114, 128)
    } else {
        ACCENT
    };
    let base_b = if dot.paused {
        Color32::from_rgb(156, 163, 175)
    } else {
        ACCENT_2
    };

    // Soft glow: stacked translucent discs.
    if r > 0.0 {
        let steps = 10;
        for n in 0..steps {
            let t = n as f32 / steps as f32;
            let color = if t < 0.45 { base_a } else { base_b };
            painter.circle_filled(c, r * 3.4 * (1.0 - t), with_alpha(color, 0.06));
        }
    }

    let ring_end = 11.0 + 8.0;
    if let Some(ring) = dot.ring {
        // Shrinking ring: tells the eyes where to go and when collection starts.
        let t = ease_out_cubic(ring) as f32;
        let radius = 64.0 + (ring_end - 64.0) * t;
        let color = if dot.retry { WARNING } else { TEXT };
        let stroke = Stroke::new(2.0, with_alpha(color, 0.25 + 0.6 * (ring / 0.15).min(1.0)));
        if dot.paused {
            let points = arc_points(c, radius, 0.0, std::f32::consts::TAU);
            painter.extend(egui::Shape::dashed_line(&points, stroke, 8.0, 5.0));
        } else {
            painter.circle_stroke(c, radius, stroke);
        }
    }
    if let Some(arc) = dot.arc {
        // Collection progress arc, clockwise from 12 o'clock.
        painter.circle_stroke(c, ring_end, Stroke::new(3.0, Color32::from_white_alpha(38)));
        if arc > 0.0 {
            let color = if dot.retry { WARNING } else { base_b };
            let sweep = std::f32::consts::TAU * clamp01(arc) as f32;
            painter.add(egui::Shape::line(
                arc_points(c, ring_end, -std::f32::consts::FRAC_PI_2, sweep),
                Stroke::new(3.0, color),
            ));
        }
    }

    if r > 0.0 {
        painter.circle_filled(c, r, base_a);
        // A crisp centre point gives the eyes something precise to fixate.
        let core = (2.6 * dot.pop as f32).max(1.5);
        painter.circle_filled(c, core, Color32::from_white_alpha(240));
    }
}

/// On the screens without the dot: an arrow towards the one that has it.
fn paint_pointer(painter: &egui::Painter, rect: UiRect, surface: &Surface, dot: &DotState) {
    let center = rect.center();
    let target = local(rect, surface, dot.target_x, dot.target_y);
    let delta = target - center;
    let dist = delta.length();
    if dist >= 1.0 {
        let unit = delta / dist;
        let reach = rect.width().min(rect.height()) * 0.28;
        let tip = center + unit * reach;
        let angle = unit.y.atan2(unit.x);
        let size = 26.0;
        let ends: Vec<Pos2> = [-1.0_f32, 1.0]
            .iter()
            .map(|sign| {
                let a = angle + std::f32::consts::PI + sign * 40.0_f32.to_radians();
                tip + Vec2::angled(a) * size
            })
            .collect();
        painter.add(egui::Shape::line(
            vec![ends[0], tip, ends[1]],
            Stroke::new(5.0, with_alpha(ACCENT_2, 0.85)),
        ));
    }
    painter.text(
        Pos2::new(rect.center().x, center.y),
        egui::Align2::CENTER_CENTER,
        "Follow the dot on the other screen",
        egui::FontId::proportional(pt(13.0)),
        MUTED,
    );
}

fn paint_footer(painter: &egui::Painter, rect: UiRect, surface: &Surface) {
    let width = (rect.width() * 0.36).min(520.0);
    let height = 4.0;
    let x0 = rect.left() + (rect.width() - width) / 2.0;
    let y0 = rect.bottom() - 72.0;
    let radius = height / 2.0;
    painter.rect_filled(
        UiRect::from_min_size(Pos2::new(x0, y0), Vec2::new(width, height)),
        radius,
        Color32::from_white_alpha(30),
    );
    let fill = width * clamp01(surface.progress) as f32;
    if fill > 0.0 {
        painter.rect_filled(
            UiRect::from_min_size(Pos2::new(x0, y0), Vec2::new(height.max(fill), height)),
            radius,
            ACCENT,
        );
    }
    let center_x = rect.center().x;
    if !surface.progress_text.is_empty() {
        painter.text(
            Pos2::new(center_x, y0 - 8.0),
            egui::Align2::CENTER_BOTTOM,
            &surface.progress_text,
            egui::FontId::proportional(pt(10.0)),
            MUTED,
        );
    }
    if !surface.keys_text.is_empty() {
        painter.text(
            Pos2::new(center_x, y0 + 14.0),
            egui::Align2::CENTER_TOP,
            &surface.keys_text,
            egui::FontId::proportional(pt(9.0)),
            MUTED,
        );
    }
    if !surface.hint.is_empty() && surface.active {
        painter.text(
            Pos2::new(center_x, y0 - 68.0),
            egui::Align2::CENTER_CENTER,
            &surface.hint,
            egui::FontId::proportional(pt(14.0)),
            surface.hint_color,
        );
    }
}

fn draw_card(
    ctx: &egui::Context,
    window: &CalibrationWindow,
    surface: &Surface,
    rect: UiRect,
    now: f64,
    input: &mut SurfaceInput,
) {
    let width = (rect.width() - 48.0).clamp(280.0, 600.0);
    egui::Area::new(egui::Id::new(("calibration-card", surface.monitor.index)))
        .anchor(
            egui::Align2::CENTER_CENTER,
            Vec2::new(0.0, -rect.height() * 0.04),
        )
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(Color32::from_white_alpha(11))
                .stroke(Stroke::new(1.0, Color32::from_white_alpha(23)))
                .corner_radius(22)
                .inner_margin(egui::Margin::symmetric(36, 32))
                .show(ui, |ui| {
                    ui.set_width(width - 72.0);
                    ui.spacing_mut().item_spacing.y = 12.0;
                    ui.vertical_centered(|ui| {
                        for row in &surface.card.rows {
                            draw_card_row(ui, window, row, now, input);
                        }
                    });
                });
        });
}

fn draw_card_row(
    ui: &mut egui::Ui,
    window: &CalibrationWindow,
    row: &CardRow,
    now: f64,
    input: &mut SurfaceInput,
) {
    match row {
        CardRow::Label(label) => {
            let color = label
                .color
                .unwrap_or(if label.muted { MUTED } else { TEXT });
            let mut text = egui::RichText::new(&label.text)
                .size(pt(label.size))
                .color(color);
            if label.bold {
                text = text.strong();
            }
            ui.add(egui::Label::new(text).wrap());
        }
        CardRow::Badge { text, color } => {
            egui::Frame::new()
                .fill(with_alpha(*color, 0.16))
                .stroke(Stroke::new(1.0, with_alpha(*color, 0.45)))
                .corner_radius(11)
                .inner_margin(egui::Margin::symmetric(12, 3))
                .show(ui, |ui| {
                    ui.label(
                        egui::RichText::new(text)
                            .size(pt(9.0))
                            .strong()
                            .color(*color),
                    );
                });
        }
        CardRow::Keys(items) => {
            let line = items
                .iter()
                .map(|(key, action)| format!("[{key}] {action}"))
                .collect::<Vec<_>>()
                .join("   ·   ");
            ui.label(egui::RichText::new(line).size(pt(10.5)).color(MUTED));
        }
        CardRow::Bars(bars) => {
            for (label, value) in bars {
                ui.horizontal(|ui| {
                    ui.add_sized(
                        [130.0, 18.0],
                        egui::Label::new(egui::RichText::new(label).color(MUTED)),
                    );
                    let (rect, _) =
                        ui.allocate_exact_size(Vec2::new(150.0, 6.0), egui::Sense::hover());
                    ui.painter()
                        .rect_filled(rect, 3.0, Color32::from_white_alpha(28));
                    if *value > 0.0 {
                        let fill = UiRect::from_min_size(
                            rect.min,
                            Vec2::new(rect.width() * clamp01(*value) as f32, rect.height()),
                        );
                        ui.painter().rect_filled(fill, 3.0, grade_color(*value));
                    }
                    ui.label(format!("{}%", percent(*value)));
                });
            }
        }
        CardRow::Buttons(buttons) => {
            ui.horizontal(|ui| {
                for button in buttons {
                    let mut widget = egui::Button::new(
                        egui::RichText::new(button.text)
                            .size(pt(11.0))
                            .color(if button.primary { Color32::WHITE } else { TEXT }),
                    )
                    .corner_radius(10);
                    widget = widget.fill(if button.primary {
                        ACCENT
                    } else {
                        Color32::from_white_alpha(20)
                    });
                    if ui.add(widget).clicked() {
                        input.buttons.push(button.name);
                    }
                }
            });
        }
        CardRow::Spinner => {
            let size = 34.0;
            let (rect, _) = ui.allocate_exact_size(Vec2::splat(size), egui::Sense::hover());
            let center = rect.center();
            let radius = size / 2.0 - 3.0;
            let angle = (window.fitting_seconds(now) / 0.9 * 360.0).rem_euclid(360.0) as f32;
            ui.painter().circle_stroke(
                center,
                radius,
                Stroke::new(3.0, Color32::from_white_alpha(36)),
            );
            ui.painter().add(egui::Shape::line(
                arc_points(center, radius, angle.to_radians(), 110.0_f32.to_radians()),
                Stroke::new(3.0, ACCENT_2),
            ));
        }
        CardRow::Spacing(points) => {
            ui.add_space(*points);
        }
    }
}

#[cfg(test)]
#[path = "calibration_window_tests.rs"]
mod tests;
