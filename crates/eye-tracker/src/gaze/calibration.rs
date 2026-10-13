//! Calibration: where to show targets, how to collect samples, how good the result is.
//!
//! The flow driven by the calibration window is:
//!
//! ```text
//! plan = make_plan(monitors, 9, 0.1)                 // dots to show, in order
//! collector = CalibrationCollector::new(plan, options)
//! collector.start(now)
//! ...every UI tick:   events = collector.update(now)
//! ...every frame:     collector.add(observation)
//! (model, report) = evaluate(collector.samples(), monitors, None, gaze_indices)
//! ```
//!
//! `gaze_indices` are the backend's gaze-direction features
//! (`gaze_feature_indices(backend.feature_names, backend.gaze_features)`).
//! Everything here is plain Rust driven by an injected clock, so the whole procedure
//! is unit-testable without a camera or a screen.

use std::collections::{BTreeMap, BTreeSet};

use nalgebra::DMatrix;
use serde_json::{Map, Value, json};
use thiserror::Error;

use super::model::{
    DEFAULT_ALPHAS, FitOptions, GazeModel, ModelError, SUPPORTED_DEGREES, lopo_predictions,
    select_model,
};
use crate::types::{Monitor, Observation, nearest_monitor, virtual_bounds};

/// What Python raised as a `ValueError`.
#[derive(Debug, Error, PartialEq)]
#[error("{0}")]
pub struct CalibrationError(pub String);

impl From<ModelError> for CalibrationError {
    fn from(error: ModelError) -> Self {
        CalibrationError(error.to_string())
    }
}

fn invalid<T>(message: impl Into<String>) -> Result<T, CalibrationError> {
    Err(CalibrationError(message.into()))
}

/// Collector phases (`PHASE_IDLE` … `PHASE_DONE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Idle,
    Settle,
    Collect,
    Done,
}

impl Phase {
    /// The Python phase string.
    pub fn value(self) -> &'static str {
        match self {
            Phase::Idle => "idle",
            Phase::Settle => "settle",
            Phase::Collect => "collect",
            Phase::Done => "done",
        }
    }
}

/// Events returned by [`CalibrationCollector::update`] (`EVENT_TARGET` …).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    Target,
    Retry,
    Finished,
}

impl Event {
    /// The Python event string.
    pub fn value(self) -> &'static str {
        match self {
            Event::Target => "target",
            Event::Retry => "retry",
            Event::Finished => "finished",
        }
    }
}

/// Minimum monitor accuracy for each grade, best first.
pub const GRADE_THRESHOLDS: [(&str, f64); 3] =
    [("excellent", 0.97), ("good", 0.90), ("fair", 0.75)];

pub const MIN_SAMPLES: usize = 10;
pub const MIN_POINTS: usize = 3;

/// Percentile of the absolute held-out error per axis that is reported as a
/// monitor's gaze error ([`CalibrationReport::per_monitor_error_px`]). The 75th
/// covers most looks without being dominated by the few dots the face was half
/// lost on.
pub const AXIS_ERROR_PERCENTILE: f64 = 75.0;

/// The defaults of [`make_plan`].
pub const DEFAULT_POINTS_PER_MONITOR: usize = 9;
pub const DEFAULT_MARGIN: f64 = 0.1;

// Timer ticks are sums of floats (11.8 + 0.8 = 12.599999…); without a little
// slack a phase could end one tick late.
const TIME_EPS: f64 = 1e-6;

// ------------------------------------------------------------------------ plan
/// One dot of the calibration sequence.
///
/// `nx`/`ny` are relative to the monitor (0..1), `x`/`y` are global
/// coordinates of the same point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibrationTarget {
    pub point_id: i64,
    pub monitor_index: i32,
    pub nx: f64,
    pub ny: f64,
    pub x: f64,
    pub y: f64,
}

/// Calibration targets for every monitor, in the order they should be shown.
///
/// `points_per_monitor`: 1 (centre), 5 (centre + 4 corners) or a square number
/// `k*k` (a `k x k` grid spanning `margin .. 1 - margin`). Monitors are
/// visited left to right (then top to bottom). Within a monitor the grid is
/// walked as a serpentine; the orientation of that walk is chosen so that it
/// starts at the corner nearest to where the previous monitor ended, which keeps
/// eye and head travel short. `point_id` counts up from 0.
pub fn make_plan(
    monitors: &[Monitor],
    points_per_monitor: usize,
    margin: f64,
) -> Result<Vec<CalibrationTarget>, CalibrationError> {
    if !(0.0..0.5).contains(&margin) {
        return invalid(format!("margin must be in [0, 0.5), got {margin:?}"));
    }
    let base = monitor_path(points_per_monitor, margin)?;
    let variants = [
        base.clone(),
        base.iter().map(|&(nx, ny)| (1.0 - nx, ny)).collect(),
        base.iter().map(|&(nx, ny)| (nx, 1.0 - ny)).collect(),
        base.iter().map(|&(nx, ny)| (1.0 - nx, 1.0 - ny)).collect(),
    ];

    let mut sorted: Vec<&Monitor> = monitors.iter().collect();
    sorted.sort_by_key(|m| (m.rect.x, m.rect.y, m.index));
    let mut plan: Vec<CalibrationTarget> = Vec::new();
    let mut last: Option<(f64, f64)> = None;
    for monitor in sorted {
        let rect = monitor.rect;
        let mut path = &variants[0];
        if let Some((last_x, last_y)) = last {
            let distance = |v: &Vec<(f64, f64)>| {
                let (x, y) = rect.denormalize(v[0].0, v[0].1);
                (x - last_x).hypot(y - last_y)
            };
            // The first of equally good variants wins, so the canonical top-left
            // start wins ties.
            let mut best = 0;
            for i in 1..variants.len() {
                if distance(&variants[i]) < distance(&variants[best]) {
                    best = i;
                }
            }
            path = &variants[best];
        }
        for &(nx, ny) in path {
            let (x, y) = rect.denormalize(nx, ny);
            plan.push(CalibrationTarget {
                point_id: plan.len() as i64,
                monitor_index: monitor.index,
                nx,
                ny,
                x,
                y,
            });
        }
        last = plan.last().map(|t| (t.x, t.y));
    }
    Ok(plan)
}

fn monitor_path(points: usize, margin: f64) -> Result<Vec<(f64, f64)>, CalibrationError> {
    let (lo, hi) = (margin, 1.0 - margin);
    if points == 1 {
        return Ok(vec![(0.5, 0.5)]);
    }
    if points == 5 {
        // Centre first (easiest to find), then the corners as a loop.
        return Ok(vec![(0.5, 0.5), (lo, lo), (hi, lo), (hi, hi), (lo, hi)]);
    }
    let k = points.isqrt();
    if k < 2 || k * k != points {
        return invalid(format!(
            "points_per_monitor must be 1, 5 or a square number >= 4, got {points}"
        ));
    }
    // numpy.linspace(lo, hi, k): the last value is exactly hi.
    let step = (hi - lo) / (k - 1) as f64;
    let coords: Vec<f64> = (0..k)
        .map(|i| if i == k - 1 { hi } else { i as f64 * step + lo })
        .collect();
    let mut path = Vec::with_capacity(points);
    for (row, &ny) in coords.iter().enumerate() {
        if row % 2 == 0 {
            path.extend(coords.iter().map(|&nx| (nx, ny)));
        } else {
            path.extend(coords.iter().rev().map(|&nx| (nx, ny)));
        }
    }
    Ok(path)
}

// --------------------------------------------------------------------- samples
/// A feature vector labelled with the global point the user was looking at.
///
/// `point_id` groups samples of one calibration dot (implicit samples learned
/// from mouse use carry negative ids). `weight` scales the sample's influence
/// on the fit.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationSample {
    pub features: Vec<f64>,
    pub x: f64,
    pub y: f64,
    pub monitor_index: i32,
    pub point_id: i64,
    pub weight: f64,
}

impl CalibrationSample {
    /// A sample with the default weight of 1.
    pub fn new(features: Vec<f64>, x: f64, y: f64, monitor_index: i32, point_id: i64) -> Self {
        Self::weighted(features, x, y, monitor_index, point_id, 1.0)
    }

    pub fn weighted(
        features: Vec<f64>,
        x: f64,
        y: f64,
        monitor_index: i32,
        point_id: i64,
        weight: f64,
    ) -> Self {
        CalibrationSample {
            features,
            x,
            y,
            monitor_index,
            point_id,
            weight,
        }
    }
}

/// `X (n, d)`, `Y (n, 2)` and the weights `(n,)` of [`samples_to_arrays`].
pub type SampleArrays = (DMatrix<f64>, DMatrix<f64>, Vec<f64>);

/// Stack samples into `X (n, d)`, targets `Y (n, 2)` and `weights (n,)`.
pub fn samples_to_arrays(samples: &[CalibrationSample]) -> Result<SampleArrays, CalibrationError> {
    let Some(first) = samples.first() else {
        return invalid("no samples");
    };
    let d = first.features.len();
    if samples.iter().any(|s| s.features.len() != d) {
        return invalid("all samples must have the same number of features");
    }
    let x = DMatrix::from_fn(samples.len(), d, |r, c| samples[r].features[c]);
    let y = DMatrix::from_fn(samples.len(), 2, |r, c| {
        if c == 0 { samples[r].x } else { samples[r].y }
    });
    let w = samples.iter().map(|s| s.weight).collect();
    Ok((x, y, w))
}

// ------------------------------------------------------------------- collector
/// The settings of a [`CalibrationCollector`] (its Python keyword arguments).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CollectorOptions {
    pub settle_s: f64,
    pub collect_s: f64,
    pub min_samples: usize,
    pub max_retries: usize,
}

impl Default for CollectorOptions {
    fn default() -> Self {
        CollectorOptions {
            settle_s: 0.8,
            collect_s: 1.0,
            min_samples: 5,
            max_retries: 1,
        }
    }
}

/// Time-based state machine that shows each target and gathers its samples.
///
/// Per target: `settle` (the eyes travel to the new dot; nothing is recorded)
/// then `collect` (usable observations are recorded). A target that gathered
/// fewer than `min_samples` when its collect window ends gets up to
/// `max_retries` extra collect windows (samples so far are kept) and is then
/// skipped, dropping its samples.
///
/// Every phase starts at the `now` of the [`update`](Self::update) call that entered it,
/// so a stalled UI thread never eats into the next target's time. Progress
/// properties describe the state as of the last [`start`](Self::start)/[`update`](Self::update).
#[derive(Debug, Clone)]
pub struct CalibrationCollector {
    plan: Vec<CalibrationTarget>,
    pub settle_s: f64,
    pub collect_s: f64,
    pub min_samples: usize,
    pub max_retries: usize,
    phase: Phase,
    index: usize,
    phase_start: f64,
    now: f64,
    retries: usize,
    pending: Vec<CalibrationSample>,
    samples: Vec<CalibrationSample>,
    skipped: Vec<i64>,
    events: Vec<Event>,
    n_features: Option<usize>,
    /// Counts per camera frame size, in order of first sight.
    frame_sizes: Vec<((u32, u32), usize)>,
}

impl CalibrationCollector {
    pub fn new(
        plan: &[CalibrationTarget],
        options: CollectorOptions,
    ) -> Result<Self, CalibrationError> {
        if options.settle_s < 0.0 || options.collect_s <= 0.0 {
            return invalid("settle_s must be >= 0 and collect_s > 0");
        }
        if options.min_samples < 1 {
            return invalid("min_samples must be >= 1 and max_retries >= 0");
        }
        Ok(CalibrationCollector {
            plan: plan.to_vec(),
            settle_s: options.settle_s,
            collect_s: options.collect_s,
            min_samples: options.min_samples,
            max_retries: options.max_retries,
            phase: Phase::Idle,
            index: 0,
            phase_start: 0.0,
            now: 0.0,
            retries: 0,
            pending: Vec::new(),
            samples: Vec::new(),
            skipped: Vec::new(),
            events: Vec::new(),
            n_features: None,
            frame_sizes: Vec::new(),
        })
    }

    fn reset(&mut self) {
        self.phase = Phase::Idle;
        self.index = 0;
        self.phase_start = 0.0;
        self.now = 0.0;
        self.retries = 0;
        self.pending.clear();
        self.samples.clear();
        self.skipped.clear();
        self.events.clear();
        self.n_features = None;
        self.frame_sizes.clear();
    }

    // ------------------------------------------------------------- control
    /// (Re)start from the first target. The next [`update`](Self::update) reports
    /// [`Event::Target`] (or [`Event::Finished`] for an empty plan).
    pub fn start(&mut self, now: f64) {
        self.reset();
        self.now = now;
        if self.plan.is_empty() {
            self.phase = Phase::Done;
            self.events.push(Event::Finished);
            return;
        }
        self.enter(Phase::Settle, now);
        self.events.push(Event::Target);
    }

    /// Advance the timers; returns the events since the last call, in order:
    /// [`Event::Target`] (a new target is current), [`Event::Retry`] (the current
    /// target gets another collect window) and [`Event::Finished`].
    pub fn update(&mut self, now: f64) -> Vec<Event> {
        if matches!(self.phase, Phase::Settle | Phase::Collect) {
            self.now = self.now.max(now);
            // Loop so zero-length phases (settle_s == 0) pass in a single call.
            while self.step(self.now) {}
        }
        std::mem::take(&mut self.events)
    }

    /// Offer an observation; returns true if it was recorded for the current target.
    ///
    /// Only usable observations that were really analysed are accepted: motion-gate
    /// copies (`skipped`) may repeat a frame from before the target appeared.
    pub fn add(&mut self, obs: &Observation) -> bool {
        let Some(target) = self.current() else {
            return false;
        };
        if self.phase != Phase::Collect || !obs.usable() || obs.skipped {
            return false;
        }
        let Some(features) = &obs.features else {
            return false;
        };
        if features.is_empty() || !features.iter().all(|v| v.is_finite()) {
            return false;
        }
        match self.n_features {
            None => self.n_features = Some(features.len()),
            Some(n) if n != features.len() => {
                log::warn!(
                    "Ignoring observation with {} features (expected {n})",
                    features.len()
                );
                return false;
            }
            Some(_) => {}
        }
        self.pending.push(CalibrationSample::new(
            features.clone(),
            target.x,
            target.y,
            target.monitor_index,
            target.point_id,
        ));
        let (width, height) = obs.frame_size;
        if width > 0 && height > 0 {
            match self
                .frame_sizes
                .iter_mut()
                .find(|(s, _)| *s == (width, height))
            {
                Some((_, count)) => *count += 1,
                None => self.frame_sizes.push(((width, height), 1)),
            }
        }
        true
    }

    // ------------------------------------------------------------- properties
    pub fn plan(&self) -> &[CalibrationTarget] {
        &self.plan
    }

    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// Target being shown, or `None` before start and after the last target.
    pub fn current(&self) -> Option<CalibrationTarget> {
        if matches!(self.phase, Phase::Settle | Phase::Collect) {
            self.plan.get(self.index).copied()
        } else {
            None
        }
    }

    /// Position of the current target in the plan.
    pub fn current_index(&self) -> usize {
        self.index
    }

    /// 0..1 through the current phase (a retry restarts the collect phase).
    pub fn phase_progress(&self) -> f64 {
        let elapsed = self.now - self.phase_start;
        match self.phase {
            Phase::Settle => {
                if self.settle_s == 0.0 {
                    1.0
                } else {
                    clamp01(elapsed / self.settle_s)
                }
            }
            Phase::Collect => clamp01(elapsed / self.collect_s),
            Phase::Done => 1.0,
            Phase::Idle => 0.0,
        }
    }

    /// 0..1 through the current target, settle and collect time combined.
    pub fn point_progress(&self) -> f64 {
        let total = self.settle_s + self.collect_s;
        match self.phase {
            Phase::Settle => self.phase_progress() * self.settle_s / total,
            Phase::Collect => (self.settle_s + self.phase_progress() * self.collect_s) / total,
            Phase::Done => 1.0,
            Phase::Idle => 0.0,
        }
    }

    /// 0..1 through the whole plan.
    pub fn progress(&self) -> f64 {
        if self.phase == Phase::Done {
            return 1.0;
        }
        if self.phase == Phase::Idle || self.plan.is_empty() {
            return 0.0;
        }
        (self.index as f64 + self.point_progress()) / self.plan.len() as f64
    }

    /// Samples recorded so far for the current target.
    pub fn current_sample_count(&self) -> usize {
        self.pending.len()
    }

    /// Samples of all completed (not skipped) targets.
    pub fn samples(&self) -> &[CalibrationSample] {
        &self.samples
    }

    /// Point ids that were skipped for lack of usable observations.
    pub fn skipped_points(&self) -> &[i64] {
        &self.skipped
    }

    /// Camera frame size `(w, h)` of the recorded observations, `(0, 0)` if unknown.
    ///
    /// The most common one if the camera changed its size mid-calibration. Store
    /// it in `CalibrationData::frame_size` so a later change of the camera's
    /// aspect ratio is noticed.
    pub fn frame_size(&self) -> (u32, u32) {
        let mut best: Option<&((u32, u32), usize)> = None;
        for entry in &self.frame_sizes {
            // Counter.most_common(1): the first of the most common sizes.
            if best.is_none_or(|b| entry.1 > b.1) {
                best = Some(entry);
            }
        }
        best.map_or((0, 0), |b| b.0)
    }

    // ------------------------------------------------------------- internals
    fn enter(&mut self, phase: Phase, now: f64) {
        self.phase = phase;
        self.phase_start = now;
    }

    /// Perform at most one transition; true if one happened.
    fn step(&mut self, now: f64) -> bool {
        let elapsed = now - self.phase_start + TIME_EPS;
        if self.phase == Phase::Settle {
            if elapsed < self.settle_s {
                return false;
            }
            self.enter(Phase::Collect, now);
            return true;
        }
        if self.phase != Phase::Collect || elapsed < self.collect_s {
            return false;
        }

        let target = self.plan[self.index];
        if self.pending.len() >= self.min_samples {
            self.samples.append(&mut self.pending);
        } else if self.retries < self.max_retries {
            self.retries += 1;
            self.enter(Phase::Collect, now);
            self.events.push(Event::Retry);
            log::debug!(
                "Calibration point {}: {}/{} samples, retrying",
                target.point_id,
                self.pending.len(),
                self.min_samples
            );
            return true;
        } else {
            self.skipped.push(target.point_id);
            log::info!(
                "Calibration point {} skipped ({} usable samples)",
                target.point_id,
                self.pending.len()
            );
        }

        self.pending.clear();
        self.retries = 0;
        self.index += 1;
        if self.index >= self.plan.len() {
            self.enter(Phase::Done, now);
            self.events.push(Event::Finished);
            return false;
        }
        self.enter(Phase::Settle, now);
        self.events.push(Event::Target);
        true
    }
}

// ------------------------------------------------------------------ evaluation
/// Quality of a calibration, estimated by leave-one-point-out cross-validation.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationReport {
    /// Fraction of samples whose held-out prediction is nearest to the right monitor.
    pub monitor_accuracy: f64,
    pub mean_error_px: f64,
    pub median_error_px: f64,
    pub per_monitor_accuracy: BTreeMap<i32, f64>,
    pub n_samples: usize,
    pub n_points: usize,
    pub alpha: f64,
    pub grade: String,
    /// Polynomial degree chosen for the model (see `gaze::model`).
    pub degree: usize,
    /// Monitors without a single surviving calibration dot (usually the face was
    /// not found while looking at them). They appear in `per_monitor_accuracy`
    /// with 0.0 and cap the grade at "fair".
    pub uncovered_monitors: Vec<i32>,
    /// Gaze error per monitor and axis, `{monitor: (x_px, y_px)}`: the
    /// [`AXIS_ERROR_PERCENTILE`]th percentile of the absolute held-out error
    /// of that monitor's samples. Split panes are only told apart where they are
    /// large compared with it. Empty in reports saved before it existed (see
    /// `gaze::store::axis_error`) and for uncovered monitors.
    pub per_monitor_error_px: BTreeMap<i32, (f64, f64)>,
}

impl CalibrationReport {
    /// A report with the defaults of the optional fields (`degree` 2, nothing uncovered,
    /// no per-axis errors).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        monitor_accuracy: f64,
        mean_error_px: f64,
        median_error_px: f64,
        per_monitor_accuracy: BTreeMap<i32, f64>,
        n_samples: usize,
        n_points: usize,
        alpha: f64,
        grade: &str,
    ) -> Self {
        CalibrationReport {
            monitor_accuracy,
            mean_error_px,
            median_error_px,
            per_monitor_accuracy,
            n_samples,
            n_points,
            alpha,
            grade: grade.to_string(),
            degree: 2,
            uncovered_monitors: Vec::new(),
            per_monitor_error_px: BTreeMap::new(),
        }
    }

    /// One human-readable line, e.g. `"Excellent — 99% monitor accuracy, 180 samples"`.
    pub fn summary(&self) -> String {
        // Floor, so "100%" is only ever shown for a perfect result.
        let pct = (self.monitor_accuracy * 100.0 + 1e-9).floor() as i64;
        let mut grade = self.grade.to_lowercase();
        if let Some(first) = grade.get_mut(..1) {
            first.make_ascii_uppercase();
        }
        let mut text = format!(
            "{grade} — {pct}% monitor accuracy, {} samples",
            self.n_samples
        );
        if self.mean_error_px.is_finite() {
            text += &format!(", mean error {:.0} px", self.mean_error_px);
        }
        if !self.uncovered_monitors.is_empty() {
            let n = self.uncovered_monitors.len();
            text += &format!(
                "; {n} screen{} not calibrated",
                if n > 1 { "s" } else { "" }
            );
        }
        text
    }

    /// JSON-friendly dict (monitor keys become strings, non-finite numbers `null`).
    pub fn to_dict(&self) -> Map<String, Value> {
        let accuracy: Map<String, Value> = self
            .per_monitor_accuracy
            .iter()
            .map(|(k, v)| (k.to_string(), json!(v)))
            .collect();
        let errors: Map<String, Value> = self
            .per_monitor_error_px
            .iter()
            .map(|(k, &(x, y))| (k.to_string(), json!([x, y])))
            .collect();
        let mut out = Map::new();
        out.insert("monitor_accuracy".into(), json!(self.monitor_accuracy));
        out.insert("mean_error_px".into(), json!(self.mean_error_px));
        out.insert("median_error_px".into(), json!(self.median_error_px));
        out.insert("per_monitor_accuracy".into(), Value::Object(accuracy));
        out.insert("n_samples".into(), json!(self.n_samples));
        out.insert("n_points".into(), json!(self.n_points));
        out.insert("alpha".into(), json!(self.alpha));
        out.insert("grade".into(), json!(self.grade));
        out.insert("degree".into(), json!(self.degree));
        out.insert("uncovered_monitors".into(), json!(self.uncovered_monitors));
        out.insert("per_monitor_error_px".into(), Value::Object(errors));
        out
    }

    /// Inverse of [`to_dict`](Self::to_dict). Fails with `invalid calibration report: …` on
    /// malformed input.
    pub fn from_dict(data: &Map<String, Value>) -> Result<CalibrationReport, CalibrationError> {
        Self::parse(data)
            .map_err(|exc| CalibrationError(format!("invalid calibration report: {exc}")))
    }

    fn parse(data: &Map<String, Value>) -> Result<CalibrationReport, String> {
        let required = |key: &str| data.get(key).ok_or_else(|| format!("missing key '{key}'"));
        let mut per_monitor_accuracy = BTreeMap::new();
        match data.get("per_monitor_accuracy") {
            None | Some(Value::Null) => {}
            Some(Value::Object(map)) => {
                for (k, v) in map {
                    per_monitor_accuracy.insert(parse_key(k)?, json_float(v)?);
                }
            }
            Some(other) => return Err(format!("cannot convert {other} to a dict")),
        }
        let degree = match data.get("degree") {
            None => 2,
            Some(v) => usize::try_from(json_int(v)?).map_err(|_| "degree out of range")?,
        };
        let mut uncovered_monitors = Vec::new();
        match data.get("uncovered_monitors") {
            None | Some(Value::Null) => {}
            Some(Value::Array(list)) => {
                for v in list {
                    uncovered_monitors.push(json_i32(v)?);
                }
            }
            Some(other) => return Err(format!("uncovered_monitors must be a list, not {other}")),
        }
        Ok(CalibrationReport {
            monitor_accuracy: float_or_nan(required("monitor_accuracy")?)?,
            mean_error_px: float_or_nan(required("mean_error_px")?)?,
            median_error_px: float_or_nan(required("median_error_px")?)?,
            per_monitor_accuracy,
            n_samples: usize::try_from(json_int(required("n_samples")?)?)
                .map_err(|_| "n_samples out of range")?,
            n_points: usize::try_from(json_int(required("n_points")?)?)
                .map_err(|_| "n_points out of range")?,
            alpha: float_or_nan(required("alpha")?)?,
            grade: match required("grade")? {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            },
            degree,
            uncovered_monitors,
            per_monitor_error_px: axis_errors_from_dict(data.get("per_monitor_error_px"))
                .map_err(|e| e.0)?,
        })
    }
}

/// `per_monitor_error_px` as stored (`{"0": [x, y]}`); absent or null gives `{}`.
///
/// Entries that are not two finite, non-negative numbers are dropped: a damaged
/// entry only means "unknown", which callers estimate otherwise. A value that is
/// not an object at all is an error, like the rest of a malformed report.
pub fn axis_errors_from_dict(
    value: Option<&Value>,
) -> Result<BTreeMap<i32, (f64, f64)>, CalibrationError> {
    let map = match value {
        None | Some(Value::Null) => return Ok(BTreeMap::new()),
        Some(Value::Object(map)) => map,
        Some(_) => return invalid("per_monitor_error_px must be an object"),
    };
    let mut out = BTreeMap::new();
    for (key, pair) in map {
        let Some([ex, ey]) = pair.as_array().map(Vec::as_slice) else {
            continue;
        };
        let (Ok(ex), Ok(ey), Ok(index)) = (json_float(ex), json_float(ey), parse_key(key)) else {
            continue;
        };
        if [ex, ey].iter().all(|v| v.is_finite() && *v >= 0.0) {
            out.insert(index, (ex, ey));
        }
    }
    Ok(out)
}

/// `{monitor: (x_px, y_px)}`: the [`AXIS_ERROR_PERCENTILE`]th percentile of
/// the absolute error on each axis over each monitor's samples.
///
/// `predictions` and `targets` are `(n, 2)` matrices (held-out predictions and
/// the points looked at); `monitor_indices` gives each row's monitor. Rows with
/// a non-finite prediction are ignored; a monitor without a finite row is left out.
pub fn per_axis_errors(
    predictions: &DMatrix<f64>,
    targets: &DMatrix<f64>,
    monitor_indices: &[i32],
) -> Result<BTreeMap<i32, (f64, f64)>, CalibrationError> {
    let n = predictions.nrows();
    if predictions.ncols() != 2
        || targets.ncols() != 2
        || n != targets.nrows()
        || n != monitor_indices.len()
    {
        return invalid("predictions, targets and monitor indices differ in length");
    }
    let mut columns: BTreeMap<i32, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    for (row, &owner) in monitor_indices.iter().enumerate() {
        let ex = (predictions[(row, 0)] - targets[(row, 0)]).abs();
        let ey = (predictions[(row, 1)] - targets[(row, 1)]).abs();
        if ex.is_finite() && ey.is_finite() {
            let (xs, ys) = columns.entry(owner).or_default();
            xs.push(ex);
            ys.push(ey);
        }
    }
    Ok(columns
        .into_iter()
        .map(|(idx, (xs, ys))| {
            (
                idx,
                (
                    percentile(xs, AXIS_ERROR_PERCENTILE),
                    percentile(ys, AXIS_ERROR_PERCENTILE),
                ),
            )
        })
        .collect())
}

/// `numpy.percentile` (linear interpolation) of a non-empty list.
fn percentile(mut values: Vec<f64>, q: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    let position = q / 100.0 * (values.len() - 1) as f64;
    let below = position.floor() as usize;
    let above = (below + 1).min(values.len() - 1);
    let t = position - below as f64;
    let diff = values[above] - values[below];
    if t >= 0.5 {
        values[above] - diff * (1.0 - t)
    } else {
        values[below] + diff * t
    }
}

/// `"excellent"` (>= 0.97), `"good"` (>= 0.9), `"fair"` (>= 0.75) or `"poor"`.
pub fn grade_for(monitor_accuracy: f64) -> &'static str {
    for (name, threshold) in GRADE_THRESHOLDS {
        if monitor_accuracy >= threshold {
            return name;
        }
    }
    "poor"
}

/// Fit the final model and estimate how well it will work.
///
/// Hyper-parameters are chosen by leave-one-point-out cross-validation (every
/// fold hides all samples of one calibration dot): `alpha` always, and the
/// polynomial degree too unless `degree` is given. The report is computed from
/// the held-out predictions of the chosen settings, so it reflects accuracy on
/// gaze points the model has not seen. The returned model is fitted on all samples.
///
/// `nonlinear` should be the backend's gaze-direction feature indices (see
/// `gaze::model::gaze_feature_indices`): only they get polynomial terms, which
/// keeps the model accurate when the user's posture changes. `None` expands
/// every feature (the behaviour before backends declared them).
///
/// A monitor without any calibration dot (possible with three or more monitors)
/// is reported with 0 % accuracy in `per_monitor_accuracy`, listed in
/// `uncovered_monitors` and caps the grade at "fair".
///
/// Fails with `not enough calibration data` with fewer than 10
/// samples, fewer than 3 points, or points on fewer than two monitors (one on a
/// single-monitor desk).
pub fn evaluate(
    samples: &[CalibrationSample],
    monitors: &[Monitor],
    degree: Option<usize>,
    nonlinear: Option<&[usize]>,
) -> Result<(GazeModel, CalibrationReport), CalibrationError> {
    if monitors.is_empty() {
        return invalid("no monitors");
    }
    let known: BTreeSet<i32> = monitors.iter().map(|m| m.index).collect();
    let usable: Vec<CalibrationSample> = samples
        .iter()
        .filter(|s| known.contains(&s.monitor_index))
        .cloned()
        .collect();
    if usable.len() < samples.len() {
        log::warn!(
            "Ignoring {} samples of unknown monitors",
            samples.len() - usable.len()
        );
    }

    let point_ids: Vec<i64> = usable.iter().map(|s| s.point_id).collect();
    let n_points = point_ids.iter().collect::<BTreeSet<_>>().len();
    let covered: BTreeSet<i32> = usable.iter().map(|s| s.monitor_index).collect();
    if usable.len() < MIN_SAMPLES || n_points < MIN_POINTS || covered.len() < 2.min(monitors.len())
    {
        return invalid("not enough calibration data");
    }

    let (x, y, w) = samples_to_arrays(&usable)?;
    let bounds = virtual_bounds(monitors).map_err(|e| CalibrationError(e.to_string()))?;
    let degrees: Vec<usize> = match degree {
        None => SUPPORTED_DEGREES.to_vec(),
        Some(d) => vec![d],
    };
    let options = FitOptions {
        bounds: Some(bounds),
        weights: Some(&w),
        nonlinear,
    };
    let selection = select_model(&x, &y, &point_ids, &DEFAULT_ALPHAS, &degrees, &options)?;
    let preds = match selection.predictions.clone() {
        Some(preds) => preds,
        // Every candidate diverged; still report honest held-out errors.
        None => lopo_predictions(
            &x,
            &y,
            &point_ids,
            selection.alpha,
            selection.degree,
            &options,
        )?,
    };

    let truth: Vec<i32> = usable.iter().map(|s| s.monitor_index).collect();
    let correct: Vec<bool> = (0..preds.nrows())
        .map(|i| predicted_monitor(monitors, preds[(i, 0)], preds[(i, 1)]) == Some(truth[i]))
        .collect();
    let errors: Vec<f64> = (0..preds.nrows())
        .map(|i| (preds[(i, 0)] - y[(i, 0)]).hypot(preds[(i, 1)] - y[(i, 1)]))
        .filter(|e| e.is_finite())
        .collect();

    // A monitor without dots was never calibrated: the model only extrapolates
    // there, and gaze at it is often where the face is lost. An "Excellent"
    // grade computed over the other monitors would hide that.
    let uncovered: Vec<i32> = known.difference(&covered).copied().collect();
    let mut per_monitor: BTreeMap<i32, f64> = covered
        .iter()
        .map(|&idx| {
            let rows: Vec<bool> = truth
                .iter()
                .zip(&correct)
                .filter(|(t, _)| **t == idx)
                .map(|(_, c)| *c)
                .collect();
            (idx, fraction(&rows))
        })
        .collect();
    per_monitor.extend(uncovered.iter().map(|&idx| (idx, 0.0)));
    let monitor_accuracy = fraction(&correct);
    let mut grade = grade_for(monitor_accuracy);
    if !uncovered.is_empty() && (grade == "excellent" || grade == "good") {
        grade = "fair";
    }
    if !uncovered.is_empty() {
        log::warn!("Calibration has no points on monitor(s) {uncovered:?}");
    }

    let report = CalibrationReport {
        monitor_accuracy,
        mean_error_px: if errors.is_empty() {
            f64::NAN
        } else {
            errors.iter().sum::<f64>() / errors.len() as f64
        },
        median_error_px: median(&errors),
        per_monitor_accuracy: per_monitor,
        n_samples: usable.len(),
        n_points,
        alpha: selection.alpha,
        grade: grade.to_string(),
        degree: selection.degree,
        uncovered_monitors: uncovered,
        per_monitor_error_px: per_axis_errors(&preds, &y, &truth)?,
    };
    let regions: Vec<_> = monitors.iter().map(|m| m.rect).collect();
    let model = GazeModel::new(selection.degree, selection.alpha, nonlinear)?.fit(
        &x,
        &y,
        Some(&w),
        Some(bounds),
        Some(&regions),
    )?;
    log::info!(
        "Calibration evaluated: {} (degree {}, alpha {})",
        report.summary(),
        report.degree,
        report.alpha
    );
    Ok((model, report))
}

fn predicted_monitor(monitors: &[Monitor], px: f64, py: f64) -> Option<i32> {
    if !(px.is_finite() && py.is_finite()) {
        return None;
    }
    nearest_monitor(monitors, px, py).ok().map(|(m, _)| m.index)
}

/// The share of `true` values (`numpy.mean` of a boolean array).
fn fraction(values: &[bool]) -> f64 {
    values.iter().filter(|v| **v).count() as f64 / values.len() as f64
}

/// `numpy.median`; `NaN` for no values.
fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

// ------------------------------------------------------------ JSON conversions
// What Python's `float(v)` and `int(v)` accept of a JSON value, restricted to numbers (as in
// `gaze::model`): floats truncate for `int`, non-finite or huge ones fail.
pub(crate) fn json_float(v: &Value) -> Result<f64, String> {
    v.as_f64()
        .ok_or_else(|| format!("float() argument must be a number, not {v}"))
}

pub(crate) fn json_int(v: &Value) -> Result<i64, String> {
    if let Some(i) = v.as_i64() {
        return Ok(i);
    }
    let f = json_float(v)?;
    if !f.is_finite() || f.abs() >= 9.2e18 {
        return Err("cannot convert float to integer".into());
    }
    Ok(f.trunc() as i64)
}

pub(crate) fn json_i32(v: &Value) -> Result<i32, String> {
    i32::try_from(json_int(v)?).map_err(|_| format!("integer {v} is out of range"))
}

/// A dictionary key as `int(key)`.
fn parse_key(key: &str) -> Result<i32, String> {
    key.trim()
        .parse()
        .map_err(|_| format!("invalid literal for int(): '{key}'"))
}

/// `_float`: `null` is NaN.
fn float_or_nan(v: &Value) -> Result<f64, String> {
    match v {
        Value::Null => Ok(f64::NAN),
        other => json_float(other),
    }
}

#[cfg(test)]
#[path = "calibration_tests.rs"]
mod tests;
