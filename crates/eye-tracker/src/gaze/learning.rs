//! Keeping the calibration accurate from everyday use.
//!
//! People look at the mouse pointer when they place it: after a manual cursor move
//! comes to rest (to click, select or read), the eyes are almost always near the
//! pointer. Each such moment is a free, weakly labelled calibration sample. Mixing
//! them into the fit (at a lower weight than deliberate calibration samples) lets
//! the model follow slow changes such as a different chair height or a moved
//! camera. The same moments reveal when the model has drifted: if its prediction
//! keeps disagreeing with where the user puts the cursor, it is time to recalibrate.
//!
//! Not every settle is a good label: the user may click "Run" on one monitor and
//! already watch the output on the other. [`plausible_label`] rejects labels
//! far from what the model currently predicts, and [`ImplicitLearner::discard_last`]
//! takes such a sample back out of the training set. The gate is generous on
//! purpose, so the samples that let the model follow a posture change still pass.

use std::collections::{BTreeMap, VecDeque};

use super::calibration::{CalibrationError, CalibrationSample, samples_to_arrays};
use super::model::GazeModel;
use crate::types::{Monitor, Observation, Rect, nearest_monitor};

/// A settle only counts if the cursor was moved within this many seconds.
pub const RECENT_MOVE_S: f64 = 1.5;

/// A learned label is implausible when the model's current prediction is farther
/// from it than this fraction of the diagonal of the cursor's monitor (about
/// 1100 px on a 1920x1080 monitor). In synthetic tests this removes every label
/// of "cursor parked on one monitor, reading the other" while keeping over 90 %
/// of the samples that correct an 8 cm change of posture; requiring the
/// prediction to be on the cursor's monitor would drop half of the latter.
pub const IMPLICIT_MAX_ERROR: f64 = 0.5;

/// The default cooldown of [`DriftMonitor::should_alert`], in seconds.
pub const ALERT_COOLDOWN_S: f64 = 1800.0;

/// Maps a cursor position to a monitor index (`None` outside every monitor).
pub type MonitorLookup<'a> = dyn Fn(f64, f64) -> Option<i32> + 'a;

/// The settings of an [`ImplicitLearner`] (its Python keyword arguments).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LearnerOptions {
    pub max_samples: usize,
    pub weight: f64,
    pub settle_s: f64,
    pub refit_every: usize,
}

impl Default for LearnerOptions {
    fn default() -> Self {
        LearnerOptions {
            max_samples: 400,
            weight: 0.5,
            settle_s: 0.35,
            refit_every: 25,
        }
    }
}

/// Adds weak labels from natural mouse use: when the user moves the cursor manually and it
/// settles, they are very likely looking near it.
///
/// Call [`on_manual_cursor`](Self::on_manual_cursor) for every poll that saw a user-initiated
/// cursor move and [`on_observation`](Self::on_observation) for every observation. A *settle*
/// is the moment the cursor has not moved for `settle_s` seconds after a move; the
/// first usable observation within `1.5 s` of the last move becomes one sample
/// (one sample per settle, never more).
///
/// Implicit samples get negative `point_id` values (`-1, -2, …`), one per
/// settle, so they never collide with calibration point ids. When full, the
/// oldest sample of the monitor with the most samples is dropped, so one
/// heavily used monitor cannot push the others out of the training set.
#[derive(Debug, Clone)]
pub struct ImplicitLearner {
    pub max_samples: usize,
    pub weight: f64,
    pub settle_s: f64,
    pub refit_every: usize,
    samples: Vec<CalibrationSample>,
    last_move: Option<f64>,
    cursor: (f64, f64),
    /// A settle is pending and has not produced a sample yet.
    armed: bool,
    since_refit: usize,
    next_id: i64,
    // Undo information for discard_last(): the latest sample (found again by its unique
    // point id), whether it still counts towards the next refit, and the (index, sample)
    // pairs evicted to make room for it, in eviction order.
    last_added: Option<CalibrationSample>,
    last_counted: bool,
    last_evicted: Vec<(usize, CalibrationSample)>,
}

impl Default for ImplicitLearner {
    fn default() -> Self {
        Self::from_options(LearnerOptions::default())
    }
}

impl ImplicitLearner {
    pub fn new(options: LearnerOptions) -> Result<Self, CalibrationError> {
        if options.weight <= 0.0 || options.settle_s < 0.0 || options.refit_every < 1 {
            return Err(CalibrationError(
                "max_samples must be >= 0, weight > 0, settle_s >= 0 and refit_every >= 1".into(),
            ));
        }
        Ok(Self::from_options(options))
    }

    fn from_options(options: LearnerOptions) -> Self {
        ImplicitLearner {
            max_samples: options.max_samples,
            weight: options.weight,
            settle_s: options.settle_s,
            refit_every: options.refit_every,
            samples: Vec::new(),
            last_move: None,
            cursor: (0.0, 0.0),
            armed: false,
            since_refit: 0,
            next_id: -1,
            last_added: None,
            last_counted: false,
            last_evicted: Vec::new(),
        }
    }

    // -------------------------------------------------------------- events
    /// Record a user-initiated cursor position (called on each manual move sample).
    pub fn on_manual_cursor(&mut self, x: f64, y: f64, now: f64) {
        self.last_move = Some(now);
        self.cursor = (x, y);
        self.armed = true;
    }

    /// Turn the observation into a sample if the cursor has just settled.
    ///
    /// `monitor_index_at(x, y)` maps the cursor position to a monitor index
    /// (`None` outside every monitor). Returns the new sample, or `None`.
    pub fn on_observation(
        &mut self,
        obs: &Observation,
        now: f64,
        monitor_index_at: &MonitorLookup,
    ) -> Option<CalibrationSample> {
        let last_move = self.last_move?;
        if !self.armed || self.max_samples == 0 {
            return None;
        }
        let since_move = now - last_move;
        if since_move < self.settle_s {
            return None; // still moving (or about to move again)
        }
        if since_move > RECENT_MOVE_S {
            self.armed = false; // this settle is too old to trust
            return None;
        }
        if !obs.usable() || obs.skipped {
            return None; // keep waiting for a good frame within the window
        }
        let features = obs.features.as_ref()?;
        if features.is_empty() || !features.iter().all(|v| v.is_finite()) {
            return None;
        }

        let (x, y) = self.cursor;
        let monitor = monitor_index_at(x, y);
        self.armed = false;
        let monitor = monitor?;

        if let Some(first) = self.samples.first()
            && first.features.len() != features.len()
        {
            log::warn!(
                "Feature length changed ({} -> {}); discarding learned samples",
                first.features.len(),
                features.len()
            );
            self.samples.clear();
        }
        let sample =
            CalibrationSample::weighted(features.clone(), x, y, monitor, self.next_id, self.weight);
        self.next_id -= 1;
        self.samples.push(sample.clone());
        let evicted = self.trim();
        self.since_refit += 1;
        self.last_added = Some(sample.clone());
        self.last_counted = true;
        self.last_evicted = evicted;
        Some(sample)
    }

    /// Take back the sample most recently returned by [`on_observation`](Self::on_observation).
    ///
    /// For a label the caller finds implausible (see [`plausible_label`]): the
    /// sample leaves the training set, no longer counts towards the next refit,
    /// and the samples evicted to make room for it are restored. Returns the
    /// discarded sample, or `None` if there is nothing to undo (it was already
    /// discarded, or the samples were replaced, cleared or trimmed since).
    pub fn discard_last(&mut self) -> Option<CalibrationSample> {
        let sample = self.last_added.take()?;
        // Learned ids are unique (see `load`), so the id finds the very sample.
        self.samples.retain(|s| s.point_id != sample.point_id);
        // Evictions shifted later indices; undoing them in reverse order restores
        // the list exactly (a victim always precedes the newest sample).
        for (index, victim) in std::mem::take(&mut self.last_evicted).into_iter().rev() {
            if victim.point_id != sample.point_id {
                self.samples.insert(index.min(self.samples.len()), victim);
            }
        }
        if self.last_counted {
            self.since_refit = self.since_refit.saturating_sub(1);
        }
        self.forget_last();
        Some(sample)
    }

    // -------------------------------------------------------------- samples
    /// Learned samples, oldest first.
    pub fn samples(&self) -> &[CalibrationSample] {
        &self.samples
    }

    /// Samples added since the last [`mark_refit`](Self::mark_refit).
    pub fn new_since_refit(&self) -> usize {
        self.since_refit
    }

    /// Replace the learned samples (e.g. with those saved in the calibration file).
    pub fn load(&mut self, samples: impl IntoIterator<Item = CalibrationSample>) {
        self.samples = samples.into_iter().collect();
        self.trim();
        let lowest = self
            .samples
            .iter()
            .map(|s| s.point_id)
            .filter(|&id| id < 0)
            .min()
            .unwrap_or(0);
        self.next_id = lowest - 1;
        self.since_refit = 0;
        self.forget_last();
    }

    /// Forget every learned sample (after a new calibration).
    pub fn clear(&mut self) {
        self.samples.clear();
        self.since_refit = 0;
        self.next_id = -1;
        self.armed = false;
        self.forget_last();
    }

    /// Change the capacity, dropping samples if needed (0 disables learning).
    ///
    /// Returns true if samples were dropped. The model fitted with them then
    /// still carries their influence, so the caller should refit it (with
    /// [`refit_model`] and the remaining [`samples`](Self::samples); with none left
    /// that reproduces the calibration-only fit) and save the result.
    pub fn set_max_samples(&mut self, max_samples: usize) -> bool {
        self.max_samples = max_samples;
        let dropped = !self.trim().is_empty();
        if dropped {
            self.forget_last();
        }
        dropped
    }

    /// True once `refit_every` new samples have arrived since the last refit.
    pub fn should_refit(&self) -> bool {
        self.since_refit >= self.refit_every
    }

    pub fn mark_refit(&mut self) {
        self.since_refit = 0;
        self.last_counted = false;
    }

    fn forget_last(&mut self) {
        self.last_added = None;
        self.last_counted = false;
        self.last_evicted.clear();
    }

    /// Drop samples beyond the capacity; returns the `(index, sample)` evicted, in order.
    fn trim(&mut self) -> Vec<(usize, CalibrationSample)> {
        let mut evicted = Vec::new();
        while self.samples.len() > self.max_samples {
            let mut counts: BTreeMap<i32, usize> = BTreeMap::new();
            for s in &self.samples {
                *counts.entry(s.monitor_index).or_default() += 1;
            }
            let busiest = counts.values().copied().max().unwrap_or(0);
            // Samples are oldest first, so the first sample of a busiest monitor is
            // the oldest among them; ties between monitors go to the older sample.
            let victim = self
                .samples
                .iter()
                .position(|s| counts[&s.monitor_index] == busiest)
                .unwrap_or(0);
            evicted.push((victim, self.samples.remove(victim)));
        }
        evicted
    }
}

/// Fit a new model on calibration plus learned samples.
///
/// The template supplies the degree, `alpha` and nonlinear features chosen at
/// calibration time and, unless `bounds` is given, the target normalisation.
/// Sample weights are honoured, so learned samples count less than calibration
/// samples. Without learned samples this reproduces the calibration-only fit.
/// The look-away regions (`GazeModel::away_regions`) are learned for
/// `monitors` (the calibrated layout), or else for the monitors the template
/// had them for.
pub fn refit_model(
    base: &[CalibrationSample],
    implicit: &[CalibrationSample],
    template: &GazeModel,
    bounds: Option<Rect>,
    monitors: Option<&[Monitor]>,
) -> Result<GazeModel, CalibrationError> {
    let items: Vec<CalibrationSample> = base.iter().chain(implicit).cloned().collect();
    let (x, y, w) = samples_to_arrays(&items)?;
    let regions: Vec<Rect> = match monitors {
        Some(monitors) => monitors.iter().map(|m| m.rect).collect(),
        None => template.away_regions().iter().map(|r| r.rect).collect(),
    };
    let model = GazeModel::new(template.degree, template.alpha, template.nonlinear())?;
    Ok(model.fit(
        &x,
        &y,
        Some(&w),
        bounds.or_else(|| template.bounds()),
        Some(&regions),
    )?)
}

/// Whether a learned sample's label (the settled cursor) is plausible.
///
/// `predicted` is the current model's gaze point for `sample.features`. The
/// label is implausible when it lies more than `max_error` times the diagonal
/// of the cursor's monitor (`sample.monitor_index`, else the monitor nearest
/// to the cursor) away from the prediction. The prediction need not be on the
/// same monitor, so labels that correct a drifted model still count. A
/// non-finite prediction or no monitors gives no evidence either way: true.
pub fn plausible_label(
    predicted: &[f64],
    sample: &CalibrationSample,
    monitors: &[Monitor],
    max_error: f64,
) -> bool {
    let (px, py) = (predicted[0], predicted[1]);
    if !(px.is_finite() && py.is_finite()) || monitors.is_empty() {
        return true;
    }
    let monitor = monitors
        .iter()
        .find(|m| m.index == sample.monitor_index)
        .or_else(|| {
            nearest_monitor(monitors, sample.x, sample.y)
                .ok()
                .map(|(m, _)| m)
        });
    let Some(monitor) = monitor else {
        return true;
    };
    (px - sample.x).hypot(py - sample.y) <= max_error * monitor.rect.diagonal()
}

/// Tracks disagreement between prediction and manual cursor placement.
///
/// Each event is either a comparison between the monitor the model predicts and
/// the monitor where the user settled the cursor, or the outcome of an automatic
/// switch: undone by the user within a moment (an error) or kept (correct).
/// Recording both outcomes of switches matters: with adaptive learning off there
/// are no settle comparisons, and counting only the undone switches would make
/// every event an error. When at least `min_events` of the last `window`
/// events exist and at least `alert_ratio` of them are errors,
/// [`should_alert`](Self::should_alert) fires (at most once per cooldown).
#[derive(Debug, Clone)]
pub struct DriftMonitor {
    pub alert_ratio: f64,
    pub min_events: usize,
    window: usize,
    events: VecDeque<bool>,
    last_alert: Option<f64>,
}

impl Default for DriftMonitor {
    fn default() -> Self {
        DriftMonitor {
            alert_ratio: 0.35,
            min_events: 15,
            window: 40,
            events: VecDeque::new(),
            last_alert: None,
        }
    }
}

impl DriftMonitor {
    pub fn new(
        window: usize,
        alert_ratio: f64,
        min_events: usize,
    ) -> Result<Self, CalibrationError> {
        if window < 1 || !(alert_ratio > 0.0 && alert_ratio <= 1.0) || min_events < 1 {
            return Err(CalibrationError(
                "window >= 1, 0 < alert_ratio <= 1 and min_events >= 1 required".into(),
            ));
        }
        Ok(DriftMonitor {
            alert_ratio,
            min_events,
            window,
            events: VecDeque::new(),
            last_alert: None,
        })
    }

    fn push(&mut self, error: bool) {
        if self.events.len() == self.window {
            self.events.pop_front();
        }
        self.events.push_back(error);
    }

    /// Compare a prediction with where the user put the cursor.
    ///
    /// `predicted_monitor` `None` means there was no prediction (no face,
    /// blink); such events carry no information and are ignored.
    pub fn record(&mut self, predicted_monitor: Option<i32>, actual_monitor: i32) {
        if let Some(predicted) = predicted_monitor {
            self.push(predicted != actual_monitor);
        }
    }

    /// Outcome of an automatic switch: `ok` false if the user undid it at once
    /// (moved the mouse back within the wrong-switch window), true once that
    /// window has passed without such a correction.
    pub fn record_switch(&mut self, ok: bool) {
        self.push(!ok);
    }

    /// The user moved the mouse back within 2 s of an automatic switch.
    pub fn record_wrong_switch(&mut self) {
        self.record_switch(false);
    }

    /// Fraction of recent events that were errors (0 without events).
    pub fn error_rate(&self) -> f64 {
        if self.events.is_empty() {
            return 0.0;
        }
        self.events.iter().filter(|e| **e).count() as f64 / self.events.len() as f64
    }

    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// True if accuracy has dropped and no alert was given in the last `cooldown_s`
    /// ([`ALERT_COOLDOWN_S`] by default).
    ///
    /// Returning true starts the cooldown.
    pub fn should_alert(&mut self, now: f64, cooldown_s: f64) -> bool {
        if self.events.len() < self.min_events || self.error_rate() < self.alert_ratio {
            return false;
        }
        if self.last_alert.is_some_and(|last| now - last < cooldown_s) {
            return false;
        }
        self.last_alert = Some(now);
        log::info!(
            "Gaze accuracy dropped: {:.0}% recent disagreement",
            self.error_rate() * 100.0
        );
        true
    }

    /// Forget all events and the alert cooldown (after a new calibration).
    pub fn reset(&mut self) {
        self.events.clear();
        self.last_alert = None;
    }
}

#[cfg(test)]
#[path = "learning_tests.rs"]
mod tests;
