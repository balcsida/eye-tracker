//! Regression from a vision feature vector to a point on the virtual desktop.
//!
//! The mapping between head pose / iris position and the point being looked at is
//! smooth but non-linear (perspective, eye-ball rotation, head/eye coordination) and
//! differs per person and per desk set-up. A ridge regression on standardised
//! features with low-order polynomial terms captures it well from a few hundred
//! calibration samples, fits in about a millisecond and predicts in microseconds,
//! which matters because prediction runs for every analysed frame.
//!
//! Design matrices by `degree`:
//!
//! * `1`: `[1, z_i]`
//! * `2`: `[1, z_i, z_j*z_k (j <= k)]`
//! * `3`: degree 2 plus the pure cubes `z_j**3`. Looking at a flat screen from
//!   close up maps angles to pixels through `tan`, which bends noticeably beyond
//!   about 30 degrees (wide or triple-monitor desks). Pure cubes model that odd,
//!   one-sided curvature at the cost of only `k` extra terms; the full cubic
//!   expansion would add `O(k**3)` terms and overfit.
//!
//! `j` and `k` run over the *nonlinear* features only ([`GazeModel::nonlinear`];
//! all features when it is `None`, the behaviour of models saved before it
//! existed). A backend declares which of its features encode gaze direction (head
//! rotation, iris offsets) and only those get products and cubes. Head position and
//! roll barely vary while the user calibrates, so cross-validation, which holds out
//! a dot but never a posture, cannot tell that a curved fit through them is noise,
//! and such a fit falls apart as soon as the user sits lower or leans back. Their
//! linear terms still correct the gaze for moderate posture changes.
//!
//! Four details keep the model well behaved:
//!
//! * Targets are normalised to the virtual-desktop bounds, so the regularisation
//!   strength `alpha` means the same thing on a 1080p laptop and a triple-4K desk.
//! * The intercept is not penalised, so shrinkage pulls predictions towards the
//!   mean gaze point rather than towards the desktop origin.
//! * Features are clipped to the calibrated range (widened by a margin) before the
//!   polynomial expansion. A pose never seen during calibration (for example a head
//!   roll the user did not make while calibrating) would otherwise be squared or
//!   cubed and could throw the prediction thousands of pixels off.
//! * Clipping hides how far outside the calibrated range a frame lies, and the
//!   clipped polynomial may even bend back onto the screens. [`GazeModel::looks_away`]
//!   therefore reports gaze far off every monitor (the user reads a phone on the
//!   desk or turns to a colleague), so such frames are treated as looking away
//!   instead of as gaze at the nearest monitor.
//!
//! Looking away is judged from the *combined* gaze direction: a second, purely
//! linear ridge fit on the same standardised features, evaluated without clipping
//! the gaze-direction features (head position and roll stay clipped, so leaning
//! back or sitting lower cannot throw it off). Linear extrapolation is monotone,
//! so gaze well below or beside the screens lands well outside them. Head rotation
//! and eye-in-head offset are complementary - the same point can be looked at with
//! more head and less eye - and a fit on both keeps its estimate on the screens
//! when the user shares the work differently than while calibrating, where testing
//! each feature against its own calibrated range would call such gaze "away". The
//! strong ridge keeps the fit from leaning on either of two nearly collinear
//! features. Gaze at a monitor whose calibration dots were all skipped (which
//! `calibration::evaluate` reports) is no longer "away" merely because it lies
//! beyond the calibrated range; the clipped model places it on or next to that
//! monitor. Models saved before this estimate existed fall back to the per-feature
//! range test (see [`LOOK_AWAY_EXCESS`]).
//!
//! One linear map serves the whole desk, but a turn of the gaze covers more pixels
//! on a dense monitor (a 4K panel next to a 1080p one: Windows and X11 report
//! native pixels) than on a coarse one. The map then overshoots on the coarse
//! monitor, whose outer edge it places well outside that monitor, most of all
//! when the user leans in. [`GazeModel::fit`] therefore also learns, from the
//! calibration samples of each monitor, where the linear estimate actually puts
//! that monitor ([`AwayRegion`]); gaze near that place is never "away"
//! either. On a desk of equal monitors the place is the monitor itself, so the
//! test only ever becomes more lenient than with the monitor rectangles alone.

use std::fmt;

use nalgebra::DMatrix;
use serde_json::{Map, Value, json};
use thiserror::Error;

use crate::types::{Monitor, Rect, nearest_monitor};

/// Identifier written by [`GazeModel::to_dict`]; lets future model types coexist.
pub const MODEL_KIND: &str = "ridge-poly";

/// Standard deviations below this are treated as this value (constant features).
pub const STD_FLOOR: f64 = 1e-6;

/// Features are clipped to the training range widened by this fraction of the
/// range on each side. Calibration targets stop short of the screen edges, so a
/// generous margin is needed to still reach the edges and corners.
pub const CLIP_MARGIN: f64 = 0.5;

pub const DEFAULT_ALPHAS: [f64; 5] = [0.01, 0.1, 1.0, 10.0, 100.0];
pub const SUPPORTED_DEGREES: [usize; 3] = [1, 2, 3];

/// A frame looks away from every monitor when the linear gaze estimate (see the
/// module docstring) lies more than this fraction of the nearest monitor's
/// diagonal outside that monitor: about 12 cm beyond the edge of a 24" screen.
/// In synthetic tests (two, three and stacked monitors; posture changes of
/// 5-12 cm, and users who turn the head 25 % more or less than while
/// calibrating) at most 0.1 % of on-screen frames, edges and corners included,
/// exceed it; the worst case tried - nodding 40 % more while sitting 5 cm
/// higher - flags up to 5 % of the frames on the taskbar, where the per-feature
/// test flagged up to 80 %. Over 90 % of the frames of gaze 30 cm below the
/// screens and practically all at 40-80 cm (a phone or papers on the desk)
/// exceed it. Treated as "no gaze", every such frame restarts the switch dwell,
/// so even partial detection keeps a glance at the phone from switching monitors.
pub const LOOK_AWAY_MARGIN: f64 = 0.2;

/// Ridge strength of that linear estimate (standardised features, targets
/// normalised to the desktop). Stronger than the gaze model's usual choice:
/// head and eye features are nearly collinear during a calibration, and an
/// even split between them keeps the estimate on the screens when the user
/// later moves the head more or less, at a negligible cost in slope.
pub const LOOK_AWAY_ALPHA: f64 = 10.0;

/// Fallback for models saved without the linear estimate: a frame looks away
/// when one of its gaze-direction features lies more than this fraction of its
/// calibrated range beyond that range. Calibration dots stop 10 % short of the
/// screen edges, so gaze at an edge lies about 0.125 of the range beyond it.
/// This test is weaker (about half of the frames of gaze 40 cm below the
/// screens) and flags on-screen gaze once the user moves the head more or less
/// than while calibrating; recalibrating, or a refit by adaptive learning,
/// replaces it with the linear estimate.
pub const LOOK_AWAY_EXCESS: f64 = 0.25;

/// An [`AwayRegion`] is learned from a monitor's samples only when their
/// targets (at least three distinct ones) span at least this fraction of the
/// monitor in both directions; otherwise the corners would be extrapolated from
/// too narrow a base.
pub const REGION_MIN_SPAN: f64 = 0.25;

/// ...and only when the image of the monitor is at most this many times its
/// diagonal: a larger one is a fit gone wrong, not a density difference (4K
/// next to 1080p scales by about 1.3).
pub const REGION_MAX_SCALE: f64 = 3.0;

/// A more complex candidate (higher degree, smaller alpha) must beat the simpler
/// one by this relative margin in cross-validation to be chosen.
const PREFER_SIMPLER: f64 = 1e-3;

/// What Python raised: `Value` is a `ValueError`, `NotFitted` the `RuntimeError`.
#[derive(Debug, Error, PartialEq)]
pub enum ModelError {
    #[error("{0}")]
    Value(String),
    #[error("GazeModel is not fitted")]
    NotFitted,
}

fn invalid<T>(message: impl Into<String>) -> Result<T, ModelError> {
    Err(ModelError::Value(message.into()))
}

/// Where the linear look-away estimate places one calibrated monitor.
///
/// `rect` is the monitor as calibrated. `x0, y0, x1, y1` bound both the
/// rectangle and its image under the linear estimate (an affine fit through
/// the monitor's samples); `diagonal` is the larger of the two diagonals.
/// Gaze within `margin * diagonal` of the box does not look away (see
/// [`GazeModel::looks_away`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AwayRegion {
    pub rect: Rect,
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
    pub diagonal: f64,
}

impl AwayRegion {
    /// The region of a monitor whose image is unknown: the monitor itself.
    pub fn of(rect: Rect) -> AwayRegion {
        AwayRegion {
            rect,
            x0: f64::from(rect.x),
            y0: f64::from(rect.y),
            x1: f64::from(rect.right()),
            y1: f64::from(rect.bottom()),
            diagonal: rect.diagonal(),
        }
    }

    /// Euclidean distance from the point to the box (0 when inside).
    pub fn distance_outside(&self, px: f64, py: f64) -> f64 {
        let dx = (self.x0 - px).max(0.0).max(px - self.x1);
        let dy = (self.y0 - py).max(0.0).max(py - self.y1);
        dx.hypot(dy)
    }

    pub fn to_dict(&self) -> Value {
        json!({
            "rect": self.rect.to_list(),
            "box": [self.x0, self.y0, self.x1, self.y1],
            "diagonal": self.diagonal,
        })
    }

    /// Inverse of [`AwayRegion::to_dict`]. Fails with a `ValueError` on malformed input.
    pub fn from_dict(d: &Value) -> Result<AwayRegion, ModelError> {
        away_region_from_json(d).map_err(ModelError::Value)
    }
}

fn away_region_from_json(d: &Value) -> Result<AwayRegion, String> {
    let Some(map) = d.as_object() else {
        return Err("away region must be an object".into());
    };
    let parse = || -> Result<(Rect, [f64; 4], f64), String> {
        let rect = json_rect(map.get("rect").ok_or("missing key 'rect'")?)?;
        let list = map
            .get("box")
            .and_then(Value::as_array)
            .ok_or("box must be a list of 4 numbers")?;
        if list.len() != 4 {
            return Err("box must be a list of 4 numbers".into());
        }
        let mut boxed = [0.0; 4];
        for (slot, v) in boxed.iter_mut().zip(list) {
            *slot = json_float(v)?;
        }
        let diagonal = json_float(map.get("diagonal").ok_or("missing key 'diagonal'")?)?;
        Ok((rect, boxed, diagonal))
    };
    let (rect, [x0, y0, x1, y1], diagonal) =
        parse().map_err(|exc| format!("invalid away region: {exc}"))?;
    if ![x0, y0, x1, y1, diagonal].iter().all(|v| v.is_finite()) {
        return Err("away region contains non-finite numbers".into());
    }
    if rect.w <= 0 || rect.h <= 0 || x1 < x0 || y1 < y0 || diagonal <= 0.0 {
        return Err("away region is inconsistent".into());
    }
    Ok(AwayRegion {
        rect,
        x0,
        y0,
        x1,
        y1,
        diagonal,
    })
}

/// What a fit produces; the model is unfitted while this is absent.
#[derive(Debug, Clone)]
struct Fitted {
    mean: Vec<f64>,
    std: Vec<f64>,
    /// The calibrated range `(lo, hi)`; absent in files written before it was stored.
    range: Option<(Vec<f64>, Vec<f64>)>,
    /// `(n_terms, 2)`, in normalised target space.
    coef: DMatrix<f64>,
    /// `(1 + d, 2)`: the linear estimate behind `looks_away` (absent: saved without it).
    linear_coef: Option<DMatrix<f64>>,
    bounds: Rect,
}

/// Maps a feature vector to a gaze point on the virtual desktop (ridge regression on
/// standardised features with optional polynomial terms, see the module docstring).
///
/// `nonlinear` lists the indices of the features that get quadratic and cubic
/// terms (the backend's gaze-direction features, see [`gaze_feature_indices`]);
/// the others enter linearly. `None` expands every feature. Indices are sorted
/// and de-duplicated; they are checked against the feature count when fitting.
#[derive(Clone)]
pub struct GazeModel {
    pub degree: usize,
    pub alpha: f64,
    nonlinear: Option<Vec<usize>>,
    fitted: Option<Fitted>,
    /// Where the linear estimate places each calibrated monitor (empty: fitted without
    /// the monitors, or saved before the regions existed).
    away_regions: Vec<AwayRegion>,
}

impl fmt::Debug for GazeModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = if self.is_fitted() {
            format!("fitted, {} features", self.n_features())
        } else {
            "unfitted".to_string()
        };
        let nonlinear = match &self.nonlinear {
            None => String::new(),
            Some(v) => format!(", nonlinear={v:?}"),
        };
        write!(
            f,
            "GazeModel(degree={}, alpha={}{nonlinear}, {state})",
            self.degree, self.alpha
        )
    }
}

impl GazeModel {
    pub fn new(
        degree: usize,
        alpha: f64,
        nonlinear: Option<&[usize]>,
    ) -> Result<GazeModel, ModelError> {
        check_degree(degree as i64)?;
        if !alpha.is_finite() || alpha < 0.0 {
            return invalid(format!("alpha must be a finite number >= 0, got {alpha}"));
        }
        Ok(GazeModel {
            degree,
            alpha,
            nonlinear: as_indices(nonlinear),
            fitted: None,
            away_regions: Vec::new(),
        })
    }

    // ---------------------------------------------------------------- properties
    pub fn is_fitted(&self) -> bool {
        self.fitted.is_some()
    }

    /// Length of the feature vectors the model was fitted on (0 if unfitted).
    pub fn n_features(&self) -> usize {
        self.fitted.as_ref().map_or(0, |f| f.mean.len())
    }

    /// Indices of the features with polynomial terms (`None`: all of them).
    pub fn nonlinear(&self) -> Option<&[usize]> {
        self.nonlinear.as_deref()
    }

    /// Virtual-desktop rectangle used to normalise targets (`None` if unfitted).
    pub fn bounds(&self) -> Option<Rect> {
        self.fitted.as_ref().map(|f| f.bounds)
    }

    /// Where the linear look-away estimate places each monitor passed to
    /// [`GazeModel::fit`] as `regions` (see [`AwayRegion`]).
    pub fn away_regions(&self) -> &[AwayRegion] {
        &self.away_regions
    }

    /// Whether [`GazeModel::looks_away`] can use the combined gaze direction (models
    /// fitted by this version; older files fall back to the per-feature test).
    pub fn has_linear_estimate(&self) -> bool {
        self.fitted
            .as_ref()
            .is_some_and(|f| f.linear_coef.is_some())
    }

    // ---------------------------------------------------------------- fitting
    /// Fit the model and return it.
    ///
    /// `x` is `(n, d)` features, `y` is `(n, 2)` targets in global pixels and
    /// `weights` optional non-negative per-sample weights (e.g. 0.5 for samples
    /// learned from mouse use). `bounds` should be the virtual desktop; when
    /// omitted the bounding box of `y` is used. `regions` are the monitor
    /// rectangles the samples were taken on; for each one, where the linear
    /// look-away estimate places it is learned ([`GazeModel::away_regions`]).
    pub fn fit(
        mut self,
        x: &DMatrix<f64>,
        y: &DMatrix<f64>,
        weights: Option<&[f64]>,
        bounds: Option<Rect>,
        regions: Option<&[Rect]>,
    ) -> Result<GazeModel, ModelError> {
        as_matrix(x, "X")?;
        as_targets(y, x.nrows())?;
        let w = as_weights(weights, x.nrows())?;
        let bounds = check_bounds(bounds, y)?;
        check_indices(self.nonlinear.as_deref(), x.ncols())?;

        let (mean, std) = standardisation(x);
        let a = design(&standardise(x, &mean, &std), self.degree, self.nonlinear());
        let aw = scale_rows(&a, &w);
        let lhs = a.transpose() * &aw;
        let yn = normalise(y, bounds);
        let rhs = aw.transpose() * &yn;
        let coef = solve_ridge(&lhs, &rhs, self.alpha);
        // Every design starts with the linear columns [1, z], so the linear fit
        // behind looks_away is the leading block of the same normal equations.
        let k = 1 + x.ncols();
        let linear_coef = solve_ridge(
            &lhs.view((0, 0), (k, k)).into_owned(),
            &rhs.rows(0, k).into_owned(),
            LOOK_AWAY_ALPHA,
        );
        let range = (column_extreme(x, f64::min), column_extreme(x, f64::max));
        // The training samples lie within the calibrated range, so their linear
        // estimate needs no clipping: it is the leading block of the design.
        let estimates = denormalise(&(a.columns(0, k) * &linear_coef), bounds);
        self.away_regions = regions
            .unwrap_or_default()
            .iter()
            .map(|&rect| away_region(rect, y, &estimates, &w))
            .collect();
        self.fitted = Some(Fitted {
            mean,
            std,
            range: Some(range),
            coef,
            linear_coef: Some(linear_coef),
            bounds,
        });
        Ok(self)
    }

    // ---------------------------------------------------------------- prediction
    /// Predict the gaze point in global pixels for one feature vector.
    pub fn predict(&self, x: &[f64]) -> Result<[f64; 2], ModelError> {
        let m = DMatrix::from_row_slice(1, x.len(), x);
        let y = self.predict_many(&m)?;
        Ok([y[(0, 0)], y[(0, 1)]])
    }

    /// Predict gaze points in global pixels: `(n, d) -> (n, 2)`.
    pub fn predict_many(&self, x: &DMatrix<f64>) -> Result<DMatrix<f64>, ModelError> {
        let fitted = self.fitted.as_ref().ok_or(ModelError::NotFitted)?;
        self.check_features(x.ncols(), || format!("({}, {})", x.nrows(), x.ncols()))?;
        let mut x = x.clone();
        if let Some((lo, hi)) = &fitted.range {
            for (c, mut col) in x.column_iter_mut().enumerate() {
                let (low, high) = widened(lo[c], hi[c]);
                col.iter_mut().for_each(|v| *v = clip(*v, low, high));
            }
        }
        let z = standardise(&x, &fitted.mean, &fitted.std);
        let yn = design(&z, self.degree, self.nonlinear()) * &fitted.coef;
        Ok(denormalise(&yn, fitted.bounds))
    }

    // ---------------------------------------------------------------- range checks
    /// How far each feature lies outside the calibrated range, in units of that range.
    ///
    /// `0` inside `[lo, hi]` (the minimum and maximum seen when fitting),
    /// `0.25` a quarter of the range below `lo` or above `hi`, and so on. All
    /// zeros for a model loaded from a file written before the range was stored,
    /// where it is unknown.
    pub fn extrapolation(&self, x: &[f64]) -> Result<Vec<f64>, ModelError> {
        let fitted = self.fitted.as_ref().ok_or(ModelError::NotFitted)?;
        self.check_features(x.len(), || format!("({},)", x.len()))?;
        Ok(match &fitted.range {
            None => vec![0.0; x.len()],
            Some((lo, hi)) => x
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let span = (hi[i] - lo[i]).max(STD_FLOOR);
                    nan_max(nan_max(lo[i] - v, v - hi[i]), 0.0) / span
                })
                .collect(),
        })
    }

    /// [`GazeModel::extrapolation`] for the rows of an `(n, d)` matrix.
    pub fn extrapolation_many(&self, x: &DMatrix<f64>) -> Result<DMatrix<f64>, ModelError> {
        let rows: Vec<Vec<f64>> = x
            .row_iter()
            .map(|r| {
                let row: Vec<f64> = r.iter().copied().collect();
                self.extrapolation(&row)
            })
            .collect::<Result<_, _>>()?;
        if rows.is_empty() {
            self.fitted.as_ref().ok_or(ModelError::NotFitted)?;
            self.check_features(x.ncols(), || format!("({}, {})", x.nrows(), x.ncols()))?;
        }
        Ok(DMatrix::from_fn(x.nrows(), x.ncols(), |r, c| rows[r][c]))
    }

    /// Whether the feature vector `x` `(d,)` shows gaze away from every monitor.
    ///
    /// With `monitors` (the layout the model was calibrated for) and a model
    /// that has the linear estimate ([`GazeModel::has_linear_estimate`]): true when
    /// that estimate of the gaze point lies more than `margin` times the
    /// nearest monitor's diagonal outside it (see [`LOOK_AWAY_MARGIN`] and
    /// the module docstring), and more than `margin` times the diagonal of
    /// every [`GazeModel::away_regions`] entry of those monitors outside that region.
    /// The gaze-direction `features` (indices; default [`GazeModel::nonlinear`])
    /// extrapolate freely; the others are clipped to their calibrated range,
    /// since leaning back is not looking away.
    ///
    /// Otherwise the fallback: true when one of the gaze-direction features
    /// lies more than `threshold` ([`LOOK_AWAY_EXCESS`]) of its calibrated range
    /// outside that range.
    ///
    /// Either way this is false when the gaze-direction features are unknown
    /// (`features` and [`GazeModel::nonlinear`] both `None`, as for models saved
    /// before `nonlinear` existed; pass the backend's indices for those), and
    /// for non-finite features: a broken frame never reads as "looking away".
    pub fn looks_away(
        &self,
        x: &[f64],
        features: Option<&[usize]>,
        threshold: f64,
        monitors: Option<&[Monitor]>,
        margin: f64,
    ) -> Result<bool, ModelError> {
        let indices = match features {
            None => self.nonlinear.clone(),
            Some(_) => as_indices(features),
        }
        .unwrap_or_default();
        let monitors = monitors.filter(|m| !m.is_empty());
        if let (Some(monitors), true) = (monitors, self.has_linear_estimate()) {
            self.check_features(x.len(), || format!("({},)", x.len()))?;
            if indices.is_empty() {
                return Ok(false);
            }
            check_indices(Some(&indices), x.len())?;
            let Some((px, py)) = self.linear_estimate(x, &indices) else {
                return Ok(false);
            };
            let Ok((monitor, distance)) = nearest_monitor(monitors, px, py) else {
                return Ok(false);
            };
            if distance <= margin * monitor.rect.diagonal() {
                return Ok(false);
            }
            // Regions of other layouts (a profile used elsewhere) do not apply.
            return Ok(!self.away_regions.iter().any(|region| {
                monitors.iter().any(|m| m.rect == region.rect)
                    && region.distance_outside(px, py) <= margin * region.diagonal
            }));
        }
        let excess = self.extrapolation(x)?;
        if indices.is_empty() {
            return Ok(false);
        }
        check_indices(Some(&indices), excess.len())?;
        // NaN compares false, so a broken feature never reads as "looking away".
        let worst = indices
            .iter()
            .map(|&i| excess[i])
            .fold(f64::NEG_INFINITY, nan_max);
        Ok(worst > threshold)
    }

    /// The linear gaze estimate for `x` `(d,)` in global pixels, with only the
    /// features other than `gaze_indices` clipped (`None` if not finite).
    fn linear_estimate(&self, x: &[f64], gaze_indices: &[usize]) -> Option<(f64, f64)> {
        let fitted = self.fitted.as_ref()?;
        let linear = fitted.linear_coef.as_ref()?;
        let mut px = linear[(0, 0)];
        let mut py = linear[(0, 1)];
        for (i, &v) in x.iter().enumerate() {
            let v = match &fitted.range {
                Some((lo, hi)) if !gaze_indices.contains(&i) => {
                    let (low, high) = widened(lo[i], hi[i]);
                    clip(v, low, high)
                }
                _ => v,
            };
            let z = (v - fitted.mean[i]) / fitted.std[i];
            px += z * linear[(1 + i, 0)];
            py += z * linear[(1 + i, 1)];
        }
        let b = fitted.bounds;
        let px = px * f64::from(b.w) + f64::from(b.x);
        let py = py * f64::from(b.h) + f64::from(b.y);
        (px.is_finite() && py.is_finite()).then_some((px, py))
    }

    /// Checks the feature count against the fitted one (`shape` describes the input).
    fn check_features(
        &self,
        cols: usize,
        shape: impl FnOnce() -> String,
    ) -> Result<(), ModelError> {
        if cols != self.n_features() {
            return invalid(format!(
                "expected feature vector(s) of length {}, got shape {}",
                self.n_features(),
                shape()
            ));
        }
        Ok(())
    }

    // ---------------------------------------------------------------- persistence
    /// JSON-safe representation (plain lists and numbers).
    ///
    /// Like the Python model, one without a calibrated range (loaded from a very old file)
    /// is written without its fitted arrays.
    pub fn to_dict(&self) -> Value {
        let mut out = Map::new();
        out.insert("kind".into(), json!(MODEL_KIND));
        out.insert("degree".into(), json!(self.degree));
        out.insert("alpha".into(), json!(self.alpha));
        if let Some(nonlinear) = &self.nonlinear {
            // Absent means "every feature", which is how older files are read.
            out.insert("nonlinear".into(), json!(nonlinear));
        }
        if let Some(Fitted {
            mean,
            std,
            range: Some((lo, hi)),
            coef,
            linear_coef,
            bounds,
        }) = &self.fitted
        {
            out.insert("mean".into(), json!(mean));
            out.insert("std".into(), json!(std));
            out.insert("lo".into(), json!(lo));
            out.insert("hi".into(), json!(hi));
            out.insert("coef".into(), matrix_json(coef));
            out.insert("bounds".into(), json!(bounds.to_list()));
            if let Some(linear) = linear_coef {
                // Absent in older files, which then use the per-feature test.
                out.insert("linear_coef".into(), matrix_json(linear));
            }
            if !self.away_regions.is_empty() {
                // Absent in older files: the monitor rectangles alone are used.
                let regions: Vec<Value> =
                    self.away_regions.iter().map(AwayRegion::to_dict).collect();
                out.insert("away_regions".into(), Value::Array(regions));
            }
        }
        Value::Object(out)
    }

    /// Inverse of [`GazeModel::to_dict`]. Fails with a `ValueError` on malformed input.
    pub fn from_dict(d: &Value) -> Result<GazeModel, ModelError> {
        let Some(map) = d.as_object() else {
            return invalid("model must be an object");
        };
        let kind = map.get("kind").unwrap_or(&Value::Null);
        if map.contains_key("kind") && kind.as_str() != Some(MODEL_KIND) {
            return invalid(format!("unsupported model kind {}", python_repr(kind)));
        }
        let parameters = || -> Result<GazeModel, String> {
            let degree = match map.get("degree") {
                None => 2,
                Some(v) => json_int(v)?,
            };
            let alpha = match map.get("alpha") {
                None => 1.0,
                Some(v) => json_float(v)?,
            };
            let nonlinear = json_index_list(map.get("nonlinear"))?;
            check_degree(degree).map_err(|e| e.to_string())?;
            GazeModel::new(degree as usize, alpha, nonlinear.as_deref()).map_err(|e| e.to_string())
        };
        let mut model = parameters()
            .map_err(|exc| ModelError::Value(format!("invalid model parameters: {exc}")))?;
        if !map.contains_key("coef") {
            return Ok(model); // an unfitted model was saved
        }
        let parsed = parse_arrays(map)
            .map_err(|exc| ModelError::Value(format!("invalid model data: {exc}")))?;
        let n = parsed.mean.len();
        if n == 0 || parsed.std.len() != n {
            return invalid("model arrays have inconsistent shapes");
        }
        check_indices(model.nonlinear(), n)?;
        let terms = n_terms(n, model.degree, model.nonlinear());
        if (parsed.coef.rows, parsed.coef.cols) != (terms, 2) {
            return invalid("model arrays have inconsistent shapes");
        }
        if parsed.lo.is_some() != parsed.hi.is_some() {
            return invalid("model clip range is incomplete");
        }
        let mut arrays: Vec<&[f64]> = vec![&parsed.mean, &parsed.std, &parsed.coef.data];
        let range = match (parsed.lo, parsed.hi) {
            (Some(lo), Some(hi)) => {
                if lo.len() != n || hi.len() != n || lo.iter().zip(&hi).any(|(l, h)| h < l) {
                    return invalid("model clip range is inconsistent");
                }
                Some((lo, hi))
            }
            _ => None,
        };
        if let Some((lo, hi)) = &range {
            arrays.extend([lo.as_slice(), hi.as_slice()]);
        }
        if let Some(linear) = &parsed.linear {
            if (linear.rows, linear.cols) != (1 + n, 2) {
                return invalid("model arrays have inconsistent shapes");
            }
            arrays.push(&linear.data);
        }
        if !arrays.iter().all(|a| a.iter().all(|v| v.is_finite())) {
            return invalid("model contains non-finite numbers");
        }
        if parsed.std.iter().any(|&s| s <= 0.0) || parsed.bounds.w <= 0 || parsed.bounds.h <= 0 {
            return invalid("model scale parameters must be positive");
        }
        model.fitted = Some(Fitted {
            mean: parsed.mean,
            std: parsed.std,
            range,
            coef: parsed.coef.into_matrix(),
            linear_coef: parsed.linear.map(Table::into_matrix),
            bounds: parsed.bounds,
        });
        model.away_regions = parsed.regions;
        Ok(model)
    }
}

/// A JSON matrix as `numpy.asarray(..., dtype=float64)` sees it: `rows` x `cols`, row-major.
struct Table {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Table {
    fn into_matrix(self) -> DMatrix<f64> {
        DMatrix::from_row_slice(self.rows, self.cols, &self.data)
    }
}

struct ParsedModel {
    mean: Vec<f64>,
    std: Vec<f64>,
    coef: Table,
    bounds: Rect,
    lo: Option<Vec<f64>>,
    hi: Option<Vec<f64>>,
    linear: Option<Table>,
    regions: Vec<AwayRegion>,
}

fn parse_arrays(map: &Map<String, Value>) -> Result<ParsedModel, String> {
    let get = |key: &str| map.get(key).ok_or(format!("missing key '{key}'"));
    let mean = json_vector(get("mean")?, "mean")?;
    let std = json_vector(get("std")?, "std")?;
    let coef = json_table(get("coef")?)?;
    let bounds = json_rect(get("bounds")?)?;
    let lo = map.get("lo").map(|v| json_vector(v, "lo")).transpose()?;
    let hi = map.get("hi").map(|v| json_vector(v, "hi")).transpose()?;
    let linear = match map.get("linear_coef") {
        None | Some(Value::Null) => None,
        Some(v) => Some(json_table(v)?),
    };
    let regions = match map.get("away_regions") {
        None => Vec::new(),
        Some(v) if is_falsy(v) => Vec::new(),
        Some(Value::Array(items)) => items
            .iter()
            .map(away_region_from_json)
            .collect::<Result<_, _>>()?,
        Some(_) => return Err("away_regions must be a list".into()),
    };
    Ok(ParsedModel {
        mean,
        std,
        coef,
        bounds,
        lo,
        hi,
        linear,
        regions,
    })
}

fn is_falsy(v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::Bool(b) => !b,
        Value::Number(n) => n.as_f64() == Some(0.0),
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
    }
}

fn python_repr(v: &Value) -> String {
    match v {
        Value::String(s) => format!("'{s}'"),
        other => other.to_string(),
    }
}

/// `float(v)`: numbers only.
fn json_float(v: &Value) -> Result<f64, String> {
    v.as_f64()
        .ok_or_else(|| format!("float() argument must be a number, not {}", python_repr(v)))
}

/// `int(v)`: numbers only, floats truncate.
fn json_int(v: &Value) -> Result<i64, String> {
    if let Some(i) = v.as_i64() {
        return Ok(i);
    }
    let f = json_float(v)?;
    if !f.is_finite() || f.abs() >= 9.2e18 {
        return Err("cannot convert float to integer".into());
    }
    Ok(f.trunc() as i64)
}

fn json_rect(v: &Value) -> Result<Rect, String> {
    let list = v.as_array().ok_or("rect must be a list of 4 integers")?;
    if list.len() != 4 {
        return Err("rect must be a list of 4 integers".into());
    }
    let mut out = [0i32; 4];
    for (slot, item) in out.iter_mut().zip(list) {
        *slot = i32::try_from(json_int(item)?).map_err(|_| "rect value out of range")?;
    }
    Ok(Rect::from_list(out))
}

/// One element of a numeric array; `null` reads as NaN, as numpy converts `None`.
fn json_number(v: &Value) -> Result<f64, String> {
    match v {
        Value::Null => Ok(f64::NAN),
        other => json_float(other),
    }
}

fn json_vector(v: &Value, name: &str) -> Result<Vec<f64>, String> {
    let list = v
        .as_array()
        .ok_or_else(|| format!("{name} must be a list of numbers"))?;
    list.iter().map(json_number).collect()
}

fn json_table(v: &Value) -> Result<Table, String> {
    let list = v.as_array().ok_or("array must be a list of rows")?;
    let Some(Value::Array(first)) = list.first() else {
        // Empty, or one-dimensional: never the shape the callers expect.
        let data = list.iter().map(json_number).collect::<Result<_, _>>()?;
        return Ok(Table {
            rows: list.len(),
            cols: 0,
            data,
        });
    };
    let cols = first.len();
    let mut data = Vec::with_capacity(list.len() * cols);
    for row in list {
        let row = row
            .as_array()
            .filter(|r| r.len() == cols)
            .ok_or("array rows have different lengths")?;
        for item in row {
            data.push(json_number(item)?);
        }
    }
    Ok(Table {
        rows: list.len(),
        cols,
        data,
    })
}

/// `nonlinear` as read from JSON: absent/null or a list of integers.
fn json_index_list(v: Option<&Value>) -> Result<Option<Vec<usize>>, String> {
    let list = match v {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Array(list)) => list,
        Some(_) => return Err("nonlinear must be a list of feature indices".into()),
    };
    let mut out = Vec::with_capacity(list.len());
    for item in list {
        // Floats (even 1.0) are not integers here; as_i64 refuses them.
        let Some(i) = item.as_i64() else {
            return Err(format!("feature indices must be integers, got {item}"));
        };
        if i < 0 {
            return Err(format!("feature indices must be >= 0, got {i}"));
        }
        out.push(i as usize);
    }
    Ok(as_indices(Some(&out)))
}

fn matrix_json(m: &DMatrix<f64>) -> Value {
    Value::Array(
        m.row_iter()
            .map(|r| Value::Array(r.iter().map(|&v| json!(v)).collect()))
            .collect(),
    )
}

// ------------------------------------------------------------ model selection
/// Indices of `gaze_features` within `feature_names` (a backend's
/// `VisionBackend::feature_names` and `gaze_features`).
///
/// The result is what [`GazeModel`], [`select_model`] and `calibration::evaluate`
/// take as `nonlinear` and [`GazeModel::looks_away`] as `features`. `None` (every
/// feature is nonlinear, gaze direction unknown) when the backend declares no
/// gaze features. Names missing from `feature_names` are ignored with a warning
/// rather than raised on, so a backend declaration error degrades the model
/// instead of stopping tracking.
pub fn gaze_feature_indices<S: AsRef<str>>(
    feature_names: &[S],
    gaze_features: &[S],
) -> Option<Vec<usize>> {
    let mut indices = Vec::new();
    for name in gaze_features {
        match feature_names
            .iter()
            .position(|n| n.as_ref() == name.as_ref())
        {
            Some(i) => indices.push(i),
            None => {
                let names: Vec<&str> = feature_names.iter().map(AsRef::as_ref).collect();
                log::warn!(
                    "Gaze feature {:?} is not one of the features {:?}; ignored",
                    name.as_ref(),
                    names
                );
            }
        }
    }
    indices.sort_unstable();
    indices.dedup();
    (!indices.is_empty()).then_some(indices)
}

/// What the three model-selection functions take besides their data: `bounds`
/// (the virtual desktop; the bounding box of the targets when `None`), per-sample
/// `weights` and the `nonlinear` feature indices, as for [`GazeModel`].
#[derive(Debug, Clone, Copy, Default)]
pub struct FitOptions<'a> {
    pub bounds: Option<Rect>,
    pub weights: Option<&'a [f64]>,
    pub nonlinear: Option<&'a [usize]>,
}

/// Pick the ridge strength with the lowest leave-one-group-out mean pixel error.
///
/// `groups` holds one id per sample (the calibration point id), so every fold
/// predicts a point the model has never seen. That mirrors real use far better
/// than per-sample cross-validation, where near-duplicate frames of the same point
/// leak into the training set. Returns the middle alpha when fewer than two groups
/// exist. Near-ties go to the larger alpha (the smoother model). `options.nonlinear`
/// is passed to the candidate models (see [`GazeModel`]).
pub fn select_alpha(
    x: &DMatrix<f64>,
    y: &DMatrix<f64>,
    groups: &[i64],
    alphas: &[f64],
    degree: usize,
    options: &FitOptions,
) -> Result<f64, ModelError> {
    Ok(select_model(x, y, groups, alphas, &[degree], options)?.alpha)
}

/// Outcome of [`select_model`].
#[derive(Clone)]
pub struct ModelSelection {
    pub degree: usize,
    pub alpha: f64,
    /// Mean leave-one-group-out error of the winner (`NaN` without cross-validation).
    pub mean_error_px: f64,
    /// Leave-one-group-out predictions `(n, 2)` of the winner (`None` without CV).
    pub predictions: Option<DMatrix<f64>>,
}

impl fmt::Debug for ModelSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelSelection")
            .field("degree", &self.degree)
            .field("alpha", &self.alpha)
            .field("mean_error_px", &self.mean_error_px)
            .finish_non_exhaustive()
    }
}

/// Choose `degree` and `alpha` jointly by leave-one-group-out cross-validation.
///
/// Simpler candidates win near-ties (lower degree first, then larger alpha). With
/// fewer than two groups no cross-validation is possible and the lowest listed
/// degree (at most 2) with the middle alpha is returned. `options.nonlinear`
/// selects the features with polynomial terms, as for [`GazeModel`]; build the
/// final model with the same value.
pub fn select_model(
    x: &DMatrix<f64>,
    y: &DMatrix<f64>,
    groups: &[i64],
    alphas: &[f64],
    degrees: &[usize],
    options: &FitOptions,
) -> Result<ModelSelection, ModelError> {
    let mut alpha_list = alphas.to_vec();
    alpha_list.sort_by(|a, b| b.total_cmp(a));
    alpha_list.dedup();
    let mut degree_list = degrees.to_vec();
    degree_list.sort_unstable();
    degree_list.dedup();
    if alpha_list.is_empty() || degree_list.is_empty() {
        return invalid("alphas and degrees must not be empty");
    }
    for &a in &alpha_list {
        if !a.is_finite() || a < 0.0 {
            return invalid(format!("alpha must be a finite number >= 0, got {a}"));
        }
    }
    for &deg in &degree_list {
        check_degree(deg as i64)?;
    }

    as_matrix(x, "X")?;
    as_targets(y, x.nrows())?;
    as_groups(groups, x.nrows())?;
    let w = as_weights(options.weights, x.nrows())?;
    let indices = as_indices(options.nonlinear);
    check_indices(indices.as_deref(), x.ncols())?;
    // One normalisation for every fold, so alpha means the same in each of them
    // and in the final fit.
    let bounds = check_bounds(options.bounds, y)?;
    // The list is descending; the middle of the ascending one is index len / 2.
    let middle_alpha = alpha_list[alpha_list.len() - 1 - alpha_list.len() / 2];

    if distinct_groups(groups) < 2 {
        let degree = degree_list[0].min(2);
        log::debug!("select_model: fewer than two groups; degree={degree} alpha={middle_alpha}");
        return Ok(ModelSelection {
            degree,
            alpha: middle_alpha,
            mean_error_px: f64::NAN,
            predictions: None,
        });
    }

    let mut best: Option<ModelSelection> = None;
    for &degree in &degree_list {
        let folds = LopoFolds::new(x, y, groups, &w, degree, bounds, indices.as_deref());
        for &alpha in &alpha_list {
            let preds = folds.predict(alpha);
            let err = mean_error(&preds, y);
            log::debug!("select_model: degree={degree} alpha={alpha} -> {err:.1} px");
            if !err.is_finite() {
                continue;
            }
            if best
                .as_ref()
                .is_none_or(|b| err < b.mean_error_px * (1.0 - PREFER_SIMPLER))
            {
                best = Some(ModelSelection {
                    degree,
                    alpha,
                    mean_error_px: err,
                    predictions: Some(preds),
                });
            }
        }
    }
    Ok(best.unwrap_or_else(|| {
        log::warn!("select_model: every candidate diverged; using the defaults");
        ModelSelection {
            degree: degree_list[0].min(2),
            alpha: middle_alpha,
            mean_error_px: f64::NAN,
            predictions: None,
        }
    }))
}

/// Leave-one-group-out predictions `(n, 2)` in global pixels.
///
/// Each group is predicted by a model trained on all other groups. Rows of a
/// group whose complement carries no weight are `NaN`. Needs two groups.
pub fn lopo_predictions(
    x: &DMatrix<f64>,
    y: &DMatrix<f64>,
    groups: &[i64],
    alpha: f64,
    degree: usize,
    options: &FitOptions,
) -> Result<DMatrix<f64>, ModelError> {
    as_matrix(x, "X")?;
    as_targets(y, x.nrows())?;
    as_groups(groups, x.nrows())?;
    let w = as_weights(options.weights, x.nrows())?;
    let indices = as_indices(options.nonlinear);
    check_indices(indices.as_deref(), x.ncols())?;
    if distinct_groups(groups) < 2 {
        return invalid("leave-one-group-out needs at least two groups");
    }
    check_degree(degree as i64)?;
    let bounds = check_bounds(options.bounds, y)?;
    let folds = LopoFolds::new(x, y, groups, &w, degree, bounds, indices.as_deref());
    Ok(folds.predict(alpha))
}

/// Leave-one-group-out cross-validation that costs little more than one fit.
///
/// The normal equations of a fold are the full ones minus the held-out group's
/// contribution (`AᵀWA - A_gᵀW_gA_g`), so the design matrix is built once and
/// each (fold, alpha) pair needs only a small `p x p` solve. Standardisation
/// uses all samples; the resulting leakage of one mean/std estimate into the
/// folds is negligible compared with the per-point effects being measured.
/// Held-out rows are not clipped to the fold's feature range the way
/// [`GazeModel::predict`] would; that only matters for a held-out point far
/// outside every other point, where the estimate is then somewhat pessimistic.
struct LopoFolds {
    bounds: Rect,
    n: usize,
    folds: Vec<Fold>,
}

struct Fold {
    rows: Vec<usize>,
    lhs: DMatrix<f64>,
    rhs: DMatrix<f64>,
    held_out: DMatrix<f64>,
    trainable: bool,
}

impl LopoFolds {
    fn new(
        x: &DMatrix<f64>,
        y: &DMatrix<f64>,
        groups: &[i64],
        w: &[f64],
        degree: usize,
        bounds: Rect,
        nonlinear: Option<&[usize]>,
    ) -> LopoFolds {
        let (mean, std) = standardisation(x);
        let a = design(&standardise(x, &mean, &std), degree, nonlinear);
        let yn = normalise(y, bounds);
        let aw = scale_rows(&a, w);
        let lhs = a.transpose() * &aw;
        let rhs = aw.transpose() * &yn;
        let total_w: f64 = w.iter().sum();
        let mut ids = groups.to_vec();
        ids.sort_unstable();
        ids.dedup();
        let folds = ids
            .into_iter()
            .map(|gid| {
                let rows: Vec<usize> = (0..groups.len()).filter(|&i| groups[i] == gid).collect();
                let ag = a.select_rows(&rows);
                let wg: Vec<f64> = rows.iter().map(|&i| w[i]).collect();
                let agw = scale_rows(&ag, &wg);
                let trainable = total_w - wg.iter().sum::<f64>() > 1e-12;
                Fold {
                    lhs: &lhs - ag.transpose() * &agw,
                    rhs: &rhs - agw.transpose() * yn.select_rows(&rows),
                    held_out: ag,
                    rows,
                    trainable,
                }
            })
            .collect();
        LopoFolds {
            bounds,
            n: x.nrows(),
            folds,
        }
    }

    fn predict(&self, alpha: f64) -> DMatrix<f64> {
        let mut preds = DMatrix::from_element(self.n, 2, f64::NAN);
        for fold in self.folds.iter().filter(|f| f.trainable) {
            let out = &fold.held_out * solve_ridge(&fold.lhs, &fold.rhs, alpha);
            for (k, &row) in fold.rows.iter().enumerate() {
                preds[(row, 0)] = out[(k, 0)];
                preds[(row, 1)] = out[(k, 1)];
            }
        }
        denormalise(&preds, self.bounds)
    }
}

// --------------------------------------------------------------------- internals
/// Where the linear look-away estimate places the monitor `rect`.
///
/// `y` are the training targets, `estimates` their linear estimates and `w`
/// the sample weights. The estimates of the samples on `rect` are fitted as an
/// affine function of their targets (weighted least squares), which maps the
/// corners of `rect` to its image. The region bounds the rectangle and that
/// image, so it is never smaller than the monitor itself; without enough
/// samples to place the corners reliably it is the monitor itself.
fn away_region(rect: Rect, y: &DMatrix<f64>, estimates: &DMatrix<f64>, w: &[f64]) -> AwayRegion {
    let region = AwayRegion::of(rect);
    let on_rect: Vec<usize> = (0..y.nrows())
        .filter(|&i| {
            let (px, py) = (y[(i, 0)], y[(i, 1)]);
            px >= f64::from(rect.x)
                && px < f64::from(rect.right())
                && py >= f64::from(rect.y)
                && py < f64::from(rect.bottom())
                && w[i] > 0.0
        })
        .collect();
    let mut distinct: Vec<(f64, f64)> = on_rect.iter().map(|&i| (y[(i, 0)], y[(i, 1)])).collect();
    let (min_x, max_x) = min_max(distinct.iter().map(|p| p.0));
    let (min_y, max_y) = min_max(distinct.iter().map(|p| p.1));
    distinct.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    distinct.dedup();
    if distinct.len() < 3 {
        return region;
    }
    if max_x - min_x < REGION_MIN_SPAN * f64::from(rect.w)
        || max_y - min_y < REGION_MIN_SPAN * f64::from(rect.h)
    {
        return region;
    }
    let m = on_rect.len();
    let design = DMatrix::from_fn(m, 3, |r, c| {
        let root_w = w[on_rect[r]].sqrt();
        root_w * if c < 2 { y[(on_rect[r], c)] } else { 1.0 }
    });
    let observed = DMatrix::from_fn(m, 2, |r, c| {
        w[on_rect[r]].sqrt() * estimates[(on_rect[r], c)]
    });
    let svd = design.svd(true, true);
    let cutoff = f64::EPSILON * m.max(3) as f64 * svd.singular_values.max();
    if svd.rank(cutoff) < 3 {
        return region;
    }
    let Ok(affine) = svd.solve(&observed, cutoff) else {
        return region;
    };
    let (x0, y0, x1, y1) = (
        f64::from(rect.x),
        f64::from(rect.y),
        f64::from(rect.right()),
        f64::from(rect.bottom()),
    );
    let corners =
        DMatrix::from_row_slice(4, 3, &[x0, y0, 1.0, x1, y0, 1.0, x1, y1, 1.0, x0, y1, 1.0]);
    let image = corners * affine;
    if !image.iter().all(|v| v.is_finite()) {
        return region;
    }
    let dist =
        |a: usize, b: usize| (image[(a, 0)] - image[(b, 0)]).hypot(image[(a, 1)] - image[(b, 1)]);
    let diagonal = dist(2, 0).max(dist(3, 1));
    if diagonal > REGION_MAX_SCALE * rect.diagonal() {
        log::debug!("Ignoring an implausible look-away region for {rect:?}");
        return region;
    }
    let (ix0, ix1) = min_max(image.column(0).iter().copied());
    let (iy0, iy1) = min_max(image.column(1).iter().copied());
    AwayRegion {
        rect,
        x0: x0.min(ix0),
        y0: y0.min(iy0),
        x1: x1.max(ix1),
        y1: y1.max(iy1),
        diagonal: rect.diagonal().max(diagonal),
    }
}

fn min_max(values: impl Iterator<Item = f64>) -> (f64, f64) {
    values.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
        (lo.min(v), hi.max(v))
    })
}

fn n_terms(n_features: usize, degree: usize, nonlinear: Option<&[usize]>) -> usize {
    let k = nonlinear.map_or(n_features, <[usize]>::len);
    let mut n = 1 + n_features;
    if degree >= 2 {
        n += k * (k + 1) / 2;
    }
    if degree >= 3 {
        n += k;
    }
    n
}

/// A column of the design matrix, in the order the coefficients are stored: the
/// intercept, the features, the products `z_j*z_k (j <= k)` of the nonlinear
/// ones (row-major upper triangle), then their cubes.
#[derive(Clone, Copy)]
enum Term {
    One,
    Linear(usize),
    Product(usize, usize),
    Cube(usize),
}

fn design(z: &DMatrix<f64>, degree: usize, nonlinear: Option<&[usize]>) -> DMatrix<f64> {
    let d = z.ncols();
    let mut terms = vec![Term::One];
    terms.extend((0..d).map(Term::Linear));
    if degree >= 2 {
        let all: Vec<usize>;
        let picked = match nonlinear {
            Some(v) => v,
            None => {
                all = (0..d).collect();
                &all
            }
        };
        for (a, &j) in picked.iter().enumerate() {
            terms.extend(picked[a..].iter().map(|&k| Term::Product(j, k)));
        }
        if degree >= 3 {
            terms.extend(picked.iter().map(|&j| Term::Cube(j)));
        }
    }
    DMatrix::from_fn(z.nrows(), terms.len(), |r, c| match terms[c] {
        Term::One => 1.0,
        Term::Linear(j) => z[(r, j)],
        Term::Product(j, k) => z[(r, j)] * z[(r, k)],
        Term::Cube(j) => z[(r, j)].powi(3),
    })
}

fn standardisation(x: &DMatrix<f64>) -> (Vec<f64>, Vec<f64>) {
    let n = x.nrows() as f64;
    let mean: Vec<f64> = x.column_iter().map(|c| c.sum() / n).collect();
    let std = x
        .column_iter()
        .zip(&mean)
        .map(|(c, m)| {
            let var = c.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n;
            var.sqrt().max(STD_FLOOR)
        })
        .collect();
    (mean, std)
}

fn standardise(x: &DMatrix<f64>, mean: &[f64], std: &[f64]) -> DMatrix<f64> {
    DMatrix::from_fn(x.nrows(), x.ncols(), |r, c| (x[(r, c)] - mean[c]) / std[c])
}

fn column_extreme(x: &DMatrix<f64>, pick: fn(f64, f64) -> f64) -> Vec<f64> {
    x.column_iter()
        .map(|c| c.iter().copied().reduce(pick).unwrap_or(f64::NAN))
        .collect()
}

/// `m` with row `i` multiplied by `w[i]`.
fn scale_rows(m: &DMatrix<f64>, w: &[f64]) -> DMatrix<f64> {
    DMatrix::from_fn(m.nrows(), m.ncols(), |r, c| m[(r, c)] * w[r])
}

/// Solve `(lhs + αI')β = rhs` where `I'` is the identity without the intercept.
fn solve_ridge(lhs: &DMatrix<f64>, rhs: &DMatrix<f64>, alpha: f64) -> DMatrix<f64> {
    let mut system = lhs.clone();
    for i in 1..system.nrows() {
        system[(i, i)] += alpha;
    }
    if let Some(coef) = system.clone().lu().solve(rhs)
        && coef.iter().all(|v| v.is_finite())
    {
        return coef;
    }
    // Singular system (alpha == 0 with collinear or too few samples): the
    // minimum-norm solution is the most conservative choice.
    log::debug!("ridge system singular; falling back to least squares");
    let svd = system.svd(true, true);
    let cutoff =
        f64::EPSILON * svd.u.as_ref().map_or(1, |u| u.nrows()) as f64 * svd.singular_values.max();
    svd.solve(rhs, cutoff)
        .unwrap_or_else(|_| DMatrix::zeros(rhs.nrows(), rhs.ncols()))
}

/// Mean of the finite-or-not but non-`NaN` errors; `NaN` when every row is `NaN`.
fn mean_error(preds: &DMatrix<f64>, y: &DMatrix<f64>) -> f64 {
    let errors: Vec<f64> = (0..y.nrows())
        .map(|i| (preds[(i, 0)] - y[(i, 0)]).hypot(preds[(i, 1)] - y[(i, 1)]))
        .filter(|e| !e.is_nan())
        .collect();
    if errors.is_empty() {
        return f64::NAN;
    }
    errors.iter().sum::<f64>() / errors.len() as f64
}

fn normalise(y: &DMatrix<f64>, b: Rect) -> DMatrix<f64> {
    DMatrix::from_fn(y.nrows(), 2, |r, c| {
        if c == 0 {
            (y[(r, 0)] - f64::from(b.x)) / f64::from(b.w)
        } else {
            (y[(r, 1)] - f64::from(b.y)) / f64::from(b.h)
        }
    })
}

fn denormalise(yn: &DMatrix<f64>, b: Rect) -> DMatrix<f64> {
    DMatrix::from_fn(yn.nrows(), 2, |r, c| {
        if c == 0 {
            yn[(r, 0)] * f64::from(b.w) + f64::from(b.x)
        } else {
            yn[(r, 1)] * f64::from(b.h) + f64::from(b.y)
        }
    })
}

/// The clip range of a feature: its calibrated range widened by [`CLIP_MARGIN`].
fn widened(lo: f64, hi: f64) -> (f64, f64) {
    let span = hi - lo;
    (lo - CLIP_MARGIN * span, hi + CLIP_MARGIN * span)
}

/// `numpy.clip`: NaN stays NaN.
fn clip(v: f64, low: f64, high: f64) -> f64 {
    if v < low {
        low
    } else if v > high {
        high
    } else {
        v
    }
}

/// `numpy.maximum`: NaN wins, where `f64::max` would ignore it.
fn nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

fn check_degree(degree: i64) -> Result<(), ModelError> {
    if !SUPPORTED_DEGREES.iter().any(|&d| d as i64 == degree) {
        return invalid(format!(
            "degree must be one of {SUPPORTED_DEGREES:?}, got {degree}"
        ));
    }
    Ok(())
}

fn check_bounds(bounds: Option<Rect>, y: &DMatrix<f64>) -> Result<Rect, ModelError> {
    match bounds {
        None => {
            let (x0, x1) = min_max(y.column(0).iter().copied());
            let (y0, y1) = min_max(y.column(1).iter().copied());
            let (x0, y0) = (x0.floor(), y0.floor());
            Ok(Rect::new(
                x0 as i32,
                y0 as i32,
                ((x1.ceil() - x0) as i32).max(1),
                ((y1.ceil() - y0) as i32).max(1),
            ))
        }
        Some(b) if b.w <= 0 || b.h <= 0 => {
            invalid(format!("bounds must have a positive size, got {b:?}"))
        }
        Some(b) => Ok(b),
    }
}

fn as_matrix(x: &DMatrix<f64>, name: &str) -> Result<(), ModelError> {
    if x.nrows() == 0 || x.ncols() == 0 {
        return invalid(format!(
            "{name} must be a non-empty (n, d) array, got shape ({}, {})",
            x.nrows(),
            x.ncols()
        ));
    }
    if !x.iter().all(|v| v.is_finite()) {
        return invalid(format!("{name} contains non-finite values"));
    }
    Ok(())
}

fn as_targets(y: &DMatrix<f64>, n: usize) -> Result<(), ModelError> {
    if (y.nrows(), y.ncols()) != (n, 2) {
        return invalid(format!(
            "Y must have shape ({n}, 2), got ({}, {})",
            y.nrows(),
            y.ncols()
        ));
    }
    if !y.iter().all(|v| v.is_finite()) {
        return invalid("Y contains non-finite values");
    }
    Ok(())
}

fn as_weights(weights: Option<&[f64]>, n: usize) -> Result<Vec<f64>, ModelError> {
    let Some(w) = weights else {
        return Ok(vec![1.0; n]);
    };
    if w.len() != n {
        return invalid(format!(
            "weights must have shape ({n},), got ({},)",
            w.len()
        ));
    }
    if !w.iter().all(|v| v.is_finite()) || w.iter().any(|&v| v < 0.0) || !w.iter().any(|&v| v > 0.0)
    {
        return invalid("weights must be finite, non-negative and not all zero");
    }
    Ok(w.to_vec())
}

fn as_groups(groups: &[i64], n: usize) -> Result<(), ModelError> {
    if groups.len() != n {
        return invalid(format!(
            "groups must have shape ({n},), got ({},)",
            groups.len()
        ));
    }
    Ok(())
}

fn distinct_groups(groups: &[i64]) -> usize {
    let mut ids = groups.to_vec();
    ids.sort_unstable();
    ids.dedup();
    ids.len()
}

/// Sorted, de-duplicated feature indices (`None` passes through).
fn as_indices(indices: Option<&[usize]>) -> Option<Vec<usize>> {
    indices.map(|i| {
        let mut out = i.to_vec();
        out.sort_unstable();
        out.dedup();
        out
    })
}

fn check_indices(indices: Option<&[usize]>, n_features: usize) -> Result<(), ModelError> {
    // `indices` may arrive unsorted from a caller; the largest decides.
    if let Some(&last) = indices.and_then(|i| i.iter().max())
        && last >= n_features
    {
        return invalid(format!(
            "feature index {last} is out of range for {n_features} features"
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
