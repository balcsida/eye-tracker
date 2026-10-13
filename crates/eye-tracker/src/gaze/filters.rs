//! Smoothing for the gaze point.
//!
//! Gaze estimates jitter by tens of pixels from frame to frame. A fixed low-pass
//! filter would either leave that jitter visible or make monitor switches feel
//! sluggish. The One Euro filter (Casiez, Roussel & Vogel, CHI 2012) adapts its cut-off
//! to the speed of the signal: heavy smoothing while the gaze rests, almost none
//! while it jumps to another monitor. It is time-aware, which matters here because
//! the frame rate changes all the time (adaptive rate, motion gate).

use std::f64::consts::PI;

use thiserror::Error;

// PointFilter tuning; both parameters are interpolated by `smoothing` (0..1).
// beta is in pixel units: at 0.004 a 2000 px/s gaze jump raises the cut-off by
// 8 Hz. Frame-to-frame jitter of a webcam gaze estimate also "moves" at roughly
// 1000 px/s, so a large beta would keep the cut-off high while the gaze rests and
// heavy smoothing would barely smooth. Heavy smoothing therefore uses a small
// beta; a monitor jump (~20000 px/s at 12 fps) still opens the filter within a frame.
const MIN_CUTOFF_RAW: f64 = 8.0;
const MIN_CUTOFF_SMOOTH: f64 = 0.3;
const BETA_RAW: f64 = 0.004;
const BETA_SMOOTH: f64 = 0.0005;
const D_CUTOFF: f64 = 1.0;

/// Returned by [`OneEuroFilter::new`] for parameters outside their range.
#[derive(Debug, Error, PartialEq, Eq)]
#[error("min_cutoff and d_cutoff must be > 0 and beta >= 0")]
pub struct InvalidFilterParameters;

/// One-dimensional One Euro filter.
///
/// `min_cutoff` (Hz) sets the smoothing of a slow signal, `beta` how quickly
/// the cut-off rises with speed and `d_cutoff` (Hz) the smoothing of the speed
/// estimate itself. Timestamps are seconds; a sample whose timestamp does not
/// advance is ignored and the previous output is returned.
#[derive(Debug, Clone)]
pub struct OneEuroFilter {
    pub min_cutoff: f64,
    pub beta: f64,
    pub d_cutoff: f64,
    x: Option<f64>,
    dx: f64,
    t: Option<f64>,
}

impl Default for OneEuroFilter {
    /// `min_cutoff=1.0, beta=0.0, d_cutoff=1.0`.
    fn default() -> Self {
        Self::unchecked(1.0, 0.0, 1.0)
    }
}

impl OneEuroFilter {
    pub fn new(min_cutoff: f64, beta: f64, d_cutoff: f64) -> Result<Self, InvalidFilterParameters> {
        if min_cutoff <= 0.0 || d_cutoff <= 0.0 || beta < 0.0 {
            return Err(InvalidFilterParameters);
        }
        Ok(Self::unchecked(min_cutoff, beta, d_cutoff))
    }

    fn unchecked(min_cutoff: f64, beta: f64, d_cutoff: f64) -> Self {
        Self {
            min_cutoff,
            beta,
            d_cutoff,
            x: None,
            dx: 0.0,
            t: None,
        }
    }

    /// Last filtered value (`None` before the first sample or after a reset).
    pub fn value(&self) -> Option<f64> {
        self.x
    }

    pub fn reset(&mut self) {
        self.x = None;
        self.dx = 0.0;
        self.t = None;
    }

    /// Filter the sample `x` taken at time `t` (seconds) and return the smoothed value.
    pub fn update(&mut self, x: f64, t: f64) -> f64 {
        if !x.is_finite() {
            // A NaN would poison the state forever; hold the last value instead.
            return self.x.unwrap_or(x);
        }
        let (Some(prev), Some(prev_t)) = (self.x, self.t) else {
            self.x = Some(x);
            self.dx = 0.0;
            self.t = Some(t);
            return x;
        };
        let dt = t - prev_t;
        if dt <= 0.0 {
            return prev;
        }
        let dx = (x - prev) / dt;
        let edx = self.dx + smoothing_factor(self.d_cutoff, dt) * (dx - self.dx);
        let cutoff = self.min_cutoff + self.beta * edx.abs();
        let filtered = prev + smoothing_factor(cutoff, dt) * (x - prev);
        self.x = Some(filtered);
        self.dx = edx;
        self.t = Some(t);
        filtered
    }
}

/// Two-dimensional One Euro filter for gaze points in pixels.
///
/// `smoothing` 0..1 maps linearly to `min_cutoff` from 8 Hz (light) to 0.3 Hz
/// (heavy) and to `beta` from 0.004 to 0.0005. 0 returns the raw input
/// unchanged. The filters keep running at smoothing 0 so that turning smoothing
/// back on continues from the current gaze position.
#[derive(Debug, Clone)]
pub struct PointFilter {
    fx: OneEuroFilter,
    fy: OneEuroFilter,
    smoothing: f64,
}

impl PointFilter {
    pub fn new(smoothing: f64) -> Self {
        let mut filter = Self {
            fx: OneEuroFilter::unchecked(MIN_CUTOFF_RAW, BETA_RAW, D_CUTOFF),
            fy: OneEuroFilter::unchecked(MIN_CUTOFF_RAW, BETA_RAW, D_CUTOFF),
            smoothing: 0.0,
        };
        filter.set_smoothing(smoothing);
        filter
    }

    pub fn smoothing(&self) -> f64 {
        self.smoothing
    }

    /// Change the strength (clamped to 0..1) without resetting the state.
    pub fn set_smoothing(&mut self, smoothing: f64) {
        let s = if smoothing.is_finite() {
            smoothing.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.smoothing = s;
        let min_cutoff = MIN_CUTOFF_RAW + (MIN_CUTOFF_SMOOTH - MIN_CUTOFF_RAW) * s;
        let beta = BETA_RAW + (BETA_SMOOTH - BETA_RAW) * s;
        for f in [&mut self.fx, &mut self.fy] {
            f.min_cutoff = min_cutoff;
            f.beta = beta;
        }
    }

    pub fn update(&mut self, x: f64, y: f64, t: f64) -> (f64, f64) {
        let (fx, fy) = (self.fx.update(x, t), self.fy.update(y, t));
        if self.smoothing <= 0.0 {
            return (x, y);
        }
        (fx, fy)
    }

    pub fn reset(&mut self) {
        self.fx.reset();
        self.fy.reset();
    }
}

fn smoothing_factor(cutoff: f64, dt: f64) -> f64 {
    let tau = 1.0 / (2.0 * PI * cutoff);
    1.0 / (1.0 + tau / dt)
}

#[cfg(test)]
#[path = "filters_tests.rs"]
mod tests;
