//! Motion gate: skip face analysis while the camera picture is unchanged.
//!
//! Someone reading or typing sits almost perfectly still for long stretches. The
//! gate compares a tiny greyscale thumbnail of each new frame with the thumbnail of
//! the last *analysed* frame and only lets the frame through when enough has
//! changed, which removes most inference work at a cost of ~0.2 ms per frame.
//!
//! A whole-frame thumbnail cannot see a pure eye movement: at 32x24 pixels the
//! irises are a fraction of a pixel. When the caller passes the face box of the
//! analysed frame, the gate therefore also watches a small, higher-resolution
//! thumbnail of the eye band, so a glance at another monitor is noticed even when
//! the head does not move. The eyes cover only a few percent of that band, so the
//! band is scored by its most-changed cells (the top decile), not by its mean,
//! which would dilute an iris movement below the camera noise.
//!
//! The module also decides whether a frame is *blind*: so dark and featureless
//! (lens covered, privacy shutter closed, unlit room) that "no face" means "cannot
//! tell" rather than "nobody here".

use super::{Frame, Grey, Plane};

/// Normalised `(x, y, w, h)` box.
pub type NormBox = (f64, f64, f64, f64);

// Eye band inside a face box (fractions of the box). Generous enough for the
// face-mesh landmark box (eyes at ~33 % of its height) and the YuNet detection
// box (eyes at ~41 %).
const BAND_X0: f64 = 0.05;
const BAND_X1: f64 = 0.95;
const BAND_Y0: f64 = 0.15;
const BAND_Y1: f64 = 0.60;
const ROI_SIZE: (usize, usize) = (32, 16);
const MIN_ROI_PX: usize = 8;
/// Share of eye-band cells whose mean change scores the band.
pub const BAND_TOP_FRACTION: f64 = 0.10;
/// The eye band's score must exceed this multiple of the whole-frame threshold.
/// Its top-decile statistic reads a few grey levels even for a still picture
/// with camera noise, while an eye movement scores well above ten.
pub const BAND_THRESHOLD_FACTOR: f64 = 3.0;

/// Whole-frame thumbnail size used by the gate and the blindness test.
pub const THUMB_SIZE: (usize, usize) = (32, 24);
/// A thumbnail darker than this mean grey level (0-255) ...
pub const BLIND_MAX_MEAN: f64 = 32.0;
/// ... and flatter than this standard deviation shows nothing recognisable.
pub const BLIND_MAX_STD: f64 = 5.0;

fn thumb(gray: &Grey, size: (usize, usize)) -> Plane {
    gray.resize_area(size.0, size.1).to_plane()
}

/// Greyscale `f32` thumbnail of a BGR frame.
pub fn thumbnail(frame: &Frame, size: (usize, usize)) -> Plane {
    thumb(&frame.to_grey(), size)
}

/// True when a thumbnail is too dark and uniform to show a face.
///
/// Covered lenses and closed privacy shutters give near-black, featureless
/// pictures; any real scene with a person in it has far more contrast.
pub fn is_blind(thumb: &Plane) -> bool {
    let (mean, std) = thumb.mean_std();
    mean < BLIND_MAX_MEAN && std < BLIND_MAX_STD
}

fn eye_band(face_box: NormBox) -> NormBox {
    let (x, y, w, h) = face_box;
    (
        x + BAND_X0 * w,
        y + BAND_Y0 * h,
        (BAND_X1 - BAND_X0) * w,
        (BAND_Y1 - BAND_Y0) * h,
    )
}

fn roi_thumb(gray: &Grey, band: NormBox) -> Option<Plane> {
    let (width, height) = (gray.width as f64, gray.height as f64);
    let (bx, by, bw, bh) = band;
    let x0 = (bx * width).max(0.0) as usize;
    let y0 = (by * height).max(0.0) as usize;
    let x1 = (((bx + bw) * width).round_ties_even().max(0.0) as usize).min(gray.width);
    let y1 = (((by + bh) * height).round_ties_even().max(0.0) as usize).min(gray.height);
    if x1.saturating_sub(x0) < MIN_ROI_PX || y1.saturating_sub(y0) < MIN_ROI_PX / 2 {
        return None;
    }
    Some(thumb(&gray.crop(x0, y0, x1, y1), ROI_SIZE))
}

fn mean_abs_diff(a: &Plane, b: &Plane) -> f64 {
    let sum: f64 = a
        .data
        .iter()
        .zip(&b.data)
        .map(|(x, y)| f64::from((x - y).abs()))
        .sum();
    sum / a.data.len() as f64
}

/// Mean of the largest `fraction` of absolute differences.
///
/// Robust to where in the band the change happens and insensitive to the size
/// of the band, unlike the plain mean.
fn top_abs_diff(a: &Plane, b: &Plane, fraction: f64) -> f64 {
    let mut diff = a.abs_diff(b).data;
    let k = ((diff.len() as f64 * fraction) as usize).max(1);
    diff.sort_by(|x, y| y.total_cmp(x));
    diff[..k].iter().map(|&v| f64::from(v)).sum::<f64>() / k as f64
}

/// Decides whether a frame differs enough from the last analysed one.
///
/// Not thread-safe; used by the vision worker thread only.
pub struct MotionGate {
    /// Mean absolute grey-level difference (0-255 scale) of the whole-frame thumbnail that
    /// counts as motion. Camera noise after downscaling is well below 1. The eye band uses
    /// [`BAND_THRESHOLD_FACTOR`] times this value for its top-decile score.
    pub threshold: f64,
    /// Whole-frame thumbnail size `(width, height)`.
    pub size: (usize, usize),
    /// Always analyse at least this often, so that presence, blink and face-count information
    /// never goes stale.
    pub max_skip_s: f64,
    reference: Option<Plane>,
    ref_roi: Option<Plane>,
    band: Option<NormBox>,
    last_processed: Option<f64>,
    last_motion: f64,
}

impl Default for MotionGate {
    fn default() -> Self {
        Self::new(2.0, THUMB_SIZE, 2.0)
    }
}

impl MotionGate {
    pub fn new(threshold: f64, size: (usize, usize), max_skip_s: f64) -> Self {
        Self {
            threshold,
            size,
            max_skip_s,
            reference: None,
            ref_roi: None,
            band: None,
            last_processed: None,
            last_motion: 0.0,
        }
    }

    /// Motion measured by the latest [`should_process`](Self::should_process) call
    /// (diagnostics).
    ///
    /// The larger of the whole-frame difference and the eye-band score divided
    /// by [`BAND_THRESHOLD_FACTOR`], so it compares directly with `threshold`.
    pub fn last_motion(&self) -> f64 {
        self.last_motion
    }

    /// Thumbnail of the last analysed frame, if any.
    pub fn reference(&self) -> Option<&Plane> {
        self.reference.as_ref()
    }

    /// True if `frame_bgr` should be analysed.
    ///
    /// True when nothing has been analysed yet, when `max_skip_s` has passed
    /// since the last analysed frame, or when the whole frame or the eye band
    /// changed enough (see `threshold`).
    pub fn should_process(&mut self, frame_bgr: &Frame, now: f64) -> bool {
        let (Some(reference), Some(last_processed)) = (&self.reference, self.last_processed) else {
            return true;
        };
        if now - last_processed >= self.max_skip_s {
            return true;
        }
        let gray = frame_bgr.to_grey();
        let mut motion = mean_abs_diff(&thumb(&gray, self.size), reference);
        if let (Some(band), Some(ref_roi)) = (self.band, &self.ref_roi)
            && let Some(roi) = roi_thumb(&gray, band)
        {
            let score = top_abs_diff(&roi, ref_roi, BAND_TOP_FRACTION);
            motion = motion.max(score / BAND_THRESHOLD_FACTOR);
        }
        self.last_motion = motion;
        motion >= self.threshold
    }

    /// Remember `frame_bgr` as the last analysed frame.
    ///
    /// `face_box` is the normalised `(x, y, w, h)` face box found in this
    /// frame (`Observation.face_box`); it enables eye-band motion detection.
    pub fn mark_processed(&mut self, frame_bgr: &Frame, now: f64, face_box: Option<NormBox>) {
        let gray = frame_bgr.to_grey();
        self.reference = Some(thumb(&gray, self.size));
        self.last_processed = Some(now);
        self.band = None;
        self.ref_roi = None;
        if let Some(face_box) = face_box {
            let band = eye_band(face_box);
            if let Some(roi) = roi_thumb(&gray, band) {
                self.band = Some(band);
                self.ref_roi = Some(roi);
            }
        }
    }

    /// Forget the reference frame; the next frame is always analysed.
    pub fn reset(&mut self) {
        self.reference = None;
        self.ref_roi = None;
        self.band = None;
        self.last_processed = None;
        self.last_motion = 0.0;
    }
}

#[cfg(test)]
#[path = "motion_tests.rs"]
mod tests;
