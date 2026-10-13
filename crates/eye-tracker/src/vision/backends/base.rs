//! Vision backend contract.
//!
//! A backend turns one BGR camera frame into an [`Observation`]. Backends are used from a single
//! worker thread and need not be thread-safe.

use thiserror::Error;

use crate::types::Observation;
use crate::vision::Frame;

/// Raised by a backend constructor when its runtime or model is missing.
#[derive(Debug, Error)]
#[error("{0}")]
pub struct BackendUnavailable(pub String);

pub trait VisionBackend {
    /// Short identifier stored with calibrations ("facemesh", "lite").
    fn name(&self) -> &'static str;

    /// Human-readable names of the entries of `Observation::features`.
    fn feature_names(&self) -> &'static [&'static str];

    /// Changes whenever feature semantics change; calibrations with another version are
    /// rejected.
    fn feature_version(&self) -> &'static str;

    /// The subset of `feature_names` that encodes where the user looks (head rotation and eye
    /// direction, as opposed to head position or roll). The gaze model applies nonlinear terms
    /// only to these, and judges looking away from every monitor from the combined direction
    /// they give (`GazeModel::looks_away`).
    fn gaze_features(&self) -> &'static [&'static str] {
        &[]
    }

    /// Analyse one frame.
    ///
    /// `timestamp` is monotonic seconds and strictly increases between calls. The returned
    /// observation carries the same timestamp. Must not fail for frames without faces
    /// (`face_count == 0`).
    fn process(&mut self, frame_bgr: &Frame, timestamp: f64) -> Observation;

    /// Maximum number of faces to detect (2 enables the shoulder guard).
    fn set_max_faces(&mut self, _n: usize) {}

    /// Whether the last observation is final for an unchanged picture.
    ///
    /// A tracking backend returns `false` when analysing the very same picture again would
    /// still give noticeably different features (its search window was still catching up with a
    /// moved face). The vision worker then analyses the next frame instead of letting the motion
    /// gate repeat the observation.
    fn settled(&self) -> bool {
        true
    }

    /// Forget what was tracked in earlier frames; the next frame starts afresh.
    ///
    /// Called when the camera was released or replaced: the next frame may show a different
    /// scene, so nothing may be carried over from before.
    fn reset(&mut self) {}

    /// A copy of the frame with landmarks drawn (camera preview only).
    fn annotate(&self, frame_bgr: &Frame, _observation: &Observation) -> Frame {
        frame_bgr.clone()
    }

    /// Release native resources. Safe to call more than once.
    fn close(&mut self) {}
}
