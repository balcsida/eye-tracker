//! Gaze estimation: from vision features to a point on the virtual desktop.
//!
//! Pure numerics (no UI, no camera), so everything here is deterministic and unit-testable.
//!
//! * [`model`]: ridge regression with polynomial terms, cross-validated model selection.
//! * [`filters`]: One Euro smoothing of the gaze point.
//! * [`calibration`]: target plan, sample collection state machine, quality report.
//! * [`store`]: calibration file format (numbers only, never images).
//! * [`learning`]: refinement from natural mouse use and drift detection.

pub mod calibration;
pub mod filters;
pub mod learning;
pub mod model;
pub mod store;
