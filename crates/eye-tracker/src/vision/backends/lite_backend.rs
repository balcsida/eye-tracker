//! Lite backend: YuNet face detection and five-point head geometry.
//!
//! This backend needs nothing beyond a 230 kB model and runs in a few milliseconds at 320 px.
//! It has no iris landmarks, so it relies on head movement alone: the features describe where
//! the nose points relative to the eyes and mouth, plus where the face is in the frame. Blinks
//! cannot be detected.

use std::path::Path;
use std::time::Instant;

use imageproc::drawing::draw_filled_circle_mut;

use super::base::{BackendUnavailable, VisionBackend};
use super::overlay::{
    COLOR_BOX, COLOR_OTHER, COLOR_POINT, draw_box, draw_label, line_thickness, observation_label,
};
use super::yunet::{Face, YuNet, resize_area};
use crate::types::Observation;
use crate::vision::Frame;
use crate::vision::threads::{MAX_OPENCV_THREADS, limit_opencv_threads};

pub const MODEL_FILE: &str = "face_detection_yunet_2023mar.onnx";
/// Frames are downscaled to at most this width before detection.
pub const MAX_INPUT_WIDTH: usize = 320;
/// Detections below this confidence are ignored (and not counted as faces).
pub const SCORE_THRESHOLD: f32 = 0.6;
pub const NMS_THRESHOLD: f32 = 0.3;
pub const TOP_K: usize = 50;
/// For a frontal face the nose tip sits about this far from the eye line towards the mouth line
/// (fraction of the eye-to-mouth distance).
pub const NOSE_BASELINE: f64 = 0.55;
/// Gains turning the nose offsets into rough angles for display. Only the raw offsets are used
/// for gaze estimation, so these need not be exact.
pub const YAW_GAIN: f64 = 1.6;
pub const PITCH_GAIN: f64 = 2.5;
pub const BORDER_MARGIN: f64 = 0.01;

/// Create a YuNet detector for frames about `input_width` pixels wide.
///
/// `model_path` replaces the embedded model (tests). Fails with [`BackendUnavailable`] when the
/// model cannot be read or loaded.
pub fn create_yunet(
    model_path: Option<&Path>,
    input_width: usize,
) -> Result<YuNet, BackendUnavailable> {
    let owned;
    let model = match model_path {
        Some(path) => {
            owned = std::fs::read(path).map_err(|_| {
                BackendUnavailable(format!(
                    "YuNet model not found at {} (run cargo xtask fetch-models)",
                    path.display()
                ))
            })?;
            &owned[..]
        }
        None => super::yunet_model(),
    };
    let size = (
        input_width,
        ((input_width as f64 * 3.0 / 4.0).round_ties_even() as usize).max(1),
    );
    let mut detector = YuNet::from_bytes(model, size)
        .map_err(|err| BackendUnavailable(format!("could not load the YuNet model: {err}")))?;
    detector.score_threshold = SCORE_THRESHOLD;
    detector.nms_threshold = NMS_THRESHOLD;
    detector.top_k = TOP_K;
    Ok(detector)
}

/// Features and rough `(yaw, pitch)` in degrees for one YuNet detection row.
///
/// `face` is `[x, y, w, h, re_x, re_y, le_x, le_y, nose_x, nose_y, rm_x, rm_y, lm_x, lm_y,
/// score]` in pixels of a `frame_width` x `frame_height` image. YuNet's "right eye" is the one on
/// the image left, so the eye axis points to image-right for any upright face. Returns `None` for
/// degenerate geometry.
pub fn geometry_features(
    face: &Face,
    frame_width: f64,
    frame_height: f64,
) -> Option<(Vec<f64>, f64, f64)> {
    let v = |i: usize| f64::from(face[i]);
    let (x, y, w, h) = (v(0), v(1), v(2), v(3));
    let right_eye = [v(4), v(5)];
    let left_eye = [v(6), v(7)];
    let nose = [v(8), v(9)];
    let mouth_mid = [(v(10) + v(12)) / 2.0, (v(11) + v(13)) / 2.0];

    let eye_mid = [
        (right_eye[0] + left_eye[0]) / 2.0,
        (right_eye[1] + left_eye[1]) / 2.0,
    ];
    let axis = [left_eye[0] - right_eye[0], left_eye[1] - right_eye[1]];
    let ipd = axis[0].hypot(axis[1]);
    if ipd.is_nan() || ipd < 1.0 {
        return None;
    }
    let u = [axis[0] / ipd, axis[1] / ipd];
    let down = [-u[1], u[0]]; // perpendicular pointing to the chin
    let dot = |a: [f64; 2], b: [f64; 2]| a[0] * b[0] + a[1] * b[1];
    let mouth_depth = dot([mouth_mid[0] - eye_mid[0], mouth_mid[1] - eye_mid[1]], down);
    if mouth_depth.is_nan() || mouth_depth < 1.0 {
        return None;
    }

    let nose_rel = [nose[0] - eye_mid[0], nose[1] - eye_mid[1]];
    // Horizontal offset in eye distances: grows as the head turns to image-right.
    let nose_dx = dot(nose_rel, u) / ipd;
    // Vertical position between the eye line (0) and mouth line (1), relative to a frontal face.
    // Normalising by the eye-to-mouth distance keeps it independent of how far away the user
    // sits and of head yaw (which shrinks the eye distance).
    let nose_dy = dot(nose_rel, down) / mouth_depth - NOSE_BASELINE;
    // Counter-clockwise positive, matching the sign of the facemesh backend's roll.
    let roll = (-axis[1]).atan2(axis[0]).to_degrees();
    let face_cx = (x + w / 2.0) / frame_width;
    let face_cy = (y + h / 2.0) / frame_height;
    let scale = ipd / frame_width;

    let features = vec![nose_dx, nose_dy, roll, face_cx, face_cy, scale];
    if !features.iter().all(|f| f.is_finite()) {
        return None;
    }
    let yaw = (nose_dx * YAW_GAIN).clamp(-1.0, 1.0).asin().to_degrees();
    let pitch = (nose_dy * PITCH_GAIN).clamp(-1.0, 1.0).asin().to_degrees();
    Some((features, yaw, pitch))
}

fn clipped_box(face: &Face, width: f64, height: f64) -> (f64, f64, f64, f64) {
    let v = |i: usize| f64::from(face[i]);
    let (x, y, w, h) = (v(0), v(1), v(2), v(3));
    let (x0, y0) = ((x / width).max(0.0), (y / height).max(0.0));
    let (x1, y1) = (((x + w) / width).min(1.0), ((y + h) / height).min(1.0));
    (x0, y0, (x1 - x0).max(0.0), (y1 - y0).max(0.0))
}

/// Landmarks of the last processed frame, for `annotate`.
struct Detections {
    timestamp: f64,
    /// Normalised eyes, nose and mouth corners.
    points: [[f64; 2]; 5],
    /// Boxes of the other faces.
    others: Vec<(f64, f64, f64, f64)>,
}

/// Head-geometry features from the YuNet face detector.
///
/// `max_faces` is accepted for interface compatibility; YuNet always finds every face, and
/// `face_count` reports all of them.
pub struct LiteBackend {
    detector: Option<YuNet>,
    input_size: Option<(usize, usize)>,
    max_faces: usize,
    last: Option<Detections>,
}

impl LiteBackend {
    pub const NAME: &'static str = "lite";
    pub const FEATURE_NAMES: &'static [&'static str] =
        &["nose_dx", "nose_dy", "roll", "face_cx", "face_cy", "scale"];
    pub const FEATURE_VERSION: &'static str = "yunet-geom-1";
    pub const GAZE_FEATURES: &'static [&'static str] = &["nose_dx", "nose_dy"];

    /// The backend with the embedded model.
    pub fn new(max_faces: usize) -> Result<Self, BackendUnavailable> {
        Self::with_model(None, max_faces)
    }

    /// The backend with the YuNet model at `model_path` (the embedded one for `None`).
    pub fn with_model(
        model_path: Option<&Path>,
        max_faces: usize,
    ) -> Result<Self, BackendUnavailable> {
        limit_opencv_threads(MAX_OPENCV_THREADS);
        Ok(Self {
            detector: Some(create_yunet(model_path, MAX_INPUT_WIDTH)?),
            input_size: None,
            max_faces: max_faces.max(1),
            last: None,
        })
    }
}

impl VisionBackend for LiteBackend {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn feature_names(&self) -> &'static [&'static str] {
        Self::FEATURE_NAMES
    }

    fn feature_version(&self) -> &'static str {
        Self::FEATURE_VERSION
    }

    fn gaze_features(&self) -> &'static [&'static str] {
        Self::GAZE_FEATURES
    }

    fn set_max_faces(&mut self, n: usize) {
        self.max_faces = n.max(1);
    }

    fn close(&mut self) {
        self.detector = None;
        self.last = None;
    }

    /// Panics after `close()` (the Python raised `RuntimeError`).
    fn process(&mut self, frame_bgr: &Frame, timestamp: f64) -> Observation {
        let started = Instant::now();
        let Some(detector) = self.detector.as_mut() else {
            panic!("LiteBackend is closed");
        };
        let (width, height) = (frame_bgr.width, frame_bgr.height);
        let resized;
        let mut image = frame_bgr;
        if width > MAX_INPUT_WIDTH {
            let scale = MAX_INPUT_WIDTH as f64 / width as f64;
            let small_h = ((height as f64 * scale).round_ties_even() as usize).max(1);
            resized = resize_area(frame_bgr, MAX_INPUT_WIDTH, small_h);
            image = &resized;
        }
        let (small_w, small_h) = (image.width, image.height);
        let frame_size = (width as u32, height as u32);
        let mut faces: Vec<Face> = Vec::new();
        if self.input_size != Some((small_w, small_h))
            && let Err(err) = detector.set_input_size((small_w, small_h))
        {
            log::warn!("YuNet cannot run at {small_w}x{small_h}: {err}");
        } else {
            self.input_size = Some((small_w, small_h));
            match detector.detect(image) {
                Ok(rows) => faces = rows,
                Err(err) => log::warn!("YuNet detection failed: {err}"),
            }
        }
        faces.retain(|row| row[14] >= SCORE_THRESHOLD);
        if faces.is_empty() {
            self.last = None;
            let mut obs = Observation::new(timestamp, 0);
            obs.inference_ms = started.elapsed().as_secs_f64() * 1e3;
            obs.frame_size = frame_size;
            return obs;
        }

        // The first of equally large faces, like Python's max().
        let mut primary = 0;
        for (i, f) in faces.iter().enumerate() {
            if f[2] * f[3] > faces[primary][2] * faces[primary][3] {
                primary = i;
            }
        }
        let face = faces[primary];
        let score = f64::from(face[14]);
        let (sw, sh) = (small_w as f64, small_h as f64);
        let v = |i: usize| f64::from(face[i]);
        let (x0, y0) = (v(0) / sw, v(1) / sh);
        let (x1, y1) = ((v(0) + v(2)) / sw, (v(1) + v(3)) / sh);
        let face_box = clipped_box(&face, sw, sh);

        let mut obs = Observation::new(timestamp, faces.len() as u32);
        if let Some((features, head_yaw, head_pitch)) = geometry_features(&face, sw, sh) {
            let touches_border = x0 <= BORDER_MARGIN
                || y0 <= BORDER_MARGIN
                || x1 >= 1.0 - BORDER_MARGIN
                || y1 >= 1.0 - BORDER_MARGIN;
            obs.features = Some(features);
            obs.head_yaw = Some(head_yaw);
            obs.head_pitch = Some(head_pitch);
            obs.quality = if touches_border {
                score.min(0.5)
            } else {
                score
            };
        }

        self.last = Some(Detections {
            timestamp,
            points: std::array::from_fn(|i| [v(4 + 2 * i) / sw, v(5 + 2 * i) / sh]),
            others: faces
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != primary)
                .map(|(_, f)| clipped_box(f, sw, sh))
                .collect(),
        });
        obs.face_box = Some(face_box);
        obs.inference_ms = started.elapsed().as_secs_f64() * 1e3;
        obs.frame_size = frame_size;
        obs
    }

    fn annotate(&self, frame_bgr: &Frame, observation: &Observation) -> Frame {
        let mut out = frame_bgr.clone();
        let (w, h) = (out.width as f64, out.height as f64);
        let thickness = line_thickness(&out);
        if let Some(last) = &self.last
            && last.timestamp == observation.timestamp
        {
            for [px, py] in last.points {
                let center = (
                    (px * w).round_ties_even() as i32,
                    (py * h).round_ties_even() as i32,
                );
                let c = COLOR_POINT;
                draw_filled_circle_mut(
                    &mut out,
                    center,
                    thickness + 2,
                    image::Rgb([c.0, c.1, c.2]),
                );
            }
            for &box_ in &last.others {
                draw_box(&mut out, box_, COLOR_OTHER, thickness);
            }
        }
        if let Some(face_box) = observation.face_box {
            draw_box(&mut out, face_box, COLOR_BOX, thickness);
        }
        draw_label(&mut out, &observation_label(observation));
        out
    }
}
