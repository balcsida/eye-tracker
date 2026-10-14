//! Face mesh backend: 478 face landmarks (with irises) and head pose.
//!
//! This is the accurate default. It runs MediaPipe's face landmark network (the
//! `face_landmarks_detector.tflite` member of the Face Landmarker bundle, Apache-2.0, see
//! `models/NOTICE.md`) with tract. The MediaPipe runtime itself is deliberately not used: it
//! contains a usage-logging client that uploads to Google servers, and this app never touches
//! the network.
//!
//! Each analysed frame costs about one landmark inference (about 13 ms on one core of a current
//! desktop CPU, most of it in the network; a frame without a face costs one YuNet detection,
//! about 11 ms):
//!
//! 1. **Region of interest.** A square crop, rotated so that the eye line is horizontal, is
//!    tracked from the previous frame's landmarks (MediaPipe's rule: landmark bounding box, 1.5x
//!    its longer side). When there is no previous face, when the network's face-presence score
//!    drops below 0.5 or when the crop degenerates, YuNet (the lite backend's detector) finds
//!    faces and the best one (largest, then most central) seeds a new crop. When the network
//!    rejects that seed or accepts it only half-heartedly, it is also tried rotated (YuNet
//!    misjudges strong head roll) and the most face-like crop wins if the network is confident
//!    about it; when nothing works, comparable runners-up are tried, unless the unreadable face
//!    is the one tracked until then (the user's hand over it, say).
//! 2. **Landmarks.** The 256x256 crop goes through the network; its 478 points (468 mesh points
//!    plus five per iris) are mapped back into the frame. When the face moved noticeably since
//!    the crop was chosen, the landmarks are biased towards the old position, so the network runs
//!    again on the crop they ask for (and after a large jump on a crop from a fresh YuNet
//!    detection) until the crop sits on the face: a still or slowly moving face costs one
//!    inference, a posture shift between two frames two to four (see `FaceMeshBackend::settle`).
//! 3. **Head pose.** A PnP solve ([`super::pnp`], replacing `cv2.solvePnP`) fits MediaPipe's
//!    canonical face mesh to the landmarks MediaPipe itself uses for rigid fitting (its
//!    Procrustes basis), with approximate intrinsics (focal length = longer frame side,
//!    principal point = frame centre) and the previous pose as the initial guess.
//! 4. **Eyes.** Iris position inside each eye and lid opening (blinks).
//!
//! With the shoulder guard on (`max_faces >= 2`) YuNet additionally counts faces at
//! [`GUARD_DETECT_WIDTH`] pixels at most every [`GUARD_PERIOD_S`] seconds, which finds onlookers
//! several metres away. With the guard off it still looks every [`PRIMARY_CHECK_S`] seconds (one
//! 320 px detection, under 1 % of one core): either way, a clearly stronger face elsewhere in the
//! picture takes over as the primary face if the network reads it confidently, so that a poster
//! or a colleague acquired while the user was away does not keep the gaze control once the user
//! is back.
//!
//! Feature vector ([`FaceMeshBackend::FEATURE_NAMES`]):
//!
//! * `yaw`, `pitch`, `roll` in degrees. Yaw is positive when the nose points towards the right
//!   edge of the camera image, pitch when it points towards the bottom edge (head lowered), roll
//!   when the head is tilted counter-clockwise in the image. Mirroring the image flips the sign
//!   of yaw and roll.
//! * `tx`, `ty`, `tz`: position of the head centre relative to the camera in centimetres (x
//!   right, y down, z away from the camera), only as accurate as the assumed focal length.
//! * `iris_h`, `iris_v`: see [`EyeMetrics`].

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::time::Instant;

use imageproc::drawing::{draw_filled_circle_mut, draw_hollow_circle_mut, draw_line_segment_mut};
use nalgebra::{Matrix3, Point3, Vector3};
use tract_tflite::prelude::*;

use super::base::{BackendUnavailable, VisionBackend};
use super::face_geometry::{FaceGeometry, load_face_geometry, parse_face_geometry};
use super::lite_backend::create_yunet;
use super::overlay::{
    COLOR_BOX, COLOR_EYE, COLOR_IRIS, COLOR_OTHER, COLOR_POSE, Color, draw_box, draw_label,
    line_thickness, observation_label,
};
use super::pnp::{self, Camera};
use super::yunet::{YuNet, resize_area};
use crate::types::Observation;
use crate::vision::Frame;
use crate::vision::threads::{MAX_OPENCV_THREADS, limit_opencv_threads};

pub const MODEL_FILE: &str = "face_landmarks_detector.tflite";
pub const GEOMETRY_FILE: &str = "geometry_pipeline_metadata_landmarks.binarypb";

/// Side of the square network input in pixels.
pub const INPUT_SIZE: usize = 256;
/// Crop side relative to the longer side of the face (MediaPipe uses 1.5).
pub const ROI_SCALE: f64 = 1.5;
/// Face-presence probability below which the tracked face counts as lost.
pub const PRESENCE_THRESHOLD: f64 = 0.5;
/// A crop smaller than this (frame pixels) is too coarse to track.
pub const MIN_ROI_SIDE: f64 = 24.0;
/// YuNet input width for (re-)acquiring the face.
pub const DETECT_WIDTH: usize = 320;
/// YuNet input width while the shoulder guard counts faces. Measured with a face photo scaled
/// into a 640x480 frame, YuNet finds faces from about 20 px wide at 320 px input, 14 px at 480
/// and 12 px at 640. With a typical 65-78° webcam a face is roughly 60-75 px divided by its
/// distance in metres wide at 640 px, so 480 px reaches onlookers about 4 m away (320 px: about
/// 2.5-3 m) for half the cost of 640 px.
pub const GUARD_DETECT_WIDTH: usize = 480;
/// Minimum seconds between shoulder-guard face counts; the count is reused in between (the guard
/// itself waits seconds before reacting).
pub const GUARD_PERIOD_S: f64 = 0.5;
/// YuNet detections below this confidence are ignored (and not counted as faces).
pub const DETECTION_SCORE: f32 = 0.6;
/// A face elsewhere in the image whose primary score (see [`primary_score`]) is this much higher
/// than the tracked face's takes over as the primary face.
pub const PRIMARY_SWITCH_RATIO: f64 = 1.3;
/// How much less a face at the image corner counts than the same face at the centre when
/// choosing the primary face (the user usually sits in front of the camera; posters, TVs and
/// passers-by are usually off to the side). Mild on purpose: size dominates, centrality decides
/// between similar faces.
pub const PRIMARY_CENTRE_WEIGHT: f64 = 0.3;
/// With the shoulder guard off, YuNet looks for a better primary face this often (seconds) while
/// a face is tracked, so a face acquired while the user was away (a poster, a colleague) does not
/// keep the role once the user is back. A stronger face that the network then rejects is not
/// tried again for this long either, also with the guard on (which detects every
/// `GUARD_PERIOD_S`): an unreadable face in view (a profile, a hand over it) would otherwise cost
/// an extra inference at every count, doubling the inferences at 2 fps.
pub const PRIMARY_CHECK_S: f64 = 1.5;
/// ... and this soon after a face was (re-)acquired, when the user may still be on the way into
/// the picture.
pub const PRIMARY_RECHECK_S: f64 = 0.5;
/// A landmark pass whose presence logit is below this is a partial fit at best, although it
/// clears the presence threshold. Measured on a face photo: a hand over the eyes scored 3.3-3.8,
/// over the mouth 2.8-3.2 (the next pass rejected it), a face half out of the frame 0.4-4.1,
/// three quarters in it 6.4-7.8, the first, wrongly rolled pass on a face rolled by 110° 3.2;
/// faces seen in full scored 10-24 (10.3 heavily blurred, 11-12 with a hand on the chin), and
/// 11-22 as a stronger face coming into the picture. Such a fit does not take the primary role
/// from the tracked face (one took it for good: it kept clearing the presence threshold), and it
/// does not count as settled (the motion gate repeated one, 5° off, for seconds, although the
/// next pass rejected it).
pub const CONFIDENT_LOGIT: f64 = 8.0;
/// YuNet's box side divided by the side of the 468-point landmark box, for the same face
/// (measured 1.10-1.18 at 320 and 480 px input). Converts a tracked face's size to YuNet's scale
/// when YuNet itself did not report it.
pub const YUNET_TO_MESH_SIDE: f64 = 1.13;
/// When the best detection fails the landmark network (a profile, a hand over the face), up to
/// this many detections are tried in all ...
pub const MAX_SEED_CANDIDATES: usize = 3;
/// ... but only those scoring at least this share of the best one. A face much smaller than the
/// user's is a background face, and the user turning away must not hand it the gaze control. Nor
/// does any other face get it while the best detection is the face tracked until it became
/// unreadable (see `FaceMeshBackend::acquire`).
pub const SEED_CANDIDATE_MIN_RATIO: f64 = 0.5;
/// YuNet's eye points hardly follow a rolled head at 320 px (a 55° roll reads as about 5°), and
/// the landmark network rejects crops more than about 40° off, or worse, fits them with a partly
/// rolled face. A seed that fails, or passes with a presence logit below
/// `ROLL_RETRY_BELOW_LOGIT`, is therefore also tried rotated by this angle either way, and the
/// most face-like crop wins ...
pub const ROLL_RETRY_DEG: f64 = 45.0;
/// ... (aligned faces score a logit of about 12-22, 9 when heavily blurred; such partial fits
/// about 2-8). A turned crop scoring below this itself counts only if the crop its landmarks ask
/// for then reads confidently: a face the network rejects upright (a hand over the mouth) passed
/// turned by 45° with a logit of 2-4 and a fit 30-45° off in roll and yaw, and a crop 135° off an
/// upright face with 2.4, from where the passes converged on an upside-down fit. Correctly turned
/// crops of photos rolled by 45-70° scored 11-30 ...
pub const ROLL_RETRY_BELOW_LOGIT: f64 = 10.0;
/// ... at most this often (seconds) when it does not help, so a face the network can never read
/// (a profile) does not cost two extra inferences on every frame.
pub const ROLL_RETRY_PERIOD_S: f64 = 2.0;
/// A crop counts as settled on the face when the landmarks it produced ask for a crop whose
/// centre is at most this share of its side away ...
pub const ROI_SETTLE_SHIFT: f64 = 0.04;
/// ... and whose side differs by at most this share (still-face jitter stays below about 0.01
/// for both).
pub const ROI_SETTLE_SCALE: f64 = 0.035;
/// A crop whose centre is further off than this share of its side (or whose side is off by more
/// than `ROI_SETTLE_SCALE`) is re-run within the same frame. It pulls the landmarks towards where
/// the face was: by up to 10° of pitch and 0.1 of iris travel after a 5-8 cm posture shift
/// between two frames, and upward jumps show up mostly as a changed side, because the network
/// follows them only a fraction of the way. Between the two shift limits the bias stays below
/// about 1.5° and the next frame removes it (the motion gate then does not hold the observation,
/// see `VisionBackend::settled`), so a steadily moving head costs no extra inference.
pub const ROI_CATCH_UP_SHIFT: f64 = 0.08;
/// ... provided that the network is sure it sees a face in the crop (a presence logit of at least
/// this). Far behind a large jump, an upward one especially, the network follows the face only a
/// fraction of the way per pass, so a crop can ask for a shift below `ROI_CATCH_UP_SHIFT` while
/// its landmarks are still 35-45 px (12-20° of pitch) off. Measured on drawn faces after such
/// jumps, those crops scored a logit of 4-6, crops in the same band that were on the face
/// (landmarks within 1.5 px) 19-23; for photos see `ROLL_RETRY_BELOW_LOGIT`. Without this, which
/// of the two a jump ended on depended on pixel-level details (a 1 px shift of the face, sensor
/// noise, the CPU's floating-point rounding).
pub const CATCH_UP_MIN_LOGIT: f64 = 10.0;
/// After this many analysed frames in a row whose crop did not settle, the result counts as
/// settled anyway. A crop catching up with a jump settles within a frame or two; one that keeps
/// shifting slightly on a still picture is jitter, and the motion gate must be able to hold it to
/// keep the CPU use low.
pub const MAX_UNSETTLED_FRAMES: usize = 3;
/// Mean eye openness (lid gap / eye width) below which the eyes always count as closed. Open eyes
/// measure roughly 0.2-0.35.
pub const BLINK_OPENNESS: f64 = 0.12;
/// Once the user's own open-eye baseline is known, the blink threshold becomes
/// `BLINK_BASELINE_RATIO * baseline`, clamped to [`BLINK_MIN_OPENNESS`, `BLINK_OPENNESS`]:
/// narrow eyes or a downward gaze never read as a blink.
pub const BLINK_MIN_OPENNESS: f64 = 0.08;
pub const BLINK_BASELINE_RATIO: f64 = 0.5;
/// Open-eye samples needed before the baseline is trusted, and how many are kept.
pub const BLINK_BASELINE_MIN_SAMPLES: usize = 30;
pub const BLINK_BASELINE_WINDOW: usize = 90;
/// A face whose landmark box comes this close (normalised) to the image edge is probably cut
/// off, so its landmarks are less trustworthy.
pub const BORDER_MARGIN: f64 = 0.01;
pub const MAX_FACES_LIMIT: usize = 5;

const OUTPUT_NAMES: [&str; 2] = ["Identity", "Identity_1"]; // landmarks, face-presence logit
const NUM_LANDMARKS: usize = 478;
const MESH_POINTS: usize = 468;
const MIN_EYE_WIDTH_PX: f64 = 1.5;

// Per eye: (first corner, second corner, iris centre, upper lid, lower lid). Landmarks are
// numbered anatomically, and in a camera image the subject's right eye (33/133) appears on the
// left. Taking 33 -> 133 for one eye and 362 -> 263 for the other makes both corner axes point
// towards image-right, so the iris coordinate moves in the same direction in both eyes and can be
// averaged.
pub const EYE_A: [usize; 5] = [33, 133, 468, 159, 145];
pub const EYE_B: [usize; 5] = [362, 263, 473, 386, 374];
/// Outer eye corners on the image-left and image-right side: the eye line.
const EYE_LINE: (usize, usize) = (33, 263);
const NOSE_TIP: usize = 4;
/// Preview arrow showing where the face points: 5 cm out of the face (the pose model frame's -z
/// axis).
const POSE_ARROW_CM: [f64; 3] = [0.0, 0.0, -5.0];

const EYE_A_CONTOUR: [usize; 16] = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
];
const EYE_B_CONTOUR: [usize; 16] = [
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
];
const IRIS_A: [usize; 5] = [468, 469, 470, 471, 472];
const IRIS_B: [usize; 5] = [473, 474, 475, 476, 477];

/// A normalised `(x, y, w, h)` box.
pub type Box4 = (f64, f64, f64, f64);
/// A YuNet row (see [`super::yunet::Face`]) in frame pixels.
pub type Detection = [f64; 15];
/// Landmarks in frame pixels and the presence logit of the pass that produced them.
type Inferred = (Vec<[f64; 2]>, f64);

// ----------------------------------------------------------------------------
// Pure helpers (unit-tested without the network)

/// Iris position and lid opening, averaged over both eyes.
///
/// `iris_h` is the iris centre projected onto the corner-to-corner axis (0 at the image-left
/// corner, 1 at the image-right corner, ~0.5 looking straight). `iris_v` is the signed distance
/// from that axis in eye widths (positive = towards the bottom of the image). `openness` is the
/// lid gap in eye widths.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EyeMetrics {
    pub iris_h: f64,
    pub iris_v: f64,
    pub openness: f64,
}

/// Compute [`EyeMetrics`] from landmark positions in pixels.
///
/// `points_px` has 478 (or more) rows. Pixel units matter: normalised landmark coordinates are
/// anisotropic for non-square images. Returns `None` when the eyes are degenerate (too small).
pub fn eye_metrics(points_px: &[[f64; 2]]) -> Option<EyeMetrics> {
    if points_px.len() < NUM_LANDMARKS {
        return None;
    }
    eye_metrics_of([EYE_A, EYE_B].map(|eye| eye.map(|i| points_px[i])))
}

/// [`EyeMetrics`] from the five points of each eye in pixels, in the order of [`EYE_A`] and
/// [`EYE_B`]: the image-left corner, the image-right corner, the iris centre, the upper lid and
/// the lower lid. Returns `None` when an eye is degenerate (too small).
pub fn eye_metrics_of(eyes: [[[f64; 2]; 5]; 2]) -> Option<EyeMetrics> {
    let (mut h, mut v, mut openness) = (0.0, 0.0, 0.0);
    for [a, c1, iris, lid_top, lid_bottom] in eyes {
        let axis = sub(c1, a);
        let width = axis[0].hypot(axis[1]);
        if width.is_nan() || width < MIN_EYE_WIDTH_PX {
            return None;
        }
        let u = [axis[0] / width, axis[1] / width];
        let normal = [-u[1], u[0]]; // image-down for an upright face
        let rel = sub(iris, a);
        h += dot(rel, u) / width;
        v += dot(rel, normal) / width;
        let lid = sub(lid_top, lid_bottom);
        openness += lid[0].hypot(lid[1]) / width;
    }
    Some(EyeMetrics {
        iris_h: h / 2.0,
        iris_v: v / 2.0,
        openness: openness / 2.0,
    })
}

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn dot(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

/// Decides whether the eyes are closed, relative to the user's own eyes.
///
/// A fixed openness threshold misfires for narrow eyes and for a downward gaze (the lids close in
/// when looking at a monitor below the camera). The detector keeps a rolling median of open-eye
/// readings and, once it has enough of them, flags a blink below half of that baseline. The
/// threshold never rises above the absolute [`BLINK_OPENNESS`], so wide eyes behave as before.
#[derive(Debug, Clone)]
pub struct BlinkDetector {
    pub absolute: f64,
    pub minimum: f64,
    pub ratio: f64,
    pub min_samples: usize,
    window: usize,
    history: VecDeque<f64>,
}

impl Default for BlinkDetector {
    fn default() -> Self {
        Self::new(
            BLINK_OPENNESS,
            BLINK_MIN_OPENNESS,
            BLINK_BASELINE_RATIO,
            BLINK_BASELINE_MIN_SAMPLES,
            BLINK_BASELINE_WINDOW,
        )
    }
}

impl BlinkDetector {
    pub fn new(absolute: f64, minimum: f64, ratio: f64, min_samples: usize, window: usize) -> Self {
        let min_samples = min_samples.max(1);
        let window = min_samples.max(window);
        Self {
            absolute,
            minimum,
            ratio,
            min_samples,
            window,
            history: VecDeque::with_capacity(window),
        }
    }

    /// Current openness threshold below which the eyes count as closed.
    pub fn threshold(&self) -> f64 {
        if self.history.len() < self.min_samples {
            return self.absolute;
        }
        let mut values: Vec<f64> = self.history.iter().copied().collect();
        values.sort_by(f64::total_cmp);
        let n = values.len();
        let baseline = if n % 2 == 1 {
            values[n / 2]
        } else {
            (values[n / 2 - 1] + values[n / 2]) / 2.0
        };
        self.absolute.min(self.minimum.max(self.ratio * baseline))
    }

    /// Classify one reading (`true` = closed) and learn from open eyes.
    ///
    /// Until the baseline exists every finite reading is learnt, not only those above the
    /// absolute threshold: eyes that stay below it (narrow eyes, a monitor below the camera)
    /// would otherwise never teach the detector anything and read as closed forever. The median
    /// shrugs off the few blinks among them. Afterwards only open eyes are learnt, so a long
    /// stretch of closed eyes cannot drag the baseline down.
    pub fn update(&mut self, openness: f64) -> bool {
        if !openness.is_finite() {
            return false;
        }
        let closed = openness < self.threshold();
        if !closed || self.history.len() < self.min_samples {
            if self.history.len() == self.window {
                self.history.pop_front();
            }
            self.history.push_back(openness);
        }
        closed
    }

    pub fn reset(&mut self) {
        self.history.clear();
    }
}

/// A square, rotated crop of the frame that the landmark network sees.
///
/// `(cx, cy)` is the centre and `side` the side length in frame pixels; `angle` is the direction
/// of the eye line in degrees (image coordinates, y down), which the crop rotates to horizontal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Roi {
    pub cx: f64,
    pub cy: f64,
    pub side: f64,
    pub angle: f64,
}

/// A 2x3 affine matrix.
pub type Affine = [[f64; 3]; 2];

impl Roi {
    pub fn new(cx: f64, cy: f64, side: f64, angle: f64) -> Self {
        Self {
            cx,
            cy,
            side,
            angle,
        }
    }

    /// 2x3 affine matrix mapping frame pixels to network-input pixels.
    pub fn transform(&self) -> Affine {
        // cv2.getRotationMatrix2D: positive angles rotate counter-clockwise on screen, which
        // levels an eye line that descends towards image-right by that angle.
        let scale = INPUT_SIZE as f64 / self.side;
        let (sin, cos) = self.angle.to_radians().sin_cos();
        let (alpha, beta) = (scale * cos, scale * sin);
        let half = INPUT_SIZE as f64 / 2.0;
        [
            [
                alpha,
                beta,
                (1.0 - alpha) * self.cx - beta * self.cy + half - self.cx,
            ],
            [
                -beta,
                alpha,
                beta * self.cx + (1.0 - alpha) * self.cy + half - self.cy,
            ],
        ]
    }

    /// True when the crop is finite, large enough and centred inside the frame.
    pub fn valid(&self, width: usize, height: usize) -> bool {
        [self.cx, self.cy, self.side, self.angle]
            .iter()
            .all(|v| v.is_finite())
            && self.side >= MIN_ROI_SIDE
            && 0.0 <= self.cx
            && self.cx < width as f64
            && 0.0 <= self.cy
            && self.cy < height as f64
    }
}

/// `cv2.invertAffineTransform`.
fn invert_affine(m: &Affine) -> Affine {
    let d = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let d = if d != 0.0 { 1.0 / d } else { 0.0 };
    let (a11, a12, a21, a22) = (m[1][1] * d, -m[0][1] * d, -m[1][0] * d, m[0][0] * d);
    let b1 = -a11 * m[0][2] - a12 * m[1][2];
    let b2 = -a21 * m[0][2] - a22 * m[1][2];
    [[a11, a12, b1], [a21, a22, b2]]
}

fn apply(m: &Affine, [x, y]: [f64; 2]) -> [f64; 2] {
    [
        m[0][0] * x + m[0][1] * y + m[0][2],
        m[1][0] * x + m[1][1] * y + m[1][2],
    ]
}

/// `cv2.warpAffine(frame, m, (size, size), flags=INTER_LINEAR, borderMode=BORDER_CONSTANT)`:
/// bilinear sampling, black outside the frame. Returns BGR bytes.
fn warp_affine(frame: &Frame, m: &Affine, size: usize) -> Vec<u8> {
    let inv = invert_affine(m);
    let (w, h) = (frame.width as i64, frame.height as i64);
    let tap = |x: i64, y: i64, c: usize| -> f64 {
        if 0 <= x && x < w && 0 <= y && y < h {
            f64::from(frame.data[((y * w + x) * 3) as usize + c])
        } else {
            0.0
        }
    };
    let mut out = vec![0u8; size * size * 3];
    for y in 0..size {
        for x in 0..size {
            let [sx, sy] = apply(&inv, [x as f64, y as f64]);
            if !(sx > -1.0 && sx < w as f64 && sy > -1.0 && sy < h as f64) {
                continue; // all four taps outside (also NaN)
            }
            let (x0, y0) = (sx.floor(), sy.floor());
            let (fx, fy) = (sx - x0, sy - y0);
            let (x0, y0) = (x0 as i64, y0 as i64);
            for c in 0..3 {
                let top = tap(x0, y0, c) * (1.0 - fx) + tap(x0 + 1, y0, c) * fx;
                let bottom = tap(x0, y0 + 1, c) * (1.0 - fx) + tap(x0 + 1, y0 + 1, c) * fx;
                let v = top * (1.0 - fy) + bottom * fy;
                out[(y * size + x) * 3 + c] = v.round_ties_even().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

fn eye_line_angle(left: [f64; 2], right: [f64; 2]) -> f64 {
    (right[1] - left[1]).atan2(right[0] - left[0]).to_degrees()
}

/// Crop for a YuNet detection row (pixels: box, right eye, left eye, ...).
pub fn roi_from_detection(face: &Detection) -> Roi {
    let (x, y, w, h) = (face[0], face[1], face[2], face[3]);
    // YuNet's "right eye" is the one on the image left.
    let angle = eye_line_angle([face[4], face[5]], [face[6], face[7]]);
    Roi::new(x + w / 2.0, y + h / 2.0, w.max(h) * ROI_SCALE, angle)
}

/// Crop for tracking a face into the next frame (MediaPipe's rule).
pub fn roi_from_landmarks(points_px: &[[f64; 2]]) -> Roi {
    let (x0, y0, x1, y1) = bounds(&points_px[..MESH_POINTS]);
    let angle = eye_line_angle(points_px[EYE_LINE.0], points_px[EYE_LINE.1]);
    let side = (x1 - x0).max(y1 - y0) * ROI_SCALE;
    Roi::new((x0 + x1) / 2.0, (y0 + y1) / 2.0, side, angle)
}

/// `(min x, min y, max x, max y)` of some points (NaN-propagating like numpy's min/max).
fn bounds(points: &[[f64; 2]]) -> (f64, f64, f64, f64) {
    let pick = |f: fn(f64, f64) -> f64, i: usize| {
        points.iter().map(|p| p[i]).reduce(|a, b| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                f(a, b)
            }
        })
    };
    let get = |v: Option<f64>| v.unwrap_or(f64::NAN);
    (
        get(pick(f64::min, 0)),
        get(pick(f64::min, 1)),
        get(pick(f64::max, 0)),
        get(pick(f64::max, 1)),
    )
}

/// True when `wanted` (the crop the landmarks ask for) matches `used`.
///
/// Landmarks from a crop that is centred on the face are trustworthy; when the face sits
/// off-centre in the crop (it moved since the crop was chosen) they are pulled towards the crop
/// centre, and the crop they ask for is off by a noticeable share of its side. `shift` and
/// `scale` are the tolerated centre offset and side change, as shares of `used.side` (see
/// [`ROI_SETTLE_SHIFT`]; [`crop_settled`] uses the defaults).
pub fn crop_settled_within(used: &Roi, wanted: &Roi, shift: f64, scale: f64) -> bool {
    if !(used.side > 0.0
        && [wanted.cx, wanted.cy, wanted.side]
            .iter()
            .all(|v| v.is_finite()))
    {
        return false;
    }
    let offset = (wanted.cx - used.cx).hypot(wanted.cy - used.cy) / used.side;
    offset <= shift && (wanted.side / used.side - 1.0).abs() <= scale
}

/// [`crop_settled_within`] the default [`ROI_SETTLE_SHIFT`] and [`ROI_SETTLE_SCALE`].
pub fn crop_settled(used: &Roi, wanted: &Roi) -> bool {
    crop_settled_within(used, wanted, ROI_SETTLE_SHIFT, ROI_SETTLE_SCALE)
}

/// True when a frame's landmark passes may stop at the crop `used`.
///
/// That is when it is settled on the face ([`crop_settled`]), or when it is catching up with a
/// moving face (`wanted` within [`ROI_CATCH_UP_SHIFT`]) and the network is confident that it sees
/// a face in it (`logit`, the presence logit of the pass on `used`, of at least
/// [`CATCH_UP_MIN_LOGIT`]). See `FaceMeshBackend::settle`.
pub fn crop_caught_up(used: &Roi, wanted: &Roi, logit: f64) -> bool {
    if crop_settled(used, wanted) {
        return true;
    }
    logit >= CATCH_UP_MIN_LOGIT
        && crop_settled_within(used, wanted, ROI_CATCH_UP_SHIFT, ROI_SETTLE_SCALE)
}

/// How strongly a face of `side` pixels centred at `(cx, cy)` claims to be the user.
///
/// The face side, reduced by up to [`PRIMARY_CENTRE_WEIGHT`] the further the face is from the
/// image centre (0 at the centre, the full weight at a corner).
pub fn primary_score(side: f64, cx: f64, cy: f64, width: usize, height: usize) -> f64 {
    let (half_w, half_h) = (width as f64 / 2.0, height as f64 / 2.0);
    let offset = ((cx - half_w) / half_w).hypot((cy - half_h) / half_h) / std::f64::consts::SQRT_2;
    side * (1.0 - PRIMARY_CENTRE_WEIGHT * offset.min(1.0))
}

/// Head rotation (degrees) and position (centimetres), see the module docs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeadPose {
    pub yaw: f64,
    pub pitch: f64,
    pub roll: f64,
    pub tx: f64,
    pub ty: f64,
    pub tz: f64,
}

/// [`HeadPose`] from a model-to-camera rotation matrix and translation.
///
/// The model frame is the canonical face flipped to OpenCV's camera convention (x right, y down,
/// z away from the camera), so a face looking straight into the camera has the identity
/// rotation.
pub fn pose_from_rotation(r: &Matrix3<f64>, translation: &Vector3<f64>) -> HeadPose {
    // The face's forward direction, towards the camera when frontal.
    let nose = -r.column(2);
    let yaw = nose[0].atan2(-nose[2]).to_degrees();
    let pitch = nose[1].atan2(nose[0].hypot(nose[2])).to_degrees();
    // Direction of the model x axis in the image; image y points down, so a counter-clockwise
    // tilt lifts its right end (negative y).
    let roll = (-r[(1, 0)]).atan2(r[(0, 0)]).to_degrees();
    HeadPose {
        yaw,
        pitch,
        roll,
        tx: translation[0],
        ty: translation[1],
        tz: translation[2],
    }
}

/// Approximate pinhole intrinsics: focal length = longer side, centred.
pub fn camera_matrix(width: usize, height: usize) -> Camera {
    let f = width.max(height) as f64;
    Camera {
        fx: f,
        fy: f,
        cx: width as f64 / 2.0,
        cy: height as f64 / 2.0,
    }
}

/// Head pose by fitting the canonical face mesh to the landmarks.
#[derive(Debug, Clone)]
pub struct PoseEstimator {
    ids: Vec<usize>,
    object: Vec<Point3<f64>>,
    rvec: Option<Vector3<f64>>,
    tvec: Option<Vector3<f64>>,
    size: Option<(usize, usize)>,
    pub camera: Camera,
}

impl PoseEstimator {
    pub fn new(geometry: &FaceGeometry) -> Self {
        let ids = geometry.basis_ids.clone();
        // Canonical frame (x right, y up, z towards the camera) -> OpenCV-style object frame
        // (x right, y down, z away from the camera).
        let object = ids
            .iter()
            .map(|&i| {
                let [x, y, z] = geometry.vertices[i];
                Point3::new(x, -y, -z)
            })
            .collect();
        Self {
            ids,
            object,
            rvec: None,
            tvec: None,
            size: None,
            camera: camera_matrix(1, 1),
        }
    }

    pub fn rvec(&self) -> Option<Vector3<f64>> {
        self.rvec
    }

    pub fn tvec(&self) -> Option<Vector3<f64>> {
        self.tvec
    }

    /// Forget the previous pose (a new face or a new frame size).
    pub fn reset(&mut self) {
        self.rvec = None;
        self.tvec = None;
    }

    /// Pose for 478 landmark pixels in a `width` x `height` frame.
    pub fn estimate(
        &mut self,
        points_px: &[[f64; 2]],
        width: usize,
        height: usize,
    ) -> Option<HeadPose> {
        if self.size != Some((width, height)) {
            self.size = Some((width, height));
            self.camera = camera_matrix(width, height);
            self.reset();
        }
        let image: Vec<nalgebra::Point2<f64>> = self
            .ids
            .iter()
            .map(|&i| nalgebra::Point2::new(points_px[i][0], points_px[i][1]))
            .collect();
        if !image.iter().all(|p| p.x.is_finite() && p.y.is_finite()) {
            self.reset();
            return None;
        }
        let mut solution = None;
        if let (Some(r), Some(t)) = (self.rvec, self.tvec) {
            solution = self.solve(&image, Some((r, t)));
        }
        if solution.is_none() {
            // A global solver for a fresh face (or when the tracked solution drifted into an
            // implausible one).
            solution = self.solve(&image, None);
        }
        let Some((r, t)) = solution else {
            self.reset();
            return None;
        };
        self.rvec = Some(r);
        self.tvec = Some(t);
        Some(pose_from_rotation(&pnp::rodrigues(&r), &t))
    }

    fn solve(
        &self,
        image: &[nalgebra::Point2<f64>],
        guess: Option<(Vector3<f64>, Vector3<f64>)>,
    ) -> Option<(Vector3<f64>, Vector3<f64>)> {
        let (r, t) = pnp::solve_pnp(&self.object, image, &self.camera, guess)?;
        let rotation = pnp::rodrigues(&r);
        // In front of the camera and facing it (nose towards the lens).
        if t[2] <= 0.0 || rotation[(2, 2)] <= 0.0 {
            return None;
        }
        Some((r, t))
    }
}

// ----------------------------------------------------------------------------
// Model loading

/// The landmark network: an optimised tract plan and the positions of its two outputs.
pub struct LandmarkNet {
    plan: Arc<TypedRunnableModel>,
    landmarks_ix: usize,
    flag_ix: usize,
}

impl LandmarkNet {
    /// Raw landmarks (478 x 3, network-input pixels) and the presence logit for an NHWC RGB
    /// input in 0..1.
    fn forward(&self, input: Tensor) -> TractResult<(Vec<f32>, f32)> {
        let out = self.plan.run(tvec!(input.into()))?;
        let landmarks = out[self.landmarks_ix]
            .to_plain_array_view::<f32>()?
            .iter()
            .copied()
            .collect();
        let flag = out[self.flag_ix]
            .to_plain_array_view::<f32>()?
            .iter()
            .next()
            .copied()
            .unwrap_or(f32::NAN);
        Ok((landmarks, flag))
    }
}

/// The fixed input of the load-time self-test: a different gradient per channel.
///
/// Asymmetric on purpose, so that a runtime that mixes up the tensor layout or the channel order
/// produces visibly different landmarks. Laid out NHWC like the network input; channel 0 (the
/// network's red) is the x ramp, 1 the y ramp, 2 their mean.
pub fn self_test_blob() -> Tensor {
    let ramp = |i: usize| (i as f64 / (INPUT_SIZE - 1) as f64) as f32;
    tract_ndarray::Array4::from_shape_fn((1, INPUT_SIZE, INPUT_SIZE, 3), |(_, y, x, c)| match c {
        0 => ramp(x),
        1 => ramp(y),
        _ => (ramp(x) + ramp(y)) / 2.0,
    })
    .into()
}

/// Reference outputs for [`self_test_blob`], recorded with OpenCV 5.0's classic engine
/// (MediaPipe's own runtime agrees to within 0.005 of the frame on real faces): presence logit,
/// mean landmark (x, y) and iris 468 (x, y) in network-input pixels.
const SELF_TEST_FLAG: f64 = -8.30;
const SELF_TEST_MEAN: (f64, f64) = (127.64, 116.29);
const SELF_TEST_IRIS: (f64, f64) = (95.10, 85.45);
const SELF_TEST_PX_TOLERANCE: f64 = 2.0;
const SELF_TEST_FLAG_TOLERANCE: f64 = 1.5;

/// Make sure the network runs and computes what it should.
///
/// A runtime that loads the model but returns the wrong tensors would give garbage landmarks;
/// comparing one fixed input against reference values turns it into a clean "unavailable".
/// Fails when the outputs have the wrong shape or values.
fn check_landmark_net(net: &LandmarkNet) -> Result<(), String> {
    check_landmark_net_against(net, SELF_TEST_MEAN)
}

fn check_landmark_net_against(net: &LandmarkNet, reference_mean: (f64, f64)) -> Result<(), String> {
    let (landmarks, flag) = net.forward(self_test_blob()).map_err(|e| e.to_string())?;
    if landmarks.len() != NUM_LANDMARKS * 3 {
        return Err(format!(
            "unexpected output size {} (not the landmark model)",
            landmarks.len()
        ));
    }
    let points: Vec<[f64; 2]> = landmarks
        .chunks(3)
        .map(|p| [f64::from(p[0]), f64::from(p[1])])
        .collect();
    let n = points.len() as f64;
    let mean = (
        points.iter().map(|p| p[0]).sum::<f64>() / n,
        points.iter().map(|p| p[1]).sum::<f64>() / n,
    );
    let iris = points[EYE_A[2]];
    let logit = f64::from(flag);
    let worst = [
        mean.0 - reference_mean.0,
        mean.1 - reference_mean.1,
        iris[0] - SELF_TEST_IRIS.0,
        iris[1] - SELF_TEST_IRIS.1,
    ]
    .iter()
    .fold(0.0f64, |m, d| m.max(d.abs()));
    if !(worst <= SELF_TEST_PX_TOLERANCE
        && (logit - SELF_TEST_FLAG).abs() <= SELF_TEST_FLAG_TOLERANCE)
    {
        return Err(format!(
            "the model computes wrong values (landmarks off by {worst:.1} px, presence logit \
             {logit:.2} instead of {SELF_TEST_FLAG:.2})"
        ));
    }
    Ok(())
}

/// Parse and optimise the model, and find its outputs by name (checking their sizes).
fn build_landmark_net(data: &[u8]) -> TractResult<LandmarkNet> {
    let model = tract_tflite::tflite().model_for_read(&mut &data[..])?;
    let outputs = model.output_outlets()?.to_vec();
    let find = |name: &str| -> TractResult<usize> {
        outputs
            .iter()
            .position(|o| model.node(o.node).name == name)
            .ok_or_else(|| anyhow::anyhow!("no output {name} (not the landmark model)"))
    };
    let (landmarks_ix, flag_ix) = (find(OUTPUT_NAMES[0])?, find(OUTPUT_NAMES[1])?);
    for (ix, len) in [(landmarks_ix, NUM_LANDMARKS * 3), (flag_ix, 1)] {
        let fact = model.outlet_fact(outputs[ix])?;
        let size = fact
            .shape
            .as_concrete()
            .map(|s| s.iter().product::<usize>());
        if size != Some(len) {
            anyhow::bail!(
                "unexpected output shape {:?} (not the landmark model)",
                fact.shape
            );
        }
    }
    let plan = model.into_optimized()?.into_runnable()?;
    Ok(LandmarkNet {
        plan,
        landmarks_ix,
        flag_ix,
    })
}

/// Load and self-test the landmark network from the model bytes.
///
/// Fails with [`BackendUnavailable`] when the model cannot be run correctly.
pub fn load_landmark_net_from(data: &[u8]) -> Result<LandmarkNet, BackendUnavailable> {
    let net = build_landmark_net(data).map_err(|err| unrunnable(&format!("{err:#}")))?;
    check_landmark_net(&net).map_err(|err| unrunnable(&err))?;
    Ok(net)
}

fn unrunnable(detail: &str) -> BackendUnavailable {
    BackendUnavailable(format!(
        "tract cannot run the face landmark model: {detail}"
    ))
}

/// Load and self-test the landmark network at `path`.
///
/// Fails with [`BackendUnavailable`] when the model is missing or cannot be run correctly.
pub fn load_landmark_net(path: &Path) -> Result<LandmarkNet, BackendUnavailable> {
    let data = std::fs::read(path).map_err(|_| {
        BackendUnavailable(format!(
            "Face landmark model not found at {} (run cargo xtask fetch-models)",
            path.display()
        ))
    })?;
    load_landmark_net_from(&data)
}

// ----------------------------------------------------------------------------
// The backend

/// What `annotate` needs from the last processed frame.
struct Drawing {
    timestamp: f64,
    /// 478 normalised points of the primary face.
    points: Option<Vec<[f64; 2]>>,
    /// Normalised boxes of the other faces.
    others: Vec<Box4>,
    rvec: Option<Vector3<f64>>,
    tvec: Option<Vector3<f64>>,
    camera: Option<Camera>,
}

/// Clipped, normalised `(x, y, w, h)` of a YuNet row in frame pixels.
fn det_box(face: &Detection, width: usize, height: usize) -> Box4 {
    let (w_, h_) = (width as f64, height as f64);
    let (x, y, w, h) = (face[0], face[1], face[2], face[3]);
    let (x0, y0) = ((x / w_).max(0.0), (y / h_).max(0.0));
    let (x1, y1) = (((x + w) / w_).min(1.0), ((y + h) / h_).min(1.0));
    (x0, y0, (x1 - x0).max(0.0), (y1 - y0).max(0.0))
}

fn det_side(face: &Detection) -> f64 {
    face[2].max(face[3])
}

fn det_centre(face: &Detection) -> (f64, f64) {
    (face[0] + face[2] / 2.0, face[1] + face[3] / 2.0)
}

fn det_score(face: &Detection, width: usize, height: usize) -> f64 {
    let (cx, cy) = det_centre(face);
    primary_score(det_side(face), cx, cy, width, height)
}

/// True when the detection's centre lies inside the (unrotated) crop square.
fn in_crop(face: &Detection, roi: &Roi) -> bool {
    let (cx, cy) = det_centre(face);
    let half = roi.side / 2.0;
    (cx - roi.cx).abs() < half && (cy - roi.cy).abs() < half
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// The first of the highest-scoring detections (like Python's `max`).
fn best_by(faces: &[Detection], key: impl Fn(&Detection) -> f64) -> Option<&Detection> {
    let mut best: Option<(&Detection, f64)> = None;
    for face in faces {
        let k = key(face);
        if best.is_none_or(|(_, b)| k > b) {
            best = Some((face, k));
        }
    }
    best.map(|(face, _)| face)
}

fn clamp_faces(n: usize) -> usize {
    n.clamp(1, MAX_FACES_LIMIT)
}

/// Stand-ins for the Python tests' monkeypatching of `_infer`, `_detect` and
/// `roi_from_detection`.
#[cfg(test)]
#[derive(Default)]
pub(crate) struct TestHooks {
    /// Replaces `infer`; gets the crop and the real `infer` to call through.
    #[allow(clippy::type_complexity)]
    pub infer: Option<
        Box<
            dyn FnMut(
                &Frame,
                &Roi,
                &mut dyn FnMut(&Frame, &Roi) -> Option<Inferred>,
            ) -> Option<Inferred>,
        >,
    >,
    pub infer_calls: usize,
    /// The input width of every `detect` call.
    pub detect_widths: Vec<usize>,
    /// Added to the angle of every crop seeded from a detection.
    pub seed_roll: f64,
}

/// Head pose and iris features from MediaPipe's face landmark network.
///
/// Fails to construct with [`BackendUnavailable`] when a model cannot be loaded.
pub struct FaceMeshBackend {
    net: Option<LandmarkNet>,
    detector_model: Option<PathBuf>,
    detector: Option<YuNet>,
    guard_detector: Option<YuNet>,
    detector_sizes: HashMap<usize, (usize, usize)>,
    pose: PoseEstimator,
    /// Nose tip in the pose model frame, for the preview's direction line.
    nose_tip: [f64; 3],
    blink: BlinkDetector,
    max_faces: usize,
    roi: Option<Roi>,
    settled: bool,
    unsettled_frames: usize,
    frame_size: Option<(usize, usize)>,
    /// Other faces (normalised boxes) from the latest shoulder-guard count, reused until the
    /// next count is due.
    guard_others: Vec<Box4>,
    guard_ts: f64,
    /// Primary-face check with the guard off: time of the last one and the period until the
    /// next.
    primary_ts: f64,
    primary_period: f64,
    /// Last time rotated seeds were tried in vain (see `ROLL_RETRY_PERIOD_S`).
    roll_retry_ts: f64,
    /// Where the face tracked until it became unreadable is (see `acquire`).
    unreadable: Option<Roi>,
    last: Option<Drawing>,
    landmarks: Option<Vec<[f64; 2]>>,
    closed: bool,
    #[cfg(test)]
    pub(crate) hooks: TestHooks,
}

impl FaceMeshBackend {
    pub const NAME: &'static str = "facemesh";
    pub const FEATURE_NAMES: &'static [&'static str] =
        &["yaw", "pitch", "roll", "tx", "ty", "tz", "iris_h", "iris_v"];
    pub const FEATURE_VERSION: &'static str = "facemesh-pose-iris-1";
    pub const GAZE_FEATURES: &'static [&'static str] = &["yaw", "pitch", "iris_h", "iris_v"];

    /// The backend with the embedded models.
    pub fn new(max_faces: usize) -> Result<Self, BackendUnavailable> {
        Self::with_paths(None, max_faces, None, None)
    }

    /// The backend with model files from disk where given (the embedded ones otherwise):
    /// `model_path` is the landmark network, `geometry_path` the canonical face geometry
    /// (`.binarypb`), `detector_path` the YuNet face detector (`.onnx`). `max_faces` of 2 or
    /// more enables shoulder-guard counting.
    pub fn with_paths(
        model_path: Option<&Path>,
        max_faces: usize,
        geometry_path: Option<&Path>,
        detector_path: Option<&Path>,
    ) -> Result<Self, BackendUnavailable> {
        limit_opencv_threads(MAX_OPENCV_THREADS);
        let geometry = match geometry_path {
            Some(path) => load_face_geometry(path),
            None => parse_face_geometry(super::geometry_metadata()),
        }
        .map_err(|err| {
            BackendUnavailable(format!(
                "Face geometry unavailable: {err} (run cargo xtask fetch-models)"
            ))
        })?;
        let net = match model_path {
            Some(path) => load_landmark_net(path)?,
            None => load_landmark_net_from(super::landmark_model())?,
        };
        let detector_model = detector_path.map(Path::to_path_buf);
        let detector = create_yunet(detector_model.as_deref(), DETECT_WIDTH)?;
        let [x, y, z] = geometry.vertices[NOSE_TIP];
        Ok(Self {
            net: Some(net),
            detector_model,
            detector: Some(detector),
            guard_detector: None,
            detector_sizes: HashMap::new(),
            pose: PoseEstimator::new(&geometry),
            nose_tip: [x, -y, -z],
            blink: BlinkDetector::default(),
            max_faces: clamp_faces(max_faces),
            roi: None,
            settled: true,
            unsettled_frames: 0,
            frame_size: None,
            guard_others: Vec::new(),
            guard_ts: f64::NEG_INFINITY,
            primary_ts: f64::NEG_INFINITY,
            primary_period: PRIMARY_CHECK_S,
            roll_retry_ts: f64::NEG_INFINITY,
            unreadable: None,
            last: None,
            landmarks: None,
            closed: false,
            #[cfg(test)]
            hooks: TestHooks::default(),
        })
    }

    /// Most faces reported in `face_count`.
    pub fn max_faces(&self) -> usize {
        self.max_faces
    }

    /// Normalised landmarks (478 points) of the last frame's primary face.
    pub fn landmarks(&self) -> Option<Vec<[f64; 2]>> {
        self.landmarks.clone()
    }

    /// Current eye-openness threshold for blinks (diagnostics).
    pub fn blink_threshold(&self) -> f64 {
        self.blink.threshold()
    }

    /// Drop everything carried from frame to frame except the blink baseline.
    fn forget_tracking(&mut self) {
        self.roi = None;
        self.settled = true;
        self.unsettled_frames = 0;
        self.guard_others.clear();
        self.guard_ts = f64::NEG_INFINITY;
        self.primary_ts = f64::NEG_INFINITY;
        self.primary_period = PRIMARY_CHECK_S;
        self.roll_retry_ts = f64::NEG_INFINITY;
        self.unreadable = None;
        self.pose.reset();
    }

    fn note_settled(&mut self, settled: bool) {
        self.unsettled_frames = if settled {
            0
        } else {
            self.unsettled_frames + 1
        };
        self.settled = settled || self.unsettled_frames >= MAX_UNSETTLED_FRAMES;
    }

    fn primary_check_due(&self, timestamp: f64) -> bool {
        // Written like the guard's check so that a clock jumping backwards (a new timestamp
        // source) counts as due rather than as "checked recently".
        !(0.0..self.primary_period).contains(&(timestamp - self.primary_ts))
    }

    fn schedule_primary_check(&mut self, timestamp: f64, period: f64) {
        self.primary_ts = timestamp;
        self.primary_period = period;
    }

    /// `roi_from_detection` (which the tests may turn).
    fn seed(&self, face: &Detection) -> Roi {
        #[cfg(test)]
        {
            let seed = roi_from_detection(face);
            Roi {
                angle: seed.angle + self.hooks.seed_roll,
                ..seed
            }
        }
        #[cfg(not(test))]
        roi_from_detection(face)
    }

    /// Seed the landmark network from YuNet detections; the first that works wins.
    ///
    /// Candidates go best first (largest, then most central: see [`primary_score`]). When the
    /// network rejects the best one or accepts it only with a low presence logit
    /// ([`ROLL_RETRY_BELOW_LOGIT`]), it is also tried rotated by [`ROLL_RETRY_DEG`] either way
    /// (YuNet misjudges strong head roll) and the most face-like crop wins. When it still fails,
    /// comparable runners-up are tried (see [`SEED_CANDIDATE_MIN_RATIO`]), unless the best
    /// detection lies in `lost`, the crop of the face that was tracked until it became
    /// unreadable: that is the user drinking, a hand over the face, turned away, and a poster or
    /// a colleague in the background must not take the gaze control meanwhile. Such a face is
    /// followed from frame to frame until a face is read again or it is no longer the best
    /// detection. A frame where the first seed works well costs one inference, as before.
    /// Returns the landmarks, their crop and its presence logit.
    fn acquire(
        &mut self,
        frame: &Frame,
        detections: &[Detection],
        timestamp: f64,
        lost: Option<Roi>,
    ) -> Option<(Vec<[f64; 2]>, Roi, f64)> {
        let (width, height) = (frame.width, frame.height);
        if detections.is_empty() {
            // Nothing in view: whether the user is still there is unknown, so what is remembered
            // of them is kept.
            return None;
        }
        let mut ranked = detections.to_vec();
        ranked.sort_by(|a, b| det_score(b, width, height).total_cmp(&det_score(a, width, height)));
        let best_score = det_score(&ranked[0], width, height);
        let holding = lost.is_some_and(|lost| in_crop(&ranked[0], &lost));
        self.unreadable = None;
        for (index, face) in ranked.iter().take(MAX_SEED_CANDIDATES).enumerate() {
            if index > 0
                && (holding
                    || det_score(face, width, height) < SEED_CANDIDATE_MIN_RATIO * best_score)
            {
                break;
            }
            let seed = self.seed(face);
            if !seed.valid(width, height) {
                continue;
            }
            let result = self.infer(frame, &seed);
            let doubtful = result.as_ref().is_none_or(|r| r.1 < ROLL_RETRY_BELOW_LOGIT);
            if index == 0
                && doubtful
                && !(0.0..ROLL_RETRY_PERIOD_S).contains(&(timestamp - self.roll_retry_ts))
            {
                let to_beat = result.as_ref().map_or(f64::NEG_INFINITY, |r| r.1);
                if let Some(rolled) = self.better_rolled(frame, &seed, to_beat) {
                    return Some(rolled);
                }
                // Turning the crop did not help: the face is not merely rolled.
                self.roll_retry_ts = timestamp;
            }
            if let Some((points, logit)) = result {
                return Some((points, seed, logit));
            }
        }
        if holding {
            self.unreadable = Some(self.seed(&ranked[0]));
        }
        None
    }

    /// `seed` rolled by [`ROLL_RETRY_DEG`] either way, if that looks more like a face.
    ///
    /// Both rotations are run; the one with the higher presence logit is returned (landmarks,
    /// crop, logit) when it beats `logit_to_beat` (the unrotated seed's). Passing the presence
    /// threshold is not enough: a crop 50-100° off can pass with a confident, wrongly rolled fit,
    /// but measured on rolled photos it scores a logit of about 2-14 against 18-28 for the right
    /// orientation.
    ///
    /// A winner below [`ROLL_RETRY_BELOW_LOGIT`] is doubtful too: a face turned further than the
    /// retry turned the crop (still about 45° off), or no face at all. A face the network rejects
    /// upright (a hand over the mouth) passed turned with a logit of 2-4 and a fit 30-45° off in
    /// roll and yaw. One more pass on the crop its landmarks ask for tells them apart: the former
    /// then reads confidently, the latter was rejected.
    fn better_rolled(
        &mut self,
        frame: &Frame,
        seed: &Roi,
        mut logit_to_beat: f64,
    ) -> Option<(Vec<[f64; 2]>, Roi, f64)> {
        let mut best: Option<(Vec<[f64; 2]>, Roi, f64)> = None;
        for turn in [ROLL_RETRY_DEG, -ROLL_RETRY_DEG] {
            let crop = Roi {
                angle: seed.angle + turn,
                ..*seed
            };
            if let Some((points, logit)) = self.infer(frame, &crop)
                && logit > logit_to_beat
            {
                best = Some((points, crop, logit));
                logit_to_beat = logit;
            }
        }
        let best = best?;
        if best.2 >= ROLL_RETRY_BELOW_LOGIT {
            return Some(best);
        }
        let wanted = roi_from_landmarks(&best.0);
        let follow = if wanted.valid(frame.width, frame.height) {
            self.infer(frame, &wanted)
        } else {
            None
        };
        match follow {
            Some((points, logit)) if logit >= ROLL_RETRY_BELOW_LOGIT => {
                Some((points, wanted, logit))
            }
            _ => None,
        }
    }

    /// Re-run the network until its crop sits on the face.
    ///
    /// Returns the landmarks, whether their crop settled and the presence logit of the pass that
    /// produced them.
    ///
    /// `points` came from the crop `used` with the presence logit `logit`. One pass on a crop
    /// chosen from the previous frame follows a face that moved by up to about 10 % of the crop
    /// almost completely, but after a larger jump (a posture shift between two frames at the idle
    /// frame rates) the landmarks stay biased towards the old position, upward jumps especially,
    /// while the network may still report a face. So:
    ///
    /// 1. If the crop the landmarks ask for is close to `used` (within [`ROI_SETTLE_SHIFT`], or
    ///    within [`ROI_CATCH_UP_SHIFT`] with a confident presence logit, see
    ///    [`crop_caught_up`]), they are final for this frame. This is the steady state, also for
    ///    a steadily moving head: no extra cost.
    /// 2. Otherwise one more pass on that crop, which suffices for moderate jumps (about 10 ms).
    /// 3. Still not caught up, or that crop was rejected: a large jump, which the network follows
    ///    only a fraction of the way per pass. With `reseed` (tracking), YuNet finds the face
    ///    independently of the old crop (the detection nearest the wanted crop, about 7 ms unless
    ///    this frame already ran one) and the network runs on its crop and once more on the crop
    ///    its landmarks ask for. Without (the face was just found by YuNet), one more plain pass.
    ///
    /// Each step keeps the previous landmarks when the network rejects the new crop, and a
    /// rejected crop is not tried twice. The crop counts as settled when the last pass's crop
    /// matched within the stricter [`ROI_SETTLE_SHIFT`] (the result as a whole, see `settled`,
    /// only when that pass was also confident, see [`CONFIDENT_LOGIT`]).
    #[allow(clippy::too_many_arguments)]
    fn settle(
        &mut self,
        frame: &Frame,
        mut points: Vec<[f64; 2]>,
        mut used: Roi,
        mut logit: f64,
        detections: Option<&[Detection]>,
        reseed: bool,
    ) -> (Vec<[f64; 2]>, bool, f64) {
        let (width, height) = (frame.width, frame.height);
        let mut wanted = roi_from_landmarks(&points);
        if crop_caught_up(&used, &wanted, logit) || !wanted.valid(width, height) {
            let settled = crop_settled(&used, &wanted);
            return (points, settled, logit);
        }
        let second = self.infer(frame, &wanted);
        let last_pass_possible = second.is_some();
        match second {
            Some((p, l)) => {
                (points, used, logit) = (p, wanted, l);
                wanted = roi_from_landmarks(&points);
                if crop_caught_up(&used, &wanted, logit) || !wanted.valid(width, height) {
                    let settled = crop_settled(&used, &wanted);
                    return (points, settled, logit);
                }
            }
            None if !reseed => return (points, false, logit),
            None => {}
        }
        // Whether `wanted` still deserves a pass (the network just rejected it if not).
        let mut last_pass = last_pass_possible;
        if reseed {
            let fresh;
            let detections = match detections {
                Some(d) => d,
                None => {
                    fresh = self.detect(frame, DETECT_WIDTH);
                    &fresh[..]
                }
            };
            let nearby: Vec<&Detection> =
                detections.iter().filter(|f| in_crop(f, &wanted)).collect();
            let distance = |d: &Detection| {
                let (cx, cy) = det_centre(d);
                (cx - wanted.cx).hypot(cy - wanted.cy)
            };
            let mut nearest: Option<&Detection> = None;
            for face in nearby {
                if nearest.is_none_or(|n| distance(face) < distance(n)) {
                    nearest = Some(face);
                }
            }
            if let Some(face) = nearest {
                let seed = self.seed(face);
                let third = if seed.valid(width, height) {
                    self.infer(frame, &seed)
                } else {
                    None
                };
                if let Some((p, l)) = third {
                    (points, used, logit) = (p, seed, l);
                    wanted = roi_from_landmarks(&points);
                    last_pass = true;
                }
            }
        }
        if last_pass
            && wanted.valid(width, height)
            && let Some((p, l)) = self.infer(frame, &wanted)
        {
            (points, used, logit) = (p, wanted, l);
        }
        let settled = crop_settled(&used, &roi_from_landmarks(&points));
        (points, settled, logit)
    }

    /// Landmarks (frame pixels) and face-presence logit for a crop, or `None`.
    ///
    /// `None` when the presence probability is below [`PRESENCE_THRESHOLD`]. The logit is
    /// returned rather than the probability because it keeps ranking crops where the probability
    /// has long saturated at 1.0 (see `acquire`).
    fn infer(&mut self, frame: &Frame, roi: &Roi) -> Option<Inferred> {
        #[cfg(test)]
        {
            self.hooks.infer_calls += 1;
            if let Some(mut hook) = self.hooks.infer.take() {
                let result = hook(frame, roi, &mut |f, r| self.run_net(f, r));
                self.hooks.infer = Some(hook);
                return result;
            }
        }
        self.run_net(frame, roi)
    }

    fn run_net(&self, frame: &Frame, roi: &Roi) -> Option<Inferred> {
        let net = self.net.as_ref()?;
        let m = roi.transform();
        let crop = warp_affine(frame, &m, INPUT_SIZE);
        // blobFromImage(crop, 1 / 255, swapRB=True), laid out NHWC as the network expects.
        let scale = (1.0f64 / 255.0) as f32;
        let input: Tensor =
            tract_ndarray::Array4::from_shape_fn((1, INPUT_SIZE, INPUT_SIZE, 3), |(_, y, x, c)| {
                f32::from(crop[(y * INPUT_SIZE + x) * 3 + 2 - c]) * scale
            })
            .into();
        let (raw, flag) = match net.forward(input) {
            Ok(out) => out,
            Err(err) => {
                log::debug!("Landmark inference failed: {err:#}");
                return None;
            }
        };
        let logit = f64::from(flag);
        if logit.is_nan() || sigmoid(logit) < PRESENCE_THRESHOLD {
            return None;
        }
        let inverse = invert_affine(&m);
        let points: Vec<[f64; 2]> = raw
            .chunks(3)
            .take(NUM_LANDMARKS)
            .map(|p| apply(&inverse, [f64::from(p[0]), f64::from(p[1])]))
            .collect();
        if points.len() != NUM_LANDMARKS || !points.iter().flatten().all(|v| v.is_finite()) {
            return None;
        }
        Some((points, logit))
    }

    /// YuNet detections (rows in frame pixels) with score >= `DETECTION_SCORE`.
    fn detect(&mut self, frame: &Frame, input_width: usize) -> Vec<Detection> {
        #[cfg(test)]
        self.hooks.detect_widths.push(input_width);
        let (width, height) = (frame.width, frame.height);
        let scale = (input_width as f64 / width as f64).min(1.0);
        let resized;
        let mut image = frame;
        if scale < 1.0 {
            let small_h = ((height as f64 * scale).round_ties_even() as usize).max(1);
            resized = resize_area(frame, input_width, small_h);
            image = &resized;
        }
        if input_width == GUARD_DETECT_WIDTH && self.guard_detector.is_none() {
            match create_yunet(self.detector_model.as_deref(), GUARD_DETECT_WIDTH) {
                Ok(detector) => self.guard_detector = Some(detector),
                Err(err) => {
                    log::warn!("Cannot create the shoulder guard's detector: {err}");
                    return Vec::new();
                }
            }
        }
        let detector = if input_width == GUARD_DETECT_WIDTH {
            self.guard_detector.as_mut()
        } else {
            self.detector.as_mut()
        };
        let Some(detector) = detector else {
            return Vec::new();
        };
        let size = (image.width, image.height);
        if self.detector_sizes.get(&input_width) != Some(&size) {
            if let Err(err) = detector.set_input_size(size) {
                log::warn!("YuNet cannot run at {}x{}: {err}", size.0, size.1);
                return Vec::new();
            }
            self.detector_sizes.insert(input_width, size);
        }
        let rows = match detector.detect(image) {
            Ok(rows) => rows,
            Err(err) => {
                log::warn!("YuNet detection failed: {err}");
                return Vec::new();
            }
        };
        rows.iter()
            .filter(|row| row[14] >= DETECTION_SCORE)
            .map(|row| {
                let mut face: Detection = row.map(f64::from);
                face[..14].iter_mut().for_each(|v| *v /= scale);
                face
            })
            .collect()
    }

    /// Detect every face at the guard's resolution and note when.
    fn guard_count(&mut self, frame: &Frame, timestamp: f64) -> Vec<Detection> {
        let faces = self.detect(frame, GUARD_DETECT_WIDTH);
        self.guard_ts = timestamp;
        faces
    }

    /// Crop of a clearly stronger face elsewhere than the tracked `roi`, if any.
    ///
    /// Faces are compared by [`primary_score`] and must beat the tracked face by
    /// [`PRIMARY_SWITCH_RATIO`]. Like is compared with like: the tracked face's size is YuNet's
    /// box for it when this frame's detections include it, and otherwise its landmark box
    /// converted to YuNet's scale ([`YUNET_TO_MESH_SIDE`]). Comparing YuNet's box (about 13 %
    /// larger) with the landmark box would let a face only 1.15x the user's take over.
    fn primary_candidate(
        &self,
        roi: &Roi,
        detections: &[Detection],
        width: usize,
        height: usize,
    ) -> Option<Roi> {
        let others: Vec<Detection> = detections
            .iter()
            .filter(|f| !in_crop(f, roi))
            .copied()
            .collect();
        if others.is_empty() {
            return None;
        }
        let tracked: Vec<&Detection> = detections.iter().filter(|f| in_crop(f, roi)).collect();
        let tracked_score = if tracked.is_empty() {
            let mesh_side = roi.side / ROI_SCALE;
            primary_score(
                mesh_side * YUNET_TO_MESH_SIDE,
                roi.cx,
                roi.cy,
                width,
                height,
            )
        } else {
            tracked
                .iter()
                .map(|f| det_score(f, width, height))
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let best = best_by(&others, |d| det_score(d, width, height))?;
        if det_score(best, width, height) <= PRIMARY_SWITCH_RATIO * tracked_score {
            return None;
        }
        let seed = self.seed(best);
        seed.valid(width, height).then_some(seed)
    }

    /// Faces besides the primary one, as normalised boxes.
    ///
    /// A fresh guard count (`counted`, from this very frame) is split into the primary face and
    /// the others, and the others are kept until the next count. They are matched against the
    /// primary face only here, where both come from the same frame: matching stale detections
    /// against a later face position would count the user twice after a quick head movement.
    fn other_faces(
        &mut self,
        counted: Option<&[Detection]>,
        primary_px: &[[f64; 2]],
        width: usize,
        height: usize,
    ) -> Vec<Box4> {
        if self.max_faces < 2 {
            return Vec::new();
        }
        let Some(counted) = counted else {
            return self.guard_others.clone();
        };
        let (x0, y0, x1, y1) = bounds(primary_px);
        let others: Vec<Box4> = counted
            .iter()
            .filter(|face| {
                let (cx, cy) = det_centre(face);
                !(x0 <= cx && cx <= x1 && y0 <= cy && cy <= y1)
            })
            .map(|face| det_box(face, width, height))
            .collect();
        self.guard_others = others.clone();
        others
    }

    fn without_landmarks(
        &mut self,
        detections: &[Detection],
        timestamp: f64,
        width: usize,
        height: usize,
        started: Instant,
    ) -> Observation {
        self.roi = None;
        self.landmarks = None;
        // Best first, in the order the faces were tried as the primary face.
        let mut faces = detections.to_vec();
        faces.sort_by(|a, b| det_score(b, width, height).total_cmp(&det_score(a, width, height)));
        let boxes: Vec<Box4> = faces.iter().map(|f| det_box(f, width, height)).collect();
        self.last = Some(Drawing {
            timestamp,
            points: None,
            others: boxes.iter().skip(1).copied().collect(),
            rvec: None,
            tvec: None,
            camera: None,
        });
        // YuNet sees a face the landmark network rejects (a profile, heavy blur): someone is
        // there, but nothing usable for gaze.
        let mut obs = Observation::new(timestamp, faces.len().min(self.max_faces) as u32);
        obs.face_box = boxes.first().copied();
        obs.inference_ms = started.elapsed().as_secs_f64() * 1e3;
        obs.frame_size = (width as u32, height as u32);
        obs
    }

    fn with_landmarks(
        &mut self,
        points: Vec<[f64; 2]>,
        counted: Option<&[Detection]>,
        timestamp: f64,
        width: usize,
        height: usize,
        started: Instant,
    ) -> Observation {
        let (w, h) = (width as f64, height as f64);
        let mesh = &points[..MESH_POINTS];
        let (x0, y0, x1, y1) = bounds(mesh);
        let (x0, y0, x1, y1) = (x0 / w, y0 / h, x1 / w, y1 / h);
        let face_box = (
            x0.max(0.0),
            y0.max(0.0),
            (x1.min(1.0) - x0.max(0.0)).max(0.0),
            (y1.min(1.0) - y0.max(0.0)).max(0.0),
        );
        let others = self.other_faces(counted, mesh, width, height);
        let eyes = eye_metrics(&points);
        let pose = self.pose.estimate(&points, width, height);

        let mut obs = Observation::new(timestamp, (1 + others.len()).min(self.max_faces) as u32);
        if let (Some(eyes), Some(pose)) = (eyes, pose) {
            let vector = vec![
                pose.yaw,
                pose.pitch,
                pose.roll,
                pose.tx,
                pose.ty,
                pose.tz,
                eyes.iris_h,
                eyes.iris_v,
            ];
            if vector.iter().all(|v| v.is_finite()) {
                obs.features = Some(vector);
                let touches_border = x0 <= BORDER_MARGIN
                    || y0 <= BORDER_MARGIN
                    || x1 >= 1.0 - BORDER_MARGIN
                    || y1 >= 1.0 - BORDER_MARGIN;
                obs.quality = if touches_border { 0.5 } else { 1.0 };
                obs.blink = self.blink.update(eyes.openness);
            }
        }

        let normalised: Vec<[f64; 2]> = points.iter().map(|p| [p[0] / w, p[1] / h]).collect();
        self.landmarks = Some(normalised.clone());
        self.last = Some(Drawing {
            timestamp,
            points: Some(normalised),
            others,
            rvec: self.pose.rvec(),
            tvec: self.pose.tvec(),
            camera: Some(self.pose.camera),
        });
        obs.head_yaw = pose.map(|p| p.yaw);
        obs.head_pitch = pose.map(|p| p.pitch);
        obs.face_box = Some(face_box);
        obs.inference_ms = started.elapsed().as_secs_f64() * 1e3;
        obs.frame_size = (width as u32, height as u32);
        obs
    }

    fn draw_face(&self, out: &mut Frame, last: &Drawing, thickness: i32) {
        let Some(points) = &last.points else {
            return;
        };
        let (w, h) = (out.width as f64, out.height as f64);
        let pixels: Vec<[f64; 2]> = points.iter().map(|p| [p[0] * w, p[1] * h]).collect();
        let round = |p: [f64; 2]| (p[0].round_ties_even() as f32, p[1].round_ties_even() as f32);
        for contour in [EYE_A_CONTOUR, EYE_B_CONTOUR] {
            for (i, &a) in contour.iter().enumerate() {
                let b = contour[(i + 1) % contour.len()];
                thick_line(
                    out,
                    round(pixels[a]),
                    round(pixels[b]),
                    thickness,
                    COLOR_EYE,
                );
            }
        }
        for iris in [IRIS_A, IRIS_B] {
            let centre = pixels[iris[0]];
            let radius = iris[1..]
                .iter()
                .map(|&i| (pixels[i][0] - centre[0]).hypot(pixels[i][1] - centre[1]))
                .sum::<f64>()
                / 4.0;
            let c = (
                centre[0].round_ties_even() as i32,
                centre[1].round_ties_even() as i32,
            );
            let radius = (radius.round_ties_even() as i32).max(1);
            for r in radius - (thickness - 1) / 2..=radius + thickness / 2 {
                if r > 0 {
                    draw_hollow_circle_mut(out, c, r, rgb(COLOR_IRIS));
                }
            }
            draw_filled_circle_mut(out, c, thickness.max(1), rgb(COLOR_IRIS));
        }
        if let (Some(rvec), Some(tvec), Some(camera)) = (last.rvec, last.tvec, last.camera) {
            // Where the nose points: from the tip, 5 cm out of the face.
            let tip = Point3::from(self.nose_tip);
            let end = tip + Vector3::from(POSE_ARROW_CM);
            let ends = pnp::project_points(&[tip, end], &rvec, &tvec, &camera);
            let (a, b) = (ends[0], ends[1]);
            if [a.x, a.y, b.x, b.y].iter().all(|v| v.is_finite()) {
                thick_line(
                    out,
                    round([a.x, a.y]),
                    round([b.x, b.y]),
                    thickness + 1,
                    COLOR_POSE,
                );
            }
        }
    }
}

fn rgb(color: Color) -> image::Rgb<u8> {
    image::Rgb([color.0, color.1, color.2])
}

/// A line `thickness` pixels wide (round caps), like `cv2.line`.
fn thick_line(out: &mut Frame, a: (f32, f32), b: (f32, f32), thickness: i32, color: Color) {
    if thickness <= 1 {
        draw_line_segment_mut(out, a, b, rgb(color));
        return;
    }
    let n = ((b.0 - a.0).hypot(b.1 - a.1).ceil() as usize).max(1);
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let p = (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
        draw_filled_circle_mut(
            out,
            (p.0.round() as i32, p.1.round() as i32),
            thickness / 2,
            rgb(color),
        );
    }
}

impl VisionBackend for FaceMeshBackend {
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
        let n = clamp_faces(n);
        if n == self.max_faces {
            return;
        }
        self.max_faces = n;
        // Count again on the next frame with the new setting.
        self.guard_others.clear();
        self.guard_ts = f64::NEG_INFINITY;
    }

    /// `false` while the tracking crop is still catching up with a moved face.
    ///
    /// A frame re-runs the network within itself only after a large jump (see `settle`); after a
    /// moderate move, or a jump that a few passes could not quite follow, the crop the landmarks
    /// ask for is still off by more than [`ROI_SETTLE_SHIFT`] / [`ROI_SETTLE_SCALE`] and the next
    /// frame continues from it. The remaining bias is small, but the motion gate must not repeat
    /// it for seconds. Nor a doubtful fit (a presence logit below [`CONFIDENT_LOGIT`]), which the
    /// next pass may reject. `true` again after [`MAX_UNSETTLED_FRAMES`] unsettled frames in a
    /// row.
    fn settled(&self) -> bool {
        self.settled
    }

    /// Forget the tracked face; the next frame looks for faces afresh.
    ///
    /// The vision worker calls this when the camera was released (pause, privacy mode, a lock)
    /// or replaced: the picture may have changed completely meanwhile, and a crop left over from
    /// before could latch onto whatever face is there now. The blink baseline is kept, it belongs
    /// to the user rather than to a camera session.
    fn reset(&mut self) {
        self.forget_tracking();
        self.last = None;
        self.landmarks = None;
    }

    fn close(&mut self) {
        self.closed = true;
        self.net = None;
        self.detector = None;
        self.guard_detector = None;
        self.forget_tracking();
        self.last = None;
        self.landmarks = None;
        self.blink.reset();
    }

    /// Panics after `close()` (the Python raised `RuntimeError`).
    fn process(&mut self, frame: &Frame, timestamp: f64) -> Observation {
        if self.closed || self.net.is_none() {
            panic!("FaceMeshBackend is closed");
        }
        let started = Instant::now();
        let (width, height) = (frame.width, frame.height);
        if self.frame_size != Some((width, height)) {
            self.frame_size = Some((width, height));
            self.forget_tracking();
        }
        let guard = self.max_faces >= 2;

        // YuNet rows found in this very frame, if any search ran (reused below so that no frame
        // runs the detector twice at the same resolution).
        let mut detections: Option<Vec<Detection>> = None;
        // Shoulder guard: a fresh face count on this frame, if one is due.
        let mut counted: Option<Vec<Detection>> = None;
        if guard && !(0.0..GUARD_PERIOD_S).contains(&(timestamp - self.guard_ts)) {
            let faces = self.guard_count(frame, timestamp);
            counted = Some(faces.clone());
            detections = Some(faces);
        }

        // Landmarks, the crop they came from and the pass's face-presence logit.
        let mut found: Option<(Vec<[f64; 2]>, Roi, f64)> = None;
        let roi = self.roi;
        let mut tracking = roi.is_some();
        if let Some(roi) = roi {
            // With the guard on, its counts double as primary-face checks; this only stays false
            // for a while after a stronger face was rejected.
            let check = self.primary_check_due(timestamp);
            if check && detections.is_none() && !guard {
                detections = Some(self.detect(frame, DETECT_WIDTH));
                self.schedule_primary_check(timestamp, PRIMARY_CHECK_S);
            }
            let candidate = match (&detections, check) {
                (Some(d), true) => self.primary_candidate(&roi, d, width, height),
                _ => None,
            };
            if let Some(candidate) = candidate {
                // A clearly stronger face elsewhere (the user came back): it takes over if the
                // network reads it confidently; otherwise (a profile, a hand over it, half out of
                // the picture) the tracked face stays rather than being lost to it, and the next
                // try waits.
                match self.infer(frame, &candidate) {
                    Some((points, logit)) if logit >= CONFIDENT_LOGIT => {
                        self.pose.reset();
                        found = Some((points, candidate, logit));
                    }
                    _ => self.schedule_primary_check(timestamp, PRIMARY_CHECK_S),
                }
            }
            if found.is_none()
                && let Some((points, logit)) = self.infer(frame, &roi)
            {
                found = Some((points, roi, logit));
            }
        }
        if found.is_none() {
            // Lost (or never had) the face: find one. With the guard on, the search doubles as a
            // fresh face count.
            tracking = false;
            self.pose.reset();
            if detections.is_none() {
                if guard {
                    let faces = self.guard_count(frame, timestamp);
                    counted = Some(faces.clone());
                    detections = Some(faces);
                } else {
                    detections = Some(self.detect(frame, DETECT_WIDTH));
                }
            }
            if !guard {
                // The search chose the primary face; look again soon, the user may be on the way
                // into the picture.
                self.schedule_primary_check(timestamp, PRIMARY_RECHECK_S);
            }
            // The face tracked until now (lost on this very frame) or the one that was already
            // unreadable before.
            let lost = roi.or(self.unreadable);
            let faces = detections.as_deref().unwrap_or_default();
            found = self.acquire(frame, faces, timestamp, lost);
        }

        let Some((points, used, logit)) = found else {
            self.note_settled(true);
            let faces = detections.unwrap_or_default();
            return self.without_landmarks(&faces, timestamp, width, height, started);
        };
        let (points, settled, logit) =
            self.settle(frame, points, used, logit, detections.as_deref(), tracking);
        // A doubtful fit (a hand coming over the face) may not survive the next pass on the very
        // same picture, so the motion gate must not repeat it.
        self.note_settled(settled && logit >= CONFIDENT_LOGIT);
        let roi = roi_from_landmarks(&points);
        self.roi = roi.valid(width, height).then_some(roi);
        self.with_landmarks(
            points,
            counted.as_deref(),
            timestamp,
            width,
            height,
            started,
        )
    }

    fn annotate(&self, frame_bgr: &Frame, observation: &Observation) -> Frame {
        let mut out = frame_bgr.clone();
        let thickness = line_thickness(&out);
        if let Some(last) = &self.last
            && last.timestamp == observation.timestamp
        {
            self.draw_face(&mut out, last, thickness);
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

#[cfg(test)]
#[path = "facemesh_backend_tests.rs"]
mod tests;
