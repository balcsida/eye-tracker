//! Tests for the apple backend.
//!
//! The conversions, the iris and blink features and the device choice are tested on hand-made
//! [`VisionFace`]s, without Vision. The tests that call Vision run on every Mac; the shared
//! behaviour (blank frames, odd sizes, `close`) is tested in `backends_tests.rs` for every
//! backend.
//!
//! Not tested:
//! - The yaw and pitch signs: Vision reads the drawn test faces only at some sizes and gives
//!   them a yaw of 4-19° although they are frontal, so only the roll sign is checked on them.
//!   `test_real_face_agrees_with_facemesh` compares all angles on a photo (it returns early
//!   without `EYE_TRACKER_TEST_FACE`).
//! - Which device Vision picks: that depends on the Mac (the log line is checked, not its text
//!   after the prefix).

use super::*;
use crate::testing::capture_logs;
use crate::testing::face_drawing::{FaceParams, draw_face, rotate};
use crate::vision::backends::FaceMeshBackend;
use crate::vision::backends::tests::{camera_like, face_image, load};

const WIDTH: usize = 640;
const HEIGHT: usize = 480;
/// A face box (Vision's convention: normalised, origin bottom left).
const BOX: Box4 = (0.25, 0.25, 0.5, 0.5);

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

/// Pixels (y down) back to Vision's box-normalised points (y up): the inverse of `to_pixels`.
fn normalised(points: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let (x, y, w, h) = BOX;
    points
        .iter()
        .map(|p| {
            [
                (p[0] / WIDTH as f64 - x) / w,
                (1.0 - p[1] / HEIGHT as f64 - y) / h,
            ]
        })
        .collect()
}

/// A six-point eye contour like Vision's: corners at `x0` and `x0 + 40`, lids `gap` apart.
fn contour(x0: f64, gap: f64) -> Vec<[f64; 2]> {
    let y = 200.0;
    normalised(&[
        [x0, y],
        [x0 + 12.0, y - gap / 2.0],
        [x0 + 28.0, y - gap / 2.0 + 1.0],
        [x0 + 40.0, y],
        [x0 + 28.0, y + gap / 2.0 - 1.0],
        [x0 + 12.0, y + gap / 2.0],
    ])
}

/// Two eyes 80 px apart, both pupils 24 px from the image-left corner and 1 px low.
/// Vision's "left" eye is the one on the image right here, as for a subject facing the camera.
fn eyes(gap: f64) -> VisionEyes {
    let pupils = normalised(&[[284.0, 201.0], [364.0, 201.0]]);
    VisionEyes {
        left_eye: contour(340.0, gap),
        right_eye: contour(260.0, gap),
        left_pupil: pupils[1],
        right_pupil: pupils[0],
    }
}

fn face(bounding_box: Box4, eyes: Option<VisionEyes>) -> VisionFace {
    VisionFace {
        bounding_box,
        confidence: 0.9,
        roll: Some(0.05),
        yaw: Some(0.1),
        pitch: Some(-0.2),
        eyes,
    }
}

fn backend() -> AppleBackend {
    AppleBackend::new(1).unwrap()
}

// ---------------------------------------------------------------- conversions
#[test]
fn test_image_box_flips_y() {
    let (x, y, w, h) = image_box((0.1, 0.2, 0.3, 0.4));
    assert!(approx(x, 0.1, 1e-12) && approx(y, 0.4, 1e-12));
    assert!(approx(w, 0.3, 1e-12) && approx(h, 0.4, 1e-12));
    // Clipped to the image on every side.
    let (x, y, w, h) = image_box((-0.1, 0.9, 0.3, 0.3));
    assert!(approx(x, 0.0, 1e-12) && approx(y, 0.0, 1e-12));
    assert!(approx(w, 0.2, 1e-12) && approx(h, 0.1, 1e-12));
}

#[test]
fn test_to_pixels_maps_the_box_corners() {
    let bounding_box = (0.25, 0.5, 0.5, 0.25);
    assert_eq!(
        to_pixels([0.0, 0.0], bounding_box, 640, 480),
        [160.0, 240.0]
    );
    assert_eq!(
        to_pixels([1.0, 1.0], bounding_box, 640, 480),
        [480.0, 120.0]
    );
}

#[test]
fn test_angles_are_degrees() {
    let (yaw, pitch, roll) = head_angles(&face(BOX, None)).unwrap();
    assert!(approx(yaw, 0.1f64.to_degrees(), 1e-12));
    assert!(approx(pitch, -0.2f64.to_degrees(), 1e-12));
    assert!(approx(roll, 0.05f64.to_degrees(), 1e-12));
    let no_pitch = VisionFace {
        pitch: None,
        ..face(BOX, None)
    };
    assert_eq!(head_angles(&no_pitch), None);
}

// ------------------------------------------------------------- eyes and blink
#[test]
fn test_iris_features_use_the_facemesh_formula() {
    let metrics = vision_eye_metrics(&eyes(16.0), BOX, WIDTH, HEIGHT).unwrap();
    // The same five points per eye, in facemesh's order, through facemesh's function.
    let eye = |x0: f64, pupil: [f64; 2]| {
        [
            [x0, 200.0],
            [x0 + 40.0, 200.0],
            pupil,
            [x0 + 12.0, 192.0],
            [x0 + 12.0, 208.0],
        ]
    };
    let expected = eye_metrics_of([eye(260.0, [284.0, 201.0]), eye(340.0, [364.0, 201.0])]);
    let expected = expected.unwrap();
    assert!(approx(metrics.iris_h, expected.iris_h, 1e-9), "{metrics:?}");
    assert!(approx(metrics.iris_v, expected.iris_v, 1e-9), "{metrics:?}");
    assert!(
        approx(metrics.openness, expected.openness, 1e-9),
        "{metrics:?}"
    );
    assert!(approx(metrics.iris_h, 0.6, 1e-9));
    assert!(approx(metrics.iris_v, 0.025, 1e-9));
    assert!(approx(metrics.openness, 0.4, 1e-9));
}

#[test]
fn test_degenerate_eyes_give_no_features() {
    let mut flat = eyes(16.0);
    flat.left_eye.truncate(2);
    assert_eq!(vision_eye_metrics(&flat, BOX, WIDTH, HEIGHT), None);
    let mut on_top = eyes(16.0);
    on_top.left_eye = on_top.right_eye.clone();
    assert_eq!(vision_eye_metrics(&on_top, BOX, WIDTH, HEIGHT), None);
}

#[test]
fn test_closed_lids_are_a_blink() {
    let mut backend = backend();
    let open = backend.observe(&[face(BOX, Some(eyes(16.0)))], 1.0, WIDTH, HEIGHT);
    assert!(open.usable() && !open.blink);
    let closed = backend.observe(&[face(BOX, Some(eyes(2.0)))], 2.0, WIDTH, HEIGHT);
    assert!(closed.blink); // 2 px of a 40 px eye: below BLINK_OPENNESS
    assert!(!closed.usable());
}

// ---------------------------------------------------------------- observation
#[test]
fn test_observation_of_the_primary_face() {
    let mut backend = backend();
    let small = face((0.05, 0.6, 0.1, 0.1), None);
    let obs = backend.observe(&[small, face(BOX, Some(eyes(16.0)))], 3.0, WIDTH, HEIGHT);
    assert_eq!(obs.timestamp, 3.0);
    assert_eq!(obs.face_count, 2); // every face, for the shoulder guard
    assert_eq!(obs.face_box, Some(image_box(BOX)));
    assert_eq!(obs.frame_size, (640, 480));
    assert_eq!(obs.quality, 0.9);
    let features = obs.features.unwrap();
    assert_eq!(features.len(), AppleBackend::FEATURE_NAMES.len());
    assert!(approx(features[0], 0.1f64.to_degrees(), 1e-9)); // yaw
    assert!(approx(features[3], 0.6, 1e-9)); // iris_h
    assert_eq!(obs.head_yaw, Some(features[0]));
    assert_eq!(obs.head_pitch, Some(features[1]));
}

#[test]
fn test_face_without_landmarks_counts_but_has_no_features() {
    let obs = backend().observe(&[face(BOX, None)], 1.0, WIDTH, HEIGHT);
    assert_eq!(obs.face_count, 1);
    assert!(obs.face_box.is_some() && obs.head_yaw.is_some());
    assert!(obs.features.is_none() && !obs.usable());
}

#[test]
fn test_face_at_the_border_has_lower_quality() {
    // The eye points are relative to the box, so they move with it.
    let cut = face((0.0, 0.25, 0.5, 0.5), Some(eyes(16.0)));
    let obs = backend().observe(&[cut], 1.0, WIDTH, HEIGHT);
    assert!(obs.usable());
    assert_eq!(obs.quality, 0.5);
}

#[test]
fn test_no_face() {
    let obs = backend().observe(&[], 1.0, WIDTH, HEIGHT);
    assert_eq!(obs.face_count, 0);
    assert!(obs.face_box.is_none() && obs.features.is_none());
}

// ------------------------------------------------------------- compute device
#[test]
fn test_device_choice_prefers_the_neural_engine() {
    use ComputeDevice::{NeuralEngine, Other};
    assert_eq!(choose_device(&[Other, NeuralEngine, Other]), Some(1));
    assert_eq!(choose_device(&[Other, Other]), None);
    assert_eq!(choose_device(&[]), None);
}

#[test]
fn test_the_chosen_device_is_logged() {
    let logs = capture_logs();
    let _backend = backend();
    let lines: Vec<String> = logs
        .records()
        .into_iter()
        .filter(|(level, m)| *level == log::Level::Info && m.starts_with("apple backend:"))
        .map(|(_, m)| m)
        .collect();
    // macOS 14 and later: one line; before, nothing.
    assert!(lines.len() <= 1, "{lines:?}");
    for line in lines {
        assert!(line.starts_with("apple backend: landmarks on "), "{line}");
    }
}

// --------------------------------------------------------------------- Vision
#[test]
fn test_metadata() {
    let backend = backend();
    assert_eq!(backend.name(), "apple");
    assert_eq!(backend.feature_version(), "vision-pose-pupil-1");
    assert_eq!(
        backend.feature_names(),
        ["yaw", "pitch", "roll", "iris_h", "iris_v"]
    );
    assert_eq!(backend.gaze_features(), FaceMeshBackend::GAZE_FEATURES);
}

#[test]
fn test_a_vision_failure_degrades_and_warns_once() {
    let logs = capture_logs();
    let mut backend = backend();
    let empty = Frame::filled(0, 0, [0; 3]); // no CGImage can hold it
    for ts in [1.0, 2.0] {
        let obs = backend.process(&empty, ts);
        assert_eq!(obs.face_count, 0);
        assert!(obs.features.is_none());
    }
    assert_eq!(logs.messages_at_least(log::Level::Warn).len(), 1);
}

/// Vision reads the drawn face at this size (not at every size: it is trained on photos).
fn drawn_face() -> Frame {
    draw_face(
        None,
        &FaceParams {
            scale: 0.35,
            ..FaceParams::default()
        },
    )
}

#[test]
fn test_drawn_face_roll_sign() {
    let mut backend = backend();
    let ccw = backend.process(&rotate(&drawn_face(), 20.0), 1.0);
    let cw = backend.process(&rotate(&drawn_face(), -20.0), 2.0);
    let (Some(ccw), Some(cw)) = (ccw.features, cw.features) else {
        eprintln!("Vision found no drawn face on this Mac; skipping");
        return;
    };
    // Counter-clockwise positive, like facemesh (which reads +20 and -19 here).
    assert!(ccw[2] > 5.0, "{ccw:?}");
    assert!(cw[2] < -5.0, "{cw:?}");
}

#[test]
fn test_drawn_face_annotation() {
    let mut backend = backend();
    let frame = drawn_face();
    let obs = backend.process(&frame, 1.0);
    if obs.features.is_none() {
        eprintln!("Vision found no drawn face on this Mac; skipping");
        return;
    }
    let out = backend.annotate(&frame, &obs);
    let changed = out
        .data
        .chunks(3)
        .zip(frame.data.chunks(3))
        .filter(|(a, b)| a != b)
        .count();
    assert!(changed > 200, "{changed}");
}

/// Yaw and pitch within a few degrees, the iris within 0.1, on a photo.
#[test]
fn test_real_face_agrees_with_facemesh() {
    let Some(path) = face_image() else { return };
    let frame = camera_like(&load(&path), 640);
    let apple = backend().process(&frame, 1.0);
    let mut facemesh = FaceMeshBackend::new(1, false).unwrap();
    let mut reference = facemesh.process(&frame, 1.0);
    for i in 0..3 {
        reference = facemesh.process(&frame, 1.1 + 0.1 * f64::from(i));
    }
    let (a, f) = (apple.features.unwrap(), reference.features.unwrap());
    // apple: yaw, pitch, roll, iris_h, iris_v; facemesh: yaw, pitch, roll, tx, ty, tz, iris_h,
    // iris_v.
    eprintln!(
        "apple yaw {:.1} pitch {:.1} iris {:.3} {:.3} ({:.1} ms); facemesh yaw {:.1} pitch {:.1} \
         iris {:.3} {:.3} ({:.1} ms)",
        a[0], a[1], a[3], a[4], apple.inference_ms, f[0], f[1], f[6], f[7], reference.inference_ms
    );
    assert!(approx(a[0], f[0], 5.0), "yaw {} vs {}", a[0], f[0]);
    assert!(approx(a[1], f[1], 5.0), "pitch {} vs {}", a[1], f[1]);
    assert!(approx(a[3], f[6], 0.1), "iris_h {} vs {}", a[3], f[6]);
    assert!(approx(a[4], f[7], 0.1), "iris_v {} vs {}", a[4], f[7]);
}
