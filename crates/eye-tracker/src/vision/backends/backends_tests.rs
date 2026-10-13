//! Tests for backend selection, the model files and the behaviour shared by all backends.
//!
//! The face-mesh backend's own logic is tested in `facemesh_backend_tests.rs`. Tests here run on
//! drawn cartoon faces (see `testing::face_drawing`), so they need no photo; the few that need a
//! real face use [`face_image`] and return early unless `EYE_TRACKER_TEST_FACE` points at a photo
//! with one frontal face.
//!
//! Adapted or not ported:
//! - `test_available_backends_needs_every_model` monkeypatched `paths.model_path`; the models are
//!   embedded now, so the same rule is tested through `available_among` / `class_among`.
//! - `test_auto_falls_back_to_lite`, `test_auto_reports_all_failures` and
//!   `test_explicit_backend_does_not_fall_back` monkeypatched the constructors; they inject the
//!   failing constructor through `create_with`.
//! - `test_mediapipe_runtime_is_gone` / `test_mediapipe_is_not_a_dependency` check the Rust
//!   sources and `Cargo.toml` instead of `sys.modules` and `pyproject.toml`.
//! - `test_fetch_models_pins_the_same_checksums` checks `scripts/fetch_models.py` (the spec of
//!   `cargo xtask fetch-models`, which is not ported yet) instead of importing it.
//! - `test_other_frame_formats`: the `gray` and `bgra` cases do not exist in Rust, where a
//!   [`Frame`] is always BGR; `large` and `tiny` are ported.
//! - `test_close_is_idempotent_and_final` expects a panic where Python raised `RuntimeError`.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::*;
use crate::testing::face_drawing::{FaceParams, draw_face, two_faces};
use crate::types::Observation;
use crate::vision::Frame;
use crate::vision::backends::lite_backend::geometry_features;
use crate::vision::backends::yunet::{Face, resize_area};

/// Path to a photo containing one frontal face (the `face_image` fixture).
///
/// Face photos are not committed to the repository. Set `EYE_TRACKER_TEST_FACE` to an image path
/// to enable tests that need a real face; otherwise they return early.
pub(crate) fn face_image() -> Option<PathBuf> {
    let value = std::env::var_os("EYE_TRACKER_TEST_FACE")?;
    let path = PathBuf::from(value);
    if path.is_file() {
        Some(path)
    } else {
        eprintln!("set EYE_TRACKER_TEST_FACE to a face photo to run this test");
        None
    }
}

/// Decode a photo (from memory: any path).
pub(crate) fn load(path: &Path) -> Frame {
    let image = image::open(path).unwrap_or_else(|err| panic!("could not decode {path:?}: {err}"));
    Frame::from_rgb_image(&image.to_rgb8())
}

/// The photo scaled down so that its longer side is at most `longest` (INTER_AREA).
pub(crate) fn camera_like(image: &Frame, longest: usize) -> Frame {
    let scale = (longest as f64 / image.width.max(image.height) as f64).min(1.0);
    let size = |v: usize| (v as f64 * scale).round_ties_even() as usize;
    resize_area(image, size(image.width), size(image.height))
}

/// One instance of every available backend (the parametrised `backend` fixture).
fn each_backend(test: impl Fn(&mut dyn VisionBackend)) {
    for name in available_backends() {
        let mut backend = create_backend(name, 1).unwrap();
        test(backend.as_mut());
        backend.close();
    }
}

/// Process a frame a few times (trackers settle) and return the last result.
fn run(backend: &mut dyn VisionBackend, frame: &Frame, start: f64, n: usize) -> Observation {
    let mut obs = None;
    for i in 0..n {
        obs = Some(backend.process(frame, start + 0.05 * i as f64));
    }
    obs.unwrap()
}

fn face_params(cx: i32) -> FaceParams {
    FaceParams {
        cx,
        ..FaceParams::default()
    }
}

// ------------------------------------------------------------------ selection
#[test]
fn test_both_backends_are_available_with_the_committed_models() {
    assert_eq!(available_backends(), ["facemesh", "lite"]);
}

#[test]
fn test_available_backends_order() {
    assert_eq!(BACKEND_NAMES, ["facemesh", "lite"]);
    let available = available_backends();
    let ordered: Vec<&str> = BACKEND_NAMES
        .into_iter()
        .filter(|n| available.contains(n))
        .collect();
    assert_eq!(ordered, available);
}

#[test]
fn test_available_backends_needs_every_model() {
    let none = available_among(|_| false);
    assert!(none.is_empty());
    let err = class_among("auto", &none).unwrap_err();
    assert!(err.0.contains("No vision backend"), "{err}");
    // Only YuNet present: the lite backend is available, facemesh is not.
    let lite_only = available_among(|file| file == "face_detection_yunet_2023mar.onnx");
    assert_eq!(lite_only, ["lite"]);
    assert_eq!(class_among("auto", &lite_only).unwrap(), BackendClass::Lite);
}

#[test]
fn test_unknown_backend_raises() {
    let err = create_backend("tensorflow", 1).err().unwrap();
    assert!(err.0.contains("Unknown vision backend"), "{err}");
    let err = backend_class("tensorflow").unwrap_err();
    assert!(err.0.contains("auto, facemesh, lite"), "{err}");
}

#[test]
fn test_legacy_names_map_to_the_new_backends() {
    for (legacy, class) in [
        ("mediapipe", BackendClass::FaceMesh),
        ("opencv", BackendClass::Lite),
    ] {
        assert_eq!(backend_class(legacy).unwrap(), class);
        assert_eq!(backend_class(&legacy.to_uppercase()).unwrap(), class);
    }
}

#[test]
fn test_auto_prefers_facemesh() {
    let mut instance = create_backend("auto", 1).unwrap();
    assert_eq!(instance.name(), available_backends()[0]);
    assert_eq!(instance.name(), "facemesh");
    assert_eq!(backend_class("auto").unwrap().name(), instance.name());
    assert_eq!(
        backend_class(" AUTO ").unwrap().feature_version(),
        instance.feature_version()
    );
    instance.close();
}

#[test]
fn test_auto_falls_back_to_lite() {
    let instance = create_with("auto", 2, |name, max_faces| {
        if name == "facemesh" {
            Err(BackendUnavailable(
                "simulated landmark model failure".into(),
            ))
        } else {
            create(name, max_faces)
        }
    })
    .unwrap();
    assert_eq!(instance.name(), "lite");
}

#[test]
fn test_auto_reports_all_failures() {
    let broken = |name: &str, _: usize| -> Result<Box<dyn VisionBackend>, BackendUnavailable> {
        let class = if name == "facemesh" {
            "FaceMeshBackend"
        } else {
            "LiteBackend"
        };
        Err(BackendUnavailable(format!("{class} broken")))
    };
    let err = create_with("auto", 1, broken).err().unwrap();
    assert!(err.0.contains("FaceMeshBackend broken"), "{err}");
    assert!(err.0.contains("LiteBackend broken"), "{err}");
}

#[test]
fn test_explicit_backend_does_not_fall_back() {
    let err = create_with("facemesh", 1, |name, max_faces| {
        if name == "facemesh" {
            Err(BackendUnavailable("facemesh broken".into()))
        } else {
            create(name, max_faces)
        }
    })
    .err()
    .unwrap();
    assert!(err.0.contains("facemesh broken"), "{err}");
}

#[test]
fn test_lite_missing_model() {
    let dir = tempfile::tempdir().unwrap();
    let err = LiteBackend::with_model(Some(&dir.path().join("missing.onnx")), 1)
        .err()
        .unwrap();
    assert!(err.0.contains("model not found"), "{err}");
}

#[test]
fn test_backend_metadata() {
    assert_eq!(FaceMeshBackend::NAME, "facemesh");
    assert_eq!(FaceMeshBackend::FEATURE_VERSION, "facemesh-pose-iris-1");
    assert_eq!(
        FaceMeshBackend::FEATURE_NAMES,
        ["yaw", "pitch", "roll", "tx", "ty", "tz", "iris_h", "iris_v"]
    );
    assert_eq!(
        FaceMeshBackend::GAZE_FEATURES,
        ["yaw", "pitch", "iris_h", "iris_v"]
    );
    assert_eq!(LiteBackend::NAME, "lite");
    assert_eq!(LiteBackend::FEATURE_VERSION, "yunet-geom-1");
    assert_eq!(
        LiteBackend::FEATURE_NAMES,
        ["nose_dx", "nose_dy", "roll", "face_cx", "face_cy", "scale"]
    );
    assert_eq!(LiteBackend::GAZE_FEATURES, ["nose_dx", "nose_dy"]);
    for class in [BackendClass::FaceMesh, BackendClass::Lite] {
        for name in class.gaze_features() {
            assert!(class.feature_names().contains(name));
        }
    }
}

// --------------------------------------------------------------- offline rule
/// The MediaPipe runtime uploads usage logs; the app must not use or ship it.
#[test]
fn test_mediapipe_runtime_is_gone() {
    let package = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/vision");
    assert!(!package.join("backends/mediapipe_backend.rs").exists());
    let mut stack = vec![package];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let source = std::fs::read_to_string(&path).unwrap();
                // Split so that this file does not match itself.
                assert!(!source.contains(concat!("use ", "mediapipe")), "{path:?}");
                assert!(!source.contains(concat!("mediapipe", "::")), "{path:?}");
            }
        }
    }
}

#[test]
fn test_mediapipe_is_not_a_dependency() {
    let manifest = include_str!("../../../Cargo.toml");
    let dependencies = manifest.split("[dependencies]").nth(1).unwrap();
    assert!(!dependencies.to_lowercase().contains("mediapipe"));
}

// ---------------------------------------------------------------- model files
#[test]
fn test_model_files_are_committed_and_pinned() {
    let mut needed: Vec<&str> = BACKEND_MODEL_FILES
        .iter()
        .flat_map(|(_, files)| files.iter().copied())
        .collect();
    needed.sort_unstable();
    needed.dedup();
    let mut pinned: Vec<&str> = MODEL_FILES.iter().map(|(name, _)| *name).collect();
    pinned.sort_unstable();
    assert_eq!(pinned, needed);
    for (name, sha256) in MODEL_FILES {
        let bytes = model_bytes(name).unwrap();
        assert_eq!(hex::encode(Sha256::digest(bytes)), sha256, "{name}");
    }
    let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
    assert!(!models.join("face_landmarker.task").exists());
}

#[test]
fn test_embedded_models_match_the_pinned_checksums() {
    let embedded = [landmark_model(), geometry_metadata(), yunet_model()];
    for ((name, digest), bytes) in MODEL_FILES.iter().zip(embedded) {
        assert_eq!(hex::encode(Sha256::digest(bytes)), *digest, "{name}");
    }
}

#[test]
fn test_model_notice_lists_every_file() {
    let notice = include_str!("../../../models/NOTICE.md");
    for (name, sha256) in MODEL_FILES {
        assert!(notice.contains(name), "{name}");
        assert!(notice.contains(sha256), "{name}");
    }
    assert!(notice.contains("Apache"));
    assert!(notice.contains("MIT"));
}

#[test]
fn test_fetch_models_pins_the_same_checksums() {
    let script = include_str!("../../../../../scripts/fetch_models.py");
    for (name, sha256) in MODEL_FILES {
        assert!(script.contains(name), "{name}");
        assert!(script.contains(sha256), "{name}");
    }
}

// ---------------------------------------------------------- no-face behaviour
#[test]
fn test_blank_frame_has_no_face() {
    each_backend(|backend| {
        let frame = Frame::filled(640, 480, [90; 3]);
        let obs = backend.process(&frame, 1.0);
        assert_eq!(obs.timestamp, 1.0);
        assert_eq!(obs.face_count, 0);
        assert!(obs.features.is_none());
        assert_eq!(obs.quality, 0.0);
        assert!(!obs.usable());
        assert!(!obs.face_present());
        assert!(!obs.blind); // the worker, not the backend, judges blindness
        assert_eq!(obs.frame_size, (640, 480));
        assert!(obs.inference_ms > 0.0);
    });
}

#[test]
fn test_repeated_timestamps_are_tolerated() {
    each_backend(|backend| {
        let frame = Frame::filled(320, 240, [0; 3]);
        for ts in [5.0, 5.0, 4.0, 5.0005] {
            assert_eq!(backend.process(&frame, ts).face_count, 0);
        }
    });
}

#[test]
fn test_other_frame_formats() {
    each_backend(|backend| {
        for (width, height) in [(1920, 1080), (8, 8)] {
            let frame = Frame::filled(width, height, [0; 3]);
            let obs = backend.process(&frame, 1.0);
            assert_eq!(obs.face_count, 0);
            assert_eq!(obs.frame_size, (width as u32, height as u32));
        }
    });
}

#[test]
fn test_annotate_returns_a_copy() {
    each_backend(|backend| {
        let frame = Frame::filled(320, 240, [60; 3]);
        let obs = backend.process(&frame, 1.0);
        let out = backend.annotate(&frame, &obs);
        assert_eq!((out.width, out.height), (frame.width, frame.height));
        assert!(frame.data.iter().all(|&v| v == 60)); // input untouched
        assert!(out.data.iter().any(|&v| v != 60)); // "no face" label drawn
    });
}

#[test]
fn test_close_is_idempotent_and_final() {
    each_backend(|backend| {
        backend.close();
        backend.close();
        let frame = Frame::filled(10, 10, [0; 3]);
        let result = catch_unwind(AssertUnwindSafe(|| backend.process(&frame, 1.0)));
        assert!(result.is_err());
    });
}

/// The worker calls reset() when the camera was released and asks `settled`.
#[test]
fn test_reset_starts_afresh_and_a_still_face_is_settled() {
    each_backend(|backend| {
        let frame = draw_face(None, &FaceParams::default());
        assert!(run(backend, &frame, 1.0, 3).usable());
        assert!(backend.settled()); // a still face: the motion gate may repeat it
        backend.reset();
        backend.reset(); // idempotent
        assert!(backend.settled());
        let obs = backend.process(&frame, 5.0);
        assert!(obs.usable());
        assert!(backend.settled());
    });
}

// --------------------------------------------------------------- drawn faces
#[test]
fn test_drawn_face_is_found() {
    each_backend(|backend| {
        let frame = draw_face(None, &FaceParams::default());
        let obs = run(backend, &frame, 1.0, 3);
        assert_eq!(obs.face_count, 1);
        let features = obs.features.as_ref().unwrap();
        assert_eq!(features.len(), backend.feature_names().len());
        assert!(features.iter().all(|f| f.is_finite()));
        assert!(obs.quality >= 0.6);
        assert!(!obs.blink);
        assert!(obs.usable());
        assert!(obs.head_yaw.unwrap().abs() < 8.0); // drawn frontal
        let (x, y, w, h) = obs.face_box.unwrap();
        assert!(x < 0.5 && 0.5 < x + w);
        assert!(y < 0.5 && 0.5 < y + h);
        assert!(0.15 < w && w < 0.4);
        log::info!("{}: {:.1} ms per frame", backend.name(), obs.inference_ms);
    });
}

#[test]
fn test_drawn_face_annotation() {
    each_backend(|backend| {
        let frame = draw_face(None, &FaceParams::default());
        let obs = run(backend, &frame, 1.0, 3);
        let out = backend.annotate(&frame, &obs);
        assert_eq!((out.width, out.height), (frame.width, frame.height));
        let changed = out
            .data
            .chunks(3)
            .zip(frame.data.chunks(3))
            .filter(|(a, b)| a != b)
            .count();
        assert!(changed > 200, "{changed}");
        // A stale observation (another timestamp) only gets the box and label.
        let stale = backend.annotate(&frame, &Observation::new(-1.0, 0));
        assert_ne!(stale, frame);
    });
}

/// A face on the image left is turned towards image-right (it faces the lens).
#[test]
fn test_head_turn_signs() {
    each_backend(|backend| {
        let left = run(backend, &draw_face(None, &face_params(200)), 1.0, 3);
        let right = run(backend, &draw_face(None, &face_params(440)), 10.0, 3);
        let (left, right) = (left.head_yaw.unwrap(), right.head_yaw.unwrap());
        if backend.name() == "facemesh" {
            assert!(left > 3.0, "{left}");
            assert!(right < -3.0, "{right}");
        }
    });
}

#[test]
fn test_two_faces_are_counted() {
    each_backend(|backend| {
        let canvas = two_faces(0.22);
        backend.set_max_faces(2);
        let obs = run(backend, &canvas, 1.0, 3);
        assert_eq!(obs.face_count, 2);
        let (x, _, w, _) = obs.face_box.unwrap();
        assert!(x + w / 2.0 < 0.5); // the primary face is the user's, not the onlooker's
        let out = backend.annotate(&canvas, &obs);
        assert_eq!((out.width, out.height), (canvas.width, canvas.height));
    });
}

// ------------------------------------------------------------- lite geometry
struct Row {
    re: (f32, f32),
    le: (f32, f32),
    nose: (f32, f32),
    rm: (f32, f32),
    lm: (f32, f32),
    box_: (f32, f32, f32, f32),
}

impl Default for Row {
    fn default() -> Self {
        Self {
            re: (100.0, 100.0),
            le: (140.0, 100.0),
            nose: (120.0, 122.0),
            rm: (105.0, 140.0),
            lm: (135.0, 140.0),
            box_: (80.0, 60.0, 80.0, 100.0),
        }
    }
}

fn yunet_row(row: Row) -> Face {
    let Row {
        re,
        le,
        nose,
        rm,
        lm,
        box_,
    } = row;
    [
        box_.0, box_.1, box_.2, box_.3, re.0, re.1, le.0, le.1, nose.0, nose.1, rm.0, rm.1, lm.0,
        lm.1, 0.9,
    ]
}

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn test_geometry_frontal() {
    let (features, yaw, pitch) =
        geometry_features(&yunet_row(Row::default()), 320.0, 240.0).unwrap();
    let [nose_dx, nose_dy, roll, cx, cy, scale] = features[..] else {
        panic!("{features:?}");
    };
    assert!(approx(nose_dx, 0.0, 1e-6));
    assert!(approx(nose_dy, 0.0, 1e-6));
    assert!(approx(roll, 0.0, 1e-6));
    assert!(approx(cx, 120.0 / 320.0, 1e-6) && approx(cy, 110.0 / 240.0, 1e-6));
    assert!(approx(scale, 40.0 / 320.0, 1e-6));
    assert!(approx(yaw, 0.0, 1e-9) && approx(pitch, 0.0, 1e-9));
}

#[test]
fn test_geometry_signs() {
    let turned = geometry_features(
        &yunet_row(Row {
            nose: (128.0, 122.0),
            ..Row::default()
        }),
        320.0,
        240.0,
    )
    .unwrap();
    let lowered = geometry_features(
        &yunet_row(Row {
            nose: (120.0, 128.0),
            ..Row::default()
        }),
        320.0,
        240.0,
    )
    .unwrap();
    let tilted = geometry_features(
        &yunet_row(Row {
            le: (140.0, 90.0),
            ..Row::default()
        }),
        320.0,
        240.0,
    )
    .unwrap();
    assert!(turned.0[0] > 0.0);
    assert!(turned.1 > 0.0); // yaw
    assert!(lowered.0[1] > 0.0);
    assert!(lowered.2 > 0.0); // pitch
    assert!(tilted.0[2] > 0.0); // counter-clockwise roll
}

#[test]
fn test_geometry_mirror_flips_horizontal_features() {
    let row = yunet_row(Row {
        nose: (127.0, 120.0),
        le: (140.0, 95.0),
        ..Row::default()
    });
    let width = 320.0f32;
    let mut mirrored = row;
    // Mirror every x coordinate; YuNet labels eyes/mouth corners by image side.
    mirrored[0] = width - (row[0] + row[2]);
    mirrored[4..6].copy_from_slice(&[width - row[6], row[7]]);
    mirrored[6..8].copy_from_slice(&[width - row[4], row[5]]);
    mirrored[8] = width - row[8];
    mirrored[10..12].copy_from_slice(&[width - row[12], row[13]]);
    mirrored[12..14].copy_from_slice(&[width - row[10], row[11]]);
    let a = geometry_features(&row, f64::from(width), 240.0).unwrap();
    let b = geometry_features(&mirrored, f64::from(width), 240.0).unwrap();
    assert!(approx(b.0[0], -a.0[0], 1e-6)); // nose_dx
    assert!(approx(b.0[2], -a.0[2], 1e-6)); // roll
    assert!(approx(b.0[1], a.0[1], 1e-6)); // nose_dy
}

#[test]
fn test_geometry_degenerate() {
    let zero_ipd = yunet_row(Row {
        le: (100.0, 100.0),
        ..Row::default()
    });
    assert!(geometry_features(&zero_ipd, 320.0, 240.0).is_none());
    let mouth_on_eyes = yunet_row(Row {
        rm: (105.0, 100.0),
        lm: (135.0, 100.0),
        ..Row::default()
    });
    assert!(geometry_features(&mouth_on_eyes, 320.0, 240.0).is_none());
}

#[test]
fn test_lite_reports_every_face_regardless_of_max_faces() {
    let mut lite = LiteBackend::new(1).unwrap();
    assert_eq!(lite.process(&two_faces(0.22), 1.0).face_count, 2);
    lite.close();
}

// ------------------------------------------------------------------ real faces
#[test]
fn test_real_face() {
    let Some(path) = face_image() else { return };
    let frame = camera_like(&load(&path), 640);
    each_backend(|backend| {
        let obs = run(backend, &frame, 1.0, 5);
        assert_eq!(obs.face_count, 1);
        let features = obs.features.as_ref().unwrap();
        assert!(features.iter().all(|f| f.is_finite()));
        assert!(obs.quality >= 0.5);
        assert!(obs.usable());
        let (w, h) = obs.frame_size;
        log::info!(
            "{}: {:.1} ms per frame at {w}x{h}",
            backend.name(),
            obs.inference_ms
        );
        // tract without optimisation (debug builds) is far slower than the release build.
        if !cfg!(debug_assertions) {
            assert!(obs.inference_ms < 250.0);
        }
    });
}

fn mirror(frame: &Frame) -> Frame {
    let mut out = frame.clone();
    for y in 0..frame.height {
        for x in 0..frame.width {
            out.set_pixel(frame.width - 1 - x, y, frame.pixel(x, y));
        }
    }
    out
}

#[test]
fn test_real_face_mirror_flips_yaw_roll_and_iris() {
    let Some(path) = face_image() else { return };
    let frame = camera_like(&load(&path), 640);
    let a = run(&mut FaceMeshBackend::new(1).unwrap(), &frame, 1.0, 3);
    let b = run(
        &mut FaceMeshBackend::new(1).unwrap(),
        &mirror(&frame),
        1.0,
        3,
    );
    let (a, b) = (a.features.unwrap(), b.features.unwrap());
    let (yaw, pitch, roll) = (0, 1, 2);
    assert!(approx(a[yaw] + b[yaw], 0.0, 2.0));
    assert!(approx(a[roll] + b[roll], 0.0, 2.0));
    assert!(approx(a[pitch], b[pitch], 2.0));
    // Mirroring swaps the eyes and reverses the corner axes: iris_h -> 1 - iris_h.
    assert!(approx(a[6] + b[6], 1.0, 0.05));
}

#[test]
fn test_real_face_iris_is_position_and_scale_invariant() {
    let Some(path) = face_image() else { return };
    let image = load(&path);
    let base = run(
        &mut FaceMeshBackend::new(1).unwrap(),
        &camera_like(&image, 640),
        1.0,
        3,
    );
    let (w, h) = (image.width, image.height);
    let (x0, y1) = ((w as f64 * 0.06) as usize, (h as f64 * 0.92) as usize);
    let mut cropped = Frame::filled(w - x0, y1, [0; 3]);
    for y in 0..y1 {
        for x in x0..w {
            cropped.set_pixel(x - x0, y, image.pixel(x, y));
        }
    }
    let shifted = camera_like(&cropped, 640);
    let smaller = camera_like(&image, 420);
    let base = base.features.unwrap();
    for frame in [shifted, smaller] {
        let obs = run(&mut FaceMeshBackend::new(1).unwrap(), &frame, 1.0, 3);
        let features = obs.features.unwrap();
        assert!(approx(features[6], base[6], 0.05)); // iris_h
        assert!(approx(features[7], base[7], 0.05)); // iris_v
    }
}
