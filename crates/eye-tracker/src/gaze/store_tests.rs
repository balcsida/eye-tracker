// Tests for gaze::store (calibration file, profiles, robustness), ported from tests/test_store.py.
// The random streams come from testing::gaze_synth::Rng, not numpy.
//
// Differences from the Python tests:
// * The library owns its profiles and addresses them by `ProfileKey`, so "is" checks compare
//   keys (or the unique `created_at` labels), and test_library_put_replaces_same_key_and_unknown_
//   camera_profile ends with `put` of a copy of a stored profile, which returns the older copy
//   of the same key (Python returned nothing for the very same object).
// * serde_json::Value cannot hold NaN or infinity. `report` values that were NaN are `null`
//   (what the file stores), and the corruptions that wrote `inf`/`nan`/`1e400`/`10**400` use
//   strings or `null` (all rejected the same way, as a number that cannot be converted).
//   Such a file written by another tool (with the bare tokens `NaN`/`Infinity`) is not valid JSON
//   for serde_json and drops the whole file, where Python dropped only that profile.
// * Report keys are strings (`"0"`): Python's int keys become strings in the file anyway.
// * test_recomputing_an_old_profile_is_cheap_enough_for_the_gui_thread keeps the 0.25 s bound,
//   measured in the unoptimised test build.
// * New tests (not in Python): test_loads_a_calibration_file_written_by_the_python_app (the
//   fixture is a file written by the Python store, and Rust writes the same document back),
//   test_key_order_of_the_file_matches_python and test_iso_from_unix.
// * That Python reads a file written by Rust was checked by hand, not in cargo test: a library
//   written by `CalibrationLibrary::save` here (the fixture plus a freshly fitted profile) was
//   loaded with `uv run python` by eye_tracker.gaze.store.CalibrationLibrary.load: all
//   three profiles, their samples, report and fitted models came back and predicted.

use std::path::Path;
use std::sync::LazyLock;

use serde_json::{Map, Value, json};

use super::*;
use crate::testing::capture_logs;
use crate::testing::gaze_synth::{GAZE, Rng, SampleOptions, TWO, calibration_samples};
use crate::types::layout_signature;

static MONITORS: LazyLock<Vec<Monitor>> = LazyLock::new(|| {
    vec![
        Monitor {
            primary: true,
            scale: 1.5,
            ..Monitor::new(0, "DELL U2720Q", Rect::new(0, 0, 3840, 2160))
        },
        Monitor::new(1, "LG ÜltraGear", Rect::new(3840, 540, 1920, 1080)),
    ]
});
// A second desk: a laptop panel next to a 1080p monitor.
static HOME: LazyLock<Vec<Monitor>> = LazyLock::new(|| {
    vec![
        Monitor {
            primary: true,
            ..Monitor::new(0, "laptop", Rect::new(0, 0, 1920, 1200))
        },
        Monitor::new(1, "monitor", Rect::new(1920, 0, 1920, 1080)),
    ]
});

/// The arguments of `make_data`.
struct Spec<'a> {
    n_points: usize,
    per_point: usize,
    implicit: usize,
    monitors: &'a [Monitor],
    camera: &'a str,
    frame_size: (u32, u32),
    created_at: &'a str,
}

impl Default for Spec<'static> {
    fn default() -> Self {
        Spec {
            n_points: 12,
            per_point: 5,
            implicit: 4,
            monitors: &MONITORS,
            camera: "0",
            frame_size: (640, 480),
            created_at: "2026-09-30T08:15:00+00:00",
        }
    }
}

fn make_data(spec: Spec) -> CalibrationData {
    let monitors = spec.monitors;
    let mut rng = Rng::new(7);
    let mut samples = Vec::new();
    for pid in 0..spec.n_points {
        let monitor = &monitors[pid % 2];
        let nx = rng.uniform(0.1, 0.9);
        let ny = rng.uniform(0.1, 0.9);
        let (x, y) = monitor.rect.denormalize(nx, ny);
        for _ in 0..spec.per_point {
            samples.push(CalibrationSample::new(
                rng.normal_vec(0.0, 1.0, 8),
                x,
                y,
                monitor.index,
                pid as i64,
            ));
        }
    }
    let learned: Vec<CalibrationSample> = (0..spec.implicit)
        .map(|i| {
            CalibrationSample::weighted(
                rng.normal_vec(0.0, 1.0, 8),
                100.0 + i as f64,
                200.0,
                0,
                -(i as i64 + 1),
                0.5,
            )
        })
        .collect();
    let (x, y, _) = samples_to_arrays(&samples).unwrap();
    let model = GazeModel::new(3, 0.1, Some(&[0, 1, 6, 7]))
        .unwrap()
        .fit(&x, &y, None, virtual_bounds(monitors).ok(), None)
        .unwrap();
    let mut data = CalibrationData::new(
        "facemesh",
        "facemesh-pose-iris-1",
        layout_signature(monitors),
        monitors.to_vec(),
        samples,
        learned,
        model,
    );
    data.report = object(json!({
        "grade": "excellent",
        "monitor_accuracy": 0.99,
        "mean_error_px": f64::NAN,
        "per_monitor_accuracy": {"0": 1.0, "1": 0.98},
        "n_samples": 60,
    }));
    data.created_at = spec.created_at.to_string();
    data.camera = spec.camera.to_string();
    data.frame_size = spec.frame_size;
    data
}

fn data_with(camera: &str, created_at: &str) -> CalibrationData {
    make_data(Spec {
        camera,
        created_at,
        ..Spec::default()
    })
}

fn object(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        other => panic!("not an object: {other}"),
    }
}

fn assert_samples_close(a: &[CalibrationSample], b: &[CalibrationSample]) {
    assert_eq!(a.len(), b.len());
    for (s, t) in a.iter().zip(b) {
        assert_eq!(s.features.len(), t.features.len());
        for (u, v) in s.features.iter().zip(&t.features) {
            assert!((u - v).abs() <= 1e-6 + 1e-5 * v.abs());
        }
        assert!((s.x - t.x).abs() <= 1e-6 && (s.y - t.y).abs() <= 1e-6);
        assert_eq!((s.monitor_index, s.point_id), (t.monitor_index, t.point_id));
        assert!((s.weight - t.weight).abs() <= 1e-6 * t.weight.abs().max(1e-6));
    }
}

fn read_doc(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn write_doc(path: &Path, doc: &Value) {
    std::fs::write(path, serde_json::to_string(doc).unwrap()).unwrap();
}

/// The single-calibration layout written by version 1 of the file format.
fn v1_doc(data: &CalibrationData, dir: &Path) -> Value {
    let path = dir.join("v2.json");
    save_calibration(&path, data).unwrap();
    let mut profile = read_doc(&path)["profiles"][0].clone();
    std::fs::remove_file(&path).unwrap();
    for key in ["camera", "frame_size"] {
        profile.as_object_mut().unwrap().remove(key);
    }
    let mut doc = json!({"format": "eye-tracker-calibration", "version": 1});
    doc.as_object_mut()
        .unwrap()
        .extend(profile.as_object().unwrap().clone());
    doc
}

fn saved_doc(dir: &Path) -> (std::path::PathBuf, Value) {
    let path = dir.join("calibration.json");
    save_calibration(&path, &make_data(Spec::default())).unwrap();
    let doc = read_doc(&path);
    (path, doc)
}

fn temp() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

fn warnings(logs: &crate::testing::LogCapture) -> Vec<String> {
    logs.messages_at_least(log::Level::Warn)
}

fn labels(library: &CalibrationLibrary) -> Vec<String> {
    library.iter().map(|p| p.created_at.clone()).collect()
}

fn created_at(profiles: &[CalibrationData]) -> Vec<&str> {
    profiles.iter().map(|p| p.created_at.as_str()).collect()
}

fn library_of(profiles: Vec<CalibrationData>) -> CalibrationLibrary {
    CalibrationLibrary::new(profiles, MAX_PROFILES).unwrap()
}

fn compatible(
    data: &CalibrationData,
    camera: Option<&str>,
    frame: Option<(u32, u32)>,
) -> (bool, String) {
    data.is_compatible("facemesh", "facemesh-pose-iris-1", &MONITORS, camera, frame)
}

fn ok() -> (bool, String) {
    (true, String::new())
}

fn key_of(data: Option<&CalibrationData>) -> Option<ProfileKey> {
    data.map(CalibrationData::key)
}

/// Equal JSON, numbers compared by value (`0` equals `0.0`).
fn same_json(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same_json(p, q))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len()
                && x.iter()
                    .all(|(k, v)| y.get(k).is_some_and(|w| same_json(v, w)))
        }
        _ => a == b,
    }
}

fn assert_close(a: f64, b: f64, tolerance: f64) {
    assert!((a - b).abs() <= tolerance, "{a} != {b}");
}

// --------------------------------------------------------------------------- round trip
#[test]
fn test_round_trip() {
    let data = make_data(Spec::default());
    let dir = temp();
    let path = dir.path().join("sub").join("calibration.json");
    save_calibration(&path, &data).unwrap();
    let loaded = load_calibration(&path).unwrap();
    assert_eq!(loaded.backend, "facemesh");
    assert_eq!(loaded.feature_version, "facemesh-pose-iris-1");
    assert_eq!(loaded.layout_signature, data.layout_signature);
    assert_eq!(loaded.monitors, *MONITORS);
    assert_eq!(loaded.created_at, "2026-09-30T08:15:00+00:00");
    assert_eq!(
        (loaded.camera.as_str(), loaded.frame_size),
        ("0", (640, 480))
    );
    assert_samples_close(&loaded.samples, &data.samples);
    assert_samples_close(&loaded.implicit_samples, &data.implicit_samples);
    assert_eq!((loaded.model.degree, loaded.model.alpha), (3, 0.1));
    assert_eq!(loaded.model.nonlinear(), Some(&[0, 1, 6, 7][..]));
    let (x, _, _) = samples_to_arrays(&data.samples).unwrap();
    let (saved, restored) = (
        data.model.predict_many(&x).unwrap(),
        loaded.model.predict_many(&x).unwrap(),
    );
    assert!(
        saved
            .iter()
            .zip(restored.iter())
            .all(|(a, b)| (a - b).abs() <= 1e-3)
    );
    assert_eq!(loaded.report["grade"], "excellent");
    assert!(loaded.report["mean_error_px"].is_null()); // NaN is not valid JSON
    assert_eq!(
        loaded.report["per_monitor_accuracy"],
        json!({"0": 1.0, "1": 0.98})
    );
    assert_eq!(loaded.report["n_samples"], 60);
    assert_eq!(loaded.grade(), Some("excellent"));
    assert_eq!(compatible(&loaded, None, None), ok());
    assert_eq!(compatible(&loaded, Some("0"), Some((640, 480))), ok());
}

#[test]
fn test_file_is_readable_json_without_images() {
    let dir = temp();
    let path = dir.path().join("calibration.json");
    save_calibration(&path, &make_data(Spec::default())).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let doc: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(doc["format"], "eye-tracker-calibration");
    assert_eq!(doc["version"], CALIBRATION_VERSION);
    assert_eq!(CALIBRATION_VERSION, 2);
    assert_eq!(doc.as_object().unwrap().len(), 3);
    assert_eq!(doc["profiles"].as_array().unwrap().len(), 1);
    let profile = &doc["profiles"][0];
    let mut keys: Vec<&str> = profile
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    let mut expected = [
        "created_at",
        "backend",
        "feature_version",
        "layout_signature",
        "camera",
        "frame_size",
        "monitors",
        "report",
        "model",
        "samples",
        "implicit_samples",
    ];
    expected.sort_unstable();
    assert_eq!(keys, expected);
    let mut sample_keys: Vec<&str> = profile["samples"][0]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    sample_keys.sort_unstable();
    assert_eq!(
        sample_keys,
        ["features", "monitor", "point", "weight", "x", "y"]
    );
    assert_eq!(profile["frame_size"], json!([640, 480]));
    // One sample per line keeps the file diff-friendly and compact.
    assert!(text.matches('\n').count() < 120);
    assert!(text.contains("LG ÜltraGear"));
    // Sample floats are rounded to six decimals.
    for value in profile["samples"][0]["features"].as_array().unwrap() {
        let v = value.as_f64().unwrap();
        assert_eq!(round_sample(v), v);
    }
    // atomic write leaves no temp files
    let names: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    assert_eq!(names, [path]);
}

#[test]
fn test_key_order_of_the_file_matches_python() {
    let dir = temp();
    let path = dir.path().join("calibration.json");
    save_calibration(&path, &make_data(Spec::default())).unwrap();
    let doc = read_doc(&path);
    let keys = |v: &Value| -> Vec<String> { v.as_object().unwrap().keys().cloned().collect() };
    assert_eq!(keys(&doc), ["format", "version", "profiles"]);
    assert_eq!(
        keys(&doc["profiles"][0]),
        [
            "created_at",
            "backend",
            "feature_version",
            "layout_signature",
            "camera",
            "frame_size",
            "monitors",
            "report",
            "model",
            "samples",
            "implicit_samples"
        ]
    );
    assert_eq!(
        keys(&doc["profiles"][0]["monitors"][0]),
        ["index", "name", "rect", "primary", "scale"]
    );
    assert_eq!(
        keys(&doc["profiles"][0]["samples"][0]),
        ["features", "x", "y", "monitor", "point", "weight"]
    );
    assert_eq!(
        keys(&doc["profiles"][0]["model"])[..4],
        ["kind", "degree", "alpha", "nonlinear"]
    );
}

#[test]
fn test_non_finite_samples_are_not_saved() {
    let logs = capture_logs();
    let mut data = make_data(Spec::default());
    data.samples[0].features[3] = f64::NAN;
    let dir = temp();
    let path = dir.path().join("c.json");
    save_calibration(&path, &data).unwrap();
    let loaded = load_calibration(&path).unwrap();
    assert_eq!(loaded.samples.len(), data.samples.len() - 1);
    assert!(warnings(&logs).iter().any(|m| m.contains("non-finite")));
}

#[test]
fn test_defaults_and_created_at_format() {
    let data = make_data(Spec::default());
    let minimal = CalibrationData::new(
        "opencv",
        "yunet-geom-1",
        "abc",
        vec![],
        vec![],
        vec![],
        data.model.clone(),
    );
    assert!(minimal.report.is_empty());
    assert_eq!(minimal.grade(), None);
    assert_eq!((minimal.camera.as_str(), minimal.frame_size), ("", (0, 0)));
    // The renamed OpenCV backend is keyed by its current name.
    assert_eq!(
        minimal.key(),
        (
            "abc".into(),
            "lite".into(),
            "yunet-geom-1".into(),
            String::new()
        )
    );
    // ISO 8601 in UTC with second precision: 2026-09-30T08:15:00+00:00
    let shape: Vec<char> = minimal
        .created_at
        .chars()
        .map(|c| if c.is_ascii_digit() { '0' } else { c })
        .collect();
    assert_eq!(
        shape.iter().collect::<String>(),
        "0000-00-00T00:00:00+00:00"
    );
    assert!(utc_now_iso().ends_with("+00:00"));
}

#[test]
fn test_iso_from_unix() {
    assert_eq!(iso_from_unix(0), "1970-01-01T00:00:00+00:00");
    assert_eq!(iso_from_unix(1_700_000_000), "2023-11-14T22:13:20+00:00");
    assert_eq!(iso_from_unix(951_782_400), "2000-02-29T00:00:00+00:00");
    assert_eq!(iso_from_unix(1_788_077_699), "2026-08-30T08:14:59+00:00");
}

#[test]
fn test_unfitted_model_round_trip() {
    let mut data = make_data(Spec::default());
    data.model = GazeModel::new(1, 2.0, None).unwrap();
    let dir = temp();
    let path = dir.path().join("c.json");
    save_calibration(&path, &data).unwrap();
    let loaded = load_calibration(&path).unwrap();
    assert!(!loaded.model.is_fitted());
    let (fine, reason) = compatible(&loaded, None, None);
    assert!(!fine);
    assert!(reason.contains("no fitted model"));
}

#[test]
fn test_version_1_file_still_loads() {
    let data = make_data(Spec::default());
    let dir = temp();
    let path = dir.path().join("calibration.json");
    write_doc(&path, &v1_doc(&data, dir.path()));
    let loaded = load_calibration(&path).unwrap();
    assert_eq!((loaded.camera.as_str(), loaded.frame_size), ("", (0, 0))); // unknown
    assert_samples_close(&loaded.samples, &data.samples);
    // An unknown camera never makes the calibration unusable.
    assert_eq!(compatible(&loaded, Some("1"), Some((1280, 720))), ok());
}

#[test]
fn test_model_without_nonlinear_key_loads_as_all_features() {
    let mut data = make_data(Spec::default());
    let (x, y, _) = samples_to_arrays(&data.samples).unwrap();
    data.model = GazeModel::new(2, 1.0, None)
        .unwrap()
        .fit(&x, &y, None, None, None)
        .unwrap();
    let dir = temp();
    let path = dir.path().join("calibration.json");
    save_calibration(&path, &data).unwrap();
    let doc = read_doc(&path);
    assert!(doc["profiles"][0]["model"].get("nonlinear").is_none());
    let loaded = load_calibration(&path).unwrap();
    assert_eq!(loaded.model.nonlinear(), None);
}

// --------------------------------------------------------------------------- compatibility
#[test]
fn test_is_compatible_reasons() {
    let data = make_data(Spec::default());
    let (fine, reason) = data.is_compatible("lite", "facemesh-pose-iris-1", &MONITORS, None, None);
    assert!(!fine);
    assert!(reason.contains("lite"));
    assert!(reason.contains("facemesh"));
    let (fine, reason) =
        data.is_compatible("facemesh", "facemesh-pose-iris-2", &MONITORS, None, None);
    assert!(!fine);
    assert!(reason.contains("facemesh-pose-iris-2"));
    let moved = [
        MONITORS[0].clone(),
        Monitor::new(1, "LG", Rect::new(-1920, 0, 1920, 1080)),
    ];
    let (fine, reason) = data.is_compatible("facemesh", "facemesh-pose-iris-1", &moved, None, None);
    assert!(!fine);
    assert!(reason.contains("layout"));
    // Monitor order and names do not matter, only the geometry.
    let renamed = [
        Monitor::new(5, "x", MONITORS[1].rect),
        Monitor::new(6, "y", MONITORS[0].rect),
    ];
    assert_eq!(
        data.is_compatible("facemesh", "facemesh-pose-iris-1", &renamed, None, None),
        ok_pair()
    );
}

fn ok_pair() -> (bool, String) {
    ok()
}

fn legacy_profile(backend: &str, feature_version: &str) -> CalibrationData {
    let mut data = make_data(Spec::default());
    data.backend = backend.to_string();
    data.feature_version = feature_version.to_string();
    data
}

#[test]
fn test_opencv_calibrations_are_valid_for_the_renamed_lite_backend() {
    // The OpenCV/YuNet backend is now "lite" with unchanged measurements.
    let old = legacy_profile("opencv", "yunet-geom-1");
    assert_eq!(
        old.is_compatible("lite", "yunet-geom-1", &MONITORS, None, None),
        ok()
    );
    assert_eq!(old.key().1, "lite");
    // Its measurement version still has to match.
    let (fine, reason) = old.is_compatible("lite", "yunet-geom-2", &MONITORS, None, None);
    assert!(!fine);
    assert!(reason.contains("yunet-geom-2"));
    // The library finds it for "lite", and a new "lite" calibration replaces it.
    let mut library = library_of(vec![old.clone()]);
    assert_eq!(
        key_of(library.best("lite", "yunet-geom-1", &MONITORS, None, None)),
        Some(old.key())
    );
    let new = legacy_profile("lite", "yunet-geom-1");
    library.put(new.clone());
    assert_eq!(library.len(), 1);
    assert_eq!(library.profiles()[0].backend, "lite");
    // Loading a file written by the old version stores the current name.
    let dir = temp();
    let path = dir.path().join("calibration.json");
    save_calibration(&path, &old).unwrap();
    assert_eq!(read_doc(&path)["profiles"][0]["backend"], "opencv");
    let loaded = load_calibration(&path).unwrap();
    assert_eq!(loaded.backend, "lite");
    assert_eq!(
        loaded.is_compatible("lite", "yunet-geom-1", &MONITORS, None, None),
        ok()
    );
}

#[test]
fn test_mediapipe_calibrations_are_not_used_by_facemesh() {
    // The MediaPipe runtime backend measured differently from its successor.
    let old = legacy_profile("mediapipe", "mp-pose-iris-1");
    let (fine, reason) =
        old.is_compatible("facemesh", "facemesh-pose-iris-1", &MONITORS, None, None);
    assert!(!fine);
    assert!(reason.contains("mediapipe"));
    assert!(reason.contains("facemesh"));
    // Even under the same name, the measurement version differs.
    let (fine, _) = old.is_compatible("mediapipe", "facemesh-pose-iris-1", &MONITORS, None, None);
    assert!(!fine);
    let library = library_of(vec![old]);
    assert!(
        library
            .best("facemesh", "facemesh-pose-iris-1", &MONITORS, None, None)
            .is_none()
    );
}

#[test]
fn test_is_compatible_checks_the_camera() {
    let data = make_data(Spec {
        camera: "0",
        frame_size: (640, 480),
        ..Spec::default()
    });
    assert_eq!(
        compatible(&data, Some("1"), None),
        (false, "the camera changed".to_string())
    );
    assert_eq!(compatible(&data, Some(" 0 "), None), ok());
    // Unknown on either side: no reason to reject.
    assert_eq!(compatible(&data, Some(""), None), ok());
    assert_eq!(compatible(&data, None, Some((0, 0))), ok());
    let unknown = make_data(Spec {
        camera: "",
        ..Spec::default()
    });
    assert_eq!(compatible(&unknown, Some("1"), None), ok());
    // The resolution does not matter, the field of view (aspect ratio) does.
    assert_eq!(compatible(&data, Some("0"), Some((1280, 960))), ok());
    let (fine, reason) = compatible(&data, Some("0"), Some((1280, 720)));
    assert!(!fine);
    assert!(reason.contains("aspect ratio"));
    assert!(reason.contains("640x480"));
    assert!(reason.contains("1280x720"));
    assert_eq!(
        data.camera_matches(Some("1"), None),
        (false, "the camera changed".to_string())
    );
    assert_eq!(data.camera_matches(None, Some((320, 240))), ok());
}

#[test]
fn test_same_aspect() {
    assert!(same_aspect((640, 480), (1280, 960)));
    assert!(same_aspect((1920, 1080), (1280, 720)));
    assert!(same_aspect((1920, 1080), (1920, 1088))); // within 1 % (encoder padding)
    assert!(!same_aspect((640, 480), (640, 360)));
    assert!(same_aspect((0, 0), (640, 360)));
    assert!(same_aspect((640, 480), (0, 0)));
}

// --------------------------------------------------------------------------- robustness
#[test]
fn test_missing_file_returns_none_quietly() {
    let logs = capture_logs();
    let dir = temp();
    assert!(load_calibration(&dir.path().join("nope.json")).is_none());
    assert_eq!(
        CalibrationLibrary::load(&dir.path().join("nope.json")).len(),
        0
    );
    assert!(warnings(&logs).is_empty());
}

#[test]
fn test_unreadable_path_returns_none() {
    let dir = temp();
    assert!(load_calibration(dir.path()).is_none()); // a directory, not a file
}

type Corruption = fn(&Value) -> Value;

fn with(doc: &Value, key: &str, value: Value) -> Value {
    let mut out = doc.clone();
    out[key] = value;
    out
}

fn without(doc: &Value, key: &str) -> Value {
    let mut out = doc.clone();
    out.as_object_mut().unwrap().remove(key);
    out
}

fn with_model(doc: &Value, key: &str, value: Value) -> Value {
    with(doc, "model", with(&doc["model"], key, value))
}

/// Corruptions of one profile (in a version 2 file) or of a whole version 1 file.
fn profile_corruptions() -> Vec<Corruption> {
    vec![
        |doc| without(doc, "model"),
        |doc| without(doc, "backend"),
        |doc| with(doc, "backend", json!(42)),
        |doc| with(doc, "monitors", json!("all of them")),
        |doc| with(doc, "monitors", json!([{"index": 0}])),
        |doc| with(doc, "samples", json!([{"features": [1, 2]}])),
        |doc| {
            with(
                doc,
                "samples",
                json!([with(&doc["samples"][0], "features", json!([]))]),
            )
        },
        |doc| {
            with(
                doc,
                "samples",
                json!([with(&doc["samples"][0], "features", json!([1.0, 2.0]))]),
            )
        },
        |doc| {
            with(
                doc,
                "samples",
                json!([with(&doc["samples"][0], "weight", json!(-1))]),
            )
        },
        |doc| with(doc, "implicit_samples", json!({"a": 1})),
        |doc| with_model(doc, "coef", json!([[1.0, 2.0]])),
        |doc| with_model(doc, "nonlinear", json!([0, 99])),
        // Values that cannot be converted used to escape as OverflowError (Python wrote
        // infinity there; see the header).
        |doc| with_model(doc, "bounds", json!([0, 0, 1920, "inf"])),
        |doc| with_model(doc, "degree", json!("inf")),
        |doc| {
            with(
                doc,
                "monitors",
                json!([with(
                    &doc["monitors"][0],
                    "rect",
                    json!([0, 0, 1920, "1e400"])
                )]),
            )
        },
        |doc| {
            with(
                doc,
                "monitors",
                json!([with(&doc["monitors"][0], "index", json!("-inf"))]),
            )
        },
        |doc| {
            with(
                doc,
                "monitors",
                json!([with(&doc["monitors"][0], "scale", json!("inf"))]),
            )
        },
        |doc| {
            with(
                doc,
                "samples",
                json!([with(&doc["samples"][0], "point", json!("inf"))]),
            )
        },
        |doc| {
            with(
                doc,
                "samples",
                json!([with(&doc["samples"][0], "monitor", json!(null))]),
            )
        },
        |doc| {
            with(
                doc,
                "samples",
                json!([with(&doc["samples"][0], "x", json!("1e400"))]),
            )
        },
        |_| json!("not an object"),
    ]
}

fn expect_none_with_warning(path: &Path) {
    let logs = capture_logs();
    assert!(load_calibration(path).is_none());
    assert!(
        warnings(&logs)
            .iter()
            .any(|m| m.to_lowercase().contains("calibration")),
        "{:?}",
        warnings(&logs)
    );
}

#[test]
fn test_corrupt_files_return_none_with_warning() {
    let dir = temp();
    let (path, doc) = saved_doc(dir.path());
    let deep = format!(
        "{{\"format\": \"eye-tracker-calibration\", \"version\": 2, \"profiles\": {}{}}}",
        "[".repeat(100_000),
        "]".repeat(100_000)
    );
    // Deep nesting makes the JSON parser recurse until it gives up.
    let mut contents: Vec<String> = vec![
        "{ not json".into(),
        "[]".into(),
        String::new(),
        "\u{0}\u{ff} garbage".into(),
        serde_json::to_string(&with(&doc, "format", json!("something-else"))).unwrap(),
        serde_json::to_string(&with(&doc, "version", json!(0))).unwrap(),
        serde_json::to_string(&with(&doc, "version", json!("1"))).unwrap(),
        serde_json::to_string(&with(&doc, "profiles", json!("many"))).unwrap(),
        serde_json::to_string(&without(&doc, "profiles")).unwrap(),
        deep,
    ];
    for corrupt in profile_corruptions() {
        let broken = with(&doc, "profiles", json!([corrupt(&doc["profiles"][0])]));
        contents.push(serde_json::to_string(&broken).unwrap());
    }
    for text in contents {
        std::fs::write(&path, &text).unwrap();
        expect_none_with_warning(&path);
    }
}

#[test]
fn test_corrupt_version_1_files_return_none() {
    for corrupt in profile_corruptions() {
        let dir = temp();
        let path = dir.path().join("calibration.json");
        let broken = corrupt(&v1_doc(&make_data(Spec::default()), dir.path()));
        write_doc(&path, &broken);
        assert!(load_calibration(&path).is_none());
    }
}

#[test]
fn test_file_with_byte_order_mark_loads() {
    // Windows PowerShell 5.1 `Set-Content -Encoding UTF8` writes a BOM.
    let dir = temp();
    let (path, _) = saved_doc(dir.path());
    let mut bytes = vec![0xef, 0xbb, 0xbf];
    bytes.extend(std::fs::read(&path).unwrap());
    std::fs::write(&path, bytes).unwrap();
    assert_eq!(load_calibration(&path).unwrap().backend, "facemesh");
}

#[test]
fn test_newer_version_is_ignored() {
    let logs = capture_logs();
    let dir = temp();
    let (path, doc) = saved_doc(dir.path());
    write_doc(
        &path,
        &with(&doc, "version", json!(CALIBRATION_VERSION + 1)),
    );
    assert!(load_calibration(&path).is_none());
    assert!(warnings(&logs).iter().any(|m| m.contains("newer")));
}

#[test]
fn test_invalid_report_is_replaced_by_empty_dict() {
    let dir = temp();
    let (path, mut doc) = saved_doc(dir.path());
    doc["profiles"][0]["report"] = json!("excellent");
    write_doc(&path, &doc);
    assert!(load_calibration(&path).unwrap().report.is_empty());
}

#[test]
fn test_malformed_camera_metadata_means_unknown() {
    let cases = [
        (json!(7), json!([640, 480]), ("", (640, 480))),
        (json!(null), json!([640]), ("", (0, 0))),
        (json!("0"), json!("640x480"), ("0", (0, 0))),
        (json!("0"), json!([640.0, 480]), ("0", (0, 0))),
        (json!("0"), json!([-1, 480]), ("0", (0, 0))),
        (json!("0"), json!([true, 480]), ("0", (0, 0))),
    ];
    for (camera, frame_size, expected) in cases {
        let dir = temp();
        let (path, mut doc) = saved_doc(dir.path());
        doc["profiles"][0]["camera"] = camera;
        doc["profiles"][0]["frame_size"] = frame_size;
        write_doc(&path, &doc);
        // optional metadata never costs the calibration
        let loaded = load_calibration(&path).unwrap();
        assert_eq!((loaded.camera.as_str(), loaded.frame_size), expected);
    }
}

#[test]
fn test_optional_fields_default() {
    let dir = temp();
    let (path, mut doc) = saved_doc(dir.path());
    doc.as_object_mut().unwrap().remove("format");
    let profile = doc["profiles"][0].as_object_mut().unwrap();
    for key in [
        "implicit_samples",
        "report",
        "created_at",
        "camera",
        "frame_size",
    ] {
        profile.remove(key);
    }
    for sample in profile["samples"].as_array_mut().unwrap() {
        sample.as_object_mut().unwrap().remove("weight");
    }
    write_doc(&path, &doc);
    let loaded = load_calibration(&path).unwrap();
    assert!(loaded.implicit_samples.is_empty());
    assert!(loaded.samples.iter().all(|s| s.weight == 1.0));
    assert_eq!((loaded.camera.as_str(), loaded.frame_size), ("", (0, 0)));
}

// --------------------------------------------------------------------------- profiles
#[test]
fn test_save_replaces_the_same_setup_and_keeps_others() {
    let dir = temp();
    let path = dir.path().join("calibration.json");
    let office = data_with("0", "2026-09-01T08:00:00+00:00");
    save_calibration(&path, &office).unwrap();
    let home = make_data(Spec {
        monitors: &HOME,
        created_at: "2026-09-02T08:00:00+00:00",
        ..Spec::default()
    });
    save_calibration(&path, &home).unwrap();
    let library = CalibrationLibrary::load(&path);
    assert_eq!(
        labels(&library),
        ["2026-09-02T08:00:00+00:00", office.created_at.as_str()]
    );
    let loaded = load_calibration(&path).unwrap();
    assert_eq!(loaded.created_at, home.created_at); // the most recent one

    // A new calibration of the office replaces the old office profile only.
    let again = make_data(Spec {
        n_points: 4,
        created_at: "2026-09-03T08:00:00+00:00",
        ..Spec::default()
    });
    save_calibration(&path, &again).unwrap();
    let library = CalibrationLibrary::load(&path);
    assert_eq!(
        labels(&library),
        [again.created_at.as_str(), home.created_at.as_str()]
    );
    assert_eq!(library.profiles()[0].samples.len(), 20);
    let names: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(names, ["calibration.json"]);
}

#[test]
fn test_library_picks_the_profile_for_the_current_desk() {
    let office = make_data(Spec::default());
    let home = make_data(Spec {
        monitors: &HOME,
        ..Spec::default()
    });
    let library = library_of(vec![home.clone(), office.clone()]);
    let args = ("facemesh", "facemesh-pose-iris-1");
    let best = |monitors: &[Monitor], camera: Option<&str>| {
        key_of(library.best(args.0, args.1, monitors, camera, None))
    };
    assert_eq!(best(&MONITORS, Some("0")), Some(office.key()));
    assert_eq!(best(&HOME, Some("0")), Some(home.key()));
    let (found, reason) = library.match_profile(args.0, args.1, &HOME, None, None);
    assert_eq!((key_of(found), reason), (Some(home.key()), String::new()));
    // Another backend or camera: nothing fits, and the reason says why.
    let (found, reason) = library.match_profile("lite", "yunet-geom-1", &HOME, None, None);
    assert!(found.is_none());
    assert!(reason.contains("lite"));
    let (found, reason) = library.match_profile(args.0, args.1, &HOME, Some("1"), None);
    assert!(found.is_none());
    assert_eq!(reason, "the camera changed");
    let third = [Monitor::new(0, "tv", Rect::new(0, 0, 3840, 2160))];
    let (found, reason) = library.match_profile(args.0, args.1, &third, None, None);
    assert!(found.is_none());
    assert_eq!(reason, "the monitor layout changed");
    let empty = CalibrationLibrary::default();
    let (found, reason) = empty.match_profile(args.0, args.1, &HOME, None, None);
    assert!(found.is_none());
    assert_eq!(reason, "not calibrated yet");
}

#[test]
fn test_library_prefers_a_known_camera_then_the_most_recent() {
    let legacy = make_data(Spec {
        camera: "",
        frame_size: (0, 0),
        created_at: "legacy",
        ..Spec::default()
    });
    let webcam = make_data(Spec {
        camera: "1",
        frame_size: (1280, 720),
        created_at: "webcam",
        ..Spec::default()
    });
    let builtin = make_data(Spec {
        camera: "0",
        frame_size: (640, 480),
        created_at: "builtin",
        ..Spec::default()
    });
    let library = library_of(vec![legacy, webcam, builtin]); // legacy most recent
    let best = |camera: Option<&str>, frame_size: Option<(u32, u32)>| {
        library
            .best(
                "facemesh",
                "facemesh-pose-iris-1",
                &MONITORS,
                camera,
                frame_size,
            )
            .map(|p| p.created_at.clone())
            .unwrap()
    };
    assert_eq!(best(Some("1"), None), "webcam");
    assert_eq!(best(Some("0"), None), "builtin");
    assert_eq!(best(Some("2"), None), "legacy"); // unknown camera: may be right
    assert_eq!(best(None, None), "legacy"); // camera unknown: most recently used
    assert_eq!(best(None, Some((640, 480))), "builtin"); // a known match wins
    assert_eq!(best(Some("0"), Some((1280, 720))), "legacy");
}

#[test]
fn test_library_put_replaces_same_key_and_unknown_camera_profile() {
    let legacy = data_with("", "legacy");
    let other_desk = make_data(Spec {
        monitors: &HOME,
        camera: "",
        created_at: "home",
        ..Spec::default()
    });
    let mut library = library_of(vec![legacy, other_desk.clone()]);
    let fresh = data_with("0", "fresh");
    let replaced = library.put(fresh);
    // The unknown-camera profile of the same setup was most likely this camera.
    assert_eq!(created_at(&replaced), ["legacy"]);
    assert_eq!(labels(&library), ["fresh", "home"]);
    let newer = data_with("0", "newer");
    assert_eq!(created_at(&library.put(newer.clone())), ["fresh"]);
    assert_eq!(labels(&library), ["newer", "home"]);
    let external = data_with("1", "external");
    assert!(library.put(external).is_empty());
    assert_eq!(labels(&library), ["external", "newer", "home"]);
    assert_eq!(
        library.get(&newer.key()).map(|p| p.created_at.as_str()),
        Some("newer")
    );
    assert!(
        library
            .get(&("nope".into(), String::new(), String::new(), String::new()))
            .is_none()
    );
    // Putting a stored profile again (after a refit) makes it the latest; the stored copy
    // it replaces is handed back (see the header).
    assert_eq!(created_at(&library.put(other_desk)), ["home"]);
    assert_eq!(
        library.latest().map(|p| p.created_at.as_str()),
        Some("home")
    );
    assert_eq!(library.len(), 3);
}

#[test]
fn test_library_keeps_at_most_max_profiles_least_recently_used_first() {
    let mut library = CalibrationLibrary::new([], 3).unwrap();
    let datas: Vec<CalibrationData> = (0..4)
        .map(|i| data_with(&i.to_string(), &format!("c{i}")))
        .collect();
    for d in &datas[..3] {
        library.put(d.clone());
    }
    assert!(library.mark_used(&datas[0].key())); // c0 used most recently now
    let evicted = library.put(datas[3].clone());
    assert_eq!(created_at(&evicted), ["c1"]); // the least recently used
    assert_eq!(labels(&library), ["c3", "c0", "c2"]);
    let dir = temp();
    let path = dir.path().join("calibration.json");
    library.save(&path).unwrap();
    let reloaded = CalibrationLibrary::load(&path);
    assert_eq!(labels(&reloaded), ["c3", "c0", "c2"]); // order persists
    assert_eq!(CalibrationLibrary::load_limited(&path, 2).unwrap().len(), 2);
    assert_eq!(MAX_PROFILES, 8);
    let message = CalibrationLibrary::new([], 0).unwrap_err().to_string();
    assert!(message.contains("max_profiles"));
}

#[test]
fn test_library_remove_clear_and_membership() {
    let (a, b) = (data_with("0", "a"), data_with("1", "b"));
    let mut library = library_of(vec![a.clone(), b]);
    assert!(library.contains(&a.key()));
    assert!(!library.mark_used(&data_with("2", "c").key()));
    assert!(library.remove(&a.key()));
    assert!(!library.remove(&a.key()));
    assert!(!library.contains(&a.key()));
    assert_eq!(labels(&library), ["b"]);
    library.clear();
    assert!(library.latest().is_none());
    assert_eq!(library.len(), 0);
}

#[test]
fn test_corrupt_profile_is_skipped_and_the_others_kept() {
    let logs = capture_logs();
    let dir = temp();
    let path = dir.path().join("calibration.json");
    library_of(vec![data_with("0", "x"), data_with("1", "y")])
        .save(&path)
        .unwrap();
    let mut doc = read_doc(&path);
    doc["profiles"][0]["model"]["bounds"] = json!([0, 0, "wide", 10]);
    write_doc(&path, &doc);
    let library = CalibrationLibrary::load(&path);
    let cameras: Vec<&str> = library.iter().map(|p| p.camera.as_str()).collect();
    assert_eq!(cameras, ["1"]);
    assert!(warnings(&logs).iter().any(|m| m.contains("profile 1")));
}

#[test]
fn test_duplicate_keys_in_a_file_keep_the_most_recent() {
    let dir = temp();
    let path = dir.path().join("calibration.json");
    library_of(vec![make_data(Spec {
        created_at: "new",
        ..Spec::default()
    })])
    .save(&path)
    .unwrap();
    let mut doc = read_doc(&path);
    let duplicate = with(&doc["profiles"][0], "created_at", json!("old"));
    doc["profiles"].as_array_mut().unwrap().push(duplicate);
    write_doc(&path, &doc);
    assert_eq!(labels(&CalibrationLibrary::load(&path)), ["new"]);
}

#[test]
fn test_profiles_are_shared_by_reference_so_refits_persist() {
    // Python held the profiles by reference; here a change through get_mut is saved.
    let dir = temp();
    let path = dir.path().join("calibration.json");
    let mut library = library_of(vec![make_data(Spec {
        implicit: 0,
        ..Spec::default()
    })]);
    let key = library
        .best("facemesh", "facemesh-pose-iris-1", &MONITORS, None, None)
        .unwrap()
        .key();
    library.get_mut(&key).unwrap().implicit_samples = vec![CalibrationSample::weighted(
        vec![1.0; 8],
        5.0,
        6.0,
        0,
        -1,
        0.5,
    )];
    library.save(&path).unwrap();
    assert_eq!(load_calibration(&path).unwrap().implicit_samples.len(), 1);
}

#[test]
fn test_empty_library_saves_a_valid_file() {
    let dir = temp();
    let path = dir.path().join("calibration.json");
    CalibrationLibrary::default().save(&path).unwrap();
    assert_eq!(read_doc(&path)["profiles"], json!([]));
    assert!(load_calibration(&path).is_none());
}

// ------------------------------------------------------------ gaze accuracy
static SIDE_BY_SIDE: LazyLock<Vec<Monitor>> = LazyLock::new(|| {
    vec![
        Monitor {
            primary: true,
            ..Monitor::new(0, "left", Rect::new(0, 0, 1920, 1080))
        },
        Monitor::new(1, "right", Rect::new(1920, 0, 1920, 1080)),
    ]
});

/// Two monitors; the features are the gaze point in kilo-pixels plus noise.
fn linear_data(report: Value) -> CalibrationData {
    let mut rng = Rng::new(11);
    let mut samples = Vec::new();
    let mut point = 0;
    for m in SIDE_BY_SIDE.iter() {
        for nx in [0.1, 0.5, 0.9] {
            for ny in [0.1, 0.5, 0.9] {
                let (x, y) = m.rect.denormalize(nx, ny);
                for _ in 0..4 {
                    let noise = rng.normal_vec(0.0, 0.02, 2);
                    let noisy = vec![x / 1000.0 + noise[0], y / 1000.0 + noise[1]];
                    samples.push(CalibrationSample::new(noisy, x, y, m.index, point));
                }
                point += 1;
            }
        }
    }
    let (x, y, _) = samples_to_arrays(&samples).unwrap();
    let model = GazeModel::new(1, 0.01, None)
        .unwrap()
        .fit(&x, &y, None, virtual_bounds(SIDE_BY_SIDE.iter()).ok(), None)
        .unwrap();
    let mut data = CalibrationData::new(
        "lite",
        "v1",
        layout_signature(SIDE_BY_SIDE.iter()),
        SIDE_BY_SIDE.to_vec(),
        samples,
        vec![],
        model,
    );
    data.report = object(report);
    data
}

#[test]
fn test_axis_error_prefers_the_stored_value() {
    let mut data =
        linear_data(json!({"per_monitor_error_px": {"0": [12.0, 34.0]}, "mean_error_px": 99.0}));
    let monitors = data.monitors.clone();
    assert_eq!(axis_error(&mut data, &monitors[0]), Some((12.0, 34.0)));
    // Matched by rectangle: the same screen under another number.
    let renumbered = Monitor::new(5, "left", monitors[0].rect);
    assert_eq!(axis_error(&mut data, &renumbered), Some((12.0, 34.0)));
    // A monitor without a stored value falls back to the mean error / √2.
    let (x, y) = axis_error(&mut data, &monitors[1]).unwrap();
    assert_close(x, 70.0, 0.01);
    assert_close(y, 70.0, 0.01);
}

#[test]
fn test_axis_error_is_recomputed_for_old_profiles_and_saved() {
    let mut data =
        linear_data(json!({"grade": "good", "alpha": 0.01, "degree": 1, "mean_error_px": 500.0}));
    let monitors = data.monitors.clone();
    let (sx, sy) = axis_error(&mut data, &monitors[1]).unwrap_or((0.0, 0.0));
    // Feature noise of 0.02 kilo-pixels is some 20 px of gaze error, far below the
    // fallback from the (deliberately wrong) mean error.
    assert!(5.0 < sx && sx < 80.0, "{sx}");
    assert!(5.0 < sy && sy < 80.0, "{sy}");
    let stored = data.report["per_monitor_error_px"].as_object().unwrap();
    assert_eq!(stored.keys().collect::<Vec<_>>(), ["0", "1"]);
    let dir = temp();
    let path = dir.path().join("calibration.json");
    library_of(vec![data]).save(&path).unwrap();
    let mut reloaded = load_calibration(&path).unwrap();
    let (rx, ry) = axis_error(&mut reloaded, &monitors[1]).unwrap();
    assert_close(rx, sx, 1e-6 * sx);
    assert_close(ry, sy, 1e-6 * sy);
}

#[test]
fn test_axis_error_without_any_information() {
    let mut data = linear_data(json!({"mean_error_px": null}));
    data.samples.truncate(4); // a single dot: nothing can be held out
    let first = data.monitors[0].clone();
    assert_eq!(axis_error(&mut data, &first), None);
    let mut data = linear_data(json!({"mean_error_px": 42.0}));
    data.samples.clear();
    let (x, y) = axis_error(&mut data, &first).unwrap();
    assert_close(x, 29.7, 0.01);
    assert_close(y, 29.7, 0.01);
}

#[test]
fn test_recomputing_an_old_profile_is_cheap_enough_for_the_gui_thread() {
    // The first axis_error of an old profile runs on the GUI thread (see its docs):
    // a realistic two-monitor calibration (16 dots per monitor, 30 samples each,
    // degree 3, the face-mesh gaze features nonlinear) must take milliseconds.
    let options = SampleOptions {
        per_point: 30,
        points_per_monitor: 16,
        ..SampleOptions::default()
    };
    let samples = calibration_samples(&TWO, &mut Rng::new(3), 1.0, &options);
    let (model, report) =
        crate::gaze::calibration::evaluate(&samples, &TWO, Some(3), Some(&GAZE)).unwrap();
    let mut saved = report.to_dict();
    // as written by an earlier version
    let expected = saved.remove("per_monitor_error_px").unwrap();
    let mut data = CalibrationData::new(
        "facemesh",
        "v",
        layout_signature(TWO.iter()),
        TWO.to_vec(),
        samples,
        vec![],
        model,
    );
    data.report = saved;
    let started = std::time::Instant::now();
    let sigma = axis_error(&mut data, &TWO[1]).unwrap();
    let elapsed = started.elapsed().as_secs_f64();
    let want = expected["1"].as_array().unwrap();
    assert_close(sigma.0, want[0].as_f64().unwrap(), 1e-6 * sigma.0); // same as a new report
    assert_close(sigma.1, want[1].as_f64().unwrap(), 1e-6 * sigma.1);
    // Measured: ~4 ms on a desktop in Python. The bound leaves room for slow CI machines.
    assert!(elapsed < 0.25, "{:.0} ms", elapsed * 1000.0);
}

// ------------------------------------------------------------ the Python app's file
/// A calibration.json written by the Python app (`CalibrationLibrary.save` of an office profile
/// recorded with the `opencv` backend, later renamed `lite`, and a home profile; three features,
/// degree 2, look-away regions), as `uv run python` produced it.
const PYTHON_FILE: &str = r##"{
  "format": "eye-tracker-calibration",
  "version": 2,
  "profiles": [
    {
      "created_at": "2026-09-30T08:15:00+00:00",
      "backend": "opencv",
      "feature_version": "yunet-geom-1",
      "layout_signature": "e6fe4e98ba22c6ef",
      "camera": "0",
      "frame_size": [640,480],
      "monitors": [
        {"index":0,"name":"DELL U2720Q","rect":[0,0,3840,2160],"primary":true,"scale":1.5},
        {"index":1,"name":"LG ÜltraGear","rect":[3840,540,1920,1080],"primary":false,"scale":1.0}
      ],
      "report": {
        "grade": "good",
        "monitor_accuracy": 0.93,
        "mean_error_px": 84.5,
        "median_error_px": null,
        "per_monitor_accuracy": {
          "0": 1.0,
          "1": 0.86
        },
        "n_samples": 16,
        "n_points": 8,
        "alpha": 0.5,
        "degree": 2,
        "uncovered_monitors": [],
        "per_monitor_error_px": {
          "0": [
            60.25,
            41.5
          ],
          "1": [
            72.0,
            55.125
          ]
        }
      },
      "model": {
        "kind": "ridge-poly",
        "degree": 2,
        "alpha": 0.5,
        "nonlinear": [0,1],
        "mean": [-0.1093630303,-0.1671926586,0.07100450775],
        "std": [1.119894868,0.91991984,0.4617869469],
        "lo": [-2.711162479,-1.889013246,-0.6832266618],
        "hi": [2.117838755,1.294063814,1.006724315],
        "coef": [[0.5797288421,0.4380705918],[0.08646615768,0.01323926634],[-0.1090273785,-0.0003381072666],[0.1351539844,0.001884784832],[0.01693291933,0.02103057768],[-0.01639743571,0.008327099634],[-0.08506742893,-0.04825720067]],
        "bounds": [0,0,5760,2160],
        "linear_coef": [[0.5083333333,0.4125],[0.07363130833,0.01853247203],[-0.01124590668,0.004003124579],[0.03479035566,-0.008068994824]],
        "away_regions": [{"rect":[0,0,3840,2160],"box":[0,0,3840,2160],"diagonal":4405.81434},{"rect":[3840,540,1920,1080],"box":[3840,540,5760,1620],"diagonal":2202.90717}]
      },
      "samples": [
        {"features":[0.345584,0.821618,0.330437],"x":768.0,"y":648.0,"monitor":0,"point":0,"weight":1.0},
        {"features":[-1.303157,0.905356,0.446375],"x":768.0,"y":648.0,"monitor":0,"point":0,"weight":1.0},
        {"features":[-0.536953,0.581118,0.364572],"x":4224.0,"y":972.0,"monitor":1,"point":1,"weight":1.0},
        {"features":[0.294132,0.028422,0.546713],"x":4224.0,"y":972.0,"monitor":1,"point":1,"weight":1.0},
        {"features":[-0.736454,-0.16291,-0.482119],"x":1152.0,"y":1080.0,"monitor":0,"point":2,"weight":1.0},
        {"features":[0.598846,0.039722,-0.292457],"x":1152.0,"y":1080.0,"monitor":0,"point":2,"weight":1.0},
        {"features":[-0.781908,-0.257192,0.008142],"x":4416.0,"y":864.0,"monitor":1,"point":3,"weight":1.0},
        {"features":[-0.275603,1.294064,1.006724],"x":4416.0,"y":864.0,"monitor":1,"point":3,"weight":1.0},
        {"features":[-2.711162,-1.889013,-0.174772],"x":1536.0,"y":864.0,"monitor":0,"point":4,"weight":1.0},
        {"features":[-0.42219,0.213643,0.217322],"x":1536.0,"y":864.0,"monitor":0,"point":4,"weight":1.0},
        {"features":[2.117839,-1.112021,-0.377605],"x":4608.0,"y":1080.0,"monitor":1,"point":5,"weight":1.0},
        {"features":[2.042772,0.646703,0.663063],"x":4608.0,"y":1080.0,"monitor":1,"point":5,"weight":1.0},
        {"features":[-0.514006,-1.648075,0.167465],"x":1920.0,"y":648.0,"monitor":0,"point":6,"weight":1.0},
        {"features":[0.109014,-1.227352,-0.683227],"x":1920.0,"y":648.0,"monitor":0,"point":6,"weight":1.0},
        {"features":[-0.072044,-0.944752,-0.09827],"x":4800.0,"y":972.0,"monitor":1,"point":7,"weight":1.0},
        {"features":[0.095483,0.035586,-0.506292],"x":4800.0,"y":972.0,"monitor":1,"point":7,"weight":1.0}
      ],
      "implicit_samples": [
        {"features":[0.593748,0.891167,0.320848],"x":100.0,"y":200.0,"monitor":0,"point":-1,"weight":0.5},
        {"features":[-0.81823,0.731652,-0.50144],"x":101.0,"y":200.0,"monitor":0,"point":-2,"weight":0.5}
      ]
    },
    {
      "created_at": "2026-09-29T19:00:00+00:00",
      "backend": "facemesh",
      "feature_version": "facemesh-pose-iris-1",
      "layout_signature": "c15ea7c25bc78984",
      "camera": "",
      "frame_size": [0,0],
      "monitors": [
        {"index":0,"name":"laptop","rect":[0,0,1920,1200],"primary":true,"scale":1.0},
        {"index":1,"name":"monitor","rect":[1920,0,1920,1080],"primary":false,"scale":1.0}
      ],
      "report": {
        "grade": "good",
        "monitor_accuracy": 0.93,
        "mean_error_px": 84.5,
        "median_error_px": null,
        "per_monitor_accuracy": {
          "0": 1.0,
          "1": 0.86
        },
        "n_samples": 16,
        "n_points": 8,
        "alpha": 0.5,
        "degree": 2,
        "uncovered_monitors": [],
        "per_monitor_error_px": {
          "0": [
            60.25,
            41.5
          ],
          "1": [
            72.0,
            55.125
          ]
        }
      },
      "model": {
        "kind": "ridge-poly",
        "degree": 2,
        "alpha": 0.5,
        "nonlinear": [0,1],
        "mean": [-0.2948087607,0.08813512497,0.321782151],
        "std": [0.8607663017,0.8828432464,0.9931454701],
        "lo": [-2.441467383,-1.729411467,-1.504831414],
        "hi": [0.9692722011,1.799707383,2.056702818],
        "coef": [[0.300762231,0.3885629859],[0.2629182191,-0.02200062482],[0.05905425958,-0.004389422956],[0.07017473241,-0.00825196469],[-0.01903176707,0.02522558383],[0.03082414736,0.025652515],[0.1485053398,-0.04193122176]],
        "bounds": [0,0,3840,1200],
        "linear_coef": [[0.425,0.3675],[0.0851114471,0.006556361333],[-0.01520221423,-0.004876448131],[0.01530348757,-0.003790622711]],
        "away_regions": [{"rect":[0,0,1920,1200],"box":[0,0,1920,1200],"diagonal":2264.155472},{"rect":[1920,0,1920,1080],"box":[1920,0,3840,1080],"diagonal":2202.90717}]
      },
      "samples": [
        {"features":[0.189053,-0.522748,-0.413064],"x":384.0,"y":360.0,"monitor":0,"point":0,"weight":1.0},
        {"features":[-2.441467,1.799707,1.144166],"x":384.0,"y":360.0,"monitor":0,"point":0,"weight":1.0},
        {"features":[-0.325423,0.773807,0.281211],"x":2304.0,"y":432.0,"monitor":1,"point":1,"weight":1.0},
        {"features":[-0.553823,0.977567,-0.310557],"x":2304.0,"y":432.0,"monitor":1,"point":1,"weight":1.0},
        {"features":[-0.328824,-0.792147,0.454958],"x":576.0,"y":600.0,"monitor":0,"point":2,"weight":1.0},
        {"features":[-0.099198,0.545289,-0.607186],"x":576.0,"y":600.0,"monitor":0,"point":2,"weight":1.0},
        {"features":[0.126828,-0.892274,0.841465],"x":2496.0,"y":324.0,"monitor":1,"point":3,"weight":1.0},
        {"features":[0.188035,0.330571,0.410504],"x":2496.0,"y":324.0,"monitor":1,"point":3,"weight":1.0},
        {"features":[-1.010758,0.783181,2.056703],"x":768.0,"y":480.0,"monitor":0,"point":4,"weight":1.0},
        {"features":[-1.638443,-1.729411,-1.504831],"x":768.0,"y":480.0,"monitor":0,"point":4,"weight":1.0},
        {"features":[0.841459,0.128716,1.078342],"x":2688.0,"y":540.0,"monitor":1,"point":5,"weight":1.0},
        {"features":[0.722431,0.210572,0.284038],"x":2688.0,"y":540.0,"monitor":1,"point":5,"weight":1.0},
        {"features":[-0.16976,0.86846,-1.129716],"x":960.0,"y":360.0,"monitor":0,"point":6,"weight":1.0},
        {"features":[-0.421859,0.242939,1.801421],"x":960.0,"y":360.0,"monitor":0,"point":6,"weight":1.0},
        {"features":[-0.764464,-1.07906,-0.563287],"x":2880.0,"y":432.0,"monitor":1,"point":7,"weight":1.0},
        {"features":[0.969272,-0.235006,1.324347],"x":2880.0,"y":432.0,"monitor":1,"point":7,"weight":1.0}
      ],
      "implicit_samples": [
        {"features":[-1.87253,1.128523,1.034866],"x":100.0,"y":200.0,"monitor":0,"point":-1,"weight":0.5},
        {"features":[-1.418666,0.153582,1.215758],"x":101.0,"y":200.0,"monitor":0,"point":-2,"weight":0.5}
      ]
    }
  ]
}
"##;

#[test]
fn test_loads_a_calibration_file_written_by_the_python_app() {
    let dir = temp();
    let path = dir.path().join("calibration.json");
    std::fs::write(&path, PYTHON_FILE).unwrap();
    let library = CalibrationLibrary::load(&path);
    assert_eq!(library.len(), 2);

    let office = &library.profiles()[0];
    // The renamed backend is stored under its current name.
    assert_eq!(
        (office.backend.as_str(), office.feature_version.as_str()),
        ("lite", "yunet-geom-1")
    );
    assert_eq!(office.layout_signature, "e6fe4e98ba22c6ef");
    assert_eq!(office.layout_signature, layout_signature(MONITORS.iter()));
    assert_eq!(office.monitors, *MONITORS);
    assert_eq!(
        (office.camera.as_str(), office.frame_size),
        ("0", (640, 480))
    );
    assert_eq!(office.created_at, "2026-09-30T08:15:00+00:00");
    assert_eq!(
        (office.samples.len(), office.implicit_samples.len()),
        (16, 2)
    );
    assert_eq!(office.implicit_samples[1].weight, 0.5);
    assert_eq!(office.implicit_samples[1].point_id, -2);
    assert_eq!(office.grade(), Some("good"));
    assert!(office.report["median_error_px"].is_null());
    assert_eq!(office.model.away_regions().len(), 2);
    assert_eq!(office.model.nonlinear(), Some(&[0, 1][..]));
    assert_eq!(
        office.is_compatible(
            "lite",
            "yunet-geom-1",
            &MONITORS,
            Some("0"),
            Some((640, 480))
        ),
        ok()
    );
    let home = &library.profiles()[1];
    assert_eq!(
        (home.backend.as_str(), home.camera.as_str(), home.frame_size),
        ("facemesh", "", (0, 0))
    );
    assert_eq!(home.layout_signature, layout_signature(HOME.iter()));

    // The Python model's predictions for two queries.
    let queries = nalgebra::DMatrix::from_row_slice(2, 3, &[0.1, -0.2, 0.3, 1.0, 0.5, -0.5]);
    for (profile, expected) in [
        (
            office,
            [
                3844.2026468142044,
                954.9586399117812,
                2184.6291191724176,
                971.6647852442188,
            ],
        ),
        (
            home,
            [
                1565.7327315025507,
                452.50305187003096,
                2598.254196051177,
                511.44559634143064,
            ],
        ),
    ] {
        let predicted = profile.model.predict_many(&queries).unwrap();
        for (got, want) in predicted.transpose().iter().zip(expected) {
            assert_close(*got, want, 1e-6);
        }
    }

    // Rust writes the same document back (only the backend's current name differs), so the
    // Python app reads what the Rust app writes. Python wrote the boxes of regions that were
    // not learned from samples as integers and Rust writes them as floats, so numbers
    // are compared by value; the key order is checked in test_key_order_of_the_file_matches_python.
    let written: Value = serde_json::from_str(&library.to_text()).unwrap();
    let original: Value =
        serde_json::from_str(&PYTHON_FILE.replace("\"opencv\"", "\"lite\"")).unwrap();
    assert!(same_json(&written, &original));
    let rewritten = dir.path().join("rewritten.json");
    library.save(&rewritten).unwrap();
    assert_eq!(
        labels(&CalibrationLibrary::load(&rewritten)),
        labels(&library)
    );
}
