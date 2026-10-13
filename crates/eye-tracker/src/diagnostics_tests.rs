//! Tests for `diagnostics` (`doctor` and `bench`), ported from tests/test_diagnostics.py.
//!
//! Nothing here opens a camera: camera probing is faked, and the benchmark runs on a synthetic
//! image written to a temporary directory. The machine the report asks (platform, monitors,
//! cameras, hotkey manager, running instance, start at login) is a [`Host`] built from fakes; the
//! home directory is replaced through `TEST_HOMES` where Python monkeypatched `Path.home`.
//!
//! Not ported:
//! * `test_an_installed_mediapipe_is_reported_as_a_problem`: the check for unwanted Python
//!   distributions is gone with the Python environment (see the module docs).
//! * The Qt assertions of `test_collect_report_structure` (`qt.platform_plugin == "offscreen"`,
//!   `monitors.virtual is True`) and the `env` fixture's `qapp`: there is no Qt; the report has a
//!   `ui` section instead, and the monitors come from a fake host because
//!   `platform::monitors()` is still a stand-in in `diagnostics.rs`.
//! * `diagnostics._redact(None) is None` of `test_redact`: `redact` takes a `&str`; absent values
//!   are `Option::map`ped by the callers.
//! * `test_run_bench_errors` has no `ValueError`/`CameraError` classes to match: they are
//!   `BenchError::Invalid` and `BenchError::Camera`.
//!
//! New: `test_collect_report_on_this_machine` (the real [`Host`]),
//! `test_instance_status_over_ipc`, `test_format_g`, `test_the_ui_and_libraries_sections` and
//! `test_percentiles_match_numpy`.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::rc::Rc;

use parking_lot::Mutex;

use super::*;
use crate::gaze::calibration::{CalibrationSample, samples_to_arrays};
use crate::gaze::model::GazeModel;
use crate::gaze::store::save_calibration;
use crate::platform::hotkeys::{Hotkey, HotkeyCallback, HotkeyError, UnsupportedHotkeyManager};
use crate::testing::gaze_synth::Rng;
use crate::testing::{AppDirs, app_dirs};

const SECTIONS: [&str; 15] = [
    "generated_at",
    "app",
    "system",
    "ui",
    "monitors",
    "libraries",
    "backends",
    "cameras",
    "platform",
    "hotkeys",
    "autostart",
    "paths",
    "settings",
    "calibration",
    "problems",
];

// ----------------------------------------------------------------------- fakes
/// Platform services with chosen answers (capabilities all available, like `_MacLikeServices`).
struct FakeServices {
    wayland: bool,
    permissions: Vec<(&'static str, Option<bool>)>,
    accessibility: &'static str,
}

impl Default for FakeServices {
    fn default() -> Self {
        FakeServices {
            wayland: false,
            permissions: vec![("camera", Some(true)), ("accessibility", None)],
            accessibility: "unknown",
        }
    }
}

impl PlatformServices for FakeServices {
    fn name(&self) -> &'static str {
        "fake"
    }

    fn capabilities(&self) -> HashMap<&'static str, bool> {
        CAPABILITY_ORDER.iter().map(|name| (*name, true)).collect()
    }

    fn permissions(&self) -> HashMap<&'static str, Option<bool>> {
        self.permissions.iter().copied().collect()
    }

    fn accessibility_status(&self) -> &'static str {
        self.accessibility
    }

    fn is_wayland(&self) -> bool {
        self.wayland
    }
}

/// A hotkey manager that knows what it registered and which combination types a character.
#[derive(Default)]
struct FakeHotkeys {
    supported: bool,
    note: Mutex<Option<String>>,
    /// `(hotkey text, message)`
    conflict: Option<(&'static str, &'static str)>,
    registered: HashMap<String, Hotkey>,
    errors: HashMap<String, String>,
}

impl HotkeyManager for FakeHotkeys {
    fn supported(&self) -> bool {
        self.supported
    }

    fn name(&self) -> &'static str {
        "fake"
    }

    fn note(&self) -> Option<String> {
        self.note.lock().clone()
    }

    fn register(&self, _name: &str, _hotkey: HotkeyArg<'_>, _callback: HotkeyCallback) -> bool {
        false
    }

    fn unregister(&self, _name: &str) -> bool {
        false
    }

    fn unregister_all(&self) {}

    fn registered(&self) -> HashMap<String, Hotkey> {
        self.registered.clone()
    }

    fn last_error(&self, name: &str) -> Option<String> {
        self.errors.get(name).cloned()
    }

    fn layout_conflict(&self, hotkey: HotkeyArg<'_>) -> Result<Option<String>, HotkeyError> {
        let hotkey = match hotkey {
            HotkeyArg::Hotkey(hotkey) => hotkey,
            HotkeyArg::Text(text) => parse_hotkey(text)?,
        };
        Ok(self
            .conflict
            .filter(|(text, _)| hotkey.to_string() == *text)
            .map(|(_, message)| message.to_string()))
    }

    fn start(&self) {}

    fn stop(&self) {}
}

fn fake_monitors() -> Vec<Monitor> {
    vec![
        Monitor {
            primary: true,
            ..Monitor::new(0, "left", Rect::new(0, 0, 1920, 1080))
        },
        Monitor::new(1, "right", Rect::new(1920, 0, 2560, 1440)),
    ]
}

/// A machine with two monitors, no camera, no hotkey support, nothing running.
fn host() -> Host {
    Host {
        os: "linux",
        executable: PathBuf::from("/usr/bin/eye-tracker"),
        services: Arc::new(FakeServices::default()),
        lock_methods: Box::new(|| None),
        monitors: Box::new(|| Ok(fake_monitors())),
        list_cameras: Box::new(|_, _| Ok(Vec::new())),
        hotkey_manager: Box::new(|| Box::new(UnsupportedHotkeyManager::new(None))),
        instance_running: Box::new(|| false),
        instance_status: Box::new(|| None),
        autostart: Box::new(|| AutostartInfo {
            supported: true,
            status: Status::Disabled,
            enabled: false,
            location: None,
            registered: None,
            command: vec!["eye-tracker".into(), "--background".into()],
        }),
    }
}

/// The home directory of these tests (`Path.home` monkeypatched); restored on drop.
struct Home;

fn fake_home(homes: &[&str]) -> Home {
    TEST_HOMES.with(|h| *h.borrow_mut() = Some(homes.iter().map(|s| (*s).to_string()).collect()));
    Home
}

impl Drop for Home {
    fn drop(&mut self) {
        TEST_HOMES.with(|h| *h.borrow_mut() = None);
    }
}

fn object(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        other => panic!("not an object: {other}"),
    }
}

fn calibration(monitors: &[Monitor], backend: &str, feature_version: &str) -> CalibrationData {
    let mut rng = Rng::new(3);
    let mut samples = Vec::new();
    for pid in 0..8 {
        let monitor = &monitors[pid % monitors.len()];
        let (x, y) = monitor
            .rect
            .denormalize(rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9));
        for _ in 0..3 {
            samples.push(CalibrationSample::new(
                rng.normal_vec(0.0, 1.0, 6),
                x,
                y,
                monitor.index,
                pid as i64,
            ));
        }
    }
    let (features, targets, _) = samples_to_arrays(&samples).unwrap();
    let model = GazeModel::new(1, 1.0, None)
        .unwrap()
        .fit(
            &features,
            &targets,
            None,
            virtual_bounds(monitors).ok(),
            None,
        )
        .unwrap();
    let mut data = CalibrationData::new(
        backend,
        feature_version,
        layout_signature(monitors),
        monitors.to_vec(),
        samples,
        Vec::new(),
        model,
    );
    data.report = object(json!({
        "grade": "good",
        "monitor_accuracy": 0.93,
        "mean_error_px": 120.0,
        "median_error_px": 100.0,
        "per_monitor_accuracy": {"0": 0.93},
        "n_samples": 24,
        "n_points": 8,
        "alpha": 1.0,
    }));
    data.created_at = "2026-09-30T08:15:00+00:00".into();
    data
}

/// A smooth synthetic picture (no face), written for the benchmark.
fn face_free_image(path: &Path) {
    let image = image::RgbImage::from_fn(320, 240, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
    });
    image.save(path).unwrap();
}

fn save_settings(settings: &Settings) {
    settings.save(&paths::settings_file()).unwrap();
}

fn sections(report: &Value) -> BTreeSet<String> {
    report.as_object().unwrap().keys().cloned().collect()
}

fn problems_of(report: &Value) -> Vec<String> {
    report["problems"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p.as_str().unwrap().to_string())
        .collect()
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_string()).collect()
}

// ------------------------------------------------------------------------- doctor
#[test]
fn test_collect_report_structure() {
    let _dirs = app_dirs();
    let report = collect_report_with(&host(), false, None);
    assert_eq!(
        sections(&report),
        SECTIONS.iter().map(|s| s.to_string()).collect()
    );
    // The whole report is JSON-serialisable.
    serde_json::from_str::<Value>(&serde_json::to_string(&report).unwrap()).unwrap();

    assert_eq!(report["app"]["version"], VERSION);
    assert_eq!(report["app"]["instance_running"], false);
    assert_eq!(report["monitors"]["count"], 2);
    let models: BTreeSet<&str> = report["backends"]["models"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        models,
        MODEL_FILES
            .iter()
            .map(|(name, _)| *name)
            .collect::<BTreeSet<_>>()
    );
    assert!(report["backends"].get("mediapipe_installed").is_none());
    assert!(report["libraries"].get("mediapipe").is_none());
    assert_eq!(report["cameras"]["probed"], false);
    assert!(report["cameras"].get("devices").is_none());
    for capability in ["lock", "cursor", "hotkeys"] {
        assert!(report["platform"]["capabilities"].get(capability).is_some());
    }
    assert_eq!(
        report["settings"],
        json!({"file_exists": false, "non_default": {}})
    );
    assert_eq!(report["calibration"], json!({"exists": false}));
    assert!(report["autostart"]["command"].is_string());
    // The command is spelled as this copy is run (packages have no "eye-tracker").
    let command = redacted_command(&cli_command(&["calibrate"]), POSIX_SHELL);
    assert_eq!(
        report["app"]["command_line"],
        redacted_command(&cli_command(&[]), POSIX_SHELL)
    );
    let problems = problems_of(&report);
    assert!(problems.contains(&format!(
        "Not calibrated yet: choose 'Calibrate...' in the tray menu or run '{command}'."
    )));
    assert!(!problems.iter().any(|p| p.contains("Only one monitor")));
}

#[test]
fn test_collect_report_on_this_machine() {
    let _dirs = app_dirs();
    let report = collect_report(false, None);
    assert_eq!(
        sections(&report),
        SECTIONS.iter().map(|s| s.to_string()).collect()
    );
    assert_eq!(report["app"]["version"], VERSION);
    assert!(format_report(&report).starts_with(&format!("{APP_NAME} {VERSION} - diagnostics\n")));
}

#[test]
fn test_report_paths_are_relative_to_home() {
    let _dirs = app_dirs();
    let home = paths::config_dir().to_string_lossy().into_owned();
    let _home = fake_home(&[&home]);
    let report = collect_report_with(&host(), false, None);
    let settings_file = report["paths"]["settings_file"].as_str().unwrap();
    assert!(settings_file.starts_with('~'));
    assert!(!settings_file.contains(&home));
    // The socket name is a hash of the user name: easy to reverse, so left out.
    assert!(report["paths"].get("ipc_name").is_none());
    assert!(!report.to_string().contains(&paths::ipc_name()));
    assert_eq!(report["paths"]["profile"], "custom (--config-dir)");
}

#[test]
fn test_camera_probe_is_opt_in() {
    let _dirs = app_dirs();
    let calls: Rc<RefCell<Vec<(u32, String)>>> = Rc::default();
    let mut host = host();
    host.list_cameras = {
        let calls = Rc::clone(&calls);
        Box::new(move |max_index, api| {
            calls.borrow_mut().push((max_index, api.to_string()));
            Ok(vec![CameraInfo {
                index: 0,
                name: "Camera 0".into(),
                width: 640,
                height: 480,
            }])
        })
    };
    collect_report_with(&host, false, None);
    assert!(calls.borrow().is_empty());
    let report = collect_report_with(&host, true, None);
    assert_eq!(*calls.borrow(), vec![(4, "auto".to_string())]);
    assert_eq!(
        report["cameras"]["devices"],
        json!([{"index": 0, "name": "Camera 0", "width": 640, "height": 480}])
    );
    let text = format_report(&report);
    assert!(text.contains("camera 0"));
    assert!(text.contains("Camera 0 (640x480)"));

    host.list_cameras = Box::new(|_, _| Ok(Vec::new()));
    let report = collect_report_with(&host, true, None);
    assert!(problems_of(&report).contains(&"No camera delivered a picture.".to_string()));
    assert!(format_report(&report).contains("(no camera found)"));

    host.list_cameras = Box::new(|_, _| Err(CameraError("busy".into())));
    let report = collect_report_with(&host, true, None);
    assert_eq!(report["cameras"], json!({"error": "CameraError: busy"}));
}

#[test]
fn test_settings_section_lists_changes_without_side_effects() {
    let _dirs = app_dirs();
    let mut settings = Settings::default();
    settings.general.backend = "lite".into();
    settings.presence.away_timeout_s = 120;
    save_settings(&settings);
    let report = collect_report_with(&host(), false, None);
    assert_eq!(report["settings"]["valid"], true);
    assert_eq!(
        report["settings"]["non_default"],
        json!({"general.backend": "lite", "presence.away_timeout_s": 120})
    );
    assert_eq!(report["backends"]["configured"], "lite");
    assert_eq!(report["backends"]["active"], "lite");

    std::fs::write(paths::settings_file(), "{broken").unwrap();
    let report = collect_report_with(&host(), false, None);
    assert_eq!(report["settings"]["valid"], false);
    assert!(
        problems_of(&report)
            .contains(&"The settings file is not valid JSON; defaults are used.".to_string())
    );
    // Unlike Settings::load, the doctor never moves the broken file aside.
    assert_eq!(
        std::fs::read_to_string(paths::settings_file()).unwrap(),
        "{broken"
    );
}

#[test]
fn test_calibration_section() {
    let _dirs = app_dirs();
    let mut settings = Settings::default();
    settings.general.backend = "lite".into();
    save_settings(&settings);
    let current = fake_monitors();
    save_calibration(
        &paths::calibration_file(),
        &calibration(&current, "lite", "yunet-geom-1"),
    )
    .unwrap();

    let report = collect_report_with(&host(), false, None);
    let section = &report["calibration"];
    assert_eq!(section["exists"], true);
    assert_eq!(section["valid"], true);
    assert_eq!(section["compatible"], true);
    assert_eq!(section["grade"], "good");
    assert_eq!(section["samples"], 24);
    assert!(section["summary"].as_str().unwrap().starts_with("Good"));

    let other = vec![
        Monitor {
            primary: true,
            ..Monitor::new(0, "left", Rect::new(-1920, 0, 1920, 1080))
        },
        Monitor::new(1, "right", Rect::new(0, 0, 2560, 1440)),
    ];
    std::fs::remove_file(paths::calibration_file()).unwrap();
    save_calibration(
        &paths::calibration_file(),
        &calibration(&other, "lite", "yunet-geom-1"),
    )
    .unwrap();
    let report = collect_report_with(&host(), false, None);
    assert_eq!(report["calibration"]["compatible"], false);
    assert!(
        report["calibration"]["reason"]
            .as_str()
            .unwrap()
            .contains("monitor layout")
    );
    assert!(
        problems_of(&report)
            .iter()
            .any(|p| p.starts_with("The calibration no longer applies"))
    );

    std::fs::write(paths::calibration_file(), "not json").unwrap();
    let report = collect_report_with(&host(), false, None);
    assert_eq!(
        report["calibration"],
        json!({"exists": true, "valid": false})
    );
    assert!(
        problems_of(&report)
            .contains(&"The calibration file is damaged; please recalibrate.".to_string())
    );
}

#[test]
fn test_format_report() {
    let _dirs = app_dirs();
    let report = collect_report_with(&host(), false, None);
    let text = format_report(&report);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines[0], format!("Eye Tracker {VERSION} - diagnostics"));
    assert_eq!(lines[1], "=".repeat(lines[0].len()));
    for heading in [
        "Problems",
        "App",
        "UI",
        "Monitors",
        "Vision backends",
        "Paths",
    ] {
        assert!(lines.contains(&heading), "{heading}");
    }
    assert!(text.ends_with('\n'));
    // Fixed texts are ASCII so the report survives any console code page.
    assert!(!text.contains('—'));
}

#[test]
fn test_format_report_handles_errors_and_odd_values() {
    let report = json!({
        "app": {"version": "9.9"},
        "problems": [],
        "system": {"error": "OSError: nope"},
        "settings": {"non_default": {}, "flag": true, "missing": null, "ratio": 0.5},
        "cameras": {"probed": false, "devices": []},
        "monitors": {"count": 0, "items": []},
    });
    let text = format_report(&report);
    assert!(text.contains("Eye Tracker 9.9 - diagnostics"));
    assert!(text.contains("none found"));
    assert!(text.contains("error  OSError: nope"));
    assert!(text.contains("flag         yes"));
    assert!(text.contains("missing      unknown"));
    assert!(text.contains("ratio        0.5"));
    assert!(text.contains("non_default  -"));
}

#[test]
fn test_problems_from_a_synthetic_report() {
    let report = json!({
        "app": {"instance_running": true},
        "system": {"error": "RuntimeError: x"},
        "backends": {
            "available": [],
            "models": {
                "a.tflite": {"used_by": ["facemesh"], "present": false},
                "b.onnx": {"used_by": ["facemesh", "lite"], "present": true, "sha256_ok": false},
            },
            "active_error": "No vision backend is available",
        },
        "monitors": {"count": 1},
        "cameras": {"probed": true, "devices": []},
        "platform": {
            "permissions": {"camera": false, "accessibility": false},
            "capabilities": {"cursor": false, "lock": false},
        },
        "hotkeys": {"enabled": true, "note": "Wayland", "invalid": {"recalibrate": "bad"}},
        "settings": {"valid": true},
        "calibration": {"exists": true, "valid": true, "compatible": true},
    });
    assert_eq!(
        problems(&report),
        vec![
            "Could not collect system information: RuntimeError: x",
            "No vision backend is available (it needs the bundled model files).",
            "The model file a.tflite (facemesh backend) is missing.",
            "The model file b.onnx (facemesh, lite backend) is damaged (checksum).",
            "No vision backend is available",
            "Only one monitor detected: switching needs two or more \
             (walk-away and privacy features still work).",
            "No camera delivered a picture (it may be in use by Eye Tracker itself).",
            "Camera access is blocked by the operating system's privacy settings.",
            "Accessibility permission is missing: keyboard focus cannot follow your gaze.",
            "This session does not allow moving the cursor. On Wayland, sway and Hyprland \
             work directly; elsewhere install ydotool 1.x and run ydotoold.",
            "Locking the screen is not supported here.",
            "Global hotkeys: Wayland",
            "Hotkey recalibrate is invalid: bad",
        ]
    );
}

#[test]
fn test_a_missing_calibration_is_a_problem_only_where_switching_needs_one() {
    // r3-ux-docs-03: one monitor, or switching turned off, needs no calibration.
    let not_calibrated = |monitors: Value, non_default: Value| {
        let report = json!({
            "monitors": monitors,
            "settings": {"valid": true, "non_default": non_default},
            "calibration": {"exists": false},
        });
        problems(&report)
            .iter()
            .filter(|p| p.starts_with("Not calibrated"))
            .count()
    };
    assert_eq!(not_calibrated(json!({"count": 2}), json!({})), 1);
    // A headless fake screen.
    assert_eq!(
        not_calibrated(json!({"count": 1, "virtual": true}), json!({})),
        1
    );
    assert_eq!(not_calibrated(json!({"count": 1}), json!({})), 0);
    assert_eq!(
        not_calibrated(json!({"count": 2}), json!({"switching.enabled": false})),
        0
    );
}

#[test]
fn test_redact() {
    let _home = fake_home(&["/home/alice"]);
    assert_eq!(redact("/home/alice"), "~");
    assert_eq!(redact("/home/alice/x/y.log"), "~/x/y.log");
    // A sibling whose name merely starts with the home path is left alone.
    assert_eq!(redact("/home/alice2/x"), "/home/alice2/x");
    assert_eq!(redact("relative/path"), "relative/path");
}

#[test]
fn test_format_command() {
    let text = format_command(&strings(&["/usr/bin/python 3", "-m", "eye_tracker"]));
    assert!(text.contains("eye_tracker"));
    // Only the path with a space is quoted.
    assert_eq!(text.matches(['"', '\'']).count(), 2);
}

#[test]
fn test_describe_device() {
    assert_eq!(describe_device("1"), "camera 1");
    assert_eq!(describe_device(" "), "none");
    let video = Path::new("videos").join("me.mp4");
    assert_eq!(describe_device(&video.to_string_lossy()), "file me.mp4");
}

// -------------------------------------------------------------------------- bench
#[test]
fn test_run_bench_on_an_image() {
    let dir = tempfile::tempdir().unwrap();
    let image = dir.path().join("frame.png");
    face_free_image(&image);
    let options = BenchOptions {
        idle_fps: Some(20.0),
        ..BenchOptions::default()
    };
    // YuNet takes 0.3 s a frame in the unoptimised test build (Python: 0.25 s): a second leaves
    // room for the frames the gate skips.
    let result = run_bench(1.0, &image.to_string_lossy(), "lite", &options).unwrap();
    assert_eq!(result["backend"], "lite");
    assert_eq!(result["feature_version"], "yunet-geom-1");
    assert_eq!(result["device"], "file frame.png");
    assert_eq!(result["frame_size"], json!([320, 240]));
    assert_eq!(
        sections(&result["modes"]),
        ["idle", "max"].iter().map(|s| s.to_string()).collect()
    );
    let (fast, idle) = (&result["modes"]["max"], &result["modes"]["idle"]);
    assert!(fast["target_fps"].is_null());
    assert!(fast["frames"].as_u64().unwrap() > 0);
    assert_eq!(fast["analysed"], fast["frames"]);
    assert_eq!(fast["face_ratio"].as_f64(), Some(0.0));
    assert!(fast["inference_ms"]["p50"].as_f64().is_some());
    assert_eq!(idle["target_fps"].as_f64(), Some(20.0));
    // A still image: the motion gate skips almost every frame after the first.
    let (analysed, frames) = (
        idle["analysed"].as_u64().unwrap(),
        idle["frames"].as_u64().unwrap(),
    );
    assert!(1 <= analysed && analysed < frames);
    assert!(idle["skip_ratio"].as_f64().unwrap() > 0.0);
    // Both modes report the CPU time the process used.
    for mode in [fast, idle] {
        assert!(mode["cpu_percent_core"].as_f64().unwrap() >= 0.0);
        assert!(
            mode["cpu_percent"].as_f64().unwrap() <= mode["cpu_percent_core"].as_f64().unwrap()
        );
    }

    let text = format_bench(&result);
    assert!(text.starts_with("Eye Tracker benchmark - lite backend (yunet-geom-1)"));
    assert!(text.contains("max speed"));
    assert!(text.contains("idle (20 fps)"));
    assert!(text.contains("frames per second"));
}

#[test]
fn test_run_bench_errors() {
    let dir = tempfile::tempdir().unwrap();
    let options = BenchOptions::default();
    assert!(matches!(
        run_bench(0.0, "0", "lite", &options),
        Err(BenchError::Invalid(message)) if message.contains("positive")
    ));
    let missing = dir.path().join("missing.png");
    assert!(matches!(
        run_bench(0.1, &missing.to_string_lossy(), "lite", &options),
        Err(BenchError::Camera(CameraError(message))) if message.contains("not found")
    ));
    let broken = dir.path().join("broken.png");
    std::fs::write(&broken, b"this is not an image").unwrap();
    assert!(matches!(
        run_bench(0.1, &broken.to_string_lossy(), "lite", &options),
        Err(BenchError::Failed(_))
    ));
}

#[test]
fn test_format_bench_with_missing_values() {
    let result = json!({
        "backend": "lite",
        "feature_version": "v",
        "device": "camera 0",
        "frame_size": [640, 480],
        "cpu_count": 4,
        "modes": {
            "max": {
                "target_fps": null,
                "fps": 0.0,
                "analysed_fps": 0.0,
                "inference_ms": {"p50": null, "p95": null},
                "cpu_percent": null,
                "cpu_percent_core": null,
                "face_ratio": null,
            }
        },
    });
    let text = format_bench(&result);
    assert!(text.contains("max speed"));
    assert!(!text.contains("idle"));
    assert!(text.contains("inference p50 / p95   -"));
}

#[test]
fn test_percentiles_match_numpy() {
    let values = [1.0, 2.0, 3.0, 4.0, 10.0];
    assert_eq!(percentile(&values, 50.0), 3.0);
    assert!((percentile(&values, 95.0) - 8.8).abs() < 1e-12);
    assert_eq!(stats(&[]), json!({"p50": null, "p95": null, "mean": null}));
    assert_eq!(stats(&[2.0, 4.0])["mean"].as_f64(), Some(3.0));
}

// -------------------------------------------------------------------- review fixes
#[test]
fn test_calibration_section_reports_every_stored_profile() {
    let _dirs = app_dirs();
    let mut settings = Settings::default();
    settings.general.backend = "lite".into();
    save_settings(&settings);
    let current = fake_monitors();
    let other = vec![
        Monitor {
            primary: true,
            ..Monitor::new(0, "left", Rect::new(-1920, 0, 1920, 1080))
        },
        Monitor::new(1, "right", Rect::new(0, 0, 2560, 1440)),
    ];
    let mut home = calibration(&current, "lite", "yunet-geom-1");
    home.camera = "0".into();
    save_calibration(&paths::calibration_file(), &home).unwrap();
    save_calibration(
        &paths::calibration_file(),
        &calibration(&other, "lite", "yunet-geom-1"),
    )
    .unwrap();
    let section = collect_report_with(&host(), false, None)["calibration"].clone();
    assert_eq!(section["profiles"], 2);
    // The most recent profile is for another desk, but this desk has one: usable.
    assert_eq!(section["compatible"], true);
    assert!(
        section["matching_profile"]
            .as_str()
            .unwrap()
            .ends_with("camera 0, good")
    );
    assert_eq!(
        sections(&section["profile_list"]),
        ["1", "2"].iter().map(|s| s.to_string()).collect()
    );
}

#[test]
fn test_settings_in_the_report_do_not_reveal_the_home_directory() {
    // Regression (ui_app-09): a video file camera printed the full path.
    let _dirs = app_dirs();
    let _home = fake_home(&["/home/alice"]);
    let mut settings = Settings::default();
    settings.camera.device = "/home/alice/Videos/me-at-desk.mp4".into();
    settings.privacy.pause_for_apps = strings(&["/home/alice/apps/meeting.exe", "zoom.exe"]);
    save_settings(&settings);
    let report = collect_report_with(&host(), false, None);
    let non_default = &report["settings"]["non_default"];
    assert_eq!(non_default["camera.device"], "file me-at-desk.mp4");
    let apps = non_default["privacy.pause_for_apps"].as_array().unwrap();
    assert!(apps[0].as_str().unwrap().starts_with('~'));
    assert!(apps[0].as_str().unwrap().ends_with("meeting.exe"));
    assert_eq!(apps[1], "zoom.exe");
    let text = format_report(&report);
    assert!(!text.contains("/home/alice"));
    assert!(!text.contains("alice"));
}

fn hotkeys_with_conflict() -> Box<dyn HotkeyManager> {
    Box::new(FakeHotkeys {
        supported: true,
        conflict: Some((
            "ctrl+alt+t",
            "Ctrl+Alt+T is AltGr+T, which types a character on the Turkish Q layout",
        )),
        ..FakeHotkeys::default()
    })
}

#[test]
fn test_hotkeys_that_type_a_character_are_problems() {
    let _dirs = app_dirs();
    let mut host = host();
    host.hotkey_manager = Box::new(hotkeys_with_conflict);
    let mut settings = Settings::default();
    settings.hotkeys.toggle_tracking = "ctrl+alt+t".into(); // an Eye Tracker 0.1 default
    save_settings(&settings);
    let report = collect_report_with(&host, false, None);
    assert_eq!(
        sections(&report["hotkeys"]["layout_conflicts"]),
        ["toggle_tracking"].iter().map(|s| s.to_string()).collect()
    );
    assert!(
        problems_of(&report).contains(
            &"Hotkey toggle_tracking cannot be used: Ctrl+Alt+T is AltGr+T, which types a \
          character on the Turkish Q layout."
                .to_string()
        )
    );
    settings.hotkeys.enabled = false; // not registered at all: not a problem
    save_settings(&settings);
    let report = collect_report_with(&host, false, None);
    assert!(
        !problems_of(&report)
            .iter()
            .any(|p| p.contains("cannot be used"))
    );
}

#[test]
fn test_autostart_section_reports_the_status_and_registered_command() {
    let _dirs = app_dirs();
    let _home = fake_home(&["/home/alice"]);
    let mut host = host();
    host.autostart = Box::new(|| AutostartInfo {
        supported: true,
        status: Status::Stale,
        enabled: false,
        location: Some("/home/alice/.config/autostart/eye-tracker.desktop".into()),
        registered: Some(strings(&["/home/alice/old/EyeTracker.exe"])),
        command: strings(&["eye-tracker", "--background"]),
    });
    let report = collect_report_with(&host, false, None);
    let section = &report["autostart"];
    assert_eq!(section["status"], "stale");
    assert_eq!(
        section["location"],
        "~/.config/autostart/eye-tracker.desktop"
    );
    // Unquoted on every OS: a POSIX shell would not expand a quoted '~/old/...'.
    assert_eq!(section["registered"], "~/old/EyeTracker.exe");
    assert!(
        problems_of(&report)
            .iter()
            .any(|p| p.starts_with("Start at login points to a copy"))
    );
}

#[test]
fn test_redacted_commands_keep_the_home_tilde_expandable() {
    // Commands the report suggests ("run '...'") must still run when pasted into a shell.
    let _home = fake_home(&["/home/alice"]);
    let parts = strings(&[
        "/home/alice/.local/bin/eye-tracker",
        "--config-dir",
        "/home/alice/my profiles/test",
        "/home/alice",
        "/home/alice/",
        "/opt/eye tracker/x",
        "~/a literal tilde", // not from the redaction: stays quoted, like any other '~'
        "/home/alice2/x",
        "--log=/home/alice/x", // a shell expands '~' only at the start of a word
    ]);
    assert_eq!(
        redacted_command(&parts, true),
        "~/.local/bin/eye-tracker --config-dir ~/'my profiles/test' ~ ~/ \
         '/opt/eye tracker/x' '~/a literal tilde' /home/alice2/x '--log=~/x'"
    );
    let redacted: Vec<String> = parts.iter().map(|p| redact(p)).collect();
    assert_eq!(redacted_command(&parts, false), format_command(&redacted));
}

#[test]
fn test_a_stale_accessibility_grant_is_explained() {
    let mut host = host();
    host.services = Arc::new(FakeServices {
        permissions: vec![("camera", Some(true)), ("accessibility", Some(false))],
        accessibility: "stale",
        ..FakeServices::default()
    });
    let section = platform_section(&host).unwrap();
    assert_eq!(section["accessibility"], "stale");
    let problems = problems(&json!({"platform": section}));
    assert_eq!(problems.len(), 1);
    assert!(problems[0].contains("granted to an earlier version"));
    assert!(problems[0].contains("remove Eye Tracker with '-' and add it again"));
    assert!(problems[0].is_ascii());
}

#[test]
fn test_the_macos_cli_explains_why_accessibility_is_unknown() {
    let mac_like = || {
        Arc::new(FakeServices {
            permissions: vec![("camera", Some(true)), ("accessibility", Some(false))],
            accessibility: "stale",
            ..FakeServices::default()
        })
    };
    let mut host = host();
    host.os = "macos";
    host.executable = PathBuf::from("/tmp/Eye Tracker.app/Contents/MacOS/eye-tracker-cli");
    host.services = Arc::new(FakeServices::default()); // the terminal's grant: unknown
    let section = platform_section(&host).unwrap();
    assert!(section.get("accessibility").is_none()); // unknown: no verdict either way
    let note = section["notes"]["accessibility"].as_str().unwrap();
    assert!(note.contains("the terminal"));
    assert!(note.contains("Settings > Diagnostics"));
    assert!(note.is_ascii());
    assert!(problems(&json!({"platform": section})).is_empty());

    // The app itself (or a known permission) needs no such note.
    host.executable = PathBuf::from("/tmp/Eye Tracker.app/Contents/MacOS/Eye Tracker");
    assert!(platform_section(&host).unwrap().get("notes").is_none());
    host.executable = PathBuf::from("/tmp/Eye Tracker.app/Contents/MacOS/eye-tracker-cli");
    host.services = mac_like();
    assert!(platform_section(&host).unwrap().get("notes").is_none());
}

#[test]
fn test_linux_notes_explain_camera_release_and_wayland_cursor() {
    let mut host = host();
    host.services = Arc::new(FakeServices {
        wayland: true,
        ..FakeServices::default()
    });
    let section = platform_section(&host).unwrap();
    let notes = &section["notes"];
    assert!(
        notes["camera_release"]
            .as_str()
            .unwrap()
            .contains("PipeWire camera portal")
    );
    assert!(notes["cursor"].as_str().unwrap().contains("ydotool 1.x"));
    assert!(notes["cursor"].as_str().unwrap().contains("hyprctl"));
    host.os = "windows";
    assert!(platform_section(&host).unwrap().get("notes").is_none());
}

// ------------------------------------------------------------- second review fixes
#[test]
fn test_redact_replaces_the_home_directory_anywhere() {
    // Regression (r2-ui-app-03): only a leading home directory was replaced.
    let _home = fake_home(&["/home/alice"]);
    // Explorer's "Copy as path"
    assert_eq!(
        redact("\"/home/alice/AppData/Zoom.exe\""),
        "\"~/AppData/Zoom.exe\""
    );
    assert_eq!(redact("see /home/alice"), "see ~");
    // Either slash matches either separator ("C:/Users/..." from a UI toolkit).
    assert_eq!(redact("\\home\\alice\\x.log"), "~\\x.log");
    assert_eq!(redact("//home//alice/x"), "~/x");
    // A name that merely starts with the account name is someone else's.
    assert_eq!(redact("/home/alice2"), "/home/alice2");
    assert_eq!(redact("/home/alice.old"), "/home/alice.old");
    assert_eq!(redact("/home/alice-2"), "/home/alice-2");
    if cfg!(windows) || cfg!(target_os = "macos") {
        // case-insensitive file systems
        assert_eq!(redact("/HOME/ALICE"), "~");
    }
    let nested = redact_tree(&json!({"a": ["/home/alice/x", {"b": ["/home/alice"]}], "n": 3}));
    assert_eq!(nested, json!({"a": ["~/x", {"b": ["~"]}], "n": 3}));
}

#[test]
fn test_a_home_that_would_redact_everything_is_ignored() {
    let _home = fake_home(&["/", "C:\\", "/home/bob"]);
    assert_eq!(redact("/usr/lib and C:\\x"), "/usr/lib and C:\\x");
    assert_eq!(redact("/home/bob/x"), "~/x");
}

#[test]
fn test_error_texts_do_not_reveal_the_home_directory() {
    let _home = fake_home(&["/home/alice"]);
    let section = safe(|| {
        Err(SectionError::new(
            "PermissionDenied",
            "Permission denied (os error 13) at '/home/alice/cfg/settings.json'",
        ))
    });
    let error = section["error"].as_str().unwrap();
    assert!(error.starts_with("PermissionDenied: Permission denied"));
    assert!(!error.contains("alice"));
    assert!(error.contains('~'));
}

#[test]
fn test_a_panicking_section_is_reported_not_raised() {
    let section = safe(|| -> Section { panic!("boom") });
    assert_eq!(section, json!({"error": "Panic: boom"}));
}

#[test]
fn test_quoted_app_paths_and_camera_serials_stay_out_of_the_report() {
    let _dirs = app_dirs();
    let _home = fake_home(&["/home/alice"]);
    let mut settings = Settings::default();
    let zoom = "/home/alice/AppData/Roaming/Zoom/bin/Zoom.exe";
    settings.privacy.pause_for_apps = vec![format!("\"{zoom}\""), "obs64.exe".into()];
    settings.camera.device =
        "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_A1B2C3D4-video-index0".into();
    save_settings(&settings);
    let report = collect_report_with(&host(), false, None);
    let non_default = &report["settings"]["non_default"];
    let apps = non_default["privacy.pause_for_apps"].as_array().unwrap();
    assert!(apps[0].as_str().unwrap().starts_with("\"~"));
    assert!(apps[0].as_str().unwrap().ends_with("Zoom.exe\""));
    assert_eq!(apps[1], "obs64.exe");
    assert_eq!(
        non_default["camera.device"],
        "device usb-046d_HD_Pro_Webcam_C920_*-video-index0"
    );
    let text = format_report(&report);
    assert!(!text.contains("alice"));
    assert!(!text.contains("A1B2C3D4"));
}

#[test]
fn test_describe_camera_devices() {
    assert_eq!(describe_device("/dev/video2"), "device video2");
    let by_id = "/dev/v4l/by-id/usb-Generic_USB2.0_HD_UVC_WebCam_0x0001-video-index1";
    assert_eq!(
        describe_device(by_id),
        "device usb-Generic_USB2.0_HD_UVC_WebCam_*-video-index1"
    );
    // A link without a serial part is shown as it is.
    assert_eq!(describe_device("/dev/v4l/by-id/webcam"), "device webcam");
}

/// Three valid hotkeys, checked by a manager that sees no layout conflict.
fn host_with_hotkeys() -> (Host, AppDirs) {
    let dirs = app_dirs();
    let mut host = host();
    host.hotkey_manager = Box::new(|| {
        Box::new(FakeHotkeys {
            supported: true,
            ..FakeHotkeys::default()
        })
    });
    let mut settings = Settings::default();
    settings.hotkeys.toggle_tracking = "ctrl+alt+meta+t".into();
    settings.hotkeys.toggle_privacy = "ctrl+alt+meta+p".into();
    settings.hotkeys.recalibrate = "ctrl+alt+meta+c".into();
    save_settings(&settings);
    (host, dirs)
}

#[test]
fn test_doctor_shows_why_a_hotkey_was_not_registered() {
    // Regression (r2-docs-12): doctor never knew what the OS refused.
    let (host, _dirs) = host_with_hotkeys();
    let live = FakeHotkeys {
        supported: true,
        registered: HashMap::from([(
            "toggle_tracking".to_string(),
            parse_hotkey("ctrl+alt+meta+t").unwrap(),
        )]),
        errors: HashMap::from([(
            "toggle_privacy".to_string(),
            "Ctrl+Alt+Win+P is already used by another application".to_string(),
        )]),
        ..FakeHotkeys::default()
    };
    let report = collect_report_with(&host, false, Some(&live));
    assert_eq!(
        report["hotkeys"]["registration"],
        json!({
            "toggle_tracking": "registered",
            "toggle_privacy":
                "not registered: Ctrl+Alt+Win+P is already used by another application",
            "recalibrate": "not registered: reason unknown",
        })
    );
    assert!(
        problems_of(&report).contains(
            &"Hotkey toggle_privacy could not be registered: Ctrl+Alt+Win+P is already used by \
          another application."
                .to_string()
        )
    );
    assert!(format_report(&report).contains("registration.toggle_tracking"));
    // The app's manager knows limits a fresh one cannot see (r3-platform-hotkeys-02).
    *live.note.lock() = Some("Press Ctrl and Alt before Super.".into());
    let report = collect_report_with(&host, false, Some(&live));
    assert_eq!(
        report["hotkeys"]["note"],
        "Press Ctrl and Alt before Super."
    );
}

#[test]
fn test_doctor_asks_the_running_instance_about_its_hotkeys() {
    let (mut host, _dirs) = host_with_hotkeys();
    // Nothing running: nobody can know what the OS would accept.
    let report = collect_report_with(&host, false, None);
    assert_eq!(report["hotkeys"]["registration"], HOTKEYS_NOT_RUNNING);
    assert!(
        !problems_of(&report)
            .iter()
            .any(|p| p.contains("could not be registered"))
    );

    let status: Rc<RefCell<Option<Value>>> =
        Rc::new(RefCell::new(Some(json!({"state": "tracking"}))));
    host.instance_running = {
        let status = Rc::clone(&status);
        Box::new(move || status.borrow().is_some())
    };
    host.instance_status = {
        let status = Rc::clone(&status);
        Box::new(move || status.borrow().clone())
    };
    // An instance that does not report its hotkeys (an older version).
    let report = collect_report_with(&host, false, None);
    assert_eq!(report["app"]["instance_running"], true);
    assert_eq!(report["hotkeys"]["registration"], HOTKEYS_NOT_REPORTED);

    *status.borrow_mut() = Some(json!({"state": "tracking", "hotkeys": {
        "registered": ["toggle_tracking", "recalibrate"],
        "errors": {"toggle_privacy": "Ctrl+Alt+Win+P is already used by another application"},
    }}));
    let report = collect_report_with(&host, false, None);
    assert_eq!(
        report["hotkeys"]["registration"],
        json!({
            "toggle_tracking": "registered",
            "toggle_privacy":
                "not registered: Ctrl+Alt+Win+P is already used by another application",
            "recalibrate": "registered",
        })
    );

    // What only the running manager found out (r3-platform-hotkeys-02); a {name: hotkey}
    // mapping lists the registered names too.
    *status.borrow_mut() = Some(json!({"hotkeys": {
        "registered": {"toggle_tracking": "ctrl+alt+meta+t"},
        "note": "Another program uses the Super key on its own.",
    }}));
    let report = collect_report_with(&host, false, None);
    assert_eq!(
        report["hotkeys"]["note"],
        "Another program uses the Super key on its own."
    );
    assert_eq!(
        report["hotkeys"]["registration"]["toggle_tracking"],
        "registered"
    );
    assert!(
        problems_of(&report).contains(
            &"Global hotkeys: Another program uses the Super key on its own.".to_string()
        )
    );
}

#[test]
fn test_instance_status_over_ipc() {
    let _dirs = app_dirs();
    assert_eq!(system_instance_status(), None);
    let (events, received) = crossbeam_channel::unbounded();
    let mut server = ipc::InstanceServer::new(events);
    assert!(server.listen());
    std::thread::spawn(move || {
        for event in received {
            let ipc::Event::CommandReceived { reply, .. } = event;
            reply.send(r#"{"state": "tracking", "hotkeys": {"registered": ["recalibrate"]}}"#);
        }
    });
    let status = system_instance_status().unwrap();
    assert_eq!(status["state"], "tracking");
    assert!(ipc::is_running(ipc::PROBE_TIMEOUT_MS));
    server.close();
}

#[test]
fn test_linux_report_says_which_lock_tools_exist_and_that_locking_is_untested() {
    // Regression (r2-docs-13): "lock yes" read as "locking works".
    let mut host = host();
    host.lock_methods =
        Box::new(|| Some(strings(&["loginctl lock-session", "xdg-screensaver lock"])));
    let section = platform_section(&host).unwrap();
    assert_eq!(
        section["lock_methods"],
        json!(["loginctl lock-session", "xdg-screensaver lock"])
    );
    let note = section["notes"]["lock"].as_str().unwrap();
    assert!(note.contains("known when it is tried"));
    assert!(note.contains("Could not lock the screen"));
    assert!(note.is_ascii());
    // Platforms without the query report the capability alone.
    host.lock_methods = Box::new(|| None);
    assert!(
        platform_section(&host)
            .unwrap()
            .get("lock_methods")
            .is_none()
    );
}

// ------------------------------------------------------------------------ new
#[test]
fn test_format_g() {
    assert_eq!(format_g(0.0), "0");
    assert_eq!(format_g(1.0), "1");
    assert_eq!(format_g(1.5), "1.5");
    assert_eq!(format_g(2.0 / 3.0), "0.666667");
    assert_eq!(format_g(123456.0), "123456");
    assert_eq!(format_g(1234567.0), "1.23457e+06");
    assert_eq!(format_g(0.0001), "0.0001");
    assert_eq!(format_g(0.00001), "1e-05");
    assert_eq!(format_g(-2.5), "-2.5");
    assert_eq!(format_g(100.0), "100");
}

#[test]
fn test_the_ui_and_libraries_sections() {
    let libraries = libraries_section();
    for name in ["tract", "nokhwa", "egui", "eframe", "image"] {
        assert!(libraries[name].is_string(), "{name}");
    }
    let ui = ui_section();
    assert_eq!(ui["egui"], libraries["egui"]);
    assert_eq!(ui["eframe"], libraries["eframe"]);
    assert!(ui["display_backend"].is_string());
    assert_eq!(locked_version("no-such-crate"), None);
}

#[test]
fn test_a_missing_display_is_an_error_section_not_a_failure() {
    let _dirs = app_dirs();
    let mut host = host();
    host.monitors = Box::new(|| Err("no display available".into()));
    let report = collect_report_with(&host, false, None);
    assert_eq!(report["monitors"], json!({"error": "no display available"}));
    assert!(
        problems_of(&report)
            .contains(&"Could not collect monitors information: no display available".to_string())
    );
}
