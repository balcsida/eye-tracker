//! Tests of `preview.rs` (the preview tests of tests/test_ui_basic.py).
//!
//! Not ported:
//! * `test_preview_window_lifecycle`'s `window.show()` / `window.grab()` on a live widget: the
//!   lifecycle is driven through `open` / `hide` / `close` and the `SetPreview` actions (the
//!   `controller.calls` of the Python fake), and drawing by a headless smoke test;
//! * the non-frame inputs (`"not a frame"`, a float64 array): a `Frame` is typed, so only a
//!   malformed one can be ignored and logged once.

use super::*;
use crate::types::Rect;

const INFO: Option<(&str, &str)> = Some(("facemesh", "facemesh-pose-iris-1"));

fn monitors() -> Vec<Monitor> {
    vec![
        Monitor {
            primary: true,
            ..Monitor::new(0, "Left", Rect::new(0, 0, 1920, 1080))
        },
        Monitor::new(1, "Right", Rect::new(1920, 0, 1920, 1080)),
    ]
}

fn gaze(x: f64, y: f64) -> Option<GazePoint> {
    Some(GazePoint {
        x,
        y,
        timestamp: 1.0,
    })
}

#[test]
fn test_preview_window_lifecycle() {
    let monitors = monitors();
    let mut calls = Vec::new();
    let mut window = PreviewWindow::new(TrackingState::Tracking, INFO);
    assert!(!window.preview_active());
    window.open(&monitors, INFO);
    calls.extend(window.drain_actions());
    assert_eq!(calls, [PreviewAction::SetPreview(true)]);
    assert!(window.preview_active());
    assert_eq!(window.value("backend"), "facemesh (facemesh-pose-iris-1)");
    assert!(NOTE.contains("never saved"));

    window.set_frame(&Frame::filled(640, 480, [0, 0, 255])); // red
    let image = window.image().unwrap().clone();
    assert_eq!(image.size, [640, 480]);
    assert_eq!(
        image.pixels[10 * 640 + 10],
        egui::Color32::from_rgb(255, 0, 0)
    );
    // A malformed frame is ignored (and logged once).
    let broken = Frame {
        width: 2,
        height: 2,
        data: vec![0; 3],
    };
    window.set_frame(&broken);
    window.set_frame(&broken);
    assert_eq!(window.image(), Some(&image));
    assert_eq!(window.frame_errors, 2);

    let mut obs = Observation::new(1.0, 2);
    obs.features = Some(vec![0.0; 8]);
    obs.quality = 1.0;
    obs.head_yaw = Some(-12.3);
    obs.head_pitch = Some(4.0);
    window.set_observation(&obs);
    assert_eq!(window.value("faces"), "2");
    assert_eq!(window.value("pose"), "-12° / +4°");
    window.set_observation(&Observation::new(2.0, 0));
    assert_eq!(window.value("faces"), "0");
    assert_eq!(window.value("pose"), "—");
    window.set_stats(&PreviewStats {
        fps: Some(11.8),
        target_fps: Some(12.0),
        inference_ms: Some(9.44),
        skip_ratio: Some(0.35),
        cpu_percent: Some(0.84),
        backend: Some("lite".to_string()),
    });
    assert_eq!(window.value("fps"), "12 fps (target 12 fps)");
    assert_eq!(window.value("inference"), "9.4 ms");
    assert_eq!(window.value("skipped"), "35 %");
    assert_eq!(window.value("cpu"), "0.8 %");
    assert!(window.value("backend").starts_with("lite"));
    window.set_gaze(gaze(2500.0, 10.0), &monitors);
    assert_eq!(window.value("monitor"), "Monitor 2 · Right");
    window.set_gaze(gaze(-5000.0, 10.0), &monitors);
    assert_eq!(window.value("monitor"), "Off-screen");
    window.set_gaze(None, &monitors);
    assert_eq!(window.value("monitor"), "—");
    assert!(calls.len() == 1 && window.drain_actions().is_empty());

    window.close();
    calls.extend(window.drain_actions());
    assert_eq!(calls.last(), Some(&PreviewAction::SetPreview(false)));
    assert!(!window.preview_active());
    assert!(window.image().is_none());
    window.open(&monitors, INFO); // can be reopened
    calls.extend(window.drain_actions());
    assert_eq!(calls.last(), Some(&PreviewAction::SetPreview(true)));
    window.hide();
    calls.extend(window.drain_actions());
    assert_eq!(
        calls,
        [
            PreviewAction::SetPreview(true),
            PreviewAction::SetPreview(false),
            PreviewAction::SetPreview(true),
            PreviewAction::SetPreview(false),
        ]
    );
}

#[test]
fn test_preview_stops_frames_while_minimised() {
    let mut window = PreviewWindow::new(TrackingState::Tracking, None);
    window.open(&[], None);
    assert_eq!(window.drain_actions(), [PreviewAction::SetPreview(true)]);
    window.set_minimized(true); // no one is watching
    assert_eq!(window.drain_actions(), [PreviewAction::SetPreview(false)]);
    window.set_minimized(true);
    assert!(window.drain_actions().is_empty());
    window.set_minimized(false);
    assert_eq!(window.drain_actions(), [PreviewAction::SetPreview(true)]);
    // Without backend info only a name from the statistics is shown.
    assert_eq!(window.value("backend"), "—");
    window.set_stats(&PreviewStats {
        backend: Some("lite".to_string()),
        ..PreviewStats::default()
    });
    assert_eq!(window.value("backend"), "lite");
}

#[test]
fn test_preview_ignores_frames_while_hidden_and_camera_off() {
    let mut window = PreviewWindow::new(TrackingState::Tracking, INFO);
    let frame = Frame::filled(4, 4, [0, 0, 0]);
    window.set_frame(&frame);
    assert!(window.image().is_none()); // hidden: nothing converted
    window.open(&[], INFO);
    window.set_frame(&frame);
    assert!(window.image().is_some());
    window.set_state(TrackingState::Privacy);
    assert!(window.image().is_none());
    assert_eq!(window.value("state"), TrackingState::Privacy.label());
    assert_eq!(
        window.placeholder(),
        "Camera is off: privacy mode (camera off)"
    );
    window.set_state(TrackingState::Tracking);
    assert_eq!(window.placeholder(), "Waiting for the camera…");
}

#[test]
fn test_preview_ignores_bad_numbers() {
    let mut window = PreviewWindow::new(TrackingState::Tracking, None);
    window.set_stats(&PreviewStats {
        fps: Some(f64::NAN),
        inference_ms: Some(f64::INFINITY),
        cpu_percent: Some(1.25),
        ..PreviewStats::default()
    });
    assert_eq!(window.value("fps"), "—");
    assert_eq!(window.value("inference"), "—");
    assert_eq!(window.value("cpu"), "1.2 %");
    window.set_stats(&PreviewStats {
        fps: Some(0.5),
        target_fps: Some(0.0),
        ..PreviewStats::default()
    });
    assert_eq!(window.value("fps"), "0.5 fps"); // no target without a positive one
}

#[test]
fn test_preview_fits_the_frame_with_its_aspect_ratio() {
    assert_eq!(fit_aspect([640, 480], (1280.0, 480.0)), (640.0, 480.0));
    assert_eq!(fit_aspect([100, 50], (400.0, 400.0)), (400.0, 200.0));
    assert_eq!(fit_aspect([0, 50], (400.0, 400.0)), (0.0, 0.0));
}

#[test]
fn test_preview_draws_a_viewport_while_open() {
    // egui runs headless: without a backend the viewport is embedded in a window.
    let monitors = monitors();
    let mut window = PreviewWindow::new(TrackingState::Tracking, INFO);
    window.open(&monitors, INFO);
    let frame = Frame::filled(64, 48, [10, 20, 30]);
    let ctx = egui::Context::default();
    for step in 0..2 {
        let inputs = PreviewInputs {
            monitors: &monitors,
            backend_info: INFO,
            frame: (step == 0).then_some(&frame),
            ..PreviewInputs::default()
        };
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            let _ = window.show(ui.ctx(), &inputs);
        });
        output.textures_delta.clear(); // nothing renders the textures
    }
    assert!(window.image().is_some());
}

#[test]
fn test_preview_window_is_opaque() {
    // Left unset, glow makes it as transparent as the root: no title bar background on macOS.
    assert_eq!(PreviewWindow::viewport_builder().transparent, Some(false));
}
