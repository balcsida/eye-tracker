// The Python suite has no test file for types.py (its helpers are only exercised through the
// controller, gaze and platform tests, which are ported with those modules). The tests below
// are new and pin the behaviour those tests rely on.

use super::*;

fn two_monitors() -> Vec<Monitor> {
    vec![
        Monitor::new(0, "A", Rect::new(0, 0, 1920, 1080)),
        Monitor::new(1, "B", Rect::new(1920, 0, 1280, 1024)),
    ]
}

#[test]
fn test_rect_geometry() {
    let r = Rect::new(10, 20, 100, 50);
    assert_eq!((r.right(), r.bottom()), (110, 70));
    assert_eq!(r.center(), (60.0, 45.0));
    assert!(r.contains(10.0, 20.0));
    assert!(!r.contains(110.0, 30.0)); // right/bottom are exclusive
    assert_eq!(r.distance_outside(60.0, 30.0), 0.0);
    assert_eq!(r.distance_outside(113.0, 74.0), 5.0);
    assert_eq!(r.clamp(500.0, -5.0), (109, 20));
    assert_eq!(r.clamp(10.5, 20.5), (10, 20)); // halves round to even, as in Python
    assert_eq!(r.normalize(60.0, 45.0), (0.5, 0.5));
    assert_eq!(r.denormalize(0.5, 0.5), (60.0, 45.0));
    assert_eq!(Rect::from_list(r.to_list()), r);
}

#[test]
fn test_virtual_bounds_and_nearest_monitor() {
    let monitors = two_monitors();
    assert_eq!(virtual_bounds(&monitors), Ok(Rect::new(0, 0, 3200, 1080)));
    assert_eq!(virtual_bounds(&[]), Err(NoMonitors));
    assert_eq!(
        monitor_at(&monitors, 2000.0, 10.0).map(|m| m.index),
        Some(1)
    );
    assert!(monitor_at(&monitors, -1.0, 10.0).is_none());
    let (m, d) = nearest_monitor(&monitors, 1930.0, 1000.0).unwrap();
    assert_eq!((m.index, d), (1, 0.0));
    let (m, d) = nearest_monitor(&monitors, -30.0, 40.0).unwrap();
    assert_eq!((m.index, d), (0, 30.0));
    assert!(nearest_monitor(&[], 0.0, 0.0).is_err());
}

#[test]
fn test_layout_signature_is_order_independent_and_matches_python() {
    let mut monitors = two_monitors();
    let sig = layout_signature(&monitors);
    monitors.reverse();
    assert_eq!(layout_signature(&monitors), sig);
    assert_eq!(sig.len(), 16);
    // hashlib.sha1(b"[(0, 0, 1920, 1080), (1920, 0, 1280, 1024)]").hexdigest()[:16]
    assert_eq!(sig, "2ca4f1dd8234969b");
}

#[test]
fn test_observation_flags() {
    let mut obs = Observation::new(0.0, 1);
    assert!(obs.face_present());
    assert!(!obs.usable());
    obs.features = Some(vec![0.0]);
    obs.quality = 0.3;
    assert!(obs.usable());
    obs.blink = true;
    assert!(!obs.usable());
    obs.blind = true;
    assert!(!obs.face_present());
}

#[test]
fn test_tracking_state_labels_and_camera() {
    assert_eq!(TrackingState::Starting.label(), "Starting…");
    assert_eq!(TrackingState::NeedsCalibration.value(), "needs_calibration");
    assert!(TrackingState::Away.camera_active());
    assert!(!TrackingState::Paused.camera_active());
    assert!(!TrackingState::Yielded.camera_active());
}
