// Tests for the adaptive frame-rate policy (engine/scheduler.rs).

use super::*;
use crate::config::describe_settings;
use crate::testing;

const T: TrackingState = TrackingState::Tracking;

/// (active, idle, typing, noface, away, calibrating)
fn expected(profile: &str) -> [f64; 6] {
    match profile {
        "eco" => [8.0, 2.0, 1.0, 1.0, 0.5, 15.0],
        "balanced" => [12.0, 4.0, 2.0, 2.0, 1.0, 24.0],
        "responsive" => [20.0, 8.0, 4.0, 3.0, 1.0, 30.0],
        _ => unreachable!(),
    }
}

/// One context per mode.
fn contexts() -> [(&'static str, RateContext); 6] {
    let t = RateContext::new(T);
    [
        (
            "active",
            RateContext {
                pending_switch: true,
                ..t
            },
        ),
        ("idle", t),
        ("typing", RateContext { typing: true, ..t }),
        (
            "noface",
            RateContext {
                face_present: false,
                ..t
            },
        ),
        (
            "away",
            RateContext {
                face_present: false,
                ..RateContext::new(TrackingState::Away)
            },
        ),
        ("calibrating", RateContext::new(TrackingState::Calibrating)),
    ]
}

fn ctx(state: TrackingState) -> RateContext {
    RateContext::new(state)
}

#[test]
fn test_profile_rates() {
    for profile in ["balanced", "eco", "responsive"] {
        let policy = RatePolicy::new(profile);
        for ((name, ctx), fps) in contexts().iter().zip(expected(profile)) {
            assert_eq!(policy.mode(ctx), *name);
            assert_eq!(policy.fps(ctx), fps);
            assert!((policy.interval(ctx) - 1.0 / fps).abs() < 1e-9);
        }
    }
}

#[test]
fn test_every_profile_defines_every_mode() {
    for (_, rates) in PROFILES {
        assert!(MODES.iter().all(|m| rates.get(m).is_some()));
        assert!(rates.get(CAMERA_OFF_MODE).is_none());
    }
}

#[test]
fn test_profiles_match_settings_choices() {
    let row = describe_settings()
        .into_iter()
        .find(|r| r.key == "performance.profile")
        .expect("setting exists");
    let mut choices: Vec<&str> = row.choices.to_vec();
    let mut profiles: Vec<&str> = PROFILES.iter().map(|(n, _)| *n).collect();
    choices.sort_unstable();
    profiles.sort_unstable();
    assert_eq!(choices, profiles);
}

#[test]
fn test_camera_off_states_return_one_second() {
    let policy = RatePolicy::new("responsive");
    for state in CAMERA_OFF_STATES {
        let ctx = RateContext {
            pending_switch: true,
            preview: true,
            ..ctx(state)
        };
        assert_eq!(policy.interval(&ctx), 1.0);
        assert_eq!(policy.fps(&ctx), 1.0);
        assert_eq!(policy.mode(&ctx), "off");
    }
}

#[test]
fn test_preview_runs_at_calibration_rate() {
    let policy = RatePolicy::default();
    let c = RateContext {
        preview: true,
        typing: true,
        ..ctx(T)
    };
    assert_eq!(policy.fps(&c), 24.0);
}

#[test]
fn test_calibration_beats_everything() {
    let policy = RatePolicy::default();
    let c = RateContext {
        face_present: false,
        typing: true,
        ..ctx(TrackingState::Calibrating)
    };
    assert_eq!(policy.mode(&c), "calibrating");
}

#[test]
fn test_away_beats_presence_warning() {
    let policy = RatePolicy::default();
    let c = RateContext {
        presence_warning: true,
        face_present: false,
        ..ctx(TrackingState::Away)
    };
    assert_eq!(policy.mode(&c), "away");
}

#[test]
fn test_presence_warning_samples_fast_without_a_face() {
    let policy = RatePolicy::default();
    let c = RateContext {
        presence_warning: true,
        face_present: false,
        ..ctx(T)
    };
    assert_eq!(policy.mode(&c), "active");
}

#[test]
fn test_no_face_beats_pending_switch() {
    let policy = RatePolicy::default();
    let c = RateContext {
        face_present: false,
        pending_switch: true,
        ..ctx(T)
    };
    assert_eq!(policy.mode(&c), "noface");
}

#[test]
fn test_pending_switch_or_gaze_motion_beats_typing() {
    let policy = RatePolicy::default();
    let pending = RateContext {
        pending_switch: true,
        typing: true,
        ..ctx(T)
    };
    let moving = RateContext {
        gaze_moving: true,
        typing: true,
        ..ctx(T)
    };
    assert_eq!(policy.mode(&pending), "active");
    assert_eq!(policy.mode(&moving), "active");
}

#[test]
fn test_other_camera_states_use_normal_modes() {
    let policy = RatePolicy::default();
    for state in [
        TrackingState::Starting,
        TrackingState::NeedsCalibration,
        TrackingState::CameraError,
    ] {
        assert_eq!(policy.mode(&ctx(state)), "idle");
        let no_face = RateContext {
            face_present: false,
            ..ctx(state)
        };
        assert_eq!(policy.mode(&no_face), "noface");
    }
}

#[test]
fn test_unknown_profile_falls_back_to_balanced() {
    let logs = testing::capture_logs();
    let policy = RatePolicy::new("turbo");
    assert_eq!(policy.profile(), "balanced");
    assert!(
        logs.messages_at_least(log::Level::Warn)
            .iter()
            .any(|m| m.contains("turbo"))
    );
}

#[test]
fn test_set_profile_changes_rates() {
    let mut policy = RatePolicy::new("eco");
    let idle = ctx(T);
    assert_eq!(policy.fps(&idle), 2.0);
    policy.set_profile("responsive");
    assert_eq!(policy.profile(), "responsive");
    assert_eq!(policy.fps(&idle), 8.0);
}
