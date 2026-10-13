// Tests for releasing the camera to other apps (engine/camera_yield.rs).

use super::*;

fn inputs(camera: Option<bool>, running: &[&str], apps: &[&str]) -> YieldInputs {
    YieldInputs {
        camera_in_use_by_other: camera,
        running: running.iter().map(|s| s.to_string()).collect(),
        pause_for_apps: apps.iter().map(|s| s.to_string()).collect(),
    }
}

#[test]
fn test_nothing_to_yield_for() {
    let i = inputs(Some(false), &["explorer.exe"], &[]);
    assert_eq!(should_yield(&i, true), (false, String::new()));
}

#[test]
fn test_camera_in_use_yields_when_enabled() {
    let i = inputs(Some(true), &[], &[]);
    let (ok, reason) = should_yield(&i, true);
    assert!(ok);
    assert!(reason.to_lowercase().contains("camera"));
}

#[test]
fn test_camera_in_use_ignored_when_disabled() {
    let i = inputs(Some(true), &[], &[]);
    assert_eq!(should_yield(&i, false), (false, String::new()));
}

#[test]
fn test_unknown_camera_usage_does_not_yield() {
    let i = inputs(None, &[], &[]);
    assert_eq!(should_yield(&i, true), (false, String::new()));
}

#[test]
fn test_listed_app_yields_even_when_camera_yield_is_off() {
    let i = inputs(None, &["zoom.exe", "code.exe"], &["zoom.exe"]);
    assert_eq!(
        should_yield(&i, false),
        (true, "zoom.exe is running".to_string())
    );
}

#[test]
fn test_listed_app_reported_before_camera_usage() {
    let i = inputs(Some(true), &["obs64.exe"], &["obs64.exe"]);
    assert_eq!(
        should_yield(&i, true),
        (true, "obs64.exe is running".to_string())
    );
}

#[test]
fn test_app_matching_is_forgiving() {
    let cases = [
        ("zoom.exe", "zoom.exe"),
        ("Zoom.EXE", "zoom.exe"),
        ("zoom", "zoom.exe"),
        ("zoom.exe", "zoom"),
        ("Zoom.app", "zoom"),
        ("  obs64.exe ", "obs64.exe"),
        (
            r"C:\Program Files\obs-studio\bin\64bit\obs64.exe",
            "obs64.exe",
        ),
        ("/Applications/OBS.app", "obs"),
    ];
    for (configured, running) in cases {
        let apps = [configured.to_string()];
        assert_eq!(matching_app([running], &apps), Some(configured.trim()));
    }
}

#[test]
fn test_app_matching_is_exact_on_the_name() {
    let cases = [
        ("zoom", "zoomit.exe"),
        ("", "zoom.exe"),
        ("   ", "zoom.exe"),
        ("code", "vscode.exe"),
    ];
    for (configured, running) in cases {
        let apps = [configured.to_string()];
        assert_eq!(matching_app([running], &apps), None);
    }
}

#[test]
fn test_first_listed_app_wins() {
    let apps = ["b.exe".to_string(), "a.exe".to_string()];
    assert_eq!(matching_app(["a.exe", "b.exe"], &apps), Some("b.exe"));
}

#[test]
fn test_normalize_process_name() {
    let cases = [
        ("Teams.exe", "teams"),
        ("FaceTime.app", "facetime"),
        (".exe", ".exe"),
        ("python3.12", "python3.12"),
        ("dir/sub\\Tool.EXE", "tool"),
        (
            "\"C:\\Program Files\\obs-studio\\bin\\64bit\\obs64.exe\"",
            "obs64",
        ),
        (
            "'C:\\Program Files\\obs-studio\\bin\\64bit\\obs64.exe'",
            "obs64",
        ),
        ("  \"obs64.exe\"  ", "obs64"),
        ("'/Applications/zoom.us.app'", "zoom.us"),
        ("\"", ""),
    ];
    for (name, expected) in cases {
        assert_eq!(normalize_process_name(name), expected, "{name:?}");
    }
}

#[test]
fn test_quoted_path_from_copy_as_path_matches_and_is_reported_unquoted() {
    // Explorer's "Copy as path" (Ctrl+Shift+C) adds the double quotes.
    let entry = "\"C:\\Program Files\\obs-studio\\bin\\64bit\\obs64.exe\"";
    let i = inputs(None, &["obs64.exe"], &[entry]);
    let (ok, reason) = should_yield(&i, true);
    assert!(ok);
    assert_eq!(
        reason,
        "C:\\Program Files\\obs-studio\\bin\\64bit\\obs64.exe is running"
    );
}
