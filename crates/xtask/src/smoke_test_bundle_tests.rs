//! Tests of `smoke_test_bundle.rs`. They are new (the Python script had its own tests, which
//! drove a fake CLI that answered `doctor` and `bench`); these drive a fake CLI that answers
//! `--version` and `ctl`, so they need a Unix shell.
#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::*;

const SHORT: Timeouts = Timeouts {
    start: Duration::from_secs(5),
    quit: Duration::from_secs(5),
};

/// A script that behaves like the CLI: `--version`, `ctl status` (3 until the app is up),
/// `ctl quit`, and otherwise the app itself, which runs until it is told to quit.
fn fake_cli(dir: &Path, version: &str, app_body: &str) -> PathBuf {
    let path = dir.join("fake-cli");
    let script = format!(
        r#"#!/bin/sh
case " $* " in
  *" --version "*) echo "eye-tracker {version}"; exit 0 ;;
  *" ctl status "*) [ -f "$2/up" ] && exit 0; exit 3 ;;
  *" ctl quit "*) touch "$2/quit"; exit 0 ;;
esac
mkdir -p "$2"
{app_body}
"#
    );
    fs::write(&path, script).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    path
}

const RUNS_UNTIL_QUIT: &str = r#"touch "$2/up"
while [ ! -f "$2/quit" ]; do sleep 0.05; done
exit 0"#;

fn run_smoke(cli: &Path, version: &str, work: &Path) -> Result<(), String> {
    smoke_test(cli, version, work, Path::new("image.png"), SHORT)
}

#[test]
fn test_smoke_test_starts_the_app_checks_status_and_quits_it() {
    let dir = tempfile::tempdir().unwrap();
    let cli = fake_cli(dir.path(), "9.9.9", RUNS_UNTIL_QUIT);
    run_smoke(&cli, "9.9.9", dir.path()).unwrap();
    assert!(dir.path().join("config/quit").is_file());
}

#[test]
fn test_smoke_test_fails_on_another_version() {
    let dir = tempfile::tempdir().unwrap();
    let cli = fake_cli(dir.path(), "9.9.9", RUNS_UNTIL_QUIT);
    let error = run_smoke(&cli, "1.0.0", dir.path()).unwrap_err();
    assert!(error.contains("--version does not report 1.0.0"), "{error}");
}

#[test]
fn test_smoke_test_fails_when_the_app_exits_before_it_answers() {
    let dir = tempfile::tempdir().unwrap();
    let cli = fake_cli(dir.path(), "9.9.9", "echo broken >&2; exit 1");
    let error = run_smoke(&cli, "9.9.9", dir.path()).unwrap_err();
    assert!(error.contains("exited with status"), "{error}");
}

#[test]
fn test_smoke_test_fails_when_the_app_never_answers() {
    let dir = tempfile::tempdir().unwrap();
    let cli = fake_cli(dir.path(), "9.9.9", "sleep 30");
    let timeouts = Timeouts {
        start: Duration::from_millis(500),
        quit: Duration::from_secs(1),
    };
    let error =
        smoke_test(&cli, "9.9.9", dir.path(), Path::new("image.png"), timeouts).unwrap_err();
    assert!(error.contains("did not answer"), "{error}");
}

#[test]
fn test_smoke_test_fails_when_the_app_ignores_quit() {
    let dir = tempfile::tempdir().unwrap();
    let cli = fake_cli(dir.path(), "9.9.9", "touch \"$2/up\"\nsleep 30");
    let timeouts = Timeouts {
        start: Duration::from_secs(5),
        quit: Duration::from_millis(500),
    };
    let error =
        smoke_test(&cli, "9.9.9", dir.path(), Path::new("image.png"), timeouts).unwrap_err();
    assert!(error.contains("did not quit"), "{error}");
}

#[test]
fn test_main_reports_a_missing_tool_as_a_usage_error() {
    let args = Args {
        version: "9.9.9".into(),
        cli: PathBuf::from("/nonexistent/eye-tracker-cli"),
    };
    assert_eq!(main(&args), 2);
}
