//! `cargo xtask smoke-test-bundle`: start a built Eye Tracker the way users get it and check
//! that it runs. Port of `scripts/smoke_test_bundle.py`.
//!
//! `.github/workflows/build.yml` runs this for every package it makes: the Windows bundle (the
//! contents of the portable ZIP) and its installed copy, the command-line tool inside the
//! mounted macOS disk image, and the Linux AppImage. It checks that the build
//!
//! * reports the expected version (`--version`);
//! * starts with a temporary configuration folder (`--config-dir`, `--background`, and the
//!   app's own icon as the camera image, so no camera is opened), answers `ctl status`, and
//!   quits on `ctl quit` with exit status 0.
//!
//! The tool writes its settings and logs into a temporary folder, never into the user's. Exit
//! status: 0 when every check passed, 1 otherwise (the failed check and the tool's output are
//! printed), 2 on a usage error.
//!
//! The Python script also ran `doctor`, `bench` on every backend and video decoding and camera
//! probing. They tested the bundled Python libraries (OpenCV, FFmpeg); the Rust executable
//! compiles the models and image decoding in, so there is nothing separate to check.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread::sleep;
use std::time::{Duration, Instant};

use crate::repo_root;

/// `ctl` exit status when no instance is running (`cli::EXIT_NOT_RUNNING`).
const NOT_RUNNING: i32 = eye_tracker::cli::EXIT_NOT_RUNNING;
/// Seconds one short command may take (the AppImage unpacks itself on every start).
const COMMAND_TIMEOUT: Duration = Duration::from_secs(300);
const POLL: Duration = Duration::from_millis(250);

#[derive(clap::Args, Debug)]
pub struct Args {
    /// The version the build must report.
    #[arg(long)]
    pub version: String,
    /// The built command-line tool (or the AppImage, or the console executable).
    pub cli: PathBuf,
}

/// How long to wait for the app to come up and to quit again.
#[derive(Debug, Clone, Copy)]
pub struct Timeouts {
    pub start: Duration,
    pub quit: Duration,
}

impl Default for Timeouts {
    fn default() -> Self {
        Self {
            start: Duration::from_secs(120),
            quit: Duration::from_secs(60),
        }
    }
}

/// Entry point of `cargo xtask smoke-test-bundle`; returns the exit code.
pub fn main(args: &Args) -> i32 {
    let Ok(cli) = args.cli.canonicalize() else {
        eprintln!("error: {} does not exist", args.cli.display());
        return 2;
    };
    if !cli.is_file() {
        eprintln!("error: {} is not a file", cli.display());
        return 2;
    }
    let image = repo_root().join("packaging/linux/eye-tracker.png");
    let work = std::env::temp_dir().join(format!("eye-tracker-smoke-{}", std::process::id()));
    let result = fs::create_dir_all(&work)
        .map_err(|e| format!("cannot create {}: {e}", work.display()))
        .and_then(|()| smoke_test(&cli, &args.version, &work, &image, Timeouts::default()));
    // The tool may still hold a log file open for a moment on Windows.
    let _ = fs::remove_dir_all(&work);
    match result {
        Ok(()) => {
            println!("smoke test passed");
            0
        }
        Err(message) => {
            eprintln!("smoke test failed: {message}");
            1
        }
    }
}

/// Run every check against the command-line tool `cli`; `work` is its configuration folder's
/// parent and `image` the picture it analyses instead of a camera.
pub fn smoke_test(
    cli: &Path,
    version: &str,
    work: &Path,
    image: &Path,
    timeouts: Timeouts,
) -> Result<(), String> {
    let output = run(cli, &["--version"])?.trim().to_string();
    check(
        output.split_whitespace().any(|word| word == version),
        &format!("--version does not report {version}: {output:?}"),
    )?;
    println!("ok: {output}");

    let config = work.join("config");
    let config_arg = config.to_string_lossy().into_owned();
    let image_arg = image.to_string_lossy().into_owned();
    let log = work.join("app.log");
    println!(
        "$ {} --config-dir {config_arg} --camera {image_arg} --background",
        cli.display()
    );
    let mut app = start_app(cli, &config_arg, &image_arg, &log)?;
    let result = check_running_app(&mut app, cli, &config_arg, timeouts);
    if result.is_err() {
        let _ = app.kill();
        let _ = app.wait();
        if let Ok(text) = fs::read_to_string(&log) {
            eprintln!("--- output of the app ---\n{text}");
        }
    }
    result
}

fn start_app(cli: &Path, config: &str, image: &str, log: &Path) -> Result<Child, String> {
    let file =
        fs::File::create(log).map_err(|e| format!("cannot create {}: {e}", log.display()))?;
    let duplicate = file
        .try_clone()
        .map_err(|e| format!("cannot write {}: {e}", log.display()))?;
    Command::new(cli)
        .args(["--config-dir", config, "--camera", image, "--background"])
        .stdin(Stdio::null())
        .stdout(file)
        .stderr(duplicate)
        .spawn()
        .map_err(|e| format!("could not run {}: {e}", cli.display()))
}

/// Wait for `ctl status` to answer, send `ctl quit` and wait for the app to exit with status 0.
fn check_running_app(
    app: &mut Child,
    cli: &Path,
    config: &str,
    timeouts: Timeouts,
) -> Result<(), String> {
    let ctl = |command: &str, extra: &[&str]| -> Result<Option<i32>, String> {
        let mut args = vec!["--config-dir", config, "ctl", command];
        args.extend_from_slice(extra);
        status(cli, &args)
    };

    let deadline = Instant::now() + timeouts.start;
    loop {
        if let Some(code) = app.try_wait().map_err(|e| e.to_string())? {
            return Err(format!(
                "the app exited with status {code} before it answered"
            ));
        }
        match ctl("status", &["--timeout", "500"])? {
            Some(0) => break,
            Some(code) if code == NOT_RUNNING => {}
            other => return Err(format!("ctl status failed with status {other:?}")),
        }
        check(
            Instant::now() < deadline,
            &format!("ctl status did not answer within {:?}", timeouts.start),
        )?;
        sleep(POLL);
    }
    println!("ok: the app answers ctl status");

    check(
        ctl("quit", &[])? == Some(0),
        "ctl quit was not acknowledged",
    )?;
    let deadline = Instant::now() + timeouts.quit;
    loop {
        if let Some(code) = app.try_wait().map_err(|e| e.to_string())? {
            check(code.success(), &format!("the app quit with status {code}"))?;
            println!("ok: the app quit on ctl quit");
            return Ok(());
        }
        check(
            Instant::now() < deadline,
            &format!("the app did not quit within {:?}", timeouts.quit),
        )?;
        sleep(POLL);
    }
}

fn check(condition: bool, message: &str) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(message.to_string())
    }
}

/// Run `cli args`; a non-zero exit status is a failure that shows the output.
fn run(cli: &Path, args: &[&str]) -> Result<String, String> {
    println!("$ {} {}", cli.display(), args.join(" "));
    let mut child = Command::new(cli)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not run {}: {e}", cli.display()))?;
    let deadline = Instant::now() + COMMAND_TIMEOUT;
    // The pipes are small for these commands; reading after the exit cannot block them.
    loop {
        if child.try_wait().map_err(|e| e.to_string())?.is_some() {
            break;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            return Err(format!("{} timed out", cli.display()));
        }
        sleep(Duration::from_millis(50));
    }
    let output = child.wait_with_output().map_err(|e| e.to_string())?;
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    if !output.status.success() {
        return Err(format!(
            "exit status {}\n--- stdout ---\n{stdout}\n--- stderr ---\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(stdout)
}

/// Exit status of `cli args` (`None` when it was killed by a signal).
fn status(cli: &Path, args: &[&str]) -> Result<Option<i32>, String> {
    Command::new(cli)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.code())
        .map_err(|e| format!("could not run {}: {e}", cli.display()))
}

#[cfg(test)]
#[path = "smoke_test_bundle_tests.rs"]
mod tests;
