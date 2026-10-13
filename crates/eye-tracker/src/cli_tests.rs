//! Tests of the commands shown to users (tests/test_cli.py, `test_shown_commands_*` and the
//! command text of `test_ctl_says_how_to_start_this_copy`).
//!
//! `sys.executable` / `sys.frozen` / `sys.platform` / `shutil.which` are replaced by the
//! [`Context`] the commands are resolved from.
//!
//! Not ported: `test_shown_commands_from_source` (there is no source install, so no
//! `python -m eye_tracker` form), `test_help_examples_use_the_program_name` and the `ctl`
//! exit code / stderr part of `test_ctl_says_how_to_start_this_copy` (they drive the argument
//! parser, which is not ported yet; the command text it prints is covered here).

use std::fs;
use std::path::PathBuf;

use super::*;

/// A release package whose folder holds the given executables; the first is the running one.
struct Package {
    _dir: tempfile::TempDir,
    folder: PathBuf,
    exe: PathBuf,
}

fn package(running: &str, others: &[&str], folder_name: &str) -> Package {
    let dir = tempfile::tempdir().unwrap();
    let folder = dir.path().join(folder_name);
    fs::create_dir(&folder).unwrap();
    for name in std::iter::once(&running).chain(others) {
        fs::write(folder.join(name), b"").unwrap();
    }
    let exe = folder.join(running);
    Package {
        _dir: dir,
        folder,
        exe,
    }
}

fn path_of(p: &Package, name: &str) -> String {
    p.folder.join(name).to_string_lossy().into_owned()
}

fn no_path(_: &str) -> Option<PathBuf> {
    None
}

fn context<'a>(
    exe: &'a Path,
    os: &'a str,
    which: &'a dyn Fn(&str) -> Option<PathBuf>,
) -> Context<'a> {
    Context {
        exe,
        os,
        appimage: "",
        which,
        profile: None,
    }
}

fn strings(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|s| (*s).to_string()).collect()
}

#[test]
fn test_shown_commands_name_the_console_executable_of_the_windows_package() {
    let p = package("EyeTracker.exe", &["eye-tracker-cli.exe"], "Eye Tracker");
    let (cli_exe, gui_exe) = (
        path_of(&p, "eye-tracker-cli.exe"),
        path_of(&p, "EyeTracker.exe"),
    );
    let ctx = context(&p.exe, "linux", &no_path);
    // The windowed app has no console: commands it shows use the console twin.
    assert_eq!(
        cli_command_in(&ctx, &["ctl", "toggle"]),
        [cli_exe.as_str(), "ctl", "toggle"]
    );
    assert_eq!(app_command_in(&ctx), [gui_exe.as_str()]);
    assert_eq!(program_name_in(&ctx), "eye-tracker-cli.exe");

    // Run as the console executable: the same commands, and the app is started with the
    // windowed one (not tied to the terminal).
    let p = package("eye-tracker-cli.exe", &["EyeTracker.exe"], "Eye Tracker");
    let (cli_exe, gui_exe) = (
        path_of(&p, "eye-tracker-cli.exe"),
        path_of(&p, "EyeTracker.exe"),
    );
    let ctx = context(&p.exe, "linux", &no_path);
    assert_eq!(
        cli_command_in(&ctx, &["ctl", "quit"]),
        [cli_exe.as_str(), "ctl", "quit"]
    );
    assert_eq!(app_command_in(&ctx), [gui_exe.as_str()]);
}

#[test]
fn test_shown_commands_use_the_installers_command_on_the_path() {
    // The Windows installer adds eye-tracker.exe (a copy of eye-tracker-cli.exe) and puts its
    // folder on the PATH: hints then show the short command the docs use.
    let p = package(
        "EyeTracker.exe",
        &["eye-tracker-cli.exe", "eye-tracker.exe"],
        "Eye Tracker",
    );
    let gui_exe = path_of(&p, "EyeTracker.exe");
    let on_path = std::cell::RefCell::new(Some(p.folder.join("eye-tracker.exe")));
    let which = |_: &str| on_path.borrow().clone();
    let ctx = context(&p.exe, "windows", &which);
    assert_eq!(
        cli_command_in(&ctx, &["ctl", "toggle"]),
        ["eye-tracker", "ctl", "toggle"]
    );
    assert_eq!(program_name_in(&ctx), "eye-tracker");
    // The app is still started windowed, never through the console command.
    assert_eq!(app_command_in(&ctx), [gui_exe.as_str()]);

    let q = package("eye-tracker.exe", &["EyeTracker.exe"], "Eye Tracker");
    let gui_exe = path_of(&q, "EyeTracker.exe");
    *on_path.borrow_mut() = Some(q.exe.clone());
    let ctx = context(&q.exe, "windows", &which);
    assert_eq!(cli_command_in(&ctx, &["doctor"]), ["eye-tracker", "doctor"]);
    assert_eq!(app_command_in(&ctx), [gui_exe.as_str()]);

    // PATH option unticked (or not seen yet by this process): the full path.
    let ctx = context(&p.exe, "windows", &no_path);
    assert_eq!(
        cli_command_in(&ctx, &[]),
        [path_of(&p, "eye-tracker-cli.exe")]
    );
    // Another copy's eye-tracker on the PATH would control the wrong installation.
    let other = tempfile::tempdir().unwrap();
    let other_exe = other.path().join("eye-tracker.exe");
    fs::write(&other_exe, b"").unwrap();
    let elsewhere = |_: &str| Some(other_exe.clone());
    let ctx = context(&p.exe, "windows", &elsewhere);
    assert_eq!(
        cli_command_in(&ctx, &[]),
        [path_of(&p, "eye-tracker-cli.exe")]
    );
}

#[test]
fn test_shown_commands_in_the_macos_and_linux_packages() {
    // Eye Tracker.app/Contents/MacOS
    let p = package("eye-tracker-cli", &["Eye Tracker"], "Eye Tracker");
    let ctx = context(&p.exe, "macos", &no_path);
    assert_eq!(cli_command_in(&ctx, &[]), [path_of(&p, "eye-tracker-cli")]);
    assert_eq!(app_command_in(&ctx), [path_of(&p, "Eye Tracker")]);

    // one executable for both
    let tarball = package("eye-tracker", &[], "eye-tracker");
    let ctx = context(&tarball.exe, "linux", &no_path);
    assert_eq!(
        cli_command_in(&ctx, &["doctor"]),
        [path_of(&tarball, "eye-tracker").as_str(), "doctor"]
    );
    assert_eq!(app_command_in(&ctx), [path_of(&tarball, "eye-tracker")]);
}

#[test]
fn test_shown_commands_inside_an_appimage() {
    // the bundle inside the image, mounted at a random path
    let p = package("eye-tracker", &[], "Eye Tracker");
    let image_dir = tempfile::tempdir().unwrap();
    let image = image_dir.path().join("Eye_Tracker-x86_64.AppImage");
    fs::write(&image, b"").unwrap();
    let image_text = image.to_string_lossy().into_owned();
    let mut ctx = context(&p.exe, "linux", &no_path);
    ctx.appimage = &image_text;
    assert_eq!(
        cli_command_in(&ctx, &["ctl", "toggle"]),
        [image_text.as_str(), "ctl", "toggle"]
    );
    assert_eq!(app_command_in(&ctx), [image_text.as_str()]);
    assert_eq!(program_name_in(&ctx), "Eye_Tracker-x86_64.AppImage");
    // A stale variable (inherited from another AppImage) naming no file is ignored.
    let gone = image_dir.path().join("gone.AppImage");
    let gone_text = gone.to_string_lossy().into_owned();
    ctx.appimage = &gone_text;
    assert!(cli_command_in(&ctx, &[])[0].ends_with("eye-tracker"));
}

#[test]
fn test_shown_commands_keep_the_profile() {
    // Each --config-dir profile is its own instance: ctl must address this one.
    let p = package("eye-tracker", &[], "Eye Tracker");
    let exe = path_of(&p, "eye-tracker");
    let profile = tempfile::tempdir().unwrap();
    let profile_text = profile.path().to_string_lossy().into_owned();
    let mut ctx = context(&p.exe, "linux", &no_path);
    ctx.profile = Some(profile.path());
    assert_eq!(
        cli_command_in(&ctx, &["ctl", "toggle"]),
        [exe.as_str(), "--config-dir", &profile_text, "ctl", "toggle"]
    );
    assert_eq!(
        app_command_in(&ctx),
        [exe.as_str(), "--config-dir", &profile_text]
    );
}

#[test]
fn test_ctl_says_how_to_start_this_copy() {
    // The command text of the "not running" message: the windowed executable and the profile.
    let p = package("eye-tracker-cli.exe", &["EyeTracker.exe"], "Eye Tracker");
    let profile = tempfile::tempdir().unwrap();
    let mut ctx = context(&p.exe, "windows", &no_path);
    ctx.profile = Some(profile.path());
    let gui_exe = path_of(&p, "EyeTracker.exe");
    let profile_text = profile.path().to_string_lossy().into_owned();
    assert_eq!(
        app_command_in(&ctx),
        [gui_exe.as_str(), "--config-dir", &profile_text]
    );
}

#[test]
fn test_format_command_quotes_for_the_shell() {
    let parts = strings(&["/opt/Eye Tracker/eye-tracker", "it's", "", "plain-1.0"]);
    assert_eq!(
        format_command_for(&parts, false),
        "'/opt/Eye Tracker/eye-tracker' 'it'\"'\"'s' '' plain-1.0"
    );
    let parts = strings(&["C:\\Eye Tracker\\EyeTracker.exe", "a\"b", "", "dir\\ "]);
    assert_eq!(
        format_command_for(&parts, true),
        "\"C:\\Eye Tracker\\EyeTracker.exe\" a\\\"b \"\" \"dir\\ \""
    );
}
