//! Command-line entry points: `eye-tracker` (console) and `eye-tracker-gui` (windowed).
//!
//! Without a subcommand the tray app starts (`run`). Other subcommands help with setup and
//! troubleshooting and never need a display, except `doctor`, which reports monitors when one
//! is available.
//!
//! Exit codes: 0 success, 1 error, 2 usage error, 3 the app is not running (`ctl`), 130
//! interrupted.

use std::path::{Path, PathBuf};

use crate::platform::base::which;
use crate::{APP_NAME, APP_SLUG, paths};

pub const EXIT_OK: i32 = 0;
pub const EXIT_ERROR: i32 = 1;
pub const EXIT_USAGE: i32 = 2;
pub const EXIT_NOT_RUNNING: i32 = 3;
pub const EXIT_INTERRUPTED: i32 = 130;

/// Name of the console executable next to the windowed one in the Windows and macOS builds
/// (packaging/pyinstaller/eye-tracker.spec). The Linux build has one executable for both roles.
pub const FROZEN_CLI_NAME: &str = "eye-tracker-cli";

pub const BACKEND_CHOICES: [&str; 3] = ["auto", "facemesh", "lite"];
/// Backend names of Eye Tracker 0.1, still accepted (e.g. in desktop shortcuts).
pub const LEGACY_BACKENDS: [(&str, &str); 2] = [("mediapipe", "facemesh"), ("opencv", "lite")];
/// Mirrors `ipc::COMMANDS`.
pub const CTL_COMMANDS: [&str; 11] = [
    "calibrate",
    "pause",
    "privacy-off",
    "privacy-on",
    "privacy-toggle",
    "quit",
    "resume",
    "settings",
    "show",
    "status",
    "toggle",
];

// TODO(port): src/eye_tracker/cli.py: the argument parser (build_parser, _EXAMPLES, _backend_name,
// _positive_float), the subcommand handlers, main and gui_main.

// --------------------------------------------------------- commands shown to users

/// Windowed executables next to [`FROZEN_CLI_NAME`] (Windows, macOS).
const FROZEN_GUI_NAMES: [&str; 2] = ["EyeTracker", APP_NAME];

/// What the commands are resolved from; the public functions fill it from the running process.
struct Context<'a> {
    /// The running executable (`sys.executable`).
    exe: &'a Path,
    /// `std::env::consts::OS`.
    os: &'a str,
    /// The `APPIMAGE` environment variable ("" when unset).
    appimage: &'a str,
    /// Finds an executable on the PATH of this process (`shutil.which`).
    which: &'a dyn Fn(&str) -> Option<PathBuf>,
    /// The `--config-dir` profile ([`paths::base_override`]).
    profile: Option<&'a Path>,
}

/// Runs `f` with the context of the running process.
fn with_process_context<T>(f: impl FnOnce(&Context) -> T) -> T {
    // An unreadable executable path degrades to the bare command name.
    let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from(APP_SLUG));
    let appimage = std::env::var("APPIMAGE").unwrap_or_default();
    let profile = paths::base_override();
    f(&Context {
        exe: &exe,
        os: std::env::consts::OS,
        appimage: &appimage,
        which: &which,
        profile: profile.as_deref(),
    })
}

/// The command line that runs this installation's command-line interface.
///
/// Hints such as "bind `… ctl toggle` to a desktop shortcut" or "run `… doctor`" must name a
/// command that exists here. Only the Windows installer (its `eye-tracker.exe`, for terminals
/// opened after the installation, unless its PATH option was unticked) puts an `eye-tracker`
/// command on the PATH. The result, followed by `args`, is
///
/// * inside an AppImage: the `.AppImage` file (AppRun passes the arguments on; the mounted
///   bundle path changes on every run);
/// * the Windows installer's copy, when this process's PATH finds its `eye-tracker.exe`:
///   `eye-tracker`;
/// * otherwise the console executable of this copy (`eye-tracker-cli.exe` next to
///   `EyeTracker.exe`, `eye-tracker-cli` inside the macOS app, the one executable of the Linux
///   bundle).
///
/// A `--config-dir` profile is passed on (`--config-dir DIR` follows the program): each profile
/// is a separate instance with its own command socket, so `ctl` without it would talk to the
/// default profile.
pub fn cli_command(args: &[&str]) -> Vec<String> {
    with_process_context(|ctx| cli_command_in(ctx, args))
}

fn cli_command_in(ctx: &Context, args: &[&str]) -> Vec<String> {
    let mut command = cli_base(ctx);
    command.extend(profile_args(ctx));
    command.extend(args.iter().map(|a| (*a).to_string()));
    command
}

/// [`cli_command`] as the user would type it (quoted for this OS's shell).
pub fn cli_command_text(args: &[&str]) -> String {
    format_command(&cli_command(args))
}

/// The command line that starts the tray app of this installation.
///
/// Like [`cli_command`], except that the Windows and macOS builds name their windowed
/// executable: a console one (`eye-tracker-cli`, or the installer's `eye-tracker.exe`) would
/// tie the app to the terminal it was started from.
pub fn app_command() -> Vec<String> {
    with_process_context(app_command_in)
}

fn app_command_in(ctx: &Context) -> Vec<String> {
    let mut base = cli_base(ctx);
    if appimage(ctx).is_none() {
        for name in FROZEN_GUI_NAMES {
            let windowed = ctx.exe.with_file_name(format!("{name}{}", suffix(ctx.exe)));
            if windowed.is_file() {
                base = vec![windowed.to_string_lossy().into_owned()];
                break;
            }
        }
    }
    base.extend(profile_args(ctx));
    base
}

/// Render a command as the user would type it on this OS.
pub fn format_command(parts: &[String]) -> String {
    format_command_for(parts, cfg!(windows))
}

fn format_command_for(parts: &[String], windows: bool) -> String {
    if windows {
        list2cmdline(parts)
    } else {
        parts
            .iter()
            .map(|p| shlex_quote(p))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Python's `shlex.quote`.
fn shlex_quote(arg: &str) -> String {
    let safe = |c: char| c.is_ascii_alphanumeric() || "_@%+=:,./-".contains(c);
    if !arg.is_empty() && arg.chars().all(safe) {
        arg.to_string()
    } else {
        format!("'{}'", arg.replace('\'', "'\"'\"'"))
    }
}

/// Python's `subprocess.list2cmdline`: the quoting rules of the MS C runtime, for cmd.exe.
fn list2cmdline(parts: &[String]) -> String {
    let mut result = String::new();
    for (i, arg) in parts.iter().enumerate() {
        if i > 0 {
            result.push(' ');
        }
        let need_quote = arg.contains([' ', '\t']) || arg.is_empty();
        if need_quote {
            result.push('"');
        }
        let mut backslashes = 0;
        for c in arg.chars() {
            match c {
                '\\' => backslashes += 1,
                '"' => {
                    result.push_str(&"\\".repeat(backslashes * 2));
                    backslashes = 0;
                    result.push_str("\\\"");
                }
                _ => {
                    result.push_str(&"\\".repeat(backslashes));
                    backslashes = 0;
                    result.push(c);
                }
            }
        }
        result.push_str(&"\\".repeat(backslashes));
        if need_quote {
            result.push_str(&"\\".repeat(backslashes));
            result.push('"');
        }
    }
    result
}

fn profile_args(ctx: &Context) -> Vec<String> {
    match ctx.profile {
        Some(profile) => vec!["--config-dir".to_string(), profile.display().to_string()],
        None => Vec::new(),
    }
}

/// The program part of [`cli_command`] (no profile, no arguments).
fn cli_base(ctx: &Context) -> Vec<String> {
    if let Some(appimage) = appimage(ctx) {
        return vec![appimage];
    }
    if ctx.os == "windows" && installer_command_on_path(ctx) {
        return vec![APP_SLUG.to_string()];
    }
    let stem = ctx
        .exe
        .file_stem()
        .map(|s| s.to_string_lossy().to_lowercase());
    if stem.as_deref() != Some(FROZEN_CLI_NAME) {
        // The windowed executable has no console: its output would be lost.
        let console = ctx
            .exe
            .with_file_name(format!("{FROZEN_CLI_NAME}{}", suffix(ctx.exe)));
        if console.is_file() {
            return vec![console.to_string_lossy().into_owned()];
        }
    }
    vec![ctx.exe.to_string_lossy().into_owned()]
}

/// The extension of `exe` with its dot ("" when it has none), like `Path.suffix`.
fn suffix(exe: &Path) -> String {
    exe.extension()
        .map(|e| format!(".{}", e.to_string_lossy()))
        .unwrap_or_default()
}

/// The `.AppImage` file this Linux build runs from, if any.
///
/// `APPIMAGE` is inherited by every child of an AppImage (a terminal emulator AppImage, for
/// instance); a variable naming no file is ignored.
fn appimage(ctx: &Context) -> Option<String> {
    (ctx.os == "linux" && !ctx.appimage.is_empty() && Path::new(ctx.appimage).is_file())
        .then(|| ctx.appimage.to_string())
}

/// Whether `eye-tracker` on this process's PATH is the Windows installer's `eye-tracker.exe` of
/// this very copy (next to the running executable).
///
/// A process started before the installer changed the PATH (the app launched from its last
/// page) does not see the entry yet and keeps the full path.
fn installer_command_on_path(ctx: &Context) -> bool {
    (ctx.which)(APP_SLUG)
        .is_some_and(|found| same_file(&found, &ctx.exe.with_file_name(format!("{APP_SLUG}.exe"))))
}

/// How `--help` names the program: [`cli_command`] without directories.
///
/// `eye-tracker` on the PATH set by the Windows installer, `eye-tracker-cli.exe` or the
/// `.AppImage` file name in the other packages.
pub fn program_name() -> String {
    with_process_context(program_name_in)
}

fn program_name_in(ctx: &Context) -> String {
    let base = cli_base(ctx);
    Path::new(&base[0])
        .file_name()
        .map_or_else(|| base[0].clone(), |n| n.to_string_lossy().into_owned())
}

fn same_file(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

#[cfg(test)]
#[path = "cli_tests.rs"]
mod tests;
