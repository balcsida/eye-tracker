//! Command-line entry points: `eye-tracker` (console) and `eye-tracker-gui` (windowed).
//!
//! Without a subcommand the tray app starts (`run`). Other subcommands help with setup and
//! troubleshooting and never need a display, except `doctor`, which reports monitors when one
//! is available.
//!
//! Exit codes: 0 success, 1 error, 2 usage error, 3 the app is not running (`ctl`), 130
//! interrupted.

use std::ffi::OsString;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use clap::{ArgAction, Args, CommandFactory, FromArgMatches, Subcommand};
use serde_json::Value;

use crate::app::{self, RunArgs};
use crate::config::Settings;
use crate::diagnostics::{self, BenchOptions};
use crate::logging_setup::{LEVELS, setup_logging};
use crate::platform::autostart::{AutostartBackend, Status, SystemAutostart};
use crate::platform::base::which;
use crate::{APP_NAME, APP_SLUG, VERSION, ipc, paths};

pub const EXIT_OK: i32 = 0;
pub const EXIT_ERROR: i32 = 1;
pub const EXIT_USAGE: i32 = 2;
pub const EXIT_NOT_RUNNING: i32 = 3;
pub const EXIT_INTERRUPTED: i32 = 130;

/// Name of the console executable next to the windowed one in the Windows and macOS builds
/// (packaging/windows/installer.iss, packaging/macos/make_app.sh). The Linux build has one
/// executable for both roles.
pub const FROZEN_CLI_NAME: &str = "eye-tracker-cli";

/// `apple` is accepted everywhere (settings files travel between machines) and works on macOS.
pub const BACKEND_CHOICES: [&str; 4] = ["auto", "facemesh", "lite", "apple"];
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

/// `--help` examples: (arguments, explanation).
const EXAMPLES: [(&str, &str); 6] = [
    ("", "start the tray app"),
    (
        "calibrate",
        "calibrate (in the running app, if there is one)",
    ),
    (
        "ctl privacy-toggle",
        "bind this to a desktop shortcut (e.g. on Wayland)",
    ),
    ("doctor", "show a diagnostics report for bug reports"),
    (
        "bench --seconds 10",
        "measure CPU use and latency on this machine",
    ),
    ("autostart enable", "start at login"),
];

/// `CTL_COMMANDS` joined, as the help of `ctl COMMAND` lists them.
const CTL_COMMANDS_HELP: &str = "calibrate, pause, privacy-off, privacy-on, privacy-toggle, quit, \
resume, settings, show, status, toggle";

/// What `autostart status` adds for a [`Status`] value that needs an explanation (`{command}`:
/// this copy's `autostart enable`).
const AUTOSTART_STATUS_HELP: [(&str, &str); 2] = [
    (
        "stale",
        "Broken: the registered program no longer exists or is in a temporary location. Run \
'{command}' to repair it.",
    ),
    (
        "other-profile",
        "The entry starts another profile (--config-dir). Run '{command}' to start this one \
instead.",
    ),
];

/// `CreateProcess` flags of the detached app: no console, and Ctrl+C in the terminal does not
/// reach it.
const DETACHED_PROCESS: u32 = 0x0000_0008;
const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
#[cfg_attr(not(windows), allow(dead_code))]
const DETACHED_FLAGS: u32 = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP;

// ---------------------------------------------------------------------- parser
//
// The global options are `global` arguments: clap accepts them before and after the subcommand,
// a value after the subcommand wins and an absent one does not reset the other (what the Python
// parser did with suppressed defaults on the subparsers).

#[derive(clap::Parser, Debug)]
#[command(
    disable_version_flag = true,
    subcommand_help_heading = "commands",
    subcommand_value_name = "COMMAND"
)]
struct Cli {
    /// show program's version number and exit
    #[arg(long)]
    version: bool,
    #[command(flatten)]
    global: GlobalOptions,
    #[command(flatten)]
    run: RunOptions,
    #[command(subcommand)]
    command: Option<Sub>,
}

#[derive(Args, Debug, Clone, Default)]
struct GlobalOptions {
    /// keep settings, calibration and logs in DIR (portable mode; a separate instance)
    #[arg(
        long,
        global = true,
        value_name = "DIR",
        help_heading = "global options"
    )]
    config_dir: Option<String>,
    /// logging verbosity (default: from the settings, INFO)
    #[arg(
        long,
        global = true,
        value_name = "{DEBUG,INFO,WARNING,ERROR}",
        value_parser = log_level,
        hide_possible_values = true,
        help_heading = "global options"
    )]
    log_level: Option<String>,
    /// camera index (0, 1, ...) or an image file or a folder of frames; overrides the settings
    #[arg(
        long,
        global = true,
        value_name = "DEV",
        help_heading = "global options"
    )]
    camera: Option<String>,
    /// vision backend: facemesh (head and eyes), lite (head only, lightest), apple (macOS:
    /// Apple's Vision framework, head and eyes) or auto; overrides the settings
    #[arg(
        long,
        global = true,
        value_name = "{auto,facemesh,lite,apple}",
        value_parser = backend_option,
        hide_possible_values = true,
        help_heading = "global options"
    )]
    backend: Option<String>,
    /// start quietly, as at login: no first-run wizard, no startup notifications
    #[arg(long, global = true, action = ArgAction::SetTrue, help_heading = "global options")]
    background: bool,
}

#[derive(Args, Debug, Clone, Default)]
struct RunOptions {
    /// open the calibration right after start
    #[arg(long)]
    calibrate: bool,
    /// append numeric tracking data (features, head angles, gaze, pointer position, decisions;
    /// never images) to FILE as JSON lines, for tuning and bug reports; applies only when the
    /// app starts
    #[arg(long, value_name = "FILE")]
    trace: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
enum Sub {
    #[command(
        about = "start the tray app (the default)",
        long_about = "Start the tray app. If it is already running, the running instance is shown \
instead (with --background, nothing is), and --trace, --camera and --backend are not applied: \
they only take effect when the app starts."
    )]
    Run(RunOptions),
    #[command(
        about = "calibrate now",
        long_about = "Open the calibration in the running app, or start the app and calibrate."
    )]
    Calibrate,
    #[command(
        about = "print a diagnostics report",
        long_about = "Print everything useful for a bug report (paths are shown relative to ~)."
    )]
    Doctor {
        /// machine-readable output
        #[arg(long)]
        json: bool,
        /// briefly open cameras 0-3 to list the working ones (their lights may flash)
        #[arg(long)]
        probe_cameras: bool,
    },
    #[command(
        about = "measure CPU use and latency",
        long_about = "Run the capture and analysis loop without the UI, first as fast as possible \
and then at the idle rate the app uses while you sit still. Use --camera to benchmark an image file or a \
folder of frames instead of the configured camera."
    )]
    Bench {
        /// duration of each of the two runs (default: 5)
        #[arg(
            long,
            value_name = "N",
            value_parser = positive_float,
            default_value_t = 5.0,
            hide_default_value = true,
            allow_negative_numbers = true
        )]
        seconds: f64,
        /// machine-readable output
        #[arg(long)]
        json: bool,
    },
    #[command(
        about = "send a command to the running app",
        long_about = "Send a command to the running app. Bind these to desktop keyboard shortcuts \
where global hotkeys are unavailable (Wayland). 'status' prints JSON."
    )]
    Ctl {
        #[arg(
            value_name = "COMMAND",
            value_parser = clap::builder::PossibleValuesParser::new(CTL_COMMANDS),
            hide_possible_values = true,
            help = CTL_COMMANDS_HELP
        )]
        action: String,
        /// how long to wait for an answer (default: 1500)
        #[arg(
            long,
            value_name = "MS",
            default_value_t = 1500,
            hide_default_value = true,
            allow_negative_numbers = true
        )]
        timeout: i64,
    },
    #[command(
        about = "start at login: enable, disable or status",
        long_about = "Manage starting the app at login (per user, no administrator rights)."
    )]
    Autostart {
        #[arg(
            default_value = "status",
            value_parser = ["enable", "disable", "status"]
        )]
        action: String,
    },
    #[command(
        about = "delete the calibration and/or settings",
        long_about = "Delete stored data. The app must not be running."
    )]
    Reset {
        /// delete the calibration
        #[arg(long)]
        calibration: bool,
        /// restore default settings
        #[arg(long)]
        settings: bool,
        /// both of the above
        #[arg(long)]
        all: bool,
    },
}

/// A `--log-level` value: any case, one of [`LEVELS`].
fn log_level(text: &str) -> Result<String, String> {
    let level = text.to_uppercase();
    if LEVELS.contains(&level.as_str()) {
        Ok(level)
    } else {
        Err(invalid_choice(&level, &LEVELS))
    }
}

/// A `--backend` value, with the 0.1 names mapped to the current ones.
pub fn backend_name(text: &str) -> String {
    let name = text.trim().to_lowercase();
    LEGACY_BACKENDS
        .iter()
        .find(|(legacy, _)| *legacy == name)
        .map_or(name, |(_, current)| (*current).to_string())
}

fn backend_option(text: &str) -> Result<String, String> {
    let name = backend_name(text);
    if BACKEND_CHOICES.contains(&name.as_str()) {
        Ok(name)
    } else {
        Err(invalid_choice(&name, &BACKEND_CHOICES))
    }
}

fn invalid_choice(value: &str, choices: &[&str]) -> String {
    format!(
        "invalid choice: '{value}' (choose from {})",
        choices.join(", ")
    )
}

fn positive_float(text: &str) -> Result<f64, String> {
    let value: f64 = text
        .trim()
        .parse()
        .map_err(|_| format!("not a number: '{text}'"))?;
    if value > 0.0 && value.is_finite() {
        Ok(value)
    } else {
        Err("must be a positive number".to_string())
    }
}

/// Examples spelled with the name this copy is run by (see [`cli_command`]: `eye-tracker`
/// exists after the Windows installer; the other packages have `eye-tracker-cli` or the
/// `.AppImage` file).
fn epilog(prog: &str) -> String {
    let commands: Vec<String> = EXAMPLES
        .iter()
        .map(|(args, _)| format!("{prog} {args}").trim_end().to_string())
        .collect();
    let width = commands
        .iter()
        .map(|c| c.chars().count())
        .max()
        .unwrap_or(0)
        + 2;
    let lines: Vec<String> = commands
        .iter()
        .zip(EXAMPLES)
        .map(|(command, (_, explanation))| format!("  {command:<width$}{explanation}"))
        .collect();
    format!(
        "examples:\n{}\n\nexit codes: 0 ok, 1 error, 2 usage error, 3 not running (ctl), 130 interrupted\n",
        lines.join("\n")
    )
}

/// The argument parser (global options work before or after the subcommand).
fn build_command() -> clap::Command {
    let prog = program_name();
    Cli::command()
        .bin_name(prog.clone())
        .about(format!(
            "{APP_NAME}: look at a monitor and the mouse cursor and keyboard focus follow. \
Webcam-based, private and offline."
        ))
        .after_help(epilog(&prog))
}

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
    with_context_of(&exe, std::env::consts::OS, f)
}

/// Runs `f` with the context of an executable of the given OS (the process, unless a test says
/// otherwise).
fn with_context_of<T>(exe: &Path, os: &str, f: impl FnOnce(&Context) -> T) -> T {
    let appimage = std::env::var("APPIMAGE").unwrap_or_default();
    let profile = paths::base_override();
    f(&Context {
        exe,
        os,
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

// ------------------------------------------------------------------ entry points

type SpawnFn<'a> = dyn Fn(&[String]) -> io::Result<()> + 'a;

/// What the commands act on; [`Env::process`] is the running process, tests replace the parts
/// they observe.
struct Env<'a> {
    out: &'a mut dyn Write,
    err: &'a mut dyn Write,
    /// Runs the tray app (`app::run_app`).
    run_app: Box<dyn Fn(&RunArgs) -> i32 + 'a>,
    autostart: Box<dyn AutostartBackend + 'a>,
    /// Starts a detached process from a command line.
    spawn: Box<SpawnFn<'a>>,
    instance_running: Box<dyn Fn() -> bool + 'a>,
    /// `logging_setup::setup_logging` (it starts the process-wide logger, which tests must not).
    setup_logging: fn(&str, bool, bool),
    /// The running executable and `std::env::consts::OS`.
    exe: PathBuf,
    os: String,
}

impl<'a> Env<'a> {
    fn process(out: &'a mut dyn Write, err: &'a mut dyn Write) -> Self {
        Env {
            out,
            err,
            run_app: Box::new(app::run_app),
            autostart: Box::new(SystemAutostart),
            spawn: Box::new(spawn_detached),
            instance_running: Box::new(|| ipc::is_running(ipc::PROBE_TIMEOUT_MS)),
            setup_logging,
            // An unreadable executable path degrades to the bare command name.
            exe: std::env::current_exe().unwrap_or_else(|_| PathBuf::from(APP_SLUG)),
            os: std::env::consts::OS.to_string(),
        }
    }

    /// Command-line tools report on the terminal and leave the app's log file alone.
    fn setup_cli_logging(&self, global: &GlobalOptions) {
        (self.setup_logging)(
            global.log_level.as_deref().unwrap_or("WARNING"),
            true,
            false,
        );
    }

    // A windowed process may have no console to print to: a failed write is not an error.
    fn out(&mut self, message: &str) {
        let _ = writeln!(self.out, "{message}");
        let _ = self.out.flush();
    }

    fn out_raw(&mut self, text: &str) {
        let _ = write!(self.out, "{text}");
        let _ = self.out.flush();
    }

    fn err_raw(&mut self, text: &str) {
        let _ = write!(self.err, "{text}");
        let _ = self.err.flush();
    }

    fn err(&mut self, message: &str) {
        let _ = writeln!(self.err, "{message}");
        let _ = self.err.flush();
    }
}

/// Console entry point (`eye-tracker`). Returns the process exit code.
pub fn main() -> i32 {
    main_from(std::env::args_os())
}

/// [`main`] with the command line given (program name first).
pub fn main_from<I, T>(argv: I) -> i32
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let (mut out, mut err) = (io::stdout(), io::stderr());
    run_cli(argv, &mut Env::process(&mut out, &mut err))
}

/// Entry point of the windowed launcher (`eye-tracker-gui`).
///
/// Such a process may have no console, so nothing here panics on a failed print (the output
/// helpers ignore write errors, and `println!` is not used). Everything else is the same as
/// [`main`].
pub fn gui_main() -> i32 {
    main()
}

fn run_cli<I, T>(argv: I, env: &mut Env) -> i32
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let cli = match build_command()
        .try_get_matches_from(argv)
        .and_then(|matches| Cli::from_arg_matches(&matches))
    {
        Ok(cli) => cli,
        Err(error) => {
            // --help and usage errors.
            let text = error.render().to_string();
            if error.use_stderr() {
                env.err_raw(&text);
            } else {
                env.out_raw(&text);
            }
            return error.exit_code();
        }
    };
    if cli.version {
        env.out(&format!("{APP_NAME} {VERSION}"));
        return EXIT_OK;
    }
    if let Some(dir) = cli.global.config_dir.as_deref().filter(|d| !d.is_empty()) {
        paths::set_base_override(Some(Path::new(dir)));
    }
    let global = &cli.global;
    match &cli.command {
        None => cmd_run(env, run_args(global, &cli.run, &RunOptions::default())),
        Some(Sub::Run(options)) => cmd_run(env, run_args(global, &cli.run, options)),
        Some(Sub::Calibrate) => {
            let mut args = run_args(global, &cli.run, &RunOptions::default());
            // run_app forwards "calibrate" to a running instance instead of "show", so this
            // covers both "open it in the running app" and "start the app and calibrate".
            args.calibrate = true;
            cmd_run(env, args)
        }
        Some(Sub::Doctor {
            json,
            probe_cameras,
        }) => cmd_doctor(env, global, *json, *probe_cameras),
        Some(Sub::Bench { seconds, json }) => cmd_bench(env, global, *seconds, *json),
        Some(Sub::Ctl { action, timeout }) => cmd_ctl(env, global, action, *timeout),
        Some(Sub::Autostart { action }) => cmd_autostart(env, global, action),
        Some(Sub::Reset {
            calibration,
            settings,
            all,
        }) => cmd_reset(env, global, *calibration, *settings, *all),
    }
}

/// The run options of the main parser and of the `run` subcommand (a value after the subcommand
/// wins).
fn run_args(global: &GlobalOptions, top: &RunOptions, sub: &RunOptions) -> RunArgs {
    RunArgs {
        config_dir: global
            .config_dir
            .as_deref()
            .filter(|d| !d.is_empty())
            .map(PathBuf::from),
        log_level: global.log_level.clone(),
        camera: global.camera.clone(),
        backend: global.backend.clone(),
        background: global.background,
        calibrate: top.calibrate || sub.calibrate,
        trace: sub.trace.clone().or_else(|| top.trace.clone()),
    }
}

// -------------------------------------------------------------------- commands

fn cmd_run(env: &mut Env, args: RunArgs) -> i32 {
    if start_detached(env, &args) {
        return EXIT_OK;
    }
    (env.run_app)(&args)
}

/// Start the tray app as the windowed executable, detached from this console.
///
/// Only for the console executables of a Windows package (`eye-tracker.exe` on the installer's
/// PATH, `eye-tracker-cli.exe`): run there, the app would live inside the terminal, filling it
/// with log lines, and closing the terminal would kill it without a clean shutdown. A running
/// instance is handed over in this process instead (`run_app`), so that what it answers (e.g.
/// "options not applied") is printed. Returns whether the app was started; `false` means: run it
/// here.
fn start_detached(env: &mut Env, args: &RunArgs) -> bool {
    let Some(command) = detached_command(env, args) else {
        return false;
    };
    if let Err(err) = (env.spawn)(&command) {
        log::warn!(
            "Could not start {} on its own ({err}); running it here",
            command[0]
        );
        return false;
    }
    env.out(&format!(
        "{APP_NAME} is starting in the system tray; you can close this window."
    ));
    true
}

/// The command line that starts the app detached, or `None` when it is to run in this process.
fn detached_command(env: &Env, args: &RunArgs) -> Option<Vec<String>> {
    if env.os != "windows" || !is_console_executable(&env.exe) {
        return None;
    }
    let mut command = with_context_of(&env.exe, &env.os, app_command_in);
    if is_console_executable(Path::new(&command[0])) {
        return None; // no windowed executable next to this one
    }
    if (env.instance_running)() {
        return None;
    }
    let text = |value: &Option<String>| value.clone().filter(|v| !v.is_empty());
    let trace = args
        .trace
        .as_ref()
        .filter(|t| !t.as_os_str().is_empty())
        .map(|t| resolve_path(t).to_string_lossy().into_owned());
    for (flag, value) in [
        ("--log-level", text(&args.log_level)),
        ("--camera", text(&args.camera)),
        ("--backend", text(&args.backend)),
        ("--trace", trace),
    ] {
        if let Some(value) = value {
            command.extend([flag.to_string(), value]);
        }
    }
    if args.background {
        command.push("--background".to_string());
    }
    if args.calibrate {
        command.push("--calibrate".to_string());
    }
    Some(command)
}

/// Whether `exe` is `eye-tracker-cli` or `eye-tracker` (with or without `.exe`).
fn is_console_executable(exe: &Path) -> bool {
    exe.file_stem()
        .map(|s| s.to_string_lossy().to_lowercase())
        .is_some_and(|stem| stem == FROZEN_CLI_NAME || stem == APP_SLUG)
}

/// Python's `Path.resolve()`: absolute, with the symlinks of the part that exists followed.
fn resolve_path(path: &Path) -> PathBuf {
    if let Ok(resolved) = path.canonicalize() {
        return resolved;
    }
    let absolute = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    match (
        absolute.parent().and_then(|p| p.canonicalize().ok()),
        absolute.file_name(),
    ) {
        (Some(parent), Some(name)) => parent.join(name),
        _ => absolute,
    }
}

fn spawn_detached(command: &[String]) -> io::Result<()> {
    let mut process = Command::new(&command[0]);
    process
        .args(&command[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        process.creation_flags(DETACHED_FLAGS);
    }
    // Our own windowed executable, arguments from our parser.
    process.spawn().map(drop)
}

fn cmd_doctor(env: &mut Env, global: &GlobalOptions, json: bool, probe_cameras: bool) -> i32 {
    env.setup_cli_logging(global);
    let report = diagnostics::collect_report(probe_cameras, None);
    if json {
        env.out(&json_text(&report));
    } else {
        env.out_raw(&diagnostics::format_report(&report));
    }
    EXIT_OK
}

fn cmd_bench(env: &mut Env, global: &GlobalOptions, seconds: f64, json: bool) -> i32 {
    env.setup_cli_logging(global);
    let settings = read_settings();
    let device = global
        .camera
        .clone()
        .unwrap_or_else(|| settings.camera.device.clone());
    let backend = global
        .backend
        .clone()
        .filter(|b| !b.is_empty())
        .unwrap_or_else(|| settings.general.backend.clone());
    env.err(&format!("Benchmarking for 2 x {seconds} s..."));
    let options = BenchOptions {
        width: usize::try_from(settings.camera.width).unwrap_or(640),
        height: usize::try_from(settings.camera.height).unwrap_or(480),
        api: settings.camera.api.clone(),
        idle_fps: None,
        coreml: settings.general.coreml,
    };
    let result = match diagnostics::run_bench(seconds, &device, &backend, &options) {
        Ok(result) => result,
        Err(error) => {
            let mut message = error.to_string();
            let index = device.trim();
            if !index.is_empty()
                && index.chars().all(|c| c.is_ascii_digit())
                && (env.instance_running)()
            {
                message.push_str(&format!(
                    "\n{APP_NAME} is running and may be holding the camera; release it with \
'{}' and try again.",
                    cli_command_text(&["ctl", "privacy-on"])
                ));
            }
            env.err(&format!("error: {message}"));
            return EXIT_ERROR;
        }
    };
    if json {
        env.out(&json_text(&result));
    } else {
        env.out_raw(&diagnostics::format_bench(&result));
    }
    EXIT_OK
}

fn cmd_ctl(env: &mut Env, global: &GlobalOptions, action: &str, timeout: i64) -> i32 {
    env.setup_cli_logging(global);
    let timeout_ms = u64::try_from(timeout).unwrap_or(0).max(1);
    let reply = match ipc::send_command(action, timeout_ms, 0) {
        Ok(reply) => reply,
        Err(error) => {
            env.err(&format!("error: {error}"));
            return EXIT_ERROR;
        }
    };
    let Some(reply) = reply else {
        if (env.instance_running)() {
            env.err(&format!(
                "error: {APP_NAME} is running but did not answer within {timeout} ms."
            ));
            return EXIT_ERROR;
        }
        env.err(&format!(
            "{APP_NAME} is not running. Start it with '{}'.",
            format_command(&app_command())
        ));
        return EXIT_NOT_RUNNING;
    };
    if reply.starts_with("error") {
        env.err(&reply);
        return EXIT_ERROR;
    }
    env.out(&reply);
    EXIT_OK
}

fn cmd_autostart(env: &mut Env, global: &GlobalOptions, action: &str) -> i32 {
    env.setup_cli_logging(global);
    // The login entry starts this profile: with --config-dir it must pass the same directory,
    // or the default profile would start at login instead.
    let profile = global
        .config_dir
        .as_deref()
        .filter(|d| !d.is_empty())
        .map(Path::new);
    let backend = &*env.autostart;
    let mut lines = Vec::new();
    let mut errors = Vec::new();
    let code = if action == "status" {
        if !backend.is_supported() {
            lines.push("Start at login: not supported on this system".to_string());
        } else {
            let status = backend.status_for_profile(profile);
            // The value itself (enabled, disabled, stale, other-profile) is what scripts and bug
            // reports quote; the explanation says what to do.
            lines.push(format!("Start at login: {}", status.as_str()));
            if let Some((_, explanation)) = AUTOSTART_STATUS_HELP
                .iter()
                .find(|(value, _)| *value == status.as_str())
            {
                let command = cli_command_text(&["autostart", "enable"]);
                lines.push(format!("  {}", explanation.replace("{command}", &command)));
            }
            lines.push(format!(
                "Entry:      {}",
                backend.location().unwrap_or_default()
            ));
            if let Some(registered) = backend.registered_command() {
                lines.push(format!("Registered: {}", format_command(&registered)));
            }
            lines.push(format!(
                "This copy:  {}",
                format_command(&backend.launch_command(true, profile))
            ));
        }
        EXIT_OK
    } else if !backend.is_supported() {
        errors.push("error: start at login is not supported on this system.".to_string());
        EXIT_ERROR
    } else {
        let done = if action == "enable" {
            backend.enable_for_profile(true, profile).map(|()| {
                format!(
                    "Start at login enabled ({}).",
                    backend.location().unwrap_or_default()
                )
            })
        } else {
            backend.disable_for_profile(profile).map(|()| {
                if backend.status_for_profile(profile) == Status::OtherProfile {
                    "Start at login starts another profile; left unchanged.".to_string()
                } else {
                    "Start at login disabled.".to_string()
                }
            })
        };
        match done {
            Ok(message) => {
                lines.push(message);
                EXIT_OK
            }
            Err(error) => {
                errors.push(format!("error: {error}"));
                EXIT_ERROR
            }
        }
    };
    lines.iter().for_each(|line| env.out(line));
    errors.iter().for_each(|line| env.err(line));
    code
}

fn cmd_reset(
    env: &mut Env,
    global: &GlobalOptions,
    calibration: bool,
    settings: bool,
    all: bool,
) -> i32 {
    env.setup_cli_logging(global);
    let mut targets: Vec<PathBuf> = Vec::new();
    if all || settings {
        let settings_file = paths::settings_file();
        // Settings::load moves an unreadable file aside under this name.
        let mut corrupt = settings_file.clone().into_os_string();
        corrupt.push(".corrupt");
        targets.extend([settings_file, PathBuf::from(corrupt)]);
    }
    if all || calibration {
        targets.push(paths::calibration_file());
    }
    if targets.is_empty() {
        env.err("error: choose what to reset: --calibration, --settings or --all");
        return EXIT_USAGE;
    }
    if (env.instance_running)() {
        env.err(&format!(
            "error: {APP_NAME} is running and would write its data back. Quit it first ('{}'), \
then run reset again.",
            cli_command_text(&["ctl", "quit"])
        ));
        return EXIT_ERROR;
    }
    let mut removed = 0;
    for path in targets {
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => {
                env.err(&format!(
                    "error: could not delete {}: {error}",
                    path.display()
                ));
                return EXIT_ERROR;
            }
        }
        env.out(&format!("Deleted {}", path.display()));
        removed += 1;
    }
    if removed == 0 {
        env.out("Nothing to reset.");
    }
    EXIT_OK
}

// ---------------------------------------------------------------------- helpers

/// Settings as the app would see them, without moving a corrupt file aside.
fn read_settings() -> Settings {
    let path = paths::settings_file();
    match std::fs::read_to_string(&path) {
        Ok(text) => match serde_json::from_str::<Value>(&text) {
            Ok(data) => Settings::from_dict(&data),
            Err(error) => {
                log::warn!("Ignoring unreadable settings {}: {error}", path.display());
                Settings::default()
            }
        },
        Err(error) => {
            if error.kind() != io::ErrorKind::NotFound {
                log::warn!("Ignoring unreadable settings {}: {error}", path.display());
            }
            Settings::default()
        }
    }
}

/// `json.dumps(value, indent=2)`.
fn json_text(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_default()
}

#[cfg(test)]
#[path = "cli_tests.rs"]
mod tests;
