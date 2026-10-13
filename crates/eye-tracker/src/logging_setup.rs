//! Logging configuration for the app and the command-line tools.
//!
//! The tray app usually runs without a console (the windowed binary, started at login), so the
//! rotating log file is the only place where problems become visible. This module therefore
//! also routes panics (main thread and worker threads) into that file.
//!
//! `flexi_logger` can only be started once per process. [`setup_logging`] starts it with a
//! writer whose sinks can be swapped, which is how repeated calls "replace the previous
//! handlers" as they did in Python.

use std::io::{self, Write};
use std::panic;
use std::sync::{Arc, Once};

use flexi_logger::writers::{FileLogWriter, LogWriter};
use flexi_logger::{
    Cleanup, Criterion, DeferredNow, FileSpec, LogSpecification, Logger, LoggerHandle, Naming,
};
use log::{LevelFilter, Record};
use parking_lot::{Mutex, RwLock};

use crate::{APP_SLUG, paths};

/// The format of every line, in `logging` notation. [`format_line`] produces it.
pub const LOG_FORMAT: &str = "%(asctime)s %(levelname)-7s %(name)s: %(message)s";
pub const MAX_BYTES: u64 = 1_000_000;
pub const BACKUP_COUNT: usize = 3;
pub const LEVELS: [&str; 4] = ["DEBUG", "INFO", "WARNING", "ERROR"];

/// Target of the panic hook's records; printed as `CRITICAL` like the Python crash logger.
const CRASH_TARGET: &str = "eye_tracker.crash";

#[derive(Default)]
struct Sinks {
    file: Option<FileLogWriter>,
    console: bool,
}

/// The one `flexi_logger` writer; forwards to whatever sinks `setup_logging` installed.
struct SwitchWriter(Arc<RwLock<Sinks>>);

impl LogWriter for SwitchWriter {
    fn write(&self, now: &mut DeferredNow, record: &Record) -> io::Result<()> {
        let sinks = self.0.read();
        if let Some(file) = &sinks.file {
            file.write(now, record)?;
        }
        if sinks.console {
            format_line(&mut io::stderr().lock(), now, record)?;
        }
        Ok(())
    }

    fn flush(&self) -> io::Result<()> {
        match &self.0.read().file {
            Some(file) => file.flush(),
            None => Ok(()),
        }
    }

    fn max_log_level(&self) -> LevelFilter {
        LevelFilter::Trace
    }
}

struct State {
    handle: Option<LoggerHandle>,
    sinks: Arc<RwLock<Sinks>>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);
static HOOK: Once = Once::new();

/// Turn `"debug"`, `"INFO"`, `"warning"` … into a level filter (`default` if invalid).
pub fn parse_level(level: Option<&str>, default: LevelFilter) -> LevelFilter {
    match level.map(|l| l.trim().to_ascii_uppercase()).as_deref() {
        Some("DEBUG") => LevelFilter::Debug,
        Some("INFO") => LevelFilter::Info,
        Some("WARNING" | "WARN") => LevelFilter::Warn,
        Some("ERROR" | "CRITICAL" | "FATAL") => LevelFilter::Error,
        _ => default,
    }
}

/// Configure the logger. Safe to call again (previous sinks are replaced).
///
/// * `level`: `DEBUG`, `INFO`, `WARNING` or `ERROR` (case-insensitive; anything else means
///   `INFO`).
/// * `console`: also log to stderr.
/// * `log_to_file`: write to the rotating log file (`paths::log_file()`, 1 MB × 3 backups).
///   Command-line tools that only report to the terminal pass `false`.
pub fn setup_logging(level: &str, mut console: bool, log_to_file: bool) {
    let numeric = parse_level(Some(level), LevelFilter::Info);
    let mut file = None;
    if log_to_file {
        match open_log_file() {
            Ok(writer) => file = Some(writer),
            Err(err) => {
                // A read-only profile must not stop the app; fall back to stderr.
                console = true;
                eprintln!("Eye Tracker: cannot write the log file ({err})");
            }
        }
    }

    let mut guard = STATE.lock();
    let state = guard.get_or_insert_with(|| start_logger(numeric));
    *state.sinks.write() = Sinks { file, console };
    apply_level(state, numeric);
    HOOK.call_once(install_panic_hook);
}

/// Change the verbosity after [`setup_logging`] (e.g. once settings are loaded).
pub fn set_level(level: &str) {
    if let Some(state) = STATE.lock().as_ref() {
        apply_level(state, parse_level(Some(level), LevelFilter::Info));
    }
}

/// Flush and remove the sinks installed by [`setup_logging`].
///
/// Used at exit and by tests (an open log file would keep a temporary directory locked on
/// Windows).
pub fn shutdown_logging() {
    if let Some(state) = STATE.lock().as_ref() {
        let mut sinks = state.sinks.write();
        if let Some(file) = &sinks.file {
            let _ = file.flush();
        }
        *sinks = Sinks::default();
    }
}

fn start_logger(level: LevelFilter) -> State {
    let sinks = Arc::new(RwLock::new(Sinks::default()));
    let started = Logger::with(LogSpecification::builder().default(level).build())
        .log_to_writer(Box::new(SwitchWriter(sinks.clone())))
        .start();
    let handle = match started {
        Ok(handle) => Some(handle),
        Err(err) => {
            // Another logger is already installed (a test harness, an embedding host).
            eprintln!("Eye Tracker: cannot start logging ({err})");
            None
        }
    };
    State { handle, sinks }
}

fn apply_level(state: &State, level: LevelFilter) {
    if let Some(handle) = &state.handle {
        handle.set_new_spec(LogSpecification::builder().default(level).build());
    }
}

fn open_log_file() -> Result<FileLogWriter, flexi_logger::FlexiLoggerError> {
    let spec = FileSpec::default()
        .directory(paths::log_dir())
        .basename(APP_SLUG)
        .suffix("log")
        .suppress_timestamp();
    // An empty current infix keeps the live file named `eye-tracker.log` (`paths::log_file()`);
    // only the rotated backups get a timestamp.
    FileLogWriter::builder(spec)
        .format(format_line)
        .append()
        .rotate(
            Criterion::Size(MAX_BYTES),
            Naming::TimestampsCustomFormat {
                current_infix: Some(""),
                format: "r%Y-%m-%d_%H-%M-%S",
            },
            Cleanup::KeepLogFiles(BACKUP_COUNT),
        )
        .try_build()
}

/// `2026-01-31 12:00:00,123 WARNING eye_tracker.config: message`
fn format_line(w: &mut dyn Write, now: &mut DeferredNow, record: &Record) -> io::Result<()> {
    let target = record.target();
    let level = if target == CRASH_TARGET {
        "CRITICAL"
    } else {
        match record.level() {
            log::Level::Error => "ERROR",
            log::Level::Warn => "WARNING",
            log::Level::Info => "INFO",
            log::Level::Debug | log::Level::Trace => "DEBUG",
        }
    };
    writeln!(
        w,
        "{} {level:<7} {}: {}",
        now.format("%Y-%m-%d %H:%M:%S,%3f"),
        target.replace("::", "."),
        record.args()
    )
}

/// Log panics; without a console they would otherwise vanish.
fn install_panic_hook() {
    let previous = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        let thread = std::thread::current();
        let place = match thread.name() {
            Some("main") | None => String::new(),
            Some(name) => format!(" in thread {name}"),
        };
        let payload = info.payload();
        let message = payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("(non-text panic payload)");
        let location = info
            .location()
            .map_or_else(String::new, |l| format!(" at {l}"));
        let backtrace = std::backtrace::Backtrace::force_capture();
        log::error!(target: CRASH_TARGET, "Unhandled exception{place}: {message}{location}\n{backtrace}");
        previous(info);
    }));
}

#[cfg(test)]
#[path = "logging_setup_tests.rs"]
mod tests;
