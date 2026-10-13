//! The tray application: startup sequence, single instance and window management.

// TODO(port): src/eye_tracker/app.py. Until then only the entry point the CLI calls exists.

use std::path::PathBuf;

/// Command-line switches that change how the app starts (see `cli::run_args`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RunArgs {
    pub config_dir: Option<PathBuf>,
    pub log_level: Option<String>,
    pub camera: Option<String>,
    pub backend: Option<String>,
    pub background: bool,
    pub calibrate: bool,
    pub trace: Option<PathBuf>,
}

/// Start the tray app and run the event loop. Returns the exit code.
pub fn run_app(_args: &RunArgs) -> i32 {
    eprintln!("the tray app is not ported yet");
    crate::cli::EXIT_ERROR
}
