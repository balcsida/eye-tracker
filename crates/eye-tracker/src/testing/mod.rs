//! Shared test helpers (ports of tests/conftest.py, gaze_synth.py, face_drawing.py and
//! panes_fakes.py as they are needed).

use std::path::Path;

use parking_lot::{Mutex, MutexGuard};

use crate::paths;

/// `paths::set_base_override` is process-wide, so tests that use it must not overlap.
static PATHS_LOCK: Mutex<()> = Mutex::new(());

/// Holds the lock on the process-wide paths override and clears the override when dropped
/// (the `_reset_override` fixture of test_paths.py).
pub struct PathsGuard {
    _lock: MutexGuard<'static, ()>,
}

pub fn lock_paths() -> PathsGuard {
    PathsGuard {
        _lock: PATHS_LOCK.lock(),
    }
}

impl Drop for PathsGuard {
    fn drop(&mut self) {
        paths::set_base_override(None);
    }
}

/// Redirects config, data and logs into a temporary directory (the `app_dirs` fixture).
pub struct AppDirs {
    // Field order matters: the override is reset and the lock released (guard) before the
    // directory is removed.
    _guard: PathsGuard,
    dir: tempfile::TempDir,
}

pub fn app_dirs() -> AppDirs {
    let guard = lock_paths();
    let dir = tempfile::tempdir().expect("temporary directory");
    paths::set_base_override(Some(dir.path()));
    AppDirs { _guard: guard, dir }
}

impl AppDirs {
    pub fn path(&self) -> &Path {
        self.dir.path()
    }
}

/// Log records of the current thread (pytest's `caplog`). Records of other threads, and of tests
/// running in parallel, are not mixed in.
pub struct LogCapture;

struct CaptureLogger;

static CAPTURED: Mutex<Vec<(std::thread::ThreadId, log::Level, String)>> = Mutex::new(Vec::new());

impl log::Log for CaptureLogger {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }

    fn log(&self, record: &log::Record) {
        CAPTURED.lock().push((
            std::thread::current().id(),
            record.level(),
            record.args().to_string(),
        ));
    }

    fn flush(&self) {}
}

/// Start capturing (installs the capturing logger on first use) and clear this thread's records.
pub fn capture_logs() -> LogCapture {
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| {
        let _ = log::set_logger(&CaptureLogger);
        log::set_max_level(log::LevelFilter::Trace);
    });
    LogCapture.clear();
    LogCapture
}

impl LogCapture {
    pub fn records(&self) -> Vec<(log::Level, String)> {
        let me = std::thread::current().id();
        CAPTURED
            .lock()
            .iter()
            .filter(|r| r.0 == me)
            .map(|r| (r.1, r.2.clone()))
            .collect()
    }

    /// Messages of at least `level` severity (`Warn` includes `Error`).
    pub fn messages_at_least(&self, level: log::Level) -> Vec<String> {
        self.records()
            .into_iter()
            .filter(|(l, _)| *l <= level)
            .map(|(_, m)| m)
            .collect()
    }

    fn clear(&self) {
        let me = std::thread::current().id();
        CAPTURED.lock().retain(|r| r.0 != me);
    }
}

impl Drop for LogCapture {
    fn drop(&mut self) {
        self.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_dirs_redirects_and_resets() {
        let kept;
        {
            let dirs = app_dirs();
            kept = dirs.path().to_path_buf();
            assert_eq!(paths::config_dir(), paths::base_override().unwrap());
            assert!(paths::log_file().starts_with(dirs.path().canonicalize().unwrap()));
        }
        let _guard = lock_paths();
        assert_eq!(paths::base_override(), None);
        assert!(!kept.exists());
    }
}
