//! Filesystem locations (config, data, logs).
//!
//! The directories are the ones the Python app used (`platformdirs` with the app author and
//! `roaming=False`), so an upgrade keeps its settings, calibration and logs. The bundled models
//! are embedded in the binary (see `vision::backends`), so there is no resource directory.

use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::RwLock;

use crate::APP_SLUG;

static OVERRIDE: RwLock<Option<PathBuf>> = RwLock::new(None);

/// Put config, data and logs under one directory (`--config-dir`, tests).
///
/// The path is expanded (`~`) and resolved to an absolute path like Python's
/// `Path.expanduser().resolve()`.
pub fn set_base_override(path: Option<&Path>) {
    let resolved = path.map(|p| resolve(&expand_user(p)));
    *OVERRIDE.write().unwrap_or_else(|e| e.into_inner()) = resolved;
}

/// The directory set by [`set_base_override`] (resolved), or `None`.
///
/// `None` means the default per-user profile. Autostart entries and the diagnostics report use
/// it to tell which profile this process runs.
pub fn base_override() -> Option<PathBuf> {
    OVERRIDE.read().unwrap_or_else(|e| e.into_inner()).clone()
}

pub fn config_dir() -> PathBuf {
    ensure(base_override().unwrap_or_else(default_config_dir))
}

pub fn data_dir() -> PathBuf {
    ensure(base_override().map_or_else(default_data_dir, |base| base.join("data")))
}

pub fn log_dir() -> PathBuf {
    ensure(base_override().map_or_else(default_log_dir, |base| base.join("logs")))
}

pub fn settings_file() -> PathBuf {
    config_dir().join("settings.json")
}

pub fn calibration_file() -> PathBuf {
    data_dir().join("calibration.json")
}

/// What the app remembers between runs that is no setting (privacy mode).
pub fn state_file() -> PathBuf {
    data_dir().join("state.json")
}

pub fn log_file() -> PathBuf {
    log_dir().join(format!("{APP_SLUG}.log"))
}

/// Per-user name for the single-instance local socket.
pub fn ipc_name() -> String {
    let user = ["LOGNAME", "USER", "LNAME", "USERNAME"]
        .iter()
        .find_map(|name| std::env::var(name).ok().filter(|v| !v.is_empty()))
        .unwrap_or_else(|| "user".to_string());
    let scope = match base_override() {
        Some(base) => format!("{user}|{}", base.display()),
        None => user,
    };
    format!("{APP_SLUG}-{}", &sha1_hex(scope.as_bytes())[..10])
}

/// Create the directory (a failure is logged; the caller then fails on its own write).
fn ensure(path: PathBuf) -> PathBuf {
    if let Err(err) = fs::create_dir_all(&path) {
        log::warn!("Could not create {}: {err}", path.display());
    }
    path
}

fn home() -> PathBuf {
    directories::BaseDirs::new().map_or_else(|| PathBuf::from("."), |d| d.home_dir().to_path_buf())
}

#[cfg(target_os = "windows")]
fn windows_base() -> PathBuf {
    directories::BaseDirs::new()
        .map_or_else(home, |d| d.data_local_dir().to_path_buf())
        .join(crate::APP_AUTHOR)
        .join(APP_SLUG)
}

#[cfg(target_os = "windows")]
fn default_config_dir() -> PathBuf {
    windows_base()
}

#[cfg(target_os = "windows")]
fn default_data_dir() -> PathBuf {
    windows_base()
}

#[cfg(target_os = "windows")]
fn default_log_dir() -> PathBuf {
    windows_base().join("Logs")
}

#[cfg(target_os = "macos")]
fn default_config_dir() -> PathBuf {
    home().join("Library/Application Support").join(APP_SLUG)
}

#[cfg(target_os = "macos")]
fn default_data_dir() -> PathBuf {
    default_config_dir()
}

#[cfg(target_os = "macos")]
fn default_log_dir() -> PathBuf {
    home().join("Library/Logs").join(APP_SLUG)
}

/// The XDG directory named by `var`, or `fallback` below the home directory.
#[cfg(not(any(target_os = "windows", target_os = "macos")))]
fn xdg(var: &str, fallback: &str) -> PathBuf {
    match std::env::var_os(var) {
        Some(v) if Path::new(&v).is_absolute() => PathBuf::from(v),
        _ => home().join(fallback),
    }
}

#[cfg(not(any(target_os = "windows", target_os = "macos")))]
fn default_config_dir() -> PathBuf {
    xdg("XDG_CONFIG_HOME", ".config").join(APP_SLUG)
}

#[cfg(not(any(target_os = "windows", target_os = "macos")))]
fn default_data_dir() -> PathBuf {
    xdg("XDG_DATA_HOME", ".local/share").join(APP_SLUG)
}

#[cfg(not(any(target_os = "windows", target_os = "macos")))]
fn default_log_dir() -> PathBuf {
    xdg("XDG_STATE_HOME", ".local/state")
        .join(APP_SLUG)
        .join("log")
}

fn expand_user(path: &Path) -> PathBuf {
    match path.strip_prefix("~") {
        Ok(rest) => home().join(rest),
        Err(_) => path.to_path_buf(),
    }
}

/// Absolute, symlink-free and `..`-free path; the part that does not exist yet is kept as given
/// (Python's non-strict `Path.resolve`).
fn resolve(path: &Path) -> PathBuf {
    let absolute = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    let mut existing = PathBuf::new();
    let mut rest: Vec<Component> = Vec::new();
    let mut components = absolute.components();
    for c in components.by_ref() {
        match c {
            Component::ParentDir => {
                // Resolve what exists first, so a symlink is followed before `..` applies.
                existing = fs::canonicalize(&existing).unwrap_or(existing);
                existing.pop();
            }
            Component::CurDir => {}
            c => {
                existing.push(c);
                if !existing.exists() {
                    existing.pop();
                    rest.push(c);
                    break;
                }
            }
        }
    }
    let mut out = fs::canonicalize(&existing).unwrap_or(existing);
    // Whatever is left cannot exist, so only `..` and `.` need handling.
    for c in rest.into_iter().chain(components) {
        match c {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            c => out.push(c),
        }
    }
    out
}

/// SHA-1 as lowercase hex. Only used for non-secret fingerprints that must stay equal to the
/// Python app's (`layout_signature`, `ipc_name`).
pub(crate) fn sha1_hex(data: &[u8]) -> String {
    use sha1::{Digest, Sha1};
    hex::encode(Sha1::digest(data))
}

#[cfg(test)]
#[path = "paths_tests.rs"]
mod tests;
