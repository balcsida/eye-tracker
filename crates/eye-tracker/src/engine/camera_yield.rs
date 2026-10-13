//! When to let go of the camera so other apps can use it.
//!
//! Most webcams can only be opened by one process at a time. Tracking pauses and
//! releases the camera while another app is streaming from it (video calls) or
//! while an app from the user's list is running (screen recorders, games).

use std::collections::HashSet;

const SUFFIXES: [&str; 2] = [".exe", ".app"];
/// Windows Explorer's "Copy as path" wraps the path in double quotes; shells and
/// users sometimes use single quotes.
const QUOTES: [char; 2] = ['"', '\''];

fn unquote(text: &str) -> &str {
    text.trim().trim_matches(QUOTES).trim()
}

/// Facts gathered by the controller for [`should_yield`].
#[derive(Debug, Clone, Default)]
pub struct YieldInputs {
    /// Whether another process streams from a camera (`None` = cannot tell).
    pub camera_in_use_by_other: Option<bool>,
    /// Names of running processes (as from `PlatformServices::running_process_names`).
    pub running: HashSet<String>,
    /// User-configured process names that pause tracking while they run.
    pub pause_for_apps: Vec<String>,
}

/// Canonical form for matching: lower case, no quotes, no directory, no `.exe`/`.app`.
///
/// Accepts what users paste, e.g. `"C:\Program Files\obs-studio\bin\64bit\obs64.exe"`
/// (with the quotes Explorer's "Copy as path" adds) or `/Applications/zoom.us.app`.
pub fn normalize_process_name(name: &str) -> String {
    let last = unquote(name).rsplit(['\\', '/']).next().unwrap_or("");
    let base = unquote(last).to_lowercase();
    for suffix in SUFFIXES {
        if base.len() > suffix.len()
            && let Some(stem) = base.strip_suffix(suffix)
        {
            return stem.to_string();
        }
    }
    base
}

/// First entry of `pause_for_apps` (as the user wrote it) that is running.
pub fn matching_app<S: AsRef<str>>(
    running: impl IntoIterator<Item = S>,
    pause_for_apps: &[String],
) -> Option<&str> {
    let names: HashSet<String> = running
        .into_iter()
        .filter(|n| !n.as_ref().is_empty())
        .map(|n| normalize_process_name(n.as_ref()))
        .filter(|n| !n.is_empty())
        .collect();
    pause_for_apps
        .iter()
        .find(|entry| {
            let wanted = normalize_process_name(entry);
            !wanted.is_empty() && names.contains(&wanted)
        })
        .map(|entry| unquote(entry))
}

/// Decide whether tracking should release the camera.
///
/// Returns `(true, reason)` - e.g. `(true, "zoom.exe is running")` - or
/// `(false, "")`. Listed apps always pause tracking; a camera used by another
/// app only does when `yield_camera` is enabled.
pub fn should_yield(inputs: &YieldInputs, yield_camera: bool) -> (bool, String) {
    if let Some(app) = matching_app(&inputs.running, &inputs.pause_for_apps) {
        return (true, format!("{app} is running"));
    }
    if yield_camera && inputs.camera_in_use_by_other == Some(true) {
        return (true, "Another app is using the camera".to_string());
    }
    (false, String::new())
}

#[cfg(test)]
#[path = "camera_yield_tests.rs"]
mod tests;
