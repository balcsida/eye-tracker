//! Side-by-side sessions of desktop apps built on Chromium, through UI Automation.

// TODO(port): src/eye_tracker/panes/providers/chromium_apps.py

use crate::types::AppIdentity;

/// Name of the provider (`Pane::provider`, logs, `status`).
pub const PROVIDER_NAME: &str = "desktop_apps";

/// How to find the side-by-side sessions of one desktop app (the part `registry` needs).
// TODO(port): the rest of the Python `AppProfile` fields.
#[derive(Debug, PartialEq, Eq)]
pub struct AppProfile {
    /// Short name used in logs.
    pub name: &'static str,
    /// Process names (`platform::base::process_basename`) of the app.
    pub processes: &'static [&'static str],
    /// Class of the app's top-level windows.
    pub window_class: &'static str,
}

impl AppProfile {
    pub fn matches(&self, app: &AppIdentity) -> bool {
        self.processes
            .contains(&app.process.trim().to_lowercase().as_str())
            && app.app_id == self.window_class
    }
}

/// The Claude desktop app (Windows).
pub const CLAUDE: AppProfile = AppProfile {
    name: "claude",
    processes: &["claude"],
    window_class: "Chrome_WidgetWin_1",
};

/// The ChatGPT desktop app (Windows), which also hosts Codex.
pub const CHATGPT: AppProfile = AppProfile {
    name: "chatgpt",
    processes: &["chatgpt"],
    window_class: "Chrome_WidgetWin_1",
};

/// Every app with a profile; only these ever skip the registry's deny-list.
pub const PROFILES: [&AppProfile; 2] = [&CLAUDE, &CHATGPT];

/// The profile of `app` (process name and window class), if it has one.
pub fn profile_for(app: Option<&AppIdentity>) -> Option<&'static AppProfile> {
    let app = app?;
    PROFILES.into_iter().find(|p| p.matches(app))
}

#[cfg(windows)]
use std::sync::Arc;

#[cfg(windows)]
use super::super::types::{ClientRect, PaneProvider};
#[cfg(windows)]
use super::command::CommandRunner;

/// The provider for `registry::default_providers`.
// TODO(port): construct the provider; until then the desktop apps' sessions are not available.
#[cfg(windows)]
pub fn create(
    _client_rect: ClientRect,
    _runner: Arc<dyn CommandRunner>,
) -> Option<Box<dyn PaneProvider>> {
    None
}
