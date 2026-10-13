//! Which pane providers look at which windows.
//!
//! Before any provider sees a window, [`is_denied`] checks it against a hard
//! deny-list of Electron and Chromium applications (editors, browsers, chat apps).
//! Asking those about their contents through accessibility interfaces can switch
//! them into a slower screen-reader mode, so they are never inspected at all, not
//! even by a provider's [`PaneProvider::applies`]. Editors such as VS
//! Code get their own, purpose-built support later instead.
//!
//! The one exception is opt-in: with `panes.desktop_apps` on, the desktop apps
//! that have a profile in `providers::chromium_apps` (matched by process
//! name *and* window class; the Claude and ChatGPT apps) skip the deny-list and go
//! to that provider alone. Every other denied app stays denied.

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use crate::config::PaneSettings;
use crate::types::AppIdentity;

use super::providers::chromium_apps::{PROVIDER_NAME as DESKTOP_APPS_PROVIDER, profile_for};
use super::providers::command::{CommandRunner, run_command};
use super::providers::tmux::TmuxProvider;
use super::providers::wezterm::WezTermProvider;
use super::types::{ClientRect, PaneProvider};

/// Process names (lower-cased, without extension) that are never inspected.
pub const DENIED_PROCESSES: [&str; 19] = [
    "code",
    "code - insiders",
    "cursor",
    "claude",
    "chatgpt",
    "antigravity",
    "windsurf",
    "chrome",
    "google chrome",
    "chromium",
    "msedge",
    "microsoft edge",
    "firefox",
    "slack",
    "teams",
    "ms-teams",
    "discord",
    "obsidian",
    "electron",
];

/// Window classes (Windows), WM_CLASS classes (X11) and bundle ids (macOS) that
/// are never inspected. `Chrome_WidgetWin_1` is the top-level window class of
/// every Chromium and Electron application on Windows.
pub const DENIED_APP_IDS: [&str; 14] = [
    "Chrome_WidgetWin_1",
    "com.microsoft.VSCode",
    "com.microsoft.VSCodeInsiders",
    "com.todesktop.230313mzl4w4u92", // Cursor
    "com.anthropic.claudefordesktop",
    "com.openai.chat",
    "com.google.Chrome",
    "com.microsoft.edgemac",
    "org.mozilla.firefox",
    "com.tinyspeck.slackmacgap",
    "com.microsoft.teams2",
    "com.hnc.Discord",
    "md.obsidian",
    "com.github.Electron",
];

/// Whether windows of `app` must never be inspected for panes.
///
/// `desktop_apps` is the `panes.desktop_apps` setting: when it is on, an app with a desktop-app
/// profile is not denied (and only the desktop-app provider is asked about it, see
/// [`PaneRegistry::providers_for`]).
pub fn is_denied(app: &AppIdentity, desktop_apps: bool) -> bool {
    if desktop_apps && profile_for(Some(app)).is_some() {
        return false;
    }
    DENIED_PROCESSES.contains(&app.process.trim().to_lowercase().as_str())
        || DENIED_APP_IDS.contains(&app.app_id.as_str())
}

/// What a panic carried, for the log (Python logged the exception).
pub(crate) fn panic_message(payload: &(dyn Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "panic".to_string())
}

/// The enabled providers, in order of preference.
///
/// `desktop_apps` is the `panes.desktop_apps` setting (see [`is_denied`]).
#[derive(Default)]
pub struct PaneRegistry {
    providers: Vec<Box<dyn PaneProvider>>,
    desktop_apps: bool,
}

impl PaneRegistry {
    pub fn new(providers: Vec<Box<dyn PaneProvider>>) -> Self {
        Self {
            providers,
            desktop_apps: false,
        }
    }

    pub fn with_desktop_apps(mut self, desktop_apps: bool) -> Self {
        self.desktop_apps = desktop_apps;
        self
    }

    pub fn providers(&self) -> &[Box<dyn PaneProvider>] {
        &self.providers
    }

    /// The providers that may know panes of `app`; none for a denied app.
    ///
    /// The deny-list is checked before any provider is asked anything. An app with a
    /// desktop-app profile, when `desktop_apps` is on, goes to the desktop-app provider and to no
    /// other; no other app goes to that one.
    pub fn providers_for(&mut self, app: Option<&AppIdentity>) -> Vec<&mut dyn PaneProvider> {
        let Some(app) = app else {
            return Vec::new();
        };
        let desktop = if self.desktop_apps && profile_for(Some(app)).is_some() {
            true
        } else if is_denied(app, false) {
            return Vec::new();
        } else {
            false
        };
        let mut found: Vec<&mut dyn PaneProvider> = Vec::new();
        for provider in &mut self.providers {
            if (provider.name() == DESKTOP_APPS_PROVIDER) != desktop {
                continue;
            }
            match catch_unwind(AssertUnwindSafe(|| provider.applies(app))) {
                Ok(true) => found.push(provider.as_mut()),
                Ok(false) => {}
                Err(payload) => log::debug!(
                    "Pane provider {} failed in applies(): {}",
                    provider.name(),
                    panic_message(payload.as_ref())
                ),
            }
        }
        found
    }

    pub fn provider(&mut self, name: &str) -> Option<&mut dyn PaneProvider> {
        for provider in &mut self.providers {
            if provider.name() == name {
                return Some(provider.as_mut());
            }
        }
        None
    }

    /// Call `close()` of every provider (failures are logged).
    ///
    /// Called on the pane worker's thread (see `worker`).
    pub fn close(&mut self) {
        for provider in &mut self.providers {
            if let Err(payload) = catch_unwind(AssertUnwindSafe(|| provider.close())) {
                log::debug!(
                    "Closing pane provider {} failed: {}",
                    provider.name(),
                    panic_message(payload.as_ref())
                );
            }
        }
    }
}

/// Python's `sys.platform` for the running system ("win32", "darwin", "linux").
pub fn current_system() -> &'static str {
    if cfg!(windows) {
        "win32"
    } else if cfg!(target_os = "macos") {
        "darwin"
    } else {
        std::env::consts::OS
    }
}

/// An optional provider implemented in its own module. One that the platform does not have is
/// skipped.
struct OptionalSlot {
    /// Name of the provider (and of the `PaneSettings` field that enables it).
    name: &'static str,
    enabled: fn(&PaneSettings) -> bool,
    /// `sys.platform` values it runs on.
    platforms: &'static [&'static str],
}

const OPTIONAL_PROVIDERS: [OptionalSlot; 2] = [
    OptionalSlot {
        name: "windows_terminal",
        enabled: |s| s.windows_terminal,
        platforms: &["win32"],
    },
    OptionalSlot {
        name: "desktop_apps",
        enabled: |s| s.desktop_apps,
        platforms: &["win32"],
    },
];

/// Creates an optional provider by its slot name (see [`OPTIONAL_PROVIDERS`]); `None` if it is
/// not available.
type OptionalProvider<'a> =
    dyn Fn(&str, &ClientRect, &Arc<dyn CommandRunner>) -> Option<Box<dyn PaneProvider>> + 'a;

/// The providers the settings enable for `system`, most specific first.
///
/// `client_rect` is `PlatformServices::window_client_rect` (panes are mapped onto the content
/// area); `runner` runs the terminals' command-line tools (tests pass a fake; `None` runs the
/// real ones). `system` is a Python `sys.platform` value, see [`current_system`].
pub fn default_providers(
    settings: &PaneSettings,
    client_rect: ClientRect,
    runner: Option<Arc<dyn CommandRunner>>,
    system: &str,
) -> Vec<Box<dyn PaneProvider>> {
    default_providers_with(settings, client_rect, runner, system, &optional_provider)
}

fn default_providers_with(
    settings: &PaneSettings,
    client_rect: ClientRect,
    runner: Option<Arc<dyn CommandRunner>>,
    system: &str,
    optional: &OptionalProvider,
) -> Vec<Box<dyn PaneProvider>> {
    let run = runner.unwrap_or_else(|| Arc::new(run_command));
    let mut providers: Vec<Box<dyn PaneProvider>> = Vec::new();
    if settings.wezterm {
        // Before tmux: inside WezTerm its own panes are what the user sees split.
        providers.push(Box::new(WezTermProvider::new(
            client_rect.clone(),
            run.clone(),
        )));
    }
    // Before tmux too: with two or more Windows Terminal panes, those are the split
    // the user sees; with one, tmux running in it gets its turn.
    for slot in OPTIONAL_PROVIDERS {
        if !(slot.enabled)(settings) || !slot.platforms.contains(&system) {
            continue;
        }
        if let Some(provider) = optional(slot.name, &client_rect, &run) {
            providers.push(provider);
        }
    }
    if settings.tmux {
        providers.push(Box::new(TmuxProvider::new(client_rect, run)));
    }
    providers
}

#[cfg_attr(not(windows), allow(unused_variables))]
fn optional_provider(
    name: &str,
    client_rect: &ClientRect,
    runner: &Arc<dyn CommandRunner>,
) -> Option<Box<dyn PaneProvider>> {
    match name {
        #[cfg(windows)]
        "windows_terminal" => {
            super::providers::windows_terminal::create(client_rect.clone(), runner.clone())
        }
        #[cfg(windows)]
        "desktop_apps" => {
            super::providers::chromium_apps::create(client_rect.clone(), runner.clone())
        }
        _ => {
            log::debug!("Pane provider {name} not available");
            None
        }
    }
}

#[cfg(test)]
#[path = "registry_tests.rs"]
mod tests;
