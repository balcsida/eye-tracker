//! Windows Terminal panes, through UI Automation (Windows only).

// TODO(port): src/eye_tracker/panes/providers/windows_terminal.py

#[cfg(windows)]
use std::sync::Arc;

#[cfg(windows)]
use super::super::types::{ClientRect, PaneProvider};
#[cfg(windows)]
use super::command::CommandRunner;

/// The provider for `registry::default_providers`.
// TODO(port): construct the provider; until then Windows Terminal panes are not available.
#[cfg(windows)]
pub fn create(
    _client_rect: ClientRect,
    _runner: Arc<dyn CommandRunner>,
) -> Option<Box<dyn PaneProvider>> {
    None
}
