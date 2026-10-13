//! Pane providers: one module per terminal or multiplexer (see `registry`).

// TODO(port): src/eye_tracker/panes/providers/__init__.py

pub mod chromium_apps;
pub mod command;
pub mod tmux;
pub mod wezterm;
pub mod windows_terminal;
