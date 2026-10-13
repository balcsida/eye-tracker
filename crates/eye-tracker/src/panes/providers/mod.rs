//! Pane providers: one module per terminal or multiplexer (see `registry`).

pub mod chromium_apps;
pub mod command;
pub mod tmux;
pub mod wezterm;
pub mod windows_terminal;
