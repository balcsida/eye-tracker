//! Eye Tracker: look at a monitor and your cursor and keyboard focus follow.

pub const APP_NAME: &str = "Eye Tracker";
pub const APP_SLUG: &str = "eye-tracker";
pub const APP_ID: &str = "io.github.bugraskl.eyetracker";
pub const APP_AUTHOR: &str = "bugraskl";
pub const REPO_URL: &str = "https://github.com/bugraskl/eye-tracker";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod app;
pub mod cli;
pub mod config;
pub mod diagnostics;
pub mod engine;
pub mod gaze;
pub mod ipc;
pub mod logging_setup;
pub mod panes;
pub mod paths;
pub mod platform;
pub mod trace;
pub mod types;
pub mod ui;
pub mod update;
pub mod vision;

#[cfg(test)]
mod testing;
