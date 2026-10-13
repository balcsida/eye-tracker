//! Operating-system integration.

// TODO(port): src/eye_tracker/platform/__init__.py

pub mod autostart;
pub mod base;
pub mod hotkeys;
#[cfg(target_os = "linux")]
pub mod linux;
#[cfg(target_os = "macos")]
pub mod macos;
#[cfg(target_os = "windows")]
pub mod win_uia;
#[cfg(target_os = "windows")]
pub mod windows;
