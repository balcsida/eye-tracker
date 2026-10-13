//! Operating-system integration.
//!
//! Use [`get_platform`] to obtain the implementation for the running OS. It never fails: if the
//! OS-specific implementation cannot be created (an unknown OS) the do-nothing
//! [`GenericPlatform`] is returned and the app keeps working with reduced features.

use std::sync::Arc;

use parking_lot::Mutex;

pub mod autostart;
pub mod base;
pub mod hotkeys;
#[cfg(target_os = "linux")]
pub mod linux;
#[cfg(target_os = "macos")]
pub mod macos;
pub mod win_uia;
// Not cfg-gated: only its Win32 binding is, so the policy tests run on every OS.
pub mod windows;

pub use base::{GenericPlatform, PlatformServices};

use crate::types::{Monitor, Rect};

static CACHE: Mutex<Option<Arc<dyn PlatformServices>>> = Mutex::new(None);

/// The process-wide platform services singleton (created on first call).
pub fn get_platform() -> Arc<dyn PlatformServices> {
    let mut cache = CACHE.lock();
    if let Some(instance) = cache.as_ref() {
        return instance.clone();
    }
    let instance = create(std::env::consts::OS);
    log::debug!("Platform services: {}", instance.name());
    *cache = Some(instance.clone());
    instance
}

/// The implementation for `system` (a `std::env::consts::OS` value), or the do-nothing one.
fn create(system: &str) -> Arc<dyn PlatformServices> {
    #[cfg(target_os = "linux")]
    if system == "linux" {
        return match std::panic::catch_unwind(linux::LinuxPlatform::new) {
            Ok(platform) => Arc::new(platform),
            Err(panic) => {
                let err = panic
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_default();
                log::warn!("{system} integration unavailable ({err}); OS features are disabled");
                Arc::new(GenericPlatform)
            }
        };
    }
    #[cfg(target_os = "macos")]
    if system == "macos" {
        // `MacPlatform::new` cannot fail: a missing framework only degrades the call that
        // needs it, so there is no "integration unavailable" fallback to take.
        return Arc::new(macos::MacPlatform::new());
    }
    if system == "windows" {
        // Never fails: the Win32 binding is created (and its failure logged) on first use.
        return Arc::new(windows::WindowsPlatform::new());
    }
    log::info!("No OS integration for {system:?}; OS features are disabled");
    Arc::new(GenericPlatform)
}

/// The name a screen that has none gets (by position).
pub fn fallback_screen_name(index: i32) -> String {
    format!("Screen {}", index + 1)
}

/// One display as `display-info` reports it.
#[derive(Debug, Clone, PartialEq)]
struct Display {
    name: String,
    x: i32,
    y: i32,
    width: u32,
    height: u32,
    scale: f32,
    primary: bool,
}

/// The monitors of this desktop, primary first, then in the order the system lists them,
/// indexed from 0 (what `QGuiApplication.screens()` gave the Python app).
///
/// Rectangles are in the coordinates of [`crate::types`]: native pixels on Windows and X11,
/// Cocoa points on macOS. Empty when the displays cannot be read (no display server).
pub fn monitors() -> Vec<Monitor> {
    let displays = match display_info::DisplayInfo::all() {
        Ok(displays) => displays,
        Err(e) => {
            log::debug!("Reading the monitor layout failed: {e}");
            return Vec::new();
        }
    };
    let displays = displays
        .into_iter()
        .map(|d| Display {
            name: d.name,
            x: d.x,
            y: d.y,
            width: d.width,
            height: d.height,
            scale: d.scale_factor,
            primary: d.is_primary,
        })
        .collect();
    to_monitors(displays, cfg!(target_os = "linux"))
}

/// [`monitors`] from the displays. `logical` says the crate divided the positions and sizes
/// by the scale factor (X11), which is undone: the rest of the app works in native pixels.
fn to_monitors(mut displays: Vec<Display>, logical: bool) -> Vec<Monitor> {
    // Stable: the primary first, the others in the system's order.
    displays.sort_by_key(|d| !d.primary);
    let mut monitors = Vec::new();
    for d in displays {
        let scale = if d.scale.is_finite() && d.scale > 0.0 {
            f64::from(d.scale)
        } else {
            1.0
        };
        let native = |v: f64| if logical { (v * scale).round() } else { v } as i32;
        let rect = Rect::new(
            native(f64::from(d.x)),
            native(f64::from(d.y)),
            native(f64::from(d.width)),
            native(f64::from(d.height)),
        );
        if rect.w <= 0 || rect.h <= 0 {
            continue;
        }
        let index = monitors.len() as i32;
        let name = if d.name.trim().is_empty() {
            fallback_screen_name(index)
        } else {
            d.name
        };
        monitors.push(Monitor {
            index,
            name,
            rect,
            primary: d.primary,
            scale,
        });
    }
    monitors
}

/// Forget the cached singleton (tests only).
#[cfg(test)]
pub(crate) fn reset_for_tests() {
    *CACHE.lock() = None;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn display(name: &str, x: i32, width: u32, primary: bool) -> Display {
        Display {
            name: name.into(),
            x,
            y: 0,
            width,
            height: 1080,
            scale: 1.0,
            primary,
        }
    }

    #[test]
    fn test_qt_monitors_offscreen() {
        let monitors = to_monitors(
            vec![
                display("DP-2", 1920, 1920, false),
                display("", 3840, 1920, false),
                display("DP-1", 0, 1920, true),
                display("gone", 0, 0, false),
            ],
            false,
        );
        let indices: Vec<i32> = monitors.iter().map(|m| m.index).collect();
        assert_eq!(indices, [0, 1, 2]);
        assert!(monitors.iter().all(|m| m.rect.w > 0 && m.rect.h > 0));
        let names: Vec<&str> = monitors.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, ["DP-1", "DP-2", "Screen 3"]); // primary first
        assert!(monitors[0].primary);
    }

    #[test]
    fn x11_monitors_are_in_native_pixels() {
        let mut scaled = display("eDP-1", 1280, 1280, true);
        scaled.height = 720;
        scaled.scale = 1.5;
        let monitors = to_monitors(vec![scaled], true);
        assert_eq!(monitors[0].rect, Rect::new(1920, 0, 1920, 1080));
        assert_eq!(monitors[0].scale, 1.5);
    }

    #[test]
    fn get_platform_is_a_singleton_until_reset() {
        let first = get_platform();
        assert!(Arc::ptr_eq(&first, &get_platform()));
        reset_for_tests();
        assert!(!Arc::ptr_eq(&first, &get_platform()));
    }
}
