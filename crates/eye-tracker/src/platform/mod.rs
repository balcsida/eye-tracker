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
#[cfg(target_os = "windows")]
pub mod win_uia;
// Not cfg-gated: only its Win32 binding is, so the policy tests run on every OS.
pub mod windows;

pub use base::{GenericPlatform, PlatformServices};

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
    // TODO(port): "windows" => windows::WindowsPlatform and "linux" => linux::LinuxPlatform, each
    // with the Python's fallback: a constructor error logs
    // "{system} integration unavailable ({err}); OS features are disabled" at WARN and returns
    // GenericPlatform. Add the arms when those modules are ported.
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

/// Forget the cached singleton (tests only).
#[cfg(test)]
pub(crate) fn reset_for_tests() {
    *CACHE.lock() = None;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_platform_is_a_singleton_until_reset() {
        let first = get_platform();
        assert!(Arc::ptr_eq(&first, &get_platform()));
        reset_for_tests();
        assert!(!Arc::ptr_eq(&first, &get_platform()));
    }
}
