//! Tests for the default (unsupported-everywhere) PlatformServices contract.
//!
//! Not ported: `test_contract_documents_the_desktop_and_camera_semantics` (it asserted on
//! Python `__doc__` strings, which Rust does not keep at run time).

use std::cell::RefCell;
use std::collections::HashMap;

use super::*;
use crate::types::WindowHandle;

fn window() -> WindowRef {
    WindowRef {
        handle: WindowHandle::Hwnd(1),
        pid: Some(2),
        rect: Some(Rect::new(0, 0, 10, 10)),
    }
}

#[test]
fn test_defaults_are_safe_no_ops() {
    let services = GenericPlatform;
    // Only macOS knows the Accessibility permission in detail.
    assert_eq!(services.accessibility_status(), "unknown");
    // App Nap exists only on macOS: elsewhere nothing throttles us, so "in effect".
    assert!(services.set_background_activity(true));
    assert!(services.set_background_activity(false));
    assert_eq!(services.camera_in_use_by_other_app(), None);
    assert!(!services.is_window_valid(&window()));
    assert!(!services.lock_screen());
    assert!(!services.open_permission_settings("camera"));
}

#[test]
fn test_pane_queries_are_unsupported_by_default() {
    let services = GenericPlatform;
    let w = window();
    assert_eq!(services.window_app(&w), None);
    assert_eq!(services.window_client_rect(&w), None);
    assert!(!services.capabilities()["panes"]);
}

#[test]
fn test_process_basename() {
    let cases = [
        ("WindowsTerminal.exe", "windowsterminal"),
        ("C:\\Program Files\\WezTerm\\wezterm-gui.exe", "wezterm-gui"),
        ("/usr/bin/kitty", "kitty"),
        ("iTerm2", "iterm2"),
        ("  Code.EXE ", "code"),
    ];
    for (raw, name) in cases {
        assert_eq!(process_basename(raw), name, "{raw}");
    }
}

/// A process table `pid -> (create_time, name)`; counts name reads.
struct FakeProcessTable {
    table: RefCell<HashMap<u32, (u64, &'static str)>>,
    name_reads: RefCell<Vec<u32>>,
}

impl FakeProcessTable {
    fn new(entries: &[(u32, u64, &'static str)]) -> Self {
        Self {
            table: RefCell::new(entries.iter().map(|&(p, t, n)| (p, (t, n))).collect()),
            name_reads: RefCell::default(),
        }
    }

    fn lookup(&self, cache: &AppNameCache, pid: Option<i64>) -> Option<String> {
        cache.lookup(
            pid,
            |pid| self.table.borrow().get(&pid).map(|e| e.0),
            |pid| {
                self.name_reads.borrow_mut().push(pid);
                self.table.borrow().get(&pid).map(|e| e.1.to_string())
            },
        )
    }
}

#[test]
fn test_process_names_are_cached_per_process() {
    let procs = FakeProcessTable::new(&[(7, 1000, "WezTerm-GUI.exe")]);
    let cache = AppNameCache::default();
    assert_eq!(
        procs.lookup(&cache, Some(7)).as_deref(),
        Some("wezterm-gui")
    );
    assert_eq!(
        procs.lookup(&cache, Some(7)).as_deref(),
        Some("wezterm-gui")
    );
    // the second answer came from the cache
    assert_eq!(*procs.name_reads.borrow(), vec![7]);
    assert_eq!(procs.lookup(&cache, Some(404)), None);
    assert_eq!(procs.lookup(&cache, None), None);
    assert_eq!(procs.lookup(&cache, Some(0)), None);
}

/// The Claude app quit and its pid went to VS Code: VS Code must not pass as Claude.
#[test]
fn test_a_reused_pid_is_not_taken_for_the_process_that_quit() {
    let procs = FakeProcessTable::new(&[(4242, 1000, "claude.exe")]);
    let cache = AppNameCache::default();
    assert_eq!(procs.lookup(&cache, Some(4242)).as_deref(), Some("claude"));
    procs.table.borrow_mut().insert(4242, (1500, "Code.exe")); // same pid, another process
    assert_eq!(procs.lookup(&cache, Some(4242)).as_deref(), Some("code"));
    procs.table.borrow_mut().remove(&4242); // gone: nothing is answered from the cache
    assert_eq!(procs.lookup(&cache, Some(4242)), None);
    assert_eq!(cache.len(), 1); // the old entry was replaced, not kept
}

#[test]
fn test_the_trait_object_is_usable_from_any_thread() {
    let services: std::sync::Arc<dyn PlatformServices> = std::sync::Arc::new(GenericPlatform);
    let other = services.clone();
    std::thread::spawn(move || assert!(!other.lock_screen()))
        .join()
        .unwrap();
    assert!(
        services
            .running_process_names()
            .iter()
            .all(|n| n.to_lowercase() == *n)
    );
}
