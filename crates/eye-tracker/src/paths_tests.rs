// Ported from tests/test_paths.py. Not ported: the `diagnostics._profile_override()` and
// `autostart._active_profile()` assertions of both tests; they belong to the diagnostics and
// autostart modules, which are ported later (they read `paths::base_override()`).

use super::*;
use crate::testing;

#[test]
fn test_base_override_is_none_for_the_default_profile() {
    let _guard = testing::lock_paths();
    set_base_override(None);
    assert_eq!(base_override(), None);
}

#[test]
fn test_base_override_is_the_resolved_directory() {
    let _guard = testing::lock_paths();
    let tmp = tempfile::tempdir().unwrap();
    let cwd = std::env::current_dir().unwrap();
    std::env::set_current_dir(tmp.path()).unwrap();
    set_base_override(Some(Path::new("portable")));
    std::env::set_current_dir(cwd).unwrap();
    let expected = fs::canonicalize(tmp.path()).unwrap().join("portable");
    assert_eq!(base_override(), Some(expected.clone()));
    assert_eq!(config_dir(), expected);
    assert_eq!(data_dir(), expected.join("data"));
    assert_eq!(log_file(), expected.join("logs").join("eye-tracker.log"));
}

#[test]
fn test_resolve_removes_parent_components_below_missing_directories() {
    let tmp = tempfile::tempdir().unwrap();
    let base = fs::canonicalize(tmp.path()).unwrap();
    assert_eq!(resolve(&base.join("a/b/../c")), base.join("a/c"));
}

#[test]
fn test_sha1_matches_known_vectors() {
    assert_eq!(sha1_hex(b"abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
    assert_eq!(sha1_hex(b""), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    let long = [b'a'; 1000];
    assert_eq!(sha1_hex(&long), "291e9a6c66994949b57ba5e650361e98fc36b1ba");
}

#[test]
fn test_ipc_name_depends_on_the_profile() {
    let _guard = testing::lock_paths();
    set_base_override(None);
    let default = ipc_name();
    assert!(default.starts_with("eye-tracker-") && default.len() == "eye-tracker-".len() + 10);
    let tmp = tempfile::tempdir().unwrap();
    set_base_override(Some(tmp.path()));
    assert_ne!(ipc_name(), default);
}
