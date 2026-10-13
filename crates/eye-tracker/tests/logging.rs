//! `setup_logging` end to end. This is an integration test because the logger can only be
//! installed once per process.

use std::fs;

use eye_tracker::{logging_setup, paths};

fn log_text(dir: &std::path::Path) -> String {
    fs::read_dir(dir)
        .unwrap()
        .map(|e| fs::read_to_string(e.unwrap().path()).unwrap())
        .collect()
}

#[test]
fn test_log_lines_and_panics_reach_the_file() {
    let tmp = tempfile::tempdir().unwrap();
    paths::set_base_override(Some(tmp.path()));
    logging_setup::setup_logging("INFO", false, true);
    log::warn!(target: "eye_tracker::config", "hello {}", 42);
    log::debug!("below the level");
    logging_setup::set_level("DEBUG");
    log::debug!("above the level now");
    let _ = std::thread::Builder::new()
        .name("worker-x".into())
        .spawn(|| panic!("boom"))
        .unwrap()
        .join();
    logging_setup::shutdown_logging();
    let text = log_text(&tmp.path().join("logs"));
    assert!(
        text.contains(" WARNING eye_tracker.config: hello 42"),
        "{text}"
    );
    assert!(text.contains("above the level now"));
    assert!(!text.contains("below the level"));
    assert!(
        text.contains(" CRITICAL eye_tracker.crash: Unhandled exception in thread worker-x: boom")
    );
    // After shutdown nothing more is written.
    log::error!("after shutdown");
    assert!(!log_text(&tmp.path().join("logs")).contains("after shutdown"));
}
