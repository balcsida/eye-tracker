//! `setup_logging` end to end. This is an integration test because the logger can only be
//! installed once per process.

use std::fs;

use eye_tracker::{logging_setup, paths};

fn log_text() -> String {
    fs::read_to_string(paths::log_file()).unwrap_or_default()
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
    assert!(paths::log_file().is_file());
    let text = log_text();
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
    assert!(!log_text().contains("after shutdown"));
}
