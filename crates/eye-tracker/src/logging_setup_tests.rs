// The Python suite has no test file for logging_setup.py; the tests below are new.
// `setup_logging` itself (it starts the process-wide logger, which the unit tests of this crate
// replace with a capturing one) is tested in tests/logging.rs. Not ported (nothing to port): the
// Qt message handler (`install_qt_message_handler`), the OpenCV and `cv2` log-level handling
// and `captureWarnings`.

use super::*;

#[test]
fn test_parse_level() {
    assert_eq!(
        parse_level(Some(" debug "), LevelFilter::Info),
        LevelFilter::Debug
    );
    assert_eq!(
        parse_level(Some("WARNING"), LevelFilter::Info),
        LevelFilter::Warn
    );
    assert_eq!(
        parse_level(Some("critical"), LevelFilter::Info),
        LevelFilter::Error
    );
    assert_eq!(
        parse_level(Some("loud"), LevelFilter::Warn),
        LevelFilter::Warn
    );
    assert_eq!(parse_level(None, LevelFilter::Info), LevelFilter::Info);
}

#[test]
fn test_debug_keeps_the_app_and_quietens_libraries() {
    let spec = log_specification(LevelFilter::Debug);
    assert!(spec.enabled(log::Level::Debug, "eye_tracker::engine"));
    assert!(!spec.enabled(log::Level::Debug, "tract_core::ops"));
    assert!(spec.enabled(log::Level::Warn, "tract_core::ops"));
    assert!(!spec.enabled(log::Level::Info, "wgpu_hal::vulkan"));
    assert!(spec.enabled(log::Level::Info, "zbus::connection"));
    assert!(!spec.enabled(log::Level::Debug, "zbus::connection"));
}

#[test]
fn test_line_format() {
    let mut now = DeferredNow::new();
    let mut line = |level, target| {
        let mut out = Vec::new();
        let args = format_args!("hello {}", 42);
        let record = Record::builder()
            .level(level)
            .target(target)
            .args(args)
            .build();
        format_line(&mut out, &mut now, &record).unwrap();
        String::from_utf8(out).unwrap()
    };
    // "2026-01-31 12:00:00,123 WARNING eye_tracker.config: hello 42"
    let warning = line(log::Level::Warn, "eye_tracker::config");
    assert!(
        warning.ends_with(" WARNING eye_tracker.config: hello 42\n"),
        "{warning}"
    );
    assert_eq!(warning.as_bytes()[4], b'-');
    assert_eq!(warning.as_bytes()[19], b',');
    let crash = line(log::Level::Error, CRASH_TARGET);
    assert!(
        crash.ends_with(" CRITICAL eye_tracker.crash: hello 42\n"),
        "{crash}"
    );
}
