//! Tests of `about.rs` (`test_about_dialog` of tests/test_ui_basic.py).
//!
//! Changed: the dialog's rich-text labels are checked through the texts they are built from
//! (`PRIVACY_STATEMENT`, `privacy_parts`, `notices_text`); `system_info` names Rust and egui
//! where the Python line named Python and Qt (there is no Qt, PySide6 or OpenCV in the Rust
//! app), and the clipboard is checked on the copy command egui emits. Not ported:
//! `dialog.grab()` (a live widget); a headless smoke test draws the window instead.

use super::*;

#[test]
fn test_about_dialog() {
    // r3-ux-docs-07: as docs/privacy.md says, only the app's own code has none.
    assert!(PRIVACY_STATEMENT.contains("never saved"));
    assert!(!PRIVACY_STATEMENT.contains("no network code at all"));
    assert!(PRIVACY_STATEMENT.contains("own code contains no network code"));
    let (before, link, after, url) = privacy_parts();
    assert_eq!(format!("{before}{link}{after}"), PRIVACY_STATEMENT);
    assert!(url.contains("docs/privacy.md")); // links the details
    assert_eq!(link, "the privacy documentation");

    let notices = notices_text();
    for needle in [
        "MediaPipe",
        "YuNet",
        "tract",
        "egui and eframe",
        "nokhwa",
        "MIT OR Apache-2.0",
        "Apache-2.0",
        "MIT",
    ] {
        assert!(notices.contains(needle), "{needle}");
    }
    for needle in ["Qt", "PySide6", "NumPy", "psutil", "PyObjC"] {
        assert!(!notices.contains(needle), "{needle}");
    }
    // Only the MediaPipe models are shipped (run by OpenCV), not its runtime.
    assert!(notices.contains("MediaPipe Face Landmarker model"));
    assert!(notices.contains("MediaPipe runtime is not included"));
    assert!(THIRD_PARTY.len() >= 6);

    let info = system_info();
    for needle in [VERSION, "Rust", "egui", APP_NAME] {
        assert!(info.contains(needle), "{needle}");
    }
    assert_eq!(info.matches(" · ").count(), 3);

    let mut dialog = AboutDialog::new();
    assert_eq!(dialog.copy_button_text(), "Copy system info");
    let ctx = egui::Context::default();
    let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
        dialog.copy_system_info(ui.ctx())
    });
    output.textures_delta.clear(); // nothing renders the textures
    let copied: Vec<&str> = output
        .platform_output
        .commands
        .iter()
        .filter_map(|command| match command {
            egui::OutputCommand::CopyText(text) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(copied, [info.as_str()]);
    assert_eq!(dialog.copy_button_text(), "Copied ✓");
}

#[test]
fn test_about_window_opens_draws_and_closes() {
    let mut dialog = AboutDialog::new();
    assert!(!dialog.is_open());
    dialog.open();
    let ctx = egui::Context::default();
    for _ in 0..2 {
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| dialog.show(ui.ctx()));
        output.textures_delta.clear(); // nothing renders the textures
    }
    assert!(dialog.is_open());
    dialog.close();
    assert!(!dialog.is_open());
    assert_eq!(
        AboutDialog::viewport_builder().title.as_deref(),
        Some("About Eye Tracker")
    );
}
