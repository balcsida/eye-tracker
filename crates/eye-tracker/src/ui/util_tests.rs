//! Tests of `util.rs` (the util tests and the `bgr_to_qimage` frame-conversion tests of
//! tests/test_ui_basic.py).
//!
//! Not ported:
//! * `test_controller_accessors_are_defensive`: it fed duck-typed controller fakes to
//!   `controller_settings`, `controller_state` and `controller_monitors`; the UI never holds the
//!   controller now, `app.rs` passes plain values, so those accessors do not exist.
//!   `settings_from_controller`'s fallback is tested instead.
//! * `test_bgr_to_qimage_other_layouts`: greyscale, BGRA, non-contiguous and negative-stride
//!   arrays; a `Frame` is always contiguous BGR8.
//! * `test_bgr_to_qimage_single_row_or_column` for the shapes `(1, 1)` and `(1, 1, 4)` (greyscale
//!   and BGRA), and the dtype/rank cases of `test_bgr_to_qimage_rejects_bad_frames` (the type
//!   system rules them out).
//! * the `QKeySequence` internals of `test_hotkey_sequence_swaps_control_and_command_on_macos`
//!   are tested on `KeySequence` instead.

use super::*;
use crate::testing;
use crate::types::Rect;

fn left() -> Monitor {
    Monitor {
        primary: true,
        ..Monitor::new(0, "Left", Rect::new(0, 0, 1920, 1080))
    }
}

fn right() -> Monitor {
    Monitor::new(1, "Right", Rect::new(1920, 0, 1920, 1080))
}

#[test]
fn test_ui_scale_and_formatting() {
    assert!((ui_scale(None) - 1.0).abs() < 1e-9);
    assert_eq!(scaled(28.0, 1.5), 42);
    assert_eq!(scaled(0.2, 1.0), 1);
    assert_eq!(format_fps(4.2), "4 fps");
    assert_eq!(format_fps(12.0), "12 fps");
    assert_eq!(format_fps(0.5), "0.5 fps");
    assert_eq!(format_fps(f64::NAN), "0 fps");
    assert_eq!(format_cpu(0.64), "CPU 0.6 %");
    assert_eq!(format_cpu(f64::INFINITY), "CPU 0.0 %");
    assert_eq!(elide("a  b\nc", 10), "a b c");
    assert_eq!(elide("abcdefghijkl", 6), "abcde…");
    assert_eq!(monitor_label(&right()), "Monitor 2 · Right");
}

#[test]
fn test_ui_scale_follows_logical_dpi() {
    if cfg!(target_os = "macos") {
        // macOS UI is measured in points.
        assert_eq!(ui_scale(Some(2.0)), 1.0);
        return;
    }
    // The monitor's scale factor is logical DPI / 96.
    assert!((ui_scale(Some(144.0 / 96.0)) - 1.5).abs() < 1e-9);
    assert!((ui_scale(Some(192.0 / 96.0)) - 2.0).abs() < 1e-9);
    assert_eq!(ui_scale(Some(f64::NAN)), 1.0);
    assert_eq!(ui_scale(Some(0.0)), 1.0);
    assert_eq!(ui_scale(Some(10_000.0 / 96.0)), 4.0);
    assert_eq!(to_points(300.0, 1.5), 200.0);
}

#[test]
fn test_hotkey_sequence_display() {
    if cfg!(target_os = "macos") {
        return; // macOS shows ⌃⌥ glyphs instead
    }
    let text = |value: &str| hotkey_sequence_text(Some(value), None).map(|s| s.portable_text());
    assert_eq!(text("ctrl+alt+t").as_deref(), Some("Ctrl+Alt+T"));
    assert_eq!(text("Shift+F5").as_deref(), Some("Shift+F5"));
    let parsed = parse_hotkey("ctrl+shift+space").unwrap();
    assert_eq!(
        hotkey_sequence(&parsed, None).unwrap().portable_text(),
        "Ctrl+Shift+Space"
    );
    for bad in ["", "   ", "p", "ctrl+", "ctrl+f99"] {
        assert_eq!(hotkey_sequence_text(Some(bad), None), None, "{bad:?}");
    }
    assert_eq!(hotkey_sequence_text(None, None), None);
    assert_eq!(parse_hotkey_text(Some("nonsense")), None);
    assert_eq!(
        parse_hotkey_text(Some("ctrl+alt+p")),
        Some(parse_hotkey("ctrl+alt+p").unwrap())
    );
}

#[test]
fn test_hotkey_sequence_swaps_control_and_command_on_macos() {
    // The system's control modifier is Command on macOS; the physical Control key is Meta.
    let sequence = hotkey_sequence_text(Some("ctrl+meta+k"), Some(true)).unwrap();
    assert!(sequence.meta && sequence.control);
    let alt_only = hotkey_sequence_text(Some("ctrl+alt+k"), Some(true)).unwrap();
    assert!(alt_only.meta);
    assert!(!alt_only.control);
    let elsewhere = hotkey_sequence_text(Some("ctrl+alt+k"), Some(false)).unwrap();
    assert!(elsewhere.control);
    assert!(!elsewhere.meta);
}

#[test]
fn test_screen_for_monitor_matches_geometry() {
    let screens = vec![
        Monitor {
            primary: true,
            ..Monitor::new(0, "DP-1", Rect::new(0, 0, 1920, 1080))
        },
        Monitor::new(1, "DP-2", Rect::new(1920, 0, 1280, 720)),
    ];
    let first = &screens[0];
    let by_geometry = screen_for_monitor(&Monitor::new(1, "whatever", first.rect), &screens);
    assert_eq!(by_geometry, Some(first));
    let by_name = screen_for_monitor(&Monitor::new(0, "DP-2", Rect::new(1, 2, 3, 4)), &screens);
    assert_eq!(by_name, Some(&screens[1]));
    // Unknown geometry and name: fall back to the screen under the centre / by index.
    let by_centre = screen_for_monitor(&Monitor::new(1, "", Rect::new(10, 10, 50, 50)), &screens);
    assert_eq!(by_centre, Some(first));
    let by_index = screen_for_monitor(&Monitor::new(1, "", Rect::new(-90, -90, 50, 50)), &screens);
    assert_eq!(by_index, Some(&screens[1]));
    assert_eq!(
        screen_for_monitor(&Monitor::new(7, "", Rect::new(-90, -90, 50, 50)), &screens),
        None
    );
    assert_eq!(screen_for_point(-99999.0, -99999.0, &screens), Some(first));
    assert_eq!(screen_for_point(2000.5, 10.0, &screens), Some(&screens[1]));
    assert_eq!(screen_for_monitor(&left(), &[]), None);
    assert_eq!(
        primary_monitor(&[right(), left()]).map(|m| m.index),
        Some(0)
    );
    assert_eq!(primary_monitor(&[right()]).map(|m| m.index), Some(1));
    assert_eq!(primary_monitor(&[]), None);
}

#[test]
fn test_settings_from_controller_copies_or_reads_the_settings_file() {
    let mut current = Settings::default();
    current.general.notifications = false;
    assert_eq!(settings_from_controller(Some(&current)), current);

    // Without settings the file on disk is used, not the defaults.
    let _dirs = testing::app_dirs();
    let mut on_disk = Settings::default();
    on_disk.general.notifications = !on_disk.general.notifications;
    std::fs::create_dir_all(paths::config_dir()).unwrap();
    std::fs::write(
        paths::settings_file(),
        serde_json::to_string(&on_disk).unwrap(),
    )
    .unwrap();
    assert_eq!(settings_from_controller(None), on_disk);
}

#[test]
fn test_system_tray_theme_is_a_plain_guess() {
    // Only the Windows branch reads the registry, and the others fall back to the app theme.
    set_app_prefers_dark(false);
    if !cfg!(windows) && std::env::var("XDG_CURRENT_DESKTOP").is_err() {
        assert!(!system_tray_is_dark());
    }
    set_app_prefers_dark(true);
    let _ = system_tray_is_dark();
}

#[test]
fn test_qcolor_parses_hex_and_clamps_alpha() {
    assert_eq!(qcolor("#6366F1", None), [0x63, 0x66, 0xF1, 255]);
    assert_eq!(qcolor("#FFFFFF", Some(14)), [255, 255, 255, 14]);
    assert_eq!(qcolor("#000000", Some(999)), [0, 0, 0, 255]);
    assert_eq!(qcolor("#000000", Some(-4)), [0, 0, 0, 0]);
    assert_eq!(qcolor("nonsense", None), [0, 0, 0, 255]);
}

#[test]
fn test_egui_palette_matches_the_hex_palette() {
    for (hex, egui) in [
        (ACCENT, egui_colors::ACCENT),
        (SUCCESS, egui_colors::SUCCESS),
        (WARNING, egui_colors::WARNING),
        (DANGER, egui_colors::DANGER),
    ] {
        assert_eq!(color32(qcolor(hex, None)), egui);
    }
}

#[test]
fn test_bgr_to_qimage_converts_colours() {
    let mut frame = Frame::filled(3, 2, [0, 0, 0]);
    frame.data[0..3].copy_from_slice(&[255, 0, 0]); // blue in BGR
    frame.data[3..6].copy_from_slice(&[0, 255, 0]);
    frame.data[15..18].copy_from_slice(&[0, 0, 255]); // red
    let image = bgr_to_color_image(&frame).unwrap();
    assert_eq!(image.size, [3, 2]);
    assert_eq!(image.pixels[0], Color32::from_rgb(0, 0, 255));
    assert_eq!(image.pixels[1], Color32::from_rgb(0, 255, 0));
    assert_eq!(image.pixels[5], Color32::from_rgb(255, 0, 0));
    frame.data.fill(0); // the image owns a copy of the pixels
    assert_eq!(image.pixels[0], Color32::from_rgb(0, 0, 255));
}

#[test]
fn test_bgr_to_qimage_single_row_or_column() {
    for (height, width) in [(1, 1), (1, 7), (5, 1)] {
        let image = bgr_to_color_image(&Frame::filled(width, height, [77; 3])).unwrap();
        assert_eq!(image.size, [width, height]);
        assert_eq!(
            image.pixels[width * height - 1],
            Color32::from_rgb(77, 77, 77)
        );
    }
}

#[test]
fn test_bgr_to_qimage_rejects_bad_frames() {
    let empty = Frame {
        width: 4,
        height: 0,
        data: Vec::new(),
    };
    assert_eq!(bgr_to_color_image(&empty), Err(FrameError::Empty));
    let short = Frame {
        width: 2,
        height: 2,
        data: vec![0; 11],
    };
    let error = bgr_to_color_image(&short).unwrap_err();
    assert!(error.to_string().contains("shape"), "{error}");
    assert!(
        bgr_to_color_image(&empty)
            .unwrap_err()
            .to_string()
            .contains("image")
    );
}

#[test]
fn test_floating_viewports_never_take_focus_or_show_in_the_taskbar() {
    let dot = floating_viewport("dot", true, false, true);
    assert_eq!(dot.mouse_passthrough, Some(true));
    assert_eq!(dot.active, Some(false));
    assert_eq!(dot.window_level, Some(WindowLevel::AlwaysOnTop));
    assert_eq!(dot.decorations, Some(false));
    assert_eq!(dot.taskbar, Some(false));
    assert_eq!(dot.transparent, Some(true));
    let curtain = floating_viewport("curtain", false, true, false);
    assert_eq!(curtain.active, Some(true));
    assert_eq!(curtain.mouse_passthrough, Some(false));
    assert_eq!(curtain.transparent, Some(false));
    let placed = place_viewport(curtain, Rect::new(1920, 0, 1920, 1080), 1.5);
    assert_eq!(placed.position, Some(Pos2::new(1280.0, 0.0)));
    assert_eq!(placed.inner_size, Some(Vec2::new(1280.0, 720.0)));
}

/// A window the OS put elsewhere is sent back to its place; one in place is left alone.
#[test]
fn test_keep_viewport_placed_moves_a_misplaced_window_back() {
    let builder = place_viewport(
        ViewportBuilder::default(),
        Rect::new(832, 1080, 1728, 1117),
        1.0,
    );
    let commands = |outer: egui::Rect| {
        let mut input = egui::RawInput::default();
        input
            .viewports
            .entry(egui::ViewportId::ROOT)
            .or_default()
            .outer_rect = Some(outer);
        let ctx = egui::Context::default();
        let mut output = ctx.run_ui(input, |ui| keep_viewport_placed(ui.ctx(), &builder));
        output.textures_delta.clear();
        output
            .viewport_output
            .remove(&egui::ViewportId::ROOT)
            .map(|viewport| viewport.commands)
            .unwrap_or_default()
            .into_iter()
            .filter(|command| {
                matches!(
                    command,
                    egui::ViewportCommand::OuterPosition(_) | egui::ViewportCommand::InnerSize(_)
                )
            })
            .collect::<Vec<_>>()
    };
    // Where macOS put it: into the main screen's visible frame.
    let moved = commands(egui::Rect::from_min_size(
        Pos2::new(832.0, 63.0),
        Vec2::new(1728.0, 1017.0),
    ));
    assert_eq!(
        moved,
        [
            egui::ViewportCommand::OuterPosition(Pos2::new(832.0, 1080.0)),
            egui::ViewportCommand::InnerSize(Vec2::new(1728.0, 1117.0)),
        ]
    );
    let placed = egui::Rect::from_min_size(Pos2::new(832.0, 1080.0), Vec2::new(1728.0, 1117.0));
    assert!(commands(placed).is_empty());
}
