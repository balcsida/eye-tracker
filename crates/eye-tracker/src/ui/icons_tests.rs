//! Tests of `icons.rs` (the icon tests of tests/test_ui_basic.py).
//!
//! Not ported: `QIcon.pixmap` / `QIcon.cacheKey` checks as such (an `IconSet` is a plain list of
//! images; the cache is checked with `Arc::ptr_eq`), and the `QPainter` argument of
//! `paint_lock` (it paints on a `tiny_skia::Pixmap`).

use super::*;
use TrackingState as S;

const ALL_STATES: [TrackingState; 10] = [
    S::Starting,
    S::NeedsCalibration,
    S::Calibrating,
    S::Tracking,
    S::Paused,
    S::Privacy,
    S::Away,
    S::Locked,
    S::Yielded,
    S::CameraError,
];

/// The pixels whose alpha exceeds `threshold`.
fn opaque_pixels(image: &Rgba, threshold: u8) -> Vec<[u8; 4]> {
    (0..image.height)
        .flat_map(|y| (0..image.width).map(move |x| (x, y)))
        .map(|(x, y)| image.pixel(x, y))
        .filter(|pixel| pixel[3] > threshold)
        .collect()
}

/// Qt's `QColor.lightness()`.
fn lightness(pixel: [u8; 4]) -> f64 {
    let max = f64::from(pixel[..3].iter().copied().max().unwrap_or(0));
    let min = f64::from(pixel[..3].iter().copied().min().unwrap_or(0));
    (max + min) / 2.0
}

#[test]
fn test_tray_images_render_for_every_state() {
    for state in ALL_STATES {
        for dark in [true, false] {
            for mask in [false, true] {
                for size in [16, 22, 32, 64] {
                    let image = render_tray_image(state, size, dark, mask).unwrap();
                    assert_eq!((image.width, image.height), (size, size));
                    assert!(opaque_pixels(&image, 40).len() > size as usize); // a glyph was drawn
                    assert!(image.pixel(0, 0)[3] < 60); // corners stay see-through
                }
            }
        }
        let icon = tray_icon(state, Some(true), Some(false));
        assert!(!icon.images.is_empty());
        for size in [16, 24, 48] {
            assert!(icon.best(size).is_some_and(|image| image.width >= size));
        }
        assert!(!icon.is_mask);
    }
}

#[test]
fn test_tray_template_icons_are_monochrome_masks() {
    for state in ALL_STATES {
        let icon = tray_icon(state, None, Some(true));
        assert!(icon.is_mask);
        let image = render_tray_image(state, 32, true, true).unwrap();
        let colours = opaque_pixels(&image, 200);
        assert!(!colours.is_empty());
        assert!(colours.iter().all(|c| c[..3] == [0, 0, 0]));
    }
}

#[test]
fn test_tray_glyph_colour_follows_taskbar_theme() {
    let mean_lightness = |image: &Rgba| {
        let colours = opaque_pixels(image, 220);
        colours.iter().map(|c| lightness(*c)).sum::<f64>() / colours.len() as f64
    };
    let on_dark = render_tray_image(S::Paused, 32, true, false).unwrap();
    let on_light = render_tray_image(S::Paused, 32, false, false).unwrap();
    assert!(mean_lightness(&on_dark) > 180.0);
    assert!(mean_lightness(&on_light) < 90.0);
}

#[test]
fn test_tray_glyph_has_a_contrasting_halo() {
    // A light glyph (dark taskbar guess) still has a dark rim, so it stays readable if the guess
    // was wrong and the taskbar is light.
    let image = render_tray_image(S::Tracking, 32, true, false).unwrap();
    let rim: Vec<_> = opaque_pixels(&image, 20)
        .into_iter()
        .filter(|c| lightness(*c) < 60.0)
        .collect();
    assert!(!rim.is_empty());
}

#[test]
fn test_tray_states_are_visually_distinct() {
    let states = [
        S::Tracking,
        S::Paused,
        S::Privacy,
        S::Away,
        S::CameraError,
        S::NeedsCalibration,
        S::Starting,
        S::Calibrating,
    ];
    let images: Vec<Rgba> = states
        .iter()
        .map(|&s| render_tray_image(s, 32, true, false).unwrap())
        .collect();
    for (i, a) in images.iter().enumerate() {
        for (j, b) in images.iter().enumerate().skip(i + 1) {
            assert_ne!(a, b, "{:?} vs {:?}", states[i], states[j]);
        }
    }
}

#[test]
fn test_error_badge_is_red_and_calibration_badge_amber() {
    let error = render_tray_image(S::CameraError, 32, true, false).unwrap();
    let badge = error.pixel(25, 25); // centre of the badge on the 16-unit grid
    assert!(badge[0] > 200);
    assert!(badge[1] < 110);
    assert!(badge[3] > 240);
    let amber = render_tray_image(S::NeedsCalibration, 32, true, false)
        .unwrap()
        .pixel(25, 25);
    assert!(amber[0] > 200);
    assert!((121..200).contains(&amber[1]));
    assert!(amber[2] < 80);
}

#[test]
fn test_tray_icon_is_cached() {
    let first = tray_icon(S::Paused, Some(false), Some(false));
    assert!(Arc::ptr_eq(
        &first,
        &tray_icon(S::Paused, Some(false), Some(false))
    ));
    clear_cache();
    assert!(
        !tray_icon(S::Paused, Some(false), Some(false))
            .images
            .is_empty()
    );
}

#[test]
fn test_app_icon_renders_at_every_size() {
    let icon = app_icon();
    assert_eq!(icon.images.len(), APP_ICON_SIZES.len());
    for size in [16, 32, 256] {
        assert!(icon.best(size).is_some_and(|image| image.width >= size));
    }
    let image = render_app_icon_png(256, false).unwrap();
    assert_eq!(image.width, 256);
    assert_eq!(image.pixel(0, 0)[3], 0); // rounded corner
    assert_eq!(image.pixel(128, 20)[3], 255); // gradient tile
    assert!(lightness(image.pixel(70, 128)) > 240.0); // white of the eye
    let padded = render_app_icon_png(256, true).unwrap();
    assert!(padded.pixel(12, 128)[3] < image.pixel(12, 128)[3]);
    assert_eq!(
        render_app_icon_png(0, false).unwrap_err().to_string(),
        "size must be positive"
    );
    assert_eq!(
        render_tray_image(S::Tracking, 0, true, false)
            .unwrap_err()
            .to_string(),
        "size must be positive"
    );
}

#[test]
fn test_small_glyphs() {
    assert!(!status_dot_icon(util::SUCCESS).images.is_empty());
    assert!(status_dot_icon(util::SUCCESS).best(16).is_some());
    let mut pixmap = Pixmap::new(64, 64).unwrap();
    paint_lock(&mut pixmap, (0.0, 0.0, 64.0, 64.0), [255, 255, 255, 255]);
    let image = to_rgba(&pixmap);
    assert!(opaque_pixels(&image, 40).len() > 400);
    assert_eq!(image.pixel(32, 42)[3], 0); // the keyhole is cut out
    assert_eq!(render_lock(0, [0; 4]).unwrap_err(), SizeError);
}
