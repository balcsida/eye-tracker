//! Tests for `make_icons` (the make_icons part of `tests/test_build_scripts.py`).
//!
//! Not ported: `test_qimage_to_pillow_keeps_colour_and_alpha` (there is no QImage → Pillow
//! conversion any more; the renderer returns an `RgbaImage` directly).

use image::Rgba;

use super::*;

fn solid_renderer(color: [u8; 4]) -> impl Fn(u32, bool) -> RgbaImage {
    move |size, _padded| RgbaImage::from_pixel(size, size, Rgba(color))
}

#[test]
fn test_renderer_size_is_checked() {
    let wrong = |size: u32, _padded: bool| RgbaImage::new(size + 1, size);
    let error = render(&wrong, 16, false).unwrap_err();
    assert!(error.contains("renderer returned"), "{error}");
}

#[test]
fn test_write_icons_produces_every_format() {
    let tmp = tempfile::tempdir().unwrap();
    let written = write_icons(&solid_renderer([0x33, 0x66, 0xCC, 255]), tmp.path()).unwrap();
    let mut expected: Vec<PathBuf> = OUTPUTS
        .iter()
        .map(|(_, rel)| tmp.path().join(rel))
        .collect();
    let mut written_sorted = written.clone();
    written_sorted.sort();
    expected.sort();
    assert_eq!(written_sorted, expected);
    assert!(missing_outputs(tmp.path()).is_empty());

    let png = image::open(tmp.path().join(output("png"))).unwrap();
    assert_eq!((png.width(), png.height()), (1024, 1024));
    assert!(png.color().has_alpha());
    let linux = image::open(tmp.path().join(output("linux"))).unwrap();
    assert_eq!((linux.width(), linux.height()), (512, 512));

    let ico = IconDir::read(fs::File::open(tmp.path().join(output("ico"))).unwrap()).unwrap();
    let sizes: Vec<u32> = ico.entries().iter().map(|e| e.width()).collect();
    assert_eq!(sizes, ICO_SIZES);

    let icns = IconFamily::read(fs::File::open(tmp.path().join(output("icns"))).unwrap()).unwrap();
    let largest = icns
        .get_icon_with_type(IconType::RGBA32_512x512_2x)
        .unwrap();
    assert_eq!((largest.width(), largest.height()), (1024, 1024));
    assert_eq!(icns.available_icons().len(), 8);
}

#[test]
fn test_app_icon_renders_with_the_real_renderer() {
    for padded in [false, true] {
        let image = render(
            &|size, padded| render_app_icon_png(size, padded),
            64,
            padded,
        )
        .unwrap();
        assert_eq!(image.dimensions(), (64, 64));
        // The corner is transparent (rounded tile), the centre is opaque (the eye).
        assert_eq!(image.get_pixel(0, 0)[3], 0);
        assert_eq!(image.get_pixel(32, 32)[3], 255);
    }
}

#[test]
fn test_check_mode() {
    let tmp = tempfile::tempdir().unwrap();
    let args = Args {
        out: Some(tmp.path().to_path_buf()),
        check: true,
    };
    assert_eq!(main(&args), 1);
    for (_, rel) in OUTPUTS {
        let path = tmp.path().join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, b"x").unwrap();
    }
    assert_eq!(main(&args), 0);
}
