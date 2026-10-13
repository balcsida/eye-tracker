//! Render the application icon into every format the packages need.
//!
//! The artwork is drawn in code (a port of `render_app_icon_png` of the Python app's
//! `ui/icons.py`), so the app, the tray and the installers share one source of truth. It is
//! rendered at each native size, so small icons stay crisp instead of being scaled down from
//! 1024 px, and written as:
//!
//! | File                                  | Sizes     | Used by                        |
//! |---------------------------------------|-----------|--------------------------------|
//! | `assets/icon.png`                     | 1024      | README, release pages          |
//! | `packaging/windows/eye-tracker.ico`   | 16 - 256  | executables, installer         |
//! | `packaging/macos/eye-tracker.icns`    | 32 - 1024 | `Eye Tracker.app`              |
//! | `packaging/linux/eye-tracker.png`     | 512       | AppImage, `.desktop` entry     |
//!
//! The `.icns` uses the macOS icon grid (a smaller tile with a soft shadow, the
//! `padded` variant); the other formats use the full square.
//!
//! ```text
//! cargo xtask make-icons             # write all icons
//! cargo xtask make-icons --out DIR   # write under DIR instead of the repo
//! cargo xtask make-icons --check     # exit 1 if any icon file is missing
//! ```
//!
//! The artwork is rasterised by a small supersampling renderer (the Python app used Qt's
//! `QPainter`), so the pixels differ slightly from the committed files at the antialiased edges.

use std::fs;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use icns::{IconFamily, IconType};
use ico::{IconDir, IconDirEntry, IconImage, ResourceType};
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ImageEncoder, RgbaImage};

use crate::repo_root;

/// Sizes stored in the Windows .ico (Explorer, taskbar, Alt+Tab, 100-200 % DPI).
pub const ICO_SIZES: [u32; 10] = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256];
/// Pixel sizes stored in an .icns (ic07..ic14: 16-512 pt at 1x and 2x).
pub const ICNS_SIZES: [u32; 6] = [32, 64, 128, 256, 512, 1024];

/// Output files, relative to the root: png, ico, icns, linux.
pub const OUTPUTS: [(&str, &str); 4] = [
    ("png", "assets/icon.png"),
    ("ico", "packaging/windows/eye-tracker.ico"),
    ("icns", "packaging/macos/eye-tracker.icns"),
    ("linux", "packaging/linux/eye-tracker.png"),
];

fn output(key: &str) -> &'static str {
    OUTPUTS
        .iter()
        .find(|(k, _)| *k == key)
        .map_or("", |(_, path)| path)
}

/// `render(size, padded)` → a `size` × `size` image.
pub type Renderer<'a> = dyn Fn(u32, bool) -> RgbaImage + 'a;

// -------------------------------------------------------------------------------- drawing
const ACCENT: [f32; 3] = [0x63 as f32, 0x66 as f32, 0xF1 as f32];
const ACCENT_2: [f32; 3] = [0x22 as f32, 0xD3 as f32, 0xEE as f32];

/// Premultiplied RGBA of one sample, 0..255 per channel (alpha 0..1).
#[derive(Clone, Copy, Default)]
struct Sample {
    rgb: [f32; 3],
    alpha: f32,
}

impl Sample {
    /// Paint `color` (0..255, straight alpha `alpha`) over the sample.
    fn over(&mut self, color: [f32; 3], alpha: f32) {
        for (channel, source) in self.rgb.iter_mut().zip(color) {
            *channel = source * alpha + *channel * (1.0 - alpha);
        }
        self.alpha = alpha + self.alpha * (1.0 - alpha);
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn lerp_color(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
    ]
}

fn hex_color(value: u32) -> [f32; 3] {
    [
        ((value >> 16) & 0xFF) as f32,
        ((value >> 8) & 0xFF) as f32,
        (value & 0xFF) as f32,
    ]
}

/// A rounded rectangle `(x, y, w, h)` with corner radius `r`.
fn in_rounded(px: f32, py: f32, (x, y, w, h): (f32, f32, f32, f32), r: f32) -> bool {
    let (hw, hh) = (w / 2.0, h / 2.0);
    let (dx, dy) = ((px - x - hw).abs(), (py - y - hh).abs());
    if dx > hw || dy > hh {
        return false;
    }
    let (qx, qy) = (dx - (hw - r), dy - (hh - r));
    qx <= 0.0 || qy <= 0.0 || qx * qx + qy * qy <= r * r
}

fn in_circle(px: f32, py: f32, (cx, cy): (f32, f32), r: f32) -> bool {
    (px - cx).powi(2) + (py - cy).powi(2) <= r * r
}

/// The closed almond eye outline with pointed corners (`_almond`): two cubics mirrored about
/// the horizontal axis, so its half-height at each x is all that is needed to fill it.
struct Almond {
    left: f32,
    right: f32,
    k: f32,
    dx: f32,
}

impl Almond {
    fn new(x: f32, w: f32, h: f32, pinch: f32) -> Self {
        Self {
            left: x,
            right: x + w,
            // A cubic whose two control points sit at ±k peaks at 0.75 k.
            k: h / 2.0 / 0.75,
            dx: w * pinch,
        }
    }

    /// Half-height of the almond at `x` (0 outside it).
    fn half_height(&self, x: f32) -> f32 {
        if x < self.left || x > self.right {
            return 0.0;
        }
        // x(t) is monotonic for these control points: bisect t, then evaluate y(t).
        let (p0, p1, p2, p3) = (
            self.left,
            self.left + self.dx,
            self.right - self.dx,
            self.right,
        );
        let bezier = |a: f32, b: f32, c: f32, d: f32, t: f32| {
            let u = 1.0 - t;
            u * u * u * a + 3.0 * u * u * t * b + 3.0 * u * t * t * c + t * t * t * d
        };
        let (mut lo, mut hi) = (0.0f32, 1.0f32);
        for _ in 0..24 {
            let mid = (lo + hi) / 2.0;
            if bezier(p0, p1, p2, p3, mid) < x {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let t = (lo + hi) / 2.0;
        -bezier(0.0, -self.k, -self.k, 0.0, t)
    }
}

/// Stops of a gradient: `(position, color)` ascending; clamped outside.
fn gradient(stops: &[(f32, [f32; 3], f32)], t: f32) -> ([f32; 3], f32) {
    let t = t.clamp(0.0, 1.0);
    for pair in stops.windows(2) {
        let ((t0, c0, a0), (t1, c1, a1)) = (pair[0], pair[1]);
        if t <= t1 {
            let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
            return (lerp_color(c0, c1, f), lerp(a0, a1, f));
        }
    }
    let (_, c, a) = stops[stops.len() - 1];
    (c, a)
}

/// The application icon as a `size`×`size` RGBA image: a rounded square with the brand's
/// indigo→cyan gradient and a white eye (`render_app_icon_png`).
///
/// `padded` follows the macOS icon grid (smaller tile with a soft shadow); use it for `.icns`.
/// Windows and Linux icons use the full square.
///
/// # Panics
///
/// When `size` is zero.
pub fn render_app_icon_png(size: u32, padded: bool) -> RgbaImage {
    assert!(size >= 1, "size must be positive");
    let s = size as f32;
    let (bx, by, w) = if padded {
        // macOS icon grid: an 824/1024 rounded square with room for its shadow.
        let side = s * 824.0 / 1024.0;
        ((s - side) / 2.0, (s - side) / 2.0 - s * 0.01, side)
    } else {
        (0.0, 0.0, s)
    };
    let body = (bx, by, w, w);
    let radius = w * 0.225;
    let spread = s * 0.03;
    let (cx, cy) = (bx + w / 2.0, by + w / 2.0);
    let eye = Almond::new(bx + w * 0.15, w * 0.70, w * 0.43, 0.23);
    let iris_r = w * 0.155;
    let ring_width = (w * 0.012).max(1.0);

    let samples_per_side: u32 = if size > 128 { 4 } else { 8 };
    let n = samples_per_side as f32;
    // The eye's half-height depends on x only: one table entry per sample column.
    let columns = (size * samples_per_side) as usize;
    let column_x = |c: usize| (c as f32 + 0.5) / n;
    let eye_half: Vec<f32> = (0..columns).map(|c| eye.half_height(column_x(c))).collect();

    let iris_stops = [
        (0.0, hex_color(0x3B3A98), 1.0),
        (0.75, hex_color(0x26236E), 1.0),
        (1.0, hex_color(0x1B1850), 1.0),
    ];
    let tile_stops = [(0.0, ACCENT, 1.0), (1.0, ACCENT_2, 1.0)];
    let light_stops = [
        (0.0, [255.0; 3], 56.0 / 255.0),
        (0.55, [255.0; 3], 0.0),
        (1.0, [255.0; 3], 0.0),
    ];
    let diagonal = (w * w + w * w).max(f32::MIN_POSITIVE);

    let mut image = RgbaImage::new(size, size);
    for py in 0..size {
        for px in 0..size {
            let mut total = Sample::default();
            for sy in 0..samples_per_side {
                let y = py as f32 + (sy as f32 + 0.5) / n;
                for sx in 0..samples_per_side {
                    let x = px as f32 + (sx as f32 + 0.5) / n;
                    let column = (px * samples_per_side + sx) as usize;
                    let mut sample = Sample::default();
                    if padded {
                        // Cheap blurred drop shadow: stacked, slightly offset rounded rects.
                        for i in 0..10 {
                            let grow = spread * (i + 1) as f32 / 10.0;
                            let alpha =
                                ((38.0 * (1.0 - i as f32 / 10.0) / 2.2) as i32) as f32 / 255.0;
                            let rect = (
                                bx - grow,
                                by - grow + spread * 0.35,
                                w + 2.0 * grow,
                                w + 2.0 * grow,
                            );
                            if in_rounded(x, y, rect, radius + grow) {
                                sample.over([8.0, 10.0, 30.0], alpha);
                            }
                        }
                    }
                    if in_rounded(x, y, body, radius) {
                        let t = ((x - bx) * w + (y - by) * w) / diagonal;
                        let (color, _) = gradient(&tile_stops, t);
                        sample.over(color, 1.0);
                        // Gentle top light gives the flat tile some depth without looking glossy.
                        let (light, alpha) = gradient(&light_stops, (y - by) / w);
                        sample.over(light, alpha);
                    }
                    let inside_eye = |y: f32| (y - cy).abs() < eye_half[column];
                    if inside_eye(y - w * 0.018) {
                        sample.over([20.0, 18.0, 70.0], 60.0 / 255.0);
                    }
                    if inside_eye(y) {
                        sample.over([255.0; 3], 1.0);
                        if in_circle(x, y, (cx, cy), iris_r) {
                            let t = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() / iris_r;
                            sample.over(gradient(&iris_stops, t).0, 1.0);
                        }
                        let ring = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() - iris_r * 0.82;
                        if ring.abs() <= ring_width / 2.0 {
                            sample.over(ACCENT_2, 170.0 / 255.0);
                        }
                        if in_circle(x, y, (cx, cy), w * 0.068) {
                            sample.over(hex_color(0x0B0B24), 1.0);
                        }
                        if in_circle(x, y, (cx + w * 0.052, cy - w * 0.055), w * 0.03) {
                            sample.over([255.0; 3], 235.0 / 255.0);
                        }
                    }
                    for (sum, add) in total.rgb.iter_mut().zip(sample.rgb) {
                        *sum += add;
                    }
                    total.alpha += sample.alpha;
                }
            }
            let count = (samples_per_side * samples_per_side) as f32;
            let alpha = total.alpha / count;
            let straight = |sum: f32| {
                if alpha > 0.0 {
                    (sum / count / alpha).round().clamp(0.0, 255.0) as u8
                } else {
                    0
                }
            };
            image.put_pixel(
                px,
                py,
                image::Rgba([
                    straight(total.rgb[0]),
                    straight(total.rgb[1]),
                    straight(total.rgb[2]),
                    (alpha * 255.0).round() as u8,
                ]),
            );
        }
    }
    image
}

// ------------------------------------------------------------------------------- writing
fn render(renderer: &Renderer, size: u32, padded: bool) -> Result<RgbaImage, String> {
    let image = renderer(size, padded);
    if image.dimensions() != (size, size) {
        let (w, h) = image.dimensions();
        return Err(format!("renderer returned ({w}, {h}) for a {size} px icon"));
    }
    Ok(image)
}

fn save_png(image: &RgbaImage, path: &Path) -> Result<(), String> {
    let file = fs::File::create(path).map_err(|e| e.to_string())?;
    PngEncoder::new_with_quality(
        BufWriter::new(file),
        CompressionType::Best,
        FilterType::Adaptive,
    )
    .write_image(
        image.as_raw(),
        image.width(),
        image.height(),
        image::ExtendedColorType::Rgba8,
    )
    .map_err(|e| e.to_string())
}

fn icns_types(size: u32) -> &'static [IconType] {
    match size {
        32 => &[IconType::RGBA32_16x16_2x],
        64 => &[IconType::RGBA32_32x32_2x],
        128 => &[IconType::RGBA32_128x128],
        256 => &[IconType::RGBA32_256x256, IconType::RGBA32_128x128_2x],
        512 => &[IconType::RGBA32_512x512, IconType::RGBA32_256x256_2x],
        1024 => &[IconType::RGBA32_512x512_2x],
        _ => &[],
    }
}

/// Render every icon file below `root`; returns the paths written.
pub fn write_icons(renderer: &Renderer, root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut sizes: Vec<u32> = ICO_SIZES.iter().copied().chain([512, 1024]).collect();
    sizes.sort_unstable();
    sizes.dedup();
    let full = sizes
        .iter()
        .map(|&size| Ok((size, render(renderer, size, false)?)))
        .collect::<Result<Vec<_>, String>>()?;
    let full = |size: u32| &full.iter().find(|(s, _)| *s == size).expect("rendered").1;
    let mut written = Vec::new();
    let mut target = |key: &str| -> Result<PathBuf, String> {
        let path = root.join(output(key));
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        written.push(path.clone());
        Ok(path)
    };

    save_png(full(1024), &target("png")?)?;
    save_png(full(512), &target("linux")?)?;

    // Every size is rendered natively; none is downscaled from the largest, which would blur
    // the 16-32 px variants.
    let mut directory = IconDir::new(ResourceType::Icon);
    for size in ICO_SIZES {
        let icon = IconImage::from_rgba_data(size, size, full(size).as_raw().clone());
        directory.add_entry(IconDirEntry::encode_as_png(&icon).map_err(|e| e.to_string())?);
    }
    let ico = fs::File::create(target("ico")?).map_err(|e| e.to_string())?;
    directory
        .write(BufWriter::new(ico))
        .map_err(|e| e.to_string())?;

    let mut family = IconFamily::new();
    for size in ICNS_SIZES {
        let padded = render(renderer, size, true)?;
        let icon = icns::Image::from_data(icns::PixelFormat::RGBA, size, size, padded.into_raw())
            .map_err(|e| e.to_string())?;
        for &kind in icns_types(size) {
            family
                .add_icon_with_type(&icon, kind)
                .map_err(|e| e.to_string())?;
        }
    }
    let icns = fs::File::create(target("icns")?).map_err(|e| e.to_string())?;
    family
        .write(BufWriter::new(icns))
        .map_err(|e| e.to_string())?;
    Ok(written)
}

/// Icon files that do not exist below `root`.
pub fn missing_outputs(root: &Path) -> Vec<PathBuf> {
    OUTPUTS
        .iter()
        .map(|(_, rel)| root.join(rel))
        .filter(|path| !path.is_file())
        .collect()
}

#[derive(clap::Args, Debug, Clone)]
pub struct Args {
    /// Root directory for the output files (default: the repository).
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Do not render; exit 1 if any icon file is missing.
    #[arg(long)]
    pub check: bool,
}

/// Entry point of `cargo xtask make-icons`; returns the exit code.
pub fn main(args: &Args) -> i32 {
    let root = args.out.clone().unwrap_or_else(repo_root);
    if args.check {
        let missing = missing_outputs(&root);
        for path in &missing {
            println!("  missing  {}", path.display());
        }
        if !missing.is_empty() {
            eprintln!("make_icons: run `cargo xtask make-icons`");
            return 1;
        }
        println!("make_icons: OK: all {} icon files present", OUTPUTS.len());
        return 0;
    }
    match write_icons(&|size, padded| render_app_icon_png(size, padded), &root) {
        Ok(written) => {
            for path in written {
                let shown = path
                    .strip_prefix(&root)
                    .unwrap_or(&path)
                    .display()
                    .to_string();
                let bytes = fs::metadata(&path).map_or(0, |m| m.len());
                println!("  wrote {shown}  ({bytes} bytes)");
            }
            0
        }
        Err(e) => {
            eprintln!("make_icons: {e}");
            1
        }
    }
}

#[cfg(test)]
#[path = "make_icons_tests.rs"]
mod tests;
