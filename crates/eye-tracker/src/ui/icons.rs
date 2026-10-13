//! Icons drawn in code: no image files, crisp at every size.
//!
//! * [`app_icon`] - the application icon: a rounded square with the brand's indigo→cyan gradient
//!   and a white eye.
//! * [`tray_icon`] - a monochrome eye glyph per [`TrackingState`] (tracking: accent pupil;
//!   paused: pause bars; privacy: slashed eye; away: closed eye; camera error: red badge; needs
//!   calibration: amber badge).
//! * [`render_app_icon_png`] / [`render_tray_image`] - the same drawings as one [`Rgba`] image
//!   (used by `cargo xtask make-icons` and the tests).
//! * [`paint_lock`] / [`status_dot_icon`] - small glyphs for other widgets.
//!
//! Every size is painted from vector paths rather than scaled from one bitmap, so 16 px tray
//! icons stay sharp. Tray glyphs are drawn on a 16-unit grid; on macOS they are template images
//! (`IconSet::is_mask`) that the menu bar tints itself. Elsewhere the glyph colour follows the
//! taskbar ([`util::system_tray_is_dark`]) and a faint halo in the opposite colour keeps it
//! readable if that guess is wrong.
//!
//! Everything is rasterised with `tiny-skia` into [`Rgba`] buffers that the tray backends and
//! egui textures take.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use parking_lot::Mutex;
use thiserror::Error;
use tiny_skia::{
    BlendMode, FillRule, LineCap, LineJoin, Mask, Paint, Path, PathBuilder, Pixmap, PixmapPaint,
    Shader, Stroke, Transform,
};

use super::util::{self, Color8};
use crate::types::TrackingState;

/// Pixel sizes rendered into the tray icon: 16-24 px at 100-150 %, the macOS menu bar (18/22 pt
/// at 1x and 2x) and larger sizes for high-DPI taskbars.
pub const TRAY_SIZES: [u32; 12] = [16, 18, 20, 22, 24, 28, 32, 36, 40, 44, 48, 64];
/// Pixel sizes rendered into the application icon.
pub const APP_ICON_SIZES: [u32; 11] = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256, 512];
/// Pixel sizes of the small status dot.
const DOT_SIZES: [u32; 5] = [16, 20, 24, 32, 48];

const GRID: f32 = 16.0; // design grid of the tray glyphs
const STROKE: f32 = 1.5; // lid stroke on that grid

/// An RGBA8 image with straight (not premultiplied) alpha, row-major.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rgba {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl Rgba {
    /// The pixel at (`x`, `y`) as `[r, g, b, a]`.
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let at = (y as usize * self.width as usize + x as usize) * 4;
        [
            self.data[at],
            self.data[at + 1],
            self.data[at + 2],
            self.data[at + 3],
        ]
    }

    /// The image as an egui texture source.
    pub fn to_color_image(&self) -> egui::ColorImage {
        egui::ColorImage::from_rgba_unmultiplied(
            [self.width as usize, self.height as usize],
            &self.data,
        )
    }

    /// The image as a window icon.
    pub fn to_icon_data(&self) -> egui::IconData {
        egui::IconData {
            rgba: self.data.clone(),
            width: self.width,
            height: self.height,
        }
    }
}

/// An icon with one image per size (a `QIcon`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IconSet {
    pub images: Vec<Rgba>,
    /// A template image: only the alpha channel matters and the system picks the colour (macOS
    /// menu bar).
    pub is_mask: bool,
}

impl IconSet {
    /// The smallest image that is at least `size` pixels wide, else the largest.
    pub fn best(&self, size: u32) -> Option<&Rgba> {
        self.images
            .iter()
            .find(|image| image.width >= size)
            .or_else(|| self.images.last())
    }
}

/// Raised for a size below one pixel.
#[derive(Debug, Error, PartialEq, Eq)]
#[error("size must be positive")]
pub struct SizeError;

/// (base glyph, overlay) per state. Overlays are drawn over a knocked-out gap so they stay
/// legible at 16 px and in single-colour (template) mode.
fn tray_style(state: TrackingState) -> (Base, Option<Overlay>) {
    match state {
        TrackingState::Starting => (Base::Starting, None),
        TrackingState::NeedsCalibration => (Base::Open, Some(Overlay::Warning)),
        TrackingState::Calibrating => (Base::Calibrating, None),
        TrackingState::Tracking => (Base::Tracking, None),
        TrackingState::Paused | TrackingState::Yielded => (Base::Pause, None),
        TrackingState::Privacy => (Base::Open, Some(Overlay::Slash)),
        TrackingState::Away | TrackingState::Locked => (Base::Closed, None),
        TrackingState::CameraError => (Base::Open, Some(Overlay::Error)),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Base {
    Starting,
    Open,
    Calibrating,
    Tracking,
    Pause,
    Closed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Overlay {
    Slash,
    Error,
    Warning,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CacheKey {
    Tray(TrackingState, bool, bool),
    App,
    Dot(String),
}

static ICON_CACHE: LazyLock<Mutex<HashMap<CacheKey, Arc<IconSet>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Forget cached icons (e.g. after a theme change).
pub fn clear_cache() {
    ICON_CACHE.lock().clear();
}

// ----------------------------------------------------------------------- primitives
fn new_pixmap(size: u32) -> Result<Pixmap, SizeError> {
    // A pixmap this large is refused by tiny-skia; none of the sizes used come near it.
    Pixmap::new(size, size).ok_or(SizeError)
}

/// The pixmap's (premultiplied) pixels as a straight-alpha image.
pub(crate) fn to_rgba(pixmap: &Pixmap) -> Rgba {
    let mut data = Vec::with_capacity(pixmap.data().len());
    for pixel in pixmap.pixels() {
        let straight = pixel.demultiply();
        data.extend([
            straight.red(),
            straight.green(),
            straight.blue(),
            straight.alpha(),
        ]);
    }
    Rgba {
        width: pixmap.width(),
        height: pixmap.height(),
        data,
    }
}

/// The finished path. Every builder here has segments, so `finish` cannot fail.
fn build(builder: PathBuilder) -> Path {
    builder.finish().expect("a path with segments")
}

/// Closed eye outline with pointed corners, filling the rectangle's height.
fn almond(left: f32, top: f32, width: f32, height: f32, pinch: f32) -> Path {
    let right = left + width;
    let cy = top + height / 2.0;
    // A cubic whose two control points sit at +-k peaks at 0.75 k.
    let k = height / 2.0 / 0.75;
    let dx = width * pinch;
    let mut pb = PathBuilder::new();
    pb.move_to(left, cy);
    pb.cubic_to(left + dx, cy - k, right - dx, cy - k, right, cy);
    pb.cubic_to(right - dx, cy + k, left + dx, cy + k, left, cy);
    pb.close();
    build(pb)
}

/// Outline of `path` stroked with round caps and joins, as a fillable path. `resolution` is the
/// scale the path will be drawn at (for the curve precision).
fn stroke(path: &Path, width: f32, resolution: f32) -> Path {
    let style = Stroke {
        width,
        line_cap: LineCap::Round,
        line_join: LineJoin::Round,
        ..Stroke::default()
    };
    path.stroke(&style, resolution)
        .unwrap_or_else(|| path.clone())
}

fn circle(cx: f32, cy: f32, r: f32) -> Path {
    PathBuilder::from_circle(cx, cy, r).unwrap_or_else(|| build(PathBuilder::new()))
}

fn line(x0: f32, y0: f32, x1: f32, y1: f32) -> Path {
    let mut pb = PathBuilder::new();
    pb.move_to(x0, y0);
    pb.line_to(x1, y1);
    build(pb)
}

/// A rounded rectangle; like Qt the radius is limited to half of each side.
fn rounded(x: f32, y: f32, w: f32, h: f32, radius: f32) -> Path {
    let r = radius.min(w / 2.0).min(h / 2.0).max(0.0);
    let k = r * 0.552_284_8;
    let (right, bottom) = (x + w, y + h);
    let mut pb = PathBuilder::new();
    pb.move_to(x + r, y);
    pb.line_to(right - r, y);
    pb.cubic_to(right - r + k, y, right, y + r - k, right, y + r);
    pb.line_to(right, bottom - r);
    pb.cubic_to(
        right,
        bottom - r + k,
        right - r + k,
        bottom,
        right - r,
        bottom,
    );
    pb.line_to(x + r, bottom);
    pb.cubic_to(x + r - k, bottom, x, bottom - r + k, x, bottom - r);
    pb.line_to(x, y + r);
    pb.cubic_to(x, y + r - k, x + r - k, y, x + r, y);
    pb.close();
    build(pb)
}

fn paint_color(color: Color8) -> Paint<'static> {
    let mut paint = Paint {
        anti_alias: true,
        ..Paint::default()
    };
    paint.set_color(util::skia_color(color));
    paint
}

fn paint_shader(shader: Shader<'static>) -> Paint<'static> {
    Paint {
        shader,
        anti_alias: true,
        ..Paint::default()
    }
}

/// The point at fraction `t` of the *length* of a cubic Bézier (Qt's `pointAtPercent`).
fn point_at_percent(p: [(f32, f32); 4], t: f32) -> (f32, f32) {
    const STEPS: usize = 256;
    let at = |u: f32| {
        let v = 1.0 - u;
        let (a, b, c, d) = (v * v * v, 3.0 * v * v * u, 3.0 * v * u * u, u * u * u);
        (
            a * p[0].0 + b * p[1].0 + c * p[2].0 + d * p[3].0,
            a * p[0].1 + b * p[1].1 + c * p[2].1 + d * p[3].1,
        )
    };
    let points: Vec<(f32, f32)> = (0..=STEPS).map(|i| at(i as f32 / STEPS as f32)).collect();
    let lengths: Vec<f32> = points
        .windows(2)
        .map(|w| (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1))
        .collect();
    let mut remaining = t * lengths.iter().sum::<f32>();
    for (i, length) in lengths.iter().enumerate() {
        if remaining <= *length || i + 1 == lengths.len() {
            let f = if *length > 0.0 {
                remaining / length
            } else {
                0.0
            };
            let (a, b) = (points[i], points[i + 1]);
            return (a.0 + (b.0 - a.0) * f, a.1 + (b.1 - a.1) * f);
        }
        remaining -= length;
    }
    points[STEPS]
}

// ------------------------------------------------------------------------ tray glyph
#[derive(Default)]
struct Glyph {
    mono: Vec<Path>,
    accent: Vec<Path>,
    overlay: Vec<Path>,
    knockout: Vec<Path>,
}

fn tray_glyph(base: Base, overlay: Option<Overlay>, resolution: f32) -> Glyph {
    let mut glyph = Glyph::default();
    let stroked = |path: &Path, width: f32| stroke(path, width, resolution);
    if base == Base::Closed {
        let curve = [(1.6, 6.6), (4.4, 11.4), (11.6, 11.4), (14.4, 6.6)];
        let mut pb = PathBuilder::new();
        pb.move_to(curve[0].0, curve[0].1);
        pb.cubic_to(
            curve[1].0, curve[1].1, curve[2].0, curve[2].1, curve[3].0, curve[3].1,
        );
        glyph.mono.push(stroked(&build(pb), STROKE));
        for (t, dx, dy) in [(0.2, -1.0, 1.7), (0.5, 0.0, 2.1), (0.8, 1.0, 1.7)] {
            let (x, y) = point_at_percent(curve, t);
            glyph.mono.push(stroked(&line(x, y, x + dx, y + dy), 1.3));
        }
    } else {
        // Apex 3.4..12.6 leaves a clear gap between lids and a 2.4-unit pupil.
        glyph
            .mono
            .push(stroked(&almond(1.25, 3.4, 13.5, 9.2, 0.22), STROKE));
        match base {
            Base::Open => glyph.mono.push(circle(8.0, 8.0, 2.4)),
            Base::Starting => glyph.mono.push(stroked(&circle(8.0, 8.0, 2.0), 1.2)),
            Base::Tracking => glyph.accent.push(circle(8.0, 8.0, 2.6)),
            Base::Calibrating => {
                glyph.accent.push(stroked(&circle(8.0, 8.0, 2.35), 1.1));
                glyph.accent.push(circle(8.0, 8.0, 0.95));
            }
            Base::Pause => {
                // Whole-unit x edges keep the bars sharp at 16 px.
                glyph.mono.push(rounded(5.0, 5.5, 2.0, 5.0, 0.6));
                glyph.mono.push(rounded(9.0, 5.5, 2.0, 5.0, 0.6));
            }
            Base::Closed => {}
        }
    }

    match overlay {
        Some(Overlay::Slash) => {
            let slash = line(2.4, 2.4, 13.6, 13.6);
            glyph.overlay.push(stroked(&slash, STROKE));
            glyph.knockout.push(stroked(&slash, STROKE + 2.6));
        }
        Some(Overlay::Error | Overlay::Warning) => {
            glyph.overlay.push(circle(12.55, 12.55, 2.85));
            glyph.knockout.push(circle(12.55, 12.55, 2.85 + 1.25));
        }
        None => {}
    }
    glyph
}

fn paint_tray_glyph(pixmap: &mut Pixmap, state: TrackingState, size: u32, dark: bool, mask: bool) {
    let (base, overlay) = tray_style(state);
    let scale = size as f32 / GRID;
    let glyph = tray_glyph(base, overlay, scale);

    // Template image: only alpha matters; the menu bar picks the colour.
    let black = [0, 0, 0, 255];
    let (fg, accent_paint, halo, badge) = if mask {
        (black, paint_color(black), None, [black, black])
    } else {
        let fg = util::qcolor(if dark { "#F3F4F6" } else { "#1F2937" }, None);
        let gradient = util::gradient(
            (5.2, 5.2),
            (10.8, 10.8),
            &[
                (
                    0.0,
                    util::qcolor(
                        if dark {
                            util::ACCENT_LIGHT
                        } else {
                            util::ACCENT_DEEP
                        },
                        None,
                    ),
                ),
                (
                    1.0,
                    util::qcolor(
                        if dark {
                            util::ACCENT_2
                        } else {
                            util::ACCENT_2_DEEP
                        },
                        None,
                    ),
                ),
            ],
        );
        let halo = if dark {
            [0, 0, 0, 110]
        } else {
            [255, 255, 255, 150]
        };
        (
            fg,
            paint_shader(gradient),
            Some(halo),
            [
                util::qcolor(util::DANGER, None),
                util::qcolor(util::WARNING, None),
            ],
        )
    };
    let transform = Transform::from_scale(scale, scale);
    let halo_stroke = Stroke {
        width: 2.0 * GRID / size as f32, // one device pixel on each side of the edge
        line_cap: LineCap::Round,
        line_join: LineJoin::Round,
        ..Stroke::default()
    };
    let draw_halo = |pixmap: &mut Pixmap, paths: &[&Path]| {
        if let Some(color) = halo {
            let paint = paint_color(color);
            for path in paths {
                pixmap.stroke_path(path, &paint, &halo_stroke, transform, None);
            }
        }
    };

    let body: Vec<&Path> = glyph.mono.iter().chain(&glyph.accent).collect();
    draw_halo(pixmap, &body);
    let fg_paint = paint_color(fg);
    for path in &glyph.mono {
        pixmap.fill_path(path, &fg_paint, FillRule::Winding, transform, None);
    }
    for path in &glyph.accent {
        pixmap.fill_path(path, &accent_paint, FillRule::Winding, transform, None);
    }

    if !glyph.overlay.is_empty() {
        let clear = Paint {
            blend_mode: BlendMode::Clear,
            anti_alias: true,
            ..Paint::default()
        };
        for path in &glyph.knockout {
            pixmap.fill_path(path, &clear, FillRule::Winding, transform, None);
        }
        let color = match overlay {
            Some(Overlay::Error) => badge[0],
            Some(Overlay::Warning) => badge[1],
            _ => fg,
        };
        draw_halo(pixmap, &glyph.overlay.iter().collect::<Vec<_>>());
        let paint = paint_color(color);
        for path in &glyph.overlay {
            pixmap.fill_path(path, &paint, FillRule::Winding, transform, None);
        }
    }
}

/// The tray glyph for `state` as a `size`×`size` image.
///
/// `dark` means a dark taskbar (light glyph); `mask` renders a black template image for the macOS
/// menu bar.
pub fn render_tray_image(
    state: TrackingState,
    size: u32,
    dark: bool,
    mask: bool,
) -> Result<Rgba, SizeError> {
    if size < 1 {
        return Err(SizeError);
    }
    let mut pixmap = new_pixmap(size)?;
    paint_tray_glyph(&mut pixmap, state, size, dark, mask);
    Ok(to_rgba(&pixmap))
}

/// Monochrome tray icon for `state` with an image for every tray size.
///
/// `mask: None` uses template images on macOS only; `dark: None` asks
/// [`util::system_tray_is_dark`]. Icons are cached per combination.
pub fn tray_icon(state: TrackingState, dark: Option<bool>, mask: Option<bool>) -> Arc<IconSet> {
    let mask = mask.unwrap_or(cfg!(target_os = "macos"));
    let dark = dark.unwrap_or_else(|| mask || util::system_tray_is_dark());
    let key = CacheKey::Tray(state, dark, mask);
    if let Some(icon) = ICON_CACHE.lock().get(&key) {
        return icon.clone();
    }
    let images = TRAY_SIZES
        .iter()
        .filter_map(|&size| render_tray_image(state, size, dark, mask).ok())
        .collect();
    let icon = Arc::new(IconSet {
        images,
        is_mask: mask,
    });
    ICON_CACHE.lock().insert(key, icon.clone());
    icon
}

// ------------------------------------------------------------------------- app icon
/// Cheap blurred drop shadow: stacked, slightly offset translucent rounded rects.
fn soft_shadow(pixmap: &mut Pixmap, rect: (f32, f32, f32, f32), radius: f32, spread: f32) {
    let steps = 10;
    for i in 0..steps {
        let grow = spread * (i + 1) as f32 / steps as f32;
        let alpha = (38.0 * (1.0 - i as f32 / steps as f32) / 2.2) as u8;
        let dy = spread * 0.35;
        let path = rounded(
            rect.0 - grow,
            rect.1 - grow + dy,
            rect.2 + 2.0 * grow,
            rect.3 + 2.0 * grow,
            radius + grow,
        );
        pixmap.fill_path(
            &path,
            &paint_color([8, 10, 30, alpha]),
            FillRule::Winding,
            Transform::identity(),
            None,
        );
    }
}

fn paint_app_icon(pixmap: &mut Pixmap, size: u32, padded: bool) {
    let s = size as f32;
    let (bx, by, w) = if padded {
        // macOS icon grid: an 824/1024 rounded square with room for its shadow.
        let side = s * 824.0 / 1024.0;
        ((s - side) / 2.0, (s - side) / 2.0 - s * 0.01, side)
    } else {
        (0.0, 0.0, s)
    };
    let radius = w * 0.225;
    if padded {
        soft_shadow(pixmap, (bx, by, w, w), radius, s * 0.03);
    }

    let identity = Transform::identity();
    let tile = rounded(bx, by, w, w, radius);
    let tile_gradient = util::accent_gradient((bx, by), (bx + w, by + w), false);
    pixmap.fill_path(
        &tile,
        &paint_shader(tile_gradient),
        FillRule::Winding,
        identity,
        None,
    );
    // Gentle top light gives the flat tile some depth without looking glossy.
    let light = util::gradient(
        (bx, by),
        (bx, by + w),
        &[
            (0.0, [255, 255, 255, 56]),
            (0.55, [255, 255, 255, 0]),
            (1.0, [255, 255, 255, 0]),
        ],
    );
    pixmap.fill_path(
        &tile,
        &paint_shader(light),
        FillRule::Winding,
        identity,
        None,
    );

    let (cx, cy) = (bx + w / 2.0, by + w / 2.0);
    let eye = almond(bx + w * 0.15, cy - w * 0.215, w * 0.70, w * 0.43, 0.23);
    let shadow = eye
        .clone()
        .transform(Transform::from_translate(0.0, w * 0.018))
        .unwrap_or_else(|| eye.clone());
    pixmap.fill_path(
        &shadow,
        &paint_color([20, 18, 70, 60]),
        FillRule::Winding,
        identity,
        None,
    );
    pixmap.fill_path(
        &eye,
        &paint_color([255, 255, 255, 255]),
        FillRule::Winding,
        identity,
        None,
    );

    let Some(mut clip) = Mask::new(pixmap.width(), pixmap.height()) else {
        return;
    };
    clip.fill_path(&eye, FillRule::Winding, true, identity);
    let iris_r = w * 0.155;
    let iris = util::radial_gradient(
        (cx, cy),
        iris_r,
        &[
            (0.0, util::qcolor("#3B3A98", None)),
            (0.75, util::qcolor("#26236E", None)),
            (1.0, util::qcolor("#1B1850", None)),
        ],
    );
    pixmap.fill_path(
        &circle(cx, cy, iris_r),
        &paint_shader(iris),
        FillRule::Winding,
        identity,
        Some(&clip),
    );
    let ring = Stroke {
        width: (w * 0.012).max(1.0),
        ..Stroke::default()
    };
    pixmap.stroke_path(
        &circle(cx, cy, iris_r * 0.82),
        &paint_color(util::qcolor(util::ACCENT_2, Some(170))),
        &ring,
        identity,
        Some(&clip),
    );
    pixmap.fill_path(
        &circle(cx, cy, w * 0.068),
        &paint_color(util::qcolor("#0B0B24", None)),
        FillRule::Winding,
        identity,
        Some(&clip),
    );
    pixmap.fill_path(
        &circle(cx + w * 0.052, cy - w * 0.055, w * 0.03),
        &paint_color([255, 255, 255, 235]),
        FillRule::Winding,
        identity,
        Some(&clip),
    );
}

/// The application icon as a `size`×`size` image.
///
/// `padded` follows the macOS icon grid (smaller tile with a soft shadow); use it for `.icns`.
/// Windows and Linux icons use the full square.
pub fn render_app_icon_png(size: u32, padded: bool) -> Result<Rgba, SizeError> {
    if size < 1 {
        return Err(SizeError);
    }
    let mut pixmap = new_pixmap(size)?;
    paint_app_icon(&mut pixmap, size, padded);
    Ok(to_rgba(&pixmap))
}

/// The application icon with an image for every common size (cached).
pub fn app_icon() -> Arc<IconSet> {
    if let Some(icon) = ICON_CACHE.lock().get(&CacheKey::App) {
        return icon.clone();
    }
    let images = APP_ICON_SIZES
        .iter()
        .filter_map(|&size| render_app_icon_png(size, false).ok())
        .collect();
    let icon = Arc::new(IconSet {
        images,
        is_mask: false,
    });
    ICON_CACHE.lock().insert(CacheKey::App, icon.clone());
    icon
}

// --------------------------------------------------------------------- small glyphs
/// Draw a padlock filling the rectangle (`x`, `y`, `w`, `h`) of `pixmap` in `color` (keyhole cut
/// out, so any background shows).
pub fn paint_lock(pixmap: &mut Pixmap, rect: (f32, f32, f32, f32), color: Color8) {
    let (left, top, w, h) = rect;
    let Some(mut layer) = Pixmap::new(pixmap.width(), pixmap.height()) else {
        return;
    };
    let identity = Transform::identity();
    let body = (left + w * 0.14, top + h * 0.44, w * 0.72, h * 0.54);
    let centre_x = body.0 + body.2 / 2.0;
    let hole_y = body.1 + body.3 * 0.42;

    let (rx, ry) = (w * 0.2, h * 0.22);
    let (arc_x, arc_y) = (left + w * 0.5, top + h * 0.28);
    let k = 0.552_284_8;
    let mut shackle = PathBuilder::new();
    shackle.move_to(left + w * 0.3, body.1 + h * 0.02);
    shackle.line_to(left + w * 0.3, arc_y);
    // The half ellipse over the top, from the left end to the right one.
    shackle.cubic_to(
        arc_x - rx,
        arc_y - ry * k,
        arc_x - rx * k,
        arc_y - ry,
        arc_x,
        arc_y - ry,
    );
    shackle.cubic_to(
        arc_x + rx * k,
        arc_y - ry,
        arc_x + rx,
        arc_y - ry * k,
        arc_x + rx,
        arc_y,
    );
    shackle.line_to(left + w * 0.7, body.1 + h * 0.02);
    let paint = paint_color(color);
    layer.fill_path(
        &stroke(&build(shackle), w * 0.1, 1.0),
        &paint,
        FillRule::Winding,
        identity,
        None,
    );
    layer.fill_path(
        &rounded(body.0, body.1, body.2, body.3, w * 0.12),
        &paint,
        FillRule::Winding,
        identity,
        None,
    );
    // The keyhole, a circle over a slot, is cleared out of the body.
    let clear = Paint {
        blend_mode: BlendMode::Clear,
        anti_alias: true,
        ..Paint::default()
    };
    layer.fill_path(
        &circle(centre_x, hole_y, w * 0.075),
        &clear,
        FillRule::Winding,
        identity,
        None,
    );
    layer.fill_path(
        &rounded(centre_x - w * 0.03, hole_y, w * 0.06, h * 0.17, w * 0.03),
        &clear,
        FillRule::Winding,
        identity,
        None,
    );
    pixmap.draw_pixmap(
        0,
        0,
        layer.as_ref(),
        &PixmapPaint::default(),
        identity,
        None,
    );
}

/// A padlock of `size`×`size` pixels in `color` on a transparent background.
pub fn render_lock(size: u32, color: Color8) -> Result<Rgba, SizeError> {
    if size < 1 {
        return Err(SizeError);
    }
    let mut pixmap = new_pixmap(size)?;
    paint_lock(&mut pixmap, (0.0, 0.0, size as f32, size as f32), color);
    Ok(to_rgba(&pixmap))
}

/// A small filled circle (menu status indicator) in `color` (`#RRGGBB`).
pub fn status_dot_icon(color: &str) -> Arc<IconSet> {
    let key = CacheKey::Dot(color.to_string());
    if let Some(icon) = ICON_CACHE.lock().get(&key) {
        return icon.clone();
    }
    let images = DOT_SIZES
        .iter()
        .filter_map(|&size| {
            let mut pixmap = new_pixmap(size).ok()?;
            let r = size as f32 * 0.22;
            let centre = size as f32 / 2.0;
            pixmap.fill_path(
                &circle(centre, centre, r),
                &paint_color(util::qcolor(color, None)),
                FillRule::Winding,
                Transform::identity(),
                None,
            );
            Some(to_rgba(&pixmap))
        })
        .collect();
    let icon = Arc::new(IconSet {
        images,
        is_mask: false,
    });
    ICON_CACHE.lock().insert(key, icon.clone());
    icon
}

#[cfg(test)]
#[path = "icons_tests.rs"]
mod tests;
