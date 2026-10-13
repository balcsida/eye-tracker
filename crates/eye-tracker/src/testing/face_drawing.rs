//! Cartoon faces for the vision tests (port of tests/face_drawing.py).
//!
//! Face photos are not committed to the repository, but a simple drawn face (hair,
//! skin-coloured oval, eyes with irises, brows, nose and mouth) is recognised by
//! both YuNet and the MediaPipe landmark network. That lets the vision backends be
//! tested end to end on every CI runner. Drawing is deterministic, so the frames
//! are identical everywhere.
//!
//! Colours are written as BGR triples like in the Python; shapes are drawn without
//! anti-aliasing (the Gaussian blur at the end softens the edges).

use std::f64::consts::PI;

use image::{Rgb, RgbImage};
use imageproc::drawing::{
    draw_filled_circle_mut, draw_filled_ellipse_mut, draw_filled_rect_mut, draw_line_segment_mut,
    draw_polygon_mut,
};
use imageproc::filter::gaussian_blur_f32;
use imageproc::point::Point;
use imageproc::rect::Rect;

use crate::vision::Frame;

pub(crate) const WIDTH: usize = 640;
pub(crate) const HEIGHT: usize = 480;
pub(crate) const BACKGROUND: [u8; 3] = [200, 205, 210];

/// The keyword arguments of `draw_face`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FaceParams {
    pub cx: i32,
    pub cy: i32,
    pub scale: f64,
    /// Horizontal shift of both irises (pixels at scale 1).
    pub iris_dx: f64,
    pub blur: f32,
}

impl Default for FaceParams {
    fn default() -> Self {
        Self {
            cx: (WIDTH / 2) as i32,
            cy: (HEIGHT / 2) as i32,
            scale: 1.0,
            iris_dx: 0.0,
            blur: 1.2,
        }
    }
}

fn rgb(bgr: [u8; 3]) -> Rgb<u8> {
    Rgb([bgr[2], bgr[1], bgr[0]])
}

/// Points of the elliptical arc from `start` to `end` degrees (image coordinates, y down).
fn arc_points(c: (i32, i32), axes: (i32, i32), start: f64, end: f64) -> Vec<(f32, f32)> {
    let steps = (2 * (axes.0.max(axes.1)).max(8)) as usize;
    (0..=steps)
        .map(|i| {
            let a = (start + (end - start) * i as f64 / steps as f64) * PI / 180.0;
            (
                (f64::from(c.0) + f64::from(axes.0) * a.cos()) as f32,
                (f64::from(c.1) + f64::from(axes.1) * a.sin()) as f32,
            )
        })
        .collect()
}

/// A polyline `thickness` pixels wide (round joins).
fn stroke(img: &mut RgbImage, points: &[(f32, f32)], thickness: i32, color: Rgb<u8>) {
    if thickness <= 1 {
        for pair in points.windows(2) {
            draw_line_segment_mut(img, pair[0], pair[1], color);
        }
        return;
    }
    let radius = thickness / 2;
    for pair in points.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let n = ((b.0 - a.0).hypot(b.1 - a.1).ceil() as usize).max(1);
        for i in 0..=n {
            let t = i as f32 / n as f32;
            let p = (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
            draw_filled_circle_mut(img, (p.0.round() as i32, p.1.round() as i32), radius, color);
        }
    }
}

/// Draw a frontal cartoon face centred at `(cx, cy)`.
///
/// At `scale = 1` the face is 160 px wide, like a user ~60 cm from a 640x480 webcam.
/// Draws onto a copy of `image` (default: a plain 640x480 background).
pub(crate) fn draw_face(image: Option<&Frame>, params: &FaceParams) -> Frame {
    let FaceParams {
        cx,
        cy,
        scale: s,
        iris_dx,
        blur,
    } = *params;
    let base = image
        .cloned()
        .unwrap_or_else(|| Frame::filled(WIDTH, HEIGHT, BACKGROUND));
    let height = base.height as u32;
    let mut img = base.to_rgb_image();

    let p = |dx: f64, dy: f64| -> (i32, i32) {
        (
            (f64::from(cx) + dx * s).round_ties_even() as i32,
            (f64::from(cy) + dy * s).round_ties_even() as i32,
        )
    };
    let r = |v: f64| -> i32 { ((v * s).round_ties_even() as i32).max(1) };

    // Hair: the upper half of an ellipse.
    let hair = p(0.0, -40.0);
    let mut outline: Vec<Point<i32>> = arc_points(hair, (r(95.0), r(95.0)), 180.0, 360.0)
        .iter()
        .map(|&(x, y)| Point::new(x.round() as i32, y.round() as i32))
        .collect();
    outline.dedup();
    if outline.first() == outline.last() {
        outline.pop();
    }
    draw_polygon_mut(&mut img, &outline, rgb([40, 45, 60]));
    // Face.
    draw_filled_ellipse_mut(
        &mut img,
        p(0.0, 0.0),
        r(80.0),
        r(105.0),
        rgb([150, 175, 215]),
    );
    // Neck.
    let (x0, y0) = p(-35.0, 95.0);
    let (x1, _) = p(35.0, 0.0);
    if x1 > x0 && (y0 as u32) < height {
        draw_filled_rect_mut(
            &mut img,
            Rect::at(x0, y0).of_size((x1 - x0 + 1) as u32, height - y0.max(0) as u32),
            rgb([140, 165, 205]),
        );
    }
    for side in [-1.0, 1.0] {
        let (ex, ey) = (32.0 * side, -15.0);
        // Brow.
        let brow = arc_points(p(ex, ey - 20.0), (r(22.0), r(6.0)), 180.0, 360.0);
        stroke(&mut img, &brow, r(4.0), rgb([40, 50, 70]));
        // Eye: sclera, iris, pupil, outline.
        draw_filled_ellipse_mut(&mut img, p(ex, ey), r(18.0), r(8.0), rgb([235, 235, 235]));
        draw_filled_circle_mut(&mut img, p(ex + iris_dx, ey), r(7.0), rgb([60, 80, 110]));
        draw_filled_circle_mut(&mut img, p(ex + iris_dx, ey), r(3.0), rgb([10, 10, 10]));
        let outline = arc_points(p(ex, ey), (r(18.0), r(8.0)), 0.0, 360.0);
        stroke(&mut img, &outline, 1, rgb([60, 70, 90]));
    }
    // Nose.
    let (n0, n1, n2) = (p(0.0, -10.0), p(-12.0, 25.0), p(12.0, 25.0));
    let nose = [n0, n1, n2].map(|(x, y)| (x as f32, y as f32));
    stroke(&mut img, &nose, r(2.0), rgb([110, 130, 170]));
    draw_filled_circle_mut(&mut img, p(-7.0, 25.0), r(3.0), rgb([90, 105, 150]));
    draw_filled_circle_mut(&mut img, p(7.0, 25.0), r(3.0), rgb([90, 105, 150]));
    // Mouth: the lower half of an ellipse.
    let mouth = arc_points(p(0.0, 55.0), (r(28.0), r(10.0)), 0.0, 180.0);
    stroke(&mut img, &mouth, r(4.0), rgb([70, 70, 160]));

    if blur > 0.0 {
        img = gaussian_blur_f32(&img, blur);
    }
    Frame::from_rgb_image(&img)
}

/// The user (left of centre) and a small onlooker's face in the top right.
pub(crate) fn two_faces(onlooker_scale: f64) -> Frame {
    let mut canvas = draw_face(
        None,
        &FaceParams {
            cx: 260,
            ..FaceParams::default()
        },
    );
    let onlooker = draw_face(
        None,
        &FaceParams {
            cx: 560,
            cy: 150,
            scale: onlooker_scale,
            ..FaceParams::default()
        },
    );
    let radius = (200.0 * onlooker_scale).round_ties_even() as i64;
    for y in 0..canvas.height {
        for x in 0..canvas.width {
            let (dx, dy) = (x as i64 - 560, y as i64 - 150);
            if dx * dx + dy * dy <= radius * radius {
                canvas.set_pixel(x, y, onlooker.pixel(x, y));
            }
        }
    }
    canvas
}

/// `image` rotated counter-clockwise (as seen on screen) about its centre; the border
/// replicates the edge pixels.
pub(crate) fn rotate(image: &Frame, degrees: f64) -> Frame {
    let (w, h) = (image.width, image.height);
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    let (sin, cos) = (degrees * PI / 180.0).sin_cos();
    let mut out = Frame::filled(w, h, [0; 3]);
    let sample = |x: f64, y: f64| -> [f64; 3] {
        let x = x.clamp(0.0, (w - 1) as f64);
        let y = y.clamp(0.0, (h - 1) as f64);
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
        let (fx, fy) = (x - x0 as f64, y - y0 as f64);
        let mut v = [0.0; 3];
        for (c, slot) in v.iter_mut().enumerate() {
            let top = f64::from(image.pixel(x0, y0)[c]) * (1.0 - fx)
                + f64::from(image.pixel(x1, y0)[c]) * fx;
            let bottom = f64::from(image.pixel(x0, y1)[c]) * (1.0 - fx)
                + f64::from(image.pixel(x1, y1)[c]) * fx;
            *slot = top * (1.0 - fy) + bottom * fy;
        }
        v
    };
    for y in 0..h {
        for x in 0..w {
            // Inverse of cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1).
            let (dx, dy) = (x as f64 - cx, y as f64 - cy);
            let v = sample(cos * dx - sin * dy + cx, sin * dx + cos * dy + cy);
            out.set_pixel(x, y, v.map(|c| c.round().clamp(0.0, 255.0) as u8));
        }
    }
    out
}

// The Python module has no tests of its own; these only check the port draws what it should
// (the vision backends' tests, which port later, are the real consumers).
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_draw_face_is_deterministic_and_keeps_the_background_corner() {
        let a = draw_face(None, &FaceParams::default());
        assert_eq!(a, draw_face(None, &FaceParams::default()));
        assert_eq!((a.width, a.height), (WIDTH, HEIGHT));
        assert_eq!(a.pixel(2, 2), BACKGROUND);
        assert_ne!(a.pixel(WIDTH / 2, HEIGHT / 2), BACKGROUND);
    }

    #[test]
    fn test_iris_dx_moves_the_irises() {
        let centred = draw_face(None, &FaceParams::default());
        let shifted = draw_face(
            None,
            &FaceParams {
                iris_dx: 4.0,
                ..FaceParams::default()
            },
        );
        assert_ne!(centred, shifted);
    }

    #[test]
    fn test_two_faces_adds_the_onlooker() {
        let user = draw_face(
            None,
            &FaceParams {
                cx: 260,
                ..FaceParams::default()
            },
        );
        assert_ne!(two_faces(0.22), user);
    }

    #[test]
    fn test_rotate_zero_is_identity_and_ninety_turns_the_image() {
        let face = draw_face(None, &FaceParams::default());
        assert_eq!(rotate(&face, 0.0), face);
        let turned = rotate(&face, 90.0);
        assert_eq!(turned.pixel(2, 2), BACKGROUND);
        assert_ne!(turned, face);
    }
}
