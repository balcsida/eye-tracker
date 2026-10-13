// Ported from tests/test_motion.py. Not ported: test_other_pixel_formats (gray, BGRA and
// single-channel frames: a `Frame` is always BGR8, the camera layer converts) and the non-BGR
// frames of test_thumbnail_formats. Random inputs come from a small seeded generator instead of
// numpy's, so the frames differ from the Python ones but are as reproducible. The eye drawing of
// `eyes_frame` has no anti-aliasing or sub-pixel circle centres (imageproc); the blur and the
// noise soften it the same way.

use approx::assert_relative_eq;
use image::imageops::{FilterType, resize};
use imageproc::drawing::{draw_filled_circle_mut, draw_filled_ellipse_mut, draw_line_segment_mut};
use imageproc::filter::gaussian_blur_f32;

use super::*;

const FACE_BOX: NormBox = (0.3, 0.2, 0.4, 0.5);

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03)
    }

    /// xorshift64*: uniform in `[0, 1)`.
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Integer in `low..high`.
    fn int(&mut self, low: i32, high: i32) -> i32 {
        low + (self.uniform() * f64::from(high - low)) as i32
    }

    fn normal(&mut self, sigma: f64) -> f64 {
        let (u, v) = (1.0 - self.uniform(), self.uniform());
        sigma * (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// A textured 640x480 BGR frame (smooth, so small shifts matter little).
fn base() -> Frame {
    let mut rng = Rng::new(1234);
    let noise = image::RgbImage::from_fn(32, 24, |_, _| {
        image::Rgb([0, 0, 0].map(|_: u8| rng.int(0, 255) as u8))
    });
    Frame::from_rgb_image(&resize(&noise, 640, 480, FilterType::CatmullRom))
}

/// `cv2.add(frame, full_like(frame, value))`
fn add(frame: &Frame, value: u8) -> Frame {
    let mut out = frame.clone();
    out.data
        .iter_mut()
        .for_each(|v| *v = v.saturating_add(value));
    out
}

fn fill(frame: &mut Frame, x: std::ops::Range<usize>, y: std::ops::Range<usize>, value: u8) {
    for row in y {
        for col in x.clone() {
            frame.set_pixel(col, row, [value; 3]);
        }
    }
}

/// Paint two small squares where the eyes of FACE_BOX would be (like moving irises).
fn with_patches(frame: &Frame, value: u8) -> Frame {
    let mut out = frame.clone();
    // Eye band of FACE_BOX spans x 205..435, y 132..240 in a 640x480 frame.
    for cx in [270, 370] {
        fill(&mut out, cx..cx + 16, 180..196, value);
    }
    out
}

#[test]
fn test_first_frame_is_always_processed() {
    let mut gate = MotionGate::default();
    assert!(gate.should_process(&base(), 0.0));
}

#[test]
fn test_identical_frame_is_skipped() {
    let base = base();
    let mut gate = MotionGate::default();
    gate.mark_processed(&base, 0.0, None);
    assert!(!gate.should_process(&base.clone(), 0.1));
    assert_eq!(gate.last_motion(), 0.0);
}

#[test]
fn test_global_change_is_processed() {
    let base = base();
    let mut gate = MotionGate::default();
    gate.mark_processed(&base, 0.0, None);
    assert!(gate.should_process(&add(&base, 10), 0.1));
    assert!(gate.last_motion() >= 2.0);
}

#[test]
fn test_sensor_noise_is_ignored() {
    let base = base();
    let mut gate = MotionGate::default();
    gate.mark_processed(&base, 0.0, None);
    let mut rng = Rng::new(7);
    let mut noisy = base.clone();
    noisy
        .data
        .iter_mut()
        .for_each(|v| *v = (i32::from(*v) + rng.int(-4, 5)).clamp(0, 255) as u8);
    assert!(!gate.should_process(&noisy, 0.1));
    assert!(gate.last_motion() < 1.0);
}

#[test]
fn test_max_skip_forces_refresh() {
    let base = base();
    let mut gate = MotionGate::new(2.0, THUMB_SIZE, 2.0);
    gate.mark_processed(&base, 10.0, None);
    assert!(!gate.should_process(&base, 11.9));
    assert!(gate.should_process(&base, 12.0));
}

#[test]
fn test_reference_is_last_processed_frame() {
    let base = base();
    let mut gate = MotionGate::default();
    gate.mark_processed(&base, 0.0, None);
    // Slow drift: each step is small, but it accumulates against the reference.
    let step: Vec<Frame> = [1, 2, 3].iter().map(|&k| add(&base, k)).collect();
    assert!(!gate.should_process(&step[0], 0.1));
    assert!(gate.should_process(&step[2], 0.2));
    gate.mark_processed(&step[2], 0.2, None);
    assert!(!gate.should_process(&step[2], 0.3));
}

#[test]
fn test_reset_forgets_reference() {
    let base = base();
    let mut gate = MotionGate::default();
    gate.mark_processed(&base, 0.0, None);
    gate.reset();
    assert!(gate.should_process(&base, 0.1));
    assert_eq!(gate.last_motion(), 0.0);
}

#[test]
fn test_threshold_is_adjustable() {
    let base = base();
    let mut gate = MotionGate::default();
    gate.mark_processed(&base, 0.0, None);
    let brighter = add(&base, 10);
    gate.threshold = 50.0;
    assert!(!gate.should_process(&brighter, 0.1));
    gate.threshold = 5.0;
    assert!(gate.should_process(&brighter, 0.1));
}

#[test]
fn test_eye_band_detects_small_eye_movement() {
    let base = base();
    let before = with_patches(&base, 40);
    let after = with_patches(&base, 220);

    let mut without_face = MotionGate::default();
    without_face.mark_processed(&before, 0.0, None);
    assert!(!without_face.should_process(&after, 0.1));
    let global_motion = without_face.last_motion();

    let mut with_face = MotionGate::default();
    with_face.mark_processed(&before, 0.0, Some(FACE_BOX));
    assert!(with_face.should_process(&after, 0.1));
    assert!(with_face.last_motion() > global_motion);
}

#[test]
fn test_change_outside_eye_band_uses_global_threshold() {
    let base = base();
    let mut gate = MotionGate::default();
    gate.mark_processed(&base, 0.0, Some(FACE_BOX));
    let mut changed = base.clone();
    fill(&mut changed, 20..36, 400..416, 255); // small change far from the face
    assert!(!gate.should_process(&changed, 0.1));
}

#[test]
fn test_face_box_at_the_edge_is_tolerated() {
    let base = base();
    let mut gate = MotionGate::default();
    gate.mark_processed(&base, 0.0, Some((0.95, 0.95, 0.2, 0.2))); // band mostly outside
    assert!(!gate.should_process(&base, 0.1));
    gate.mark_processed(&base, 0.2, Some((0.5, 0.5, 0.0, 0.0))); // degenerate box
    assert!(!gate.should_process(&base, 0.3));
}

#[test]
fn test_face_box_is_forgotten_when_face_disappears() {
    let base = base();
    let before = with_patches(&base, 40);
    let after = with_patches(&base, 220);
    let mut gate = MotionGate::default();
    gate.mark_processed(&before, 0.0, Some(FACE_BOX));
    gate.mark_processed(&before, 0.1, None);
    assert!(!gate.should_process(&after, 0.2));
}

#[test]
fn test_mark_processed_after_should_process_same_frame() {
    let base = base();
    let mut gate = MotionGate::default();
    gate.mark_processed(&base, 0.0, None);
    let brighter = add(&base, 20);
    assert!(gate.should_process(&brighter, 0.1));
    gate.mark_processed(&brighter, 0.1, Some(FACE_BOX));
    assert!(!gate.should_process(&brighter.clone(), 0.2));
}

#[test]
fn test_frame_size_change_is_handled() {
    let base = base();
    let mut gate = MotionGate::default();
    gate.mark_processed(&base, 0.0, Some(FACE_BOX));
    let smaller = Frame::from_rgb_image(&resize(
        &base.to_rgb_image(),
        320,
        240,
        FilterType::Triangle,
    ));
    assert!(!gate.should_process(&smaller, 0.1));
}

/// A face with two eyes (sclera, dark iris) on `base`, plus its face box.
///
/// The eyes cover only a few percent of the eye band, like in a webcam frame.
fn eyes_frame(
    base: &Frame,
    iris_dx: f64,
    closed: bool,
    noise: f64,
    face_w: i32,
    seed: u64,
) -> (Frame, NormBox) {
    let mut frame = base.clone();
    let face_h = (f64::from(face_w) * 1.25).round_ties_even() as i32;
    let (x0, y0) = (320 - face_w / 2, 240 - face_h / 2);
    // Colours are in the stored BGR order.
    draw_filled_ellipse_mut(
        &mut frame,
        (320, 240),
        face_w / 2,
        face_h / 2,
        image::Rgb([150, 160, 190]),
    );
    let eye_w = 0.22 * f64::from(face_w);
    let eye_h = 0.30 * eye_w;
    let cy = f64::from(y0) + 0.40 * f64::from(face_h);
    for cx in [
        320.0 - 0.22 * f64::from(face_w),
        320.0 + 0.22 * f64::from(face_w),
    ] {
        if closed {
            let y = cy.round_ties_even() as f32;
            let (a, b) = (
                (cx - eye_w / 2.0).round_ties_even() as f32,
                (cx + eye_w / 2.0).round_ties_even() as f32,
            );
            for dy in [0.0, 1.0] {
                draw_line_segment_mut(
                    &mut frame,
                    (a, y + dy),
                    (b, y + dy),
                    image::Rgb([90, 90, 110]),
                );
            }
            continue;
        }
        let centre = (cx.round_ties_even() as i32, cy.round_ties_even() as i32);
        draw_filled_ellipse_mut(
            &mut frame,
            centre,
            (eye_w / 2.0).round_ties_even() as i32,
            (eye_h / 2.0).round_ties_even() as i32,
            image::Rgb([190, 190, 190]),
        );
        draw_filled_circle_mut(
            &mut frame,
            ((cx + iris_dx).round_ties_even() as i32, centre.1),
            (0.2 * eye_w).round_ties_even() as i32,
            image::Rgb([60, 50, 40]),
        );
    }
    let mut rng = Rng::new(seed);
    let mut noisy = frame;
    noisy.data.iter_mut().for_each(|v| {
        *v = (f64::from(*v) + rng.normal(noise))
            .round()
            .clamp(0.0, 255.0) as u8;
    });
    let blurred = gaussian_blur_f32(&noisy.to_rgb_image(), 0.8);
    let box_ = (
        f64::from(x0) / 640.0,
        f64::from(y0) / 480.0,
        f64::from(face_w) / 640.0,
        f64::from(face_h) / 480.0,
    );
    (Frame::from_rgb_image(&blurred), box_)
}

#[test]
fn test_eye_only_glance_passes_the_gate() {
    // A 4 px iris shift (an eye-only glance at another monitor) must be analysed.
    let base = base();
    for face_w in [90, 140, 220] {
        let (reference, box_) = eyes_frame(&base, 0.0, false, 1.0, face_w, 1);
        let (glance, _) = eyes_frame(&base, 4.0, false, 1.0, face_w, 2);
        let mut gate = MotionGate::default();
        gate.mark_processed(&reference, 0.0, Some(box_));
        assert!(gate.should_process(&glance, 0.1), "{face_w}");
        // The same change is invisible to the whole-frame thumbnail alone.
        let mut blind_to_eyes = MotionGate::default();
        blind_to_eyes.mark_processed(&reference, 0.0, None);
        assert!(!blind_to_eyes.should_process(&glance, 0.1), "{face_w}");
    }
}

#[test]
fn test_noisy_still_face_is_skipped() {
    // Camera noise (up to a dim-light sigma of 6) must not defeat the gate.
    let base = base();
    for noise in [1.0, 3.0, 6.0] {
        let (reference, box_) = eyes_frame(&base, 0.0, false, noise, 90, 1);
        let mut gate = MotionGate::default();
        gate.mark_processed(&reference, 0.0, Some(box_));
        for seed in 2..6 {
            let (still, _) = eyes_frame(&base, 0.0, false, noise, 90, seed);
            assert!(!gate.should_process(&still, 0.1), "{noise} {seed}");
            assert!(gate.last_motion() < 2.0);
        }
    }
}

#[test]
fn test_eyes_reopening_passes_the_gate() {
    let base = base();
    let (closed, box_) = eyes_frame(&base, 0.0, true, 1.0, 140, 1);
    let (reopened, _) = eyes_frame(&base, 0.0, false, 1.0, 140, 2);
    let mut gate = MotionGate::default();
    gate.mark_processed(&closed, 0.0, Some(box_));
    assert!(gate.should_process(&reopened, 0.1));
}

#[test]
fn test_band_score_is_reported_on_the_threshold_scale() {
    let base = base();
    let (reference, box_) = eyes_frame(&base, 0.0, false, 1.0, 140, 1);
    let (glance, _) = eyes_frame(&base, 4.0, false, 1.0, 140, 1);
    let mut gate = MotionGate::default();
    gate.mark_processed(&reference, 0.0, Some(box_));
    gate.should_process(&glance, 0.1);
    let roi_ref = roi_thumb(&reference.to_grey(), eye_band(box_)).unwrap();
    let roi_new = roi_thumb(&glance.to_grey(), eye_band(box_)).unwrap();
    let band = top_abs_diff(&roi_new, &roi_ref, BAND_TOP_FRACTION);
    assert_relative_eq!(
        gate.last_motion(),
        band / BAND_THRESHOLD_FACTOR,
        max_relative = 1e-6
    );
}

#[test]
fn test_top_abs_diff_is_robust_to_where_the_change_is() {
    let a = Plane {
        width: 32,
        height: 16,
        data: vec![0.0; 512],
    };
    let mut b = a.clone();
    for row in 2..4 {
        for col in 3..5 {
            b.data[row * 32 + col] = 100.0; // 4 of 512 cells change
        }
    }
    let small = top_abs_diff(&a, &b, BAND_TOP_FRACTION);
    assert_relative_eq!(small, 100.0 * 4.0 / 51.0, max_relative = 1e-6); // top 10 % = 51 cells
    assert_relative_eq!(
        mean_abs_diff(&a, &b),
        100.0 * 4.0 / 512.0,
        max_relative = 1e-6
    );
    let mut moved = b.clone(); // np.roll(b, 10, axis=1)
    for row in 0..16 {
        for col in 0..32 {
            moved.data[row * 32 + (col + 10) % 32] = b.data[row * 32 + col];
        }
    }
    assert_relative_eq!(
        top_abs_diff(&a, &moved, BAND_TOP_FRACTION),
        small,
        max_relative = 1e-6
    );
}

// -------------------------------------------------------------------- blindness
#[test]
fn test_thumbnail_formats() {
    let thumb = thumbnail(&base(), THUMB_SIZE);
    assert_eq!((thumb.height, thumb.width), (24, 32));
    assert_eq!(thumb.data.len(), 24 * 32);
}

#[test]
fn test_covered_lens_is_blind() {
    let black = Frame::filled(640, 480, [0; 3]);
    assert!(is_blind(&thumbnail(&black, THUMB_SIZE)));
    // A closed shutter at full sensor gain: dark, noisy, but featureless.
    let mut rng = Rng::new(0);
    let mut shutter = black;
    shutter
        .data
        .iter_mut()
        .for_each(|v| *v = (18.0 + rng.normal(8.0)).round().clamp(0.0, 255.0) as u8);
    assert!(is_blind(&thumbnail(&shutter, THUMB_SIZE)));
}

#[test]
fn test_a_hand_over_the_lens_is_blind() {
    let mut glow = Frame::filled(640, 480, [0; 3]);
    for y in 0..480 {
        let level = 8.0 + 14.0 * y as f32 / 479.0;
        let px = [0.6, 0.8, 1.0].map(|k| (level * k) as u8);
        for x in 0..640 {
            glow.set_pixel(x, y, px);
        }
    }
    assert!(is_blind(&thumbnail(&glow, THUMB_SIZE)));
}

#[test]
fn test_dim_scene_with_structure_is_not_blind() {
    let mut dim = base(); // mean ~25, but textured
    dim.data
        .iter_mut()
        .for_each(|v| *v = (f32::from(*v) * 0.2) as u8);
    assert!(!is_blind(&thumbnail(&dim, THUMB_SIZE)));
}

#[test]
fn test_bright_flat_scene_is_not_blind() {
    // A white wall behind an empty chair means 'nobody here', not 'cannot tell'.
    let wall = Frame::filled(640, 480, [200; 3]);
    assert!(!is_blind(&thumbnail(&wall, THUMB_SIZE)));
}

#[test]
fn test_cost_is_small() {
    let base = base();
    let mut gate = MotionGate::default();
    gate.mark_processed(&base, 0.0, Some(FACE_BOX));
    let frames: Vec<Frame> = (0..4).map(|k| add(&base, k % 2)).collect();
    for frame in &frames {
        // warm-up
        gate.should_process(frame, 0.1);
    }
    let runs = 200;
    let started = std::time::Instant::now();
    for i in 0..runs {
        gate.should_process(&frames[i % 4], 0.1);
    }
    let per_call_ms = started.elapsed().as_secs_f64() * 1e3 / runs as f64;
    // ~0.2 ms on a desktop CPU in a release build; generous bound for slow CI runners (and for
    // the unoptimised test profile).
    let bound = if cfg!(debug_assertions) { 20.0 } else { 2.0 };
    assert!(per_call_ms < bound, "{per_call_ms} ms");
}
