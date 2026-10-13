//! Synthetic webcam features for the gaze tests (port of tests/gaze_synth.py).
//!
//! A user ~65 cm from 24" 1080p panels (53 cm wide). The head turns part of the way
//! towards the target (a random share, as people do) and the eyes do the rest; the
//! webcam reports head pose, head position and iris offsets like the face-mesh
//! backend. Flat screens make pixels ~ tan(angle): the mapping is non-linear.
//!
//! Desks whose monitors differ in pixel density (a 4K panel next to a 1080p one;
//! Windows and X11 report native pixels) are described by [`Panel`]: where
//! each monitor physically is and how large its pixels are.
//!
//! Differences from the Python module: points and features are `(n, k)` [`DMatrix`] values
//! (the numpy arrays), the numpy generator is replaced by the seeded [`Rng`] (so the streams
//! differ from numpy's), the keyword arguments of `synth_features` and `features_at_cm` are
//! the fields of [`SynthOptions`] and [`SampleOptions`]. `outer_strip` comes from test_model.py.

// Shared by the tests of several modules; not every helper is used by every build.
#![allow(dead_code)]

use std::sync::LazyLock;

use nalgebra::DMatrix;

use crate::gaze::calibration::{CalibrationSample, make_plan};
use crate::types::{Monitor, Rect};

pub(crate) const PX_CM: f64 = 53.0 / 1920.0;
pub(crate) const FEATURE_NAMES: [&str; 8] =
    ["yaw", "pitch", "roll", "tx", "ty", "tz", "iris_h", "iris_v"];
/// The gaze-direction features (head rotation and iris offsets).
pub(crate) const GAZE: [usize; 4] = [0, 1, 6, 7];
pub(crate) const NOISE_SCALE: [f64; 8] = [1.0, 1.0, 1.0, 0.3, 0.3, 0.5, 0.01, 0.006];

pub(crate) static TWO: LazyLock<Vec<Monitor>> = LazyLock::new(|| {
    vec![
        Monitor {
            primary: true,
            ..Monitor::new(0, "left", Rect::new(0, 0, 1920, 1080))
        },
        Monitor::new(1, "right", Rect::new(1920, 0, 1920, 1080)),
    ]
});
pub(crate) static THREE: LazyLock<Vec<Monitor>> = LazyLock::new(|| {
    (0..3)
        .map(|i| Monitor::new(i, format!("m{i}"), Rect::new(1920 * (i - 1), 0, 1920, 1080)))
        .collect()
});
pub(crate) static STACKED: LazyLock<Vec<Monitor>> = LazyLock::new(|| {
    vec![
        Monitor::new(0, "top", Rect::new(0, -1080, 1920, 1080)),
        Monitor::new(1, "bottom", Rect::new(0, 0, 1920, 1080)),
    ]
});

/// Where a monitor physically is: the top-left corner of its picture in cm from
/// the camera (x to the right, y downwards) and the size of one pixel in cm.
#[derive(Debug, Clone)]
pub(crate) struct Panel {
    pub monitor: Monitor,
    pub left_cm: f64,
    pub top_cm: f64,
    pub cm_per_px: f64,
}

/// A 24" 1080p panel (left) next to a 27" 4K panel at 100 % (right), centred on
/// each other with the camera on the seam. Both are about 65 cm away, so the same
/// turn of the gaze covers 1.78x as many pixels on the 4K panel.
pub(crate) static MIXED: LazyLock<Vec<Monitor>> = LazyLock::new(|| {
    vec![
        Monitor {
            primary: true,
            ..Monitor::new(0, "fhd", Rect::new(0, 0, 1920, 1080))
        },
        Monitor::new(1, "uhd", Rect::new(1920, 0, 3840, 2160)),
    ]
});
pub(crate) static MIXED_PANELS: LazyLock<Vec<Panel>> = LazyLock::new(|| {
    vec![
        Panel {
            monitor: MIXED[0].clone(),
            left_cm: -53.1,
            top_cm: 3.85,
            cm_per_px: 53.1 / 1920.0,
        },
        Panel {
            monitor: MIXED[1].clone(),
            left_cm: 0.0,
            top_cm: 2.0,
            cm_per_px: 59.8 / 3840.0,
        },
    ]
});

/// Seeded random numbers (xoshiro256** seeded through splitmix64), standing in for
/// `numpy.random.Generator`. Vector methods draw their values in order.
pub(crate) struct Rng {
    state: [u64; 4],
}

impl Rng {
    pub(crate) fn new(seed: u64) -> Rng {
        let mut z = seed;
        let mut state = [0u64; 4];
        for slot in &mut state {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut v = z;
            v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *slot = v ^ (v >> 31);
        }
        Rng { state }
    }

    fn next_u64(&mut self) -> u64 {
        let s = &mut self.state;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub(crate) fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.unit()
    }

    pub(crate) fn normal(&mut self, mean: f64, std: f64) -> f64 {
        let u1 = 1.0 - self.unit(); // (0, 1], so the logarithm is finite
        let u2 = self.unit();
        mean + std * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// An integer in `[0, n)`.
    pub(crate) fn integers(&mut self, n: usize) -> usize {
        ((self.unit() * n as f64) as usize).min(n - 1)
    }

    pub(crate) fn uniform_vec(&mut self, low: f64, high: f64, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.uniform(low, high)).collect()
    }

    pub(crate) fn normal_vec(&mut self, mean: f64, std: f64, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.normal(mean, std)).collect()
    }

    /// An `(rows, cols)` matrix of standard normal values, drawn row by row.
    pub(crate) fn normal_matrix(&mut self, rows: usize, cols: usize) -> DMatrix<f64> {
        let values = self.normal_vec(0.0, 1.0, rows * cols);
        DMatrix::from_row_slice(rows, cols, &values)
    }

    pub(crate) fn permutation<T>(&mut self, mut items: Vec<T>) -> Vec<T> {
        for i in (1..items.len()).rev() {
            let j = self.integers(i + 1);
            items.swap(i, j);
        }
        items
    }
}

/// Physical position (cm from the camera; x right, y down) of desktop `points` `(n, 2)`.
///
/// A point outside every monitor is placed with the pixel size of the nearest one.
pub(crate) fn to_cm(points: &DMatrix<f64>, panels: &[Panel]) -> (Vec<f64>, Vec<f64>) {
    let mut xs = Vec::with_capacity(points.nrows());
    let mut ys = Vec::with_capacity(points.nrows());
    for r in 0..points.nrows() {
        let (px, py) = (points[(r, 0)], points[(r, 1)]);
        let panel = panels
            .iter()
            .min_by(|a, b| {
                let da = a.monitor.rect.distance_outside(px, py);
                let db = b.monitor.rect.distance_outside(px, py);
                da.total_cmp(&db)
            })
            .expect("at least one panel");
        let rect = panel.monitor.rect;
        xs.push(panel.left_cm + (px - f64::from(rect.x)) * panel.cm_per_px);
        ys.push(panel.top_cm + (py - f64::from(rect.y)) * panel.cm_per_px);
    }
    (xs, ys)
}

/// The keyword arguments of `synth_features` / `features_at_cm`.
///
/// `head_offset` (cm; right, down, back) moves the user's head away from where
/// it was while calibrating: `(0, 8, 0)` sits 8 cm lower, `(0, 0, 12)` leans
/// back. `head_share` is the mean share of a gaze shift the head makes (the
/// eyes do the rest), `pitch_share` scales it for vertical shifts: changing
/// them models a user who moves the head more or less than while calibrating.
/// `panels` places the monitors physically (mixed pixel densities; `camera_x`
/// is then ignored); by default every pixel is [`PX_CM`] wide and the camera
/// sits at `camera_x`, 2 cm above `y = 0`. None of these change the random
/// stream, so equal seeds stay comparable.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SynthOptions<'a> {
    pub camera_x: f64,
    pub head_offset: (f64, f64, f64),
    pub head_share: f64,
    pub pitch_share: f64,
    pub panels: Option<&'a [Panel]>,
}

impl Default for SynthOptions<'_> {
    fn default() -> Self {
        SynthOptions {
            camera_x: 1920.0,
            head_offset: (0.0, 0.0, 0.0),
            head_share: 0.55,
            pitch_share: 0.8,
            panels: None,
        }
    }
}

/// 8-D features (yaw, pitch, roll, tx, ty, tz, iris_h, iris_v) `(n, 8)` for gaze `points` `(n, 2)`.
pub(crate) fn synth_features(
    points: &DMatrix<f64>,
    rng: &mut Rng,
    noise: f64,
    options: &SynthOptions,
) -> DMatrix<f64> {
    let (target_x, target_y) = match options.panels {
        None => (
            (0..points.nrows())
                .map(|r| (points[(r, 0)] - options.camera_x) * PX_CM)
                .collect(),
            (0..points.nrows())
                .map(|r| points[(r, 1)] * PX_CM + 2.0)
                .collect(),
        ),
        Some(panels) => to_cm(points, panels),
    };
    features_at_cm(&target_x, &target_y, rng, noise, options)
}

/// [`synth_features`] for gaze targets given in cm from the camera (x right,
/// y down, in the plane of the screens): things that are not on a monitor.
pub(crate) fn features_at_cm(
    target_x: &[f64],
    target_y: &[f64],
    rng: &mut Rng,
    noise: f64,
    options: &SynthOptions,
) -> DMatrix<f64> {
    let n = target_x.len();
    let (dx, dy, dz) = options.head_offset;
    let hx: Vec<f64> = rng.normal_vec(0.0, 2.5, n).iter().map(|v| v + dx).collect();
    let hy: Vec<f64> = rng.normal_vec(0.0, 1.5, n).iter().map(|v| v + dy).collect();
    let hz: Vec<f64> = rng
        .normal_vec(0.0, 3.0, n)
        .iter()
        .map(|v| 65.0 + v + dz)
        .collect();
    let gaze_yaw: Vec<f64> = (0..n)
        .map(|i| (target_x[i] - hx[i]).atan2(hz[i]).to_degrees())
        .collect();
    let gaze_pitch: Vec<f64> = (0..n)
        .map(|i| (target_y[i] - hy[i]).atan2(hz[i]).to_degrees())
        .collect();
    let share: Vec<f64> = rng
        .normal_vec(0.0, 0.12, n)
        .iter()
        .map(|v| (options.head_share + v).clamp(0.1, 0.95))
        .collect();
    let yaw_noise = rng.normal_vec(0.0, 2.0, n);
    let pitch_noise = rng.normal_vec(0.0, 1.5, n);
    let roll = rng.normal_vec(0.0, 1.5, n);
    let mut feats = DMatrix::zeros(n, 8);
    for i in 0..n {
        let yaw = share[i] * gaze_yaw[i] + yaw_noise[i];
        let pitch = options.pitch_share * share[i] * gaze_pitch[i] + pitch_noise[i];
        let iris_h = 0.5 + 0.42 * (gaze_yaw[i] - yaw).to_radians().sin();
        let iris_v = 0.05 + 0.20 * (gaze_pitch[i] - pitch).to_radians().sin();
        let row = [yaw, pitch, roll[i], hx[i], hy[i], -hz[i], iris_h, iris_v];
        for (c, v) in row.into_iter().enumerate() {
            feats[(i, c)] = v;
        }
    }
    let jitter = rng.normal_matrix(n, 8);
    for i in 0..n {
        for c in 0..8 {
            feats[(i, c)] += jitter[(i, c)] * NOISE_SCALE[c] * noise;
        }
    }
    feats
}

/// Calibration-like data: a 3x3 grid per monitor, `per_point` samples each.
///
/// Returns the features `(n, 8)`, the points `(n, 2)` and the group (calibration point) of each row.
pub(crate) fn grid_dataset(
    monitors: &[Monitor],
    rng: &mut Rng,
    noise: f64,
    per_point: usize,
    camera_x: f64,
    panels: Option<&[Panel]>,
) -> (DMatrix<f64>, DMatrix<f64>, Vec<i64>) {
    let mut points: Vec<(f64, f64)> = Vec::new();
    let mut groups: Vec<i64> = Vec::new();
    for m in monitors {
        for ny in [0.1, 0.5, 0.9] {
            for nx in [0.1, 0.5, 0.9] {
                let group = (groups.len() / per_point) as i64;
                points.extend(std::iter::repeat_n(m.rect.denormalize(nx, ny), per_point));
                groups.extend(std::iter::repeat_n(group, per_point));
            }
        }
    }
    let p = points_matrix(&points);
    let options = SynthOptions {
        camera_x,
        panels,
        ..SynthOptions::default()
    };
    (synth_features(&p, rng, noise, &options), p, groups)
}

/// The keyword arguments of `calibration_samples`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SampleOptions<'a> {
    pub per_point: usize,
    pub camera_x: f64,
    pub points_per_monitor: usize,
    pub panels: Option<&'a [Panel]>,
}

impl Default for SampleOptions<'_> {
    fn default() -> Self {
        SampleOptions {
            per_point: 20,
            camera_x: 1920.0,
            points_per_monitor: 9,
            panels: None,
        }
    }
}

/// Samples as the calibration collector would record them for `make_plan(monitors)`.
pub(crate) fn calibration_samples(
    monitors: &[Monitor],
    rng: &mut Rng,
    noise: f64,
    options: &SampleOptions,
) -> Vec<CalibrationSample> {
    let plan = make_plan(monitors, options.points_per_monitor, 0.1).expect("a valid plan");
    let synth = SynthOptions {
        camera_x: options.camera_x,
        panels: options.panels,
        ..SynthOptions::default()
    };
    let mut samples = Vec::new();
    for t in plan {
        let target = points_matrix(&vec![(t.x, t.y); options.per_point]);
        let feats = synth_features(&target, rng, noise, &synth);
        for i in 0..feats.nrows() {
            samples.push(CalibrationSample::new(
                row_vec(&feats, i),
                t.x,
                t.y,
                t.monitor_index,
                t.point_id,
            ));
        }
    }
    samples
}

/// Uniform points on the monitors, avoiding the outer `margin` (5 %: like real use).
pub(crate) fn random_points(
    monitors: &[Monitor],
    rng: &mut Rng,
    n: usize,
    margin: f64,
) -> DMatrix<f64> {
    let idx: Vec<usize> = (0..n).map(|_| rng.integers(monitors.len())).collect();
    let nx = rng.uniform_vec(margin, 1.0 - margin, n);
    let ny = rng.uniform_vec(margin, 1.0 - margin, n);
    let points: Vec<(f64, f64)> = (0..n)
        .map(|i| monitors[idx[i]].rect.denormalize(nx[i], ny[i]))
        .collect();
    points_matrix(&points)
}

/// Points in the outer (left) `fraction` of `rect`, top to bottom.
pub(crate) fn outer_strip(rect: Rect, rng: &mut Rng, n: usize, fraction: f64) -> DMatrix<f64> {
    let left = f64::from(rect.x);
    let xs = rng.uniform_vec(left, left + fraction * f64::from(rect.w), n);
    let ys = rng.uniform_vec(f64::from(rect.y), f64::from(rect.bottom()), n);
    points_matrix(&xs.into_iter().zip(ys).collect::<Vec<_>>())
}

/// Points `cm` below the bottom edge of the monitors (a phone or papers on the desk).
pub(crate) fn below_points(monitors: &[Monitor], rng: &mut Rng, n: usize, cm: f64) -> DMatrix<f64> {
    let left = monitors.iter().map(|m| m.rect.x).min().unwrap_or(0);
    let right = monitors.iter().map(|m| m.rect.right()).max().unwrap_or(0);
    let bottom = monitors.iter().map(|m| m.rect.bottom()).max().unwrap_or(0);
    let xs = rng.uniform_vec(f64::from(left), f64::from(right), n);
    let y = f64::from(bottom) + cm / PX_CM;
    points_matrix(&xs.into_iter().map(|x| (x, y)).collect::<Vec<_>>())
}

/// [`below_points`] for physically placed monitors, in cm for
/// [`features_at_cm`]: `cm` below the lowest bottom edge, across the desk.
pub(crate) fn below_panels_cm(
    panels: &[Panel],
    rng: &mut Rng,
    n: usize,
    cm: f64,
) -> (Vec<f64>, Vec<f64>) {
    let left = panels
        .iter()
        .map(|p| p.left_cm)
        .fold(f64::INFINITY, f64::min);
    let right = panels
        .iter()
        .map(|p| p.left_cm + f64::from(p.monitor.rect.w) * p.cm_per_px)
        .fold(f64::NEG_INFINITY, f64::max);
    let bottom = panels
        .iter()
        .map(|p| p.top_cm + f64::from(p.monitor.rect.h) * p.cm_per_px)
        .fold(f64::NEG_INFINITY, f64::max);
    (rng.uniform_vec(left, right, n), vec![bottom + cm; n])
}

pub(crate) fn pixel_errors(pred: &DMatrix<f64>, truth: &DMatrix<f64>) -> Vec<f64> {
    (0..pred.nrows())
        .map(|i| (pred[(i, 0)] - truth[(i, 0)]).hypot(pred[(i, 1)] - truth[(i, 1)]))
        .collect()
}

/// `(x, y)` pairs as an `(n, 2)` matrix.
pub(crate) fn points_matrix(points: &[(f64, f64)]) -> DMatrix<f64> {
    DMatrix::from_fn(points.len(), 2, |r, c| {
        if c == 0 { points[r].0 } else { points[r].1 }
    })
}

/// Row `i` of a matrix as a vector.
pub(crate) fn row_vec(m: &DMatrix<f64>, i: usize) -> Vec<f64> {
    m.row(i).iter().copied().collect()
}

/// The median of the values (mean of the middle two for an even count), as `numpy.median`.
pub(crate) fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

pub(crate) fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}
