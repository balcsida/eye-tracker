//! `cv2.Rodrigues`, `cv2.projectPoints` (pinhole, no distortion) and `cv2.solvePnP` replacements.
//!
//! `solve_pnp` refines with Levenberg-Marquardt over the 6 pose parameters
//! (`SOLVEPNP_ITERATIVE` with `useExtrinsicGuess` when `guess` is given); without a
//! guess it starts from a normalised DLT, which stands in for `SOLVEPNP_SQPNP`.
//! The DLT needs non-coplanar points, which a face mesh is.
use nalgebra::{
    Matrix3, Matrix6, Point2, Point3, Rotation3, SMatrix, SymmetricEigen, UnitQuaternion, Vector3,
    Vector6,
};

/// Pinhole intrinsics, like OpenCV's camera matrix without skew.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

/// cv2.Rodrigues, rotation vector -> matrix.
pub fn rodrigues(rvec: &Vector3<f64>) -> Matrix3<f64> {
    Rotation3::from_scaled_axis(*rvec).into_inner()
}

/// cv2.Rodrigues, matrix -> rotation vector (the matrix must be a rotation).
pub fn rotation_vector(r: &Matrix3<f64>) -> Vector3<f64> {
    scaled_axis(&Rotation3::from_matrix_unchecked(*r))
}

/// `Rotation3::scaled_axis` through a quaternion: the matrix form takes `acos` of the trace,
/// which is NaN when rounding pushes a near-identity trace past 3 (exact DLT data does).
fn scaled_axis(r: &Rotation3<f64>) -> Vector3<f64> {
    UnitQuaternion::from_rotation_matrix(r).scaled_axis()
}

/// cv2.projectPoints without distortion.
pub fn project_points(
    points: &[Point3<f64>],
    rvec: &Vector3<f64>,
    tvec: &Vector3<f64>,
    cam: &Camera,
) -> Vec<Point2<f64>> {
    let r = rodrigues(rvec);
    points
        .iter()
        .map(|p| {
            let x = r * p.coords + tvec;
            Point2::new(cam.fx * x.x / x.z + cam.cx, cam.fy * x.y / x.z + cam.cy)
        })
        .collect()
}

/// Pose `(rvec, tvec)` mapping object points into the camera frame, or `None` when
/// the solve fails (fewer than 6 points, degenerate geometry, non-finite result).
pub fn solve_pnp(
    object: &[Point3<f64>],
    image: &[Point2<f64>],
    cam: &Camera,
    guess: Option<(Vector3<f64>, Vector3<f64>)>,
) -> Option<(Vector3<f64>, Vector3<f64>)> {
    if object.len() != image.len() || object.len() < 6 {
        return None;
    }
    let (r, t) = match guess {
        Some((rvec, tvec)) => (Rotation3::from_scaled_axis(rvec), tvec),
        None => dlt(object, image, cam)?,
    };
    let (r, t) = refine_lm(object, image, cam, r, t);
    let rvec = scaled_axis(&r);
    (rvec.iter().chain(t.iter()).all(|v| v.is_finite())).then_some((rvec, t))
}

/// Direct linear transform on normalised image coordinates and centred, scaled
/// object points. Needs non-coplanar points (a face mesh is).
fn dlt(
    object: &[Point3<f64>],
    image: &[Point2<f64>],
    cam: &Camera,
) -> Option<(Rotation3<f64>, Vector3<f64>)> {
    let n = object.len() as f64;
    let mean = object.iter().fold(Vector3::zeros(), |s, p| s + p.coords) / n;
    let spread = object.iter().map(|p| (p.coords - mean).norm()).sum::<f64>() / n;
    if spread.is_nan() || spread <= 0.0 {
        return None;
    }
    let k = 3f64.sqrt() / spread;
    // Accumulate A^T A (12x12) directly instead of the 2N x 12 matrix.
    let mut ata = SMatrix::<f64, 12, 12>::zeros();
    for (p, q) in object.iter().zip(image) {
        let pn = (p.coords - mean) * k;
        let h = [pn.x, pn.y, pn.z, 1.0];
        let (u, v) = ((q.x - cam.cx) / cam.fx, (q.y - cam.cy) / cam.fy);
        for (coord, offset) in [(u, 0), (v, 4)] {
            let mut row = SMatrix::<f64, 1, 12>::zeros();
            for j in 0..4 {
                row[offset + j] = h[j];
                row[8 + j] = -coord * h[j];
            }
            ata += row.transpose() * row;
        }
    }
    let eig = SymmetricEigen::new(ata);
    let (min_ix, _) = eig
        .eigenvalues
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))?;
    let m = eig.eigenvectors.column(min_ix);
    let mut a = Matrix3::from_fn(|i, j| m[i * 4 + j]);
    let mut last = Vector3::new(m[3], m[7], m[11]);
    if a.determinant() < 0.0 {
        a = -a;
        last = -last;
    }
    let svd = a.svd(true, true);
    let (u, vt) = (svd.u?, svd.v_t?);
    let r = u * vt;
    if r.determinant() < 0.0 {
        return None;
    }
    // a = (lambda / k) R, last = lambda (R mean + t).
    let scale = svd.singular_values.mean();
    let t = last / (scale * k) - r * mean;
    Some((Rotation3::from_matrix_unchecked(r), t))
}

/// Levenberg-Marquardt on the reprojection error; rotation updates are
/// left-multiplied small rotations, so there is no singularity at any angle.
fn refine_lm(
    object: &[Point3<f64>],
    image: &[Point2<f64>],
    cam: &Camera,
    mut r: Rotation3<f64>,
    mut t: Vector3<f64>,
) -> (Rotation3<f64>, Vector3<f64>) {
    let cost = |r: &Rotation3<f64>, t: &Vector3<f64>| -> f64 {
        object
            .iter()
            .zip(image)
            .map(|(p, q)| {
                let x = r * p.coords + t;
                (cam.fx * x.x / x.z + cam.cx - q.x).powi(2)
                    + (cam.fy * x.y / x.z + cam.cy - q.y).powi(2)
            })
            .sum()
    };
    let mut current = cost(&r, &t);
    let mut lambda = 1e-3;
    for _ in 0..100 {
        let mut jtj = Matrix6::<f64>::zeros();
        let mut jtr = Vector6::<f64>::zeros();
        for (p, q) in object.iter().zip(image) {
            let rp = r * p.coords;
            let x = rp + t;
            let iz = 1.0 / x.z;
            let res = [
                cam.fx * x.x * iz + cam.cx - q.x,
                cam.fy * x.y * iz + cam.cy - q.y,
            ];
            // d(u, v)/dX, then dX/d(omega) = -[R p]x and dX/dt = I.
            let du = Vector3::new(cam.fx * iz, 0.0, -cam.fx * x.x * iz * iz);
            let dv = Vector3::new(0.0, cam.fy * iz, -cam.fy * x.y * iz * iz);
            for (d, e) in [(du, res[0]), (dv, res[1])] {
                let rot = rp.cross(&d); // d^T (-[rp]x) = (rp x d)^T
                let j = Vector6::new(rot.x, rot.y, rot.z, d.x, d.y, d.z);
                jtj += j * j.transpose();
                jtr += j * e;
            }
        }
        let mut step = None;
        while lambda < 1e10 {
            let mut a = jtj;
            for i in 0..6 {
                a[(i, i)] += lambda * jtj[(i, i)].max(1e-12);
            }
            let Some(chol) = a.cholesky() else {
                lambda *= 10.0;
                continue;
            };
            let delta = -chol.solve(&jtr);
            let r_new = Rotation3::from_scaled_axis(Vector3::new(delta[0], delta[1], delta[2])) * r;
            let t_new = t + Vector3::new(delta[3], delta[4], delta[5]);
            let c = cost(&r_new, &t_new);
            if c < current {
                lambda = (lambda / 10.0).max(1e-9);
                step = Some((r_new, t_new, c, delta.norm()));
                break;
            }
            lambda *= 10.0;
        }
        let Some((r_new, t_new, c, size)) = step else {
            break;
        };
        let improvement = (current - c) / current.max(1e-300);
        (r, t, current) = (r_new, t_new, c);
        if size < 1e-10 || improvement < 1e-12 {
            break;
        }
    }
    (r, t)
}

/// Synthetic-data noise for the tests: xorshift64* and Box-Muller,
/// reproducible without a rand dependency.
#[cfg(test)]
pub(crate) struct Rng(pub u64);
#[cfg(test)]
impl Rng {
    pub(crate) fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64
    }
    pub(crate) fn gauss(&mut self) -> f64 {
        let (a, b) = (self.uniform().max(1e-300), self.uniform());
        (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
    }
}

#[cfg(test)]
#[path = "pnp_tests.rs"]
mod tests;
