//! Unit tests from the vision spike (synthetic poses; the face-mesh tests cover real geometry).

use super::*;

const CAM: Camera = Camera {
    fx: 640.0,
    fy: 640.0,
    cx: 320.0,
    cy: 240.0,
};

#[test]
fn rodrigues_round_trip_matches_opencv_convention() {
    let rvec = Vector3::new(0.3, -1.2, 2.5);
    assert!((rotation_vector(&rodrigues(&rvec)) - rvec).norm() < 1e-12);
    // 90 degrees about z maps x to y (OpenCV: [[0,-1,0],[1,0,0],[0,0,1]]).
    let r = rodrigues(&Vector3::new(0.0, 0.0, std::f64::consts::FRAC_PI_2));
    assert!((r * Vector3::x() - Vector3::y()).norm() < 1e-12);
}

#[test]
fn project_points_is_pinhole() {
    let p = project_points(
        &[Point3::new(1.0, -2.0, 0.0)],
        &Vector3::zeros(),
        &Vector3::new(0.0, 0.0, 10.0),
        &CAM,
    );
    assert!((p[0] - Point2::new(384.0, 112.0)).norm() < 1e-12);
}

#[test]
fn solve_pnp_recovers_noisy_pose_with_and_without_guess() {
    let mut rng = Rng(7);
    let object: Vec<Point3<f64>> = (0..468)
        .map(|_| {
            Point3::new(
                14.0 * rng.uniform() - 7.0,
                18.0 * rng.uniform() - 9.0,
                8.0 * rng.uniform() - 4.0,
            )
        })
        .collect();
    let (rvec, tvec) = (Vector3::new(0.2, -0.35, 0.1), Vector3::new(3.0, -2.0, 50.0));
    let image: Vec<Point2<f64>> = project_points(&object, &rvec, &tvec, &CAM)
        .into_iter()
        .map(|p| p + nalgebra::Vector2::new(0.5 * rng.gauss(), 0.5 * rng.gauss()))
        .collect();
    let guess = (
        rvec + Vector3::new(0.05, -0.05, 0.03),
        tvec + Vector3::new(1.0, 1.0, -3.0),
    );
    for g in [None, Some(guess)] {
        let (r, t) = solve_pnp(&object, &image, &CAM, g).unwrap();
        let angle = (Rotation3::from_scaled_axis(r) * Rotation3::from_scaled_axis(rvec).inverse())
            .angle()
            .to_degrees();
        assert!(angle < 0.5, "rotation off by {angle} deg");
        assert!(
            (t - tvec).norm() / tvec.norm() < 0.01,
            "translation {t} vs {tvec}"
        );
    }
}

/// Noise-free points: the DLT is exact and its rotation's trace can round past 3.
#[test]
fn solve_pnp_handles_exact_points_at_the_identity() {
    let mut rng = Rng(3);
    let object: Vec<Point3<f64>> = (0..33)
        .map(|_| Point3::new(rng.gauss(), rng.gauss(), rng.gauss()))
        .collect();
    let tvec = Vector3::new(0.0, 0.0, 50.0);
    let image = project_points(&object, &Vector3::zeros(), &tvec, &CAM);
    let (r, t) = solve_pnp(&object, &image, &CAM, None).unwrap();
    assert!(r.norm() < 1e-9 && (t - tvec).norm() < 1e-6, "{r} {t}");
}
