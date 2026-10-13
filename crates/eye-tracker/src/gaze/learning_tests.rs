// Tests for gaze::learning (implicit samples, refit, drift detection), ported from
// tests/test_learning.py. The random streams come from testing::gaze_synth::Rng, not numpy.
//
// Not ported:
// * test_invalid_learner_arguments: the `max_samples=-1` case and `set_max_samples(-5)`;
//   both take a `usize`, so a negative capacity cannot be written (and `set_max_samples`
//   has no error to return). The other three arguments are covered.
// * The discard_last tests compare the returned sample by value where Python used `is`
//   (learned samples carry unique point ids).

use nalgebra::DMatrix;

use super::*;
use crate::gaze::calibration::evaluate;
use crate::testing::capture_logs;
use crate::testing::gaze_synth::{
    GAZE, Rng, SampleOptions, SynthOptions, TWO, calibration_samples, median, random_points,
    row_vec, synth_features,
};
use crate::types::monitor_at;

static MONITORS: std::sync::LazyLock<Vec<Monitor>> = std::sync::LazyLock::new(|| {
    vec![
        Monitor::new(0, "left", Rect::new(0, 0, 1920, 1080)),
        Monitor::new(1, "right", Rect::new(1920, 0, 1920, 1080)),
    ]
});

fn lookup(x: f64, y: f64) -> Option<i32> {
    monitor_at(MONITORS.iter(), x, y).map(|m| m.index)
}

fn obs() -> Observation {
    obs_with(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
}

fn obs_with(features: Vec<f64>) -> Observation {
    Observation {
        quality: 1.0,
        features: Some(features),
        ..Observation::new(0.0, 1)
    }
}

fn learner_with(options: LearnerOptions) -> ImplicitLearner {
    ImplicitLearner::new(options).unwrap()
}

/// A short drag ending at (x, y) at time t, then an observation 0.4 s later.
fn move_and_settle(
    learner: &mut ImplicitLearner,
    x: f64,
    y: f64,
    t: f64,
) -> Option<CalibrationSample> {
    learner.on_manual_cursor(x - 30.0, y - 10.0, t - 0.1);
    learner.on_manual_cursor(x, y, t);
    learner.on_observation(&obs(), t + 0.4, &lookup)
}

fn error_of<T: std::fmt::Debug>(result: Result<T, CalibrationError>) -> String {
    result.unwrap_err().to_string()
}

fn xs_of(learner: &ImplicitLearner) -> Vec<f64> {
    learner.samples().iter().map(|s| s.x).collect()
}

fn ids_of(learner: &ImplicitLearner) -> Vec<i64> {
    learner.samples().iter().map(|s| s.point_id).collect()
}

fn gaze_options() -> SampleOptions<'static> {
    SampleOptions {
        per_point: 8,
        ..SampleOptions::default()
    }
}

fn base_samples(monitors: &[Monitor], seed: u64, per_point: usize) -> Vec<CalibrationSample> {
    let options = SampleOptions {
        per_point,
        ..SampleOptions::default()
    };
    calibration_samples(monitors, &mut Rng::new(seed), 1.0, &options)
}

fn features_matrix(samples: &[CalibrationSample]) -> DMatrix<f64> {
    samples_to_arrays(samples).unwrap().0
}

fn medians_to(pred: &DMatrix<f64>, x: f64, y: f64) -> f64 {
    median(
        &(0..pred.nrows())
            .map(|i| (pred[(i, 0)] - x).hypot(pred[(i, 1)] - y))
            .collect::<Vec<_>>(),
    )
}

// --------------------------------------------------------------------------- ImplicitLearner
#[test]
fn test_no_sample_without_a_manual_move() {
    let mut learner = ImplicitLearner::default();
    assert_eq!(learner.on_observation(&obs(), 10.0, &lookup), None);
    assert!(learner.samples().is_empty());
}

#[test]
fn test_one_sample_per_settle() {
    let mut learner = learner_with(LearnerOptions {
        weight: 0.5,
        settle_s: 0.35,
        ..LearnerOptions::default()
    });
    learner.on_manual_cursor(100.0, 100.0, 0.0);
    learner.on_manual_cursor(2500.0, 300.0, 0.2);
    // not settled yet
    assert_eq!(learner.on_observation(&obs(), 0.3, &lookup), None);
    let sample = learner
        .on_observation(&obs_with(vec![3.0; 8]), 0.6, &lookup)
        .unwrap();
    assert_eq!(
        (sample.x, sample.y, sample.monitor_index),
        (2500.0, 300.0, 1)
    );
    assert_eq!(sample.weight, 0.5);
    assert_eq!(sample.point_id, -1);
    assert!(sample.features.iter().all(|v| *v == 3.0));
    // Further observations of the same settle are duplicates.
    assert_eq!(learner.on_observation(&obs(), 0.7, &lookup), None);
    assert_eq!(learner.on_observation(&obs(), 1.0, &lookup), None);
    assert_eq!(learner.samples().len(), 1);

    // A new move starts a new settle with its own (negative) point id.
    let second = move_and_settle(&mut learner, 500.0, 500.0, 3.0).unwrap();
    assert_eq!(second.point_id, -2);
    assert_eq!(second.monitor_index, 0);
    assert_eq!(learner.samples().len(), 2);
}

#[test]
fn test_settle_expires_after_recent_window() {
    let mut learner = learner_with(LearnerOptions {
        settle_s: 0.35,
        ..LearnerOptions::default()
    });
    learner.on_manual_cursor(100.0, 100.0, 0.0);
    // > 1.5 s since the move
    assert_eq!(learner.on_observation(&obs(), 1.6, &lookup), None);
    assert_eq!(learner.on_observation(&obs(), 1.7, &lookup), None);
    assert!(learner.samples().is_empty());
}

#[test]
fn test_waits_for_a_usable_observation_within_the_window() {
    let mut learner = ImplicitLearner::default();
    learner.on_manual_cursor(100.0, 100.0, 0.0);
    let blink = Observation {
        blink: true,
        ..obs()
    };
    assert_eq!(learner.on_observation(&blink, 0.4, &lookup), None);
    let blurred = Observation {
        quality: 0.0,
        ..obs()
    };
    assert_eq!(learner.on_observation(&blurred, 0.5, &lookup), None);
    let skipped = Observation {
        skipped: true,
        ..obs()
    };
    assert_eq!(learner.on_observation(&skipped, 0.6, &lookup), None);
    assert_eq!(
        learner.on_observation(&Observation::new(0.7, 0), 0.7, &lookup),
        None
    );
    let broken = obs_with(vec![f64::INFINITY; 8]);
    assert_eq!(learner.on_observation(&broken, 0.8, &lookup), None);
    assert!(learner.on_observation(&obs(), 0.9, &lookup).is_some());
}

#[test]
fn test_cursor_outside_every_monitor_gives_no_sample() {
    let mut learner = ImplicitLearner::default();
    learner.on_manual_cursor(-500.0, -500.0, 0.0);
    assert_eq!(learner.on_observation(&obs(), 0.5, &lookup), None);
    // settle consumed
    assert_eq!(learner.on_observation(&obs(), 0.6, &lookup), None);
    assert!(learner.samples().is_empty());
}

#[test]
fn test_moving_again_postpones_the_sample() {
    let mut learner = learner_with(LearnerOptions {
        settle_s: 0.35,
        ..LearnerOptions::default()
    });
    learner.on_manual_cursor(100.0, 100.0, 0.0);
    learner.on_manual_cursor(150.0, 100.0, 0.3);
    // only 0.2 s since last move
    assert_eq!(learner.on_observation(&obs(), 0.5, &lookup), None);
    let sample = learner.on_observation(&obs(), 0.7, &lookup).unwrap();
    assert_eq!(sample.x, 150.0);
}

#[test]
fn test_capacity_is_shared_fairly_between_monitors() {
    let mut learner = learner_with(LearnerOptions {
        max_samples: 4,
        ..LearnerOptions::default()
    });
    let mut t = 0.0;
    // mostly the left monitor
    for x in [100.0, 2000.0, 200.0, 300.0, 400.0, 500.0] {
        assert!(move_and_settle(&mut learner, x, 100.0, t).is_some());
        t += 2.0;
    }
    let xs = xs_of(&learner);
    assert_eq!(xs.len(), 4);
    assert!(xs.contains(&2000.0)); // the lone right-monitor sample survives
    assert_eq!(xs, [2000.0, 300.0, 400.0, 500.0]); // the oldest left samples went first
}

#[test]
fn test_zero_capacity_disables_learning() {
    let mut learner = learner_with(LearnerOptions {
        max_samples: 0,
        ..LearnerOptions::default()
    });
    assert_eq!(move_and_settle(&mut learner, 100.0, 100.0, 1.0), None);
    assert!(learner.samples().is_empty());
}

#[test]
fn test_refit_schedule() {
    let mut learner = learner_with(LearnerOptions {
        refit_every: 3,
        ..LearnerOptions::default()
    });
    let mut t = 0.0;
    for i in 0..2 {
        move_and_settle(&mut learner, 100.0 + f64::from(i), 100.0, t);
        t += 2.0;
    }
    assert!(!learner.should_refit());
    move_and_settle(&mut learner, 300.0, 100.0, t);
    assert!(learner.should_refit());
    assert_eq!(learner.new_since_refit(), 3);
    learner.mark_refit();
    assert!(!learner.should_refit());
    assert_eq!(learner.new_since_refit(), 0);
}

#[test]
fn test_load_clear_and_ids_continue() {
    let mut learner = learner_with(LearnerOptions {
        max_samples: 3,
        ..LearnerOptions::default()
    });
    let saved: Vec<CalibrationSample> = (0..5)
        .map(|i| {
            CalibrationSample::weighted(
                vec![0.0; 8],
                10.0 * f64::from(i),
                10.0,
                0,
                -i64::from(i + 1),
                0.5,
            )
        })
        .collect();
    learner.load(saved);
    assert_eq!(ids_of(&learner), [-3, -4, -5]); // newest kept
    assert!(!learner.should_refit());
    let sample = move_and_settle(&mut learner, 2500.0, 100.0, 1.0).unwrap();
    assert_eq!(sample.point_id, -6);
    learner.set_max_samples(1);
    assert_eq!(ids_of(&learner), [-6]);
    learner.clear();
    assert!(learner.samples().is_empty());
    let sample = move_and_settle(&mut learner, 100.0, 100.0, 5.0).unwrap();
    assert_eq!(sample.point_id, -1);
}

#[test]
fn test_feature_length_change_discards_old_samples() {
    let logs = capture_logs();
    let mut learner = ImplicitLearner::default();
    learner.load([CalibrationSample::weighted(
        vec![0.0; 6],
        1.0,
        1.0,
        0,
        -1,
        0.5,
    )]);
    assert!(move_and_settle(&mut learner, 100.0, 100.0, 1.0).is_some());
    assert_eq!(learner.samples().len(), 1);
    assert_eq!(learner.samples()[0].features.len(), 8);
    let warnings = logs.messages_at_least(log::Level::Warn);
    assert!(warnings.iter().any(|m| m.contains("discarding")));
}

#[test]
fn test_discard_last_takes_the_sample_back() {
    let mut learner = learner_with(LearnerOptions {
        refit_every: 2,
        ..LearnerOptions::default()
    });
    let first = move_and_settle(&mut learner, 100.0, 100.0, 1.0).unwrap();
    let second = move_and_settle(&mut learner, 200.0, 100.0, 3.0).unwrap();
    assert!(learner.should_refit());
    assert_eq!(learner.discard_last(), Some(second));
    assert_eq!(learner.samples(), std::slice::from_ref(&first));
    assert_eq!(learner.new_since_refit(), 1);
    assert!(!learner.should_refit());
    // only the latest can be taken back
    assert_eq!(learner.discard_last(), None);
    assert_eq!(learner.samples(), [first]);
    let third = move_and_settle(&mut learner, 300.0, 100.0, 5.0).unwrap();
    assert_eq!(third.point_id, -3); // ids are never reused
}

#[test]
fn test_discard_last_restores_the_sample_evicted_for_it() {
    let mut learner = learner_with(LearnerOptions {
        max_samples: 2,
        ..LearnerOptions::default()
    });
    let mut t = 0.0;
    for x in [100.0, 200.0, 2500.0] {
        move_and_settle(&mut learner, x, 100.0, t);
        t += 2.0;
    }
    assert_eq!(xs_of(&learner), [200.0, 2500.0]); // 100 made room
    learner.discard_last();
    assert_eq!(xs_of(&learner), [100.0, 200.0]);
}

#[test]
fn test_discard_last_after_refit_or_reset() {
    let mut learner = learner_with(LearnerOptions {
        refit_every: 1,
        ..LearnerOptions::default()
    });
    let sample = move_and_settle(&mut learner, 100.0, 100.0, 1.0);
    learner.mark_refit();
    // still leaves the training set …
    assert_eq!(learner.discard_last(), sample);
    // … but was already accounted for
    assert_eq!(learner.new_since_refit(), 0);
    let resets: [fn(&mut ImplicitLearner); 3] = [
        |l| l.load([]),
        |l| l.clear(),
        |l| {
            l.set_max_samples(0);
        },
    ];
    for reset in resets {
        learner.set_max_samples(400);
        move_and_settle(&mut learner, 100.0, 100.0, 10.0);
        reset(&mut learner);
        assert_eq!(learner.discard_last(), None);
    }
}

#[test]
fn test_set_max_samples_reports_dropped_samples() {
    let mut learner = learner_with(LearnerOptions {
        max_samples: 5,
        ..LearnerOptions::default()
    });
    for i in 0..3 {
        move_and_settle(
            &mut learner,
            100.0 + f64::from(i),
            100.0,
            2.0 * f64::from(i),
        );
    }
    assert!(!learner.set_max_samples(3));
    assert!(!learner.set_max_samples(10));
    assert!(learner.set_max_samples(1));
    assert_eq!(learner.samples().len(), 1);
    assert!(learner.set_max_samples(0));
    assert!(learner.samples().is_empty());
    assert!(!learner.set_max_samples(0));
}

#[test]
fn test_invalid_learner_arguments() {
    let cases = [
        LearnerOptions {
            weight: 0.0,
            ..LearnerOptions::default()
        },
        LearnerOptions {
            settle_s: -0.1,
            ..LearnerOptions::default()
        },
        LearnerOptions {
            refit_every: 0,
            ..LearnerOptions::default()
        },
    ];
    for options in cases {
        assert!(error_of(ImplicitLearner::new(options)).contains("max_samples"));
    }
}

// --------------------------------------------------------------------------- refit_model
#[test]
fn test_refit_model_uses_template_settings_and_weights() {
    let mut rng = Rng::new(3);
    let x = rng.normal_matrix(40, 3);
    let map = DMatrix::from_row_slice(3, 2, &[500.0, 0.0, 0.0, 300.0, 50.0, 50.0]);
    let mut y = &x * &map;
    for i in 0..40 {
        y[(i, 0)] += 1900.0;
        y[(i, 1)] += 540.0;
    }
    let base: Vec<CalibrationSample> = (0..40)
        .map(|i| CalibrationSample::new(row_vec(&x, i), y[(i, 0)], y[(i, 1)], 0, (i / 4) as i64))
        .collect();
    let template = GazeModel::new(1, 0.01, None)
        .unwrap()
        .fit(&x, &y, None, Some(Rect::new(0, 0, 3840, 1080)), None)
        .unwrap();

    // Learned samples that contradict the calibration pull the fit, but weakly.
    let wrong: Vec<CalibrationSample> = (0..10)
        .map(|i| CalibrationSample::weighted(row_vec(&x, i), 3500.0, 900.0, 1, -1 - i as i64, 0.01))
        .collect();
    let model = refit_model(&base, &wrong, &template, None, None).unwrap();
    assert_eq!((model.degree, model.alpha), (1, 0.01));
    assert_eq!(model.bounds(), Some(Rect::new(0, 0, 3840, 1080)));
    let pred = model.predict_many(&x).unwrap();
    let err: Vec<f64> = (0..40)
        .map(|i| (pred[(i, 0)] - y[(i, 0)]).hypot(pred[(i, 1)] - y[(i, 1)]))
        .collect();
    assert!(median(&err) < 50.0);

    let heavy: Vec<CalibrationSample> = wrong
        .iter()
        .map(|s| CalibrationSample::weighted(s.features.clone(), s.x, s.y, 1, s.point_id, 50.0))
        .collect();
    let pulled = refit_model(
        &base,
        &heavy,
        &template,
        Some(Rect::new(0, 0, 4000, 1200)),
        None,
    )
    .unwrap();
    assert_eq!(pulled.bounds(), Some(Rect::new(0, 0, 4000, 1200)));
    let first_ten = x.rows(0, 10).into_owned();
    assert!(
        medians_to(&pulled.predict_many(&first_ten).unwrap(), 3500.0, 900.0)
            < medians_to(&model.predict_many(&first_ten).unwrap(), 3500.0, 900.0)
    );
}

fn allclose(a: &DMatrix<f64>, b: &DMatrix<f64>, atol: f64) -> bool {
    a.shape() == b.shape()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| (x - y).abs() <= atol + 1e-5 * y.abs())
}

#[test]
fn test_refit_without_learned_samples_restores_the_calibration_fit() {
    // Lowering "max learned samples" (to 0) must be able to undo their influence.
    let base = calibration_samples(&TWO, &mut Rng::new(60), 1.0, &gaze_options());
    let (calibrated, _) = evaluate(&base, &TWO, None, Some(&GAZE)).unwrap();
    let mut rng = Rng::new(61);
    let points = random_points(&TWO[..1], &mut rng, 80, 0.05);
    let feats = synth_features(&points, &mut rng, 1.0, &SynthOptions::default());
    let wrong: Vec<CalibrationSample> = (0..80)
        .map(|i| {
            CalibrationSample::weighted(row_vec(&feats, i), 3500.0, 900.0, 1, -1 - i as i64, 0.5)
        })
        .collect();
    let polluted = refit_model(&base, &wrong, &calibrated, None, None).unwrap();
    assert_eq!(polluted.nonlinear(), Some(&GAZE[..]));
    let mut learner = ImplicitLearner::default();
    learner.load(wrong);
    assert!(learner.set_max_samples(0));
    let restored = refit_model(&base, learner.samples(), &polluted, None, None).unwrap();
    let x = features_matrix(&base);
    assert!(!allclose(
        &polluted.predict_many(&x).unwrap(),
        &calibrated.predict_many(&x).unwrap(),
        1.0
    ));
    assert!(allclose(
        &restored.predict_many(&x).unwrap(),
        &calibrated.predict_many(&x).unwrap(),
        1e-6
    ));
    assert_eq!(
        (
            restored.degree,
            restored.alpha,
            restored.nonlinear(),
            restored.bounds()
        ),
        (
            calibrated.degree,
            calibrated.alpha,
            calibrated.nonlinear(),
            calibrated.bounds()
        )
    );
    assert_eq!(restored.away_regions(), calibrated.away_regions());
}

#[test]
fn test_refit_learns_the_look_away_regions_of_the_calibrated_monitors() {
    let base = calibration_samples(&TWO, &mut Rng::new(62), 1.0, &gaze_options());
    let (calibrated, _) = evaluate(&base, &TWO, None, Some(&GAZE)).unwrap();
    let rects: Vec<Rect> = calibrated.away_regions().iter().map(|r| r.rect).collect();
    assert_eq!(rects, TWO.iter().map(|m| m.rect).collect::<Vec<_>>());
    // A model saved before the regions existed gains them with the next refit...
    let mut data = calibrated.to_dict();
    data.as_object_mut().unwrap().remove("away_regions");
    let legacy = GazeModel::from_dict(&data).unwrap();
    assert!(
        refit_model(&base, &[], &legacy, None, None)
            .unwrap()
            .away_regions()
            .is_empty()
    );
    let upgraded = refit_model(&base, &[], &legacy, None, Some(&TWO)).unwrap();
    assert_eq!(upgraded.away_regions(), calibrated.away_regions());
    // ...and a refit without monitors keeps those of the template.
    assert_eq!(
        refit_model(&base, &[], &upgraded, None, None)
            .unwrap()
            .away_regions(),
        calibrated.away_regions()
    );
}

// --------------------------------------------------------------------------- plausible labels
fn labelled(x: f64, y: f64, monitor: i32) -> CalibrationSample {
    CalibrationSample::weighted(vec![0.0; 8], x, y, monitor, -1, 0.5)
}

#[test]
fn test_plausible_label() {
    let diagonal = MONITORS[1].rect.diagonal();
    let label = labelled(2300.0, 900.0, 1);
    let plausible = |predicted: [f64; 2], monitors: &[Monitor], max_error: f64| {
        plausible_label(&predicted, &label, monitors, max_error)
    };
    assert!(plausible([2400.0, 800.0], &MONITORS, IMPLICIT_MAX_ERROR));
    // On the other monitor but close to the label: a drifted model the label corrects.
    assert!(plausible([1800.0, 900.0], &MONITORS, IMPLICIT_MAX_ERROR));
    // The cursor was parked on the right while the user read the left monitor.
    assert!(!plausible([500.0, 400.0], &MONITORS, IMPLICIT_MAX_ERROR));
    let limit = IMPLICIT_MAX_ERROR * diagonal;
    assert!(plausible(
        [2300.0 - limit + 1.0, 900.0],
        &MONITORS,
        IMPLICIT_MAX_ERROR
    ));
    assert!(!plausible(
        [2300.0 - limit - 1.0, 900.0],
        &MONITORS,
        IMPLICIT_MAX_ERROR
    ));
    assert!(plausible([500.0, 400.0], &MONITORS, 2.0));
    // No evidence either way.
    assert!(plausible([f64::NAN, 400.0], &MONITORS, IMPLICIT_MAX_ERROR));
    assert!(plausible([500.0, 400.0], &[], IMPLICIT_MAX_ERROR));
    // An index that is not in the list: the monitor under the cursor decides.
    let small = [
        Monitor::new(0, "a", Rect::new(0, 0, 800, 600)),
        Monitor::new(1, "b", Rect::new(800, 0, 800, 600)),
    ];
    assert!(!plausible_label(
        &[700.0, 300.0],
        &labelled(1500.0, 300.0, 9),
        &small,
        IMPLICIT_MAX_ERROR
    ));
}

#[test]
fn test_gated_learning_resists_parked_cursor_labels() {
    // Regression: every settle used to be learned, so a habit such as clicking "Run" on the
    // right monitor and reading the output on the left one biased the model towards the right.
    let base = base_samples(&TWO, 70, 12);
    let (model, _) = evaluate(&base, &TWO, None, Some(&GAZE)).unwrap();
    let mut rng = Rng::new(71);

    let mut learn = |gate: bool| -> GazeModel {
        let mut learner = learner_with(LearnerOptions {
            max_samples: 400,
            refit_every: 10_000,
            ..LearnerOptions::default()
        });
        let mut t = 0.0;
        for i in 0..400 {
            let mut gaze = random_points(&TWO, &mut rng, 1, 0.05);
            let parked = i % 5 == 0; // 20 % of the settles
            if parked {
                gaze = random_points(&TWO[..1], &mut rng, 1, 0.05); // reading the left monitor
            }
            let cursor = if parked {
                (3500.0, 900.0)
            } else {
                (gaze[(0, 0)], gaze[(0, 1)])
            };
            let features = synth_features(&gaze, &mut rng, 1.0, &SynthOptions::default());
            learner.on_manual_cursor(cursor.0, cursor.1, t);
            let o = Observation {
                timestamp: t + 0.4,
                ..obs_with(row_vec(&features, 0))
            };
            let sample = learner.on_observation(&o, t + 0.4, &lookup).unwrap();
            if gate
                && !plausible_label(
                    &model.predict(&sample.features).unwrap(),
                    &sample,
                    &TWO,
                    IMPLICIT_MAX_ERROR,
                )
            {
                assert_eq!(learner.discard_last(), Some(sample));
            }
            t += 2.0;
        }
        refit_model(&base, learner.samples(), &model, None, None).unwrap()
    };

    let points = random_points(&TWO, &mut Rng::new(72), 1500, 0.05);
    let x = synth_features(&points, &mut Rng::new(73), 1.0, &SynthOptions::default());
    let accuracy = |m: &GazeModel| -> f64 {
        let predicted = m.predict_many(&x).unwrap();
        let hits = (0..points.nrows())
            .filter(|&i| {
                let side = |px: f64, py: f64| nearest_monitor(TWO.iter(), px, py).unwrap().0.index;
                side(predicted[(i, 0)], predicted[(i, 1)]) == side(points[(i, 0)], points[(i, 1)])
            })
            .count();
        hits as f64 / points.nrows() as f64
    };

    let (ungated, gated) = {
        let ungated = accuracy(&learn(false));
        (ungated, accuracy(&learn(true)))
    };
    assert!(gated >= 0.99, "gated {gated}");
    assert!(gated >= accuracy(&model) - 0.005);
    assert!(ungated < gated - 0.02, "ungated {ungated}, gated {gated}");
}

// --------------------------------------------------------------------------- DriftMonitor
fn drift(window: usize, alert_ratio: f64, min_events: usize) -> DriftMonitor {
    DriftMonitor::new(window, alert_ratio, min_events).unwrap()
}

fn assert_close(a: f64, b: f64, tolerance: f64) {
    assert!((a - b).abs() <= tolerance, "{a} != {b}");
}

#[test]
fn test_drift_error_rate_and_min_events() {
    let mut d = drift(40, 0.35, 15);
    assert_eq!(d.error_rate(), 0.0);
    for _ in 0..10 {
        d.record(Some(0), 1); // every prediction wrong
    }
    assert_eq!(d.error_rate(), 1.0);
    assert!(!d.should_alert(100.0, ALERT_COOLDOWN_S)); // fewer than min_events
    d.record(None, 1); // no prediction: ignored
    assert_eq!(d.event_count(), 10);
    for _ in 0..5 {
        d.record(Some(1), 1);
    }
    assert_close(d.error_rate(), 10.0 / 15.0, 1e-6);
    assert!(d.should_alert(100.0, ALERT_COOLDOWN_S));
}

#[test]
fn test_drift_does_not_alert_when_accurate() {
    let mut d = drift(40, 0.35, 15);
    for i in 0..40 {
        d.record(Some(0), if i % 5 == 0 { 1 } else { 0 }); // 20 % disagreement
    }
    assert_close(d.error_rate(), 0.2, 1e-6);
    assert!(!d.should_alert(0.0, ALERT_COOLDOWN_S));
}

#[test]
fn test_drift_cooldown() {
    let mut d = drift(20, 0.5, 5);
    for _ in 0..5 {
        d.record_wrong_switch();
    }
    assert!(d.should_alert(1000.0, ALERT_COOLDOWN_S));
    assert!(!d.should_alert(1001.0, ALERT_COOLDOWN_S));
    assert!(!d.should_alert(1000.0 + 1799.0, ALERT_COOLDOWN_S));
    assert!(d.should_alert(1000.0 + 1800.0, ALERT_COOLDOWN_S));
    assert!(!d.should_alert(3000.0, 300.0));
    assert!(d.should_alert(3100.0, 300.0));
}

#[test]
fn test_switch_outcomes_count_both_ways() {
    // With adaptive learning off only switches produce events. 5 % of 300 switches
    // undone is an accurate model, not a reason to alert.
    let mut d = DriftMonitor::default();
    let mut alerts = 0;
    for i in 0..300 {
        d.record_switch(i % 20 != 0);
        alerts += i32::from(d.should_alert(f64::from(i * 600), ALERT_COOLDOWN_S));
    }
    assert_eq!(alerts, 0);
    assert_close(d.error_rate(), 0.05, 0.03);
    d.record_switch(false);
    d.record_wrong_switch();
    assert_eq!(d.event_count(), 40);
}

#[test]
fn test_mostly_undone_switches_still_alert() {
    let mut d = drift(20, 0.35, 15);
    for i in 0..20 {
        d.record_switch(i % 2 == 0);
    }
    assert_eq!(d.error_rate(), 0.5);
    assert!(d.should_alert(0.0, ALERT_COOLDOWN_S));
}

#[test]
fn test_drift_window_rolls_over() {
    let mut d = drift(10, 0.5, 5);
    for _ in 0..10 {
        d.record(Some(0), 1);
    }
    for _ in 0..10 {
        d.record(Some(1), 1);
    }
    assert_eq!(d.event_count(), 10);
    assert_eq!(d.error_rate(), 0.0);
    assert!(!d.should_alert(0.0, ALERT_COOLDOWN_S));
}

#[test]
fn test_drift_reset_clears_events_and_cooldown() {
    let mut d = drift(40, 0.5, 3);
    for _ in 0..3 {
        d.record(Some(0), 1);
    }
    assert!(d.should_alert(10.0, ALERT_COOLDOWN_S));
    d.reset();
    assert_eq!(d.event_count(), 0);
    for _ in 0..3 {
        d.record(Some(0), 1);
    }
    assert!(d.should_alert(11.0, ALERT_COOLDOWN_S)); // cooldown was reset too
}

#[test]
fn test_invalid_drift_arguments() {
    let cases = [(0, 0.35, 15), (40, 0.0, 15), (40, 1.5, 15), (40, 0.35, 0)];
    for (window, alert_ratio, min_events) in cases {
        let message = error_of(DriftMonitor::new(window, alert_ratio, min_events));
        assert!(message.contains("window"), "{message}");
    }
}
