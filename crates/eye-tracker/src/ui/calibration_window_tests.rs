// Tests of ui/calibration_window.rs, ported from the calibration part of tests/test_ui_dialogs.py.
//
// The Python tests drove Qt widgets through a FakeController (signals and methods) and a
// FakeClock. Here `Env` plays both: it owns the fake clock and the controller's plain values
// that `Inputs` carries, and it applies the `CalibrationAction`s the way the controller would
// (`begun`, `finished`, `notifications`, `states`, `results`). The random streams come from
// testing::gaze_synth::Rng, not numpy.
//
// Not ported (they exercised Qt itself):
// * test_calibration_opens_one_frameless_topmost_surface_per_monitor: the Qt window flags and
//   `isVisible()` are replaced by the viewport builder's decorations, window level, position and
//   size.
// * test_full_calibration_produces_calibration_data: the `processEvents()` calls and the checks
//   that every card label is visible, tall enough, and that the card is as high as its layout
//   needs concern QLayout; egui sizes the card while drawing.
// * test_screen_geometry_and_primary_changes_cancel: the Qt `geometryChanged` and
//   `primaryScreenChanged` signals are replaced by `on_screens_changed` and the comparison of
//   `Inputs::monitors` with the monitors the calibration started on.

use super::*;
use crate::config::Settings;
use crate::testing::gaze_synth::Rng;
use crate::types::Rect;

const LEFT_RECT: Rect = Rect::new(0, 0, 1920, 1080);
const RIGHT_RECT: Rect = Rect::new(1920, 0, 1920, 1080);
const BACKEND: (&str, &str) = ("facemesh", "facemesh-pose-iris-1");

fn left() -> Monitor {
    Monitor {
        primary: true,
        ..Monitor::new(0, "Left", LEFT_RECT)
    }
}

fn right() -> Monitor {
    Monitor::new(1, "Right", RIGHT_RECT)
}

/// The fake controller and clock of the Python tests.
struct Env {
    window: CalibrationWindow,
    now: f64,
    monitors: Vec<Monitor>,
    backend: (String, String),
    gaze_indices: Option<Vec<usize>>,
    camera: String,
    begun: usize,
    finished: Vec<Option<CalibrationData>>,
    notifications: Vec<(String, String)>,
    states: Vec<CalibrationState>,
    results: Vec<bool>,
}

impl Env {
    fn new() -> Env {
        Env::with(vec![left(), right()], false)
    }

    fn with(monitors: Vec<Monitor>, fit_in_thread: bool) -> Env {
        Env {
            window: CalibrationWindow::new(CalibrationOptions {
                points_per_monitor: 5,
                settle_s: 0.2,
                collect_s: 0.3,
                min_samples: 3,
                max_retries: 1,
                fit_in_thread,
                ..CalibrationOptions::default()
            }),
            now: 1000.0,
            monitors,
            backend: (BACKEND.0.to_string(), BACKEND.1.to_string()),
            // The synthetic observations encode the dot's position in features 0 and 1.
            gaze_indices: Some(vec![0, 1]),
            camera: Settings::default().camera.device,
            begun: 0,
            finished: Vec::new(),
            notifications: Vec::new(),
            states: Vec::new(),
            results: Vec::new(),
        }
    }

    fn inputs(&self) -> Inputs {
        Inputs {
            now: self.now,
            monitors: self.monitors.clone(),
            backend_info: self.backend.clone(),
            gaze_feature_indices: self.gaze_indices.clone(),
            camera_device: self.camera.clone(),
            observations: Vec::new(),
        }
    }

    /// The controller's side: what the window asked for.
    fn pump(&mut self) {
        for action in self.window.take_actions() {
            match action {
                CalibrationAction::Started => self.begun += 1,
                CalibrationAction::StateChanged(state) => self.states.push(state),
                CalibrationAction::Notify { title, message } => {
                    self.notifications.push((title, message));
                }
                CalibrationAction::Finished { data, grade } => {
                    assert_eq!(Some(grade.as_str()), data.grade());
                    self.finished.push(Some(*data));
                    self.results.push(true);
                }
                CalibrationAction::Cancelled => {
                    self.finished.push(None);
                    self.results.push(false);
                }
                CalibrationAction::NotStarted => self.results.push(false),
            }
        }
    }

    fn advance(&mut self, seconds: f64) {
        self.now += seconds;
    }

    fn start(&mut self) -> bool {
        let started = self.window.start(&self.inputs());
        self.pump();
        started
    }

    fn begin(&mut self) {
        self.window.begin(&self.inputs());
        self.pump();
    }

    fn tick(&mut self) {
        self.window.tick(&self.inputs());
        self.pump();
    }

    fn key(&mut self, key: Key) -> bool {
        let handled = self.window.handle_key(&self.inputs(), key);
        self.pump();
        handled
    }

    fn save(&mut self) -> bool {
        let saved = self.window.save(&self.inputs());
        self.pump();
        saved
    }

    fn cancel(&mut self) {
        self.window.cancel();
        self.pump();
    }

    fn check_idle(&mut self) -> bool {
        let closed = self.window.check_idle(self.now);
        self.pump();
        closed
    }

    /// `controller.observation.emit(obs)`.
    fn emit(&mut self, obs: Observation) {
        self.window.on_observation(self.now, &obs);
    }

    fn hint(&self) -> String {
        self.window.hint(self.now)
    }

    /// Advance the fake clock tick by tick, feeding one observation per tick.
    fn run_dots(
        &mut self,
        make: &mut dyn FnMut(&Env, Option<CalibrationTarget>) -> Option<Observation>,
    ) {
        for _ in 0..2000 {
            if self.window.state() != CalibrationState::Running {
                return;
            }
            if let Some(obs) = make(self, self.window.current_target()) {
                self.emit(obs);
            }
            self.advance(0.05);
            self.tick();
        }
        panic!("calibration did not finish");
    }

    /// `_run_to_result`: start, begin, run every dot and tick once more.
    fn run_to_result(
        &mut self,
        make: &mut dyn FnMut(&Env, Option<CalibrationTarget>) -> Option<Observation>,
    ) {
        self.start();
        self.begin();
        self.run_dots(make);
        self.tick();
    }
}

struct Look {
    face: bool,
    usable: bool,
    skipped: bool,
    frame_size: (u32, u32),
}

impl Default for Look {
    fn default() -> Self {
        Look {
            face: true,
            usable: true,
            skipped: false,
            frame_size: (0, 0),
        }
    }
}

/// A synthetic observation whose features encode the dot being looked at.
fn observation(
    target: Option<CalibrationTarget>,
    now: f64,
    rng: &mut Rng,
    look: Look,
) -> Observation {
    let features = match target {
        Some(target) if look.face && look.usable => {
            let (x, y) = (target.x / 1000.0, target.y / 1000.0);
            Some(vec![
                x + rng.normal(0.0, 0.004),
                y + rng.normal(0.0, 0.004),
                0.3 * x - 0.2 * y + rng.normal(0.0, 0.004),
                1.0 + rng.normal(0.0, 0.004),
            ])
        }
        _ => None,
    };
    Observation {
        features,
        quality: if look.face { 1.0 } else { 0.0 },
        skipped: look.skipped,
        frame_size: look.frame_size,
        ..Observation::new(now, u32::from(look.face))
    }
}

fn no_face(now: f64, rng: &mut Rng) -> Observation {
    observation(
        None,
        now,
        rng,
        Look {
            face: false,
            ..Look::default()
        },
    )
}

/// An observation of the dot currently shown, as a `make` for `run_dots`.
fn looking_at_the_dot(
    seed: u64,
) -> impl FnMut(&Env, Option<CalibrationTarget>) -> Option<Observation> {
    let mut rng = Rng::new(seed);
    move |env, target| Some(observation(target, env.now, &mut rng, Look::default()))
}

fn only_skipped(seed: u64) -> impl FnMut(&Env, Option<CalibrationTarget>) -> Option<Observation> {
    let mut rng = Rng::new(seed);
    move |env, target| {
        Some(observation(
            target,
            env.now,
            &mut rng,
            Look {
                skipped: true,
                ..Look::default()
            },
        ))
    }
}

/// Wait for the thread-backed fit like `_wait_until` did: tick every 30 ms of fake time.
fn wait_for_the_fit(env: &mut Env) -> bool {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while std::time::Instant::now() < deadline {
        env.advance(0.03);
        env.tick();
        if env.window.state() != CalibrationState::Fitting {
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    false
}

fn feed_dot_observations(env: &mut Env, rng: &mut Rng, count: usize) {
    for _ in 0..count {
        let obs = observation(env.window.current_target(), env.now, rng, Look::default());
        env.emit(obs);
        env.advance(0.05);
        env.tick();
    }
}

// ---------------------------------------------------------------------------- tests
#[test]
fn test_calibration_opens_one_frameless_topmost_surface_per_monitor() {
    let mut env = Env::new();
    assert!(env.start());
    assert_eq!(env.begun, 1);
    assert_eq!(env.window.state(), CalibrationState::Intro);
    assert_eq!(env.states, [CalibrationState::Intro]);
    let monitors: Vec<&Monitor> = env.window.surfaces().iter().map(|s| &s.monitor).collect();
    assert_eq!(monitors, [&left(), &right()]);
    for surface in env.window.surfaces() {
        let builder = surface_viewport_builder(&surface.monitor);
        assert_eq!(builder.decorations, Some(false));
        assert_eq!(
            builder.window_level,
            Some(egui::viewport::WindowLevel::AlwaysOnTop)
        );
        let rect = surface.monitor.rect;
        assert_eq!(
            builder.position,
            Some(Pos2::new(rect.x as f32, rect.y as f32))
        );
        assert_eq!(
            builder.inner_size,
            Some(Vec2::new(rect.w as f32, rect.h as f32))
        );
    }
    // Calling start() again only raises the windows.
    assert!(env.start());
    assert_eq!(env.begun, 1);
}

#[test]
fn test_full_calibration_produces_calibration_data() {
    let mut env = Env::new();
    let mut rng = Rng::new(7);
    env.start();
    // The intro shows whether the camera sees a face.
    let obs = observation(None, env.now, &mut rng, Look::default());
    env.emit(obs);
    env.tick();
    for surface in env.window.surfaces() {
        assert!(
            surface
                .card
                .label("face")
                .unwrap()
                .contains("Face detected")
        );
    }

    assert!(env.key(Key::Space));
    assert_eq!(env.window.state(), CalibrationState::Running);
    assert_eq!(env.window.plan().len(), 10);
    let mut seen_monitors = std::collections::BTreeSet::new();
    let progress_before = env.window.progress();
    env.run_dots(&mut |env, target| {
        if let Some(target) = target {
            seen_monitors.insert(target.monitor_index);
        }
        Some(observation(target, env.now, &mut rng, Look::default()))
    });
    assert_eq!(seen_monitors, [0, 1].into());
    assert!(progress_before < 0.1);
    assert_eq!(env.window.state(), CalibrationState::Fitting);
    env.tick(); // the fit runs on the tick after "Calculating…" was shown
    assert_eq!(env.window.state(), CalibrationState::Result);
    let report = env.window.report().unwrap().clone();
    assert_eq!(report.grade, "excellent");
    assert_eq!(
        report.per_monitor_accuracy,
        BTreeMap::from([(0, 1.0), (1, 1.0)])
    );
    let card = &env.window.surfaces()[0].card;
    assert_eq!(card.button_names(), ["save", "retry", "cancel"]);
    assert_eq!(card.label("badge"), Some("EXCELLENT"));

    assert!(env.key(Key::Enter));
    assert_eq!(env.window.state(), CalibrationState::Closed);
    assert_eq!(env.results, [true]);
    assert!(env.window.surfaces().is_empty());
    assert_eq!(env.finished.len(), 1);
    let data = env.finished[0].as_ref().unwrap();
    assert_eq!(
        (data.backend.as_str(), data.feature_version.as_str()),
        BACKEND
    );
    assert_eq!(data.monitors, [left(), right()]);
    assert_eq!(data.layout_signature, layout_signature(&[left(), right()]));
    assert!(data.implicit_samples.is_empty());
    assert_eq!(data.samples.len(), report.n_samples);
    assert!(data.samples.len() >= 30);
    let saved_points: std::collections::BTreeSet<i64> =
        data.samples.iter().map(|s| s.point_id).collect();
    let plan_points: std::collections::BTreeSet<i64> =
        env.window.plan().iter().map(|t| t.point_id).collect();
    assert_eq!(saved_points, plan_points);
    assert!(data.model.is_fitted());
    assert_eq!(data.report["grade"], "excellent");
    assert_eq!(data.grade(), Some("excellent"));
    assert!(data.created_at.ends_with("+00:00") || data.created_at.ends_with('Z'));
    assert_eq!(env.window.data().unwrap().created_at, data.created_at);
    let (ok, reason) = data.is_compatible(BACKEND.0, BACKEND.1, &[left(), right()], None, None);
    assert!(ok, "{reason}");
}

#[test]
fn test_calibration_fits_on_a_background_thread() {
    let mut env = Env::with(vec![left(), right()], true);
    env.start();
    env.begin();
    env.run_dots(&mut looking_at_the_dot(3));
    assert_eq!(env.window.state(), CalibrationState::Fitting);
    assert!(env.window.surfaces()[0].card.has_spinner());
    assert!(wait_for_the_fit(&mut env));
    assert_eq!(env.window.state(), CalibrationState::Result);
    assert!(env.save());
    assert!(matches!(env.finished.last(), Some(Some(_))));
}

#[test]
fn test_escape_cancels_the_calibration() {
    let mut env = Env::new();
    env.start();
    env.begin();
    feed_dot_observations(&mut env, &mut Rng::new(1), 5);
    assert!(env.key(Key::Escape)); // any screen works
    assert_eq!(env.window.state(), CalibrationState::Closed);
    assert!(matches!(env.finished.as_slice(), [None]));
    assert_eq!(env.results, [false]);
    assert!(env.window.surfaces().is_empty());
    env.cancel(); // idempotent
    assert!(matches!(env.finished.as_slice(), [None]));
}

#[test]
fn test_closing_a_surface_cancels() {
    let mut env = Env::new();
    env.start();
    env.window.surface_closed(); // e.g. Alt+F4
    env.pump();
    assert_eq!(env.window.state(), CalibrationState::Closed);
    assert!(matches!(env.finished.as_slice(), [None]));
}

#[test]
fn test_missing_face_pauses_and_shows_a_hint() {
    let mut env = Env::new();
    let mut rng = Rng::new(2);
    env.start();
    env.begin();
    feed_dot_observations(&mut env, &mut rng, 5);
    for _ in 0..40 {
        // 2 s of frames without a face (the grace period is 1.5 s)
        let obs = no_face(env.now, &mut rng);
        env.emit(obs);
        env.advance(0.05);
        env.tick();
    }
    assert!(env.window.paused());
    assert_eq!(env.hint(), "Can't see your face — check the camera");
    // While paused the dots stand still.
    let progress = env.window.progress();
    for _ in 0..40 {
        let obs = no_face(env.now, &mut rng);
        env.emit(obs);
        env.advance(0.05);
        env.tick();
    }
    assert_eq!(env.window.progress(), progress);
    // No frames at all: the camera itself is the problem.
    env.advance(3.0);
    env.tick();
    assert_eq!(env.hint(), "Waiting for the camera…");
    // The face returns: the dots continue.
    let obs = observation(
        env.window.current_target(),
        env.now,
        &mut rng,
        Look::default(),
    );
    env.emit(obs);
    env.advance(0.05);
    env.tick();
    assert!(!env.window.paused());
    assert_eq!(env.hint(), "");
}

#[test]
fn test_space_pauses_manually() {
    let mut env = Env::new();
    let mut rng = Rng::new(4);
    env.start();
    env.begin();
    env.key(Key::Space);
    assert!(env.window.paused());
    let progress = env.window.progress();
    feed_dot_observations(&mut env, &mut rng, 20);
    assert_eq!(env.window.progress(), progress);
    assert!(env.hint().contains("Paused"));
    env.key(Key::Space);
    assert!(!env.window.paused());
}

#[test]
fn test_skipped_observations_are_never_recorded_and_retry_works() {
    let mut env = Env::new();
    env.start();
    env.begin();
    // A face is visible (no pause) but every frame is a motion-gate copy.
    env.run_dots(&mut only_skipped(5));
    assert_eq!(env.window.state(), CalibrationState::Fitting);
    env.tick();
    assert_eq!(env.window.state(), CalibrationState::Error);
    assert!(env.window.error().contains("10 of 10 dots"));
    assert!(env.finished.is_empty()); // still open: the user may retry
    // R starts over.
    env.key(Key::R);
    assert_eq!(env.window.state(), CalibrationState::Running);
    assert_eq!(env.window.error(), "");
    env.run_dots(&mut looking_at_the_dot(50));
    env.tick();
    assert_eq!(env.window.state(), CalibrationState::Result);
}

#[test]
fn test_retry_from_the_result_screen() {
    let mut env = Env::new();
    env.start();
    env.begin();
    env.run_dots(&mut looking_at_the_dot(6));
    env.tick();
    assert_eq!(env.window.state(), CalibrationState::Result);
    env.key(Key::R);
    assert_eq!(env.window.state(), CalibrationState::Running);
    assert!(env.window.report().is_none());
    assert!(env.window.progress() < 0.1);
    assert!(env.finished.is_empty());
}

#[test]
fn test_save_refuses_without_backend_info() {
    let mut env = Env::new();
    env.start();
    env.begin();
    env.run_dots(&mut looking_at_the_dot(8));
    env.tick();
    env.backend = (String::new(), String::new());
    assert!(!env.save());
    assert_eq!(env.window.state(), CalibrationState::Error);
    assert!(env.finished.is_empty());
}

#[test]
fn test_calibration_without_monitors_finishes_immediately() {
    let mut env = Env::with(Vec::new(), false);
    assert!(!env.start());
    assert_eq!(env.results, [false]);
    assert_eq!(env.begun, 0);
    assert!(!env.window.is_active());
}

#[test]
fn test_calibration_fits_with_the_gaze_features_and_records_the_camera() {
    let mut env = Env::new();
    env.camera = "1".into();
    let mut rng = Rng::new(12);
    env.run_to_result(&mut |env, target| {
        Some(observation(
            target,
            env.now,
            &mut rng,
            Look {
                frame_size: (640, 480),
                ..Look::default()
            },
        ))
    });
    assert_eq!(env.window.state(), CalibrationState::Result);
    // controller.gaze_feature_indices() reached the fit.
    assert_eq!(env.window.model().unwrap().nonlinear(), Some(&[0, 1][..]));
    assert!(env.save());
    let data = env.finished.last().unwrap().as_ref().unwrap();
    assert_eq!(data.camera, "1");
    assert_eq!(data.frame_size, (640, 480));
    assert_eq!(
        data.camera_matches(Some("1"), Some((1280, 960))),
        (true, String::new())
    );
}

fn backend_inputs(name: &str) -> Inputs {
    Inputs {
        backend_info: (name.to_string(), "v".to_string()),
        ..Inputs::default()
    }
}

#[test]
fn test_gaze_features_come_from_the_backend_class_without_a_controller_api() {
    let lite = backend_class("lite").unwrap();
    let expected = gaze_feature_indices(lite.feature_names(), lite.gaze_features());
    assert!(expected.is_some());
    assert_eq!(controller_gaze_features(&backend_inputs("lite")), expected);
    assert_eq!(
        controller_gaze_features(&backend_inputs("no-such-backend")),
        None
    );
    assert_eq!(controller_gaze_features(&backend_inputs("")), None);
    // No backend info at all (the Python `object()`).
    assert_eq!(controller_gaze_features(&Inputs::default()), None);
    // Indices that do not fit the recorded features degrade to "all nonlinear".
    let sample = CalibrationSample::new(vec![0.0; 4], 0.0, 0.0, 0, 0);
    assert_eq!(
        fit_features(Some(vec![0, 7]), std::slice::from_ref(&sample)),
        None
    );
    assert_eq!(
        fit_features(Some(vec![0, 1]), std::slice::from_ref(&sample)),
        Some(vec![0, 1])
    );
    assert_eq!(fit_features(None, &[sample]), None);
}

#[test]
fn test_an_untouched_intro_closes_itself() {
    // Regression (ui_app-06): an abandoned calibration suspended walk-away locking.
    let mut env = Env::new();
    env.start();
    env.advance(IDLE_TIMEOUT_S - 1.0);
    env.tick();
    assert_eq!(env.window.state(), CalibrationState::Intro);
    env.key(Key::Other); // any key: someone is there
    env.advance(IDLE_TIMEOUT_S - 1.0);
    assert!(!env.check_idle());
    env.advance(2.0);
    assert!(env.check_idle());
    assert_eq!(env.window.state(), CalibrationState::Closed);
    assert!(matches!(env.finished.as_slice(), [None])); // the controller leaves CALIBRATING
    assert_eq!(env.results, [false]);
    let (title, text) = env.notifications.last().unwrap();
    assert_eq!(title, "Calibration closed");
    assert!(text.contains("Calibrate now"));
}

#[test]
fn test_an_unsaved_result_closes_itself() {
    let mut env = Env::new();
    env.run_to_result(&mut looking_at_the_dot(11));
    assert_eq!(env.window.state(), CalibrationState::Result);
    // The animation timer is stopped on this screen; the watchdog keeps running.
    assert!(env.window.timer_ms().is_none());
    assert!(env.window.watchdog_active());
    env.advance(IDLE_TIMEOUT_S + 1.0);
    assert!(env.check_idle());
    assert!(matches!(env.finished.as_slice(), [None])); // nothing saved without Enter
}

#[test]
fn test_an_error_screen_closes_itself() {
    let mut env = Env::new();
    env.run_to_result(&mut only_skipped(13));
    assert_eq!(env.window.state(), CalibrationState::Error);
    assert!(env.window.watchdog_active());
    env.advance(IDLE_TIMEOUT_S + 1.0);
    assert!(env.check_idle());
    assert_eq!(env.window.state(), CalibrationState::Closed);
}

#[test]
fn test_dots_paused_without_a_face_close_the_calibration() {
    let mut env = Env::new();
    let mut rng = Rng::new(14);
    env.start();
    env.begin();
    for _ in 0..40 {
        // 2 s without a face: the dots pause
        let obs = no_face(env.now, &mut rng);
        env.emit(obs);
        env.advance(0.05);
        env.tick();
    }
    assert!(env.window.paused());
    env.advance(NO_FACE_TIMEOUT_S - 3.0);
    env.tick();
    assert_eq!(env.window.state(), CalibrationState::Running);
    env.advance(3.0);
    env.tick();
    assert_eq!(env.window.state(), CalibrationState::Closed);
    assert!(env.notifications.last().unwrap().1.contains("No face"));
}

#[test]
fn test_a_manual_pause_left_alone_closes_the_calibration() {
    let mut env = Env::new();
    let mut rng = Rng::new(15);
    env.start();
    env.begin();
    let obs = observation(
        env.window.current_target(),
        env.now,
        &mut rng,
        Look::default(),
    );
    env.emit(obs);
    env.key(Key::Space);
    assert!(env.window.paused());
    for _ in 0..3 {
        env.advance(IDLE_TIMEOUT_S / 4.0);
        let obs = observation(
            env.window.current_target(),
            env.now,
            &mut rng,
            Look::default(),
        );
        env.emit(obs);
        env.tick();
    }
    assert_eq!(env.window.state(), CalibrationState::Running); // a face alone is not a key press
    env.advance(IDLE_TIMEOUT_S / 4.0 + 1.0);
    env.tick();
    assert_eq!(env.window.state(), CalibrationState::Closed);
}

#[test]
fn test_a_layout_change_at_save_cancels_instead_of_offering_a_retry() {
    // Regression (ui_app-05): retrying planned on the stale layout forever.
    let mut env = Env::new();
    env.run_to_result(&mut looking_at_the_dot(11));
    assert_eq!(env.window.state(), CalibrationState::Result);
    env.monitors = vec![
        left(),
        Monitor::new(1, "Right", Rect::new(1920, 0, 2560, 1440)),
    ];
    assert!(!env.save());
    assert_eq!(env.window.state(), CalibrationState::Closed);
    assert!(matches!(env.finished.as_slice(), [None]));
    assert_eq!(env.notifications.last().unwrap().0, "Calibration cancelled");
}

#[test]
fn test_retry_rechecks_the_monitor_layout() {
    let mut env = Env::new();
    env.run_to_result(&mut only_skipped(16));
    assert_eq!(env.window.state(), CalibrationState::Error);
    // The controller reports the new layout only after its debounce.
    env.monitors = vec![
        Monitor {
            primary: true,
            ..Monitor::new(0, "Left", Rect::new(0, 0, 2560, 1440))
        },
        right(),
    ];
    env.key(Key::R);
    assert_eq!(env.window.state(), CalibrationState::Closed);
    assert_eq!(env.notifications.last().unwrap().0, "Calibration cancelled");
}

#[test]
fn test_screen_geometry_and_primary_changes_cancel() {
    let mut env = Env::new();
    env.start();
    env.window.on_screens_changed(); // resolution change or rearrangement
    env.pump();
    assert_eq!(env.window.state(), CalibrationState::Closed);
    assert!(matches!(env.finished.as_slice(), [None]));
    env.window.on_screens_changed(); // closed: nothing happens
    env.pump();
    assert_eq!(env.finished.len(), 1);

    // The controller's monitors differ from the ones being calibrated: a new primary.
    let mut second = Env::new();
    second.start();
    second.monitors = vec![
        Monitor::new(0, "Left", LEFT_RECT),
        Monitor {
            primary: true,
            ..right()
        },
    ];
    // Same rectangles, so the layout signature alone would not notice.
    assert!(!second.window.layout_changed(&second.inputs()));
    let inputs = second.inputs();
    second.window.update(&inputs);
    second.pump();
    assert_eq!(second.window.state(), CalibrationState::Closed);

    // A moved or resized monitor is noticed while the window is open.
    let mut third = Env::new();
    third.start();
    third.monitors = vec![
        left(),
        Monitor::new(1, "Right", Rect::new(1920, 0, 1280, 720)),
    ];
    let inputs = third.inputs();
    third.window.update(&inputs);
    third.pump();
    assert_eq!(third.window.state(), CalibrationState::Closed);
    assert_eq!(
        third.notifications.last().unwrap().0,
        "Calibration cancelled"
    );
}

#[test]
fn test_a_screen_without_usable_dots_is_named_on_the_result() {
    let third = Monitor::new(2, "Third", Rect::new(3840, 0, 1920, 1080));
    let mut env = Env::with(vec![left(), right(), third.clone()], false);
    let mut rng = Rng::new(17);
    env.run_to_result(&mut |env, target| {
        // The face is seen but not measurable while looking at the third screen.
        let usable = target.is_none_or(|t| t.monitor_index != third.index);
        Some(observation(
            target,
            env.now,
            &mut rng,
            Look {
                usable,
                ..Look::default()
            },
        ))
    });
    assert_eq!(env.window.state(), CalibrationState::Result);
    let report = env.window.report().unwrap();
    assert_eq!(report.uncovered_monitors, [third.index]);
    assert!(report.grade == "fair" || report.grade == "poor");
    let tip = env.window.surfaces()[0].card.label("tip").unwrap();
    assert!(tip.starts_with("Screen 3 was not calibrated"));
    assert!(tip.contains("press R"));
}
