//! Adaptive camera frame rate - the main lever for low CPU use.
//!
//! Face analysis dominates the app's CPU cost, so the rate at which frames are
//! analysed is chosen from what is going on: fast while a switch is being
//! considered or during calibration, slow while the user reads or types on one
//! monitor, slower still when nobody is there. Each profile maps these *modes* to
//! frames per second.

use crate::types::TrackingState;

/// Modes a [`RatePolicy`] distinguishes (the keys of every profile).
pub const MODES: [&str; 6] = ["calibrating", "away", "active", "noface", "typing", "idle"];

/// Mode reported for states in which the worker does not use the camera.
pub const CAMERA_OFF_MODE: &str = "off";

/// Frames per second of one profile, by mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rates {
    pub active: f64,
    pub idle: f64,
    pub typing: f64,
    pub noface: f64,
    pub away: f64,
    pub calibrating: f64,
}

impl Rates {
    /// The rate of a mode name from [`MODES`].
    pub fn get(&self, mode: &str) -> Option<f64> {
        Some(match mode {
            "active" => self.active,
            "idle" => self.idle,
            "typing" => self.typing,
            "noface" => self.noface,
            "away" => self.away,
            "calibrating" => self.calibrating,
            _ => return None,
        })
    }
}

/// Frames per second for each profile and mode.
pub const PROFILES: [(&str, Rates); 3] = [
    (
        "eco",
        Rates {
            active: 8.0,
            idle: 2.0,
            typing: 1.0,
            noface: 1.0,
            away: 0.5,
            calibrating: 15.0,
        },
    ),
    (
        "balanced",
        Rates {
            active: 12.0,
            idle: 4.0,
            typing: 2.0,
            noface: 2.0,
            away: 1.0,
            calibrating: 24.0,
        },
    ),
    (
        "responsive",
        Rates {
            active: 20.0,
            idle: 8.0,
            typing: 4.0,
            noface: 3.0,
            away: 1.0,
            calibrating: 30.0,
        },
    ),
];

pub const DEFAULT_PROFILE: &str = "balanced";

/// States in which the camera is released; the rate is irrelevant there.
pub const CAMERA_OFF_STATES: [TrackingState; 4] = [
    TrackingState::Paused,
    TrackingState::Privacy,
    TrackingState::Locked,
    TrackingState::Yielded,
];

/// Value returned by [`RatePolicy::interval`] / [`RatePolicy::fps`] for
/// camera-off states (the worker is inactive anyway).
pub const CAMERA_OFF_VALUE: f64 = 1.0;

/// What the controller knows right now that matters for the frame rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RateContext {
    pub state: TrackingState,
    /// The switch decider is considering a switch (`Decision::pending`).
    pub pending_switch: bool,
    /// The gaze point is moving quickly (a switch may be coming).
    pub gaze_moving: bool,
    /// The user typed recently.
    pub typing: bool,
    /// A face was seen in the latest observation.
    pub face_present: bool,
    /// The walk-away countdown is running (sample fast so it cancels quickly).
    pub presence_warning: bool,
    /// The camera preview window is open.
    pub preview: bool,
}

impl RateContext {
    /// A context for `state` with the Python defaults (a face present, nothing else going on).
    pub fn new(state: TrackingState) -> Self {
        Self {
            state,
            pending_switch: false,
            gaze_moving: false,
            typing: false,
            face_present: true,
            presence_warning: false,
            preview: false,
        }
    }
}

/// Chooses the interval between analysed frames from a [`RateContext`].
#[derive(Debug, Clone)]
pub struct RatePolicy {
    profile: &'static str,
    rates: Rates,
}

impl Default for RatePolicy {
    fn default() -> Self {
        Self::new(DEFAULT_PROFILE)
    }
}

impl RatePolicy {
    pub fn new(profile: &str) -> Self {
        let mut policy = Self {
            profile: PROFILES[1].0,
            rates: PROFILES[1].1,
        };
        policy.set_profile(profile);
        policy
    }

    pub fn profile(&self) -> &str {
        self.profile
    }

    /// Switch profile; unknown names fall back to `"balanced"` with a warning.
    pub fn set_profile(&mut self, profile: &str) {
        let found = PROFILES.iter().find(|(n, _)| *n == profile).or_else(|| {
            log::warn!("Unknown performance profile {profile:?}; using {DEFAULT_PROFILE:?}");
            PROFILES.iter().find(|(n, _)| *n == DEFAULT_PROFILE)
        });
        if let Some((name, rates)) = found {
            self.profile = name;
            self.rates = *rates;
        }
    }

    /// Name of the mode selected for `ctx` (one of [`MODES`] or `"off"`).
    pub fn mode(&self, ctx: &RateContext) -> &'static str {
        let state = ctx.state;
        if CAMERA_OFF_STATES.contains(&state) {
            return CAMERA_OFF_MODE;
        }
        if state == TrackingState::Calibrating || ctx.preview {
            return "calibrating";
        }
        if state == TrackingState::Away {
            return "away";
        }
        if ctx.presence_warning {
            // Fast so the countdown is cancelled as soon as the user is back.
            return "active";
        }
        if !ctx.face_present {
            return "noface";
        }
        if ctx.pending_switch || ctx.gaze_moving {
            return "active";
        }
        if ctx.typing {
            return "typing";
        }
        "idle"
    }

    /// Target analysed frames per second for `ctx`.
    pub fn fps(&self, ctx: &RateContext) -> f64 {
        // Every mode but "off" is a key of the rates.
        self.rates.get(self.mode(ctx)).unwrap_or(CAMERA_OFF_VALUE)
    }

    /// Seconds between analysed frames for `ctx`.
    pub fn interval(&self, ctx: &RateContext) -> f64 {
        match self.rates.get(self.mode(ctx)) {
            Some(fps) => 1.0 / fps,
            None => CAMERA_OFF_VALUE,
        }
    }
}

#[cfg(test)]
#[path = "scheduler_tests.rs"]
mod tests;
