//! Walk-away detection with a cancellable countdown.
//!
//! The monitor is fed a verdict about the camera image (face or no face) and the
//! time since the last keyboard/mouse input. When neither shows the user for long
//! enough it moves through `PRESENT -> WARNING -> AWAY` and emits events that the
//! controller turns into UI (countdown toast) and actions (lock, displays off).
//! Any sign of the user cancels the countdown or signals their return.
//!
//! Keyboard/mouse input counts as presence (by default) because the camera can
//! miss a face - a user leaning back, bad light, a turned head - and locking the
//! screen under someone who is typing is the worst possible failure.

use crate::config::PresenceSettings;

/// Input more recent than this counts as proof that someone is at the computer.
pub const INPUT_EVIDENCE_S: f64 = 1.5;

/// The countdown never starts before the face has been missing this long, even
/// when `warning_s >= away_timeout_s`; otherwise a single frame without a
/// detected face would flash the countdown.
pub const MIN_ABSENCE_BEFORE_WARNING_S: f64 = 1.0;

/// Where the user is, as far as the camera and input can tell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresenceState {
    Present,
    Warning,
    Away,
}

impl PresenceState {
    pub fn value(self) -> &'static str {
        match self {
            Self::Present => "present",
            Self::Warning => "warning",
            Self::Away => "away",
        }
    }
}

/// What a [`PresenceEvent`] reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresenceEventKind {
    /// Countdown started; `remaining_s` seconds until *away*.
    Warn,
    /// Countdown cancelled (the user is back, or presence detection was
    /// paused/disabled).
    Cancel,
    /// The user is gone; perform the configured action.
    Away,
    /// The user came back after being *away*.
    Return,
}

/// A state transition reported by [`PresenceMonitor::update`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PresenceEvent {
    pub kind: PresenceEventKind,
    pub remaining_s: f64,
}

impl PresenceEvent {
    pub fn new(kind: PresenceEventKind, remaining_s: f64) -> Self {
        Self { kind, remaining_s }
    }

    fn of(kind: PresenceEventKind) -> Self {
        Self::new(kind, 0.0)
    }
}

/// Tuning of [`PresenceMonitor`] (seconds).
#[derive(Debug, Clone, PartialEq)]
pub struct PresenceConfig {
    pub enabled: bool,
    /// Total time without presence evidence before the user counts as away.
    pub away_timeout_s: f64,
    /// Length of the cancellable countdown at the end of `away_timeout_s`
    /// (0 = no countdown).
    pub warning_s: f64,
    /// Recent keyboard/mouse input counts as presence.
    pub require_input_idle: bool,
}

impl Default for PresenceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            away_timeout_s: 45.0,
            warning_s: 10.0,
            require_input_idle: true,
        }
    }
}

impl PresenceConfig {
    /// Build from user settings (`s.action` is handled by the controller).
    pub fn from_settings(s: &PresenceSettings) -> Self {
        Self {
            enabled: s.enabled,
            away_timeout_s: s.away_timeout_s as f64,
            warning_s: s.warning_s as f64,
            require_input_idle: s.require_input_idle,
        }
    }
}

/// State machine `PRESENT -> WARNING -> AWAY` driven by face and input evidence.
#[derive(Debug)]
pub struct PresenceMonitor {
    config: PresenceConfig,
    state: PresenceState,
    last_seen: f64,
    /// Set when timing must restart on the next update (re-enabled after a
    /// period in which nothing was observed).
    restart: bool,
}

impl PresenceMonitor {
    pub fn new(config: PresenceConfig, now: f64) -> Self {
        Self {
            config,
            state: PresenceState::Present,
            last_seen: now,
            restart: false,
        }
    }

    pub fn state(&self) -> PresenceState {
        self.state
    }

    pub fn config(&self) -> &PresenceConfig {
        &self.config
    }

    /// Time of the most recent presence evidence (or end of the last unknown period).
    pub fn last_seen(&self) -> f64 {
        self.last_seen
    }

    /// Change the tuning; the state is re-evaluated on the next update.
    ///
    /// Disabling while *WARNING*/*AWAY* makes the next update emit `Cancel` /
    /// `Return` so that UI driven by earlier events is torn down.
    pub fn set_config(&mut self, config: PresenceConfig) {
        if !self.config.enabled && config.enabled {
            // Nothing was observed while disabled; that time is not absence.
            self.restart = true;
        }
        self.config = config;
    }

    /// Back to *PRESENT* without emitting events (e.g. after the session was unlocked).
    pub fn reset(&mut self, now: f64) {
        self.state = PresenceState::Present;
        self.last_seen = now;
        self.restart = false;
    }

    /// Seconds until *AWAY* (meaningful while *WARNING*; 0 when away, inf when disabled).
    pub fn remaining(&self, now: f64) -> f64 {
        if !self.config.enabled {
            return f64::INFINITY;
        }
        if self.state == PresenceState::Away {
            return 0.0;
        }
        let absent = if self.restart {
            0.0
        } else {
            (now - self.last_seen).max(0.0)
        };
        (self.config.away_timeout_s - absent).max(0.0)
    }

    /// Advance the state machine and return the transitions that happened.
    ///
    /// `face_present` is `None` when the camera cannot tell (camera off,
    /// privacy mode, paused): timers are frozen so that time without a camera is
    /// never counted as absence. `seconds_since_input` is `None` when the OS
    /// cannot report input idle time.
    pub fn update(
        &mut self,
        now: f64,
        face_present: Option<bool>,
        seconds_since_input: Option<f64>,
    ) -> Vec<PresenceEvent> {
        use PresenceEventKind::{Away, Cancel, Return, Warn};
        let cfg = &self.config;
        if !cfg.enabled || self.restart {
            self.restart = false;
            self.last_seen = now;
            return self.back_to_present(Some(Cancel), Some(Return));
        }

        let Some(face_present) = face_present else {
            self.last_seen = now;
            // A countdown cannot continue without a camera; take it down rather
            // than leave the toast frozen on screen. AWAY stays AWAY: the
            // controller resets presence when the session is unlocked.
            if self.state == PresenceState::Warning {
                return self.back_to_present(Some(Cancel), None);
            }
            return Vec::new();
        };

        let evidence = face_present
            || (cfg.require_input_idle
                && seconds_since_input.is_some_and(|s| s.is_finite() && s < INPUT_EVIDENCE_S));
        if evidence {
            self.last_seen = now;
            return self.back_to_present(Some(Cancel), Some(Return));
        }

        if self.state == PresenceState::Away {
            return Vec::new();
        }

        let timeout = cfg.away_timeout_s.max(0.0);
        let warning = cfg.warning_s.max(0.0);
        let threshold = if warning > 0.0 {
            (timeout - warning).max(MIN_ABSENCE_BEFORE_WARNING_S.min(timeout))
        } else {
            timeout
        };
        let mut absent = (now - self.last_seen).max(0.0);

        if absent < threshold {
            if self.state == PresenceState::Warning {
                // Only reachable after a config change lengthened the timeout.
                return self.back_to_present(Some(Cancel), None);
            }
            return Vec::new();
        }

        if self.state == PresenceState::Present && warning > 0.0 {
            if absent >= timeout {
                // Updates were sparse (process suspended, very low frame rate) and
                // the whole countdown elapsed unseen. Never act without warning:
                // restart the countdown so that the user always gets `warning_s`.
                self.last_seen = now - threshold;
                absent = threshold;
            }
            self.state = PresenceState::Warning;
            let remaining = (timeout - absent).max(0.0);
            log::info!("No presence for {absent:.0} s; away in {remaining:.0} s");
            return vec![PresenceEvent::new(Warn, remaining)];
        }

        if absent >= timeout {
            self.state = PresenceState::Away;
            log::info!("User away (no presence for {absent:.0} s)");
            return vec![PresenceEvent::of(Away)];
        }
        Vec::new()
    }

    fn back_to_present(
        &mut self,
        from_warning: Option<PresenceEventKind>,
        from_away: Option<PresenceEventKind>,
    ) -> Vec<PresenceEvent> {
        let previous = self.state;
        self.state = PresenceState::Present;
        if let (PresenceState::Warning, Some(kind)) = (previous, from_warning) {
            log::info!("Presence countdown cancelled");
            return vec![PresenceEvent::of(kind)];
        }
        if let (PresenceState::Away, Some(kind)) = (previous, from_away) {
            log::info!("User returned");
            return vec![PresenceEvent::of(kind)];
        }
        Vec::new()
    }
}

#[cfg(test)]
#[path = "presence_tests.rs"]
mod tests;
