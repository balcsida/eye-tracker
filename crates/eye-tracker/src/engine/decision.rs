//! Decide *when* the cursor should jump to the monitor the user is looking at.
//!
//! The gaze estimate is noisy, the user glances around constantly and a wrong
//! switch is far more annoying than a late one. [`SwitchDecider`] therefore
//! only fires after the gaze has passed a chain of filters, evaluated in order on
//! every update:
//!
//! 1. **Gaze available** - no gaze (no face, blink, not calibrated) resets dwell.
//!    So does the caller's verdict that the user looks away from every monitor
//!    (`looking_away`), which is reported as *off screen*.
//! 2. **Candidate monitor** - the monitor containing the gaze point. A gaze point
//!    slightly outside every monitor (estimation error near the outer edges) is
//!    attributed to the nearest monitor; one far outside (a glance at the phone or
//!    the keyboard) is ignored as *off screen*.
//! 3. **Same monitor** - nothing to do.
//! 4. **Hysteresis** - the gaze must clearly favour the other monitor: it must be
//!    nearer to it than to the current monitor by a margin. For a gaze point on
//!    the other monitor that margin is simply how far it is past the bezel; for a
//!    point outside every monitor (below the seam of two side-by-side monitors,
//!    where the keyboard usually is) it is the difference of the two distances,
//!    so jitter around the seam is held there just as on screen.
//! 5. **Dwell** - the candidate must be stable for `dwell_s` across consecutive
//!    updates. A long gap between updates restarts the dwell because nothing is
//!    known about where the user looked in between.
//! 6. **Guards** - recent mouse use, recent typing and a cooldown after the last
//!    switch block the switch. The dwell keeps accumulating while guarded, so the
//!    switch happens the moment the guard expires if the user is still looking.
//!
//! # Reading while typing
//!
//! Copying from a document on one monitor into an editor on the other means
//! looking at the source while the keystrokes go to the current monitor. With the
//! plain typing guard, the first reading pause longer than `typing_grace_s`
//! would move the cursor - and the keyboard focus - to the source, and the next
//! keystrokes would land there. The decider therefore recognises a *reading
//! monitor*: another monitor the user kept typing while (or right after) looking
//! at. Evidence is a look at that monitor (two or more consecutive updates with
//! the gaze clearly on it) followed by typing that continues at least
//! [`READING_EVIDENCE_S`] after the look began and resumes no later than
//! [`READING_RETURN_S`] after it ended. A single noisy gaze sample never
//! counts, and neither does the usual "last keystroke while the eyes already move
//! on" of a user who is about to work on the other monitor.
//!
//! For a reading monitor the typing guard lasts `reading_grace_s` after the last
//! keystroke instead of `typing_grace_s`: reading pauses keep the focus where the
//! user types, glances back at the editor do not reset anything, and a user who
//! really wants to work on the reading monitor gets there after a longer look
//! (or at once by moving the mouse). The reading monitor is forgotten when the
//! cursor changes monitor, when typing has paused for `reading_grace_s` or when
//! the gaze has not been on it for [`READING_FORGET_S`]. Switches to any other
//! monitor are unaffected. The rule itself is [`ReadingTracker`](super::reading::ReadingTracker),
//! which the split-pane decider shares.
//!
//! The decider is pure logic: time is injected and it has no side effects, which
//! makes it cheap to test exhaustively.

use std::collections::HashMap;

use crate::config::SwitchingSettings;
use crate::types::{Monitor, Rect};

// The reading rule's constants (READING_*) live in `reading` and are re-exported
// here: they describe this decider's behaviour too.
use super::reading::ReadingTracker;
pub use super::reading::{
    READING_EVIDENCE_S, READING_FORGET_S, READING_MAX_GAP_S, READING_RETURN_S,
};

/// A switch counts as `pending` (so the scheduler raises the frame rate) only
/// when it could fire within this many seconds. A switch that is blocked for a
/// long time - typically the user reads one monitor while typing on the other -
/// must not keep the camera at the fast rate.
pub const PENDING_HORIZON_S: f64 = 0.5;

/// Default of [`SwitchConfig::reading_grace_s`].
pub const DEFAULT_READING_GRACE_S: f64 = 6.0;

/// Absorbs float rounding in time differences (`10.3 - 10.0 < 0.3` is possible).
const EPS: f64 = 1e-6;

/// Every value [`Decision::reason`] can take.
pub const REASONS: [&str; 10] = [
    "switch",
    "same",
    "dwell",
    "no_gaze",
    "off_screen",
    "mouse",
    "typing",
    "cooldown",
    "hysteresis",
    "disabled",
];

/// Tuning of [`SwitchDecider`]. Durations in seconds, fractions of monitor size.
#[derive(Debug, Clone, PartialEq)]
pub struct SwitchConfig {
    /// How long the gaze must stay on another monitor before switching (0 = immediately).
    pub dwell_s: f64,
    /// How much nearer the gaze must be to the other monitor than to the current
    /// one, as a fraction of the other monitor's smaller side. For a gaze point
    /// on the other monitor this is how far past the bezel it must be.
    pub hysteresis: f64,
    /// Gaze further than this fraction of the nearest monitor's diagonal outside
    /// every monitor is treated as looking away from the screens.
    pub off_screen_margin: f64,
    /// Minimum time between two switches.
    pub cooldown_s: f64,
    /// No switching this long after the user moved the mouse.
    pub mouse_grace_s: f64,
    /// No switching this long after the user typed.
    pub typing_grace_s: f64,
    /// A longer pause between two updates restarts the dwell (0 disables the check).
    pub max_gap_s: f64,
    /// Typing grace toward a *reading monitor* (see the module documentation).
    /// A value not above `typing_grace_s` turns the reading rule off.
    pub reading_grace_s: f64,
}

impl Default for SwitchConfig {
    fn default() -> Self {
        Self {
            dwell_s: 0.3,
            hysteresis: 0.06,
            off_screen_margin: 0.35,
            cooldown_s: 0.6,
            mouse_grace_s: 1.5,
            typing_grace_s: 2.0,
            max_gap_s: 0.75,
            reading_grace_s: DEFAULT_READING_GRACE_S,
        }
    }
}

impl SwitchConfig {
    /// Build the configuration from the user-facing (millisecond based) settings.
    pub fn from_settings(s: &SwitchingSettings) -> Self {
        Self {
            dwell_s: s.dwell_ms as f64 / 1000.0,
            hysteresis: s.hysteresis,
            off_screen_margin: s.off_screen_margin,
            cooldown_s: s.cooldown_ms as f64 / 1000.0,
            mouse_grace_s: s.mouse_grace_ms as f64 / 1000.0,
            typing_grace_s: s.typing_grace_ms as f64 / 1000.0,
            reading_grace_s: s.reading_grace_ms as f64 / 1000.0,
            ..Self::default()
        }
    }
}

/// Outcome of one [`SwitchDecider::update`] call.
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    /// Monitor index to switch to *now*; `None` means do nothing.
    pub target: Option<i32>,
    /// Monitor the gaze currently favours (may equal the current monitor);
    /// `None` when there is no gaze or it is off screen.
    pub candidate: Option<i32>,
    /// Dwell progress toward `candidate` in `[0, 1]` (1 once the dwell is complete).
    pub progress: f64,
    /// Why this decision was taken; one of [`REASONS`].
    pub reason: &'static str,
    /// A switch may fire soon; the scheduler should sample faster.
    pub pending: bool,
}

impl Decision {
    /// `true` when the caller should switch to [`target`](Self::target) now.
    pub fn fired(&self) -> bool {
        self.target.is_some()
    }
}

/// Stateful filter from gaze points to monitor switches.
///
/// Monitors are identified by [`Monitor::index`]. The caller reports the
/// monitor the cursor is on (`current_monitor`); when the decider fires, the
/// caller performs the switch and calls [`notify_switched`](Self::notify_switched).
#[derive(Debug)]
pub struct SwitchDecider {
    config: SwitchConfig,
    monitors: Vec<Monitor>,
    by_index: HashMap<i32, usize>,
    dwell_candidate: Option<i32>,
    dwell_start: f64,
    last_update: Option<f64>,
    last_switch: f64,
    last_target: Option<i32>,
    // Reading-while-typing state; it belongs to the monitor the cursor was
    // on (`context`) and is dropped whenever that changes.
    context: Option<i32>,
    reading: ReadingTracker<i32>,
}

impl SwitchDecider {
    pub fn new(monitors: &[Monitor], config: SwitchConfig) -> Self {
        let mut decider = Self {
            config,
            monitors: Vec::new(),
            by_index: HashMap::new(),
            dwell_candidate: None,
            dwell_start: 0.0,
            last_update: None,
            last_switch: f64::NEG_INFINITY,
            last_target: None,
            context: None,
            reading: ReadingTracker::new(),
        };
        decider.set_monitors(monitors);
        decider
    }

    // ---------------------------------------------------------------- config
    pub fn config(&self) -> &SwitchConfig {
        &self.config
    }

    pub fn monitors(&self) -> &[Monitor] {
        &self.monitors
    }

    /// Time of the most recent switch (`-inf` if none yet).
    pub fn last_switch_time(&self) -> f64 {
        self.last_switch
    }

    /// Monitor index of the most recent switch.
    pub fn last_switch_target(&self) -> Option<i32> {
        self.last_target
    }

    /// The monitor currently treated as the user's reading monitor, if any.
    pub fn reading_monitor(&self) -> Option<i32> {
        self.reading.reading().copied()
    }

    /// Replace the monitor layout. A dwell in progress is discarded.
    pub fn set_monitors(&mut self, monitors: &[Monitor]) {
        self.monitors = monitors.to_vec();
        // Like the Python dict, the last monitor with a given index wins.
        self.by_index = self
            .monitors
            .iter()
            .enumerate()
            .map(|(i, m)| (m.index, i))
            .collect();
        if self.by_index.len() != self.monitors.len() {
            let indices: Vec<i32> = monitors.iter().map(|m| m.index).collect();
            log::warn!("Duplicate monitor indices in layout: {indices:?}");
        }
        self.clear_dwell();
        self.clear_reading();
    }

    /// Change the tuning; takes effect on the next update.
    pub fn set_config(&mut self, config: SwitchConfig) {
        self.config = config;
    }

    /// Forget all transient state, including the cooldown.
    pub fn reset(&mut self) {
        self.clear_dwell();
        self.clear_reading();
        self.context = None;
        self.last_update = None;
        self.last_switch = f64::NEG_INFINITY;
        self.last_target = None;
    }

    // ---------------------------------------------------------------- update
    /// Feed one gaze sample and decide whether to switch now.
    ///
    /// `current_monitor` is the monitor the cursor is on (`None` when it is
    /// on no monitor; then any candidate is a switch target). Activity times are
    /// `-inf` when there has never been any activity. `looking_away` is the
    /// caller's verdict that the user looks away from every monitor (e.g. the
    /// gaze model extrapolates far beyond its calibration); it overrides
    /// `gaze` and is reported as `"off_screen"`.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        now: f64,
        gaze: Option<(f64, f64)>,
        current_monitor: Option<i32>,
        last_mouse_activity: f64,
        last_key_activity: f64,
        enabled: bool,
        looking_away: bool,
    ) -> Decision {
        let cfg = self.config.clone();
        if let Some(last_update) = self.last_update {
            let gap = now - last_update;
            // A clock going backwards is treated like a gap: the dwell history is
            // no longer trustworthy.
            if gap < 0.0 || (cfg.max_gap_s > 0.0 && gap > cfg.max_gap_s + EPS) {
                self.clear_dwell();
            }
            if gap < 0.0 {
                self.clear_reading();
            } else if gap > cfg.max_gap_s.max(READING_MAX_GAP_S) + EPS {
                self.reading.end_look();
            }
        }
        self.last_update = Some(now);

        if !enabled || self.monitors.is_empty() {
            self.clear_dwell();
            self.clear_reading();
            return Decision {
                target: None,
                candidate: None,
                progress: 0.0,
                reason: "disabled",
                pending: false,
            };
        }

        if current_monitor != self.context {
            // The keyboard focus most likely moved with the cursor: whatever the
            // user was reading beside the old monitor says nothing about the new one.
            self.clear_reading();
            self.context = current_monitor;
        }

        let current = current_monitor.and_then(|i| self.monitor(i));
        let (candidate, held) = self.classify(gaze, current, looking_away);
        match (candidate, held) {
            (Some(c), None) => self.reading.observe(c, now),
            _ => self.reading.end_look(),
        }
        self.reading.update(
            now,
            last_key_activity,
            cfg.typing_grace_s,
            cfg.reading_grace_s,
            &self.context,
            self.context.is_some_and(|c| self.by_index.contains_key(&c)),
        );

        let candidate = match (candidate, held) {
            (Some(c), None) => c,
            (candidate, held) => {
                // (classify always gives a reason when there is no candidate.)
                self.clear_dwell();
                return Decision {
                    target: None,
                    candidate,
                    progress: 0.0,
                    reason: held.unwrap_or("off_screen"),
                    pending: false,
                };
            }
        };

        if self.dwell_candidate != Some(candidate) {
            self.dwell_candidate = Some(candidate);
            self.dwell_start = now;
        }
        let elapsed = now - self.dwell_start;
        let mut dwell_left = (cfg.dwell_s - elapsed).max(0.0);
        if dwell_left <= EPS {
            dwell_left = 0.0;
        }
        let progress = if cfg.dwell_s <= 0.0 {
            1.0
        } else {
            (elapsed / cfg.dwell_s).clamp(0.0, 1.0)
        };

        let (guard_reason, guard_left) =
            self.guard(now, last_mouse_activity, last_key_activity, candidate);

        if dwell_left > 0.0 {
            // Keep sampling fast while dwelling, unless a guard will block the
            // switch well beyond the end of the dwell anyway.
            let pending = guard_left <= dwell_left + PENDING_HORIZON_S;
            return Decision {
                target: None,
                candidate: Some(candidate),
                progress,
                reason: "dwell",
                pending,
            };
        }

        if let Some(reason) = guard_reason {
            return Decision {
                target: None,
                candidate: Some(candidate),
                progress: 1.0,
                reason,
                pending: guard_left <= PENDING_HORIZON_S,
            };
        }

        // Fire. The dwell restarts and the cooldown is armed right here (not only
        // in notify_switched) so that a switch the caller could not perform, e.g.
        // cursor warping refused on Wayland, is not retried on every frame.
        self.clear_dwell();
        self.clear_reading();
        self.last_switch = now;
        self.last_target = Some(candidate);
        if let Some((gx, gy)) = gaze {
            log::debug!("Switch to monitor {candidate} (gaze {gx:.0}, {gy:.0})");
        }
        Decision {
            target: Some(candidate),
            candidate: Some(candidate),
            progress: 1.0,
            reason: "switch",
            pending: false,
        }
    }

    /// Record that the caller switched to `monitor_index` (arms the cooldown).
    pub fn notify_switched(&mut self, now: f64, monitor_index: i32) {
        self.last_switch = now;
        self.last_target = Some(monitor_index);
        self.clear_dwell();
        self.clear_reading();
    }

    // ------------------------------------------------------------- internals
    fn clear_dwell(&mut self) {
        self.dwell_candidate = None;
    }

    fn clear_reading(&mut self) {
        self.reading.clear();
    }

    fn monitor(&self, index: i32) -> Option<&Monitor> {
        self.by_index.get(&index).map(|&i| &self.monitors[i])
    }

    /// The candidate monitor index and, unless the gaze clearly favours it, why not.
    ///
    /// Returns `(Some(candidate), None)` when the gaze is clearly on another monitor
    /// and `(candidate_or_None, Some(reason))` for `"off_screen"`, `"no_gaze"`,
    /// `"same"` and `"hysteresis"`.
    fn classify(
        &self,
        gaze: Option<(f64, f64)>,
        current: Option<&Monitor>,
        looking_away: bool,
    ) -> (Option<i32>, Option<&'static str>) {
        if looking_away {
            return (None, Some("off_screen"));
        }
        let Some((gx, gy)) = gaze.filter(|(x, y)| x.is_finite() && y.is_finite()) else {
            return (None, Some("no_gaze"));
        };
        let Some((candidate, rect)) = self.candidate(gx, gy, current) else {
            return (None, Some("off_screen"));
        };
        let Some(current) = current else {
            return (Some(candidate), None);
        };
        if candidate == current.index {
            return (Some(candidate), Some("same"));
        }
        // How much nearer the gaze is to the candidate than to the current
        // monitor. On the candidate this is the distance past the bezel; off
        // screen it compares both distances, so a point just across the
        // bisector below or above a seam is held like one just past a bezel.
        let margin = current.rect.distance_outside(gx, gy) - rect.distance_outside(gx, gy);
        let needed = self.config.hysteresis * f64::from(rect.w.min(rect.h));
        if margin + EPS < needed {
            return (Some(candidate), Some("hysteresis"));
        }
        (Some(candidate), None)
    }

    fn candidate(&self, gx: f64, gy: f64, current: Option<&Monitor>) -> Option<(i32, Rect)> {
        // Prefer the current monitor on ties (overlapping/mirrored displays, or a
        // point equidistant from two monitors) so ambiguity never causes a switch.
        if let Some(current) = current.filter(|c| c.rect.contains(gx, gy)) {
            return Some((current.index, current.rect));
        }
        if let Some(m) = self.monitors.iter().find(|m| m.rect.contains(gx, gy)) {
            return Some((m.index, m.rect));
        }
        let mut best: Option<&Monitor> = None;
        let mut best_d = f64::INFINITY;
        for m in &self.monitors {
            let d = m.rect.distance_outside(gx, gy);
            if d < best_d || (d == best_d && current.is_some_and(|c| m.index == c.index)) {
                best = Some(m);
                best_d = d;
            }
        }
        let best = best?;
        if best_d > self.config.off_screen_margin * best.rect.diagonal() {
            return None;
        }
        Some((best.index, best.rect))
    }

    /// First blocking guard (spec order) and the time until *all* guards expire.
    fn guard(
        &self,
        now: f64,
        last_mouse: f64,
        last_key: f64,
        candidate: i32,
    ) -> (Option<&'static str>, f64) {
        let cfg = &self.config;
        let mut typing_grace = cfg.typing_grace_s;
        if self.reading.reading() == Some(&candidate) {
            typing_grace = typing_grace.max(cfg.reading_grace_s);
        }
        let checks = [
            ("mouse", cfg.mouse_grace_s - (now - last_mouse)),
            ("typing", typing_grace - (now - last_key)),
            ("cooldown", cfg.cooldown_s - (now - self.last_switch)),
        ];
        let mut reason = None;
        let mut longest = 0.0_f64;
        for (name, left) in checks {
            if left > EPS {
                reason = reason.or(Some(name));
                longest = longest.max(left);
            }
        }
        (reason, longest)
    }
}

#[cfg(test)]
#[path = "decision_tests.rs"]
mod tests;
