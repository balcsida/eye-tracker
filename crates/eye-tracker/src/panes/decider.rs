//! Decide *when* keyboard focus should move to the split pane the user looks at.
//!
//! This runs only while the monitor-level decider is content (the gaze is on the
//! monitor the cursor is on), for the focused terminal window on that monitor.
//! Panes are much smaller than monitors and the gaze estimate is the same, so
//! [`PaneDecider`] is stricter than `SwitchDecider` (`engine::decision`) and measures everything
//! in units of the calibration's gaze error `σ` on each axis (`gaze::store::axis_error`).
//! Filters, in order:
//!
//! 1. **Fresh snapshot** - the pane layout must be at most [`PaneConfig::fresh_s`]
//!    old (`"stale"`), and there must be one (`"no_panes"`).
//! 2. **Inside the window** - gaze outside the terminal window does nothing.
//! 3. **Candidate pane** - the pane under the gaze (`"no_pane"` on a border or
//!    the tab bar).
//! 4. **Same pane** - the focused pane: nothing to do.
//! 5. **Size gate** - the candidate `P` and the focused pane `C` are separated
//!    along one axis: x for panes side by side, y for stacked ones, the dominant
//!    one for diagonal neighbours. `P` qualifies only if its extent on that axis
//!    is at least `max(precision × σ_axis, min_pane_px)` (`"small"`). A pane
//!    the gaze cannot reliably be placed in is never switched to.
//! 6. **Hysteresis** - the gaze must be at least `hysteresis × σ_axis` past the
//!    divider between `C` and `P`.
//! 7. **Dwell, by majority** - at least [`PaneConfig::majority`] of the samples
//!    of the last `dwell_s` must have passed 5 and 6 for `P`, and the history
//!    must span the whole dwell. A few stray samples do not restart it; a long gap
//!    between updates does.
//! 8. **Guards** - recent mouse use, recent typing (longer toward a *reading
//!    pane*, see `engine::reading`), the cooldown after the last
//!    pane switch, and a grace after the focused pane changed without us (the
//!    user switched panes by hand: their choice wins for a while).
//!
//! The decider is pure logic: time is injected and it has no side effects.

use std::collections::VecDeque;

use crate::config::{PaneSettings, SwitchingSettings};
use crate::engine::reading::{READING_MAX_GAP_S, ReadingTracker};
use crate::types::{Rect, WindowHandle};

use super::types::{Pane, PaneId, PaneSnapshot};

/// A pane switch counts as `pending` (the scheduler raises the frame rate)
/// only when it could fire within this many seconds (as for monitor switches).
pub const PENDING_HORIZON_S: f64 = 0.5;

/// Gaze error assumed when the calibration says nothing about it (pixels).
pub const FALLBACK_GAZE_ERROR_PX: f64 = 150.0;

/// A focus change seen within this time after we asked for it is ours.
pub const OWN_CHANGE_S: f64 = 5.0;

/// Panes overlapping by at most this many pixels on an axis are not overlapping.
const TOUCH_PX: i32 = 2;

const EPS: f64 = 1e-6;

/// Every value [`PaneDecision::reason`] can take.
pub const PANE_REASONS: [&str; 15] = [
    "switch",
    "same",
    "dwell",
    "no_gaze",
    "no_panes",
    "stale",
    "outside",
    "no_pane",
    "small",
    "hysteresis",
    "mouse",
    "typing",
    "cooldown",
    "manual",
    "disabled",
];

/// Tuning of [`PaneDecider`]. Durations in seconds, sizes in pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct PaneConfig {
    pub dwell_s: f64,
    pub typing_grace_s: f64,
    pub reading_grace_s: f64,
    pub cooldown_s: f64,
    pub mouse_grace_s: f64,
    /// Grace after the focused pane changed without us. Set from `panes.typing_grace_ms`
    /// (documented there): switching panes by hand is a keyboard shortcut or a click, so it
    /// gets the grace of typing, without a setting of its own.
    pub manual_grace_s: f64,
    /// Minimum pane extent in units of the gaze error on the separating axis.
    pub precision: f64,
    /// Distance past the divider, in units of the gaze error.
    pub hysteresis: f64,
    pub min_pane_px: f64,
    /// Oldest pane snapshot that is still trusted.
    pub fresh_s: f64,
    /// A longer pause between two updates restarts the dwell.
    pub max_gap_s: f64,
    /// Share of the dwell window's samples that must be on the candidate.
    pub majority: f64,
}

impl Default for PaneConfig {
    fn default() -> Self {
        Self {
            dwell_s: 0.6,
            typing_grace_s: 3.0,
            reading_grace_s: 8.0,
            cooldown_s: 1.0,
            mouse_grace_s: 1.5,
            manual_grace_s: 3.0,
            precision: 2.5,
            hysteresis: 0.5,
            min_pane_px: 240.0,
            fresh_s: 1.5,
            max_gap_s: 0.75,
            majority: 0.8,
        }
    }
}

impl PaneConfig {
    /// From the user-facing (millisecond based) settings; the mouse grace is shared.
    pub fn from_settings(panes: &PaneSettings, switching: &SwitchingSettings) -> Self {
        Self {
            dwell_s: panes.dwell_ms as f64 / 1000.0,
            typing_grace_s: panes.typing_grace_ms as f64 / 1000.0,
            reading_grace_s: panes.reading_grace_ms as f64 / 1000.0,
            cooldown_s: panes.cooldown_ms as f64 / 1000.0,
            mouse_grace_s: switching.mouse_grace_ms as f64 / 1000.0,
            manual_grace_s: panes.typing_grace_ms as f64 / 1000.0,
            precision: panes.precision,
            hysteresis: panes.hysteresis,
            min_pane_px: panes.min_pane_px as f64,
            ..Self::default()
        }
    }
}

/// Outcome of one [`PaneDecider::update`] call.
#[derive(Debug, Clone, PartialEq)]
pub struct PaneDecision {
    /// Pane to focus *now*; `None` means do nothing.
    pub target: Option<Pane>,
    /// Id of the pane under the gaze (may be the focused one); `None` if none.
    pub candidate: Option<PaneId>,
    /// Dwell progress toward `candidate` in `[0, 1]`.
    pub progress: f64,
    /// Why this decision was taken; one of [`PANE_REASONS`].
    pub reason: &'static str,
    /// A pane switch may fire soon; the scheduler should sample faster.
    pub pending: bool,
}

impl PaneDecision {
    pub fn fired(&self) -> bool {
        self.target.is_some()
    }
}

/// The axis that separates panes `a` and `b`: 0 (x, side by side) or 1 (y, stacked).
/// Diagonal neighbours get the axis of the larger centre offset.
pub fn separation_axis(a: &Rect, b: &Rect) -> usize {
    let overlap_x = a.right().min(b.right()) - a.x.max(b.x);
    let overlap_y = a.bottom().min(b.bottom()) - a.y.max(b.y);
    let apart_x = overlap_x <= TOUCH_PX;
    let apart_y = overlap_y <= TOUCH_PX;
    if apart_x && !apart_y {
        return 0;
    }
    if apart_y && !apart_x {
        return 1;
    }
    let (ax, ay) = a.center();
    let (bx, by) = b.center();
    if (bx - ax).abs() >= (by - ay).abs() {
        0
    } else {
        1
    }
}

fn extent(rect: &Rect, axis: usize) -> i32 {
    if axis == 0 { rect.w } else { rect.h }
}

fn on_axis(pair: (f64, f64), axis: usize) -> f64 {
    if axis == 0 { pair.0 } else { pair.1 }
}

/// How far `gaze` is past the divider between `current` and `candidate` on `axis`, toward
/// `candidate` (negative on the `current` side).
pub fn divider_margin(gaze: (f64, f64), current: &Rect, candidate: &Rect, axis: usize) -> f64 {
    let g = on_axis(gaze, axis);
    let (c0, c1) = if axis == 0 {
        (current.x, current.right())
    } else {
        (current.y, current.bottom())
    };
    let (p0, p1) = if axis == 0 {
        (candidate.x, candidate.right())
    } else {
        (candidate.y, candidate.bottom())
    };
    if p0 + p1 >= c0 + c1 {
        // candidate after the current pane
        let divider = if c1 <= p0 {
            f64::from(c1 + p0) / 2.0
        } else {
            f64::from(p0)
        };
        return g - divider;
    }
    let divider = if p1 <= c0 {
        f64::from(p1 + c0) / 2.0
    } else {
        f64::from(p1)
    };
    divider - g
}

/// The size gate: is `candidate` large enough to be told apart from `current`?
pub fn pane_qualifies(
    candidate: &Pane,
    current: &Pane,
    sigma: (f64, f64),
    config: &PaneConfig,
) -> bool {
    let axis = separation_axis(&current.rect, &candidate.rect);
    let needed = (config.precision * on_axis(sigma, axis)).max(config.min_pane_px);
    f64::from(extent(&candidate.rect, axis)) + EPS >= needed
}

/// How many panes besides the focused one pass the size gate.
pub fn eligible_panes(
    snapshot: Option<&PaneSnapshot>,
    sigma: (f64, f64),
    config: &PaneConfig,
) -> usize {
    let Some(snapshot) = snapshot else { return 0 };
    let Some(current) = snapshot.focused() else {
        return 0;
    };
    snapshot
        .panes
        .iter()
        .filter(|p| p.id != current.id && pane_qualifies(p, current, sigma, config))
        .count()
}

/// Stateful filter from gaze points to pane focus changes (see the module docs).
///
/// The caller reports the focused window's rectangle and its latest pane snapshot with every
/// update; when the decider fires, the caller asks the pane's provider to focus it and calls
/// [`notify_switched`](Self::notify_switched).
#[derive(Debug)]
pub struct PaneDecider {
    config: PaneConfig,
    samples: VecDeque<(f64, Option<PaneId>)>,
    last_update: Option<f64>,
    last_switch: f64,
    last_target: Option<PaneId>,
    manual_at: f64,
    window: Option<WindowHandle>,
    focused: Option<PaneId>,
    reading: ReadingTracker<PaneId>,
}

impl Default for PaneDecider {
    fn default() -> Self {
        Self::new(PaneConfig::default())
    }
}

impl PaneDecider {
    pub fn new(config: PaneConfig) -> Self {
        Self {
            config,
            samples: VecDeque::new(),
            last_update: None,
            last_switch: f64::NEG_INFINITY,
            last_target: None,
            manual_at: f64::NEG_INFINITY,
            window: None,
            focused: None,
            reading: ReadingTracker::new(),
        }
    }

    // ---------------------------------------------------------------- config
    pub fn config(&self) -> &PaneConfig {
        &self.config
    }

    pub fn set_config(&mut self, config: PaneConfig) {
        self.config = config;
    }

    pub fn reading_pane(&self) -> Option<&PaneId> {
        self.reading.reading()
    }

    pub fn last_switch_time(&self) -> f64 {
        self.last_switch
    }

    /// Forget everything: dwell, reading pane, the window seen and the cooldown.
    pub fn reset(&mut self) {
        self.samples.clear();
        self.reading.clear();
        self.last_update = None;
        self.last_switch = f64::NEG_INFINITY;
        self.last_target = None;
        self.manual_at = f64::NEG_INFINITY;
        self.window = None;
        self.focused = None;
    }

    /// The caller focused `pane_id` (arms the cooldown).
    pub fn notify_switched(&mut self, now: f64, pane_id: &PaneId) {
        self.last_switch = now;
        self.last_target = Some(pane_id.clone());
        self.samples.clear();
        self.reading.clear();
    }

    // ---------------------------------------------------------------- update
    /// Feed one gaze sample and decide whether to focus another pane now.
    ///
    /// `window` is the focused window's rectangle and `snapshot` its latest panes (`None` when
    /// unknown); `sigma` is the gaze error `(x, y)` in pixels on this monitor (`None`:
    /// [`FALLBACK_GAZE_ERROR_PX`]). Activity times are `-inf` when there has never been any.
    #[allow(clippy::too_many_arguments)] // the Python signature, one value per input
    pub fn update(
        &mut self,
        now: f64,
        gaze: Option<(f64, f64)>,
        window: Option<Rect>,
        snapshot: Option<&PaneSnapshot>,
        sigma: Option<(f64, f64)>,
        last_mouse_activity: f64,
        last_key_activity: f64,
        enabled: bool,
    ) -> PaneDecision {
        let cfg = self.config.clone();
        if let Some(last) = self.last_update {
            let gap = now - last;
            if gap < 0.0 || (cfg.max_gap_s > 0.0 && gap > cfg.max_gap_s + EPS) {
                self.samples.clear();
            }
            if gap < 0.0 {
                self.reading.clear();
            } else if gap > cfg.max_gap_s.max(READING_MAX_GAP_S) + EPS {
                self.reading.end_look();
            }
        }
        self.last_update = Some(now);

        if !enabled {
            self.samples.clear();
            self.reading.clear();
            return PaneDecision {
                target: None,
                candidate: None,
                progress: 0.0,
                reason: "disabled",
                pending: false,
            };
        }
        let Some((gx, gy)) = gaze.filter(|g| g.0.is_finite() && g.1.is_finite()) else {
            return self.hold(now, last_key_activity, None, "no_gaze");
        };
        let Some(snapshot) = snapshot.filter(|s| !s.panes.is_empty()) else {
            return self.hold(now, last_key_activity, None, "no_panes");
        };
        if now - snapshot.taken_at > cfg.fresh_s + EPS {
            return self.hold(now, last_key_activity, None, "stale");
        }
        let Some(current) = snapshot.focused() else {
            return self.hold(now, last_key_activity, None, "no_panes");
        };
        self.observe_focus(now, &snapshot.window_handle, &current.id);
        if let Some(window) = window
            && !window.contains(gx, gy)
        {
            return self.hold(now, last_key_activity, None, "outside");
        }
        let Some(candidate) = snapshot.panes.iter().find(|p| p.rect.contains(gx, gy)) else {
            return self.hold(now, last_key_activity, None, "no_pane");
        };
        if candidate.id == current.id {
            return self.hold(now, last_key_activity, Some(candidate.id.clone()), "same");
        }

        let (sx, sy) = sigma.unwrap_or((FALLBACK_GAZE_ERROR_PX, FALLBACK_GAZE_ERROR_PX));
        let axis = separation_axis(&current.rect, &candidate.rect);
        let s_axis = on_axis((sx, sy), axis);
        if !pane_qualifies(candidate, current, (sx, sy), &cfg) {
            return self.hold(now, last_key_activity, Some(candidate.id.clone()), "small");
        }
        let margin = divider_margin((gx, gy), &current.rect, &candidate.rect, axis);
        if margin + EPS < cfg.hysteresis * s_axis {
            return self.hold(
                now,
                last_key_activity,
                Some(candidate.id.clone()),
                "hysteresis",
            );
        }

        // The gaze is clearly on the candidate.
        self.record(now, Some(candidate.id.clone()));
        self.reading.observe(candidate.id.clone(), now);
        self.update_reading(now, last_key_activity, Some(&current.id));
        let (share, span) = self.dwell(now, &candidate.id);
        let progress = if cfg.dwell_s <= 0.0 {
            1.0
        } else {
            (span / cfg.dwell_s).min(1.0) * (share / cfg.majority.max(EPS)).min(1.0)
        };
        let (guard_reason, guard_left) = self.guard(
            now,
            last_mouse_activity,
            last_key_activity,
            Some(&candidate.id),
        );
        if span + EPS < cfg.dwell_s || share + EPS < cfg.majority {
            let dwell_left = (cfg.dwell_s * (1.0 - progress)).max(0.0);
            let pending = guard_left <= dwell_left + PENDING_HORIZON_S;
            return PaneDecision {
                target: None,
                candidate: Some(candidate.id.clone()),
                progress,
                reason: "dwell",
                pending,
            };
        }
        if let Some(reason) = guard_reason {
            let pending = guard_left <= PENDING_HORIZON_S;
            return PaneDecision {
                target: None,
                candidate: Some(candidate.id.clone()),
                progress: 1.0,
                reason,
                pending,
            };
        }

        // Fire; the cooldown is armed here so that a focus request that fails is
        // not repeated on every frame.
        self.notify_switched(now, &candidate.id);
        log::debug!("Focus pane {} of {}", candidate.id, candidate.provider);
        PaneDecision {
            target: Some(candidate.clone()),
            candidate: Some(candidate.id.clone()),
            progress: 1.0,
            reason: "switch",
            pending: false,
        }
    }

    // ------------------------------------------------------------- internals
    /// No clear look at another pane on this update.
    fn hold(
        &mut self,
        now: f64,
        last_key: f64,
        candidate: Option<PaneId>,
        reason: &'static str,
    ) -> PaneDecision {
        self.record(now, None);
        self.reading.end_look();
        let focused = self.focused.clone();
        self.update_reading(now, last_key, focused.as_ref());
        PaneDecision {
            target: None,
            candidate,
            progress: 0.0,
            reason,
            pending: false,
        }
    }

    fn record(&mut self, now: f64, pane_id: Option<PaneId>) {
        self.samples.push_back((now, pane_id));
        // Keep one sample at or before the start of the dwell window, so the
        // history can be seen to span it.
        let start = now - self.config.dwell_s;
        while self.samples.len() > 1 && self.samples[1].0 <= start + EPS {
            self.samples.pop_front();
        }
    }

    /// (share of samples on `pane_id`, time spanned by the history).
    fn dwell(&self, now: f64, pane_id: &PaneId) -> (f64, f64) {
        let hits = self
            .samples
            .iter()
            .filter(|(_, pid)| pid.as_ref() == Some(pane_id))
            .count();
        (
            hits as f64 / self.samples.len() as f64,
            now - self.samples[0].0,
        )
    }

    /// Notice a new window, and focus changes we did not cause.
    fn observe_focus(&mut self, now: f64, window: &WindowHandle, focused: &PaneId) {
        if self.window.as_ref() != Some(window) {
            self.window = Some(window.clone());
            self.focused = Some(focused.clone());
            self.samples.clear();
            self.reading.clear();
            return;
        }
        if self.focused.as_ref() == Some(focused) {
            return;
        }
        let ours =
            self.last_target.as_ref() == Some(focused) && now - self.last_switch <= OWN_CHANGE_S;
        if !ours {
            log::debug!("Pane focus changed by the user (to {focused})");
            self.manual_at = now;
        }
        self.focused = Some(focused.clone());
        self.samples.clear();
        self.reading.clear(); // the keyboard focus moved
    }

    fn update_reading(&mut self, now: f64, last_key: f64, context: Option<&PaneId>) {
        let cfg = &self.config;
        self.reading.update(
            now,
            last_key,
            cfg.typing_grace_s,
            cfg.reading_grace_s,
            &context,
            context.is_some(),
        );
    }

    /// First blocking guard and the time until *all* guards expire.
    fn guard(
        &self,
        now: f64,
        last_mouse: f64,
        last_key: f64,
        candidate: Option<&PaneId>,
    ) -> (Option<&'static str>, f64) {
        let cfg = &self.config;
        let mut typing_grace = cfg.typing_grace_s;
        if candidate == self.reading.reading() {
            typing_grace = typing_grace.max(cfg.reading_grace_s);
        }
        let checks = [
            ("mouse", cfg.mouse_grace_s - (now - last_mouse)),
            ("typing", typing_grace - (now - last_key)),
            ("cooldown", cfg.cooldown_s - (now - self.last_switch)),
            ("manual", cfg.manual_grace_s - (now - self.manual_at)),
        ];
        let mut reason = None;
        let mut longest: f64 = 0.0;
        for (name, left) in checks {
            if left > EPS {
                if reason.is_none() {
                    reason = Some(name);
                }
                longest = longest.max(left);
            }
        }
        (reason, longest)
    }
}

#[cfg(test)]
#[path = "decider_tests.rs"]
mod tests;
