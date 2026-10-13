//! Mouse vs keyboard activity, derived without any input hooks.
//!
//! Global keyboard hooks would need special permissions on every OS and look like
//! a key logger, so they are deliberately avoided. Instead:
//!
//! * **Mouse** activity is detected by polling the cursor position. Moves the app
//!   makes itself (after a monitor switch) are announced through
//!   [`InputTracker::note_programmatic_move`] and are not user activity. Slow,
//!   precise motion (a few pixels per poll, e.g. dragging a slider) counts as soon
//!   as the pointer has drifted [`DRIFT_DISTANCE_PX`] within
//!   [`DRIFT_WINDOW_S`]; one-pixel jitter back and forth never does.
//! * **Keyboard** activity comes from the OS "seconds since last key press" where
//!   one exists (macOS). Elsewhere it is inferred: the OS idle timer was reset but
//!   the cursor did not move, so the input was most likely a key press. Only
//!   timestamps are ever read, never key contents.
//!
//! Consequence of the inference: clicks and scrolling without moving the pointer
//! also count as "keyboard". Both block switching for a moment, which is the
//! desired effect anyway.

/// Platform idle-time provider: seconds since the last input, `None` when
/// unsupported or failing.
pub type IdleProvider = Box<dyn FnMut() -> Option<f64>>;

/// A cursor displacement larger than this between two polls is a mouse move.
pub const MOVE_THRESHOLD_PX: f64 = 2.0;
/// A move landing within this distance of a programmatic warp target is ours.
pub const WARP_TOLERANCE_PX: f64 = 3.0;
/// How long after a warp the arrival of the cursor is attributed to the warp.
pub const WARP_WINDOW_S: f64 = 0.5;
/// The OS idle timer must jump forward by more than this to count as new input
/// (absorbs the jitter between sampling `now` and sampling the idle time).
pub const IDLE_RESET_EPS_S: f64 = 0.05;
/// Pointer steps too small to be a move on their own count once the pointer has
/// drifted further than this from where the steps began...
pub const DRIFT_DISTANCE_PX: f64 = 4.0;
/// ...within this time. Measuring the net displacement (not the path length)
/// keeps jitter around one spot from ever adding up.
pub const DRIFT_WINDOW_S: f64 = 1.0;

type Point = (i32, i32);

fn dist(a: Point, b: Point) -> f64 {
    f64::from(a.0 - b.0).hypot(f64::from(a.1 - b.1))
}

/// Keeps timestamps of the user's last mouse, keyboard and any input.
///
/// `seconds_since_input` / `seconds_since_key_input` are the platform's
/// idle-time providers (they may return `None` when unsupported, and they
/// must not return a stale cached value unchanged - see [`InputTracker::poll`]).
/// All timestamps are in the caller's clock (a monotonic clock) and are
/// `-inf` until the first activity is known.
pub struct InputTracker {
    idle_fn: IdleProvider,
    key_idle_fn: IdleProvider,
    cursor: Point,
    has_cursor: bool,
    last_mouse: f64,
    last_key: f64,
    last_any: f64,
    manual_move: bool,
    warp: Option<Point>,
    warp_time: f64,
    warp_polled: bool,
    /// Where a run of small pointer steps began, and when (see `track_drift`).
    drift_anchor: Option<Point>,
    drift_since: f64,
    /// Latest OS input time already attributed to some activity.
    os_seen: Option<f64>,
    prev_idle: Option<f64>,
}

impl InputTracker {
    pub fn new(seconds_since_input: IdleProvider, seconds_since_key_input: IdleProvider) -> Self {
        Self {
            idle_fn: seconds_since_input,
            key_idle_fn: seconds_since_key_input,
            cursor: (0, 0),
            has_cursor: false,
            last_mouse: f64::NEG_INFINITY,
            last_key: f64::NEG_INFINITY,
            last_any: f64::NEG_INFINITY,
            manual_move: false,
            warp: None,
            warp_time: f64::NEG_INFINITY,
            warp_polled: false,
            drift_anchor: None,
            drift_since: f64::NEG_INFINITY,
            os_seen: None,
            prev_idle: None,
        }
    }

    /// Time of the last user-initiated cursor move.
    pub fn last_mouse_activity(&self) -> f64 {
        self.last_mouse
    }

    /// Time of the last (detected or inferred) keyboard input.
    pub fn last_key_activity(&self) -> f64 {
        self.last_key
    }

    /// Time of the last input of any kind.
    pub fn last_any_activity(&self) -> f64 {
        self.last_any
    }

    /// Cursor position seen by the latest poll (`(0, 0)` before the first poll).
    pub fn cursor(&self) -> (i32, i32) {
        self.cursor
    }

    /// `true` if the most recent poll saw a user-initiated cursor move.
    pub fn manual_move(&self) -> bool {
        self.manual_move
    }

    /// Seconds since the last input of any kind, `None` if nothing is known yet.
    pub fn seconds_since_any(&self, now: f64) -> Option<f64> {
        if self.last_any.is_infinite() {
            return None;
        }
        Some((now - self.last_any).max(0.0))
    }

    /// Announce that the app itself is warping the cursor to `pos`.
    pub fn note_programmatic_move(&mut self, pos: (i32, i32), now: f64) {
        self.warp = Some(pos);
        self.warp_time = now;
        self.warp_polled = false;
    }

    /// Sample the cursor position and the OS idle timers.
    pub fn poll(&mut self, now: f64, cursor: (i32, i32)) {
        let (x, y) = cursor;
        let had_cursor = self.has_cursor;
        let prev = self.cursor;
        self.cursor = (x, y);
        self.has_cursor = true;

        let changed = had_cursor && (x, y) != prev;
        self.manual_move = false;
        if had_cursor && dist((x, y), prev) > MOVE_THRESHOLD_PX {
            self.drift_anchor = None;
            if !self.explained_by_warp(x, y, now) {
                self.register_manual_move(now);
            }
            self.warp = None;
        } else if changed {
            self.track_drift(prev, x, y, now);
        }

        // A warp is given at least one poll to show up, even if polls are slow.
        if self.warp.is_some() {
            if self.warp_polled && now - self.warp_time > WARP_WINDOW_S {
                self.warp = None;
            } else {
                self.warp_polled = true;
            }
        }

        let key_idle = read(&mut self.key_idle_fn);
        if let Some(key_idle) = key_idle {
            self.last_key = self.last_key.max(now - key_idle);
            self.last_any = self.last_any.max(self.last_key);
        }

        let Some(idle) = read(&mut self.idle_fn) else {
            return;
        };
        // An identical reading while time has passed is a cached/stale value; it
        // carries no information and would otherwise look like fresh input.
        let stale = self.prev_idle == Some(idle);
        self.prev_idle = Some(idle);
        if stale {
            if let (true, Some(seen)) = (changed, self.os_seen) {
                // The pointer moved, so this poll did see input although the idle
                // timer repeats its value: it ticks in coarse steps (15.6 ms on
                // Windows), so two polls during continuous motion can both read 0.
                // Consume the reading, or once the pointer stops the next poll
                // would mistake this very input for a key press.
                self.os_seen = Some(seen.max(now - idle));
            }
            return;
        }
        let input_time = now - idle;
        if let (true, Some(seen), false) = (key_idle.is_none(), self.os_seen, changed)
            && input_time > seen + IDLE_RESET_EPS_S
        {
            // The idle timer was reset but the pointer did not move: a key press
            // (or a click/scroll in place).
            self.last_key = self.last_key.max(input_time);
        }
        self.os_seen = Some(self.os_seen.map_or(input_time, |s| s.max(input_time)));
        self.last_any = self.last_any.max(input_time);
    }

    fn register_manual_move(&mut self, now: f64) {
        self.manual_move = true;
        self.last_mouse = now;
        self.last_any = self.last_any.max(now);
    }

    /// Count slow pointer motion whose single steps are below the move threshold.
    ///
    /// Without this a slow drag (a slider, a precise selection) would register
    /// as neither mouse nor keyboard use and the cursor could be warped away
    /// in the middle of it.
    fn track_drift(&mut self, prev: Point, x: i32, y: i32, now: f64) {
        if let Some(warp) = self.warp
            && dist((x, y), warp) <= WARP_TOLERANCE_PX
        {
            self.drift_anchor = None; // our own warp settling onto its target
            return;
        }
        let anchor = match self.drift_anchor {
            Some(a) if now - self.drift_since <= DRIFT_WINDOW_S => a,
            _ => {
                self.drift_since = now;
                prev
            }
        };
        self.drift_anchor = Some(anchor);
        if dist((x, y), anchor) > DRIFT_DISTANCE_PX {
            self.drift_anchor = None;
            self.register_manual_move(now);
        }
    }

    fn explained_by_warp(&self, x: i32, y: i32, now: f64) -> bool {
        let Some(warp) = self.warp else {
            return false;
        };
        if dist((x, y), warp) > WARP_TOLERANCE_PX {
            return false;
        }
        (now - self.warp_time) <= WARP_WINDOW_S || !self.warp_polled
    }
}

/// One idle reading; anything that is not a finite non-negative number is "unknown".
fn read(provider: &mut IdleProvider) -> Option<f64> {
    provider().filter(|s| s.is_finite() && *s >= 0.0)
}

#[cfg(test)]
#[path = "input_state_tests.rs"]
mod tests;
