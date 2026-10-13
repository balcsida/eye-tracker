//! Per-monitor memory of the cursor position and the focused window.
//!
//! When the user looks back at a monitor they usually want to continue where they
//! left off there: the cursor returns to where it was and keyboard focus goes back
//! to the window they were using, instead of both landing in the middle of the
//! screen.

use std::collections::HashMap;

use crate::types::{Monitor, Rect, WindowRef};

/// Accepted values of `mode` in [`choose_cursor_target`].
pub const CURSOR_MODES: [&str; 3] = ["last", "center", "gaze"];

/// A gaze-placed cursor keeps this fraction of the monitor size (but at least
/// [`GAZE_INSET_MIN_PX`]) away from the monitor's edges: a pointer parked on
/// the outermost pixel row, column or corner opens auto-hidden taskbars, docks
/// and panels and fires hot corners (GNOME Activities, KWin, macOS actions).
pub const GAZE_INSET_FRACTION: f64 = 0.02;
pub const GAZE_INSET_MIN_PX: i64 = 16;

/// Last cursor position and last focused window, per monitor index.
#[derive(Debug, Default)]
pub struct WindowMemory {
    cursor: HashMap<i32, (i32, i32)>,
    window: HashMap<i32, WindowRef>,
}

impl WindowMemory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Remember where the user left the cursor on a monitor.
    pub fn record_cursor(&mut self, monitor_index: i32, pos: (i32, i32)) {
        self.cursor.insert(monitor_index, pos);
    }

    pub fn last_cursor(&self, monitor_index: i32) -> Option<(i32, i32)> {
        self.cursor.get(&monitor_index).copied()
    }

    /// Remember the window the user last worked in on a monitor.
    pub fn record_window(&mut self, monitor_index: i32, window: WindowRef) {
        self.window.insert(monitor_index, window);
    }

    pub fn last_window(&self, monitor_index: i32) -> Option<&WindowRef> {
        self.window.get(&monitor_index)
    }

    /// Drop the remembered window (e.g. it was closed or moved elsewhere).
    pub fn forget_window(&mut self, monitor_index: i32) {
        self.window.remove(&monitor_index);
    }

    /// Forget everything (e.g. after the monitor layout changed).
    pub fn clear(&mut self) {
        self.cursor.clear();
        self.window.clear();
    }
}

/// Where to put the cursor when switching to `monitor`.
///
/// * `"last"` - the remembered position on that monitor; else the centre of
///   the part of `window_rect` (the window about to get focus) that lies on
///   the monitor; else the monitor centre.
/// * `"center"` - the monitor centre.
/// * `"gaze"` - the gaze point, kept a small inset away from the monitor's
///   edges (see [`GAZE_INSET_FRACTION`]); else the monitor centre.
///
/// The result always lies inside `monitor.rect`. Unknown modes behave like
/// `"center"`.
pub fn choose_cursor_target(
    mode: &str,
    monitor: &Monitor,
    memory: &WindowMemory,
    gaze: Option<(f64, f64)>,
    window_rect: Option<Rect>,
) -> (i32, i32) {
    let rect = monitor.rect;
    match mode {
        "last" => {
            if let Some((x, y)) = memory.last_cursor(monitor.index)
                && rect.contains(f64::from(x), f64::from(y))
            {
                return rect.clamp(f64::from(x), f64::from(y));
            }
            if let Some(visible) = window_rect.and_then(|w| intersection(&w, &rect)) {
                let (cx, cy) = visible.center();
                return rect.clamp(cx, cy);
            }
        }
        "gaze" => {
            if let Some((gx, gy)) = gaze
                && gx.is_finite()
                && gy.is_finite()
            {
                return clamp_inset(&rect, gx, gy);
            }
        }
        "center" => {}
        _ => log::warn!("Unknown cursor target mode {mode:?}; using the monitor centre"),
    }
    let (cx, cy) = rect.center();
    rect.clamp(cx, cy)
}

/// Nearest integer point of `rect` shrunk by the gaze inset on every side.
///
/// For monitors too small for the inset the range collapses to the centre, so
/// the result always stays inside `rect`.
fn clamp_inset(rect: &Rect, px: f64, py: f64) -> (i32, i32) {
    let inset = |size: i32| -> i64 {
        let wanted = ((GAZE_INSET_FRACTION * f64::from(size)).round_ties_even() as i64)
            .max(GAZE_INSET_MIN_PX);
        wanted.min(i64::from((size - 1) / 2).max(0))
    };
    let (ix, iy) = (inset(rect.w), inset(rect.h));
    let x = (px.round_ties_even() as i64)
        .max(i64::from(rect.x) + ix)
        .min(i64::from(rect.right()) - 1 - ix);
    let y = (py.round_ties_even() as i64)
        .max(i64::from(rect.y) + iy)
        .min(i64::from(rect.bottom()) - 1 - iy);
    (x as i32, y as i32)
}

fn intersection(a: &Rect, b: &Rect) -> Option<Rect> {
    let (x0, y0) = (a.x.max(b.x), a.y.max(b.y));
    let (x1, y1) = (a.right().min(b.right()), a.bottom().min(b.bottom()));
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some(Rect::new(x0, y0, x1 - x0, y1 - y0))
}

#[cfg(test)]
#[path = "window_memory_tests.rs"]
mod tests;
