//! Gaze dot: shows where the app thinks you are looking (a testing aid).
//!
//! Each monitor gets its own small, click-through, always-on-top window (an egui immediate
//! viewport) that only ever moves within that monitor. The dot is shown on the monitor that
//! contains the gaze point and hidden on all others. Small windows keep the cost negligible,
//! since a full-screen translucent overlay would recomposite a whole display on every update.
//! Keeping each window on its own monitor also avoids DPI changes when a window crosses between
//! displays.
//!
//! A gaze point outside every monitor (looking at the desk or a phone) is shown as a hollow ring
//! clamped to the edge of the nearest monitor.
//!
//! [`GazeOverlay`] holds the state and decides what is shown where; [`GazeOverlay::show`] only
//! draws it. The viewports are transparent: the app's `clear_color` must be transparent too.

use std::collections::BTreeMap;

use egui::{
    Color32, ColorImage, Context, TextureHandle, TextureOptions, ViewportCommand, ViewportId,
};
use tiny_skia::{FillRule, Paint, PathBuilder, Pixmap, Stroke, Transform};

use super::icons::{self, Rgba};
use super::util;
use crate::types::{GazePoint, Monitor, TrackingState, monitor_at, nearest_monitor};

/// Dot diameter in design pixels (scaled per monitor with `ui_scale`).
pub const DOT_DIAMETER: f64 = 28.0;
/// Window side as a multiple of the dot diameter (room for the glow).
const GLOW: f64 = 2.0;

/// States in which the controller produces gaze estimates.
fn is_gaze_state(state: TrackingState) -> bool {
    state == TrackingState::Tracking
}

/// A small click-through window that draws the dot for one monitor.
pub struct DotWindow {
    pub monitor: Monitor,
    pub scale: f64,
    pub radius: f64,
    /// Window side in native pixels.
    pub side: i32,
    /// Window position in native pixels.
    pub x: i32,
    pub y: i32,
    /// The dot's centre within the window.
    pub dot: (f64, f64),
    pub off_screen: bool,
    texture: Option<TextureHandle>,
    drawn: Option<((f64, f64), bool)>,
    sent_position: Option<(i32, i32)>,
}

impl DotWindow {
    pub fn new(monitor: &Monitor, scale: f64) -> Self {
        let side = (DOT_DIAMETER * GLOW * scale).ceil() as i32;
        Self {
            monitor: monitor.clone(),
            scale,
            radius: DOT_DIAMETER * scale / 2.0,
            side,
            x: 0,
            y: 0,
            dot: (f64::from(side) / 2.0, f64::from(side) / 2.0),
            off_screen: false,
            texture: None,
            drawn: None,
            sent_position: None,
        }
    }

    /// The window as (x, y, width, height) in native pixels.
    pub fn geometry(&self) -> (i32, i32, i32, i32) {
        (self.x, self.y, self.side, self.side)
    }

    /// Centre the dot on (`x`, `y`), clamped into this window's monitor.
    pub fn place(&mut self, x: f64, y: f64, off_screen: bool) {
        let rect = self.monitor.rect;
        let (cx, cy) = rect.clamp(x, y);
        let side = self.side;
        // The window never leaves its monitor; near an edge the dot moves within the window
        // instead (and may be partly cut off, like the real edge).
        let left = (cx - side / 2)
            .max(rect.x)
            .min(rect.x.max(rect.right() - side));
        let top = (cy - side / 2)
            .max(rect.y)
            .min(rect.y.max(rect.bottom() - side));
        self.x = left;
        self.y = top;
        self.dot = (f64::from(cx - left), f64::from(cy - top));
        self.off_screen = off_screen;
    }

    /// The window: frameless, always on top, click-through, never focused, see-through.
    pub fn viewport_builder(&self) -> egui::ViewportBuilder {
        let (x, y, side, _) = self.geometry();
        util::place_viewport(
            util::floating_viewport("Gaze dot", true, false, true),
            crate::types::Rect::new(x, y, side, side),
            self.scale,
        )
    }

    /// The dot as a `side`×`side` image (the Python `paintEvent`).
    pub fn render(&self) -> Option<Rgba> {
        let side = self.side.max(1) as u32;
        let mut pixmap = Pixmap::new(side, side)?;
        paint_dot(
            &mut pixmap,
            self.dot,
            self.radius,
            self.scale,
            self.off_screen,
        );
        Some(icons::to_rgba(&pixmap))
    }

    fn texture_id(&mut self, ctx: &Context) -> Option<egui::TextureId> {
        let key = (self.dot, self.off_screen);
        if self.texture.is_none() || self.drawn != Some(key) {
            let image: ColorImage = self.render()?.to_color_image();
            let options = TextureOptions::LINEAR;
            match &mut self.texture {
                Some(texture) => texture.set(image, options),
                None => {
                    let name = format!("gaze-dot-{}", self.monitor.index);
                    self.texture = Some(ctx.load_texture(name, image, options));
                }
            }
            self.drawn = Some(key);
        }
        self.texture.as_ref().map(TextureHandle::id)
    }
}

fn fill_circle(
    pixmap: &mut Pixmap,
    centre: (f64, f64),
    radius: f64,
    paint: &Paint,
    stroke: Option<(&Paint, f32)>,
) {
    let Some(path) = PathBuilder::from_circle(centre.0 as f32, centre.1 as f32, radius as f32)
    else {
        return;
    };
    pixmap.fill_path(&path, paint, FillRule::Winding, Transform::identity(), None);
    if let Some((pen, width)) = stroke {
        let style = Stroke {
            width,
            ..Stroke::default()
        };
        pixmap.stroke_path(&path, pen, &style, Transform::identity(), None);
    }
}

fn solid(color: util::Color8) -> Paint<'static> {
    let mut paint = Paint {
        anti_alias: true,
        ..Paint::default()
    };
    paint.set_color(util::skia_color(color));
    paint
}

fn shaded(shader: tiny_skia::Shader<'static>) -> Paint<'static> {
    Paint {
        shader,
        anti_alias: true,
        ..Paint::default()
    }
}

fn paint_dot(pixmap: &mut Pixmap, c: (f64, f64), r: f64, scale: f64, off_screen: bool) {
    let pen_width = (2.0 * scale).max(1.5) as f32;
    if off_screen {
        fill_circle(
            pixmap,
            c,
            r * 0.8,
            &solid(util::qcolor(util::ACCENT, Some(70))),
            Some((&solid([255, 255, 255, 150]), pen_width)),
        );
        return;
    }
    let (cx, cy) = (c.0 as f32, c.1 as f32);
    let glow = util::radial_gradient(
        (cx, cy),
        (r * GLOW) as f32,
        &[
            (0.0, util::qcolor(util::ACCENT, Some(110))),
            (0.55, util::qcolor(util::ACCENT, Some(40))),
            (1.0, util::qcolor(util::ACCENT, Some(0))),
        ],
    );
    fill_circle(pixmap, c, r * GLOW, &shaded(glow), None);
    let rf = r as f32;
    let body = util::accent_gradient((cx - rf, cy - rf), (cx + rf, cy + rf), false);
    fill_circle(
        pixmap,
        c,
        r * 0.82,
        &shaded(body),
        Some((&solid([255, 255, 255, 230]), pen_width)),
    );
    fill_circle(pixmap, c, r * 0.2, &solid([255, 255, 255, 235]), None);
}

/// What [`GazeOverlay::show`] reads each frame (the Python widget read these from the controller
/// and its signals).
#[derive(Debug, Clone, Copy)]
pub struct OverlayInputs<'a> {
    /// The controller's monitor layout.
    pub monitors: &'a [Monitor],
    /// `settings.ui.show_gaze_overlay`.
    pub show_gaze_overlay: bool,
    pub state: TrackingState,
    /// The latest gaze estimate, `None` when there is none.
    pub gaze: Option<GazePoint>,
    /// Seconds on any monotonic clock.
    pub now: f64,
}

/// Shows the controller's gaze as a soft dot on the monitor being looked at.
///
/// By default it follows `settings.ui.show_gaze_overlay` (initially and when it changes);
/// [`set_enabled`](Self::set_enabled) switches it directly. The dot hides when the gaze is
/// `None`, when tracking stops, or when no gaze update arrives for `stale_ms` (a frozen dot would
/// be misleading).
pub struct GazeOverlay {
    follow_settings: bool,
    enabled: bool,
    monitors: Vec<Monitor>,
    windows: BTreeMap<i32, DotWindow>,
    visible: Option<i32>,
    stale_s: f64,
    stale_due: Option<f64>,
    last_setting: Option<bool>,
    last_state: Option<TrackingState>,
    last_gaze: Option<GazePoint>,
}

impl GazeOverlay {
    /// A disabled overlay; `follow_settings` makes [`update`](Self::update) follow the setting.
    pub fn new(follow_settings: bool, stale_ms: u64) -> Self {
        Self {
            follow_settings,
            enabled: false,
            monitors: Vec::new(),
            windows: BTreeMap::new(),
            visible: None,
            stale_s: stale_ms.max(1) as f64 / 1000.0,
            stale_due: None,
            last_setting: None,
            last_state: None,
            last_gaze: None,
        }
    }

    /// An overlay that follows the setting, starting with its current value.
    pub fn from_settings(show_gaze_overlay: bool) -> Self {
        let mut overlay = Self::new(true, 2500);
        overlay.last_setting = Some(show_gaze_overlay);
        overlay.set_enabled(show_gaze_overlay);
        overlay
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Index of the monitor currently showing the dot, if any.
    pub fn visible_monitor(&self) -> Option<i32> {
        self.visible
    }

    /// The per-monitor dot windows created so far (by monitor index).
    pub fn windows(&self) -> &BTreeMap<i32, DotWindow> {
        &self.windows
    }

    /// Turn the overlay on or off (off also destroys its windows).
    ///
    /// Windows are created lazily by the first gaze update, so an enabled overlay costs nothing
    /// while tracking is paused.
    pub fn set_enabled(&mut self, enabled: bool) {
        if enabled == self.enabled {
            return;
        }
        self.enabled = enabled;
        if !enabled {
            self.stale_due = None;
            self.destroy_windows();
        }
        log::debug!(
            "Gaze overlay {}",
            if enabled { "enabled" } else { "disabled" }
        );
    }

    /// Move the dot to `point` (global coordinates); `None` hides it. `monitors` is the
    /// controller's current layout and `now` the time in seconds.
    pub fn set_gaze(&mut self, point: Option<GazePoint>, monitors: &[Monitor], now: f64) {
        if !self.enabled {
            return;
        }
        let Some(point) = point.filter(|p| p.x.is_finite() && p.y.is_finite()) else {
            self.stale_due = None;
            self.show_on(None);
            return;
        };
        // Re-read the layout on every update: it is a short list copy, and the controller
        // rebuilds it (debounced) some time after a screen change.
        self.refresh_monitors(monitors);
        if self.monitors.is_empty() {
            return;
        }
        let at = monitor_at(&self.monitors, point.x, point.y).cloned();
        let off_screen = at.is_none();
        let monitor = match at {
            Some(monitor) => monitor,
            None => match nearest_monitor(&self.monitors, point.x, point.y) {
                Ok((monitor, _distance)) => monitor.clone(),
                Err(_) => return,
            },
        };
        let scale = util::ui_scale(Some(monitor.scale));
        let window = self
            .windows
            .entry(monitor.index)
            .or_insert_with(|| DotWindow::new(&monitor, scale));
        if window.monitor != monitor {
            *window = DotWindow::new(&monitor, scale);
        }
        window.place(point.x, point.y, off_screen);
        self.show_on(Some(monitor.index));
        self.stale_due = Some(now + self.stale_s);
    }

    /// Re-read the monitor layout; windows are rebuilt on change.
    pub fn refresh_monitors(&mut self, monitors: &[Monitor]) {
        if monitors == self.monitors.as_slice() {
            return;
        }
        self.destroy_windows();
        self.monitors = monitors.to_vec();
    }

    /// Hide and destroy every window (the overlay can be re-enabled later).
    pub fn close(&mut self) {
        self.stale_due = None;
        self.destroy_windows();
    }

    /// Hide the dot once no gaze update arrived for `stale_ms`.
    pub fn tick(&mut self, now: f64) {
        if self.stale_due.is_some_and(|due| now >= due) {
            self.stale_due = None;
            self.show_on(None);
        }
    }

    pub fn on_state_changed(&mut self, state: TrackingState) {
        if !is_gaze_state(state) {
            self.stale_due = None;
            self.show_on(None);
        }
    }

    pub fn on_settings_changed(&mut self, show_gaze_overlay: bool) {
        if self.follow_settings {
            self.set_enabled(show_gaze_overlay);
        }
    }

    /// The screens were added, removed or resized. A window on a removed or resized screen would
    /// be moved somewhere by the windowing system; drop them all. The next gaze update recreates
    /// them for the new layout (with the right screen and DPI scale).
    pub fn on_screens_changed(&mut self) {
        self.stale_due = None;
        self.destroy_windows();
    }

    /// Apply this frame's inputs: what the controller's signals told the Python widget.
    pub fn update(&mut self, inputs: &OverlayInputs<'_>) {
        if self.follow_settings && self.last_setting != Some(inputs.show_gaze_overlay) {
            self.last_setting = Some(inputs.show_gaze_overlay);
            self.on_settings_changed(inputs.show_gaze_overlay);
        }
        if self.last_state != Some(inputs.state) {
            self.last_state = Some(inputs.state);
            self.on_state_changed(inputs.state);
        }
        if self.last_gaze != inputs.gaze {
            self.last_gaze = inputs.gaze;
            self.set_gaze(inputs.gaze, inputs.monitors, inputs.now);
        }
        self.tick(inputs.now);
    }

    /// Apply `inputs` and draw the dot window of the monitor being looked at.
    pub fn show(&mut self, ctx: &Context, inputs: &OverlayInputs<'_>) {
        self.update(inputs);
        let Some(index) = self.visible else {
            return;
        };
        if let Some(due) = self.stale_due {
            ctx.request_repaint_after(std::time::Duration::from_secs_f64(
                (due - inputs.now).max(0.0),
            ));
        }
        let Some(window) = self.windows.get_mut(&index) else {
            return;
        };
        let id = ViewportId::from_hash_of(("gaze-dot", index));
        let (x, y, ..) = window.geometry();
        let builder = window.viewport_builder();
        if window.sent_position.is_some_and(|sent| sent != (x, y)) {
            let position = egui::pos2(
                util::to_points(f64::from(x), window.scale),
                util::to_points(f64::from(y), window.scale),
            );
            ctx.send_viewport_cmd_to(id, ViewportCommand::OuterPosition(position));
        }
        window.sent_position = Some((x, y));
        let Some(texture) = window.texture_id(ctx) else {
            return;
        };
        ctx.show_viewport_immediate(id, builder, |ui, _class| {
            let rect = ui.max_rect();
            ui.painter().image(
                texture,
                rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                Color32::WHITE,
            );
        });
    }

    fn show_on(&mut self, index: Option<i32>) {
        self.visible = index;
    }

    fn destroy_windows(&mut self) {
        self.windows.clear();
        self.visible = None;
    }
}

#[cfg(test)]
#[path = "overlay_tests.rs"]
mod tests;
