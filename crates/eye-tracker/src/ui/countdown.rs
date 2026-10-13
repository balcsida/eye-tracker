//! Walk-away countdown toast.
//!
//! When the controller sees nobody at the desk it announces the configured action (lock, displays
//! off, …) a few seconds in advance. The toast shows that countdown with a shrinking ring and
//! tells the user how to cancel it. It never takes keyboard focus, so it cannot interrupt typing.
//! It hides as soon as the controller reports `away_cancelled` or the state leaves the tracking
//! states.
//!
//! [`CountdownToast`] holds the state (what to show where, the text, the timing) and
//! [`CountdownToast::show`] draws it as an egui viewport.

use std::cell::Cell;
use std::sync::Arc;
use std::time::Duration;

use egui::{Align2, Color32, Context, FontId, TextureHandle, TextureOptions, ViewportId};
use tiny_skia::{LineCap, Paint, Path, PathBuilder, Pixmap, Stroke, Transform};

use super::icons::{self, Rgba};
use super::util;
use crate::config::Settings;
use crate::engine::controller::away_action_for;
use crate::platform::PlatformServices;
use crate::types::{Monitor, Rect, TrackingState, monitor_at};

/// The title per presence action (`{n}` is the seconds left); other actions show nothing.
fn title_template(action: &str) -> Option<&'static str> {
    match action {
        "lock" | "lock_and_display_off" => Some("Locking in {n} s"),
        "display_off" => Some("Turning displays off in {n} s"),
        "notify" => Some("Marking you as away in {n} s"),
        _ => None,
    }
}

/// The title once the countdown reached zero.
fn now_title(action: &str) -> &'static str {
    match action {
        "display_off" => "Turning displays off…",
        "notify" => "Marking you as away…",
        _ => "Locking now…",
    }
}

/// States in which a pending countdown can no longer complete.
fn cancels_countdown(state: TrackingState) -> bool {
    matches!(
        state,
        TrackingState::Away
            | TrackingState::Paused
            | TrackingState::Privacy
            | TrackingState::Locked
            | TrackingState::Yielded
            | TrackingState::Calibrating
    )
}

/// Seconds between repaints while a countdown runs (the Python tick of 250 ms).
const TICK_S: f64 = 0.25;
const FADE_S: f64 = 0.16;

/// The settings the toast reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CountdownSettings {
    /// `settings.presence.action`.
    pub action: String,
    /// `settings.general.first_run_done`.
    pub first_run_done: bool,
    /// `settings.presence.require_input_idle`.
    pub require_input_idle: bool,
}

impl CountdownSettings {
    pub fn from_settings(settings: &Settings) -> Self {
        Self {
            action: settings.presence.action.clone(),
            first_run_done: settings.general.first_run_done,
            require_input_idle: settings.presence.require_input_idle,
        }
    }
}

/// Where the toast goes: what it needs to know about the screens and the mouse.
#[derive(Debug, Clone, Copy)]
pub struct ToastEnv<'a> {
    /// The controller's monitor layout.
    pub monitors: &'a [Monitor],
    /// Every screen the system knows (`platform::monitors()`).
    pub screens: &'a [Monitor],
    /// The mouse position (`PlatformServices::cursor_position`).
    pub cursor: Option<(f64, f64)>,
}

/// Text sizes in points, for fitting the toast around its text.
pub trait TextMeasure {
    fn title_width(&self, text: &str) -> f64;
    fn hint_width(&self, text: &str) -> f64;
    fn title_height(&self) -> f64;
    fn hint_height(&self) -> f64;
}

/// Everything [`CountdownToast::show`] reads each frame (what the controller's signals told the
/// Python toast).
#[derive(Debug, Clone, Copy)]
pub struct CountdownInputs<'a> {
    pub env: ToastEnv<'a>,
    pub state: TrackingState,
    pub settings: &'a CountdownSettings,
    /// `away_warning(remaining_s)` seen since the last frame (the controller may repeat it).
    pub warning: Option<f64>,
    /// `away_cancelled` seen since the last frame.
    pub cancelled: bool,
    /// Seconds on any monotonic clock.
    pub now: f64,
}

/// Small rounded always-on-top toast: "Locking in 8 s — move the mouse or …".
///
/// It keeps its own deadline, so the controller may send the warning once or repeat it. The text
/// names the action that will really run (until the first-run setup is finished a lock is only a
/// notification). For `"none"` nothing is shown, because nothing is going to happen. The hint
/// offers only what can cancel here: keyboard and mouse use cannot be observed on every desktop
/// (Wayland outside GNOME). Time is an argument (`now`, in seconds) everywhere.
pub struct CountdownToast {
    platform: Option<Arc<dyn PlatformServices>>,
    settings: CountdownSettings,
    /// Whether keyboard and mouse use can be observed (asked once, when needed).
    input_observable: Cell<Option<bool>>,
    deadline: Option<f64>,
    total: f64,
    visible: bool,
    shown_at: f64,
    needs_place: bool,
    scale: f64,
    /// The toast's rectangle in native pixels once placed.
    geometry: Option<Rect>,
    last_state: Option<TrackingState>,
    last_settings: Option<CountdownSettings>,
    ring: Option<((u32, u32), TextureHandle)>,
}

impl CountdownToast {
    pub fn new(platform: Option<Arc<dyn PlatformServices>>, settings: CountdownSettings) -> Self {
        Self {
            platform,
            settings,
            input_observable: Cell::new(None),
            deadline: None,
            total: 1.0,
            visible: false,
            shown_at: 0.0,
            needs_place: false,
            scale: 1.0,
            geometry: None,
            last_state: None,
            last_settings: None,
            ring: None,
        }
    }

    /// The presence action being counted down to: `settings.presence.action`, or `"notify"` for
    /// a lock or blanked displays while the setup is unfinished.
    pub fn action(&self) -> &str {
        away_action_for(&self.settings.action, self.settings.first_run_done)
    }

    /// Whether a countdown is running (the toast is shown).
    pub fn active(&self) -> bool {
        self.deadline.is_some()
    }

    pub fn visible(&self) -> bool {
        self.visible
    }

    /// The toast's rectangle in native pixels (after [`place`](Self::place)).
    pub fn geometry(&self) -> Option<Rect> {
        self.geometry
    }

    /// Start or update the countdown: the action happens in `remaining_s`.
    pub fn show_warning(&mut self, remaining_s: f64, now: f64) {
        if title_template(self.action()).is_none() {
            self.cancel();
            return;
        }
        let remaining = if remaining_s.is_finite() {
            remaining_s.max(0.0)
        } else {
            0.0
        };
        // A later deadline than the current one means a new countdown.
        if self.deadline.is_none() || remaining > self.remaining(now) + 0.5 {
            self.total = remaining.max(1e-6);
        }
        self.deadline = Some(now + remaining);
        if !self.visible {
            self.visible = true;
            self.shown_at = now;
            self.needs_place = true;
        }
    }

    /// Stop the countdown and hide the toast.
    pub fn cancel(&mut self) {
        self.deadline = None;
        self.visible = false;
        self.needs_place = false;
    }

    /// Seconds left until the action (0 when not counting down).
    pub fn remaining(&self, now: f64) -> f64 {
        self.deadline
            .map_or(0.0, |deadline| (deadline - now).max(0.0))
    }

    /// Remaining whole seconds as displayed (rounded up).
    pub fn seconds_left(&self, now: f64) -> i64 {
        ((self.remaining(now) - 1e-6).ceil() as i64).max(0)
    }

    /// E.g. `"Locking in 8 s"` (`"Locking now…"` at zero).
    pub fn title_text(&self, now: f64) -> String {
        let action = self.known_action();
        let n = self.seconds_left(now);
        match title_template(action) {
            Some(template) if n > 0 => template.replace("{n}", &n.to_string()),
            _ => now_title(action).to_string(),
        }
    }

    /// How to cancel: input counts as presence only when the settings say so and the system lets
    /// the app observe it.
    pub fn hint_text(&self) -> &'static str {
        if self.settings.require_input_idle && self.input_is_observable() {
            "Move the mouse or look at the camera to cancel"
        } else {
            "Look at the camera to cancel"
        }
    }

    /// The platform's `input_idle` capability (true when it cannot be read).
    ///
    /// Wayland desktops other than GNOME offer no idle time: keystrokes are never seen there,
    /// and pointer moves only over X11 (XWayland) windows.
    fn input_is_observable(&self) -> bool {
        if let Some(observable) = self.input_observable.get() {
            return observable;
        }
        let capabilities = self
            .platform
            .as_ref()
            .map(|platform| platform.capabilities());
        let observable = capabilities
            .and_then(|c| c.get("input_idle").copied())
            .unwrap_or(true);
        self.input_observable.set(Some(observable));
        observable
    }

    /// The whole message on one line (also the accessible name).
    pub fn text(&self, now: f64) -> String {
        let hint = self.hint_text();
        let mut chars = hint.chars();
        let lowered: String = chars
            .next()
            .map(|first| first.to_lowercase().chain(chars).collect())
            .unwrap_or_default();
        format!("{} — {lowered}", self.title_text(now))
    }

    /// The action whose texts are shown (a lock for any action without a countdown).
    fn known_action(&self) -> &str {
        let action = self.action();
        if title_template(action).is_some() {
            action
        } else {
            "lock"
        }
    }

    fn target_rect(env: &ToastEnv<'_>) -> Option<Rect> {
        // Where the cursor is, is where the user last worked.
        let at_cursor = env.cursor.and_then(|(x, y)| monitor_at(env.monitors, x, y));
        at_cursor
            .or_else(|| util::primary_monitor(env.monitors))
            .or_else(|| util::primary_monitor(env.screens))
            .map(|monitor| monitor.rect)
    }

    /// Size the toast around its text and centre it on the monitor under the cursor.
    pub fn place(&mut self, env: &ToastEnv<'_>, measure: &dyn TextMeasure) {
        self.needs_place = false;
        let rect = Self::target_rect(env);
        let scale = rect
            .and_then(|r| {
                let (cx, cy) = r.center();
                util::screen_for_point(cx, cy, env.screens)
                    .or_else(|| util::screen_for_point(cx, cy, env.monitors))
            })
            .map(|screen| screen.scale);
        let s = util::ui_scale(scale);
        self.scale = s;
        let action = self.known_action();
        let n = 10.max(self.total.ceil() as i64);
        let widest_title = measure
            .title_width(
                &title_template(action)
                    .unwrap_or_default()
                    .replace("{n}", &n.to_string()),
            )
            .max(measure.title_width(now_title(action)));
        let text_w = widest_title.max(measure.hint_width(self.hint_text())) * s;
        let width = (22.0 * s + 52.0 * s + 18.0 * s + text_w + 26.0 * s).ceil();
        let text_h = (measure.title_height() + measure.hint_height()) * s;
        let height = (92.0 * s).max(text_h + 44.0 * s).ceil();
        let Some(rect) = rect else {
            return;
        };
        let (cx, cy) = rect.center();
        self.geometry = Some(Rect::new(
            (cx - width / 2.0).round_ties_even() as i32,
            (cy - height / 2.0).round_ties_even() as i32,
            width as i32,
            height as i32,
        ));
    }

    pub fn on_state_changed(&mut self, state: TrackingState) {
        if cancels_countdown(state) {
            self.cancel();
        }
    }

    pub fn on_settings_changed(&mut self, settings: &CountdownSettings) {
        self.settings = settings.clone();
        if title_template(self.action()).is_none() {
            self.cancel();
        } else if self.visible {
            self.needs_place = true;
        }
    }

    /// The fade-in opacity (0..1; ease-out cubic over 160 ms).
    pub fn opacity(&self, now: f64) -> f32 {
        let t = ((now - self.shown_at) / FADE_S).clamp(0.0, 1.0);
        (1.0 - (1.0 - t).powi(3)) as f32
    }

    /// The ring's remaining share (0..1).
    pub fn fraction(&self, now: f64) -> f64 {
        if self.total > 0.0 {
            (self.remaining(now) / self.total).min(1.0)
        } else {
            0.0
        }
    }

    /// The window: frameless, always on top, see-through and never focused.
    pub fn viewport_builder(&self) -> egui::ViewportBuilder {
        let builder = util::floating_viewport("Countdown", false, false, true);
        match self.geometry {
            Some(rect) => util::place_viewport(builder, rect, self.scale),
            None => builder,
        }
    }

    /// Apply this frame's inputs: what the controller's signals told the Python toast.
    pub fn update(&mut self, inputs: &CountdownInputs<'_>) {
        if self.last_settings.as_ref() != Some(inputs.settings) {
            self.last_settings = Some(inputs.settings.clone());
            self.on_settings_changed(inputs.settings);
        }
        if self.last_state != Some(inputs.state) {
            self.last_state = Some(inputs.state);
            self.on_state_changed(inputs.state);
        }
        if inputs.cancelled {
            self.cancel();
        }
        if let Some(remaining) = inputs.warning {
            self.show_warning(remaining, inputs.now);
        }
    }

    /// Apply `inputs` and draw the toast while a countdown runs.
    pub fn show(&mut self, ctx: &Context, inputs: &CountdownInputs<'_>) {
        self.update(inputs);
        if !self.visible || self.deadline.is_none() {
            return;
        }
        let fonts = FontSet::new(ctx);
        if self.needs_place || self.geometry.is_none() {
            self.place(&inputs.env, &EguiMeasure { ctx, fonts: &fonts });
        }
        let now = inputs.now;
        let opacity = self.opacity(now);
        let ring = self.ring_texture(ctx, now);
        let (title, hint) = (self.title_text(now), self.hint_text());
        let number = self.seconds_left(now).to_string();
        let id = ViewportId::from_hash_of("countdown-toast");
        let builder = self.viewport_builder();
        ctx.show_viewport_immediate(id, builder.clone(), |ui, _class| {
            util::keep_viewport_placed(ui.ctx(), &builder);
            draw_toast(ui, &fonts, opacity, ring.as_ref(), &title, hint, &number);
        });
        let next = if opacity < 1.0 { 0.016 } else { TICK_S };
        ctx.request_repaint_after(Duration::from_secs_f64(next));
    }

    fn ring_texture(&mut self, ctx: &Context, now: f64) -> Option<TextureHandle> {
        let side = (52.0 * self.scale).ceil().max(1.0) as u32;
        // 360 steps are finer than a pixel of the ring and keep the texture reuse cheap.
        let steps = (self.fraction(now) * 360.0).round() as u32;
        let key = (side, steps);
        if self.ring.as_ref().is_none_or(|(have, _)| *have != key) {
            let image = render_ring(side, f64::from(steps) / 360.0)?.to_color_image();
            match &mut self.ring {
                Some((have, texture)) => {
                    texture.set(image, TextureOptions::LINEAR);
                    *have = key;
                }
                None => {
                    let texture = ctx.load_texture("countdown-ring", image, TextureOptions::LINEAR);
                    self.ring = Some((key, texture));
                }
            }
        }
        self.ring.as_ref().map(|(_, texture)| texture.clone())
    }
}

// ----------------------------------------------------------------------------- drawing
/// An arc of the circle (`cx`, `cy`, `r`) from `start_deg` by `sweep_deg`, with Qt's angle
/// convention (counter-clockwise from 3 o'clock; a negative sweep runs clockwise).
fn arc_path(cx: f32, cy: f32, r: f32, start_deg: f64, sweep_deg: f64) -> Option<Path> {
    if sweep_deg == 0.0 {
        return None;
    }
    let segments = (sweep_deg.abs() / 90.0).ceil().max(1.0);
    let step = (sweep_deg / segments).to_radians();
    let k = 4.0 / 3.0 * (step / 4.0).tan();
    let point = |theta: f64| {
        (
            f64::from(cx) + f64::from(r) * theta.cos(),
            f64::from(cy) - f64::from(r) * theta.sin(),
        )
    };
    let slope = |theta: f64| (-f64::from(r) * theta.sin(), -f64::from(r) * theta.cos());
    let mut theta = start_deg.to_radians();
    let mut builder = PathBuilder::new();
    let first = point(theta);
    builder.move_to(first.0 as f32, first.1 as f32);
    for _ in 0..segments as u32 {
        let next = theta + step;
        let (p0, p3) = (point(theta), point(next));
        let (d0, d3) = (slope(theta), slope(next));
        builder.cubic_to(
            (p0.0 + k * d0.0) as f32,
            (p0.1 + k * d0.1) as f32,
            (p3.0 - k * d3.0) as f32,
            (p3.1 - k * d3.1) as f32,
            p3.0 as f32,
            p3.1 as f32,
        );
        theta = next;
    }
    builder.finish()
}

/// The countdown ring of `side` pixels: a faint track and an accent arc that shrinks clockwise
/// from 12 o'clock as `fraction` (remaining / total) falls.
pub fn render_ring(side: u32, fraction: f64) -> Option<Rgba> {
    let mut pixmap = Pixmap::new(side, side)?;
    let s = side as f32 / 52.0;
    let pen_w = (4.0 * s).max(2.0);
    let centre = side as f32 / 2.0;
    let radius = centre - pen_w / 2.0;
    let identity = Transform::identity();
    let mut track = Paint {
        anti_alias: true,
        ..Paint::default()
    };
    track.set_color(util::skia_color([255, 255, 255, 34]));
    let style = Stroke {
        width: pen_w,
        ..Stroke::default()
    };
    pixmap.stroke_path(
        &PathBuilder::from_circle(centre, centre, radius)?,
        &track,
        &style,
        identity,
        None,
    );
    let fraction = fraction.min(1.0);
    if fraction > 0.0 {
        let arc = Paint {
            shader: util::accent_gradient((0.0, 0.0), (side as f32, side as f32), false),
            anti_alias: true,
            ..Paint::default()
        };
        let style = Stroke {
            width: pen_w,
            line_cap: LineCap::Round,
            ..Stroke::default()
        };
        if let Some(path) = arc_path(centre, centre, radius, 90.0, -fraction * 360.0) {
            pixmap.stroke_path(&path, &arc, &style, identity, None);
        }
    }
    Some(icons::to_rgba(&pixmap))
}

/// The fonts of the toast: title, hint and the number in the ring.
struct FontSet {
    title: FontId,
    hint: FontId,
    number: FontId,
}

impl FontSet {
    fn new(ctx: &Context) -> Self {
        let base = ctx.global_style().text_styles[&egui::TextStyle::Body].size;
        Self {
            title: FontId::proportional(base * 1.45),
            hint: FontId::proportional(base * 1.08),
            number: FontId::proportional(base * 1.6),
        }
    }
}

struct EguiMeasure<'a> {
    ctx: &'a Context,
    fonts: &'a FontSet,
}

impl EguiMeasure<'_> {
    fn width(&self, text: &str, font: &FontId) -> f64 {
        self.ctx.fonts_mut(|fonts| {
            f64::from(
                fonts
                    .layout_no_wrap(text.to_string(), font.clone(), Color32::WHITE)
                    .size()
                    .x,
            )
        })
    }
}

impl TextMeasure for EguiMeasure<'_> {
    fn title_width(&self, text: &str) -> f64 {
        self.width(text, &self.fonts.title)
    }

    fn hint_width(&self, text: &str) -> f64 {
        self.width(text, &self.fonts.hint)
    }

    fn title_height(&self) -> f64 {
        self.ctx
            .fonts_mut(|fonts| f64::from(fonts.row_height(&self.fonts.title)))
    }

    fn hint_height(&self) -> f64 {
        self.ctx
            .fonts_mut(|fonts| f64::from(fonts.row_height(&self.fonts.hint)))
    }
}

fn draw_toast(
    ui: &mut egui::Ui,
    fonts: &FontSet,
    opacity: f32,
    ring: Option<&TextureHandle>,
    title: &str,
    hint: &str,
    number: &str,
) {
    let mut painter = ui.painter().clone();
    painter.set_opacity(opacity);
    let bounds = ui.max_rect();
    util::paint_surface(
        &painter,
        bounds,
        14.0,
        util::SURFACE,
        util::SURFACE_ALPHA,
        26,
    );

    // Countdown ring: a faint track and an accent arc that shrinks clockwise.
    let d = 52.0;
    let ring_rect = egui::Rect::from_min_size(
        bounds.min + egui::vec2(22.0, (bounds.height() - d) / 2.0),
        egui::Vec2::splat(d),
    );
    if let Some(ring) = ring {
        painter.image(
            ring.id(),
            ring_rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            Color32::WHITE,
        );
    }
    let text = util::color32(util::qcolor(util::TEXT, None));
    painter.text(
        ring_rect.center(),
        Align2::CENTER_CENTER,
        number,
        fonts.number.clone(),
        text,
    );

    let left = ring_rect.right() + 18.0;
    let (title_h, hint_h) = ui
        .ctx()
        .fonts_mut(|f| (f.row_height(&fonts.title), f.row_height(&fonts.hint)));
    let gap = 4.0;
    let top = bounds.min.y + (bounds.height() - (title_h + gap + hint_h)) / 2.0;
    painter.text(
        egui::pos2(left, top + title_h / 2.0),
        Align2::LEFT_CENTER,
        title,
        fonts.title.clone(),
        text,
    );
    painter.text(
        egui::pos2(left, top + title_h + gap + hint_h / 2.0),
        Align2::LEFT_CENTER,
        hint,
        fonts.hint.clone(),
        util::color32(util::qcolor(util::TEXT_MUTED, None)),
    );
}

#[cfg(test)]
#[path = "countdown_tests.rs"]
mod tests;
