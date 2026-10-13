//! Privacy curtain: covers every monitor when someone looks over your shoulder.
//!
//! Shown when the controller reports the guard active: the shoulder guard triggered with
//! `settings.privacy.guard_action == "curtain"`, or the `"lock"` action could not lock the
//! screen. Hidden when the guard clears, when the camera is switched off (the guard could never
//! clear then), when the guard is turned off in the settings, or when the user dismisses it with
//! Esc or the button.
//!
//! Esc only works in the window that has keyboard focus. The curtain appears in response to the
//! camera, not to input to this app, so Windows' foreground lock may refuse to activate it: the
//! curtain then asks the platform layer to activate it (as gaze switching does for other apps'
//! windows), and while none of its windows has keyboard focus the text says to click Dismiss
//! instead of promising that Esc works; the Esc would go to the application hidden underneath.
//!
//! [`PrivacyCurtain`] holds the state and decides what is shown where; [`PrivacyCurtain::show`]
//! draws one opaque egui viewport per monitor and returns what the owner has to do
//! ([`CurtainAction`]).

use egui::{
    Align2, Color32, Context, FontId, Key, TextureHandle, TextureOptions, ViewportCommand,
    ViewportId,
};
use tiny_skia::{FillRule, Paint, PathBuilder, Pixmap, Stroke, Transform};

use super::icons::{self, Rgba};
use super::util;
use crate::types::{Monitor, Rect, TrackingState, monitor_at};

const TITLE: &str = "Someone is looking over your shoulder";
/// Shown while a curtain window has keyboard focus (Esc reaches it).
pub const HINT: &str = "Your screens stay covered until they leave. Press Esc to dismiss.";
/// Shown while keyboard focus is elsewhere: Esc would go to the hidden window.
pub const HINT_NO_KEYBOARD: &str =
    "Your screens stay covered until they leave. Click Dismiss to close.";
const BUTTON_TEXT: &str = "Dismiss  (Esc)";
/// Activation completes asynchronously on some platforms, and a refused one reports no focus
/// change at all, so the hint is checked again after this many seconds.
pub const FOCUS_RECHECK_S: f64 = 0.3;

/// States in which the camera is off, so the guard can no longer clear.
fn camera_off(state: TrackingState) -> bool {
    matches!(
        state,
        TrackingState::Paused
            | TrackingState::Privacy
            | TrackingState::Locked
            | TrackingState::Yielded
    )
}

/// What the owner has to do after [`PrivacyCurtain::show`] (what the Python curtain emitted or
/// called on the platform layer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurtainAction {
    /// The user closed the curtain (`dismissed`).
    Dismissed,
    /// Qt's activation was refused: activate this viewport's native window through
    /// `PlatformServices::activate_window` (with this process id), as gaze switching does.
    ActivateWindow { viewport: ViewportId, pid: u32 },
}

/// Everything [`PrivacyCurtain::show`] reads each frame.
#[derive(Debug, Clone, Copy)]
pub struct CurtainInputs<'a> {
    /// The controller's monitor layout.
    pub monitors: &'a [Monitor],
    /// Every screen the system knows right now (`platform::monitors()`).
    pub screens: &'a [Monitor],
    /// The mouse position (`PlatformServices::cursor_position`).
    pub cursor: Option<(f64, f64)>,
    /// The controller's guard flag (`guard_changed`).
    pub guard_active: bool,
    pub state: TrackingState,
    /// `settings.privacy.shoulder_guard`.
    pub shoulder_guard: bool,
    /// `settings.privacy.guard_action`.
    pub guard_action: &'a str,
    /// Seconds on any monotonic clock.
    pub now: f64,
}

/// Where the parts of a curtain window go, in the window's own units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContentLayout {
    /// The lock badge as (x, y, side).
    pub badge: (f64, f64, f64),
    /// The title as (y, height) across the window width.
    pub title: (f64, f64),
    /// The hint as (y, height) across the window width.
    pub hint: (f64, f64),
    /// Top of the dismiss button.
    pub button_top: f64,
}

/// Lock badge, title and hint plus the button top, centred in a `width`×`height` window.
pub fn content_layout(
    width: f64,
    height: f64,
    scale: f64,
    title_h: f64,
    hint_h: f64,
    button_h: f64,
) -> ContentLayout {
    let s = scale;
    let badge = 96.0 * s;
    let block = badge + 28.0 * s + title_h + 10.0 * s + hint_h + 30.0 * s + button_h;
    let top = ((height - block) / 2.0).max(0.0);
    let title_y = top + badge + 28.0 * s;
    let hint_y = title_y + title_h + 10.0 * s;
    ContentLayout {
        badge: ((width - badge) / 2.0, top, badge),
        title: (title_y, title_h),
        hint: (hint_y, hint_h),
        button_top: hint_y + hint_h + 30.0 * s,
    }
}

/// An opaque full-monitor window with a lock, a message and a dismiss button.
#[derive(Debug, Clone, PartialEq)]
pub struct CurtainWindow {
    pub monitor: Monitor,
    pub scale: f64,
    /// What the window tells the user about dismissing it (see
    /// [`set_keyboard_ready`](Self::set_keyboard_ready)).
    pub hint: &'static str,
    /// Whether the system reports keyboard focus in this window.
    pub focused: bool,
}

impl CurtainWindow {
    fn new(monitor: &Monitor, screens: &[Monitor]) -> Self {
        let screen = util::screen_for_monitor(monitor, screens);
        Self {
            monitor: monitor.clone(),
            scale: util::ui_scale(screen.map(|s| s.scale)),
            hint: HINT_NO_KEYBOARD,
            focused: false,
        }
    }

    /// The text assistive technology gets (the Python accessible description).
    pub fn accessible_description(&self) -> String {
        format!("{TITLE}. {}", self.hint)
    }

    /// Promise that Esc works only while a curtain window receives the keyboard.
    pub fn set_keyboard_ready(&mut self, ready: bool) {
        self.hint = if ready { HINT } else { HINT_NO_KEYBOARD };
    }

    /// Whether `key` dismisses the curtain.
    pub fn handles_key(key: Key) -> bool {
        key == Key::Escape
    }

    /// The viewport of the curtain window with this index.
    pub fn viewport_id(index: usize) -> ViewportId {
        ViewportId::from_hash_of(("privacy-curtain", index))
    }

    /// The window: frameless, always on top, opaque, and allowed to take keyboard focus.
    pub fn viewport_builder(&self) -> egui::ViewportBuilder {
        util::place_viewport(
            util::floating_viewport("Privacy curtain", false, true, false),
            self.monitor.rect,
            self.scale,
        )
    }
}

/// Covers every monitor while the shoulder guard is active.
///
/// Reacts to the guard flag, the tracking state and the settings it is given each frame
/// ([`update`](Self::update)); [`show_curtain`](Self::show_curtain) /
/// [`hide_curtain`](Self::hide_curtain) can also be called directly. The user closing it
/// becomes a [`CurtainAction::Dismissed`].
pub struct PrivacyCurtain {
    windows: Vec<CurtainWindow>,
    /// The window that should get keyboard focus (until that is checked).
    focus_target: Option<usize>,
    /// A window to bring to the front on the next frame.
    focus_command: Option<usize>,
    rebuild_pending: bool,
    focus_check_due: Option<f64>,
    hints_due: Option<f64>,
    actions: Vec<CurtainAction>,
    monitors: Vec<Monitor>,
    screens: Vec<Monitor>,
    cursor: Option<(f64, f64)>,
    last_guard: bool,
    last_state: Option<TrackingState>,
    last_settings: Option<(bool, String)>,
    glow: Option<TextureHandle>,
    badge: Option<(u32, TextureHandle)>,
}

impl Default for PrivacyCurtain {
    fn default() -> Self {
        Self::new()
    }
}

impl PrivacyCurtain {
    pub fn new() -> Self {
        Self {
            windows: Vec::new(),
            focus_target: None,
            focus_command: None,
            rebuild_pending: false,
            focus_check_due: None,
            hints_due: None,
            actions: Vec::new(),
            monitors: Vec::new(),
            screens: Vec::new(),
            cursor: None,
            last_guard: false,
            last_state: None,
            last_settings: None,
            glow: None,
            badge: None,
        }
    }

    pub fn is_showing(&self) -> bool {
        !self.windows.is_empty()
    }

    /// The curtain windows currently shown (one per monitor).
    pub fn windows(&self) -> &[CurtainWindow] {
        &self.windows
    }

    /// Whether a curtain window has keyboard focus (so Esc dismisses it).
    pub fn has_keyboard(&self) -> bool {
        self.windows.iter().any(|w| w.focused)
    }

    /// What the owner has to do now (the Python signals and platform calls).
    pub fn drain_actions(&mut self) -> Vec<CurtainAction> {
        std::mem::take(&mut self.actions)
    }

    /// Tell the curtain about the monitor layout, the system's screens and the mouse position it
    /// works from. A change of `screens` while the curtain is up rebuilds it.
    pub fn set_environment(
        &mut self,
        monitors: &[Monitor],
        screens: &[Monitor],
        cursor: Option<(f64, f64)>,
    ) {
        if self.is_showing() && screens != self.screens.as_slice() {
            self.on_screens_changed();
        }
        self.monitors = monitors.to_vec();
        self.screens = screens.to_vec();
        self.cursor = cursor;
    }

    /// Cover every monitor (idempotent).
    pub fn show_curtain(&mut self, now: f64) {
        let monitors = self.curtain_monitors();
        if !self.windows.is_empty() && self.windows.iter().map(|w| &w.monitor).eq(monitors.iter()) {
            return;
        }
        self.destroy_windows();
        self.windows = monitors
            .iter()
            .map(|monitor| CurtainWindow::new(monitor, &self.screens))
            .collect();
        self.focus_window_under_cursor(now);
        log::info!("Privacy curtain shown on {} monitor(s)", self.windows.len());
    }

    /// Remove the curtain (idempotent).
    pub fn hide_curtain(&mut self) {
        if !self.windows.is_empty() {
            log::info!("Privacy curtain hidden");
        }
        self.destroy_windows();
    }

    /// The user closed the curtain.
    pub fn dismiss(&mut self) {
        if self.windows.is_empty() {
            return;
        }
        self.hide_curtain();
        self.actions.push(CurtainAction::Dismissed);
    }

    pub fn close(&mut self) {
        self.destroy_windows();
    }

    /// The controller's monitors plus any screen they do not cover yet.
    ///
    /// The controller rebuilds its layout a moment after a screen change; for privacy every
    /// screen the system knows about right now must be covered.
    fn curtain_monitors(&self) -> Vec<Monitor> {
        let mut monitors = self.monitors.clone();
        let mut covered: Vec<Rect> = monitors.iter().map(|m| m.rect).collect();
        for screen in &self.screens {
            let rect = screen.rect;
            if covered.contains(&rect) || rect.w <= 0 || rect.h <= 0 {
                continue;
            }
            covered.push(rect);
            monitors.push(Monitor {
                index: monitors.len() as i32,
                name: screen.name.clone(),
                rect,
                primary: screen.primary,
                scale: screen.scale,
            });
        }
        monitors
    }

    /// The screens changed: the windows are rebuilt on the next [`tick`](Self::tick) (deferred,
    /// because the system moves windows off a removed screen before this settles).
    pub fn on_screens_changed(&mut self) {
        if !self.windows.is_empty() && !self.rebuild_pending {
            self.rebuild_pending = true;
        }
    }

    fn focus_window_under_cursor(&mut self, now: f64) {
        if self.windows.is_empty() {
            return;
        }
        let monitors: Vec<Monitor> = self.windows.iter().map(|w| w.monitor.clone()).collect();
        let at = self
            .cursor
            .and_then(|(x, y)| monitor_at(&monitors, x, y))
            .and_then(|monitor| self.windows.iter().position(|w| &w.monitor == monitor));
        let target = at.unwrap_or(0);
        // Keyboard focus is what makes Esc work; the button works without it.
        self.focus_command = Some(target);
        self.focus_target = Some(target);
        self.update_hints();
        // Activation is reported asynchronously; a refused one never is.
        self.focus_check_due = Some(now + FOCUS_RECHECK_S);
    }

    /// If the activation was refused, ask the platform layer to activate the curtain.
    pub fn ensure_keyboard(&mut self, now: f64) {
        let target = self.focus_target.take();
        if let Some(index) = target.filter(|&i| i < self.windows.len())
            && !self.has_keyboard()
        {
            // The toolkit's request is a plain SetForegroundWindow on Windows, which the
            // foreground lock refuses while the user works in another app.
            self.actions.push(CurtainAction::ActivateWindow {
                viewport: CurtainWindow::viewport_id(index),
                pid: std::process::id(),
            });
            self.hints_due = Some(now + FOCUS_RECHECK_S);
        }
        self.update_hints();
    }

    /// The system reports that window `index` gained or lost keyboard focus.
    pub fn set_window_focus(&mut self, index: usize, focused: bool) {
        if let Some(window) = self.windows.get_mut(index)
            && window.focused != focused
        {
            window.focused = focused;
            self.update_hints();
        }
    }

    pub fn update_hints(&mut self) {
        let ready = self.has_keyboard();
        for window in &mut self.windows {
            window.set_keyboard_ready(ready);
        }
    }

    fn destroy_windows(&mut self) {
        self.focus_target = None;
        self.focus_command = None;
        self.windows.clear();
    }

    /// The controller's guard flag changed.
    ///
    /// The controller decides when a curtain is due: for the "curtain" action, and as the
    /// fallback when the "lock" action could not lock the screen.
    pub fn on_guard_changed(&mut self, active: bool, now: f64) {
        if active {
            self.show_curtain(now);
        } else {
            self.hide_curtain();
        }
    }

    pub fn on_state_changed(&mut self, state: TrackingState) {
        if camera_off(state) {
            self.hide_curtain();
        }
    }

    pub fn on_settings_changed(&mut self, shoulder_guard: bool, guard_action: &str) {
        // Guard switched off, or switched to notifications only: nothing should keep the
        // screens covered any more.
        if !shoulder_guard || guard_action == "notify" {
            self.hide_curtain();
        }
    }

    /// Run the timers: the deferred rebuild and the two focus checks.
    pub fn tick(&mut self, now: f64) {
        if self.rebuild_pending {
            self.rebuild_pending = false;
            if !self.windows.is_empty() {
                self.show_curtain(now);
            }
        }
        if self.focus_check_due.is_some_and(|due| now >= due) {
            self.focus_check_due = None;
            self.ensure_keyboard(now);
        }
        if self.hints_due.is_some_and(|due| now >= due) {
            self.hints_due = None;
            self.update_hints();
        }
    }

    /// Apply this frame's inputs: what the controller's signals told the Python curtain.
    pub fn update(&mut self, inputs: &CurtainInputs<'_>) {
        self.set_environment(inputs.monitors, inputs.screens, inputs.cursor);
        let settings = (inputs.shoulder_guard, inputs.guard_action.to_string());
        if self.last_settings.as_ref() != Some(&settings) {
            self.on_settings_changed(settings.0, &settings.1);
            self.last_settings = Some(settings);
        }
        if self.last_state != Some(inputs.state) {
            self.last_state = Some(inputs.state);
            self.on_state_changed(inputs.state);
        }
        if self.last_guard != inputs.guard_active {
            self.last_guard = inputs.guard_active;
            self.on_guard_changed(inputs.guard_active, inputs.now);
        }
        self.tick(inputs.now);
    }

    /// Apply `inputs`, draw one viewport per monitor and return what the owner has to do.
    pub fn show(&mut self, ctx: &Context, inputs: &CurtainInputs<'_>) -> Vec<CurtainAction> {
        self.update(inputs);
        let mut dismissed = false;
        let mut focus_changes = Vec::new();
        for index in 0..self.windows.len() {
            let window = self.windows[index].clone();
            let id = CurtainWindow::viewport_id(index);
            let glow = self.glow_texture(ctx);
            let badge = self.badge_texture(ctx, &window);
            let builder = window.viewport_builder();
            let report = ctx.show_viewport_immediate(id, builder.clone(), |ui, _class| {
                util::keep_viewport_placed(ui.ctx(), &builder);
                draw_window(ui, &window, glow.as_ref(), badge.as_ref())
            });
            dismissed |= report.dismiss;
            if report.focused != window.focused {
                focus_changes.push((index, report.focused));
            }
            if self.focus_command == Some(index) {
                self.focus_command = None;
                ctx.send_viewport_cmd_to(id, ViewportCommand::Focus);
            }
        }
        for (index, focused) in focus_changes {
            self.set_window_focus(index, focused);
        }
        if dismissed {
            self.dismiss();
        }
        if self.is_showing() {
            ctx.request_repaint_after(std::time::Duration::from_millis(250));
        }
        self.drain_actions()
    }

    fn glow_texture(&mut self, ctx: &Context) -> Option<TextureHandle> {
        if self.glow.is_none() {
            let image = render_glow(256)?.to_color_image();
            self.glow = Some(ctx.load_texture("curtain-glow", image, TextureOptions::LINEAR));
        }
        self.glow.clone()
    }

    fn badge_texture(&mut self, ctx: &Context, window: &CurtainWindow) -> Option<TextureHandle> {
        let side = (96.0 * window.scale).ceil().max(1.0) as u32;
        if self.badge.as_ref().is_none_or(|(size, _)| *size != side) {
            let image = render_badge(side)?.to_color_image();
            let texture = ctx.load_texture("curtain-badge", image, TextureOptions::LINEAR);
            self.badge = Some((side, texture));
        }
        self.badge.as_ref().map(|(_, texture)| texture.clone())
    }
}

/// A barely visible glow behind the message so the screen does not look dead: the accent colour
/// fading from 26/255 at the centre to nothing at the edge of the `side`×`side` image.
fn render_glow(side: u32) -> Option<Rgba> {
    let mut pixmap = Pixmap::new(side, side)?;
    let half = side as f32 / 2.0;
    let shader = util::radial_gradient(
        (half, half),
        half,
        &[
            (0.0, util::qcolor(util::ACCENT, Some(26))),
            (1.0, util::qcolor(util::ACCENT, Some(0))),
        ],
    );
    let paint = Paint {
        shader,
        anti_alias: true,
        ..Paint::default()
    };
    let rect = tiny_skia::Rect::from_xywh(0.0, 0.0, side as f32, side as f32)?;
    pixmap.fill_rect(rect, &paint, Transform::identity(), None);
    Some(icons::to_rgba(&pixmap))
}

/// The lock badge of `side` pixels: a faint white disc with an accent ring and a lock.
fn render_badge(side: u32) -> Option<Rgba> {
    let mut pixmap = Pixmap::new(side, side)?;
    let s = side as f32;
    let design_scale = s / 96.0;
    let identity = Transform::identity();
    let mut white = Paint {
        anti_alias: true,
        ..Paint::default()
    };
    white.set_color(util::skia_color(util::qcolor("#FFFFFF", Some(14))));
    pixmap.fill_path(
        &PathBuilder::from_circle(s / 2.0, s / 2.0, s / 2.0)?,
        &white,
        FillRule::Winding,
        identity,
        None,
    );
    let ring = Paint {
        shader: util::accent_gradient((0.0, 0.0), (s, s), false),
        anti_alias: true,
        ..Paint::default()
    };
    let style = Stroke {
        width: (2.0 * design_scale).max(1.5),
        ..Stroke::default()
    };
    pixmap.stroke_path(
        &PathBuilder::from_circle(s / 2.0, s / 2.0, s / 2.0 - 1.0)?,
        &ring,
        &style,
        identity,
        None,
    );
    icons::paint_lock(
        &mut pixmap,
        (s * 0.3, s * 0.27, s * 0.4, s * 0.43),
        util::qcolor(util::TEXT, None),
    );
    Some(icons::to_rgba(&pixmap))
}

/// What drawing a curtain window found out.
struct WindowReport {
    dismiss: bool,
    focused: bool,
}

fn draw_window(
    ui: &mut egui::Ui,
    window: &CurtainWindow,
    glow: Option<&TextureHandle>,
    badge: Option<&TextureHandle>,
) -> WindowReport {
    let rect = ui.max_rect();
    let (w, h) = (f64::from(rect.width()), f64::from(rect.height()));
    let painter = ui.painter().clone();
    painter.rect_filled(rect, 0.0, util::color32(util::qcolor(util::CURTAIN, None)));
    if let Some(glow) = glow {
        let radius = rect.width().max(rect.height()) * 0.55;
        let glow_rect =
            egui::Rect::from_center_size(rect.center(), egui::Vec2::splat(2.0 * radius));
        painter.image(glow.id(), glow_rect, unit_uv(), Color32::WHITE);
    }

    let base = egui::TextStyle::Body.resolve(ui.style()).size;
    let title_font = FontId::proportional(base * 2.1);
    let hint_font = FontId::proportional(base * 1.2);
    let button_font = FontId::proportional(base);
    let (title_h, hint_h, button_text_h, button_text_w) = ui.ctx().fonts_mut(|fonts| {
        let galley =
            fonts.layout_no_wrap(BUTTON_TEXT.to_string(), button_font.clone(), Color32::WHITE);
        (
            f64::from(fonts.row_height(&title_font)),
            f64::from(fonts.row_height(&hint_font)),
            f64::from(galley.size().y),
            f64::from(galley.size().x),
        )
    });
    let (button_w, button_h) = (button_text_w + 44.0 + 2.0, button_text_h + 18.0 + 2.0);
    let layout = content_layout(w, h, 1.0, title_h, hint_h, button_h);

    let (bx, by, side) = layout.badge;
    let badge_rect = egui::Rect::from_min_size(
        rect.min + egui::vec2(bx as f32, by as f32),
        egui::Vec2::splat(side as f32),
    );
    if let Some(badge) = badge {
        painter.image(badge.id(), badge_rect, unit_uv(), Color32::WHITE);
    }
    let centre_x = rect.center().x;
    let text = util::color32(util::qcolor(util::TEXT, None));
    let muted = util::color32(util::qcolor(util::TEXT_MUTED, None));
    painter.text(
        egui::pos2(
            centre_x,
            rect.min.y + (layout.title.0 + layout.title.1 / 2.0) as f32,
        ),
        Align2::CENTER_CENTER,
        TITLE,
        title_font,
        text,
    );
    painter.text(
        egui::pos2(
            centre_x,
            rect.min.y + (layout.hint.0 + layout.hint.1 / 2.0) as f32,
        ),
        Align2::CENTER_CENTER,
        window.hint,
        hint_font,
        muted,
    );

    let button_rect = egui::Rect::from_min_size(
        egui::pos2(
            centre_x - button_w as f32 / 2.0,
            rect.min.y + layout.button_top as f32,
        ),
        egui::vec2(button_w as f32, button_h as f32),
    );
    let mut clicked = false;
    ui.scope(|ui| {
        let visuals = &mut ui.style_mut().visuals.widgets;
        let translucent = |alpha: u8| Color32::from_white_alpha(alpha);
        visuals.inactive.weak_bg_fill = translucent(20);
        visuals.hovered.weak_bg_fill = translucent(36);
        visuals.active.weak_bg_fill = translucent(51);
        for state in [
            &mut visuals.inactive,
            &mut visuals.hovered,
            &mut visuals.active,
        ] {
            state.bg_stroke = egui::Stroke::new(1.0, translucent(41));
            state.fg_stroke = egui::Stroke::new(1.0, text);
            state.corner_radius = egui::CornerRadius::same(15);
        }
        let button = egui::Button::new(egui::RichText::new(BUTTON_TEXT).font(button_font.clone()));
        let response = ui
            .put(button_rect, button)
            .on_hover_cursor(egui::CursorIcon::PointingHand);
        clicked = response.clicked();
    });
    let (escape, focused) = ui.input(|i| {
        let pressed = i.events.iter().any(|event| {
            matches!(event, egui::Event::Key { key, pressed: true, .. } if CurtainWindow::handles_key(*key))
        });
        (pressed, i.viewport().focused.unwrap_or(false))
    });
    WindowReport {
        dismiss: clicked || escape,
        focused,
    }
}

fn unit_uv() -> egui::Rect {
    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0))
}

#[cfg(test)]
#[path = "curtain_tests.rs"]
mod tests;
