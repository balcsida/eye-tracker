//! Live camera preview with what the tracker sees.
//!
//! Opening the window asks the controller for annotated preview frames
//! ([`PreviewAction::SetPreview`]`(true)`); hiding, minimising or closing it stops them again, so
//! the extra drawing work only happens while someone is watching. Frames are shown here only:
//! they are converted to an egui image in memory and never saved.
//!
//! [`PreviewWindow`] holds the state (what is shown, the lifecycle) and [`PreviewWindow::show`]
//! draws it as a viewport with decorations.

use egui::{ColorImage, Context, TextureHandle, TextureOptions, ViewportId};

use super::icons;
use super::util;
use crate::APP_NAME;
use crate::types::{GazePoint, Monitor, Observation, TrackingState, monitor_at};
use crate::vision::Frame;

pub const NOTE: &str = "Frames are shown only here and are never saved.";
const DASH: &str = "—";
const WAITING: &str = "Waiting for the camera…";

/// (key, caption) of the details panel, in reading order.
const FIELDS: [(&str, &str); 9] = [
    ("state", "State"),
    ("backend", "Backend"),
    ("faces", "Faces"),
    ("pose", "Head yaw / pitch"),
    ("monitor", "Looking at"),
    ("fps", "Camera rate"),
    ("inference", "Analysis time"),
    ("skipped", "Frames skipped"),
    ("cpu", "CPU"),
];

/// What the owner has to tell the controller (what the Python window called on it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreviewAction {
    /// `controller.set_preview(enabled, owner)`: this window is one owner among others (the
    /// setup assistant), so hiding it must not stop frames another consumer still wants.
    SetPreview(bool),
}

/// The statistics the details panel shows (the keys of the controller's `stats_changed` dict).
/// A missing or non-finite number leaves the field as it was.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PreviewStats {
    pub fps: Option<f64>,
    pub target_fps: Option<f64>,
    pub inference_ms: Option<f64>,
    pub skip_ratio: Option<f64>,
    pub cpu_percent: Option<f64>,
    pub backend: Option<String>,
}

/// Everything [`PreviewWindow::show`] reads each frame: the values the Python window got from
/// the controller's signals (`Some` = it arrived since the last frame) and its methods.
#[derive(Debug, Clone, Copy, Default)]
pub struct PreviewInputs<'a> {
    /// The controller's monitor layout.
    pub monitors: &'a [Monitor],
    /// `controller.backend_info()`: backend name and version.
    pub backend_info: Option<(&'a str, &'a str)>,
    pub state: Option<TrackingState>,
    /// `preview_frame`.
    pub frame: Option<&'a Frame>,
    /// `observation`.
    pub observation: Option<&'a Observation>,
    /// `stats_changed`.
    pub stats: Option<&'a PreviewStats>,
    /// `gaze_changed` (`Some(None)`: the gaze was lost).
    pub gaze: Option<Option<GazePoint>>,
}

/// Camera preview plus live details: backend, head pose, faces, rates, gaze monitor.
///
/// Closing only hides it, so the app can keep one instance.
pub struct PreviewWindow {
    preview_on: bool,
    frame_errors: u32,
    monitors: Vec<Monitor>,
    state: TrackingState,
    values: Vec<String>,
    placeholder: String,
    image: Option<ColorImage>,
    image_changed: bool,
    visible: bool,
    minimized: bool,
    backend_info: Option<(String, String)>,
    actions: Vec<PreviewAction>,
    frame_texture: Option<TextureHandle>,
    lock_texture: Option<TextureHandle>,
}

impl PreviewWindow {
    pub fn new(state: TrackingState, backend_info: Option<(&str, &str)>) -> Self {
        let mut window = Self {
            preview_on: false,
            frame_errors: 0,
            monitors: Vec::new(),
            state,
            values: vec![DASH.to_string(); FIELDS.len()],
            placeholder: WAITING.to_string(),
            image: None,
            image_changed: false,
            visible: false,
            minimized: false,
            backend_info: None,
            actions: Vec::new(),
            frame_texture: None,
            lock_texture: None,
        };
        window.set_backend_info(backend_info);
        window.show_backend(None);
        window.set_state(state);
        window
    }

    // ------------------------------------------------------------------ public API
    /// Whether this window has asked the controller for preview frames.
    pub fn preview_active(&self) -> bool {
        self.preview_on
    }

    /// Whether the window is open (not hidden and not minimised).
    pub fn is_shown(&self) -> bool {
        self.visible && !self.minimized
    }

    /// The text of a details field (`"state"`, `"backend"`, `"faces"`, `"pose"`, `"monitor"`,
    /// `"fps"`, `"inference"`, `"skipped"` or `"cpu"`).
    pub fn value(&self, key: &str) -> &str {
        FIELDS
            .iter()
            .position(|(k, _)| *k == key)
            .map_or("", |i| &self.values[i])
    }

    /// The frame being shown, if any.
    pub fn image(&self) -> Option<&ColorImage> {
        self.image.as_ref()
    }

    /// What the frame view says while it has no frame.
    pub fn placeholder(&self) -> &str {
        &self.placeholder
    }

    /// What the owner has to tell the controller now.
    pub fn drain_actions(&mut self) -> Vec<PreviewAction> {
        std::mem::take(&mut self.actions)
    }

    /// Open the window (the Python `showEvent`): it asks for preview frames.
    pub fn open(&mut self, monitors: &[Monitor], backend_info: Option<(&str, &str)>) {
        self.visible = true;
        self.minimized = false;
        self.monitors = monitors.to_vec();
        self.set_backend_info(backend_info);
        self.show_backend(None);
        self.sync_preview();
    }

    /// Hide the window (the Python `hideEvent`; also when it is minimised: no one is watching
    /// then).
    pub fn hide(&mut self) {
        self.visible = false;
        self.sync_preview();
    }

    /// Close the window: stop the frames and drop the last one.
    pub fn close(&mut self) {
        self.visible = false;
        self.sync_preview();
        self.set_image(None);
    }

    /// The window was minimised or restored.
    pub fn set_minimized(&mut self, minimized: bool) {
        self.minimized = minimized;
        self.sync_preview();
    }

    /// Show an annotated BGR frame.
    pub fn set_frame(&mut self, frame: &Frame) {
        if !self.is_shown() {
            return;
        }
        match util::bgr_to_color_image(frame) {
            Ok(image) => self.set_image(Some(image)),
            Err(error) => {
                self.frame_errors += 1;
                if self.frame_errors == 1 {
                    // A broken stream would flood the log.
                    log::warn!("Cannot show preview frame: {error}");
                }
            }
        }
    }

    pub fn set_observation(&mut self, obs: &Observation) {
        self.set("faces", obs.face_count.to_string());
        if let (Some(yaw), Some(pitch)) = (obs.head_yaw, obs.head_pitch) {
            self.set("pose", format!("{yaw:+.0}° / {pitch:+.0}°"));
        } else if !obs.face_present() {
            self.set("pose", DASH.to_string());
        }
    }

    pub fn set_stats(&mut self, stats: &PreviewStats) {
        let finite = |value: Option<f64>| value.filter(|v| v.is_finite());
        let (fps, target) = (finite(stats.fps), finite(stats.target_fps));
        if let Some(fps) = fps {
            let mut text = util::format_fps(fps);
            if let Some(target) = target.filter(|t| *t > 0.0) {
                text.push_str(&format!(" (target {})", util::format_fps(target)));
            }
            self.set("fps", text);
        }
        if let Some(inference) = finite(stats.inference_ms) {
            self.set("inference", format!("{inference:.1} ms"));
        }
        if let Some(skipped) = finite(stats.skip_ratio) {
            self.set("skipped", format!("{:.0} %", skipped * 100.0));
        }
        if let Some(cpu) = finite(stats.cpu_percent) {
            self.set("cpu", format!("{cpu:.1} %"));
        }
        if let Some(backend) = stats.backend.as_deref().filter(|b| !b.is_empty()) {
            self.show_backend(Some(backend));
        }
    }

    /// The gaze moved (`None`: it was lost). `monitors` is the controller's current layout, used
    /// when none was known yet.
    pub fn set_gaze(&mut self, point: Option<GazePoint>, monitors: &[Monitor]) {
        let Some(point) = point else {
            self.set("monitor", DASH.to_string());
            return;
        };
        if self.monitors.is_empty() {
            self.monitors = monitors.to_vec();
        }
        let label = monitor_at(&self.monitors, point.x, point.y)
            .map_or_else(|| "Off-screen".to_string(), util::monitor_label);
        self.set("monitor", label);
    }

    pub fn set_state(&mut self, state: TrackingState) {
        self.state = state;
        self.set("state", state.label().to_string());
        if state.camera_active() {
            self.placeholder = WAITING.to_string();
        } else {
            self.placeholder = format!("Camera is off: {}", state.label().to_lowercase());
            self.set_image(None);
        }
    }

    // ------------------------------------------------------------------ internals
    fn set(&mut self, key: &str, text: String) {
        if let Some(i) = FIELDS.iter().position(|(k, _)| *k == key) {
            self.values[i] = text;
        }
    }

    fn set_image(&mut self, image: Option<ColorImage>) {
        self.image = image;
        self.image_changed = true;
    }

    fn set_backend_info(&mut self, info: Option<(&str, &str)>) {
        if let Some((name, version)) = info {
            self.backend_info = Some((name.to_string(), version.to_string()));
        }
    }

    fn show_backend(&mut self, name: Option<&str>) {
        let (backend, version) = match &self.backend_info {
            Some((backend, version)) => (Some(backend.clone()), version.clone()),
            None => (None, String::new()),
        };
        let name = name.map(str::to_string).or(backend);
        if let Some(name) = name.filter(|n| !n.is_empty()) {
            let text = if version.is_empty() {
                name
            } else {
                format!("{name} ({version})")
            };
            self.set("backend", text);
        }
    }

    /// Ask the controller for frames while the window is open, and stop when it is not.
    fn sync_preview(&mut self) {
        let wanted = self.is_shown();
        if wanted != self.preview_on {
            self.preview_on = wanted;
            self.actions.push(PreviewAction::SetPreview(wanted));
        }
    }

    /// Apply this frame's inputs: what the controller's signals told the Python window.
    pub fn update(&mut self, inputs: &PreviewInputs<'_>) {
        if let Some(state) = inputs.state {
            self.set_state(state);
        }
        if let Some(frame) = inputs.frame {
            self.set_frame(frame);
        }
        if let Some(obs) = inputs.observation {
            self.set_observation(obs);
        }
        if let Some(stats) = inputs.stats {
            self.set_stats(stats);
        }
        if let Some(gaze) = inputs.gaze {
            self.set_gaze(gaze, inputs.monitors);
        }
    }

    /// The preview's window.
    pub fn viewport_id() -> ViewportId {
        ViewportId::from_hash_of("camera-preview")
    }

    /// The window: titled, resizable, opaque, with the app icon.
    pub fn viewport_builder() -> egui::ViewportBuilder {
        let icon = icons::app_icon();
        let builder = egui::ViewportBuilder::default()
            .with_title(format!("Camera preview — {APP_NAME}"))
            .with_inner_size([680.0, 660.0])
            .with_min_inner_size([360.0, 480.0])
            // Under glow a child viewport left unset inherits the transparent root's flag.
            .with_transparent(false);
        match icon.best(256) {
            Some(image) => builder.with_icon(image.to_icon_data()),
            None => builder,
        }
    }

    /// Apply `inputs`, draw the window while it is open and return what the controller has to be
    /// told.
    pub fn show(&mut self, ctx: &Context, inputs: &PreviewInputs<'_>) -> Vec<PreviewAction> {
        self.update(inputs);
        if self.visible {
            self.draw(ctx);
        }
        self.drain_actions()
    }

    fn draw(&mut self, ctx: &Context) {
        let frame_texture = self.frame_texture(ctx);
        let lock = self.lock_texture(ctx);
        let id = Self::viewport_id();
        let mut closed = false;
        let mut minimized = None;
        ctx.show_viewport_immediate(id, Self::viewport_builder(), |ui, _class| {
            let (close_requested, is_minimized) =
                ui.input(|i| (i.viewport().close_requested(), i.viewport().minimized));
            closed = close_requested;
            minimized = is_minimized;
            self.draw_contents(ui, frame_texture.as_ref(), lock.as_ref());
        });
        if closed {
            self.close();
        } else if let Some(minimized) = minimized {
            self.set_minimized(minimized);
        }
    }

    fn frame_texture(&mut self, ctx: &Context) -> Option<TextureHandle> {
        if self.image_changed {
            self.image_changed = false;
            self.frame_texture = self
                .image
                .clone()
                .map(|image| ctx.load_texture("preview-frame", image, TextureOptions::LINEAR));
        }
        self.frame_texture.clone()
    }

    fn lock_texture(&mut self, ctx: &Context) -> Option<TextureHandle> {
        if self.lock_texture.is_none() {
            let lock = icons::render_lock(32, util::qcolor(util::ACCENT, None)).ok()?;
            self.lock_texture = Some(ctx.load_texture(
                "preview-lock",
                lock.to_color_image(),
                TextureOptions::LINEAR,
            ));
        }
        self.lock_texture.clone()
    }

    fn draw_contents(
        &self,
        ui: &mut egui::Ui,
        frame: Option<&TextureHandle>,
        lock: Option<&TextureHandle>,
    ) {
        egui::Panel::bottom("preview-info").show(ui, |ui| {
            let rows = FIELDS.len().div_ceil(2);
            egui::Grid::new("preview-details")
                .num_columns(4)
                .spacing([18.0, 6.0])
                .show(ui, |ui| {
                    for row in 0..rows {
                        for column in 0..2 {
                            let Some(((_, caption), value)) = FIELDS
                                .get(column * rows + row)
                                .zip(self.values.get(column * rows + row))
                            else {
                                continue;
                            };
                            ui.label(
                                egui::RichText::new(*caption).color(util::muted_color(ui, 0.62)),
                            );
                            ui.add(egui::Label::new(value.as_str()).selectable(true));
                        }
                        ui.end_row();
                    }
                });
        });
        egui::CentralPanel::default().show(ui, |ui| {
            ui.horizontal(|ui| {
                if let Some(lock) = lock {
                    ui.add(egui::Image::new((lock.id(), egui::vec2(16.0, 16.0))));
                }
                ui.label(NOTE);
            });
            let size = ui.available_size().max(egui::vec2(320.0, 240.0));
            let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
            self.draw_frame_view(ui, rect, frame);
        });
    }

    /// The frame scaled to fit (aspect kept) on rounded dark ground, or the placeholder text.
    fn draw_frame_view(&self, ui: &egui::Ui, bounds: egui::Rect, frame: Option<&TextureHandle>) {
        let radius = 12.0;
        let painter = ui.painter();
        painter.rect_filled(bounds, radius, util::color32(util::qcolor("#0B0D12", None)));
        match (frame, &self.image) {
            (Some(texture), Some(image)) => {
                let (w, h) = fit_aspect(image.size, (bounds.width(), bounds.height()));
                let target = egui::Rect::from_center_size(bounds.center(), egui::vec2(w, h));
                egui::Image::new((texture.id(), target.size()))
                    .corner_radius(radius)
                    .paint_at(ui, target);
            }
            _ => {
                painter.text(
                    bounds.center(),
                    egui::Align2::CENTER_CENTER,
                    &self.placeholder,
                    egui::TextStyle::Body.resolve(ui.style()),
                    util::color32(util::qcolor(util::TEXT_MUTED, None)),
                );
            }
        }
    }
}

/// `size` scaled to fit inside `bounds` with its aspect ratio kept (it may grow).
pub fn fit_aspect(size: [usize; 2], bounds: (f32, f32)) -> (f32, f32) {
    let (w, h) = (size[0] as f32, size[1] as f32);
    if w <= 0.0 || h <= 0.0 {
        return (0.0, 0.0);
    }
    let factor = (bounds.0 / w).min(bounds.1 / h);
    (w * factor, h * factor)
}

#[cfg(test)]
#[path = "preview_tests.rs"]
mod tests;
