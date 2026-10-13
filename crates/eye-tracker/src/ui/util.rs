//! Shared helpers for the user interface.
//!
//! * DPI: [`ui_scale`] turns a design size in pixels into device pixels for a monitor. On Windows
//!   and Linux/X11 the app is per-monitor DPI aware (so coordinates equal native pixels, see
//!   [`crate::types`]); hand-sized windows must therefore be scaled explicitly, and
//!   [`to_points`] converts native pixels back into the points egui positions windows in.
//!   Text sizes need no scaling: egui draws in points.
//! * Screens: [`screen_for_monitor`] maps the controller's [`Monitor`] objects back to the
//!   operating system's screens (the `platform::monitors()` list).
//! * Palette: the colours of the app's calm dark surfaces and its indigo→cyan accent.
//! * Small, UI-only formatting helpers and the BGR camera frame → egui image conversion.

use std::sync::atomic::{AtomicBool, Ordering};

use egui::{Color32, Pos2, Stroke, StrokeKind, Vec2, ViewportBuilder, WindowLevel};
use thiserror::Error;
use tiny_skia::{
    Color, GradientStop, LinearGradient, Point, RadialGradient, Shader, SpreadMode, Transform,
};

use crate::config::Settings;
use crate::paths;
use crate::platform::hotkeys::{Hotkey, parse_hotkey};
use crate::types::{Monitor, Rect};
use crate::vision::Frame;

// --------------------------------------------------------------------------- palette
/// Accent gradient start (indigo 500) and end (cyan 400).
pub const ACCENT: &str = "#6366F1";
pub const ACCENT_2: &str = "#22D3EE";
/// A lighter indigo for accents drawn on dark backgrounds.
pub const ACCENT_LIGHT: &str = "#818CF8";
/// Deeper accent pair with enough contrast on light backgrounds.
pub const ACCENT_DEEP: &str = "#4F46E5";
pub const ACCENT_2_DEEP: &str = "#0891B2";

/// Dark translucent surface used by toasts and floating panels.
pub const SURFACE: &str = "#12141C";
pub const SURFACE_ALPHA: u8 = 238;
/// Near-black, fully opaque background of the privacy curtain.
pub const CURTAIN: &str = "#07080C";

pub const TEXT: &str = "#F5F7FA";
pub const TEXT_MUTED: &str = "#A0A8B8";

pub const SUCCESS: &str = "#34D399";
pub const WARNING: &str = "#F59E0B";
pub const DANGER: &str = "#EF4444";

/// A straight (not premultiplied) RGBA colour.
pub type Color8 = [u8; 4];

/// A colour from a `#RRGGBB` string with an optional 0..255 alpha (black for invalid text, as
/// an invalid `QColor` paints).
pub fn qcolor(value: &str, alpha: Option<i64>) -> Color8 {
    let hex = value.strip_prefix('#').unwrap_or(value);
    let channel = |i: usize| {
        hex.get(2 * i..2 * i + 2)
            .and_then(|pair| u8::from_str_radix(pair, 16).ok())
    };
    let [r, g, b] = if hex.len() == 6 {
        match (channel(0), channel(1), channel(2)) {
            (Some(r), Some(g), Some(b)) => [r, g, b],
            _ => [0, 0, 0],
        }
    } else {
        [0, 0, 0]
    };
    let a = alpha.map_or(255, |a| a.clamp(0, 255) as u8);
    [r, g, b, a]
}

/// The egui colour of `color`.
pub fn color32(color: Color8) -> Color32 {
    Color32::from_rgba_unmultiplied(color[0], color[1], color[2], color[3])
}

/// The tiny-skia colour of `color`.
pub fn skia_color(color: Color8) -> Color {
    Color::from_rgba8(color[0], color[1], color[2], color[3])
}

/// The indigo→cyan brand gradient between two points (in the coordinates the shape is filled
/// in). `deep` selects the darker pair, which keeps its contrast on light surfaces.
pub fn accent_gradient(start: (f32, f32), end: (f32, f32), deep: bool) -> Shader<'static> {
    let (from, to) = if deep {
        (ACCENT_DEEP, ACCENT_2_DEEP)
    } else {
        (ACCENT, ACCENT_2)
    };
    gradient(
        start,
        end,
        &[(0.0, qcolor(from, None)), (1.0, qcolor(to, None))],
    )
}

/// A linear gradient with the given (position, colour) stops; a solid colour when the two points
/// coincide.
pub fn gradient(start: (f32, f32), end: (f32, f32), stops: &[(f32, Color8)]) -> Shader<'static> {
    let skia_stops = stops
        .iter()
        .map(|&(at, color)| GradientStop::new(at, skia_color(color)))
        .collect();
    LinearGradient::new(
        Point::from_xy(start.0, start.1),
        Point::from_xy(end.0, end.1),
        skia_stops,
        SpreadMode::Pad,
        Transform::identity(),
    )
    .unwrap_or_else(|| Shader::SolidColor(skia_color(stops.last().map_or([0; 4], |s| s.1))))
}

/// A radial gradient from the stops' first colour at `center` to the last at `radius`; a solid
/// colour for a degenerate radius.
pub fn radial_gradient(
    center: (f32, f32),
    radius: f32,
    stops: &[(f32, Color8)],
) -> Shader<'static> {
    let skia_stops = stops
        .iter()
        .map(|&(at, color)| GradientStop::new(at, skia_color(color)))
        .collect();
    let point = Point::from_xy(center.0, center.1);
    RadialGradient::new(
        point,
        point,
        radius,
        skia_stops,
        SpreadMode::Pad,
        Transform::identity(),
    )
    .unwrap_or_else(|| Shader::SolidColor(skia_color(stops.last().map_or([0; 4], |s| s.1))))
}

// ------------------------------------------------------------------------------- DPI
/// Factor from design pixels (96 DPI) to device pixels on a monitor whose scale factor is `scale`
/// (`None`: unknown, the primary screen).
///
/// macOS always returns 1.0: coordinates are points there and the system handles Retina scaling
/// itself. Elsewhere the factor is the monitor's scale (1.5 on a 150 % display), clamped to
/// 0.5..4; a missing, non-finite or non-positive scale gives 1.0.
pub fn ui_scale(scale: Option<f64>) -> f64 {
    if cfg!(target_os = "macos") {
        return 1.0;
    }
    match scale {
        Some(scale) if scale.is_finite() && scale > 0.0 => scale.clamp(0.5, 4.0),
        _ => 1.0,
    }
}

/// Round a design size to whole device pixels (never less than 1).
pub fn scaled(pixels: f64, scale: f64) -> i64 {
    ((pixels * scale).round_ties_even() as i64).max(1)
}

/// Native pixels as the points egui positions and sizes windows in (`scale` is the
/// [`ui_scale`] of the monitor).
pub fn to_points(pixels: f64, scale: f64) -> f32 {
    (pixels / scale) as f32
}

// --------------------------------------------------------------------------- screens
/// The screen that corresponds to a controller [`Monitor`].
///
/// Matching is by exact geometry first (monitor rects are built from the screen list), then by
/// name, then by the screen containing the monitor's centre, then by index.
pub fn screen_for_monitor<'a>(monitor: &Monitor, screens: &'a [Monitor]) -> Option<&'a Monitor> {
    if let Some(screen) = screens.iter().find(|s| s.rect == monitor.rect) {
        return Some(screen);
    }
    if !monitor.name.is_empty()
        && let Some(screen) = screens.iter().find(|s| s.name == monitor.name)
    {
        return Some(screen);
    }
    let (cx, cy) = monitor.rect.center();
    if let Some(screen) = screen_at(screens, cx.trunc(), cy.trunc()) {
        return Some(screen);
    }
    usize::try_from(monitor.index)
        .ok()
        .and_then(|i| screens.get(i))
}

/// The screen containing a global point.
pub fn screen_at(screens: &[Monitor], x: f64, y: f64) -> Option<&Monitor> {
    screens.iter().find(|s| s.rect.contains(x, y))
}

/// Screen containing a global point, else the primary screen.
pub fn screen_for_point(x: f64, y: f64, screens: &[Monitor]) -> Option<&Monitor> {
    screen_at(screens, x.trunc(), y.trunc()).or_else(|| primary_monitor(screens))
}

/// The primary monitor, else the first one, else `None`.
pub fn primary_monitor(monitors: &[Monitor]) -> Option<&Monitor> {
    monitors
        .iter()
        .find(|m| m.primary)
        .or_else(|| monitors.first())
}

/// Human label such as `"Monitor 2 · DELL U2720Q"` (indices shown from 1).
pub fn monitor_label(monitor: &Monitor) -> String {
    let mut label = format!("Monitor {}", monitor.index + 1);
    let name = monitor.name.trim();
    if !name.is_empty() {
        label.push_str(" · ");
        label.push_str(name);
    }
    label
}

// ------------------------------------------------------------------------ formatting
/// `"12 fps"`, `"0.5 fps"` (one decimal only below 1 fps).
pub fn format_fps(fps: f64) -> String {
    if !fps.is_finite() || fps <= 0.0 {
        return "0 fps".to_string();
    }
    if fps < 0.95 {
        return format!("{fps:.1} fps");
    }
    format!("{} fps", fps.round_ties_even() as i64)
}

/// `"CPU 0.6 %"` (share of the whole machine).
pub fn format_cpu(percent: f64) -> String {
    let percent = if percent.is_finite() && percent >= 0.0 {
        percent
    } else {
        0.0
    };
    format!("CPU {percent:.1} %")
}

/// Shorten `text` to at most `limit` characters with a trailing ellipsis.
pub fn elide(text: &str, limit: usize) -> String {
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if text.chars().count() <= limit {
        return text;
    }
    let kept: String = text.chars().take(limit.saturating_sub(1)).collect();
    format!("{}…", kept.trim_end())
}

// ------------------------------------------------------------------------ settings
/// A private copy of the controller's current settings (`current`, as the app read them from
/// it). Falls back to the settings file on disk, which the controller keeps up to date, rather
/// than to defaults: applying defaults would silently reset the user's configuration.
pub fn settings_from_controller(current: Option<&Settings>) -> Settings {
    if let Some(settings) = current {
        return settings.clone();
    }
    log::warn!(
        "Controller exposes no settings; reading {}",
        paths::settings_file().display()
    );
    Settings::load(&paths::settings_file())
}

// --------------------------------------------------------------------------- theming
static APP_PREFERS_DARK: AtomicBool = AtomicBool::new(true);

/// Tell [`system_tray_is_dark`] whether the app's own theme is dark (egui's system theme); the
/// last resort for desktops that offer no better hint. Dark until told otherwise.
pub fn set_app_prefers_dark(dark: bool) {
    APP_PREFERS_DARK.store(dark, Ordering::Relaxed);
}

/// Best guess whether the taskbar / panel behind the tray icon is dark.
///
/// The tray icon is monochrome, so it has to pick a glyph colour. (macOS does not need this: its
/// menu-bar icon is a template image the system tints.)
///
/// * Windows: the taskbar follows `SystemUsesLightTheme`, which is separate from the app theme.
/// * GNOME: the top bar is dark whatever the theme.
/// * otherwise: the app's theme ([`set_app_prefers_dark`]).
///
/// The icons also draw a faint contrasting halo, so a wrong guess still leaves a readable icon.
pub fn system_tray_is_dark() -> bool {
    if cfg!(windows) {
        // A missing value is Windows 10's default dark taskbar.
        return windows_personalize_value("SystemUsesLightTheme") != Some(1);
    }
    if cfg!(target_os = "linux") {
        let desktop = std::env::var("XDG_CURRENT_DESKTOP")
            .unwrap_or_default()
            .to_lowercase();
        if desktop.contains("gnome") || desktop.contains("unity") {
            return true;
        }
    }
    APP_PREFERS_DARK.load(Ordering::Relaxed)
}

/// A DWORD under `HKCU\...\Themes\Personalize`; `None` when unreadable.
#[cfg(windows)]
fn windows_personalize_value(name: &str) -> Option<u32> {
    use std::ffi::c_void;

    use windows::Win32::Foundation::ERROR_SUCCESS;
    use windows::Win32::System::Registry::{HKEY_CURRENT_USER, RRF_RT_REG_DWORD, RegGetValueW};
    use windows::core::{HSTRING, PCWSTR, w};

    let name = HSTRING::from(name);
    let mut value = 0u32;
    let mut size = std::mem::size_of::<u32>() as u32;
    // SAFETY: the key path and `name` are NUL-terminated UTF-16 strings that outlive the call,
    // and `value` is a valid out buffer of the `size` bytes the call is told.
    let status = unsafe {
        RegGetValueW(
            HKEY_CURRENT_USER,
            w!(r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"),
            PCWSTR(name.as_ptr()),
            RRF_RT_REG_DWORD,
            None,
            Some(std::ptr::from_mut(&mut value).cast::<c_void>()),
            Some(&mut size),
        )
    };
    (status == ERROR_SUCCESS).then_some(value)
}

#[cfg(not(windows))]
fn windows_personalize_value(_name: &str) -> Option<u32> {
    None
}

// --------------------------------------------------------------------------- hotkeys
/// A hotkey as the platform's menus display it (Qt's `QKeySequence` for one combination).
///
/// The modifiers use Qt's names: on macOS Qt's `control` is the Command key and `meta` the
/// physical Control key, so [`hotkey_sequence`] swaps our `ctrl` and `meta` there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeySequence {
    pub control: bool,
    pub alt: bool,
    pub shift: bool,
    pub meta: bool,
    /// The key's portable name: `"T"`, `"F5"`, `"Space"`, `"Esc"`.
    pub key: String,
}

impl KeySequence {
    /// Qt's portable text: `"Ctrl+Alt+T"` (modifiers in the order Meta, Ctrl, Alt, Shift).
    pub fn portable_text(&self) -> String {
        let mut parts = Vec::new();
        for (set, name) in [
            (self.meta, "Meta"),
            (self.control, "Ctrl"),
            (self.alt, "Alt"),
            (self.shift, "Shift"),
        ] {
            if set {
                parts.push(name);
            }
        }
        parts.push(&self.key);
        parts.join("+")
    }
}

/// The portable name of a hotkey key (`None` for a key without one).
fn qt_key_name(name: &str) -> Option<String> {
    let mut chars = name.chars();
    let (first, rest) = (chars.next()?, chars.as_str());
    if rest.is_empty() && (first.is_ascii_lowercase() || first.is_ascii_digit()) {
        return Some(first.to_ascii_uppercase().to_string());
    }
    if first == 'f' && !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()) {
        let number: u32 = rest.parse().ok()?;
        return (1..=35).contains(&number).then(|| format!("F{number}"));
    }
    let text = match name {
        "space" => "Space",
        "enter" => "Return",
        "tab" => "Tab",
        "escape" => "Esc",
        "backspace" => "Backspace",
        "insert" => "Ins",
        "delete" => "Del",
        "home" => "Home",
        "end" => "End",
        "pageup" => "PgUp",
        "pagedown" => "PgDown",
        "left" => "Left",
        "right" => "Right",
        "up" => "Up",
        "down" => "Down",
        "minus" => "-",
        "equal" => "=",
        "bracketleft" => "[",
        "bracketright" => "]",
        "backslash" => "\\",
        "semicolon" => ";",
        "quote" => "'",
        "grave" => "`",
        "comma" => ",",
        "period" => ".",
        "slash" => "/",
        _ => return None,
    };
    Some(text.to_string())
}

/// [`parse_hotkey`] that returns `None` instead of failing (empty or invalid text).
pub fn parse_hotkey_text(text: Option<&str>) -> Option<Hotkey> {
    let text = text?;
    if text.trim().is_empty() {
        return None;
    }
    parse_hotkey(text).ok()
}

/// The [`KeySequence`] for a hotkey setting such as `"ctrl+alt+t"` (`None` for empty or invalid
/// text).
pub fn hotkey_sequence_text(text: Option<&str>, macos: Option<bool>) -> Option<KeySequence> {
    hotkey_sequence(&parse_hotkey_text(text)?, macos)
}

/// A [`KeySequence`] for a parsed hotkey.
///
/// Used to *display* global hotkeys next to menu items, where the platform renders them
/// natively. On macOS the system's control modifier is the Command key, so our `ctrl` (the
/// physical Control key) maps to the meta modifier and `meta` (Command) to the control one.
/// `macos = None` follows the running system.
pub fn hotkey_sequence(hotkey: &Hotkey, macos: Option<bool>) -> Option<KeySequence> {
    let key = qt_key_name(hotkey.key())?;
    let mac = macos.unwrap_or(cfg!(target_os = "macos"));
    let modifiers = hotkey.modifiers();
    let (ctrl, meta) = (modifiers.contains("ctrl"), modifiers.contains("meta"));
    Some(KeySequence {
        control: if mac { meta } else { ctrl },
        alt: modifiers.contains("alt"),
        shift: modifiers.contains("shift"),
        meta: if mac { ctrl } else { meta },
        key,
    })
}

// ------------------------------------------------------------------ floating windows
/// A frameless, always-on-top helper window.
///
/// * `click_through`: mouse input passes to whatever is below (gaze dot).
/// * `accept_focus`: the window may take keyboard focus (privacy curtain); otherwise it never
///   activates, so it cannot steal focus from the user's app.
/// * `translucent`: per-pixel alpha background (rounded, see-through corners).
///
/// Kept out of the taskbar and Alt+Tab. Give it a place with [`place_viewport`].
pub fn floating_viewport(
    title: &str,
    click_through: bool,
    accept_focus: bool,
    translucent: bool,
) -> ViewportBuilder {
    ViewportBuilder::default()
        .with_title(title)
        .with_decorations(false)
        .with_window_level(WindowLevel::AlwaysOnTop)
        .with_taskbar(false)
        .with_resizable(false)
        .with_transparent(translucent)
        .with_mouse_passthrough(click_through)
        .with_active(accept_focus)
}

/// Put a window on the native-pixel rectangle `rect` of a monitor whose [`ui_scale`] is `scale`.
pub fn place_viewport(builder: ViewportBuilder, rect: Rect, scale: f64) -> ViewportBuilder {
    builder
        .with_position(Pos2::new(
            to_points(f64::from(rect.x), scale),
            to_points(f64::from(rect.y), scale),
        ))
        .with_inner_size(Vec2::new(
            to_points(f64::from(rect.w), scale),
            to_points(f64::from(rect.h), scale),
        ))
}

/// A softer shade of the theme's text colour: readable on light and dark themes, unlike a
/// disabled widget, and the text stays selectable.
pub fn muted_color(ui: &egui::Ui, opacity: f32) -> Color32 {
    ui.visuals()
        .text_color()
        .gamma_multiply(opacity.clamp(0.0, 1.0))
}

/// Paint the app's dark translucent rounded surface with a hairline border.
pub fn paint_surface(
    painter: &egui::Painter,
    rect: egui::Rect,
    radius: f32,
    color: &str,
    alpha: u8,
    border_alpha: u8,
) {
    let inner = rect.shrink(0.5);
    let fill = color32(qcolor(color, Some(i64::from(alpha))));
    let border = if border_alpha > 0 {
        Stroke::new(1.0, Color32::from_white_alpha(border_alpha))
    } else {
        Stroke::NONE
    };
    painter.rect(inner, radius, fill, border, StrokeKind::Middle);
}

// ------------------------------------------------------------------------ frames
/// Why a camera frame cannot be shown.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum FrameError {
    #[error("empty image")]
    Empty,
    #[error("unsupported image shape ({height}, {width}, 3) for {len} bytes")]
    Shape {
        width: usize,
        height: usize,
        len: usize,
    },
}

/// Convert a BGR camera [`Frame`] to an egui image that owns its pixels.
///
/// Fails for an empty frame or one whose buffer does not hold exactly `width * height` BGR
/// pixels.
pub fn bgr_to_color_image(frame: &Frame) -> Result<egui::ColorImage, FrameError> {
    if frame.width == 0 || frame.height == 0 {
        return Err(FrameError::Empty);
    }
    if frame.data.len() != frame.width * frame.height * 3 {
        return Err(FrameError::Shape {
            width: frame.width,
            height: frame.height,
            len: frame.data.len(),
        });
    }
    let pixels = frame
        .data
        .as_chunks::<3>()
        .0
        .iter()
        .map(|bgr| Color32::from_rgb(bgr[2], bgr[1], bgr[0]))
        .collect();
    Ok(egui::ColorImage::new([frame.width, frame.height], pixels))
}

#[cfg(test)]
#[path = "util_tests.rs"]
mod tests;
