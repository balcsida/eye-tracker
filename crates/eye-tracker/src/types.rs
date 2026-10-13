//! Shared value types used by the vision, gaze, engine, platform and UI layers.
//!
//! # Coordinate convention
//!
//! Every screen coordinate in this code base is a *global screen coordinate*:
//!
//! * Windows and Linux/X11: the app is per-monitor DPI aware (see
//!   `PlatformServices::prepare_process`) and nothing is scaled, so coordinates are native
//!   physical pixels, identical to Win32 / X11 coordinates.
//! * macOS: coordinates are Cocoa points with a top-left origin, identical to the coordinates
//!   used by Quartz events and the Accessibility API.
//!
//! Because everything agrees, positions can be passed between the UI and the platform layer
//! without conversion.

use std::collections::HashMap;

use thiserror::Error;

use crate::paths::sha1_hex;

/// Returned by the monitor helpers when the list of monitors is empty.
#[derive(Debug, Error, PartialEq, Eq)]
#[error("no monitors")]
pub struct NoMonitors;

/// Axis-aligned rectangle in global screen coordinates (`right`/`bottom` exclusive).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

impl Rect {
    pub const fn new(x: i32, y: i32, w: i32, h: i32) -> Self {
        Self { x, y, w, h }
    }

    pub const fn right(&self) -> i32 {
        self.x + self.w
    }

    pub const fn bottom(&self) -> i32 {
        self.y + self.h
    }

    pub fn center(&self) -> (f64, f64) {
        (
            f64::from(self.x) + f64::from(self.w) / 2.0,
            f64::from(self.y) + f64::from(self.h) / 2.0,
        )
    }

    pub fn diagonal(&self) -> f64 {
        f64::from(self.w).hypot(f64::from(self.h))
    }

    pub fn contains(&self, px: f64, py: f64) -> bool {
        f64::from(self.x) <= px
            && px < f64::from(self.right())
            && f64::from(self.y) <= py
            && py < f64::from(self.bottom())
    }

    /// Euclidean distance from the point to the rectangle (0 when inside).
    pub fn distance_outside(&self, px: f64, py: f64) -> f64 {
        let dx = (f64::from(self.x) - px)
            .max(0.0)
            .max(px - f64::from(self.right()));
        let dy = (f64::from(self.y) - py)
            .max(0.0)
            .max(py - f64::from(self.bottom()));
        dx.hypot(dy)
    }

    /// Nearest integer point that lies inside the rectangle.
    pub fn clamp(&self, px: f64, py: f64) -> (i32, i32) {
        // Python's round() rounds halves to even; so does this.
        let cx = (px.round_ties_even() as i32)
            .max(self.x)
            .min(self.right() - 1);
        let cy = (py.round_ties_even() as i32)
            .max(self.y)
            .min(self.bottom() - 1);
        (cx, cy)
    }

    pub fn normalize(&self, px: f64, py: f64) -> (f64, f64) {
        (
            (px - f64::from(self.x)) / f64::from(self.w),
            (py - f64::from(self.y)) / f64::from(self.h),
        )
    }

    pub fn denormalize(&self, nx: f64, ny: f64) -> (f64, f64) {
        (
            f64::from(self.x) + nx * f64::from(self.w),
            f64::from(self.y) + ny * f64::from(self.h),
        )
    }

    pub fn union(&self, other: &Rect) -> Rect {
        let x0 = self.x.min(other.x);
        let y0 = self.y.min(other.y);
        let x1 = self.right().max(other.right());
        let y1 = self.bottom().max(other.bottom());
        Rect::new(x0, y0, x1 - x0, y1 - y0)
    }

    pub fn to_list(&self) -> [i32; 4] {
        [self.x, self.y, self.w, self.h]
    }

    pub fn from_list(values: [i32; 4]) -> Rect {
        let [x, y, w, h] = values;
        Rect::new(x, y, w, h)
    }
}

/// A physical display.
#[derive(Debug, Clone, PartialEq)]
pub struct Monitor {
    pub index: i32,
    pub name: String,
    pub rect: Rect,
    pub primary: bool,
    pub scale: f64,
}

impl Monitor {
    /// A non-primary monitor with scale 1.0.
    pub fn new(index: i32, name: impl Into<String>, rect: Rect) -> Self {
        Self {
            index,
            name: name.into(),
            rect,
            primary: false,
            scale: 1.0,
        }
    }
}

/// Bounding rectangle of the whole virtual desktop.
pub fn virtual_bounds<'a>(
    monitors: impl IntoIterator<Item = &'a Monitor>,
) -> Result<Rect, NoMonitors> {
    let mut rects = monitors.into_iter().map(|m| m.rect);
    let first = rects.next().ok_or(NoMonitors)?;
    Ok(rects.fold(first, |acc, r| acc.union(&r)))
}

/// Stable fingerprint of a monitor arrangement (order independent).
///
/// A calibration is only valid for the layout it was recorded on. The text hashed is Python's
/// `repr` of the sorted list of rectangle tuples, so signatures stored by the Python app still
/// match.
pub fn layout_signature<'a>(monitors: impl IntoIterator<Item = &'a Monitor>) -> String {
    let mut parts: Vec<[i32; 4]> = monitors.into_iter().map(|m| m.rect.to_list()).collect();
    parts.sort_unstable();
    let items: Vec<String> = parts
        .iter()
        .map(|[x, y, w, h]| format!("({x}, {y}, {w}, {h})"))
        .collect();
    let digest = sha1_hex(format!("[{}]", items.join(", ")).as_bytes());
    digest[..16].to_string()
}

pub fn monitor_at<'a>(
    monitors: impl IntoIterator<Item = &'a Monitor>,
    px: f64,
    py: f64,
) -> Option<&'a Monitor> {
    monitors.into_iter().find(|m| m.rect.contains(px, py))
}

/// Monitor closest to the point and the distance to it (0 when inside).
pub fn nearest_monitor<'a>(
    monitors: impl IntoIterator<Item = &'a Monitor>,
    px: f64,
    py: f64,
) -> Result<(&'a Monitor, f64), NoMonitors> {
    let mut best: Option<(&Monitor, f64)> = None;
    for m in monitors {
        let d = m.rect.distance_outside(px, py);
        if best.is_none_or(|(_, bd)| d < bd) {
            best = Some((m, d));
        }
    }
    best.ok_or(NoMonitors)
}

/// Result of analysing one camera frame.
///
/// `features` is a backend-specific 1-D float vector (see `VisionBackend::feature_names`)
/// describing head pose and eye state of the primary (largest) face. It is `None` when no
/// usable face was found.
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    pub timestamp: f64,
    pub face_count: u32,
    pub features: Option<Vec<f64>>,
    pub quality: f64,
    pub blink: bool,
    pub head_yaw: Option<f64>,
    pub head_pitch: Option<f64>,
    pub face_box: Option<(f64, f64, f64, f64)>,
    pub skipped: bool,
    pub inference_ms: f64,
    pub frame_size: (u32, u32),
    /// The frame is too dark or uniform to judge (lens covered, shutter closed, unlit room).
    /// `face_count` is then meaningless: presence and the shoulder guard must treat the frame
    /// as "cannot tell", not "nobody here".
    pub blind: bool,
}

impl Observation {
    pub fn new(timestamp: f64, face_count: u32) -> Self {
        Self {
            timestamp,
            face_count,
            features: None,
            quality: 0.0,
            blink: false,
            head_yaw: None,
            head_pitch: None,
            face_box: None,
            skipped: false,
            inference_ms: 0.0,
            frame_size: (0, 0),
            blind: false,
        }
    }

    pub fn face_present(&self) -> bool {
        self.face_count > 0 && !self.blind
    }

    /// True when the observation can drive gaze estimation.
    pub fn usable(&self) -> bool {
        self.features.is_some() && !self.blink && self.quality >= 0.3
    }
}

/// Estimated gaze position on the virtual desktop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GazePoint {
    pub x: f64,
    pub y: f64,
    pub timestamp: f64,
}

/// What a [`WindowRef`] points at, in the platform's own terms.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WindowHandle {
    /// An HWND.
    Hwnd(isize),
    /// An X11 window id.
    X11(u32),
    /// A macOS process id and the address of its AXUIElement.
    Ax { pid: i32, element: usize },
}

/// Opaque handle to a top-level window, owned by the platform layer.
///
/// Compare with `PlatformServices::same_window` rather than `==` (so there is no `PartialEq`).
#[derive(Debug, Clone)]
pub struct WindowRef {
    pub handle: WindowHandle,
    pub pid: Option<u32>,
    pub rect: Option<Rect>,
}

/// Which application a window belongs to (`PlatformServices::window_app`).
///
/// `process` is the lower-cased executable name without its extension (`"windowsterminal"`,
/// `"wezterm-gui"`). `app_id` is the window class name on Windows
/// (`"CASCADIA_HOSTING_WINDOW_CLASS"`), the `WM_CLASS` class on X11 and the bundle identifier
/// on macOS; `""` when unknown. Titles are never part of it.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct AppIdentity {
    pub process: String,
    pub app_id: String,
}

/// High-level state shown in the tray and used by the rate scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrackingState {
    Starting,
    NeedsCalibration,
    Calibrating,
    Tracking,
    Paused,
    Privacy,
    Away,
    Locked,
    Yielded,
    CameraError,
}

impl TrackingState {
    /// The Python enum's value.
    pub fn value(self) -> &'static str {
        match self {
            Self::Starting => "starting",
            Self::NeedsCalibration => "needs_calibration",
            Self::Calibrating => "calibrating",
            Self::Tracking => "tracking",
            Self::Paused => "paused",
            Self::Privacy => "privacy",
            Self::Away => "away",
            Self::Locked => "locked",
            Self::Yielded => "yielded",
            Self::CameraError => "camera_error",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Starting => "Starting…",
            Self::NeedsCalibration => "Needs calibration",
            Self::Calibrating => "Calibrating",
            Self::Tracking => "Tracking",
            Self::Paused => "Paused",
            Self::Privacy => "Privacy mode (camera off)",
            Self::Away => "Away",
            Self::Locked => "Screen locked (camera off)",
            Self::Yielded => "Camera in use by another app",
            Self::CameraError => "Camera unavailable",
        }
    }

    /// Whether the camera should be open in this state.
    pub fn camera_active(self) -> bool {
        matches!(
            self,
            Self::Starting
                | Self::NeedsCalibration
                | Self::Calibrating
                | Self::Tracking
                | Self::Away
                | Self::CameraError
        )
    }
}

/// Rolling statistics published by the vision worker.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WorkerStats {
    pub fps: f64,
    pub target_fps: f64,
    pub inference_ms: f64,
    pub skip_ratio: f64,
    pub camera_open: bool,
    pub last_error: Option<String>,
    pub extra: HashMap<String, serde_json::Value>,
}

#[cfg(test)]
#[path = "types_tests.rs"]
mod tests;
