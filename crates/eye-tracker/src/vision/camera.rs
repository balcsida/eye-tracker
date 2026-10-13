//! Frame sources: webcams through nokhwa and image files or frame directories for tests and demos.
//!
//! Every source hands out BGR8 [`Frame`]s that live only in memory. Nothing in this module
//! writes image data anywhere.
//!
//! Sources are used from the vision worker thread. [`FrameSource::open`] may block for a few
//! seconds (some Windows camera drivers are slow to start) but never fails loudly; failures are
//! reported through [`FrameSource::last_error`].
//!
//! Differences from the Python app, which read cameras and videos through OpenCV:
//!
//! * Cameras are opened with nokhwa (Media Foundation on Windows, AVFoundation on macOS, V4L2 on
//!   Linux). It has no DirectShow backend, so the `dshow` API setting only keeps its format
//!   policy (see [`format_preference`]) and the `api` setting selects nothing else.
//! * **Video files are not supported.** A [`FileSource`] plays a still image (png, jpg, bmp, ...)
//!   or a directory of image frames, played in file-name order and looped like a video. There
//!   is no FFmpeg, so no container or codec is ever opened.
//! * A camera is chosen by its index in the list [`list_cameras`] returns or by its name (for
//!   example `FaceTime HD Camera`). On macOS index 0 is often a phone's Continuity Camera, so
//!   settings should store the name.
//!
//! Offline guarantee: [`open_source`] and [`FileSource`] refuse URLs and other protocol
//! specifications, so only local files are ever opened.
//!
//! macOS: nokhwa maps every native format to YUYV and asks AVFoundation to convert to it
//! (`yuvs`), so no planar NV12 buffer reaches the decoder there; NV12 from other platforms is
//! converted by [`decode_frame`] as well.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use thiserror::Error;

use super::Frame;
use crate::panes::worker::monotonic;

/// File suffixes treated as still images by [`FileSource`].
pub const IMAGE_SUFFIXES: [&str; 10] = [
    ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".pgm", ".ppm", ".pbm",
];

// A camera that fails this many reads in a row, or keeps failing for this long, is considered
// gone (unplugged, taken over by another application or its driver crashed). The time bound
// matters for drivers whose reads block for seconds before failing (Media Foundation, V4L2
// select timeouts).
const MAX_READ_FAILURES: u32 = 5;
const MAX_FAILURE_S: f64 = 3.0;
// Reads attempted while opening; the first frames of some cameras are empty.
const WARMUP_READS: usize = 5;
// YUY2 bandwidth (bytes/s) above which DirectShow is asked for MJPG: USB 2 cameras cannot
// deliver more uncompressed, while below it the uncompressed format is cheaper to handle than
// DirectShow's MJPEG decoder.
const DSHOW_MJPG_ABOVE_BYTES_S: f64 = 20e6;
// A video plays at this rate when nothing better is known (frame directories have no rate).
const DEFAULT_PLAYBACK_FPS: f64 = 30.0;

/// A frame source could not be created (for example a missing file).
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{0}")]
pub struct CameraError(pub String);

/// A camera found by [`list_cameras`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CameraInfo {
    pub index: u32,
    pub name: String,
    pub width: usize,
    pub height: usize,
}

/// Anything that produces BGR frames for the vision worker.
pub trait FrameSource {
    /// Start capturing. Returns `false` (and sets `last_error`) on failure.
    fn open(&mut self) -> bool;

    /// The most recent BGR frame, or `None` if none is available.
    fn read(&mut self) -> Option<Frame>;

    /// Stop capturing and free the device. Safe to call more than once.
    fn release(&mut self);

    fn is_open(&self) -> bool;

    fn last_error(&self) -> Option<String>;

    /// The settings string this source was configured with (`"0"`, `"/dev/v4l/by-id/…"`, a file
    /// path); identifies the camera for calibrations. `None` for sources that have none.
    fn device(&self) -> Option<String> {
        None
    }
}

// ------------------------------------------------------------------ capture API

/// A capture API name from the settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureApi {
    Any,
    DShow,
    Msmf,
    AvFoundation,
    V4l2,
}

const API_BY_NAME: [(&str, CaptureApi); 5] = [
    ("any", CaptureApi::Any),
    ("dshow", CaptureApi::DShow),
    ("msmf", CaptureApi::Msmf),
    ("avfoundation", CaptureApi::AvFoundation),
    ("v4l2", CaptureApi::V4l2),
];

/// Map a capture API name from the settings to a [`CaptureApi`].
///
/// `"auto"` picks the API that opens fastest and supports MJPG on each OS: DirectShow on
/// Windows (Media Foundation can take many seconds to open), AVFoundation on macOS and V4L2 on
/// Linux.
pub fn api_preference(api: &str) -> Result<CaptureApi, CameraError> {
    let key = api.trim().to_lowercase();
    if key == "auto" {
        return Ok(if cfg!(windows) {
            CaptureApi::DShow
        } else if cfg!(target_os = "macos") {
            CaptureApi::AvFoundation
        } else if cfg!(target_os = "linux") {
            CaptureApi::V4l2
        } else {
            CaptureApi::Any
        });
    }
    match API_BY_NAME.iter().find(|(name, _)| *name == key) {
        Some((_, value)) => Ok(*value),
        None => {
            let choices = ["auto", "any", "dshow", "msmf", "avfoundation", "v4l2"].join(", ");
            Err(CameraError(format!(
                "Unknown camera API '{api}'; expected one of {choices}"
            )))
        }
    }
}

// ------------------------------------------------------------ pixel formats

/// The formats a camera can deliver and [`decode_frame`] converts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Mjpeg,
    /// Packed 4:2:2, `Y0 U Y1 V` (YUY2).
    Yuyv,
    /// Planar luma followed by interleaved `U V` at half resolution.
    Nv12,
    Gray,
    Rgb,
    Bgr,
}

/// The formats to ask a camera for, best first; the device's own default is the last resort.
///
/// MJPG lets USB 2 cameras deliver 640x480 at full rate, and cameras without it simply keep a
/// native format. It is requested everywhere except where the Python app did not ask for it:
/// DirectShow below USB 2 bandwidth, where uncompressed frames are cheaper than the MJPEG
/// decoder, and AVFoundation, which offers no MJPEG at all (macOS converts to YUYV itself).
pub fn format_preference(
    api: CaptureApi,
    width: usize,
    height: usize,
    fps: u32,
) -> Vec<PixelFormat> {
    use PixelFormat::{Mjpeg, Nv12, Yuyv};
    let avfoundation =
        api == CaptureApi::AvFoundation || (api == CaptureApi::Any && cfg!(target_os = "macos"));
    if avfoundation {
        return vec![Yuyv];
    }
    if api == CaptureApi::DShow
        && (width * height * 2 * fps as usize) as f64 <= DSHOW_MJPG_ABOVE_BYTES_S
    {
        return vec![Yuyv, Nv12, Mjpeg];
    }
    vec![Mjpeg, Yuyv, Nv12]
}

fn clamp_u8(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

/// BT.601 limited-range YUV to `[b, g, r]`, the conversion OpenCV applies to camera YUV.
fn yuv_to_bgr(y: u8, u: u8, v: u8) -> [u8; 3] {
    let c = 298 * (i32::from(y) - 16) + 128;
    let d = i32::from(u) - 128;
    let e = i32::from(v) - 128;
    [
        clamp_u8((c + 516 * d) >> 8),
        clamp_u8((c - 100 * d - 208 * e) >> 8),
        clamp_u8((c + 409 * e) >> 8),
    ]
}

/// The bytes between two rows: drivers may pad rows, in which case the buffer is longer than
/// `rows * min_stride` and a whole number of rows.
fn row_stride(len: usize, rows: usize, min_stride: usize) -> usize {
    if rows > 0 && len.is_multiple_of(rows) && len / rows >= min_stride {
        len / rows
    } else {
        min_stride
    }
}

/// Convert a raw camera buffer into a BGR [`Frame`]; `None` if the buffer is too short or cannot
/// be decoded.
pub fn decode_frame(
    format: PixelFormat,
    width: usize,
    height: usize,
    data: &[u8],
) -> Option<Frame> {
    if width == 0 || height == 0 {
        return None;
    }
    let pixels = width * height;
    let mut out = Vec::with_capacity(pixels * 3);
    match format {
        PixelFormat::Mjpeg => {
            let image = image::load_from_memory_with_format(data, image::ImageFormat::Jpeg).ok()?;
            return Some(Frame::from_rgb_image(&image.to_rgb8()));
        }
        PixelFormat::Yuyv => {
            let stride = row_stride(data.len(), height, width * 2);
            if data.len() < stride * height || !width.is_multiple_of(2) {
                return None;
            }
            for row in data.chunks_exact(stride).take(height) {
                for quad in row[..width * 2].as_chunks::<4>().0 {
                    let (u, v) = (quad[1], quad[3]);
                    out.extend(yuv_to_bgr(quad[0], u, v));
                    out.extend(yuv_to_bgr(quad[2], u, v));
                }
            }
        }
        PixelFormat::Nv12 => {
            let stride = row_stride(data.len(), height + height / 2, width);
            if data.len() < stride * (height + height / 2)
                || !width.is_multiple_of(2)
                || !height.is_multiple_of(2)
            {
                return None;
            }
            let (luma, chroma) = data.split_at(stride * height);
            for y in 0..height {
                let uv = &chroma[(y / 2) * stride..];
                for x in 0..width {
                    let pair = (x / 2) * 2;
                    out.extend(yuv_to_bgr(luma[y * stride + x], uv[pair], uv[pair + 1]));
                }
            }
        }
        PixelFormat::Gray => {
            let stride = row_stride(data.len(), height, width);
            if data.len() < stride * height {
                return None;
            }
            for row in data.chunks_exact(stride).take(height) {
                out.extend(row[..width].iter().flat_map(|&g| [g, g, g]));
            }
        }
        PixelFormat::Rgb | PixelFormat::Bgr => {
            let stride = row_stride(data.len(), height, width * 3);
            if data.len() < stride * height {
                return None;
            }
            for row in data.chunks_exact(stride).take(height) {
                for px in row[..width * 3].as_chunks::<3>().0 {
                    if format == PixelFormat::Rgb {
                        out.extend([px[2], px[1], px[0]]);
                    } else {
                        out.extend(px);
                    }
                }
            }
        }
    }
    Frame::from_bgr(width, height, out)
}

// ---------------------------------------------------------- capture back ends

/// What to ask the driver for when opening a camera.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenRequest {
    pub api: CaptureApi,
    pub width: usize,
    pub height: usize,
    pub fps: u32,
}

/// Which camera to open.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CameraTarget {
    /// Position in the list of [`CaptureBackend::enumerate`].
    Index(u32),
    /// The device name, compared ignoring case.
    Name(String),
}

/// A device found by [`CaptureBackend::enumerate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    pub index: u32,
    pub name: String,
}

/// An open capture stream (the part of `cv2.VideoCapture` the camera uses).
pub trait Capture {
    /// Discard the oldest queued frame. `false` if none arrived.
    fn grab(&mut self) -> bool;

    /// The next frame, decoded to BGR.
    fn read(&mut self) -> Option<Frame>;

    /// Stop the stream.
    fn release(&mut self);

    /// Backend, size and format for the debug log.
    fn describe(&self) -> String;
}

/// The camera API underneath [`Camera`] (replaceable in tests).
pub trait CaptureBackend {
    /// The cameras the system lists.
    fn enumerate(&self) -> Result<Vec<DeviceInfo>, String>;

    /// Open a camera and start its stream.
    fn open(
        &self,
        target: &CameraTarget,
        request: &OpenRequest,
    ) -> Result<Box<dyn Capture>, String>;
}

/// The backend the app uses: nokhwa.
pub fn default_backend() -> Arc<dyn CaptureBackend> {
    Arc::new(nokhwa_backend::NokhwaBackend)
}

mod nokhwa_backend {
    use std::time::Duration;

    use nokhwa::utils::{
        ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType,
        Resolution,
    };

    use super::{
        CameraTarget, Capture, CaptureBackend, DeviceInfo, Frame, OpenRequest, PixelFormat,
        decode_frame, format_preference,
    };

    const ALL_FORMATS: [FrameFormat; 6] = [
        FrameFormat::MJPEG,
        FrameFormat::YUYV,
        FrameFormat::NV12,
        FrameFormat::GRAY,
        FrameFormat::RAWRGB,
        FrameFormat::RAWBGR,
    ];
    // How long to wait for the user to answer the macOS camera permission prompt.
    const ACCESS_PROMPT_TIMEOUT: Duration = Duration::from_secs(30);

    impl PixelFormat {
        fn to_nokhwa(self) -> FrameFormat {
            match self {
                PixelFormat::Mjpeg => FrameFormat::MJPEG,
                PixelFormat::Yuyv => FrameFormat::YUYV,
                PixelFormat::Nv12 => FrameFormat::NV12,
                PixelFormat::Gray => FrameFormat::GRAY,
                PixelFormat::Rgb => FrameFormat::RAWRGB,
                PixelFormat::Bgr => FrameFormat::RAWBGR,
            }
        }

        fn from_nokhwa(format: FrameFormat) -> Self {
            match format {
                FrameFormat::MJPEG => PixelFormat::Mjpeg,
                FrameFormat::YUYV => PixelFormat::Yuyv,
                FrameFormat::NV12 => PixelFormat::Nv12,
                FrameFormat::GRAY => PixelFormat::Gray,
                FrameFormat::RAWRGB => PixelFormat::Rgb,
                FrameFormat::RAWBGR => PixelFormat::Bgr,
            }
        }
    }

    /// Ask for camera access (a prompt on macOS, a no-op elsewhere) and report whether it is
    /// granted.
    fn ensure_access() -> bool {
        if nokhwa::nokhwa_check() {
            return true;
        }
        let (tx, rx) = crossbeam_channel::bounded(1);
        nokhwa::nokhwa_initialize(move |granted| {
            let _ = tx.try_send(granted);
        });
        rx.recv_timeout(ACCESS_PROMPT_TIMEOUT).unwrap_or(false)
    }

    pub struct NokhwaBackend;

    impl NokhwaBackend {
        fn query() -> Result<Vec<(u32, String)>, String> {
            if !ensure_access() {
                return Err("camera access was not granted".into());
            }
            let infos = nokhwa::query(ApiBackend::Auto).map_err(|e| e.to_string())?;
            Ok(infos
                .iter()
                .filter_map(|info| Some((info.index().as_index().ok()?, info.human_name())))
                .collect())
        }

        fn resolve(target: &CameraTarget) -> Result<u32, String> {
            match target {
                CameraTarget::Index(index) => Ok(*index),
                CameraTarget::Name(name) => Self::query()?
                    .into_iter()
                    .find(|(_, found)| found.eq_ignore_ascii_case(name))
                    .map(|(index, _)| index)
                    .ok_or_else(|| format!("no camera is named {name}")),
            }
        }
    }

    impl CaptureBackend for NokhwaBackend {
        fn enumerate(&self) -> Result<Vec<DeviceInfo>, String> {
            Ok(Self::query()?
                .into_iter()
                .map(|(index, name)| DeviceInfo { index, name })
                .collect())
        }

        fn open(
            &self,
            target: &CameraTarget,
            request: &OpenRequest,
        ) -> Result<Box<dyn Capture>, String> {
            let index = CameraIndex::Index(Self::resolve(target)?);
            let size = Resolution::new(request.width as u32, request.height as u32);
            let mut last_error = String::from("the camera offers no usable format");
            let mut camera = None;
            // Closest() only matches the exact format asked for, so each candidate is tried in
            // turn; the device's first format is the last resort ("keep the default format").
            // AVFoundation only accepts a frame rate that is the top of one of the device's rate
            // ranges (30 fps for most webcams), so 15 fps falls back to the usual rates.
            let mut rates = vec![request.fps];
            rates.extend(
                [30, 60, 25, 24, 20, 15, 10, 5]
                    .into_iter()
                    .filter(|r| *r != request.fps),
            );
            'formats: for format in
                format_preference(request.api, request.width, request.height, request.fps)
            {
                let wanted = [format.to_nokhwa()];
                for &rate in &rates {
                    let closest = CameraFormat::new(size, format.to_nokhwa(), rate);
                    let asked = RequestedFormat::with_formats(
                        RequestedFormatType::Closest(closest),
                        &wanted,
                    );
                    match nokhwa::Camera::new(index.clone(), asked) {
                        Ok(found) => {
                            camera = Some(found);
                            break 'formats;
                        }
                        Err(e) => last_error = e.to_string(),
                    }
                }
            }
            if camera.is_none() {
                let asked = RequestedFormat::with_formats(RequestedFormatType::None, &ALL_FORMATS);
                match nokhwa::Camera::new(index, asked) {
                    Ok(found) => camera = Some(found),
                    Err(e) => last_error = e.to_string(),
                }
            }
            let mut camera = camera.ok_or(last_error)?;
            camera.open_stream().map_err(|e| e.to_string())?;
            Ok(Box::new(NokhwaCapture { camera }))
        }
    }

    struct NokhwaCapture {
        camera: nokhwa::Camera,
    }

    impl Capture for NokhwaCapture {
        fn grab(&mut self) -> bool {
            self.camera.frame().is_ok()
        }

        fn read(&mut self) -> Option<Frame> {
            let buffer = self.camera.frame().ok()?;
            let size = buffer.resolution();
            decode_frame(
                PixelFormat::from_nokhwa(buffer.source_frame_format()),
                size.width() as usize,
                size.height() as usize,
                buffer.buffer(),
            )
        }

        fn release(&mut self) {
            if let Err(e) = self.camera.stop_stream() {
                log::debug!("Stopping the camera stream failed: {e}");
            }
        }

        fn describe(&self) -> String {
            let format = self.camera.camera_format();
            format!("{:?} {format}", self.camera.backend())
        }
    }
}

// ------------------------------------------------------------------- Camera

/// What [`Camera::new`] asks the driver for.
#[derive(Debug, Clone)]
pub struct CameraOptions {
    /// Requested capture size. Small sizes keep USB bandwidth and decoding cost low; the
    /// backends downscale further anyway.
    pub width: usize,
    pub height: usize,
    /// Capture API name, see [`api_preference`].
    pub api: String,
    /// Requested camera frame rate. Some drivers convert every captured frame, so asking for
    /// fewer frames saves CPU even between reads.
    pub fps: u32,
    /// Buffered frames to drop before a read that follows a pause, so that a slow reader never
    /// analyses a stale picture.
    pub flush: u32,
}

impl Default for CameraOptions {
    fn default() -> Self {
        Self {
            width: 640,
            height: 480,
            api: "auto".into(),
            fps: 15,
            flush: 1,
        }
    }
}

/// A webcam opened through a [`CaptureBackend`].
pub struct Camera {
    pub target: CameraTarget,
    pub width: usize,
    pub height: usize,
    pub api: String,
    pub fps: u32,
    pub flush: u32,
    /// The settings string this camera was configured with (`"0"`, `"/dev/v4l/by-id/…"`);
    /// identifies the camera for calibrations.
    pub device: String,
    api_pref: CaptureApi,
    backend: Arc<dyn CaptureBackend>,
    capture: Option<Box<dyn Capture>>,
    error: Option<String>,
    failures: u32,
    fail_since: Option<f64>,
    last_read: f64,
    frame_size: Option<(usize, usize)>,
    clock: Box<dyn Fn() -> f64>,
}

impl Camera {
    pub fn new(target: CameraTarget, options: CameraOptions) -> Result<Self, CameraError> {
        let device = match &target {
            CameraTarget::Index(index) => index.to_string(),
            CameraTarget::Name(name) => name.clone(),
        };
        Ok(Self {
            api_pref: api_preference(&options.api)?,
            target,
            width: options.width,
            height: options.height,
            api: options.api,
            fps: options.fps.max(1),
            flush: options.flush,
            device,
            backend: default_backend(),
            capture: None,
            error: None,
            failures: 0,
            fail_since: None,
            last_read: f64::NEG_INFINITY,
            frame_size: None,
            clock: Box::new(monotonic),
        })
    }

    /// Use another camera API (tests).
    pub fn with_backend(mut self, backend: Arc<dyn CaptureBackend>) -> Self {
        self.backend = backend;
        self
    }

    /// `"Camera 2"`, or the device name when the camera was chosen by name.
    pub fn name(&self) -> String {
        match &self.target {
            CameraTarget::Index(index) => format!("Camera {index}"),
            CameraTarget::Name(name) => name.clone(),
        }
    }

    /// `(width, height)` of the delivered frames once the camera is open.
    pub fn frame_size(&self) -> Option<(usize, usize)> {
        self.frame_size
    }

    fn request(&self) -> OpenRequest {
        OpenRequest {
            api: self.api_pref,
            width: self.width,
            height: self.height,
            fps: self.fps,
        }
    }
}

impl FrameSource for Camera {
    fn open(&mut self) -> bool {
        if self.capture.is_some() {
            return true;
        }
        self.failures = 0;
        self.fail_since = None;
        let started = Instant::now();
        // The OS privacy switches (Windows "Let desktop apps access your camera", macOS camera
        // access) make opening fail exactly like a busy camera.
        let unavailable = format!(
            "{} could not be opened — it may be in use by another app or blocked by the \
             system's camera privacy settings",
            self.name()
        );
        let mut capture = match self.backend.open(&self.target, &self.request()) {
            Ok(capture) => capture,
            Err(e) => {
                log::debug!(
                    "{} did not open ({:.0} ms): {e}",
                    self.name(),
                    started.elapsed().as_secs_f64() * 1e3
                );
                self.error = Some(unavailable);
                return false;
            }
        };

        // Some drivers report success on open but never deliver a frame when another
        // application holds the device, so only a real frame counts as success.
        let frame = (0..WARMUP_READS).find_map(|_| capture.read());
        let Some(frame) = frame else {
            capture.release();
            self.error = Some(format!(
                "{} opened but delivers no frames — it may be in use by another app or \
                 blocked by the system's camera privacy settings",
                self.name()
            ));
            log::debug!("{} delivered no frames", self.name());
            return false;
        };

        self.error = None;
        self.last_read = (self.clock)();
        self.frame_size = Some((frame.width, frame.height));
        log::debug!(
            "{} opened via {} at {}x{} in {:.0} ms",
            self.name(),
            capture.describe(),
            frame.width,
            frame.height,
            started.elapsed().as_secs_f64() * 1e3
        );
        self.capture = Some(capture);
        true
    }

    fn read(&mut self) -> Option<Frame> {
        let capture = self.capture.as_mut()?;
        // Drop buffered frames only after a pause longer than ~1.5 frame periods. When reading
        // at camera speed nothing stale can be queued, and grabbing anyway would block for an
        // extra frame period and halve the frame rate. Never while failing: a stalled driver
        // would block once per grab.
        let mut flushed = true;
        let paused = (self.clock)() - self.last_read > 1.5 / f64::from(self.fps);
        if self.flush > 0 && paused && self.failures == 0 {
            for _ in 0..self.flush {
                if !capture.grab() {
                    flushed = false;
                    break;
                }
            }
        }
        let frame = if flushed { capture.read() } else { None };
        let now = (self.clock)();
        self.last_read = now;

        if frame.is_none() {
            self.failures += 1;
            let since = *self.fail_since.get_or_insert(now);
            if self.failures >= MAX_READ_FAILURES || now - since >= MAX_FAILURE_S {
                self.error = Some(format!(
                    "{} stopped delivering frames — was it unplugged or taken by another app?",
                    self.name()
                ));
                log::debug!("{} stopped delivering frames; closing it", self.name());
                self.release();
            }
            return None;
        }
        self.failures = 0;
        self.fail_since = None;
        frame
    }

    fn release(&mut self) {
        let capture = self.capture.take();
        self.failures = 0;
        self.fail_since = None;
        if let Some(mut capture) = capture {
            capture.release();
            log::debug!("{} released", self.name());
        }
    }

    fn is_open(&self) -> bool {
        self.capture.is_some()
    }

    fn last_error(&self) -> Option<String> {
        self.error.clone()
    }

    fn device(&self) -> Option<String> {
        Some(self.device.clone())
    }
}

impl std::fmt::Debug for Camera {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Camera(target={:?}, api={:?}, open={})",
            self.target,
            self.api,
            self.is_open()
        )
    }
}

// ----------------------------------------------------------------- FileSource

fn has_image_suffix(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| IMAGE_SUFFIXES.contains(&format!(".{}", ext.to_lowercase()).as_str()))
}

fn file_name(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    )
}

/// Decode an image file. Reading the bytes ourselves keeps non-ASCII paths working everywhere.
fn imread(path: &Path) -> Option<Frame> {
    let data = match std::fs::read(path) {
        Ok(data) if !data.is_empty() => data,
        Ok(_) => return None,
        Err(e) => {
            log::debug!("Could not read {}: {e}", path.display());
            return None;
        }
    };
    let image = image::load_from_memory(&data).ok()?;
    Some(Frame::from_rgb_image(&image.to_rgb8()))
}

/// Frames from a still image or from a directory of image frames (looping forever).
///
/// Intended for tests, demos and benchmarks without a camera. A directory plays its images in
/// file-name order. Video files are not supported (see the module docs).
pub struct FileSource {
    pub path: PathBuf,
    /// Optional bounding box. Larger frames are scaled down to fit inside it (aspect ratio
    /// preserved, never enlarged), mimicking the capture size a camera would deliver.
    pub width: Option<usize>,
    pub height: Option<usize>,
    /// For frame directories, advance at [`video_fps`](Self::video_fps) (like a live camera:
    /// slow readers skip frames). When `false` every `read()` returns the next frame.
    pub realtime: bool,
    /// The settings string this source was configured with.
    pub device: String,
    /// Playback rate of a frame directory in realtime mode.
    pub video_fps: f64,
    image: Option<Frame>,
    frames: Vec<PathBuf>,
    position: usize,
    playing: bool,
    error: Option<String>,
    last_frame: Option<Frame>,
    t_prev: Option<f64>,
    debt: f64,
    // Replaceable for deterministic tests of realtime playback.
    now: Box<dyn Fn() -> f64>,
}

impl FileSource {
    pub fn new(
        path: impl Into<PathBuf>,
        width: Option<usize>,
        height: Option<usize>,
        realtime: bool,
    ) -> Self {
        let path = path.into();
        Self {
            device: path.display().to_string(),
            path,
            width,
            height,
            realtime,
            video_fps: DEFAULT_PLAYBACK_FPS,
            image: None,
            frames: Vec::new(),
            position: 0,
            playing: false,
            error: None,
            last_frame: None,
            t_prev: None,
            debt: 0.0,
            now: Box::new(monotonic),
        }
    }

    pub fn is_image(&self) -> bool {
        has_image_suffix(&self.path)
    }

    fn frame_files(&self) -> Vec<PathBuf> {
        let Ok(entries) = std::fs::read_dir(&self.path) else {
            return Vec::new();
        };
        let mut files: Vec<PathBuf> = entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.is_file() && has_image_suffix(path))
            .collect();
        files.sort_by_key(|path| file_name(path));
        files
    }

    /// Advance one frame, looping at the end. Returns the frame if decoded.
    fn step(&mut self, decode: bool) -> Option<Frame> {
        if !self.playing {
            return None;
        }
        if !decode {
            self.position = (self.position + 1) % self.frames.len();
            return None;
        }
        for _ in 0..self.frames.len() {
            let path = &self.frames[self.position];
            self.position = (self.position + 1) % self.frames.len();
            if let Some(frame) = imread(path) {
                self.last_frame = Some(frame.clone());
                return Some(frame);
            }
        }
        self.error = Some(format!(
            "Frames of {} stopped delivering images",
            file_name(&self.path)
        ));
        self.release();
        None
    }

    fn read_realtime(&mut self) -> Option<Frame> {
        let now = (self.now)();
        let (Some(t_prev), Some(last)) = (self.t_prev, &self.last_frame) else {
            self.t_prev = Some(now);
            self.debt = 0.0;
            return self.step(true);
        };
        self.debt += (now - t_prev) * self.video_fps;
        self.t_prev = Some(now);
        let steps = self.debt as usize;
        self.debt -= steps as f64;
        if steps == 0 {
            return Some(last.clone());
        }
        // After a long pause jump ahead a little instead of decoding a backlog.
        let steps = steps.min(30);
        for _ in 0..steps - 1 {
            self.step(false);
            if !self.playing {
                return None;
            }
        }
        self.step(true)
    }

    fn fit(&self, frame: Frame) -> Frame {
        let mut scale: f64 = 1.0;
        if let Some(width) = self.width {
            scale = scale.min(width as f64 / frame.width as f64);
        }
        if let Some(height) = self.height {
            scale = scale.min(height as f64 / frame.height as f64);
        }
        if scale >= 1.0 {
            return frame;
        }
        let size = |side: usize| ((side as f64 * scale).round_ties_even() as u32).max(1);
        let resized = image::imageops::resize(
            &frame.to_rgb_image(),
            size(frame.width),
            size(frame.height),
            image::imageops::FilterType::Triangle,
        );
        Frame::from_rgb_image(&resized)
    }
}

impl FrameSource for FileSource {
    fn open(&mut self) -> bool {
        if self.is_open() {
            return true;
        }
        if url_like(&self.path.to_string_lossy()) {
            self.error = Some(format!(
                "Not a local file: {} (network sources are not supported)",
                self.path.display()
            ));
            return false;
        }
        let name = file_name(&self.path);
        if self.path.is_dir() {
            let files = self.frame_files();
            let Some(first) = files.first() else {
                self.error = Some(format!("Directory {name} contains no image frames"));
                return false;
            };
            if imread(first).is_none() {
                self.error = Some(format!("Directory {name} contains no readable frames"));
                return false;
            }
            self.frames = files;
            self.position = 0;
            self.playing = true;
            self.last_frame = None;
            self.t_prev = None;
            self.debt = 0.0;
            self.error = None;
            return true;
        }
        if !self.path.is_file() {
            self.error = Some(format!("Video source not found: {}", self.path.display()));
            return false;
        }
        if !self.is_image() {
            self.error = Some(format!(
                "Could not open {name}: video files are not supported \
                 (use an image or a directory of image frames)"
            ));
            return false;
        }
        let Some(image) = imread(&self.path) else {
            self.error = Some(format!("Could not decode image {name}"));
            return false;
        };
        self.image = Some(self.fit(image));
        self.error = None;
        true
    }

    fn read(&mut self) -> Option<Frame> {
        if let Some(image) = &self.image {
            // A copy, so a consumer drawing on the frame cannot alter later reads.
            return Some(image.clone());
        }
        if !self.playing {
            return None;
        }
        let frame = if self.realtime {
            self.read_realtime()
        } else {
            self.step(true)
        }?;
        Some(self.fit(frame))
    }

    fn release(&mut self) {
        self.playing = false;
        self.image = None;
        self.last_frame = None;
    }

    fn is_open(&self) -> bool {
        self.image.is_some() || self.playing
    }

    fn last_error(&self) -> Option<String> {
        self.error.clone()
    }

    fn device(&self) -> Option<String> {
        Some(self.device.clone())
    }
}

impl std::fmt::Debug for FileSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FileSource({:?}, open={})",
            file_name(&self.path),
            self.is_open()
        )
    }
}

// ---------------------------------------------------------------- open_source

/// What FFmpeg would open as a protocol rather than a file (see its `url_find_protocol`):
/// `<scheme>:…` where the scheme is letters, digits, `+`, `-` or `.` (`rtsp://…`, `http:…`,
/// `tcp:…`, `concat:…`), except a single drive letter (`C:\…`); `subfile,…:` wraps another
/// protocol; and `://` anywhere catches nested URLs.
///
/// FFmpeg is gone, but the settings keep refusing everything that is not a local path.
fn url_like(spec: &str) -> bool {
    let lower = spec.to_lowercase();
    if lower.starts_with("subfile,") || lower.contains("://") {
        return true;
    }
    let scheme_len = lower
        .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '+' | '.' | '-')))
        .unwrap_or(lower.len());
    scheme_len >= 2 && lower[scheme_len..].starts_with(':')
}

/// The operating-system facts [`open_source`] looks at (replaceable in tests).
pub trait DeviceFs {
    fn exists(&self, path: &str) -> bool;
    fn is_file(&self, path: &str) -> bool;
    /// `os.path.realpath`
    fn resolve_link(&self, path: &str) -> String;
}

struct RealFs;

impl DeviceFs for RealFs {
    fn exists(&self, path: &str) -> bool {
        Path::new(path).exists()
    }

    fn is_file(&self, path: &str) -> bool {
        Path::new(path).is_file()
    }

    fn resolve_link(&self, path: &str) -> String {
        // Like realpath, a link to a missing node still resolves (to the missing node).
        let resolved = std::fs::canonicalize(path).or_else(|_| {
            let target = std::fs::read_link(path)?;
            Ok::<_, std::io::Error>(
                Path::new(path)
                    .parent()
                    .unwrap_or(Path::new("/"))
                    .join(target),
            )
        });
        resolved.map_or_else(|_| path.to_string(), |p| p.to_string_lossy().into_owned())
    }
}

/// Everything [`open_source_in`] depends on besides its arguments.
pub struct SourceEnv {
    pub backend: Arc<dyn CaptureBackend>,
    pub fs: Box<dyn DeviceFs>,
    pub linux: bool,
}

impl SourceEnv {
    pub fn system() -> Self {
        Self {
            backend: default_backend(),
            fs: Box::new(RealFs),
            linux: cfg!(target_os = "linux"),
        }
    }
}

/// A source created by [`open_source`].
#[derive(Debug)]
pub enum Source {
    Camera(Camera),
    File(FileSource),
}

impl Source {
    fn inner(&mut self) -> &mut dyn FrameSource {
        match self {
            Source::Camera(source) => source,
            Source::File(source) => source,
        }
    }

    fn inner_ref(&self) -> &dyn FrameSource {
        match self {
            Source::Camera(source) => source,
            Source::File(source) => source,
        }
    }
}

impl FrameSource for Source {
    fn open(&mut self) -> bool {
        self.inner().open()
    }

    fn read(&mut self) -> Option<Frame> {
        self.inner().read()
    }

    fn release(&mut self) {
        self.inner().release();
    }

    fn is_open(&self) -> bool {
        self.inner_ref().is_open()
    }

    fn last_error(&self) -> Option<String> {
        self.inner_ref().last_error()
    }

    fn device(&self) -> Option<String> {
        self.inner_ref().device()
    }
}

fn expand_user(spec: &str) -> PathBuf {
    let rest = spec
        .strip_prefix("~/")
        .or_else(|| spec.strip_prefix("~\\"))
        .or_else(|| (spec == "~").then_some(""));
    match (rest, directories::BaseDirs::new()) {
        (Some(rest), Some(dirs)) => dirs.home_dir().join(rest),
        _ => PathBuf::from(spec),
    }
}

/// Create (but do not open) the frame source described by a settings string.
///
/// `"0"`, `"1"`, … select a camera by index; a camera name (`FaceTime HD Camera`) selects it
/// by name. On Linux `/dev/videoN` and stable links to it (`/dev/v4l/by-id/…`,
/// `/dev/v4l/by-path/…`) are accepted too; links are resolved on every call, so a camera that
/// was re-plugged under another number is still found. Anything else is a path to a local image
/// or a directory of image frames. URLs and other protocols are refused: Eye Tracker never
/// opens network streams.
///
/// Fails with a [`CameraError`] when the file does not exist, is a URL, or the settings are
/// invalid.
pub fn open_source(
    device: &str,
    width: usize,
    height: usize,
    api: &str,
    fps: u32,
    realtime: bool,
) -> Result<Source, CameraError> {
    open_source_in(
        &SourceEnv::system(),
        device,
        width,
        height,
        api,
        fps,
        realtime,
    )
}

/// [`open_source`] against another environment.
pub fn open_source_in(
    env: &SourceEnv,
    device: &str,
    width: usize,
    height: usize,
    api: &str,
    fps: u32,
    realtime: bool,
) -> Result<Source, CameraError> {
    let spec = device.trim();
    if spec.is_empty() {
        return Err(CameraError("No camera configured".into()));
    }
    if url_like(spec) {
        return Err(CameraError(format!(
            "Network or protocol video sources are not supported ('{spec}'); \
             use a camera index or a local file"
        )));
    }
    let options = CameraOptions {
        width,
        height,
        api: api.to_string(),
        fps,
        ..CameraOptions::default()
    };
    let camera = |target: CameraTarget| -> Result<Source, CameraError> {
        let mut camera = Camera::new(target, options.clone())?.with_backend(env.backend.clone());
        camera.device = spec.to_string();
        Ok(Source::Camera(camera))
    };

    let mut index = None;
    if !spec.is_empty() && spec.bytes().all(|b| b.is_ascii_digit()) {
        index = spec.parse::<u32>().ok();
    } else if env.linux && spec.starts_with("/dev/") && !env.fs.is_file(spec) {
        index = Some(v4l2_index(spec, env.fs.as_ref())?);
    }
    if let Some(index) = index {
        return camera(CameraTarget::Index(index));
    }

    let path = expand_user(spec);
    if path.is_file() || path.is_dir() {
        let mut source = FileSource::new(path, Some(width), Some(height), realtime);
        source.device = spec.to_string();
        return Ok(Source::File(source));
    }
    if !spec.contains(['/', '\\']) {
        let known = env.backend.enumerate().unwrap_or_default();
        if let Some(found) = known.iter().find(|d| d.name.eq_ignore_ascii_case(spec)) {
            return camera(CameraTarget::Name(found.name.clone()));
        }
    }
    Err(CameraError(format!(
        "Video source not found: {}",
        path.display()
    )))
}

/// The camera index a `camera.device` setting selects right now, or `None`.
///
/// `"0"`, `"1"`, … and, on Linux, `/dev/videoN` or a stable link to one
/// (`/dev/v4l/by-id/…`, resolved now). `None` for an image, a directory or a camera name and
/// for a device path that does not lead to a camera.
pub fn device_index(device: &str) -> Option<u32> {
    device_index_in(&SourceEnv::system(), device)
}

/// [`device_index`] against another environment.
pub fn device_index_in(env: &SourceEnv, device: &str) -> Option<u32> {
    let spec = device.trim();
    if !spec.is_empty() && spec.bytes().all(|b| b.is_ascii_digit()) {
        return spec.parse().ok();
    }
    if env.linux && spec.starts_with("/dev/") && !env.fs.is_file(spec) {
        return v4l2_index(spec, env.fs.as_ref()).ok();
    }
    None
}

/// `N` of `/dev/videoN`.
fn dev_video_number(path: &str) -> Option<u32> {
    let digits = path.strip_prefix("/dev/video")?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// Camera index of a Linux device path: `/dev/videoN` or a link to one.
///
/// Fails with a [`CameraError`] when the path is missing or not a V4L2 capture node.
fn v4l2_index(spec: &str, fs: &dyn DeviceFs) -> Result<u32, CameraError> {
    if let Some(index) = dev_video_number(spec) {
        return Ok(index);
    }
    let target = fs.resolve_link(spec);
    if let Some(index) = dev_video_number(&target) {
        return Ok(index);
    }
    if !fs.exists(spec) {
        return Err(CameraError(format!(
            "Video device not found: {spec} (is the camera plugged in?)"
        )));
    }
    Err(CameraError(format!(
        "Not a V4L2 video device: {spec} (resolves to {target})"
    )))
}

/// Probe the cameras the system lists (those with an index below `max_index`) and return the
/// ones that deliver frames.
///
/// Each camera is opened briefly (its activity light may flash). This can take several seconds;
/// call it from a background thread. A camera currently held by another application may be
/// missing.
///
/// `skip` lists indices that must not be probed, above all the camera this app's own worker is
/// using: on DirectShow, probing an open camera "succeeds" and releasing the probe then stops
/// the device under the worker.
pub fn list_cameras(
    max_index: u32,
    api: &str,
    skip: &[u32],
) -> Result<Vec<CameraInfo>, CameraError> {
    list_cameras_with(default_backend().as_ref(), max_index, api, skip)
}

/// [`list_cameras`] against another camera API.
pub fn list_cameras_with(
    backend: &dyn CaptureBackend,
    max_index: u32,
    api: &str,
    skip: &[u32],
) -> Result<Vec<CameraInfo>, CameraError> {
    let request = OpenRequest {
        api: api_preference(api)?,
        width: 640,
        height: 480,
        fps: 15,
    };
    let devices = backend.enumerate().unwrap_or_else(|e| {
        log::debug!("Listing cameras failed: {e}");
        Vec::new()
    });
    let mut found = Vec::new();
    for device in devices {
        if device.index >= max_index || skip.contains(&device.index) {
            continue;
        }
        let mut capture = match backend.open(&CameraTarget::Index(device.index), &request) {
            Ok(capture) => capture,
            Err(e) => {
                log::debug!("Probing camera {} failed: {e}", device.index);
                continue;
            }
        };
        match capture.read() {
            Some(frame) => found.push(CameraInfo {
                index: device.index,
                name: device.name,
                width: frame.width,
                height: frame.height,
            }),
            None => log::debug!("Camera {} opened but returned no frame", device.index),
        }
        capture.release();
    }
    Ok(found)
}

#[cfg(test)]
#[path = "camera_tests.rs"]
mod tests;
