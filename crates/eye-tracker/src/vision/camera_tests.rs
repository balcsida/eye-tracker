// Ported from tests/test_camera.py (no physical camera required), with `cv2.VideoCapture`
// replaced by a fake `CaptureBackend`.
//
// Not ported:
// - test_camera_rejects_negative_index: a `u32` index cannot be negative.
// - test_camera_open_configures_capture: the FOURCC / size / fps / BUFFERSIZE property calls
//   do not exist with nokhwa; the request handed to the backend is checked instead.
// - test_dshow_sets_only_what_differs and test_dshow_asks_for_mjpg_last_at_high_bandwidth: the
//   DirectShow property order is gone; the format policy they protected is tested through
//   `format_preference` (test_dshow_*).
// - test_camera_open_when_constructor_raises: a backend reports failure through `Err`, covered by
//   test_camera_open_failure_reports_error.
// - test_file_source_video_loops, test_file_source_video_realtime and the FFmpeg tests
//   (test_file_source_reads_videos_with_ffmpeg_only, test_local_playlist_cannot_reach_the_network):
//   video files are not supported. Looping and realtime playback are tested on a directory of
//   image frames instead.
// - test_v4l2_select_timeout_is_shortened: an OpenCV environment variable.
//
// Added: test_video_files_are_refused, the frame directory tests, the pixel decoding tests, the
// camera-by-name tests and the real FaceTime camera test (ignored; run with `--ignored`).

use std::cell::Cell;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use parking_lot::Mutex;

use super::*;

// --------------------------------------------------------------------------- fakes
/// A camera the fake backend opened: its index, the request and what the capture saw.
type Created = (u32, OpenRequest, Arc<Mutex<CaptureLog>>);

/// What a fake capture saw.
#[derive(Default)]
struct CaptureLog {
    grabs: usize,
    read_calls: usize,
    released: usize,
}

struct FakeConfig {
    reads: Vec<bool>,
    size: (usize, usize),
    grab_ok: bool,
}

impl Default for FakeConfig {
    fn default() -> Self {
        Self {
            reads: Vec::new(),
            size: (64, 48),
            grab_ok: true,
        }
    }
}

struct FakeCapture {
    config: FakeConfig,
    log: Arc<Mutex<CaptureLog>>,
}

impl Capture for FakeCapture {
    fn grab(&mut self) -> bool {
        self.log.lock().grabs += 1;
        self.config.grab_ok
    }

    fn read(&mut self) -> Option<Frame> {
        let mut log = self.log.lock();
        log.read_calls += 1;
        let ok = if self.config.reads.is_empty() {
            true
        } else {
            self.config.reads.remove(0)
        };
        let (w, h) = self.config.size;
        ok.then(|| Frame::filled(w, h, [(log.read_calls % 256) as u8; 3]))
    }

    fn release(&mut self) {
        self.log.lock().released += 1;
    }

    fn describe(&self) -> String {
        "FAKE".into()
    }
}

/// Stand-in for the camera API: `config(index)` scripts each camera, `opened(index)` says whether
/// opening it works.
struct FakeBackend {
    devices: Vec<DeviceInfo>,
    config: Box<dyn Fn(u32) -> FakeConfig + Send + Sync>,
    opened: Box<dyn Fn(u32) -> bool + Send + Sync>,
    created: Mutex<Vec<Created>>,
}

impl FakeBackend {
    fn new(config: impl Fn(u32) -> FakeConfig + Send + Sync + 'static) -> Arc<Self> {
        Arc::new(Self {
            devices: (0..4)
                .map(|index| DeviceInfo {
                    index,
                    name: format!("Camera {index}"),
                })
                .collect(),
            config: Box::new(config),
            opened: Box::new(|_| true),
            created: Mutex::new(Vec::new()),
        })
    }

    fn with(config: FakeConfig) -> Arc<Self> {
        let config = Mutex::new(Some(config));
        Self::new(move |_| config.lock().take().unwrap_or_default())
    }

    fn plain() -> Arc<Self> {
        Self::new(|_| FakeConfig::default())
    }

    fn failing() -> Arc<Self> {
        Arc::new(Self {
            devices: Vec::new(),
            config: Box::new(|_| FakeConfig::default()),
            opened: Box::new(|_| false),
            created: Mutex::new(Vec::new()),
        })
    }

    fn logs(&self) -> Vec<Arc<Mutex<CaptureLog>>> {
        self.created.lock().iter().map(|c| c.2.clone()).collect()
    }

    fn indices(&self) -> Vec<u32> {
        self.created.lock().iter().map(|c| c.0).collect()
    }

    fn request(&self, n: usize) -> OpenRequest {
        self.created.lock()[n].1
    }
}

impl CaptureBackend for FakeBackend {
    fn enumerate(&self) -> Result<Vec<DeviceInfo>, String> {
        Ok(self.devices.clone())
    }

    fn open(
        &self,
        target: &CameraTarget,
        request: &OpenRequest,
    ) -> Result<Box<dyn Capture>, String> {
        let index = match target {
            CameraTarget::Index(index) => *index,
            CameraTarget::Name(name) => self
                .devices
                .iter()
                .find(|d| d.name.eq_ignore_ascii_case(name))
                .map(|d| d.index)
                .ok_or("unknown name")?,
        };
        let log = Arc::new(Mutex::new(CaptureLog::default()));
        self.created.lock().push((index, *request, log.clone()));
        if !(self.opened)(index) {
            return Err("cannot open".into());
        }
        Ok(Box::new(FakeCapture {
            config: (self.config)(index),
            log,
        }))
    }
}

fn camera_on(backend: &Arc<FakeBackend>, index: u32, options: CameraOptions) -> Camera {
    Camera::new(CameraTarget::Index(index), options)
        .expect("valid options")
        .with_backend(backend.clone())
}

fn options(api: &str) -> CameraOptions {
    CameraOptions {
        api: api.into(),
        ..CameraOptions::default()
    }
}

/// A clock the test moves by hand (`FakeClock` of the Python tests).
#[derive(Clone)]
struct FakeClock(Rc<Cell<f64>>);

impl FakeClock {
    fn new() -> Self {
        Self(Rc::new(Cell::new(1000.0)))
    }

    fn advance(&self, seconds: f64) {
        self.0.set(self.0.get() + seconds);
    }

    fn install(&self, camera: &mut Camera) {
        let now = self.0.clone();
        camera.clock = Box::new(move || now.get());
    }
}

fn assert_is_frame_source<T: FrameSource>(_: &T) {}

// ------------------------------------------------------------------ api_preference
#[test]
fn test_api_preference_auto_matches_platform() {
    let expected = if cfg!(windows) {
        CaptureApi::DShow
    } else if cfg!(target_os = "macos") {
        CaptureApi::AvFoundation
    } else if cfg!(target_os = "linux") {
        CaptureApi::V4l2
    } else {
        CaptureApi::Any
    };
    assert_eq!(api_preference("auto").unwrap(), expected);
    assert_eq!(api_preference(" AUTO ").unwrap(), expected);
}

#[test]
fn test_api_preference_named() {
    for (name, value) in [
        ("any", CaptureApi::Any),
        ("dshow", CaptureApi::DShow),
        ("msmf", CaptureApi::Msmf),
        ("avfoundation", CaptureApi::AvFoundation),
        ("v4l2", CaptureApi::V4l2),
    ] {
        assert_eq!(api_preference(name).unwrap(), value, "{name}");
    }
}

#[test]
fn test_api_preference_unknown_raises() {
    let error = api_preference("gstreamer-magic").unwrap_err();
    assert!(error.0.contains("Unknown camera API"), "{error}");
}

// ------------------------------------------------------------------------ Camera
#[test]
fn test_camera_is_a_frame_source() {
    let mut cam = Camera::new(CameraTarget::Index(0), CameraOptions::default()).unwrap();
    assert_is_frame_source(&cam);
    assert!(!cam.is_open());
    assert!(cam.read().is_none());
    assert!(cam.last_error().is_none());
}

#[test]
fn test_camera_open_configures_capture() {
    let backend = FakeBackend::with(FakeConfig {
        size: (320, 240),
        ..FakeConfig::default()
    });
    let mut cam = camera_on(
        &backend,
        2,
        CameraOptions {
            width: 320,
            height: 240,
            api: "any".into(),
            fps: 12,
            flush: 1,
        },
    );
    assert!(cam.open());
    assert!(cam.is_open());
    assert!(cam.last_error().is_none());
    assert_eq!(cam.frame_size(), Some((320, 240)));
    assert_eq!(backend.indices(), [2]);
    assert_eq!(
        backend.request(0),
        OpenRequest {
            api: CaptureApi::Any,
            width: 320,
            height: 240,
            fps: 12,
        }
    );
    // Opening twice is a no-op.
    assert!(cam.open());
    assert_eq!(backend.indices().len(), 1);
    cam.release();
    assert_eq!(backend.logs()[0].lock().released, 1);
}

#[test]
fn test_camera_open_failure_reports_error() {
    let backend = FakeBackend::failing();
    let mut cam = camera_on(&backend, 3, CameraOptions::default());
    assert!(!cam.open());
    assert!(!cam.is_open());
    let error = cam.last_error().expect("an error");
    assert!(error.contains("Camera 3"), "{error}");
    assert!(error.contains("another app"), "{error}");
    // A camera blocked by the OS privacy switch fails the same way.
    assert!(error.contains("privacy settings"), "{error}");
    assert!(cam.read().is_none());
}

#[test]
fn test_dshow_leaves_the_format_alone_below_usb2_bandwidth() {
    // Every DirectShow property change rebuilds the capture graph; uncompressed 640x480@15 fits
    // USB 2, so MJPG is not asked for first.
    let wanted = format_preference(CaptureApi::DShow, 640, 480, 15);
    assert_eq!(wanted[0], PixelFormat::Yuyv);
}

#[test]
fn test_dshow_asks_for_mjpg_at_high_bandwidth() {
    let wanted = format_preference(CaptureApi::DShow, 1280, 720, 15);
    assert_eq!(wanted[0], PixelFormat::Mjpeg);
}

#[test]
fn test_other_apis_ask_for_the_format_first() {
    for api in [CaptureApi::V4l2, CaptureApi::Msmf] {
        assert_eq!(format_preference(api, 640, 480, 15)[0], PixelFormat::Mjpeg);
    }
}

#[test]
fn test_avfoundation_never_asks_for_mjpeg() {
    let wanted = format_preference(CaptureApi::AvFoundation, 640, 480, 15);
    assert_eq!(wanted, [PixelFormat::Yuyv]);
}

#[test]
fn test_camera_open_without_frames_fails() {
    let backend = FakeBackend::with(FakeConfig {
        reads: vec![false; 10],
        ..FakeConfig::default()
    });
    let mut cam = camera_on(&backend, 1, CameraOptions::default());
    assert!(!cam.open());
    let error = cam.last_error().expect("an error");
    assert!(error.contains("no frames"), "{error}");
    assert_eq!(backend.logs()[0].lock().released, 1);
}

#[test]
fn test_camera_open_tolerates_empty_first_frames() {
    let backend = FakeBackend::with(FakeConfig {
        reads: vec![false, false, true],
        ..FakeConfig::default()
    });
    let mut cam = camera_on(&backend, 0, CameraOptions::default());
    assert!(cam.open());
}

#[test]
fn test_camera_read_flushes_only_after_a_pause() {
    let backend = FakeBackend::plain();
    // fps=1 makes the "pause" window 1.5 s, so back-to-back reads never flush.
    let mut cam = camera_on(
        &backend,
        0,
        CameraOptions {
            fps: 1,
            flush: 2,
            ..CameraOptions::default()
        },
    );
    assert!(cam.open());
    let log = backend.logs()[0].clone();
    assert!(cam.read().is_some());
    assert_eq!(log.lock().grabs, 0);
    cam.last_read -= 10.0; // pretend the reader paused for 10 s
    assert!(cam.read().is_some());
    assert_eq!(log.lock().grabs, 2);
}

#[test]
fn test_camera_read_without_flush() {
    let backend = FakeBackend::plain();
    let mut cam = camera_on(
        &backend,
        0,
        CameraOptions {
            flush: 0,
            ..CameraOptions::default()
        },
    );
    assert!(cam.open());
    cam.last_read -= 10.0;
    assert!(cam.read().is_some());
    assert_eq!(backend.logs()[0].lock().grabs, 0);
}

#[test]
fn test_camera_closes_after_consecutive_failures() {
    // One successful warm-up read, then failures.
    let mut reads = vec![true];
    reads.extend([false; 5]);
    let backend = FakeBackend::with(FakeConfig {
        reads,
        ..FakeConfig::default()
    });
    let mut cam = camera_on(&backend, 4, CameraOptions::default());
    assert!(cam.open());
    for _ in 0..4 {
        assert!(cam.read().is_none());
        assert!(cam.is_open());
    }
    assert!(cam.read().is_none());
    assert!(!cam.is_open());
    let error = cam.last_error().expect("an error");
    assert!(error.contains("stopped delivering frames"), "{error}");
    assert_eq!(backend.logs()[0].lock().released, 1);
    assert!(cam.read().is_none());
}

#[test]
fn test_camera_gives_up_after_failing_for_seconds() {
    // Drivers whose reads block for seconds must not keep the worker for a minute.
    let clock = FakeClock::new();
    let mut reads = vec![true];
    reads.extend([false; 10]);
    let backend = FakeBackend::with(FakeConfig {
        reads,
        ..FakeConfig::default()
    });
    let mut cam = camera_on(&backend, 0, CameraOptions::default());
    clock.install(&mut cam);
    assert!(cam.open());
    clock.advance(0.05);
    assert!(cam.read().is_none()); // first failure
    clock.advance(2.0); // a read that blocked for 2 s
    assert!(cam.read().is_none());
    assert!(cam.is_open());
    clock.advance(2.0);
    assert!(cam.read().is_none());
    assert!(!cam.is_open()); // 4 s of failures, only 3 reads
    let error = cam.last_error().expect("an error");
    assert!(error.contains("stopped delivering frames"), "{error}");
    assert_eq!(backend.logs()[0].lock().released, 1);
}

#[test]
fn test_camera_does_not_flush_while_failing() {
    let clock = FakeClock::new();
    let backend = FakeBackend::with(FakeConfig {
        reads: vec![true, false, false, true],
        ..FakeConfig::default()
    });
    let mut cam = camera_on(
        &backend,
        0,
        CameraOptions {
            fps: 10,
            flush: 1,
            ..CameraOptions::default()
        },
    );
    clock.install(&mut cam);
    assert!(cam.open());
    let log = backend.logs()[0].clone();
    clock.advance(1.0); // paused: the next read flushes
    assert!(cam.read().is_none());
    assert_eq!(log.lock().grabs, 1);
    clock.advance(1.0); // paused again, but failing: no extra blocking grab
    assert!(cam.read().is_none());
    assert_eq!(log.lock().grabs, 1);
    clock.advance(0.01);
    assert!(cam.read().is_some());
}

#[test]
fn test_failed_flush_counts_as_a_failed_read() {
    let clock = FakeClock::new();
    let backend = FakeBackend::with(FakeConfig {
        grab_ok: false,
        ..FakeConfig::default()
    });
    let mut cam = camera_on(
        &backend,
        0,
        CameraOptions {
            fps: 10,
            flush: 2,
            ..CameraOptions::default()
        },
    );
    clock.install(&mut cam);
    assert!(cam.open());
    let log = backend.logs()[0].clone();
    let reads_after_open = log.lock().read_calls;
    clock.advance(1.0);
    assert!(cam.read().is_none());
    assert_eq!(log.lock().grabs, 1); // stopped at the first failed grab
    assert_eq!(log.lock().read_calls, reads_after_open); // and did not block in read() as well
}

#[test]
fn test_camera_failure_counter_resets_on_success() {
    let backend = FakeBackend::with(FakeConfig {
        reads: vec![true, false, false, false, false, true, false, false],
        ..FakeConfig::default()
    });
    let mut cam = camera_on(&backend, 0, CameraOptions::default());
    assert!(cam.open());
    for _ in 0..4 {
        cam.read();
    }
    assert!(cam.read().is_some());
    cam.read();
    cam.read();
    assert!(cam.is_open());
}

#[test]
fn test_camera_release_is_idempotent() {
    let backend = FakeBackend::plain();
    let mut cam = camera_on(&backend, 0, CameraOptions::default());
    cam.release();
    assert!(cam.open());
    cam.release();
    cam.release();
    assert_eq!(backend.logs()[0].lock().released, 1);
    assert!(!cam.is_open());
}

#[test]
fn test_missing_real_camera_fails_gracefully() {
    if cfg!(target_os = "macos") {
        return; // AVFoundation may block on a camera permission prompt in headless runs
    }
    let mut cam = Camera::new(CameraTarget::Index(63), CameraOptions::default()).unwrap();
    assert!(!cam.open());
    let error = cam.last_error().expect("an error");
    assert!(error.contains("Camera 63"), "{error}");
    assert!(cam.read().is_none());
    cam.release();
}

#[test]
fn test_camera_by_name_uses_the_name_in_messages() {
    let backend = FakeBackend::failing();
    let mut cam = Camera::new(
        CameraTarget::Name("FaceTime HD Camera".into()),
        options("any"),
    )
    .unwrap()
    .with_backend(backend);
    assert!(!cam.open());
    let error = cam.last_error().expect("an error");
    assert!(
        error.starts_with("FaceTime HD Camera could not be opened"),
        "{error}"
    );
}

// ------------------------------------------------------------------ list_cameras
#[test]
fn test_list_cameras_reports_working_indices() {
    let backend = Arc::new(FakeBackend {
        devices: (0..4)
            .map(|index| DeviceInfo {
                index,
                name: format!("Camera {index}"),
            })
            .collect(),
        config: Box::new(|index| FakeConfig {
            reads: vec![index != 3], // camera 3 opens but delivers nothing
            size: (160 * (index as usize + 1), 120 * (index as usize + 1)),
            grab_ok: true,
        }),
        opened: Box::new(|index| matches!(index, 0 | 2 | 3)),
        created: Mutex::new(Vec::new()),
    });
    let found = list_cameras_with(backend.as_ref(), 4, "any", &[]).unwrap();
    assert_eq!(
        found,
        [
            CameraInfo {
                index: 0,
                name: "Camera 0".into(),
                width: 160,
                height: 120,
            },
            CameraInfo {
                index: 2,
                name: "Camera 2".into(),
                width: 480,
                height: 360,
            },
        ]
    );
    assert_eq!(backend.indices(), [0, 1, 2, 3]);
    // Every camera that opened was released (camera 1 never opened).
    for (index, _, log) in backend.created.lock().iter() {
        assert_eq!(
            log.lock().released,
            usize::from(*index != 1),
            "camera {index}"
        );
    }
}

#[test]
fn test_list_cameras_never_touches_skipped_indices() {
    // Probing the worker's own DirectShow camera would stop it under the worker.
    let backend = FakeBackend::plain();
    let found = list_cameras_with(backend.as_ref(), 3, "any", &[0]).unwrap();
    assert_eq!(backend.indices(), [1, 2]);
    assert_eq!(found.iter().map(|c| c.index).collect::<Vec<_>>(), [1, 2]);
}

#[test]
fn test_list_cameras_names_the_devices_the_system_lists() {
    let backend = Arc::new(FakeBackend {
        devices: vec![
            DeviceInfo {
                index: 0,
                name: "iPhone Camera".into(),
            },
            DeviceInfo {
                index: 1,
                name: "FaceTime HD Camera".into(),
            },
        ],
        config: Box::new(|_| FakeConfig::default()),
        opened: Box::new(|_| true),
        created: Mutex::new(Vec::new()),
    });
    let found = list_cameras_with(backend.as_ref(), 4, "any", &[]).unwrap();
    assert_eq!(found[1].name, "FaceTime HD Camera");
}

#[test]
fn test_list_cameras_rejects_an_unknown_api() {
    let error = list_cameras_with(FakeBackend::plain().as_ref(), 4, "bogus", &[]).unwrap_err();
    assert!(error.0.contains("Unknown camera API"), "{error}");
}

// -------------------------------------------------------------------- FileSource
fn write_png(path: &Path, frame: &Frame) {
    frame.to_rgb_image().save(path).expect("write a png");
}

fn mean(frame: &Frame) -> f64 {
    frame.data.iter().map(|&v| f64::from(v)).sum::<f64>() / frame.data.len() as f64
}

fn file_source(path: &Path) -> FileSource {
    FileSource::new(path, None, None, false)
}

#[test]
fn test_file_source_still_image() {
    let dir = tempfile::tempdir().unwrap();
    let mut image = Frame::filled(80, 60, [0, 0, 0]);
    for y in 10..20 {
        for x in 10..30 {
            image.set_pixel(x, y, [10, 200, 30]);
        }
    }
    let path = dir.path().join("still.png");
    write_png(&path, &image);
    let mut src = file_source(&path);
    assert_is_frame_source(&src);
    assert!(src.open());
    let mut first = src.read().expect("a frame");
    assert_eq!(first, image);
    first.data.fill(255); // consumers may draw on frames; later reads must be unaffected
    assert_eq!(src.read().expect("a frame"), image);
    src.release();
    assert!(!src.is_open());
    assert!(src.read().is_none());
}

#[test]
fn test_file_source_non_ascii_path() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("görüntü şğ.png");
    write_png(&path, &Frame::filled(16, 12, [7, 7, 7]));
    let mut src = file_source(&path);
    assert!(src.open(), "{:?}", src.last_error());
    let frame = src.read().expect("a frame");
    assert_eq!((frame.height, frame.width), (12, 16));
}

#[test]
fn test_file_source_fits_bounding_box() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("big.png");
    write_png(&path, &Frame::filled(800, 600, [0, 0, 0]));
    let mut src = FileSource::new(&path, Some(400), Some(400), false);
    assert!(src.open());
    let frame = src.read().expect("a frame");
    assert_eq!((frame.height, frame.width), (300, 400)); // aspect ratio preserved
    let small = dir.path().join("small.png");
    write_png(&small, &Frame::filled(40, 30, [0, 0, 0]));
    let mut src = FileSource::new(&small, Some(640), Some(480), false);
    assert!(src.open());
    let frame = src.read().expect("a frame");
    assert_eq!((frame.height, frame.width), (30, 40)); // never enlarged
}

#[test]
fn test_file_source_missing_file() {
    let dir = tempfile::tempdir().unwrap();
    let mut src = file_source(&dir.path().join("nope.png"));
    assert!(!src.open());
    let error = src.last_error().expect("an error");
    assert!(error.contains("not found"), "{error}");
    assert!(src.read().is_none());
}

#[test]
fn test_file_source_undecodable_image() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("broken.png");
    std::fs::write(&path, b"not a png at all").unwrap();
    let mut src = file_source(&path);
    assert!(!src.open());
    assert!(src.last_error().is_some());
}

#[test]
fn test_video_files_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("clip.avi");
    std::fs::write(&path, [0u8; 64]).unwrap();
    let mut src = file_source(&path);
    assert!(!src.open());
    let error = src.last_error().expect("an error");
    assert!(error.contains("video files are not supported"), "{error}");
}

const LEVELS: [u8; 6] = [0, 50, 100, 150, 200, 250];

/// A directory of solid frames named `frame0.png`, `frame1.png`, ... in `LEVELS` order.
fn write_frames(dir: &Path) {
    for (i, level) in LEVELS.iter().enumerate() {
        write_png(
            &dir.join(format!("frame{i}.png")),
            &Frame::filled(64, 48, [*level; 3]),
        );
    }
}

fn level(frame: Option<Frame>) -> u8 {
    let value = mean(&frame.expect("a frame"));
    *LEVELS
        .iter()
        .min_by(|a, b| {
            (f64::from(**a) - value)
                .abs()
                .total_cmp(&(f64::from(**b) - value).abs())
        })
        .unwrap()
}

#[test]
fn test_file_source_directory_loops() {
    let dir = tempfile::tempdir().unwrap();
    write_frames(dir.path());
    let mut src = file_source(dir.path());
    assert!(src.open(), "{:?}", src.last_error());
    let seen: Vec<u8> = (0..LEVELS.len() + 2).map(|_| level(src.read())).collect();
    let mut expected = LEVELS.to_vec();
    expected.extend([LEVELS[0], LEVELS[1]]);
    assert_eq!(seen, expected);
    src.release();
    assert!(src.read().is_none());
}

#[test]
fn test_file_source_directory_plays_in_name_order() {
    let dir = tempfile::tempdir().unwrap();
    // Created in reverse, and with a file that is not an image among them.
    for (name, level) in [("c.png", 200u8), ("a.png", 0), ("b.png", 100)] {
        write_png(&dir.path().join(name), &Frame::filled(8, 8, [level; 3]));
    }
    std::fs::write(dir.path().join("notes.txt"), "not a frame").unwrap();
    let mut src = file_source(dir.path());
    assert!(src.open());
    let means: Vec<f64> = (0..4).map(|_| mean(&src.read().unwrap()).round()).collect();
    assert_eq!(means, [0.0, 100.0, 200.0, 0.0]);
}

#[test]
fn test_file_source_empty_directory() {
    let dir = tempfile::tempdir().unwrap();
    let mut src = file_source(dir.path());
    assert!(!src.open());
    let error = src.last_error().expect("an error");
    assert!(error.contains("no image frames"), "{error}");
}

#[test]
fn test_file_source_directory_realtime() {
    let dir = tempfile::tempdir().unwrap();
    write_frames(dir.path());
    let mut src = FileSource::new(dir.path(), None, None, true);
    src.video_fps = 10.0;
    assert!(src.open(), "{:?}", src.last_error());
    let clock = Rc::new(Cell::new(100.0));
    let now = clock.clone();
    src.now = Box::new(move || now.get());
    assert_eq!(level(src.read()), LEVELS[0]);
    clock.set(clock.get() + 0.05); // half a frame period later: still the same frame
    assert_eq!(level(src.read()), LEVELS[0]);
    clock.set(clock.get() + 0.26); // 0.31 s after the start -> frame 3
    assert_eq!(level(src.read()), LEVELS[3]);
    clock.set(clock.get() + 0.4); // 0.71 s: frame 7 of a 6-frame clip -> wrapped to frame 1
    assert_eq!(level(src.read()), LEVELS[1]);
}

// ------------------------------------------------------------------- open_source
fn env_with(backend: Arc<FakeBackend>) -> SourceEnv {
    SourceEnv {
        backend,
        fs: Box::new(FakeDev::default()),
        linux: false,
    }
}

fn open(env: &SourceEnv, device: &str) -> Result<Source, CameraError> {
    open_source_in(env, device, 640, 480, "auto", 15, true)
}

#[test]
fn test_open_source_camera_index() {
    let env = env_with(FakeBackend::plain());
    let src = open_source_in(&env, " 1 ", 320, 240, "any", 15, true).unwrap();
    let Source::Camera(camera) = src else {
        panic!("expected a camera");
    };
    assert_eq!(camera.target, CameraTarget::Index(1));
    assert_eq!((camera.width, camera.height), (320, 240));
    assert!(!camera.is_open()); // created, not opened
}

#[test]
fn test_open_source_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("frame.png");
    write_png(&path, &Frame::filled(10, 10, [0, 0, 0]));
    let env = env_with(FakeBackend::plain());
    let Source::File(source) = open(&env, path.to_str().unwrap()).unwrap() else {
        panic!("expected a file source");
    };
    assert!(source.realtime);
}

#[test]
fn test_open_source_directory() {
    let dir = tempfile::tempdir().unwrap();
    write_frames(dir.path());
    let env = env_with(FakeBackend::plain());
    assert!(matches!(
        open(&env, dir.path().to_str().unwrap()),
        Ok(Source::File(_))
    ));
}

#[test]
fn test_open_source_errors() {
    let env = env_with(FakeBackend::plain());
    for device in ["", "   ", "definitely/not/here.mp4"] {
        assert!(open(&env, device).is_err(), "{device:?}");
    }
}

#[test]
fn test_open_source_bad_api() {
    let env = env_with(FakeBackend::plain());
    let error = open_source_in(&env, "0", 640, 480, "bogus", 15, true).unwrap_err();
    assert!(error.0.contains("Unknown camera API"), "{error}");
}

#[test]
fn test_open_source_records_the_device_string() {
    let dir = tempfile::tempdir().unwrap();
    let env = env_with(FakeBackend::plain());
    let src = open(&env, " 2 ").unwrap();
    assert!(matches!(&src, Source::Camera(_)));
    assert_eq!(src.device().as_deref(), Some("2"));
    let path = dir.path().join("frame.png");
    write_png(&path, &Frame::filled(10, 10, [0, 0, 0]));
    let src = open(&env, path.to_str().unwrap()).unwrap();
    assert_eq!(src.device(), Some(path.to_str().unwrap().to_string()));
}

#[test]
fn test_open_source_finds_a_camera_by_name() {
    let env = env_with(FakeBackend::plain());
    let src = open(&env, "camera 2").unwrap();
    let Source::Camera(camera) = src else {
        panic!("expected a camera");
    };
    assert_eq!(camera.target, CameraTarget::Name("Camera 2".into()));
    assert_eq!(camera.device, "camera 2");
    let error = open(&env, "No Such Camera").map(|_| ()).unwrap_err();
    assert!(error.0.contains("not found"), "{error}");
}

#[test]
fn test_open_source_refuses_network_and_protocol_sources() {
    // Offline guarantee: nothing but a local path is ever opened.
    let env = env_with(FakeBackend::plain());
    for device in [
        "rtsp://192.168.1.20/stream",
        "http://example.com/clip.mp4",
        "HTTPS://example.com/clip.mp4",
        "https:example.com/clip.mp4",
        "tcp:10.0.0.1:5000",
        "udp://@:1234",
        "srt://host:9000",
        "concat:a.mp4|b.mp4",
        "file:clip.mp4",
        "rtmp+tls:host",
        "subfile,,start,0,end,0,,:video.mp4",
        "C:/videos/../x://y",
    ] {
        let error = open(&env, device).map(|_| ()).unwrap_err();
        assert!(error.0.contains("not supported"), "{device}: {error}");
    }
}

#[test]
fn test_file_source_refuses_urls() {
    for device in ["rtsp://camera/stream", "tcp:10.0.0.1:5000"] {
        let mut src = file_source(Path::new(device));
        assert!(!src.open());
        let error = src.last_error().expect("an error");
        assert!(
            error.contains("network sources are not supported"),
            "{error}"
        );
    }
}

#[test]
fn test_windows_drive_paths_are_files() {
    let missing = if cfg!(windows) {
        "C:\\definitely\\missing.mp4"
    } else {
        "/definitely/missing.mp4"
    };
    let env = env_with(FakeBackend::plain());
    let error = open(&env, missing).map(|_| ()).unwrap_err();
    assert!(error.0.contains("not found"), "{error}");
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("clip.png");
    write_png(&path, &Frame::filled(10, 10, [0, 0, 0]));
    let resolved = path.canonicalize().unwrap();
    assert!(matches!(
        open(&env, resolved.to_str().unwrap()),
        Ok(Source::File(_))
    ));
}

// ------------------------------------------------------------------ Linux devices
/// A fake `/dev` (link -> target): the `linux` fixture of the Python tests.
#[derive(Default, Clone)]
struct FakeDev {
    links: HashMap<String, String>,
}

impl DeviceFs for FakeDev {
    fn exists(&self, path: &str) -> bool {
        self.links.contains_key(path)
    }

    fn is_file(&self, _path: &str) -> bool {
        false
    }

    fn resolve_link(&self, path: &str) -> String {
        self.links
            .get(path)
            .cloned()
            .unwrap_or_else(|| path.to_string())
    }
}

fn linux_env(links: &[(&str, &str)]) -> SourceEnv {
    SourceEnv {
        backend: FakeBackend::plain(),
        fs: Box::new(FakeDev {
            links: links
                .iter()
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect(),
        }),
        linux: true,
    }
}

fn camera_index(source: Source) -> (Option<u32>, Option<String>) {
    let Source::Camera(camera) = source else {
        panic!("expected a camera");
    };
    let index = match camera.target {
        CameraTarget::Index(index) => Some(index),
        CameraTarget::Name(_) => None,
    };
    (index, camera.device())
}

#[test]
fn test_open_source_dev_video() {
    let env = linux_env(&[]);
    let (index, device) = camera_index(open(&env, "/dev/video2").unwrap());
    assert_eq!(index, Some(2));
    assert_eq!(device.as_deref(), Some("/dev/video2"));
}

#[test]
fn test_stable_v4l_links_resolve_to_the_camera() {
    let link = "/dev/v4l/by-id/usb-Logitech_C920_1234-video-index0";
    let env = linux_env(&[(link, "/dev/video7")]);
    let (index, device) = camera_index(open(&env, link).unwrap());
    assert_eq!(index, Some(7));
    assert_eq!(device.as_deref(), Some(link)); // the stable name, for calibrations and settings
    // Re-plugged: resolved again on the next open.
    let env = linux_env(&[(link, "/dev/video2")]);
    assert_eq!(camera_index(open(&env, link).unwrap()).0, Some(2));
}

#[test]
fn test_missing_and_non_video_devices() {
    let env = linux_env(&[("/dev/v4l/by-id/weird", "/dev/snd/pcmC0D0c")]);
    let error = open(&env, "/dev/v4l/by-id/usb-unplugged-video-index0")
        .map(|_| ())
        .unwrap_err();
    assert!(error.0.contains("not found"), "{error}");
    let error = open(&env, "/dev/v4l/by-id/weird").map(|_| ()).unwrap_err();
    assert!(error.0.contains("Not a V4L2 video device"), "{error}");
}

#[test]
fn test_device_index_follows_stable_links() {
    // What the settings dialog must not probe: the index the device setting leads to.
    let link = "/dev/v4l/by-id/usb-Logitech_C920_1234-video-index0";
    let env = linux_env(&[
        (link, "/dev/video3"),
        ("/dev/v4l/by-id/weird", "/dev/snd/pcmC0D0c"),
    ]);
    assert_eq!(device_index_in(&env, " 1 "), Some(1));
    assert_eq!(device_index_in(&env, "/dev/video5"), Some(5));
    assert_eq!(device_index_in(&env, link), Some(3));
    // Unplugged, or not a camera: no index.
    assert_eq!(
        device_index_in(&env, "/dev/v4l/by-id/usb-unplugged-video-index0"),
        None
    );
    assert_eq!(device_index_in(&env, "/dev/v4l/by-id/weird"), None);
    assert_eq!(device_index_in(&env, "clip.png"), None); // an image file
}

#[test]
fn test_device_paths_are_files_off_linux() {
    let env = env_with(FakeBackend::plain());
    assert_eq!(device_index_in(&env, "/dev/video5"), None);
    assert_eq!(device_index_in(&env, "2"), Some(2));
}

#[cfg(target_os = "linux")]
#[test]
fn test_real_symlink_to_a_video_node() {
    let dir = tempfile::tempdir().unwrap();
    let link = dir.path().join("by-id-camera");
    std::os::unix::fs::symlink("/dev/video7", &link).unwrap();
    assert_eq!(v4l2_index(link.to_str().unwrap(), &RealFs).unwrap(), 7);
}

// -------------------------------------------------------------- pixel decoding
#[test]
fn test_url_like() {
    for spec in ["rtsp://x", "ab:c", "subfile,x", "x://y", "file:clip.mp4"] {
        assert!(url_like(spec), "{spec}");
    }
    for spec in ["C:\\x.png", "/tmp/x:y.png", "0", "frames", "a:b"] {
        assert!(!url_like(spec), "{spec}");
    }
}

#[test]
fn test_decode_yuyv_known_colours() {
    // White, black, then a saturated red pair: Y U Y V quadruples (BT.601 limited range).
    let data = [235, 128, 235, 128, 16, 128, 16, 128, 81, 90, 81, 240];
    let frame = decode_frame(PixelFormat::Yuyv, 6, 1, &data).expect("a frame");
    assert_eq!(frame.pixel(0, 0), [255, 255, 255]);
    assert_eq!(frame.pixel(2, 0), [0, 0, 0]);
    let [b, g, r] = frame.pixel(4, 0);
    assert!(r > 240 && g < 20 && b < 20, "{:?}", [b, g, r]);
}

#[test]
fn test_decode_nv12_matches_yuyv() {
    // The same 4x2 picture in both layouts.
    let luma = [235u8, 200, 100, 16, 180, 160, 60, 40];
    let (u, v) = ([100u8, 140], [200u8, 60]);
    let mut nv12 = luma.to_vec();
    nv12.extend([u[0], v[0], u[1], v[1]]);
    let mut yuyv = Vec::new();
    for row in 0..2 {
        for pair in 0..2 {
            yuyv.extend([
                luma[row * 4 + pair * 2],
                u[pair],
                luma[row * 4 + pair * 2 + 1],
                v[pair],
            ]);
        }
    }
    let a = decode_frame(PixelFormat::Nv12, 4, 2, &nv12).expect("an NV12 frame");
    let b = decode_frame(PixelFormat::Yuyv, 4, 2, &yuyv).expect("a YUYV frame");
    assert_eq!(a, b);
}

#[test]
fn test_decode_skips_row_padding() {
    let row = [235u8, 128, 235, 128];
    let mut padded = Vec::new();
    for _ in 0..2 {
        padded.extend(row);
        padded.extend([9, 9, 9, 9]);
    }
    let frame = decode_frame(PixelFormat::Yuyv, 2, 2, &padded).expect("a frame");
    assert_eq!(frame, Frame::filled(2, 2, [255, 255, 255]));
}

#[test]
fn test_decode_rejects_short_buffers() {
    assert!(decode_frame(PixelFormat::Yuyv, 4, 4, &[0; 10]).is_none());
    assert!(decode_frame(PixelFormat::Nv12, 4, 4, &[0; 10]).is_none());
    assert!(decode_frame(PixelFormat::Mjpeg, 4, 4, b"not a jpeg").is_none());
    assert!(decode_frame(PixelFormat::Gray, 0, 4, &[]).is_none());
}

#[test]
fn test_decode_gray_and_rgb() {
    let gray = decode_frame(PixelFormat::Gray, 2, 1, &[10, 20]).expect("a frame");
    assert_eq!(gray.data, [10, 10, 10, 20, 20, 20]);
    let rgb = decode_frame(PixelFormat::Rgb, 1, 1, &[1, 2, 3]).expect("a frame");
    assert_eq!(rgb.pixel(0, 0), [3, 2, 1]);
    let bgr = decode_frame(PixelFormat::Bgr, 1, 1, &[1, 2, 3]).expect("a frame");
    assert_eq!(bgr.pixel(0, 0), [1, 2, 3]);
}

#[test]
fn test_decode_mjpeg() {
    let image = image::RgbImage::from_pixel(16, 16, image::Rgb([200, 40, 40]));
    let mut jpeg = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 95)
        .encode_image(&image)
        .unwrap();
    let frame = decode_frame(PixelFormat::Mjpeg, 16, 16, &jpeg).expect("a frame");
    assert_eq!((frame.width, frame.height), (16, 16));
    let [b, g, r] = frame.pixel(8, 8);
    assert!(r > 180 && g < 70 && b < 70, "{:?}", [b, g, r]);
}

// ---------------------------------------------------------------- real hardware
/// Opens the built-in FaceTime HD camera by name (never index 0: on a Mac that is often a
/// phone's Continuity Camera) and checks that real pictures come out. Run with
/// `cargo test real_facetime_camera -- --ignored --nocapture`. Frames stay in memory.
#[test]
#[ignore = "needs the built-in FaceTime HD camera and camera access"]
fn real_facetime_camera() {
    const NAME: &str = "FaceTime HD Camera";
    std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_secs(30));
        eprintln!("watchdog: the camera test took longer than 30 s");
        std::process::exit(2);
    });
    let backend = default_backend();
    let started = Instant::now();
    let devices = backend.enumerate().expect("the camera list");
    println!(
        "{} camera(s) in {:.0} ms:",
        devices.len(),
        started.elapsed().as_secs_f64() * 1e3
    );
    for device in &devices {
        println!("  {}: {}", device.index, device.name);
    }
    assert!(devices.iter().any(|d| d.name == NAME), "{NAME} is listed");

    let mut camera =
        Camera::new(CameraTarget::Name(NAME.into()), CameraOptions::default()).unwrap();
    let opened = camera.open();
    println!("open by name: {opened}, error {:?}", camera.last_error());
    if !opened {
        let request = OpenRequest {
            api: api_preference("auto").unwrap(),
            width: 640,
            height: 480,
            fps: 15,
        };
        let error = backend
            .open(&CameraTarget::Name(NAME.into()), &request)
            .map(|_| ())
            .unwrap_err();
        panic!("the camera did not open: {error}");
    }
    let (width, height) = camera.frame_size().expect("a frame size");
    let described = camera.capture.as_ref().map(|c| c.describe());
    println!("opened {NAME} at {width}x{height}: {described:?}");
    assert!(width >= 640 && height >= 480, "{width}x{height}");

    // The first frames are dark while the camera adjusts its exposure.
    for _ in 0..3 {
        camera.read().expect("a frame");
    }
    for n in 0..5 {
        let frame = camera.read().expect("a frame");
        assert_eq!((frame.width, frame.height), (width, height));
        let channel_mean = |c: usize| {
            let sum: u64 = frame
                .data
                .iter()
                .skip(c)
                .step_by(3)
                .map(|&v| u64::from(v))
                .sum();
            sum as f64 / (frame.width * frame.height) as f64
        };
        let (b, g, r) = (channel_mean(0), channel_mean(1), channel_mean(2));
        let brightness = (b + g + r) / 3.0;
        println!("frame {n}: mean {brightness:.1} (b {b:.1}, g {g:.1}, r {r:.1})");
        assert!(brightness > 5.0, "the frame is black: {brightness}");
        assert!(
            (b - g).abs() > 0.5 || (g - r).abs() > 0.5,
            "the three channels are identical: {b} {g} {r}"
        );
    }
    camera.release();
    assert!(!camera.is_open());
}
