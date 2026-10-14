//! Apple backend (macOS only): face rectangles, head pose and landmarks from Apple's Vision
//! framework.
//!
//! Vision runs Apple's own face models, on the Neural Engine where the landmarks request allows
//! it, so this backend ships no model files and keeps tract off the CPU. Per frame:
//!
//! 1. The BGR frame becomes a `CGImage` in memory (it is never written anywhere).
//! 2. `VNDetectFaceRectanglesRequest` (revision 3) finds every face with its roll, yaw and pitch.
//! 3. `VNDetectFaceLandmarksRequest` (revision 3, 76-point constellation) reads the primary face
//!    (largest, then most central, as in the facemesh backend): eye contours and pupils.
//!
//! Feature vector ([`AppleBackend::FEATURE_NAMES`]): `yaw`, `pitch`, `roll` in degrees with the
//! facemesh backend's signs, and `iris_h`, `iris_v` computed by the facemesh backend's own
//! [`eye_metrics_of`] from each eye's corners, lids and pupil. Vision keeps no state between
//! frames, so every result is settled.

use std::time::Instant;

use imageproc::drawing::draw_filled_circle_mut;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{NSObjectProtocol, ProtocolObject};
use objc2::{AnyThread, ClassType, msg_send, sel};
use objc2_core_foundation::{CFData, CFRetained};
use objc2_core_graphics::{
    CGBitmapInfo, CGColorRenderingIntent, CGColorSpace, CGDataProvider, CGImage, CGImageAlphaInfo,
    CGImageByteOrderInfo,
};
use objc2_core_ml::{MLComputeDeviceProtocol, MLNeuralEngineComputeDevice};
use objc2_foundation::{NSArray, NSDictionary, NSIndexSet};
use objc2_vision::{
    VNComputeStageMain, VNDetectFaceLandmarksRequest, VNDetectFaceRectanglesRequest,
    VNFaceLandmarkRegion2D, VNFaceLandmarks2D, VNFaceObservation, VNFaceObservationAccepting,
    VNImageRequestHandler, VNRequest, VNRequestFaceLandmarksConstellation,
};

use super::base::{BackendUnavailable, VisionBackend};
use super::facemesh_backend::{BlinkDetector, Box4, EyeMetrics, eye_metrics_of, primary_score};
use super::overlay::{
    COLOR_BOX, COLOR_IRIS, COLOR_OTHER, COLOR_POINT, draw_box, draw_label, line_thickness,
    observation_label,
};
use crate::types::Observation;
use crate::vision::Frame;

/// Revision 3 of both requests (macOS 12): the rectangles carry roll, yaw and pitch, the
/// landmarks support the 76-point constellation.
const REVISION: usize = 3;
/// A face box this close (normalised) to the image edge is probably cut off.
pub const BORDER_MARGIN: f64 = 0.01;

/// One face as Vision reports it: `bounding_box` is normalised with the origin at the bottom
/// left (y up), angles are in radians, landmark points are normalised to the bounding box.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VisionFace {
    pub bounding_box: Box4,
    pub confidence: f64,
    pub roll: Option<f64>,
    pub yaw: Option<f64>,
    pub pitch: Option<f64>,
    pub eyes: Option<VisionEyes>,
}

/// Eye contours and pupils of a [`VisionFace`] (Vision's "left" and "right").
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VisionEyes {
    pub left_eye: Vec<[f64; 2]>,
    pub right_eye: Vec<[f64; 2]>,
    pub left_pupil: [f64; 2],
    pub right_pupil: [f64; 2],
}

/// Where Vision may run the landmarks request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeDevice {
    NeuralEngine,
    Other,
}

/// The device to ask for: the first Neural Engine, if Vision lists one.
pub fn choose_device(devices: &[ComputeDevice]) -> Option<usize> {
    devices
        .iter()
        .position(|&d| d == ComputeDevice::NeuralEngine)
}

/// Vision's bottom-left-origin normalised rectangle as the observation's top-left-origin
/// `face_box`, clipped to the image.
pub fn image_box((x, y, w, h): Box4) -> Box4 {
    let (x0, y0) = (x.max(0.0), (1.0 - y - h).max(0.0));
    let (x1, y1) = ((x + w).min(1.0), (1.0 - y).min(1.0));
    (x0, y0, (x1 - x0).max(0.0), (y1 - y0).max(0.0))
}

/// A landmark point (normalised to `bounding_box`, y up) in image pixels (y down).
pub fn to_pixels(point: [f64; 2], (x, y, w, h): Box4, width: usize, height: usize) -> [f64; 2] {
    [
        (x + point[0] * w) * width as f64,
        (1.0 - (y + point[1] * h)) * height as f64,
    ]
}

/// The five points [`eye_metrics_of`] takes for one eye contour (pixels): the corners are the
/// contour's extremes along `u` (the image-left one first), the lids its extremes across it.
fn eye_points(contour: &[[f64; 2]], pupil: [f64; 2], u: [f64; 2]) -> [[f64; 2]; 5] {
    let along = |p: [f64; 2]| p[0] * u[0] + p[1] * u[1];
    let across = |p: [f64; 2]| p[1] * u[0] - p[0] * u[1]; // image-down for an upright face
    let lowest = |key: &dyn Fn([f64; 2]) -> f64| {
        contour
            .iter()
            .copied()
            .min_by(|&a, &b| key(a).total_cmp(&key(b)))
            .unwrap_or(pupil)
    };
    [
        lowest(&along),
        lowest(&|p| -along(p)),
        pupil,
        lowest(&across),
        lowest(&|p| -across(p)),
    ]
}

fn centre(points: &[[f64; 2]]) -> [f64; 2] {
    let n = points.len() as f64;
    let sum = points
        .iter()
        .fold([0.0, 0.0], |s, p| [s[0] + p[0], s[1] + p[1]]);
    [sum[0] / n, sum[1] / n]
}

/// Iris position and lid opening from Vision's eye contours and pupils, with the facemesh
/// backend's formula. Eyes are told apart by their place in the image, not by Vision's names.
pub fn vision_eye_metrics(
    eyes: &VisionEyes,
    bounding_box: Box4,
    width: usize,
    height: usize,
) -> Option<EyeMetrics> {
    let px = |points: &[[f64; 2]]| -> Vec<[f64; 2]> {
        points
            .iter()
            .map(|&p| to_pixels(p, bounding_box, width, height))
            .collect()
    };
    let mut contours = [px(&eyes.left_eye), px(&eyes.right_eye)];
    if contours.iter().any(|c| c.len() < 3) {
        return None;
    }
    let pupils = px(&[eyes.left_pupil, eyes.right_pupil]);
    let mut centres = contours.each_ref().map(|c| centre(c));
    if centres[0][0] > centres[1][0] {
        contours.swap(0, 1);
        centres.swap(0, 1);
    }
    let axis = [centres[1][0] - centres[0][0], centres[1][1] - centres[0][1]];
    let length = axis[0].hypot(axis[1]);
    if length.is_nan() || length < 1.0 {
        return None;
    }
    let u = [axis[0] / length, axis[1] / length];
    let distance = |p: [f64; 2], c: [f64; 2]| (p[0] - c[0]).hypot(p[1] - c[1]);
    let eye = |i: usize| {
        let c = centres[i];
        let pupil = if distance(pupils[0], c) <= distance(pupils[1], c) {
            pupils[0]
        } else {
            pupils[1]
        };
        eye_points(&contours[i], pupil, u)
    };
    eye_metrics_of([eye(0), eye(1)])
}

/// `(yaw, pitch, roll)` in degrees. Vision measures them in a right-handed frame with y up and
/// z towards the camera, which gives the facemesh backend's signs (see its module docs): roll
/// is positive for a counter-clockwise tilt, as the drawn-face test checks.
pub fn head_angles(face: &VisionFace) -> Option<(f64, f64, f64)> {
    Some((
        face.yaw?.to_degrees(),
        face.pitch?.to_degrees(),
        face.roll?.to_degrees(),
    ))
}

/// The face the features come from: largest, then most central.
fn primary_index(faces: &[VisionFace], width: usize, height: usize) -> Option<usize> {
    let score = |f: &VisionFace| {
        let (x, y, w, h) = image_box(f.bounding_box);
        let (wf, hf) = (width as f64, height as f64);
        primary_score(
            (w * wf).max(h * hf),
            (x + w / 2.0) * wf,
            (y + h / 2.0) * hf,
            width,
            height,
        )
    };
    (0..faces.len()).reduce(|best, i| {
        if score(&faces[i]) > score(&faces[best]) {
            i
        } else {
            best
        }
    })
}

/// What `annotate` needs from the last processed frame.
struct Drawing {
    timestamp: f64,
    /// Normalised (top-left origin) eye contours and pupils of the primary face.
    eyes: Vec<[f64; 2]>,
    pupils: Vec<[f64; 2]>,
    /// Boxes of the other faces.
    others: Vec<Box4>,
}

/// The Vision requests, created once and reused for every frame.
struct Requests {
    rectangles: Retained<VNDetectFaceRectanglesRequest>,
    landmarks: Retained<VNDetectFaceLandmarksRequest>,
}

/// Head pose and iris features from Apple's Vision framework.
///
/// `max_faces` is accepted for interface compatibility; Vision always finds every face, and
/// `face_count` reports all of them.
pub struct AppleBackend {
    requests: Option<Requests>,
    blink: BlinkDetector,
    last: Option<Drawing>,
    warned: bool,
}

impl AppleBackend {
    pub const NAME: &'static str = "apple";
    pub const FEATURE_NAMES: &'static [&'static str] =
        &["yaw", "pitch", "roll", "iris_h", "iris_v"];
    pub const FEATURE_VERSION: &'static str = "vision-pose-pupil-1";
    pub const GAZE_FEATURES: &'static [&'static str] = &["yaw", "pitch", "iris_h", "iris_v"];

    /// Fails with [`BackendUnavailable`] when Vision lacks revision 3 of the requests (before
    /// macOS 12).
    pub fn new(_max_faces: usize) -> Result<Self, BackendUnavailable> {
        let requests = autoreleasepool(|_| create_requests())?;
        Ok(Self {
            requests: Some(requests),
            blink: BlinkDetector::default(),
            last: None,
            warned: false,
        })
    }

    fn warn_once(&mut self, message: &str) {
        if !self.warned {
            self.warned = true;
            log::warn!("apple backend: {message}");
        }
    }

    /// The observation for Vision's faces (the pure half of `process`).
    fn observe(
        &mut self,
        faces: &[VisionFace],
        timestamp: f64,
        width: usize,
        height: usize,
    ) -> Observation {
        let mut obs = Observation::new(timestamp, faces.len() as u32);
        obs.frame_size = (width as u32, height as u32);
        let Some(primary) = primary_index(faces, width, height) else {
            self.last = None;
            return obs;
        };
        let face = &faces[primary];
        let face_box = image_box(face.bounding_box);
        obs.face_box = Some(face_box);
        let angles = head_angles(face);
        obs.head_yaw = angles.map(|a| a.0);
        obs.head_pitch = angles.map(|a| a.1);
        let eyes = face
            .eyes
            .as_ref()
            .and_then(|e| vision_eye_metrics(e, face.bounding_box, width, height));
        if let (Some((yaw, pitch, roll)), Some(eyes)) = (angles, eyes) {
            let vector = vec![yaw, pitch, roll, eyes.iris_h, eyes.iris_v];
            if vector.iter().all(|v| v.is_finite()) {
                obs.features = Some(vector);
                let (x, y, w, h) = face.bounding_box;
                let touches_border = x <= BORDER_MARGIN
                    || y <= BORDER_MARGIN
                    || x + w >= 1.0 - BORDER_MARGIN
                    || y + h >= 1.0 - BORDER_MARGIN;
                obs.quality = if touches_border {
                    face.confidence.min(0.5)
                } else {
                    face.confidence
                };
                obs.blink = self.blink.update(eyes.openness);
            }
        }
        let normalised = |points: &[[f64; 2]]| -> Vec<[f64; 2]> {
            points
                .iter()
                .map(|&p| to_pixels(p, face.bounding_box, 1, 1))
                .collect()
        };
        self.last = Some(Drawing {
            timestamp,
            eyes: face.eyes.as_ref().map_or_else(Vec::new, |e| {
                [normalised(&e.left_eye), normalised(&e.right_eye)].concat()
            }),
            pupils: face
                .eyes
                .as_ref()
                .map_or_else(Vec::new, |e| normalised(&[e.left_pupil, e.right_pupil])),
            others: faces
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != primary)
                .map(|(_, f)| image_box(f.bounding_box))
                .collect(),
        });
        obs
    }
}

impl VisionBackend for AppleBackend {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn feature_names(&self) -> &'static [&'static str] {
        Self::FEATURE_NAMES
    }

    fn feature_version(&self) -> &'static str {
        Self::FEATURE_VERSION
    }

    fn gaze_features(&self) -> &'static [&'static str] {
        Self::GAZE_FEATURES
    }

    /// Vision keeps nothing between frames; only the drawing goes. The blink baseline is kept,
    /// it belongs to the user rather than to a camera session (as in the facemesh backend).
    fn reset(&mut self) {
        self.last = None;
    }

    fn close(&mut self) {
        self.requests = None;
        self.last = None;
        self.blink.reset();
    }

    /// Panics after `close()`, like the other backends.
    fn process(&mut self, frame: &Frame, timestamp: f64) -> Observation {
        let started = Instant::now();
        let Some(requests) = &self.requests else {
            panic!("AppleBackend is closed");
        };
        let (width, height) = (frame.width, frame.height);
        let faces = autoreleasepool(|_| detect_faces(requests, frame));
        let mut obs = match faces {
            Ok(faces) => self.observe(&faces, timestamp, width, height),
            Err(err) => {
                self.warn_once(&err);
                self.last = None;
                let mut obs = Observation::new(timestamp, 0);
                obs.frame_size = (width as u32, height as u32);
                obs
            }
        };
        obs.inference_ms = started.elapsed().as_secs_f64() * 1e3;
        obs
    }

    fn annotate(&self, frame_bgr: &Frame, observation: &Observation) -> Frame {
        let mut out = frame_bgr.clone();
        let (w, h) = (out.width as f64, out.height as f64);
        let thickness = line_thickness(&out);
        if let Some(last) = &self.last
            && last.timestamp == observation.timestamp
        {
            for (points, color, radius) in [
                (&last.eyes, COLOR_POINT, thickness),
                (&last.pupils, COLOR_IRIS, thickness + 1),
            ] {
                for [x, y] in points {
                    let c = ((x * w).round() as i32, (y * h).round() as i32);
                    draw_filled_circle_mut(
                        &mut out,
                        c,
                        radius,
                        image::Rgb([color.0, color.1, color.2]),
                    );
                }
            }
            for &box_ in &last.others {
                draw_box(&mut out, box_, COLOR_OTHER, thickness);
            }
        }
        if let Some(face_box) = observation.face_box {
            draw_box(&mut out, face_box, COLOR_BOX, thickness);
        }
        draw_label(&mut out, &observation_label(observation));
        out
    }
}

// ----------------------------------------------------------------------------
// Vision calls

fn create_requests() -> Result<Requests, BackendUnavailable> {
    // SAFETY: plain constructors and class queries; `supportedRevisions` is a class method of
    // every VNRequest subclass that returns an NSIndexSet.
    let (rectangles, landmarks, revisions_ok) = unsafe {
        let rectangles = VNDetectFaceRectanglesRequest::new();
        let landmarks = VNDetectFaceLandmarksRequest::new();
        let revisions: Retained<NSIndexSet> =
            msg_send![VNDetectFaceRectanglesRequest::class(), supportedRevisions];
        let ok = revisions.containsIndex(REVISION)
            && VNDetectFaceLandmarksRequest::revision_supportsConstellation(
                REVISION,
                VNRequestFaceLandmarksConstellation::Constellation76Points,
            );
        (rectangles, landmarks, ok)
    };
    if !revisions_ok {
        return Err(BackendUnavailable(
            "Vision lacks revision 3 of the face requests (macOS 12 or later needed)".into(),
        ));
    }
    // SAFETY: the revision and constellation were checked as supported above.
    unsafe {
        rectangles.setRevision(REVISION);
        landmarks.setRevision(REVISION);
        landmarks.setConstellation(VNRequestFaceLandmarksConstellation::Constellation76Points);
    }
    prefer_neural_engine(&landmarks);
    Ok(Requests {
        rectangles,
        landmarks,
    })
}

/// Ask for the Neural Engine for the landmarks request when Vision offers it (macOS 14+).
fn prefer_neural_engine(request: &VNRequest) {
    if !request.respondsToSelector(sel!(supportedComputeStageDevicesAndReturnError:)) {
        return;
    }
    // SAFETY: a static NSString constant of Vision, present where the selector above is.
    let Some(stage) = (unsafe { VNComputeStageMain }) else {
        return;
    };
    // SAFETY: the method exists (checked above) and takes no arguments besides the error.
    let supported = unsafe { request.supportedComputeStageDevicesAndReturnError() };
    let devices = supported
        .ok()
        .and_then(|by_stage| by_stage.objectForKey(stage))
        .map(|list| list.to_vec())
        .unwrap_or_default();
    let kinds: Vec<ComputeDevice> = devices.iter().map(|d| device_kind(d)).collect();
    match choose_device(&kinds) {
        Some(i) => {
            // SAFETY: the device comes from the request's own list for this stage.
            unsafe { request.setComputeDevice_forComputeStage(Some(&devices[i]), stage) };
            log::info!("apple backend: landmarks on Neural Engine");
        }
        None => log::info!("apple backend: landmarks on CPU/GPU"),
    }
}

fn device_kind(device: &ProtocolObject<dyn MLComputeDeviceProtocol>) -> ComputeDevice {
    if device.isKindOfClass(MLNeuralEngineComputeDevice::class()) {
        ComputeDevice::NeuralEngine
    } else {
        ComputeDevice::Other
    }
}

/// The frame as a 32-bit BGRX `CGImage`, in memory.
fn cg_image(frame: &Frame) -> Option<CFRetained<CGImage>> {
    let (width, height) = (frame.width, frame.height);
    let mut bgrx = Vec::with_capacity(width * height * 4);
    for px in frame.data.as_chunks::<3>().0 {
        bgrx.extend_from_slice(&[px[0], px[1], px[2], 255]);
    }
    let provider = CGDataProvider::with_cf_data(Some(&CFData::from_bytes(&bgrx)))?;
    let space = CGColorSpace::new_device_rgb()?;
    let info =
        CGBitmapInfo(CGImageByteOrderInfo::Order32Little.0 | CGImageAlphaInfo::NoneSkipFirst.0);
    // SAFETY: the provider holds `height` rows of `width * 4` bytes in the layout `info`
    // describes; `decode` may be null.
    unsafe {
        CGImage::new(
            width,
            height,
            8,
            32,
            width * 4,
            Some(&space),
            info,
            Some(&provider),
            std::ptr::null(),
            false,
            CGColorRenderingIntent::RenderingIntentDefault,
        )
    }
}

/// Every face in the frame; the primary one with landmarks.
fn detect_faces(requests: &Requests, frame: &Frame) -> Result<Vec<VisionFace>, String> {
    let image = cg_image(frame).ok_or("could not wrap the frame in a CGImage")?;
    // SAFETY: an empty options dictionary is of the right type.
    let handler = unsafe {
        VNImageRequestHandler::initWithCGImage_options(
            VNImageRequestHandler::alloc(),
            &image,
            &NSDictionary::new(),
        )
    };
    let rectangles: &VNRequest = &requests.rectangles;
    handler
        .performRequests_error(&NSArray::from_slice(&[rectangles]))
        .map_err(|err| format!("face detection failed: {}", err.localizedDescription()))?;
    // SAFETY: the request has been performed; `results` is a plain getter.
    let found = unsafe { requests.rectangles.results() }.map_or_else(Vec::new, |r| r.to_vec());
    let mut faces: Vec<VisionFace> = found.iter().map(|f| vision_face(f)).collect();
    let Some(primary) = primary_index(&faces, frame.width, frame.height) else {
        return Ok(faces);
    };
    // SAFETY: the array holds a face observation of this very image.
    unsafe {
        requests
            .landmarks
            .setInputFaceObservations(Some(&NSArray::from_slice(&[&*found[primary]])));
    }
    let landmarks: &VNRequest = &requests.landmarks;
    handler
        .performRequests_error(&NSArray::from_slice(&[landmarks]))
        .map_err(|err| format!("face landmarks failed: {}", err.localizedDescription()))?;
    // SAFETY: as above.
    let read = unsafe { requests.landmarks.results() }.and_then(|r| r.firstObject());
    if let Some(read) = read {
        faces[primary] = vision_face(&read);
    }
    Ok(faces)
}

fn vision_face(face: &VNFaceObservation) -> VisionFace {
    // SAFETY: plain getters of a finished observation.
    let (rect, confidence, roll, yaw, pitch, landmarks) = unsafe {
        (
            face.boundingBox(),
            face.confidence(),
            face.roll(),
            face.yaw(),
            face.pitch(),
            face.landmarks(),
        )
    };
    let eyes = landmarks.and_then(|l| vision_eyes(&l));
    VisionFace {
        bounding_box: (
            rect.origin.x,
            rect.origin.y,
            rect.size.width,
            rect.size.height,
        ),
        confidence: f64::from(confidence),
        roll: roll.map(|n| n.as_f64()),
        yaw: yaw.map(|n| n.as_f64()),
        pitch: pitch.map(|n| n.as_f64()),
        eyes,
    }
}

fn vision_eyes(landmarks: &VNFaceLandmarks2D) -> Option<VisionEyes> {
    // SAFETY: plain getters of a finished observation's landmarks.
    let (left_eye, right_eye, left_pupil, right_pupil) = unsafe {
        (
            landmarks.leftEye()?,
            landmarks.rightEye()?,
            landmarks.leftPupil()?,
            landmarks.rightPupil()?,
        )
    };
    Some(VisionEyes {
        left_eye: region_points(&left_eye),
        right_eye: region_points(&right_eye),
        left_pupil: *region_points(&left_pupil).first()?,
        right_pupil: *region_points(&right_pupil).first()?,
    })
}

fn region_points(region: &VNFaceLandmarkRegion2D) -> Vec<[f64; 2]> {
    // SAFETY: `normalizedPoints` points at `pointCount` points owned by the region, which
    // outlives this call.
    unsafe {
        let (points, count) = (region.normalizedPoints(), region.pointCount());
        if points.is_null() {
            return Vec::new();
        }
        std::slice::from_raw_parts(points, count)
            .iter()
            .map(|p| [p.x, p.y])
            .collect()
    }
}
