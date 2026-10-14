//! Vision backends and backend selection.
//!
//! `facemesh` (478 landmarks with irises: head pose + eye direction) is preferred; `lite`
//! (YuNet's five landmarks: head pose only, lowest CPU) is the fallback. Both run in-process with
//! tract (the TFLite landmark network and the ONNX YuNet detector), and nothing touches the
//! network. The model files are embedded in the binary; nothing is downloaded at run time. On
//! macOS, `apple` (only when chosen) runs Apple's Vision framework instead: its own face models,
//! head pose and pupils, no model files.

#[cfg(target_os = "macos")]
pub mod apple_backend;
pub mod base;
pub mod face_geometry;
pub mod facemesh_backend;
pub mod lite_backend;
pub mod overlay;
pub mod pnp;
pub mod yunet;

#[cfg(target_os = "macos")]
pub use apple_backend::AppleBackend;
pub use base::{BackendUnavailable, VisionBackend};
pub use facemesh_backend::FaceMeshBackend;
pub use lite_backend::LiteBackend;

/// Backend names in order of preference (the order `"auto"` tries them in).
#[cfg(not(target_os = "macos"))]
pub const BACKEND_NAMES: &[&str] = &["facemesh", "lite"];
#[cfg(target_os = "macos")]
pub const BACKEND_NAMES: &[&str] = &["facemesh", "lite", "apple"];

/// Every bundled model file and its pinned SHA-256 (see `models/NOTICE.md` for provenance;
/// `cargo xtask fetch-models` pins the same values).
pub const MODEL_FILES: [(&str, &str); 3] = [
    (
        "face_landmarks_detector.tflite",
        "c7d54204ce0448474c7f3fa9af494787c0965cbdd6f20fc72867e43046bd43d5",
    ),
    (
        "geometry_pipeline_metadata_landmarks.binarypb",
        "bdbcda96dfcb7da883da124aaa2c55dee49770d934f0fcc71747f8c21bdc75b4",
    ),
    (
        "face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
];

/// The model files each backend needs.
pub const BACKEND_MODEL_FILES: &[(&str, &[&str])] = &[
    (
        "facemesh",
        &[
            "face_landmarks_detector.tflite",
            "geometry_pipeline_metadata_landmarks.binarypb",
            "face_detection_yunet_2023mar.onnx",
        ],
    ),
    ("lite", &["face_detection_yunet_2023mar.onnx"]),
    #[cfg(target_os = "macos")]
    ("apple", &[]),
];

// Names used by Eye Tracker 0.1 settings files and scripts.
const LEGACY_NAMES: [(&str, &str); 2] = [("mediapipe", "facemesh"), ("opencv", "lite")];

/// MediaPipe face landmark network (TFLite).
pub fn landmark_model() -> &'static [u8] {
    include_bytes!("../../../models/face_landmarks_detector.tflite")
}

/// YuNet face detector (ONNX).
pub fn yunet_model() -> &'static [u8] {
    include_bytes!("../../../models/face_detection_yunet_2023mar.onnx")
}

/// MediaPipe's canonical face geometry (serialized protobuf).
pub fn geometry_metadata() -> &'static [u8] {
    include_bytes!("../../../models/geometry_pipeline_metadata_landmarks.binarypb")
}

/// The embedded bytes of a model file named in [`MODEL_FILES`].
pub fn model_bytes(name: &str) -> Option<&'static [u8]> {
    match name {
        "face_landmarks_detector.tflite" => Some(landmark_model()),
        "geometry_pipeline_metadata_landmarks.binarypb" => Some(geometry_metadata()),
        "face_detection_yunet_2023mar.onnx" => Some(yunet_model()),
        _ => None,
    }
}

/// What the Python app's backend classes told without an instance: the metadata of the backend
/// [`create_backend`] would create.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendClass {
    FaceMesh,
    Lite,
    #[cfg(target_os = "macos")]
    Apple,
}

impl BackendClass {
    pub fn name(self) -> &'static str {
        match self {
            Self::FaceMesh => FaceMeshBackend::NAME,
            Self::Lite => LiteBackend::NAME,
            #[cfg(target_os = "macos")]
            Self::Apple => AppleBackend::NAME,
        }
    }

    pub fn feature_names(self) -> &'static [&'static str] {
        match self {
            Self::FaceMesh => FaceMeshBackend::FEATURE_NAMES,
            Self::Lite => LiteBackend::FEATURE_NAMES,
            #[cfg(target_os = "macos")]
            Self::Apple => AppleBackend::FEATURE_NAMES,
        }
    }

    pub fn feature_version(self) -> &'static str {
        match self {
            Self::FaceMesh => FaceMeshBackend::FEATURE_VERSION,
            Self::Lite => LiteBackend::FEATURE_VERSION,
            #[cfg(target_os = "macos")]
            Self::Apple => AppleBackend::FEATURE_VERSION,
        }
    }

    pub fn gaze_features(self) -> &'static [&'static str] {
        match self {
            Self::FaceMesh => FaceMeshBackend::GAZE_FEATURES,
            Self::Lite => LiteBackend::GAZE_FEATURES,
            #[cfg(target_os = "macos")]
            Self::Apple => AppleBackend::GAZE_FEATURES,
        }
    }
}

/// Names of the backends whose model files are present.
///
/// This is a cheap check; a backend listed here can still fail to load (for example a corrupt
/// model), which [`create_backend`] reports. With the models embedded, all are always listed.
pub fn available_backends() -> Vec<&'static str> {
    available_among(|file| model_bytes(file).is_some_and(|bytes| !bytes.is_empty()))
}

fn available_among(present: impl Fn(&str) -> bool) -> Vec<&'static str> {
    BACKEND_MODEL_FILES
        .iter()
        .filter(|(_, files)| files.iter().all(|f| present(f)))
        .map(|&(name, _)| name)
        .collect()
}

/// The backend class `create_backend(name)` would try first, without creating it.
///
/// Useful to learn `feature_version` (for calibration compatibility checks) before the worker
/// thread has created the backend.
///
/// Fails with [`BackendUnavailable`] for an unknown name, or `"auto"` with no backend available.
pub fn backend_class(name: &str) -> Result<BackendClass, BackendUnavailable> {
    class_among(name, &available_backends())
}

fn class_among(name: &str, available: &[&str]) -> Result<BackendClass, BackendUnavailable> {
    let mut key = normalise(name)?;
    if key == "auto" {
        key = available
            .first()
            .copied()
            .ok_or_else(|| BackendUnavailable(nothing_available_message()))?;
    }
    Ok(class(key))
}

/// Create a vision backend.
///
/// `"auto"` tries `facemesh` first and falls back to `lite` (and on macOS then to `apple`). Call
/// this on the thread that will use the backend.
///
/// Fails with [`BackendUnavailable`] when the requested backend (or, for `"auto"`, every backend)
/// cannot be created. The message says why.
pub fn create_backend(
    name: &str,
    max_faces: usize,
) -> Result<Box<dyn VisionBackend>, BackendUnavailable> {
    create_with(name, max_faces, create)
}

fn create_with(
    name: &str,
    max_faces: usize,
    create: impl Fn(&str, usize) -> Result<Box<dyn VisionBackend>, BackendUnavailable>,
) -> Result<Box<dyn VisionBackend>, BackendUnavailable> {
    let key = normalise(name)?;
    if key != "auto" {
        return create(key, max_faces);
    }
    let mut reasons = Vec::new();
    for &candidate in BACKEND_NAMES {
        match create(candidate, max_faces) {
            Ok(backend) => {
                log::info!("Using the {} vision backend", backend.name());
                return Ok(backend);
            }
            Err(err) => {
                log::info!("Vision backend {candidate} unavailable: {err}");
                reasons.push(format!("{candidate}: {err}"));
            }
        }
    }
    Err(BackendUnavailable(format!(
        "No vision backend is available — {}",
        reasons.join("; ")
    )))
}

fn normalise(name: &str) -> Result<&'static str, BackendUnavailable> {
    let key = name.trim().to_lowercase();
    let key = LEGACY_NAMES
        .iter()
        .find(|(legacy, _)| *legacy == key)
        .map_or(key.as_str(), |&(_, new)| new);
    if key == "auto" {
        return Ok("auto");
    }
    #[cfg(not(target_os = "macos"))]
    if key == "apple" {
        return Err(BackendUnavailable(
            "The apple vision backend needs macOS".into(),
        ));
    }
    BACKEND_NAMES
        .iter()
        .copied()
        .find(|&n| n == key)
        .ok_or_else(|| {
            let choices = ["auto"].iter().chain(BACKEND_NAMES).copied();
            BackendUnavailable(format!(
                "Unknown vision backend {name:?}; choose one of {}",
                choices.collect::<Vec<_>>().join(", ")
            ))
        })
}

fn class(name: &str) -> BackendClass {
    match name {
        "facemesh" => BackendClass::FaceMesh,
        #[cfg(target_os = "macos")]
        "apple" => BackendClass::Apple,
        _ => BackendClass::Lite,
    }
}

fn create(name: &str, max_faces: usize) -> Result<Box<dyn VisionBackend>, BackendUnavailable> {
    match name {
        "facemesh" => Ok(Box::new(FaceMeshBackend::new(max_faces)?)),
        #[cfg(target_os = "macos")]
        "apple" => Ok(Box::new(AppleBackend::new(max_faces)?)),
        _ => Ok(Box::new(LiteBackend::new(max_faces)?)),
    }
}

fn nothing_available_message() -> String {
    "No vision backend is available: the models must be present (cargo xtask fetch-models)"
        .to_string()
}

#[cfg(test)]
#[path = "backends_tests.rs"]
pub(crate) mod tests;
