//! Vision backends and backend selection.
//!
//! `facemesh` (478 landmarks with irises: head pose + eye direction) is preferred; `lite`
//! (YuNet's five landmarks: head pose only, lowest CPU) is the fallback. The model files are
//! embedded in the binary; nothing is downloaded at run time.

// TODO(port): src/eye_tracker/vision/backends/__init__.py (backend selection)

pub mod base;
pub mod face_geometry;
pub mod facemesh_backend;
pub mod lite_backend;
pub mod overlay;

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

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn test_embedded_models_match_the_pinned_checksums() {
        let embedded = [landmark_model(), geometry_metadata(), yunet_model()];
        for ((name, digest), bytes) in MODEL_FILES.iter().zip(embedded) {
            assert_eq!(hex::encode(Sha256::digest(bytes)), *digest, "{name}");
        }
    }
}
