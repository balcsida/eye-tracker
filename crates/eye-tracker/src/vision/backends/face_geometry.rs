//! MediaPipe's canonical face model, read from the Face Landmarker bundle.
//!
//! `geometry_pipeline_metadata_landmarks.binarypb` is a member of MediaPipe's
//! `face_landmarker.task` bundle (Apache-2.0, see `models/NOTICE.md`). It is a
//! serialised `GeometryPipelineMetadata` protocol buffer, defined in
//! `mediapipe/tasks/cc/vision/face_geometry/proto/geometry_pipeline_metadata.proto`
//! and `mesh_3d.proto`:
//!
//! ```text
//! message GeometryPipelineMetadata {
//!   optional InputSource input_source = 3;
//!   optional Mesh3d canonical_mesh = 1;
//!   repeated WeightedLandmarkRef procrustes_landmark_basis = 2;
//! }
//! message WeightedLandmarkRef { optional uint32 landmark_id = 1; optional float weight = 2; }
//! message Mesh3d {
//!   optional VertexType vertex_type = 1;        // VERTEX_PT = 0: x, y, z, u, v per vertex
//!   optional PrimitiveType primitive_type = 2;  // TRIANGLE = 0
//!   repeated float vertex_buffer = 3;
//!   repeated uint32 index_buffer = 4;
//! }
//! ```
//!
//! Only the canonical vertex positions and the Procrustes landmark basis (the
//! landmarks MediaPipe itself uses to fit the rigid head pose, with weights) are
//! needed, so a few lines of protobuf wire-format decoding replace a protobuf
//! dependency.
//!
//! The canonical mesh uses centimetres in a head-centred frame: `x` points to the
//! right of a camera image showing the face (the subject's left), `y` up and
//! `z` out of the face towards the camera.

use std::path::Path;

use thiserror::Error;

/// Landmarks in the canonical mesh (the landmark model adds 10 iris points).
pub const MESH_VERTICES: usize = 468;
/// VERTEX_PT: position (3) + texture coordinates (2).
const FLOATS_PER_VERTEX: usize = 5;

// Protobuf wire types.
const VARINT: u64 = 0;
const I64: u64 = 1;
const LEN: u64 = 2;
const I32: u64 = 5;

/// The geometry file is missing, truncated or not the expected message.
#[derive(Debug, Error, PartialEq, Eq)]
#[error("{0}")]
pub struct GeometryError(pub String);

fn error<T>(message: impl Into<String>) -> Result<T, GeometryError> {
    Err(GeometryError(message.into()))
}

/// Canonical face vertices and the landmark basis for rigid pose fitting.
#[derive(Debug, Clone, PartialEq)]
pub struct FaceGeometry {
    /// `(468, 3)` canonical vertex positions in centimetres (see module docs).
    pub vertices: Vec<[f64; 3]>,
    /// Landmark indices MediaPipe fits the head pose to (stable, rigid points).
    pub basis_ids: Vec<usize>,
    /// Relative weight of each basis landmark in MediaPipe's Procrustes fit.
    pub basis_weights: Vec<f64>,
}

/// The value of one protobuf field: a varint, or the raw bytes of a fixed-width or
/// length-delimited field.
#[derive(Clone, Copy)]
enum Value<'a> {
    Varint(u64),
    Bytes(&'a [u8]),
}

fn varint(buf: &[u8], mut pos: usize) -> Result<(u64, usize), GeometryError> {
    let (mut result, mut shift) = (0u64, 0u32);
    loop {
        if pos >= buf.len() || shift > 63 {
            return error("truncated varint");
        }
        let byte = buf[pos];
        pos += 1;
        result |= u64::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            return Ok((result, pos));
        }
        shift += 7;
    }
}

/// `(field number, wire type, value)` for every field of a message.
fn fields(buf: &[u8]) -> Result<Vec<(u64, u64, Value<'_>)>, GeometryError> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < buf.len() {
        let (key, next) = varint(buf, pos)?;
        pos = next;
        let (number, wire) = (key >> 3, key & 7);
        let size = match wire {
            VARINT => {
                let (value, next) = varint(buf, pos)?;
                pos = next;
                out.push((number, wire, Value::Varint(value)));
                continue;
            }
            I64 => 8,
            LEN => {
                let (size, next) = varint(buf, pos)?;
                pos = next;
                usize::try_from(size).unwrap_or(usize::MAX)
            }
            I32 => 4,
            _ => return error(format!("unsupported protobuf wire type {wire}")),
        };
        match pos.checked_add(size).filter(|&end| end <= buf.len()) {
            Some(end) => {
                out.push((number, wire, Value::Bytes(&buf[pos..end])));
                pos = end;
            }
            None => return error("truncated field"),
        }
    }
    Ok(out)
}

/// Decode a `repeated float` entry, packed or not.
fn floats(wire: u64, value: Value<'_>) -> Result<Vec<f64>, GeometryError> {
    match value {
        Value::Bytes(bytes) if (wire == I32 || wire == LEN) && bytes.len() % 4 == 0 => Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect()),
        _ => error("malformed float field"),
    }
}

/// Decode a serialised `GeometryPipelineMetadata` message.
///
/// Fails with [`GeometryError`] when the data is not a valid landmark geometry message.
pub fn parse_face_geometry(data: &[u8]) -> Result<FaceGeometry, GeometryError> {
    let mut vertex_buffer: Vec<f64> = Vec::new();
    let mut basis: Vec<(u64, f64)> = Vec::new();
    for (number, wire, value) in fields(data)? {
        let (1 | 2, LEN, Value::Bytes(message)) = (number, wire, value) else {
            continue;
        };
        if number == 1 {
            // canonical_mesh
            for (sub, sub_wire, sub_value) in fields(message)? {
                if sub == 1 && !matches!(sub_value, Value::Varint(0)) {
                    return error(format!(
                        "unsupported mesh vertex type {}",
                        match sub_value {
                            Value::Varint(v) => v.to_string(),
                            Value::Bytes(b) => format!("{b:?}"),
                        }
                    ));
                }
                if sub == 3 {
                    vertex_buffer.extend(floats(sub_wire, sub_value)?);
                }
            }
        } else {
            // procrustes_landmark_basis
            let (mut landmark_id, mut weight) = (0, 0.0);
            for (sub, sub_wire, sub_value) in fields(message)? {
                match (sub, sub_value) {
                    (1, Value::Varint(id)) => landmark_id = id,
                    (2, _) => {
                        let values = floats(sub_wire, sub_value)?;
                        if values.len() != 1 {
                            return error("malformed landmark weight");
                        }
                        weight = values[0];
                    }
                    _ => {}
                }
            }
            basis.push((landmark_id, weight));
        }
    }

    if vertex_buffer.len() != MESH_VERTICES * FLOATS_PER_VERTEX {
        return error(format!(
            "expected {MESH_VERTICES} mesh vertices, found {}",
            vertex_buffer.len() as f64 / FLOATS_PER_VERTEX as f64
        ));
    }
    let vertices: Vec<[f64; 3]> = vertex_buffer
        .as_chunks::<FLOATS_PER_VERTEX>()
        .0
        .iter()
        .map(|v| [v[0], v[1], v[2]])
        .collect();
    if !vertices.iter().flatten().all(|v| v.is_finite()) {
        return error("non-finite mesh vertices");
    }
    basis.retain(|&(id, weight)| id < MESH_VERTICES as u64 && weight > 0.0);
    if basis.len() < 6 {
        return error(format!(
            "only {} usable landmarks in the Procrustes basis",
            basis.len()
        ));
    }
    Ok(FaceGeometry {
        vertices,
        basis_ids: basis.iter().map(|&(id, _)| id as usize).collect(),
        basis_weights: basis.iter().map(|&(_, weight)| weight).collect(),
    })
}

/// Read and decode the geometry file at `path`.
///
/// Fails with [`GeometryError`] when the file cannot be read or decoded.
pub fn load_face_geometry(path: &Path) -> Result<FaceGeometry, GeometryError> {
    match std::fs::read(path) {
        Ok(data) => parse_face_geometry(&data),
        Err(err) => error(format!("cannot read {}: {err}", path.display())),
    }
}

#[cfg(test)]
#[path = "face_geometry_tests.rs"]
mod tests;
