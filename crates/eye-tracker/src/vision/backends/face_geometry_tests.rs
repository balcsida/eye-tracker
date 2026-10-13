// Ported from tests/test_face_geometry.py (a minimal protobuf decoder). The pytest parametrize
// cases are loops; `paths.model_path(GEOMETRY_FILE)` is `geometry_metadata()`, the embedded copy.

use super::*;
use crate::vision::backends::geometry_metadata;

// ------------------------------------------------------------ protobuf encoder
fn varint(mut value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            out.push(byte | 0x80);
        } else {
            out.push(byte);
            return out;
        }
    }
}

fn key(number: u64, wire: u64) -> Vec<u8> {
    varint(number << 3 | wire)
}

fn len_field(number: u64, payload: &[u8]) -> Vec<u8> {
    [
        key(number, 2),
        varint(payload.len() as u64),
        payload.to_vec(),
    ]
    .concat()
}

/// A `Mesh3d` message with VERTEX_PT vertices (u, v set to zero).
fn mesh(vertices: &[[f32; 3]], packed: bool, vertex_type: u64) -> Vec<u8> {
    let floats: Vec<f32> = vertices
        .iter()
        .flat_map(|v| [v[0], v[1], v[2], 0.0, 0.0])
        .collect();
    let mut body = [key(1, 0), varint(vertex_type)].concat();
    if packed {
        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        body.extend(len_field(3, &bytes));
    } else {
        for f in &floats {
            body.extend(key(3, 5));
            body.extend(f.to_le_bytes());
        }
    }
    [body, key(4, 0), varint(7)].concat() // an index-buffer entry, ignored
}

fn landmark(landmark_id: u64, weight: f32) -> Vec<u8> {
    [
        key(1, 0),
        varint(landmark_id),
        key(2, 5),
        weight.to_le_bytes().to_vec(),
    ]
    .concat()
}

fn message(vertices: &[[f32; 3]], basis: &[(u64, f32)], packed: bool, vertex_type: u64) -> Vec<u8> {
    let mut out = [key(3, 0), varint(1)].concat(); // input_source, ignored
    out.extend(len_field(1, &mesh(vertices, packed, vertex_type)));
    for &(landmark_id, weight) in basis {
        out.extend(len_field(2, &landmark(landmark_id, weight)));
    }
    out
}

const BASIS: [(u64, f32); 7] = [
    (4, 0.07),
    (6, 0.12),
    (33, 0.05),
    (133, 0.05),
    (263, 0.05),
    (362, 0.05),
    (1, 0.01),
];

/// Reproducible pseudo-random vertices (an LCG standing in for numpy's generator).
fn vertices() -> Vec<[f32; 3]> {
    let mut state = 3u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 17.0
    };
    (0..MESH_VERTICES)
        .map(|_| [next(), next(), next()])
        .collect()
}

fn parse_error(data: &[u8]) -> String {
    parse_face_geometry(data).unwrap_err().to_string()
}

// -------------------------------------------------------------------- parsing
#[test]
fn test_parse_round_trip() {
    for packed in [true, false] {
        let vertices = vertices();
        let geometry = parse_face_geometry(&message(&vertices, &BASIS, packed, 0)).unwrap();
        for (got, want) in geometry.vertices.iter().zip(&vertices) {
            for (g, w) in got.iter().zip(want) {
                assert_eq!(*g, f64::from(*w));
            }
        }
        assert_eq!(geometry.vertices.len(), vertices.len());
        let ids: Vec<usize> = BASIS.iter().map(|&(i, _)| i as usize).collect();
        assert_eq!(geometry.basis_ids, ids);
        let weights: Vec<f64> = BASIS.iter().map(|&(_, w)| f64::from(w)).collect();
        assert_eq!(geometry.basis_weights, weights);
    }
}

#[test]
fn test_unusable_basis_entries_are_dropped() {
    let mut basis = BASIS.to_vec();
    basis.extend([(MESH_VERTICES as u64 + 3, 0.5), (10, 0.0)]);
    let geometry = parse_face_geometry(&message(&vertices(), &basis, true, 0)).unwrap();
    let ids: Vec<usize> = BASIS.iter().map(|&(i, _)| i as usize).collect();
    assert_eq!(geometry.basis_ids, ids);
}

#[test]
fn test_malformed_data() {
    let cases: [(Vec<u8>, &str); 4] = [
        (vec![], "expected 468 mesh vertices"),
        (vec![0x80], "truncated varint"),
        (key(9, 3), "unsupported protobuf wire type"),
        (
            [key(1, 2), varint(50), b"abc".to_vec()].concat(),
            "truncated field",
        ),
    ];
    for (data, expected) in cases {
        let message = parse_error(&data);
        assert!(message.contains(expected), "{message}");
    }
}

#[test]
fn test_wrong_vertex_type() {
    let message = parse_error(&message(&vertices(), &BASIS, true, 1));
    assert!(message.contains("vertex type"), "{message}");
}

#[test]
fn test_too_few_vertices() {
    let message = parse_error(&message(&vertices()[..100], &BASIS, true, 0));
    assert!(message.contains("expected 468"), "{message}");
}

#[test]
fn test_too_small_basis() {
    let message = parse_error(&message(&vertices(), &BASIS[..3], true, 0));
    assert!(message.contains("Procrustes basis"), "{message}");
}

#[test]
fn test_non_finite_vertices() {
    let mut vertices = vertices();
    vertices[5][1] = f32::NAN;
    let message = parse_error(&message(&vertices, &BASIS, true, 0));
    assert!(message.contains("non-finite"), "{message}");
}

#[test]
fn test_missing_file() {
    let tmp = tempfile::tempdir().unwrap();
    let err = load_face_geometry(&tmp.path().join("missing.binarypb")).unwrap_err();
    assert!(err.to_string().contains("cannot read"), "{err}");
}

// ---------------------------------------------------------- the bundled model
#[test]
fn test_bundled_canonical_face() {
    let geometry = parse_face_geometry(geometry_metadata()).unwrap();
    let v = &geometry.vertices;
    assert_eq!(v.len(), MESH_VERTICES);
    assert!(v.iter().flatten().all(|c| c.is_finite()));
    // Centimetres, head-centred: a face is ~15 cm wide and symmetric about x = 0.
    let xs = || v.iter().map(|p| p[0]);
    let width = xs().fold(f64::MIN, f64::max) - xs().fold(f64::MAX, f64::min);
    assert!(10.0 < width && width < 20.0);
    assert!((xs().sum::<f64>() / v.len() as f64).abs() < 0.5);
    // Orientation: the subject's right eye corner (33) is on the image left,
    // y points up (forehead above chin) and z towards the camera (nose tip in front).
    assert!(v[33][0] < 0.0 && 0.0 < v[263][0]);
    assert!(v[10][1] > v[152][1]);
    let mean_z = v.iter().map(|p| p[2]).sum::<f64>() / v.len() as f64;
    assert!(v[4][2] > mean_z);
    // MediaPipe fits the pose to a set of stable landmarks, with positive weights.
    let ids = &geometry.basis_ids;
    assert!(6 <= ids.len() && ids.len() < MESH_VERTICES);
    let unique: std::collections::HashSet<_> = ids.iter().collect();
    assert_eq!(unique.len(), ids.len());
    assert!(geometry.basis_weights.iter().all(|&w| w > 0.0));
}
