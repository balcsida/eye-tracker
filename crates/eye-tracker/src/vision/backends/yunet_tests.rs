//! Unit tests from the vision spike, plus the detector on drawn and blank images.

use super::*;
use crate::testing::face_drawing::{FaceParams, draw_face};
use crate::vision::backends::yunet_model;

fn face(x: f32, y: f32, side: f32, score: f32) -> Face {
    let mut f = [0f32; 15];
    f[..4].copy_from_slice(&[x, y, side, side]);
    f[14] = score;
    f
}

#[test]
fn nms_keeps_best_of_overlapping_and_separate_faces() {
    let faces = vec![
        face(10.0, 10.0, 100.0, 0.92),
        face(15.0, 12.0, 100.0, 0.97),
        face(300.0, 10.0, 80.0, 0.95),
        face(0.0, 0.0, 50.0, 0.5),
    ];
    let kept = nms(faces, 0.9, 0.3, 5000);
    let scores: Vec<f32> = kept.iter().map(|f| f[14]).collect();
    assert_eq!(scores, vec![0.97, 0.95]);
}

#[test]
fn iou_matches_hand_computation() {
    // 100x100 boxes offset by 50 px horizontally: 5000 / 15000.
    assert!((iou([0, 0, 100, 100], [50, 0, 100, 100]) - 1.0 / 3.0).abs() < 1e-6);
    assert_eq!(iou([0, 0, 10, 10], [20, 20, 10, 10]), 0.0);
}

#[test]
fn pad32_rounds_up_like_opencv() {
    assert_eq!(
        (pad32(320), pad32(240), pad32(321), pad32(480)),
        (320, 256, 352, 480)
    );
}

#[test]
fn blank_image_has_no_face_and_a_drawn_face_one() {
    let mut net = YuNet::from_bytes(yunet_model(), (320, 240)).unwrap();
    net.score_threshold = 0.6;
    assert!(
        net.detect(&Frame::filled(320, 240, [0; 3]))
            .unwrap()
            .is_empty()
    );
    let small = resize_area(&draw_face(None, &FaceParams::default()), 320, 240);
    let faces = net.detect(&small).unwrap();
    assert_eq!(faces.len(), 1);
    let f = faces[0];
    // Centred, about 80 px wide; the right eye (image left) left of the left eye.
    assert!((f[0] + f[2] / 2.0 - 160.0).abs() < 10.0, "{f:?}");
    assert!(f[2] > 50.0 && f[2] < 120.0, "{f:?}");
    assert!(f[4] < f[6], "{f:?}");
}

#[test]
fn detect_rejects_an_image_of_another_size() {
    let net = YuNet::from_bytes(yunet_model(), (320, 240)).unwrap();
    assert!(net.detect(&Frame::filled(64, 48, [0; 3])).is_err());
}

#[test]
fn resize_area_halves_by_block_means() {
    let mut frame = Frame::filled(4, 2, [0; 3]);
    frame.set_pixel(0, 0, [100, 0, 200]);
    frame.set_pixel(1, 1, [100, 4, 0]);
    let small = resize_area(&frame, 2, 1);
    assert_eq!((small.width, small.height), (2, 1));
    assert_eq!(small.pixel(0, 0), [50, 1, 50]);
    assert_eq!(small.pixel(1, 0), [0, 0, 0]);
}
