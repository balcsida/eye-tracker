//! YuNet face detection (`face_detection_yunet_2023mar.onnx`) under tract-onnx, with the
//! post-processing of OpenCV's `FaceDetectorYN` (`modules/objdetect/src/face_detect.cpp`).
//!
//! The Python app ran YuNet through `cv2.FaceDetectorYN`; this module reproduces what it did:
//! the image is zero-padded right and bottom to multiples of 32 and fed as an NCHW BGR float blob
//! (0..255, no scaling, no channel swap), every prior is decoded, and `cv::dnn::NMSBoxes` runs on
//! the truncated integer boxes. One optimised plan is built per padded input size
//! ([`YuNet::set_input_size`], like `setInputSize`).

use tract_onnx::prelude::*;
use tract_onnx::tract_core::internal::ensure;

use crate::vision::{Frame, Grey};

const STRIDES: [usize; 3] = [8, 16, 32];

/// One detection, in OpenCV's row layout: `[x, y, w, h, re_x, re_y, le_x, le_y, nose_x, nose_y,
/// rm_x, rm_y, lm_x, lm_y, score]` in input-image pixels. YuNet's "right eye" is the one on the
/// image left.
pub type Face = [f32; 15];

pub struct YuNet {
    model: InferenceModel,
    plan: Option<(usize, usize, Arc<TypedRunnableModel>)>,
    input_size: (usize, usize),
    pub score_threshold: f32,
    pub nms_threshold: f32,
    pub top_k: usize,
}

impl YuNet {
    /// Load the ONNX model from memory for images of `input_size` (width, height), with the
    /// defaults of `cv2.FaceDetectorYN.create` (score 0.9, NMS 0.3, top_k 5000).
    pub fn from_bytes(model: &[u8], input_size: (usize, usize)) -> TractResult<YuNet> {
        // The file pins intermediate shapes to 640x640; dropping them makes the input size free.
        let model = tract_onnx::onnx()
            .with_ignore_value_info(true)
            .with_ignore_output_shapes(true)
            .model_for_read(&mut &model[..])?;
        let mut net = YuNet {
            model,
            plan: None,
            input_size: (0, 0),
            score_threshold: 0.9,
            nms_threshold: 0.3,
            top_k: 5000,
        };
        net.set_input_size(input_size)?;
        Ok(net)
    }

    /// Like `FaceDetectorYN::setInputSize`: (width, height) of the images passed to `detect`.
    pub fn set_input_size(&mut self, (w, h): (usize, usize)) -> TractResult<()> {
        let (pw, ph) = (pad32(w), pad32(h));
        if !matches!(self.plan, Some((a, b, _)) if (a, b) == (pw, ph)) {
            let plan = self
                .model
                .clone()
                .with_input_fact(0, f32::fact([1, 3, ph, pw]).into())?
                .into_optimized()?
                .into_runnable()?;
            self.plan = Some((pw, ph, plan));
        }
        self.input_size = (w, h);
        Ok(())
    }

    pub fn input_size(&self) -> (usize, usize) {
        self.input_size
    }

    /// The raw network outputs for an image of the input size, in OpenCV's order:
    /// cls_8/16/32, obj_8/16/32, bbox_8/16/32, kps_8/16/32.
    pub fn forward(&self, image: &Frame) -> TractResult<TVec<TValue>> {
        let (w, h) = self.input_size;
        ensure!(
            (image.width, image.height) == (w, h),
            "expected a {w}x{h} image, got {}x{}",
            image.width,
            image.height
        );
        let Some((pw, ph, plan)) = self.plan.as_ref() else {
            anyhow::bail!("no input size set");
        };
        let bgr = &image.data;
        // blobFromImage of the zero-padded image: NCHW, BGR, 0..255, no scaling.
        let input = tract_ndarray::Array4::from_shape_fn((1, 3, *ph, *pw), |(_, c, y, x)| {
            if x < w && y < h {
                f32::from(bgr[(y * w + x) * 3 + c])
            } else {
                0.0
            }
        });
        plan.run(tvec!(Tensor::from(input).into()))
    }

    /// `FaceDetectorYN::detect`: the faces after thresholding and NMS, best first.
    pub fn detect(&self, image: &Frame) -> TractResult<Vec<Face>> {
        let outputs = self.forward(image)?;
        let Some((pw, ph, _)) = self.plan.as_ref() else {
            anyhow::bail!("no input size set");
        };
        let faces = decode(&outputs, *pw, *ph)?;
        Ok(nms(
            faces,
            self.score_threshold,
            self.nms_threshold,
            self.top_k,
        ))
    }
}

fn pad32(v: usize) -> usize {
    ((v.max(1) - 1) / 32 + 1) * 32
}

/// Every prior's face, before thresholding (`FaceDetectorYN::postProcess`).
pub fn decode(outputs: &[TValue], pad_w: usize, pad_h: usize) -> TractResult<Vec<Face>> {
    ensure!(
        outputs.len() == 12,
        "YuNet has 12 outputs, got {}",
        outputs.len()
    );
    let mut faces = vec![];
    for (i, &stride) in STRIDES.iter().enumerate() {
        let (cols, rows) = (pad_w / stride, pad_h / stride);
        let view = |k: usize| outputs[i + 3 * k].to_plain_array_view::<f32>();
        let (cls, obj, bbox, kps) = (view(0)?, view(1)?, view(2)?, view(3)?);
        let (Some(cls), Some(obj), Some(bbox), Some(kps)) = (
            cls.as_slice(),
            obj.as_slice(),
            bbox.as_slice(),
            kps.as_slice(),
        ) else {
            anyhow::bail!("non-contiguous YuNet output at stride {stride}");
        };
        let n = rows * cols;
        ensure!(
            cls.len() == n && obj.len() == n && bbox.len() == n * 4 && kps.len() == n * 10,
            "unexpected output size at stride {stride}"
        );
        let s = stride as f32;
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                let mut face = [0f32; 15];
                face[14] = (cls[idx].clamp(0.0, 1.0) * obj[idx].clamp(0.0, 1.0)).sqrt();
                let b = &bbox[idx * 4..idx * 4 + 4];
                let (cx, cy) = ((c as f32 + b[0]) * s, (r as f32 + b[1]) * s);
                let (bw, bh) = (b[2].exp() * s, b[3].exp() * s);
                face[..4].copy_from_slice(&[cx - bw / 2.0, cy - bh / 2.0, bw, bh]);
                for n in 0..5 {
                    face[4 + 2 * n] = (kps[idx * 10 + 2 * n] + c as f32) * s;
                    face[5 + 2 * n] = (kps[idx * 10 + 2 * n + 1] + r as f32) * s;
                }
                faces.push(face);
            }
        }
    }
    Ok(faces)
}

/// `cv::dnn::NMSBoxes` on the integer boxes OpenCV builds (`Rect2i` of truncated values),
/// eta = 1.
pub fn nms(faces: Vec<Face>, score_threshold: f32, nms_threshold: f32, top_k: usize) -> Vec<Face> {
    let mut candidates: Vec<Face> = faces
        .into_iter()
        .filter(|f| f[14] > score_threshold)
        .collect();
    // Stable sort, highest score first, like cv::dnn's GetMaxScoreIndex.
    candidates.sort_by(|a, b| b[14].total_cmp(&a[14]));
    candidates.truncate(top_k);
    let rect = |f: &Face| [f[0] as i32, f[1] as i32, f[2] as i32, f[3] as i32];
    let mut kept: Vec<Face> = vec![];
    for f in candidates {
        let a = rect(&f);
        if kept.iter().all(|k| iou(a, rect(k)) <= nms_threshold) {
            kept.push(f);
        }
    }
    kept
}

fn iou(a: [i32; 4], b: [i32; 4]) -> f32 {
    let w = (a[0] + a[2]).min(b[0] + b[2]) - a[0].max(b[0]);
    let h = (a[1] + a[3]).min(b[1] + b[3]) - a[1].max(b[1]);
    if w <= 0 || h <= 0 {
        return 0.0;
    }
    let inter = (w * h) as f32;
    inter / ((a[2] * a[3] + b[2] * b[3]) as f32 - inter)
}

/// `cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)` for a BGR frame (the
/// downscaling before detection), one channel at a time.
pub fn resize_area(frame: &Frame, width: usize, height: usize) -> Frame {
    let planes: Vec<Grey> = (0..3)
        .map(|c| {
            Grey {
                width: frame.width,
                height: frame.height,
                data: frame.data.iter().skip(c).step_by(3).copied().collect(),
            }
            .resize_area(width, height)
        })
        .collect();
    let data = (0..width * height)
        .flat_map(|i| [planes[0].data[i], planes[1].data[i], planes[2].data[i]])
        .collect();
    Frame {
        width,
        height,
        data,
    }
}

#[cfg(test)]
#[path = "yunet_tests.rs"]
mod tests;
