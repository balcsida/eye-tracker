//! Camera capture and face analysis.
//!
//! The pipeline works on [`Frame`]s: BGR8 images like the ones OpenCV handed around in the
//! Python app. The helpers below (greyscale, `INTER_AREA` resizing, absolute difference, mean and
//! standard deviation) are the few image operations the motion gate needs.

pub mod camera;
pub mod motion;
pub mod threads;
pub mod worker;

pub mod backends;

/// A BGR8 image, row-major, three bytes per pixel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl Frame {
    /// A frame of one colour, given as `[b, g, r]`.
    pub fn filled(width: usize, height: usize, bgr: [u8; 3]) -> Self {
        Self {
            width,
            height,
            data: bgr.repeat(width * height),
        }
    }

    /// Wrap a BGR buffer; `None` unless it holds exactly `width * height` pixels.
    pub fn from_bgr(width: usize, height: usize, data: Vec<u8>) -> Option<Self> {
        (data.len() == width * height * 3).then_some(Self {
            width,
            height,
            data,
        })
    }

    pub fn from_rgb_image(image: &image::RgbImage) -> Self {
        let mut data = image.as_raw().clone();
        data.as_chunks_mut::<3>()
            .0
            .iter_mut()
            .for_each(|px| px.swap(0, 2));
        Self {
            width: image.width() as usize,
            height: image.height() as usize,
            data,
        }
    }

    pub fn to_rgb_image(&self) -> image::RgbImage {
        image::RgbImage::from_fn(self.width as u32, self.height as u32, |x, y| {
            let [b, g, r] = self.pixel(x as usize, y as usize);
            image::Rgb([r, g, b])
        })
    }

    /// `[b, g, r]` of one pixel.
    pub fn pixel(&self, x: usize, y: usize) -> [u8; 3] {
        let i = (y * self.width + x) * 3;
        [self.data[i], self.data[i + 1], self.data[i + 2]]
    }

    pub fn set_pixel(&mut self, x: usize, y: usize, bgr: [u8; 3]) {
        let i = (y * self.width + x) * 3;
        self.data[i..i + 3].copy_from_slice(&bgr);
    }

    /// Greyscale with OpenCV's fixed-point `COLOR_BGR2GRAY` weights.
    pub fn to_grey(&self) -> Grey {
        let data = self
            .data
            .as_chunks::<3>()
            .0
            .iter()
            .map(|px| {
                let (b, g, r) = (u32::from(px[0]), u32::from(px[1]), u32::from(px[2]));
                ((b * 1868 + g * 9617 + r * 4899 + (1 << 13)) >> 14) as u8
            })
            .collect();
        Grey {
            width: self.width,
            height: self.height,
            data,
        }
    }
}

/// imageproc draws onto a frame directly; the `Rgb` triple is the stored `[b, g, r]` order.
impl imageproc::drawing::Canvas for Frame {
    type Pixel = image::Rgb<u8>;

    fn dimensions(&self) -> (u32, u32) {
        (self.width as u32, self.height as u32)
    }

    fn get_pixel(&self, x: u32, y: u32) -> Self::Pixel {
        image::Rgb(self.pixel(x as usize, y as usize))
    }

    fn draw_pixel(&mut self, x: u32, y: u32, color: Self::Pixel) {
        self.set_pixel(x as usize, y as usize, color.0);
    }
}

/// An 8-bit greyscale image, row-major.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grey {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl Grey {
    /// The pixels `x0..x1` × `y0..y1` (the caller keeps them inside the image).
    pub fn crop(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> Grey {
        let mut data = Vec::with_capacity((x1 - x0) * (y1 - y0));
        for row in y0..y1 {
            data.extend_from_slice(&self.data[row * self.width + x0..row * self.width + x1]);
        }
        Grey {
            width: x1 - x0,
            height: y1 - y0,
            data,
        }
    }

    /// `cv2.resize(..., INTER_AREA)`: every output pixel is the mean of the source area it
    /// covers when shrinking, and linear interpolation when enlarging; the result is rounded
    /// back to 8 bits.
    pub fn resize_area(&self, width: usize, height: usize) -> Grey {
        let columns = axis_weights(self.width, width);
        let rows = axis_weights(self.height, height);
        // Horizontal pass into f32 rows, then vertical.
        let mut tmp = vec![0f32; width * self.height];
        for y in 0..self.height {
            let src = &self.data[y * self.width..(y + 1) * self.width];
            for (x, taps) in columns.iter().enumerate() {
                tmp[y * width + x] = taps.iter().map(|&(i, w)| f32::from(src[i]) * w).sum();
            }
        }
        let mut data = Vec::with_capacity(width * height);
        for taps in &rows {
            for x in 0..width {
                let v: f32 = taps.iter().map(|&(i, w)| tmp[i * width + x] * w).sum();
                data.push(v.round().clamp(0.0, 255.0) as u8);
            }
        }
        Grey {
            width,
            height,
            data,
        }
    }

    pub fn to_plane(&self) -> Plane {
        Plane {
            width: self.width,
            height: self.height,
            data: self.data.iter().map(|&v| f32::from(v)).collect(),
        }
    }
}

/// For each output index the `(source index, weight)` taps of an `INTER_AREA` resize.
fn axis_weights(src: usize, dst: usize) -> Vec<Vec<(usize, f32)>> {
    let scale = src as f64 / dst as f64;
    (0..dst)
        .map(|i| {
            if scale >= 1.0 {
                let (start, end) = (i as f64 * scale, (i + 1) as f64 * scale);
                let first = start.floor() as usize;
                let last = (end.ceil() as usize).min(src);
                (first..last)
                    .map(|s| {
                        let overlap = end.min((s + 1) as f64) - start.max(s as f64);
                        (s, (overlap / scale) as f32)
                    })
                    .filter(|&(_, w)| w > 0.0)
                    .collect()
            } else {
                let pos = ((i as f64 + 0.5) * scale - 0.5).max(0.0);
                let lo = (pos.floor() as usize).min(src - 1);
                let hi = (lo + 1).min(src - 1);
                let frac = (pos - lo as f64) as f32;
                vec![(lo, 1.0 - frac), (hi, frac)]
            }
        })
        .collect()
}

/// A single-channel `f32` image (the greyscale thumbnails of the motion gate).
#[derive(Debug, Clone, PartialEq)]
pub struct Plane {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl Plane {
    /// `cv2.absdiff`
    pub fn abs_diff(&self, other: &Plane) -> Plane {
        Plane {
            width: self.width,
            height: self.height,
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| (a - b).abs())
                .collect(),
        }
    }

    /// `cv2.meanStdDev` (population standard deviation).
    pub fn mean_std(&self) -> (f64, f64) {
        let n = self.data.len() as f64;
        let mean = self.data.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let var = self
            .data
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / n;
        (mean, var.sqrt())
    }
}
