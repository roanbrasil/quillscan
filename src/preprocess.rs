//! Image → tensor conversion for the two models.

use image::{imageops::FilterType, RgbImage};

/// PP-OCR detection normalisation: ImageNet mean/std on `x / 255`.
const DET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const DET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Resize so the longest side is at most `max_side` and both sides are
/// multiples of 32 (the detector's stride). Returns the resized image plus the
/// `(sx, sy)` factors that map resized coordinates back to the original.
pub fn resize_for_detection(img: &RgbImage, max_side: u32) -> (RgbImage, f32, f32) {
    let (w, h) = (img.width(), img.height());
    let longest = w.max(h) as f32;
    let ratio = if longest > max_side as f32 {
        max_side as f32 / longest
    } else {
        1.0
    };
    let round32 = |v: f32| -> u32 { (((v / 32.0).round() as u32).max(1)) * 32 };
    let nw = round32(w as f32 * ratio);
    let nh = round32(h as f32 * ratio);
    let resized = if (nw, nh) == (w, h) {
        img.clone()
    } else {
        image::imageops::resize(img, nw, nh, FilterType::Triangle)
    };
    (resized, w as f32 / nw as f32, h as f32 / nh as f32)
}

/// Pack an RGB image into a `[1, 3, H, W]` f32 buffer, normalised per channel
/// as `(x/255 - mean) / std`.
pub fn to_nchw(img: &RgbImage, mean: [f32; 3], std: [f32; 3]) -> Vec<f32> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let plane = w * h;
    let mut out = vec![0f32; 3 * plane];
    let scale = [1.0 / (255.0 * std[0]), 1.0 / (255.0 * std[1]), 1.0 / (255.0 * std[2])];
    let shift = [mean[0] / std[0], mean[1] / std[1], mean[2] / std[2]];
    for (i, px) in img.pixels().enumerate() {
        out[i] = px.0[0] as f32 * scale[0] - shift[0];
        out[plane + i] = px.0[1] as f32 * scale[1] - shift[1];
        out[2 * plane + i] = px.0[2] as f32 * scale[2] - shift[2];
    }
    out
}

pub fn det_tensor(img: &RgbImage) -> Vec<f32> {
    to_nchw(img, DET_MEAN, DET_STD)
}

/// Recognition normalisation is simply `(x/255 - 0.5) / 0.5`.
pub fn rec_tensor(img: &RgbImage) -> Vec<f32> {
    to_nchw(img, [0.5; 3], [0.5; 3])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_resize_is_multiple_of_32() {
        let img = RgbImage::new(1234, 777);
        let (r, sx, sy) = resize_for_detection(&img, 960);
        assert_eq!(r.width() % 32, 0);
        assert_eq!(r.height() % 32, 0);
        assert!(r.width() <= 960 && r.height() <= 960);
        assert!((sx * r.width() as f32 - 1234.0).abs() < 1e-3);
        assert!((sy * r.height() as f32 - 777.0).abs() < 1e-3);
    }

    #[test]
    fn nchw_layout_and_normalisation() {
        let mut img = RgbImage::new(2, 1);
        img.put_pixel(0, 0, image::Rgb([255, 0, 0]));
        img.put_pixel(1, 0, image::Rgb([0, 0, 255]));
        let t = rec_tensor(&img);
        assert_eq!(t.len(), 6);
        assert!((t[0] - 1.0).abs() < 1e-6); // R of pixel 0
        assert!((t[1] + 1.0).abs() < 1e-6); // R of pixel 1
        assert!((t[5] - 1.0).abs() < 1e-6); // B of pixel 1
    }
}
