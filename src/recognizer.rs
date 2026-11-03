//! Text recognition: crop → `48 × W` tensor → CTC greedy decoding.

use crate::engine::Model;
use crate::preprocess;
use anyhow::{Context, Result};
use image::{imageops::FilterType, RgbImage};
use std::fs;
use std::path::Path;

/// PP-OCRv4 recognition models take 48-pixel-high strips.
pub const REC_HEIGHT: u32 = 48;

/// Character set for CTC decoding. Index 0 is the CTC blank; the last index
/// is the space character, which PaddleOCR appends after the dictionary file.
pub struct Charset {
    chars: Vec<String>,
}

impl Charset {
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path).with_context(|| format!("reading charset {}", path.display()))?;
        Ok(Self::from_lines(text.lines()))
    }

    pub fn from_lines<'a>(lines: impl Iterator<Item = &'a str>) -> Self {
        let mut chars = vec![String::new()]; // blank
        chars.extend(lines.map(|l| l.to_string()));
        chars.push(" ".to_string());
        Self { chars }
    }

    pub fn len(&self) -> usize {
        self.chars.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }

    /// Greedy CTC decode of a `[T, C]` logit/probability matrix.
    /// Returns the text and the mean confidence over emitted characters.
    pub fn ctc_decode(&self, probs: &[f32], timesteps: usize, classes: usize) -> (String, f32) {
        let mut text = String::new();
        let mut conf_sum = 0.0f32;
        let mut emitted = 0usize;
        let mut prev = usize::MAX;
        for t in 0..timesteps {
            let row = &probs[t * classes..(t + 1) * classes];
            let (idx, p) = row
                .iter()
                .enumerate()
                .fold((0usize, f32::MIN), |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc });
            if idx != 0 && idx != prev {
                if let Some(ch) = self.chars.get(idx) {
                    text.push_str(ch);
                    conf_sum += p;
                    emitted += 1;
                }
            }
            prev = idx;
        }
        let conf = if emitted == 0 { 0.0 } else { conf_sum / emitted as f32 };
        (text, conf)
    }
}

/// Input widths the recogniser is run at. A crop is resized to height 48,
/// then placed at the left of the smallest bucket that fits it, with the rest
/// of the tensor zero (mid-grey after normalisation). Bucketing keeps the
/// number of concrete plans tract has to build small (see `engine.rs`) and is
/// the same trick PaddleOCR uses when it pads a batch to a common width.
pub const WIDTH_BUCKETS: [u32; 10] = [128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072];

/// Smallest bucket that fits `w`; the last bucket for anything wider.
pub fn bucket_width(w: u32) -> u32 {
    WIDTH_BUCKETS
        .iter()
        .copied()
        .find(|&b| b >= w)
        .unwrap_or(*WIDTH_BUCKETS.last().unwrap())
}

/// Resize a crop to the recogniser's input height, keeping aspect ratio, pad
/// it to a bucket width, and return the tensor plus that width.
pub fn crop_to_tensor(crop: &RgbImage) -> (Vec<f32>, usize) {
    let ratio = crop.width() as f32 / crop.height().max(1) as f32;
    let w = ((REC_HEIGHT as f32 * ratio).ceil() as u32).max(8);
    let bucket = bucket_width(w);
    let w = w.min(bucket); // very wide crops are squeezed into the last bucket
    let resized = image::imageops::resize(crop, w, REC_HEIGHT, FilterType::Triangle);
    let strip = preprocess::rec_tensor(&resized);
    if w == bucket {
        return (strip, w as usize);
    }
    // Copy each channel row into the wider zero-filled tensor.
    let (h, w, bw) = (REC_HEIGHT as usize, w as usize, bucket as usize);
    let mut out = vec![0f32; 3 * h * bw];
    for c in 0..3 {
        for y in 0..h {
            let src = &strip[c * h * w + y * w..c * h * w + (y + 1) * w];
            out[c * h * bw + y * bw..c * h * bw + y * bw + w].copy_from_slice(src);
        }
    }
    (out, bw)
}

pub fn recognise(model: &Model, charset: &Charset, crop: &RgbImage) -> Result<(String, f32)> {
    let (tensor, w) = crop_to_tensor(crop);
    let (out, shape) = model.run(tensor, REC_HEIGHT as usize, w)?;
    anyhow::ensure!(shape.len() == 3, "unexpected recogniser output shape {shape:?}");
    let (t, c) = (shape[1], shape[2]);
    anyhow::ensure!(
        c == charset.len(),
        "charset has {} entries but model emits {c} classes",
        charset.len()
    );
    Ok(charset.ctc_decode(&out, t, c))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_round_up() {
        assert_eq!(bucket_width(100), 128);
        assert_eq!(bucket_width(128), 128);
        assert_eq!(bucket_width(129), 192);
        assert_eq!(bucket_width(9999), 3072);
    }

    #[test]
    fn padded_tensor_keeps_pixels_left_aligned() {
        let mut crop = RgbImage::new(20, 10);
        for p in crop.pixels_mut() {
            *p = image::Rgb([255, 255, 255]);
        }
        let (t, w) = crop_to_tensor(&crop);
        assert_eq!(w, 128);
        assert_eq!(t.len(), 3 * 48 * 128);
        assert!((t[0] - 1.0).abs() < 1e-6); // first pixel of row 0: white
        assert_eq!(t[127], 0.0); // last column of row 0: padding
    }

    #[test]
    fn ctc_collapses_repeats_and_blanks() {
        let cs = Charset::from_lines(["a", "b"].into_iter());
        assert_eq!(cs.len(), 4); // blank, a, b, space
                                 // timesteps: a a blank a b b blank space
        let seq = [1, 1, 0, 1, 2, 2, 0, 3];
        let mut probs = vec![0.0f32; seq.len() * 4];
        for (t, &c) in seq.iter().enumerate() {
            probs[t * 4 + c] = 0.9;
        }
        let (text, conf) = cs.ctc_decode(&probs, seq.len(), 4);
        assert_eq!(text, "aab ");
        assert!((conf - 0.9).abs() < 1e-6);
    }
}
