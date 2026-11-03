//! The end-to-end pipeline: image in, text lines out.

use crate::contours::{self, BoxOptions};
use crate::engine::{InputShape, Model};
use crate::geometry::Quad;
use crate::recognizer::{self, Charset};
use crate::{layout, preprocess, warp};
use anyhow::{Context, Result};
use image::{GrayImage, RgbImage};
use rayon::prelude::*;
use serde::Serialize;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Where the model files live.
#[derive(Clone, Debug)]
pub struct ModelPaths {
    pub detector: PathBuf,
    pub recognizer: PathBuf,
    pub charset: PathBuf,
}

impl ModelPaths {
    /// The PP-OCRv4 files as shipped in this repository's `models/` folder.
    pub fn ppocr_v4(dir: impl Into<PathBuf>) -> Self {
        let dir = dir.into();
        Self {
            detector: dir.join("ch_PP-OCRv4_det_infer.onnx"),
            recognizer: dir.join("ch_PP-OCRv4_rec_infer.onnx"),
            charset: dir.join("ppocr_keys_v1.txt"),
        }
    }
}

/// Per-call knobs. Plain data, no shared state: safe to build on the fly for
/// every request while the [`Ocr`] instance stays shared.
#[derive(Clone, Copy, Debug)]
pub struct RunOptions {
    /// Longest side of the detector input, in pixels.
    pub max_side: u32,
    /// Pixel threshold on the probability map.
    pub det_thresh: f32,
    /// Minimum mean probability for a box to be kept.
    pub box_thresh: f32,
    /// Box expansion ratio.
    pub unclip_ratio: f32,
    /// Drop recognised lines whose confidence is below this.
    pub min_text_conf: f32,
    /// Recognise crops in parallel.
    pub parallel: bool,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            max_side: 960,
            det_thresh: 0.3,
            box_thresh: 0.6,
            unclip_ratio: 1.6,
            min_text_conf: 0.5,
            parallel: true,
        }
    }
}

/// Boxes with their detector scores.
pub type Boxes = Vec<(Quad, f32)>;

/// One recognised line of text, in original-image coordinates.
#[derive(Clone, Debug, Serialize)]
pub struct TextLine {
    pub text: String,
    /// Mean per-character confidence from the recogniser.
    pub confidence: f32,
    /// Mean probability of the detector over the region.
    pub det_score: f32,
    pub quad: Quad,
}

/// Wall-clock time spent in each stage of the last run.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Timings {
    pub preprocess: Duration,
    pub detect: Duration,
    pub postprocess: Duration,
    pub crop: Duration,
    pub recognise: Duration,
    pub total: Duration,
}

/// A loaded OCR engine. Immutable after construction; share it with `Arc`.
pub struct Ocr {
    det: Model,
    rec: Model,
    charset: Charset,
}

impl Ocr {
    pub fn load(paths: &ModelPaths) -> Result<Self> {
        let det = Model::load(&paths.detector, InputShape::DynamicHW { stride: 32 }).context("detector")?;
        let rec = Model::load(
            &paths.recognizer,
            InputShape::FixedHeight(recognizer::REC_HEIGHT as usize),
        )
        .context("recogniser")?;
        let charset = Charset::from_file(&paths.charset)?;
        Ok(Self { det, rec, charset })
    }

    /// Pre-build the recogniser plans for every width bucket, and the
    /// detector plan for an image of `(width, height)` pixels, so the first
    /// real request does not pay for graph optimisation. Optional; the
    /// plans are built lazily otherwise.
    pub fn warm_up(&self, image_size: Option<(u32, u32)>, opts: &RunOptions) -> Result<()> {
        if let Some((w, h)) = image_size {
            let (resized, _, _) = preprocess::resize_for_detection(&RgbImage::new(w, h), opts.max_side);
            self.det.warm_up(resized.height() as usize, resized.width() as usize)?;
        }
        recognizer::WIDTH_BUCKETS
            .par_iter()
            .try_for_each(|&bw| self.rec.warm_up(recognizer::REC_HEIGHT as usize, bw as usize))
    }

    /// Run detection only. Returns boxes in original-image coordinates with
    /// their detector scores, in reading order.
    pub fn detect(&self, img: &RgbImage, opts: &RunOptions) -> Result<(Boxes, Timings)> {
        self.detect_inner(img, opts, None)
    }

    /// Like [`detect`](Self::detect), but also returns the raw probability map
    /// as a grayscale image at the detector's input resolution. Useful for
    /// debugging thresholds and for illustrations.
    pub fn detect_with_map(&self, img: &RgbImage, opts: &RunOptions) -> Result<(Boxes, Timings, GrayImage)> {
        let mut map = None;
        let (boxes, t) = self.detect_inner(img, opts, Some(&mut map))?;
        Ok((boxes, t, map.expect("map requested")))
    }

    fn detect_inner(
        &self,
        img: &RgbImage,
        opts: &RunOptions,
        prob_map: Option<&mut Option<GrayImage>>,
    ) -> Result<(Boxes, Timings)> {
        let mut t = Timings::default();
        let t0 = Instant::now();

        let (resized, sx, sy) = preprocess::resize_for_detection(img, opts.max_side);
        let tensor = preprocess::det_tensor(&resized);
        let (rw, rh) = (resized.width() as usize, resized.height() as usize);
        t.preprocess = t0.elapsed();

        let t1 = Instant::now();
        let (prob, shape) = self.det.run(tensor, rh, rw)?;
        anyhow::ensure!(
            shape.len() == 4 && shape[2] == rh && shape[3] == rw,
            "unexpected detector output shape {shape:?} for input {rh}x{rw}"
        );
        t.detect = t1.elapsed();
        if let Some(slot) = prob_map {
            let px: Vec<u8> = prob.iter().map(|p| (p * 255.0).round() as u8).collect();
            *slot = GrayImage::from_raw(rw as u32, rh as u32, px);
        }

        let t2 = Instant::now();
        let comps = contours::find_components(&prob, rw, rh, opts.det_thresh, 4);
        let box_opts = BoxOptions {
            box_thresh: opts.box_thresh,
            unclip_ratio: opts.unclip_ratio,
            min_side: 3.0,
        };
        let mut boxes: Vec<(Quad, f32)> = contours::components_to_boxes(&comps, rw, rh, &box_opts)
            .into_iter()
            .map(|(q, s)| (q.scale(sx, sy).clamp(img.width() as f32, img.height() as f32), s))
            .collect();
        let order = layout::reading_order(&boxes.iter().map(|b| b.0).collect::<Vec<_>>());
        boxes = order.into_iter().map(|i| boxes[i]).collect();
        t.postprocess = t2.elapsed();
        t.total = t0.elapsed();
        Ok((boxes, t))
    }

    /// Full pipeline: detect, crop, recognise.
    pub fn run(&self, img: &RgbImage, opts: &RunOptions) -> Result<(Vec<TextLine>, Timings)> {
        let t0 = Instant::now();
        let (boxes, mut t) = self.detect(img, opts)?;

        let t3 = Instant::now();
        let crops: Vec<RgbImage> = if opts.parallel {
            boxes.par_iter().map(|(q, _)| warp::rectify(img, q)).collect()
        } else {
            boxes.iter().map(|(q, _)| warp::rectify(img, q)).collect()
        };
        t.crop = t3.elapsed();

        let t4 = Instant::now();
        let results: Vec<Result<(String, f32)>> = if opts.parallel {
            crops
                .par_iter()
                .map(|c| recognizer::recognise(&self.rec, &self.charset, c))
                .collect()
        } else {
            crops
                .iter()
                .map(|c| recognizer::recognise(&self.rec, &self.charset, c))
                .collect()
        };
        t.recognise = t4.elapsed();

        let mut lines = Vec::with_capacity(boxes.len());
        for ((quad, det_score), r) in boxes.into_iter().zip(results) {
            let (text, confidence) = r?;
            if text.is_empty() || confidence < opts.min_text_conf {
                continue;
            }
            lines.push(TextLine {
                text,
                confidence,
                det_score,
                quad,
            });
        }
        t.total = t0.elapsed();
        Ok((lines, t))
    }
}
