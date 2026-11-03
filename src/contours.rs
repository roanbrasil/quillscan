//! From a probability map to candidate text boxes, without `findContours`.
//!
//! OpenCV's approach is border following (Suzuki & Abe, 1985): walk around
//! every blob and emit its outline as a polygon. For DBNet post-processing we
//! never need the outline itself; we need the **minimum-area rectangle** of
//! each blob and a **confidence score** for it. So this module does something
//! simpler and easier to get right:
//!
//! 1. threshold the map into a bitmap;
//! 2. label 8-connected components with a flood fill;
//! 3. for each component, keep only the pixels that touch the background
//!    (the "edge pixels") — those are enough to compute the convex hull;
//! 4. hand the edge pixels to [`geometry::min_area_rect`].
//!
//! The score of a box is the mean probability over the component's pixels.

use crate::geometry::{self, Point, Quad};

/// A connected blob of "text" pixels in the probability map.
#[derive(Debug)]
pub struct Component {
    /// Pixels on the blob's border (adjacent to at least one background pixel).
    pub edge: Vec<Point>,
    /// Number of pixels in the blob.
    pub size: usize,
    /// Mean probability over the blob's pixels.
    pub score: f32,
}

/// Threshold `prob` (row-major, `w × h`) at `thresh`, then label connected
/// components. Components smaller than `min_pixels` are discarded.
pub fn find_components(prob: &[f32], w: usize, h: usize, thresh: f32, min_pixels: usize) -> Vec<Component> {
    assert_eq!(prob.len(), w * h);
    let mut mask: Vec<bool> = prob.iter().map(|&p| p > thresh).collect();
    let mut out = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut pixels: Vec<usize> = Vec::new();

    for start in 0..w * h {
        if !mask[start] {
            continue;
        }
        // Flood-fill this component, clearing the mask as we go so every
        // pixel is visited exactly once over the whole image.
        pixels.clear();
        stack.clear();
        stack.push(start);
        mask[start] = false;
        while let Some(idx) = stack.pop() {
            pixels.push(idx);
            let (x, y) = (idx % w, idx / w);
            let x0 = x.saturating_sub(1);
            let y0 = y.saturating_sub(1);
            let x1 = (x + 1).min(w - 1);
            let y1 = (y + 1).min(h - 1);
            for ny in y0..=y1 {
                for nx in x0..=x1 {
                    let n = ny * w + nx;
                    if mask[n] {
                        mask[n] = false;
                        stack.push(n);
                    }
                }
            }
        }
        if pixels.len() < min_pixels {
            continue;
        }

        // Edge pixels: any pixel with a 4-neighbour below the threshold (or
        // on the image border). We re-test against `prob` because `mask` has
        // been consumed.
        let is_text = |x: isize, y: isize| -> bool {
            x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h && prob[y as usize * w + x as usize] > thresh
        };
        let mut edge = Vec::new();
        let mut sum = 0.0f32;
        for &idx in &pixels {
            let (x, y) = ((idx % w) as isize, (idx / w) as isize);
            sum += prob[idx];
            if !is_text(x - 1, y) || !is_text(x + 1, y) || !is_text(x, y - 1) || !is_text(x, y + 1) {
                edge.push(Point::new(x as f32, y as f32));
            }
        }
        out.push(Component {
            edge,
            size: pixels.len(),
            score: sum / pixels.len() as f32,
        });
    }
    out
}

/// Options for turning components into boxes.
#[derive(Clone, Copy, Debug)]
pub struct BoxOptions {
    /// Discard boxes whose score is below this.
    pub box_thresh: f32,
    /// Expansion ratio applied to every box (DBNet's `unclip_ratio`).
    pub unclip_ratio: f32,
    /// Discard boxes whose shorter side (after expansion) is below this, in
    /// probability-map pixels.
    pub min_side: f32,
}

impl Default for BoxOptions {
    fn default() -> Self {
        Self {
            box_thresh: 0.6,
            unclip_ratio: 1.6,
            min_side: 3.0,
        }
    }
}

/// Convert components to expanded, ordered, image-clamped quadrilaterals in
/// probability-map coordinates. Returns `(quad, score)` pairs.
pub fn components_to_boxes(comps: &[Component], w: usize, h: usize, opts: &BoxOptions) -> Vec<(Quad, f32)> {
    let mut boxes = Vec::with_capacity(comps.len());
    for c in comps {
        if c.score < opts.box_thresh {
            continue;
        }
        let Some(rect) = geometry::min_area_rect(&c.edge) else {
            continue;
        };
        // Pixel centres sit at integer coordinates; add half a pixel on each
        // side so the rectangle covers the blob rather than its centres.
        let rect = geometry::offset_quad(&rect, 0.5).clamp(w as f32, h as f32);
        let (sw, sh) = rect.side_lengths();
        if sw.min(sh) < 1.0 {
            continue;
        }
        let expanded = geometry::expand_quad(&rect, opts.unclip_ratio).clamp(w as f32, h as f32);
        let (ew, eh) = expanded.side_lengths();
        if ew.min(eh) < opts.min_side {
            continue;
        }
        boxes.push((expanded, c.score));
    }
    boxes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_two_separate_blobs() {
        let (w, h) = (20usize, 10usize);
        let mut prob = vec![0.0f32; w * h];
        for y in 2..5 {
            for x in 2..8 {
                prob[y * w + x] = 0.9;
            }
            for x in 12..18 {
                prob[y * w + x] = 0.8;
            }
        }
        let comps = find_components(&prob, w, h, 0.3, 1);
        assert_eq!(comps.len(), 2);
        assert_eq!(comps[0].size, 18);
        assert!((comps[0].score - 0.9).abs() < 1e-5);
        // A 6×3 blob has 2*6 + 2*1 = 14 border pixels.
        assert_eq!(comps[0].edge.len(), 14);

        let boxes = components_to_boxes(
            &comps,
            w,
            h,
            &BoxOptions {
                box_thresh: 0.5,
                unclip_ratio: 0.0,
                min_side: 1.0,
            },
        );
        assert_eq!(boxes.len(), 2);
        let (bw, bh) = boxes[0].0.side_lengths();
        // Pixel centres span 5×2; the half-pixel pad makes it 6×3.
        assert!((bw - 6.0).abs() < 1e-3 && (bh - 3.0).abs() < 1e-3, "{bw}x{bh}");
    }

    #[test]
    fn diagonal_pixels_are_connected() {
        let (w, h) = (4usize, 4usize);
        let mut prob = vec![0.0f32; w * h];
        for i in 0..4 {
            prob[i * w + i] = 1.0;
        }
        let comps = find_components(&prob, w, h, 0.5, 1);
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0].size, 4);
    }
}
