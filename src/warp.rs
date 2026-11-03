//! Perspective rectification of a text quadrilateral into a horizontal strip.
//!
//! This replaces `cv::getPerspectiveTransform` + `cv::warpPerspective`.
//! We solve the 8-parameter homography that maps the *destination* rectangle
//! onto the *source* quad, then walk every destination pixel, map it back
//! into the source image and sample it bilinearly. Inverse mapping means no
//! holes and no overdraw, and it parallelises trivially by row.

use crate::geometry::{Point, Quad};
use image::{Rgb, RgbImage};

/// 3×3 homography stored row-major.
#[derive(Clone, Copy, Debug)]
pub struct Homography(pub [f32; 9]);

impl Homography {
    /// Homography mapping the four `src` points onto the four `dst` points
    /// (same ordering). Solves the standard 8×8 linear system with Gaussian
    /// elimination and partial pivoting.
    pub fn from_points(src: &[Point; 4], dst: &[Point; 4]) -> Option<Self> {
        // Unknowns: h0..h7 (h8 = 1).
        let mut a = [[0f64; 9]; 8];
        for i in 0..4 {
            let (x, y) = (src[i].x as f64, src[i].y as f64);
            let (u, v) = (dst[i].x as f64, dst[i].y as f64);
            a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
            a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
        }
        // Forward elimination.
        for col in 0..8 {
            let pivot = (col..8).max_by(|&r1, &r2| a[r1][col].abs().partial_cmp(&a[r2][col].abs()).unwrap())?;
            if a[pivot][col].abs() < 1e-12 {
                return None;
            }
            a.swap(col, pivot);
            for r in (col + 1)..8 {
                let f = a[r][col] / a[col][col];
                let (pivot_row, rest) = a.split_at_mut(col + 1);
                let pivot_row = &pivot_row[col];
                let row = &mut rest[r - col - 1];
                for (x, p) in row.iter_mut().zip(pivot_row.iter()).skip(col) {
                    *x -= f * p;
                }
            }
        }
        // Back substitution.
        let mut h = [0f64; 8];
        for r in (0..8).rev() {
            let mut s = a[r][8];
            for c in (r + 1)..8 {
                s -= a[r][c] * h[c];
            }
            h[r] = s / a[r][r];
        }
        Some(Homography([
            h[0] as f32,
            h[1] as f32,
            h[2] as f32,
            h[3] as f32,
            h[4] as f32,
            h[5] as f32,
            h[6] as f32,
            h[7] as f32,
            1.0,
        ]))
    }

    #[inline]
    pub fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        let m = &self.0;
        let w = m[6] * x + m[7] * y + m[8];
        ((m[0] * x + m[1] * y + m[2]) / w, (m[3] * x + m[4] * y + m[5]) / w)
    }
}

#[inline]
fn sample_bilinear(img: &RgbImage, x: f32, y: f32) -> Rgb<u8> {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let px = |xi: i32, yi: i32| -> [f32; 3] {
        let xi = xi.clamp(0, w - 1) as u32;
        let yi = yi.clamp(0, h - 1) as u32;
        let p = img.get_pixel(xi, yi).0;
        [p[0] as f32, p[1] as f32, p[2] as f32]
    };
    let (p00, p10, p01, p11) = (px(x0, y0), px(x0 + 1, y0), px(x0, y0 + 1), px(x0 + 1, y0 + 1));
    let mut out = [0u8; 3];
    for c in 0..3 {
        let top = p00[c] * (1.0 - fx) + p10[c] * fx;
        let bot = p01[c] * (1.0 - fx) + p11[c] * fx;
        out[c] = (top * (1.0 - fy) + bot * fy).round().clamp(0.0, 255.0) as u8;
    }
    Rgb(out)
}

/// Cut `quad` out of `img` and straighten it into a `w × h` strip, where `w`
/// and `h` are the quad's own side lengths. If the result is much taller than
/// wide (vertical text, or a box that came out rotated by 90°), rotate it so
/// the recogniser always sees a horizontal line.
pub fn rectify(img: &RgbImage, quad: &Quad) -> RgbImage {
    let (fw, fh) = quad.side_lengths();
    let dw = fw.round().max(1.0) as u32;
    let dh = fh.round().max(1.0) as u32;
    let dst = [
        Point::new(0.0, 0.0),
        Point::new(dw as f32 - 1.0, 0.0),
        Point::new(dw as f32 - 1.0, dh as f32 - 1.0),
        Point::new(0.0, dh as f32 - 1.0),
    ];
    // Map destination -> source, so we can inverse-map every output pixel.
    let h = match Homography::from_points(&dst, &quad.0) {
        Some(h) => h,
        None => return RgbImage::new(dw, dh),
    };
    let mut out = RgbImage::new(dw, dh);
    for y in 0..dh {
        for x in 0..dw {
            let (sx, sy) = h.apply(x as f32, y as f32);
            out.put_pixel(x, y, sample_bilinear(img, sx, sy));
        }
    }
    if dh as f32 / dw as f32 >= 1.5 {
        image::imageops::rotate90(&out)
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_homography() {
        let pts = [
            Point::new(0.0, 0.0),
            Point::new(9.0, 0.0),
            Point::new(9.0, 4.0),
            Point::new(0.0, 4.0),
        ];
        let h = Homography::from_points(&pts, &pts).unwrap();
        let (x, y) = h.apply(3.0, 2.0);
        assert!((x - 3.0).abs() < 1e-4 && (y - 2.0).abs() < 1e-4);
    }

    #[test]
    fn homography_maps_corners() {
        let src = [
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 5.0),
            Point::new(0.0, 5.0),
        ];
        let dst = [
            Point::new(3.0, 4.0),
            Point::new(14.0, 6.0),
            Point::new(13.0, 12.0),
            Point::new(2.0, 10.0),
        ];
        let h = Homography::from_points(&src, &dst).unwrap();
        for i in 0..4 {
            let (x, y) = h.apply(src[i].x, src[i].y);
            assert!((x - dst[i].x).abs() < 1e-3 && (y - dst[i].y).abs() < 1e-3, "corner {i}");
        }
    }

    #[test]
    fn rectify_axis_aligned_crop_is_exact() {
        let mut img = RgbImage::new(20, 20);
        for y in 0..20 {
            for x in 0..20 {
                img.put_pixel(x, y, Rgb([x as u8 * 10, y as u8 * 10, 0]));
            }
        }
        let q = Quad([
            Point::new(2.0, 3.0),
            Point::new(11.0, 3.0),
            Point::new(11.0, 6.0),
            Point::new(2.0, 6.0),
        ]);
        let out = rectify(&img, &q);
        assert_eq!((out.width(), out.height()), (9, 3));
        assert_eq!(out.get_pixel(0, 0).0, [20, 30, 0]);
        assert_eq!(out.get_pixel(8, 2).0, [110, 60, 0]);
    }
}
