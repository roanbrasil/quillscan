//! Small 2-D geometry toolkit: convex hull, minimum-area bounding rectangle,
//! and the box expansion step DBNet needs after detection.
//!
//! Everything here works on plain `f32` points. There is no external geometry
//! crate: the three algorithms we need (monotone chain, rotating calipers,
//! axis-aligned offset of a rectangle) are each under fifty lines.

use serde::Serialize;

/// A point in image coordinates (x to the right, y down).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn dist(self, other: Point) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

/// A quadrilateral with corners ordered **top-left, top-right, bottom-right,
/// bottom-left**. The ordering matters: the perspective warp relies on it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Quad(pub [Point; 4]);

impl Quad {
    /// Length of the top edge and left edge, respectively.
    pub fn side_lengths(&self) -> (f32, f32) {
        let [tl, tr, br, bl] = self.0;
        let w = tl.dist(tr).max(bl.dist(br));
        let h = tl.dist(bl).max(tr.dist(br));
        (w, h)
    }

    /// Shoelace area (always non-negative).
    pub fn area(&self) -> f32 {
        polygon_area(&self.0)
    }

    pub fn perimeter(&self) -> f32 {
        let p = &self.0;
        (0..4).map(|i| p[i].dist(p[(i + 1) % 4])).sum()
    }

    pub fn centroid(&self) -> Point {
        let p = &self.0;
        Point::new(
            (p[0].x + p[1].x + p[2].x + p[3].x) / 4.0,
            (p[0].y + p[1].y + p[2].y + p[3].y) / 4.0,
        )
    }

    /// Axis-aligned bounding box as `(min_x, min_y, max_x, max_y)`.
    pub fn bbox(&self) -> (f32, f32, f32, f32) {
        let mut b = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for p in &self.0 {
            b.0 = b.0.min(p.x);
            b.1 = b.1.min(p.y);
            b.2 = b.2.max(p.x);
            b.3 = b.3.max(p.y);
        }
        b
    }

    /// Clamp every corner into `[0, w) × [0, h)`.
    pub fn clamp(&self, w: f32, h: f32) -> Quad {
        Quad(
            self.0
                .map(|p| Point::new(p.x.clamp(0.0, w - 1.0), p.y.clamp(0.0, h - 1.0))),
        )
    }

    /// Multiply every coordinate by `(sx, sy)`. Used to map boxes found on the
    /// resized detection input back to the original image.
    pub fn scale(&self, sx: f32, sy: f32) -> Quad {
        Quad(self.0.map(|p| Point::new(p.x * sx, p.y * sy)))
    }
}

/// Shoelace formula for an arbitrary simple polygon.
pub fn polygon_area(pts: &[Point]) -> f32 {
    let n = pts.len();
    if n < 3 {
        return 0.0;
    }
    let mut s = 0.0f32;
    for i in 0..n {
        let a = pts[i];
        let b = pts[(i + 1) % n];
        s += a.x * b.y - b.x * a.y;
    }
    (s / 2.0).abs()
}

fn cross(o: Point, a: Point, b: Point) -> f32 {
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
}

/// Convex hull by Andrew's monotone chain. Returns the hull in
/// counter-clockwise order (in image coordinates, where y grows downward,
/// that means visually clockwise). Collinear points are dropped.
pub fn convex_hull(points: &[Point]) -> Vec<Point> {
    if points.len() < 3 {
        return points.to_vec();
    }
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap().then(a.y.partial_cmp(&b.y).unwrap()));
    pts.dedup();

    let mut hull: Vec<Point> = Vec::with_capacity(pts.len() * 2);
    for &p in &pts {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    let lower_len = hull.len() + 1;
    for &p in pts.iter().rev().skip(1) {
        while hull.len() >= lower_len && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop();
    hull
}

/// Minimum-area bounding rectangle of a point set (rotating calipers over the
/// convex hull edges). Returns the rectangle as an ordered [`Quad`].
///
/// The idea: the minimum-area rectangle of a convex polygon always has one
/// side flush with a polygon edge. So we try every hull edge as the
/// orientation, project all hull points onto that edge's direction and its
/// normal, and keep the orientation with the smallest extent product.
pub fn min_area_rect(points: &[Point]) -> Option<Quad> {
    let hull = convex_hull(points);
    if hull.is_empty() {
        return None;
    }
    if hull.len() == 1 {
        let p = hull[0];
        return Some(Quad([p, p, p, p]));
    }
    if hull.len() == 2 {
        // Degenerate: a segment. Return a zero-height rectangle along it.
        let (a, b) = (hull[0], hull[1]);
        return Some(order_corners([a, b, b, a]));
    }

    let mut best_area = f32::MAX;
    let mut best: Option<[Point; 4]> = None;

    for i in 0..hull.len() {
        let a = hull[i];
        let b = hull[(i + 1) % hull.len()];
        let len = a.dist(b);
        if len < 1e-6 {
            continue;
        }
        // Unit direction along the edge and its normal.
        let ux = (b.x - a.x) / len;
        let uy = (b.y - a.y) / len;
        let (nx, ny) = (-uy, ux);

        let (mut min_u, mut max_u, mut min_n, mut max_n) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for p in &hull {
            let du = p.x * ux + p.y * uy;
            let dn = p.x * nx + p.y * ny;
            min_u = min_u.min(du);
            max_u = max_u.max(du);
            min_n = min_n.min(dn);
            max_n = max_n.max(dn);
        }
        let area = (max_u - min_u) * (max_n - min_n);
        if area < best_area {
            best_area = area;
            let corner = |u: f32, n: f32| Point::new(u * ux + n * nx, u * uy + n * ny);
            best = Some([
                corner(min_u, min_n),
                corner(max_u, min_n),
                corner(max_u, max_n),
                corner(min_u, max_n),
            ]);
        }
    }
    best.map(order_corners)
}

/// Reorder four corners into top-left, top-right, bottom-right, bottom-left.
///
/// Sort by x, split into the two leftmost and two rightmost, then order each
/// pair by y. This is robust for rectangles up to ±45° of rotation, which is
/// the regime text lines live in.
pub fn order_corners(mut pts: [Point; 4]) -> Quad {
    pts.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());
    let (mut left, mut right) = ([pts[0], pts[1]], [pts[2], pts[3]]);
    if left[0].y > left[1].y {
        left.swap(0, 1);
    }
    if right[0].y > right[1].y {
        right.swap(0, 1);
    }
    Quad([left[0], right[0], right[1], left[1]])
}

/// Expand a detected box outward.
///
/// DBNet is trained on text regions shrunk toward their core, so detections
/// come out tight. PaddleOCR's post-processing grows each box by an offset
/// `d = area × ratio / perimeter` (this is the formula from the DBNet paper,
/// applied in reverse). PaddleOCR computes that offset with a general polygon
/// offsetting library; because our boxes are always rectangles, the offset of
/// a rectangle by `d` is just a rectangle whose sides moved outward by `d`,
/// so we do it with four edge normals and no polygon-clipping machinery.
pub fn expand_quad(q: &Quad, ratio: f32) -> Quad {
    let perimeter = q.perimeter();
    if perimeter < 1e-6 {
        return *q;
    }
    let d = q.area() * ratio / perimeter;
    offset_quad(q, d)
}

/// Move every edge of the quad outward by `d` pixels.
pub fn offset_quad(q: &Quad, d: f32) -> Quad {
    let c = q.centroid();
    let p = q.0;
    let mut out = [Point::new(0.0, 0.0); 4];
    for i in 0..4 {
        // Each corner is the intersection of two edges; moving both edges
        // outward by `d` moves the corner along the sum of their outward
        // unit normals. For a rectangle those normals are orthogonal, so the
        // corner shifts by (±d, ±d) in the rectangle's own frame.
        let prev = p[(i + 3) % 4];
        let cur = p[i];
        let next = p[(i + 1) % 4];
        let n1 = outward_normal(prev, cur, c);
        let n2 = outward_normal(cur, next, c);
        out[i] = Point::new(cur.x + (n1.x + n2.x) * d, cur.y + (n1.y + n2.y) * d);
    }
    Quad(out)
}

fn outward_normal(a: Point, b: Point, centroid: Point) -> Point {
    let len = a.dist(b).max(1e-6);
    let (ux, uy) = ((b.x - a.x) / len, (b.y - a.y) / len);
    let (mut nx, mut ny) = (-uy, ux);
    // Flip if the normal points toward the centroid.
    let mid = Point::new((a.x + b.x) / 2.0, (a.y + b.y) / 2.0);
    if (mid.x - centroid.x) * nx + (mid.y - centroid.y) * ny < 0.0 {
        nx = -nx;
        ny = -ny;
    }
    Point::new(nx, ny)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hull_of_square_has_four_points() {
        let pts = [
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(2.0, 2.0),
            Point::new(0.0, 2.0),
            Point::new(1.0, 1.0),
            Point::new(1.0, 0.0),
        ];
        let hull = convex_hull(&pts);
        assert_eq!(hull.len(), 4);
    }

    #[test]
    fn min_area_rect_of_axis_aligned_points() {
        let pts: Vec<Point> = (0..10)
            .flat_map(|x| (0..3).map(move |y| Point::new(x as f32, y as f32)))
            .collect();
        let q = min_area_rect(&pts).unwrap();
        let (w, h) = q.side_lengths();
        assert!((w - 9.0).abs() < 1e-3, "w={w}");
        assert!((h - 2.0).abs() < 1e-3, "h={h}");
        assert!((q.area() - 18.0).abs() < 1e-2);
    }

    #[test]
    fn min_area_rect_follows_rotation() {
        // A thin bar rotated 30°: the min-area rect should be ~ 20×2, not the
        // axis-aligned bbox.
        let (c, s) = (30f32.to_radians().cos(), 30f32.to_radians().sin());
        let pts: Vec<Point> = (0..40)
            .flat_map(|i| {
                (0..5).map(move |j| {
                    let (x, y) = (i as f32 * 0.5, j as f32 * 0.5);
                    Point::new(x * c - y * s, x * s + y * c)
                })
            })
            .collect();
        let q = min_area_rect(&pts).unwrap();
        assert!(q.area() < 45.0, "area={}", q.area());
    }

    #[test]
    fn expand_grows_rectangle_uniformly() {
        let q = Quad([
            Point::new(10.0, 10.0),
            Point::new(50.0, 10.0),
            Point::new(50.0, 20.0),
            Point::new(10.0, 20.0),
        ]);
        let e = expand_quad(&q, 1.5);
        // d = (400 * 1.5) / 100 = 6
        assert!((e.0[0].x - 4.0).abs() < 1e-3);
        assert!((e.0[0].y - 4.0).abs() < 1e-3);
        assert!((e.0[2].x - 56.0).abs() < 1e-3);
        assert!((e.0[2].y - 26.0).abs() < 1e-3);
    }

    #[test]
    fn corners_are_ordered() {
        let q = order_corners([
            Point::new(5.0, 5.0),
            Point::new(0.0, 0.0),
            Point::new(0.0, 5.0),
            Point::new(5.0, 0.0),
        ]);
        assert_eq!(q.0[0], Point::new(0.0, 0.0));
        assert_eq!(q.0[1], Point::new(5.0, 0.0));
        assert_eq!(q.0[2], Point::new(5.0, 5.0));
        assert_eq!(q.0[3], Point::new(0.0, 5.0));
    }
}
