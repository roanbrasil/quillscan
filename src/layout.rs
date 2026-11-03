//! Reading order: group boxes into lines and sort them left to right.
//!
//! Detection returns boxes in whatever order the flood fill found them. For a
//! document, people expect top-to-bottom, left-to-right. We cluster boxes by
//! vertical overlap: two boxes belong to the same line when their vertical
//! centres are within half the smaller box height of each other.

use crate::geometry::Quad;

/// Returns indices into `quads` in reading order.
pub fn reading_order(quads: &[Quad]) -> Vec<usize> {
    struct Item {
        idx: usize,
        cx: f32,
        cy: f32,
        h: f32,
    }
    let mut items: Vec<Item> = quads
        .iter()
        .enumerate()
        .map(|(idx, q)| {
            let (x0, y0, x1, y1) = q.bbox();
            Item {
                idx,
                cx: (x0 + x1) / 2.0,
                cy: (y0 + y1) / 2.0,
                h: (y1 - y0).max(1.0),
            }
        })
        .collect();
    items.sort_by(|a, b| a.cy.partial_cmp(&b.cy).unwrap());

    let mut lines: Vec<Vec<usize>> = Vec::new(); // indices into `items`
    let mut line_cy: Vec<f32> = Vec::new();
    for (i, it) in items.iter().enumerate() {
        match (0..lines.len()).find(|&li| {
            let ref_h = items[lines[li][0]].h.min(it.h);
            (line_cy[li] - it.cy).abs() < ref_h * 0.5
        }) {
            Some(li) => {
                lines[li].push(i);
                let n = lines[li].len() as f32;
                line_cy[li] = line_cy[li] + (it.cy - line_cy[li]) / n;
            }
            None => {
                lines.push(vec![i]);
                line_cy.push(it.cy);
            }
        }
    }
    for l in &mut lines {
        l.sort_by(|&a, &b| items[a].cx.partial_cmp(&items[b].cx).unwrap());
    }
    lines.into_iter().flatten().map(|i| items[i].idx).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Point;

    fn q(x: f32, y: f32, w: f32, h: f32) -> Quad {
        Quad([
            Point::new(x, y),
            Point::new(x + w, y),
            Point::new(x + w, y + h),
            Point::new(x, y + h),
        ])
    }

    #[test]
    fn orders_two_lines_left_to_right() {
        let quads = vec![
            q(100.0, 50.0, 40.0, 10.0), // line 2, right
            q(10.0, 10.0, 40.0, 10.0),  // line 1, left
            q(10.0, 52.0, 40.0, 10.0),  // line 2, left (slightly offset)
            q(100.0, 12.0, 40.0, 10.0), // line 1, right
        ];
        assert_eq!(reading_order(&quads), vec![1, 3, 2, 0]);
    }
}
