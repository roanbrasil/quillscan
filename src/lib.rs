//! quillscan — an OpenCV-free OCR pipeline in pure Rust.
//!
//! The pipeline is the classic two-stage design used by PP-OCR:
//!
//! 1. **Detection**: a DBNet-style model turns the page into a probability map
//!    of "text here" pixels.
//! 2. **Post-processing**: the probability map becomes a list of quadrilaterals.
//!    This is the part that normally lives in OpenCV (`findContours`,
//!    `minAreaRect`, polygon offsetting, `warpPerspective`). Here it is written
//!    from scratch in [`contours`], [`geometry`] and [`warp`].
//! 3. **Recognition**: every quadrilateral is rectified into a horizontal strip
//!    and fed to a CRNN/SVTR model whose output is decoded with greedy CTC.
//!
//! Inference runs on [`tract`](https://github.com/sonos/tract), a pure-Rust
//! ONNX runtime, so the whole thing builds with `cargo build` and links no C or
//! C++ code.

pub mod contours;
pub mod engine;
pub mod geometry;
pub mod layout;
pub mod pipeline;
pub mod preprocess;
pub mod recognizer;
pub mod warp;

pub use geometry::{Point, Quad};
pub use pipeline::{ModelPaths, Ocr, RunOptions, TextLine, Timings};
