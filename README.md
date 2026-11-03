# quillscan

OCR in pure Rust, with no OpenCV and no C++ inference runtime.

quillscan runs the PP-OCRv4 detection and recognition models on
[tract](https://github.com/sonos/tract), a pure-Rust ONNX runtime, and
replaces every piece of OpenCV that an OCR pipeline normally depends on with a
few hundred lines of plain Rust:

| OpenCV call | quillscan replacement | Where |
|---|---|---|
| `cv::resize` + mean/std normalisation | `image` crate resize + hand-written NCHW packing | `src/preprocess.rs` |
| `cv::findContours` | flood-fill connected components; only edge pixels are kept | `src/contours.rs` |
| `cv::minAreaRect` | convex hull (monotone chain) + rotating calipers | `src/geometry.rs` |
| Vatti polygon offsetting (`pyclipper`) | edge-normal offset of a rectangle (exact for quads) | `src/geometry.rs` |
| `cv::getPerspectiveTransform` + `cv::warpPerspective` | 8×8 Gaussian elimination + inverse-mapped bilinear sampling | `src/warp.rs` |
| `cv::rotate` | `image::imageops::rotate90` | `src/warp.rs` |

The project exists to test a claim, not to sell one: *can a document OCR
pipeline drop OpenCV and a C++ runtime without losing speed or accuracy?* The
`bench` subcommand and `bench/` scripts measure exactly that on your own
machine. The numbers below come from a 2-vCPU cloud VM; treat them as one
data point and run your own.

| 1240×1754 page, 2 vCPU | quillscan (tract) | rapidocr (OpenCV + ORT) |
|---|---|---|
| p50 latency | 1,710 ms | 2,667 ms |
| peak RSS | 222 MB | 345 MB |
| time to first result | 3.0 s | 3.7 s |
| lines exact | 31 / 31 | 31 / 31 |

## Quick start

```bash
git clone https://github.com/roanbrasil/quillscan
cd quillscan
python3 scripts/fetch_models.py      # ~15 MB, from the rapidocr_onnxruntime wheel on PyPI
python3 scripts/make_fixture.py      # renders the test invoice (needs Pillow)
cargo run --release -- run tests/fixtures/invoice_synthetic.png
```

```
INVOICE #2026-0411
Daath.ai Consulting Inc.
...
[31 lines | det 683.1 ms | post 2.0 ms | rec 902.7 ms | total 1711.2 ms]   # 2-vCPU Xeon, warm
```

Other subcommands:

```bash
quillscan run image.png --json          # lines with quads, confidences, per-stage timings
quillscan detect image.png              # boxes only
quillscan bench image.png --iters 20    # latency distribution, stage breakdown, peak RSS
```

Tuning flags (all global): `--max-side 960`, `--det-thresh 0.3`,
`--box-thresh 0.6`, `--unclip-ratio 1.6`, `--min-text-conf 0.5`, `--sequential`.

## Library use

```rust
use quillscan::{ModelPaths, Ocr, RunOptions};
use std::sync::Arc;

let ocr = Arc::new(Ocr::load(&ModelPaths::ppocr_v4("models"))?);
let img = image::open("receipt.jpg")?.into_rgb8();

// `Ocr` is immutable after load; `RunOptions` is plain data. Share the
// engine across threads and build options per request.
let (lines, timings) = ocr.run(&img, &RunOptions::default())?;
for line in &lines {
    println!("{:.2}  {}", line.confidence, line.text);
}
```

## How the post-processing works

DBNet outputs a probability map at the detector's input resolution. Turning
that into text boxes is usually done with OpenCV; here it is done like this:

1. **Threshold** the map (`det_thresh`) into a bitmap.
2. **Flood-fill** 8-connected components. While filling, record each
   component's mean probability (that is the box score) and keep only the
   pixels that touch background (the edge pixels).
3. **Minimum-area rectangle** of the edge pixels: convex hull by monotone
   chain, then rotating calipers over the hull edges. One hull side is always
   flush with the optimal rectangle, so trying each edge as the orientation is
   exact.
4. **Expand** the rectangle by `d = area × unclip_ratio / perimeter`, the
   DBNet formula. PaddleOCR does this with a general polygon-offsetting
   library because its boxes can be polygons; for rectangles the offset is just
   a bigger rectangle, so it is four edge normals and no clipping code.
5. **Reading order**: cluster boxes by vertical centre, sort each cluster by x.
6. **Rectify** each box with a homography solved by Gaussian elimination and
   inverse-mapped bilinear sampling, rotating 90° when the crop is much taller
   than wide.

Crops are rectified and recognised in parallel with Rayon. The CTC decoder is
greedy: argmax per time step, collapse repeats, drop blanks.

## Dynamic shapes and tract (read this before benchmarking)

Both models take variable-size inputs. tract can compile a single plan with
symbolic `H`/`W`, and that is what I built first; on PP-OCRv4 it runs about
**5× slower** than a plan optimised for a concrete size (on a 2.8 GHz Xeon
vCPU: detector 3.5 s → 0.65 s, recogniser 4.9 s → 0.78 s for the test page).
So the default is:

- **Detector**: one plan per distinct input size, built on first use (~0.3 s)
  and cached. Document pipelines see few distinct sizes.
- **Recogniser**: crop widths are rounded up to one of ten buckets
  (`128 … 3072`), the crop is left-aligned and the rest zero-padded, and one
  plan per bucket is built on first use (~0.1 s each). This is the same
  pad-to-common-width idea PaddleOCR uses for batches.
- `Ocr::warm_up()` pre-builds all recogniser plans and the detector plan for a
  given page size, for servers that care about the first request.

`QUILLSCAN_SYMBOLIC_SHAPES=1` switches back to the single symbolic plan; it is
kept for experiments. Getting that plan to compile needs two tricks that are
documented in `src/engine.rs` (declare the input as `32·h × 32·w`, and assert
`h >= 1` so tract's integer-division simplifier can prove the FPN shapes
match). The recogniser additionally needs the `Cast` surgery described under
"Models and licensing", with or without symbolic shapes.

What is deliberately **not** here: an angle classifier (PP-OCR's optional
180° detector), polygon (non-rectangular) boxes, batching of crops by aspect
ratio, and INT8 models. Each is a reasonable next step.

## Benchmarking against the OpenCV pipeline

The baseline is `rapidocr`, which runs the *same* ONNX files
through OpenCV + ONNX Runtime. That isolates the runtime and glue code from
the models.

```bash
mkdir -p bench/out
cargo run --release -- bench tests/fixtures/invoice_synthetic.png --iters 20 --json > bench/out/quillscan.json

python3 -m venv .venv && . .venv/bin/activate
pip install rapidocr onnxruntime opencv-python-headless
python3 bench/baseline_rapidocr.py tests/fixtures/invoice_synthetic.png --iters 20 > bench/out/rapidocr.json

python3 bench/compare.py bench/out/quillscan.json bench/out/rapidocr.json
```

`compare.py` prints a Markdown table of load time, mean/p50/p95 latency, peak
RSS and line count for both engines. Also worth recording: binary size
(`ls -la target/release/quillscan`) versus the baseline's installed footprint
(`du -sh .venv`), a Docker image built from `Dockerfile` versus one from
`bench/Dockerfile.baseline`, and the `cold_first_run_ms` field, which is where
quillscan pays for building tract plans.

Things that make the comparison fair:

- same models, same `max_side`, angle classification off on both sides;
- warm-up iterations before timing;
- report the distribution (p50/p95), not only the mean;
- note the CPU, core count and whether SIMD features were enabled
  (`RUSTFLAGS="-C target-cpu=native"` changes tract's numbers a lot).

## Accuracy check

`scripts/make_fixture.py` renders `tests/fixtures/invoice_synthetic.png` from
known strings (the `EXPECTED` list in `tests/pipeline.rs`). `cargo test --release -- --ignored`
runs the full pipeline on it and asserts that every expected line is
recognised. For a real accuracy comparison use a public benchmark set
(ICDAR 2015, or the receipts in SROIE) and compare line-level exact match
against the baseline.

## Dependencies

`tract-onnx`, `image`, `rayon`, `clap`, `serde`, `anyhow`, `thiserror`.
Nothing in `cargo tree` builds or links C or C++ (the only `-sys` crate is
`linux-raw-sys`, which is Rust constants for syscalls). A plain `cargo build`
is the whole build.

## Models and licensing

Code: MIT. Models: PP-OCRv4 from PaddleOCR (Apache-2.0), not stored in this
repository. `python3 scripts/fetch_models.py` writes them to `models/`:

| File | What |
|---|---|
| `ch_PP-OCRv4_det_infer.onnx` | DBNet-style text detector (PP-OCRv4 mobile), byte-identical to upstream |
| `ch_PP-OCRv4_rec_infer.onnx` | SVTR-LCNet text recogniser (PP-OCRv4 mobile), Chinese + Latin, with the edit below |
| `ppocr_keys_v1.txt` | 6623-entry character dictionary, read from the recogniser's ONNX metadata (`character` key) |

Source: the ONNX exports redistributed in the `rapidocr_onnxruntime` 1.3.24
wheel on PyPI.

The recogniser has one structural edit: paddle2onnx emits
`Shape → Cast(int32) → Slice → Cast(int64)` chains that feed `Reshape`. The
int32 round trip is a no-op on the values but prevents tract from propagating
symbolic dimensions, so `fetch_models.py` bypasses those eight casts. No
weights and no activation ops are changed; outputs are identical to the
upstream model.

## Related work

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — the models.
- [RapidOCR](https://github.com/RapidAI/RapidOCR) — the OpenCV + ONNX Runtime
  baseline used in `bench/`.
- [ocrs](https://github.com/robertknight/ocrs) and
  [rusto-rs](https://github.com/byrizki/rusto-rs) — other Rust OCR engines;
  both use the `rten` runtime. quillscan was written independently on `tract`
  and shares no code with either.
