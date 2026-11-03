#!/usr/bin/env python3
"""Baseline: the conventional OpenCV + ONNX Runtime pipeline (RapidOCR).

Runs the *same* PP-OCRv4 ONNX files from `models/` through `rapidocr`, which
uses OpenCV for pre/post-processing and ONNX Runtime (C++) for inference. So
the comparison with quillscan isolates the runtime and the glue code, not the
models.

Usage:
    python3 -m venv .venv && . .venv/bin/activate
    pip install rapidocr onnxruntime opencv-python-headless
    python3 bench/baseline_rapidocr.py tests/fixtures/invoice_synthetic.png --iters 20

Outputs a JSON object with the same keys as `quillscan bench --json`, so the
two can be put side by side by bench/compare.py.

Settings mirror quillscan's defaults: longest side 960, det thresh 0.3,
box thresh 0.6, unclip 1.6, no dilation, no angle classifier.
"""
import argparse
import json
import pathlib
import resource
import statistics
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--models", default="models")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--max-side", type=int, default=960)
    ap.add_argument("--threads", type=int, default=-1, help="ONNX Runtime intra-op threads (-1 = all)")
    args = ap.parse_args()
    models = pathlib.Path(args.models)

    t0 = time.perf_counter()
    import cv2  # noqa: E402
    from rapidocr import RapidOCR  # noqa: E402
    from rapidocr.utils.typings import ModelType, OCRVersion  # noqa: E402

    engine = RapidOCR(
        params={
            "Global.use_cls": False,
            "Global.use_preprocess_img": False,
            "Global.text_score": 0.5,
            "EngineConfig.onnxruntime.intra_op_num_threads": args.threads,
            "Det.ocr_version": OCRVersion.PPOCRV4,
            "Det.model_type": ModelType.MOBILE,
            "Det.model_path": str(models / "ch_PP-OCRv4_det_infer.onnx"),
            "Det.limit_side_len": args.max_side,
            "Det.limit_type": "max",
            "Det.mean": [0.485, 0.456, 0.406],
            "Det.std": [0.229, 0.224, 0.225],
            "Det.thresh": 0.3,
            "Det.box_thresh": 0.6,
            "Det.unclip_ratio": 1.6,
            "Det.use_dilation": False,
            "Rec.ocr_version": OCRVersion.PPOCRV4,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.model_path": str(models / "ch_PP-OCRv4_rec_infer.onnx"),
            "Rec.rec_keys_path": str(models / "ppocr_keys_v1.txt"),
        }
    )
    load_ms = (time.perf_counter() - t0) * 1000

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"could not read {args.image}")

    t = time.perf_counter()
    engine(img)
    cold_first_run_ms = (time.perf_counter() - t) * 1000
    for _ in range(1, args.warmup):
        engine(img)

    totals = []
    lines = 0
    for _ in range(args.iters):
        t = time.perf_counter()
        result = engine(img)
        totals.append((time.perf_counter() - t) * 1000)
        lines = len(result.txts or []) if result is not None and result.txts is not None else 0

    totals.sort()
    pct = lambda p: totals[round((len(totals) - 1) * p)]  # noqa: E731
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss_kb / 1024 if sys.platform != "darwin" else rss_kb / (1024 * 1024)

    report = {
        "engine": "rapidocr (OpenCV + ONNX Runtime)",
        "image": args.image,
        "image_size": [img.shape[1], img.shape[0]],
        "iters": args.iters,
        "lines_found": lines,
        "load_ms": load_ms,
        "cold_first_run_ms": cold_first_run_ms,
        "mean_ms": statistics.fmean(totals),
        "p50_ms": pct(0.5),
        "p95_ms": pct(0.95),
        "min_ms": totals[0],
        "peak_rss_mb": rss_mb,
        "texts": list(result.txts) if result is not None and result.txts is not None else [],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
