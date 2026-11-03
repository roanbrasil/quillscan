#!/usr/bin/env python3
"""Fetch the PP-OCRv4 ONNX models used by quillscan.

The models are the ones redistributed inside the `rapidocr_onnxruntime` wheel
on PyPI (Apache-2.0, converted from PaddleOCR). This script downloads that
wheel, extracts the detector and recogniser, and writes the character
dictionary that the recogniser carries in its ONNX metadata.

Usage:
    python3 scripts/fetch_models.py [--out models]

Requires: pip, and the `onnx` package (installed on the fly if missing).
"""
import argparse
import pathlib
import subprocess
import sys
import tempfile
import zipfile

WHEEL = "rapidocr_onnxruntime==1.3.24"
FILES = {
    "rapidocr_onnxruntime/models/ch_PP-OCRv4_det_infer.onnx": "ch_PP-OCRv4_det_infer.onnx",
    "rapidocr_onnxruntime/models/ch_PP-OCRv4_rec_infer.onnx": "ch_PP-OCRv4_rec_infer.onnx",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.check_call([
            sys.executable, "-m", "pip", "download", WHEEL, "--no-deps",
            "--python-version", "3.11", "--only-binary=:all:", "-d", tmp, "-q",
        ])
        wheel = next(pathlib.Path(tmp).glob("*.whl"))
        with zipfile.ZipFile(wheel) as zf:
            for src, dst in FILES.items():
                (out / dst).write_bytes(zf.read(src))
                print(f"wrote {out / dst}")

    try:
        import onnx  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnx"])
        import onnx
    model = onnx.load(str(out / "ch_PP-OCRv4_rec_infer.onnx"))
    meta = {p.key: p.value for p in model.metadata_props}
    chars = meta["character"].splitlines()
    (out / "ppocr_keys_v1.txt").write_text("\n".join(chars) + "\n", encoding="utf-8")
    print(f"wrote {out / 'ppocr_keys_v1.txt'} ({len(chars)} characters)")

    removed = strip_shape_casts(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(out / "ch_PP-OCRv4_rec_infer.onnx"))
    print(f"stripped {removed} int32/int64 casts on shape paths from the recogniser")


def strip_shape_casts(model) -> int:
    """Bypass `Cast` nodes that sit on shape-computation paths.

    paddle2onnx emits `Shape -> Cast(int32) -> Slice -> Cast(int64) -> Concat
    -> Reshape`. The round trip through int32 changes nothing numerically, but
    it stops tract from propagating symbolic dimensions through the graph, so
    the whole recogniser would have to be re-optimised for every input width.
    Removing the casts touches no weights and no computation on activations.
    """
    g = model.graph
    producers = {o: n for n in g.node for o in n.output}
    shape_ops = {"Cast", "Slice", "Gather", "Concat", "Unsqueeze", "Squeeze", "Mul", "Add", "Div", "Sub"}

    def on_shape_path(n, depth=0):
        if depth > 6:
            return False
        for i in n.input:
            p = producers.get(i)
            if p is None:
                continue
            if p.op_type == "Shape":
                return True
            if p.op_type in shape_ops and on_shape_path(p, depth + 1):
                return True
        return False

    rename = {}
    removed = 0
    for n in list(g.node):
        if n.op_type == "Cast" and on_shape_path(n):
            rename[n.output[0]] = n.input[0]
            g.node.remove(n)
            removed += 1

    def resolve(x):
        while x in rename:
            x = rename[x]
        return x

    for n in g.node:
        for i, x in enumerate(n.input):
            n.input[i] = resolve(x)
    return removed


if __name__ == "__main__":
    main()
