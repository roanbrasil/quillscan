#!/usr/bin/env python3
"""Put a quillscan bench report and a baseline report side by side.

    cargo run --release -- bench tests/fixtures/invoice_synthetic.png --json > bench/out/quillscan.json
    python3 bench/baseline_rapidocr.py tests/fixtures/invoice_synthetic.png > bench/out/rapidocr.json
    python3 bench/compare.py bench/out/quillscan.json bench/out/rapidocr.json

Prints a Markdown table you can paste into a write-up. Numbers are whatever
the two reports contain; nothing is estimated.
"""
import json
import sys


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    reports = [json.load(open(p)) for p in sys.argv[1:]]
    names = [r.get("engine", "quillscan (pure Rust, tract)") for r in reports]
    rows = [
        ("Model load (ms)", "load_ms", "{:.0f}"),
        ("Cold first run (ms)", "cold_first_run_ms", "{:.0f}"),
        ("Time to first result (ms)", "_ttfr", "{:.0f}"),
        ("Mean latency (ms)", "mean_ms", "{:.1f}"),
        ("p50 latency (ms)", "p50_ms", "{:.1f}"),
        ("p95 latency (ms)", "p95_ms", "{:.1f}"),
        ("Min latency (ms)", "min_ms", "{:.1f}"),
        ("Peak RSS (MB)", "peak_rss_mb", "{:.0f}"),
        ("Lines found", "lines_found", "{}"),
    ]
    print("| Metric | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for r in reports:
        if r.get("load_ms") is not None and r.get("cold_first_run_ms") is not None:
            r["_ttfr"] = r["load_ms"] + r["cold_first_run_ms"]
    for label, key, fmt in rows:
        cells = []
        for r in reports:
            v = r.get(key)
            cells.append(fmt.format(v) if v is not None else "n/a")
        print(f"| {label} | " + " | ".join(cells) + " |")
    size = reports[0].get("image_size")
    if size:
        print(f"\nImage: {reports[0]['image']} ({size[0]}×{size[1]}), {reports[0]['iters']} iterations after warm-up.")


if __name__ == "__main__":
    main()
