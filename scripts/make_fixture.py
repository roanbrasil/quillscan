#!/usr/bin/env python3
"""Render the synthetic invoice used by the end-to-end test.

Draws the strings listed in tests/pipeline.rs (EXPECTED) on an A4-sized page
(1240x1754, 150 dpi), so the test has an image with known ground truth. Keep
the two in sync.

Usage:
    python3 scripts/make_fixture.py [--out tests/fixtures/invoice_synthetic.png]

Requires: Pillow. Uses DejaVu Sans if it is installed, else Pillow's built-in
scalable font.
"""
import argparse
import pathlib

from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/Library/Fonts/DejaVuSans.ttf",
    "C:/Windows/Fonts/DejaVuSans.ttf",
    "DejaVuSans.ttf",
]


def font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


# (x, y, size, text). Columns follow a plain invoice table.
COLS = (80, 600, 760, 980)
ROWS = [
    ("Description", "Qty", "Unit", "Total"),
    ("OCR pipeline review", "8", "150.00", "1200.00"),
    ("Rust performance audit", "12", "150.00", "1800.00"),
    ("Deployment hardening", "4", "175.00", "700.00"),
    ("Subtotal", "", "", "3700.00"),
    ("Tax GST 5%", "", "", "185.00"),
    ("Tax QST 9.975%", "", "", "369.08"),
    ("TOTAL DUE", "", "", "4254.08"),
]


def layout():
    items = [
        (80, 80, 28, "INVOICE #2026-0411"),
        (80, 140, 28, "Daath.ai Consulting Inc."),
        (80, 180, 20, "123 Rue Saint-Charles, Longueuil QC"),
    ]
    y = 250
    for row in ROWS:
        for x, text in zip(COLS, row):
            if text:
                items.append((x, y, 20, text))
        y += 44
    for i, text in enumerate(
        ["Payment terms: Net 30 days", "Bank transfer reference QS-0411", "Thank you for your business"]
    ):
        items.append((80, 662 + 40 * i, 20, text))
    items.append((700, 1500, 20, "Page 1 of 1"))
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/fixtures/invoice_synthetic.png")
    args = ap.parse_args()

    img = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(img)
    for x, y, size, text in layout():
        draw.text((x, y), text, fill="black", font=font(size))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
