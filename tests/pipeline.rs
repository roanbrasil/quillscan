//! End-to-end check on the synthetic invoice. Needs the models, so it is
//! `#[ignore]`d by default: `cargo test --release -- --ignored`.
//! Render the image first with `python3 scripts/make_fixture.py`.

use quillscan::{ModelPaths, Ocr, RunOptions};

/// Every line `scripts/make_fixture.py` draws on the invoice.
const EXPECTED: &[&str] = &[
    "INVOICE #2026-0411",
    "Daath.ai Consulting Inc.",
    "123 Rue Saint-Charles, Longueuil QC",
    "Description",
    "Qty",
    "Unit",
    "Total",
    "OCR pipeline review",
    "8",
    "150.00",
    "1200.00",
    "Rust performance audit",
    "12",
    "150.00",
    "1800.00",
    "Deployment hardening",
    "4",
    "175.00",
    "700.00",
    "Subtotal",
    "3700.00",
    "Tax GST 5%",
    "185.00",
    "Tax QST 9.975%",
    "369.08",
    "TOTAL DUE",
    "4254.08",
    "Payment terms: Net 30 days",
    "Bank transfer reference QS-0411",
    "Thank you for your business",
    "Page 1 of 1",
];

#[test]
#[ignore]
fn synthetic_invoice_is_read_exactly() {
    let ocr = Ocr::load(&ModelPaths::ppocr_v4("models")).expect("models present");
    let img = image::open("tests/fixtures/invoice_synthetic.png").unwrap().into_rgb8();
    let (lines, _) = ocr.run(&img, &RunOptions::default()).unwrap();
    let got: Vec<String> = lines.iter().map(|l| l.text.trim().to_string()).collect();
    let expected: Vec<String> = EXPECTED.iter().map(|s| s.to_string()).collect();
    let mut missing = Vec::new();
    for e in &expected {
        if !got.contains(e) {
            missing.push(e.clone());
        }
    }
    assert!(
        missing.is_empty(),
        "lines not recognised exactly: {missing:?}\nrecognised: {got:?}"
    );
    assert_eq!(got.len(), expected.len(), "unexpected extra lines: {got:?}");
}
