use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use quillscan::{ModelPaths, Ocr, RunOptions, Timings};
use serde::Serialize;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(name = "quillscan", version, about = "OpenCV-free OCR in pure Rust")]
struct Cli {
    /// Directory containing the PP-OCRv4 model files.
    #[arg(long, default_value = "models", global = true)]
    models: PathBuf,
    /// Longest side of the detector input.
    #[arg(long, default_value_t = 960, global = true)]
    max_side: u32,
    #[arg(long, default_value_t = 0.3, global = true)]
    det_thresh: f32,
    #[arg(long, default_value_t = 0.6, global = true)]
    box_thresh: f32,
    #[arg(long, default_value_t = 1.6, global = true)]
    unclip_ratio: f32,
    #[arg(long, default_value_t = 0.5, global = true)]
    min_text_conf: f32,
    /// Disable Rayon parallelism in crop and recognition stages.
    #[arg(long, global = true)]
    sequential: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// OCR one image and print the text (or JSON).
    Run {
        image: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Detect boxes only, printed as JSON.
    Detect {
        image: PathBuf,
        /// Also write the detector's probability map as a grayscale PNG.
        #[arg(long)]
        dump_prob: Option<PathBuf>,
    },
    /// Time the pipeline on one image, report per-stage latency and peak RSS.
    Bench {
        image: PathBuf,
        #[arg(long, default_value_t = 20)]
        iters: usize,
        #[arg(long, default_value_t = 3)]
        warmup: usize,
        #[arg(long)]
        json: bool,
    },
}

fn opts_from(cli: &Cli) -> RunOptions {
    RunOptions {
        max_side: cli.max_side,
        det_thresh: cli.det_thresh,
        box_thresh: cli.box_thresh,
        unclip_ratio: cli.unclip_ratio,
        min_text_conf: cli.min_text_conf,
        parallel: !cli.sequential,
    }
}

fn load_image(p: &PathBuf) -> Result<image::RgbImage> {
    Ok(image::open(p)
        .with_context(|| format!("opening {}", p.display()))?
        .into_rgb8())
}

#[derive(Serialize)]
struct BenchReport {
    image: String,
    image_size: [u32; 2],
    iters: usize,
    lines_found: usize,
    load_ms: f64,
    /// Very first run, which includes building tract plans for every new
    /// input size (detector size + recogniser width buckets).
    cold_first_run_ms: f64,
    mean_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    min_ms: f64,
    stage_mean_ms: StageMs,
    peak_rss_mb: Option<f64>,
    threads: usize,
}

#[derive(Serialize, Default)]
struct StageMs {
    preprocess: f64,
    detect: f64,
    postprocess: f64,
    crop: f64,
    recognise: f64,
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Peak resident set size, from /proc on Linux. `None` elsewhere.
fn peak_rss_mb() -> Option<f64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = s.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024.0)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let opts = opts_from(&cli);
    let paths = ModelPaths::ppocr_v4(&cli.models);

    match &cli.cmd {
        Cmd::Run { image, json } => {
            let ocr = Ocr::load(&paths)?;
            let img = load_image(image)?;
            let (lines, t) = ocr.run(&img, &opts)?;
            if *json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({ "lines": lines, "timings_ms": timings_ms(&t) }))?
                );
            } else {
                for l in &lines {
                    println!("{}", l.text);
                }
                eprintln!(
                    "[{} lines | det {:.1} ms | post {:.1} ms | rec {:.1} ms | total {:.1} ms]",
                    lines.len(),
                    ms(t.detect),
                    ms(t.postprocess),
                    ms(t.recognise),
                    ms(t.total)
                );
            }
        }
        Cmd::Detect { image, dump_prob } => {
            let ocr = Ocr::load(&paths)?;
            let img = load_image(image)?;
            let (boxes, t) = match dump_prob {
                Some(path) => {
                    let (boxes, t, map) = ocr.detect_with_map(&img, &opts)?;
                    map.save(path).with_context(|| format!("writing {}", path.display()))?;
                    (boxes, t)
                }
                None => ocr.detect(&img, &opts)?,
            };
            let out: Vec<_> = boxes
                .iter()
                .map(|(q, s)| serde_json::json!({ "quad": q, "score": s }))
                .collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({ "boxes": out, "timings_ms": timings_ms(&t) }))?
            );
        }
        Cmd::Bench {
            image,
            iters,
            warmup,
            json,
        } => {
            let tl = Instant::now();
            let ocr = Ocr::load(&paths)?;
            let load_ms = ms(tl.elapsed());
            let img = load_image(image)?;
            let tc = Instant::now();
            ocr.run(&img, &opts)?;
            let cold_first_run_ms = ms(tc.elapsed());
            for _ in 1..*warmup {
                ocr.run(&img, &opts)?;
            }
            let mut totals: Vec<f64> = Vec::with_capacity(*iters);
            let mut stage = StageMs::default();
            let mut lines_found = 0;
            for _ in 0..*iters {
                let (lines, t) = ocr.run(&img, &opts)?;
                lines_found = lines.len();
                totals.push(ms(t.total));
                stage.preprocess += ms(t.preprocess);
                stage.detect += ms(t.detect);
                stage.postprocess += ms(t.postprocess);
                stage.crop += ms(t.crop);
                stage.recognise += ms(t.recognise);
            }
            let n = *iters as f64;
            for v in [
                &mut stage.preprocess,
                &mut stage.detect,
                &mut stage.postprocess,
                &mut stage.crop,
                &mut stage.recognise,
            ] {
                *v /= n;
            }
            totals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let pct = |p: f64| totals[((totals.len() - 1) as f64 * p).round() as usize];
            let report = BenchReport {
                image: image.display().to_string(),
                image_size: [img.width(), img.height()],
                iters: *iters,
                lines_found,
                load_ms,
                cold_first_run_ms,
                mean_ms: totals.iter().sum::<f64>() / n,
                p50_ms: pct(0.5),
                p95_ms: pct(0.95),
                min_ms: totals[0],
                stage_mean_ms: stage,
                peak_rss_mb: peak_rss_mb(),
                threads: rayon::current_num_threads(),
            };
            if *json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!(
                    "image      : {} ({}x{})",
                    report.image, report.image_size[0], report.image_size[1]
                );
                println!("model load : {:.1} ms (parse only)", report.load_ms);
                println!(
                    "cold run   : {:.1} ms (first image, builds plans)",
                    report.cold_first_run_ms
                );
                println!("lines      : {}", report.lines_found);
                println!(
                    "latency    : mean {:.1} | p50 {:.1} | p95 {:.1} | min {:.1} ms  (n={})",
                    report.mean_ms, report.p50_ms, report.p95_ms, report.min_ms, report.iters
                );
                println!(
                    "stages     : pre {:.1} | det {:.1} | post {:.1} | crop {:.1} | rec {:.1} ms",
                    report.stage_mean_ms.preprocess,
                    report.stage_mean_ms.detect,
                    report.stage_mean_ms.postprocess,
                    report.stage_mean_ms.crop,
                    report.stage_mean_ms.recognise
                );
                match report.peak_rss_mb {
                    Some(r) => println!("peak RSS   : {r:.0} MB"),
                    None => println!("peak RSS   : n/a on this OS"),
                }
                println!("threads    : {}", report.threads);
            }
        }
    }
    Ok(())
}

fn timings_ms(t: &Timings) -> serde_json::Value {
    serde_json::json!({
        "preprocess": ms(t.preprocess),
        "detect": ms(t.detect),
        "postprocess": ms(t.postprocess),
        "crop": ms(t.crop),
        "recognise": ms(t.recognise),
        "total": ms(t.total),
    })
}
