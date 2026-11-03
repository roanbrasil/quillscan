//! Thin wrapper around `tract-onnx`.
//!
//! Both PP-OCR models have dynamic spatial dimensions (the detector takes any
//! `H × W` multiple of 32; the recogniser takes `48 × W`). There are two ways
//! to handle that with tract:
//!
//! * **Concrete plans, cached per input size** (the default). The ONNX graph
//!   is parsed once; for each new `(h, w)` tract optimises a plan with the
//!   geometry fully known. Costs ~0.3 s per new size for the detector and
//!   ~0.1 s per width for the recogniser on a 2.8 GHz Xeon, then runs fast.
//!   The recogniser bounds the number of sizes by bucketing crop widths.
//! * **One symbolic plan** (`QUILLSCAN_SYMBOLIC_SHAPES=1`). No per-size cost,
//!   but on PP-OCRv4 the resulting plan runs about 5x slower, because the
//!   convolutions fall back to generic kernels when the geometry is a symbol.
//!   Getting it to optimise at all needs two tricks: declare the input as
//!   `32·h × 32·w` so the FPN's `⌊(H+31)/32⌋`-style expressions collapse, and
//!   assert `h >= 1` so tract's truncating-division simplifier can fire.
//!
//! Plans are immutable once built, so one `Model` can be shared across
//! threads behind an `Arc`; the cache is a mutex around a small map.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tract_onnx::prelude::*;

type Plan = Arc<TypedRunnableModel>;

#[allow(clippy::large_enum_variant)]
enum Inner {
    /// One plan with symbolic spatial dims.
    Symbolic(Plan),
    /// Symbolic optimisation failed; keep the typed graph and build one plan
    /// per concrete `(h, w)` on demand.
    PerShape {
        inference: InferenceModel,
        plans: Mutex<HashMap<(usize, usize), Plan>>,
    },
}

pub struct Model {
    inner: Inner,
}

/// Which input dimensions are fixed and which are free.
pub enum InputShape {
    /// `[1, 3, H, W]` with both free, each a multiple of `stride`.
    DynamicHW { stride: usize },
    /// `[1, 3, h, W]` with a fixed height and free width.
    FixedHeight(usize),
}

impl Model {
    pub fn load(path: &Path, shape: InputShape) -> Result<Self> {
        let mut model = tract_onnx::onnx()
            .model_for_path(path)
            .with_context(|| format!("loading ONNX model {}", path.display()))?;

        let h = model.symbols.sym("h");
        let w = model.symbols.sym("w");
        // tract only collapses `(32h + 31) / 32` to `h` when it can prove
        // `h >= 0`; these assertions give it that fact.
        model.symbols.add_assertion("h >= 1")?;
        model.symbols.add_assertion("w >= 1")?;
        let dims: TVec<TDim> = match shape {
            InputShape::DynamicHW { stride } => {
                tvec![1.to_dim(), 3.to_dim(), h.to_dim() * stride, w.to_dim() * stride]
            }
            InputShape::FixedHeight(hh) => tvec![1.to_dim(), 3.to_dim(), hh.to_dim(), w.to_dim()],
        };
        let inference = model.clone();
        model.set_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), dims))?;

        // Default: one plan per concrete input size, built on first use and
        // cached. Measured on PP-OCRv4, tract's plans optimised for a concrete
        // size run about 5x faster than a single plan with symbolic H/W (the
        // convolution kernels take a slower generic path when the geometry is
        // not known at optimisation time). The symbolic plan costs nothing per
        // new size, so it is kept as an opt-in for experiments.
        let symbolic = std::env::var_os("QUILLSCAN_SYMBOLIC_SHAPES").is_some();
        if !symbolic {
            return Ok(Self {
                inner: Inner::PerShape {
                    inference,
                    plans: Mutex::new(HashMap::new()),
                },
            });
        }

        match model.into_optimized().and_then(|m| m.into_runnable()) {
            Ok(plan) => Ok(Self {
                inner: Inner::Symbolic(plan),
            }),
            Err(e) => {
                eprintln!(
                    "quillscan: symbolic optimisation of {} failed ({e:#}); falling back to per-shape plans",
                    path.display()
                );
                Ok(Self {
                    inner: Inner::PerShape {
                        inference,
                        plans: Mutex::new(HashMap::new()),
                    },
                })
            }
        }
    }

    /// Build and cache the plan for one input size without running it.
    pub fn warm_up(&self, h: usize, w: usize) -> Result<()> {
        self.plan_for(h, w).map(|_| ())
    }

    /// Number of plans currently cached (1 for a symbolic model).
    pub fn cached_plans(&self) -> usize {
        match &self.inner {
            Inner::Symbolic(_) => 1,
            Inner::PerShape { plans, .. } => plans.lock().unwrap().len(),
        }
    }

    fn plan_for(&self, h: usize, w: usize) -> Result<Plan> {
        match &self.inner {
            Inner::Symbolic(p) => Ok(p.clone()),
            Inner::PerShape { inference, plans } => {
                if let Some(p) = plans.lock().unwrap().get(&(h, w)) {
                    return Ok(p.clone());
                }
                let mut model = inference.clone();
                model.set_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec![1, 3, h, w]))?;
                let plan = model
                    .into_optimized()
                    .context("optimising concretised graph")?
                    .into_runnable()?;
                plans.lock().unwrap().insert((h, w), plan.clone());
                Ok(plan)
            }
        }
    }

    /// Run on a single `[1, 3, h, w]` f32 buffer. Returns the first output as
    /// a flat `Vec<f32>` plus its shape.
    pub fn run(&self, data: Vec<f32>, h: usize, w: usize) -> Result<(Vec<f32>, Vec<usize>)> {
        let plan = self.plan_for(h, w)?;
        let input = Tensor::from_shape(&[1, 3, h, w], &data)?;
        let outputs = plan.run(tvec![input.into()])?;
        let out = outputs.into_iter().next().context("model produced no output")?;
        let shape = out.shape().to_vec();
        let data = out.into_tensor().into_plain_array::<f32>()?.into_raw_vec_and_offset().0;
        Ok((data, shape))
    }
}
