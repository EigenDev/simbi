// =============================================================================
// field_layout.rs
//
// regimespec-driven field naming. `RegimeSpec.fields` declares every
// conservation law (`den`, `mom`, `nrg`, `mag`) with its
// `FieldKind::{Scalar, DimVector, FixedVector}`, so the i/o layer walks
// THAT to spell the on-disk names rather than hardcoding `rho`, `v1..vD`,
// `den`, `m1..mD`, `nrg`, `b1..bD`.
//
// this module exposes the canonical naming convention as DATA — one match
// per (FieldKind × per-component index) instead of fan-out across writer and
// reader. callers ask `cons_field_names(D, regime_spec)` and get the exact
// list of strings the on-disk layout uses.
// =============================================================================

use symbi_hydro::{FieldKind, FieldSpec};

/// the canonical on-disk dataset name for one COMPONENT of one FieldSpec.
/// - `Scalar`           → `fs.name` ("den", "nrg", "rho", ...)
/// - `DimVector`        → `m1..mD` for momentum, `v1..vD` for velocity, ...
/// - `FixedVector { n }`→ `b1..bn`  (e.g. magnetic, always 3-component)
///
/// the (NAME → SUFFIX) mapping is the canonical on-disk convention — existing
/// checkpoint files + every `scripts/plot_*.py` read identical paths.
pub fn dataset_name(fs: &FieldSpec, idx: usize) -> String {
    match fs.kind {
        FieldKind::Scalar => fs.name.to_string(),
        FieldKind::DimVector => format!("{}{}", short_prefix(fs.name), idx + 1),
        FieldKind::FixedVector { .. } => format!("{}{}", short_prefix(fs.name), idx + 1),
    }
}

/// `den` → `m`, `vel` → `v`, `mag` → `b` — the canonical single-letter
/// on-disk prefix. momentum uses `m` (conserved) and `v` for primitive
/// velocity; the FieldSpec on cons carries `"mom"` and on prim carries
/// `"vel"`.
fn short_prefix(field_name: &str) -> &'static str {
    match field_name {
        "mom" => "m",
        "vel" => "v",
        "mag" => "b",
        "bcell" => "b",      // primitive cell-centered B
        "bface" => "B",      // face-centered B (CT ground truth)
        // unknown — fall back to the field name itself (least surprising for
        // a future regime).
        other => Box::leak(other.to_string().into_boxed_str()),
    }
}

/// the canonical iteration count for a FieldSpec at simulation dim D.
pub fn component_count(fs: &FieldSpec, ndim: usize) -> usize {
    match fs.kind {
        FieldKind::Scalar => 1,
        FieldKind::DimVector => ndim,
        FieldKind::FixedVector { components } => components as usize,
    }
}

/// produce every (component-index, dataset-name) pair for one FieldSpec at
/// dimension `D`. for `mom @ D=3` returns `[(0, "m1"), (1, "m2"), (2, "m3")]`.
pub fn iter_components<'a>(fs: &'a FieldSpec, ndim: usize)
    -> impl Iterator<Item = (usize, String)> + 'a
{
    (0..component_count(fs, ndim)).map(move |k| (k, dataset_name(fs, k)))
}
