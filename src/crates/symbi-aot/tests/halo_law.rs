// =============================================================================
// halo_law.rs
//
// the ghost-width law: every registered kernel's
// stencil reach — read off its FieldLoadAt index expressions in the serialized
// neutral IR — must fit inside the allocated ghost halo. a stencil widened
// without widening the halo fails here at test time, naming the kernel, the
// field, and the axis at test time, before a widened stencil reads garbage at run time.
//
// three kernel families are exempted by design because they index through
// runtime-computed addressing, not fixed-offset stencils:
//   - ghost fills: the source coord picks periodic/reflect/outflow through a
//     runtime lattice-map select; the map keeps reads in range by construction
//   - refinement transfers (refine_*, wb_cf_*, wb_band_*): cross-grid addressing through
//     scaled coords (fine = 2*coarse); the transfer layer computes its own reach. the
//     departure encode/decode pairs carry the wb_cf_ and wb_band_ prefixes rather than
//     refine_, and are the same family: they live beside the restriction kernels, are
//     dispatched by the same transfer layer, and share one mechanical-chain walk.
//
//     that walk is why they are exempt rather than bounded. it reads at
//     `select(span >= t, base + sgn*t, coord)`, where `base` is the thread's coordinate
//     clamped into the uncovered reference row, `sgn` is the leg's direction as data, and
//     `span` is its length. a step past the leg's end takes the `coord` arm, and a step
//     within it walks from the reference toward the thread's own cell without overshooting
//     -- so every read lands on a cell the walk has already visited. that holds because
//     `sgn`, `span` and the mask are built from one subtraction and agree by construction,
//     which is a semantic invariant of the leg. a syntactic reach analysis reading index
//     expressions cannot establish it: taken apart, `base + sgn*t` is only bounded by the
//     unrolled chain length (WB_CF_CHAIN_MAX = 4, doubled for the band encode), and the
//     host dispatch asserts every band extent against that bound before launching.
//   - field_axpy_shift: reads at a runtime shift parameter bounded by dispatch
// an unbounded kernel outside these families fails the law.
// =============================================================================

use symbi_aot::IR_BLOBS;
use symbi_ir::prepared_from_ir;
use symbi_ir::{AxisReach, stencil_reach};

// the ghost halo every sim allocates (the SimBuilder default). the plm
// reconstruction's -2..+1 fan on the flux axis fits exactly.
const NG: u32 = 2;

// the halo a ppm sim must allocate: the monotonized-parabola face pair loads
// -3..+2 along the sweep, so the `_ppm` kernel family contracts for one more
// ghost cell than the plm default. dispatching a `_ppm` kernel into an ng = 2
// allocation is refused at kernel-set construction.
const NG_PPM: u32 = 3;

/// the halo a kernel's name contracts for: the `_ppm` tag widens the law's
/// bound; every other kernel holds the plm default.
fn expected_ng(name: &str) -> u32 {
    if name.contains("_ppm") { NG_PPM } else { NG }
}

// kernel-name families whose index expressions are runtime-directed rather
// than fixed-offset stencils; unbounded reach is their design.
const UNBOUNDED_BY_DESIGN: &[&str] =
    &["ghost_fill", "refine_", "wb_cf_", "wb_band_", "field_axpy_shift"];

// diagnostic census of the whole registry: reach per kernel/field/axis.
#[test]
#[ignore = "diagnostic census, run explicitly with -- --ignored --nocapture"]
fn halo_census() {
    for (name, ir) in IR_BLOBS {
        let report = stencil_reach(&prepared_from_ir(ir).scalarized);
        if report.per_field.is_empty() {
            continue;
        }
        println!("{name}:");
        for (field, axes) in &report.per_field {
            println!("  {field}: {axes:?}");
        }
    }
}

#[test]
fn every_registered_kernel_fits_the_ghost_halo() {
    let mut violations = Vec::new();
    for (name, ir) in IR_BLOBS {
        let exempt = UNBOUNDED_BY_DESIGN.iter().any(|fam| name.contains(fam));
        let ng = expected_ng(name);
        let report = stencil_reach(&prepared_from_ir(ir).scalarized);
        for (field, axes) in &report.per_field {
            for (axis, reach) in axes.iter().enumerate() {
                match reach {
                    AxisReach::Bounded(w) if *w > ng => violations.push(format!(
                        "{name}: field '{field}' axis {axis} reaches {w} > ng = {ng}"
                    )),
                    AxisReach::Bounded(_) => {}
                    AxisReach::Unbounded if !exempt => violations.push(format!(
                        "{name}: field '{field}' axis {axis} has reach the analysis cannot \
                         bound, and the kernel is not in an unbounded-by-design family"
                    )),
                    AxisReach::Unbounded => {}
                }
            }
        }
    }
    assert!(
        violations.is_empty(),
        "the ghost-width law failed:\n{}",
        violations.join("\n"),
    );
}

// the positive control: confirms the law fires on a real stencil, not a
// vacuous pass. plm reconstruction's -2..+1 fan on the flux axis is a pinned
// fact of the discretization — if the analysis stops seeing it, the law
// above has gone blind while still reporting clean.
#[test]
fn plm_face_flux_reach_is_two_on_the_flux_axis() {
    let report = stencil_reach(&prepared_from_ir(flux_blob()).scalarized);
    assert_eq!(
        report.per_field["prim_rho"],
        vec![AxisReach::Bounded(2), AxisReach::Bounded(0)],
        "plm stencil reach on the axis-0 face flux",
    );
    assert!(report.unbounded().is_empty());
}

// the ppm counterpart: the -3..+2 parabola fan must be visible to the analysis
// as reach exactly 3 on the sweep axis and 0 transverse — wider means a stencil
// bug, narrower means the parabola quietly collapsed toward the linear fan.
#[test]
fn ppm_face_flux_reach_is_three_on_the_flux_axis() {
    let blob = IR_BLOBS
        .iter()
        .find(|(name, _)| *name == "adiabatic_face_flux_ppm_2d_0")
        .map(|(_, ir)| *ir)
        .expect("adiabatic_face_flux_ppm_2d_0 missing from the registry");
    let report = stencil_reach(&prepared_from_ir(blob).scalarized);
    assert_eq!(
        report.per_field["prim_rho"],
        vec![AxisReach::Bounded(3), AxisReach::Bounded(0)],
        "ppm stencil reach on the axis-0 face flux",
    );
    assert!(report.unbounded().is_empty());
}

// bug injection through the serialized artifact: widen the real blob's -2
// stencil offset to -3 and assert the analysis reports the violation with the
// field and axis attached. this exercises the whole audit path — serde, the
// let-environment resolution, the reach join, and the classifier.
#[test]
fn widened_stencil_in_a_real_blob_breaks_the_law() {
    let widened = flux_blob().replace("\"I32\":-2", "\"I32\":-3");
    assert_ne!(widened, flux_blob(), "injection site missing from the blob");
    let report = stencil_reach(&prepared_from_ir(&widened).scalarized);
    let over_halo: Vec<(&String, usize, u32)> = report
        .per_field
        .iter()
        .flat_map(|(field, axes)| {
            axes.iter()
                .enumerate()
                .filter_map(move |(axis, r)| match r {
                    AxisReach::Bounded(w) if *w > NG => Some((field, axis, *w)),
                    _ => None,
                })
        })
        .collect();
    assert!(
        over_halo.iter().all(|(_, axis, w)| *axis == 0 && *w == 3) && !over_halo.is_empty(),
        "the widened flux-axis offset must surface as reach 3 on axis 0, got {over_halo:?}",
    );
}

fn flux_blob() -> &'static str {
    IR_BLOBS
        .iter()
        .find(|(name, _)| *name == "adiabatic_face_flux_2d_0")
        .map(|(_, ir)| *ir)
        .expect("adiabatic_face_flux_2d_0 missing from the registry")
}
