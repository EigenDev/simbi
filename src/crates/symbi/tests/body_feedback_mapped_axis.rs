// =============================================================================
// body_feedback_mapped_axis.rs
//
// the body-feedback receipt books the force the gas exerts on a gravitating
// body: the sum over cells of rho dV GM r / (|r|^2 + eps^2)^{3/2}, r the cell
// centroid relative to the body. on a 2D cartesian grid with a log-spaced x
// axis the centroids and cell volumes come from the axis map,
// `face(i) = x_lo 10^(i s)`, while the grid stores the nominal linear width as
// its spacing. the expected force is summed here from the face positions; a
// uniform x axis is the control.
//
// run: cargo test -p symbi --test body_feedback_mapped_axis
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{AxisMap, Cartesian};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection};
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

const NX: usize = 16;
const NY: usize = 8;
const X_LO: f64 = 1.0;
const X_HI: f64 = 10.0;
const Y_LO: f64 = -1.0;
const Y_HI: f64 = 1.0;
const SOFT: f64 = 0.05;

fn face(log: bool, i: usize) -> f64 {
    if log { X_LO * (X_HI / X_LO).powf(i as f64 / NX as f64) } else { X_LO + (X_HI - X_LO) * i as f64 / NX as f64 }
}

/// the booked x force on a unit-mass body at the origin and the force summed from the face
/// positions, for uniform unit-density gas at rest
fn booked_and_expected(log: bool) -> (f64, f64) {
    let dy = (Y_HI - Y_LO) / NY as f64;
    let maps = log.then(|| {
        [AxisMap::Log { start: X_LO, log_slope: (X_HI / X_LO).log10() / NX as f64 }, AxisMap::Uniform { start: Y_LO, dx: dy }]
    });
    let hole = Body::black_hole(0, Tensor::new([0.0, 0.0]), Tensor::zeros(), 1.0, 0.1, SOFT, 0.0, 0.3, 0.1);
    let s = Sim::build(Newtonian, IdealGas { gamma: 1.4 }, Cartesian)
        .cells([NX, NY])
        .origin([X_LO, Y_LO])
        .spacing([(X_HI - X_LO) / NX as f64, dy])
        .coord_maps(maps)
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim")
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0]), Pressure(1.0)))
        .build()
        .with_bodies(BodyCollection::new().add(hole));
    let kernels = AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(1.4, 0.4, &s.geom.allocated);
    kernels.c2p(&s);
    kernels.ghost_fill(&s);
    kernels.body_feedback(&s, 0.01);
    let booked = s.immersed.as_ref().expect("bodies").diagnostics.consolidate()[0].force_delta[0];
    let mut expected = 0.0;
    for i in 0..NX {
        let (a, b) = (face(log, i), face(log, i + 1));
        let x = 0.5 * (a + b);
        for j in 0..NY {
            let y = Y_LO + (j as f64 + 0.5) * dy;
            expected += (b - a) * dy * x / (x * x + y * y + SOFT * SOFT).powf(1.5);
        }
    }
    (booked, expected)
}

#[test]
fn the_feedback_force_sums_over_the_mapped_cells() {
    // the premise: the log cells differ from the uniform ones
    assert!((face(true, 1) - face(false, 1)).abs() > 0.1, "the log faces equal the uniform faces; the gate is vacuous");
    for (label, log) in [("uniform", false), ("log", true)] {
        let (booked, expected) = booked_and_expected(log);
        let rel = (booked.abs() - expected).abs() / expected;
        eprintln!("{label}: booked {booked:.12e} expected magnitude {expected:.12e} relative {rel:.3e}");
        assert!(rel < 1e-12, "{label}: the booked force {booked:.6e} departs from {expected:.6e} by {rel:.3e}");
    }
}
