// =============================================================================
// typed_source_mapped_axis.rs
//
// the additive point-mass gravity source evaluates its acceleration at each
// cell's centroid, `S_mom = -rho GM (x - xm) / |x - xm|^3`. on a log-spaced
// cartesian axis the centroid comes from the axis map, `face(i) = x_lo 10^(i s)`,
// while the grid stores the nominal linear width as its spacing. one small euler
// step from uniform gas at rest isolates the source (the flux divergence
// vanishes), and the momentum gained per unit time is compared with the
// acceleration at the face midpoint written out here. a uniform axis is the
// control.
//
// run: cargo test -p symbi --test typed_source_mapped_axis
// =============================================================================

use symbi::regimes::substrate_kernels::FusedSourceBinding;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::step_once;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{AxisMap, Cartesian};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

const N: usize = 16;
const X_LO: f64 = 1.0;
const X_HI: f64 = 10.0;

fn face(log: bool, i: usize) -> f64 {
    if log { X_LO * (X_HI / X_LO).powf(i as f64 / N as f64) } else { X_LO + (X_HI - X_LO) * i as f64 / N as f64 }
}

/// the largest relative error of the momentum gain against the point-mass acceleration at the
/// face midpoint, over the cells away from the outflow walls
fn worst_gain_error(log: bool) -> f64 {
    let maps = log.then(|| [AxisMap::Log { start: X_LO, log_slope: (X_HI / X_LO).log10() / N as f64 }]);
    let mut s = Sim::build(Newtonian, IdealGas { gamma: 1.4 }, Cartesian)
        .cells([N])
        .origin([X_LO])
        .spacing([(X_HI - X_LO) / N as f64])
        .coord_maps(maps)
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim")
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0]), Pressure(1.0)))
        .build();
    s.timestepping = Timestepping::Euler;
    let gravity = FusedSourceBinding::new("point_mass_grav", &[("gm", 1.0), ("eps", 0.0), ("xm_0", 0.0)]);
    let kernels = AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(1.4, 0.4, &s.geom.allocated).with_additive_source(gravity);
    kernels.c2p(&s);
    kernels.ghost_fill(&s);
    let dt = 1e-7;
    step_once(&mut s, &kernels, dt);
    let lo = s.geom.interior.spaces[0].lo;
    let mut worst = 0.0f64;
    for c in s.geom.interior.iter() {
        let i = (c[0] - lo) as usize;
        if i == 0 || i == N - 1 {
            continue;
        }
        let x = 0.5 * (face(log, i) + face(log, i + 1));
        let gain = *s.fields.cons.mom[0].view().at(c) / dt;
        let want = -1.0 / (x * x);
        worst = worst.max(((gain - want) / want).abs());
    }
    worst
}

#[test]
fn a_point_mass_source_acts_at_the_mapped_cell_centroid() {
    // the premise: on the log axis the face midpoints differ from the nominal uniform ones
    assert!((face(true, 1) - face(false, 1)).abs() > 0.1, "the log faces equal the uniform faces; the gate is vacuous");
    for (label, log) in [("uniform", false), ("log", true)] {
        let worst = worst_gain_error(log);
        assert!(worst < 1e-12, "{label}: the momentum gain departs from the point-mass acceleration by {worst:.3e}");
    }
}
