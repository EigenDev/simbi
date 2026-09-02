//! Joint space-time convergence of the complete composite algorithm.
//!
//! A diagonal entropy wave crosses all four faces of a static refined box.  Resolution is doubled
//! at fixed CFL, so both `dx` and every level's subcycled `dt` halve together.  The gate measures
//! the composite leaf norm, a fixed-physical-width seam norm, Fourier phase, and conservation.

use std::f64::consts::TAU;
use symbi_hydro::quantity::{Density, Pressure};

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::regime::Regime;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.35;
const T_FINAL: f64 = 0.35;
const VELOCITY: [f64; 2] = [0.7, 0.4];

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
type Hier = Hierarchy<Newtonian, 2, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset>;

fn primitive(x: [f64; 2], time: f64) -> Prim<f64, 2> {
    let phase = TAU * (x[0] + x[1] - (VELOCITY[0] + VELOCITY[1]) * time);
    Prim::adiabatic(
        Density(1.0 + 0.1 * phase.sin()),
        Tensor::new(VELOCITY),
        Pressure(1.0),
    )
}

fn seed(sim: &Sim) {
    let nrg = sim.fields.cons.nrg_field().unwrap();
    for cell in sim.geom.interior.iter() {
        let cons = Regime::to_conserved(
            &sim.physics.regime,
            &sim.physics.eos,
            &primitive(sim.geom.centroid(cell), 0.0),
        );
        sim.fields.cons.den.view_mut().set(cell, cons.den());
        for dd in 0..2 {
            sim.fields.cons.mom[dd].view_mut().set(cell, cons.mom()[dd]);
        }
        nrg.view_mut().set(cell, cons.nrg());
    }
}

fn run(n: usize) -> Hier {
    let dx = 1.0 / n as f64;
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([n; 2])
        .spacing([dx; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(CFL)
        .allocate()
        .unwrap()
        .set_initial(|x| primitive(x, 0.0))
        .build();
    let kernels = Kset::new(GAMMA, CFL, &coarse.geom.allocated);
    let mut hierarchy = Hierarchy::with_refinement(
        coarse,
        kernels,
        &[RefinementRegion {
            x_lo: [0.25; 2],
            x_hi: [0.75; 2],
        }],
        ProlongOrder::Ppm,
        |state| Kset::new(GAMMA, CFL, &state.geom.allocated),
    )
    .unwrap();
    seed(&hierarchy.levels[1].state);
    hierarchy.evolve(T_FINAL).unwrap();
    hierarchy
}

#[derive(Debug)]
struct Error {
    global: f64,
    seam: f64,
    phase: f64,
    mass_drift: f64,
}

fn errors(hierarchy: &Hier) -> Error {
    let mut global = 0.0;
    let mut global_volume = 0.0;
    let mut seam = 0.0;
    let mut seam_volume = 0.0;
    let mut mass = 0.0;
    let mut mode_re = 0.0;
    let mut mode_im = 0.0;
    for level in &hierarchy.levels {
        let volume: f64 = level.state.geom.dx.iter().product();
        for cell in level.state.geom.interior.iter() {
            if !level.state.composite_ownership.owns_leaf(cell) {
                continue;
            }
            let x = level.state.geom.centroid(cell);
            let rho = *level.state.fields.cons.den.view().at(cell);
            let exact = primitive(x, T_FINAL).rho();
            let point_error = (rho - exact).abs();
            global += point_error * volume;
            global_volume += volume;
            mass += rho * volume;
            let theta = TAU * (x[0] + x[1]);
            mode_re += (rho - 1.0) * theta.cos() * volume;
            mode_im -= (rho - 1.0) * theta.sin() * volume;
            let distance = (0..2)
                .map(|dd| (x[dd] - 0.25).abs().min((x[dd] - 0.75).abs()))
                .fold(f64::INFINITY, f64::min);
            if distance < 0.08 {
                seam += point_error * volume;
                seam_volume += volume;
            }
        }
    }
    let exact_phase = -(VELOCITY[0] + VELOCITY[1]) * TAU * T_FINAL - TAU / 4.0;
    let numerical_phase = mode_im.atan2(mode_re);
    let phase = (numerical_phase - exact_phase).sin().abs();
    Error {
        global: global / global_volume,
        seam: seam / seam_volume,
        phase,
        mass_drift: (mass - 1.0).abs(),
    }
}

fn order(coarse: f64, fine: f64) -> f64 {
    (coarse / fine).log2()
}

#[test]
fn composite_wave_converges_jointly_in_space_and_time() {
    let e24 = errors(&run(24));
    let e48 = errors(&run(48));
    let e96 = errors(&run(96));
    let global_order = order(e48.global, e96.global);
    let seam_order = order(e48.seam, e96.seam);
    let phase_order = order(e48.phase, e96.phase);
    eprintln!(
        "[amr-joint] N=24 {e24:?}\n[amr-joint] N=48 {e48:?}\n\
         [amr-joint] N=96 {e96:?}\n[amr-joint] orders: global={global_order:.3} \
         seam={seam_order:.3} phase={phase_order:.3}"
    );
    assert!(
        global_order > 1.7,
        "global joint order fell to {global_order:.3}"
    );
    assert!(
        seam_order > 1.4,
        "coarse-fine seam order fell to {seam_order:.3}"
    );
    assert!(phase_order > 1.6, "phase order fell to {phase_order:.3}");
    for error in [&e24, &e48, &e96] {
        assert!(
            error.mass_drift < 2.0e-12,
            "composite mass drifted by {:.3e}",
            error.mass_drift
        );
    }
}
