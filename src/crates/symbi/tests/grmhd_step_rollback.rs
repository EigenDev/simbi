// =============================================================================
// grmhd_step_rollback.rs
//
// the GRMHD rejected-step transaction law. the physical-constraint-preserving
// fofc redo replays a whole explicit step at half the timestep when its
// source-free low-order anchor is itself inadmissible, so the step must be a
// TRANSACTION: `restore_step` has to put back everything an accepted step
// touched, or the replay starts from a hybrid of the rejected attempt and the
// step-entry state.
//
// gates:
// - a full accepted step followed by `restore_step` reproduces the step-entry
//   conserved gas state, cell-centered B, staggered face B, AND the primitives
//   BIT-FOR-BIT (the retried step re-enters at wave_speeds/flux, which
//   reconstruct from prim, so a conserved-only rollback would replay from the
//   rejected attempt's primitives);
// - the rollback storage exists exactly where a step can be rejected: a curved
//   magnetized background allocates it, flat MHD does not.
// =============================================================================

use std::f64::consts::PI;

use symbi::regimes::substrate_kernels::Solver;
use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet3D;
use symbi::sim::evolve::step_once;
use symbi::sim::state::*;
use symbi::sim::substrate_seam::KernelSet;
use symbi_algebra::Tensor;
use symbi_geometry::{Cartesian, KerrKSCartesian};
use symbi_grid::Field;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const N: usize = 6;
const GAMMA: f64 = 4.0 / 3.0;
const CFL: f64 = 0.3;
const MASS: f64 = 0.2;
const SPIN: f64 = 0.5;
const B0: f64 = 0.1;
const X_LO: f64 = 1.2;
const DT: f64 = 5.0e-3;

type KerrSim = SimState<Rmhd, 3, KerrKSCartesian<f64>, IdealGas<f64>, CpuSpace, HostMemory>;
type FlatSim = SimState<Rmhd, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

fn swirl_prim(x: f64, y: f64, z: f64) -> MhdPrim<f64, 3> {
    let s = 2.0 * PI;
    MhdPrim {
        hydro: Prim {
            rho: 1.0,
            vel: Tensor::new([
                0.1 * (s * y).sin(),
                0.1 * (s * z).sin(),
                0.1 * (s * x).sin(),
            ]),
            pre: 0.5,
        },
        mag: Tensor::new([B0, 0.0, 0.0]),
    }
}

/// every value of a field over its OWN domain (cell fields on the allocated
/// domain, each face field on its staggered domain).
fn dump(field: &Field<f64, 3, HostMemory>) -> Vec<f64> {
    let view = field.view();
    field.domain().iter().map(|c| *view.at(c)).collect()
}

/// the full observable state of a magnetized run: the conserved gas vector,
/// both magnetic representations, and the primitives derived from them.
fn dump_state(sim: &KerrSim) -> Vec<Vec<f64>> {
    let mhd = sim.fields.mhd.as_ref().expect("magnetized run");
    let mut out = vec![dump(&sim.fields.cons.den)];
    for cc in 0..3 {
        out.push(dump(&sim.fields.cons.mom[cc]));
    }
    out.push(dump(sim.fields.cons.nrg_field().expect("energy regime")));
    out.push(dump(&sim.fields.prim.rho));
    for cc in 0..3 {
        out.push(dump(&sim.fields.prim.vel[cc]));
    }
    out.push(dump(sim.fields.prim.pre_field().expect("energy regime")));
    for cc in 0..3 {
        out.push(dump(&mhd.bcell[cc]));
    }
    for dd in 0..3 {
        out.push(dump(&mhd.bface[dd]));
    }
    out
}

fn kerr_sim() -> KerrSim {
    let dx = 1.0 / N as f64;
    let metric = KerrKSCartesian {
        mass: MASS,
        spin: SPIN,
    };
    // the box sits outside the horizon (x_lo = 1.2 puts every cell at
    // r >= sqrt(3) * 1.2, well beyond r_+ < 2M = 0.4). seed the DENSITIZED face
    // flux uniform so the staggered divergence starts at machine zero.
    KerrSim::build(Rmhd, IdealGas { gamma: GAMMA }, metric)
        .cells([N; 3])
        .origin([X_LO; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .expect("kerr sim")
        .set_initial(|[x, y, z]| swirl_prim(x, y, z))
        .seed_faces(|axis, [x, y, z]| {
            if axis == 0 {
                B0 / symbi_geometry::Metric::<f64, 3>::sqrt_det_gamma(&metric, Tensor::new([x, y, z]))
            } else {
                0.0
            }
        })
        .build()
}

fn kernels(sim: &KerrSim) -> RmhdSubstrateKernelSet3D<HostMemory, f64> {
    RmhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, CFL, 1.0, &sim.geom.allocated)
        .with_solver(Solver::Hlld)
        .expect("hlld")
        .ct_method(CtMethod::Uct)
}

#[test]
fn rejecting_a_step_restores_the_full_magnetized_entry_state() {
    let mut sim = kerr_sim();
    let kern = kernels(&sim);

    // a rejection is raised mid-step, so the entry state must be a genuine step
    // boundary: seeded initial data has never been through the ghost fill that
    // closes every stage, and comparing against it would measure the fill
    // rather than the rollback.
    step_once(&mut sim, &kern, DT);

    kern.snapshot_retry(&sim);
    let entry = dump_state(&sim);

    step_once(&mut sim, &kern, DT);

    // WITHOUT this the rollback comparison is vacuous: a step that moved
    // nothing would "restore" trivially.
    let moved = dump_state(&sim);
    assert!(
        entry
            .iter()
            .zip(moved.iter())
            .any(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| x != y)),
        "the accepted step left the state untouched; the rollback gate is vacuous"
    );

    kern.restore_step(&sim);
    let rolled = dump_state(&sim);

    let labels = [
        "cons.den", "cons.mom_0", "cons.mom_1", "cons.mom_2", "cons.nrg", "prim.rho", "prim.vel_0",
        "prim.vel_1", "prim.vel_2", "prim.pre", "bcell_0", "bcell_1", "bcell_2", "bface_0",
        "bface_1", "bface_2",
    ];
    assert_eq!(labels.len(), entry.len());
    for (kk, label) in labels.iter().enumerate() {
        assert_eq!(
            entry[kk].len(),
            rolled[kk].len(),
            "{label}: rollback changed the field extent"
        );
        for (ii, (before, after)) in entry[kk].iter().zip(rolled[kk].iter()).enumerate() {
            assert_eq!(
                before, after,
                "{label}[{ii}]: a rejected step must restore the entry value exactly"
            );
        }
    }
}

#[test]
fn rollback_storage_exists_exactly_where_a_step_can_be_rejected() {
    let curved = kerr_sim();
    assert!(
        curved
            .fields
            .mhd
            .as_ref()
            .expect("magnetized run")
            .step_snapshot
            .is_some(),
        "a curved magnetized background can reject a step and needs the rollback snapshot"
    );

    let dx = 1.0 / N as f64;
    let flat: FlatSim = FlatSim::build(Rmhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 3])
        .origin([0.0; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .expect("flat sim")
        .set_initial(|[x, y, z]| swirl_prim(x, y, z))
        .seed_faces(|axis, _| if axis == 0 { B0 } else { 0.0 })
        .build();
    assert!(
        flat.fields
            .mhd
            .as_ref()
            .expect("magnetized run")
            .step_snapshot
            .is_none(),
        "flat MHD accepts every step; the rollback snapshot would be a dead allocation the \
         size of the whole conserved + magnetic state"
    );
}
