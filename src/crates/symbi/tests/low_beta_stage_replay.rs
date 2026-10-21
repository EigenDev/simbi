// =============================================================================
// low_beta_stage_replay.rs
//
// one stage of the 2.5D newtonian MHD update replayed on a captured 13 x 13 block
// (fixtures/low_beta_stage_entry.rs) around a cell whose thermal energy is about
// 1e-4 of its magnetic energy. the stage is the first of an SSP RK2 step
// (a0 = 0, ac = 1, stage tag 1): PLM minmod reconstruction, HLLE fluxes, the
// contact edge EMF, the godunov update of the gas and of the cell B_z, the
// constrained-transport curl of the faces and the face-to-cell average. the
// center cell depends on cells at most four away, so the outflow ghosts beyond
// the block do not reach it. the replay reproduces the production state of the
// center cell bit for bit, and its internal energy E - |m|^2 / 2 rho - |B|^2 / 2,
// positive at entry, is negative after the stage: the total energy the fluxes
// deliver falls short of the kinetic gain plus the field energy the curl adds.
// the stage is the update without a body source; the captured run's sink pulled
// on no gas.
//
// the expected values characterize the scheme the block was captured under.
// a change to the update that leaves this cell admissible changes them, and
// this test then fails as the record of that change: the entry block stays,
// the expectation becomes the new scheme's stage. the negative internal energy
// is the observed failure, never a property to preserve.
//
// fixtures/low_beta_stage_reference.py recomputes the same stage from the
// intended discrete formulas (minmod PLM, HLLE with davis speeds, the contact
// EMF, the godunov update and the curl), independent of the production
// kernels, and agrees with the production slots to 4e-16.
//
// run: cargo test -p symbi --test low_beta_stage_replay
//      python3 crates/symbi/tests/fixtures/low_beta_stage_reference.py
// =============================================================================

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

include!("fixtures/low_beta_stage_entry.rs");

type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;
type Kern = NewtonianMhdSubstrateKernelSet<HostMemory, f64, 2>;

const GAMMA: f64 = 5.0 / 3.0;

/// the internal energy of the cell at `c`: total less kinetic and cell-field energy
fn eint(s: &Sim, c: [isize; 2]) -> f64 {
    let cons = &s.fields.cons;
    let m = s.fields.mhd.as_ref().expect("mhd");
    let rho = *cons.den.view().at(c);
    let kin: f64 = 0.5 * (0..3).map(|k| cons.mom[k].view().at(c).powi(2)).sum::<f64>() / rho;
    let mag: f64 = 0.5 * (0..3).map(|k| m.bcell[k].view().at(c).powi(2)).sum::<f64>();
    *cons.nrg.as_ref().expect("nrg").view().at(c) - kin - mag
}

fn loaded() -> (Sim, Kern, [isize; 2]) {
    let s = Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .spacing([1.0 / 256.0; 2])
        .origin([0.0; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(0.4)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim")
        .set_initial(|_| MhdPrim::new(Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)), Tensor::new([0.4, 0.1, 0.1])))
        .seed_faces_uniform([0.4, 0.1])
        .build();
    let lo = [s.geom.interior.spaces[0].lo, s.geom.interior.spaces[1].lo];
    let at = |i: usize, j: usize| [lo[0] + i as isize, lo[1] + j as isize];
    let cons = &s.fields.cons;
    let prim = &s.fields.prim;
    let m = s.fields.mhd.as_ref().expect("mhd");
    let cell: Vec<(&symbi_grid::Field<f64, 2, HostMemory>, &[[f64; N]; N])> = vec![
        (&cons.den, &ENTRY_DEN),
        (&cons.mom[0], &ENTRY_MOM0),
        (&cons.mom[1], &ENTRY_MOM1),
        (&cons.mom[2], &ENTRY_MOM2),
        (cons.nrg.as_ref().expect("nrg"), &ENTRY_NRG),
        (&prim.rho, &ENTRY_RHO),
        (&prim.vel[0], &ENTRY_VEL0),
        (&prim.vel[1], &ENTRY_VEL1),
        (&prim.vel[2], &ENTRY_VEL2),
        (prim.pre.as_ref().expect("pre"), &ENTRY_PRE),
        (&m.bcell[0], &ENTRY_BCELL0),
        (&m.bcell[1], &ENTRY_BCELL1),
        (&m.bcell[2], &ENTRY_BCELL2),
    ];
    for (field, values) in cell {
        for j in 0..N {
            for i in 0..N {
                field.view_mut().set(at(i, j), values[j][i]);
            }
        }
    }
    for j in 0..N {
        for i in 0..=N {
            m.bface[0].view_mut().set(at(i, j), ENTRY_BFACE0[j][i]);
        }
    }
    for j in 0..=N {
        for i in 0..N {
            m.bface[1].view_mut().set(at(i, j), ENTRY_BFACE1[j][i]);
        }
    }
    let kernels = Kern::new(GAMMA, 0.4, 1.0, &s.geom.allocated);
    kernels.ghost_fill(&s);
    (s, kernels, at(N / 2, N / 2))
}

#[test]
fn the_replayed_stage_reproduces_the_production_cell_and_its_negative_internal_energy() {
    let (mut s, kernels, center) = loaded();
    let entry = eint(&s, center);
    assert!(entry > 0.0, "the captured cell enters the stage with internal energy {entry:.3e}; the premise is gone");
    kernels.snapshot(&s);
    s.dt = DT;
    for d in 0..2 {
        kernels.flux(&s, d);
    }
    kernels.efield(&s);
    kernels.godunov_stage(&s, DT, 0.0, 1.0);
    kernels.stage_emf(&s, DT, 1);
    kernels.stage_curl(&s, DT, 1);
    let m = s.fields.mhd.as_ref().expect("mhd");
    let cons = &s.fields.cons;
    let replayed = [
        *cons.den.view().at(center),
        *cons.mom[0].view().at(center),
        *cons.mom[1].view().at(center),
        *cons.mom[2].view().at(center),
        *cons.nrg.as_ref().expect("nrg").view().at(center),
        *m.bcell[0].view().at(center),
        *m.bcell[1].view().at(center),
        *m.bcell[2].view().at(center),
    ];
    for k in 0..8 {
        assert_eq!(replayed[k].to_bits(), PRODUCTION_AFTER_CURL[k].to_bits(), "slot {k}: replay {} against production {}", replayed[k], PRODUCTION_AFTER_CURL[k]);
    }
    let after = eint(&s, center);
    assert!(after < 0.0, "the replayed stage leaves internal energy {after:.3e}");
}
