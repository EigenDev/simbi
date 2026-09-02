// =============================================================================
// refinement_uniform_preservation.rs
//
// well-balancedness of the coarse-fine coupling: a uniform gas at rest with a
// static refined level and no sources must stay uniform. every flux across
// every face (including the coarse-fine seam ghosts) is the flux of identical
// states, so any drift — especially a ring at the level boundary — is a
// defect in prolongation, restriction, or the subcycle sequencing, not
// physics. gated for both the adiabatic and the isothermal kernel sets: the
// coupling must be regime-generic.
// =============================================================================

use symbi::regimes::substrate::IsoSubstrateKernelSet;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::{IdealGas, Isothermal};
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::{Prim, PrimG};
use symbi_xpu::{CpuSpace, HostMemory};

const CFL: f64 = 0.4;
const N: usize = 16;
const RHO0: f64 = 2.0;
const STEPS: u64 = 6;

#[test]
fn uniform_gas_stays_uniform_across_the_level_seam_adiabatic() {
    type Sim = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 3>;
    const GAMMA: f64 = 5.0 / 3.0;
    let dx = 1.0 / N as f64;
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 3])
        .origin([0.0; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(CFL)
        .allocate()
        .unwrap()
        .set_initial(|_x: [f64; 3]| {
            Prim::adiabatic(Density(RHO0), Tensor::new([0.0; 3]), Pressure(1.0))
        })
        .build();
    let ck = Kset::new(GAMMA, CFL, &coarse.geom.allocated);
    let regions = [RefinementRegion {
        x_lo: [0.25; 3],
        x_hi: [0.75; 3],
    }];
    let mut hier = Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, |s| {
        Kset::new(GAMMA, CFL, &s.geom.allocated)
    })
    .unwrap();

    // seed the fine level's gas (harness convention: with_refinement's IC
    // prolongation is not relied on here). uniform pre = 1 at rest.
    {
        let fine = &hier.levels[1].state;
        let nrg = fine.fields.cons.nrg_field().unwrap();
        for c in fine.geom.interior.iter() {
            fine.fields.cons.den.view_mut().set(c, RHO0);
            nrg.view_mut().set(c, 1.0 / (GAMMA - 1.0));
            for dd in 0..3 {
                fine.fields.cons.mom[dd].view_mut().set(c, 0.0);
            }
        }
    }

    hier.evolve_steps(STEPS).unwrap();

    for (lvl, level) in hier.levels.iter().enumerate() {
        let sim = &level.state;
        for c in sim.geom.interior.iter() {
            let den = *sim.fields.cons.den.view().at(c);
            let drift = (den - RHO0).abs();
            assert!(
                drift < 1e-12,
                "adiabatic level {lvl}: den drifted by {drift:e} at {c:?} (den = {den})"
            );
            for dd in 0..3 {
                let mom = *sim.fields.cons.mom[dd].view().at(c);
                assert!(
                    mom.abs() < 1e-12,
                    "adiabatic level {lvl}: mom[{dd}] drifted to {mom:e} at {c:?}"
                );
            }
        }
    }
}

#[test]
fn uniform_gas_stays_uniform_across_the_level_seam_iso() {
    type ISim = SimState<IsoNewtonian, 3, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
    type IKset = IsoSubstrateKernelSet<HostMemory, f64, 3>;
    let cs = 1.0;
    let dx = 1.0 / N as f64;
    let coarse = ISim::build(IsoNewtonian, Isothermal { cs }, Cartesian)
        .cells([N; 3])
        .origin([0.0; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(CFL)
        .allocate()
        .unwrap()
        .set_initial(|_x: [f64; 3]| {
            PrimG::<f64, 3, IsoModel>::isothermal(Density(RHO0), Tensor::new([0.0; 3]))
        })
        .build();
    let ck = IKset::new(cs, CFL, &coarse.geom.allocated);
    let regions = [RefinementRegion {
        x_lo: [0.25; 3],
        x_hi: [0.75; 3],
    }];
    let mut hier = Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, |s| {
        IKset::new(cs, CFL, &s.geom.allocated)
    })
    .unwrap();

    // seed the fine level's gas (harness convention: with_refinement's IC
    // prolongation is not relied on here); iso has no energy slot.
    {
        let fine = &hier.levels[1].state;
        for c in fine.geom.interior.iter() {
            fine.fields.cons.den.view_mut().set(c, RHO0);
            for dd in 0..3 {
                fine.fields.cons.mom[dd].view_mut().set(c, 0.0);
            }
        }
    }

    hier.evolve_steps(STEPS).unwrap();

    for (lvl, level) in hier.levels.iter().enumerate() {
        let sim = &level.state;
        for c in sim.geom.interior.iter() {
            let den = *sim.fields.cons.den.view().at(c);
            let drift = (den - RHO0).abs();
            assert!(
                drift < 1e-12,
                "iso level {lvl}: den drifted by {drift:e} at {c:?} (den = {den})"
            );
            for dd in 0..3 {
                let mom = *sim.fields.cons.mom[dd].view().at(c);
                assert!(
                    mom.abs() < 1e-12,
                    "iso level {lvl}: mom[{dd}] drifted to {mom:e} at {c:?}"
                );
            }
        }
    }
}
