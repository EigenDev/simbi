// =============================================================================
// refinement_dye.rs
//
// the passive scalar across a coarse-fine interface.
//
// a dye is carried by the mass flux, so refinement exposes it in three separate places: the fine
// level's ghost band has to receive the concentration by prolongation, the covered coarse cells
// have to receive it back by restriction, and the conserved dye at the interface has to be
// refluxed from the interface flux MISMATCH the way mass is. drop any one of those and the dye
// still advects, still stays bounded, and still looks like a plausible tracer field — it is only
// the total that goes wrong, quietly, in proportion to how much dye crossed the interface.
//
// so the gate is the composite total: dye summed over leaf cells (coarse outside the refined
// patch, fine inside it), which on a source-free periodic domain is exactly conserved. a bump
// that stopped short of the patch would conserve that total trivially, so the setup also asserts
// the dye crossed.
//
// run: cargo test -p symbi --test refinement_dye
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_amr::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<Newtonian, 1, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Hier = Hierarchy<
    Newtonian,
    1,
    1,
    Cartesian,
    IdealGas<f64>,
    CpuSpace,
    HostMemory,
    AdiabaticSubstrateKernelSet<HostMemory, f64, 1>,
>;

const N: usize = 128;
const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const V: f64 = 0.5;
// the refined patch, in domain coordinates.
const PATCH_LO: f64 = 0.375;
const PATCH_HI: f64 = 0.625;
// the dye starts as a step occupying [0, FRONT) and advects right at V.
const FRONT: f64 = 0.25;

fn kset(sim: &Sim) -> AdiabaticSubstrateKernelSet<HostMemory, f64, 1> {
    AdiabaticSubstrateKernelSet::new(GAMMA, CFL, &sim.geom.allocated)
}

fn coarse_sim() -> Sim {
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N])
        .spacing([1.0 / N as f64])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(CFL)
        .allocate()
        .expect("sim")
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([V]),
            pre: 1.0,
        })
        .build()
        .with_passive_scalar()
        .expect("chi alloc")
}

/// seed the dye step over the WHOLE allocated grid of one level, so the first stage's upwind
/// stencil reads consistent ghosts before any fill runs.
fn seed_chi(sim: &Sim, dx: f64, x_origin: f64) {
    let cons_chi = sim.fields.cons.chi_field().expect("cons chi");
    let prim_chi = sim.fields.prim.chi_field().expect("prim chi");
    for c in sim.geom.allocated.clone().iter() {
        let x = x_origin + (c[0] as f64 + 0.5) * dx;
        let chi = if x < FRONT { 1.0 } else { 0.0 };
        let rho = *sim.fields.cons.den.view().at(c);
        cons_chi.view_mut().set(c, rho * chi);
        prim_chi.view_mut().set(c, chi);
    }
}

/// conserved dye summed over LEAF cells: coarse cells outside the refined coverage plus every
/// fine cell, each weighted by its own cell width. this is the quantity a missing reflux or a
/// missing restriction breaks.
fn composite_dye(hier: &Hier) -> f64 {
    let coarse = &hier.levels[0].state;
    let dx_c = coarse.geom.dx[0];
    let coverage = hier.levels[0]
        .coverage
        .as_ref()
        .expect("a refined hierarchy has coverage on the root");
    let mut total = 0.0;
    let cchi = coarse.fields.cons.chi_field().expect("coarse chi");
    for c in coarse.geom.interior.iter() {
        if !coverage.contains(c) {
            total += *cchi.view().at(c) * dx_c;
        }
    }
    let fine = &hier.levels[1].state;
    let dx_f = fine.geom.dx[0];
    let fchi = fine.fields.cons.chi_field().expect("fine chi");
    for c in fine.geom.interior.iter() {
        total += *fchi.view().at(c) * dx_f;
    }
    total
}

/// the dye mass held by the fine level alone — the probe for whether the bump reached the patch.
fn fine_dye(hier: &Hier) -> f64 {
    let fine = &hier.levels[1].state;
    let dx_f = fine.geom.dx[0];
    let fchi = fine.fields.cons.chi_field().expect("fine chi");
    fine.geom
        .interior
        .iter()
        .map(|c| *fchi.view().at(c) * dx_f)
        .sum()
}

fn refined() -> Hier {
    let coarse = coarse_sim();
    let ck = kset(&coarse);
    let regions = [RefinementRegion {
        x_lo: [PATCH_LO],
        x_hi: [PATCH_HI],
    }];
    let hier =
        Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, kset).expect("refine");
    seed_chi(&hier.levels[0].state, 1.0 / N as f64, 0.0);
    // a fine level is built empty; give it the same uniform gas the root carries.
    hier.levels[1].state.seed_cells(|_| Prim {
        rho: 1.0,
        vel: Tensor::new([V]),
        pre: 1.0,
    });
    let fine = &hier.levels[1].state;
    let dx_f = fine.geom.dx[0];
    let x0 = fine.geom.x_lo[0];
    seed_chi(fine, dx_f, x0);
    hier
}

#[test]
fn dye_is_conserved_across_a_refinement_interface() {
    let mut hier = refined();
    let before = composite_dye(&hier);
    let fine_before = fine_dye(&hier);
    assert!(
        before > 0.0,
        "no dye seeded; the gate would pass on an empty field"
    );

    // long enough for the front to enter the patch (t = 0.25) and leave it again (t = 0.75), so
    // the dye crosses both interfaces and clears the patch.
    let t_final = 0.9;
    hier.evolve(t_final).expect("refined dye advection");

    let after = composite_dye(&hier);
    let fine_after = fine_dye(&hier);

    // the premise: the dye traversed the refined patch. a bump that stopped short would conserve
    // the total trivially and leave every transfer path unexercised.
    assert!(
        fine_after > 1e-6 || fine_before > 1e-6,
        "no dye ever occupied the refined patch; the interface was never exercised"
    );
    assert!(
        (fine_after - fine_before).abs() > 1e-6,
        "the fine-level dye never changed (before {fine_before}, after {fine_after}); \
         the bump did not cross the interface"
    );

    // the conserved dye on a periodic domain with no sources: exactly conserved over leaf cells.
    let drift = (after - before).abs() / before;
    assert!(
        drift < 1e-10,
        "composite dye drifted by {drift:e} across the refinement interface \
         (before {before}, after {after})"
    );
}
