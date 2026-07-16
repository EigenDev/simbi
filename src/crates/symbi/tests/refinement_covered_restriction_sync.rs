// =============================================================================
// refinement_covered_restriction_sync.rs
//
// after every root step the hierarchy overwrites the covered coarse cells with the
// conservative restriction of their fine children (level_restrict_reflux), so BETWEEN
// root steps — where a checkpoint is written — the covered coarse conserved state must
// BE the restriction, bit-for-bit. this gate evolves a rotating density bump across a
// level seam and asserts that re-running the restriction into a scratch field changes
// nothing, for the adiabatic AND the isothermal kernel sets. a mismatch means some
// post-restriction pass (reflux, source, drain, eos) modified covered cells out of
// sync with the fine level.
// =============================================================================

use symbi::regimes::substrate::IsoSubstrateKernelSet;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::transfer::restrict_cell_field;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi::symbi_grid::Field;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::{IdealGas, Isothermal};
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::{Prim, PrimG};
use symbi_xpu::{CpuSpace, HostMemory};

const CFL: f64 = 0.4;
const N: usize = 32;
const STEPS: u64 = 8;

// a smooth density bump in solid-body rotation about the domain center: gradients and a
// genuinely evolving flow cross the level seam, so covered cells change every step.
fn bump(x: f64, y: f64) -> f64 {
    1.0 + 0.5 * (-((x - 0.5).powi(2) + (y - 0.5).powi(2)) / 0.02).exp()
}
fn rot(x: f64, y: f64) -> [f64; 2] {
    let om = 1.0;
    [-om * (y - 0.5), om * (x - 0.5)]
}

// compare the live covered coarse field against a fresh restriction of the fine field
// into a zeroed scratch; any covered cell where they differ is out of sync.
fn max_covered_mismatch(
    fine: &Field<f64, 2, HostMemory>,
    coarse: &Field<f64, 2, HostMemory>,
    alloc: &symbi_algebra::Domain<2>,
    coverage: &symbi_algebra::Domain<2>,
) -> f64 {
    let scratch = Field::<f64, 2, HostMemory>::zeros(alloc).unwrap();
    restrict_cell_field(fine, &scratch, coverage);
    let mut worst = 0.0_f64;
    for c in coverage.iter() {
        let live = *coarse.view().at(c);
        let want = *scratch.view().at(c);
        worst = worst.max((live - want).abs() / want.abs().max(1e-300));
    }
    worst
}

#[test]
fn covered_coarse_cells_are_the_restriction_adiabatic() {
    type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
    const GAMMA: f64 = 1.4;
    let dx = 1.0 / N as f64;
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 2])
        .origin([0.0; 2])
        .spacing([dx; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .unwrap()
        .set_initial(|[x, y]: [f64; 2]| Prim {
            rho: bump(x, y),
            vel: Tensor::new(rot(x, y)),
            pre: bump(x, y),
        })
        .build();
    let ck = Kset::new(GAMMA, CFL, &coarse.geom.allocated);
    let regions = [RefinementRegion { x_lo: [0.25; 2], x_hi: [0.75; 2] }];
    let mut hier = Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, |s| {
        Kset::new(GAMMA, CFL, &s.geom.allocated)
    })
    .unwrap();

    hier.evolve_steps(STEPS).unwrap();

    let (lo, hi) = hier.levels.split_at(1);
    let (c, f) = (&lo[0].state, &hi[0].state);
    let cov = lo[0].coverage.as_ref().expect("coarse level has coverage");
    let alloc = &c.geom.allocated;
    let m_den = max_covered_mismatch(&f.fields.cons.den, &c.fields.cons.den, alloc, cov);
    let m_m0 = max_covered_mismatch(&f.fields.cons.mom[0], &c.fields.cons.mom[0], alloc, cov);
    let m_nrg = max_covered_mismatch(
        f.fields.cons.nrg_field().unwrap(),
        c.fields.cons.nrg_field().unwrap(),
        alloc,
        cov,
    );
    assert!(m_den < 1e-13, "adiabatic covered den out of sync with restriction: {m_den:e}");
    assert!(m_m0 < 1e-13, "adiabatic covered mom out of sync with restriction: {m_m0:e}");
    assert!(m_nrg < 1e-13, "adiabatic covered nrg out of sync with restriction: {m_nrg:e}");
}

#[test]
fn covered_coarse_cells_are_the_restriction_iso() {
    type ISim = SimState<IsoNewtonian, 2, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
    type IKset = IsoSubstrateKernelSet<HostMemory, f64, 2>;
    let cs = 1.0;
    let dx = 1.0 / N as f64;
    let coarse = ISim::build(IsoNewtonian, Isothermal { cs }, Cartesian)
        .cells([N; 2])
        .origin([0.0; 2])
        .spacing([dx; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .unwrap()
        .set_initial(|[x, y]: [f64; 2]| PrimG::<f64, 2, IsoModel> {
            rho: bump(x, y),
            vel: Tensor::new(rot(x, y)),
            pre: Default::default(),
        })
        .build();
    let ck = IKset::new(cs, CFL, &coarse.geom.allocated);
    let regions = [RefinementRegion { x_lo: [0.25; 2], x_hi: [0.75; 2] }];
    let mut hier = Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, |s| {
        IKset::new(cs, CFL, &s.geom.allocated)
    })
    .unwrap();

    hier.evolve_steps(STEPS).unwrap();

    let (lo, hi) = hier.levels.split_at(1);
    let (c, f) = (&lo[0].state, &hi[0].state);
    let cov = lo[0].coverage.as_ref().expect("coarse level has coverage");
    let alloc = &c.geom.allocated;
    let m_den = max_covered_mismatch(&f.fields.cons.den, &c.fields.cons.den, alloc, cov);
    let m_m0 = max_covered_mismatch(&f.fields.cons.mom[0], &c.fields.cons.mom[0], alloc, cov);
    let m_m1 = max_covered_mismatch(&f.fields.cons.mom[1], &c.fields.cons.mom[1], alloc, cov);
    assert!(m_den < 1e-13, "iso covered den out of sync with restriction: {m_den:e}");
    assert!(m_m0 < 1e-13, "iso covered mom_x out of sync with restriction: {m_m0:e}");
    assert!(m_m1 < 1e-13, "iso covered mom_y out of sync with restriction: {m_m1:e}");
}
