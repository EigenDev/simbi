// =============================================================================
// refine_temporal_convergence.rs
//
// temporal-order gate for the coarse-fine ghost coupling: a smooth entropy
// wave advecting diagonally through a static nested box (2d, periodic, rk2),
// dt self-convergence on a FIXED composite grid — the same hierarchy run at
// cfl, cfl/2, and a cfl/16 reference, compared cell-by-cell at the same
// physical time, so the (fixed) spatial error cancels and only the temporal
// error remains. the boundary-sensitive norm (cells within two cells of the
// coarse-fine interface, both sides) isolates the ghost coupling:
// substep-start-frozen ghosts make the interface flux first-order in time
// (measured 1.14); stage-correct shu-osher time interpolation
// (hierarchy.rs::stage_time_fractions) restores near-second order (measured
// 1.62 at the production cfl pair; the slope saturates near 1.3 at deep dt,
// where the mixed space-time prolongation term dominates).
//
// usage:
//  cargo test -p symbi --release --test refine_temporal_convergence
// =============================================================================

use std::f64::consts::TAU;

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
const N: usize = 32;
const T_FINAL: f64 = 0.2;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
type Hier = Hierarchy<Newtonian, 2, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset>;

fn fill(sim: &Sim) {
    let cnrg = sim.fields.cons.nrg_field().unwrap();
    for c in sim.geom.interior.iter() {
        let x = sim.geom.centroid(c);
        let prim = Prim {
            rho: 1.0 + 0.2 * (TAU * (x[0] + x[1])).sin(),
            vel: Tensor::new([1.0, 0.5]),
            pre: 1.0,
        };
        let cons = Regime::to_conserved(&sim.physics.regime, &sim.physics.eos, &prim);
        sim.fields.cons.den.view_mut().set(c, cons.den);
        for dd in 0..2 {
            sim.fields.cons.mom[dd].view_mut().set(c, cons.mom[dd]);
        }
        cnrg.view_mut().set(c, cons.nrg);
    }
}

fn evolved(cfl: f64) -> Hier {
    let dx = 1.0 / N as f64;
    // the same fill() closure the fine level uses, folded in as the prim ic.
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 2])
        .spacing([dx; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(cfl)
        .allocate()
        .unwrap()
        .set_initial(|x: [f64; 2]| Prim {
            rho: 1.0 + 0.2 * (TAU * (x[0] + x[1])).sin(),
            vel: Tensor::new([1.0, 0.5]),
            pre: 1.0,
        })
        .build();
    let ck = Kset::new(GAMMA, cfl, &coarse.geom.allocated);
    let regions = [RefinementRegion {
        x_lo: [0.25; 2],
        x_hi: [0.75; 2],
    }];
    let mut hier = Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, |s| {
        Kset::new(GAMMA, cfl, &s.geom.allocated)
    })
    .unwrap();
    fill(&hier.levels[1].state);
    hier.evolve(T_FINAL).unwrap();
    hier
}

/// is this cell within `width` cells of the coarse-fine interface of its
/// level (fine skin inside, coarse reflux shell outside)?
fn near_interface(hier: &Hier, ll: usize, c: [isize; 2], width: isize) -> bool {
    if ll == 0 {
        let cov = hier.levels[0].coverage.as_ref().unwrap();
        if cov.contains(c) {
            return false;
        }
        (0..2).all(|ax| c[ax] >= cov.spaces[ax].lo - width && c[ax] < cov.spaces[ax].hi + width)
            && (0..2).any(|ax| c[ax] < cov.spaces[ax].lo || c[ax] >= cov.spaces[ax].hi)
    } else {
        let int = &hier.levels[1].state.geom.interior;
        (0..2).any(|ax| c[ax] < int.spaces[ax].lo + width || c[ax] >= int.spaces[ax].hi - width)
    }
}

/// l1 density error between two same-grid hierarchies over the interface
/// skin (composite: covered coarse cells excluded on l0).
fn skin_error(a: &Hier, b: &Hier) -> f64 {
    let mut err = 0.0;
    let mut count = 0u64;
    for ll in 0..2 {
        for c in a.levels[ll].state.geom.interior.iter() {
            if !near_interface(a, ll, c, 2) {
                continue;
            }
            let da = *a.levels[ll].state.fields.cons.den.view().at(c);
            let db = *b.levels[ll].state.fields.cons.den.view().at(c);
            err += (da - db).abs();
            count += 1;
        }
    }
    err / count as f64
}

#[test]
fn cf_ghost_coupling_is_second_order_in_time() {
    let reference = evolved(0.025);
    let coarse_dt = evolved(0.4);
    let half_dt = evolved(0.2);

    let e1 = skin_error(&coarse_dt, &reference);
    let e2 = skin_error(&half_dt, &reference);
    let order = (e1 / e2).log2();
    eprintln!(
        "[amr-temporal] skin l1: e(cfl 0.4) = {e1:.3e}, e(cfl 0.2) = {e2:.3e}, order = {order:.2}"
    );
    // measured baselines on this problem: substep-start-frozen ghosts give
    // 1.14 (and 4-14x larger absolute errors); stage-correct interpolation
    // gives 1.62 at this cfl pair. the deep-dt slope saturates near 1.3 — a
    // mixed spatio-temporal term in the prolongation (ppm in space x linear
    // in time), first-order in dt only at FIXED dx; under cfl-coupled
    // refinement (dt ~ dx) it contributes at third order. the 1.4 bound
    // separates the broken and correct couplings with margin on both sides.
    assert!(
        order > 1.4,
        "coarse-fine temporal order {order:.2} (e1 {e1:.3e}, e2 {e2:.3e}) — \
         the ghost coupling has degraded toward first order (frozen-ghost \
         baseline is 1.14, stage-correct is 1.62)"
    );
}
