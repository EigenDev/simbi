// =============================================================================
// refine_divb_under_evolve.rs
//
// div(B)-preservation gate: a 2-level static nesting of the newtonian-mhd
// substrate (orszag-tang, periodic, the fine level covering the vortex core)
// keeps the discrete staggered div(B) at machine zero on both levels under
// evolve — the area-weighted bface restriction + the edge-EMF reflux preserve
// the constraint across the coarse-fine boundary — while the composite gas
// mass and momentum stay conserved (the hydro flux registers, magnetic stress
// included). total energy is not asserted to machine precision: the CT
// magnetic-energy correction (bcell_from_bface) is deliberately
// non-conservative, on a single level exactly as here.
//
// mirrors nmhd_divb_under_evolve.rs (the single-level gate): same analytic
// div-free staggered OT initial condition, same checker, levels in absolute
// indices so the fine IC closure is the same formula.
// =============================================================================

use std::f64::consts::PI;
use std::sync::atomic::Ordering;

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet3D;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::regime::Regime;
use symbi_hydro::state::{Cons, Prim};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<NewtonianMhd, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = NewtonianMhdSubstrateKernelSet3D<HostMemory, f64>;
type Hier = Hierarchy<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset>;

const N: usize = 16;
const NZ: usize = 2;
const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.3;
const V0: f64 = 0.5;
const B0: f64 = 1.0;
const DIVB_TOL: f64 = 1e-12;

/// the orszag-tang field with face-averaged staggered B, z-invariant: exact
/// face integrals of the continuum div-free field are discretely div-free on
/// every level and coarse-fine consistent (a coarse face integral equals the
/// mean of its fine sub-face integrals identically) — the consistency the
/// init restriction + divB gate require. point sampling would be div-free per
/// level but inconsistent across levels. the same closure fills every level
/// (absolute coordinates).
fn fill_ot(sim: &Sim) {
    let rho0 = GAMMA * GAMMA;
    let p0 = GAMMA;
    let mhd = sim.fields.mhd.as_ref().expect("nmhd must allocate mhd");
    let dy = sim.geom.dx[1];
    let dxx = sim.geom.dx[0];

    // bx = -B0 sin(2 pi y): the x-face average over its y extent.
    for c in &sim.geom.interior.extend(0, 0, 1) {
        let y0 = sim.geom.x_lo[1] + c[1] as f64 * dy;
        let y1 = y0 + dy;
        let avg = B0 * ((2.0 * PI * y1).cos() - (2.0 * PI * y0).cos()) / (2.0 * PI * dy);
        mhd.bface[0].view_mut().set(c, avg);
    }
    // by = B0 sin(4 pi x): the y-face average over its x extent.
    for c in &sim.geom.interior.extend(1, 0, 1) {
        let x0 = sim.geom.x_lo[0] + c[0] as f64 * dxx;
        let x1 = x0 + dxx;
        let avg = B0 * ((4.0 * PI * x0).cos() - (4.0 * PI * x1).cos()) / (4.0 * PI * dxx);
        mhd.bface[1].view_mut().set(c, avg);
    }
    for c in &sim.geom.interior.extend(2, 0, 1) {
        mhd.bface[2].view_mut().set(c, 0.0);
    }
    mhd.bface_initialized.store(true, Ordering::Relaxed);

    for c in sim.geom.interior.iter() {
        let xc = sim.geom.centroid(c);
        let (x, y) = (xc[0], xc[1]);
        let vx = -V0 * (2.0 * PI * y).sin();
        let vy = V0 * (2.0 * PI * x).sin();
        let bx_c = -B0 * (2.0 * PI * y).sin();
        let by_c = B0 * (4.0 * PI * x).sin();
        let prim = MhdPrim {
            hydro: Prim {
                rho: rho0,
                vel: Tensor::new([vx, vy, 0.0]),
                pre: p0,
            },
            mag: Tensor::new([bx_c, by_c, 0.0]),
        };
        let cons = sim.physics.regime.to_conserved(&sim.physics.eos, &prim);
        sim.fields.cons.scatter(
            c,
            Cons {
                chi: Default::default(),
                den: cons.den,
                mom: cons.mom,
                nrg: cons.nrg,
            },
        );
        mhd.bcell[0].view_mut().set(c, bx_c);
        mhd.bcell[1].view_mut().set(c, by_c);
        mhd.bcell[2].view_mut().set(c, 0.0);
    }
}

/// max |divB| (staggered) and max |B| over a level's interior.
fn max_divb_and_b(sim: &Sim) -> (f64, f64, [isize; 3]) {
    let mhd = sim.fields.mhd.as_ref().expect("mhd");
    let inv: [f64; 3] = std::array::from_fn(|ax| 1.0 / sim.geom.dx[ax]);
    let mut max_div = 0.0_f64;
    let mut max_b = 0.0_f64;
    let mut worst = [0_isize; 3];
    for c in sim.geom.interior.iter() {
        let mut div = 0.0;
        let mut bsq = 0.0;
        for aa in 0..3 {
            let lo = *mhd.bface[aa].view().at(c);
            let mut ch = c;
            ch[aa] += 1;
            let hi = *mhd.bface[aa].view().at(ch);
            div += (hi - lo) * inv[aa];
            bsq += lo * lo;
        }
        if div.abs() > max_div {
            max_div = div.abs();
            worst = c;
        }
        max_b = max_b.max(bsq.sqrt());
    }
    (max_div, max_b, worst)
}

/// composite gas mass + x-momentum (coarse outside the coverage + fine).
fn composite_mass_momx(hier: &Hier) -> (f64, f64) {
    let mut mass = 0.0;
    let mut momx = 0.0;
    for lvl in &hier.levels {
        let vol: f64 = lvl.state.geom.dx.iter().product();
        let cov = lvl.coverage.as_ref();
        for c in lvl.state.geom.interior.iter() {
            if let Some(cov) = cov {
                if cov.contains(c) {
                    continue;
                }
            }
            mass += *lvl.state.fields.cons.den.view().at(c) * vol;
            momx += *lvl.state.fields.cons.mom[0].view().at(c) * vol;
        }
    }
    (mass, momx)
}

/// the single-level relative mass drift on the same problem to time `t` — the
/// CT periodic-wrap non-conservation the composite bound calibrates against.
fn single_level_control_drift(t: f64) -> f64 {
    let dx = 1.0 / N as f64;
    let dz = 1.0 / NZ as f64;
    let mut sim = Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N, NZ])
        .spacing([dx, dx, dz])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(CFL)
        .finish()
        .unwrap();
    fill_ot(&sim);
    let k = Kset::new(GAMMA, CFL, 1.0, &sim.geom.allocated);
    let vol = dx * dx * dz;
    let mass = |s: &Sim| -> f64 {
        s.geom
            .interior
            .iter()
            .map(|c| *s.fields.cons.den.view().at(c) * vol)
            .sum()
    };
    let m0 = mass(&sim);
    symbi::sim::evolve::evolve(&mut sim, &k, t).unwrap();
    ((mass(&sim) - m0) / m0).abs()
}

#[test]
fn nmhd_two_level_preserves_divb_across_the_interface() {
    let dx = 1.0 / N as f64;
    let dz = 1.0 / NZ as f64;
    let coarse = Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N, NZ])
        .spacing([dx, dx, dz])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(CFL)
        .finish()
        .unwrap();
    fill_ot(&coarse);
    let ck = Kset::new(GAMMA, CFL, 1.0, &coarse.geom.allocated);

    // refine the vortex core in x/y; z spans the whole (thin periodic) axis.
    let regions = [RefinementRegion {
        x_lo: [0.25, 0.25, 0.0],
        x_hi: [0.75, 0.75, 1.0],
    }];
    let mut hier = Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, |s| {
        Kset::new(GAMMA, CFL, 1.0, &s.geom.allocated)
    })
    .unwrap();
    fill_ot(&hier.levels[1].state);

    // both ICs are analytically div-free.
    for lvl in &hier.levels {
        let (div0, b0, _) = max_divb_and_b(&lvl.state);
        assert!(div0 / b0.max(1.0) < 1e-13, "IC not div-free: {div0:e}");
    }

    let (m0, p0) = composite_mass_momx(&hier);

    // march 10 root steps, asserting divB on both levels after every step —
    // any leak at the coarse-fine interface compounds and trips immediately.
    let mut max_rel = 0.0_f64;
    for step in 0..10u64 {
        hier.evolve_steps(1).unwrap();
        for (ll, lvl) in hier.levels.iter().enumerate() {
            let (max_div, max_b, worst) = max_divb_and_b(&lvl.state);
            let rel = max_div / max_b.max(1.0);
            assert!(
                rel < DIVB_TOL,
                "divB grew at step {step} level {ll} cell {worst:?}: \
                 max|divB|={max_div:e} max|B|={max_b:e} rel={rel:e} — \
                 the coarse-fine B transfer / emf reflux is broken"
            );
            max_rel = max_rel.max(rel);
        }
    }

    // composite gas conservation: the hydro reflux is exact (the budget
    // identity holds to 1e-15 on the fine side), but the MHD substrate itself
    // drifts mass ~1e-9/step across periodic walls even on A single level —
    // the flux reads the CT-evolved bface, and the two stored copies of each
    // periodic wrap face drift apart under the boundary-edge EMF stencils, so
    // the wrap telescoping leaks. measure that single-level control drift on
    // the same problem and assert the 2-level composite stays at its scale —
    // the amr machinery must add nothing.
    let single_drift = single_level_control_drift(hier.levels[0].state.time);
    let (m1, p1) = composite_mass_momx(&hier);
    let rel = |a: f64, b: f64, s: f64| ((a - b) / s).abs();
    let bound = (20.0 * single_drift).max(1e-11);
    assert!(
        rel(m1, m0, m0) < bound,
        "mass drift {:e} exceeds the single-level CT wrap-drift scale {bound:e}",
        rel(m1, m0, m0)
    );
    assert!(
        rel(p1, p0, m0) < bound,
        "momentum drift {:e} exceeds the single-level CT wrap-drift scale {bound:e}",
        rel(p1, p0, m0)
    );

    // the fine bface transverse halo at the CF sides is prolonged coarse data
    // (it was allocation zeros before the staggered CF prolongation): spot a
    // halo row of bface[0] (y = fine interior lo - 1) — the OT bx field is
    // O(B0) there, so a stale halo reads exactly 0.
    {
        let fine = &hier.levels[1].state;
        let fmhd = fine.fields.mhd.as_ref().unwrap();
        let y_halo = fine.geom.interior.spaces[1].lo - 1;
        let x_mid = fine.geom.interior.spaces[0].lo + 4;
        let v = *fmhd.bface[0].view().at([x_mid, y_halo, 0]);
        assert!(
            v.is_finite() && v.abs() > 1e-3,
            "fine bface CF halo looks stale (bx at halo row: {v:e})"
        );
    }

    // physical state everywhere (the prims are post-c2p current).
    for (ll, lvl) in hier.levels.iter().enumerate() {
        let pre = lvl.state.fields.prim.pre_field().unwrap();
        for c in lvl.state.geom.interior.iter() {
            let rho = *lvl.state.fields.prim.rho.view().at(c);
            let p = *pre.view().at(c);
            assert!(rho.is_finite() && rho > 0.0, "level {ll} {c:?}: rho={rho}");
            assert!(p.is_finite() && p > 0.0, "level {ll} {c:?}: p={p}");
        }
    }

    eprintln!("[refine_divb] 10 root steps, max rel divB seen = {max_rel:e} (tol {DIVB_TOL:e})");
}
