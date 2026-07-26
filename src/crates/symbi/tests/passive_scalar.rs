// =============================================================================
// passive_scalar.rs
//
// the eulerian passive scalar (dye) gates, end to end through evolve():
//   - a uniform concentration chi = c stays c to rounding through a shocked
//     flow (the dye flux is c times the mass flux, so the update telescopes
//     with the mass update; only non-associativity separates them)
//   - total rho*chi is conserved to machine precision on a periodic domain
//   - a chi step function advects with the flow: monotone, bounded in [0, 1],
//     front position tracks v*t
//   - an undyed run reports has_passive_scalar() = false and allocates nothing
//
// run: cargo test -p symbi --test passive_scalar
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const N: usize = 64;
const L: f64 = 1.0;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

fn build(bc: BoundaryType, ic: impl Fn([f64; 2]) -> Prim<f64, 2>) -> Sim {
    let dx = 2.0 * L / N as f64;
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([-L, -L])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(bc))
        .allocate()
        .expect("sim")
        .set_initial(ic)
        .build()
        .with_passive_scalar()
        .expect("chi alloc")
}

// seed chi = f(x, y) over the WHOLE allocated (ghost-padded) grid so the first
// stage's upwind stencil reads consistent ghosts before any fill runs.
fn seed_chi(sim: &Sim, f: impl Fn(f64, f64) -> f64) {
    let dx = 2.0 * L / N as f64;
    let dom = sim.geom.allocated.clone();
    let cons_chi = sim.fields.cons.chi_field().expect("cons chi");
    let prim_chi = sim.fields.prim.chi_field().expect("prim chi");
    for c in dom.iter() {
        let x = -L + (c[0] as f64 + 0.5) * dx;
        let y = -L + (c[1] as f64 + 0.5) * dx;
        let chi = f(x, y);
        let rho = *sim.fields.cons.den.view().at(c);
        cons_chi.view_mut().set(c, rho * chi);
        prim_chi.view_mut().set(c, chi);
    }
}

fn chi_stats(sim: &Sim) -> (f64, f64, f64) {
    let prim_chi = sim.fields.prim.chi_field().expect("prim chi");
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut total = 0.0;
    let cons_chi = sim.fields.cons.chi_field().expect("cons chi");
    for c in sim.geom.interior.iter() {
        let v = *prim_chi.view().at(c);
        lo = lo.min(v);
        hi = hi.max(v);
        total += *cons_chi.view().at(c);
    }
    (lo, hi, total)
}

#[test]
fn uniform_dye_stays_uniform_through_a_shock() {
    // an outward blast: shocks, rarefactions, fofc-eligible gradients — the
    // dye must not see any of it.
    const C: f64 = 0.7;
    let mut sim = build(BoundaryType::Outflow, |[x, y]| {
        let r2 = x * x + y * y;
        Prim {
            rho: 1.0,
            vel: Tensor::new([0.0, 0.0]),
            pre: if r2 < 0.1 { 10.0 } else { 0.1 },
        }
    });
    seed_chi(&sim, |_, _| C);
    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.4, &sim.geom.allocated);
    evolve(&mut sim, &sub, 0.2).expect("blast with dye");
    let (lo, hi, _) = chi_stats(&sim);
    assert!(
        (lo - C).abs() < 1e-12 && (hi - C).abs() < 1e-12,
        "uniform dye drifted: [{lo}, {hi}] vs {C}"
    );
}

#[test]
fn dye_mass_is_conserved_on_a_periodic_domain() {
    let mut sim = build(BoundaryType::Periodic, |[x, y]| Prim {
        rho: 1.0 + 0.3 * (2.0 * std::f64::consts::PI * x / L).sin().abs(),
        vel: Tensor::new([0.4, -0.25 * (std::f64::consts::PI * y / L).cos()]),
        pre: 1.0,
    });
    seed_chi(&sim, |x, y| if x * x + y * y < 0.25 { 1.0 } else { 0.0 });
    let (_, _, total0) = chi_stats(&sim);
    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.4, &sim.geom.allocated);
    evolve(&mut sim, &sub, 0.5).expect("periodic dye advection");
    let (lo, hi, total1) = chi_stats(&sim);
    assert!(
        (total1 - total0).abs() < 1e-12 * total0.abs().max(1.0),
        "dye mass drifted: {total0} -> {total1}"
    );
    // donor-cell upwinding is monotone: the dye stays inside its initial range.
    assert!(
        lo > -1e-12 && hi < 1.0 + 1e-12,
        "dye left [0,1]: [{lo}, {hi}]"
    );
}

#[test]
fn dye_front_advects_at_the_flow_speed() {
    // uniform flow +x at v = 0.5; a dye front at x = -0.5 must arrive near
    // x = -0.5 + v t (donor-cell smears it; the mid-level crossing tracks).
    const V: f64 = 0.5;
    let mut sim = build(BoundaryType::Periodic, |_| Prim {
        rho: 1.0,
        vel: Tensor::new([V, 0.0]),
        pre: 1.0,
    });
    seed_chi(
        &sim,
        |x, _| if (-0.5..0.0).contains(&x) { 1.0 } else { 0.0 },
    );
    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.4, &sim.geom.allocated);
    let t_final = 0.6;
    evolve(&mut sim, &sub, t_final).expect("uniform advection");

    // the dye's x center of mass moves exactly v*t in a uniform flow.
    let dx = 2.0 * L / N as f64;
    let cons_chi = sim.fields.cons.chi_field().expect("cons chi");
    let (mut m, mut mx) = (0.0, 0.0);
    for c in sim.geom.interior.iter() {
        let x = -L + (c[0] as f64 + 0.5) * dx;
        let v = *cons_chi.view().at(c);
        m += v;
        mx += v * x;
    }
    let com = mx / m;
    let expected = -0.25 + V * t_final;
    assert!(
        (com - expected).abs() < 2.0 * dx,
        "dye center of mass at {com}, expected {expected}"
    );
    let (lo, hi, _) = chi_stats(&sim);
    assert!(
        lo > -1e-12 && hi < 1.0 + 1e-12,
        "dye left [0,1]: [{lo}, {hi}]"
    );
}

// the python driver ALWAYS wraps the sim in a single-level hierarchy (fofc
// lives there), which sequences its stages by hand — a chi phase present only
// in the uni-grid pipeline is invisible to every python run. this gate drives
// the SAME hierarchy loop and pins that the dye actually moves under it.
#[test]
fn dye_advects_under_the_hierarchy_driver() {
    use symbi::sim::refinement::Hierarchy;
    const V: f64 = 0.5;
    let sim = build(BoundaryType::Periodic, |_| Prim {
        rho: 1.0,
        vel: Tensor::new([V, 0.0]),
        pre: 1.0,
    });
    seed_chi(
        &sim,
        |x, _| if (-0.5..0.0).contains(&x) { 1.0 } else { 0.0 },
    );
    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.4, &sim.geom.allocated);
    let mut hier = Hierarchy::single(sim, sub);
    let t_final = 0.6;
    hier.evolve(t_final).expect("hierarchy dye advection");
    let sim = &hier.levels[0].state;

    let dx = 2.0 * L / N as f64;
    let cons_chi = sim.fields.cons.chi_field().expect("cons chi");
    let (mut m, mut mx) = (0.0, 0.0);
    for c in sim.geom.interior.iter() {
        let x = -L + (c[0] as f64 + 0.5) * dx;
        let v = *cons_chi.view().at(c);
        m += v;
        mx += v * x;
    }
    let com = mx / m;
    let expected = -0.25 + V * t_final;
    assert!(
        (com - expected).abs() < 2.0 * dx,
        "hierarchy-driven dye center of mass at {com}, expected {expected} (frozen dye sits at -0.25)"
    );
}

#[test]
fn undyed_run_carries_no_scalar() {
    let dx = 2.0 * L / N as f64;
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([-L, -L])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim")
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([0.0, 0.0]),
            pre: 1.0,
        })
        .build();
    assert!(!sim.has_passive_scalar());
    assert!(sim.fields.cons.chi_field().is_none());
    assert!(sim.fields.prim.chi_field().is_none());
}
