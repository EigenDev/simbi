// =============================================================================
// refine_conservation.rs
//
// 2-level STATIC nesting gate: subcycling +
// refluxing conserves the composite-grid totals (coarse cells outside the
// coverage + fine cells inside) to machine precision while waves cross the
// coarse-fine interface, the restriction keeps coarse == average(fine
// children) in the overlap, and a static-nested run matches the equivalent
// uniform-fine run on a smooth problem to truncation error.
//
// absolute indices: the fine level's cell f IS the uniform-fine run's cell f
// (same global origin, same dx), so the smooth-problem comparison reads
// matching absolute coordinates directly.
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;

type Sim<const D: usize> = SimState<Newtonian, D, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset<const D: usize> = AdiabaticSubstrateKernelSet<HostMemory, f64, D>;

fn kset<const D: usize>(sim: &Sim<D>) -> Kset<D>
where
    Newtonian: symbi_hydro::regime::Regime<f64, D>,
    Cartesian: symbi_geometry::Metric<f64, D>,
{
    Kset::<D>::new(GAMMA, CFL, &sim.geom.allocated)
}

/// composite totals (mass, momentum-x, energy): coarse interior outside the
/// coverage + fine interior, volume-weighted (cartesian uniform per level).
fn composite_totals<const D: usize>(
    hier: &Hierarchy<Newtonian, D, D, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset<D>>,
) -> (f64, f64, f64)
where
    Newtonian: symbi_hydro::regime::Regime<f64, D>,
    Cartesian: symbi_geometry::Metric<f64, D>,
{
    let mut mass = 0.0;
    let mut momx = 0.0;
    let mut nrg = 0.0;
    for (ll, lvl) in hier.levels.iter().enumerate() {
        let vol: f64 = lvl.state.geom.dx.iter().product();
        let cov = lvl.coverage.as_ref();
        let cons = &lvl.state.fields.cons;
        let cnrg = cons.nrg_field().unwrap();
        for c in lvl.state.geom.interior.iter() {
            if let Some(cov) = cov {
                if cov.contains(c) {
                    continue;
                }
            }
            mass += *cons.den.view().at(c) * vol;
            momx += *cons.mom[0].view().at(c) * vol;
            nrg += *cnrg.view().at(c) * vol;
        }
        let _ = ll;
    }
    (mass, momx, nrg)
}

fn assert_finite_positive<const D: usize>(
    hier: &Hierarchy<Newtonian, D, D, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset<D>>,
) where
    Newtonian: symbi_hydro::regime::Regime<f64, D>,
    Cartesian: symbi_geometry::Metric<f64, D>,
{
    for (ll, lvl) in hier.levels.iter().enumerate() {
        for c in lvl.state.geom.interior.iter() {
            let den = *lvl.state.fields.cons.den.view().at(c);
            assert!(den.is_finite() && den > 0.0, "level {ll} {c:?}: bad density {den}");
        }
    }
}

/// coarse == average of fine children over the coverage (the restriction ran
/// at the end of every coarse step).
fn assert_restriction_consistency<const D: usize>(
    hier: &Hierarchy<Newtonian, D, D, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset<D>>,
) where
    Newtonian: symbi_hydro::regime::Regime<f64, D>,
    Cartesian: symbi_geometry::Metric<f64, D>,
{
    let cov = hier.levels[0].coverage.as_ref().unwrap();
    let fine = &hier.levels[1].state.fields.cons;
    for c in cov.iter() {
        let coarse_den = *hier.levels[0].state.fields.cons.den.view().at(c);
        let mut sum = 0.0;
        let mut count = 0.0;
        visit_children::<D>(&c, &mut |fc| {
            sum += *fine.den.view().at(*fc);
            count += 1.0;
        });
        let avg = sum / count;
        let rel = ((coarse_den - avg) / coarse_den).abs();
        assert!(
            rel < 1e-12,
            "restriction drift at {c:?}: coarse {coarse_den:e} vs child avg {avg:e} (rel {rel:e})"
        );
    }
}

fn visit_children<const D: usize>(c: &[isize; D], f: &mut impl FnMut(&[isize; D])) {
    let mut fc = [0isize; D];
    rec::<D>(c, &mut fc, 0, f);
    fn rec<const D: usize>(
        c: &[isize; D],
        fc: &mut [isize; D],
        ax: usize,
        f: &mut impl FnMut(&[isize; D]),
    ) {
        if ax == D {
            f(fc);
            return;
        }
        for o in 0..2isize {
            fc[ax] = 2 * c[ax] + o;
            rec::<D>(c, fc, ax + 1, f);
        }
    }
}

// =============================================================================
// 1d sod: composite conservation to machine precision across the interface
// =============================================================================

#[test]
fn sod_1d_two_level_conserves_composite_totals() {
    // periodic walls: mass, momentum AND energy are exactly conserved (an
    // outflow boundary leaks the pressure flux (p_l - p_r)*t into momentum).
    // the coverage edge at x=0.6 sits in the shock's path — by t=0.1 the
    // shock (speed ~1.75) has CROSSED the coarse-fine interface, so the
    // conservation check exercises the refluxed interface, not quiescence.
    let n = 200usize;
    let dx = 1.0 / n as f64;
    let ic = |x: [f64; 1]| {
        let (rho, pre) = if x[0] < 0.5 { (1.0, 1.0) } else { (0.125, 0.1) };
        Prim { rho, vel: symbi_algebra::Tensor::new([0.0]), pre }
    };
    let coarse = Sim::<1>::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([n])
        .spacing([dx])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(CFL)
        .allocate()
        .unwrap()
        .set_initial(ic)
        .build();
    let ck = kset(&coarse);

    let regions = [RefinementRegion { x_lo: [0.4], x_hi: [0.6] }];
    let mut hier =
        Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, |s| kset(s)).unwrap();
    hier.levels[1].state.seed_cells(ic);

    let (m0, p0, e0) = composite_totals(&hier);
    hier.evolve(0.1).unwrap();
    let (m1, p1, e1) = composite_totals(&hier);

    assert!(hier.levels[0].state.iteration > 5);
    assert_finite_positive(&hier);
    assert_restriction_consistency(&hier);

    let rel = |a: f64, b: f64, s: f64| ((a - b) / s).abs();
    assert!(rel(m1, m0, m0) < 1e-12, "mass drift {:e}", rel(m1, m0, m0));
    assert!(rel(e1, e0, e0) < 1e-12, "energy drift {:e}", rel(e1, e0, e0));
    // momentum starts at zero — measure against the mass scale.
    assert!(rel(p1, p0, m0) < 1e-12, "momentum drift {:e}", rel(p1, p0, m0));

    // the shock genuinely left the refined region: density just outside the
    // hi interface departed from its initial 0.125.
    let outside = *hier.levels[0].state.fields.cons.den.view().at([(0.63 / dx) as isize]);
    assert!(
        (outside - 0.125).abs() > 0.05,
        "shock never crossed the coarse-fine interface (den at x=0.63: {outside})"
    );
}

// =============================================================================
// 1d smooth pulse: static nesting matches the uniform-fine run to truncation
// =============================================================================

#[test]
fn smooth_pulse_two_level_matches_uniform_fine() {
    let n = 128usize;
    let dx = 1.0 / n as f64;
    let ic = |x: [f64; 1]| {
        let g = (-((x[0] - 0.5) / 0.06).powi(2)).exp();
        Prim { rho: 1.0 + 0.01 * g, vel: symbi_algebra::Tensor::new([0.0]), pre: 1.0 + 0.014 * g }
    };
    let t_final = 0.04;

    // the 2-level run: coarse n, fine over [0.25, 0.75).
    let coarse = Sim::<1>::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([n])
        .spacing([dx])
        .cfl(CFL)
        .allocate()
        .unwrap()
        .set_initial(ic)
        .build();
    let ck = kset(&coarse);
    let regions = [RefinementRegion { x_lo: [0.25], x_hi: [0.75] }];
    let mut hier =
        Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, |s| kset(s)).unwrap();
    hier.levels[1].state.seed_cells(ic);
    hier.evolve(t_final).unwrap();

    // the uniform-fine reference: 2n cells at dx/2 over the whole domain.
    let mut fine_ref = Sim::<1>::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([2 * n])
        .spacing([dx / 2.0])
        .cfl(CFL)
        .allocate()
        .unwrap()
        .set_initial(ic)
        .build();
    let fk = kset(&fine_ref);
    evolve(&mut fine_ref, &fk, t_final).unwrap();

    // compare the fine level against the reference at matching ABSOLUTE fine
    // indices, well inside the coverage (the pulse stays in [0.35, 0.65]).
    let fine = &hier.levels[1].state;
    let lo = (0.35 / (dx / 2.0)).round() as isize;
    let hi = (0.65 / (dx / 2.0)).round() as isize;
    let mut max_err = 0.0f64;
    for ii in lo..hi {
        let a = *fine.fields.cons.den.view().at([ii]);
        let b = *fine_ref.fields.cons.den.view().at([ii]);
        max_err = max_err.max((a - b).abs());
    }
    // truncation-level agreement: the refined region resolves the pulse at the
    // reference resolution; the residual is the coarse-fine boundary coupling,
    // far below the pulse amplitude (1e-2).
    assert!(
        max_err < 2e-5,
        "static-nested vs uniform-fine density max err {max_err:e} (pulse amplitude 1e-2)"
    );
}

// =============================================================================
// 3d blast: composite conservation with a fully interior coarse-fine box
// =============================================================================

#[test]
fn blast_3d_two_level_conserves_composite_totals() {
    // periodic walls: an outflow boundary passes the (numerically smeared)
    // blast precursor out of the domain after a few steps — a physical loss
    // that would mask the interface bookkeeping this gate pins.
    let n = 16usize;
    let dx = 1.0 / n as f64;
    let ic = |x: [f64; 3]| {
        let r2 = x.iter().map(|&q| (q - 0.5) * (q - 0.5)).sum::<f64>();
        let pre = if r2 < 0.01 { 10.0 } else { 0.1 };
        Prim { rho: 1.0, vel: symbi_algebra::Tensor::new([0.0; 3]), pre }
    };
    let coarse = Sim::<3>::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([n; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(CFL)
        .allocate()
        .unwrap()
        .set_initial(ic)
        .build();
    let ck = kset(&coarse);

    // a tight box around the blast: the shock crosses the coarse-fine faces
    // (0.125 from the initial sphere edge) well before t_final, so the 3d
    // reflux is exercised on all six interface faces.
    let regions = [RefinementRegion { x_lo: [0.375; 3], x_hi: [0.625; 3] }];
    let mut hier =
        Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, |s| kset(s)).unwrap();
    hier.levels[1].state.seed_cells(ic);

    let (m0, p0, e0) = composite_totals(&hier);
    hier.evolve(0.08).unwrap();
    let (m1, p1, e1) = composite_totals(&hier);

    assert!(hier.levels[0].state.iteration > 2);
    // the blast genuinely left the refined box: pressure just outside it rose.
    let probe = [(0.70 / dx) as isize, (0.5 / dx) as isize, (0.5 / dx) as isize];
    let outside = *hier.levels[0].state.fields.prim.pre_field().unwrap().view().at(probe);
    assert!(
        outside > 0.12,
        "blast never crossed the coarse-fine box (pre at probe: {outside})"
    );
    assert_finite_positive(&hier);
    assert_restriction_consistency(&hier);

    let rel = |a: f64, b: f64, s: f64| ((a - b) / s).abs();
    assert!(rel(m1, m0, m0) < 1e-12, "mass drift {:e}", rel(m1, m0, m0));
    assert!(rel(e1, e0, e0) < 1e-12, "energy drift {:e}", rel(e1, e0, e0));
    assert!(rel(p1, p0, m0) < 1e-12, "momentum drift {:e}", rel(p1, p0, m0));
}
