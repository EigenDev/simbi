// =============================================================================
// refinement_band_consistency.rs
//
// the balanced restriction band's deviation from conservation, measured under
// refinement on a column that is off the mechanical class.
//
// the band rewrites covered coarse cells at a seam as
//   p_band = <p_fine - p_eq_fine> + p_eq_coarse,
// where p_eq_fine chains the piecewise-constant-density path integral of
// -rho dphi through the fine cells and p_eq_coarse chains it through the coarse
// cells. conservation gives <p_fine>, so the band's deviation is exactly
//   p_eq_coarse - <p_eq_fine>.
// both chains integrate the same field, and the chain is nonlinear in the
// density distribution: eight fine cells carrying a grid-scale density
// fluctuation integrate to something other than one coarse cell carrying their
// mean. a consistent restriction leaves that difference at second order in the
// mesh spacing, so halving h must cut the relative deviation by about four.
//
// the seed is deliberately off-class: an in-class column drives both chains to
// reproduce the class exactly, the deviation cancels identically, and the
// measurement is vacuous however wrong the operator is.
//
// run: cargo test -p symbi --test refinement_band_consistency -- --nocapture
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const K0: f64 = 1.0;
/// the point-mass strength. the band's deviation saturates in GM: raising it
/// 256x moves the n = 16 deviation only 4.9e-4 -> 1.2e-3 and leaves the
/// convergence rate between 4.3 and 5.3, so the chain carries a steep potential
/// across the seam without loss of order.
const GM: f64 = 100.0;
const BODY: [f64; 3] = [-1.0, 0.5, 0.5];
/// the amplitude of the grid-scale density fluctuation that takes the column off
/// the mechanical class. large enough that the deviation clears roundoff at both
/// resolutions, small enough to keep the column physical.
const RHO_RIPPLE: f64 = 0.05;

type Sim = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 3>;
type Hier = Hierarchy<Newtonian, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset>;

fn phi(x: [f64; 3]) -> f64 {
    let r2 = (0..3).map(|a| (x[a] - BODY[a]).powi(2)).sum::<f64>();
    -GM / r2.sqrt()
}

fn line_density(x: [f64; 3]) -> f64 {
    let a = (GAMMA - 1.0) / (GAMMA * K0);
    let c = 1.0 / a + phi([1.0, x[1], x[2]]);
    (a * (c - phi(x))).powf(1.0 / (GAMMA - 1.0))
}

fn class_line(n: usize, h: f64, y: f64, z: f64) -> Vec<(f64, f64)> {
    let center = |k: usize| [(k as f64 + 0.5) * h, y, z];
    let face = |k: usize| [k as f64 * h, y, z];
    let mut col = vec![(0.0_f64, 0.0_f64); n];
    let rho_out = line_density(center(n - 1));
    col[n - 1] = (rho_out, K0 * rho_out.powf(GAMMA));
    for k in (0..n - 1).rev() {
        let (ra, rb) = (line_density(center(k)), line_density(center(k + 1)));
        let pre = col[k + 1].1
            + rb * (phi(center(k + 1)) - phi(face(k + 1)))
            + ra * (phi(face(k + 1)) - phi(center(k)));
        col[k] = (ra, pre);
    }
    col
}

/// the fine lattice's density ripple: one cell-scale oscillation per axis, so the
/// eight children under a coarse cell carry genuinely different densities while
/// their mean stays near the class value.
fn ripple(x: [f64; 3], h_fine: f64) -> f64 {
    let phase: f64 = (0..3)
        .map(|a| (std::f64::consts::PI * x[a] / h_fine).sin())
        .product();
    1.0 + RHO_RIPPLE * phase
}

fn build(n: usize) -> Hier {
    let h = 1.0 / n as f64;
    let h_fine = 0.5 * h;
    let seed = move |x: [f64; 3], fine: bool| -> Prim<f64, 3> {
        let col = class_line(n, h, x[1], x[2]);
        let j = ((x[0] / h) as usize).min(n - 1);
        let (rho, pre_parent) = col[j];
        let pre = if fine {
            let xc = [(j as f64 + 0.5) * h, x[1], x[2]];
            pre_parent + rho * (phi(xc) - phi(x))
        } else {
            pre_parent
        };
        // the ripple rides the fine lattice alone; the coarse level takes the
        // class density and receives the fine mean through the restriction.
        let rho = if fine { rho * ripple(x, h_fine) } else { rho };
        Prim {
            rho,
            vel: symbi_algebra::Tensor::zeros(),
            pre,
        }
    };
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([n, n, n])
        .spacing([h, h, h])
        .boundaries(Boundaries::uniform(BoundaryType::Reflect))
        .cfl(CFL)
        .allocate()
        .expect("sim construction failed")
        .set_initial(move |x| seed(x, false))
        .build();
    let make =
        move |s: &Sim| Kset::new(GAMMA, CFL, &s.geom.allocated).well_balanced_reconstruction(true);
    let ck = make(&coarse);
    let region = RefinementRegion {
        x_lo: [0.25, 0.0, 0.0],
        x_hi: [0.75, 1.0, 1.0],
    };
    let hier = Hierarchy::with_refinement(coarse, ck, &[region], ProlongOrder::Ppm, make)
        .unwrap()
        .with_bodies(
            symbi_ib::BodyCollection::new().add(symbi_ib::Body::gravitational(
                0,
                symbi_algebra::Tensor::new(BODY),
                symbi_algebra::Tensor::zeros(),
                GM,
                1.0e-3,
                0.0,
            )),
        );
    for lvl in 1..hier.levels.len() {
        hier.levels[lvl].state.seed_cells(move |x| seed(x, true));
    }
    hier
}

/// the largest relative gap between a covered coarse cell's conserved energy and
/// the mean of its eight children, over the coverage, after one root step.
///
/// energy is the clean probe. the band rebuilds `nrg` from the restricted density
/// and velocity together with its rewritten pressure, and c2p derived the
/// restricted pressure from those same two fields, so the kinetic terms are
/// common to both and cancel: what survives is the band's pressure rewrite
/// divided by (gamma - 1). the deep covered cells keep the conservative average
/// and sit at roundoff, so the maximum reports the band alone.
fn band_deviation(hier: &mut Hier) -> (f64, [isize; 3]) {
    hier.evolve_steps(1).expect("one root step");

    let fine_nrg = hier.levels[1]
        .state
        .fields
        .cons
        .nrg_field()
        .expect("adiabatic")
        .view();
    let coarse_nrg = hier.levels[0]
        .state
        .fields
        .cons
        .nrg_field()
        .expect("adiabatic")
        .view();
    let coverage = hier.levels[0]
        .coverage
        .clone()
        .expect("the root level is covered");

    let mut worst = 0.0_f64;
    let mut where_ = [0isize; 3];
    for ii in coverage.spaces[0].lo..coverage.spaces[0].hi {
        for jj in coverage.spaces[1].lo..coverage.spaces[1].hi {
            for kk in coverage.spaces[2].lo..coverage.spaces[2].hi {
                let mut mean = 0.0;
                for di in 0..2 {
                    for dj in 0..2 {
                        for dk in 0..2 {
                            mean += *fine_nrg.at([2 * ii + di, 2 * jj + dj, 2 * kk + dk]);
                        }
                    }
                }
                mean /= 8.0;
                let dev = ((*coarse_nrg.at([ii, jj, kk]) - mean) / mean).abs();
                if dev > worst {
                    worst = dev;
                    where_ = [ii, jj, kk];
                }
            }
        }
    }
    (worst, where_)
}

#[test]
fn the_balanced_restriction_band_converges_to_the_conservative_average() {
    let (d_coarse, at_c) = band_deviation(&mut build(16));
    let (d_fine, at_f) = band_deviation(&mut build(32));
    let rate = d_coarse / d_fine.max(f64::MIN_POSITIVE);
    println!(
        "band deviation from the conservative average:\n  \
         n = 16: {d_coarse:.4e} at {at_c:?}\n  \
         n = 32: {d_fine:.4e} at {at_f:?}\n  \
         ratio: {rate:.3}  (second order -> ~4)"
    );
    // the ripple is what puts the two chains on different integrands; without a
    // measurable deviation at the coarse resolution the convergence claim below
    // is about roundoff and decides nothing.
    assert!(
        d_coarse > 1.0e-12,
        "the off-class seed left the band at roundoff ({d_coarse:.3e}); the setup no longer \
         exercises the chain mismatch it is built to measure"
    );
    assert!(
        rate > 3.0,
        "the band's deviation from conservation fell by {rate:.3} under one refinement \
         ({d_coarse:.3e} -> {d_fine:.3e}); a consistent restriction converges at second order \
         (~4). the covered coarse cells are being rewritten with an error that does not vanish \
         with the mesh."
    );
}

/// the covered band strips of the low-x seam in coarse absolute indices.
const BAND: usize = 4;

/// the band decode's per-cell contract, pinned by direct dispatch with
/// hand-written departures: an admissible sum decodes to departure plus chain,
/// and an inadmissible one keeps the cell's conservative pressure bit for bit.
/// the conservative value stands in for a drain-evacuated fine average, fifty
/// times below the class, so a fallback onto the class or the anchor reads as
/// a class-scale pressure and fails outright.
#[test]
fn an_inadmissible_band_decode_keeps_the_conservative_pressure() {
    let mut hier = build(16);
    hier.prime();

    let coverage = hier.levels[0].coverage.clone().expect("covered root");
    let mut band_c = coverage.clone();
    band_c.spaces[0].hi = band_c.spaces[0].lo + BAND as isize;
    let uncovered = band_c.spaces[0].lo - 1;

    // the conservative stand-in: the class pressure scaled to a drained band.
    let evac = 0.02;
    let st = &hier.levels[0].state;
    let cons: std::collections::HashMap<[isize; 3], f64> = {
        let pre_v = st.fields.prim.pre.as_ref().expect("adiabatic").view();
        let mut m = std::collections::HashMap::new();
        for c in band_c.iter() {
            m.insert(c, *pre_v.at(c) * evac);
        }
        m
    };
    {
        let mut pre_v = st.fields.prim.pre.as_ref().expect("adiabatic").view_mut();
        for (c, v) in &cons {
            *pre_v.at_mut(*c) = *v;
        }
    }

    // departures by parity of the x index: a value far below any pressure in
    // the problem forces the inadmissible arm, and zero is the in-class value
    // whose decode is the positive chain.
    let scratch = symbi_grid::Field::<f64, 3, symbi_xpu::HostMemory>::zeros(&st.geom.allocated)
        .expect("scratch");
    {
        let mut dep_v = scratch.view_mut();
        for c in band_c.iter() {
            *dep_v.at_mut(c) = if c[0] % 2 == 0 { -1.0e6 } else { 0.0 };
        }
    }

    // the dispatch mirrors the transfer's band decode: the uncovered row as the
    // reference clamp, the coarse lattice scalars, then the body slots in
    // declared order (pos, mass, softening, softening kind per slot; an absent
    // slot carries zero mass with unit softening).
    let mut lo: Vec<i32> = (0..3)
        .map(|a| st.geom.interior.spaces[a].lo as i32)
        .collect();
    let mut hi: Vec<i32> = (0..3)
        .map(|a| st.geom.interior.spaces[a].hi as i32 - 1)
        .collect();
    lo[0] = uncovered as i32;
    hi[0] = uncovered as i32;
    let ints: Vec<i32> = lo.iter().chain(hi.iter()).copied().collect();
    let mut scalars = Vec::new();
    scalars.extend_from_slice(&st.geom.x_lo);
    scalars.extend_from_slice(&st.geom.dx);
    let bodies = &st.immersed.as_ref().unwrap().bodies;
    for b in 0..symbi_ib::collection::MAX_SOURCE_BODIES {
        if b < bodies.len() {
            let body = bodies.get(b);
            for ax in 0..3 {
                scalars.push(body.position[ax]);
            }
            scalars.push(if body.has_gravity() { body.mass } else { 0.0 });
            scalars.push(body.softening().unwrap_or(1.0));
            scalars.push(body.softening_kind().unwrap_or(0.0));
        } else {
            scalars.extend_from_slice(&[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        }
    }
    symbi::regimes::substrate_kernels::dispatch_fields_each::<f64, symbi_xpu::HostMemory, 3>(
        symbi_ir::KernelId::WbBandDecode { ndim: 3 }.name(),
        &band_c,
        &[&st.fields.prim.rho, &scratch],
        &[st.fields.prim.pre.as_ref().expect("adiabatic")],
        &ints,
        &scalars,
    );

    let pre_v = st.fields.prim.pre.as_ref().expect("adiabatic").view();
    let mut abstained = 0usize;
    let mut decoded = 0usize;
    for (c, v) in &cons {
        let p = *pre_v.at(*c);
        if c[0] % 2 == 0 {
            assert!(
                p == *v,
                "band cell {c:?} was forced inadmissible and holds {p:.6e}; the fallback \
                 must return the conservative pressure {v:.6e} bit for bit"
            );
            abstained += 1;
        } else {
            assert!(
                p > 0.0 && p.is_finite() && p != *v,
                "band cell {c:?} carries a zero departure; its decode is the chain value, \
                 which differs from the drained conservative stand-in {v:.6e}, got {p:.6e}"
            );
            assert!(
                p > 10.0 * v,
                "band cell {c:?} decoded to {p:.6e}; a zero-departure decode lands on the \
                 class chain, which sits far above the drained stand-in {v:.6e}"
            );
            decoded += 1;
        }
    }
    println!("band decode contract: {abstained} abstained bitwise, {decoded} decoded on the chain");
    assert!(abstained > 0 && decoded > 0);
}
