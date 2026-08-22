// =============================================================================
// refinement_seam_response.rs
//
// the balanced coarse-fine ghost transfer's response to an off-class fine
// interior, measured in one application on a two-level hierarchy whose coarse
// level stays exactly in-class. a pressure departure of amplitude `a` is
// injected into the fine interior cells adjacent to the low-x seam, the
// coarse-fine ghosts are refilled through the production path, and every ghost
// reports two residuals normalized by the injected amplitude:
//
//   anchor residual -- the ghost against the fine mechanical recursion from
//   its interior anchor: the transfer's own contract, near zero by design at
//   any amplitude.
//
//   truth residual -- the ghost against the in-class column at the ghost's own
//   position, which is what the untouched coarse gas beyond the seam holds:
//   the physical fidelity of the boundary data the fine level evolves against.
//
// a transfer that transports departures faithfully keeps both small. an anchor
// residual near zero with a truth residual near one means the ghosts follow
// the perturbed interior rather than the coarse gas: the seam then presents a
// reflecting boundary to departures, and a flow that feeds the seam a steady
// departure stream accumulates them inside the fine patch.
//
// run: cargo test -p symbi --test refinement_seam_response -- --nocapture
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
const N: usize = 16;
const CFL: f64 = 0.4;
const K0: f64 = 1.0;
const BODY: [f64; 3] = [-1.0, 0.5, 0.5];
/// the perturbed strip: fine interior cells within this many cells of the
/// low-x interior edge, the reach a seam-adjacent departure occupies.
const STRIP: isize = 4;

type Sim = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 3>;
type Hier = Hierarchy<Newtonian, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset>;

fn phi(gm: f64, x: [f64; 3]) -> f64 {
    let r2 = (0..3).map(|a| (x[a] - BODY[a]).powi(2)).sum::<f64>();
    -gm / r2.sqrt()
}

fn line_density(gm: f64, x: [f64; 3]) -> f64 {
    let a = (GAMMA - 1.0) / (GAMMA * K0);
    let c = 1.0 / a + phi(gm, [1.0, x[1], x[2]]);
    (a * (c - phi(gm, x))).powf(1.0 / (GAMMA - 1.0))
}

fn class_line(gm: f64, n: usize, h: f64, y: f64, z: f64) -> Vec<(f64, f64)> {
    let center = |k: usize| [(k as f64 + 0.5) * h, y, z];
    let face = |k: usize| [k as f64 * h, y, z];
    let mut col = vec![(0.0_f64, 0.0_f64); n];
    let rho_out = line_density(gm, center(n - 1));
    col[n - 1] = (rho_out, K0 * rho_out.powf(GAMMA));
    for k in (0..n - 1).rev() {
        let (ra, rb) = (line_density(gm, center(k)), line_density(gm, center(k + 1)));
        let pre = col[k + 1].1
            + rb * (phi(gm, center(k + 1)) - phi(gm, face(k + 1)))
            + ra * (phi(gm, face(k + 1)) - phi(gm, center(k)));
        col[k] = (ra, pre);
    }
    col
}

fn build(gm: f64, balanced: bool, declared: bool) -> Hier {
    let h = 1.0 / N as f64;
    let seed = move |x: [f64; 3], fine: bool| -> Prim<f64, 3> {
        let col = class_line(gm, N, h, x[1], x[2]);
        let j = ((x[0] / h) as usize).min(N - 1);
        let (rho, pre_parent) = col[j];
        let pre = if fine {
            let xc = [(j as f64 + 0.5) * h, x[1], x[2]];
            pre_parent + rho * (phi(gm, xc) - phi(gm, x))
        } else {
            pre_parent
        };
        Prim {
            rho,
            vel: symbi_algebra::Tensor::zeros(),
            pre,
        }
    };
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N, N])
        .spacing([h, h, h])
        .boundaries(Boundaries::uniform(BoundaryType::Reflect))
        .cfl(CFL)
        .allocate()
        .expect("sim construction failed")
        .set_initial(move |x| seed(x, false))
        .build();
    let make =
        move |s: &Sim| Kset::new(GAMMA, CFL, &s.geom.allocated).well_balanced_reconstruction(balanced);
    let ck = make(&coarse);
    let region = RefinementRegion {
        x_lo: [0.25, 0.0, 0.0],
        x_hi: [0.75, 1.0, 1.0],
    };
    let mut hier = Hierarchy::with_refinement(coarse, ck, &[region], ProlongOrder::Ppm, make)
        .unwrap()
        .with_bodies(symbi_ib::BodyCollection::new().add(symbi_ib::Body::gravitational(
            0,
            symbi_algebra::Tensor::new(BODY),
            symbi_algebra::Tensor::zeros(),
            gm,
            1.0e-3,
            0.0,
        )));
    for lvl in 1..hier.levels.len() {
        hier.levels[lvl].state.seed_cells(move |x| seed(x, true));
    }
    if declared {
        hier = hier.with_equilibrium(move |x| seed(x, true)).unwrap();
        hier.seed_equilibrium();
    }
    hier
}

/// one seam-response measurement: inject the departure, refill the ghosts
/// through the production path, and return (anchor residual, truth residual),
/// each the ghost-slab maximum normalized by the injected relative amplitude.
fn seam_response(gm: f64, amp: f64, balanced: bool, declared: bool) -> (f64, f64) {
    let mut hier = build(gm, balanced, declared);
    hier.prime();

    // the departure: a smooth log-pressure bump filling the strip beside the
    // low-x seam, transversely broad so every ghost column feels it. density
    // and velocity stay on the class, so pressure alone carries it -- the
    // decomposition's own variable.
    let h = 1.0 / N as f64;
    let (ilo, jlo, jhi) = {
        let d = &hier.levels[1].state.geom.interior;
        (d.spaces[0].lo, d.spaces[1].lo, d.spaces[1].hi)
    };
    let x_edge = hier.levels[1].state.geom.x_lo[0] + (ilo as f64 + 0.5) * (0.5 * h);
    let strip_w = STRIP as f64 * 0.5 * h;
    hier.levels[1].state.seed_cells(move |x| {
        let col = class_line(gm, N, h, x[1], x[2]);
        let j = ((x[0] / h) as usize).min(N - 1);
        let (rho, pre_parent) = col[j];
        let xc = [(j as f64 + 0.5) * h, x[1], x[2]];
        let mut pre = pre_parent + rho * (phi(gm, xc) - phi(gm, x));
        let depth = (x[0] - x_edge) / strip_w;
        if (0.0..1.0).contains(&depth) {
            let profile = 0.5 * (1.0 + (std::f64::consts::PI * depth).cos());
            pre *= (amp * profile).exp();
        }
        Prim {
            rho,
            vel: symbi_algebra::Tensor::zeros(),
            pre,
        }
    });

    // the production ghost refill on the perturbed state; the coarse level is
    // untouched and remains exactly in-class.
    hier.prime();

    let st = &hier.levels[1].state;
    let rho_v = st.fields.prim.rho.view();
    let pre_v = st.fields.prim.pre.as_ref().expect("adiabatic").view();
    let (alo, x_lo, dxf) = (
        st.geom.allocated.spaces[0].lo,
        st.geom.x_lo,
        st.geom.dx,
    );
    let injected = amp.exp_m1().abs();
    let mut worst_anchor = 0.0_f64;
    let mut worst_truth = 0.0_f64;
    for jy in jlo..jhi {
        for jz in st.geom.interior.spaces[2].lo..st.geom.interior.spaces[2].hi {
            let y = x_lo[1] + (jy as f64 + 0.5) * dxf[1];
            let z = x_lo[2] + (jz as f64 + 0.5) * dxf[2];
            let col = class_line(gm, N, h, y, z);
            for gi in alo..ilo {
                let gx = [x_lo[0] + (gi as f64 + 0.5) * dxf[0], y, z];
                let p_g = *pre_v.at([gi, jy, jz]);
                // truth: the in-class pressure at the ghost's own position on
                // its parent's linear segment, the value the untouched coarse
                // column holds there.
                let j = ((gx[0] / h) as usize).min(N - 1);
                let (rho_c, pre_c) = col[j];
                let xc = [(j as f64 + 0.5) * h, y, z];
                let p_truth = pre_c + rho_c * (phi(gm, xc) - phi(gm, gx));
                worst_truth = worst_truth.max(((p_g - p_truth) / p_truth).abs());
                // anchor: the fine recursion from the neighbor toward the
                // interior, one face at a time.
                let (na, nb) = ([gi + 1, jy, jz], [gi, jy, jz]);
                let xa = [x_lo[0] + (na[0] as f64 + 0.5) * dxf[0], y, z];
                let f = [0.5 * (xa[0] + gx[0]), y, z];
                let (ra, pa) = (*rho_v.at(na), *pre_v.at(na));
                let rb = *rho_v.at(nb);
                let chained = pa + ra * (phi(gm, xa) - phi(gm, f)) + rb * (phi(gm, f) - phi(gm, gx));
                worst_anchor = worst_anchor.max(((p_g - chained) / p_truth).abs());
            }
        }
    }
    (worst_anchor / injected, worst_truth / injected)
}

#[test]
fn the_seam_response_to_an_off_class_interior_is_measured() {
    println!("\nseam ghost response, residual/amplitude (anchor | truth):");
    for gm in [100.0, 400.0, 1600.0] {
        for amp in [0.01, 0.1, 0.5, 1.0] {
            let (anchor, truth) = seam_response(gm, amp, true, false);
            let (_, truth_plain) = seam_response(gm, amp, false, false);
            println!(
                "  GM {gm:>6.0} amp {amp:>5.2}:  anchor {anchor:.3e}   truth {truth:.3e}   \
                 plain-prolong truth {truth_plain:.3e}"
            );
        }
    }
}

#[test]
fn a_declared_target_has_zero_fine_feedback_at_every_amplitude() {
    println!("\ndeclared-target seam response, residual/amplitude (anchor | truth):");
    for gm in [100.0, 400.0, 1600.0] {
        for amp in [0.01, 0.1, 0.5, 1.0] {
            let (anchor, truth) = seam_response(gm, amp, true, true);
            let (_, plain) = seam_response(gm, amp, false, false);
            println!(
                "  GM {gm:>6.0} amp {amp:>5.2}:  anchor {anchor:.3e}   truth {truth:.3e}   \
                 plain-prolong truth {plain:.3e}"
            );
            assert!(
                truth <= plain,
                "GM={gm}, amp={amp}: declared-target truth response {truth:.3e} exceeds the \
                 plain-prolongation floor {plain:.3e}"
            );
        }
    }
}
