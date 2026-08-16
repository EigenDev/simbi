// =============================================================================
// perturb_seed.rs
//
// per-level primitive perturbation (`Hierarchy::perturb_cells` + `sync_perturbed`):
// a closure evaluated at every level's own cell centers must land each level's
// full resolvable content, where the prolongation-only IC path hands fine levels
// nothing below the coarse nyquist.
//
// gates:
// - the fine level's stored velocity matches the analytic field at its own cell
//   centers to conversion roundoff, including a mode above the root nyquist that
//   prolongation cannot deliver;
// - a velocity-only perturbation leaves every level's density field bit-identical
//   (the primitive round-trip must not smear rho or pre through the kinetic term);
// - the curl-of-tapered-potential construction stays divergence-free under the
//   grid's own central difference, envelope term included: rms(div v) stays far
//   below rms(curl v) — the property that lets a radial taper ride inside the
//   potential with no projection correction.
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.3;
const N: usize = 16;

type Sim = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 3>;
type Hier = Hierarchy<Newtonian, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset>;

/// the analytic test field: v = curl(f(r) A) for a two-mode potential, one mode at
/// 8 root cells per wavelength (resolvable everywhere) and one at 2 root cells per
/// wavelength — the root nyquist, representable only on the refined level — under a
/// radial envelope f(r) = 1 - (r/r0)^2 (clamped at 0), whose gradient term exercises
/// the non-oscillatory part of the curl.
fn seed_velocity(x: [f64; 3]) -> [f64; 3] {
    let center = [0.5, 0.5, 0.5];
    let d = [x[0] - center[0], x[1] - center[1], x[2] - center[2]];
    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let r0 = 0.45;
    let (f, df) = if r < r0 {
        (1.0 - (r / r0).powi(2), -2.0 * r / (r0 * r0))
    } else {
        (0.0, 0.0)
    };
    if f == 0.0 && df == 0.0 {
        return [0.0; 3];
    }
    let rhat = if r > 0.0 {
        [d[0] / r, d[1] / r, d[2] / r]
    } else {
        [0.0; 3]
    };
    let modes: [([f64; 3], [f64; 3], f64, f64); 2] = [
        // 8 root cells per wavelength along x, potential along z
        (
            [2.0 * std::f64::consts::PI * 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            0.03,
            0.3,
        ),
        // 2 root cells per wavelength along y (the root nyquist), potential along x
        (
            [0.0, 2.0 * std::f64::consts::PI * 8.0, 0.0],
            [1.0, 0.0, 0.0],
            0.01,
            1.1,
        ),
    ];
    let mut v = [0.0f64; 3];
    for (k, e, amp, phase) in modes {
        let theta = k[0] * x[0] + k[1] * x[1] + k[2] * x[2] + phase;
        let kxe = [
            k[1] * e[2] - k[2] * e[1],
            k[2] * e[0] - k[0] * e[2],
            k[0] * e[1] - k[1] * e[0],
        ];
        let rxe = [
            rhat[1] * e[2] - rhat[2] * e[1],
            rhat[2] * e[0] - rhat[0] * e[2],
            rhat[0] * e[1] - rhat[1] * e[0],
        ];
        for ax in 0..3 {
            v[ax] += amp * (f * kxe[ax] * theta.cos() + df * rxe[ax] * theta.sin());
        }
    }
    v
}

/// a 2-level hierarchy on [0,1)^3 (middle half refined) holding a uniform gas at
/// rest, fine level seeded by prolongation, then perturbed by the analytic field.
fn perturbed_hierarchy() -> Hier {
    let dx = 1.0 / N as f64;
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 3])
        .origin([0.0; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .unwrap()
        .set_initial(move |_x: [f64; 3]| Prim {
            rho: 1.0,
            vel: Tensor::new([0.0; 3]),
            pre: 1.0,
        })
        .build();
    let ck = Kset::new(GAMMA, CFL, &coarse.geom.allocated);
    let regions = [RefinementRegion {
        x_lo: [0.25; 3],
        x_hi: [0.75; 3],
    }];
    let hier = Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Plm, |s| {
        Kset::new(GAMMA, CFL, &s.geom.allocated)
    })
    .unwrap();
    hier.seed_fine_from_coarse().unwrap();
    hier.perturb_cells(|x, p: Prim<f64, 3>| {
        let dv = seed_velocity(x);
        let mut p = p;
        for ax in 0..3 {
            p.vel[ax] += dv[ax];
        }
        p
    });
    hier
}

#[test]
fn every_level_carries_its_own_resolvable_content() {
    let mut hier = perturbed_hierarchy();
    hier.sync_perturbed();
    // the fine level's stored velocity equals the analytic field at its own centers,
    // nyquist-of-the-root mode included; conversion round-trip is the only error.
    let fine = &hier.levels[1].state;
    let mut worst = 0.0f64;
    for c in fine.geom.interior.iter() {
        let x = fine.geom.cell_coord(c);
        let p = fine.prim_at(c);
        let dv = seed_velocity(x);
        for ax in 0..3 {
            worst = worst.max((p.vel[ax] - dv[ax]).abs());
        }
    }
    assert!(
        worst < 1.0e-12,
        "fine-level velocity departs from the analytic seed by {worst:.3e}; the \
         per-level evaluation is not landing each level's own content"
    );
}

#[test]
fn velocity_perturbation_leaves_density_untouched() {
    let hier = perturbed_hierarchy();
    for (lv, level) in hier.levels.iter().enumerate() {
        let state = &level.state;
        for c in state.geom.interior.iter() {
            let den = *state.fields.cons.den.view().at(c);
            assert!(
                (den - 1.0).abs() < 1.0e-14,
                "level {lv}: density moved to {den:.16} under a velocity-only \
                 perturbation"
            );
        }
    }
}

#[test]
fn curl_construction_is_divergence_free_under_the_grid_difference() {
    let mut hier = perturbed_hierarchy();
    hier.sync_perturbed();
    // central-difference div and curl magnitude over the fine interior (one cell in
    // from the edge). the low-k mode at 16 fine cells per wavelength carries
    // (k dx)^2 / 6 ~ 2.6e-2 discrete-operator error; the nyquist-of-the-root mode at
    // 4 fine cells per wavelength carries ~0.4 but rides at a third the amplitude.
    // the ratio gate at 0.15 catches a broken construction (a taper without its
    // gradient term reads ~1) while admitting the operator's own truncation.
    let fine = &hier.levels[1].state;
    let dx = fine.geom.dx;
    let vel_at = |c: [isize; 3]| -> [f64; 3] {
        let p = fine.prim_at(c);
        [p.vel[0], p.vel[1], p.vel[2]]
    };
    let mut div2 = 0.0f64;
    let mut curl2 = 0.0f64;
    let mut cells = 0usize;
    let lo: [isize; 3] = std::array::from_fn(|ax| fine.geom.interior.spaces[ax].lo);
    let hi: [isize; 3] = std::array::from_fn(|ax| fine.geom.interior.spaces[ax].hi);
    for c in fine.geom.interior.iter() {
        if (0..3).any(|ax| c[ax] == lo[ax] || c[ax] + 1 >= hi[ax]) {
            continue;
        }
        let mut grad = [[0.0f64; 3]; 3];
        for ax in 0..3 {
            let mut cp = c;
            let mut cm = c;
            cp[ax] += 1;
            cm[ax] -= 1;
            let vp = vel_at(cp);
            let vm = vel_at(cm);
            for comp in 0..3 {
                grad[comp][ax] = (vp[comp] - vm[comp]) / (2.0 * dx[ax]);
            }
        }
        let div = grad[0][0] + grad[1][1] + grad[2][2];
        let curl = [
            grad[2][1] - grad[1][2],
            grad[0][2] - grad[2][0],
            grad[1][0] - grad[0][1],
        ];
        div2 += div * div;
        curl2 += curl[0] * curl[0] + curl[1] * curl[1] + curl[2] * curl[2];
        cells += 1;
    }
    assert!(cells > 0, "the divergence gate sampled no interior cells");
    let ratio = (div2 / curl2).sqrt();
    assert!(
        ratio < 0.15,
        "rms(div v)/rms(curl v) = {ratio:.3}; the curl construction (or its \
         envelope-gradient term) is broken"
    );
}
