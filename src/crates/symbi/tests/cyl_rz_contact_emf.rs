// =============================================================================
// cyl_rz_contact_emf.rs
//
// the Contact corner EMF E_phi on the cylindrical r-z plane.
// - the corner (r_{i-1/2}, z_{j-1/2}) meets the z-faces (i-1, j), (i, j) and
//   the r-faces (i, j-1), (i, j): each face family pairs across the other grid
//   axis. E_phi's cyclic plane is (p1, p2) = (z, r) on grid axes
//   (g1, g2) = (1, 0), so the z-face mass and B_r fluxes pair across r and the
//   r-face mass and B_z fluxes across z. every EMF input is filled with an
//   asymmetric pseudo-random pattern and the production edge EMF must equal a
//   host evaluation of the Gardiner & Stone (2005) contact formula under that
//   pairing, while the transposed pairing departs by O(1).
// - a z-uniform magnetized equilibrium at rest, B_z = B0 sin(k (r - r_min))
//   between reflecting radial walls with a periodic z, carries a 1e-14
//   z-alternating radial velocity. the seed reaches the face field within the
//   first steps and the face field stays z-uniform to 1e-12 B0 through
//   t = 0.3: a grid-scale mode in the seed decays under a consistent corner
//   stencil and grows by orders of magnitude under a transposed one.
//
// run: cargo test -p symbi --test cyl_rz_contact_emf
// =============================================================================

use std::f64::consts::PI;
use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::evolve::{KernelSet, evolve_with_callback};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cylindrical;
use symbi_grid::Field;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim =
    SimStateGeneric<NewtonianMhd, 2, 3, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory, f64>;
type Kern = NewtonianMhdSubstrateKernelSet<HostMemory, f64, 2>;
type F = Field<f64, 2, HostMemory>;

const GAMMA: f64 = 5.0 / 3.0;
const R_MIN: f64 = 1.0;
const R_MAX: f64 = 3.0;
const Z_MAX: f64 = 1.0;
const B0: f64 = 1e-2;
const K: f64 = 4.0 * PI / (R_MAX - R_MIN);

/// a deterministic value in [-0.5, 0.5] per (cell, salt)
fn rnd(i: isize, j: isize, salt: u64) -> f64 {
    let mut x = (i as i64 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (j as i64 as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
        ^ salt.wrapping_mul(0x2545_F491_4F6C_DD1D);
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    (x as f64 / u64::MAX as f64) - 0.5
}

fn sim(n: usize, seed: f64) -> Sim {
    Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cylindrical)
        .cells([n, n])
        .bounds([R_MIN, 0.0], [R_MAX, Z_MAX])
        .boundaries(Boundaries::per_axis([
            [BoundaryType::Reflect, BoundaryType::Reflect],
            [BoundaryType::Periodic, BoundaryType::Periodic],
        ]))
        .cfl(0.3)
        .allocate()
        .expect("cylindrical r-z sim")
        .set_initial(move |[r, z]| {
            let j = (z / (Z_MAX / n as f64)).floor() as i64;
            let vr = if j % 2 == 0 { seed } else { -seed };
            MhdPrim::new(
                Prim::adiabatic(Density(1.0), Tensor::new([vr, 0.0, 0.0]), Pressure(1.0)),
                Tensor::new([0.0, 0.0, B0 * (K * (r - R_MIN)).sin()]),
            )
        })
        .seed_faces(|axis, [r, _]| {
            if axis == 1 {
                B0 * (K * (r - R_MIN)).sin()
            } else {
                0.0
            }
        })
        .build()
}

fn kernels(s: &Sim) -> Kern {
    Kern::new(GAMMA, 0.3, 1.0, &s.geom.allocated).ct_method(CtMethod::Contact)
}

fn fill(f: &F, salt: u64) {
    for c in f.domain().iter() {
        *f.view_mut().at_mut(c) = rnd(c[0], c[1], salt);
    }
}

/// the Gardiner & Stone (2005) contact corner EMF from its face EMFs [n, s, e, w], cell EMFs
/// [ne, nw, se, sw] and density fluxes [n, s, e, w], with the edge kernel's upwind blend
fn contact(face: [f64; 4], cell: [f64; 4], dflux: [f64; 4]) -> f64 {
    let [en, es, ee, ew] = face;
    let [ene, enw, ese, esw] = cell;
    let [fnf, fs, fe, fw] = dflux;
    let scale = fnf.abs().max(fs.abs()).max(fe.abs()).max(fw.abs());
    let eps = 32.0 * f64::EPSILON * scale;
    let soft = |f: f64, a: f64, b: f64| {
        let d = f.abs() + eps;
        let s = if d > 0.0 { f / d } else { 0.0 };
        0.5 * ((a + b) + s * (a - b))
    };
    let de_jl = soft(fw, 2.0 * (es - esw), 2.0 * (en - enw));
    let de_jr = soft(fe, 2.0 * (ese - es), 2.0 * (ene - en));
    let de_kl = soft(fs, 2.0 * (ew - esw), 2.0 * (ee - ese));
    let de_kr = soft(fnf, 2.0 * (enw - ew), 2.0 * (ene - ee));
    0.25 * (es + en + ew + ee) + 0.125 * (de_jl - de_jr + de_kl - de_kr)
}

/// E_phi at corner `c` with the plane (p1, p2) = (z, r) bound as the runtime binds it and the
/// stencil walking grid axes (g1, g2): cell EMF v_r B_z - v_z B_r, face EMFs -F_z(B_r) and
/// +F_r(B_z), density fluxes F_z(rho) and F_r(rho)
fn host_ephi(s: &Sim, c: [isize; 2], g1: usize, g2: usize) -> f64 {
    let m = s.fields.mhd.as_ref().unwrap();
    let p = &s.fields.prim;
    let back = |o: &[usize]| {
        let mut x = c;
        for &ax in o {
            x[ax] -= 1;
        }
        x
    };
    let at = |f: &F, x: [isize; 2]| *f.view().at(x);
    let cell = |x: [isize; 2]| {
        at(&p.vel[0], x) * at(&m.bcell.b[2], x) - at(&p.vel[2], x) * at(&m.bcell.b[0], x)
    };
    let (bflux_a, bflux_b) = (&m.bflux[1].f[0], &m.bflux[0].f[2]);
    let (fden_p1, fden_p2) = (&s.fields.flux[1].den, &s.fields.flux[0].den);
    contact(
        [
            -at(bflux_a, c),
            -at(bflux_a, back(&[g2])),
            at(bflux_b, c),
            at(bflux_b, back(&[g1])),
        ],
        [
            cell(c),
            cell(back(&[g1])),
            cell(back(&[g2])),
            cell(back(&[g1, g2])),
        ],
        [
            at(fden_p1, c),
            at(fden_p1, back(&[g2])),
            at(fden_p2, c),
            at(fden_p2, back(&[g1])),
        ],
    )
}

#[test]
fn the_cylindrical_ephi_pairs_each_face_family_across_its_own_axis() {
    let s = sim(8, 0.0);
    let k = kernels(&s);
    let m = s.fields.mhd.as_ref().unwrap();
    let p = &s.fields.prim;
    for (salt, f) in [
        (1, &p.vel[0]),
        (2, &p.vel[2]),
        (3, &m.bcell.b[0]),
        (4, &m.bcell.b[2]),
    ] {
        fill(f, salt);
    }
    fill(&m.bflux[1].f[0], 5);
    fill(&m.bflux[0].f[2], 6);
    fill(&s.fields.flux[1].den, 7);
    fill(&s.fields.flux[0].den, 8);
    KernelSet::efield(&k, &s);

    let e = &m.efield.e[0];
    let lo = [s.geom.interior.spaces[0].lo, s.geom.interior.spaces[1].lo];
    let hi = [s.geom.interior.spaces[0].hi, s.geom.interior.spaces[1].hi];
    let (mut scale, mut matched, mut transposed) = (0.0f64, 0.0f64, 0.0f64);
    for i in lo[0]..=hi[0] {
        for j in lo[1]..=hi[1] {
            let got = *e.view().at([i, j]);
            scale = scale.max(got.abs());
            matched = matched.max((got - host_ephi(&s, [i, j], 1, 0)).abs());
            transposed = transposed.max((got - host_ephi(&s, [i, j], 0, 1)).abs());
        }
    }
    assert!(scale > 0.1, "degenerate inputs: max |E_phi| = {scale:e}");
    assert!(
        matched <= 1e-13 * scale,
        "E_phi departs from the contact formula on (g1, g2) = (1, 0) by {:.3e} of max |E_phi|",
        matched / scale
    );
    assert!(
        transposed > 1e-2 * scale,
        "the transposed pairing (0, 1) agrees to {:.3e} of max |E_phi|; the inputs do not distinguish the axes",
        transposed / scale
    );
}

/// max over rows r of max_z F - min_z F on the interior
fn z_departure(s: &Sim, f: &F) -> f64 {
    let i = &s.geom.interior.spaces;
    (i[0].lo..i[0].hi)
        .map(|ir| {
            let (lo, hi) =
                (i[1].lo..i[1].hi).fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), iz| {
                    let v = *f.view().at([ir, iz]);
                    (lo.min(v), hi.max(v))
                });
            hi - lo
        })
        .fold(0.0, f64::max)
}

#[test]
fn a_z_uniform_equilibrium_keeps_its_face_field_z_uniform() {
    let mut s = sim(48, 1e-14);
    let k = kernels(&s);
    let steps = std::cell::RefCell::new(Vec::<(f64, f64)>::new());
    evolve_with_callback(&mut s, &k, 0.3, 1, |st| {
        let m = st.fields.mhd.as_ref().unwrap();
        let dep = z_departure(st, &m.bface.b[0]).max(z_departure(st, &m.bface.b[1])) / B0;
        steps.borrow_mut().push((st.time, dep));
    })
    .expect("cylindrical equilibrium evolve");
    let steps = steps.into_inner();
    assert!(
        steps.len() > 40,
        "only {} steps; the mode had no time to grow",
        steps.len()
    );
    assert!(
        steps.iter().take(4).any(|(_, d)| *d > 0.0),
        "the seed never reached the face field in the first four steps; the gate is vacuous"
    );
    let (t, worst) = steps.iter().fold(
        (0.0, 0.0f64),
        |acc, &(t, d)| if d > acc.1 { (t, d) } else { acc },
    );
    assert!(
        worst < 1e-12,
        "the face field departed from z-uniformity by {worst:.3e} B0 at t = {t:.4}"
    );
}
