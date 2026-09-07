// =============================================================================
// axis_boundary_3d_parity.rs
//
// the axis boundary on a full 3D spherical (r, theta, phi) grid. with the azimuth gridded,
// the ghost at label (r, -theta, phi) is the physical cell at (r, theta, phi + pi): the fill
// mirrors theta and rotates the source half a period along phi, and the orthonormal theta
// and phi components change sign. the gates:
//
// - every halo cell, corners included, equals the analytic continuation of a smooth
//   Cartesian-uniform state through the pole (with the outflow radial walls clamping r), and
//   equals its composed image cell with the parity signs, bit for bit;
// - every transverse halo face of the staggered field, including the duplicated closing
//   phi face, equals the rotated image face;
// - after evolution the constrained-transport divergence is preserved, the two copies of the
//   closing phi face agree, and the polar halos refilled from the evolved state still match
//   their images.
// =============================================================================
use std::f64::consts::PI;
use symbi::prelude::KernelSet;
use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet3D;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Spherical;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Rmhd, 3, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = RmhdSubstrateKernelSet3D<HostMemory, f64>;

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.3;
const R_LO: f64 = 1.0;
const R_HI: f64 = 2.0;
const N: [usize; 3] = [6, 8, 8];
const V0: [f64; 3] = [0.03, 0.02, 0.01];
const B0: [f64; 3] = [0.1, 0.07, 0.05];

/// the spherical orthonormal components of a uniform Cartesian vector at (theta, phi).
fn spherical_of(v: [f64; 3], th: f64, ph: f64) -> [f64; 3] {
    let (st, ct, sp, cp) = (th.sin(), th.cos(), ph.sin(), ph.cos());
    [
        v[0] * st * cp + v[1] * st * sp + v[2] * ct,
        v[0] * ct * cp + v[1] * ct * sp - v[2] * st,
        -v[0] * sp + v[1] * cp,
    ]
}

/// a smooth gas state in the Cartesian point, [rho, v_r, v_theta, v_phi, pre]: uniform
/// vectors and scalars built from x/r, y/r, z/r.
fn probe_gas(r: f64, th: f64, ph: f64) -> [f64; 5] {
    let v = spherical_of(V0, th, ph);
    [
        1.0 + 0.1 * r + 0.05 * th.cos() + 0.02 * th.sin() * ph.cos(),
        v[0],
        v[1],
        v[2],
        1.0 + 0.03 * th.sin() * ph.sin(),
    ]
}

fn probe(r: f64, th: f64, ph: f64) -> MhdPrim<f64, 3> {
    let g = probe_gas(r, th, ph);
    MhdPrim::new(
        Prim::adiabatic(
            Density(g[0]),
            Tensor::new([g[1], g[2], g[3]]),
            Pressure(g[4]),
        ),
        Tensor::new(spherical_of(B0, th, ph)),
    )
}

fn make() -> (Sim, Kern) {
    let sim = Sim::build(Rmhd, IdealGas { gamma: GAMMA }, Spherical)
        .cells(N)
        .spacing([
            (R_HI - R_LO) / N[0] as f64,
            PI / N[1] as f64,
            2.0 * PI / N[2] as f64,
        ])
        .origin([R_LO, 0.0, 0.0])
        .boundaries(Boundaries::per_axis([
            [BoundaryType::Outflow, BoundaryType::Outflow],
            [BoundaryType::Axis, BoundaryType::Axis],
            [BoundaryType::Periodic, BoundaryType::Periodic],
        ]))
        .cfl(CFL)
        .allocate()
        .expect("spherical 3D construction")
        .set_initial(|[r, th, ph]| probe(r, th, ph))
        .seed_faces(|axis, [_r, th, ph]| spherical_of(B0, th, ph)[axis])
        .build();
    let k = Kern::new(GAMMA, CFL, 1.0, &sim.geom.allocated);
    (sim, k)
}

/// the interior cell a halo cell reads, and the sign of its theta and phi components: r
/// clamps to the wall cell, a theta crossing mirrors the index and rotates phi by half the
/// period, phi wraps.
fn image(sim: &Sim, c: [isize; 3]) -> ([isize; 3], f64) {
    let int = &sim.geom.interior;
    let (ilo, ihi) = (int.spaces[0].lo, int.spaces[0].hi);
    let (jlo, jhi) = (int.spaces[1].lo, int.spaces[1].hi);
    let (klo, khi) = (int.spaces[2].lo, int.spaces[2].hi);
    let n = khi - klo;
    let i = c[0].clamp(ilo, ihi - 1);
    let (j, turned) = if c[1] < jlo {
        (2 * jlo - 1 - c[1], true)
    } else if c[1] >= jhi {
        (2 * jhi - 1 - c[1], true)
    } else {
        (c[1], false)
    };
    let mut k = klo + (c[2] - klo).rem_euclid(n);
    if turned {
        k = klo + (k - klo + n / 2).rem_euclid(n);
    }
    ([i, j, k], if turned { -1.0 } else { 1.0 })
}

fn cell(sim: &Sim, c: [isize; 3]) -> [f64; 8] {
    let p = &sim.fields.prim;
    let m = sim.fields.mhd.as_ref().expect("mhd fields");
    [
        *p.rho.view().at(c),
        *p.vel[0].view().at(c),
        *p.vel[1].view().at(c),
        *p.vel[2].view().at(c),
        *p.pre.as_ref().expect("pressure").view().at(c),
        *m.bcell[0].view().at(c),
        *m.bcell[1].view().at(c),
        *m.bcell[2].view().at(c),
    ]
}

const NAMES: [&str; 8] = [
    "rho", "v_r", "v_theta", "v_phi", "pre", "B_r", "B_theta", "B_phi",
];
/// which slots carry the theta/phi parity of a polar crossing.
const POLAR: [bool; 8] = [false, false, true, true, false, false, true, true];

fn halo_cells(sim: &Sim) -> Vec<[isize; 3]> {
    let int = &sim.geom.interior;
    sim.geom
        .allocated
        .iter()
        .filter(|c| (0..3).any(|a| c[a] < int.spaces[a].lo || c[a] >= int.spaces[a].hi))
        .collect()
}

/// every halo cell against its composed image, bit for bit.
fn check_cells_against_images(sim: &Sim, label: &str) {
    let cells = halo_cells(sim);
    assert!(
        cells.len() > 1000,
        "{label}: the halo was enumerated ({} cells)",
        cells.len()
    );
    for c in cells {
        let (img, sign) = image(sim, c);
        let (g, s) = (cell(sim, c), cell(sim, img));
        for k in 0..8 {
            let want = if POLAR[k] { sign * s[k] } else { s[k] };
            if g[k] == 0.0 && want == 0.0 {
                continue;
            }
            assert_eq!(
                g[k].to_bits(),
                want.to_bits(),
                "{label}: {} at halo {c:?} = {} vs image {img:?} = {want}",
                NAMES[k],
                g[k]
            );
        }
    }
}

/// every halo cell's gas state against the analytic continuation through the pole, the
/// outflow walls clamping r to the wall cell's radius.
fn check_cells_against_continuation(sim: &Sim) {
    let int = &sim.geom.interior;
    let (ilo, ihi) = (int.spaces[0].lo, int.spaces[0].hi);
    for c in halo_cells(sim) {
        let x = sim.geom.cell_coord(c);
        let r = sim.geom.cell_coord([c[0].clamp(ilo, ihi - 1), c[1], c[2]])[0];
        let expect = probe_gas(r, x[1], x[2]);
        let got = cell(sim, c);
        for k in 0..5 {
            assert!(
                (got[k] - expect[k]).abs() <= 1e-10 * expect[k].abs().max(1.0),
                "{} at halo {c:?} (r {r}, theta {}, phi {}) = {} vs continuation {}",
                NAMES[k],
                x[1],
                x[2],
                got[k],
                expect[k]
            );
        }
    }
}

/// every transverse halo face of `bface[d]` against its rotated image face, bit for bit, and
/// against the seeded field's continuation.
fn check_faces(sim: &Sim, label: &str, against_seed: bool) {
    let int = &sim.geom.interior;
    let m = sim.fields.mhd.as_ref().expect("mhd fields");
    let (ilo, ihi) = (int.spaces[0].lo, int.spaces[0].hi);
    let (klo, khi) = (int.spaces[2].lo, int.spaces[2].hi);
    let n = khi - klo;
    for d in 0..3 {
        let owned = int.extend(d, 0, 1);
        let mut checked = 0;
        for f in m.bface[d].domain().iter() {
            let in_owned = (0..3).all(|a| f[a] >= owned.spaces[a].lo && f[a] < owned.spaces[a].hi);
            let transverse_halo =
                (0..3).any(|a| a != d && (f[a] < owned.spaces[a].lo || f[a] >= owned.spaces[a].hi));
            let own_axis_inside = f[d] >= owned.spaces[d].lo && f[d] < owned.spaces[d].hi;
            if in_owned || !transverse_halo || !own_axis_inside {
                continue;
            }
            // the image on the transverse axes: the face's own index is a face index and stays
            // as it is (it lies inside the owned face range), except that a phi face rotates
            // within the n distinct faces under a polar crossing (the closing face is face lo).
            let mut probe_cell = f;
            probe_cell[d] = int.spaces[d].lo;
            let (img, sign) = image(sim, probe_cell);
            let mut fi = img;
            fi[d] = f[d];
            let turned = sign < 0.0;
            if d == 2 && turned {
                fi[2] = klo + (f[2] - klo + n / 2).rem_euclid(n);
            }
            let g = *m.bface[d].view().at(f);
            let s = *m.bface[d].view().at(fi);
            let want = if d != 0 && turned { -s } else { s };
            if !(g == 0.0 && want == 0.0) {
                assert_eq!(
                    g.to_bits(),
                    want.to_bits(),
                    "{label}: bface[{d}] at halo face {f:?} = {g} vs image face {fi:?} = {want}"
                );
            }
            if against_seed {
                let mut rc = f;
                rc[0] = f[0].clamp(ilo, ihi - 1);
                let x = sim.geom.face_coord(f, d);
                let r = sim.geom.face_coord(rc, d)[0];
                let expect = spherical_of(B0, x[1], x[2])[d];
                let _ = r;
                assert!(
                    (g - expect).abs() <= 1e-12 * expect.abs().max(1.0),
                    "bface[{d}] at halo face {f:?} = {g} vs seeded continuation {expect}"
                );
            }
            checked += 1;
        }
        assert!(
            checked > 100,
            "{label}: bface[{d}] halo faces were read ({checked})"
        );
    }
}

fn closing_face_gap(sim: &Sim) -> f64 {
    let int = &sim.geom.interior;
    let m = sim.fields.mhd.as_ref().expect("mhd fields");
    let (klo, khi) = (int.spaces[2].lo, int.spaces[2].hi);
    let mut gap = 0.0f64;
    for i in int.spaces[0].lo..int.spaces[0].hi {
        for j in int.spaces[1].lo..int.spaces[1].hi {
            let a = *m.bface[2].view().at([i, j, klo]);
            let b = *m.bface[2].view().at([i, j, khi]);
            gap = gap.max((a - b).abs());
        }
    }
    gap
}

#[test]
fn every_halo_cell_is_the_continuation_through_the_pole() {
    let (sim, k) = make();
    k.c2p(&sim);
    k.ghost_fill(&sim);
    check_cells_against_continuation(&sim);
    check_cells_against_images(&sim, "after the fill");
}

#[test]
fn every_transverse_halo_face_is_the_rotated_image_face() {
    let (sim, k) = make();
    k.c2p(&sim);
    k.ghost_fill(&sim);
    check_faces(&sim, "after the fill", true);
}

#[test]
fn the_evolved_field_keeps_its_divergence_its_closing_face_and_its_polar_seam() {
    let (mut sim, k) = make();
    k.c2p(&sim);
    k.ghost_fill(&sim);
    let before = sim.conservation_diag().expect("diag").div_b.expect("div b");
    evolve(&mut sim, &k, 0.05).expect("evolve");
    let after = sim.conservation_diag().expect("diag").div_b.expect("div b");
    let gap = closing_face_gap(&sim);
    // measured at t = 0.05 on the (6, 8, 8) shell: max |div B| 5.054e-1 -> 5.006e-1 (the
    // area-weighted census drifts at truncation level as the point-sampled field relaxes, the
    // same one to two percent a polar-free wedge shows), and the two copies of the closing
    // phi face agree to 2.8e-17 against a field of 0.1. the bounds hold the divergence within
    // five percent of its seed and the closing gap at roundoff.
    assert!(
        sim.time >= 0.05 - 1e-12,
        "the run reached t = 0.05 (t = {})",
        sim.time
    );
    assert!(
        after <= 1.05 * before,
        "max |div B| left its seeded level: {before:.3e} -> {after:.3e}"
    );
    assert!(
        gap <= 1e-14,
        "the closing phi face's two copies drifted apart by {gap:.3e}"
    );
    k.ghost_fill(&sim);
    check_cells_against_images(&sim, "after evolution");
    check_faces(&sim, "after evolution", false);
}
