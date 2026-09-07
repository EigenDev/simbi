// =============================================================================
// axis_boundary_parity.rs
//
// the axis boundary on a 2.5D spherical (r, theta) grid. crossing the polar axis is a
// half-turn about it, so a vector's theta and phi components change sign there while its
// radial component and every scalar continue. the gates read the production ghost fill:
//
// - on the pole face declared `Axis`, the ghost of (rho, pre, v_r, B_r) is the mirror
//   image and the ghost of (v_theta, B_theta, v_phi, B_phi) is its negative, exactly;
// - on the equatorial face declared `Reflect` the azimuthal components keep the wall's
//   even continuation, so the two kinds differ only in the out-of-plane sign;
// - a driven inner-radius face owns its corner with the pole: the corner ghost is the
//   prescription evaluated at the corner cell's own coordinate;
// - a toroidal field B_phi = B0 r sin(theta) balanced by p = p0 - B0^2 r^2 sin^2(theta)
//   is held through the geometric update with a residual that converges under refinement,
//   with the axis fill keeping the near-pole residual at the level of the bulk.
// =============================================================================
use std::f64::consts::PI;
use symbi::prelude::KernelSet;
use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Spherical;
use symbi_hydro::RMHD_SPEC;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::Prim;
use symbi_source_compile::SourceConfig;
use symbi_source_compile::expr_bridge::build_boundary_dag;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<Rmhd, 2, 3, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = RmhdSubstrateKernelSet<HostMemory, f64, 2>;

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.3;
const R_LO: f64 = 1.0;
const R_HI: f64 = 2.0;

/// a smooth, asymmetric state with every component nonzero, so each parity is a real check.
fn probe_state(r: f64, th: f64) -> MhdPrim<f64, 3> {
    MhdPrim::new(
        Prim::adiabatic(
            Density(1.0 + 0.1 * r + 0.05 * th),
            Tensor::new([
                0.1 * th.sin() + 0.02 * r,
                0.05 * th + 0.01 * r,
                0.2 * th + 0.03 * r,
            ]),
            Pressure(1.0 + 0.1 * th),
        ),
        Tensor::new([0.1 + 0.02 * th, 0.05 * th, 0.3 * th + 0.04 * r]),
    )
}

/// the inner-radius prescription [rho, v_r, v_theta, v_phi, pre, B_r, B_theta, B_phi] as a
/// function of theta (VARIABLE_X2): rho = 2 + 0.2 theta^2, v = (0.5, 0, 0.3 theta), p = 3,
/// B = (0, 0, 0.2 theta). the prescription is regular across the pole, its scalars even and
/// its azimuthal components odd in theta, so its value at a mirrored coordinate is its pole
/// image.
fn driven_json() -> String {
    r#"{
        "kind": "dirichlet", "dim": 3, "outputs": [5, 6, 7, 10, 8, 7, 7, 11], "params": [],
        "nodes": [ {"op": "VARIABLE_X2"}, {"op": "CONSTANT", "value": 0.2},
                   {"op": "MULTIPLY", "left": 0, "right": 0},
                   {"op": "MULTIPLY", "left": 2, "right": 1}, {"op": "CONSTANT", "value": 2.0},
                   {"op": "ADD", "left": 4, "right": 3},
                   {"op": "CONSTANT", "value": 0.5}, {"op": "CONSTANT", "value": 0.0},
                   {"op": "CONSTANT", "value": 3.0}, {"op": "CONSTANT", "value": 0.3},
                   {"op": "MULTIPLY", "left": 0, "right": 9},
                   {"op": "MULTIPLY", "left": 0, "right": 1} ]
    }"#
    .to_string()
}

fn driven_values(th: f64) -> [f64; 8] {
    [
        2.0 + 0.2 * th * th,
        0.5,
        0.0,
        0.3 * th,
        3.0,
        0.0,
        0.0,
        0.2 * th,
    ]
}

fn probe_sim(n: [usize; 2], pole: BoundaryType) -> (Sim, Kern) {
    let dth = 0.5 * PI / n[1] as f64;
    let sim = Sim::build(Rmhd, IdealGas { gamma: GAMMA }, Spherical)
        .cells(n)
        .spacing([(R_HI - R_LO) / n[0] as f64, dth])
        .origin([R_LO, 0.0])
        .boundaries(Boundaries::per_axis([
            [BoundaryType::Driven(0), BoundaryType::Outflow],
            [pole, BoundaryType::Reflect],
        ]))
        .cfl(CFL)
        .allocate()
        .expect("spherical 2.5D construction")
        .set_initial(|[r, th]| probe_state(r, th))
        .seed_faces(|axis, [r, th]| match axis {
            0 => 0.1 + 0.02 * th,
            _ => 0.05 * th + 0.001 * r,
        })
        .build();
    let cfg = SourceConfig::from_json(&driven_json()).expect("parse the driven prescription");
    let built = build_boundary_dag(&cfg, &RMHD_SPEC).expect("lower the rmhd boundary dag");
    let (k, id) = Kern::new(GAMMA, CFL, 1.0, &sim.geom.allocated)
        .with_driven_boundary(built, cfg.params.clone());
    assert_eq!(id, 0);
    (sim, k)
}

/// the eight cell fields in prescription order.
fn cell(sim: &Sim, c: [isize; 2]) -> [f64; 8] {
    let p = &sim.fields.prim;
    let m = sim.fields.mhd.as_ref().expect("mhd fields");
    [
        *p.rho.view().at(c),
        *p.vel[0].view().at(c),
        *p.vel[1].view().at(c),
        *p.vel[2].view().at(c),
        *p.pre.as_ref().expect("adiabatic pressure").view().at(c),
        *m.bcell[0].view().at(c),
        *m.bcell[1].view().at(c),
        *m.bcell[2].view().at(c),
    ]
}

const NAMES: [&str; 8] = [
    "rho", "v_r", "v_theta", "v_phi", "pre", "B_r", "B_theta", "B_phi",
];

/// the parity of each field across a pole (theta and phi components odd) and across an
/// equatorial wall (theta components odd, everything else even).
const POLE_SIGN: [f64; 8] = [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0];
const WALL_SIGN: [f64; 8] = [1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0];

fn assert_mirror(sim: &Sim, ghost: [isize; 2], image: [isize; 2], sign: &[f64; 8], label: &str) {
    let g = cell(sim, ghost);
    let s = cell(sim, image);
    for k in 0..8 {
        // a vanishing component mirrors to a vanishing one whatever its sign bit.
        if g[k] == 0.0 && s[k] == 0.0 {
            continue;
        }
        assert_eq!(
            g[k].to_bits(),
            (sign[k] * s[k]).to_bits(),
            "{label}: {} at ghost {ghost:?} = {} vs {} x image {image:?} = {}",
            NAMES[k],
            g[k],
            sign[k],
            s[k]
        );
    }
}

/// every theta ghost row at interior radii, against its mirror image through the given face.
fn check_theta_band(sim: &Sim, hi: bool, sign: &[f64; 8], label: &str) -> usize {
    let (alloc, int) = (&sim.geom.allocated, &sim.geom.interior);
    let (jlo, jhi) = (int.spaces[1].lo, int.spaces[1].hi);
    let rows: Vec<isize> = if hi {
        (jhi..alloc.spaces[1].hi).collect()
    } else {
        (alloc.spaces[1].lo..jlo).collect()
    };
    let mut n = 0;
    for i in int.spaces[0].lo..int.spaces[0].hi {
        for &j in &rows {
            let jm = if hi { 2 * jhi - 1 - j } else { 2 * jlo - 1 - j };
            assert_mirror(sim, [i, j], [i, jm], sign, label);
            n += 1;
        }
    }
    n
}

#[test]
fn the_pole_flips_the_polar_and_azimuthal_components_and_the_wall_only_the_polar() {
    let (sim, k) = probe_sim([8, 8], BoundaryType::Axis);
    k.c2p(&sim);
    k.ghost_fill(&sim);
    let n_pole = check_theta_band(&sim, false, &POLE_SIGN, "axis pole");
    let n_wall = check_theta_band(&sim, true, &WALL_SIGN, "reflecting equator");
    assert!(
        n_pole >= 16 && n_wall >= 16,
        "the halo bands were read: {n_pole} + {n_wall} cells"
    );
    // the interior state feeding the check is the seeded asymmetric probe, so every odd
    // component is nonzero on the first row and the sign assertions were live.
    let int = &sim.geom.interior;
    let first = cell(&sim, [int.spaces[0].lo, int.spaces[1].lo]);
    for k in [2usize, 3, 6, 7] {
        assert!(
            first[k] != 0.0,
            "{} vanished on the first interior row; the parity check would be vacuous",
            NAMES[k]
        );
    }
}

#[test]
fn a_reflecting_pole_keeps_the_azimuthal_components_even() {
    // the control: the same grid with the pole declared a wall carries the wall parity there,
    // which is the even continuation of v_phi and B_phi the axis boundary exists to replace.
    let (sim, k) = probe_sim([8, 8], BoundaryType::Reflect);
    k.c2p(&sim);
    k.ghost_fill(&sim);
    check_theta_band(&sim, false, &WALL_SIGN, "reflecting pole");
}

#[test]
fn the_driven_face_owns_its_corner_with_the_pole() {
    // the driven pass runs after the lattice sweep over the whole radial ghost band, theta
    // ghosts included, so the corner holds the prescription at the corner cell's own
    // coordinate: for this odd-in-theta prescription that is also the axis image of the
    // driven column, and the two agree.
    let (sim, k) = probe_sim([8, 8], BoundaryType::Axis);
    k.c2p(&sim);
    k.ghost_fill(&sim);
    let (alloc, int) = (&sim.geom.allocated, &sim.geom.interior);
    let (ilo, jlo) = (int.spaces[0].lo, int.spaces[1].lo);
    let dth = 0.5 * PI / 8.0;
    let mut corners = 0;
    for i in alloc.spaces[0].lo..ilo {
        for j in alloc.spaces[1].lo..alloc.spaces[1].hi {
            let th = ((j - jlo) as f64 + 0.5) * dth;
            let want = driven_values(th);
            let got = cell(&sim, [i, j]);
            for k in 0..8 {
                assert!(
                    (got[k] - want[k]).abs() <= 1e-12 * (1.0 + want[k].abs()),
                    "{} at driven ghost {:?} (theta = {th}) = {} vs prescription {}",
                    NAMES[k],
                    [i, j],
                    got[k],
                    want[k]
                );
            }
            if j < jlo {
                corners += 1;
                // the corner cell is also the pole image of the driven column at the mirrored theta.
                assert_mirror(
                    &sim,
                    [i, j],
                    [i, 2 * jlo - 1 - j],
                    &POLE_SIGN,
                    "driven corner vs pole image",
                );
            }
        }
    }
    assert!(corners >= 4, "the corner block was read: {corners} cells");
}

// -----------------------------------------------------------------------------
// the balanced toroidal field
// -----------------------------------------------------------------------------

const B0: f64 = 0.2;
const P0: f64 = 1.0;
/// short enough that the waves the outflow radial walls shed (fast speed below 1.4) stay
/// within 0.07 of the walls, inside the margin the residual excludes.
const T_FINAL: f64 = 0.05;
const WALL_MARGIN: f64 = 0.2;

/// B_phi = B0 r sin(theta) carries the uniform axial current J_z = 2 B0, whose Lorentz force
/// -2 B0^2 w (w = r sin(theta) the cylindrical radius) is the gradient of -B0^2 w^2, so the gas
/// pressure p = p0 - B0^2 w^2 balances it exactly at rest.
fn balanced_state(r: f64, th: f64) -> MhdPrim<f64, 3> {
    let w = r * th.sin();
    MhdPrim::new(
        Prim::adiabatic(
            Density(1.0),
            Tensor::new([0.0, 0.0, 0.0]),
            Pressure(P0 - B0 * B0 * w * w),
        ),
        Tensor::new([0.0, 0.0, B0 * w]),
    )
}

fn balanced_sim(n: [usize; 2], pole: BoundaryType) -> (Sim, Kern) {
    let sim = Sim::build(Rmhd, IdealGas { gamma: GAMMA }, Spherical)
        .cells(n)
        .spacing([(R_HI - R_LO) / n[0] as f64, PI / n[1] as f64])
        .origin([R_LO, 0.0])
        .boundaries(Boundaries::per_axis([
            [BoundaryType::Outflow, BoundaryType::Outflow],
            [pole, pole],
        ]))
        .cfl(CFL)
        .allocate()
        .expect("spherical 2.5D construction")
        .set_initial(|[r, th]| balanced_state(r, th))
        .seed_faces_uniform([0.0, 0.0])
        .build();
    let k = Kern::new(GAMMA, CFL, 1.0, &sim.geom.allocated);
    (sim, k)
}

/// the largest speed over the interior cells whose radius is at least `WALL_MARGIN` from either
/// radial wall, split into the two rows touching the poles and the rest.
fn residual(sim: &Sim) -> (f64, f64) {
    let int = &sim.geom.interior;
    let (ilo, ihi) = (int.spaces[0].lo, int.spaces[0].hi);
    let (jlo, jhi) = (int.spaces[1].lo, int.spaces[1].hi);
    let dr = (R_HI - R_LO) / (ihi - ilo) as f64;
    let (mut pole, mut bulk) = (0.0f64, 0.0f64);
    for i in ilo..ihi {
        let r = R_LO + ((i - ilo) as f64 + 0.5) * dr;
        if r < R_LO + WALL_MARGIN || r > R_HI - WALL_MARGIN {
            continue;
        }
        for j in jlo..jhi {
            let c = [i, j];
            let v = (0..3)
                .map(|k| sim.fields.prim.vel[k].view().at(c).abs())
                .fold(0.0, f64::max);
            if j == jlo || j == jhi - 1 {
                pole = pole.max(v);
            } else {
                bulk = bulk.max(v);
            }
        }
    }
    (pole, bulk)
}

fn run_balanced(n: [usize; 2], pole: BoundaryType) -> (f64, f64) {
    let (mut sim, k) = balanced_sim(n, pole);
    evolve(&mut sim, &k, T_FINAL).expect("evolve");
    let (p, b) = residual(&sim);
    eprintln!("{pole:?} {n:?}: pole rows {p:.3e}, bulk {b:.3e}");
    assert!(p.is_finite() && b.is_finite());
    (p, b)
}

#[test]
fn the_balanced_toroidal_field_holds_at_rest_through_the_pole() {
    // a rest state reads the azimuthal ghost sign only through B_phi^2 in the face total
    // pressure, which is even either way, so this gate pins the geometric update through the
    // axis and its convergence; the sign itself is pinned by the ghost and induction-flux gates.
    // measured residual speeds at t = 0.05, pole rows / bulk: 1.61e-5 / 2.37e-5 at (16, 32)
    // and 8.91e-6 / 1.08e-5 at (32, 64), a first-order decay set by the point-sampled initial
    // state against the area-weighted balance. the bounds below hold that decay (0.65 per
    // refinement against 0.55 and 0.45 measured), the pole rows at the level of the bulk
    // (0.83 measured against 1.5), and a ceiling twice the fine measurement.
    let (pole_c, bulk_c) = run_balanced([16, 32], BoundaryType::Axis);
    let (pole_f, bulk_f) = run_balanced([32, 64], BoundaryType::Axis);
    assert!(
        pole_f <= 0.65 * pole_c,
        "pole rows: {pole_c:.3e} -> {pole_f:.3e} did not decay"
    );
    assert!(
        bulk_f <= 0.65 * bulk_c,
        "bulk: {bulk_c:.3e} -> {bulk_f:.3e} did not decay"
    );
    assert!(
        pole_f <= 1.5 * bulk_f,
        "the pole rows ({pole_f:.3e}) exceed the bulk ({bulk_f:.3e})"
    );
    assert!(
        pole_f < 2.0e-5,
        "the fine pole residual {pole_f:.3e} left its measured level"
    );
}

// -----------------------------------------------------------------------------
// the induction flux at the first polar interface
// -----------------------------------------------------------------------------

const V0: f64 = 0.1;

/// a meridional flow v_theta = V0 sin(theta) over the toroidal field B_phi = B0 r sin(theta):
/// both odd across the pole, so the induction flux v_theta B_phi is even and grows as
/// sin^2(theta) away from it.
fn sheared_state(r: f64, th: f64) -> MhdPrim<f64, 3> {
    MhdPrim::new(
        Prim::adiabatic(
            Density(1.0),
            Tensor::new([0.0, V0 * th.sin(), 0.0]),
            Pressure(1.0),
        ),
        Tensor::new([0.0, 0.0, B0 * r * th.sin()]),
    )
}

fn sheared_sim(n: [usize; 2], pole: BoundaryType) -> (Sim, Kern) {
    let sim = Sim::build(Rmhd, IdealGas { gamma: GAMMA }, Spherical)
        .cells(n)
        .spacing([(R_HI - R_LO) / n[0] as f64, 0.5 * PI / n[1] as f64])
        .origin([R_LO, 0.0])
        .boundaries(Boundaries::per_axis([
            [BoundaryType::Outflow, BoundaryType::Outflow],
            [pole, BoundaryType::Reflect],
        ]))
        .cfl(CFL)
        .allocate()
        .expect("spherical 2.5D construction")
        .set_initial(|[r, th]| sheared_state(r, th))
        .seed_faces_uniform([0.0, 0.0])
        .build();
    let k = Kern::new(GAMMA, CFL, 1.0, &sim.geom.allocated);
    (sim, k)
}

/// the B_phi flux through the theta faces of the middle radial column, from the pole face
/// outward, after one production flux sweep along theta, each divided by the pointwise flux
/// V0 B0 r_c sin^2(theta_face) of the seeded state. the pole face itself is returned raw in
/// slot zero, where the reference vanishes.
fn polar_bphi_flux_ratios(n: [usize; 2], pole: BoundaryType, faces: usize) -> Vec<f64> {
    let (sim, k) = sheared_sim(n, pole);
    k.c2p(&sim);
    k.ghost_fill(&sim);
    k.flux(&sim, 1);
    let int = &sim.geom.interior;
    let ncol = int.spaces[0].hi - int.spaces[0].lo;
    let i = int.spaces[0].lo + ncol / 2;
    let rc = R_LO + ((ncol / 2) as f64 + 0.5) * (R_HI - R_LO) / n[0] as f64;
    let dth = 0.5 * PI / n[1] as f64;
    let jlo = int.spaces[1].lo;
    let m = sim.fields.mhd.as_ref().expect("mhd fields");
    (0..faces as isize)
        .map(|f| {
            let flux = *m.bflux[1].f[2].view().at([i, jlo + f]);
            let th = f as f64 * dth;
            if f == 0 {
                flux
            } else {
                flux / (V0 * B0 * rc * th.sin() * th.sin())
            }
        })
        .collect()
}

#[test]
fn the_induction_flux_at_the_first_polar_interface_is_reconstructed_through_the_axis() {
    // the odd continuation gives the first interior cell its true slope, so the flux through
    // the first interior interface is built like every interface further out; the even wall
    // continuation flattens that cell and halves the flux there. measured on the (8, 16) and
    // (8, 32) grids: the axis ratios at faces 1..4 agree with one another to 0.3 percent,
    // while the wall's first-face ratio is 0.500 of the axis's, and the flux through the pole
    // face itself, which carries no area, is 6e-6 (axis) against 2e-3 (wall) of the first
    // interior face's.
    for n in [[8usize, 16usize], [8, 32]] {
        let axis = polar_bphi_flux_ratios(n, BoundaryType::Axis, 5);
        let wall = polar_bphi_flux_ratios(n, BoundaryType::Reflect, 5);
        let outer = axis[2..].iter().copied().fold(f64::NAN, f64::max);
        let inner = axis[2..].iter().copied().fold(f64::NAN, f64::min);
        assert!(
            (outer - inner) <= 0.01 * outer,
            "{n:?}: the outer faces disagree among themselves: {:?}",
            &axis[2..]
        );
        assert!(
            (axis[1] - outer).abs() <= 0.01 * outer,
            "{n:?}: the first polar interface ({}) is reconstructed unlike the outer faces ({outer})",
            axis[1]
        );
        assert!(
            wall[1] <= 0.6 * axis[1],
            "{n:?}: the wall continuation no longer flattens the first cell: wall {} vs axis {}",
            wall[1],
            axis[1]
        );
        for f in 2..5 {
            assert_eq!(
                axis[f].to_bits(),
                wall[f].to_bits(),
                "{n:?}: face {f} reads no ghost"
            );
        }
        let first = V0 * B0 * axis[1];
        assert!(
            axis[0].abs() <= 1e-4 * first.abs(),
            "{n:?}: pole-face flux {} against {first}",
            axis[0]
        );
    }
}
