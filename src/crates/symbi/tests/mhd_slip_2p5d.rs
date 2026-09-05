// =============================================================================
// mhd_slip_2p5d.rs
//
// the explicit 2.5D magnetic-slip operator on the mixed complex X = F_x + F_y + C_z (in-plane
// staggered faces, cell-centered B_z), periodic cartesian x-y grid. the operator is L = R* A R with
//   R B = (D_y B_z, -D_x B_z, G_z(B_x, B_y)),
// D_x, D_y central differences of the cell B_z, G_z the 1/4-weighted gather of the four corner
// z-edge currents, and R* its exact transpose under the plain face-plus-cell inner product (every
// weight is the one uniform cell volume): the z-component scatters to the corner EMF the CT curl
// consumes, the in-plane components return through B_z -= dt (D_x F_y - D_y F_x) in flux form.
// the gates: the cell pass against a host reference, the 3D gather reducing to R on a z-invariant
// state, the mixed adjoint channel by channel, the uniform no-op, div B under the in-plane
// transport, the explicit energy identity, convergence to a no-op on a discretely force-free field,
// and symmetry with semidefiniteness of the frozen operator on the mixed complex.
// =============================================================================

use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_grid::Field;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::drain::{local_drain_rate, sound_speed_from_cons};
use symbi_ib::magnetic_slip::{chi_shell, slip_apply, slip_coefficient};
use symbi_ib::{Body, BodyCollection, MagneticSpec, SurfaceSpec};
use symbi_substrate::regimes::mhd_substrate::{body_slip_emf_2p5d, ct_curl};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;
type Mhd = symbi_sim::state::MhdStaggeredFields<2, 3, HostMemory, f64>;
type Cell = Field<f64, 2, HostMemory>;

const GAMMA: f64 = 5.0 / 3.0;
const BODY: [f64; 2] = [0.5, 0.5];
const R_BODY: f64 = 0.22;
const D_B: f64 = 2.0;
const W: f64 = 0.12;
const ELL_RATIO: f64 = 1.5;
const B0: f64 = 0.1;
const PLACEMENT: f64 = 0.0;
const DT: f64 = 1e-3;

fn slip_spec() -> MagneticSpec {
    MagneticSpec::Slip {
        diffusivity_ratio: D_B,
        shell_width: W,
        slip_length_ratio: ELL_RATIO,
        field_regularization: B0,
        placement: PLACEMENT,
    }
}

fn build_sim(n: usize) -> Sim {
    let dx = 1.0 / n as f64;
    let sim = SimStateGeneric::<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>::build(
        NewtonianMhd,
        IdealGas { gamma: GAMMA },
        Cartesian,
    )
    .cells([n, n])
    .origin([0.0, 0.0])
    .spacing([dx, dx])
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .cfl(0.3)
    .allocate()
    .expect("2.5D sim construction")
    .set_initial(|_| {
        MhdPrim::new(
            Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)),
            Tensor::new([0.0, 0.0, 0.0]),
        )
    })
    .seed_faces(|_, _| 0.0)
    .build();
    sim.with_bodies(
        BodyCollection::new().add(
            Body::black_hole(0, Tensor::new(BODY), Tensor::zeros(), 1.0, R_BODY, 0.05, 1.0, 1.0, R_BODY)
                .with_surface(SurfaceSpec::Drain)
                .with_magnetic(slip_spec()),
        ),
    )
}

fn n_of(sim: &Sim) -> isize {
    sim.geom.interior.spaces[0].size() as isize
}

// a deterministic pseudo-random value in [-0.5, 0.5] of a wrapped 2D index, asymmetric under
// axis reflection so no stencil cancellation hides a transposition error.
fn rnd(c: [isize; 2], salt: u64) -> f64 {
    let mut h = salt.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    for (k, v) in c.iter().enumerate() {
        h ^= (*v as u64).wrapping_add(0x1234_5678 + 977 * k as u64);
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= h >> 29;
    }
    ((h >> 11) as f64 / (1u64 << 53) as f64) - 0.5
}

// seed every stored face, cell field, and the total energy from periodic analytic functions:
// faces `f(d, c)` (d = 0, 1) and the cell `bz(c)` at wrapped indices, the in-plane cell field as
// the exact average of its bounding faces, and nrg = p/(gamma-1) + |B_cell|^2/2 at rest.
fn seed(sim: &Sim, f: impl Fn(usize, [isize; 2]) -> f64, bz: impl Fn([isize; 2]) -> f64) {
    let n = n_of(sim);
    let wrap = |c: [isize; 2]| [c[0].rem_euclid(n), c[1].rem_euclid(n)];
    let m = sim.fields.mhd.as_ref().unwrap();
    for d in 0..2 {
        for c in m.bface[d].domain().iter() {
            m.bface[d].set(c, f(d, wrap(c)));
        }
    }
    for c in m.bcell[2].domain().iter() {
        m.bcell[2].set(c, bz(wrap(c)));
        for d in 0..2 {
            let mut up = c;
            up[d] += 1;
            m.bcell[d].set(c, 0.5 * (f(d, wrap(c)) + f(d, wrap(up))));
        }
    }
    let nrg = sim.fields.cons.nrg_field().unwrap();
    for c in sim.geom.interior.iter() {
        let b_sq: f64 = (0..3).map(|d| (*m.bcell[d].at(c)).powi(2)).sum();
        nrg.set(c, 1.0 / (GAMMA - 1.0) + 0.5 * b_sq);
    }
}

fn zero_efield(m: &Mhd) {
    for f in m.efield.e.iter() {
        for c in f.domain().iter() {
            f.set(c, 0.0);
        }
    }
}

fn cell_zeros(sim: &Sim) -> Cell {
    Field::zeros(&sim.geom.allocated).unwrap()
}

fn copy_cell(src: &Cell, dst: &Cell) {
    for c in src.domain().iter() {
        dst.set(c, *src.at(c));
    }
}

// the explicit operator's producer: stage the gas internal energy the coefficient reads, then the
// cell pass, the corner scatter into a cleared EMF, and the B_z update on `bz_out` (which holds
// the operand's B_z on entry). the caller applies the CT curl for the in-plane update.
fn slip_2p5d(sim: &Sim, dt: f64, bz_op: &Cell, bz_out: &Cell) {
    let m = sim.fields.mhd.as_ref().unwrap();
    let ws = m.magnetic_slip.as_ref().expect("slip workspace");
    let nrg = sim.fields.cons.nrg_field().unwrap();
    for c in sim.geom.interior.iter() {
        let den = *sim.fields.cons.den.at(c);
        let mom_sq: f64 = (0..3).map(|k| (*sim.fields.cons.mom[k].at(c)).powi(2)).sum();
        let m_cell: f64 = (0..3).map(|d| 0.5 * (*m.bcell[d].at(c)).powi(2)).sum();
        ws.gas_energy.set(c, *nrg.at(c) - 0.5 * mom_sq / den - m_cell);
    }
    zero_efield(m);
    body_slip_emf_2p5d::<2, 3, HostMemory, f64>(sim, GAMMA, dt, bz_op, bz_out);
}

// ---- host references ---------------------------------------------------------------------------

fn bface(m: &Mhd, d: usize, c: [isize; 2]) -> f64 {
    *m.bface[d].at(c)
}

// the corner z-edge current at the lower-left corner of cell `c`, from backward face differences.
fn curl_edge(m: &Mhd, c: [isize; 2], inv_dx: f64) -> f64 {
    (bface(m, 1, c) - bface(m, 1, [c[0] - 1, c[1]])) * inv_dx
        - (bface(m, 0, c) - bface(m, 0, [c[0], c[1] - 1])) * inv_dx
}

// R B at cell `c`: (D_y B_z, -D_x B_z, G_z).
fn r_ref(m: &Mhd, bz: &Cell, c: [isize; 2], inv_dx: f64) -> [f64; 3] {
    let jx = (*bz.at([c[0], c[1] + 1]) - *bz.at([c[0], c[1] - 1])) * 0.5 * inv_dx;
    let jy = -(*bz.at([c[0] + 1, c[1]]) - *bz.at([c[0] - 1, c[1]])) * 0.5 * inv_dx;
    let jz = 0.25
        * (curl_edge(m, c, inv_dx)
            + curl_edge(m, [c[0] + 1, c[1]], inv_dx)
            + curl_edge(m, [c[0], c[1] + 1], inv_dx)
            + curl_edge(m, [c[0] + 1, c[1] + 1], inv_dx));
    [jx, jy, jz]
}

// the frozen dyad coefficient a_B chi_B at cell `c`, exactly as the kernel forms it.
fn coeff_ref(sim: &Sim, m: &Mhd, c: [isize; 2]) -> f64 {
    let dx = sim.geom.dx[0];
    let b_sq: f64 = (0..3).map(|d| (*m.bcell[d].at(c)).powi(2)).sum();
    let den = *sim.fields.cons.den.at(c);
    let nrg = *sim.fields.cons.nrg_field().unwrap().at(c);
    let cs = sound_speed_from_cons(den, 0.0, nrg - 0.5 * b_sq, GAMMA);
    let lam = local_drain_rate(cs, b_sq, den, 1.0 / dx, 1.0, R_BODY);
    let x: [f64; 2] = std::array::from_fn(|a| sim.geom.x_lo[a] + (c[a] as f64 + 0.5) * dx);
    let dist = ((x[0] - BODY[0]).powi(2) + (x[1] - BODY[1]).powi(2)).sqrt();
    let chi_b = chi_shell(dist - R_BODY, W, PLACEMENT);
    slip_coefficient(ELL_RATIO * W, b_sq, B0, D_B, 1.0 / lam) * chi_b
}

fn f_ref(sim: &Sim, m: &Mhd, bz: &Cell, c: [isize; 2]) -> [f64; 3] {
    let inv_dx = 1.0 / sim.geom.dx[0];
    let b_q = Tensor::new([*m.bcell[0].at(c), *m.bcell[1].at(c), *m.bcell[2].at(c)]);
    let j = r_ref(m, bz, c, inv_dx);
    let f = slip_apply(coeff_ref(sim, m, c), &b_q, &Tensor::new(j));
    [f[0], f[1], f[2]]
}

fn interior_max<F: Fn([isize; 2]) -> f64>(sim: &Sim, f: F) -> f64 {
    sim.geom.interior.iter().map(f).fold(0.0_f64, f64::max)
}

// ---- gates ---------------------------------------------------------------------------------------

#[test]
fn the_2p5d_cell_pass_matches_the_reference_elementwise() {
    let sim = build_sim(13);
    seed(&sim, |d, c| rnd(c, d as u64 + 1), |c| rnd(c, 3));
    let m = sim.fields.mhd.as_ref().unwrap();
    let bz = cell_zeros(&sim);
    copy_cell(&m.bcell[2], &bz);
    let out = cell_zeros(&sim);
    copy_cell(&bz, &out);
    slip_2p5d(&sim, DT, &bz, &out);
    let fq = m.slip_quadrature.as_ref().unwrap();
    let scale = interior_max(&sim, |c| f_ref(&sim, m, &bz, c).iter().fold(0.0_f64, |a, x| a.max(x.abs())));
    assert!(scale > 1e-6, "vacuous: the reference quadrature vanishes");
    let worst = interior_max(&sim, |c| {
        let r = f_ref(&sim, m, &bz, c);
        (0..3).map(|d| (*fq[d].at(c) - r[d]).abs()).fold(0.0_f64, f64::max)
    });
    assert!(worst <= 1e-12 * scale, "cell pass departs from the reference by {worst:.3e} (scale {scale:.3e})");
}

// the 3D gather on a z-invariant state, with the 2.5D cell B_z identified with a z-uniform set
// of 3D z-faces, reduces exactly to (D_y B_z, -D_x B_z, G_z): the x- and y-edge curls lose their
// d/dz terms and the four-corner average of backward differences collapses to the central
// difference.
#[test]
fn the_3d_gather_on_a_z_invariant_state_is_the_2p5d_current() {
    let sim = build_sim(13);
    seed(&sim, |d, c| rnd(c, d as u64 + 1), |c| rnd(c, 3));
    let m = sim.fields.mhd.as_ref().unwrap();
    let bz = cell_zeros(&sim);
    copy_cell(&m.bcell[2], &bz);
    let inv_dx = 1.0 / sim.geom.dx[0];
    // the extruded 3D staggered field: in-plane faces copied along z, z-faces carrying the cell B_z.
    let bf3 = |comp: usize, c: [isize; 3]| -> f64 {
        match comp {
            0 | 1 => bface(m, comp, [c[0], c[1]]),
            _ => *bz.at([c[0], c[1]]),
        }
    };
    let curl3 = |d: usize, e: [isize; 3]| -> f64 {
        let p1 = (d + 1) % 3;
        let p2 = (d + 2) % 3;
        let mut m1 = e;
        m1[p1] -= 1;
        let mut m2 = e;
        m2[p2] -= 1;
        (bf3(p2, e) - bf3(p2, m1)) * inv_dx - (bf3(p1, e) - bf3(p1, m2)) * inv_dx
    };
    let gather3 = |d: usize, c: [isize; 3]| -> f64 {
        let p1 = (d + 1) % 3;
        let p2 = (d + 2) % 3;
        let mut e1 = c;
        e1[p1] += 1;
        let mut e2 = c;
        e2[p2] += 1;
        let mut e12 = c;
        e12[p1] += 1;
        e12[p2] += 1;
        0.25 * (curl3(d, c) + curl3(d, e1) + curl3(d, e2) + curl3(d, e12))
    };
    let mut scale = 0.0_f64;
    let mut worst = 0.0_f64;
    for c in sim.geom.interior.iter() {
        let r = r_ref(m, &bz, c, inv_dx);
        for k in 0..2 {
            for d in 0..3 {
                let g = gather3(d, [c[0], c[1], k]);
                scale = scale.max(r[d].abs());
                worst = worst.max((g - r[d]).abs());
            }
        }
    }
    assert!(scale > 1e-6, "vacuous: the current vanishes");
    assert!(worst <= 1e-13 * scale, "the 3D gather departs from the 2.5D current by {worst:.3e}");
}

// the mixed adjoint, channel by channel: <(D_y B_z, -D_x B_z), (F_x, F_y)> = <B_z, D_x F_y - D_y F_x>
// with the right side read from the kernel's B_z update, and <G_z B, F_z> = <curl B, scatter F_z>
// with the right side read from the kernel's corner EMF. the sum is the quadratic form <R B, A R B>,
// nonnegative.
#[test]
fn the_mixed_adjoint_closes_channel_by_channel_on_the_periodic_domain() {
    let sim = build_sim(13);
    seed(&sim, |d, c| rnd(c, d as u64 + 1), |c| rnd(c, 3));
    let m = sim.fields.mhd.as_ref().unwrap();
    let bz = cell_zeros(&sim);
    copy_cell(&m.bcell[2], &bz);
    let out = cell_zeros(&sim);
    copy_cell(&bz, &out);
    let dt = 1.0;
    slip_2p5d(&sim, dt, &bz, &out);
    let fq = m.slip_quadrature.as_ref().unwrap();
    let inv_dx = 1.0 / sim.geom.dx[0];
    let (mut cell_xy, mut cell_z, mut bz_form, mut edge_form) = (0.0, 0.0, 0.0, 0.0);
    for c in sim.geom.interior.iter() {
        let r = r_ref(m, &bz, c, inv_dx);
        cell_xy += r[0] * *fq[0].at(c) + r[1] * *fq[1].at(c);
        cell_z += r[2] * *fq[2].at(c);
        // (R* F)_z = (B_z - B_z') / dt from the kernel's update.
        bz_form += *bz.at(c) * (*bz.at(c) - *out.at(c)) / dt;
        edge_form += curl_edge(m, c, inv_dx) * *m.efield[0].at(c);
    }
    assert!(cell_xy.abs() > 1e-8 && cell_z.abs() > 1e-8, "vacuous: a channel's form vanishes");
    assert!(
        (cell_xy - bz_form).abs() <= 1e-10 * cell_xy.abs(),
        "the out-of-plane channel is not adjoint: <R_xy B, F_xy> = {cell_xy:.12e}, <B_z, R*_z F> = {bz_form:.12e}"
    );
    assert!(
        (cell_z - edge_form).abs() <= 1e-10 * cell_z.abs(),
        "the in-plane channel is not adjoint: <G_z B, F_z> = {cell_z:.12e}, <curl B, E_z> = {edge_form:.12e}"
    );
    assert!(cell_xy + cell_z >= -1e-12, "the quadratic form is negative: {}", cell_xy + cell_z);
}

#[test]
fn a_uniform_field_is_an_exact_no_op() {
    let sim = build_sim(13);
    seed(&sim, |d, _| [0.3, -0.2][d], |_| 0.45);
    let m = sim.fields.mhd.as_ref().unwrap();
    let bz = cell_zeros(&sim);
    copy_cell(&m.bcell[2], &bz);
    let out = cell_zeros(&sim);
    copy_cell(&bz, &out);
    let faces_before: Vec<f64> = (0..2).flat_map(|d| m.bface[d].domain().iter().map(move |c| bface(m, d, c))).collect();
    slip_2p5d(&sim, DT, &bz, &out);
    ct_curl::<2, 3, HostMemory, f64>(&sim, DT);
    let faces_after: Vec<f64> = (0..2).flat_map(|d| m.bface[d].domain().iter().map(move |c| bface(m, d, c))).collect();
    assert!(faces_before == faces_after, "a uniform field moved an in-plane face");
    for c in sim.geom.interior.iter() {
        assert!(*out.at(c) == *bz.at(c), "a uniform field moved B_z at {c:?}");
        assert!(*m.efield[0].at(c) == 0.0, "a uniform field produced an EMF at {c:?}");
    }
}

#[test]
fn the_in_plane_transport_keeps_the_divergence_at_roundoff() {
    let sim = build_sim(13);
    seed(&sim, |d, c| rnd(c, d as u64 + 1), |c| rnd(c, 3));
    let m = sim.fields.mhd.as_ref().unwrap();
    let inv_dx = 1.0 / sim.geom.dx[0];
    let div = |c: [isize; 2]| -> f64 {
        (bface(m, 0, [c[0] + 1, c[1]]) - bface(m, 0, c)) * inv_dx
            + (bface(m, 1, [c[0], c[1] + 1]) - bface(m, 1, c)) * inv_dx
    };
    let before: Vec<f64> = sim.geom.interior.iter().map(div).collect();
    let bz = cell_zeros(&sim);
    copy_cell(&m.bcell[2], &bz);
    let out = cell_zeros(&sim);
    copy_cell(&bz, &out);
    slip_2p5d(&sim, DT, &bz, &out);
    ct_curl::<2, 3, HostMemory, f64>(&sim, DT);
    let scale = interior_max(&sim, |c| bface(m, 0, c).abs()) * inv_dx;
    for (b, c) in before.iter().zip(sim.geom.interior.iter()) {
        assert!((div(c) - b).abs() <= 1e-12 * scale, "the in-plane transport changed div B at {c:?} by {:.3e}", div(c) - b);
    }
}

// the explicit Euler step B' = B - dt R* F on the mixed complex changes the mixed energy
// W = |B|^2/2 by dW = -dt <R B, F> + (dt^2/2) |R* F|^2, exact to roundoff.
#[test]
fn the_complete_explicit_energy_identity_closes_to_roundoff() {
    let sim = build_sim(13);
    seed(&sim, |d, c| rnd(c, d as u64 + 1), |c| rnd(c, 3));
    let m = sim.fields.mhd.as_ref().unwrap();
    let inv_dx = 1.0 / sim.geom.dx[0];
    let bz = cell_zeros(&sim);
    copy_cell(&m.bcell[2], &bz);
    let out = cell_zeros(&sim);
    copy_cell(&bz, &out);
    let faces_before: Vec<[f64; 2]> = sim.geom.interior.iter().map(|c| [bface(m, 0, c), bface(m, 1, c)]).collect();
    slip_2p5d(&sim, DT, &bz, &out);
    let fq = m.slip_quadrature.as_ref().unwrap();
    let mut form = 0.0;
    for c in sim.geom.interior.iter() {
        let r = r_ref(m, &bz, c, inv_dx);
        form += (0..3).map(|d| r[d] * *fq[d].at(c)).sum::<f64>();
    }
    ct_curl::<2, 3, HostMemory, f64>(&sim, DT);
    let (mut dw, mut dsq) = (0.0, 0.0);
    for (b, c) in faces_before.iter().zip(sim.geom.interior.iter()) {
        for d in 0..2 {
            let after = bface(m, d, c);
            dw += 0.5 * (after * after - b[d] * b[d]);
            dsq += (after - b[d]).powi(2);
        }
        let (z0, z1) = (*bz.at(c), *out.at(c));
        dw += 0.5 * (z1 * z1 - z0 * z0);
        dsq += (z1 - z0).powi(2);
    }
    let law = -DT * form + 0.5 * dsq;
    assert!(form > 1e-8, "vacuous: the dissipation form vanishes");
    assert!(
        (dw - law).abs() <= 1e-12 * (DT * form).abs(),
        "the explicit energy identity does not close: dW = {dw:.12e}, law = {law:.12e}"
    );
}

// the released fraction of the current inside the shell, |J_perp| / |J| read from F = coeff |B|^2
// J_perp where the shell is active, for a field seeded from face and cell functions of the index.
fn released_fraction(n: usize, f: impl Fn(usize, [isize; 2]) -> f64, bz_of: impl Fn([isize; 2]) -> f64) -> f64 {
    let sim = build_sim(n);
    seed(&sim, f, bz_of);
    let m = sim.fields.mhd.as_ref().unwrap();
    let bz = cell_zeros(&sim);
    copy_cell(&m.bcell[2], &bz);
    let out = cell_zeros(&sim);
    copy_cell(&bz, &out);
    slip_2p5d(&sim, DT, &bz, &out);
    let fq = m.slip_quadrature.as_ref().unwrap();
    let inv_dx = 1.0 / sim.geom.dx[0];
    let mut worst = 0.0_f64;
    let mut active = 0usize;
    for c in sim.geom.interior.iter() {
        let coeff = coeff_ref(&sim, m, c);
        if coeff < 1e-3 {
            continue;
        }
        active += 1;
        let b_sq: f64 = (0..3).map(|d| (*m.bcell[d].at(c)).powi(2)).sum();
        let j = r_ref(m, &bz, c, inv_dx);
        let jn = j.iter().map(|x| x * x).sum::<f64>().sqrt();
        let fn_ = (0..3).map(|d| (*fq[d].at(c)).powi(2)).sum::<f64>().sqrt();
        worst = worst.max(fn_ / (coeff * b_sq * jn.max(1e-300)));
    }
    assert!(active > 0, "the shell mask is inactive on every cell");
    worst
}

// B = (0, cos(k x), sin(k x)) is force-free in the continuum and, on this complex, discretely:
// the four-corner gather of backward differences of a y-face field that varies in x alone
// collapses to the same central difference that forms -D_x B_z, so both current components carry
// one sinc factor and R B is exactly parallel to B_q. the slip releases nothing, to roundoff.
#[test]
fn a_discretely_force_free_field_is_an_exact_no_op() {
    let n = 16usize;
    let k = 2.0 * std::f64::consts::PI;
    let dx = 1.0 / n as f64;
    let released = released_fraction(
        n,
        move |d, c| if d == 1 { (k * (c[0] as f64 + 0.5) * dx).cos() } else { 0.0 },
        move |c| (k * (c[0] as f64 + 0.5) * dx).sin(),
    );
    assert!(released <= 1e-13, "a discretely force-free field released a current fraction {released:.3e}");
}

// the linear force-free field B = (-d_y psi, d_x psi, alpha psi), psi = cos(k x) cos(k y),
// alpha = sqrt(2) k, has J = alpha B in the continuum; on the staggered complex its discrete
// current departs from alignment at O(dx^2), so the released fraction of the current inside the
// shell converges to zero at second order.
#[test]
fn a_force_free_field_converges_to_a_no_op_at_second_order() {
    let released = |n: usize| -> f64 {
        let k = 2.0 * std::f64::consts::PI;
        let alpha = 2.0_f64.sqrt() * k;
        let dx = 1.0 / n as f64;
        released_fraction(
            n,
            move |d, c| {
                // face d of cell c sits at the face position along d and the cell center across.
                let x = if d == 0 { c[0] as f64 * dx } else { (c[0] as f64 + 0.5) * dx };
                let y = if d == 1 { c[1] as f64 * dx } else { (c[1] as f64 + 0.5) * dx };
                match d {
                    0 => k * (k * x).cos() * (k * y).sin(),
                    _ => -k * (k * x).sin() * (k * y).cos(),
                }
            },
            move |c| {
                let (x, y) = ((c[0] as f64 + 0.5) * dx, (c[1] as f64 + 0.5) * dx);
                alpha * (k * x).cos() * (k * y).cos()
            },
        )
    };
    let (r16, r32, r64) = (released(16), released(32), released(64));
    assert!(r16 > 1e-6, "vacuous: no current is released at the coarsest grid");
    let (q1, q2) = (r16 / r32, r32 / r64);
    assert!(
        q1 > 3.0 && q2 > 3.0,
        "the released current does not vanish at second order: {r16:.3e} {r32:.3e} {r64:.3e} (ratios {q1:.2} {q2:.2})"
    );
}

// the frozen operator L p on the mixed complex: production faces <- p_xy, the operand's B_z <- p_z,
// the coefficient frozen on production bcell; L p = p - (p - R* A R p) read from the curl with
// dt = 1 and from the B_z update.
fn apply_frozen(sim: &Sim, p_faces: &[Vec<f64>; 2], p_z: &Cell, out_z: &Cell) -> ([Vec<f64>; 2], Vec<f64>) {
    let m = sim.fields.mhd.as_ref().unwrap();
    let n = n_of(sim);
    let wrap = |c: [isize; 2]| [c[0].rem_euclid(n), c[1].rem_euclid(n)];
    for d in 0..2 {
        for c in m.bface[d].domain().iter() {
            let w = wrap(c);
            m.bface[d].set(c, p_faces[d][(w[0] * n + w[1]) as usize]);
        }
    }
    copy_cell(p_z, out_z);
    zero_efield(m);
    body_slip_emf_2p5d::<2, 3, HostMemory, f64>(sim, GAMMA, 1.0, p_z, out_z);
    ct_curl::<2, 3, HostMemory, f64>(sim, 1.0);
    let lp_faces: [Vec<f64>; 2] = std::array::from_fn(|d| {
        sim.geom
            .interior
            .iter()
            .map(|c| p_faces[d][(c[0] * n + c[1]) as usize] - bface(m, d, c))
            .collect()
    });
    let lp_z: Vec<f64> = sim.geom.interior.iter().map(|c| *p_z.at(c) - *out_z.at(c)).collect();
    (lp_faces, lp_z)
}

#[test]
fn the_frozen_operator_is_symmetric_and_semidefinite_on_the_mixed_complex() {
    let sim = build_sim(13);
    seed(&sim, |d, c| rnd(c, d as u64 + 1), |c| rnd(c, 3));
    let m = sim.fields.mhd.as_ref().unwrap();
    let ws = m.magnetic_slip.as_ref().unwrap();
    let nrg = sim.fields.cons.nrg_field().unwrap();
    for c in sim.geom.interior.iter() {
        let m_cell: f64 = (0..3).map(|d| 0.5 * (*m.bcell[d].at(c)).powi(2)).sum();
        ws.gas_energy.set(c, *nrg.at(c) - m_cell);
    }
    let n = n_of(&sim);
    let vector = |salt: u64| -> ([Vec<f64>; 2], Cell) {
        let faces: [Vec<f64>; 2] = std::array::from_fn(|d| {
            (0..n * n).map(|i| rnd([i / n, i % n], salt + d as u64)).collect()
        });
        let z = cell_zeros(&sim);
        for c in z.domain().iter() {
            z.set(c, rnd([c[0].rem_euclid(n), c[1].rem_euclid(n)], salt + 7));
        }
        (faces, z)
    };
    let (pf, pz) = vector(11);
    let (qf, qz) = vector(23);
    let out = cell_zeros(&sim);
    let (lpf, lpz) = apply_frozen(&sim, &pf, &pz, &out);
    let (lqf, lqz) = apply_frozen(&sim, &qf, &qz, &out);
    let dot = |af: &[Vec<f64>; 2], az: &dyn Fn([isize; 2]) -> f64, bf: &[Vec<f64>; 2], bz: &[f64]| -> f64 {
        let mut s = 0.0;
        for (i, c) in sim.geom.interior.iter().enumerate() {
            let idx = (c[0] * n + c[1]) as usize;
            for d in 0..2 {
                s += af[d][idx] * bf[d][i];
            }
            s += az(c) * bz[i];
        }
        s
    };
    let p_lq = dot(&pf, &|c| *pz.at(c), &lqf, &lqz);
    let q_lp = dot(&qf, &|c| *qz.at(c), &lpf, &lpz);
    let p_lp = dot(&pf, &|c| *pz.at(c), &lpf, &lpz);
    let scale = p_lq.abs().max(q_lp.abs());
    assert!(scale > 1e-8, "vacuous: the operator annihilates the test vectors");
    assert!(
        (p_lq - q_lp).abs() <= 1e-10 * scale,
        "the frozen operator is not symmetric on the mixed complex: <p, L q> = {p_lq:.12e}, <L p, q> = {q_lp:.12e}"
    );
    assert!(p_lp >= -1e-12 * scale, "the frozen operator is not positive semidefinite: <p, L p> = {p_lp:.3e}");
}
