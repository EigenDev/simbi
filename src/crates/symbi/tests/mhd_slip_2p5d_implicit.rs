// =============================================================================
// mhd_slip_2p5d_implicit.rs
//
// the implicit magnetic-slip midpoint on the mixed complex X = F_x + F_y + C_z: one block conjugate
// gradient with one step length, one conjugation coefficient, and one residual norm across the
// in-plane faces and the cell B_z. the gates: the mixed inner product counts each periodic physical
// face once and each cell once; the frozen operator is symmetric and semidefinite while the operand
// varies in all three components; the predictor heat contracts all three current components; the
// commit deposits dM_xy,interp + (B_z^1)^2/2 - (B_z^0)^2/2 plus the nonnegative midpoint heat and
// conserves the extended energy; a failed solve restores faces, cell B_z, interpolated in-plane B,
// and energy exactly; the complete M map is second order in each storage complex; the discretely
// force-free state is an exact fixed point of the implicit map; and the receipt's residual is the
// mixed residual of the iterate.
// =============================================================================

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_grid::Field;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, MagneticSpec, SurfaceSpec};
use symbi_substrate::regimes::mhd_substrate::{
    body_slip_emf_2p5d, magnetic_slip_apply_operator_mixed, magnetic_slip_commit,
    magnetic_slip_mixed_dot, magnetic_slip_solve,
};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;
type Cell = Field<f64, 2, HostMemory>;

const GAMMA: f64 = 5.0 / 3.0;
const BODY: [f64; 2] = [0.5, 0.5];
const R_BODY: f64 = 0.22;
const DT: f64 = 1e-3;

fn slip_spec() -> MagneticSpec {
    MagneticSpec::Slip {
        diffusivity_ratio: 2.0,
        shell_width: 0.12,
        slip_length_ratio: 1.5,
        field_regularization: 0.1,
        placement: 0.0,
    }
}

fn with_body(sim: Sim) -> Sim {
    sim.with_bodies(
        BodyCollection::new().add(
            Body::black_hole(0, Tensor::new(BODY), Tensor::zeros(), 1.0, R_BODY, 0.05, 1.0, 1.0, R_BODY)
                .with_surface(SurfaceSpec::Drain)
                .with_magnetic(slip_spec()),
        ),
    )
}

fn builder(n: usize) -> SimBuilder<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64, NeedsCells> {
    let dx = 1.0 / n as f64;
    SimStateGeneric::<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>::build(
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
}

// an empty field seeded afterwards by `seed`.
fn build_blank(n: usize) -> Sim {
    with_body(
        builder(n)
            .set_initial(|_| {
                MhdPrim::new(
                    Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)),
                    Tensor::new([0.0, 0.0, 0.0]),
                )
            })
            .seed_faces(|_, _| 0.0)
            .build(),
    )
}

// a smooth divergence-free in-plane field with a smooth out-of-plane component, seeded through the
// two-representation contract: the cell field is the exact average of its bounding faces, and
// B_z is the cell value, so the primed state is a fixed point of the face-to-cell projection.
fn build_smooth(n: usize) -> Sim {
    let dx = 1.0 / n as f64;
    let k = 2.0 * std::f64::consts::PI;
    let (a0, b0) = (0.3, 0.2);
    with_body(
        builder(n)
            .set_initial(move |[x, y]| {
                let bx = |xf: f64| -a0 * (k * xf).cos() * (k * y).sin();
                let by = |yf: f64| a0 * (k * x).sin() * (k * yf).cos();
                MhdPrim::new(
                    Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)),
                    Tensor::new([
                        0.5 * (bx(x - 0.5 * dx) + bx(x + 0.5 * dx)),
                        0.5 * (by(y - 0.5 * dx) + by(y + 0.5 * dx)),
                        b0 * (k * x).cos() * (k * y).cos(),
                    ]),
                )
            })
            .seed_faces(move |axis, [x, y]| match axis {
                0 => -a0 * (k * x).cos() * (k * y).sin(),
                _ => a0 * (k * x).sin() * (k * y).cos(),
            })
            .build(),
    )
}

fn n_of(sim: &Sim) -> isize {
    sim.geom.interior.spaces[0].size() as isize
}

fn rnd(c: [isize; 2], salt: u64) -> f64 {
    let mut h = salt.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    for (k, v) in c.iter().enumerate() {
        h ^= (*v as u64).wrapping_add(0x1234_5678 + 977 * k as u64);
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= h >> 29;
    }
    ((h >> 11) as f64 / (1u64 << 53) as f64) - 0.5
}

// seed every stored face and cell from periodic functions of the wrapped index, the in-plane cell
// field as the exact face average, and nrg = p/(gamma-1) + |B_cell|^2/2 at rest.
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

fn random_seed(sim: &Sim) {
    seed(sim, |d, c| rnd(c, d as u64 + 1), |c| rnd(c, 3));
}

// ---- state snapshots -----------------------------------------------------------------------------

fn faces_all(sim: &Sim) -> Vec<f64> {
    let m = sim.fields.mhd.as_ref().unwrap();
    (0..2).flat_map(|d| m.bface[d].domain().iter().map(move |c| *m.bface[d].at(c))).collect()
}
fn cells_all(sim: &Sim, d: usize) -> Vec<f64> {
    let m = sim.fields.mhd.as_ref().unwrap();
    m.bcell[d].domain().iter().map(|c| *m.bcell[d].at(c)).collect()
}
fn nrg_interior(sim: &Sim) -> Vec<f64> {
    let nrg = sim.fields.cons.nrg_field().unwrap();
    sim.geom.interior.iter().map(|c| *nrg.at(c)).collect()
}

// the mixed magnetic energy over the degrees of freedom: interior faces of each orientation and
// interior cells, one uniform volume.
fn mixed_energy(sim: &Sim) -> f64 {
    let m = sim.fields.mhd.as_ref().unwrap();
    let mut w = 0.0;
    for c in sim.geom.interior.iter() {
        for d in 0..2 {
            w += 0.5 * (*m.bface[d].at(c)).powi(2);
        }
        w += 0.5 * (*m.bcell[2].at(c)).powi(2);
    }
    w
}
// the in-plane face-to-cell defect delta_c = 1/8 sum_d (B_{d,+} - B_{d,-})^2 over the interior.
fn defect(sim: &Sim) -> f64 {
    let m = sim.fields.mhd.as_ref().unwrap();
    let mut e = 0.0;
    for c in sim.geom.interior.iter() {
        for d in 0..2 {
            let mut up = c;
            up[d] += 1;
            let g = *m.bface[d].at(up) - *m.bface[d].at(c);
            e += 0.125 * g * g;
        }
    }
    e
}
fn eint(sim: &Sim, c: [isize; 2]) -> f64 {
    let m = sim.fields.mhd.as_ref().unwrap();
    let nrg = *sim.fields.cons.nrg_field().unwrap().at(c);
    nrg - (0..3).map(|d| 0.5 * (*m.bcell[d].at(c)).powi(2)).sum::<f64>()
}

// ---- gates ---------------------------------------------------------------------------------------

#[test]
fn the_mixed_inner_product_counts_each_face_and_cell_once() {
    let sim = build_blank(13);
    let n = n_of(&sim) as f64;
    let m = sim.fields.mhd.as_ref().unwrap();
    let ws = m.magnetic_slip.as_ref().unwrap();
    let z = ws.z.as_ref().expect("a 2.5D workspace carries the cell complex");
    for d in 0..2 {
        for c in ws.iterate.b[d].domain().iter() {
            ws.iterate.b[d].set(c, 1.0);
        }
    }
    for c in z.iterate.domain().iter() {
        z.iterate.set(c, 1.0);
    }
    let dot = magnetic_slip_mixed_dot::<2, 3, HostMemory, f64>(&sim, &ws.iterate, Some(&z.iterate), &ws.iterate, Some(&z.iterate));
    assert!(dot == 3.0 * n * n, "the mixed inner product counts {dot} entries; expected 2 N^2 faces + N^2 cells = {}", 3.0 * n * n);
    let faces_only = magnetic_slip_mixed_dot::<2, 3, HostMemory, f64>(&sim, &ws.iterate, None, &ws.iterate, None);
    assert!(faces_only == 2.0 * n * n, "the face inner product counts {faces_only}; expected {}", 2.0 * n * n);
}

// the coefficient stays frozen on production bcell while the operand varies in every component.
#[test]
fn the_frozen_operator_is_symmetric_and_semidefinite_on_the_mixed_complex() {
    let sim = build_blank(13);
    random_seed(&sim);
    let m = sim.fields.mhd.as_ref().unwrap();
    let ws = m.magnetic_slip.as_ref().unwrap();
    let z = ws.z.as_ref().unwrap();
    let nrg = sim.fields.cons.nrg_field().unwrap();
    for c in sim.geom.interior.iter() {
        let m_cell: f64 = (0..3).map(|d| 0.5 * (*m.bcell[d].at(c)).powi(2)).sum();
        ws.gas_energy.set(c, *nrg.at(c) - m_cell);
    }
    let n = n_of(&sim);
    let fill = |faces: &symbi_sim::state::BfaceFields<2, HostMemory, f64>, cell: &Cell, salt: u64| {
        for d in 0..2 {
            for c in faces.b[d].domain().iter() {
                faces.b[d].set(c, rnd([c[0].rem_euclid(n), c[1].rem_euclid(n)], salt + d as u64));
            }
        }
        for c in cell.domain().iter() {
            cell.set(c, rnd([c[0].rem_euclid(n), c[1].rem_euclid(n)], salt + 7));
        }
    };
    fill(&ws.rhs, &z.rhs, 11);
    fill(&ws.residual, &z.residual, 23);
    magnetic_slip_apply_operator_mixed::<2, 3, HostMemory, f64>(&sim, GAMMA, &ws.rhs, Some(&z.rhs), &ws.direction, Some(&z.direction));
    magnetic_slip_apply_operator_mixed::<2, 3, HostMemory, f64>(&sim, GAMMA, &ws.residual, Some(&z.residual), &ws.operator_direction, Some(&z.operator_direction));
    let p_lq = magnetic_slip_mixed_dot::<2, 3, HostMemory, f64>(&sim, &ws.rhs, Some(&z.rhs), &ws.operator_direction, Some(&z.operator_direction));
    let lp_q = magnetic_slip_mixed_dot::<2, 3, HostMemory, f64>(&sim, &ws.direction, Some(&z.direction), &ws.residual, Some(&z.residual));
    let p_lp = magnetic_slip_mixed_dot::<2, 3, HostMemory, f64>(&sim, &ws.rhs, Some(&z.rhs), &ws.direction, Some(&z.direction));
    // the out-of-plane member of L p is nonzero, so the operand's B_z takes part.
    let lz: f64 = sim.geom.interior.iter().map(|c| (*z.direction.at(c)).abs()).fold(0.0, f64::max);
    assert!(lz > 1e-8, "the operator ignores the operand's B_z");
    let scale = p_lq.abs().max(lp_q.abs());
    assert!(scale > 1e-8, "vacuous: the operator annihilates the test vectors");
    assert!((p_lq - lp_q).abs() <= 1e-10 * scale, "not symmetric: <p, L q> = {p_lq:.12e}, <L p, q> = {lp_q:.12e}");
    assert!(p_lp >= -1e-12 * scale, "not positive semidefinite: <p, L p> = {p_lp:.3e}");
}

// the predictor lifts the gas energy by (dt/2) qdot^0 with qdot^0 = (R B^0) . F_q(B^0) contracting
// all three current components, F_q(B^0) read from the explicit pass on the same state.
#[test]
fn the_predictor_heat_contracts_all_three_current_components() {
    let sim = build_blank(13);
    random_seed(&sim);
    let m = sim.fields.mhd.as_ref().unwrap();
    let ws = m.magnetic_slip.as_ref().unwrap();
    let dx = sim.geom.dx[0];
    let inv_dx = 1.0 / dx;
    let e_g0: Vec<f64> = sim.geom.interior.iter().map(|c| eint(&sim, c)).collect();
    // the explicit pass at B^0 on scratch copies of B_z, leaving production untouched.
    let bz: Cell = Field::zeros(&sim.geom.allocated).unwrap();
    let out: Cell = Field::zeros(&sim.geom.allocated).unwrap();
    for c in bz.domain().iter() {
        bz.set(c, *m.bcell[2].at(c));
        out.set(c, *m.bcell[2].at(c));
    }
    for (c, e) in sim.geom.interior.iter().zip(&e_g0) {
        ws.gas_energy.set(c, *e);
    }
    for k in 0..2 {
        for c in m.efield.e[k].domain().iter() {
            m.efield.e[k].set(c, 0.0);
        }
    }
    body_slip_emf_2p5d::<2, 3, HostMemory, f64>(&sim, GAMMA, DT, &bz, &out);
    let fq = m.slip_quadrature.as_ref().unwrap();
    let qdot_ref: Vec<f64> = sim
        .geom
        .interior
        .iter()
        .map(|c| {
            let jx = (*bz.at([c[0], c[1] + 1]) - *bz.at([c[0], c[1] - 1])) * 0.5 * inv_dx;
            let jy = -(*bz.at([c[0] + 1, c[1]]) - *bz.at([c[0] - 1, c[1]])) * 0.5 * inv_dx;
            let curl = |e: [isize; 2]| {
                (*m.bface[1].at(e) - *m.bface[1].at([e[0] - 1, e[1]])) * inv_dx
                    - (*m.bface[0].at(e) - *m.bface[0].at([e[0], e[1] - 1])) * inv_dx
            };
            let jz = 0.25 * (curl(c) + curl([c[0] + 1, c[1]]) + curl([c[0], c[1] + 1]) + curl([c[0] + 1, c[1] + 1]));
            jx * *fq[0].at(c) + jy * *fq[1].at(c) + jz * *fq[2].at(c)
        })
        .collect();
    // the explicit pass moved no production state; a fresh solve now stages e_g* = e_g^0 + dt/2 qdot^0.
    let receipt = magnetic_slip_solve::<2, 3, HostMemory, f64>(&sim, DT, GAMMA, 1e-12, 500);
    assert!(receipt.converged);
    let scale = qdot_ref.iter().fold(0.0_f64, |a, q| a.max(q.abs()));
    assert!(scale > 1e-8, "vacuous: no dissipation");
    let mut worst = 0.0_f64;
    for ((c, e0), q) in sim.geom.interior.iter().zip(&e_g0).zip(&qdot_ref) {
        let staged = *ws.gas_energy.at(c);
        worst = worst.max((staged - (e0 + 0.5 * DT * q)).abs());
        assert!(*q >= -1e-12 * scale, "negative predicted heat at {c:?}: {q:.3e}");
    }
    assert!(worst <= 1e-12 * DT * scale, "the staged midpoint gas energy departs from e_g^0 + dt/2 qdot^0 by {worst:.3e}");
}

// the commit deposits dM_cell = dM_xy,interp + (B_z^1)^2/2 - (B_z^0)^2/2 plus the nonnegative
// midpoint heat: the gas heat equals the mixed face-Hodge loss Q_h, cell by cell nonnegative, and
// the extended energy sum(E + delta_xy) is conserved.
#[test]
fn the_commit_thermalizes_the_mixed_dissipation_and_conserves_the_extended_energy() {
    let sim = build_blank(13);
    random_seed(&sim);
    let eh_before: f64 = nrg_interior(&sim).iter().sum::<f64>() + defect(&sim);
    let w_before = mixed_energy(&sim);
    let eint_before: Vec<f64> = sim.geom.interior.iter().map(|c| eint(&sim, c)).collect();
    let receipt = magnetic_slip_solve::<2, 3, HostMemory, f64>(&sim, DT, GAMMA, 1e-13, 500);
    assert!(receipt.converged, "CG did not converge: {} iterations, residual {:.3e}", receipt.iterations, receipt.final_residual_norm);
    magnetic_slip_commit::<2, 3, HostMemory, f64>(&sim, DT, GAMMA);
    let eh_after: f64 = nrg_interior(&sim).iter().sum::<f64>() + defect(&sim);
    let q_h = w_before - mixed_energy(&sim);
    let eint_after: Vec<f64> = sim.geom.interior.iter().map(|c| eint(&sim, c)).collect();
    let heat: f64 = eint_after.iter().zip(&eint_before).map(|(a, b)| a - b).sum();
    let min_cell_heat = eint_after.iter().zip(&eint_before).map(|(a, b)| a - b).fold(f64::INFINITY, f64::min);
    let scale = q_h.abs().max(1.0);
    assert!(q_h > 1e-4, "vacuous: no dissipation ({q_h:.3e})");
    assert!((heat - q_h).abs() < 1e-8 * scale, "gas heat != Q_h: {heat:.9e} vs {q_h:.9e}");
    assert!(min_cell_heat >= -1e-12 * scale, "a cell cooled: min cell heat {min_cell_heat:.3e}");
    assert!((eh_after - eh_before).abs() < 1e-9 * eh_before.abs(), "extended energy drift {:.3e}", eh_after - eh_before);
}

#[test]
fn a_failed_solve_restores_every_stored_field_exactly() {
    let sim = build_blank(13);
    random_seed(&sim);
    // one projecting solve so the interpolated in-plane cell field is the exact face average.
    assert!(magnetic_slip_solve::<2, 3, HostMemory, f64>(&sim, DT, GAMMA, 1e-12, 500).converged);
    let faces = faces_all(&sim);
    let cells: Vec<Vec<f64>> = (0..3).map(|d| cells_all(&sim, d)).collect();
    let energy = nrg_interior(&sim);
    let receipt = magnetic_slip_solve::<2, 3, HostMemory, f64>(&sim, DT, GAMMA, 1e-12, 0);
    assert!(!receipt.converged, "a zero-iteration solve reported convergence");
    assert!(faces == faces_all(&sim), "a failed solve moved a stored face");
    for d in 0..3 {
        assert!(cells[d] == cells_all(&sim, d), "a failed solve moved cell component {d}");
    }
    assert!(energy == nrg_interior(&sim), "a failed solve moved the total energy");
}

#[test]
fn the_discretely_force_free_state_is_an_exact_fixed_point_of_the_implicit_map() {
    let sim = build_blank(16);
    let k = 2.0 * std::f64::consts::PI;
    let dx = sim.geom.dx[0];
    seed(
        &sim,
        move |d, c| if d == 1 { (k * (c[0] as f64 + 0.5) * dx).cos() } else { 0.0 },
        move |c| (k * (c[0] as f64 + 0.5) * dx).sin(),
    );
    assert!(magnetic_slip_solve::<2, 3, HostMemory, f64>(&sim, DT, GAMMA, 1e-12, 500).converged);
    let faces = faces_all(&sim);
    let bz = cells_all(&sim, 2);
    let energy = nrg_interior(&sim);
    let receipt = magnetic_slip_solve::<2, 3, HostMemory, f64>(&sim, DT, GAMMA, 1e-12, 500);
    assert!(receipt.converged);
    magnetic_slip_commit::<2, 3, HostMemory, f64>(&sim, DT, GAMMA);
    let worst = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max);
    assert!(worst(&faces, &faces_all(&sim)) <= 1e-13, "the implicit map moved a face of the force-free state");
    assert!(worst(&bz, &cells_all(&sim, 2)) <= 1e-13, "the implicit map moved B_z of the force-free state");
    assert!(worst(&energy, &nrg_interior(&sim)) <= 1e-13, "the implicit map heated the force-free state");
}

// the receipt's residual is the mixed residual |rhs - (I + dt/2 L*) x| of the iterate, recomputed
// here through the public operator with the coefficient re-frozen on the predictor.
#[test]
fn the_receipt_reports_the_mixed_residual() {
    let sim = build_blank(13);
    random_seed(&sim);
    let receipt = magnetic_slip_solve::<2, 3, HostMemory, f64>(&sim, DT, GAMMA, 1e-12, 1);
    assert!(!receipt.converged && receipt.iterations == 1);
    let m = sim.fields.mhd.as_ref().unwrap();
    let ws = m.magnetic_slip.as_ref().unwrap();
    let z = ws.z.as_ref().unwrap();
    // re-freeze the coefficient on the predictor state the solve used.
    for d in 0..3 {
        for c in sim.geom.interior.iter() {
            m.bcell[d].set(c, *ws.frozen_bcell.b[d].at(c));
        }
    }
    // rhs = (I - dt/2 L*) B^0 and A x for the current iterate, through the public operator.
    magnetic_slip_apply_operator_mixed::<2, 3, HostMemory, f64>(&sim, GAMMA, &ws.input, Some(&z.input), &ws.direction, Some(&z.direction));
    magnetic_slip_apply_operator_mixed::<2, 3, HostMemory, f64>(&sim, GAMMA, &ws.iterate, Some(&z.iterate), &ws.operator_direction, Some(&z.operator_direction));
    let (mut r2, mut rz2) = (0.0, 0.0);
    for c in sim.geom.interior.iter() {
        for d in 0..2 {
            let rhs = *ws.input.b[d].at(c) - 0.5 * DT * *ws.direction.b[d].at(c);
            let ax = *ws.iterate.b[d].at(c) + 0.5 * DT * *ws.operator_direction.b[d].at(c);
            r2 += (rhs - ax).powi(2);
        }
        let rhs = *z.input.at(c) - 0.5 * DT * *z.direction.at(c);
        let ax = *z.iterate.at(c) + 0.5 * DT * *z.operator_direction.at(c);
        rz2 += (rhs - ax).powi(2);
    }
    let mixed = (r2 + rz2).sqrt();
    assert!(rz2.sqrt() > 1e-6 * mixed, "vacuous: the out-of-plane residual vanishes");
    assert!(
        (mixed - receipt.final_residual_norm).abs() <= 1e-10 * mixed,
        "the receipt residual {:.12e} is not the mixed residual {mixed:.12e} (face part {:.3e}, cell part {:.3e})",
        receipt.final_residual_norm,
        r2.sqrt(),
        rz2.sqrt()
    );
}

// the complete production M map (solve, commit, primitive recovery, ghost fill) is second order in
// time in each storage complex: the in-plane faces, the cell B_z, the total energy, the pressure.
#[test]
fn the_production_m_map_is_second_order_in_each_storage_complex() {
    let run = |dt: f64, nsteps: usize| -> [Vec<f64>; 4] {
        let sim = build_smooth(16);
        let sub = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.3, 1.0, &sim.geom.allocated);
        let mut hier = Hierarchy::single(sim, sub);
        hier.prime();
        for _ in 0..nsteps {
            assert!(!hier.magnetic_slip_map(0, dt), "the M map diverged at dt = {dt}");
        }
        let (bf, bc, nrg, pre) = hier.slip_state_snapshots(0);
        let ncells = nrg.len();
        let bz = bc[2 * ncells..3 * ncells].to_vec();
        [bf, bz, nrg, pre]
    };
    let l2 = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
    let dt = 4.0e-3;
    let (u1, u2, u3) = (run(dt, 4), run(dt / 2.0, 8), run(dt / 4.0, 16));
    for (name, i) in [("bface", 0), ("bz", 1), ("energy", 2), ("pressure", 3)] {
        let (e1, e2) = (l2(&u1[i], &u2[i]), l2(&u2[i], &u3[i]));
        let ratio = e1 / e2.max(1e-300);
        println!("2.5D M map {name:>8}: ratio {ratio:.2}");
        assert!(e2 > 1e-14, "vacuous M-map measurement in {name}");
        assert!(ratio > 3.4, "the 2.5D M map is not second order in {name}: ratio {ratio:.2}");
    }
}
