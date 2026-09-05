// =============================================================================
// mhd_slip_boundary_faces.rs
//
// the implicit magnetic-slip solve on a domain with non-periodic boundaries. the solve's degrees of
// freedom are the interior faces; the closing face of each component along its normal and the
// halo faces are derived storage under a periodic wrap and boundary values otherwise. the solve is
// transactional over every stored face, and the commit leaves the closing faces on an outflow
// boundary at their substep-input values while the interior faces move.
// =============================================================================

use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, MagneticSpec, SurfaceSpec};
use symbi_substrate::regimes::mhd_substrate::{magnetic_slip_commit, magnetic_slip_solve};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;

const N: usize = 12;
const GAMMA: f64 = 5.0 / 3.0;
const BODY: [f64; 3] = [0.5, 0.5, 0.5];
const R_BODY: f64 = 0.22;
const DT: f64 = 2.0e-3;

fn build(boundary: BoundaryType) -> Sim {
    let dx = 1.0 / N as f64;
    let k = 2.0 * std::f64::consts::PI;
    let a0 = 0.3;
    let sim = SimStateGeneric::<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>::build(
        NewtonianMhd,
        IdealGas { gamma: GAMMA },
        Cartesian,
    )
    .cells([N, N, N])
    .origin([0.0, 0.0, 0.0])
    .spacing([dx, dx, dx])
    .boundaries(Boundaries::uniform(boundary))
    .cfl(0.3)
    .allocate()
    .expect("sim construction")
    // the cell field is the exact average of its bounding faces, so the seeding deposits the
    // magnetic energy the first primitive recovery subtracts.
    .set_initial(move |[x, y, _z]| {
        let bx = |xf: f64| -a0 * (k * xf).cos() * (k * y).sin();
        let by = |yf: f64| a0 * (k * x).sin() * (k * yf).cos();
        MhdPrim::new(
            Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)),
            Tensor::new([
                0.5 * (bx(x - 0.5 * dx) + bx(x + 0.5 * dx)),
                0.5 * (by(y - 0.5 * dx) + by(y + 0.5 * dx)),
                0.0,
            ]),
        )
    })
    .seed_faces(move |axis, [x, y, _z]| match axis {
        0 => -a0 * (k * x).cos() * (k * y).sin(),
        1 => a0 * (k * x).sin() * (k * y).cos(),
        _ => 0.0,
    })
    .build();
    sim.with_bodies(
        BodyCollection::new().add(
            Body::black_hole(0, Tensor::new(BODY), Tensor::zeros(), 1.0, R_BODY, 0.05, 1.0, 1.0, R_BODY)
                .with_surface(SurfaceSpec::Drain)
                .with_magnetic(MagneticSpec::Slip {
                    diffusivity_ratio: 2.0,
                    shell_width: 0.12,
                    slip_length_ratio: 1.0,
                    field_regularization: 0.1,
                    placement: 0.0,
                }),
        ),
    )
}

// every stored face of every component, keyed (component, coordinate).
fn all_faces(sim: &Sim) -> Vec<((usize, [isize; 3]), f64)> {
    let m = sim.fields.mhd.as_ref().unwrap();
    let mut v = Vec::new();
    for d in 0..3 {
        for c in m.bface[d].domain().iter() {
            v.push(((d, c), *m.bface[d].at(c)));
        }
    }
    v
}

fn interior_cells(sim: &Sim, f: impl Fn([isize; 3]) -> f64) -> Vec<f64> {
    sim.geom.interior.iter().map(f).collect()
}

// the solve mutates neither a physical face, a halo face, a cell field, nor the total energy on a
// domain whose boundaries carry no wrap. the seeded cell field agrees with the face average to
// the last bit of the face coordinate arithmetic; one solve first projects it onto the exact
// average, and the transactional claim is read from the second solve, bit for bit.
#[test]
fn the_solve_is_transactional_over_every_stored_face_on_an_outflow_domain() {
    let sim = build(BoundaryType::Outflow);
    let sim = &sim;
    let receipt = magnetic_slip_solve::<3, 3, HostMemory, f64>(sim, DT, GAMMA, 1e-12, 500);
    assert!(receipt.converged, "the projecting solve did not converge");
    let faces_before = all_faces(sim);
    let m = sim.fields.mhd.as_ref().unwrap();
    let bcell_before = interior_cells(sim, |c| (0..3).map(|d| *m.bcell[d].at(c)).sum::<f64>());
    let nrg = sim.fields.cons.nrg_field().unwrap();
    let nrg_before = interior_cells(sim, |c| *nrg.at(c));

    let receipt = magnetic_slip_solve::<3, 3, HostMemory, f64>(sim, DT, GAMMA, 1e-12, 500);
    assert!(receipt.converged, "the solve did not converge on the outflow domain");

    let faces_after = all_faces(sim);
    for ((k, before), (_, after)) in faces_before.iter().zip(&faces_after) {
        assert!(before == after, "the solve moved stored face {k:?}: {before} -> {after}");
    }
    let bcell_after = interior_cells(sim, |c| (0..3).map(|d| *m.bcell[d].at(c)).sum::<f64>());
    assert!(bcell_before == bcell_after, "the solve moved the cell field");
    let nrg_after = interior_cells(sim, |c| *nrg.at(c));
    assert!(nrg_before == nrg_after, "the solve moved the total energy");
}

// the commit moves the interior faces through the operator and leaves every closing face on an
// outflow boundary, and every halo face there, at its substep-input value: those faces are
// boundary values, outside the solve's degrees of freedom.
#[test]
fn the_commit_leaves_outflow_boundary_faces_at_their_input_values() {
    let sim = build(BoundaryType::Outflow);
    let sim = &sim;
    let faces_before = all_faces(sim);
    let receipt = magnetic_slip_solve::<3, 3, HostMemory, f64>(sim, DT, GAMMA, 1e-12, 500);
    assert!(receipt.converged);
    magnetic_slip_commit::<3, 3, HostMemory, f64>(sim, DT, GAMMA);
    let faces_after = all_faces(sim);

    let interior = &sim.geom.interior;
    let mut moved_interior = 0usize;
    for ((k, before), (_, after)) in faces_before.iter().zip(&faces_after) {
        let (d, c) = *k;
        let inside = (0..3).all(|a| c[a] >= interior.spaces[a].lo && c[a] < interior.spaces[a].hi);
        if inside {
            if before != after {
                moved_interior += 1;
            }
        } else {
            assert!(
                before == after,
                "the commit moved boundary-side face ({d}, {c:?}) on an outflow domain: {before} -> {after}"
            );
        }
    }
    assert!(moved_interior > 0, "the commit moved no interior face; the operator never acted");
}

// on a periodic domain the closing face of each component is the wrap image of its first face
// after the commit, so the cell interpolation of the last cell reads a consistent field.
#[test]
fn the_commit_keeps_the_closing_faces_periodic_images_on_a_periodic_domain() {
    let sim = build(BoundaryType::Periodic);
    let sim = &sim;
    let receipt = magnetic_slip_solve::<3, 3, HostMemory, f64>(sim, DT, GAMMA, 1e-12, 500);
    assert!(receipt.converged);
    magnetic_slip_commit::<3, 3, HostMemory, f64>(sim, DT, GAMMA);
    let m = sim.fields.mhd.as_ref().unwrap();
    let interior = &sim.geom.interior;
    for d in 0..3 {
        for c in interior.iter() {
            if c[d] != interior.spaces[d].lo {
                continue;
            }
            let mut closing = c;
            closing[d] = interior.spaces[d].hi;
            let (first, last) = (*m.bface[d].at(c), *m.bface[d].at(closing));
            assert!(first == last, "component {d} at {c:?}: closing face {last} is not the image of {first}");
        }
    }
}

// ---- the 2.5D mixed complex on an outflow domain -------------------------------------------------

type Sim2 = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;

fn build_2p5d(boundary: BoundaryType) -> Sim2 {
    let dx = 1.0 / N as f64;
    let k = 2.0 * std::f64::consts::PI;
    let (a0, b0) = (0.3, 0.2);
    let sim = SimStateGeneric::<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>::build(
        NewtonianMhd,
        IdealGas { gamma: GAMMA },
        Cartesian,
    )
    .cells([N, N])
    .origin([0.0, 0.0])
    .spacing([dx, dx])
    .boundaries(Boundaries::uniform(boundary))
    .cfl(0.3)
    .allocate()
    .expect("2.5D sim construction")
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
    .build();
    sim.with_bodies(
        BodyCollection::new().add(
            Body::black_hole(0, Tensor::new([0.5, 0.5]), Tensor::zeros(), 1.0, R_BODY, 0.05, 1.0, 1.0, R_BODY)
                .with_surface(SurfaceSpec::Drain)
                .with_magnetic(MagneticSpec::Slip {
                    diffusivity_ratio: 2.0,
                    shell_width: 0.12,
                    slip_length_ratio: 1.0,
                    field_regularization: 0.1,
                    placement: 0.0,
                }),
        ),
    )
}

// every stored face, every stored cell of every component, and the interior energy.
fn state_2p5d(sim: &Sim2) -> (Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    let m = sim.fields.mhd.as_ref().unwrap();
    let faces = (0..2).flat_map(|d| m.bface[d].domain().iter().map(move |c| *m.bface[d].at(c))).collect();
    let cells = (0..3).map(|d| m.bcell[d].domain().iter().map(|c| *m.bcell[d].at(c)).collect()).collect();
    let nrg = sim.fields.cons.nrg_field().unwrap();
    (faces, cells, sim.geom.interior.iter().map(|c| *nrg.at(c)).collect())
}

// the solve on the mixed complex is transactional over every stored face and cell on an outflow
// domain, read from the second solve after one projecting solve.
#[test]
fn the_2p5d_solve_is_transactional_over_every_stored_field_on_an_outflow_domain() {
    let sim = build_2p5d(BoundaryType::Outflow);
    assert!(magnetic_slip_solve::<2, 3, HostMemory, f64>(&sim, DT, GAMMA, 1e-12, 500).converged);
    let before = state_2p5d(&sim);
    let receipt = magnetic_slip_solve::<2, 3, HostMemory, f64>(&sim, DT, GAMMA, 1e-12, 500);
    assert!(receipt.converged, "the 2.5D solve did not converge on the outflow domain");
    let after = state_2p5d(&sim);
    assert!(before.0 == after.0, "the solve moved a stored face");
    assert!(before.1 == after.1, "the solve moved a stored cell component");
    assert!(before.2 == after.2, "the solve moved the total energy");
}

// the commit moves interior faces and cells and leaves the boundary-side faces and the cell B_z halo
// of an outflow domain at their input values.
#[test]
fn the_2p5d_commit_leaves_outflow_boundary_storage_at_its_input_values() {
    let sim = build_2p5d(BoundaryType::Outflow);
    let m = sim.fields.mhd.as_ref().unwrap();
    let before = state_2p5d(&sim);
    assert!(magnetic_slip_solve::<2, 3, HostMemory, f64>(&sim, DT, GAMMA, 1e-12, 500).converged);
    magnetic_slip_commit::<2, 3, HostMemory, f64>(&sim, DT, GAMMA);
    let after = state_2p5d(&sim);
    let interior = &sim.geom.interior;
    let inside = |c: [isize; 2]| (0..2).all(|a| c[a] >= interior.spaces[a].lo && c[a] < interior.spaces[a].hi);
    let mut moved_faces = 0usize;
    let mut idx = 0usize;
    for d in 0..2 {
        for c in m.bface[d].domain().iter() {
            let (b, a) = (before.0[idx], after.0[idx]);
            idx += 1;
            if inside(c) {
                if b != a {
                    moved_faces += 1;
                }
            } else {
                assert!(b == a, "the commit moved boundary-side face ({d}, {c:?}) on an outflow domain");
            }
        }
    }
    // the cell B_z halo is derived storage: on an outflow side it is the edge cell's value, the same
    // image the production ghost fill forms, regenerated here from the committed interior.
    let mut moved_bz = 0usize;
    for (i, c) in m.bcell[2].domain().iter().enumerate() {
        let (b, a) = (before.1[2][i], after.1[2][i]);
        if inside(c) {
            if b != a {
                moved_bz += 1;
            }
        } else {
            let edge: [isize; 2] = std::array::from_fn(|k| c[k].clamp(interior.spaces[k].lo, interior.spaces[k].hi - 1));
            let image = *m.bcell[2].at(edge);
            assert!(a == image, "the B_z halo at {c:?} is {a}, not the outflow image {image} of {edge:?}");
        }
    }
    assert!(moved_faces > 0 && moved_bz > 0, "the commit moved no interior storage; the operator never acted");
}
