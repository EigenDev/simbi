// =============================================================================
// mhd_slip_imhd.rs
//
// the magnetic slip under the isothermal closure: the same force-selective operator and implicit
// midpoint solve as the adiabatic sink, with the dissipated magnetic energy exported to the
// cooling bath and booked per body as its slip-heat receipt. an isothermal store carries no energy
// field and no gas-energy staging; a force-free field is an exact no-op; the exported heat is
// nonnegative cell by cell and equals the face-Hodge magnetic energy loss to roundoff, on a 3D grid
// and on the 2.5D mixed complex with the vertical channel; the gas state is untouched by the slip
// and stays admissible under the coupled step; transparent and resistive sinks carry no slip
// storage; two sinks book their heat separately and the ledger closes under either closure.
// =============================================================================

use symbi::regimes::substrate_isothermal_mhd::IsothermalMhdSubstrateKernelSet;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::{IdealGas, Isothermal};
use symbi_hydro::isothermal_mhd::IsothermalMhd;
use symbi_hydro::mhd_state::{MhdPrim, MhdPrimG};
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::{Prim, PrimG};
use symbi_ib::{Body, BodyCollection, BodyKind, MagneticSpec, SurfaceSpec};
use symbi_sim::state::FieldStore;
use symbi_substrate::regimes::mhd_substrate::{magnetic_slip_commit, magnetic_slip_solve};
use symbi_xpu::{CpuSpace, HostMemory};

type Iso3 = SimStateGeneric<IsothermalMhd, 3, 3, Cartesian, Isothermal<f64>, CpuSpace, HostMemory, f64>;
type Iso2 = SimStateGeneric<IsothermalMhd, 2, 3, Cartesian, Isothermal<f64>, CpuSpace, HostMemory, f64>;
type Adi3 = SimStateGeneric<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;

const N: usize = 16;
const CS: f64 = 1.0;
const GAMMA: f64 = 5.0 / 3.0;
const DT: f64 = 1.0e-3;
const R_BODY: f64 = 0.22;
const W: f64 = 0.12;

fn slip_spec() -> MagneticSpec {
    MagneticSpec::Slip {
        diffusivity_ratio: 2.0,
        shell_width: W,
        slip_length_ratio: 1.5,
        field_regularization: 0.1,
        placement: 0.0,
    }
}

fn sink(idx: usize, position: [f64; 3], r_acc: f64, magnetic: MagneticSpec) -> Body<f64, 3> {
    Body::black_hole(idx, Tensor::new(position), Tensor::zeros(), 1.0, r_acc, 0.05, 1.0, 1.0, r_acc)
        .with_surface(SurfaceSpec::Drain)
        .with_magnetic(magnetic)
}

// a discretely solenoidal in-plane field from a corner vector potential; the phase keeps the box
// center off the field's null.
fn potential(amp: f64) -> impl Fn(f64, f64) -> f64 + Copy {
    let k = 2.0 * std::f64::consts::PI;
    let phase = 0.25 * std::f64::consts::PI;
    move |x: f64, y: f64| amp / k * (k * x + phase).sin() * (k * y + phase).sin()
}
fn face_of_potential(az: impl Fn(f64, f64) -> f64 + Copy, dx: f64) -> impl Fn(usize, [f64; 3]) -> f64 + Copy {
    move |axis: usize, [x, y, _z]: [f64; 3]| match axis {
        0 => (az(x, y + 0.5 * dx) - az(x, y - 0.5 * dx)) / dx,
        1 => -(az(x + 0.5 * dx, y) - az(x - 0.5 * dx, y)) / dx,
        _ => 0.0,
    }
}

// seed every stored face, halo included, from the analytic face field at the face's coordinate:
// the builder's face seed fills the interior faces alone, and the operator's current gather reads
// the halo.
fn seed_all_faces<const DOF: usize, R, E>(sim: &SimStateGeneric<R, 3, DOF, Cartesian, E, CpuSpace, HostMemory, f64>, face: impl Fn(usize, [f64; 3]) -> f64)
where
    R: symbi_hydro::regime::Regime<f64, 3>,
    E: symbi_hydro::eos::EosFor<f64, <R as symbi_hydro::regime::Regime<f64, 3>>::Energy>,
{
    let m = sim.fields.mhd.as_ref().unwrap();
    let (x_lo, dx) = (sim.geom.x_lo, sim.geom.dx);
    for d in 0..3 {
        for c in m.bface[d].domain().iter() {
            let x: [f64; 3] = std::array::from_fn(|a| x_lo[a] + (c[a] as f64 + if a == d { 0.0 } else { 0.5 }) * dx[a]);
            m.bface[d].set(c, face(d, x));
        }
    }
}

// a 3D isothermal periodic unit box threaded by the given face field (uniform when `amp` is 0
// and `b_uniform` finite), with the given bodies attached.
fn iso3(uniform: [f64; 3], amp: f64, bodies: BodyCollection<f64, 3>) -> Iso3 {
    let dx = 1.0 / N as f64;
    let face = face_of_potential(potential(amp), dx);
    let sim = Iso3::build(IsothermalMhd, Isothermal { cs: CS }, Cartesian)
        .cells([N; 3])
        .origin([0.0; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .allocate()
        .expect("iso sim construction")
        .set_initial(move |[x, y, z]| {
            let bx = uniform[0] + 0.5 * (face(0, [x - 0.5 * dx, y, z]) + face(0, [x + 0.5 * dx, y, z]));
            let by = uniform[1] + 0.5 * (face(1, [x, y - 0.5 * dx, z]) + face(1, [x, y + 0.5 * dx, z]));
            MhdPrimG::<f64, 3, IsoModel>::new(PrimG::isothermal(Density(1.0), Tensor::new([0.0; 3])), Tensor::new([bx, by, uniform[2]]))
        })
        .seed_faces(move |axis, x| uniform[axis] + face(axis, x))
        .build();
    seed_all_faces(&sim, move |axis, x| uniform[axis] + face(axis, x));
    sim.with_bodies(bodies)
}

fn adi3(amp: f64, bodies: BodyCollection<f64, 3>) -> Adi3 {
    let dx = 1.0 / N as f64;
    let face = face_of_potential(potential(amp), dx);
    let sim = Adi3::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 3])
        .origin([0.0; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .allocate()
        .expect("adiabatic sim construction")
        .set_initial(move |[x, y, z]| {
            let bx = 0.5 * (face(0, [x - 0.5 * dx, y, z]) + face(0, [x + 0.5 * dx, y, z]));
            let by = 0.5 * (face(1, [x, y - 0.5 * dx, z]) + face(1, [x, y + 0.5 * dx, z]));
            MhdPrim::new(Prim::adiabatic(Density(1.0), Tensor::new([0.0; 3]), Pressure(1.0)), Tensor::new([bx, by, 0.0]))
        })
        .seed_faces(face)
        .build();
    seed_all_faces(&sim, face);
    sim.with_bodies(bodies)
}

// a 2.5D isothermal periodic unit square with in-plane faces zero and a sheared vertical field.
fn iso2(bodies: BodyCollection<f64, 2>) -> Iso2 {
    let dx = 1.0 / N as f64;
    let k = 2.0 * std::f64::consts::PI;
    let bz = move |x: f64, y: f64| 0.3 * (1.0 + 0.5 * (k * x).sin() * (k * y).cos());
    let sim = Iso2::build(IsothermalMhd, Isothermal { cs: CS }, Cartesian)
        .cells([N; 2])
        .origin([0.0; 2])
        .spacing([dx; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .allocate()
        .expect("2.5D iso sim construction")
        .set_initial(move |[x, y]| {
            MhdPrimG::<f64, 3, IsoModel>::new(PrimG::isothermal(Density(1.0), Tensor::new([0.0; 3])), Tensor::new([0.0, 0.0, bz(x, y)]))
        })
        .seed_faces(|_, _| 0.0)
        .build();
    sim.with_bodies(bodies)
}

fn one_sink3(magnetic: MagneticSpec) -> BodyCollection<f64, 3> {
    BodyCollection::new().add(sink(0, [0.5; 3], R_BODY, magnetic))
}

// the face-Hodge magnetic energy over the lower faces of every interior cell (each physical face
// of a periodic box once), volume-weighted.
fn face_energy<const D: usize, const DOF: usize>(sim: &FieldStore<D, DOF, HostMemory, f64>) -> f64 {
    let m = sim.fields.mhd.as_ref().unwrap();
    let vol: f64 = sim.geom.dx.iter().product();
    let mut e = 0.0;
    for d in 0..D {
        for c in sim.geom.interior.iter() {
            e += 0.5 * m.bface[d].at(c).powi(2) * vol;
        }
    }
    e
}

// the cell magnetic energy of the out-of-plane channel of a 2.5D grid, volume-weighted.
fn bz_energy(sim: &Iso2) -> f64 {
    let m = sim.fields.mhd.as_ref().unwrap();
    let vol: f64 = sim.geom.dx.iter().product();
    sim.geom.interior.iter().map(|c| 0.5 * m.bcell[2].at(c).powi(2) * vol).sum()
}

fn cell_energy(sim: &Adi3) -> f64 {
    let m = sim.fields.mhd.as_ref().unwrap();
    let vol: f64 = sim.geom.dx.iter().product();
    sim.geom.interior.iter().map(|c| 0.5 * (0..3).map(|d| m.bcell[d].at(c).powi(2)).sum::<f64>() * vol).sum()
}

fn total_energy(sim: &Adi3) -> f64 {
    let vol: f64 = sim.geom.dx.iter().product();
    sim.geom.interior.iter().map(|c| *sim.fields.cons.nrg_field().unwrap().at(c) * vol).sum()
}

fn heat_receipts<const D: usize, const DOF: usize>(sim: &FieldStore<D, DOF, HostMemory, f64>) -> Vec<f64> {
    sim.immersed.as_ref().unwrap().diagnostics.consolidate().iter().map(|d| d.slip_heat_delta).collect()
}

fn faces_snapshot<const D: usize, const DOF: usize>(sim: &FieldStore<D, DOF, HostMemory, f64>) -> Vec<u64> {
    let m = sim.fields.mhd.as_ref().unwrap();
    let mut v = Vec::new();
    for d in 0..D {
        for c in m.bface[d].domain().iter() {
            v.push(m.bface[d].at(c).to_bits());
        }
    }
    v
}

fn gas_snapshot<const D: usize, const DOF: usize>(sim: &FieldStore<D, DOF, HostMemory, f64>) -> Vec<u64> {
    let mut v = Vec::new();
    for c in sim.geom.allocated.iter() {
        v.push(sim.fields.cons.den.at(c).to_bits());
        for k in 0..DOF {
            v.push(sim.fields.cons.mom[k].at(c).to_bits());
        }
        v.push(sim.fields.prim.rho.at(c).to_bits());
        if let Some(p) = sim.fields.prim.pre_field() {
            v.push(p.at(c).to_bits());
        }
    }
    v
}

// the slip step on a single grid: the solve then the commit, as the kernel set runs them.
fn slip_step_iso3(sim: &Iso3, dt: f64) {
    let receipt = magnetic_slip_solve::<3, 3, HostMemory, f64>(sim, dt, CS, 1e-12, 500);
    assert!(receipt.converged, "the isothermal slip solve did not converge: {receipt:?}");
    magnetic_slip_commit::<3, 3, HostMemory, f64>(sim, dt, CS);
}

#[test]
fn an_isothermal_store_carries_no_energy_field_and_no_gas_energy_staging() {
    let sim = iso3([0.3, 0.0, 0.0], 0.2, one_sink3(slip_spec()));
    assert!(sim.fields.cons.nrg_field().is_none(), "an isothermal store carries an energy field");
    let ws = sim.fields.mhd.as_ref().unwrap().magnetic_slip.as_ref().expect("slip workspace");
    assert!(ws.gas_energy.is_none(), "an isothermal slip workspace staged a gas energy");
    let adiabatic = adi3(0.2, one_sink3(slip_spec()));
    let ws = adiabatic.fields.mhd.as_ref().unwrap().magnetic_slip.as_ref().expect("slip workspace");
    assert!(ws.gas_energy.is_some(), "an adiabatic slip workspace stages the predicted gas energy");
}

// a uniform field has no current, so the operator vanishes: the faces, the gas, and the receipt
// are untouched bit for bit.
#[test]
fn a_force_free_field_is_an_exact_no_op_under_the_isothermal_slip() {
    let sim = iso3([0.3, 0.2, 0.1], 0.0, one_sink3(slip_spec()));
    let (faces, gas) = (faces_snapshot(&sim), gas_snapshot(&sim));
    let receipt = magnetic_slip_solve::<3, 3, HostMemory, f64>(&sim, DT, CS, 1e-12, 500);
    println!("force-free receipt {receipt:?}");
    let ws = sim.fields.mhd.as_ref().unwrap().magnetic_slip.as_ref().unwrap();
    let m = sim.fields.mhd.as_ref().unwrap();
    let mut moved = std::collections::BTreeMap::new();
    for d in 0..3 {
        for c in m.bface[d].domain().iter() {
            if ws.candidate.b[d].at(c).to_bits() != ws.input.b[d].at(c).to_bits() {
                *moved.entry((d, c[d])).or_insert(0usize) += 1;
            }
        }
    }
    println!("candidate vs input: moved faces by (axis, index along axis): {moved:?}");
    magnetic_slip_commit::<3, 3, HostMemory, f64>(&sim, DT, CS);
    let after = faces_snapshot(&sim);
    assert!(after == faces, "the slip moved a force-free field");
    assert!(gas_snapshot(&sim) == gas, "the slip touched the gas state");
    assert_eq!(heat_receipts(&sim)[0], 0.0, "a force-free field exported heat");
}

// the isothermal theorem: the face-Hodge magnetic energy decreases by exactly the heat the sink
// exports, the per-cell rate is nonnegative, and the gas state is untouched by the slip.
#[test]
fn the_exported_heat_is_nonnegative_and_equals_the_face_energy_loss() {
    let sim = iso3([0.3, 0.0, 0.0], 0.4, one_sink3(slip_spec()));
    let gas = gas_snapshot(&sim);
    let m0 = face_energy(&sim);
    slip_step_iso3(&sim, DT);
    let dm = face_energy(&sim) - m0;
    let q = heat_receipts(&sim)[0];
    let ws = sim.fields.mhd.as_ref().unwrap().magnetic_slip.as_ref().unwrap();
    let rates: Vec<f64> = sim.geom.interior.iter().map(|c| *ws.dissipation.at(c)).collect();
    let peak = rates.iter().cloned().fold(0.0, f64::max);
    println!("isothermal slip: exported heat {q:.9e}, -dM_face {:.9e}, residual {:.3e}, peak rate {peak:.3e}", -dm, (q + dm).abs());
    assert!(q > 0.0, "the sink exported no heat; the field is force-free at the shell");
    assert!(rates.iter().all(|r| *r >= -1e-14 * peak), "a cell's dissipation rate is negative");
    assert!((q + dm).abs() < 1e-10 * q, "the isothermal theorem fails: exported heat {q:.12e}, -dM_face {:.12e}", -dm);
    assert!(gas_snapshot(&sim) == gas, "the slip touched the gas state");
}

// the coupled step under the isothermal closure: the drain, the slip, and the ideal-MHD step on a
// draining sink; the density and velocity stay admissible, the pressure stays the closure's
// cs^2 rho, the sink accretes and exports heat.
#[test]
fn the_coupled_isothermal_step_keeps_the_gas_admissible_and_books_the_heat() {
    let sim = iso3([0.3, 0.0, 0.0], 0.2, one_sink3(slip_spec()));
    let kset = IsothermalMhdSubstrateKernelSet::<HostMemory, f64, 3>::new(CS, 0.3, 1.0, &sim.geom.allocated);
    let mut hier = Hierarchy::single(sim, kset);
    hier.prime();
    for _ in 0..5 {
        hier.step_root_with_dt(2.0 * DT);
    }
    let sim = &hier.levels[0].state;
    for c in sim.geom.interior.iter() {
        let rho = *sim.fields.prim.rho.at(c);
        assert!(rho.is_finite() && rho > 0.0, "inadmissible density {rho} at {c:?}");
        for k in 0..3 {
            assert!(sim.fields.prim.vel[k].at(c).is_finite(), "non-finite velocity at {c:?}");
        }
        if let Some(p) = sim.fields.prim.pre_field() {
            let pre = *p.at(c);
            assert!((pre - CS * CS * rho).abs() <= 1e-12 * pre.abs().max(1e-300), "the pressure left the isothermal closure at {c:?}: {pre} vs {}", CS * CS * rho);
        }
    }
    let body = sim.immersed.as_ref().unwrap().bodies.get(0);
    let accreted = match body.kind {
        BodyKind::BlackHole { total_accreted_mass, .. } => total_accreted_mass,
        _ => 0.0,
    };
    assert!(accreted > 0.0, "the sink accreted nothing");
    assert!(body.slip_heat_total > 0.0, "the body booked no exported heat over the coupled steps");
    assert!(body.slip_heat_rate > 0.0, "the body books no heat rate");
}

#[test]
fn transparent_and_resistive_isothermal_sinks_carry_no_slip_storage() {
    for (label, magnetic) in [("transparent", MagneticSpec::None), ("resistive", MagneticSpec::Resistive { eta: 0.02 })] {
        let sim = iso3([0.3, 0.0, 0.0], 0.2, one_sink3(magnetic));
        assert!(sim.fields.mhd.as_ref().unwrap().magnetic_slip.is_none(), "{label}: slip storage allocated");
        let kset = IsothermalMhdSubstrateKernelSet::<HostMemory, f64, 3>::new(CS, 0.3, 1.0, &sim.geom.allocated);
        let mut hier = Hierarchy::single(sim, kset);
        hier.prime();
        for _ in 0..3 {
            hier.step_root_with_dt(2.0 * DT);
        }
        let sim = &hier.levels[0].state;
        assert!(sim.geom.interior.iter().all(|c| sim.fields.prim.rho.at(c).is_finite()), "{label}: non-finite density");
        assert_eq!(sim.immersed.as_ref().unwrap().bodies.get(0).slip_heat_total, 0.0, "{label}: a sink without the slip booked heat");
    }
}

// the 2.5D mixed complex under the isothermal closure with a sheared vertical field: the in-plane
// faces stay zero, the cell B_z channel dissipates, and the exported heat equals the loss of the
// mixed magnetic energy (faces plus cell B_z) to roundoff.
#[test]
fn the_vertical_field_channel_exports_heat_on_a_2p5d_isothermal_grid() {
    let sim = iso2(BodyCollection::new().add(
        Body::black_hole(0, Tensor::new([0.5, 0.5]), Tensor::zeros(), 1.0, R_BODY, 0.05, 1.0, 1.0, R_BODY)
            .with_surface(SurfaceSpec::Drain)
            .with_magnetic(slip_spec()),
    ));
    assert!(sim.fields.cons.nrg_field().is_none());
    let faces = faces_snapshot(&sim);
    let m0 = face_energy(&sim) + bz_energy(&sim);
    let bz0: Vec<f64> = sim.geom.interior.iter().map(|c| *sim.fields.mhd.as_ref().unwrap().bcell[2].at(c)).collect();
    let receipt = magnetic_slip_solve::<2, 3, HostMemory, f64>(&sim, DT, CS, 1e-12, 500);
    assert!(receipt.converged, "the 2.5D isothermal solve did not converge: {receipt:?}");
    magnetic_slip_commit::<2, 3, HostMemory, f64>(&sim, DT, CS);
    let dm = face_energy(&sim) + bz_energy(&sim) - m0;
    let q = heat_receipts(&sim)[0];
    let bz1: Vec<f64> = sim.geom.interior.iter().map(|c| *sim.fields.mhd.as_ref().unwrap().bcell[2].at(c)).collect();
    println!("2.5D isothermal slip: exported heat {q:.9e}, -dM_mixed {:.9e}, residual {:.3e}", -dm, (q + dm).abs());
    assert!(q > 0.0, "the vertical channel exported no heat");
    assert!(bz0 != bz1, "the cell B_z channel did not move");
    assert!(faces_snapshot(&sim) == faces, "an in-plane face moved under a purely vertical field");
    assert!((q + dm).abs() < 1e-10 * q, "the 2.5D isothermal theorem fails: exported heat {q:.12e}, -dM_mixed {:.12e}", -dm);
}

// two slip sinks under the isothermal closure: each books its own heat, both positive, and the
// sum is the face-Hodge magnetic energy loss.
#[test]
fn two_isothermal_slip_sinks_book_their_heat_separately_and_the_ledger_closes() {
    let bodies = BodyCollection::new()
        .add(sink(0, [0.3, 0.5, 0.5], 0.15, slip_spec()))
        .add(sink(1, [0.7, 0.5, 0.5], 0.15, slip_spec()));
    let sim = iso3([0.3, 0.0, 0.0], 0.4, bodies);
    let m0 = face_energy(&sim);
    slip_step_iso3(&sim, DT);
    let dm = face_energy(&sim) - m0;
    let q = heat_receipts(&sim);
    println!("two isothermal sinks: heat {:.9e} + {:.9e}, -dM_face {:.9e}", q[0], q[1], -dm);
    assert!(q[0] > 0.0 && q[1] > 0.0, "a sink exported no heat: {q:?}");
    assert!((q[0] + q[1] + dm).abs() < 1e-10 * (q[0] + q[1]), "the two-sink ledger does not close: {:.12e} vs {:.12e}", q[0] + q[1], -dm);
}

// two slip sinks under the adiabatic closure: the extended energy E - M_cell + M_face is invariant
// to roundoff, so the heat deposited in the gas is the sum of both shells' dissipation.
#[test]
fn two_adiabatic_slip_sinks_keep_the_extended_energy_invariant() {
    let bodies = BodyCollection::new()
        .add(sink(0, [0.3, 0.5, 0.5], 0.15, slip_spec()))
        .add(sink(1, [0.7, 0.5, 0.5], 0.15, slip_spec()));
    let sim = adi3(0.4, bodies);
    let x0 = total_energy(&sim) - cell_energy(&sim) + face_energy(&sim);
    let m0 = face_energy(&sim);
    let receipt = magnetic_slip_solve::<3, 3, HostMemory, f64>(&sim, DT, GAMMA, 1e-12, 500);
    assert!(receipt.converged);
    magnetic_slip_commit::<3, 3, HostMemory, f64>(&sim, DT, GAMMA);
    let x1 = total_energy(&sim) - cell_energy(&sim) + face_energy(&sim);
    let q = heat_receipts(&sim);
    let dm = face_energy(&sim) - m0;
    println!("two adiabatic sinks: heat {:.9e} + {:.9e}, -dM_face {:.9e}, extended residual {:.3e}", q[0], q[1], -dm, x1 - x0);
    assert!(q[0] > 0.0 && q[1] > 0.0, "a sink released no heat: {q:?}");
    assert!((x1 - x0).abs() < 1e-11 * x0.abs(), "the extended energy is not invariant with two sinks: {x0:.12e} -> {x1:.12e}");
    assert!((q[0] + q[1] + dm).abs() < 1e-10 * (q[0] + q[1]), "the two-sink receipts do not sum to the face loss");
}

