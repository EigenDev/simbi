// =============================================================================
// refinement_mhd_2p5d.rs
//
// static refinement of the 2.5D magnetized gas on the mixed complex F_x + F_y + C_z: the in-plane
// staggered field transfers by face restriction and prolongation with the corner-EMF reflux, the
// out-of-plane cell field B_z by conservative cell transfer with its induction-flux reflux. under
// periodic boundaries the ideal-MHD hierarchy operator conserves the leaf-domain gas totals and the
// leaf-domain vertical flux, keeps the covered coarse state the restriction and the coarse
// divergence at machine zero, and is second order in the timestep in the fine bulk. with a
// magnetic-slip sink on the finest under either closure the palindrome runs, the finest slip
// half-step satisfies the mixed magnetic-energy theorem, a rejected attempt replays cleanly, and
// a checkpoint resumes bit for bit.
// =============================================================================

use symbi::regimes::substrate_isothermal_mhd::IsothermalMhdSubstrateKernelSet;
use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::refinement::hierarchy::{slip_failure_arm, slip_schedule_arm, slip_schedule_take, SlipFailPoint};
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_grid::Field;
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::{IdealGas, Isothermal};
use symbi_hydro::isothermal_mhd::IsothermalMhd;
use symbi_hydro::mhd_state::{MhdPrim, MhdPrimG};
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::{Prim, PrimG};
use symbi_ib::{Body, BodyCollection, BodyKind, MagneticSpec, SurfaceSpec};
use symbi_io::Metadata;
use symbi_refinement::refinement::transfer::{restrict_bface, restrict_cell_field};
use symbi_sim::state::FieldStore;
use symbi_substrate::regimes::mhd_substrate::{magnetic_slip_commit, magnetic_slip_solve};
use symbi_xpu::{CpuSpace, HostMemory};

type Adi = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;
type Iso = SimStateGeneric<IsothermalMhd, 2, 3, Cartesian, Isothermal<f64>, CpuSpace, HostMemory, f64>;
type AdiK = NewtonianMhdSubstrateKernelSet<HostMemory, f64, 2>;
type IsoK = IsothermalMhdSubstrateKernelSet<HostMemory, f64, 2>;
type AdiH = Hierarchy<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, AdiK>;
type IsoH = Hierarchy<IsothermalMhd, 2, 3, Cartesian, Isothermal<f64>, CpuSpace, HostMemory, IsoK>;

const N: usize = 32;
const GAMMA: f64 = 5.0 / 3.0;
const CS: f64 = 1.0;
const DX_FINE: f64 = 1.0 / (2.0 * N as f64);
const DT: f64 = 1.0e-3;
const VEL: [f64; 3] = [0.3, 0.2, 0.0];
const REGION: [f64; 2] = [-0.4375, 0.4375];

fn slip_spec() -> MagneticSpec {
    MagneticSpec::Slip {
        diffusivity_ratio: 2.0,
        shell_width: DX_FINE,
        slip_length_ratio: 1.0,
        field_regularization: 0.1,
        placement: 0.0,
    }
}

// the in-plane field from a corner potential (discretely solenoidal) and a sheared vertical field.
fn potential(x: f64, y: f64) -> f64 {
    let k = 2.0 * std::f64::consts::PI;
    let phase = 0.25 * std::f64::consts::PI;
    0.3 / k * (k * x + phase).sin() * (k * y + phase).sin()
}
fn face(dx: f64) -> impl Fn(usize, [f64; 2]) -> f64 + Copy {
    move |axis: usize, [x, y]: [f64; 2]| match axis {
        0 => (potential(x, y + 0.5 * dx) - potential(x, y - 0.5 * dx)) / dx,
        _ => -(potential(x + 0.5 * dx, y) - potential(x - 0.5 * dx, y)) / dx,
    }
}
fn vertical(x: f64, y: f64) -> f64 {
    let k = 2.0 * std::f64::consts::PI;
    0.2 * (1.0 + 0.3 * (k * x).sin() * (k * y).cos())
}

fn sink(magnetic: MagneticSpec) -> BodyCollection<f64, 2> {
    BodyCollection::new().add(
        Body::black_hole(0, Tensor::new([0.0, 0.0]), Tensor::zeros(), 1.0, 2.0 * DX_FINE, 0.05, 1.0, 1.0, 2.0 * DX_FINE)
            .with_surface(SurfaceSpec::Drain)
            .with_magnetic(magnetic),
    )
}

fn adi(bodies: Option<BodyCollection<f64, 2>>) -> AdiH {
    let dx = 1.0 / N as f64;
    let k = 2.0 * std::f64::consts::PI;
    let f = face(dx);
    let kset = |s: &Adi| AdiK::new(GAMMA, 0.3, 1.0, &s.geom.allocated);
    let coarse = Adi::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 2])
        .origin([-0.5; 2])
        .spacing([dx; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .allocate()
        .expect("root")
        .set_initial(move |[x, y]| {
            let bx = 0.5 * (f(0, [x - 0.5 * dx, y]) + f(0, [x + 0.5 * dx, y]));
            let by = 0.5 * (f(1, [x, y - 0.5 * dx]) + f(1, [x, y + 0.5 * dx]));
            MhdPrim::new(Prim::adiabatic(Density(1.0), Tensor::new(VEL), Pressure(1.0 + 0.1 * (k * x).sin())), Tensor::new([bx, by, vertical(x, y)]))
        })
        .seed_faces(f)
        .build();
    let ck = kset(&coarse);
    let regions = [RefinementRegion { x_lo: [REGION[0]; 2], x_hi: [REGION[1]; 2] }];
    let hier = Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, kset).unwrap();
    hier.seed_fine_from_coarse().expect("fine seed");
    let mut hier = match bodies {
        Some(b) => hier.with_bodies(b),
        None => hier,
    };
    hier.prime();
    hier
}

fn iso(bodies: Option<BodyCollection<f64, 2>>) -> IsoH {
    let dx = 1.0 / N as f64;
    let k = 2.0 * std::f64::consts::PI;
    let f = face(dx);
    let kset = |s: &Iso| IsoK::new(CS, 0.3, 1.0, &s.geom.allocated);
    let coarse = Iso::build(IsothermalMhd, Isothermal { cs: CS }, Cartesian)
        .cells([N; 2])
        .origin([-0.5; 2])
        .spacing([dx; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .allocate()
        .expect("root")
        .set_initial(move |[x, y]| {
            let bx = 0.5 * (f(0, [x - 0.5 * dx, y]) + f(0, [x + 0.5 * dx, y]));
            let by = 0.5 * (f(1, [x, y - 0.5 * dx]) + f(1, [x, y + 0.5 * dx]));
            MhdPrimG::<f64, 3, IsoModel>::new(PrimG::isothermal(Density(1.0 + 0.1 * (k * x).sin()), Tensor::new(VEL)), Tensor::new([bx, by, vertical(x, y)]))
        })
        .seed_faces(f)
        .build();
    let ck = kset(&coarse);
    let regions = [RefinementRegion { x_lo: [REGION[0]; 2], x_hi: [REGION[1]; 2] }];
    let hier = Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, kset).unwrap();
    hier.seed_fine_from_coarse().expect("fine seed");
    let mut hier = match bodies {
        Some(b) => hier.with_bodies(b),
        None => hier,
    };
    hier.prime();
    hier
}

fn compensated_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let (mut sum, mut comp) = (0.0f64, 0.0f64);
    for v in values {
        let t = sum + v;
        comp += if sum.abs() >= v.abs() { (sum - t) + v } else { (v - t) + sum };
        sum = t;
    }
    sum + comp
}

// leaf-domain totals: mass, three momenta, energy when carried, and the vertical flux.
fn leaf_totals<const DOF: usize>(levels: &[(&FieldStore<2, DOF, HostMemory, f64>, Option<&symbi_algebra::Domain<2>>)]) -> [f64; 6] {
    let mut parts: [Vec<f64>; 6] = Default::default();
    for (sim, cov) in levels {
        let vol: f64 = sim.geom.dx.iter().product();
        let m = sim.fields.mhd.as_ref().unwrap();
        for c in sim.geom.interior.iter() {
            if cov.is_some_and(|cv| cv.contains(c)) {
                continue;
            }
            parts[0].push(*sim.fields.cons.den.at(c) * vol);
            for a in 0..3 {
                parts[1 + a].push(*sim.fields.cons.mom[a].at(c) * vol);
            }
            parts[4].push(sim.fields.cons.nrg_field().map_or(0.0, |n| *n.at(c)) * vol);
            parts[5].push(*m.bcell[2].at(c) * vol);
        }
    }
    std::array::from_fn(|i| compensated_sum(parts[i].iter().copied()))
}

fn max_div_b(sim: &FieldStore<2, 3, HostMemory, f64>) -> f64 {
    let m = sim.fields.mhd.as_ref().unwrap();
    let mut worst = 0.0f64;
    for c in sim.geom.interior.iter() {
        let (ix, iy) = ([c[0] + 1, c[1]], [c[0], c[1] + 1]);
        let div = (*m.bface[0].at(ix) - *m.bface[0].at(c)) / sim.geom.dx[0] + (*m.bface[1].at(iy) - *m.bface[1].at(c)) / sim.geom.dx[1];
        worst = worst.max(div.abs());
    }
    worst
}

// the covered coarse gas, vertical field, and in-plane faces are the restrictions of the fine
// level's, and the coarse divergence is at machine zero everywhere, the refluxed shell included.
fn assert_transfer_invariants<const DOF: usize>(coarse: &FieldStore<2, DOF, HostMemory, f64>, fine: &FieldStore<2, DOF, HostMemory, f64>, cov: &symbi_algebra::Domain<2>, label: &str) {
    let scratch: Field<f64, 2, HostMemory> = Field::zeros(&coarse.geom.allocated).unwrap();
    let (cm, fm) = (coarse.fields.mhd.as_ref().unwrap(), fine.fields.mhd.as_ref().unwrap());
    for (f, c, name) in [(&fine.fields.cons.den, &coarse.fields.cons.den, "density"), (&fm.bcell[2], &cm.bcell[2], "vertical field")] {
        restrict_cell_field(f, &scratch, cov);
        for cell in cov.iter() {
            assert!(c.at(cell).to_bits() == scratch.at(cell).to_bits(), "{label}: {name} at {cell:?} is not the restriction");
        }
    }
    let faces = BfaceFields::<2, HostMemory, f64> {
        b: std::array::from_fn(|d| Field::zeros(cm.bface[d].domain()).unwrap()),
    };
    restrict_bface(&fm.bface, &faces, cov);
    for d in 0..2 {
        for cell in cov.iter() {
            assert!(cm.bface[d].at(cell).to_bits() == faces.b[d].at(cell).to_bits(), "{label}: face {d} at {cell:?} is not the area average");
        }
    }
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / a.abs().max(b.abs()).max(1e-300)
}

// with no body, under periodic boundaries, over two root steps: the leaf-domain mass, momenta,
// energy, and vertical flux are conserved to roundoff, the covered coarse state is the
// restriction, and the coarse divergence stays at machine zero.
#[test]
fn the_refined_2p5d_operator_conserves_the_leaf_totals_and_the_vertical_flux() {
    let mut hier = adi(None);
    let totals = |h: &AdiH| leaf_totals(&[(&h.levels[0].state, h.levels[0].coverage.as_ref()), (&h.levels[1].state, None)]);
    let before = totals(&hier);
    for _ in 0..2 {
        hier.step_root_with_dt(2.0 * DT);
    }
    let after = totals(&hier);
    println!("2.5D leaf totals before {before:?}\n                  after {after:?}");
    for (i, name) in [(0, "mass"), (4, "energy"), (5, "vertical flux")] {
        assert!(rel(before[i], after[i]) < 1e-12, "{name} drifts on the leaf domain: {} -> {}", before[i], after[i]);
    }
    for a in 0..3 {
        assert!((after[1 + a] - before[1 + a]).abs() < 1e-12 * before[0], "momentum {a} drifts on the leaf domain");
    }
    let (coarse, fine) = (&hier.levels[0].state, &hier.levels[1].state);
    assert_transfer_invariants(coarse, fine, hier.levels[0].coverage.as_ref().unwrap(), "adiabatic");
    let (dc, df) = (max_div_b(coarse), max_div_b(fine));
    println!("2.5D divergence after the steps: coarse {dc:.3e}, fine {df:.3e}");
    assert!(dc < 1e-12 && df < 1e-12, "div B left machine zero: coarse {dc:.3e}, fine {df:.3e}");
}

#[test]
fn the_refined_2p5d_isothermal_operator_conserves_the_leaf_totals_and_the_vertical_flux() {
    let mut hier = iso(None);
    let totals = |h: &IsoH| leaf_totals(&[(&h.levels[0].state, h.levels[0].coverage.as_ref()), (&h.levels[1].state, None)]);
    let before = totals(&hier);
    for _ in 0..2 {
        hier.step_root_with_dt(2.0 * DT);
    }
    let after = totals(&hier);
    for (i, name) in [(0, "mass"), (5, "vertical flux")] {
        assert!(rel(before[i], after[i]) < 1e-12, "{name} drifts on the leaf domain: {} -> {}", before[i], after[i]);
    }
    let (coarse, fine) = (&hier.levels[0].state, &hier.levels[1].state);
    assert_transfer_invariants(coarse, fine, hier.levels[0].coverage.as_ref().unwrap(), "isothermal");
    assert!(max_div_b(coarse) < 1e-12 && max_div_b(fine) < 1e-12, "div B left machine zero");
}

fn face_bits<const DOF: usize>(sim: &FieldStore<2, DOF, HostMemory, f64>) -> Vec<(usize, [isize; 2])> {
    let m = sim.fields.mhd.as_ref().unwrap();
    let interior = &sim.geom.interior;
    let mut v = Vec::new();
    for d in 0..2 {
        for c in m.bface[d].domain().iter() {
            if (0..2).all(|a| c[a] >= interior.spaces[a].lo && c[a] < interior.spaces[a].hi + (a == d) as isize) {
                v.push((d, c));
            }
        }
    }
    v
}

// the mixed magnetic energy of a level: the in-plane face-Hodge energy over the physical faces
// plus the vertical cell energy, as per-entry values for exact differencing.
fn mixed_energy_values<const DOF: usize>(sim: &FieldStore<2, DOF, HostMemory, f64>, faces: &[(usize, [isize; 2])]) -> Vec<f64> {
    let m = sim.fields.mhd.as_ref().unwrap();
    let mut v: Vec<f64> = faces.iter().map(|(d, c)| *m.bface[*d].at(*c)).collect();
    v.extend(sim.geom.interior.iter().map(|c| *m.bcell[2].at(c)));
    v
}
fn energy_change(before: &[f64], after: &[f64], vol: f64) -> f64 {
    before.iter().zip(after).map(|(b0, b1)| 0.5 * (b1 - b0) * (b1 + b0) * vol).sum()
}

// the finest slip half-step under the isothermal closure loses exactly the heat it exports from
// the mixed complex, faces and vertical cell field together, and the synchronization restricts.
#[test]
fn the_finest_2p5d_isothermal_slip_half_step_loses_the_mixed_energy_it_exports() {
    let hier = iso(Some(sink(slip_spec())));
    let fi = hier.levels.len() - 1;
    let l = &hier.levels[fi];
    let faces = face_bits(&l.state);
    let before = mixed_energy_values(&l.state, &faces);
    let receipt = magnetic_slip_solve::<2, 3, HostMemory, f64>(&l.state, 0.5 * DT, CS, 1e-14, 2000);
    assert!(receipt.converged, "{receipt:?}");
    magnetic_slip_commit::<2, 3, HostMemory, f64>(&l.state, 0.5 * DT, CS);
    let after = mixed_energy_values(&l.state, &faces);
    let vol: f64 = l.state.geom.dx.iter().product();
    let dm = energy_change(&before, &after, vol);
    let booked = l.state.immersed.as_ref().unwrap().diagnostics.consolidate()[0].slip_heat_delta;
    println!("refined 2.5D isothermal slip: exported {booked:.9e}, -dM_mixed {:.9e}, residual {:.3e}", -dm, (booked + dm).abs());
    assert!(booked > 0.0, "no heat exported");
    assert!((booked + dm).abs() < 1e-8 * booked, "the mixed theorem fails on the finest: exported {booked:.12e}, -dM {:.12e}", -dm);
    hier.sync_all_fine_to_coarse();
    assert_transfer_invariants(&hier.levels[0].state, &hier.levels[1].state, hier.levels[0].coverage.as_ref().unwrap(), "after the slip and the sync");
}

// the finest slip half-step under the adiabatic closure keeps the extended energy
// E - M_cell,in-plane + M_face,in-plane invariant: the vertical cell energy is common to both
// representations and cancels.
#[test]
fn the_finest_2p5d_adiabatic_slip_half_step_keeps_the_extended_energy_invariant() {
    let hier = adi(Some(sink(slip_spec())));
    let fi = hier.levels.len() - 1;
    let l = &hier.levels[fi];
    let sim = &l.state;
    let m = sim.fields.mhd.as_ref().unwrap();
    let vol: f64 = sim.geom.dx.iter().product();
    let extended = || -> f64 {
        let mut parts = Vec::new();
        for c in sim.geom.interior.iter() {
            parts.push((*sim.fields.cons.nrg_field().unwrap().at(c) - 0.5 * (m.bcell[0].at(c).powi(2) + m.bcell[1].at(c).powi(2))) * vol);
        }
        for (d, c) in face_bits(sim) {
            parts.push(0.5 * m.bface[d].at(c).powi(2) * vol);
        }
        compensated_sum(parts)
    };
    let x0 = extended();
    let receipt = magnetic_slip_solve::<2, 3, HostMemory, f64>(sim, 0.5 * DT, GAMMA, 1e-14, 2000);
    assert!(receipt.converged, "{receipt:?}");
    magnetic_slip_commit::<2, 3, HostMemory, f64>(sim, 0.5 * DT, GAMMA);
    let x1 = extended();
    let booked = sim.immersed.as_ref().unwrap().diagnostics.consolidate()[0].slip_heat_delta;
    println!("refined 2.5D adiabatic slip: heat {booked:.9e}, extended residual {:.3e}", x1 - x0);
    assert!(booked > 0.0, "no heat released");
    assert!((x1 - x0).abs() < 1e-11 * x0.abs(), "the extended energy is not invariant: {x0:.12e} -> {x1:.12e}");
}

fn snapshot(hier: &IsoH) -> Vec<u64> {
    let mut v = Vec::new();
    for l in &hier.levels {
        let sim = &l.state;
        let m = sim.fields.mhd.as_ref().unwrap();
        for c in sim.geom.allocated.iter() {
            v.push(sim.fields.cons.den.at(c).to_bits());
            for k in 0..3 {
                v.push(sim.fields.cons.mom[k].at(c).to_bits());
                v.push(m.bcell[k].at(c).to_bits());
            }
        }
        for d in 0..2 {
            for c in m.bface[d].domain().iter() {
                v.push(m.bface[d].at(c).to_bits());
            }
        }
        v.push(sim.time.to_bits());
        v.push(sim.iteration);
        if let Some(im) = sim.immersed.as_ref() {
            let b = im.bodies.get(0);
            if let BodyKind::BlackHole { total_accreted_mass, accretion_rate, .. } = b.kind {
                v.push(total_accreted_mass.to_bits());
                v.push(accretion_rate.to_bits());
            }
            v.push(b.slip_heat_total.to_bits());
        }
    }
    v
}

fn live_snapshot(hier: &IsoH) -> Vec<u64> {
    let mut v = Vec::new();
    for l in &hier.levels {
        let sim = &l.state;
        let m = sim.fields.mhd.as_ref().unwrap();
        for c in sim.geom.interior.iter() {
            v.push(sim.fields.cons.den.at(c).to_bits());
            for k in 0..3 {
                v.push(sim.fields.cons.mom[k].at(c).to_bits());
                v.push(m.bcell[k].at(c).to_bits());
            }
        }
        for (d, c) in face_bits(sim) {
            v.push(m.bface[d].at(c).to_bits());
        }
        v.push(sim.time.to_bits());
        v.push(sim.iteration);
        if let Some(im) = sim.immersed.as_ref() {
            v.push(im.bodies.get(0).slip_heat_total.to_bits());
        }
    }
    v
}

// one root step is one palindrome with the fine level subcycled inside H; the sink accretes and
// exports heat on the finest; a rejection at each point restores every level and replays as a
// clean step at the halved timestep.
#[test]
fn the_refined_2p5d_isothermal_palindrome_runs_and_replays_cleanly_after_a_rejection() {
    let mut hier = iso(Some(sink(slip_spec())));
    let fi = hier.levels.len() - 1;
    let fine_iter = hier.levels[fi].state.iteration;
    slip_schedule_arm();
    hier.step_root_with_dt(DT);
    let trace = slip_schedule_take().expect("armed");
    let ops: Vec<&str> = trace.iter().map(|(op, _)| *op).collect();
    assert_eq!(ops, ["D", "M", "H", "M", "D"], "{ops:?}");
    assert_eq!(hier.levels[fi].state.iteration, fine_iter + 2);
    let b = hier.levels[fi].state.immersed.as_ref().unwrap().bodies.get(0);
    assert!(b.slip_heat_total > 0.0, "no exported heat on the finest");
    assert!(matches!(b.kind, BodyKind::BlackHole { total_accreted_mass, .. } if total_accreted_mass > 0.0), "no accretion");
    for point in [SlipFailPoint::AfterOpening, SlipFailPoint::FineSubstep(2), SlipFailPoint::ClosingSolve] {
        let mut probe = iso(Some(sink(slip_spec())));
        let mut clean = iso(Some(sink(slip_spec())));
        slip_schedule_arm();
        slip_failure_arm(point);
        probe.step_root_with_dt(DT);
        let _ = slip_schedule_take();
        clean.step_root_with_dt(0.5 * DT);
        assert!(snapshot(&probe) == snapshot(&clean), "{point:?}: the replay differs from a clean step");
    }
}

#[test]
fn a_refined_2p5d_isothermal_slip_run_resumes_from_a_checkpoint_bit_for_bit() {
    let mut uninterrupted = iso(Some(sink(slip_spec())));
    for _ in 0..2 {
        uninterrupted.step_root_with_dt(DT);
    }
    let path = std::env::temp_dir().join(format!("symbi_refined_2p5d_restart_{}.h5", std::process::id()));
    let levels: Vec<&Iso> = uninterrupted.levels.iter().map(|l| &l.state).collect();
    symbi::sim::checkpoint::write_hierarchy_checkpoint(&levels, path.to_str().unwrap(), &Metadata::new()).expect("written");
    let mut resumed = iso(Some(sink(slip_spec())));
    resumed.restore_from_checkpoint(path.to_str().unwrap()).expect("restored");
    let _ = std::fs::remove_file(&path);
    resumed.prime();
    assert!(live_snapshot(&uninterrupted) == live_snapshot(&resumed), "the restored hierarchy differs from the written one");
    for _ in 0..2 {
        uninterrupted.step_root_with_dt(DT);
        resumed.step_root_with_dt(DT);
    }
    assert!(snapshot(&uninterrupted) == snapshot(&resumed), "the resumed run diverges");
}

fn l2_rel(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f64 = b.iter().map(|y| y * y).sum();
    (num / den.max(1e-300)).sqrt()
}

// fixed grid, refined timestep, the fine bulk at least three cells from the fine boundary: the
// gas, the in-plane faces, and the vertical cell field quarter their fixed-time difference when
// the timestep halves.
#[test]
fn the_refined_2p5d_induction_is_second_order_in_the_timestep_per_field() {
    let run = |dt: f64, nsteps: usize| -> Vec<(&'static str, Vec<f64>)> {
        let mut hier = adi(None);
        for _ in 0..nsteps {
            hier.step_root_with_dt(dt);
        }
        let sim = &hier.levels[1].state;
        let m = sim.fields.mhd.as_ref().unwrap();
        let interior = &sim.geom.interior;
        let deep = |c: &[isize; 2]| (0..2).all(|k| c[k] - interior.spaces[k].lo >= 3 && interior.spaces[k].hi - 1 - c[k] >= 3);
        let cells: Vec<[isize; 2]> = interior.iter().filter(|c| deep(c)).collect();
        let den: Vec<f64> = cells.iter().map(|c| *sim.fields.cons.den.at(*c)).collect();
        let bz: Vec<f64> = cells.iter().map(|c| *m.bcell[2].at(*c)).collect();
        let mut faces = Vec::new();
        for d in 0..2 {
            for c in &cells {
                faces.push(*m.bface[d].at(*c));
            }
        }
        vec![("fine density", den), ("fine in-plane faces", faces), ("fine vertical field", bz)]
    };
    let dt = 2.0 * DT;
    let runs = [run(dt, 4), run(dt / 2.0, 8), run(dt / 4.0, 16), run(dt / 8.0, 32)];
    for f in 0..runs[0].len() {
        let name = runs[0][f].0;
        let e: Vec<f64> = (0..3).map(|i| l2_rel(&runs[i][f].1, &runs[i + 1][f].1)).collect();
        let ratios = [e[0] / e[1].max(1e-300), e[1] / e[2].max(1e-300)];
        println!("{name}: diffs {:.3e} {:.3e} {:.3e} ratios {:.2} {:.2}", e[0], e[1], e[2], ratios[0], ratios[1]);
        assert!(e[0] > 1e-12, "{name}: vacuous");
        assert!(ratios[1] > 3.4, "{name}: not second order in the timestep: ratios {:.2} {:.2}", ratios[0], ratios[1]);
    }
}

// the fine seed is divergence-free: the face prolongation closes each fine cell's divergence
// whenever the parent's is closed, and the fine CT keeps it so.
#[test]
fn the_fine_seed_and_its_evolution_are_divergence_free() {
    let mut hier = adi(None);
    for (label, steps) in [("after seed + prime", 0), ("after two root steps", 2)] {
        for _ in 0..steps {
            hier.step_root_with_dt(DT);
        }
        let (dc, df) = (max_div_b(&hier.levels[0].state), max_div_b(&hier.levels[1].state));
        println!("[{label}] max|div B|: coarse {dc:.3e}, fine {df:.3e}");
        assert!(dc < 1e-12 && df < 1e-12, "{label}: div B left machine zero: coarse {dc:.3e}, fine {df:.3e}");
    }
}
