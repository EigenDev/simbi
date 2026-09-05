// =============================================================================
// refinement_mhd_slip_imhd.rs
//
// the refined magnetic-slip palindrome under the isothermal closure: a two-level Cartesian
// hierarchy whose finest level owns a slip sink on an isothermal gas. no level carries an energy
// field or a gas-energy staging; the finest level alone drains, slips, and books the exported
// heat while the proxy books nothing; the slip half-step on the finest loses exactly the heat it
// exports and the synchronization restricts it; a rejected attempt at any point restores every
// level and the receipts and replays as a clean step; a checkpoint resumes bit for bit with the
// receipts; the single-level composition is the manual sequence of its operators bit for bit;
// transparent and resistive sinks keep the ordinary advance.
// =============================================================================

use symbi::regimes::substrate_isothermal_mhd::IsothermalMhdSubstrateKernelSet;
use symbi::sim::refinement::hierarchy::{slip_failure_arm, slip_schedule_arm, slip_schedule_take, SlipFailPoint};
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::Isothermal;
use symbi_hydro::isothermal_mhd::IsothermalMhd;
use symbi_hydro::mhd_state::MhdPrimG;
use symbi_hydro::quantity::Density;
use symbi_hydro::state::PrimG;
use symbi_ib::{Body, BodyCollection, BodyKind, MagneticSpec, SurfaceSpec};
use symbi_io::Metadata;
use symbi_refinement::refinement::transfer::restrict_cell_field;
use symbi_grid::Field;
use symbi_sim::substrate_seam::KernelSet;
use symbi_substrate::regimes::mhd_substrate::{magnetic_slip_commit, magnetic_slip_solve};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<IsothermalMhd, 3, 3, Cartesian, Isothermal<f64>, CpuSpace, HostMemory, f64>;
type Kset = IsothermalMhdSubstrateKernelSet<HostMemory, f64, 3>;
type Hier = Hierarchy<IsothermalMhd, 3, 3, Cartesian, Isothermal<f64>, CpuSpace, HostMemory, Kset>;

const N: usize = 32;
const CS: f64 = 1.0;
const DX_FINE: f64 = 1.0 / (2.0 * N as f64);
const DT: f64 = 1.0e-3;
const VEL: [f64; 3] = [0.3, 0.2, 0.1];

fn slip_spec() -> MagneticSpec {
    MagneticSpec::Slip {
        diffusivity_ratio: 2.0,
        shell_width: DX_FINE,
        slip_length_ratio: 1.0,
        field_regularization: 0.1,
        placement: 0.0,
    }
}

fn potential(amp: f64, k: f64) -> impl Fn(f64, f64) -> f64 + Copy {
    let phase = 0.25 * std::f64::consts::PI;
    move |x: f64, y: f64| amp * (k * x + phase).sin() * (k * y + phase).sin()
}
fn face_of_potential(az: impl Fn(f64, f64) -> f64 + Copy, dx: f64) -> impl Fn(usize, [f64; 3]) -> f64 + Copy {
    move |axis: usize, [x, y, _z]: [f64; 3]| match axis {
        0 => (az(x, y + 0.5 * dx) - az(x, y - 0.5 * dx)) / dx,
        1 => -(az(x + 0.5 * dx, y) - az(x - 0.5 * dx, y)) / dx,
        _ => 0.0,
    }
}

fn sink(idx: usize, position: [f64; 3], r_acc: f64, magnetic: MagneticSpec) -> Body<f64, 3> {
    Body::black_hole(idx, Tensor::new(position), Tensor::zeros(), 1.0, r_acc, 0.05, 1.0, 1.0, r_acc)
        .with_surface(SurfaceSpec::Drain)
        .with_magnetic(magnetic)
}

fn one_sink(magnetic: MagneticSpec) -> BodyCollection<f64, 3> {
    BodyCollection::new().add(sink(0, [0.0; 3], 2.0 * DX_FINE, magnetic))
}

fn root(n: usize) -> Sim {
    let dx = 1.0 / n as f64;
    let k = 2.0 * std::f64::consts::PI;
    let face = face_of_potential(potential(0.3 / k, k), dx);
    Sim::build(IsothermalMhd, Isothermal { cs: CS }, Cartesian)
        .cells([n; 3])
        .origin([-0.5; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .allocate()
        .expect("root construction")
        .set_initial(move |[x, y, z]| {
            let bx = 0.5 * (face(0, [x - 0.5 * dx, y, z]) + face(0, [x + 0.5 * dx, y, z]));
            let by = 0.5 * (face(1, [x, y - 0.5 * dx, z]) + face(1, [x, y + 0.5 * dx, z]));
            MhdPrimG::<f64, 3, IsoModel>::new(
                PrimG::isothermal(Density(1.0 + 0.1 * (k * x).sin()), Tensor::new(VEL)),
                Tensor::new([bx, by, 0.0]),
            )
        })
        .seed_faces(face)
        .build()
}

// a 32^3 isothermal root over [-1/2, 1/2]^3 refined once over [-7/16, 7/16], the fine level
// seeded by the divergence-free prolongation, the bodies on every level, primed.
fn two_level(bodies: BodyCollection<f64, 3>) -> Hier {
    let coarse = root(N);
    let kset = |s: &Sim| Kset::new(CS, 0.3, 1.0, &s.geom.allocated);
    let ck = kset(&coarse);
    let regions = [RefinementRegion {
        x_lo: [-0.4375; 3],
        x_hi: [0.4375; 3],
    }];
    let hier = Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, kset).unwrap();
    hier.seed_fine_from_coarse().expect("fine seed");
    let mut hier = hier.with_bodies(bodies);
    hier.prime();
    hier
}

fn finest(hier: &Hier) -> usize {
    hier.levels.len() - 1
}

fn body_state(hier: &Hier, level: usize) -> (f64, f64, f64) {
    let b = hier.levels[level].state.immersed.as_ref().unwrap().bodies.get(0);
    let accreted = match b.kind {
        BodyKind::BlackHole { total_accreted_mass, .. } => total_accreted_mass,
        _ => 0.0,
    };
    (accreted, b.slip_heat_total, b.slip_heat_rate)
}

// the physical faces of a level: the lower faces of its interior cells plus the closing face
// along each axis (the finest level's boundary faces are coarse-fine faces, each physical once).
fn physical_faces(hier: &Hier, level: usize) -> Vec<(usize, [isize; 3])> {
    let sim = &hier.levels[level].state;
    let m = sim.fields.mhd.as_ref().unwrap();
    let interior = &sim.geom.interior;
    let mut v = Vec::new();
    for d in 0..3 {
        for c in m.bface[d].domain().iter() {
            let inside = (0..3).all(|a| c[a] >= interior.spaces[a].lo && c[a] < interior.spaces[a].hi + (a == d) as isize);
            if inside {
                v.push((d, c));
            }
        }
    }
    v
}

fn face_values(hier: &Hier, level: usize, faces: &[(usize, [isize; 3])]) -> Vec<f64> {
    let m = hier.levels[level].state.fields.mhd.as_ref().unwrap();
    faces.iter().map(|(d, c)| *m.bface[*d].at(*c)).collect()
}

// the change of the face-Hodge magnetic energy, summed face by face so the sum carries the
// roundoff of the changes alone, volume-weighted.
fn face_energy_change(hier: &Hier, level: usize, faces: &[(usize, [isize; 3])], before: &[f64]) -> f64 {
    let vol: f64 = hier.levels[level].state.geom.dx.iter().product();
    // (b1 - b0)(b1 + b0)/2: the difference of two nearby values is exact, so each term carries
    // the roundoff of the change alone.
    face_values(hier, level, faces).iter().zip(before).map(|(b1, b0)| 0.5 * (b1 - b0) * (b1 + b0) * vol).sum()
}

// every stored value of every level's conserved, primitive, cell, and face fields, the clocks,
// and the body state with its receipts, each with its label.
fn labeled_snapshot(hier: &Hier) -> Vec<(String, u64)> {
    let mut v = Vec::new();
    for (ll, l) in hier.levels.iter().enumerate() {
        let sim = &l.state;
        let m = sim.fields.mhd.as_ref().unwrap();
        let interior = &sim.geom.interior;
        for c in sim.geom.allocated.iter() {
            let inside = (0..3).all(|a| c[a] >= interior.spaces[a].lo && c[a] < interior.spaces[a].hi);
            let where_ = if inside { "interior" } else { "ghost" };
            v.push((format!("L{ll} den {where_} {c:?}"), sim.fields.cons.den.at(c).to_bits()));
            for k in 0..3 {
                v.push((format!("L{ll} mom{k} {where_} {c:?}"), sim.fields.cons.mom[k].at(c).to_bits()));
            }
            v.push((format!("L{ll} rho {where_} {c:?}"), sim.fields.prim.rho.at(c).to_bits()));
            for d in 0..3 {
                v.push((format!("L{ll} bc{d} {where_} {c:?}"), m.bcell[d].at(c).to_bits()));
            }
        }
        for d in 0..3 {
            for c in m.bface[d].domain().iter() {
                let inside = (0..3).all(|a| c[a] >= interior.spaces[a].lo && c[a] < interior.spaces[a].hi + (a == d) as isize);
                let where_ = if inside { "interior" } else { "ghost" };
                v.push((format!("L{ll} bface{d} {where_} {c:?}"), m.bface[d].at(c).to_bits()));
            }
        }
        v.push((format!("L{ll} time"), sim.time.to_bits()));
        v.push((format!("L{ll} iteration"), sim.iteration));
        if let Some(im) = sim.immersed.as_ref() {
            for bb in 0..im.bodies.len() {
                let b = im.bodies.get(bb);
                for a in 0..3 {
                    v.push((format!("L{ll} body{bb} pos{a}"), b.position[a].to_bits()));
                    v.push((format!("L{ll} body{bb} vel{a}"), b.velocity[a].to_bits()));
                }
                if let BodyKind::BlackHole { total_accreted_mass, accretion_rate, .. } = b.kind {
                    v.push((format!("L{ll} body{bb} accreted"), total_accreted_mass.to_bits()));
                    v.push((format!("L{ll} body{bb} accretion rate"), accretion_rate.to_bits()));
                }
                v.push((format!("L{ll} body{bb} slip heat"), b.slip_heat_total.to_bits()));
                v.push((format!("L{ll} body{bb} slip heat rate"), b.slip_heat_rate.to_bits()));
            }
        }
    }
    v
}

fn snapshot(hier: &Hier) -> Vec<u64> {
    labeled_snapshot(hier).into_iter().map(|(_, b)| b).collect()
}

fn first_mismatch(a: &[(String, u64)], b: &[(String, u64)], live_only: bool) -> Option<String> {
    a.iter().zip(b).find(|((la, x), (_, y))| x != y && !(live_only && la.contains("ghost"))).map(|((la, x), (_, y))| format!("{la}: {:e} vs {:e}", f64::from_bits(*x), f64::from_bits(*y)))
}

fn assert_covered_is_restriction(hier: &Hier, label: &str) {
    let coarse = &hier.levels[0].state;
    let fine = &hier.levels[1].state;
    let cov = hier.levels[0].coverage.as_ref().unwrap();
    let scratch: Field<f64, 3, HostMemory> = Field::zeros(&coarse.geom.allocated).unwrap();
    restrict_cell_field(&fine.fields.cons.den, &scratch, cov);
    for cell in cov.iter() {
        assert!(*coarse.fields.cons.den.at(cell) == *scratch.at(cell), "{label}: density at {cell:?} is not the restriction");
    }
}

#[test]
fn no_level_carries_an_energy_field_or_a_gas_energy_staging() {
    let hier = two_level(one_sink(slip_spec()));
    for (ll, l) in hier.levels.iter().enumerate() {
        assert!(l.state.fields.cons.nrg_field().is_none(), "level {ll} carries an energy field");
        if let Some(ws) = l.state.fields.mhd.as_ref().unwrap().magnetic_slip.as_ref() {
            assert!(ws.gas_energy.is_none(), "level {ll} stages a gas energy");
            assert_eq!(ll, finest(&hier), "a coarser level carries slip storage");
        }
    }
    assert!(hier.levels[finest(&hier)].state.fields.mhd.as_ref().unwrap().magnetic_slip.is_some(), "the finest carries no slip workspace");
}

// one root step is one palindrome with the fine level subcycled inside H; the finest sink
// accretes and books exported heat, the proxy is gravitational and books nothing.
#[test]
fn the_refined_isothermal_root_step_runs_one_palindrome_and_books_the_heat_on_the_finest() {
    let mut hier = two_level(one_sink(slip_spec()));
    let fi = finest(&hier);
    let fine_iter = hier.levels[fi].state.iteration;
    slip_schedule_arm();
    hier.step_root_with_dt(DT);
    let trace = slip_schedule_take().expect("armed");
    let ops: Vec<&str> = trace.iter().map(|(op, _)| *op).collect();
    assert_eq!(ops, ["D", "M", "H", "M", "D"], "the root step is not one palindrome: {ops:?}");
    assert_eq!(hier.levels[fi].state.iteration, fine_iter + 2, "the fine level was not subcycled twice inside H");
    let (accreted, heat, rate) = body_state(&hier, fi);
    assert!(accreted > 0.0, "the finest sink accreted nothing");
    assert!(heat > 0.0 && rate > 0.0, "the finest sink exported no heat: total {heat}, rate {rate}");
    let proxy = hier.levels[0].state.immersed.as_ref().unwrap().bodies.get(0);
    assert!(matches!(proxy.kind, BodyKind::Gravitational { .. }), "the proxy carries a sink");
    assert_eq!(proxy.slip_heat_total, 0.0, "the proxy booked heat");
    assert_covered_is_restriction(&hier, "after the root step");
}

// one slip half-step on the finest: the face-Hodge magnetic energy of the finest decreases by
// exactly the heat the sink books, and the synchronization restricts the covered coarse state.
// the identity holds to the solve's convergence: the candidate satisfies its system to a residual
// of tol times the right-hand side's norm, of order |B|, while this sink's heat is eight orders
// below the field energy, so the solve runs at a tolerance of 1e-14 here and the identity is read
// to 1e-8 of the heat.
#[test]
fn the_finest_slip_half_step_loses_exactly_the_heat_it_exports() {
    let hier = two_level(one_sink(slip_spec()));
    let fi = finest(&hier);
    let faces = physical_faces(&hier, fi);
    let before = face_values(&hier, fi, &faces);
    let l = &hier.levels[fi];
    let receipt = magnetic_slip_solve::<3, 3, HostMemory, f64>(&l.state, 0.5 * DT, CS, 1e-14, 2000);
    assert!(receipt.converged, "the isothermal slip solve did not converge: {receipt:?}");
    magnetic_slip_commit::<3, 3, HostMemory, f64>(&l.state, 0.5 * DT, CS);
    let dm = face_energy_change(&hier, fi, &faces, &before);
    let booked = l.state.immersed.as_ref().unwrap().diagnostics.consolidate()[0].slip_heat_delta;
    println!("refined isothermal slip: exported heat {booked:.9e}, -dM_face {:.9e}, residual {:.3e}", -dm, (booked + dm).abs());
    assert!(booked > 0.0, "the finest sink exported no heat");
    assert!((booked + dm).abs() < 1e-8 * booked, "the isothermal theorem fails on the finest: exported {booked:.12e}, -dM_face {:.12e}", -dm);
    hier.sync_all_fine_to_coarse();
    assert_covered_is_restriction(&hier, "after the slip and the synchronization");
}

// a rejected attempt at each point of the composition restores every level, its clocks, the body
// state, and the heat receipts bit for bit, and the replay at the halved timestep equals a clean
// step taken directly at that timestep.
#[test]
fn a_rejected_isothermal_attempt_restores_the_receipts_and_replays_cleanly() {
    for point in [SlipFailPoint::AfterOpening, SlipFailPoint::FineSubstep(2), SlipFailPoint::ClosingSolve] {
        let mut probe = two_level(one_sink(slip_spec()));
        let mut clean = two_level(one_sink(slip_spec()));
        slip_schedule_arm();
        slip_failure_arm(point);
        probe.step_root_with_dt(DT);
        let trace = slip_schedule_take().expect("armed");
        let accepted: Vec<&str> = trace.iter().rev().take(5).map(|(op, _)| *op).collect::<Vec<_>>().into_iter().rev().collect();
        assert_eq!(accepted, ["D", "M", "H", "M", "D"], "{point:?}: the replay is not a complete palindrome");
        clean.step_root_with_dt(0.5 * DT);
        assert!(body_state(&probe, finest(&probe)).1 > 0.0, "{point:?}: the replay booked no heat");
        assert!(snapshot(&probe) == snapshot(&clean), "{point:?}: the replay after a rejection differs from a clean step at the halved timestep");
    }
}

// a checkpoint written after accepted steps resumes bit for bit, the exported-heat receipts
// included.
#[test]
fn a_refined_isothermal_slip_run_resumes_from_a_checkpoint_bit_for_bit() {
    let mut uninterrupted = two_level(one_sink(slip_spec()));
    for _ in 0..2 {
        uninterrupted.step_root_with_dt(DT);
    }
    let fi = finest(&uninterrupted);
    let (_, heat_at_write, rate_at_write) = body_state(&uninterrupted, fi);
    assert!(heat_at_write > 0.0, "nothing exported before the checkpoint; the restart of the receipt is vacuous");
    let path = std::env::temp_dir().join(format!("symbi_refined_iso_slip_restart_{}.h5", std::process::id()));
    let levels: Vec<&Sim> = uninterrupted.levels.iter().map(|l| &l.state).collect();
    symbi::sim::checkpoint::write_hierarchy_checkpoint(&levels, path.to_str().unwrap(), &Metadata::new()).expect("checkpoint written");
    let mut resumed = two_level(one_sink(slip_spec()));
    resumed.restore_from_checkpoint(path.to_str().unwrap()).expect("checkpoint restored");
    let _ = std::fs::remove_file(&path);
    let (_, heat, rate) = body_state(&resumed, fi);
    assert_eq!(heat.to_bits(), heat_at_write.to_bits(), "the exported heat did not survive the restart");
    assert_eq!(rate.to_bits(), rate_at_write.to_bits(), "the heat rate did not survive the restart");
    resumed.prime();
    if let Some(m) = first_mismatch(&labeled_snapshot(&uninterrupted), &labeled_snapshot(&resumed), true) {
        panic!("the restored hierarchy differs from the written one at {m}");
    }
    for _ in 0..2 {
        uninterrupted.step_root_with_dt(DT);
        resumed.step_root_with_dt(DT);
    }
    if let Some(m) = first_mismatch(&labeled_snapshot(&uninterrupted), &labeled_snapshot(&resumed), false) {
        panic!("the resumed run diverges from the uninterrupted one at {m}");
    }
}

// on a single grid the coupled step is the manual sequence D(dt/2) M(dt/2) H(dt) M(dt/2) D(dt/2)
// of the level's own operators, each followed by the primitive and ghost rebuild, bit for bit.
#[test]
fn the_single_level_isothermal_composition_is_the_manual_operator_sequence_bit_for_bit() {
    let build = || {
        let sim = root(16).with_bodies(one_sink(slip_spec()));
        let kset = Kset::new(CS, 0.3, 1.0, &sim.geom.allocated);
        let mut hier = Hierarchy::single(sim, kset);
        hier.prime();
        hier
    };
    let mut production = build();
    let mut manual = build();
    production.step_root_with_dt(DT);
    {
        let rebuild = |h: &Hier| {
            let l = &h.levels[0];
            l.kernels.c2p(&l.state);
            l.kernels.ghost_fill(&l.state);
        };
        manual.drain_and_rebuild(0, 0.5 * DT);
        let l = &manual.levels[0];
        assert!(l.kernels.magnetic_slip_step(&l.state, 0.5 * DT));
        rebuild(&manual);
        assert!(!manual.hydro_map(0, DT), "H rejected on the smooth field");
        let l = &manual.levels[0];
        assert!(l.kernels.magnetic_slip_step(&l.state, 0.5 * DT));
        rebuild(&manual);
        manual.drain_and_rebuild(0, 0.5 * DT);
        manual.root_body_step(DT);
    }
    let fields = |h: &Hier| -> Vec<u64> {
        let sim = &h.levels[0].state;
        let m = sim.fields.mhd.as_ref().unwrap();
        let mut v = Vec::new();
        for c in sim.geom.interior.iter() {
            v.push(sim.fields.cons.den.at(c).to_bits());
            for k in 0..3 {
                v.push(sim.fields.cons.mom[k].at(c).to_bits());
                v.push(m.bcell[k].at(c).to_bits());
                v.push(m.bface[k].at(c).to_bits());
            }
        }
        let b = sim.immersed.as_ref().unwrap().bodies.get(0);
        v.push(b.slip_heat_total.to_bits());
        v
    };
    assert!(body_state(&production, 0).1 > 0.0, "the single-level step exported no heat");
    assert!(fields(&production) == fields(&manual), "the single-level composition differs from the manual operator sequence");
}

#[test]
fn transparent_and_resistive_refined_isothermal_sinks_keep_the_ordinary_advance() {
    for (label, magnetic) in [("transparent", MagneticSpec::None), ("resistive", MagneticSpec::Resistive { eta: 0.02 })] {
        let mut hier = two_level(one_sink(magnetic));
        slip_schedule_arm();
        hier.step_root_with_dt(DT);
        let trace = slip_schedule_take().expect("armed");
        assert!(trace.is_empty(), "{label}: the ordinary advance recorded a coupled schedule");
        let (accreted, heat, _) = body_state(&hier, finest(&hier));
        assert!(accreted > 0.0, "{label}: no accretion on the finest");
        assert_eq!(heat, 0.0, "{label}: a sink without the slip booked heat");
        assert!(snapshot(&hier).iter().all(|b| f64::from_bits(*b).is_finite() || *b < 1u64 << 52), "{label}: non-finite state");
    }
}
