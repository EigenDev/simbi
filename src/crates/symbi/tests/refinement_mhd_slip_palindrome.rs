// =============================================================================
// refinement_mhd_slip_palindrome.rs
//
// the coupled root step D(dt/2) M(dt/2) H(dt) M(dt/2) D(dt/2) on a refined hierarchy: the
// finest level alone drains and slips, the ideal-MHD step H runs the complete level recursion
// with its fine subcycling while every tail's drain is withheld, the covered parent regions are
// synchronized after each split pair, body feedback and motion happen once per accepted root
// step, and a rejected attempt at any point of the composition restores every level bit for bit
// and replays as a clean step at the reduced timestep.
// =============================================================================

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::refinement::hierarchy::{slip_failure_arm, slip_schedule_arm, slip_schedule_take, SlipFailPoint};
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, BodyKind, MagneticSpec, SurfaceSpec};
use symbi_refinement::refinement::transfer::restrict_cell_field;
use symbi_grid::Field;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;
type Kset = NewtonianMhdSubstrateKernelSet<HostMemory, f64, 3>;
type Hier = Hierarchy<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset>;

const N: usize = 32;
const GAMMA: f64 = 5.0 / 3.0;
const DX_FINE: f64 = 1.0 / (2.0 * N as f64);
const DT: f64 = 1.0e-3;

fn slip_spec() -> MagneticSpec {
    MagneticSpec::Slip {
        diffusivity_ratio: 2.0,
        shell_width: 0.125 * DX_FINE,
        slip_length_ratio: 1.0,
        field_regularization: 0.1,
        placement: 0.0,
    }
}

// the vector potential's phase keeps the sink at the box center off the field's null: at the
// center sin(k x + pi/4) sin(k y + pi/4) = -1/2, so the in-plane field is finite where the slip acts.
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

// a discretely solenoidal field on a 32^3 root refined over [1/8, 7/8], a sink of half a fine
// cell at the center wearing the given coupling; the fine level seeded by the divergence-free
// prolongation, then primed.
fn two_level(magnetic: MagneticSpec) -> Hier {
    let dx = 1.0 / N as f64;
    let k = 2.0 * std::f64::consts::PI;
    let face = face_of_potential(potential(0.3 / k, k), dx);
    let kset = |s: &Sim| Kset::new(GAMMA, 0.3, 1.0, &s.geom.allocated);
    let coarse = Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 3])
        .origin([0.0; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .allocate()
        .expect("root construction")
        .set_initial(move |[x, y, z]| {
            let bx = 0.5 * (face(0, [x - 0.5 * dx, y, z]) + face(0, [x + 0.5 * dx, y, z]));
            let by = 0.5 * (face(1, [x, y - 0.5 * dx, z]) + face(1, [x, y + 0.5 * dx, z]));
            MhdPrim::new(
                Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0 + 0.1 * (k * x).sin())),
                Tensor::new([bx, by, 0.0]),
            )
        })
        .seed_faces(face)
        .build();
    let ck = kset(&coarse);
    let regions = [RefinementRegion {
        x_lo: [0.125; 3],
        x_hi: [0.875; 3],
    }];
    let hier = Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, kset).unwrap();
    hier.seed_fine_from_coarse().expect("fine seed");
    let body = Body::black_hole(0, Tensor::new([0.5; 3]), Tensor::zeros(), 1.0, 0.5 * DX_FINE, 0.05, 1.0, 1.0, 0.5 * DX_FINE)
        .with_surface(SurfaceSpec::Drain)
        .with_magnetic(magnetic);
    let mut hier = hier.with_bodies(BodyCollection::new().add(body));
    hier.prime();
    hier
}

// every stored value of every level's conserved, cell, and face fields, plus the body state.
fn snapshot(hier: &Hier) -> Vec<f64> {
    let mut v = Vec::new();
    for l in &hier.levels {
        let sim = &l.state;
        let m = sim.fields.mhd.as_ref().unwrap();
        for c in sim.geom.allocated.iter() {
            v.push(*sim.fields.cons.den.at(c));
            for k in 0..3 {
                v.push(*sim.fields.cons.mom[k].at(c));
            }
            v.push(*sim.fields.cons.nrg_field().unwrap().at(c));
            v.push(*sim.fields.prim.rho.at(c));
            v.push(*sim.fields.prim.pre_field().unwrap().at(c));
            for d in 0..3 {
                v.push(*m.bcell[d].at(c));
            }
        }
        for d in 0..3 {
            for c in m.bface[d].domain().iter() {
                v.push(*m.bface[d].at(c));
            }
        }
        v.push(sim.time);
        v.push(sim.iteration as f64);
        if let Some(im) = sim.immersed.as_ref() {
            let b = im.bodies.get(0);
            for a in 0..3 {
                v.push(b.position[a]);
                v.push(b.velocity[a]);
            }
            if let BodyKind::BlackHole { total_accreted_mass, accretion_rate, .. } = b.kind {
                v.push(total_accreted_mass);
                v.push(accretion_rate);
            }
        }
    }
    v
}

// the label of every snapshot entry: (field, level, coordinate or tag).
fn labels(hier: &Hier) -> Vec<(String, String)> {
    let mut v = Vec::new();
    for (ll, l) in hier.levels.iter().enumerate() {
        let sim = &l.state;
        let m = sim.fields.mhd.as_ref().unwrap();
        let interior = &sim.geom.interior;
        for c in sim.geom.allocated.iter() {
            let inside = (0..3).all(|a| c[a] >= interior.spaces[a].lo && c[a] < interior.spaces[a].hi);
            let where_ = if inside { "interior" } else { "ghost" };
            for nm in ["den", "mom0", "mom1", "mom2", "nrg", "rho", "pre", "bc0", "bc1", "bc2"] {
                v.push((format!("L{ll} {nm} {where_}"), format!("{c:?}")));
            }
        }
        for d in 0..3 {
            for c in m.bface[d].domain().iter() {
                v.push((format!("L{ll} bface{d}"), format!("{c:?}")));
            }
        }
        v.push((format!("L{ll} time"), String::new()));
        v.push((format!("L{ll} iteration"), String::new()));
        if sim.immersed.is_some() {
            for a in 0..3 {
                v.push((format!("L{ll} body pos{a}"), String::new()));
                v.push((format!("L{ll} body vel{a}"), String::new()));
            }
            if ll + 1 == hier.levels.len() {
                v.push((format!("L{ll} accreted mass"), String::new()));
                v.push((format!("L{ll} accretion rate"), String::new()));
            }
        }
    }
    v
}

fn accreted(hier: &Hier) -> f64 {
    let fi = hier.levels.len() - 1;
    match hier.levels[fi].state.immersed.as_ref().unwrap().bodies.get(0).kind {
        BodyKind::BlackHole { total_accreted_mass, .. } => total_accreted_mass,
        _ => panic!("the finest body is not a sink"),
    }
}

#[test]
fn the_refined_root_step_runs_one_palindrome_with_the_fine_level_subcycled_inside_h() {
    let mut hier = two_level(slip_spec());
    let (root_iter, fine_iter) = (hier.levels[0].state.iteration, hier.levels[1].state.iteration);
    let mass_before = accreted(&hier);
    slip_schedule_arm();
    hier.step_root_with_dt(DT);
    let trace = slip_schedule_take().expect("armed");
    let ops: Vec<&str> = trace.iter().map(|(op, _)| *op).collect();
    assert_eq!(ops, ["D", "M", "H", "M", "D"], "the root step is not one palindrome: {ops:?}");
    let sum = |name: &str| -> f64 { trace.iter().filter(|(op, _)| *op == name).map(|(_, d)| *d).sum() };
    for name in ["D", "M", "H"] {
        assert!((sum(name) - DT).abs() < 1e-15, "{name} advances {} instead of dt", sum(name));
    }
    assert_eq!(trace.iter().filter(|(op, _)| *op == "M").count(), 2, "M ran inside the hierarchy operator");
    assert_eq!(hier.levels[0].state.iteration, root_iter + 1, "the root clock did not advance once");
    assert_eq!(hier.levels[1].state.iteration, fine_iter + 2, "the fine level was not subcycled twice inside H");
    assert!(accreted(&hier) > mass_before, "the finest sink recorded no accretion");
    assert!(matches!(hier.levels[0].state.immersed.as_ref().unwrap().bodies.get(0).kind, BodyKind::Gravitational { .. }));
}

// after the accepted root step the covered coarse state is the restriction of the fine state: the
// closing synchronization left the redundancy exact.
#[test]
fn the_covered_coarse_state_is_the_restriction_after_the_refined_root_step() {
    let mut hier = two_level(slip_spec());
    hier.step_root_with_dt(DT);
    let coarse = &hier.levels[0].state;
    let fine = &hier.levels[1].state;
    let cov = hier.levels[0].coverage.as_ref().unwrap();
    let scratch: Field<f64, 3, HostMemory> = Field::zeros(&coarse.geom.allocated).unwrap();
    for (f, c, name) in [
        (&fine.fields.cons.den, &coarse.fields.cons.den, "density"),
        (fine.fields.cons.nrg_field().unwrap(), coarse.fields.cons.nrg_field().unwrap(), "energy"),
    ] {
        restrict_cell_field(f, &scratch, cov);
        for cell in cov.iter() {
            assert!(*c.at(cell) == *scratch.at(cell), "{name} at {cell:?} is not the restriction after the step");
        }
    }
}

// a transparent or resistive sink on the refined hierarchy takes the ordinary advance: no
// schedule trace, the fine level subcycled, accretion on the finest, and the covered coarse
// state the restriction (the ordinary path's own invariant).
#[test]
fn a_refined_sink_without_the_slip_keeps_the_ordinary_advance() {
    for (label, magnetic) in [("transparent", MagneticSpec::None), ("resistive", MagneticSpec::Resistive { eta: 0.05 })] {
        let mut hier = two_level(magnetic);
        let fine_iter = hier.levels[1].state.iteration;
        slip_schedule_arm();
        hier.step_root_with_dt(DT);
        let trace = slip_schedule_take().expect("armed");
        assert!(trace.is_empty(), "{label}: the ordinary advance recorded a coupled schedule: {trace:?}");
        assert_eq!(hier.levels[1].state.iteration, fine_iter + 2, "{label}: the fine level was not subcycled");
        assert!(accreted(&hier) > 0.0, "{label}: no accretion on the finest");
    }
}

// a rejected attempt at each point of the composition restores every level, its clocks, and the
// body state bit for bit, and the replay at the halved timestep equals a clean step taken
// directly at that timestep.
#[test]
fn a_rejected_attempt_at_any_point_restores_the_hierarchy_and_replays_cleanly() {
    for point in [SlipFailPoint::AfterOpening, SlipFailPoint::FineSubstep(2), SlipFailPoint::ClosingSolve] {
        let mut probe = two_level(slip_spec());
        let mut clean = two_level(slip_spec());
        assert!(snapshot(&probe) == snapshot(&clean), "the fixture is not deterministic");
        slip_schedule_arm();
        slip_failure_arm(point);
        probe.step_root_with_dt(DT);
        let trace = slip_schedule_take().expect("armed");
        // the rejected attempt and the accepted replay both trace: two palindromes, or a
        // truncated first one followed by a complete second.
        assert!(trace.iter().filter(|(op, _)| *op == "H").count() >= 1, "{point:?}: no attempt ran");
        let accepted: Vec<&(&str, f64)> = trace.iter().rev().take(5).collect::<Vec<_>>().into_iter().rev().collect();
        assert_eq!(accepted.iter().map(|(op, _)| *op).collect::<Vec<_>>(), ["D", "M", "H", "M", "D"], "{point:?}: the replay is not a complete palindrome");
        assert!((accepted[2].1 - 0.5 * DT).abs() < 1e-15, "{point:?}: the replay did not halve the timestep");
        clean.step_root_with_dt(0.5 * DT);
        let (a, b) = (snapshot(&probe), snapshot(&clean));
        let labels = labels(&probe);
        let mismatches: Vec<usize> = a.iter().zip(&b).enumerate().filter(|(_, (x, y))| x != y).map(|(i, _)| i).collect();
        if let Some(&i) = mismatches.first() {
            let mut by_label: std::collections::BTreeMap<String, usize> = Default::default();
            for &j in &mismatches {
                *by_label.entry(labels[j].0.clone()).or_default() += 1;
            }
            panic!(
                "{point:?}: the replay departs from a clean step at entry {i} = {:?} ({} vs {}); mismatches by field: {by_label:?}",
                labels[i], a[i], b[i]
            );
        }
    }
}
