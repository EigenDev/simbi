// =============================================================================
// refinement_mhd_slip_bookkeeping.rs
//
// the ledgers of the refined magnetic-slip palindrome. on a two-level Cartesian hierarchy whose
// finest level owns a slip sink: the ideal-MHD hierarchy operator conserves the leaf-domain totals,
// the finest drain removes exactly the mass and energy it books and the proxy books nothing, the
// root step's leaf-domain mass ledger closes on the sink's receipt, the slip half-step conserves
// the extended energy on the finest and the synchronization restricts it, a checkpoint resumes the
// refined run bit for bit, the composition is second order in time per field, and the composition
// runs under outflow boundaries and for a prescribed binary of slip sinks.
// =============================================================================

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_grid::Field;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{BinaryParams, Body, BodyCollection, BodyKind, MagneticSpec, SurfaceSpec};
use symbi_io::Metadata;
use symbi_refinement::refinement::transfer::restrict_cell_field;
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;
type Kset = NewtonianMhdSubstrateKernelSet<HostMemory, f64, 3>;
type Hier = Hierarchy<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset>;

const N: usize = 32;
const GAMMA: f64 = 5.0 / 3.0;
const DX_FINE: f64 = 1.0 / (2.0 * N as f64);
const DT: f64 = 1.0e-3;
// a uniform drift keeps every mass flux's sign definite, so the contact EMF's upwind selection is a
// smooth function of the state and the ideal-MHD operator's temporal order is readable.
const VEL: [f64; 3] = [0.3, 0.2, 0.1];

fn slip_spec() -> MagneticSpec {
    MagneticSpec::Slip {
        diffusivity_ratio: 2.0,
        shell_width: 0.125 * DX_FINE,
        slip_length_ratio: 1.0,
        field_regularization: 0.1,
        placement: 0.0,
    }
}

// the vector potential's phase keeps a sink at the origin off the field's null: sin(pi/4)^2 = 1/2
// there, so the in-plane field is finite where the slip acts.
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

// a sink of radius `r_acc` at `position` with the given mass, velocity, and coupling.
fn sink_of_radius(idx: usize, position: [f64; 3], velocity: [f64; 3], mass: f64, r_acc: f64, magnetic: MagneticSpec) -> Body<f64, 3> {
    Body::black_hole(idx, Tensor::new(position), Tensor::new(velocity), mass, r_acc, 0.05, 1.0, 1.0, r_acc)
        .with_surface(SurfaceSpec::Drain)
        .with_magnetic(magnetic)
}

// a sink of half a fine cell at `position` with the given mass, velocity, and coupling.
fn sink(idx: usize, position: [f64; 3], velocity: [f64; 3], mass: f64, magnetic: MagneticSpec) -> Body<f64, 3> {
    sink_of_radius(idx, position, velocity, mass, 0.5 * DX_FINE, magnetic)
}

// a sink of three fine cells with a slip shell one fine cell wide: the mask seam and the shell are
// resolved, so the staggered field's temporal order reads at this resolution.
fn resolved_slip_sink() -> BodyCollection<f64, 3> {
    let magnetic = MagneticSpec::Slip {
        diffusivity_ratio: 2.0,
        shell_width: DX_FINE,
        slip_length_ratio: 1.0,
        field_regularization: 0.1,
        placement: 0.0,
    };
    BodyCollection::new().add(sink_of_radius(0, [0.0; 3], [0.0; 3], 1.0, 3.0 * DX_FINE, magnetic))
}

// a discretely solenoidal field on a 32^3 root over [-1/2, 1/2]^3 refined over `region` (the same
// interval on every axis), the given root boundary, the fine level seeded by the divergence-free
// prolongation; the bodies attached to every level when given; primed.
fn build(boundary: BoundaryType, region: [f64; 2], bodies: Option<BodyCollection<f64, 3>>) -> Hier {
    let dx = 1.0 / N as f64;
    let k = 2.0 * std::f64::consts::PI;
    let face = face_of_potential(potential(0.3 / k, k), dx);
    let kset = |s: &Sim| Kset::new(GAMMA, 0.3, 1.0, &s.geom.allocated);
    let coarse = Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 3])
        .origin([-0.5; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(boundary))
        .cfl(0.3)
        .allocate()
        .expect("root construction")
        .set_initial(move |[x, y, z]| {
            let bx = 0.5 * (face(0, [x - 0.5 * dx, y, z]) + face(0, [x + 0.5 * dx, y, z]));
            let by = 0.5 * (face(1, [x, y - 0.5 * dx, z]) + face(1, [x, y + 0.5 * dx, z]));
            MhdPrim::new(
                Prim::adiabatic(Density(1.0), Tensor::new(VEL), Pressure(1.0 + 0.1 * (k * x).sin())),
                Tensor::new([bx, by, 0.0]),
            )
        })
        .seed_faces(face)
        .build();
    let ck = kset(&coarse);
    let regions = [RefinementRegion {
        x_lo: [region[0]; 3],
        x_hi: [region[1]; 3],
    }];
    let hier = Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, kset).unwrap();
    hier.seed_fine_from_coarse().expect("fine seed");
    let mut hier = match bodies {
        Some(b) => hier.with_bodies(b),
        None => hier,
    };
    hier.prime();
    hier
}

fn one_slip_sink() -> BodyCollection<f64, 3> {
    BodyCollection::new().add(sink(0, [0.0; 3], [0.0; 3], 1.0, slip_spec()))
}

fn finest(hier: &Hier) -> usize {
    hier.levels.len() - 1
}

fn accreted(hier: &Hier, level: usize, body: usize) -> f64 {
    match hier.levels[level].state.immersed.as_ref().unwrap().bodies.get(body).kind {
        BodyKind::BlackHole { total_accreted_mass, .. } => total_accreted_mass,
        _ => 0.0,
    }
}

// the volume integrals of the conserved fields (mass, momentum, energy) over the leaf cells: every
// fine interior cell and every coarse interior cell outside the coverage.
fn leaf_totals(hier: &Hier) -> [f64; 5] {
    let mut parts: [Vec<f64>; 5] = Default::default();
    for lvl in &hier.levels {
        let vol: f64 = lvl.state.geom.dx.iter().product();
        let cons = &lvl.state.fields.cons;
        let nrg = cons.nrg_field().unwrap();
        for c in lvl.state.geom.interior.iter() {
            if lvl.coverage.as_ref().is_some_and(|cov| cov.contains(c)) {
                continue;
            }
            parts[0].push(*cons.den.at(c) * vol);
            for a in 0..3 {
                parts[1 + a].push(*cons.mom[a].at(c) * vol);
            }
            parts[4].push(*nrg.at(c) * vol);
        }
    }
    std::array::from_fn(|i| compensated_sum(parts[i].iter().copied()))
}

// the finest interior's (mass, energy, cell magnetic energy, face magnetic energy): the cell
// magnetic energy from the interpolated field, the face energy over the lower faces of every
// interior cell plus the closing face along each axis.
fn finest_energies(hier: &Hier) -> [f64; 4] {
    let sim = &hier.levels[finest(hier)].state;
    let m = sim.fields.mhd.as_ref().unwrap();
    let vol: f64 = sim.geom.dx.iter().product();
    let interior = &sim.geom.interior;
    let mut parts: [Vec<f64>; 4] = Default::default();
    for c in interior.iter() {
        parts[0].push(*sim.fields.cons.den.at(c) * vol);
        parts[1].push(*sim.fields.cons.nrg_field().unwrap().at(c) * vol);
        parts[2].push(0.5 * (0..3).map(|d| m.bcell[d].at(c).powi(2)).sum::<f64>() * vol);
    }
    for d in 0..3 {
        for c in m.bface[d].domain().iter() {
            let inside = (0..3).all(|a| c[a] >= interior.spaces[a].lo && c[a] < interior.spaces[a].hi + (a == d) as isize);
            if inside {
                parts[3].push(0.5 * m.bface[d].at(c).powi(2) * vol);
            }
        }
    }
    std::array::from_fn(|i| compensated_sum(parts[i].iter().copied()))
}

// max over the level's interior of |bcell_d - (bface_d(lower) + bface_d(upper)) / 2|.
fn max_cell_face_defect(hier: &Hier, level: usize) -> f64 {
    let sim = &hier.levels[level].state;
    let m = sim.fields.mhd.as_ref().unwrap();
    let mut worst = 0.0f64;
    for c in sim.geom.interior.iter() {
        for d in 0..3 {
            let mut up = c;
            up[d] += 1;
            let interp = 0.5 * (*m.bface[d].at(c) + *m.bface[d].at(up));
            worst = worst.max((*m.bcell[d].at(c) - interp).abs());
        }
    }
    worst
}

// a compensated sum (Neumaier), exact to a few units of roundoff for the sums below.
fn compensated_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let (mut sum, mut comp) = (0.0f64, 0.0f64);
    for v in values {
        let t = sum + v;
        comp += if sum.abs() >= v.abs() { (sum - t) + v } else { (v - t) + sum };
        sum = t;
    }
    sum + comp
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / a.abs().max(b.abs()).max(1e-300)
}

// the covered coarse conserved density and energy equal the restriction of the fine fields.
fn assert_covered_is_restriction(hier: &Hier, label: &str) {
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
            assert!(*c.at(cell) == *scratch.at(cell), "{label}: {name} at {cell:?} is not the restriction");
        }
    }
}

// every stored value of every level's conserved, primitive, cell, and face fields, the clocks, and
// the body state.
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
            for bb in 0..im.bodies.len() {
                let b = im.bodies.get(bb);
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
    }
    v
}

// the label of every snapshot entry, in snapshot order.
fn labels(hier: &Hier) -> Vec<String> {
    let mut v = Vec::new();
    for (ll, l) in hier.levels.iter().enumerate() {
        let sim = &l.state;
        let m = sim.fields.mhd.as_ref().unwrap();
        let interior = &sim.geom.interior;
        for c in sim.geom.allocated.iter() {
            let inside = (0..3).all(|a| c[a] >= interior.spaces[a].lo && c[a] < interior.spaces[a].hi);
            let covered = l.coverage.as_ref().is_some_and(|cov| cov.contains(c));
            let where_ = if !inside { "ghost" } else if covered { "covered" } else { "leaf" };
            for nm in ["den", "mom0", "mom1", "mom2", "nrg", "rho", "pre", "bc0", "bc1", "bc2"] {
                v.push(format!("L{ll} {nm} {where_} {c:?}"));
            }
        }
        for d in 0..3 {
            for c in m.bface[d].domain().iter() {
                let inside = (0..3).all(|a| c[a] >= interior.spaces[a].lo && c[a] < interior.spaces[a].hi + (a == d) as isize);
                let where_ = if inside { "interior" } else { "ghost" };
                v.push(format!("L{ll} bface{d} {where_} {c:?}"));
            }
        }
        v.push(format!("L{ll} time"));
        v.push(format!("L{ll} iteration"));
        if let Some(im) = sim.immersed.as_ref() {
            for bb in 0..im.bodies.len() {
                for a in 0..3 {
                    v.push(format!("L{ll} body{bb} pos{a}"));
                    v.push(format!("L{ll} body{bb} vel{a}"));
                }
                if matches!(im.bodies.get(bb).kind, BodyKind::BlackHole { .. }) {
                    v.push(format!("L{ll} body{bb} accreted mass"));
                    v.push(format!("L{ll} body{bb} accretion rate"));
                }
            }
        }
    }
    v
}

fn first_mismatch(a: &[f64], b: &[f64]) -> Option<(usize, f64, f64)> {
    assert_eq!(a.len(), b.len(), "snapshots differ in length");
    a.iter().zip(b).enumerate().find(|(_, (x, y))| x.to_bits() != y.to_bits()).map(|(i, (x, y))| (i, *x, *y))
}

// the first mismatch outside the ghost bands, which every operator refills before reading.
fn first_live_mismatch(a: &[f64], b: &[f64], labels: &[String]) -> Option<(usize, f64, f64)> {
    assert_eq!(a.len(), b.len(), "snapshots differ in length");
    a.iter().zip(b).enumerate().find(|(i, (x, y))| x.to_bits() != y.to_bits() && !labels[*i].contains("ghost")).map(|(i, (x, y))| (i, *x, *y))
}

// with no body the root step is the ideal-MHD hierarchy operator alone: fine subcycling, flux and
// EMF reflux, restriction. under periodic boundaries it conserves the leaf-domain mass, momentum,
// and energy to roundoff.
#[test]
fn the_refined_ideal_mhd_operator_conserves_the_leaf_domain_totals() {
    let mut hier = build(BoundaryType::Periodic, [-0.375, 0.375], None);
    let before = leaf_totals(&hier);
    for _ in 0..2 {
        hier.step_root_with_dt(2.0 * DT);
    }
    let after = leaf_totals(&hier);
    println!("leaf totals before {before:?}\n             after {after:?}");
    for (name, i) in [("mass", 0), ("energy", 4)] {
        assert!(rel(before[i], after[i]) < 1e-12, "{name} drifts on the leaf domain: {} -> {}", before[i], after[i]);
    }
    for a in 0..3 {
        assert!((after[1 + a] - before[1 + a]).abs() < 1e-12 * before[0], "momentum {a} drifts on the leaf domain: {} -> {}", before[1 + a], after[1 + a]);
    }
    assert_covered_is_restriction(&hier, "hierarchy operator");
}

// one drain operation on the finest removes exactly the mass and energy it books for the body; the
// coarse proxy books nothing.
#[test]
fn the_finest_drain_removes_exactly_what_it_books_and_the_proxy_books_nothing() {
    let hier = build(BoundaryType::Periodic, [-0.375, 0.375], Some(one_slip_sink()));
    let fi = finest(&hier);
    let sim = &hier.levels[fi].state;
    let vol: f64 = sim.geom.dx.iter().product();
    let cells: Vec<[isize; 3]> = sim.geom.interior.iter().collect();
    let before: Vec<(f64, f64)> = cells.iter().map(|c| (*sim.fields.cons.den.at(*c), *sim.fields.cons.nrg_field().unwrap().at(*c))).collect();
    hier.drain_and_rebuild(fi, DT);
    // the removal summed cell by cell, so the sum carries the roundoff of the removed amounts alone.
    let (mut removed_mass, mut removed_energy) = (0.0, 0.0);
    for (c, (d0, e0)) in cells.iter().zip(&before) {
        removed_mass += (d0 - *sim.fields.cons.den.at(*c)) * vol;
        removed_energy += (e0 - *sim.fields.cons.nrg_field().unwrap().at(*c)) * vol;
    }
    let booked = hier.levels[fi].state.immersed.as_ref().unwrap().diagnostics.consolidate();
    let proxy = hier.levels[0].state.immersed.as_ref().unwrap().diagnostics.consolidate();
    println!("drain removed mass {removed_mass:.12e} energy {removed_energy:.12e}; booked mass {:.12e} energy {:.12e}", booked[0].mass_delta, booked[0].energy_delta);
    assert!(removed_mass > 0.0, "the drain removed nothing");
    assert!(rel(removed_mass, booked[0].mass_delta) < 1e-12, "the booked mass {:.12e} is not the removed mass {removed_mass:.12e}", booked[0].mass_delta);
    assert!(rel(removed_energy, booked[0].energy_delta) < 1e-12, "the booked energy {:.12e} is not the removed energy {removed_energy:.12e}", booked[0].energy_delta);
    assert_eq!(proxy[0].mass_delta, 0.0, "the coarse proxy booked mass");
    assert_eq!(proxy[0].energy_delta, 0.0, "the coarse proxy booked energy");
}

// over accepted root steps the leaf-domain mass decreases by exactly the finest sink's accreted
// mass: the slip and the hierarchy operator move no mass, the drain's removal is its receipt, and
// the proxy accretes nothing.
#[test]
fn the_root_step_mass_ledger_closes_on_the_finest_sink_receipt() {
    let mut hier = build(BoundaryType::Periodic, [-0.375, 0.375], Some(one_slip_sink()));
    let fi = finest(&hier);
    let (mass0, acc0) = (leaf_totals(&hier)[0], accreted(&hier, fi, 0));
    for _ in 0..3 {
        hier.step_root_with_dt(DT);
    }
    let (mass1, acc1) = (leaf_totals(&hier)[0], accreted(&hier, fi, 0));
    let (removed, receipt) = (mass0 - mass1, acc1 - acc0);
    println!("leaf mass removed {removed:.12e}, receipt {receipt:.12e}");
    assert!(receipt > 0.0, "the finest sink accreted nothing");
    // the removal is the difference of two leaf-domain sums of order the total mass, so its
    // roundoff is that of the sums.
    assert!((removed - receipt).abs() < 1e-12 * mass0, "the mass ledger does not close: removed {removed:.12e}, receipt {receipt:.12e}");
    assert!(matches!(hier.levels[0].state.immersed.as_ref().unwrap().bodies.get(0).kind, BodyKind::Gravitational { .. }), "the proxy carries a sink");
}

// one slip half-step on the finest: the face magnetic energy decreases, the extended energy
// E - M_cell + M_face over the finest interior is invariant to roundoff, and the synchronization
// then leaves the covered coarse state the restriction.
#[test]
fn the_slip_half_step_conserves_the_extended_energy_on_the_finest_and_the_sync_restricts_it() {
    let hier = build(BoundaryType::Periodic, [-0.375, 0.375], Some(one_slip_sink()));
    let fi = finest(&hier);
    println!("finest cell field vs face interpolation after priming: max |bcell - interp| = {:.3e}", max_cell_face_defect(&hier, fi));
    let b = finest_energies(&hier);
    let l = &hier.levels[fi];
    assert!(l.kernels.magnetic_slip_step(&l.state, 0.5 * DT), "the slip solve did not converge");
    let a = finest_energies(&hier);
    let extended = |t: [f64; 4]| t[1] - t[2] + t[3];
    let (x0, x1) = (extended(b), extended(a));
    println!("slip: dM_face {:.6e}, dM_cell {:.6e}, dE {:.6e}, extended residual {:.3e}", a[3] - b[3], a[2] - b[2], a[1] - b[1], x1 - x0);
    assert!(a[3] < b[3], "the slip did not dissipate face magnetic energy");
    assert!(b[0] == a[0], "the slip moved mass");
    assert!((x1 - x0).abs() < 1e-11 * x0.abs(), "the extended energy is not invariant: {x0:.12e} -> {x1:.12e}");
    hier.sync_all_fine_to_coarse();
    assert_covered_is_restriction(&hier, "after the slip and the synchronization");
}

// a refined slip run written to a checkpoint and resumed by a fresh hierarchy continues bit for
// bit with the uninterrupted run, the finest sink's accreted mass included.
#[test]
fn a_refined_slip_run_resumes_from_a_checkpoint_bit_for_bit() {
    let mut uninterrupted = build(BoundaryType::Periodic, [-0.375, 0.375], Some(one_slip_sink()));
    for _ in 0..2 {
        uninterrupted.step_root_with_dt(DT);
    }
    let fi = finest(&uninterrupted);
    let accreted_at_write = accreted(&uninterrupted, fi, 0);
    assert!(accreted_at_write > 0.0, "nothing accreted before the checkpoint; the restart of the receipt is vacuous");
    let path = std::env::temp_dir().join(format!("symbi_refined_slip_restart_{}.h5", std::process::id()));
    let levels: Vec<&Sim> = uninterrupted.levels.iter().map(|l| &l.state).collect();
    symbi::sim::checkpoint::write_hierarchy_checkpoint(&levels, path.to_str().unwrap(), &Metadata::new()).expect("checkpoint written");

    let mut resumed = build(BoundaryType::Periodic, [-0.375, 0.375], Some(one_slip_sink()));
    let loaded = resumed.restore_from_checkpoint(path.to_str().unwrap()).expect("checkpoint restored");
    let _ = std::fs::remove_file(&path);
    assert_eq!(loaded, 2);
    assert_eq!(accreted(&resumed, fi, 0).to_bits(), accreted_at_write.to_bits(), "the finest sink's accreted mass did not survive the restart");
    resumed.prime();
    let names = labels(&resumed);
    if let Some((i, x, y)) = first_live_mismatch(&snapshot(&uninterrupted), &snapshot(&resumed), &names) {
        panic!("the restored hierarchy differs from the written one at {}: {x:e} vs {y:e}", names[i]);
    }
    for _ in 0..2 {
        uninterrupted.step_root_with_dt(DT);
        resumed.step_root_with_dt(DT);
    }
    if let Some((i, x, y)) = first_mismatch(&snapshot(&uninterrupted), &snapshot(&resumed)) {
        panic!("the resumed run diverges from the uninterrupted one at {}: {x:e} vs {y:e}", labels(&resumed)[i]);
    }
}

// per-field flat snapshots of the fine bulk: the finest level's cells and faces at least three
// cells from its boundary. the interface layers carry the coarse-fine coupling's residual of
// order dt times dx, first order in the timestep alone at a fixed grid, and the coarse leaf shell
// of this box lies within the reconstruction reach of the coverage throughout; both are excluded
// so the composition's own temporal order reads.
fn bulk_snapshots(hier: &Hier) -> Vec<(&'static str, Vec<f64>)> {
    let sim = &hier.levels[finest(hier)].state;
    let m = sim.fields.mhd.as_ref().unwrap();
    let interior = &sim.geom.interior;
    let deep = |c: &[isize; 3]| (0..3).all(|k| c[k] - interior.spaces[k].lo >= 3 && interior.spaces[k].hi - 1 - c[k] >= 3);
    let cells: Vec<[isize; 3]> = interior.iter().filter(|c| deep(c)).collect();
    let den: Vec<f64> = cells.iter().map(|c| *sim.fields.cons.den.at(*c)).collect();
    let nrg: Vec<f64> = cells.iter().map(|c| *sim.fields.cons.nrg_field().unwrap().at(*c)).collect();
    let mut face = Vec::new();
    for d in 0..3 {
        for c in &cells {
            face.push(*m.bface[d].at(*c));
        }
    }
    vec![("fine density", den), ("fine energy", nrg), ("fine face field", face)]
}

fn l2_rel(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f64 = b.iter().map(|y| y * y).sum();
    (num / den.max(1e-300)).sqrt()
}

// fixed grid, refined timestep: the spatial error is common to every run and cancels to leading
// order, so the fixed-time difference between successive timesteps isolates the temporal order
// and quarters when the timestep halves. four timestep levels give two successive ratios, so an
// asymptotic second-order trend is separated from a single coincidence. the wave crosses the
// coarse-fine interface; the bulk of both levels is read.
#[test]
fn the_refined_coupled_step_is_second_order_in_time_per_field() {
    let run = |dt: f64, nsteps: usize| -> Vec<(&'static str, Vec<f64>)> {
        let mut hier = build(BoundaryType::Periodic, [-0.4375, 0.4375], Some(resolved_slip_sink()));
        for _ in 0..nsteps {
            hier.step_root_with_dt(dt);
        }
        bulk_snapshots(&hier)
    };
    let dt = 2.0 * DT;
    let runs = [run(dt, 4), run(dt / 2.0, 8), run(dt / 4.0, 16), run(dt / 8.0, 32)];
    for f in 0..runs[0].len() {
        let name = runs[0][f].0;
        let e: Vec<f64> = (0..3).map(|i| l2_rel(&runs[i][f].1, &runs[i + 1][f].1)).collect();
        let ratios = [e[0] / e[1].max(1e-300), e[1] / e[2].max(1e-300)];
        println!("{name}: diffs {:.3e} {:.3e} {:.3e}  ratios {:.2} {:.2}  (-> 4 if second order)", e[0], e[1], e[2], ratios[0], ratios[1]);
        assert!(e[0] > 1e-10, "{name}: vacuous temporal-order test (diff {})", e[0]);
        assert!(ratios[1] > 3.4, "{name}: the refined coupled step is not second order in time: ratios {:.2} {:.2}", ratios[0], ratios[1]);
    }
}

// the composition under outflow root boundaries: the sink accretes, the fields stay finite, the
// covered coarse state is the restriction.
#[test]
fn the_refined_palindrome_runs_under_outflow_boundaries() {
    let mut hier = build(BoundaryType::Outflow, [-0.375, 0.375], Some(one_slip_sink()));
    for _ in 0..2 {
        hier.step_root_with_dt(DT);
    }
    assert!(accreted(&hier, finest(&hier), 0) > 0.0, "the finest sink accreted nothing under outflow boundaries");
    assert!(snapshot(&hier).iter().all(|v| v.is_finite()), "the state is not finite under outflow boundaries");
    assert_covered_is_restriction(&hier, "outflow");
}

// a prescribed circular binary of slip sinks inside the finest: both sinks orbit, both accrete, the
// proxies follow the finest kinematics, and the covered coarse state is the restriction.
#[test]
fn a_prescribed_binary_of_slip_sinks_orbits_and_accretes_on_the_finest() {
    let (a, mass): (f64, f64) = (0.1, 1.0);
    let v = 0.5 * (mass / a).sqrt();
    let bodies = BodyCollection::new()
        .add(sink(0, [0.5 * a, 0.0, 0.0], [0.0, v, 0.0], 0.5 * mass, slip_spec()))
        .add(sink(1, [-0.5 * a, 0.0, 0.0], [0.0, -v, 0.0], 0.5 * mass, slip_spec()))
        .as_binary()
        .with_binary_params(BinaryParams::new(mass, a, 0.0, 1.0));
    let mut hier = build(BoundaryType::Periodic, [-0.4375, 0.4375], Some(bodies));
    let fi = finest(&hier);
    let start: Vec<Tensor<f64, 3>> = (0..2).map(|b| hier.levels[fi].state.immersed.as_ref().unwrap().bodies.get(b).position).collect();
    for _ in 0..3 {
        hier.step_root_with_dt(DT);
    }
    for b in 0..2 {
        let fine_body = hier.levels[fi].state.immersed.as_ref().unwrap().bodies.get(b);
        let proxy = hier.levels[0].state.immersed.as_ref().unwrap().bodies.get(b);
        let moved: f64 = (0..3).map(|k| (fine_body.position[k] - start[b][k]).powi(2)).sum::<f64>().sqrt();
        assert!(moved > 1e-6, "sink {b} did not move on its prescribed orbit");
        assert!((0..3).all(|k| proxy.position[k].to_bits() == fine_body.position[k].to_bits()), "the proxy of sink {b} does not follow the finest kinematics");
        assert!(accreted(&hier, fi, b) > 0.0, "sink {b} accreted nothing");
        assert!(matches!(proxy.kind, BodyKind::Gravitational { .. }), "the proxy of sink {b} carries a sink");
    }
    assert_covered_is_restriction(&hier, "binary");
}

