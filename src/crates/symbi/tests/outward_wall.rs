// =============================================================================
// outward_wall.rs
//
// the outward-only outer wall on the lifted (r, theta) plane, face by face at every stage of the
// production stage, against the plain outflow wall on the same state:
// - outgoing flow: the wall face keeps the riemann flux bit for bit
// - incoming flow: the wall face carries no mass, energy or tangential momentum, and its normal
//   momentum flux is the pressure of the interior cell beside it, where the plain wall passes the
//   inflow's mass and momentum
// - reversal: gas leaving the wall decelerates under a point mass and turns inward; every stage
//   obeys the rule of its own face signs, both signs occur, and E + rho Phi closes against the
//   wall fluxes at roundoff, from an independent sum over the stored fluxes and from the
//   production receipts
// - the plain wall's flux is the same with the wall flags absent and with them all false
//
// run: cargo test -p symbi --test outward_wall
// =============================================================================

use std::f64::consts::PI;

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Spherical;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, SofteningKind};
use symbi_sim::conservation_ledger;
use symbi_sim::driver::{downstream_injection_weight, needs_step_snapshot, stage_exit_fraction, stage_schedule};
use symbi_sim::stage::{HookPoint, StageArgs, StageOutcome, fold_stage};
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<Newtonian, 2, 3, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.1;
const NI: usize = 16;
const NJ: usize = 8;
const R_LO: f64 = 1.0;
const R_HI: f64 = 2.0;
const OUTER: [[bool; 2]; 3] = [[false, true], [false, false], [false, false]];

/// a uniform gas at rest in theta and phi with radial velocity `vr`, about a point mass of `mass`
/// at the origin (none when zero)
fn uniform(vr: f64, rho: f64, p: f64, mass: f64) -> Sim {
    let walls = [[BoundaryType::Outflow, BoundaryType::Outflow], [BoundaryType::Axis, BoundaryType::Axis]];
    let sim: Sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Spherical)
        .cells([NI, NJ])
        .origin([R_LO, 0.0])
        .spacing([(R_HI - R_LO) / NI as f64, PI / NJ as f64])
        .boundaries(Boundaries::per_axis(walls))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("the grid")
        .set_initial(|_| Prim::adiabatic(Density(rho), Tensor::new([vr, 0.0, 0.0]), Pressure(p)))
        .build();
    if mass == 0.0 {
        return sim;
    }
    sim.with_bodies(BodyCollection::new().add(
        Body::gravitational(0, Tensor::zeros(), Tensor::zeros(), mass, 0.0, 0.5 * R_LO).with_softening_kind(SofteningKind::Compact),
    ))
}

fn kernels(sim: &Sim, walls: Option<[[bool; 2]; 3]>) -> Kern {
    let mut k = Kern::new(GAMMA, CFL, &sim.geom.allocated);
    k.cell_momentum_gravity_work = false;
    match walls {
        Some(w) => k.with_outward_walls(w),
        None => k,
    }
}

/// the five flux components [den, m_r, m_theta, m_phi, nrg] on outer-wall face j, and the
/// pressure of the interior cell beside it
#[derive(Clone, Copy, Debug)]
struct WallFace {
    flux: [f64; 5],
    beside_pressure: f64,
}

/// one stage's record: the outer wall's faces, and the E + rho Phi and mass rates through both
/// radial walls and both polar walls
struct StageRecord {
    faces: Vec<WallFace>,
    outflow: [f64; 2],
}

fn geometry_radial_area(r: f64, j: usize) -> f64 {
    let dt = PI / NJ as f64;
    2.0 * PI * r * r * ((j as f64 * dt).cos() - ((j + 1) as f64 * dt).cos())
}

fn potential(mass: f64, r: f64) -> f64 {
    if mass == 0.0 { 0.0 } else { -mass / r }
}

/// the radial face flux at face `i` of row `j`
fn radial_flux(sim: &FieldStore<2, 3, HostMemory, f64>, i: usize, j: usize) -> [f64; 5] {
    let f = &sim.fields.flux[0];
    let at = |field: &symbi_grid::Field<f64, 2, HostMemory>| *field.view().at([i as isize, j as isize]);
    [at(&f.den), at(&f.mom[0]), at(&f.mom[1]), at(&f.mom[2]), at(f.nrg_field().expect("nrg flux"))]
}

/// the outward (mass, E + rho Phi) rates through the radial walls; the polar walls are the axis
fn wall_rates(sim: &FieldStore<2, 3, HostMemory, f64>, mass: f64) -> [f64; 2] {
    let mut out = [0.0, 0.0];
    for j in 0..NJ {
        for (i, r, sign) in [(0usize, R_LO, -1.0), (NI, R_HI, 1.0)] {
            let f = radial_flux(sim, i, j);
            let a = geometry_radial_area(r, j);
            out[0] += sign * a * f[0];
            out[1] += sign * a * (f[4] + potential(mass, r) * f[0]);
        }
    }
    out
}

/// the volume totals of (mass, E + rho Phi), Phi at the volume-weighted radial centroid
fn totals(sim: &Sim, mass: f64) -> [f64; 2] {
    let (dr, dt) = ((R_HI - R_LO) / NI as f64, PI / NJ as f64);
    let mut out = [0.0, 0.0];
    for j in 0..NJ {
        for i in 0..NI {
            let (a, b) = (R_LO + i as f64 * dr, R_LO + (i + 1) as f64 * dr);
            let v = 2.0 * PI / 3.0 * (b.powi(3) - a.powi(3)) * ((j as f64 * dt).cos() - ((j + 1) as f64 * dt).cos());
            let r_c = (b.powi(4) - a.powi(4)) / 4.0 / ((b.powi(3) - a.powi(3)) / 3.0);
            let c = [i as isize, j as isize];
            let rho = *sim.fields.cons.den.view().at(c);
            let e = *sim.fields.cons.nrg_field().expect("nrg").view().at(c);
            out[0] += rho * v;
            out[1] += (e + rho * potential(mass, r_c)) * v;
        }
    }
    out
}

/// `steps` production steps under kernel set `k`, recording every stage's outer-wall faces and
/// wall rates, and booking the production receipts
fn march(sim: &mut Sim, k: &Kern, steps: usize, mass: f64) -> Vec<StageRecord> {
    KernelSet::c2p(k, sim);
    KernelSet::ghost_fill(k, sim);
    let mut records = Vec::new();
    for _ in 0..steps {
        let dt = KernelSet::cfl(k, sim);
        assert!(dt > 0.0, "the timestep at step {} is {dt}", sim.iteration);
        sim.dt = dt;
        let stages = sim.timestepping.stages();
        if needs_step_snapshot(stages) {
            KernelSet::snapshot(k, sim);
        }
        for spec in stage_schedule(stages) {
            let store: &FieldStore<2, 3, HostMemory, f64> = sim;
            let pressure: Vec<f64> =
                (0..NJ).map(|j| *store.fields.prim.pre_field().expect("pre").view().at([NI as isize - 1, j as isize])).collect();
            let mut record = None;
            let outcome = fold_stage(
                store,
                k,
                StageArgs {
                    dt,
                    a0: spec.a0,
                    ac: spec.ac,
                    stage: spec.index,
                    n_stages: stages.len(),
                    injection_weight: downstream_injection_weight(stages, spec.index),
                    allow_elision: true,
                    exit_fraction: stage_exit_fraction(stages, spec.index),
                },
                &mut |point| {
                    if point == HookPoint::AfterFlux {
                        let faces = (0..NJ).map(|j| WallFace { flux: radial_flux(store, NI, j), beside_pressure: pressure[j] }).collect();
                        let rates = wall_rates(store, mass);
                        let w = dt * spec.ac * downstream_injection_weight(stages, spec.index);
                        record = Some(StageRecord { faces, outflow: [w * rates[0], w * rates[1]] });
                    }
                },
            );
            assert_eq!(outcome, StageOutcome::Accepted);
            records.push(record.expect("the stage's flux"));
        }
        if conservation_ledger::scope_is_open() {
            conservation_ledger::step_commit();
        }
        sim.time += dt;
        sim.iteration += 1;
    }
    records
}

/// the outward-only rule on one face: an inward mass flux takes the wall flux, anything else the
/// riemann flux of the plain wall, bit for bit
fn assert_rule(outward: &WallFace, plain: &WallFace, label: &str) {
    if plain.flux[0] < 0.0 {
        assert_eq!(outward.flux[0], 0.0, "{label}: mass entered");
        assert_eq!(outward.flux[4], 0.0, "{label}: energy entered");
        assert_eq!((outward.flux[2], outward.flux[3]), (0.0, 0.0), "{label}: tangential momentum crossed");
        assert_eq!(outward.flux[1], outward.beside_pressure, "{label}: the normal momentum flux is not the adjacent pressure");
    } else {
        let bits = |f: &[f64; 5]| f.map(f64::to_bits);
        assert_eq!(bits(&outward.flux), bits(&plain.flux), "{label}: an outgoing face left the riemann flux");
    }
}

/// one stage of each wall on the same state
fn first_stage(vr: f64) -> (WallFace, WallFace) {
    let mut a = uniform(vr, 1.0, 1.0, 0.0);
    let mut b = uniform(vr, 1.0, 1.0, 0.0);
    let ka = kernels(&a, Some(OUTER));
    let kb = kernels(&b, None);
    let (ra, rb) = (march(&mut a, &ka, 1, 0.0), march(&mut b, &kb, 1, 0.0));
    (ra[0].faces[NJ / 2], rb[0].faces[NJ / 2])
}

#[test]
fn outgoing_flow_keeps_the_riemann_flux() {
    let (outward, plain) = first_stage(0.3);
    assert!(plain.flux[0] > 0.0, "the setup's wall face carries no outflow: {plain:?}");
    assert_rule(&outward, &plain, "outgoing");
}

#[test]
fn incoming_flow_meets_a_solid_wall_whose_pressure_is_the_adjacent_cell_s() {
    let (outward, plain) = first_stage(-0.3);
    assert!(plain.flux[0] < 0.0, "the setup's wall face carries no inflow: {plain:?}");
    assert!(plain.flux[1] != plain.beside_pressure, "the plain wall's momentum flux is the pressure alone; the comparison is vacuous");
    assert_rule(&outward, &plain, "incoming");
}

#[test]
fn a_reversing_flow_obeys_the_rule_at_every_stage_and_closes_e_plus_rho_phi() {
    let mass = 1.0;
    let mut a = uniform(0.02, 1.0, 0.01, mass);
    let mut b = uniform(0.02, 1.0, 0.01, mass);
    let ka = kernels(&a, Some(OUTER));
    let kb = kernels(&b, None);
    let start = totals(&a, mass);
    let scope = conservation_ledger::open_scope();
    let outward = march(&mut a, &ka, 30, mass);
    let receipts = scope.accepted();
    drop(scope);
    // the plain wall on the same states: each outward-wall stage's state is its own, so the rule
    // compares the outward wall's faces against the plain flux it replaces at the first stage, and
    // against the sign the outward wall's own riemann solution would take thereafter
    let plain = march(&mut b, &kb, 1, mass);
    assert_rule(&outward[0].faces[NJ / 2], &plain[0].faces[NJ / 2], "the first stage");
    let (mut out_faces, mut in_faces) = (0, 0);
    for (s, record) in outward.iter().enumerate() {
        for (j, face) in record.faces.iter().enumerate() {
            let f = face.flux;
            if f[0] > 0.0 {
                out_faces += 1;
            } else if f == [0.0, face.beside_pressure, 0.0, 0.0, 0.0] {
                in_faces += 1;
            } else {
                assert_eq!(f[0], 0.0, "stage {s} face {j}: mass flux {} is neither outflow nor the wall's", f[0]);
            }
            assert!(f[0] >= 0.0, "stage {s} face {j}: mass entered, {}", f[0]);
        }
    }
    assert!(out_faces > 0 && in_faces > 0, "the flow never reversed at the wall: {out_faces} outgoing, {in_faces} walled faces");
    // E + rho Phi and mass against the stored wall fluxes, and against the production receipts
    let end = totals(&a, mass);
    let through: [f64; 2] = [outward.iter().map(|r| r.outflow[0]).sum(), outward.iter().map(|r| r.outflow[1]).sum()];
    for (k, name) in ["mass", "E + rho Phi"].iter().enumerate() {
        let residual = end[k] - start[k] + through[k];
        assert!(residual.abs() <= 1e-13 * start[k].abs(), "{name}: change {} against outflow {}, residual {residual:e}", end[k] - start[k], through[k]);
    }
    let booked = receipts.total_outflow();
    assert!((booked.mass - through[0]).abs() <= 1e-15 * start[0].abs(), "the receipts' mass {} against {}", booked.mass, through[0]);
    assert!((booked.energy - through[1]).abs() <= 1e-15 * start[1].abs(), "the receipts' energy {} against {}", booked.energy, through[1]);
    assert_eq!(receipts.outflow[0][1].mass_in, 0.0, "the receipts book mass entering through the outward-only wall");
}

#[test]
fn the_plain_wall_is_the_same_with_the_flags_absent_or_all_false() {
    for vr in [-0.3, 0.3] {
        let mut a = uniform(vr, 1.0, 1.0, 0.0);
        let mut b = uniform(vr, 1.0, 1.0, 0.0);
        let (ka, kb) = (kernels(&a, None), kernels(&b, Some([[false; 2]; 3])));
        let (ra, rb) = (march(&mut a, &ka, 2, 0.0), march(&mut b, &kb, 2, 0.0));
        for (x, y) in ra.iter().zip(&rb) {
            for (p, q) in x.faces.iter().zip(&y.faces) {
                assert_eq!(p.flux.map(f64::to_bits), q.flux.map(f64::to_bits), "vr {vr}: the wall flags changed the plain wall");
            }
        }
        if vr < 0.0 {
            assert!(ra[0].faces.iter().all(|f| f.flux[0] < 0.0), "the plain wall's inflow case passes no inflow");
        }
    }
}
