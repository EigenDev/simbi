// =============================================================================
// refinement_balanced_rollback.rs
//
// the rollback of a balanced refined root step to its entry. the balanced
// restriction writes each seam band's coarse pressure and rebuilds the band's
// energy from it, so the step-entry pressure there differs from the recovery
// of the entry's conserved state by its rounding. a step rejected after the
// band was rewritten replays from the entry with that pressure: the replay's
// entry equals the rejected attempt's in every field, and the rejecting run
// equals the clean run whose timestep at that step is capped to the one the
// replay accepted, bit for bit, on the single hierarchy and on host tiles.
//
// run: cargo test -p symbi --test refinement_balanced_rollback
// =============================================================================
#![allow(dead_code)]
use symbi::sim::substrate_seam::KernelSet as _;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, Partition, unflatten};
use symbi::sim::refinement::launched::receiving_faces_of_tile;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion, evolve_hierarchy_decomposed, seed_decomposed_fine_from_coarse};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_discretize::Recon;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::atlas::TileId;
use symbi_xpu::{CpuSpace, HostMemory};
use symbi_refinement::refinement::phase_probe::{self, PhaseSample};

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
type Hier = Hierarchy<Newtonian, 2, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kern>;
const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const N: usize = 32;
const DX: f64 = 1.0 / N as f64;
const GM: f64 = 100.0;
const C: [f64; 2] = [0.5, -1.0];
// about seventy-five root steps: the seam bands are rewritten every root step and the coarse-fine
// transfer every fine substep
const T_FINAL: f64 = 0.16;
const PATCH: [(usize, usize); 2] = [(8, 24), (8, 24)];

fn atmosphere([x, y]: [f64; 2]) -> Prim<f64, 2> {
    // isentropic, p = rho^gamma at rest: gamma/(gamma-1) rho^(gamma-1) - GM/r is uniform, rho = 1
    // at (0.5, 1)
    let r = ((x - C[0]).powi(2) + (y - C[1]).powi(2)).sqrt();
    let k = GAMMA / (GAMMA - 1.0);
    let cst = k - GM / 2.0;
    let rho = ((cst + GM / r) / k).powf(1.0 / (GAMMA - 1.0));
    Prim::adiabatic(Density(rho), Tensor::new([0.0, 0.0]), Pressure(rho.powf(GAMMA)))
}

fn kset(balanced: bool) -> impl Fn(&Sim) -> Kern + Copy {
    move |s: &Sim| Kern::new(GAMMA, CFL, &s.geom.allocated).reconstruction(Recon::Plm).well_balanced_reconstruction(balanced).with_solver(symbi::regimes::substrate_kernels::Solver::Hllc).unwrap()
}

fn root(cells: [usize; 2], offset: [usize; 2], bnd: Boundaries<2>) -> Sim {
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells).spacing([DX; 2]).origin([0.0; 2])
        .index_offset(std::array::from_fn(|a| offset[a] as isize))
        .boundaries(bnd).cfl(CFL).ghosts(2).timestepping(Timestepping::Rk2)
        .allocate().unwrap().set_initial(atmosphere).build()
}

fn bodies() -> symbi_ib::BodyCollection<f64, 2> {
    symbi_ib::BodyCollection::new().add(symbi_ib::Body::gravitational(0, Tensor::new(C), Tensor::zeros(), GM, 1.0e-3, 0.0))
}

fn region() -> RefinementRegion<2> {
    RefinementRegion { x_lo: [0.25, 0.25], x_hi: [0.75, 0.75] }
}

fn mono(balanced: bool) -> Hier {
    let r = root([N, N], [0, 0], Boundaries::uniform(BoundaryType::Reflect));
    let k = kset(balanced)(&r);
    let h = Hier::with_refinement(r, k, &[region()], ProlongOrder::Ppm, kset(balanced)).unwrap();
    h.seed_fine_from_coarse().unwrap();
    let mut h = h.with_bodies(bodies());
    h.prime();
    h
}


fn tiles(partition: &Partition<2>) -> Vec<Hier> {
    tiles_with(partition, std::env::var("PROBE_PLAIN").is_err())
}

fn tiles_with(partition: &Partition<2>, balanced: bool) -> Vec<Hier> {
    let counts = partition.counts();
    let mut tiles = Vec::new();
    for flat in 0..partition.n_tiles() {
        let tc = unflatten(flat, counts);
        let ext = partition.tile_extents(tc);
        let m = [ext[0].1, ext[1].1];
        let off = [ext[0].0, ext[1].0];
        let bnd = Boundaries(std::array::from_fn(|a| {
            let face = |edge: bool| if edge { BoundaryType::Outflow } else { BoundaryType::CoarseFine };
            [face(tc[a] == 0), face(tc[a] == counts[a] - 1)]
        }));
        let r = root(m, off, bnd);
        let lo: [f64; 2] = std::array::from_fn(|a| off[a] as f64 * DX);
        let hi: [f64; 2] = std::array::from_fn(|a| (off[a] + m[a]) as f64 * DX);
        let w = region();
        let cx = [w.x_lo[0].max(lo[0]), w.x_hi[0].min(hi[0])];
        let cy = [w.x_lo[1].max(lo[1]), w.x_hi[1].min(hi[1])];
        let k = kset(balanced)(&r);
        let h = if cx[0] < cx[1] && cy[0] < cy[1] {
            let c = RefinementRegion { x_lo: [cx[0], cy[0]], x_hi: [cx[1], cy[1]] };
            Hier::with_refinement_of_contributing(r, k, &[c], &[w], ProlongOrder::Ppm, kset(balanced)).unwrap()
        } else {
            match receiving_faces_of_tile(partition, PATCH, TileId(flat as u32)) {
                Some((p, f)) => Hier::single(r, k).with_receiving_register(&p, f).unwrap(),
                None => Hier::single(r, k),
            }
        };
        tiles.push(h);
    }
    seed_decomposed_fine_from_coarse(&tiles, counts, &vec![0; tiles.len()], &LocalCopy).unwrap();
    tiles.into_iter().map(|h| { let mut h = h.with_bodies(bodies()); h.prime(); h }).collect()
}


/// every level's interior conserved state and primitive pressure, as bits
fn state_bits(h: &Hier) -> Vec<(usize, &'static str, [isize; 2], u64)> {
    let mut out = Vec::new();
    for (ll, l) in h.levels.iter().enumerate() {
        let s = &l.state;
        let c = &s.fields.cons;
        let pre = s.fields.prim.pre_field().unwrap();
        for x in s.geom.interior.iter() {
            let at = [x[0], x[1]];
            out.push((ll, "den", at, c.den.view().at(x).to_bits()));
            out.push((ll, "m0", at, c.mom[0].view().at(x).to_bits()));
            out.push((ll, "m1", at, c.mom[1].view().at(x).to_bits()));
            out.push((ll, "nrg", at, c.nrg_field().unwrap().view().at(x).to_bits()));
            out.push((ll, "pre", at, pre.view().at(x).to_bits()));
        }
    }
    out
}

fn assert_same(a: &[(usize, &'static str, [isize; 2], u64)], b: &[(usize, &'static str, [isize; 2], u64)], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: different cell sets");
    let differing: Vec<_> = a.iter().zip(b).filter(|(x, y)| x.3 != y.3).map(|(x, _)| (x.0, x.1, x.2)).collect();
    assert!(differing.is_empty(), "{label}: {} values differ, first {:?}", differing.len(), differing.first());
}

/// the decomposed march for `steps` root steps from the tiles' clock
fn run_tiles(t: &mut [Hier], partition: &Partition<2>, steps: u64) {
    let devices = vec![0; t.len()];
    let t0 = t[0].levels[0].state.time;
    evolve_hierarchy_decomposed(t, partition.counts(), &devices, &LocalCopy, Timestepping::Rk2, t0, 10.0, 1, |it, _, _| {
        if it >= steps { std::ops::ControlFlow::Break(()) } else { std::ops::ControlFlow::Continue(()) }
    });
}

const STEP: u64 = 2;

/// the step-entry band pressure the premise rests on: the recovery of the entry state moves it
fn assert_band_pressure_is_not_the_recovery(h: &Hier) {
    let s = &h.levels[0].state;
    let pre = s.fields.prim.pre_field().unwrap();
    let before: Vec<u64> = s.geom.allocated.iter().map(|c| pre.view().at(c).to_bits()).collect();
    h.levels[0].kernels.c2p(s);
    let moved = s.geom.allocated.iter().zip(&before).filter(|(c, b)| pre.view().at(*c).to_bits() != **b).count();
    assert!(moved > 0, "the recovery reproduces every band pressure; the rollback is not exercised");
}

#[test]
fn a_replayed_balanced_step_enters_with_the_rejected_attempts_state() {
    // premise: the band leaves the entry pressure off the recovery of the entry state
    {
        let mut h = mono(true);
        h.evolve_steps(STEP).unwrap();
        assert_band_pressure_is_not_the_recovery(&h);
    }
    let mut rejecting = mono(true);
    rejecting.reject_fine_once_at(STEP, 1, 1);
    phase_probe::start();
    rejecting.evolve_steps(STEP + 1).unwrap();
    let log = phase_probe::take();
    assert!(!rejecting.forced_fine_rejection_pending(), "the step was never rejected");
    let begins: Vec<&PhaseSample> = log.iter().filter(|s| s.phase == "begin" && s.level == 0).collect();
    assert_eq!(begins.len() as u64, STEP + 2, "the rejected step was not entered twice");
    let (attempt, replay) = (begins[STEP as usize], begins[STEP as usize + 1]);
    for ((name, a), (_, b)) in attempt.fields.iter().zip(&replay.fields) {
        let n = a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        assert_eq!(n, 0, "{name}: the replay entered with {n} values off the rejected attempt's entry");
    }
}

#[test]
fn a_rejected_balanced_step_replays_to_the_capped_clean_run_on_the_single_hierarchy() {
    let mut rejecting = mono(true);
    rejecting.reject_fine_once_at(STEP, 1, 1);
    rejecting.evolve_steps(STEP + 1).unwrap();
    assert!(!rejecting.forced_fine_rejection_pending(), "the step was never rejected");
    let accepted = rejecting.levels[0].state.dt;
    let mut clean = mono(true);
    clean.evolve_steps(STEP + 1).unwrap();
    assert!(accepted < clean.levels[0].state.dt, "the rejection left the timestep unreduced");
    let mut capped = mono(true);
    capped.cap_root_dt_once_at(STEP, accepted);
    capped.evolve_steps(STEP + 1).unwrap();
    assert_eq!(capped.levels[0].state.dt.to_bits(), accepted.to_bits(), "the capped step took another timestep");
    assert_same(&state_bits(&rejecting), &state_bits(&capped), "single hierarchy");
}

#[test]
fn a_rejected_balanced_step_replays_to_the_capped_clean_run_on_host_tiles() {
    // the cut at root cell 12 leaves the x band 8..12 on the first tile and runs through the patch
    let partition = Partition::explicit([N, N], [vec![12], vec![]]).unwrap();
    let mut rejecting = tiles(&partition);
    for h in rejecting.iter_mut().filter(|h| h.levels.len() > 1) {
        h.reject_fine_once_at(STEP, 1, 1);
    }
    run_tiles(&mut rejecting, &partition, STEP + 1);
    assert!(rejecting.iter().all(|h| !h.forced_fine_rejection_pending()), "the step was never rejected");
    let accepted = rejecting[0].levels[0].state.dt;
    // the decomposed march resumes from its state as it stands, so the control caps the step by
    // the user clamp after the steps before it
    let mut capped = tiles(&partition);
    run_tiles(&mut capped, &partition, STEP);
    for h in capped.iter_mut() {
        h.levels[0].state.max_dt = accepted;
    }
    run_tiles(&mut capped, &partition, 1);
    assert_eq!(capped[0].levels[0].state.dt.to_bits(), accepted.to_bits(), "the capped step took another timestep");
    let bits = |t: &[Hier]| {
        let mut all: Vec<_> = t.iter().enumerate().flat_map(|(i, h)| state_bits(h).into_iter().map(move |(l, n, c, b)| (l, n, [c[0] + 1000 * i as isize, c[1]], b))).collect();
        all.sort_by(|a, b| (a.0, a.1, a.2).cmp(&(b.0, b.1, b.2)));
        all
    };
    assert_same(&bits(&rejecting), &bits(&capped), "host tiles");
}
