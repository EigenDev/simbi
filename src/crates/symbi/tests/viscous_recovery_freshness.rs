// =============================================================================
// viscous_recovery_freshness.rs
//
// the viscous transport writes the conserved momentum after the step's last
// recovery, so the stored primitives predate the conserved state until they
// are recovered, and the recovered primitives are stale in every cut halo
// until they are exchanged. the post-step recipe declares both requirements
// on the viscous phase and this gate holds them apart: the pass leaves the
// primitives inconsistent and the recovery does not touch the conserved
// fields; with the recipe as declared, two tiles cut through a viscous vortex
// array equal one tile bit for bit over several steps, and a run of eight
// steps equals four steps, a restart's recovery, and four more; a table
// without the halo refresh breaks the two-tile equality, and a table without
// the recovery breaks the restart equality.
//
// run: cargo test -p symbi --test viscous_recovery_freshness
// =============================================================================
use std::f64::consts::PI;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, Partition, Schedule, Topology, exchange_grid, unflatten};
use symbi_sim::stage::{
    FieldSet, PostStepError, Recover, Refresh, RefreshAt, StepContext, StepOp, StepPhase, fold_post_step_with,
    fold_stage, StageArgs,
};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::substrate_seam::{KernelSet, WithViscosity};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
type Store = FieldStore<2, 2, HostMemory, f64>;

const N: usize = 32;
const H: f64 = 1.0 / N as f64;
const GAMMA: f64 = 1.4;
const NU: f64 = 0.02;
const STEPS: usize = 8;

/// the domain's boundary condition: periodic on every axis, or open (outflow) faces
#[derive(Clone, Copy)]
enum Outer {
    Periodic,
    Open,
}

impl Outer {
    fn topology(self) -> Topology<2> {
        match self {
            Outer::Periodic => Topology::wrapping([true; 2]),
            Outer::Open => Topology::open(),
        }
    }
}

/// a vortex array whose velocity varies along both axes, so the viscous stress reads across
/// every cut, in a density ripple. a cut face is `CoarseFine` (filled by the exchange); an
/// uncut periodic axis wraps within the tile; an open domain's outer faces are outflow
fn tile(partition: &Partition<2>, flat: usize, outer: Outer) -> (Sim, Kern) {
    let counts = partition.counts();
    let tc = unflatten(flat, counts);
    let ext = partition.tile_extents(tc);
    let bnd = Boundaries(std::array::from_fn(|a| {
        let face = |on_edge: bool| match (outer, on_edge) {
            (Outer::Periodic, _) if counts[a] > 1 => BoundaryType::CoarseFine,
            (Outer::Periodic, _) => BoundaryType::Periodic,
            (Outer::Open, true) => BoundaryType::Outflow,
            (Outer::Open, false) => BoundaryType::CoarseFine,
        };
        [face(tc[a] == 0), face(tc[a] + 1 == counts[a])]
    }));
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([ext[0].1, ext[1].1])
        .origin([0.0, 0.0])
        .spacing([H, H])
        .index_offset([ext[0].0 as isize, ext[1].0 as isize])
        .boundaries(bnd)
        .cfl(0.3)
        .timestepping(Timestepping::Euler)
        .allocate()
        .expect("sim")
        .set_initial(|[x, y]| {
            Prim::adiabatic(
                Density(1.0 + 0.1 * (2.0 * PI * x).cos() * (2.0 * PI * y).cos()),
                Tensor::new([0.3 * (2.0 * PI * y).sin() * (2.0 * PI * x).cos(), -0.3 * (2.0 * PI * x).sin() * (2.0 * PI * y).cos()]),
                Pressure(1.0),
            )
        })
        .build();
    let k = Kern::new(GAMMA, 0.3, &sim.geom.allocated).with_viscosity(NU);
    (sim, k)
}

/// the post-step context of the tiles in this process: work in place, the count its own
/// maximum, the halos refreshed through the schedule's exchange and the physical fill
struct TilesStep<'a> {
    schedule: &'a Schedule<2>,
    kernels: &'a [&'a Kern],
}

impl<'a> StepContext<2, 2, HostMemory> for TilesStep<'a> {
    fn on_tile(&self, _k: usize, f: &mut dyn FnMut()) {
        f()
    }
    fn combine_max(&mut self, local: usize) -> usize {
        local
    }
    fn refresh_halos(&mut self, stores: &[&Store], _at: RefreshAt) {
        let devices = vec![0i32; stores.len()];
        exchange_grid(stores, self.schedule, &devices, &LocalCopy);
        for (k, store) in stores.iter().enumerate() {
            self.kernels[k].ghost_fill(store);
        }
    }
}

/// the viscous phase under one declaration: the recipe's, or one with a requirement dropped
fn viscous_table(recover: Recover, refresh: Refresh) -> [StepPhase; 1] {
    [StepPhase {
        name: "viscous",
        op: StepOp::Viscous,
        reads: FieldSet::PRIM,
        writes: FieldSet::CONS,
        recover,
        refresh,
        property: "viscous transport",
    }]
}

/// prime every tile (recovery, physical fill, the cut halos) as the drivers do
fn prime(stores: &[&Store], kernels: &[&Kern], schedule: &Schedule<2>) {
    for (k, store) in stores.iter().enumerate() {
        kernels[k].c2p(store);
        kernels[k].ghost_fill(store);
    }
    exchange_grid(stores, schedule, &vec![0; stores.len()], &LocalCopy);
    for (k, store) in stores.iter().enumerate() {
        kernels[k].ghost_fill(store);
    }
}

/// `steps` forward-euler steps over the tiles: the stage fold per tile, the halo exchange,
/// then the post-step table
fn march(stores: &[&Store], kernels: &[&Kern], schedule: &Schedule<2>, table: &[StepPhase], steps: usize) -> Result<(), PostStepError> {
    let stages = Timestepping::Euler.stages();
    for _ in 0..steps {
        let dt = stores.iter().zip(kernels).map(|(s, k)| k.cfl(s)).fold(f64::INFINITY, f64::min);
        for (ii, &(a0, ac)) in stages.iter().enumerate() {
            for (k, store) in stores.iter().enumerate() {
                let args = StageArgs {
                    dt,
                    a0,
                    ac,
                    stage: ii,
                    n_stages: stages.len(),
                    injection_weight: symbi_sim::driver::downstream_injection_weight(stages, ii),
                    allow_elision: false,
                    exit_fraction: symbi_sim::driver::stage_exit_fraction(stages, ii),
                };
                let outcome = fold_stage(store, kernels[k], args, &mut |_| {});
                assert_eq!(outcome, symbi_sim::stage::StageOutcome::Accepted, "a stage of the vortex array asked for a retry");
            }
            exchange_grid(stores, schedule, &vec![0; stores.len()], &LocalCopy);
            for (k, store) in stores.iter().enumerate() {
                kernels[k].ghost_fill(store);
            }
        }
        let mut cx = TilesStep { schedule, kernels };
        fold_post_step_with(table, stores, kernels, dt, &mut cx, |_| true)?;
    }
    Ok(())
}

/// the conserved density and momentum of every cell on the whole grid
fn assembled(stores: &[&Store], partition: &Partition<2>) -> Vec<u64> {
    let counts = partition.counts();
    let mut out = vec![0u64; 3 * N * N];
    for (flat, store) in stores.iter().enumerate() {
        let ext = partition.tile_extents(unflatten(flat, counts));
        let lo = [store.geom.interior.spaces[0].lo, store.geom.interior.spaces[1].lo];
        let (den, m0, m1) = (store.fields.cons.den.view(), store.fields.cons.mom[0].view(), store.fields.cons.mom[1].view());
        for c in store.geom.interior.iter() {
            let g = (ext[0].0 + (c[0] - lo[0]) as usize) * N + ext[1].0 + (c[1] - lo[1]) as usize;
            out[3 * g] = den.at(c).to_bits();
            out[3 * g + 1] = m0.at(c).to_bits();
            out[3 * g + 2] = m1.at(c).to_bits();
        }
    }
    out
}

fn run(partition: &Partition<2>, table: &[StepPhase], steps: usize) -> Vec<u64> {
    run_with(partition, table, steps, Outer::Periodic)
}

fn run_with(partition: &Partition<2>, table: &[StepPhase], steps: usize, outer: Outer) -> Vec<u64> {
    let mut built: Vec<(Sim, Kern)> = (0..partition.n_tiles()).map(|f| tile(partition, f, outer)).collect();
    let schedule = Schedule::derive(partition.counts(), built[0].0.geom.ng, &outer.topology());
    let stores: Vec<&Store> = built.iter().map(|(s, _)| &**s).collect();
    let kernels: Vec<&Kern> = built.iter().map(|(_, k)| k).collect();
    prime(&stores, &kernels, &schedule);
    march(&stores, &kernels, &schedule, table, steps).expect("the accepted state recovered");
    let out = assembled(&stores, partition);
    drop(stores);
    drop(kernels);
    built.clear();
    out
}

/// a restart's recovery: the primitives rebuilt from the conserved fields, the physical
/// ghosts filled and the cut halos exchanged, as a resumed run does before its first step
fn restart(stores: &[&Store], kernels: &[&Kern], schedule: &Schedule<2>) {
    prime(stores, kernels, schedule);
}

fn one_tile() -> Partition<2> {
    Partition::explicit([N, N], [Vec::new(), Vec::new()]).unwrap()
}

fn two_tiles() -> Partition<2> {
    Partition::explicit([N, N], [vec![N / 2], Vec::new()]).unwrap()
}

/// the pass changes the conserved momentum and leaves the primitives inconsistent with it;
/// the recovery changes no conserved field
#[test]
fn the_viscous_pass_leaves_the_primitives_behind_and_the_recovery_keeps_the_conserved_state() {
    let partition = one_tile();
    let (sim, kern) = tile(&partition, 0, Outer::Periodic);
    let schedule = Schedule::derive([1, 1], sim.geom.ng, &Topology::wrapping([true; 2]));
    prime(&[&sim], &[&kern], &schedule);
    let dt = kern.cfl(&sim);
    let before: Vec<u64> = assembled(&[&sim], &partition);
    let vel_before: Vec<u64> = sim.geom.interior.iter().map(|c| sim.fields.prim.vel[0].view().at(c).to_bits()).collect();
    kern.viscous(&sim, dt);
    let after = assembled(&[&sim], &partition);
    assert_ne!(before, after, "the viscous pass changed no conserved field; the premise is vacuous");
    let vel_after: Vec<u64> = sim.geom.interior.iter().map(|c| sim.fields.prim.vel[0].view().at(c).to_bits()).collect();
    assert_eq!(vel_before, vel_after, "the pass rewrote the primitives itself");
    kern.c2p(&sim);
    let vel_recovered: Vec<u64> = sim.geom.interior.iter().map(|c| sim.fields.prim.vel[0].view().at(c).to_bits()).collect();
    assert_ne!(vel_after, vel_recovered, "the stored primitives already agreed with the conserved state; the premise is vacuous");
    assert_eq!(after, assembled(&[&sim], &partition), "the recovery changed a conserved field");
}

/// the recipe as declared: two tiles equal one tile over several steps, and eight steps equal
/// four, a restart's recovery, and four more
#[test]
fn the_declared_recipe_holds_across_a_cut_and_across_a_restart() {
    let table = viscous_table(Recover::Primitives, Refresh::Halos);
    let one = run(&one_tile(), &table, STEPS);
    let two = run(&two_tiles(), &table, STEPS);
    assert_eq!(one, two, "two tiles differ from one tile under the declared recipe");
    let partition = one_tile();
    let (sim, kern) = tile(&partition, 0, Outer::Periodic);
    let schedule = Schedule::derive([1, 1], sim.geom.ng, &Topology::wrapping([true; 2]));
    prime(&[&sim], &[&kern], &schedule);
    march(&[&sim], &[&kern], &schedule, &table, STEPS / 2).unwrap();
    restart(&[&sim], &[&kern], &schedule);
    march(&[&sim], &[&kern], &schedule, &table, STEPS / 2).unwrap();
    assert_eq!(assembled(&[&sim], &partition), one, "a restarted run differs from the uninterrupted one under the declared recipe");
}

/// a table without the halo refresh: the exchanged halos hold primitives that predate the
/// recovery, and two tiles depart from one
#[test]
fn without_the_halo_refresh_two_tiles_depart_from_one() {
    let table = viscous_table(Recover::Primitives, Refresh::None);
    let one = run(&one_tile(), &table, STEPS);
    let two = run(&two_tiles(), &table, STEPS);
    assert_ne!(one, two, "two tiles equal one tile without the halo refresh; the gate is vacuous");
}

/// a table without the recovery: the uninterrupted run carries stale primitives into the
/// next reconstruction and a restarted run recovers fresh ones, so they depart
#[test]
fn without_the_recovery_a_restart_departs_from_the_uninterrupted_run() {
    let table = viscous_table(Recover::None, Refresh::Halos);
    let uninterrupted = run(&one_tile(), &table, STEPS);
    let partition = one_tile();
    let (sim, kern) = tile(&partition, 0, Outer::Periodic);
    let schedule = Schedule::derive([1, 1], sim.geom.ng, &Topology::wrapping([true; 2]));
    prime(&[&sim], &[&kern], &schedule);
    march(&[&sim], &[&kern], &schedule, &table, STEPS / 2).unwrap();
    restart(&[&sim], &[&kern], &schedule);
    march(&[&sim], &[&kern], &schedule, &table, STEPS / 2).unwrap();
    assert_ne!(assembled(&[&sim], &partition), uninterrupted, "a restart equals the uninterrupted run without the recovery; the gate is vacuous");
}

/// the cut on the second axis and the two-axis cut, under periodic and open outer faces:
/// the viscous stress reads the cross derivative at every face, so a tile corner across two
/// cuts feeds the pass, and so does the corner beyond a cut side and an open outer face
#[test]
fn the_declared_recipe_holds_across_a_cut_on_each_axis() {
    let table = viscous_table(Recover::Primitives, Refresh::Halos);
    let y_cut = Partition::explicit([N, N], [Vec::new(), vec![N / 2]]).unwrap();
    let quad = Partition::explicit([N, N], [vec![N / 2], vec![N / 2]]).unwrap();
    for outer in [Outer::Periodic, Outer::Open] {
        let one = run_with(&one_tile(), &table, STEPS, outer);
        assert_eq!(one, run_with(&y_cut, &table, STEPS, outer), "a cut on the second axis departs from one tile");
        assert_eq!(one, run_with(&quad, &table, STEPS, outer), "the two-axis cut departs from one tile");
    }
}
