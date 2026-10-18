// =============================================================================
// worker_refined_processes.rs
//
// the refined hydro transaction across worker processes: a root grid with one
// refined patch of ratio 2 spanning tile cuts, each worker holding its root
// tiles and the fine tiles under them, advanced through the fabric by
// `evolve_refined_worker`, against the single hierarchy driven with the same
// timesteps in a process of its own. the comparison covers every conserved and
// primitive component of every interior cell on both levels (the covered root
// cells included) and every cut halo primitive against the single hierarchy's
// interior value at that cell. arms: two ragged layouts with cuts off the
// patch's center over several root steps; a layout where two of three workers
// hold no fine tile; a rejection forced at a root stage and one in the second
// fine substep, each against a clean run capped to the accepted smaller
// timestep, on both layouts; an invalid timestep candidate on the fine level
// under a positive user cap; a fine cell flagged troubled against the fine cut
// so the first-order correction crosses the cut one-sided; a patch face lying
// on a cut, whose flux-register units cross to the neighbor holding the
// corrected cells, with a duplicated and a dropped contribution each ending the
// run. every worker reports its grant high-water per peer.
//
// run: cargo test -p symbi --test worker_refined_processes
// =============================================================================
use std::collections::HashMap;
use std::io::Read;
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use symbi::regimes::fofc::{fofc_reset_stats, fofc_stats};
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{FinePartition, LocalCopy, Partition, Topology, flatten, plan_cons_schema, plan_schema, unflatten};
use symbi::sim::refinement::launched::{
    ClockField,
    Held, RefinedConfig, RefinedGrids, RefinedPlans, evolve_refined_worker, load_refined_tiles, prim_stencil_boxes, refined_checkpoint,
};
use symbi_io::{Hdf5Backend, IoBackend, Metadata, TreeBuf};
use symbi_sim::checkpoint::{PhysicsIdentity, write_hierarchy_checkpoint};
use symbi_sim::distributed_checkpoint::CheckpointRequest;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_fabric::rendezvous::{Identity, Rendezvous, connect};
use symbi_fabric::{Digest, Fnv, WorkerId};
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::atlas::{ExchangePlan, FieldSchema, Placement, Span, TileId};
use symbi_sim::phase_table::PhaseTable;
use symbi_sim::plan_exchange::{PlanExchange, PlanRole, StagingFault, fabric_for};
use symbi_sim::substrate_seam::RegimeKind;
use symbi_sim::worker::{Injection, WorkerConfig, WorkerError};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
type Hier = Hierarchy<Newtonian, 2, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kern>;

const ROLE: &str = "SYMBI_FABRIC_TEST_ROLE";
const N: usize = 32;
const DX: f64 = 1.0 / N as f64;
const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const NG: usize = 2;
const RATIO: usize = 2;
const STEPS: u64 = 8;
const T_FINAL: f64 = 10.0;
// the refined patch of most arms: root cells 12..20 on both axes
const PATCH: [(usize, usize); 2] = [(12, 20), (12, 20)];
/// the register components of an adiabatic 2D run: den, two momenta, energy
const N_COMP: usize = 4;
const TRANSFER_DEADLINE: Duration = Duration::from_secs(10);
const STARTUP_DEADLINE: Duration = Duration::from_secs(10);
const WATCHDOG: Duration = Duration::from_secs(120);

fn bump(x: f64, y: f64) -> f64 {
    0.3 * (-(((x - 0.5) / 0.08).powi(2) + ((y - 0.5) / 0.08).powi(2))).exp()
}

/// a root tile of `cells` placed as production places it: the whole grid's origin and the index
/// offset of its first cell
fn build_root(cells: [usize; 2], offset: [usize; 2], bnd: Boundaries<2>, ng: usize) -> Sim {
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 2])
        .origin([0.0; 2])
        .index_offset([offset[0] as isize, offset[1] as isize])
        .boundaries(bnd)
        .cfl(CFL)
        .ghosts(ng)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("root sim")
        .set_initial(|[x, y]| {
            let b = bump(x, y);
            Prim::adiabatic(Density(1.0 + b), Tensor::new([0.0, 0.0]), Pressure(1.0 + b))
        })
        .build()
}

fn kset(sim: &Sim) -> Kern {
    Kern::new(GAMMA, CFL, &sim.geom.allocated)
}

fn patch_region(patch: [(usize, usize); 2]) -> RefinementRegion<2> {
    RefinementRegion {
        x_lo: [patch[0].0 as f64 * DX, patch[1].0 as f64 * DX],
        x_hi: [patch[0].1 as f64 * DX, patch[1].1 as f64 * DX],
    }
}

/// the single hierarchy: the whole root with the whole patch
fn build_mono(patch: [(usize, usize); 2], ng: usize) -> Hier {
    let root = build_root([N, N], [0, 0], Boundaries::uniform(BoundaryType::Outflow), ng);
    let k = kset(&root);
    let mut h = Hier::with_refinement(root, k, &[patch_region(patch)], ProlongOrder::Ppm, kset).expect("mono hier");
    h.seed_fine_from_coarse().expect("seed fine");
    h.prime();
    h
}

#[derive(Clone)]
struct Arm {
    cuts0: Vec<usize>,
    cuts1: Vec<usize>,
    owner: Vec<u32>,
    steps: u64,
    /// the refined patch in root cells per axis
    patch: [(usize, usize); 2],
    /// the root ghost depth: a patch face on a cut prolongs its fine ghosts from the root
    /// halo, which then has to hold the prolongation stencil
    ng: usize,
}

impl Arm {
    fn env(&self) -> Vec<(&'static str, String)> {
        let list = |v: &[usize]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
        vec![
            ("W_CUTS0", list(&self.cuts0)),
            ("W_CUTS1", list(&self.cuts1)),
            ("W_OWNER", self.owner.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")),
            ("W_STEPS", self.steps.to_string()),
            ("W_PATCH", format!("{},{},{},{}", self.patch[0].0, self.patch[0].1, self.patch[1].0, self.patch[1].1)),
            ("W_NG", self.ng.to_string()),
        ]
    }

    fn from_env() -> Self {
        let list = |name: &str| -> Vec<usize> { env_list(name).into_iter().map(|x| x as usize).collect() };
        Self {
            cuts0: list("W_CUTS0"),
            cuts1: list("W_CUTS1"),
            owner: list("W_OWNER").into_iter().map(|x| x as u32).collect(),
            steps: env_u64("W_STEPS").unwrap(),
            patch: {
                let p = env_list("W_PATCH");
                [(p[0] as usize, p[1] as usize), (p[2] as usize, p[3] as usize)]
            },
            ng: env_u64("W_NG").unwrap() as usize,
        }
    }

    fn with_steps(mut self, steps: u64) -> Self {
        self.steps = steps;
        self
    }

    fn with_patch(mut self, patch: [(usize, usize); 2], ng: usize) -> Self {
        self.patch = patch;
        self.ng = ng;
        self
    }

    fn partition(&self) -> Partition<2> {
        Partition::explicit([N, N], [self.cuts0.clone(), self.cuts1.clone()]).unwrap()
    }

    fn workers(&self) -> u32 {
        self.owner.iter().max().unwrap() + 1
    }

}

/// the session's parent prim stencils, from the partition alone as the launched driver derives them
fn prim_stencil_boxes_of(partition: &Partition<2>, patch: [(usize, usize); 2], ng: usize) -> Vec<Option<[Span; 2]>> {
    prim_stencil_boxes(partition, patch, ng, ng + 1, ProlongOrder::Ppm, [false; 2])
}

/// a tile keeping, at every attempt's entry, the root cells it gives to the session's prim stencils;
/// a refined tile's own stencil equals the session's entry for it
fn with_prim_entry(mut h: Hier, partition: &Partition<2>, flat: usize, patch: [(usize, usize); 2], ng: usize) -> Hier {
    let boxes = prim_stencil_boxes_of(partition, patch, ng);
    let ext = partition.tile_extents(unflatten(flat, partition.counts()));
    let held = h.prim_stencil().map(|st| -> [Span; 2] {
        std::array::from_fn(|a| Span { lo: st.domain.spaces[a].lo + ext[a].0 as isize, hi: st.domain.spaces[a].hi + ext[a].0 as isize })
    });
    assert_eq!(held, boxes[flat], "tile {flat}: the builder's prim stencil differs from the session's");
    if let Some(cells) = ExchangePlan::cell_stencil_sources(partition, &Topology::open(), &boxes, ng, TileId(flat as u32)) {
        let cells = symbi_algebra::Domain::new(std::array::from_fn(|a| symbi_algebra::Space { name: axis_name(a), lo: cells[a].lo, hi: cells[a].hi }));
        h.enable_prim_entry(&cells).expect("prim entry");
    }
    h
}

/// one root tile's hierarchy: the patch clipped to the tile with the whole patch beside it,
/// a patch face on a cut as a contributing register face; a tile beside such a face with no
/// part of the patch carries the receiving register of the cells that face corrects
fn tile(partition: &Partition<2>, flat: usize, patch: [(usize, usize); 2], ng: usize) -> Hier {
    let counts = partition.counts();
    let tc = unflatten(flat, counts);
    let ext = partition.tile_extents(tc);
    let m: [usize; 2] = [ext[0].1, ext[1].1];
    let offset: [usize; 2] = [ext[0].0, ext[1].0];
    let lo: [f64; 2] = std::array::from_fn(|a| offset[a] as f64 * DX);
    let hi: [f64; 2] = std::array::from_fn(|a| lo[a] + m[a] as f64 * DX);
    let bnd = Boundaries(std::array::from_fn(|a| {
        let face = |edge: bool| if edge { BoundaryType::Outflow } else { BoundaryType::CoarseFine };
        [face(tc[a] == 0), face(tc[a] == counts[a] - 1)]
    }));
    let root = build_root(m, offset, bnd, ng);
    let whole = patch_region(patch);
    let cx = [whole.x_lo[0].max(lo[0]), whole.x_hi[0].min(hi[0])];
    let cy = [whole.x_lo[1].max(lo[1]), whole.x_hi[1].min(hi[1])];
    let k = kset(&root);
    if cx[0] < cx[1] && cy[0] < cy[1] {
        let clipped = RefinementRegion { x_lo: [cx[0], cy[0]], x_hi: [cx[1], cy[1]] };
        return with_prim_entry(
            Hier::with_refinement_of_donating(root, k, &[clipped], &[whole], ProlongOrder::Ppm, kset).expect("tile hier"),
            partition,
            flat,
            patch,
            ng,
        );
    }
    // the patch faces this tile sits beside: beyond a hi face the tile starts at it, beyond
    // a lo face it ends at it, its transverse extent meeting the patch
    let mut faces = [[false; 2]; 2];
    for ax in 0..2 {
        for side in 0..2 {
            let face_pos = if side == 0 { patch[ax].0 } else { patch[ax].1 };
            let beside = if side == 1 { ext[ax].0 == face_pos } else { ext[ax].0 + ext[ax].1 == face_pos };
            let across = (0..2).all(|aa| aa == ax || (ext[aa].0 < patch[aa].1 && ext[aa].0 + ext[aa].1 > patch[aa].0));
            faces[ax][side] = beside && across;
        }
    }
    // a tile whose interior meets the one-cell shell around the patch recovers its cells
    // beside the patch at the restriction point, faces or none
    let meets_shell = (0..2).all(|aa| ext[aa].0 < patch[aa].1 + 1 && ext[aa].0 + ext[aa].1 > patch[aa].0.saturating_sub(1));
    let h = Hier::single(root, k);
    if faces.iter().flatten().any(|&f| f) || meets_shell {
        // the patch box in this tile's index space, clipped to its transverse extent
        let patch_local = symbi_algebra::Domain::new(std::array::from_fn(|aa| {
            let origin = ext[aa].0 as isize;
            let (lo, hi) = if faces[aa].iter().any(|&f| f) {
                (patch[aa].0 as isize - origin, patch[aa].1 as isize - origin)
            } else {
                (patch[aa].0.max(ext[aa].0) as isize - origin, patch[aa].1.min(ext[aa].0 + ext[aa].1) as isize - origin)
            };
            symbi_algebra::Space { name: axis_name(aa), lo, hi }
        }));
        with_prim_entry(h.with_receiving_register(&patch_local, faces).expect("receiving register"), partition, flat, patch, ng)
    } else {
        with_prim_entry(h, partition, flat, patch, ng)
    }
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

/// the fine face fault of this worker: the step from `W_PERTURB_FINE_FACE` on `W_PERTURB_WORKER`
fn env_perturb() -> Option<u64> {
    let me = env_u64("W_ME")?;
    (env_u64("W_PERTURB_WORKER") == Some(me)).then(|| env_u64("W_PERTURB_FINE_FACE")).flatten()
}

/// the fine clock fault of this worker: (step, time|dt|iteration, local fine tile ordinal) from
/// `W_CLOCK_FAULT` on `W_CLOCK_WORKER`
fn env_clock_fault() -> Option<(u64, ClockField, usize)> {
    let me = env_u64("W_ME")?;
    if env_u64("W_CLOCK_WORKER") != Some(me) {
        return None;
    }
    let spec = std::env::var("W_CLOCK_FAULT").ok()?;
    let parts: Vec<&str> = spec.split(',').collect();
    let which = match parts[1] {
        "time" => ClockField::Time,
        "dt" => ClockField::Dt,
        _ => ClockField::Iteration,
    };
    Some((parts[0].parse().unwrap(), which, parts[2].parse().unwrap()))
}

fn env_list(name: &str) -> Vec<u64> {
    std::env::var(name)
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect()
}

fn injection_from_env(me: WorkerId) -> Injection {
    let mut injection = Injection::default();
    if let Some(worker) = env_u64("W_REJECT_ROOT_WORKER") {
        if worker == u64::from(me.0) {
            let step = env_u64("W_REJECT_ROOT_STEP").unwrap();
            let stage = env_u64("W_REJECT_ROOT_STAGE").unwrap() as usize;
            injection.reject_at = vec![(step, stage)];
        }
    }
    // a root cell raised troubled at (step, stage) on whichever worker holds it:
    // `W_ROOT_TROUBLE` = step, stage, i, j on the root grid
    let root = env_list("W_ROOT_TROUBLE");
    if root.len() == 4 {
        injection.trouble_at = Some((root[0], root[1] as usize, [root[2] as isize, root[3] as isize, 0]));
    }
    if let Some(worker) = env_u64("W_CAP_WORKER") {
        if worker == u64::from(me.0) || worker == u64::MAX {
            let step = env_u64("W_CAP_STEP").unwrap();
            let bits = env_u64("W_CAP_BITS").unwrap();
            injection.cap_dt_at = Some((step, bits));
        }
    }
    injection
}

/// the forced fine rejection of this worker: (step, substep, stage) from `W_REJECT_FINE`
fn reject_fine_from_env(me: WorkerId) -> Option<(u64, u8, usize)> {
    let worker = env_u64("W_REJECT_FINE_WORKER")?;
    if worker != u64::from(me.0) {
        return None;
    }
    let parts = env_list("W_REJECT_FINE");
    Some((parts[0], parts[1] as u8, parts[2] as usize))
}

/// the fine cell every worker carries as troubled, raised by the one holding it: (step,
/// substep, stage, i, j) from `W_FINE_TROUBLE`, the cell on the fine plan's grid
fn fine_trouble_from_env() -> Option<(u64, u8, usize, [isize; 3])> {
    let parts = env_list("W_FINE_TROUBLE");
    (parts.len() == 5).then(|| (parts[0], parts[1] as u8, parts[2] as usize, [parts[3] as isize, parts[4] as isize, 0]))
}

/// the flux contribution fault of this worker: (step, substep, stage, dup|drop) from
/// `W_CONTRIB_FAULT`
fn contribution_fault_from_env(me: WorkerId) -> Option<(u64, u8, usize, StagingFault)> {
    let worker = env_u64("W_CONTRIB_FAULT_WORKER")?;
    if worker != u64::from(me.0) {
        return None;
    }
    let spec = std::env::var("W_CONTRIB_FAULT").ok()?;
    let parts: Vec<&str> = spec.split(',').collect();
    let fault = if parts[3] == "dup" { StagingFault::Duplicate } else { StagingFault::Drop };
    Some((parts[0].parse().unwrap(), parts[1].parse().unwrap(), parts[2].parse().unwrap(), fault))
}

/// the invalid timestep candidate of this worker: (step, level) from `W_BAD_CFL`
fn bad_cfl_from_env(me: WorkerId) -> Option<(u64, usize)> {
    let worker = env_u64("W_BAD_CFL_WORKER")?;
    if worker != u64::from(me.0) {
        return None;
    }
    let parts = env_list("W_BAD_CFL");
    Some((parts[0], parts[1] as usize))
}

/// one cell's value in the dump: `kind` 0 = interior conserved, 1 = interior primitive,
/// 2 = cut halo primitive; `(i, j)` global on the record's level; `field` in schema order
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Key {
    kind: u8,
    level: u8,
    i: usize,
    j: usize,
    field: u8,
}

const RECORD_BYTES: usize = 48;

fn encode(records: &[(Key, u64)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(records.len() * RECORD_BYTES);
    for (k, bits) in records {
        for w in [k.kind as u64, k.level as u64, k.i as u64, k.j as u64, k.field as u64, *bits] {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }
    out
}

fn decode(bytes: &[u8]) -> Vec<(Key, u64)> {
    bytes
        .chunks_exact(RECORD_BYTES)
        .map(|c| {
            let u = |k: usize| u64::from_le_bytes(c[8 * k..8 * k + 8].try_into().unwrap());
            (Key { kind: u(0) as u8, level: u(1) as u8, i: u(2) as usize, j: u(3) as usize, field: u(4) as u8 }, u(5))
        })
        .collect()
}

/// every conserved and primitive component of a level's cells: the interior as kinds 0 and
/// 1, the cut halos (the ghost cells inside the level's global domain) as kind 2
fn dump_level(
    out: &mut Vec<(Key, u64)>,
    state: &FieldStore<2, 2, HostMemory, f64>,
    level: u8,
    global: impl Fn(&[isize; 2]) -> [isize; 2],
    domain_lo: [isize; 2],
    domain_hi: [isize; 2],
) {
    let cons = &state.fields.cons;
    let prim = &state.fields.prim;
    let mut cons_fields = vec![&cons.den, &cons.mom[0], &cons.mom[1]];
    cons_fields.extend(cons.nrg_field());
    cons_fields.extend(cons.chi_field());
    let mut prim_fields = vec![&prim.rho, &prim.vel[0], &prim.vel[1]];
    prim_fields.extend(prim.pre_field());
    prim_fields.extend(prim.chi_field());
    for c in state.geom.allocated.iter() {
        let g = global(&c);
        if (0..2).any(|a| g[a] < domain_lo[a] || g[a] >= domain_hi[a]) {
            continue;
        }
        let (i, j) = (g[0] as usize, g[1] as usize);
        if state.geom.interior.contains(c) {
            for (f, field) in cons_fields.iter().enumerate() {
                out.push((Key { kind: 0, level, i, j, field: f as u8 }, field.view().at(c).to_bits()));
            }
            for (f, field) in prim_fields.iter().enumerate() {
                out.push((Key { kind: 1, level, i, j, field: f as u8 }, field.view().at(c).to_bits()));
            }
        } else {
            for (f, field) in prim_fields.iter().enumerate() {
                out.push((Key { kind: 2, level, i, j, field: f as u8 }, field.view().at(c).to_bits()));
            }
        }
    }
}

/// the dump of one hierarchy whose root tile starts at root cell `ext` of the whole grid
fn dump(h: &Hier, ext: [(usize, usize); 2], patch: [(usize, usize); 2]) -> Vec<(Key, u64)> {
    let mut out = Vec::new();
    let root = &h.levels[0].state;
    let rlo: [isize; 2] = std::array::from_fn(|a| root.geom.interior.spaces[a].lo);
    dump_level(
        &mut out,
        root,
        0,
        |c| std::array::from_fn(|a| ext[a].0 as isize + c[a] - rlo[a]),
        [0, 0],
        [N as isize, N as isize],
    );
    if h.levels.len() > 1 {
        let fine = &h.levels[1].state;
        let flo: [isize; 2] = std::array::from_fn(|a| fine.geom.interior.spaces[a].lo);
        let clo: [isize; 2] = std::array::from_fn(|a| h.levels[0].coverage.as_ref().unwrap().spaces[a].lo);
        dump_level(
            &mut out,
            fine,
            1,
            |c| std::array::from_fn(|a| (ext[a].0 as isize + clo[a] - rlo[a]) * RATIO as isize + c[a] - flo[a]),
            std::array::from_fn(|a| (patch[a].0 * RATIO) as isize),
            std::array::from_fn(|a| (patch[a].1 * RATIO) as isize),
        );
    }
    out
}

fn run_worker(arm: &Arm, me: WorkerId, out: PathBuf) -> Result<(), WorkerError> {
    let partition = arm.partition();
    let counts = partition.counts();
    let n_root = partition.n_tiles();
    let owner: Vec<WorkerId> = arm.owner.iter().map(|&w| WorkerId(w)).collect();
    let placement = Placement::new(arm.workers(), owner.clone()).expect("placement");
    let fine_partition = FinePartition::of(&partition, arm.patch, RATIO).expect("fine partition");
    let fine_counts = fine_partition.partition.counts();
    let n_fine = fine_partition.partition.n_tiles();
    let fine_owner: Vec<WorkerId> = (0..n_fine)
        .map(|f| owner[flatten(fine_partition.root_tile_of(unflatten(f, fine_counts)), counts)])
        .collect();
    let fine_placement = Placement::with_idle_workers(arm.workers(), fine_owner).expect("fine placement");
    let mine: Vec<usize> = (0..n_root).filter(|&f| placement.owner_of(TileId(f as u32)) == me).collect();
    let mut tiles: Vec<Hier> = mine.iter().map(|&f| tile(&partition, f, arm.patch, arm.ng)).collect();
    if let Some(bits) = env_u64("W_MAX_DT_BITS") {
        for h in tiles.iter_mut() {
            h.levels[0].state.max_dt = f64::from_bits(bits);
        }
    }
    if let Some((step, substep, stage)) = reject_fine_from_env(me) {
        let refined = tiles.iter_mut().find(|h| h.levels.len() > 1).expect("the rejecting worker holds a fine tile");
        refined.reject_fine_once_at(step, substep, stage);
    }
    let root_ids: Vec<TileId> = mine.iter().map(|&f| TileId(f as u32)).collect();
    let fine_ids: Vec<Option<TileId>> = mine
        .iter()
        .map(|&f| fine_partition.fine_tile_of(unflatten(f, counts)).map(|tc| TileId(flatten(tc, fine_counts) as u32)))
        .collect();
    let root_store = &*tiles[0].levels[0].state;
    let schema = plan_schema(root_store);
    let cons_schema = plan_cons_schema(root_store);
    let ng = arm.ng;
    let root_plan = ExchangePlan::compile(&partition, &Topology::open(), &schema, ng).expect("root plan");
    let seed_plan = ExchangePlan::compile(&partition, &Topology::open(), &cons_schema, ng).expect("seed plan");
    // a fine store indexes its cells in its parent's frame, whose interior starts at zero:
    // its first interior cell sits at the ratio times the patch's offset inside the tile
    let fine_lo = |t: TileId| -> [isize; 2] {
        let tc = unflatten(t.0 as usize, fine_counts);
        let root_ext = partition.tile_extents(fine_partition.root_tile_of(tc));
        std::array::from_fn(|a| (RATIO * arm.patch[a].0.saturating_sub(root_ext[a].0)) as isize)
    };
    let fine_plan = ExchangePlan::compile_shifted(&fine_partition.partition, &Topology::open(), &schema, ng + 1, &fine_lo)
        .expect("fine plan");
    let flux_plan = ExchangePlan::flux_contributions(&partition, &Topology::open(), arm.patch, ng, N_COMP).expect("flux plan");
    let contributions = flux_plan.transfers().count();
    let table = PhaseTable::refined(2, RATIO as u8, 2).expect("table");
    let root = PlanExchange::new(&root_plan, &placement, me, table.clone());
    let seed = PlanExchange::new(&seed_plan, &placement, me, table.clone())
        .with_role(PlanRole::RootSeed, root.lens().len() as u32);
    let fine = PlanExchange::new(&fine_plan, &fine_placement, me, table.clone())
        .with_role(PlanRole::Fine, (root.lens().len() + seed.lens().len()) as u32);
    // the one-shot unrecoverable fine cell, armed on the fine store holding it: `W_FINE_FREEZE` =
    // fine iteration, stage, i, j on the fine plan's grid
    let freeze = env_list("W_FINE_FREEZE");
    if freeze.len() == 4 {
        let cell = [freeze[2] as isize, freeze[3] as isize];
        for (k, id) in fine_ids.iter().enumerate() {
            if let Some(local) = id.and_then(|id| fine.local_cell(id, cell)) {
                *tiles[k].levels[1].state.workspace.unrecoverable_once.lock().unwrap() =
                    vec![(freeze[0], freeze[1] as usize, local, false)];
            }
        }
    }
    let flux = (contributions > 0).then(|| {
        PlanExchange::new(&flux_plan, &placement, me, table.clone())
            .with_role(PlanRole::FluxContribution, (root.lens().len() + seed.lens().len() + fine.lens().len()) as u32)
    });
    // the parent prim stencils of a patch face on a cut under a root halo shallower than the reach
    let boxes = prim_stencil_boxes_of(&partition, arm.patch, ng);
    let stencil_plan = boxes.iter().any(Option::is_some).then(|| {
        let schema = (0..N_COMP).fold(FieldSchema::new(), |schema, k| schema.cell(&format!("prim_stencil_{k}")));
        ExchangePlan::cell_stencil(&partition, &Topology::open(), &boxes, &schema, ng).expect("prim stencil plan")
    });
    let prim_stencil = stencil_plan.as_ref().map(|plan| {
        let offset = (root.lens().len() + seed.lens().len() + fine.lens().len() + flux.as_ref().map_or(0, |f| f.lens().len())) as u32;
        let old = PlanExchange::new(plan, &placement, me, table.clone()).with_role(PlanRole::PrimStencil, offset);
        let new = PlanExchange::new(plan, &placement, me, table.clone()).with_role(PlanRole::PrimStencil, offset + old.lens().len() as u32);
        (old, new)
    });
    let plan_digest = {
        let mut h = Fnv::new();
        for d in [root_plan.digest, seed_plan.digest, fine_plan.digest, flux_plan.digest].into_iter().chain(stencil_plan.as_ref().map(|p| p.digest)) {
            h.u32((d.0 >> 32) as u32);
            h.u32(d.0 as u32);
        }
        h.finish()
    };
    let r = Rendezvous {
        me,
        coordinator: "127.0.0.1:0".parse().unwrap(),
        announce: Some(PathBuf::from(std::env::var("W_COORD").unwrap())),
        bind: None,
        advertise: None,
        identity: Identity {
            credential: 0xBEEF,
            build_id: "worker_refined_processes".into(),
            config_digest: Digest(1),
            plan_digest,
            placement_digest: placement.digest(),
            phase_table_digest: table.digest(),
            ng: ng as u32,
            workers: arm.workers(),
            block_credit: 4096,
        },
        startup: STARTUP_DEADLINE,
    };
    let mut all: Vec<&PlanExchange<'_, 2>> = vec![&root, &seed, &fine];
    all.extend(flux.as_ref());
    if let Some((old, new)) = prim_stencil.as_ref() {
        all.extend([old, new]);
    }
    let max_payload = all.iter().flat_map(|p| p.lens()).max().unwrap_or(0) * 8;
    // a checkpoint block travels on the same link: the payload bound covers the agreed credit
    let (link, session) = connect(&r, max_payload.max(4096))?;
    // the block credit the handshake agreed, which a checkpoint's blocks are sent under
    let mut fabric = fabric_for(&all, link, session).with_block_credit(4096);
    let plans = RefinedPlans { root, seed, fine, flux, emf: None, faces: None, corrected: None, stencil: None, prim_stencil, nested: Vec::new(), coefficients: Vec::new(), bands: Vec::new(), band_seams: Vec::new(), slip_shared: None, slip_cells: None };
    let cfg = RefinedConfig {
        base: WorkerConfig {
            regime: RegimeKind::of::<f64, 2, Newtonian>(),
            timestepping: Timestepping::Rk2,
            start_time: 0.0,
            t_final: T_FINAL,
            max_steps: arm.steps,
            deadline: TRANSFER_DEADLINE,
            injection: injection_from_env(me),
            audit_every_point: false,
        },
        invalid_level_cfl_at: bad_cfl_from_env(me),
        fine_trouble_at: fine_trouble_from_env(),
        contribution_fault: contribution_fault_from_env(me),
        nested_contribution_fault: None,
        restart: std::env::var_os("W_RESTART_FROM").is_some(),
        perturb_fine_face_after: env_perturb(),
        clock_fault: env_clock_fault(),
    };
    let root_devices = vec![0i32; n_root];
    let fine_devices = vec![0i32; n_fine];
    let held = Held {
        root_ids: &root_ids,
        fine_ids: &fine_ids,
        n_root,
        n_fine,
        root_devices: &root_devices,
        fine_devices: &fine_devices,
        nested: &[],
    };
    let grids = RefinedGrids::of([N, N], [0.0, 0.0], [DX; 2], arm.ng, arm.patch);
    let mut cfg = cfg;
    if let Ok(path) = std::env::var("W_RESTART_FROM") {
        load_refined_tiles(&mut tiles, &held, &plans, &grids, &path)?;
        cfg.base.start_time = tiles[0].levels[0].state.time;
    }
    let checkpoint_after = env_u64("W_CKPT_AFTER");
    let checkpoint_path = std::env::var("W_CKPT_PATH").unwrap_or_default();
    let extras = Metadata::new();
    fofc_reset_stats();
    let report = evolve_refined_worker(&mut tiles, &held, &plans, &LocalCopy, &mut fabric, &cfg, None, |_| Ok(()), |accepted, _, tiles, fabric| {
        if checkpoint_after == Some(accepted) {
            let req = CheckpointRequest {
                path: &checkpoint_path,
                extras: &extras,
                budget: 256,
                deadline: TRANSFER_DEADLINE,
                drop_block: None,
                duplicate_first_block: false,
                vote_no: false,
            };
            let identity = PhysicsIdentity::of(&tiles[0].levels[0].state);
            refined_checkpoint(tiles, &held, &plans, &grids, &identity, &req, fabric, accepted, true)?;
        }
        Ok(ControlFlow::Continue(()))
    })?;
    fabric.finish(TRANSFER_DEADLINE)?;
    let mut records = Vec::new();
    for (&flat, h) in mine.iter().zip(&tiles) {
        records.extend(dump(h, partition.tile_extents(unflatten(flat, counts)), arm.patch));
    }
    std::fs::write(&out, encode(&records)).unwrap();
    let (fallbacks, freezes) = fofc_stats();
    println!(
        "REPORT steps={} rejections={} time={} fine_troubled={} fine_neighbor={} neighbor={} fofc={fallbacks},{freezes} contrib={contributions} hw={} dts={}",
        report.steps,
        report.rejections,
        report.time.to_bits(),
        report.fine_troubled_exchanges,
        report.fine_neighbor_corrections,
        report.neighbor_corrections,
        report.grant_high_water.iter().map(|h| h.to_string()).collect::<Vec<_>>().join(","),
        report.dt_sequence.iter().map(|d| d.to_bits().to_string()).collect::<Vec<_>>().join(",")
    );
    // a march every worker agreed to halt ends the worker failing, its state the step entry
    match report.halt {
        Some(cause) => Err(WorkerError::Numerics(format!("halted: {cause}"))),
        None => Ok(()),
    }
}

/// the single hierarchy driven with the given timesteps, dumped like a worker; a process of
/// its own so the correction census is its own
fn mono_main(arm: &Arm, out: PathBuf) {
    let mut mono = build_mono(arm.patch, arm.ng);
    fofc_reset_stats();
    let write_after = env_u64("W_MONO_CKPT_AFTER");
    for (step, bits) in env_list("W_MONO_DTS").into_iter().enumerate() {
        mono.step_root_with_dt(f64::from_bits(bits));
        if write_after == Some(step as u64 + 1) {
            let path = std::env::var("W_MONO_CKPT_PATH").unwrap();
            write_hierarchy_checkpoint(&[&mono.levels[0].state, &mono.levels[1].state], &path, &Metadata::new()).unwrap();
        }
    }
    let (fallbacks, freezes) = fofc_stats();
    std::fs::write(&out, encode(&dump(&mono, [(0, N), (0, N)], arm.patch))).unwrap();
    println!("REPORT steps=0 rejections=0 time=0 fine_troubled=0 fine_neighbor=0 neighbor=0 fofc={fallbacks},{freezes} contrib=0 hw= dts=");
}

fn worker_main() -> ! {
    let arm = Arm::from_env();
    let out = PathBuf::from(std::env::var("W_OUT").unwrap());
    if std::env::var_os("W_MONO_DTS").is_some() {
        mono_main(&arm, out);
        std::process::exit(0);
    }
    let me = WorkerId(env_u64("W_ME").unwrap() as u32);
    match run_worker(&arm, me, out) {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            eprintln!("worker {me:?}: {e}");
            std::process::exit(2);
        }
    }
}

fn in_worker_role() -> bool {
    std::env::var(ROLE).is_ok()
}

struct Spawned {
    child: Child,
    out: PathBuf,
}

fn spawn(test: &str, me: u32, arm: &Arm, coord: &std::path::Path, dir: &std::path::Path, extra: &[(&str, String)]) -> Spawned {
    let out = dir.join(format!("worker_{me}.cells"));
    let mut cmd = Command::new(std::env::current_exe().unwrap());
    cmd.args([test, "--exact", "--nocapture", "--test-threads=1"])
        .env(ROLE, "worker")
        .env("W_ME", me.to_string())
        .env("W_COORD", coord)
        .env("W_OUT", &out)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (k, v) in arm.env() {
        cmd.env(k, v);
    }
    for (k, v) in extra {
        cmd.env(k, v);
    }
    let child = cmd.spawn().unwrap();
    Spawned { child, out }
}

#[derive(Debug, Clone, PartialEq)]
struct Report {
    steps: u64,
    rejections: u64,
    time: u64,
    fine_troubled: u64,
    fine_neighbor: u64,
    neighbor: u64,
    /// the correction census of the process: (fallback cells, frozen cells)
    fofc: (u64, u64),
    /// the flux contributions of the session's plan
    contrib: u64,
    high_water: Vec<usize>,
    dts: Vec<u64>,
}

struct Outcome {
    code: Option<i32>,
    stderr: String,
    report: Option<Report>,
    records: Vec<(Key, u64)>,
}

fn parse_report(stdout: &str) -> Option<Report> {
    let line = stdout.lines().find(|l| l.contains("REPORT "))?;
    let line = &line[line.find("REPORT ")?..];
    let field = |name: &str| -> Option<String> {
        line.split_whitespace().find_map(|kv| kv.strip_prefix(&format!("{name}=")).map(str::to_string))
    };
    let list = |name: &str| -> Option<Vec<u64>> {
        Some(field(name)?.split(',').filter(|s| !s.is_empty()).map(|s| s.parse().unwrap()).collect())
    };
    let fofc = list("fofc")?;
    Some(Report {
        steps: field("steps")?.parse().ok()?,
        rejections: field("rejections")?.parse().ok()?,
        time: field("time")?.parse().ok()?,
        fine_troubled: field("fine_troubled")?.parse().ok()?,
        fine_neighbor: field("fine_neighbor")?.parse().ok()?,
        neighbor: field("neighbor")?.parse().ok()?,
        fofc: (fofc[0], fofc[1]),
        contrib: field("contrib")?.parse().ok()?,
        high_water: list("hw")?.into_iter().map(|h| h as usize).collect(),
        dts: list("dts")?,
    })
}

fn collect(mut s: Spawned) -> Outcome {
    let status = s.child.wait().unwrap();
    let mut stdout = String::new();
    s.child.stdout.take().unwrap().read_to_string(&mut stdout).unwrap();
    let mut stderr = String::new();
    s.child.stderr.take().unwrap().read_to_string(&mut stderr).unwrap();
    let records = std::fs::read(&s.out).map(|bytes| decode(&bytes)).unwrap_or_default();
    Outcome { code: status.code(), stderr, report: parse_report(&stdout), records }
}

fn run_dir(test: &str, tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("symbi_refined_worker_{}_{test}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn run_arm(test: &str, tag: &str, arm: &Arm, extra: &[(&str, String)]) -> Vec<Outcome> {
    let dir = run_dir(test, tag);
    let coord = dir.join("rendezvous");
    let mut spawned: Vec<Spawned> = (0..arm.workers()).map(|w| spawn(test, w, arm, &coord, &dir, extra)).collect();
    let start = Instant::now();
    loop {
        if spawned.iter_mut().all(|s| s.child.try_wait().unwrap().is_some()) {
            break;
        }
        if start.elapsed() > WATCHDOG {
            for s in spawned.iter_mut() {
                let _ = s.child.kill();
            }
            panic!("the watchdog had to intervene");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    spawned.into_iter().map(collect).collect()
}

fn assert_all_ok(outcomes: &[Outcome], label: &str) {
    for (w, o) in outcomes.iter().enumerate() {
        assert_eq!(o.code, Some(0), "{label}: worker {w} failed:\n{}", o.stderr);
    }
}

/// the single hierarchy driven with the launched run's timesteps, as a record map, with its
/// correction census
struct Reference {
    values: HashMap<Key, u64>,
    fofc: (u64, u64),
}

impl Reference {
    /// the interior records of a run with no cut: the single hierarchy or one worker
    fn of(outcomes: &[Outcome], label: &str) -> Self {
        let mut values = HashMap::new();
        let mut fofc = (0, 0);
        for o in outcomes {
            assert_eq!(o.code, Some(0), "{label} failed:\n{}", o.stderr);
            for &(k, bits) in &o.records {
                assert!(k.kind < 2, "{label} holds a cut halo");
                assert!(values.insert(k, bits).is_none());
            }
            let f = o.report.as_ref().unwrap().fofc;
            fofc = (fofc.0 + f.0, fofc.1 + f.1);
        }
        Reference { values, fofc }
    }
}

fn reference(test: &str, tag: &str, arm: &Arm, dts: &[u64]) -> Reference {
    let dir = run_dir(test, &format!("{tag}_mono"));
    let coord = dir.join("rendezvous");
    let dts = ("W_MONO_DTS", dts.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","));
    Reference::of(&[collect(spawn(test, 0, arm, &coord, &dir, &[dts]))], "the single hierarchy")
}

/// the name of a differing record class for the failure message
fn class(k: &Key) -> String {
    let kind = ["conserved", "primitive", "halo primitive"][k.kind as usize];
    format!("level {} {kind} field {}", k.level, k.field)
}

/// every interior cell of both levels held exactly once across the workers and equal to
/// the single hierarchy in every conserved and primitive component; every cut halo
/// primitive equal to the single hierarchy's interior value at that cell
fn assert_state_equal(outcomes: &[Outcome], expected: &Reference, label: &str) {
    let mut held: HashMap<Key, u64> = HashMap::new();
    let mut halos = 0usize;
    // per class: the count of differing cells and the index box they span
    let mut differing: HashMap<String, (usize, [usize; 2], [usize; 2])> = HashMap::new();
    for (w, o) in outcomes.iter().enumerate() {
        for &(k, bits) in &o.records {
            let want = if k.kind == 2 {
                halos += 1;
                expected.values.get(&Key { kind: 1, ..k })
            } else {
                assert!(held.insert(k, bits).is_none(), "{label}: {k:?} held by two workers");
                expected.values.get(&k)
            };
            let Some(&want) = want else {
                panic!("{label}: worker {w} holds {k:?}, which the single hierarchy has no value for");
            };
            if want != bits {
                let e = differing.entry(class(&k)).or_insert((0, [usize::MAX; 2], [0; 2]));
                e.0 += 1;
                e.1 = [e.1[0].min(k.i), e.1[1].min(k.j)];
                e.2 = [e.2[0].max(k.i), e.2[1].max(k.j)];
            }
        }
    }
    assert_eq!(
        held.len(),
        expected.values.len(),
        "{label}: the workers hold {} interior values, the single hierarchy {}",
        held.len(),
        expected.values.len()
    );
    let mut report: Vec<String> = differing
        .iter()
        .map(|(c, (n, lo, hi))| format!("{c}: {n} cells in [{}, {}] x [{}, {}]", lo[0], hi[0], lo[1], hi[1]))
        .collect();
    report.sort();
    assert!(differing.is_empty(), "{label}: values differing from the single hierarchy by class:\n{}", report.join("\n"));
    assert!(
        halos > 0 || outcomes.len() == 1,
        "{label}: no cut halo was dumped; the freshness comparison is vacuous"
    );
}

/// the launched run agrees with itself across workers and with the single hierarchy
fn assert_matches_reference(test: &str, tag: &str, outcomes: &[Outcome], arm: &Arm, label: &str) -> Reference {
    assert_matches_reference_after(test, tag, outcomes, arm, label, &[])
}

/// `assert_matches_reference` for a run resumed after the steps of `before`: the single
/// hierarchy runs those steps and then the resumed run's
fn assert_matches_reference_after(test: &str, tag: &str, outcomes: &[Outcome], arm: &Arm, label: &str, before: &[u64]) -> Reference {
    assert_all_ok(outcomes, label);
    let reports: Vec<&Report> = outcomes.iter().map(|o| o.report.as_ref().expect("report")).collect();
    for (w, r) in reports.iter().enumerate() {
        assert_eq!(r.dts, reports[0].dts, "{label}: worker {w} took other timesteps");
        // the cap is the absolute root iteration: a resumed run takes the steps the checkpoint left
        assert_eq!(r.steps, arm.steps - before.len() as u64, "{label}: worker {w} took {} steps", r.steps);
        assert!(
            r.high_water.iter().all(|&h| h <= 2),
            "{label}: worker {w} held more grants than one per axis per peer: {:?}",
            r.high_water
        );
        eprintln!("{label}: worker {w} grant high-water {:?}", r.high_water);
    }
    let dts: Vec<u64> = before.iter().chain(&reports[0].dts).copied().collect();
    let expected = reference(test, tag, arm, &dts);
    assert_state_equal(outcomes, &expected, label);
    expected
}

fn ragged_17_13() -> Arm {
    Arm { cuts0: vec![17], cuts1: vec![13], owner: vec![0, 1, 2, 3], steps: STEPS, patch: PATCH, ng: NG }
}

fn ragged_18_14() -> Arm {
    Arm { cuts0: vec![18], cuts1: vec![14], owner: vec![0, 1, 2, 3], steps: STEPS, patch: PATCH, ng: NG }
}

/// one worker holding the whole grid: the launched path with no cut
fn uncut() -> Arm {
    Arm { cuts0: vec![], cuts1: vec![], owner: vec![0], steps: STEPS, patch: PATCH, ng: NG }
}

/// the patch's hi face on axis 0 lies on the cut at 17: workers 0 and 1 hold the fine faces
/// of that face, workers 2 and 3 the corrected root cells beside it. the fine ghosts beyond
/// that face are prolonged from the root halo, which then holds the PPM stencil at five cells
fn patch_face_on_cut() -> Arm {
    ragged_17_13().with_patch([(12, 17), (12, 20)], 5)
}

/// the patch's lo face on axis 0 on the cut at 12 and its hi face on axis 1 on the cut at 20, under
/// the production root halo of two cells: the fine ghosts past both faces prolong from the parent
/// prim stencil gathered from the tiles across the cuts, the corner from the diagonal tile
fn production_halo_faces() -> Arm {
    Arm { cuts0: vec![12], cuts1: vec![20], owner: vec![0, 1, 2, 3], steps: STEPS, patch: PATCH, ng: NG }
}

/// the same faces with a three-cell column beside the lo face: the stencil's four cells past that
/// face come from two tiles on two workers, the corner from a third
fn production_halo_two_owners() -> Arm {
    Arm { cuts0: vec![9, 12], cuts1: vec![20], owner: vec![0, 1, 2, 2, 0, 1], steps: STEPS, patch: PATCH, ng: NG }
}

/// three workers on a 3x2 tile grid: the patch lies in the middle column, whose two tiles
/// worker 1 owns, so workers 0 and 2 hold no fine tile
fn no_fine_on_two_workers() -> Arm {
    Arm { cuts0: vec![10, 22], cuts1: vec![16], owner: vec![0, 0, 1, 1, 2, 2], steps: STEPS, patch: PATCH, ng: NG }
}

#[test]
fn a_ragged_layout_matches_the_single_hierarchy_over_several_root_steps() {
    if in_worker_role() {
        worker_main();
    }
    let test = "a_ragged_layout_matches_the_single_hierarchy_over_several_root_steps";
    for (tag, arm) in [("17_13", ragged_17_13()), ("18_14", ragged_18_14())] {
        assert_matches_reference(test, tag, &run_arm(test, tag, &arm, &[]), &arm, &format!("cuts {tag}"));
    }
}

#[test]
fn workers_without_fine_tiles_walk_the_transaction_with_the_others() {
    if in_worker_role() {
        worker_main();
    }
    let test = "workers_without_fine_tiles_walk_the_transaction_with_the_others";
    let arm = no_fine_on_two_workers();
    let outcomes = run_arm(test, "no_fine", &arm, &[]);
    assert_matches_reference(test, "no_fine", &outcomes, &arm, "two of three workers without fine tiles");
}

/// a clean run capped to `bits` at `step` on every worker
fn capped(test: &str, tag: &str, arm: &Arm, step: u64, bits: u64) -> Vec<Outcome> {
    run_arm(
        test,
        tag,
        arm,
        &[
            ("W_CAP_WORKER", u64::MAX.to_string()),
            ("W_CAP_STEP", step.to_string()),
            ("W_CAP_BITS", bits.to_string()),
        ],
    )
}

/// the rejecting run took the same timesteps as its clean control, rejected once on every
/// worker, and holds the same records; both equal the single hierarchy
fn assert_rejection_matches_capped_control(test: &str, arm: &Arm, rejecting: Vec<Outcome>, step: u64, label: &str) {
    assert_all_ok(&rejecting, label);
    let reports: Vec<&Report> = rejecting.iter().map(|o| o.report.as_ref().unwrap()).collect();
    for (w, r) in reports.iter().enumerate() {
        assert_eq!(r.rejections, 1, "{label}: worker {w} reports {} rejections", r.rejections);
    }
    let dts = reports[0].dts.clone();
    assert_matches_reference(test, "rejecting", &rejecting, arm, label);
    let clean = capped(test, "clean", arm, step, dts[step as usize]);
    assert_matches_reference(test, "clean", &clean, arm, &format!("{label}, the clean control"));
    let clean_reports: Vec<&Report> = clean.iter().map(|o| o.report.as_ref().unwrap()).collect();
    assert_eq!(clean_reports[0].dts, dts, "{label}: the capped clean run took other timesteps than the rejecting one");
    assert!(clean_reports.iter().all(|r| r.rejections == 0));
    for (w, (a, b)) in rejecting.iter().zip(&clean).enumerate() {
        assert!(a.records == b.records, "{label}: worker {w} holds other records than in the clean control");
    }
}

/// worker 1 rejects the last stage of step 3 on the root; every worker rolls the step back
/// and replays it at the halved timestep
#[test]
fn a_rejection_at_a_root_stage_rolls_every_worker_back() {
    if in_worker_role() {
        worker_main();
    }
    let test = "a_rejection_at_a_root_stage_rolls_every_worker_back";
    let arm = ragged_17_13();
    let rejecting = run_arm(
        test,
        "rejecting",
        &arm,
        &[
            ("W_REJECT_ROOT_WORKER", "1".into()),
            ("W_REJECT_ROOT_STEP", "3".into()),
            ("W_REJECT_ROOT_STAGE", "1".into()),
        ],
    );
    assert_rejection_matches_capped_control(test, &arm, rejecting, 3, "the root-rejecting run");
}

/// worker 1 rejects the last stage of the second fine substep of step 3; every worker rolls
/// the root step back and replays it at the halved timestep, and the run equals a clean run
/// whose timestep at that step is capped to the accepted one
#[test]
fn a_rejection_in_the_second_fine_substep_rolls_every_worker_back() {
    if in_worker_role() {
        worker_main();
    }
    let test = "a_rejection_in_the_second_fine_substep_rolls_every_worker_back";
    let arm = ragged_17_13();
    let rejecting = run_arm(
        test,
        "rejecting",
        &arm,
        &[("W_REJECT_FINE_WORKER", "1".into()), ("W_REJECT_FINE", "3,1,1".into())],
    );
    assert_rejection_matches_capped_control(test, &arm, rejecting, 3, "the fine-rejecting run");
}

/// the only worker holding fine tiles rejects in the second fine substep while the other
/// two hold none: the sequence closes through their empty fine phases and every worker
/// rolls back
#[test]
fn a_fine_rejection_rolls_back_workers_without_fine_tiles() {
    if in_worker_role() {
        worker_main();
    }
    let test = "a_fine_rejection_rolls_back_workers_without_fine_tiles";
    let arm = no_fine_on_two_workers();
    let rejecting = run_arm(
        test,
        "rejecting",
        &arm,
        &[("W_REJECT_FINE_WORKER", "1".into()), ("W_REJECT_FINE", "3,1,1".into())],
    );
    assert_rejection_matches_capped_control(test, &arm, rejecting, 3, "the fine-rejecting run with empty fine workers");
}

/// a NaN timestep candidate on the fine level of one worker under a positive user cap halts
/// every worker at the entry of step 3: the cap applies after the candidate is judged, the
/// worker holding it names it and its level, every other worker the peer's, and each holds the
/// state of a clean run capped at three steps
#[test]
fn an_invalid_fine_candidate_under_a_positive_cap_halts_every_worker() {
    if in_worker_role() {
        worker_main();
    }
    let test = "an_invalid_fine_candidate_under_a_positive_cap_halts_every_worker";
    let arm = ragged_17_13();
    let cap = ("W_MAX_DT_BITS", 0.005f64.to_bits().to_string());
    let started = Instant::now();
    let outcomes = run_arm(test, "nan", &arm, &[cap.clone(), ("W_BAD_CFL_WORKER", "1".into()), ("W_BAD_CFL", "3,1".into())]);
    assert!(started.elapsed() < TRANSFER_DEADLINE * 3, "termination took {:?}", started.elapsed());
    for (w, o) in outcomes.iter().enumerate() {
        assert_ne!(o.code, Some(0), "worker {w} exited zero: the cap hid the invalid candidate");
        let r = o.report.as_ref().unwrap_or_else(|| panic!("worker {w} published no report:\n{}", o.stderr));
        assert_eq!(r.steps, 3, "worker {w} halted after {} steps", r.steps);
        let named = if w == 1 { "halted: invalid CFL candidate" } else { "halted: an invalid CFL candidate on another worker at iter 3" };
        assert!(o.stderr.contains(named), "worker {w}:\n{}", o.stderr);
    }
    assert!(outcomes[1].stderr.contains("on level 1"), "{}", outcomes[1].stderr);
    let clean = run_arm(test, "clean", &Arm { steps: 3, ..arm.clone() }, &[cap]);
    assert_all_ok(&clean, "the clean run capped at three steps");
    for (w, (a, b)) in outcomes.iter().zip(&clean).enumerate() {
        assert!(!a.records.is_empty() && a.records == b.records, "worker {w} holds other records than the clean run at the halted step entry");
    }
}

/// a fine cell against the fine cut is flagged troubled on the worker holding it, in the
/// second fine substep, with the worker across the cut clean: the flags cross the cut, the
/// clean worker redoes its side of the shared faces, the freeze tier stays silent, and the
/// run equals one worker holding the whole grid with the same flag raised
#[test]
fn trouble_on_one_side_of_a_fine_cut_is_corrected_on_both() {
    if in_worker_role() {
        worker_main();
    }
    let test = "trouble_on_one_side_of_a_fine_cut_is_corrected_on_both";
    // the fine cut of the cuts (17, 13) layout sits at fine column 10 of the patch's grid;
    // the flagged cell is the column left of it, in a row above the fine cut at row 2
    let flag = ("W_FINE_TROUBLE", "3,1,0,9,7".to_string());
    let uncut_clean = run_arm(test, "uncut_clean", &uncut(), &[]);
    let uncut_flagged = run_arm(test, "uncut_flagged", &uncut(), &[flag.clone()]);
    let clean = Reference::of(&uncut_clean, "the uncut clean run");
    let expected = Reference::of(&uncut_flagged, "the uncut flagged run");
    assert!(clean.values != expected.values, "the flag left the uncut run unchanged; the gate is vacuous");
    assert_eq!(expected.fofc.1, 0, "the freeze tier acted in the uncut run");
    let arm = ragged_17_13();
    let outcomes = run_arm(test, "cut", &arm, &[flag]);
    assert_all_ok(&outcomes, "the cut flagged run");
    let reports: Vec<&Report> = outcomes.iter().map(|o| o.report.as_ref().unwrap()).collect();
    let uncut_report = uncut_flagged[0].report.as_ref().unwrap();
    assert_eq!(uncut_report.dts, reports[0].dts, "the cut run took other timesteps than the uncut one");
    for (w, r) in reports.iter().enumerate() {
        assert_eq!(r.dts, reports[0].dts, "worker {w} took other timesteps");
        assert_eq!(r.fine_troubled, 1, "worker {w} saw {} fine troubled exchanges", r.fine_troubled);
        assert_eq!(r.fofc.1, 0, "worker {w} froze {} cells", r.fofc.1);
        assert!(r.high_water.iter().all(|&h| h <= 2), "worker {w} grant high-water {:?}", r.high_water);
        eprintln!("worker {w}: fine neighbor corrections {}, fofc {:?}", r.fine_neighbor, r.fofc);
    }
    // the worker across the fine cut holds a clean fine interior and corrects once from the
    // exchanged flag; the flagged worker corrects from its own mark
    let neighbors: Vec<usize> = (0..4).filter(|&w| reports[w].fine_neighbor > 0).collect();
    assert_eq!(neighbors, vec![3], "the workers correcting a clean fine tile from the flag");
    assert_eq!(reports[3].fine_neighbor, 1);
    assert_state_equal(&outcomes, &expected, "the cut flagged run");
}

/// the run under `flag` on the (17, 13) layout equals one worker holding the whole grid under
/// the same flag, bit for bit, with the same timesteps and the freeze tier silent; the flag
/// changes the one-worker run
fn assert_flag_matches_uncut(test: &str, flag: (&'static str, String)) {
    let uncut_clean = run_arm(test, "uncut_clean", &uncut(), &[]);
    let uncut_flagged = run_arm(test, "uncut_flagged", &uncut(), &[flag.clone()]);
    let clean = Reference::of(&uncut_clean, "the uncut clean run");
    let expected = Reference::of(&uncut_flagged, "the uncut flagged run");
    assert!(clean.values != expected.values, "{test}: the flag left the uncut run unchanged; the gate is vacuous");
    assert_eq!(expected.fofc.1, 0, "{test}: the freeze tier acted in the uncut run");
    let outcomes = run_arm(test, "cut", &ragged_17_13(), &[flag]);
    assert_all_ok(&outcomes, "the cut flagged run");
    let reports: Vec<&Report> = outcomes.iter().map(|o| o.report.as_ref().unwrap()).collect();
    let uncut_report = uncut_flagged[0].report.as_ref().unwrap();
    for (w, r) in reports.iter().enumerate() {
        assert_eq!(r.dts, uncut_report.dts, "{test}: worker {w} took other timesteps than the uncut run");
        assert_eq!(r.fofc.1, 0, "{test}: worker {w} froze {} cells", r.fofc.1);
    }
    assert_state_equal(&outcomes, &expected, "the cut flagged run");
}

/// a fine cell made unrecoverable in the predictor of the third root step's second fine substep,
/// six fine cells from the fine column cut and eight from the row cut, so no neighbor holds a
/// copy of its poisoned state: the correcting select freezes it, the freeze is accepted, and the
/// run equals one worker holding the whole grid under the same poison with the same freeze. the
/// frozen cell holds its poisoned stage input, so the next step would refuse its timestep
/// candidate: both runs end with the step that freezes
#[test]
fn an_accepted_fine_freeze_matches_the_single_hierarchy() {
    if in_worker_role() {
        worker_main();
    }
    let test = "an_accepted_fine_freeze_matches_the_single_hierarchy";
    let poison = ("W_FINE_FREEZE", "5,0,4,10".to_string());
    let last = |arm: Arm| Arm { steps: 3, ..arm };
    let uncut_clean = run_arm(test, "uncut_clean", &last(uncut()), &[]);
    let uncut_frozen = run_arm(test, "uncut_frozen", &last(uncut()), &[poison.clone()]);
    let clean = Reference::of(&uncut_clean, "the uncut clean run");
    let expected = Reference::of(&uncut_frozen, "the uncut frozen run");
    assert!(clean.values != expected.values, "the poison left the uncut run unchanged; the gate is vacuous");
    assert!(expected.fofc.1 >= 1, "the uncut run froze no cell; the freeze never happened");
    let outcomes = run_arm(test, "cut", &last(ragged_17_13()), &[poison]);
    assert_all_ok(&outcomes, "the cut frozen run");
    let reports: Vec<&Report> = outcomes.iter().map(|o| o.report.as_ref().unwrap()).collect();
    let uncut_report = uncut_frozen[0].report.as_ref().unwrap();
    let frozen: u64 = reports.iter().map(|r| r.fofc.1).sum();
    assert_eq!(frozen, expected.fofc.1, "the cut run froze {frozen} cells against {} uncut", expected.fofc.1);
    for (w, r) in reports.iter().enumerate() {
        assert_eq!(r.dts, uncut_report.dts, "worker {w} took other timesteps than the uncut run");
    }
    assert_state_equal(&outcomes, &expected, "the cut frozen run");
}

/// the fine cell of the fine-cut arm above, flagged in the corrector of the second fine substep
#[test]
fn trouble_on_one_side_of_a_fine_cut_in_the_corrector_is_corrected_on_both() {
    if in_worker_role() {
        worker_main();
    }
    assert_flag_matches_uncut(
        "trouble_on_one_side_of_a_fine_cut_in_the_corrector_is_corrected_on_both",
        ("W_FINE_TROUBLE", "3,1,1,9,7".to_string()),
    );
}

/// a fine cell in the corner of the fine cuts (fine column 10, fine row 2), flagged in the
/// predictor and in the corrector: the flags cross both cuts and the diagonal neighbor
#[test]
fn trouble_in_a_fine_tile_corner_is_corrected_on_every_side() {
    if in_worker_role() {
        worker_main();
    }
    for stage in [0, 1] {
        assert_flag_matches_uncut(
            "trouble_in_a_fine_tile_corner_is_corrected_on_every_side",
            ("W_FINE_TROUBLE", format!("3,1,{stage},9,1")),
        );
    }
}

/// a root cell against the root cut at 17, outside the refined patch, flagged in the predictor
/// and in the corrector of a root step
#[test]
fn trouble_against_a_root_cut_outside_the_patch_is_corrected_on_both() {
    if in_worker_role() {
        worker_main();
    }
    for stage in [0, 1] {
        assert_flag_matches_uncut(
            "trouble_against_a_root_cut_outside_the_patch_is_corrected_on_both",
            ("W_ROOT_TROUBLE", format!("3,{stage},16,4")),
        );
    }
}

/// the patch's hi face lies on a cut: the fine faces' units cross to the neighbor holding
/// the corrected cells at every fine stage and the run equals the single hierarchy in every
/// component; a rejection in the second fine substep rolls the registers back with the rest
#[test]
fn a_patch_face_on_a_cut_refluxes_through_the_neighbor() {
    if in_worker_role() {
        worker_main();
    }
    let test = "a_patch_face_on_a_cut_refluxes_through_the_neighbor";
    let arm = patch_face_on_cut();
    let outcomes = run_arm(test, "cut_face", &arm, &[]);
    assert_matches_reference(test, "cut_face", &outcomes, &arm, "the patch face on the cut");
    for (w, o) in outcomes.iter().enumerate() {
        let r = o.report.as_ref().unwrap();
        assert_eq!(r.contrib, 2 * N_COMP as u64, "worker {w}: the plan holds {} contributions", r.contrib);
    }
    let rejecting = run_arm(
        test,
        "rejecting",
        &arm,
        &[("W_REJECT_FINE_WORKER", "1".into()), ("W_REJECT_FINE", "3,1,1".into())],
    );
    assert_rejection_matches_capped_control(test, &arm, rejecting, 3, "the fine-rejecting run with a patch face on the cut");
}

/// patch faces on cuts under the production root halo: every arm's stencil reaches past it, the
/// run equals the single hierarchy in every component, a rejection in the second fine substep
/// replays from a regathered stencil, and the quad layout's checkpoint resumed onto the layout
/// continues bitwise
#[test]
fn patch_faces_on_cuts_at_the_production_halo_match_through_the_prim_stencil() {
    if in_worker_role() {
        worker_main();
    }
    let test = "patch_faces_on_cuts_at_the_production_halo_match_through_the_prim_stencil";
    for (tag, arm) in [("faces", production_halo_faces()), ("two_owners", production_halo_two_owners())] {
        assert!(
            prim_stencil_boxes_of(&arm.partition(), arm.patch, arm.ng).iter().any(Option::is_some),
            "{tag}: no tile reads a prim stencil; the gate is vacuous"
        );
        let outcomes = run_arm(test, tag, &arm, &[]);
        assert_matches_reference(test, tag, &outcomes, &arm, &format!("{tag}: patch faces on cuts at the production halo"));
    }
    let arm = production_halo_faces();
    let rejecting = run_arm(test, "rejecting", &arm, &[("W_REJECT_FINE_WORKER", "2".into()), ("W_REJECT_FINE", "3,1,1".into())]);
    assert_rejection_matches_capped_control(test, &arm, rejecting, 3, "the fine-rejecting run with patch faces on cuts at the production halo");
    let dir = run_dir(test, "files");
    let (ckpt, before) = quad_checkpoint(test, &dir);
    let resumed = run_arm(test, "resumed", &arm, &[("W_RESTART_FROM", ckpt.to_str().unwrap().into())]);
    assert_matches_reference_after(test, "resumed", &resumed, &arm, "resumed onto patch faces on cuts at the production halo", &before);
}

/// a contribution sent twice and one never sent each end the session on every worker
/// before the reflux, with no state published
#[test]
fn a_duplicated_or_missing_contribution_ends_the_run_before_the_reflux() {
    if in_worker_role() {
        worker_main();
    }
    let test = "a_duplicated_or_missing_contribution_ends_the_run_before_the_reflux";
    let arm = patch_face_on_cut();
    // a duplicate is caught where the phase opens: the grant of the receiving worker
    // carries the plan's send count and the phase lists twice as many
    for (tag, spec, marker) in [("dup", "3,1,0,dup", "this phase sends it"), ("drop", "3,1,0,drop", "deadline")] {
        let started = Instant::now();
        let outcomes = run_arm(
            test,
            tag,
            &arm,
            &[("W_CONTRIB_FAULT_WORKER", "0".into()), ("W_CONTRIB_FAULT", spec.into())],
        );
        assert!(started.elapsed() < TRANSFER_DEADLINE * 4, "{tag}: termination took {:?}", started.elapsed());
        for (w, o) in outcomes.iter().enumerate() {
            assert_ne!(o.code, Some(0), "{tag}: worker {w} exited zero");
            assert!(o.records.is_empty(), "{tag}: worker {w} published its state");
        }
        assert!(
            outcomes.iter().any(|o| o.stderr.contains(marker)),
            "{tag}: no worker reported the fault:\n{}",
            outcomes.iter().map(|o| o.stderr.clone()).collect::<Vec<_>>().join("\n---\n")
        );
    }
}


/// every attribute, dataset and group of two checkpoint trees equal, the datasets bit for bit
fn assert_trees_equal(a: &TreeBuf, b: &TreeBuf, path: &str) {
    assert_eq!(a.attrs.len(), b.attrs.len(), "{path}: attribute count");
    for (x, y) in a.attrs.iter().zip(&b.attrs) {
        assert_eq!(x.0, y.0, "{path}: attribute name");
        assert_eq!(format!("{:?}", x.1), format!("{:?}", y.1), "{path}: attribute {}", x.0);
    }
    assert_eq!(a.datasets.len(), b.datasets.len(), "{path}: dataset count");
    for (x, y) in a.datasets.iter().zip(&b.datasets) {
        assert_eq!(x.name, y.name, "{path}: dataset name");
        assert_eq!(x.shape, y.shape, "{path}/{}: shape", x.name);
        match (x.data.as_f64(), y.data.as_f64()) {
            (Some(p), Some(q)) => {
                let differing: Vec<usize> = p.iter().zip(q).enumerate().filter(|(_, (u, v))| u.to_bits() != v.to_bits()).map(|(i, _)| i).collect();
                assert!(
                    differing.is_empty(),
                    "{path}/{}: {} of {} values differ, first at {}",
                    x.name,
                    differing.len(),
                    p.len(),
                    differing[0]
                );
            }
            _ => assert_eq!(format!("{:?}", x.data), format!("{:?}", y.data), "{path}/{}", x.name),
        }
    }
    assert_eq!(
        a.groups.iter().map(|g| g.name.clone()).collect::<Vec<_>>(),
        b.groups.iter().map(|g| g.name.clone()).collect::<Vec<_>>(),
        "{path}: groups"
    );
    for (x, y) in a.groups.iter().zip(&b.groups) {
        assert_trees_equal(x, y, &format!("{path}/{}", x.name));
    }
}

/// the checkpoint the quad layout writes after four root steps: every attribute and dataset of
/// both levels equal to the file the single hierarchy writes after the same four steps
fn quad_checkpoint(test: &str, dir: &std::path::Path) -> (PathBuf, Vec<u64>) {
    let ckpt = dir.join("refined.h5");
    let arm = ragged_17_13().with_steps(4);
    let written = run_arm(
        test,
        "write",
        &arm,
        &[("W_CKPT_AFTER", "4".into()), ("W_CKPT_PATH", ckpt.to_str().unwrap().into())],
    );
    assert_all_ok(&written, "the checkpointing run");
    let dts = written[0].report.as_ref().unwrap().dts.clone();
    assert!(ckpt.exists(), "the checkpoint was not published");
    let mono_path = dir.join("mono.h5");
    let mono_dir = run_dir(test, "write_mono");
    let coord = mono_dir.join("rendezvous");
    let mono = collect(spawn(
        test,
        0,
        &arm,
        &coord,
        &mono_dir,
        &[
            ("W_MONO_DTS", dts.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",")),
            ("W_MONO_CKPT_AFTER", "4".into()),
            ("W_MONO_CKPT_PATH", mono_path.to_str().unwrap().into()),
        ],
    ));
    assert_eq!(mono.code, Some(0), "the single hierarchy failed:\n{}", mono.stderr);
    let got = Hdf5Backend.read(&ckpt).unwrap();
    let want = Hdf5Backend.read(&mono_path).unwrap();
    assert_trees_equal(&got, &want, "");
    (ckpt, dts)
}

#[test]
fn a_refined_checkpoint_equals_the_single_hierarchys_file() {
    if in_worker_role() {
        worker_main();
    }
    let test = "a_refined_checkpoint_equals_the_single_hierarchys_file";
    let dir = run_dir(test, "files");
    quad_checkpoint(test, &dir);
}

/// the quad layout's checkpoint after four steps resumed on another ragged quad and on a line
/// of three workers of which two hold no fine tile: four more steps equal the single hierarchy
/// run uninterrupted through eight, in every value on both levels
#[test]
fn a_refined_restart_onto_another_partition_continues_bitwise() {
    if in_worker_role() {
        worker_main();
    }
    let test = "a_refined_restart_onto_another_partition_continues_bitwise";
    let dir = run_dir(test, "files");
    let (ckpt, before) = quad_checkpoint(test, &dir);
    for (tag, arm) in [("18_14", ragged_18_14()), ("no_fine", no_fine_on_two_workers())] {
        // the cap is the absolute root iteration: the checkpoint's four steps and the rest
        let arm = arm.with_steps(STEPS);
        let resumed = run_arm(test, tag, &arm, &[("W_RESTART_FROM", ckpt.to_str().unwrap().into())]);
        assert_matches_reference_after(test, tag, &resumed, &arm, &format!("resumed on {tag}"), &before);
    }
}

/// the fine level's clock differing across its tiles: time, dt and iteration each moved on one
/// tile, across workers on the quad layout (each worker one fine tile) and within one worker on a
/// line where worker 1 holds both fine tiles and the others hold none. every worker ends before
/// the file opens, naming the fine clock
#[test]
fn a_fine_clock_disagreement_ends_the_run_before_publication() {
    if in_worker_role() {
        worker_main();
    }
    let test = "a_fine_clock_disagreement_ends_the_run_before_publication";
    let line = Arm { owner: vec![0, 0, 1, 1, 2, 2], ..no_fine_on_two_workers() };
    let cases: [(&str, Arm, &str, &str); 6] = [
        ("time_across", ragged_17_13(), "2", "4,time,0"),
        ("dt_across", ragged_17_13(), "2", "4,dt,0"),
        ("iteration_across", ragged_17_13(), "0", "4,iteration,0"),
        ("time_within", line.clone(), "1", "4,time,1"),
        ("dt_within", line.clone(), "1", "4,dt,1"),
        ("iteration_within", line, "1", "4,iteration,1"),
    ];
    for (tag, arm, worker, fault) in cases {
        let dir = run_dir(test, &format!("{tag}_files"));
        let ckpt = dir.join("refined.h5");
        let outcomes = run_arm(
            test,
            tag,
            &arm,
            &[
                ("W_CLOCK_WORKER", worker.into()),
                ("W_CLOCK_FAULT", fault.into()),
                ("W_CKPT_AFTER", "4".into()),
                ("W_CKPT_PATH", ckpt.to_str().unwrap().into()),
            ],
        );
        for (w, o) in outcomes.iter().enumerate() {
            assert_ne!(o.code, Some(0), "{tag}: worker {w} exited zero");
            assert!(o.records.is_empty(), "{tag}: worker {w} published its state");
        }
        assert!(
            outcomes.iter().any(|o| o.stderr.contains("the clock of level 1 differs")),
            "{tag}: no worker named the fine clock:\n{}",
            outcomes.iter().map(|o| o.stderr.clone()).collect::<Vec<_>>().join("\n---\n")
        );
        assert!(!ckpt.exists(), "{tag}: a checkpoint was published");
    }
}
