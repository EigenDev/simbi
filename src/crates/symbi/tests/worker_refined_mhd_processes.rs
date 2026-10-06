// =============================================================================
// worker_refined_mhd_processes.rs
//
// the refined ideal-MHD transaction across worker processes: a magnetized
// root with one refined patch of ratio 2, advanced through the fabric by
// `evolve_refined_worker` with the flux and EMF registers crossing the cuts
// the patch's faces lie on, against the single hierarchy driven with the same
// timesteps in a process of its own. the comparison covers every conserved,
// primitive and cell-field component of every interior cell on both levels,
// every cut halo primitive, and every staggered face on both levels, a face
// on a cut recorded by both of its tiles and required equal; div B is at
// round-off on every tile from its own faces and on the assembled root. every
// refined tile reads its coarse-fine face halo from a parent face stencil
// gathered from the faces' owners; the stencil at the last step's entry is
// compared value for value with the single hierarchy's faces there, the
// values gathered from other tiles included. arms: the patch inside the tiles
// across fine cuts on both axes and on one axis at a time; a rejection in the
// second fine substep; a patch face on a cut and a patch corner on two cuts,
// each with a rejection in the second fine substep; a fine cell against a fine
// cut flagged troubled in the predictor and in the corrector under the Contact
// and the UCT edge EMF, the four-worker run held to one worker under the same
// flag; a magnetized run without the stencil refused by name.
//
// run: cargo test -p symbi --test worker_refined_mhd_processes
// =============================================================================
use std::collections::HashMap;
use std::io::Read;
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use symbi::regimes::fofc::{fofc_reset_stats, fofc_stats};
use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::decomp::{FinePartition, LocalCopy, Partition, Topology, flatten, plan_cons_schema, plan_schema, unflatten};
use symbi::sim::refinement::launched::{
    ClockField,
    Held, RefinedConfig, RefinedGrids, RefinedPlans, evolve_refined_worker, face_stencil_boxes_of_tile, load_refined_tiles,
    refined_checkpoint,
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
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::atlas::{ExchangePlan, Placement, Span, TileId};
use symbi_sim::phase_table::PhaseTable;
use symbi_sim::plan_exchange::{PlanExchange, PlanRole, fabric_for};
use symbi_sim::substrate_seam::RegimeKind;
use symbi_sim::worker::{Injection, WorkerConfig, WorkerError};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = NewtonianMhdSubstrateKernelSet<HostMemory, f64, 2>;
type Hier = Hierarchy<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kern>;

const ROLE: &str = "SYMBI_FABRIC_TEST_ROLE";
const N: usize = 32;
const DX: f64 = 1.0 / N as f64;
const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const RATIO: usize = 2;
const STEPS: u64 = 8;
const T_FINAL: f64 = 10.0;
/// the uniform field the bump stirs: its cell and face components
const B0: [f64; 3] = [0.5, 0.25, 0.2];
/// the flux register components of a 2.5D magnetized run: den, three momenta, energy, and
/// the out-of-plane cell field
const N_COMP: usize = 6;
/// the bound on |div B| per tile and assembled, from each tile's own faces
const DIV_TOL: f64 = 1.0e-10;
const TRANSFER_DEADLINE: Duration = Duration::from_secs(10);
const STARTUP_DEADLINE: Duration = Duration::from_secs(10);
const WATCHDOG: Duration = Duration::from_secs(180);

fn bump(x: f64, y: f64) -> f64 {
    0.3 * (-(((x - 0.5) / 0.08).powi(2) + ((y - 0.5) / 0.08).powi(2))).exp()
}

/// a root tile on the whole grid's origin with its first cell at whole-grid index `offset`
fn build_root(cells: [usize; 2], offset: [usize; 2], bnd: Boundaries<2>, ng: usize) -> Sim {
    Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 2])
        .origin([0.0; 2])
        .index_offset(std::array::from_fn(|a| offset[a] as isize))
        .boundaries(bnd)
        .cfl(CFL)
        .ghosts(ng)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("root sim")
        .set_initial(|[x, y]| {
            let b = bump(x, y);
            MhdPrim::new(
                Prim::adiabatic(Density(1.0 + b), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0 + b)),
                Tensor::new(B0),
            )
        })
        .seed_faces_uniform([B0[0], B0[1]])
        .build()
        .with_rollback_snapshot()
        .expect("rollback snapshot")
}

/// the kernel set, its edge EMF Contact or, under `W_CT=uct`, UCT
fn kset(sim: &Sim) -> Kern {
    let k = Kern::new(GAMMA, CFL, 1.0, &sim.geom.allocated);
    if std::env::var("W_CT").is_ok_and(|v| v == "uct") { k.ct_method(CtMethod::Uct) } else { k }
}

fn patch_region(patch: [(usize, usize); 2]) -> RefinementRegion<2> {
    RefinementRegion {
        x_lo: [patch[0].0 as f64 * DX, patch[1].0 as f64 * DX],
        x_hi: [patch[0].1 as f64 * DX, patch[1].1 as f64 * DX],
    }
}

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
    patch: [(usize, usize); 2],
    ng: usize,
}

impl Arm {
    fn env(&self) -> Vec<(&'static str, String)> {
        let list = |v: &[usize]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
        vec![
            ("W_CUTS0", list(&self.cuts0)),
            ("W_CUTS1", list(&self.cuts1)),
            ("W_OWNER", self.owner.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")),
            ("W_PATCH", format!("{},{},{},{}", self.patch[0].0, self.patch[0].1, self.patch[1].0, self.patch[1].1)),
            ("W_NG", self.ng.to_string()),
        ]
    }

    fn from_env() -> Self {
        let list = |name: &str| -> Vec<usize> { env_list(name).into_iter().map(|x| x as usize).collect() };
        let p = env_list("W_PATCH");
        Self {
            cuts0: list("W_CUTS0"),
            cuts1: list("W_CUTS1"),
            owner: list("W_OWNER").into_iter().map(|x| x as u32).collect(),
            patch: [(p[0] as usize, p[1] as usize), (p[2] as usize, p[3] as usize)],
            ng: env_u64("W_NG").unwrap() as usize,
        }
    }

    fn partition(&self) -> Partition<2> {
        Partition::explicit([N, N], [self.cuts0.clone(), self.cuts1.clone()]).unwrap()
    }

    fn workers(&self) -> u32 {
        self.owner.iter().max().unwrap() + 1
    }
}

/// one root tile's hierarchy: the patch clipped to the tile with the whole patch beside it,
/// a patch face on a cut as a contributing register face; a tile whose interior meets the
/// shell around the patch, faces or none, carries the receiving registers of that shell
fn tile(partition: &Partition<2>, flat: usize, patch: [(usize, usize); 2], ng: usize) -> Hier {
    let counts = partition.counts();
    let tc = unflatten(flat, counts);
    let ext = partition.tile_extents(tc);
    let m: [usize; 2] = [ext[0].1, ext[1].1];
    let offset: [usize; 2] = [ext[0].0, ext[1].0];
    // the tile's physical box, which clips the patch
    let lo: [f64; 2] = std::array::from_fn(|a| ext[a].0 as f64 * DX);
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
        let mut h = Hier::with_refinement_of_contributing(root, k, &[clipped], &[whole], ProlongOrder::Ppm, kset)
            .expect("tile hier");
        h.enable_face_stencil().expect("face stencil");
        return h;
    }
    let mut faces = [[false; 2]; 2];
    for ax in 0..2 {
        for side in 0..2 {
            let face_pos = if side == 0 { patch[ax].0 } else { patch[ax].1 };
            let beside = if side == 1 { ext[ax].0 == face_pos } else { ext[ax].0 + ext[ax].1 == face_pos };
            let across = (0..2).all(|aa| aa == ax || (ext[aa].0 < patch[aa].1 && ext[aa].0 + ext[aa].1 > patch[aa].0));
            faces[ax][side] = beside && across;
        }
    }
    let meets_shell = (0..2).all(|aa| ext[aa].0 < patch[aa].1 + 1 && ext[aa].0 + ext[aa].1 > patch[aa].0.saturating_sub(1));
    let mut h = Hier::single(root, k);
    h.enable_face_stencil().expect("face entry");
    if faces.iter().flatten().any(|&f| f) || meets_shell {
        let patch_local = symbi_algebra::Domain::new(std::array::from_fn(|aa| {
            let origin = ext[aa].0 as isize;
            let (lo, hi) = if faces[aa].iter().any(|&f| f) {
                (patch[aa].0 as isize - origin, patch[aa].1 as isize - origin)
            } else {
                (patch[aa].0.max(ext[aa].0) as isize - origin, patch[aa].1.min(ext[aa].0 + ext[aa].1) as isize - origin)
            };
            symbi_algebra::Space { name: axis_name(aa), lo, hi }
        }));
        h.with_receiving_register(&patch_local, faces).expect("receiving register")
    } else {
        h
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
    // a root cell raised troubled at (step, stage) on whichever worker holds it:
    // `W_ROOT_TROUBLE` = step, stage, i, j on the root grid
    if let Some(worker) = env_u64("W_REJECT_ROOT_WORKER") {
        if worker == u64::from(me.0) {
            injection.reject_at = vec![(env_u64("W_REJECT_ROOT_STEP").unwrap(), env_u64("W_REJECT_ROOT_STAGE").unwrap() as usize)];
        }
    }
    let root = env_list("W_ROOT_TROUBLE");
    if root.len() == 4 {
        injection.trouble_at = Some((root[0], root[1] as usize, [root[2] as isize, root[3] as isize, 0]));
    }
    if let Some(worker) = env_u64("W_CAP_WORKER") {
        if worker == u64::from(me.0) || worker == u64::MAX {
            injection.cap_dt_at = Some((env_u64("W_CAP_STEP").unwrap(), env_u64("W_CAP_BITS").unwrap()));
        }
    }
    injection
}

fn reject_fine_from_env(me: WorkerId) -> Option<(u64, u8, usize)> {
    let worker = env_u64("W_REJECT_FINE_WORKER")?;
    if worker != u64::from(me.0) {
        return None;
    }
    let parts = env_list("W_REJECT_FINE");
    Some((parts[0], parts[1] as u8, parts[2] as usize))
}

/// one value in the dump: `kind` 0 = interior conserved, 1 = interior primitive, 2 = cut
/// halo primitive, 3 = interior cell field, 4 = staggered face (both copies of a face on a
/// cut); `(i, j)` global on the record's level; `field` the component
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

/// every component of a level: the interior conserved, primitive and cell-field values,
/// the cut halo primitives, and the staggered faces of the tile's own face range with the
/// closing copies on its cuts
fn dump_level(
    out: &mut Vec<(Key, u64)>,
    state: &FieldStore<2, 3, HostMemory, f64>,
    level: u8,
    global: impl Fn(&[isize; 2]) -> [isize; 2],
    domain_lo: [isize; 2],
    domain_hi: [isize; 2],
) {
    let cons = &state.fields.cons;
    let prim = &state.fields.prim;
    let mhd = state.fields.mhd.as_ref().expect("a magnetized level");
    let cons_fields = vec![&cons.den, &cons.mom[0], &cons.mom[1], &cons.mom[2], cons.nrg_field().unwrap()];
    let prim_fields = vec![&prim.rho, &prim.vel[0], &prim.vel[1], &prim.vel[2], prim.pre_field().unwrap()];
    let bcell: Vec<_> = mhd.bcell.b.iter().collect();
    let inside = |g: [isize; 2]| (0..2).all(|a| g[a] >= domain_lo[a] && g[a] < domain_hi[a]);
    for c in state.geom.allocated.iter() {
        let g = global(&c);
        if !inside(g) {
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
            for (f, field) in bcell.iter().enumerate() {
                out.push((Key { kind: 3, level, i, j, field: f as u8 }, field.view().at(c).to_bits()));
            }
        } else {
            for (f, field) in prim_fields.iter().enumerate() {
                out.push((Key { kind: 2, level, i, j, field: f as u8 }, field.view().at(c).to_bits()));
            }
        }
    }
    // the faces: along axis d the interior's face range holds one face past the last cell
    let interior = &state.geom.interior;
    for d in 0..2 {
        let view = mhd.bface.b[d].view();
        let faces = interior.extend(d, 0, 1);
        for c in faces.iter() {
            let g = global(&c);
            let ok = (0..2).all(|a| g[a] >= domain_lo[a] && g[a] < domain_hi[a] + isize::from(a == d));
            if !ok {
                continue;
            }
            out.push((Key { kind: 4, level, i: g[0] as usize, j: g[1] as usize, field: d as u8 }, view.at(c).to_bits()));
        }
    }
}

fn dump(h: &Hier, ext: [(usize, usize); 2], patch: [(usize, usize); 2]) -> Vec<(Key, u64)> {
    let mut out = Vec::new();
    let root = &h.levels[0].state;
    let rlo: [isize; 2] = std::array::from_fn(|a| root.geom.interior.spaces[a].lo);
    dump_level(&mut out, root, 0, |c| std::array::from_fn(|a| ext[a].0 as isize + c[a] - rlo[a]), [0, 0], [N as isize, N as isize]);
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

/// the largest |div B| over a level's interior cells from that level's own faces
fn level_div(state: &FieldStore<2, 3, HostMemory, f64>) -> f64 {
    let mhd = state.fields.mhd.as_ref().unwrap();
    let (bx, by) = (mhd.bface.b[0].view(), mhd.bface.b[1].view());
    let dx = state.geom.dx;
    let mut worst = 0.0f64;
    for c in state.geom.interior.iter() {
        let div = (*bx.at([c[0] + 1, c[1]]) - *bx.at(c)) / dx[0] + (*by.at([c[0], c[1] + 1]) - *by.at(c)) / dx[1];
        worst = worst.max(div.abs());
    }
    worst
}

fn hierarchy_div(h: &Hier) -> f64 {
    h.levels.iter().map(|l| level_div(&l.state)).fold(0.0, f64::max)
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
    let open = Topology::open();
    let root_plan = ExchangePlan::compile(&partition, &open, &schema, ng).expect("root plan");
    let seed_plan = ExchangePlan::compile(&partition, &open, &cons_schema, ng).expect("seed plan");
    let fine_lo = |t: TileId| -> [isize; 2] {
        let tc = unflatten(t.0 as usize, fine_counts);
        let root_ext = partition.tile_extents(fine_partition.root_tile_of(tc));
        std::array::from_fn(|a| (RATIO * arm.patch[a].0.saturating_sub(root_ext[a].0)) as isize)
    };
    let fine_plan = ExchangePlan::compile_shifted(&fine_partition.partition, &open, &schema, ng + 1, &fine_lo).expect("fine plan");
    let flux_plan = ExchangePlan::flux_contributions(&partition, &open, arm.patch, ng, N_COMP).expect("flux plan");
    let emf_plan = ExchangePlan::emf_contributions(&partition, &open, arm.patch, ng).expect("emf plan");
    let faces_plan = ExchangePlan::interface_faces(&partition, &open, arm.patch, ng).expect("faces plan");
    let corrected_plan = ExchangePlan::corrected_faces(&partition, &open, arm.patch, ng).expect("corrected plan");
    // every refined tile's stencil box in whole-grid face indices, derived from the partition
    // and the patch; each held tile's hierarchy allocated exactly those boxes
    for (&f, h) in mine.iter().zip(&tiles) {
        let derived = face_stencil_boxes_of_tile(&partition, arm.patch, TileId(f as u32));
        let built = h.face_stencil().map(|_| h.face_stencil_boxes());
        let spans = |b: Option<[Option<symbi_algebra::Domain<2>>; 2]>| {
            b.map(|b| b.map(|d| d.map(|d| [(d.spaces[0].lo, d.spaces[0].hi), (d.spaces[1].lo, d.spaces[1].hi)])))
        };
        assert_eq!(spans(derived), spans(built), "tile {f}: the derived stencil boxes differ from the hierarchy's");
    }
    let boxes: Vec<Option<Vec<[Span; 2]>>> = (0..n_root)
        .map(|f| {
            let ext = partition.tile_extents(unflatten(f, counts));
            face_stencil_boxes_of_tile(&partition, arm.patch, TileId(f as u32)).map(|boxes| {
                boxes
                    .iter()
                    .map(|d| {
                        let d = d.as_ref().expect("every component");
                        std::array::from_fn(|a| Span { lo: d.spaces[a].lo + ext[a].0 as isize, hi: d.spaces[a].hi + ext[a].0 as isize })
                    })
                    .collect()
            })
        })
        .collect();
    let stencil_plan = ExchangePlan::face_stencil(&partition, &open, &boxes, ng).expect("stencil plan");
    let table = PhaseTable::refined(2, RATIO as u8, 2).expect("table");
    // the plans occupy disjoint fabric id ranges in this order
    let root = PlanExchange::new(&root_plan, &placement, me, table.clone());
    let mut offset = root.lens().len() as u32;
    let seed = PlanExchange::new(&seed_plan, &placement, me, table.clone()).with_role(PlanRole::RootSeed, offset);
    offset += seed.lens().len() as u32;
    let fine = PlanExchange::new(&fine_plan, &fine_placement, me, table.clone()).with_role(PlanRole::Fine, offset);
    offset += fine.lens().len() as u32;
    let transported = flux_plan.transfers().count() > 0;
    let flux = transported.then(|| PlanExchange::new(&flux_plan, &placement, me, table.clone()).with_role(PlanRole::FluxContribution, offset));
    offset += flux.as_ref().map_or(0, |p| p.lens().len() as u32);
    let emf = transported.then(|| PlanExchange::new(&emf_plan, &placement, me, table.clone()).with_role(PlanRole::EmfContribution, offset));
    offset += emf.as_ref().map_or(0, |p| p.lens().len() as u32);
    let faces = transported.then(|| PlanExchange::new(&faces_plan, &placement, me, table.clone()).with_role(PlanRole::InterfaceFace, offset));
    offset += faces.as_ref().map_or(0, |p| p.lens().len() as u32);
    let corrected = (corrected_plan.transfers().count() > 0)
        .then(|| PlanExchange::new(&corrected_plan, &placement, me, table.clone()).with_role(PlanRole::CorrectedFace, offset));
    offset += corrected.as_ref().map_or(0, |p| p.lens().len() as u32);
    let with_stencil = std::env::var_os("W_NO_STENCIL").is_none();
    let stencil_old = with_stencil.then(|| PlanExchange::new(&stencil_plan, &placement, me, table.clone()).with_role(PlanRole::FaceStencil, offset));
    offset += stencil_old.as_ref().map_or(0, |p| p.lens().len() as u32);
    let stencil_new = with_stencil.then(|| PlanExchange::new(&stencil_plan, &placement, me, table.clone()).with_role(PlanRole::FaceStencil, offset));
    let plan_digest = {
        let mut h = Fnv::new();
        for d in [root_plan.digest, seed_plan.digest, fine_plan.digest, flux_plan.digest, emf_plan.digest, faces_plan.digest, corrected_plan.digest, stencil_plan.digest] {
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
            credential: 0xB0B0,
            build_id: "worker_refined_mhd_processes".into(),
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
    all.extend(emf.as_ref());
    all.extend(faces.as_ref());
    all.extend(corrected.as_ref());
    all.extend(stencil_old.as_ref());
    all.extend(stencil_new.as_ref());
    let max_payload = all.iter().flat_map(|p| p.lens()).max().unwrap_or(0) * 8;
    // a checkpoint block travels on the same link: the payload bound covers the agreed credit
    let (link, session) = connect(&r, max_payload.max(4096))?;
    // the block credit the handshake agreed, which a checkpoint's blocks are sent under
    let mut fabric = fabric_for(&all, link, session).with_block_credit(4096);
    let stencil = stencil_old.zip(stencil_new);
    let plans = RefinedPlans { root, seed, fine, flux, emf, faces, corrected, stencil, nested: Vec::new(), coefficients: Vec::new() };
    let cfg = RefinedConfig {
        base: WorkerConfig {
            regime: RegimeKind::of::<f64, 2, NewtonianMhd>(),
            timestepping: Timestepping::Rk2,
            start_time: 0.0,
            t_final: T_FINAL,
            max_steps: env_u64("W_STEPS").unwrap_or(STEPS),
            deadline: TRANSFER_DEADLINE,
            injection: injection_from_env(me),
            audit_every_point: std::env::var_os("W_NO_STAGE_AUDIT").is_none(),
        },
        invalid_level_cfl_at: None,
        fine_trouble_at: {
            let p = env_list("W_FINE_TROUBLE");
            (p.len() == 5).then(|| (p[0], p[1] as u8, p[2] as usize, [p[3] as isize, p[4] as isize, 0]))
        },
        contribution_fault: None,
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
    let mut off_tile = 0u64;
    for (&flat, h) in mine.iter().zip(&tiles) {
        let ext = partition.tile_extents(unflatten(flat, counts));
        records.extend(dump(h, ext, arm.patch));
        // the stencil's old level: the root faces at the last attempt's entry, by whole-grid
        // face index, kind 5
        if let Some(stencil) = h.face_stencil() {
            for (d, field) in stencil.old.iter().enumerate() {
                let lattice = h.levels[0].state.fields.mhd.as_ref().unwrap().bface[d].domain().clone();
                for c in field.domain().iter() {
                    let g = [c[0] + ext[0].0 as isize, c[1] + ext[1].0 as isize];
                    off_tile += u64::from(!lattice.contains(c));
                    records.push((Key { kind: 5, level: 0, i: g[0] as usize, j: g[1] as usize, field: d as u8 }, field.view().at(c).to_bits()));
                }
            }
        }
    }
    std::fs::write(&out, encode(&records)).unwrap();
    let div = tiles.iter().map(hierarchy_div).fold(0.0, f64::max);
    println!(
        "REPORT steps={} rejections={} audits={} div={} contrib={} emf={} corrected={} hw={} dts={} corner=0 off_tile={off_tile} oop=0 fine_troubled={} fine_neighbor={} fofc={},{}",
        report.steps,
        report.rejections,
        report.audits,
        div.to_bits(),
        flux_plan.transfers().count(),
        emf_plan.transfers().count(),
        corrected_plan.transfers().count(),
        report.grant_high_water.iter().map(|h| h.to_string()).collect::<Vec<_>>().join(","),
        report.dt_sequence.iter().map(|d| d.to_bits().to_string()).collect::<Vec<_>>().join(","),
        report.fine_troubled_exchanges,
        report.fine_neighbor_corrections,
        fofc_stats().0,
        fofc_stats().1
    );
    Ok(())
}

/// the single hierarchy driven with the given timesteps, dumped like a worker, with its
/// divergence and the EMF register's value at the patch's upper corner node
fn mono_main(arm: &Arm, out: PathBuf) {
    let mut mono = build_mono(arm.patch, arm.ng);
    let dts = env_list("W_MONO_DTS");
    let mut entry: Vec<(Key, u64)> = Vec::new();
    for (step, bits) in dts.iter().enumerate() {
        if step + 1 == dts.len() {
            let mhd = mono.levels[0].state.fields.mhd.as_ref().unwrap();
            for d in 0..2 {
                for c in mhd.bface[d].domain().iter() {
                    if (0..2).all(|a| c[a] >= 0 && c[a] <= N as isize) {
                        entry.push((Key { kind: 5, level: 0, i: c[0] as usize, j: c[1] as usize, field: d as u8 }, mhd.bface[d].view().at(c).to_bits()));
                    }
                }
            }
        }
        mono.step_root_with_dt(f64::from_bits(*bits));
        if env_u64("W_MONO_CKPT_AFTER") == Some(step as u64 + 1) {
            let path = std::env::var("W_MONO_CKPT_PATH").unwrap();
            write_hierarchy_checkpoint(&[&mono.levels[0].state, &mono.levels[1].state], &path, &Metadata::new()).unwrap();
        }
    }
    let mut records = dump(&mono, [(0, N), (0, N)], arm.patch);
    records.extend(entry);
    std::fs::write(&out, encode(&records)).unwrap();
    let corner = mono.emf_registers[0]
        .as_ref()
        .and_then(|reg| reg.slab_of(2, 0, 1).map(|(field, _)| *field.view().at([arm.patch[0].1 as isize, arm.patch[1].1 as isize])))
        .unwrap_or(0.0);
    // the largest out-of-plane flux mismatch on the patch's hi face along axis 0 at the last step
    let oop = mono.flux_registers[0]
        .oop_face(0, 1)
        .map(|(fields, dom)| dom.iter().map(|c| fields[0].view().at(c).abs()).fold(0.0, f64::max))
        .unwrap_or(0.0);
    println!(
        "REPORT steps=0 rejections=0 audits=0 div={} contrib=0 emf=0 corrected=0 hw= dts= corner={} off_tile=0 oop={} fine_troubled=0 fine_neighbor=0 fofc=0,0",
        hierarchy_div(&mono).to_bits(),
        corner.to_bits(),
        oop.to_bits()
    );
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
    Spawned { child: cmd.spawn().unwrap(), out }
}

#[derive(Debug, Clone, PartialEq)]
struct Report {
    steps: u64,
    rejections: u64,
    audits: u64,
    div: f64,
    contrib: u64,
    emf: u64,
    corrected: u64,
    corner: f64,
    /// the single hierarchy's largest out-of-plane flux mismatch on the patch's hi face
    oop: f64,
    /// the stencil values the worker gathered from faces past its own tiles' lattices
    off_tile: u64,
    fine_troubled: u64,
    fine_neighbor: u64,
    /// the correction census of the process: (fallback cells, frozen cells)
    fofc: (u64, u64),
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
    let bits = |name: &str| -> Option<f64> { Some(f64::from_bits(field(name)?.parse().ok()?)) };
    Some(Report {
        steps: field("steps")?.parse().ok()?,
        rejections: field("rejections")?.parse().ok()?,
        audits: field("audits")?.parse().ok()?,
        div: bits("div")?,
        contrib: field("contrib")?.parse().ok()?,
        emf: field("emf")?.parse().ok()?,
        corrected: field("corrected")?.parse().ok()?,
        corner: bits("corner")?,
        oop: bits("oop")?,
        off_tile: field("off_tile")?.parse().ok()?,
        fine_troubled: field("fine_troubled")?.parse().ok()?,
        fine_neighbor: field("fine_neighbor")?.parse().ok()?,
        fofc: {
            let f = list("fofc")?;
            (f[0], f[1])
        },
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
    let dir = std::env::temp_dir().join(format!("symbi_refined_mhd_worker_{}_{test}_{tag}", std::process::id()));
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
    let failed: Vec<String> = outcomes
        .iter()
        .enumerate()
        .filter(|(_, o)| o.code != Some(0))
        .map(|(w, o)| format!("worker {w} exited {:?}:\n{}", o.code, o.stderr))
        .collect();
    assert!(failed.is_empty(), "{label}: {} worker(s) failed:\n{}", failed.len(), failed.join("\n---\n"));
}

struct Reference {
    values: HashMap<Key, u64>,
    report: Report,
}

fn reference(test: &str, tag: &str, arm: &Arm, dts: &[u64]) -> Reference {
    let dir = run_dir(test, &format!("{tag}_mono"));
    let coord = dir.join("rendezvous");
    let dts = ("W_MONO_DTS", dts.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","));
    let o = collect(spawn(test, 0, arm, &coord, &dir, &[dts]));
    assert_eq!(o.code, Some(0), "the single hierarchy failed:\n{}", o.stderr);
    let mut values = HashMap::new();
    for (k, bits) in o.records {
        assert!(k.kind != 2, "the single hierarchy holds no cut halo");
        if k.kind == 5 && values.contains_key(&k) {
            continue;
        }
        assert!(values.insert(k, bits).is_none(), "the single hierarchy recorded {k:?} twice");
    }
    Reference { values, report: o.report.unwrap() }
}

fn class(k: &Key) -> String {
    let kind = ["conserved", "primitive", "halo primitive", "cell field", "face", "stencil face"][k.kind as usize];
    format!("level {} {kind} field {}", k.level, k.field)
}

/// every interior value held exactly once across the workers and equal to the single
/// hierarchy's; every cut halo primitive equal to the interior value there; every face held
/// by two tiles equal in both copies and equal to the single hierarchy's
fn assert_state_equal(outcomes: &[Outcome], expected: &Reference, label: &str) {
    assert_state_equal_with(outcomes, expected, label, true)
}

/// `assert_state_equal` against a reference that holds no stencil values
fn assert_state_equal_without_stencil(outcomes: &[Outcome], expected: &Reference, label: &str) {
    assert_state_equal_with(outcomes, expected, label, false)
}

fn assert_state_equal_with(outcomes: &[Outcome], expected: &Reference, label: &str, with_stencil: bool) {
    let mut held: HashMap<Key, u64> = HashMap::new();
    let mut copies: HashMap<Key, Vec<u64>> = HashMap::new();
    let mut halos = 0usize;
    let mut stencil = 0usize;
    let mut differing: HashMap<String, (usize, [usize; 2], [usize; 2])> = HashMap::new();
    let mut worst: HashMap<String, f64> = HashMap::new();
    let mut note = |k: &Key, got: u64, want: u64| {
        let e = differing.entry(class(k)).or_insert((0, [usize::MAX; 2], [0; 2]));
        e.0 += 1;
        e.1 = [e.1[0].min(k.i), e.1[1].min(k.j)];
        e.2 = [e.2[0].max(k.i), e.2[1].max(k.j)];
        let (g, w) = (f64::from_bits(got), f64::from_bits(want));
        let rel = (g - w).abs() / w.abs().max(1e-300);
        let slot = worst.entry(class(k)).or_insert(0.0);
        *slot = slot.max(rel);
    };
    for (w, o) in outcomes.iter().enumerate() {
        for &(k, bits) in &o.records {
            match k.kind {
                2 => {
                    halos += 1;
                    let want = expected.values.get(&Key { kind: 1, ..k });
                    let Some(&want) = want else { panic!("{label}: worker {w} holds a halo {k:?} the single hierarchy has no value for") };
                    if want != bits {
                        note(&k, bits, want);
                    }
                }
                4 => copies.entry(k).or_default().push(bits),
                5 => {
                    stencil += 1;
                    let Some(&want) = expected.values.get(&k) else { panic!("{label}: worker {w} gathers a stencil face {k:?} the single hierarchy has no value for") };
                    if want != bits {
                        note(&k, bits, want);
                    }
                }
                _ => {
                    assert!(held.insert(k, bits).is_none(), "{label}: {k:?} held by two workers");
                    let Some(&want) = expected.values.get(&k) else { panic!("{label}: worker {w} holds {k:?}, which the single hierarchy has no value for") };
                    if want != bits {
                        note(&k, bits, want);
                    }
                }
            }
        }
    }
    let mut shared = 0usize;
    for (k, values) in &copies {
        if values.len() > 1 {
            shared += 1;
            assert!(values.iter().all(|v| v == &values[0]), "{label}: the copies of {k:?} differ: {values:?}");
        }
        let Some(&want) = expected.values.get(k) else { panic!("{label}: the workers hold {k:?}, which the single hierarchy has no value for") };
        if want != values[0] {
            note(k, values[0], want);
        }
    }
    let interior_expected = expected.values.iter().filter(|(k, _)| k.kind < 4).count();
    let faces_expected = expected.values.iter().filter(|(k, _)| k.kind == 4).count();
    assert_eq!(held.len(), interior_expected, "{label}: the workers hold {} interior values, the single hierarchy {interior_expected}", held.len());
    assert_eq!(copies.len(), faces_expected, "{label}: the workers hold {} faces, the single hierarchy {faces_expected}", copies.len());
    let mut report: Vec<String> = differing
        .iter()
        .map(|(c, (n, lo, hi))| format!("{c}: {n} values in [{}, {}] x [{}, {}], worst relative {:e}", lo[0], hi[0], lo[1], hi[1], worst[c]))
        .collect();
    report.sort();
    assert!(differing.is_empty(), "{label}: values differing from the single hierarchy by class:\n{}", report.join("\n"));
    assert!(halos > 0 || outcomes.len() == 1, "{label}: no cut halo was dumped; the freshness comparison is vacuous");
    assert!(shared > 0 || outcomes.len() == 1, "{label}: no face was held by two tiles");
    assert!(stencil > 0 || !with_stencil, "{label}: no parent face stencil was gathered");
}

/// the largest |div B| on the assembled root grid from the workers' faces
fn assembled_root_div(outcomes: &[Outcome]) -> f64 {
    let mut faces: [HashMap<(usize, usize), f64>; 2] = [HashMap::new(), HashMap::new()];
    for o in outcomes {
        for &(k, bits) in &o.records {
            if k.kind == 4 && k.level == 0 {
                faces[k.field as usize].insert((k.i, k.j), f64::from_bits(bits));
            }
        }
    }
    let mut worst = 0.0f64;
    for i in 0..N {
        for j in 0..N {
            let div = (faces[0][&(i + 1, j)] - faces[0][&(i, j)]) / DX + (faces[1][&(i, j + 1)] - faces[1][&(i, j)]) / DX;
            worst = worst.max(div.abs());
        }
    }
    worst
}

fn assert_matches_reference(test: &str, tag: &str, outcomes: &[Outcome], arm: &Arm, label: &str) -> Reference {
    assert_matches_reference_after(test, tag, outcomes, arm, label, &[], STEPS)
}

/// `assert_matches_reference` for a run of `expected_steps` resumed after the steps of
/// `before`: the single hierarchy runs those steps and then the resumed run's
fn assert_matches_reference_after(
    test: &str,
    tag: &str,
    outcomes: &[Outcome],
    arm: &Arm,
    label: &str,
    before: &[u64],
    expected_steps: u64,
) -> Reference {
    assert_all_ok(outcomes, label);
    let reports: Vec<&Report> = outcomes.iter().map(|o| o.report.as_ref().expect("report")).collect();
    for (w, r) in reports.iter().enumerate() {
        assert_eq!(r.dts, reports[0].dts, "{label}: worker {w} took other timesteps");
        assert_eq!(r.steps, expected_steps, "{label}: worker {w} took {} steps", r.steps);
        assert!(r.high_water.iter().all(|&h| h <= 2), "{label}: worker {w} grant high-water {:?}", r.high_water);
        assert!(r.div <= DIV_TOL, "{label}: worker {w} holds |div B| = {:e} on its own faces", r.div);
        eprintln!("{label}: worker {w} audits {}, |div B| {:e}, grant high-water {:?}", r.audits, r.div, r.high_water);
    }
    let dts: Vec<u64> = before.iter().chain(&reports[0].dts).copied().collect();
    let expected = reference(test, tag, arm, &dts);
    assert!(expected.report.div <= DIV_TOL, "{label}: the single hierarchy holds |div B| = {:e}", expected.report.div);
    assert_state_equal(outcomes, &expected, label);
    let div = assembled_root_div(outcomes);
    assert!(div <= DIV_TOL, "{label}: the assembled root holds |div B| = {div:e}");
    expected
}

fn ragged_17_13() -> Arm {
    Arm { cuts0: vec![17], cuts1: vec![13], owner: vec![0, 1, 2, 3], patch: [(12, 20), (12, 20)], ng: 2 }
}

fn patch_face_on_cut() -> Arm {
    Arm { patch: [(12, 17), (12, 20)], ng: 5, ..ragged_17_13() }
}

/// the patch's hi faces on both cuts: worker 0 holds the fine faces, workers 1 and 2 the
/// corrected cells and faces beside each face, worker 3 the cell at the corner beyond both
fn patch_corner_on_two_cuts() -> Arm {
    Arm { patch: [(12, 17), (8, 13)], ng: 5, ..ragged_17_13() }
}

fn capped(test: &str, tag: &str, arm: &Arm, step: u64, bits: u64) -> Vec<Outcome> {
    run_arm(
        test,
        tag,
        arm,
        &[("W_CAP_WORKER", u64::MAX.to_string()), ("W_CAP_STEP", step.to_string()), ("W_CAP_BITS", bits.to_string())],
    )
}

#[test]
fn a_magnetized_patch_inside_the_tiles_matches_the_single_hierarchy() {
    if in_worker_role() {
        worker_main();
    }
    let test = "a_magnetized_patch_inside_the_tiles_matches_the_single_hierarchy";
    let arm = ragged_17_13();
    let outcomes = run_arm(test, "inside", &arm, &[]);
    let expected = assert_matches_reference(test, "inside", &outcomes, &arm, "the magnetized patch across fine cuts");
    for (w, o) in outcomes.iter().enumerate() {
        let r = o.report.as_ref().unwrap();
        assert_eq!((r.contrib, r.emf, r.corrected), (0, 0, 0), "worker {w}: a plan carries transfers with no patch face on a cut");
        assert!(r.audits > 0, "worker {w} audited no shared face");
    }
    let _ = expected;
}

/// fine cuts across the patch on one axis at a time: the fine level's faces on the cut take
/// their halo from the fine exchange, the prolongation leaves those sides alone
#[test]
fn a_magnetized_patch_across_one_fine_cut_matches_the_single_hierarchy() {
    if in_worker_role() {
        worker_main();
    }
    let test = "a_magnetized_patch_across_one_fine_cut_matches_the_single_hierarchy";
    for (tag, cuts0, cuts1) in [("x17", vec![17], vec![]), ("y13", vec![], vec![13])] {
        let arm = Arm { cuts0, cuts1, owner: vec![0, 1], patch: [(12, 20), (12, 20)], ng: 2 };
        let outcomes = run_arm(test, tag, &arm, &[]);
        assert_matches_reference(test, tag, &outcomes, &arm, tag);
    }
}

/// a rejection in the second fine substep of step 3 on the quad layout: every level's cells,
/// cell field and faces return to the root attempt's entry
#[test]
fn a_magnetized_rejection_in_the_second_fine_substep_rolls_every_worker_back() {
    if in_worker_role() {
        worker_main();
    }
    let test = "a_magnetized_rejection_in_the_second_fine_substep_rolls_every_worker_back";
    assert_rejection_matches_capped_control(test, &ragged_17_13(), 1, "the quad layout");
}

/// worker 1 rejects the last root stage of step 3; every worker rolls the root step back with its
/// cell field and faces and replays it at the halved timestep
#[test]
fn a_magnetized_rejection_at_a_root_stage_rolls_every_worker_back() {
    if in_worker_role() {
        worker_main();
    }
    let test = "a_magnetized_rejection_at_a_root_stage_rolls_every_worker_back";
    let rejection = [
        ("W_REJECT_ROOT_WORKER", "1".to_string()),
        ("W_REJECT_ROOT_STEP", "3".to_string()),
        ("W_REJECT_ROOT_STAGE", "1".to_string()),
    ];
    assert_rejecting_run_matches_capped_control(test, &ragged_17_13(), &rejection, "the quad layout, root stage");
}

/// the patch's hi face on a cut: the fine faces beyond it prolong from parent faces the tile
/// gathers from its neighbors, the flux and EMF units cross to the tiles holding the
/// corrected cells and faces, and a rejection in the second fine substep after the first
/// substep's units were folded rolls every register back
#[test]
fn a_magnetized_patch_face_on_a_cut_refluxes_through_the_neighbor() {
    if in_worker_role() {
        worker_main();
    }
    let test = "a_magnetized_patch_face_on_a_cut_refluxes_through_the_neighbor";
    let arm = patch_face_on_cut();
    let outcomes = run_arm(test, "cut_face", &arm, &[]);
    let expected = assert_matches_reference(test, "cut_face", &outcomes, &arm, "the magnetized patch face on the cut");
    assert!(expected.report.oop > 0.0, "the out-of-plane flux mismatch on the face on the cut is zero; its transport is untested");
    let reports: Vec<&Report> = outcomes.iter().map(|o| o.report.as_ref().unwrap()).collect();
    assert!(reports.iter().any(|r| r.off_tile > 0), "no stencil face came from past a tile's lattice; the gather is untested");
    for (w, r) in reports.iter().enumerate() {
        assert_eq!(r.contrib, 2 * N_COMP as u64, "worker {w}: the flux plan holds {} contributions", r.contrib);
        assert_eq!(r.emf, 2, "worker {w}: the EMF plan holds {} contributions", r.emf);
    }
    assert_rejection_matches_capped_control(test, &arm, 1, "the magnetized patch face on the cut");
}

/// the patch's corner on two cuts: the tile beyond both holds copies of a face each
/// neighbor's received units correct, which reach it at the corrected-faces point before its
/// recovery; the corner edge's EMF mismatch is nonzero, and a rejection rolls every level back
#[test]
fn a_magnetized_patch_corner_on_two_cuts_corrects_every_shared_face() {
    if in_worker_role() {
        worker_main();
    }
    let test = "a_magnetized_patch_corner_on_two_cuts_corrects_every_shared_face";
    let arm = patch_corner_on_two_cuts();
    let outcomes = run_arm(test, "corner", &arm, &[]);
    let expected = assert_matches_reference(test, "corner", &outcomes, &arm, "the magnetized patch corner on two cuts");
    assert_ne!(expected.report.corner, 0.0, "the corner edge's EMF mismatch is zero; the gate is vacuous");
    for (w, o) in outcomes.iter().enumerate() {
        let r = o.report.as_ref().unwrap();
        // one slab of every register component per face on a cut, two faces
        assert_eq!(r.contrib, 2 * N_COMP as u64, "worker {w}: the flux plan holds {} contributions", r.contrib);
        assert_eq!(r.emf, 2, "worker {w}: the EMF plan holds {} contributions", r.emf);
        assert_eq!(r.corrected, 2, "worker {w}: the corrected-faces plan holds {} transfers", r.corrected);
    }
    // the patch lies on worker 0 alone
    assert_rejection_matches_capped_control(test, &arm, 0, "the magnetized patch corner on two cuts");
}

/// `worker`, holding a fine tile, rejects the last stage of the second fine substep of step 3:
/// every worker rolls back, and the run equals a clean run capped to the accepted timestep in
/// every record
fn assert_rejection_matches_capped_control(test: &str, arm: &Arm, worker: u32, label: &str) {
    let rejection = [("W_REJECT_FINE_WORKER", worker.to_string()), ("W_REJECT_FINE", "3,1,1".into())];
    assert_rejecting_run_matches_capped_control(test, arm, &rejection, label);
}

/// the run under `rejection`, which rejects step 3 once, rolls every worker back and replays
/// the step at the halved timestep: it equals the single hierarchy and a clean run whose
/// timestep at step 3 is capped to the accepted one
fn assert_rejecting_run_matches_capped_control(test: &str, arm: &Arm, rejection: &[(&str, String)], label: &str) {
    let rejecting = run_arm(test, "rejecting", arm, rejection);
    assert_all_ok(&rejecting, label);
    let reports: Vec<&Report> = rejecting.iter().map(|o| o.report.as_ref().unwrap()).collect();
    for (w, r) in reports.iter().enumerate() {
        assert_eq!(r.rejections, 1, "{label}: worker {w} reports {} rejections", r.rejections);
    }
    let dts = reports[0].dts.clone();
    assert_matches_reference(test, "rejecting", &rejecting, arm, &format!("{label}, rejecting"));
    let clean = capped(test, "clean", arm, 3, dts[3]);
    assert_matches_reference(test, "clean", &clean, arm, &format!("{label}, the clean control"));
    assert_eq!(clean[0].report.as_ref().unwrap().dts, dts, "{label}: the capped clean run took other timesteps than the rejecting one");
    for (w, (a, b)) in rejecting.iter().zip(&clean).enumerate() {
        assert!(a.records == b.records, "{label}: worker {w} holds other records than in the clean control");
    }
}

/// a magnetized refined run without the parent face stencil is refused by name on every
/// worker before any step, with nothing published
#[test]
fn a_magnetized_run_without_the_face_stencil_is_refused_by_name() {
    if in_worker_role() {
        worker_main();
    }
    let test = "a_magnetized_run_without_the_face_stencil_is_refused_by_name";
    let outcomes = run_arm(test, "no_stencil", &patch_face_on_cut(), &[("W_NO_STENCIL", "1".into())]);
    for (w, o) in outcomes.iter().enumerate() {
        assert_ne!(o.code, Some(0), "worker {w} exited zero");
        assert!(o.records.is_empty(), "worker {w} published its state");
    }
    assert!(
        outcomes.iter().any(|o| o.stderr.contains("without the parent face stencil")),
        "no worker named the refusal:\n{}",
        outcomes.iter().map(|o| o.stderr.clone()).collect::<Vec<_>>().join("\n---\n")
    );
}

/// the record map of a one-worker run, the reference a flagged cut run is held to; its stencil
/// values are left out, since each layout gathers a stencil of its own shape
fn uncut_reference(outcomes: &[Outcome], label: &str) -> Reference {
    assert_all_ok(outcomes, label);
    let o = &outcomes[0];
    let values = o.records.iter().filter(|(k, _)| k.kind != 5).cloned().collect();
    Reference { values, report: o.report.clone().unwrap() }
}

/// a fine cell against the fine cut, flagged troubled in `stage` of the second fine substep of
/// step 3 on worker 1, which holds it; worker 3 across the cut holds a clean fine tile. the
/// flag changes the one-worker run, the correction runs on both sides with no freeze, and the
/// four-worker run equals the one-worker run under the same flag in every value on both levels
fn assert_fine_trouble_matches_uncut(test: &str, ct: &str, stage: usize) {
    let label = format!("{ct}, fine stage {stage}");
    let base = vec![("W_CT", ct.to_string())];
    let mut flagged = base.clone();
    // the fine plan's grid: the patch [12, 20)^2 at ratio 2, the fine cuts at column 10 and row 2
    flagged.push(("W_FINE_TROUBLE", format!("3,1,{stage},9,7")));
    let uncut = Arm { cuts0: vec![], cuts1: vec![], owner: vec![0], patch: [(12, 20), (12, 20)], ng: 2 };
    let clean = run_arm(test, &format!("clean_{ct}_{stage}"), &uncut, &base);
    let clean_ref = uncut_reference(&clean, &format!("{label}, the clean one-worker run"));
    assert_eq!(clean_ref.report.fofc, (0, 0), "{label}: the clean run corrected by itself");
    let one = run_arm(test, &format!("uncut_{ct}_{stage}"), &uncut, &flagged);
    let expected = uncut_reference(&one, &format!("{label}, the flagged one-worker run"));
    let changed = expected.values.iter().filter(|(k, v)| clean_ref.values.get(k) != Some(v)).count();
    assert!(changed > 0, "{label}: the flag left the one-worker run unchanged; the gate is vacuous");
    assert!(expected.report.fofc.0 > 0, "{label}: the one-worker run never corrected");
    assert_eq!(expected.report.fofc.1, 0, "{label}: the one-worker run froze a cell");
    assert!(expected.report.div <= DIV_TOL, "{label}: one-worker |div B| {:e}", expected.report.div);

    let arm = ragged_17_13();
    let mut outcomes = run_arm(test, &format!("cut_{ct}_{stage}"), &arm, &flagged);
    assert_all_ok(&outcomes, &label);
    for (w, o) in outcomes.iter().enumerate() {
        let r = o.report.as_ref().unwrap();
        assert_eq!(r.dts, expected.report.dts, "{label}: worker {w} took other timesteps than the one-worker run");
        assert_eq!(r.fine_troubled, 1, "{label}: worker {w} saw {} fine troubled exchanges", r.fine_troubled);
        assert_eq!(r.fofc.1, 0, "{label}: worker {w} froze {} cells", r.fofc.1);
        assert!(r.div <= DIV_TOL, "{label}: worker {w} holds |div B| = {:e} on its own faces", r.div);
        assert!(r.high_water.iter().all(|&h| h <= 2), "{label}: worker {w} grant high-water {:?}", r.high_water);
    }
    let reports: Vec<&Report> = outcomes.iter().map(|o| o.report.as_ref().unwrap()).collect();
    assert!(reports[1].fofc.0 > 0, "{label}: the worker holding the flag never corrected");
    let neighbors: Vec<usize> = (0..4).filter(|&w| reports[w].fine_neighbor > 0).collect();
    assert_eq!(neighbors, vec![3], "{label}: the workers correcting a clean fine tile from the flag");
    // the correction census counts a tile's own troubled cells: the neighbor's fine tile is
    // clean, and it corrects its side of the shared faces from the exchanged flag alone
    assert_eq!(reports[3].fofc.0, 0, "{label}: the neighbor's fine tile flagged a cell of its own");
    assert_eq!(reports[3].fine_neighbor, 1, "{label}: the neighbor corrected a clean fine tile {} times", reports[3].fine_neighbor);
    for o in outcomes.iter_mut() {
        o.records.retain(|(k, _)| k.kind != 5);
    }
    let reference = Reference { values: expected.values.into_iter().filter(|(k, _)| k.kind != 5).collect(), report: expected.report };
    assert_state_equal_without_stencil(&outcomes, &reference, &label);
    let div = assembled_root_div(&outcomes);
    assert!(div <= DIV_TOL, "{label}: the assembled root holds |div B| = {div:e}");
}

/// the four-worker run under `flag` and the corner EMF `ct` equals the one-worker run under the
/// same flag in every value on both levels, with the same timesteps, no freeze and div B at
/// round-off on every tile and on the assembled root; the flag changes the one-worker run
fn assert_flag_matches_uncut(test: &str, ct: &str, flag: (&'static str, String)) {
    let label = format!("{ct}, {}={}", flag.0, flag.1);
    let base = vec![("W_CT", ct.to_string())];
    let mut flagged = base.clone();
    flagged.push(flag);
    let uncut = Arm { cuts0: vec![], cuts1: vec![], owner: vec![0], patch: [(12, 20), (12, 20)], ng: 2 };
    let tag = label.replace([',', ' ', '='], "_");
    let clean = run_arm(test, &format!("clean_{tag}"), &uncut, &base);
    let clean_ref = uncut_reference(&clean, &format!("{label}, the clean one-worker run"));
    let one = run_arm(test, &format!("uncut_{tag}"), &uncut, &flagged);
    let expected = uncut_reference(&one, &format!("{label}, the flagged one-worker run"));
    let changed = expected.values.iter().filter(|(k, v)| clean_ref.values.get(k) != Some(v)).count();
    assert!(changed > 0, "{label}: the flag left the one-worker run unchanged; the gate is vacuous");
    assert_eq!(expected.report.fofc.1, 0, "{label}: the one-worker run froze a cell");
    let mut outcomes = run_arm(test, &format!("cut_{tag}"), &ragged_17_13(), &flagged);
    assert_all_ok(&outcomes, &label);
    for (w, o) in outcomes.iter().enumerate() {
        let r = o.report.as_ref().unwrap();
        assert_eq!(r.dts, expected.report.dts, "{label}: worker {w} took other timesteps than the one-worker run");
        assert_eq!(r.fofc.1, 0, "{label}: worker {w} froze {} cells", r.fofc.1);
        assert!(r.div <= DIV_TOL, "{label}: worker {w} holds |div B| = {:e} on its own faces", r.div);
    }
    for o in outcomes.iter_mut() {
        o.records.retain(|(k, _)| k.kind != 5);
    }
    let reference = Reference { values: expected.values.into_iter().filter(|(k, _)| k.kind != 5).collect(), report: expected.report };
    assert_state_equal_without_stencil(&outcomes, &reference, &label);
    let div = assembled_root_div(&outcomes);
    assert!(div <= DIV_TOL, "{label}: the assembled root holds |div B| = {div:e}");
}

/// a fine cell in the corner of the fine cuts (fine column 10, fine row 2), flagged in the
/// predictor and in the corrector of the second fine substep of step 3
#[test]
fn fine_trouble_in_a_tile_corner_matches_the_one_worker_run() {
    if in_worker_role() {
        worker_main();
    }
    for ct in ["contact", "uct"] {
        for stage in [0, 1] {
            assert_flag_matches_uncut(
                "fine_trouble_in_a_tile_corner_matches_the_one_worker_run",
                ct,
                ("W_FINE_TROUBLE", format!("3,1,{stage},9,1")),
            );
        }
    }
}

/// a root cell against the root cut at 17, outside the refined patch, flagged in the predictor
/// and in the corrector of step 3
#[test]
fn root_trouble_against_a_cut_outside_the_patch_matches_the_one_worker_run() {
    if in_worker_role() {
        worker_main();
    }
    for ct in ["contact", "uct"] {
        for stage in [0, 1] {
            assert_flag_matches_uncut(
                "root_trouble_against_a_cut_outside_the_patch_matches_the_one_worker_run",
                ct,
                ("W_ROOT_TROUBLE", format!("3,{stage},16,4")),
            );
        }
    }
}

#[test]
fn contact_fine_trouble_in_the_predictor_is_corrected_on_both_sides_of_a_fine_cut() {
    if in_worker_role() {
        worker_main();
    }
    assert_fine_trouble_matches_uncut("contact_fine_trouble_in_the_predictor_is_corrected_on_both_sides_of_a_fine_cut", "contact", 0);
}

#[test]
fn contact_fine_trouble_in_the_corrector_is_corrected_on_both_sides_of_a_fine_cut() {
    if in_worker_role() {
        worker_main();
    }
    assert_fine_trouble_matches_uncut("contact_fine_trouble_in_the_corrector_is_corrected_on_both_sides_of_a_fine_cut", "contact", 1);
}

#[test]
fn uct_fine_trouble_in_the_predictor_is_corrected_on_both_sides_of_a_fine_cut() {
    if in_worker_role() {
        worker_main();
    }
    assert_fine_trouble_matches_uncut("uct_fine_trouble_in_the_predictor_is_corrected_on_both_sides_of_a_fine_cut", "uct", 0);
}

#[test]
fn uct_fine_trouble_in_the_corrector_is_corrected_on_both_sides_of_a_fine_cut() {
    if in_worker_role() {
        worker_main();
    }
    assert_fine_trouble_matches_uncut("uct_fine_trouble_in_the_corrector_is_corrected_on_both_sides_of_a_fine_cut", "uct", 1);
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
                let differing = p.iter().zip(q).filter(|(u, v)| u.to_bits() != v.to_bits()).count();
                assert_eq!(differing, 0, "{path}/{}: {differing} of {} values differ", x.name, p.len());
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

/// the patch face on a cut, checkpointed by four workers after four root steps: the file
/// equals the single hierarchy's after the same steps in every attribute and dataset of both
/// levels, the staggered faces included; resumed onto a partition that keeps the patch face
/// off every cut, and onto the original partition with a rejection in the second fine substep
/// of the second resumed step, four more steps equal the single hierarchy run uninterrupted
/// through eight in every value, the parent face stencil gathered afresh after the restart
#[test]
fn a_magnetized_refined_restart_onto_another_partition_continues_bitwise() {
    if in_worker_role() {
        worker_main();
    }
    let test = "a_magnetized_refined_restart_onto_another_partition_continues_bitwise";
    let dir = run_dir(test, "files");
    let ckpt = dir.join("refined.h5");
    let arm = patch_face_on_cut();
    let written = run_arm(
        test,
        "write",
        &arm,
        &[("W_STEPS", "4".into()), ("W_CKPT_AFTER", "4".into()), ("W_CKPT_PATH", ckpt.to_str().unwrap().into())],
    );
    assert_all_ok(&written, "the checkpointing run");
    let before = written[0].report.as_ref().unwrap().dts.clone();
    let mono_path = dir.join("mono.h5");
    let mono_dir = run_dir(test, "write_mono");
    let mono = collect(spawn(
        test,
        0,
        &arm,
        &mono_dir.join("rendezvous"),
        &mono_dir,
        &[
            ("W_MONO_DTS", before.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",")),
            ("W_MONO_CKPT_AFTER", "4".into()),
            ("W_MONO_CKPT_PATH", mono_path.to_str().unwrap().into()),
        ],
    ));
    assert_eq!(mono.code, Some(0), "the single hierarchy failed:\n{}", mono.stderr);
    assert_trees_equal(&Hdf5Backend.read(&ckpt).unwrap(), &Hdf5Backend.read(&mono_path).unwrap(), "");

    // the patch face at 17 lies inside the middle column of a three-by-two layout
    let off_cut = Arm { cuts0: vec![10, 22], cuts1: vec![16], owner: vec![0, 0, 1, 1, 2, 2], ..patch_face_on_cut() };
    let resume = ("W_RESTART_FROM", ckpt.to_str().unwrap().to_string());
    let steps = ("W_STEPS", "4".to_string());
    let resumed = run_arm(test, "off_cut", &off_cut, &[resume.clone(), steps.clone()]);
    assert_matches_reference_after(test, "off_cut", &resumed, &off_cut, "resumed with the face off the cut", &before, 4);
    let rejecting = run_arm(
        test,
        "rejecting",
        &arm,
        &[resume, steps, ("W_REJECT_FINE_WORKER", "1".into()), ("W_REJECT_FINE", "5,1,1".into())],
    );
    for (w, o) in rejecting.iter().enumerate() {
        let r = o.report.as_ref().unwrap_or_else(|| panic!("worker {w} failed:\n{}", o.stderr));
        assert_eq!(r.rejections, 1, "worker {w} reports {} rejections after the restart", r.rejections);
    }
    assert_matches_reference_after(test, "rejecting", &rejecting, &arm, "resumed with a late rejection", &before, 4);
}

/// a fine shared-face copy moved by one unit in the last place, with the per-point audits off:
/// after four steps, the checkpoint's audit of both levels names it; after initialization, the
/// prime audit of both levels names it. every worker ends before any file is published
#[test]
fn a_fine_shared_face_disagreement_ends_the_run_before_publication() {
    if in_worker_role() {
        worker_main();
    }
    let test = "a_fine_shared_face_disagreement_ends_the_run_before_publication";
    // worker 3 holds the fine tile whose low side of axis 0 is the fine cut at root 17
    for (tag, step) in [("after_steps", "4"), ("initial", "0")] {
        let dir = run_dir(test, &format!("{tag}_files"));
        let ckpt = dir.join("refined.h5");
        let outcomes = run_arm(
            test,
            tag,
            &ragged_17_13(),
            &[
                ("W_NO_STAGE_AUDIT", "1".into()),
                ("W_PERTURB_WORKER", "3".into()),
                ("W_PERTURB_FINE_FACE", step.into()),
                ("W_CKPT_AFTER", step.into()),
                ("W_CKPT_PATH", ckpt.to_str().unwrap().into()),
            ],
        );
        for (w, o) in outcomes.iter().enumerate() {
            assert_ne!(o.code, Some(0), "{tag}: worker {w} exited zero");
            assert!(o.records.is_empty(), "{tag}: worker {w} published its state");
        }
        assert!(
            outcomes.iter().any(|o| o.stderr.contains("shared-face mismatch")),
            "{tag}: no worker named the face:\n{}",
            outcomes.iter().map(|o| o.stderr.clone()).collect::<Vec<_>>().join("\n---\n")
        );
        assert!(!ckpt.exists(), "{tag}: a checkpoint was published");
        let temps = std::fs::read_dir(&dir).unwrap().filter(|e| e.as_ref().unwrap().file_name().to_string_lossy().contains(".tmp.")).count();
        assert_eq!(temps, 0, "{tag}: a temporary file was left");
    }
}
