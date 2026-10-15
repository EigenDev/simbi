// =============================================================================
// session_counts.rs
//
// the send counts a session's fabric expects at every phase, built compactly:
// each distinct [peer][axis] row held once and every phase naming its row. on
// a nested refined session of real plans (a root cut into six uneven tiles over
// three workers, three nested boxes crossing the cuts, and every plan role the
// table schedules counts for), every phase's compact counts equal the dense sum
// of the plans' counts for every worker, at depths three to five and with the
// slip points spliced in; the rows are shared along the substep paths. the
// tables of twelve to fourteen levels build, index every phase and code each
// once, and a fourteen-level session's counts share a handful of rows. the
// tables' digests tell apart depths, stage counts and the slip points.
//
// run: cargo test -p symbi-sim --test session_counts
// =============================================================================

use symbi_fabric::WorkerId;
use symbi_sim::atlas::{ExchangePlan, FieldSchema, Placement, TileId};
use symbi_sim::decomp::{FinePartition, Partition, Topology, flatten, unflatten};
use symbi_sim::phase_table::PhaseTable;
use symbi_sim::plan_exchange::{PlanExchange, PlanRole, session_phase_counts};

const N: usize = 32;
const RATIO: usize = 2;
const NG: usize = 2;
const WORKERS: u32 = 3;

fn schema() -> FieldSchema {
    FieldSchema::new().cell("den").cell("m1").cell("m2").cell("nrg").cell_band("flags", 1).face("b1", 0).face("b2", 1)
}

/// every level's partition and placement: the root, then each box refining its parent over
/// `boxes[l]`, its tiles owned by the worker owning the root tile beneath
struct Levels {
    partitions: Vec<Partition<2>>,
    placements: Vec<Placement>,
}

fn levels(depth: usize) -> Levels {
    let root = Partition::explicit([N, N], [vec![10, 20], vec![14]]).expect("root partition");
    let owner: Vec<WorkerId> = (0..root.n_tiles()).map(|t| WorkerId(t as u32 % WORKERS)).collect();
    let mut partitions = vec![root.clone()];
    let mut placements = vec![Placement::new(WORKERS, owner.clone()).expect("root placement")];
    // the first box spans root cells 6..26; each deeper one the middle half of its 40-cell parent
    let boxes: Vec<[(usize, usize); 2]> = std::iter::once([(6, 26), (6, 26)]).chain(std::iter::repeat([(10, 30), (10, 30)])).take(depth).collect();
    let mut roots: Vec<usize> = (0..root.n_tiles()).collect();
    for l in 1..depth {
        let parent = partitions[l - 1].clone();
        let fine = FinePartition::of(&parent, boxes[l - 1], RATIO).expect("level partition");
        let counts = fine.partition.counts();
        let parent_counts = parent.counts();
        let level_roots: Vec<usize> = (0..fine.partition.n_tiles())
            .map(|t| roots[flatten(fine.root_tile_of(unflatten(t, counts)), parent_counts)])
            .collect();
        let level_owner = level_roots.iter().map(|&r| owner[r]).collect();
        placements.push(Placement::with_idle_workers(WORKERS, level_owner).expect("level placement"));
        partitions.push(fine.partition);
        roots = level_roots;
    }
    Levels { partitions, placements }
}

/// add `plan` under `role` (the root state's own role when `None`) at the next id range
fn push<'p>(
    plans: &mut Vec<PlanExchange<'p, 2>>,
    offset: &mut u32,
    plan: &'p ExchangePlan<2>,
    placement: &'p Placement,
    me: WorkerId,
    table: &PhaseTable,
    role: Option<PlanRole>,
) {
    let px = PlanExchange::new(plan, placement, me, table.clone());
    let px = match role {
        Some(role) => px.with_role(role, *offset),
        None => px,
    };
    *offset += px.lens().len() as u32;
    plans.push(px);
}

/// the dense counts of a session, `[byte][peer][axis]`: every plan's counts at every phase summed
fn dense(plans: &[&PlanExchange<'_, 2>]) -> Vec<Vec<Vec<u32>>> {
    let mut out: Vec<Vec<Vec<u32>>> = Vec::new();
    for plan in plans {
        for (byte, per_peer) in plan.phase_counts().into_iter().enumerate() {
            if out.len() <= byte {
                out.push(vec![vec![0; 2]; WORKERS as usize]);
            }
            for (peer, per_axis) in per_peer.into_iter().enumerate() {
                for (axis, c) in per_axis.into_iter().enumerate() {
                    out[byte][peer][axis] += c;
                }
            }
        }
    }
    out
}

fn assert_compact_equals_dense(depth: usize, slip: bool) {
    let lv = levels(depth);
    let schema = schema();
    let open = Topology::open();
    let frame = |_: TileId| [0isize; 2];
    let root = &lv.partitions[0];
    let patch = [(6, 26), (6, 26)];
    let state: Vec<ExchangePlan<2>> = lv
        .partitions
        .iter()
        .enumerate()
        .map(|(l, p)| if l == 0 { ExchangePlan::compile(p, &open, &schema, NG) } else { ExchangePlan::compile_shifted(p, &open, &schema, NG + 1, &frame) })
        .collect::<Result<_, _>>()
        .expect("state plans");
    // each nested level's seed gathers its parent's conserved halos on the parent's partition
    let seeds: Vec<ExchangePlan<2>> = (2..depth)
        .map(|l| ExchangePlan::compile_shifted(&lv.partitions[l - 1], &open, &schema, NG + 1, &frame))
        .collect::<Result<_, _>>()
        .expect("seed plans");
    let flux = ExchangePlan::flux_contributions(root, &open, patch, NG, 4).expect("flux plan");
    let emf = ExchangePlan::emf_contributions(root, &open, patch, NG).expect("emf plan");
    let faces = ExchangePlan::interface_faces(root, &open, patch, NG).expect("faces plan");
    let corrected = ExchangePlan::corrected_faces(root, &open, patch, NG).expect("corrected plan");
    let table = PhaseTable::nested(2, RATIO as u8, depth as u8).expect("table");
    let table = if slip { table.with_slip().expect("slip table") } else { table };
    for me in 0..WORKERS {
        let me = WorkerId(me);
        let mut plans: Vec<PlanExchange<'_, 2>> = Vec::new();
        let mut offset = 0u32;
        let t = &table;
        push(&mut plans, &mut offset, &state[0], &lv.placements[0], me, t, None);
        push(&mut plans, &mut offset, &state[0], &lv.placements[0], me, t, Some(PlanRole::RootSeed));
        push(&mut plans, &mut offset, &state[1], &lv.placements[1], me, t, Some(PlanRole::Fine));
        push(&mut plans, &mut offset, &flux, &lv.placements[0], me, t, Some(PlanRole::FluxContribution));
        push(&mut plans, &mut offset, &emf, &lv.placements[0], me, t, Some(PlanRole::EmfContribution));
        push(&mut plans, &mut offset, &faces, &lv.placements[0], me, t, Some(PlanRole::InterfaceFace));
        push(&mut plans, &mut offset, &corrected, &lv.placements[0], me, t, Some(PlanRole::CorrectedFace));
        push(&mut plans, &mut offset, &state[0], &lv.placements[0], me, t, Some(PlanRole::TracerCoefficients));
        for l in 0..depth {
            push(&mut plans, &mut offset, &state[l], &lv.placements[l], me, t, Some(PlanRole::LevelCoefficients { level: l as u8 }));
        }
        for l in 2..depth {
            push(&mut plans, &mut offset, &state[l], &lv.placements[l], me, t, Some(PlanRole::Nested { level: l as u8 }));
            push(&mut plans, &mut offset, &seeds[l - 2], &lv.placements[l - 1], me, t, Some(PlanRole::NestedSeed { level: l as u8 }));
        }
        let refs: Vec<&PlanExchange<'_, 2>> = plans.iter().collect();
        let compact = session_phase_counts(&refs);
        let want = dense(&refs);
        assert_eq!(compact.len(), table.phases().len(), "one row index per phase");
        assert_eq!(want.len(), table.phases().len());
        for (byte, row) in want.iter().enumerate() {
            assert_eq!(compact.get(byte), Some(row), "depth {depth} slip {slip} worker {me:?}: phase {byte} {:?}", table.point_of(byte as u32));
        }
        assert!(want.iter().flatten().flatten().any(|&c| c > 0), "worker {me:?} sends nothing; the comparison is vacuous");
        if depth >= 4 {
            assert!(compact.classes() * 4 < table.phases().len(), "depth {depth}: {} rows for {} phases; the paths share none", compact.classes(), table.phases().len());
        }
    }
}

#[test]
fn compact_counts_equal_the_dense_sum_at_every_phase() {
    for depth in [3, 4, 5] {
        for slip in [false, true] {
            assert_compact_equals_dense(depth, slip);
        }
    }
}

#[test]
fn the_table_digest_tells_apart_depths_stages_and_the_slip_points() {
    let d = |stages: u8, depth: u8| PhaseTable::nested(stages, 2, depth).unwrap();
    let digests = [d(2, 4).digest(), d(2, 5).digest(), d(3, 5).digest(), d(3, 5).with_slip().unwrap().digest()];
    for (i, a) in digests.iter().enumerate() {
        for b in &digests[i + 1..] {
            assert_ne!(a, b, "two different tables share a digest");
        }
    }
    assert_eq!(d(3, 5).digest(), d(3, 5).digest());
}

/// twelve to fourteen levels: every phase reads back from its wire index and its index from it,
/// and no two phases share a code
#[test]
fn the_deepest_tables_index_and_code_every_phase_once() {
    for stages in [2u8, 3] {
        for depth in [12u8, 13, 14] {
            let t = PhaseTable::nested(stages, RATIO as u8, depth).expect("deep table");
            let n = t.phases().len();
            assert!(n > 40_000, "{stages} stages, {depth} levels: {n} phases");
            for (i, phase) in t.phases().iter().enumerate() {
                assert_eq!(t.byte_of(phase.point), Some(i as u32));
                assert_eq!(t.point_of(i as u32), Some(phase.point));
            }
            let mut codes: Vec<u64> = t.phases().iter().map(|p| p.point.kind_code()).collect();
            codes.sort_unstable();
            codes.dedup();
            assert_eq!(codes.len(), n, "{stages} stages, {depth} levels: two phases share a code");
        }
    }
}

/// a fourteen-level session of real plans holds a handful of count rows and one index per phase
#[test]
fn a_fourteen_level_session_shares_a_handful_of_rows() {
    let depth = 14;
    let lv = levels(depth);
    let schema = schema();
    let open = Topology::open();
    let frame = |_: TileId| [0isize; 2];
    let state: Vec<ExchangePlan<2>> = lv
        .partitions
        .iter()
        .enumerate()
        .map(|(l, p)| if l == 0 { ExchangePlan::compile(p, &open, &schema, NG) } else { ExchangePlan::compile_shifted(p, &open, &schema, NG + 1, &frame) })
        .collect::<Result<_, _>>()
        .expect("state plans");
    let table = PhaseTable::nested(3, RATIO as u8, depth as u8).expect("table");
    let mut plans: Vec<PlanExchange<'_, 2>> = Vec::new();
    let mut offset = 0u32;
    push(&mut plans, &mut offset, &state[0], &lv.placements[0], WorkerId(0), &table, None);
    push(&mut plans, &mut offset, &state[1], &lv.placements[1], WorkerId(0), &table, Some(PlanRole::Fine));
    for l in 2..depth {
        push(&mut plans, &mut offset, &state[l], &lv.placements[l], WorkerId(0), &table, Some(PlanRole::Nested { level: l as u8 }));
    }
    let refs: Vec<&PlanExchange<'_, 2>> = plans.iter().collect();
    let compact = session_phase_counts(&refs);
    assert_eq!(compact.len(), table.phases().len());
    assert!(compact.classes() <= 16, "{} rows for {} phases", compact.classes(), table.phases().len());
    assert!((0..compact.len()).any(|b| compact.get(b).is_some_and(|row| row.iter().flatten().any(|&c| c > 0))), "the session sends nothing");
}
