// =============================================================================
// decomp_plan_seam.rs
//
// the plan-driven halo exchange across workers: tiles split into owner groups,
// every cross-group transfer carried by the in-process fabric under grants and
// epochs, every within-group transfer moved by the local copy. the ghosts must
// match, cell for cell and face for face, what the fused whole-ownership
// exchange writes with direct copies. the staggered face fields ride the same
// plan, so a three-cell cell halo with the face field's own two-face halo is
// exchanged from one descriptor family.
//
// run: cargo test -p symbi --test decomp_plan_seam
// =============================================================================

use std::ops::Deref;
use symbi::sim::decomp::{
    LocalCopy, Partition, Schedule, Topology, exchange_grid, plan_fields, plan_schema, unflatten,
};
use symbi::sim::state::{Boundaries, BoundaryType, FieldStore, SimStateGeneric, Timestepping};
use symbi_fabric::{Fabric, Loopback, Progress, SessionId};
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_sim::atlas::{ExchangePlan, ExchangePoint, Placement, WorkerId};
use symbi_sim::plan_exchange::PlanExchange;
use symbi_xpu::{CpuSpace, HostMemory};

type Hydro = SimStateGeneric<Newtonian, 2, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Mhd = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

const N: usize = 64;

fn hydro_tile(p: &Partition<2>, flat: usize, ng: usize) -> Hydro {
    let ext = p.tile_extents(unflatten(flat, p.counts()));
    Hydro::new_at(
        Newtonian,
        IdealGas { gamma: 1.4 },
        Cartesian,
        [0, 0],
        [ext[0].1, ext[1].1],
        [ext[0].0 as f64, ext[1].0 as f64],
        [1.0, 1.0],
        ng,
        Boundaries::uniform(BoundaryType::Outflow),
        0.4,
        Timestepping::Rk2,
        0,
    )
    .unwrap()
}

fn mhd_tile(p: &Partition<2>, flat: usize, ng: usize) -> Mhd {
    let ext = p.tile_extents(unflatten(flat, p.counts()));
    Mhd::new_at(
        NewtonianMhd,
        IdealGas { gamma: 1.4 },
        Cartesian,
        [0, 0],
        [ext[0].1, ext[1].1],
        [ext[0].0 as f64, ext[1].0 as f64],
        [1.0, 1.0],
        ng,
        Boundaries::uniform(BoundaryType::Outflow),
        0.4,
        Timestepping::Rk2,
        0,
    )
    .unwrap()
}

/// a per-tile, per-field, per-cell value: distinct across tiles, so a ghost filled from the
/// wrong neighbor, the wrong field, or in the wrong order is visibly wrong, and distinct from
/// the neighbor's interior, so the exchange changes cells.
fn seed(flat: usize, tag: usize, c: [isize; 2]) -> f64 {
    1.0 + 0.01 * tag as f64 + 0.1 * flat as f64 + 0.001 * c[0] as f64 + 0.0007 * c[1] as f64
}

fn fill<const DOF: usize>(store: &FieldStore<2, DOF, HostMemory, f64>, flat: usize) {
    for (tag, field) in plan_fields(store).into_iter().enumerate() {
        let mut view = field.view_mut();
        for c in field.domain().iter() {
            *view.at_mut(c) = seed(flat, tag, c);
        }
    }
}

fn all_values<const DOF: usize>(stores: &[&FieldStore<2, DOF, HostMemory, f64>]) -> Vec<f64> {
    let mut out = Vec::new();
    for store in stores {
        for field in plan_fields(store) {
            let view = field.view();
            for c in field.domain().iter() {
                out.push(*view.at(c));
            }
        }
    }
    out
}

struct Arm<'a> {
    cuts: [Vec<usize>; 2],
    topology: Topology<2>,
    owner: &'a [u32],
    ng: usize,
}

/// run the fused exchange on one tile set and the plan-driven exchange across owner groups on
/// another, then compare every cell and face of every tile bitwise. `after_open` runs on the
/// split tiles once axis 0 has opened on every worker and returns a restore action applied
/// once axis 0 has finished.
fn assert_matches<const DOF: usize, S, F>(
    arm: &Arm<'_>,
    build: F,
    after_open: impl Fn(
        &[&FieldStore<2, DOF, HostMemory, f64>],
    ) -> Box<dyn Fn(&[&FieldStore<2, DOF, HostMemory, f64>])>,
) where
    S: Deref<Target = FieldStore<2, DOF, HostMemory, f64>>,
    F: Fn(&Partition<2>, usize, usize) -> S,
{
    let partition = Partition::explicit([N, N], arm.cuts.clone()).unwrap();
    let counts = partition.counts();
    let n = partition.n_tiles();
    let devices: Vec<i32> = vec![0; n];

    let fused: Vec<S> = (0..n).map(|f| build(&partition, f, arm.ng)).collect();
    let fused_stores: Vec<&FieldStore<2, DOF, HostMemory, f64>> =
        fused.iter().map(|s| &**s).collect();
    for (flat, s) in fused_stores.iter().enumerate() {
        fill(s, flat);
    }
    let before = all_values(&fused_stores);
    let schedule = Schedule::derive(counts, arm.ng, &arm.topology);
    exchange_grid(&fused_stores, &schedule, &devices, &LocalCopy);
    let want = all_values(&fused_stores);
    let touched = want
        .iter()
        .zip(&before)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert!(
        touched > 0,
        "the exchange wrote no cells; the seam is not under test"
    );

    let split: Vec<S> = (0..n).map(|f| build(&partition, f, arm.ng)).collect();
    let split_stores: Vec<&FieldStore<2, DOF, HostMemory, f64>> =
        split.iter().map(|s| &**s).collect();
    for (flat, s) in split_stores.iter().enumerate() {
        fill(s, flat);
    }
    let fields: Vec<Vec<_>> = split_stores.iter().map(|s| plan_fields(s)).collect();

    let schema = plan_schema(split_stores[0]);
    let plan = ExchangePlan::compile(&partition, &arm.topology, &schema, arm.ng).unwrap();
    let workers = arm.owner.iter().max().unwrap() + 1;
    assert!(
        workers > 1,
        "the owner map puts every tile in one worker; no transfer crosses"
    );
    let placement =
        Placement::new(workers, arm.owner.iter().map(|&w| WorkerId(w)).collect()).unwrap();
    let link = Loopback::new();
    let exchanges: Vec<PlanExchange<'_, 2>> = (0..workers)
        .map(|w| PlanExchange::new(&plan, &placement, WorkerId(w)))
        .collect();
    let mut fabrics: Vec<Fabric<Loopback>> = exchanges
        .iter()
        .map(|ex| ex.fabric(link.clone(), SessionId(7)))
        .collect();

    let mut restore: Option<Box<dyn Fn(&[&FieldStore<2, DOF, HostMemory, f64>])>> = None;
    for axis in 0..2 {
        let epoch = PlanExchange::<2>::epoch(ExchangePoint::Prime, 0, 0, axis);
        for (ex, fabric) in exchanges.iter().zip(fabrics.iter_mut()) {
            ex.open_axis(&fields, &devices, &LocalCopy, fabric, epoch)
                .unwrap();
        }
        if axis == 0 {
            restore = Some(after_open(&split_stores));
        }
        let mut rounds = 0;
        loop {
            let mut done = true;
            for fabric in fabrics.iter_mut() {
                if fabric.progress().unwrap() == Progress::Pending {
                    done = false;
                }
            }
            if done {
                break;
            }
            rounds += 1;
            assert!(rounds < 64, "axis {axis} made no progress");
        }
        for (ex, fabric) in exchanges.iter().zip(fabrics.iter_mut()) {
            ex.finish_axis(&fields, fabric, axis).unwrap();
        }
        // the hook's window is axis 0 alone: its sends were packed at open and left during
        // progress, so the restore lands before axis 1 packs from the same fields.
        if let Some(restore) = restore.take() {
            restore(&split_stores);
        }
    }
    for fabric in &fabrics {
        fabric.shutdown().unwrap();
    }
    assert_eq!(link.queued(), 0, "frames left on the link");
    let got = all_values(&split_stores);
    assert_eq!(got.len(), want.len());
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        assert!(
            a.to_bits() == b.to_bits(),
            "value {i}: plan exchange {a:e} != fused exchange {b:e}"
        );
    }
}

fn no_hook<const DOF: usize>(
    _: &[&FieldStore<2, DOF, HostMemory, f64>],
) -> Box<dyn Fn(&[&FieldStore<2, DOF, HostMemory, f64>])> {
    Box::new(|_| {})
}

#[test]
fn a_cut_between_workers_moves_as_frames() {
    let arm = Arm {
        cuts: [vec![32], vec![32]],
        topology: Topology::open(),
        owner: &[0, 0, 1, 1],
        ng: 2,
    };
    assert_matches(&arm, hydro_tile, no_hook);
}

#[test]
fn every_tile_its_own_worker() {
    let arm = Arm {
        cuts: [vec![32], vec![32]],
        topology: Topology::open(),
        owner: &[0, 1, 2, 3],
        ng: 2,
    };
    assert_matches(&arm, hydro_tile, no_hook);
    let arm = Arm {
        cuts: [vec![32], vec![32]],
        topology: Topology::open(),
        owner: &[0, 1, 2, 3],
        ng: 3,
    };
    assert_matches(&arm, hydro_tile, no_hook);
}

#[test]
fn ragged_tiles_cross_workers_on_both_axes() {
    let arm = Arm {
        cuts: [vec![19, 37], vec![27]],
        topology: Topology::open(),
        owner: &[0, 1, 0, 1, 1, 0],
        ng: 3,
    };
    assert_matches(&arm, hydro_tile, no_hook);
}

#[test]
fn a_periodic_wrap_crosses_workers() {
    let arm = Arm {
        cuts: [vec![32], vec![32]],
        topology: Topology::wrapping([true, false]),
        owner: &[0, 0, 1, 1],
        ng: 2,
    };
    assert_matches(&arm, hydro_tile, no_hook);
    let arm = Arm {
        cuts: [vec![19, 37], vec![27]],
        topology: Topology::wrapping([true, true]),
        owner: &[0, 1, 0, 1, 1, 0],
        ng: 2,
    };
    assert_matches(&arm, hydro_tile, no_hook);
}

/// three-cell cell halos with the face fields' own two-face transverse halos, on ragged tiles
/// across workers: the face lift shares the plan with the cells.
#[test]
fn mhd_three_cell_halo_with_two_face_halo_crosses_workers() {
    let arm = Arm {
        cuts: [vec![19, 37], vec![27]],
        topology: Topology::open(),
        owner: &[0, 1, 0, 1, 1, 0],
        ng: 3,
    };
    assert_matches(&arm, mhd_tile, no_hook);
    let arm = Arm {
        cuts: [vec![32], vec![32]],
        topology: Topology::open(),
        owner: &[0, 1, 2, 3],
        ng: 3,
    };
    assert_matches(&arm, mhd_tile, no_hook);
}

#[test]
fn mhd_two_cell_halo_crosses_workers() {
    let arm = Arm {
        cuts: [vec![19, 37], vec![27]],
        topology: Topology::wrapping([false, true]),
        owner: &[0, 1, 0, 1, 1, 0],
        ng: 2,
    };
    assert_matches(&arm, mhd_tile, no_hook);
}

/// a packed send is a copy: overwriting the source interior after the phase opened leaves the
/// destination halos equal to the fused exchange of the original values.
#[test]
fn source_free_after_pack() {
    let arm = Arm {
        cuts: [vec![32], vec![32]],
        topology: Topology::open(),
        owner: &[0, 0, 1, 1],
        ng: 2,
    };
    assert_matches(&arm, hydro_tile, |stores| {
        let store = stores[0];
        let field = plan_fields(store)[0];
        let interior = store.geom.interior.clone();
        let saved: Vec<f64> = interior.iter().map(|c| *field.view().at(c)).collect();
        {
            let mut view = field.view_mut();
            for c in interior.iter() {
                *view.at_mut(c) = 12345.0;
            }
        }
        Box::new(move |stores| {
            let field = plan_fields(stores[0])[0];
            let mut view = field.view_mut();
            for (c, v) in interior.iter().zip(&saved) {
                *view.at_mut(c) = *v;
            }
        })
    });
}

/// four magnetized tiles whose face fields come from one function of the global face
/// index, so the two copies of every shared interface face agree. worker 0 holds the low-x
/// tiles and worker 1 the high-x tiles: the cuts along x cross workers, the cuts along y,
/// the periodic seam among them, sit inside one worker.
fn audited_tiles() -> (Partition<2>, Vec<Mhd>, Topology<2>, &'static [u32]) {
    let partition = Partition::explicit([N, N], [vec![32], vec![32]]).unwrap();
    let tiles: Vec<Mhd> = (0..partition.n_tiles())
        .map(|f| mhd_tile(&partition, f, 2))
        .collect();
    for (flat, tile) in tiles.iter().enumerate() {
        let ext = partition.tile_extents(unflatten(flat, partition.counts()));
        let mhd = tile.fields.mhd.as_ref().unwrap();
        for d in 0..2 {
            let mut view = mhd.bface.b[d].view_mut();
            for c in mhd.bface.b[d].domain().iter() {
                let g = [ext[0].0 as isize + c[0], ext[1].0 as isize + c[1]];
                // periodic along y: the closing face of the last row is the opening face of
                // the first, so the value depends on the global index modulo the period
                let gy = g[1].rem_euclid(N as isize);
                *view.at_mut(c) = 0.5 + 0.013 * g[0] as f64 + 0.0071 * gy as f64 + d as f64;
            }
        }
    }
    (partition, tiles, Topology::wrapping([false, true]), &[0, 0, 1, 1])
}

/// run one audit over every worker of the in-process link; the first error of any worker.
fn run_audit(
    partition: &Partition<2>,
    tiles: &[Mhd],
    topology: &Topology<2>,
    owner: &[u32],
) -> Result<Vec<symbi_sim::plan_exchange::AuditCost>, symbi_sim::plan_exchange::AuditError> {
    use symbi_sim::atlas::AuditAt;
    use symbi_sim::plan_exchange::AuditCost;
    let stores: Vec<&FieldStore<2, 3, HostMemory, f64>> = tiles.iter().map(|s| &**s).collect();
    let fields: Vec<Vec<_>> = stores.iter().map(|s| plan_fields(s)).collect();
    let plan = ExchangePlan::compile(partition, topology, &plan_schema(stores[0]), 2).unwrap();
    let workers = owner.iter().max().unwrap() + 1;
    let placement = Placement::new(workers, owner.iter().map(|&w| WorkerId(w)).collect()).unwrap();
    let link = Loopback::new();
    let exchanges: Vec<PlanExchange<'_, 2>> = (0..workers)
        .map(|w| PlanExchange::new(&plan, &placement, WorkerId(w)))
        .collect();
    let mut fabrics: Vec<Fabric<Loopback>> = exchanges
        .iter()
        .map(|ex| ex.fabric(link.clone(), SessionId(7)))
        .collect();
    let mut costs = vec![AuditCost::default(); workers as usize];
    for axis in 0..2 {
        let epoch =
            PlanExchange::<2>::epoch(ExchangePoint::Audit(AuditAt::Prime), 0, 0, axis);
        for ((ex, fabric), cost) in exchanges.iter().zip(fabrics.iter_mut()).zip(&mut costs) {
            ex.audit_open_axis(&fields, fabric, epoch, cost)?;
        }
        let mut rounds = 0;
        while fabrics
            .iter_mut()
            .map(|f| f.progress().unwrap())
            .any(|p| p == Progress::Pending)
        {
            rounds += 1;
            assert!(rounds < 64, "audit axis {axis} made no progress");
        }
        for ((ex, fabric), cost) in exchanges.iter().zip(fabrics.iter_mut()).zip(&mut costs) {
            ex.audit_finish_axis(&fields, fabric, axis, cost)?;
        }
    }
    Ok(costs)
}

/// equal copies pass, every shared face is compared exactly once, only the cuts that cross
/// workers put bytes on the link, and the audit leaves every field bit for bit as it was.
#[test]
fn the_interface_audit_passes_equal_copies_and_writes_nothing() {
    let (partition, tiles, topology, owner) = audited_tiles();
    let stores: Vec<&FieldStore<2, 3, HostMemory, f64>> = tiles.iter().map(|s| &**s).collect();
    let before = all_values(&stores);
    let costs = run_audit(&partition, &tiles, &topology, owner).unwrap();
    let after = all_values(&stores);
    assert!(before.iter().zip(&after).all(|(a, b)| a.to_bits() == b.to_bits()));
    // two cuts along x of 32 faces each cross workers; along y each worker holds one interior
    // cut and one periodic seam of 32 faces
    let faces: u64 = costs.iter().map(|c| c.faces).sum();
    assert_eq!(faces, 2 * 32 + 4 * 32);
    assert_eq!(costs[0].wire_bytes, 2 * 32 * 8, "the lower worker sends its two closing faces");
    assert_eq!(costs[1].wire_bytes, 0);
}

/// one unit in the last place on one side of one shared face is reported with the field,
/// both tiles, the face, and both values: across workers, inside one worker, and on the
/// periodic seam.
#[test]
fn the_interface_audit_names_a_one_bit_disagreement() {
    use symbi_sim::plan_exchange::AuditError;
    // (tile perturbed, field axis, local face, expected lower tile, expected upper tile)
    let cases: [(usize, usize, [isize; 2], u32, u32); 3] = [
        (2, 0, [0, 5], 0, 2),   // upper side of an x cut that crosses workers
        (0, 1, [7, 32], 0, 1),  // lower side of a y cut inside worker 0
        (3, 1, [9, 32], 3, 2),  // lower side of the periodic y seam inside worker 1
    ];
    for (tile, d, face, below, above) in cases {
        let (partition, tiles, topology, owner) = audited_tiles();
        let field = &tiles[tile].fields.mhd.as_ref().unwrap().bface.b[d];
        let old = *field.view().at(face);
        *field.view_mut().at_mut(face) = f64::from_bits(old.to_bits() + 1);
        let err = run_audit(&partition, &tiles, &topology, owner).unwrap_err();
        let AuditError::Mismatch(m) = err else {
            panic!("expected a mismatch, got {err:?}");
        };
        assert_eq!((m.below.0, m.above.0, m.axis as usize), (below, above, d), "{m}");
        assert_eq!(m.field, format!("bface{d}"));
        assert_eq!(
            m.below_value.to_bits().abs_diff(m.above_value.to_bits()),
            1,
            "{m}"
        );
    }
}
