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
        .enumerate()
        .map(|(w, ex)| {
            Fabric::new(
                link.clone(),
                WorkerId(w as u32),
                SessionId(7),
                workers as usize,
                2,
                &ex.lens(),
                ex.sends_per_peer_axis(),
            )
        })
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
