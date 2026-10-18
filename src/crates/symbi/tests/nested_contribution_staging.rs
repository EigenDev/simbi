// =============================================================================
// nested_contribution_staging.rs
//
// the exactly-once rule for the flux-register units of a nested patch face on a
// cut of its parent level, at the protocol layer: two workers on the in-process
// fabric each hold one tile of a level-1 partition (each tile in its own level
// frame), level 2's low face lies on the cut, and the face's units cross on the
// NestedFluxContribution plan at a level-2 stage point. a clean exchange lands
// every unit in the receiving worker's inbox; a sender that lists none of its
// sends leaves the receiver at its transfer deadline with the units pending and
// its inbox untouched; a sender that lists each send twice fails the phase where
// it opens. the deadline is short here, so the dropped unit is caught in
// milliseconds where a launch waits its full transfer deadline.
//
// run: cargo test -p symbi --test nested_contribution_staging
// =============================================================================
use std::time::Duration;

use symbi::sim::decomp::{LocalCopy, Partition, Topology};
use symbi_algebra::{Domain, Space};
use symbi_fabric::{Fabric, FabricError, Loopback, Progress, SessionId};
use symbi_grid::Field;
use symbi_sim::atlas::{ExchangePlan, ExchangePoint, FinePoint, Placement, Span, TileId, WorkerId};
use symbi_sim::phase_table::PhaseTable;
use symbi_sim::plan_exchange::{PlanExchange, PlanRole, StagingFault};
use symbi_xpu::HostMemory;

const N_COMP: usize = 4;
// level 1 over 32 x 32 cells cut at x = 16; level 2's patch over level-1 cells [16, 24) x [4, 28):
// its low x face lies on the cut, tile 1 holds the fine faces, tile 0 the corrected cells
const PATCH: [(usize, usize); 2] = [(16, 24), (4, 28)];
// each tile's level frame, distinct from its partition extent
const FRAMES: [[isize; 2]; 2] = [[6, -2], [10, 3]];

fn domain(s: &[Span; 2]) -> Domain<2> {
    Domain::new(std::array::from_fn(|a| Space { name: symbi_sim::state::axis_name(a), lo: s[a].lo, hi: s[a].hi }))
}

/// the result of one level-2 stage point's exchange: the receiver's error, if any, and every value
/// its inbox holds afterward
fn exchange(fault: Option<StagingFault>) -> (Option<FabricError>, Vec<f64>) {
    let partition = Partition::explicit([32, 32], [vec![16], vec![]]).expect("a cut inside the grid");
    let plan = ExchangePlan::flux_contributions_shifted(&partition, &Topology::open(), PATCH, 3, N_COMP, &|t: TileId| FRAMES[t.0 as usize])
        .expect("the nested contribution plan");
    assert!(plan.transfers().count() == N_COMP, "the face on the cut carries one transfer per component");
    let placement = Placement::new(2, vec![WorkerId(0), WorkerId(1)]).expect("a placement");
    let table = PhaseTable::nested(2, 2, 3).expect("a three-level table");
    let role = PlanRole::NestedFluxContribution { level: 2 };
    let exchanges: Vec<PlanExchange<'_, 2>> =
        (0..2).map(|w| PlanExchange::new(&plan, &placement, WorkerId(w), table.clone()).with_role(role, 0)).collect();
    // the register slabs as the plan lays them out: the sender's faces filled, the inbox zero
    let held: Vec<Vec<Field<f64, 2, HostMemory>>> = (0..2)
        .map(|t| plan.fields[t].iter().map(|l| Field::zeros(&domain(&l.allocated)).expect("a slab")).collect())
        .collect();
    for t in plan.transfers() {
        let field = &held[t.src.0 as usize][t.field as usize];
        for c in domain(&t.src_region).iter() {
            field.view_mut().set(c, 1.0 + t.field as f64 + 0.25 * c[1] as f64);
        }
    }
    let fields: Vec<Vec<&Field<f64, 2, HostMemory>>> = held.iter().map(|tile| tile.iter().collect()).collect();
    let link = Loopback::new();
    let mut fabrics: Vec<Fabric<Loopback>> = exchanges.iter().map(|ex| ex.fabric(link.clone(), SessionId(9))).collect();
    let point = ExchangePoint::Nested { level: 2, path: 0, point: FinePoint::Stage(0) };
    // the first step walks every phase of the table every worker opens before the session's first
    // level-2 stage point, each axis empty: the plan moves nothing before it
    let target = table.byte_of(point).expect("the table holds the level-2 stage point") as usize;
    for phase in table.phases().iter().take(target).filter(|p| !p.skippable && !p.init_only) {
        for axis in 0..2 {
            let epoch = exchanges[0].epoch(phase.point, 0, 0, axis);
            for fabric in fabrics.iter_mut() {
                fabric.open(symbi_fabric::PhaseSpec { epoch, sends: &[], receives: &[] }).expect("an empty phase");
            }
            for fabric in fabrics.iter_mut() {
                fabric.wait(Duration::from_millis(200)).expect("an empty phase completes");
                fabric.close().expect("an empty phase closes");
            }
        }
    }
    let epoch = exchanges[0].epoch(point, 0, 0, 0);
    for (w, (ex, fabric)) in exchanges.iter().zip(fabrics.iter_mut()).enumerate() {
        let fault = if w == 1 { fault } else { None };
        let (sends, receives) = ex.stage_axis_faulted(&fields, &[0, 0], &LocalCopy, fabric, epoch, fault).expect("staging");
        if w == 0 {
            assert!(!receives.is_empty(), "the receiver posts no unit; the gate is vacuous");
        }
        if let Err(e) = fabric.open(symbi_fabric::PhaseSpec { epoch, sends: &sends, receives: &receives }) {
            return (Some(e), Vec::new());
        }
    }
    let mut failure = None;
    for _ in 0..64 {
        let mut done = true;
        for fabric in fabrics.iter_mut() {
            match fabric.progress() {
                Ok(Progress::Done) => {}
                Ok(Progress::Pending) => done = false,
                Err(e) => {
                    failure.get_or_insert(e);
                }
            }
        }
        if done || failure.is_some() {
            break;
        }
    }
    let failure = failure.or_else(|| fabrics[0].wait(Duration::from_millis(200)).err());
    if failure.is_none() {
        exchanges[0].unpack_axis_at(&fields, &mut fabrics[0], 0, point).expect("unpack");
    }
    let inbox: Vec<f64> = plan
        .transfers()
        .flat_map(|t| {
            let field = &held[t.dst.0 as usize][t.field as usize];
            domain(&t.dst_region).iter().map(|c| *field.view().at(c)).collect::<Vec<_>>()
        })
        .collect();
    (failure, inbox)
}

#[test]
fn a_clean_nested_exchange_lands_every_unit() {
    let (failure, inbox) = exchange(None);
    assert!(failure.is_none(), "the clean exchange failed: {failure:?}");
    assert!(inbox.iter().all(|&v| v >= 1.0), "a unit failed to land: {inbox:?}");
}

#[test]
fn a_dropped_nested_unit_leaves_the_receiver_at_its_deadline_with_the_inbox_untouched() {
    let (failure, inbox) = exchange(Some(StagingFault::Drop));
    assert!(
        matches!(failure, Some(FabricError::Deadline { pending, .. }) if pending > 0),
        "the dropped unit was not caught at the receiver's deadline: {failure:?}"
    );
    assert!(inbox.iter().all(|&v| v == 0.0), "a unit landed although its send was dropped");
}

#[test]
fn a_duplicated_nested_unit_fails_the_phase() {
    let (failure, inbox) = exchange(Some(StagingFault::Duplicate));
    // the receiving worker's grant counts the plan's sends; the phase lists each twice
    assert!(
        matches!(&failure, Some(FabricError::Protocol { detail, .. }) if detail.contains("this phase sends it")),
        "the duplicated unit was not refused by the grant's count: {failure:?}"
    );
    assert!(inbox.iter().all(|&v| v == 0.0), "a unit landed although the phase failed");
}
