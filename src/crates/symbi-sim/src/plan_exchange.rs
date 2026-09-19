// =============================================================================
// plan_exchange.rs
//
// the halo exchange of one axis phase driven by a compiled exchange plan and a
// placement: transfers with both ends on this worker move by the local halo
// transport, sends pack their source region into the fabric's buffer for the
// transfer, receives are posted and their payloads scattered once landed. the
// fabric sees transfer ids and f64 values; the field, its region, and the
// iteration order live here.
//
// a phase runs in three calls so a driver can interleave workers: `open_axis`
// (post, grant, pack, local copies), `progress` until done, `finish_axis`
// (scatter and close).
//
// usage:
//  let exchange = PlanExchange::new(&plan, &placement, me);
//  let mut fabric = exchange.fabric(link, session);
//  exchange.open_axis(&fields, &devices, &LocalCopy, &mut fabric, epoch)?;
//  while fabric.progress()? == Progress::Pending {}
//  exchange.finish_axis(&fields, &mut fabric, axis)?;
// =============================================================================

use crate::atlas::{AuditAt, ExchangePlan, ExchangePoint, Placement, Span, TileId, domain_of};
use crate::decomp::HaloTransport;
use symbi_fabric::{Epoch, Fabric, FabricError, Link, PhaseSpec, TransferId, WorkerId};
use symbi_grid::Field;
use symbi_xpu::MemorySpace;

pub struct PlanExchange<'p, const D: usize> {
    plan: &'p ExchangePlan<D>,
    placement: &'p Placement,
    me: WorkerId,
}

impl<'p, const D: usize> PlanExchange<'p, D> {
    pub fn new(plan: &'p ExchangePlan<D>, placement: &'p Placement, me: WorkerId) -> Self {
        debug_assert_eq!(placement.owner.len(), plan.n_tiles());
        Self {
            plan,
            placement,
            me,
        }
    }

    pub fn epoch(point: ExchangePoint, step: u64, attempt: u16, axis: usize) -> Epoch {
        Epoch {
            step,
            attempt,
            point: point.to_wire(),
            axis: axis as u8,
        }
    }

    fn owns(&self, tile: TileId) -> bool {
        self.placement.owner_of(tile) == self.me
    }

    /// every transfer's value count, indexed by transfer id.
    pub fn lens(&self) -> Vec<usize> {
        let lens: Vec<usize> = self.plan.transfers().map(|t| t.len).collect();
        debug_assert!(
            self.plan
                .transfers()
                .enumerate()
                .all(|(i, t)| t.id == TransferId(i as u32)),
            "transfer ids run in plan order"
        );
        lens
    }

    /// the transfers this worker sends to `[peer][axis]` at the state exchange points.
    pub fn sends_per_peer_axis(&self) -> Vec<Vec<u32>> {
        self.sends_at(ExchangePoint::Prime)
    }

    /// the transfers this worker sends to `[peer][axis]` at a `Troubled` exchange point.
    pub fn trouble_sends_per_peer_axis(&self) -> Vec<Vec<u32>> {
        self.sends_at(ExchangePoint::Troubled(0))
    }

    /// the fabric endpoint for this plan: every transfer's buffer, the send counts a grant
    /// must carry at the state points, and the counts of the `Troubled` points.
    pub fn fabric<L: Link>(&self, link: L, session: symbi_fabric::SessionId) -> Fabric<L> {
        Fabric::new(
            link,
            self.me,
            session,
            self.placement.workers as usize,
            D,
            &self.lens(),
            self.sends_per_peer_axis(),
        )
        .with_point_sends(
            ExchangePoint::is_troubled_wire,
            self.trouble_sends_per_peer_axis(),
        )
        .with_point_sends(
            ExchangePoint::is_audit_wire,
            self.sends_at(ExchangePoint::Audit(AuditAt::Prime)),
        )
    }

    /// audit every shared interface face after the exchange `at`: the tile below each cut
    /// sends its copy of the face, the tile above compares it with its own bit for bit, and
    /// neither copy is written. both copies are the output of each tile's own curl of the
    /// edge EMFs, so they agree exactly when the two tiles started from the same face values
    /// and computed the same EMFs on the cut; the first differing face is returned as the
    /// error. a pair of tiles held by one worker is compared in place.
    pub fn audit<M: MemorySpace, L: Link>(
        &self,
        fields: &[Vec<&Field<f64, D, M>>],
        fabric: &mut Fabric<L>,
        at: AuditAt,
        step: u64,
        attempt: u16,
        deadline: std::time::Duration,
    ) -> Result<AuditCost, AuditError> {
        let clock = std::time::Instant::now();
        let mut cost = AuditCost::default();
        for axis in 0..D {
            let epoch = Self::epoch(ExchangePoint::Audit(at), step, attempt, axis);
            self.audit_open_axis(fields, fabric, epoch, &mut cost)?;
            fabric.wait(deadline)?;
            self.audit_finish_axis(fields, fabric, axis, &mut cost)?;
        }
        cost.elapsed = clock.elapsed();
        Ok(cost)
    }

    /// open the audit phase of `epoch.axis`: compare the interfaces whose two tiles this
    /// worker holds, pack the lower copies bound for another worker, post the receives.
    pub fn audit_open_axis<M: MemorySpace, L: Link>(
        &self,
        fields: &[Vec<&Field<f64, D, M>>],
        fabric: &mut Fabric<L>,
        epoch: Epoch,
        cost: &mut AuditCost,
    ) -> Result<(), AuditError> {
        let mut sends = Vec::new();
        let mut receives = Vec::new();
        for t in self.audits(epoch.axis as usize) {
            let f = t.field as usize;
            match (self.owns(t.src), self.owns(t.dst)) {
                (true, true) => {
                    let mut below = vec![0.0; t.len];
                    gather(fields[t.src.0 as usize][f], &t.src_region, &mut below);
                    compare(self.plan, t, fields[t.dst.0 as usize][f], &below)?;
                    cost.faces += t.len as u64;
                }
                (true, false) => {
                    let field = fields[t.src.0 as usize][f];
                    fabric.pack(t.id, |buf| gather(field, &t.src_region, buf))?;
                    sends.push((t.id, self.placement.owner_of(t.dst)));
                    cost.wire_bytes += 8 * t.len as u64;
                }
                (false, true) => receives.push((t.id, self.placement.owner_of(t.src))),
                (false, false) => {}
            }
        }
        Ok(fabric.open(PhaseSpec {
            epoch,
            sends: &sends,
            receives: &receives,
        })?)
    }

    /// compare every landed lower copy of `axis` with this worker's upper copy and close
    /// the phase.
    pub fn audit_finish_axis<M: MemorySpace, L: Link>(
        &self,
        fields: &[Vec<&Field<f64, D, M>>],
        fabric: &mut Fabric<L>,
        axis: usize,
        cost: &mut AuditCost,
    ) -> Result<(), AuditError> {
        for t in self.audits(axis) {
            if self.owns(t.dst) && !self.owns(t.src) {
                let field = fields[t.dst.0 as usize][t.field as usize];
                compare(self.plan, t, field, fabric.payload(t.id)?)?;
                fabric.mark_unpacked(t.id)?;
                cost.faces += t.len as u64;
            }
        }
        Ok(fabric.close()?)
    }

    /// whether the plan audits any interface: true exactly when it carries a face field
    /// and a cut along that field's axis. a plan-wide fact, so every worker agrees on it.
    pub fn has_audits(&self) -> bool {
        self.plan.transfers().any(|t| t.audit)
    }

    fn audits(&self, axis: usize) -> impl Iterator<Item = &crate::atlas::Transfer<D>> {
        self.plan.axes[axis].transfers.iter().filter(|t| t.audit)
    }

    /// the ghost regions of `tile` that a `Troubled` exchange writes, from every neighbor,
    /// local or remote: where a neighbor's troubled cell shows up on this tile.
    pub fn trouble_regions(&self, tile: TileId) -> Vec<symbi_algebra::Domain<D>> {
        self.plan
            .transfers()
            .filter(|t| t.dst == tile && self.plan.moves_at(t, ExchangePoint::Troubled(0)))
            .map(|t| domain_of(&t.dst_region))
            .collect()
    }

    /// the local index of global cell `cell` on `tile`, when the tile's interior holds it.
    pub fn local_cell(&self, tile: TileId, cell: [isize; D]) -> Option<[isize; D]> {
        let layout = &self.plan.tiles[tile.0 as usize];
        let local: [isize; D] = std::array::from_fn(|ax| cell[ax] - layout.offset[ax]);
        (0..D)
            .all(|ax| layout.interior[ax].lo <= local[ax] && local[ax] < layout.interior[ax].hi)
            .then_some(local)
    }

    fn sends_at(&self, point: ExchangePoint) -> Vec<Vec<u32>> {
        let mut counts = vec![vec![0u32; D]; self.placement.workers as usize];
        for t in self.plan.transfers().filter(|t| self.plan.moves_at(t, point)) {
            if self.owns(t.src) && !self.owns(t.dst) {
                counts[self.placement.owner_of(t.dst).0 as usize][t.axis as usize] += 1;
            }
        }
        counts
    }

    /// open the phase of `epoch.axis`: move the both-local transfers, pack the sends, post
    /// the receives. `fields[tile][field]` lists each held tile's fields in schema order;
    /// tiles this worker does not hold are never indexed.
    pub fn open_axis<M: MemorySpace, T: HaloTransport, L: Link>(
        &self,
        fields: &[Vec<&Field<f64, D, M>>],
        devices: &[i32],
        transport: &T,
        fabric: &mut Fabric<L>,
        epoch: Epoch,
    ) -> Result<(), FabricError> {
        let phase = &self.plan.axes[epoch.axis as usize];
        let point = ExchangePoint::from_wire(epoch.point).expect("an exchange epoch names a point");
        let mut sends = Vec::new();
        let mut receives = Vec::new();
        for t in phase.transfers.iter().filter(|t| self.plan.moves_at(t, point)) {
            let (src, dst) = (t.src.0 as usize, t.dst.0 as usize);
            let f = t.field as usize;
            match (self.owns(t.src), self.owns(t.dst)) {
                (true, true) => transport.copy_region(
                    fields[src][f],
                    &domain_of(&t.src_region),
                    fields[dst][f],
                    &domain_of(&t.dst_region),
                    devices[src],
                    devices[dst],
                ),
                (true, false) => {
                    let field = fields[src][f];
                    fabric.pack(t.id, |buf| gather(field, &t.src_region, buf))?;
                    sends.push((t.id, self.placement.owner_of(t.dst)));
                }
                (false, true) => receives.push((t.id, self.placement.owner_of(t.src))),
                (false, false) => {}
            }
        }
        fabric.open(PhaseSpec {
            epoch,
            sends: &sends,
            receives: &receives,
        })
    }

    /// scatter every landed receive of `axis` into its destination region and close the phase.
    pub fn finish_axis<M: MemorySpace, L: Link>(
        &self,
        fields: &[Vec<&Field<f64, D, M>>],
        fabric: &mut Fabric<L>,
        axis: usize,
    ) -> Result<(), FabricError> {
        self.finish_axis_at(fields, fabric, axis, ExchangePoint::Prime)
    }

    /// `finish_axis` for the phase opened at `point`, which selects the fields that moved.
    pub fn finish_axis_at<M: MemorySpace, L: Link>(
        &self,
        fields: &[Vec<&Field<f64, D, M>>],
        fabric: &mut Fabric<L>,
        axis: usize,
        point: ExchangePoint,
    ) -> Result<(), FabricError> {
        let moved = self.plan.axes[axis]
            .transfers
            .iter()
            .filter(|t| self.plan.moves_at(t, point));
        for t in moved {
            if self.owns(t.dst) && !self.owns(t.src) {
                let field = fields[t.dst.0 as usize][t.field as usize];
                scatter(field, &t.dst_region, fabric.payload(t.id)?);
                fabric.mark_unpacked(t.id)?;
            }
        }
        fabric.close()
    }
}

/// what one interface audit cost this worker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AuditCost {
    /// faces this worker compared
    pub faces: u64,
    /// payload bytes this worker sent
    pub wire_bytes: u64,
    pub elapsed: std::time::Duration,
}

/// two tiles hold different values for one shared interface face.
#[derive(Debug, Clone, PartialEq)]
pub struct InterfaceMismatch {
    pub below: TileId,
    pub above: TileId,
    pub axis: u8,
    pub field: String,
    /// the face's index in the upper tile's local index space
    pub face: Vec<isize>,
    pub below_value: f64,
    pub above_value: f64,
}

impl std::fmt::Display for InterfaceMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "shared-face mismatch on {} across axis {}: tile {:?} holds {:e} ({:#018x}) and tile \
             {:?} holds {:e} ({:#018x}) at face {:?} of the upper tile",
            self.field,
            self.axis,
            self.below,
            self.below_value,
            self.below_value.to_bits(),
            self.above,
            self.above_value,
            self.above_value.to_bits(),
            self.face
        )
    }
}

#[derive(Debug)]
pub enum AuditError {
    Fabric(FabricError),
    Mismatch(InterfaceMismatch),
}

impl From<FabricError> for AuditError {
    fn from(e: FabricError) -> Self {
        AuditError::Fabric(e)
    }
}

impl From<InterfaceMismatch> for AuditError {
    fn from(e: InterfaceMismatch) -> Self {
        AuditError::Mismatch(e)
    }
}

/// compare the lower tile's copy `theirs` of an audited face with the upper tile's own copy
/// in `field`, bit for bit, in region iteration order.
fn compare<const D: usize, M: MemorySpace>(
    plan: &ExchangePlan<D>,
    t: &crate::atlas::Transfer<D>,
    field: &Field<f64, D, M>,
    theirs: &[f64],
) -> Result<(), InterfaceMismatch> {
    let view = field.view();
    for (c, below) in domain_of(&t.dst_region).iter().zip(theirs) {
        let above = *view.at(c);
        if above.to_bits() != below.to_bits() {
            return Err(InterfaceMismatch {
                below: t.src,
                above: t.dst,
                axis: t.axis,
                field: plan.schema.entries[t.field as usize].name.clone(),
                face: c.to_vec(),
                below_value: *below,
                above_value: above,
            });
        }
    }
    Ok(())
}

/// read `region` of `field` into `buf` in region iteration order.
fn gather<const D: usize, M: MemorySpace>(
    field: &Field<f64, D, M>,
    region: &[Span; D],
    buf: &mut [f64],
) {
    let view = field.view();
    for (slot, c) in buf.iter_mut().zip(domain_of(region).iter()) {
        *slot = *view.at(c);
    }
}

/// write `buf` over `region` of `field` in region iteration order.
fn scatter<const D: usize, M: MemorySpace>(
    field: &Field<f64, D, M>,
    region: &[Span; D],
    buf: &[f64],
) {
    let mut view = field.view_mut();
    for (c, value) in domain_of(region).iter().zip(buf) {
        *view.at_mut(c) = *value;
    }
}
