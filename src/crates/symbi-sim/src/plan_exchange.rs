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
//  let exchange = PlanExchange::new(&plan, &placement, me, PhaseTable::flat(stages)?);
//  let mut fabric = exchange.fabric(link, session);
//  exchange.open_axis(&fields, &devices, &LocalCopy, &mut fabric, epoch)?;
//  while fabric.progress()? == Progress::Pending {}
//  exchange.finish_axis(&fields, &mut fabric, axis)?;
// =============================================================================

use crate::atlas::{AuditAt, ExchangePlan, ExchangePoint, FinePoint, Placement, Span, TileId, domain_of};
use crate::decomp::HaloTransport;
use crate::phase_table::PhaseTable;

/// which level's phases a plan moves at. the root plan carries the root state, bands and
/// audits at the root points; the seed plan carries the root conserved halos at the prime
/// alone (the fine level's seeding prolongs from them); the fine plan carries the fine
/// state, bands and audits at the fine points and the fine state at the rollback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanRole {
    RootState,
    /// the continuous-tracer coefficients: every transfer moves at the tracer-coefficient point
    TracerCoefficients,
    RootSeed,
    Fine,
    /// the flux-register contributions: every transfer moves at every fine stage point
    FluxContribution,
    /// the EMF-register contributions: every transfer moves at every fine post-tail point
    EmfContribution,
    /// the neighbor's copies of the staggered faces on a cut: every transfer moves at the
    /// last substep's post-tail point, after the restriction on the tile holding the fine faces
    InterfaceFace,
    /// the copies across a tile's own cut of the faces its received units corrected: every
    /// transfer moves at the corrected-faces point
    CorrectedFace,
    /// a refined tile's parent face stencil gathered from the faces' owners: every transfer
    /// moves at the last root stage point, after the root stages and before the fine subcycle
    FaceStencil,
    /// the state of nested level `level >= 2`: its state, bands and audits at its own points,
    /// its state after the primed restriction, after the restriction of the level below it,
    /// and at the rollback
    Nested { level: u8 },
    /// the conserved halos of level `level - 1` at the seed of nested level `level`
    NestedSeed { level: u8 },
    /// nested level `level`'s parent face stencil gathered from the faces' owners on the parent's
    /// partition: every transfer moves at the parent's last stage point of each of its substeps
    NestedFaceStencil { level: u8 },
}

/// a fault injected into one plan's staging at one point, for the gates of the
/// exactly-once rule
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StagingFault {
    /// list every send of this worker twice
    Duplicate,
    /// pack and list none of this worker's sends
    Drop,
}
use symbi_fabric::{Epoch, Fabric, FabricError, Link, PhaseSpec, TransferId, WorkerId};
use symbi_grid::Field;
use symbi_xpu::MemorySpace;

pub struct PlanExchange<'p, const D: usize> {
    plan: &'p ExchangePlan<D>,
    placement: &'p Placement,
    me: WorkerId,
    table: PhaseTable,
    role: PlanRole,
    /// the fabric transfer id of this plan's first transfer: several plans share one fabric
    /// with disjoint id ranges
    id_offset: u32,
}

impl<'p, const D: usize> PlanExchange<'p, D> {
    /// `table` is the phase table every worker of the session walks; its digest is part of
    /// the handshake identity
    pub fn new(plan: &'p ExchangePlan<D>, placement: &'p Placement, me: WorkerId, table: PhaseTable) -> Self {
        debug_assert_eq!(placement.owner.len(), plan.n_tiles());
        Self {
            plan,
            placement,
            me,
            table,
            role: PlanRole::RootState,
            id_offset: 0,
        }
    }

    /// this plan's role in the session and the fabric id its transfers start at
    pub fn with_role(mut self, role: PlanRole, id_offset: u32) -> Self {
        self.role = role;
        self.id_offset = id_offset;
        self
    }

    pub fn role(&self) -> PlanRole {
        self.role
    }

    pub fn id_offset(&self) -> u32 {
        self.id_offset
    }

    fn fabric_id(&self, t: &crate::atlas::Transfer<D>) -> TransferId {
        TransferId(t.id.0 + self.id_offset)
    }

    /// whether `t` moves at `point` under this plan's role
    pub fn moves(&self, t: &crate::atlas::Transfer<D>, point: ExchangePoint) -> bool {
        match self.role {
            PlanRole::RootState => self.plan.moves_at(t, point),
            PlanRole::TracerCoefficients => {
                point == ExchangePoint::TracerCoefficients && self.plan.moves_at(t, ExchangePoint::Stage(0))
            }
            PlanRole::RootSeed => point == ExchangePoint::Prime && self.plan.moves_at(t, point),
            PlanRole::Fine => match point {
                ExchangePoint::Fine { point: fine, .. } => {
                    let as_root = match fine {
                        FinePoint::Prolong | FinePoint::Stage(_) | FinePoint::PostTail => ExchangePoint::Stage(0),
                        FinePoint::Troubled(_) => ExchangePoint::Troubled(0),
                        FinePoint::Audit(_) | FinePoint::AuditTail => ExchangePoint::Audit(AuditAt::Prime),
                    };
                    self.plan.moves_at(t, as_root)
                }
                ExchangePoint::Rollback => self.plan.moves_at(t, ExchangePoint::Rollback),
                // the primed fine state, and the fine state after a nested level restricted into it
                ExchangePoint::Primed | ExchangePoint::NestedRestrict { level: 1, .. } => {
                    self.plan.moves_at(t, ExchangePoint::Stage(0))
                }
                // the fine level's shared faces are audited with the root's after the prime and
                // before a checkpoint
                ExchangePoint::Audit(AuditAt::Prime) | ExchangePoint::Audit(AuditAt::Checkpoint) => {
                    self.plan.moves_at(t, point)
                }
                _ => false,
            },
            PlanRole::Nested { level } => match point {
                ExchangePoint::Nested { level: l, point: fine, .. } if l == level => {
                    let as_root = match fine {
                        FinePoint::Prolong | FinePoint::Stage(_) | FinePoint::PostTail => ExchangePoint::Stage(0),
                        FinePoint::Troubled(_) => ExchangePoint::Troubled(0),
                        FinePoint::Audit(_) | FinePoint::AuditTail => ExchangePoint::Audit(AuditAt::Prime),
                    };
                    self.plan.moves_at(t, as_root)
                }
                ExchangePoint::NestedRestrict { level: l, .. } if l == level => self.plan.moves_at(t, ExchangePoint::Stage(0)),
                ExchangePoint::Primed => self.plan.moves_at(t, ExchangePoint::Stage(0)),
                ExchangePoint::Rollback => self.plan.moves_at(t, ExchangePoint::Rollback),
                ExchangePoint::Audit(AuditAt::Prime) | ExchangePoint::Audit(AuditAt::Checkpoint) => {
                    self.plan.moves_at(t, point)
                }
                _ => false,
            },
            PlanRole::NestedSeed { level } => {
                point == ExchangePoint::Seed { level } && self.plan.moves_at(t, ExchangePoint::Prime)
            }
            PlanRole::NestedFaceStencil { level } => {
                let last = self.table.fine_stages().saturating_sub(1);
                match point {
                    ExchangePoint::Fine { point: FinePoint::Stage(k), .. } => level == 2 && k == last,
                    ExchangePoint::Nested { level: l, point: FinePoint::Stage(k), .. } => l + 1 == level && k == last,
                    _ => false,
                }
            }
            PlanRole::FluxContribution => matches!(point, ExchangePoint::Fine { point: FinePoint::Stage(_), .. }),
            PlanRole::EmfContribution => matches!(point, ExchangePoint::Fine { point: FinePoint::PostTail, .. }),
            PlanRole::InterfaceFace => matches!(
                point,
                ExchangePoint::Fine { substep, point: FinePoint::PostTail } if substep + 1 == self.table.substeps()
            ),
            PlanRole::CorrectedFace => point == ExchangePoint::CorrectedFaces,
            PlanRole::FaceStencil => point == ExchangePoint::Stage(self.table.stages() - 1),
        }
    }

    /// the point at which this plan's troubled bands move, for the trouble regions
    fn troubled_point(&self) -> ExchangePoint {
        match self.role {
            PlanRole::Fine => ExchangePoint::Fine { substep: 0, point: FinePoint::Troubled(0) },
            PlanRole::Nested { level } => ExchangePoint::Nested { level, path: 0, point: FinePoint::Troubled(0) },
            _ => ExchangePoint::Troubled(0),
        }
    }

    pub fn plan(&self) -> &'p ExchangePlan<D> {
        self.plan
    }

    pub fn placement(&self) -> &'p Placement {
        self.placement
    }

    pub fn table(&self) -> &PhaseTable {
        &self.table
    }

    /// the epoch of `point`'s phase on `axis`; a point the table lacks is a driver error
    pub fn epoch(&self, point: ExchangePoint, step: u64, attempt: u16, axis: usize) -> Epoch {
        let byte = self
            .table
            .byte_of(point)
            .unwrap_or_else(|| panic!("exchange point {point:?} outside the session's phase table"));
        Epoch {
            step,
            attempt,
            point: byte,
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

    /// the send counts of every phase of the table, `[byte][peer][axis]`: what this plan
    /// moves at each point under its role
    pub fn phase_counts(&self) -> Vec<Vec<Vec<u32>>> {
        self.table
            .phases()
            .iter()
            .map(|phase| self.sends_at(phase.point))
            .collect()
    }

    /// the fabric endpoint for this plan alone: every transfer's buffer, the send counts of
    /// every phase of the table, and the table's sequence rules.
    pub fn fabric<L: Link>(&self, link: L, session: symbi_fabric::SessionId) -> Fabric<L> {
        fabric_for(&[self], link, session)
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
        self.audit_at(fields, fabric, ExchangePoint::Audit(at), step, attempt, deadline)
    }

    /// `audit` at any audit point of the table, a fine level's included
    pub fn audit_at<M: MemorySpace, L: Link>(
        &self,
        fields: &[Vec<&Field<f64, D, M>>],
        fabric: &mut Fabric<L>,
        point: ExchangePoint,
        step: u64,
        attempt: u16,
        deadline: std::time::Duration,
    ) -> Result<AuditCost, AuditError> {
        let clock = std::time::Instant::now();
        let mut cost = AuditCost::default();
        for axis in 0..D {
            let epoch = self.epoch(point, step, attempt, axis);
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
        let (sends, receives) = self.audit_stage_axis(fields, fabric, epoch.axis as usize, cost)?;
        Ok(fabric.open(PhaseSpec {
            epoch,
            sends: &sends,
            receives: &receives,
        })?)
    }

    /// the staging half of `audit_open_axis`: compare the interfaces whose two tiles this
    /// worker holds, pack the lower copies bound for another worker, and name the sends and
    /// receives the phase opens with
    #[allow(clippy::type_complexity)]
    pub fn audit_stage_axis<M: MemorySpace, L: Link>(
        &self,
        fields: &[Vec<&Field<f64, D, M>>],
        fabric: &mut Fabric<L>,
        axis: usize,
        cost: &mut AuditCost,
    ) -> Result<(Vec<(TransferId, WorkerId)>, Vec<(TransferId, WorkerId)>), AuditError> {
        let mut sends = Vec::new();
        let mut receives = Vec::new();
        for t in self.audits(axis) {
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
                    fabric.pack(self.fabric_id(t), |buf| gather(field, &t.src_region, buf))?;
                    sends.push((self.fabric_id(t), self.placement.owner_of(t.dst)));
                    cost.wire_bytes += 8 * t.len as u64;
                }
                (false, true) => receives.push((self.fabric_id(t), self.placement.owner_of(t.src))),
                (false, false) => {}
            }
        }
        Ok((sends, receives))
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
        self.audit_land_axis(fields, fabric, axis, cost)?;
        Ok(fabric.close()?)
    }

    /// compare every landed lower copy of `axis` with this worker's upper copy, the phase
    /// left open for the other plans sharing it
    pub fn audit_land_axis<M: MemorySpace, L: Link>(
        &self,
        fields: &[Vec<&Field<f64, D, M>>],
        fabric: &mut Fabric<L>,
        axis: usize,
        cost: &mut AuditCost,
    ) -> Result<(), AuditError> {
        for t in self.audits(axis) {
            if self.owns(t.dst) && !self.owns(t.src) {
                let field = fields[t.dst.0 as usize][t.field as usize];
                compare(self.plan, t, field, fabric.payload(self.fabric_id(t))?)?;
                fabric.mark_unpacked(self.fabric_id(t))?;
                cost.faces += t.len as u64;
            }
        }
        Ok(())
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
        let at = self.troubled_point();
        self.plan
            .transfers()
            .filter(|t| t.dst == tile && self.moves(t, at))
            .map(|t| domain_of(&t.dst_region))
            .collect()
    }

    /// the local index of global cell `cell` on `tile`, when the tile's interior holds it.
    pub fn local_cell(&self, tile: TileId, cell: [isize; D]) -> Option<[isize; D]> {
        let layout = &self.plan.tiles[tile.0 as usize];
        let local: [isize; D] = std::array::from_fn(|ax| layout.interior[ax].lo + (cell[ax] - layout.offset[ax]));
        (0..D)
            .all(|ax| layout.interior[ax].lo <= local[ax] && local[ax] < layout.interior[ax].hi)
            .then_some(local)
    }

    fn sends_at(&self, point: ExchangePoint) -> Vec<Vec<u32>> {
        let mut counts = vec![vec![0u32; D]; self.placement.workers as usize];
        for t in self.plan.transfers().filter(|t| self.moves(t, point)) {
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
        let (sends, receives) = self.stage_axis(fields, devices, transport, fabric, epoch)?;
        fabric.open(PhaseSpec {
            epoch,
            sends: &sends,
            receives: &receives,
        })
    }

    /// the first half of `open_axis`: the both-local transfers moved, the sends packed, and
    /// the sends and receives of the phase returned for the caller to open, so several
    /// plans can share one phase
    pub fn stage_axis<M: MemorySpace, T: HaloTransport, L: Link>(
        &self,
        fields: &[Vec<&Field<f64, D, M>>],
        devices: &[i32],
        transport: &T,
        fabric: &mut Fabric<L>,
        epoch: Epoch,
    ) -> Result<(Vec<(TransferId, WorkerId)>, Vec<(TransferId, WorkerId)>), FabricError> {
        self.stage_axis_faulted(fields, devices, transport, fabric, epoch, None)
    }

    /// `stage_axis` with a fault injected into this worker's sends
    pub fn stage_axis_faulted<M: MemorySpace, T: HaloTransport, L: Link>(
        &self,
        fields: &[Vec<&Field<f64, D, M>>],
        devices: &[i32],
        transport: &T,
        fabric: &mut Fabric<L>,
        epoch: Epoch,
        fault: Option<StagingFault>,
    ) -> Result<(Vec<(TransferId, WorkerId)>, Vec<(TransferId, WorkerId)>), FabricError> {
        self.stage_axis_into(fields, fields, devices, transport, fabric, epoch, fault)
    }

    /// `stage_axis_faulted` for a plan reading `fields` and writing `dst_fields`: a transfer
    /// held within this worker copies from the one list into the other
    #[allow(clippy::too_many_arguments)]
    pub fn stage_axis_into<M: MemorySpace, T: HaloTransport, L: Link>(
        &self,
        fields: &[Vec<&Field<f64, D, M>>],
        dst_fields: &[Vec<&Field<f64, D, M>>],
        devices: &[i32],
        transport: &T,
        fabric: &mut Fabric<L>,
        epoch: Epoch,
        fault: Option<StagingFault>,
    ) -> Result<(Vec<(TransferId, WorkerId)>, Vec<(TransferId, WorkerId)>), FabricError> {
        let phase = &self.plan.axes[epoch.axis as usize];
        let point = self
            .table
            .point_of(epoch.point)
            .expect("an exchange epoch names a point of the session's phase table");
        let mut sends = Vec::new();
        let mut receives = Vec::new();
        for t in phase.transfers.iter().filter(|t| self.moves(t, point)) {
            let (src, dst) = (t.src.0 as usize, t.dst.0 as usize);
            let f = t.field as usize;
            match (self.owns(t.src), self.owns(t.dst)) {
                (true, true) => transport.copy_region(
                    fields[src][f],
                    &domain_of(&t.src_region),
                    dst_fields[dst][f],
                    &domain_of(&t.dst_region),
                    devices[src],
                    devices[dst],
                ),
                (true, false) => {
                    if fault == Some(StagingFault::Drop) {
                        continue;
                    }
                    let field = fields[src][f];
                    fabric.pack(self.fabric_id(t), |buf| gather(field, &t.src_region, buf))?;
                    sends.push((self.fabric_id(t), self.placement.owner_of(t.dst)));
                    if fault == Some(StagingFault::Duplicate) {
                        sends.push((self.fabric_id(t), self.placement.owner_of(t.dst)));
                    }
                }
                (false, true) => receives.push((self.fabric_id(t), self.placement.owner_of(t.src))),
                (false, false) => {}
            }
        }
        Ok((sends, receives))
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
        self.unpack_axis_at(fields, fabric, axis, point)?;
        fabric.close()
    }

    /// the first half of `finish_axis_at`: every landed receive of `axis` scattered into its
    /// destination region, the phase left open for the caller to close
    pub fn unpack_axis_at<M: MemorySpace, L: Link>(
        &self,
        fields: &[Vec<&Field<f64, D, M>>],
        fabric: &mut Fabric<L>,
        axis: usize,
        point: ExchangePoint,
    ) -> Result<(), FabricError> {
        let moved = self.plan.axes[axis]
            .transfers
            .iter()
            .filter(|t| self.moves(t, point));
        for t in moved {
            if self.owns(t.dst) && !self.owns(t.src) {
                let field = fields[t.dst.0 as usize][t.field as usize];
                scatter(field, &t.dst_region, fabric.payload(self.fabric_id(t))?);
                fabric.mark_unpacked(self.fabric_id(t))?;
            }
        }
        Ok(())
    }
}

/// one fabric endpoint for several plans of one session: the plans' transfers occupy
/// disjoint id ranges in offset order (each plan's `id_offset` is the sum of the lengths of
/// the plans before it), the send counts of every phase are summed over the plans, and the
/// sequence rules come from the shared phase table. every plan names the same worker, the
/// same worker count and the same table.
pub fn fabric_for<const D: usize, L: Link>(
    plans: &[&PlanExchange<'_, D>],
    link: L,
    session: symbi_fabric::SessionId,
) -> Fabric<L> {
    let first = plans.first().expect("at least one plan");
    let mut lens = Vec::new();
    let mut counts: Vec<Vec<Vec<u32>>> = first
        .table
        .phases()
        .iter()
        .map(|_| vec![vec![0u32; D]; first.placement.workers as usize])
        .collect();
    for plan in plans {
        assert_eq!(plan.me, first.me, "every plan of a session names one worker");
        assert_eq!(plan.placement.workers, first.placement.workers);
        assert_eq!(plan.table.digest(), first.table.digest(), "every plan walks one phase table");
        assert_eq!(plan.id_offset as usize, lens.len(), "plans occupy disjoint id ranges in offset order");
        lens.extend(plan.lens());
        for (byte, per_peer) in plan.phase_counts().into_iter().enumerate() {
            for (peer, per_axis) in per_peer.into_iter().enumerate() {
                for (axis, c) in per_axis.into_iter().enumerate() {
                    counts[byte][peer][axis] += c;
                }
            }
        }
    }
    let prime = first.table.byte_of(ExchangePoint::Prime).expect("every table holds the prime") as usize;
    let default = counts[prime].clone();
    Fabric::new(link, first.me, session, first.placement.workers as usize, D, &lens, default)
        .with_phase_counts(counts)
        .with_phase_rules(first.table.rules())
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
pub fn gather<const D: usize, M: MemorySpace>(
    field: &Field<f64, D, M>,
    region: &[Span; D],
    buf: &mut [f64],
) {
    let dom = field.domain();
    for a in 0..D {
        assert!(
            region[a].lo >= dom.spaces[a].lo && region[a].hi <= dom.spaces[a].hi,
            "a transfer's source region {region:?} lies outside its field's domain {:?} on axis {a}: the \
             plan's tile layout and the tile's store disagree",
            dom.spaces
        );
    }
    let view = field.view();
    for (slot, c) in buf.iter_mut().zip(domain_of(region).iter()) {
        *slot = *view.at(c);
    }
}

/// write `buf` over `region` of `field` in region iteration order.
pub fn scatter<const D: usize, M: MemorySpace>(
    field: &Field<f64, D, M>,
    region: &[Span; D],
    buf: &[f64],
) {
    let mut view = field.view_mut();
    for (c, value) in domain_of(region).iter().zip(buf) {
        *view.at_mut(c) = *value;
    }
}

/// audit the shared interface faces of several plans of one session in one phase per axis at
/// `point`: each plan compares its local pairs and stages its remote copies, the phase opens
/// once with the union, and every landed copy is compared against its plan's upper copy. the
/// first differing face ends the audit as an error
pub fn audit_plans<const D: usize, M: MemorySpace, L: Link>(
    parts: &[(&PlanExchange<'_, D>, &[Vec<&Field<f64, D, M>>])],
    fabric: &mut Fabric<L>,
    point: ExchangePoint,
    step: u64,
    attempt: u16,
    deadline: std::time::Duration,
) -> Result<AuditCost, AuditError> {
    let clock = std::time::Instant::now();
    let mut cost = AuditCost::default();
    let first = parts.first().expect("at least one plan").0;
    for axis in 0..D {
        let epoch = first.epoch(point, step, attempt, axis);
        let mut sends = Vec::new();
        let mut receives = Vec::new();
        for (plan, fields) in parts {
            let (s, r) = plan.audit_stage_axis(fields, fabric, axis, &mut cost)?;
            sends.extend(s);
            receives.extend(r);
        }
        fabric.open(PhaseSpec { epoch, sends: &sends, receives: &receives })?;
        fabric.wait(deadline)?;
        for (plan, fields) in parts {
            plan.audit_land_axis(fields, fabric, axis, &mut cost)?;
        }
        fabric.close()?;
    }
    cost.elapsed = clock.elapsed();
    Ok(cost)
}
