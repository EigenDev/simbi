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
//  let mut fabric = Fabric::new(link, me, session, workers, D, &exchange.lens(), exchange.sends_per_peer_axis());
//  exchange.open_axis(&fields, &devices, &LocalCopy, &mut fabric, epoch)?;
//  while fabric.progress()? == Progress::Pending {}
//  exchange.finish_axis(&fields, &mut fabric, axis)?;
// =============================================================================

use crate::atlas::{ExchangePlan, ExchangePoint, Placement, Span, TileId, domain_of};
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

    /// the transfers this worker sends to `[peer][axis]`.
    pub fn sends_per_peer_axis(&self) -> Vec<Vec<u32>> {
        let mut counts = vec![vec![0u32; D]; self.placement.workers as usize];
        for t in self.plan.transfers() {
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
        let mut sends = Vec::new();
        let mut receives = Vec::new();
        for t in &phase.transfers {
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
        for t in &self.plan.axes[axis].transfers {
            if self.owns(t.dst) && !self.owns(t.src) {
                let field = fields[t.dst.0 as usize][t.field as usize];
                scatter(field, &t.dst_region, fabric.payload(t.id)?);
                fabric.mark_unpacked(t.id)?;
            }
        }
        fabric.close()
    }
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
