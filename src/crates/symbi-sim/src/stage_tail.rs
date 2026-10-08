// =============================================================================
// stage_tail.rs
//
// what a flat tiled march does after a stage is agreed and accepted, inside the
// attempt, the same on host tiles and on workers. a driver supplies its
// transport through `FlatStageTail`; `flat_stage_tail` owns the sequence:
// - the stage-end excision, every sweep and the rebuild refreshed at the
//   driver's points;
// - the stage's halo refresh: the cut halos from the neighbors' updated
//   interiors, the polar bands, the physical ghosts (and any audit the driver
//   runs on the refreshed state);
// - on a traced run, each tile on the refreshed state: the ito transport
//   accumulation, the boundary and source injection booked into its ledger,
//   and the discrete tracers' owners moved;
// - on a run carrying discrete tracers, the leavers cross to their owners'
//   tiles, the ancestry blend runs, and the reverted cross again.
// the cross-tile migration is the step no per-tile restore inverts, so a march
// running this tail refuses a rejected step with tracers attached.
//
// usage:
//  flat_stage_tail(&mut tail, &stage)?;
// =============================================================================

use std::collections::BTreeMap;

use crate::driver::StageSpec;
use crate::mass_transport::ContainerId;
use crate::state::FieldStore;
use crate::tracers::TransportLayout;
use symbi_xpu::MemorySpace;

/// the transport of a flat tiled march's stage tail on one participant's tiles
pub trait FlatStageTail<const D: usize, const DOF: usize, M: MemorySpace> {
    type Error;
    /// the stage-end excision through the driver's step context
    fn excise(&mut self) -> Result<(), Self::Error>;
    /// the cut halos, the polar bands and the physical ghosts refreshed from the stage's state
    fn refresh_halos(&mut self) -> Result<(), Self::Error>;
    /// whether the run carries discrete and continuous tracers, as every participant agrees
    fn tracers(&self) -> (bool, bool);
    fn tiles(&self) -> usize;
    /// tile `k`'s store and its injection ledger
    fn tile(&mut self, k: usize) -> (&mut FieldStore<D, DOF, M, f64>, &mut BTreeMap<ContainerId, f64>);
    /// tile `k`'s place in the whole grid, for the tracer transport
    fn layout(&self, k: usize) -> TransportLayout<D>;
    /// the leavers of every tile crossed to their owners' tiles
    fn migrate(&mut self) -> Result<(), Self::Error>;
    /// the ssp ancestry blend over the whole population, at stage weight `ac`
    fn blend(&mut self, ac: f64, stage: usize);
    /// a tracer step that failed, as this march reports it
    fn failure(&mut self, detail: String) -> Self::Error;
}

/// the tail of one accepted stage on a flat tiled march
pub fn flat_stage_tail<const D: usize, const DOF: usize, M, C>(cx: &mut C, stage: &StageSpec) -> Result<(), C::Error>
where
    M: MemorySpace,
    C: FlatStageTail<D, DOF, M>,
    symbi_geometry::Cartesian: symbi_geometry::Metric<f64, D>,
{
    cx.excise()?;
    cx.refresh_halos()?;
    let (discrete, continuous) = cx.tracers();
    if !(discrete || continuous) {
        return Ok(());
    }
    for k in 0..cx.tiles() {
        let layout = cx.layout(k);
        let (store, ledger) = cx.tile(k);
        let geometry = store.geom.block_geometry(symbi_geometry::Cartesian);
        let moved = (|| {
            if continuous {
                crate::tracers::accumulate_ito_transport_stage_store(store, &geometry, stage.ac)
                    .map_err(|e| format!("ito transport accumulation: {e}"))?;
            }
            let mut injections = crate::tracers::boundary_injection_transfers_store(&*store, &geometry, layout);
            injections.extend(crate::tracers::source_injection_transfers_store(&*store, &geometry, layout, stage.a0, stage.ac));
            crate::tracers::fold_injection_ledger(ledger, injections, stage.ac);
            if discrete {
                crate::tracers::advance_stage_mass_transport_tile(store, &geometry, layout, stage.a0, stage.ac, stage.index)
                    .map_err(|e| format!("tracer mass transport: {e}"))?;
            }
            Ok::<(), String>(())
        })();
        if let Err(detail) = moved {
            return Err(cx.failure(detail));
        }
    }
    if discrete {
        cx.migrate()?;
        cx.blend(stage.ac, stage.index);
        cx.migrate()?;
    }
    Ok(())
}
