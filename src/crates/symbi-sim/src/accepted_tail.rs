// =============================================================================
// accepted_tail.rs
//
// the immersed-body and continuous-tracer work a flat tiled march does once per
// accepted root step, the same on host tiles and on workers. a driver supplies
// its reductions and transport through `FlatBodies` / `FlatContinuous`; the
// operations here own the sequence.
// - `flat_body_step`: the domain's largest Alfven stiffness on every tile; the
//   density each sink drains from, kept for the tracers it carries; the surface
//   physics on every tile, recovered and its cut halos refreshed when a body
//   rescales the gas; the feedback of the bodies whose dynamics consume it;
//   the traced cells' tracers crossed into the sink reservoirs at the step's
//   end; the per-tile deltas summed in flat tile order and applied to every
//   tile's bodies alike.
// - `flat_continuous_step`: each tile's ito coefficients materialized and
//   their physical ghosts filled, the cut ghosts crossed, each tile's tracers
//   advanced and met at the domain's faces, the leavers crossed to their cells'
//   tiles, the held injection spawned.
// nothing here is rolled back: it follows the step's acceptance, and nothing
// after it can reject the step.
//
// usage:
//  flat_body_step(&mut bodies, dt)?;
//  flat_continuous_step(&mut continuous, dt)?;
// =============================================================================

use crate::state::FieldStore;
use crate::substrate_seam::KernelSet;
use crate::tracers::TransportLayout;
use symbi_xpu::MemorySpace;

/// the tiles one participant holds: their stores, kernel sets and devices, by local index
pub struct HeldTiles<'a, 's, const D: usize, const DOF: usize, M: MemorySpace, K> {
    pub stores: &'a mut [&'s mut FieldStore<D, DOF, M, f64>],
    pub kernels: &'a [&'a K],
    pub devices: &'a [i32],
}

/// the reductions and transport of a flat tiled march's body step on one participant's tiles
pub trait FlatBodies<'s, const D: usize, const DOF: usize, M: MemorySpace> {
    type Error;
    type Kernels: KernelSet<D, DOF, M, f64>;
    fn held(&mut self) -> HeldTiles<'_, 's, D, DOF, M, Self::Kernels>;
    /// the device queues drained before a host read of the tiles' fields
    fn drain(&mut self) {}
    /// the largest value over every tile of every participant, from this participant's tiles'
    /// values in local order
    fn max(&mut self, per_tile: &[f64]) -> Result<f64, Self::Error>;
    /// every tile's cut halos refreshed from its neighbors after a body rescaled the gas
    fn refresh_halos(&mut self) -> Result<(), Self::Error>;
    /// tile `k`'s place in the whole grid, for the tracer transport
    fn layout(&self, k: usize) -> TransportLayout<D>;
    /// every tile's exact deltas, in flat tile order across every participant
    fn gather_deltas(&mut self) -> Result<Vec<Vec<symbi_ib::ExactBodyDelta<D>>>, Self::Error>;
    /// a step that failed, as this march reports it
    fn failure(&mut self, detail: String) -> Self::Error;
}

/// the immersed bodies' accepted-step work on a flat tiled march
pub fn flat_body_step<'s, const D: usize, const DOF: usize, M, C>(cx: &mut C, dt: f64) -> Result<(), C::Error>
where
    M: MemorySpace + 's,
    C: FlatBodies<'s, D, DOF, M>,
    symbi_geometry::Cartesian: symbi_geometry::Metric<f64, D>,
{
    let (c_a2_tiles, traced) = {
        let held = cx.held();
        let per_tile: Vec<f64> = held.stores.iter().map(|s| crate::state::local_c_a2_max(s)).collect();
        (per_tile, held.stores.iter().any(|s| s.tracers.is_some() || s.continuous_tracers.is_some()))
    };
    // the wall relaxation's Alfven stiffness max|B|^2/rho is a maximum over the whole domain, so a
    // magnetized wall straddling a cut relaxes at the single grid's rate (zero off MHD)
    let c_a2 = cx.max(&c_a2_tiles)?;
    if traced {
        cx.drain();
    }
    let (accretion_density, penalizes) = {
        let held = cx.held();
        for s in held.stores.iter() {
            if let Some(im) = s.immersed.as_ref() {
                im.set_c_a2_override(c_a2);
            }
        }
        let density: Vec<_> = if traced {
            held.stores.iter().map(|s| crate::tracers::snapshot_accretion_density(s)).collect()
        } else {
            Vec::new()
        };
        // the bodies are the same on every tile, so every participant takes the same branch
        let penalizes = held.stores.iter().any(|s| s.immersed.as_ref().is_some_and(|im| im.bodies.penalizes()));
        for k in 0..held.stores.len() {
            let (store, kernel) = (&*held.stores[k], held.kernels[k]);
            symbi_xpu::with_device(held.devices[k], || {
                crate::driver::prof("penalize", || kernel.penalize(store, dt));
                if penalizes {
                    crate::driver::prof("c2p", || kernel.c2p(store));
                    crate::driver::prof("ghost_fill", || kernel.ghost_fill(store));
                }
            });
        }
        (density, penalizes)
    };
    if penalizes {
        cx.refresh_halos()?;
    }
    {
        // the feedback of the bodies whose dynamics consume the reduction (two-way or accreting)
        let held = cx.held();
        for k in 0..held.stores.len() {
            let (store, kernel) = (&*held.stores[k], held.kernels[k]);
            if store.immersed.as_ref().is_some_and(|im| im.bodies.needs_feedback()) {
                symbi_xpu::with_device(held.devices[k], || crate::driver::prof("body_feedback", || kernel.body_feedback(store, dt)));
            }
        }
    }
    cx.drain();
    // each tile's tracers in the drained cells cross into the sink reservoirs at the step's end
    if traced {
        let n = cx.held().stores.len();
        for k in 0..n {
            let layout = cx.layout(k);
            let crossed = {
                let held = cx.held();
                let store = &mut *held.stores[k];
                let geometry = store.geom.block_geometry(symbi_geometry::Cartesian);
                let crossing_time = store.time + dt;
                crate::tracers::advance_accretion_transport_store(store, &geometry, layout, &accretion_density[k], crossing_time).and_then(|_| {
                    crate::tracers::advance_continuous_accretion_transport_store(store, &geometry, layout, &accretion_density[k], crossing_time)
                })
            };
            if let Err(detail) = crossed {
                return Err(cx.failure(format!("tracer accretion transport: {detail}")));
            }
        }
    }
    let global = crate::decomp::merge_body_deltas(cx.gather_deltas()?);
    crate::decomp::apply_global_body_step(cx.held().stores, &global, dt);
    Ok(())
}

/// the transport of a flat tiled march's continuous-tracer step on one participant's tiles
pub trait FlatContinuous<'s, const D: usize, const DOF: usize, M: MemorySpace> {
    type Error;
    fn stores(&mut self) -> &mut [&'s mut FieldStore<D, DOF, M, f64>];
    /// the domain's extent per axis and the condition each face presents to a tracer
    fn domain(&self) -> ([(f64, f64); D], crate::state::Boundaries<D>);
    /// every tile's coefficient cut ghosts filled from its neighbors' interiors
    fn exchange_coefficients(&mut self) -> Result<(), Self::Error>;
    /// every tile's leavers crossed to the tiles holding their cells
    fn migrate(&mut self) -> Result<(), Self::Error>;
    /// the injection the accepted stages booked, spawned once over `dt`
    fn spawn(&mut self, dt: f64) -> Result<(), Self::Error>;
    /// a step that failed, as this march reports it
    fn failure(&mut self, detail: String) -> Self::Error;
}

/// the continuous tracers' accepted-step work on a flat tiled march
pub fn flat_continuous_step<'s, const D: usize, const DOF: usize, M, C>(cx: &mut C, dt: f64) -> Result<(), C::Error>
where
    M: MemorySpace + 's,
    C: FlatContinuous<'s, D, DOF, M>,
    symbi_geometry::Cartesian: symbi_geometry::Metric<f64, D>,
{
    for k in 0..cx.stores().len() {
        let materialized = {
            let store = &mut *cx.stores()[k];
            let geometry = store.geom.block_geometry(symbi_geometry::Cartesian);
            crate::tracers::materialize_ito_coefficients_store(store, &geometry)
                .map_err(|e| format!("ito coefficient materialization: {e}"))
                .and_then(|_| {
                    let coefficients = store.ito_coefficients.as_ref().expect("ito coefficients were materialized");
                    crate::tracers::fill_ito_coefficient_boundaries_host(coefficients, &store.geom, store.boundaries)
                        .map_err(|e| format!("ito coefficient boundaries: {e}"))
                })
        };
        if let Err(detail) = materialized {
            return Err(cx.failure(detail));
        }
    }
    cx.exchange_coefficients()?;
    let (bounds, boundaries) = cx.domain();
    for k in 0..cx.stores().len() {
        let advanced = {
            let store = &mut *cx.stores()[k];
            let Some(mut tracers) = store.continuous_tracers.take() else { continue };
            let (scale_start, scale_end, offset_start, offset_end) = crate::tracers::continuous_tracer_mesh_step(&*store, dt);
            let coefficients = store.ito_coefficients.as_ref().expect("ito coefficients were materialized");
            let advanced = crate::tracers::advance_continuous_tracers(
                &mut tracers,
                coefficients,
                &store.geom,
                scale_start,
                scale_end,
                offset_start,
                offset_end,
                dt,
            )
            .map_err(|e| format!("ito tracer advancement: {e}"))
            .and_then(|_| {
                let physical = crate::tracers::map_continuous_tracer_bounds(bounds, scale_end, offset_end);
                crate::tracers::apply_continuous_boundaries_host(&mut tracers, physical, boundaries)
                    .map_err(|e| format!("ito tracer boundaries: {e}"))
            });
            store.continuous_tracers = Some(tracers);
            advanced
        };
        if let Err(detail) = advanced {
            return Err(cx.failure(detail));
        }
    }
    cx.migrate()?;
    cx.spawn(dt)
}
