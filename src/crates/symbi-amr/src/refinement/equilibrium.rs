// =============================================================================
// equilibrium.rs
//
// the pieces for evolving a refinement level against a stationary target state: stash and restore
// the live gas state, snapshot the target's materialized flux, and hold the target's discrete
// imbalance so a stage can subtract it.
//
// a steady state solves the continuum equations, not the discrete ones. on any single grid the
// scheme leaves a residual
//   R := div_h F_h(qt) - s_h(qt),
// nonzero at truncation order, and that residual is what makes an atmosphere seeded on the exact
// hydrostatic profile start moving. worse, R is grid-dependent: a coarse grid and a fine grid
// reduce the same exact solution to different face values, so the coarse-fine flux register
// differences two unequal reconstructions and applies the difference to the coarse cells at the
// interface as a spurious force.
//
// the cure is to evolve the deviation from the target
// (Berberich, Chandrashekar & Klingenberg, Computers and Fluids 219 (2021) 104858). the numerical
// flux becomes
//   F_hat(Q_L, Q_R) := F(Q_L, Q_R) - F(qt_L, qt_R)                              (eq. 8)
// and the source `s(qt + dq) - s(qt)`, which for a source linear in the conserved state at fixed
// potential — gravity, `s = (0, rho g, m . g)` — is just `s(dq)`. summed over a cell the two
// subtractions are exactly `+R`, so the whole method reduces to adding `R` back at every stage,
// which makes the target an exact fixed point of the scheme.
//
// both halves are required and neither suffices alone. subtracting only at the coarse-fine
// interface leaves the fine cells sharing that face still transporting `F(qt)`, so what the coarse
// cell receives and what the fine cells send no longer agree and the composite leaks. correcting
// only the interiors leaves the register differencing two reconstructions of the target. together
// the interface flux is single-valued in the deviation, so the deviation is conserved exactly, and
// a time-independent target adds a constant — so the total is conserved exactly too.
//
// the subtracted flux is the numerical flux applied to the target, per level. the analytic flux at
// the physical face carries no grid dependence, cancels from both sides of the register
// difference, and would therefore remove exactly nothing.
//
// usage:
//  let saved = save_gas_state(&level.state)?;
//  level.state.seed_cells(&target);
//  // recover primitives + fill ghost bands, then materialize the fluxes
//  let flux_eq = snapshot_flux(&level.state)?;
//  restore_gas_state(&level.state, &saved);
// =============================================================================

use symbi_algebra::Domain;
use symbi_ir::KernelId;
use symbi_sim::state::{ConsFieldsGeneric, FieldStore, PrimFieldsGeneric};
use symbi_substrate::regimes::substrate_kernels::dispatch_fields_each;
use symbi_xpu::MemorySpace;

use super::transfer::copy_field;

/// the conserved flux of one level's stationary target, one entry per grid direction, over the
/// level's allocated domain.
pub type EquilibriumFlux<const NDIM: usize, const DOF: usize, Mem> =
    [ConsFieldsGeneric<NDIM, DOF, Mem>; NDIM];

/// a level's conserved and primitive gas fields over its allocated domain, ghost bands included.
pub struct GasState<const NDIM: usize, const DOF: usize, Mem: MemorySpace> {
    cons: ConsFieldsGeneric<NDIM, DOF, Mem>,
    prim: PrimFieldsGeneric<NDIM, DOF, Mem>,
}

impl<const NDIM: usize, const DOF: usize, Mem: MemorySpace> GasState<NDIM, DOF, Mem> {
    pub fn cons(&self) -> &ConsFieldsGeneric<NDIM, DOF, Mem> {
        &self.cons
    }
    pub fn into_cons(self) -> ConsFieldsGeneric<NDIM, DOF, Mem> {
        self.cons
    }
}

/// `dst = src` on every conserved component over `domain`.
pub fn overwrite<const NDIM: usize, const DOF: usize, Mem: MemorySpace>(
    src: &ConsFieldsGeneric<NDIM, DOF, Mem>,
    dst: &ConsFieldsGeneric<NDIM, DOF, Mem>,
    domain: &Domain<NDIM>,
) {
    let name = KernelId::FieldCopy { ndim: NDIM as u8 }.name();
    for (s, d) in comps(src).into_iter().zip(comps(dst)) {
        dispatch_fields_each::<f64, Mem, NDIM>(name, domain, &[s], &[d], &[], &[]);
    }
}

/// copy out the conserved and primitive state so a level can be driven through a state it does not
/// own and handed back unchanged.
pub fn save_gas_state<const NDIM: usize, const DOF: usize, Mem: MemorySpace>(
    store: &FieldStore<NDIM, DOF, Mem>,
) -> symbi_xpu::Result<GasState<NDIM, DOF, Mem>> {
    let alloc = &store.geom.allocated;
    let cons = ConsFieldsGeneric::zeros_with_energy(alloc, store.fields.cons.has_energy())?;
    let prim =
        PrimFieldsGeneric::zeros_with_pressure(alloc, store.fields.prim.pre_field().is_some())?;
    copy_cons(&store.fields.cons, &cons);
    copy_prim(&store.fields.prim, &prim);
    Ok(GasState { cons, prim })
}

/// put a saved conserved and primitive state back, component for component.
pub fn restore_gas_state<const NDIM: usize, const DOF: usize, Mem: MemorySpace>(
    store: &FieldStore<NDIM, DOF, Mem>,
    saved: &GasState<NDIM, DOF, Mem>,
) {
    copy_cons(&saved.cons, &store.fields.cons);
    copy_prim(&saved.prim, &store.fields.prim);
}

/// copy out the materialized per-direction interface fluxes.
pub fn snapshot_flux<const NDIM: usize, const DOF: usize, Mem: MemorySpace>(
    store: &FieldStore<NDIM, DOF, Mem>,
) -> symbi_xpu::Result<EquilibriumFlux<NDIM, DOF, Mem>> {
    let alloc = &store.geom.allocated;
    let has_energy = store.fields.cons.has_energy();
    let mut captured = Vec::with_capacity(NDIM);
    for dd in 0..NDIM {
        let snapshot = ConsFieldsGeneric::zeros_with_energy(alloc, has_energy)?;
        copy_cons(&store.fields.flux[dd], &snapshot);
        captured.push(snapshot);
    }
    Ok(captured.try_into().unwrap_or_else(|_| unreachable!()))
}

/// the target's discrete imbalance per unit time, `R = div_h F_h(qt) - s_h(qt)`, read off the
/// two conserved states that bracket one explicit stage of length `dt` started from the target:
/// that stage produces `qt - dt R`, so `R = (qt - advanced)/dt`.
///
/// `R` is independent of `dt` — a single explicit stage evaluates every flux and every source at
/// the target and at no other state, so the advanced state is exactly linear in `dt`.
pub fn imbalance_from_stage<const NDIM: usize, const DOF: usize, Mem: MemorySpace>(
    target: &ConsFieldsGeneric<NDIM, DOF, Mem>,
    advanced: &ConsFieldsGeneric<NDIM, DOF, Mem>,
    domain: &Domain<NDIM>,
    dt: f64,
) -> symbi_xpu::Result<ConsFieldsGeneric<NDIM, DOF, Mem>> {
    let residual = ConsFieldsGeneric::zeros_with_energy(domain, target.has_energy())?;
    for field in comps(&residual) {
        dispatch_fields_each::<f64, Mem, NDIM>(
            KernelId::FieldFill { ndim: NDIM as u8 }.name(),
            domain,
            &[],
            &[field],
            &[],
            &[0.0],
        );
    }
    accumulate(target, &residual, domain, 1.0 / dt);
    accumulate(advanced, &residual, domain, -1.0 / dt);
    Ok(residual)
}

/// the volume-weighted L1 norm of each conserved component of a residual over `domain`:
/// `sum |R_i| V_i`, one entry per component in component order.
///
/// a bulk magnitude, for reporting how large an imbalance is. it is not the statistic that decides
/// whether a target is stationary: a sum is dominated by wherever the imbalance happens to be
/// largest, which for a target with an unresolved feature is a handful of cells that carry no
/// information about the rest. that question is answered per cell — see
/// `Hierarchy::target_imbalance_convergence`.
pub fn residual_norm<const NDIM: usize, const DOF: usize, Mem: MemorySpace>(
    residual: &ConsFieldsGeneric<NDIM, DOF, Mem>,
    domain: &Domain<NDIM>,
    dx: &[f64; NDIM],
) -> Vec<f64> {
    let volume: f64 = dx.iter().product();
    comps(residual)
        .into_iter()
        .map(|field| {
            let view = field.view();
            domain.iter().map(|c| view.at(c).abs() * volume).sum::<f64>()
        })
        .collect()
}

/// `dst += w * src` on every conserved component over `domain`.
pub fn accumulate<const NDIM: usize, const DOF: usize, Mem: MemorySpace>(
    src: &ConsFieldsGeneric<NDIM, DOF, Mem>,
    dst: &ConsFieldsGeneric<NDIM, DOF, Mem>,
    domain: &Domain<NDIM>,
    w: f64,
) {
    let name = KernelId::FieldAxpyShift { ndim: NDIM as u8 }.name();
    let shift = [0i32; 3];
    for (s, d) in comps(src).into_iter().zip(comps(dst)) {
        dispatch_fields_each::<f64, Mem, NDIM>(name, domain, &[s], &[d], &shift[..NDIM], &[w]);
    }
}

/// the conserved components of a residual as a flat list, for a caller that walks them cell by
/// cell rather than through a kernel. same order as every other component operation here.
pub fn residual_components<const NDIM: usize, const DOF: usize, Mem: MemorySpace>(
    residual: &ConsFieldsGeneric<NDIM, DOF, Mem>,
) -> Vec<&symbi_grid::Field<f64, NDIM, Mem>> {
    comps(residual)
}

/// the conserved components as a flat list: den, mom[0..DOF], then energy when present. two lists
/// zip positionally, so both sides of an operation carry the same component order.
fn comps<const NDIM: usize, const DOF: usize, Mem: MemorySpace>(
    c: &ConsFieldsGeneric<NDIM, DOF, Mem>,
) -> Vec<&symbi_grid::Field<f64, NDIM, Mem>> {
    let mut v = Vec::with_capacity(DOF + 2);
    v.push(&c.den);
    for dd in 0..DOF {
        v.push(&c.mom[dd]);
    }
    if let Some(nrg) = c.nrg_field() {
        v.push(nrg);
    }
    v
}

fn copy_cons<const NDIM: usize, const DOF: usize, Mem: MemorySpace>(
    src: &ConsFieldsGeneric<NDIM, DOF, Mem>,
    dst: &ConsFieldsGeneric<NDIM, DOF, Mem>,
) {
    for (s, d) in comps(src).into_iter().zip(comps(dst)) {
        copy_field(s, d);
    }
}

fn copy_prim<const NDIM: usize, const DOF: usize, Mem: MemorySpace>(
    src: &PrimFieldsGeneric<NDIM, DOF, Mem>,
    dst: &PrimFieldsGeneric<NDIM, DOF, Mem>,
) {
    copy_field(&src.rho, &dst.rho);
    for dd in 0..DOF {
        copy_field(&src.vel[dd], &dst.vel[dd]);
    }
    if let (Some(s), Some(d)) = (src.pre_field(), dst.pre_field()) {
        copy_field(s, d);
    }
}
