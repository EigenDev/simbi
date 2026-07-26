// =============================================================================
// transfer.rs
//
// the inter-level field-transfer driver: selects the
// regions and dispatches the aot amr kernels (refine_restrict_{D}d /
// refine_prolong_{order}_{D}d, built in symbi-discretize gv_refinement.rs) per field
// component through `dispatch_fields_each` — each buffer resolves the ABSOLUTE
// level-global thread coordinate against its own lo, so no index translation
// appears here.
//
//   cf_ghost_slabs  — the coarse-fine ghost regions of a fine level (one slab
//                     per CoarseFine face, extended into CF corners, clipped at
//                     physical corners)
//   prolong_prims   — time-interpolated coarse -> fine prim fill over a slab
//   restrict_cons   — conservative fine -> coarse cons average over a coverage
//
// usage:
//  for slab in cf_ghost_slabs(&alloc, &interior, &boundaries) {
//      prolong_prims(&old, &new, &fine_prim, &slab, order, alpha);
//  }
//  restrict_cons(&fine_cons, &coarse_cons, &coverage);
// =============================================================================

use symbi_algebra::{Domain, Space};
use symbi_xpu::MemorySpace;

use symbi_ir::{KernelId, ProlongTag};
use symbi_sim::driver::prof;
use symbi_sim::state::{Boundaries, BoundaryType, ConsFieldsGeneric, PrimFieldsGeneric};
use symbi_substrate::regimes::substrate_kernels::dispatch_fields_each;

/// coarse-fine prolongation order — the driver-side selector for the aot
/// kernel instance (the same by-name seam `Solver` uses for hlle/hllc). the
/// kernels are built by symbi-discretize's `gv_refinement::ProlongOrder`; the
/// registered name tags (pcm/plm/ppm) are the contract between the two.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProlongOrder {
    /// piecewise constant (order 0). stencil halfwidth 0.
    Pcm,
    /// piecewise linear, van leer limited (order 1). stencil halfwidth 1.
    Plm,
    /// piecewise parabolic, monotonized sub-cell averages (order 2). halfwidth 2.
    Ppm,
}

impl ProlongOrder {
    /// coarse cells the 1d stencil reads per side beyond the parent.
    pub fn ghost_width(self) -> usize {
        match self {
            ProlongOrder::Pcm => 0,
            ProlongOrder::Plm => 1,
            ProlongOrder::Ppm => 2,
        }
    }
}

/// dst = src via the pointwise copy kernel (the device-aware field snapshot;
/// both fields on the same domain, which is also the exec domain).
pub fn copy_field<const D: usize, Mem: MemorySpace>(
    src: &symbi_grid::Field<f64, D, Mem>,
    dst: &symbi_grid::Field<f64, D, Mem>,
) {
    dispatch_fields_each::<f64, Mem, D>(
        KernelId::FieldCopy { ndim: D as u8 }.name(),
        src.domain(),
        &[src],
        &[dst],
        &[],
        &[],
    );
}

/// the typed prolong-kernel tag for a reconstruction order (the ABI mirror of
/// `ProlongOrder`, consumed by `KernelId::RefineProlong`).
pub fn prolong_tag(order: ProlongOrder) -> ProlongTag {
    match order {
        ProlongOrder::Pcm => ProlongTag::Pcm,
        ProlongOrder::Plm => ProlongTag::Plm,
        ProlongOrder::Ppm => ProlongTag::Ppm,
    }
}

/// the coarse-fine ghost slabs of a fine level, in absolute fine indices: one
/// slab per CoarseFine face. transverse extents include the allocated ghosts
/// on sides that are themselves CoarseFine (prolongation owns those corners)
/// and clip to the interior on physical sides (the physical ghost fill owns
/// those). CF-CF corners appear in both axis slabs — the double write is the
/// same value.
pub fn cf_ghost_slabs<const D: usize>(
    allocated: &Domain<D>,
    interior: &Domain<D>,
    boundaries: &Boundaries<D>,
) -> Vec<Domain<D>> {
    // POLICY: which ghosts come from the coarser level. `cf_region` is the
    // interior grown out to `allocated` ONLY on coarse-fine sides — physical
    // boundary ghosts are filled by the bc kernel, so those sides
    // stay clamped to `interior`. the cells to prolong are then exactly
    // `cf_region \ interior`.
    //
    // GEOMETRY: `guillotine_difference` (symbi-algebra) returns that set as the
    // minimal disjoint cover — `2*D` boxes, no overlap. this is union-equivalent
    // to overlapping per-face slabs (identical cell set; prolongation is a pure
    // function of (coarse state, fine coord), so the 2-3x edge/corner
    // double-writes of an overlapping cover are redundant yet still correct) but writes each cell ONCE: the
    // ~19% cell reduction on binary_disk, in the same `2*D` dispatches (not the
    // `3^D-1` of a maximal split, whose tiny corner boxes drown in launch cost).
    // the disjointness also makes the cover safe to fan out in one parallel pass.
    let cf_region = Domain::new(std::array::from_fn(|a| {
        let lo = if boundaries.lo(a) == BoundaryType::CoarseFine {
            allocated.spaces[a].lo
        } else {
            interior.spaces[a].lo
        };
        let hi = if boundaries.hi(a) == BoundaryType::CoarseFine {
            allocated.spaces[a].hi
        } else {
            interior.spaces[a].hi
        };
        Space {
            name: allocated.spaces[a].name,
            lo,
            hi,
        }
    }));
    cf_region.guillotine_difference(interior)
}

/// prolong one cell-centered scalar field from the time-interpolated coarse
/// state `(1 - alpha)*old + alpha*new` into a fine region. `old` and `new` may
/// be the same buffer (no time interpolation — alpha then picks either
/// endpoint exactly).
pub fn prolong_field<const D: usize, Mem: MemorySpace>(
    old: &symbi_grid::Field<f64, D, Mem>,
    new: &symbi_grid::Field<f64, D, Mem>,
    dst: &symbi_grid::Field<f64, D, Mem>,
    region: &Domain<D>,
    order: ProlongOrder,
    alpha: f64,
) {
    let name = KernelId::RefineProlong {
        order: prolong_tag(order),
        ndim: D as u8,
    }
    .name();
    dispatch_fields_each::<f64, Mem, D>(name, region, &[old, new], &[dst], &[], &[alpha]);
}

/// prolong the primitive components (rho, vel[0..DOF], pre when present).
pub fn prolong_prims<const D: usize, const DOF: usize, Mem: MemorySpace>(
    old: &PrimFieldsGeneric<D, DOF, Mem>,
    new: &PrimFieldsGeneric<D, DOF, Mem>,
    dst: &PrimFieldsGeneric<D, DOF, Mem>,
    region: &Domain<D>,
    order: ProlongOrder,
    alpha: f64,
) {
    // component order: rho, vel[0..DOF], pre (when present).
    let has_pre = old.pre_field().is_some();
    let ncomp = 1 + DOF + has_pre as usize;
    // multi-field BATCH: one dispatch (one rayon launch) over the whole prim set
    // collapsing `ncomp` separate launches — the per-dispatch fork-join was the
    // dominant prolong cost. generated for the 3D hot path (ncomp 4 = isothermal,
    // 5 = adiabatic/rhd); anything else falls back to the single-field path.
    if D == 3 && (ncomp == 4 || ncomp == 5) {
        let mut inputs: Vec<&symbi_grid::Field<f64, D, Mem>> = Vec::with_capacity(2 * ncomp);
        let mut outputs: Vec<&symbi_grid::Field<f64, D, Mem>> = Vec::with_capacity(ncomp);
        // interleaved (src_old_k, src_new_k) inputs then (dst_k) outputs — the
        // buffer order `refine_prolong_multi_gv` traces.
        inputs.push(&old.rho);
        inputs.push(&new.rho);
        outputs.push(&dst.rho);
        for kk in 0..DOF {
            inputs.push(&old.vel[kk]);
            inputs.push(&new.vel[kk]);
            outputs.push(&dst.vel[kk]);
        }
        if let (Some(po), Some(pn), Some(pd)) = (old.pre_field(), new.pre_field(), dst.pre_field())
        {
            inputs.push(po);
            inputs.push(pn);
            outputs.push(pd);
        }
        let name = KernelId::RefineProlongMulti {
            order: prolong_tag(order),
            ncomp: ncomp as u8,
            ndim: D as u8,
        }
        .name();
        dispatch_fields_each::<f64, Mem, D>(name, region, &inputs, &outputs, &[], &[alpha]);
        return;
    }
    // single-field fallback (1D/2D, or unusual component counts).
    prolong_field(&old.rho, &new.rho, &dst.rho, region, order, alpha);
    for kk in 0..DOF {
        prolong_field(
            &old.vel[kk],
            &new.vel[kk],
            &dst.vel[kk],
            region,
            order,
            alpha,
        );
    }
    if let (Some(po), Some(pn), Some(pd)) = (old.pre_field(), new.pre_field(), dst.pre_field()) {
        prolong_field(po, pn, pd, region, order, alpha);
    }
}

/// the prim batch as an ordered component list: rho, vel[0..DOF], pre (when present).
fn prim_comps<'a, const D: usize, const DOF: usize, Mem: MemorySpace>(
    p: &'a PrimFieldsGeneric<D, DOF, Mem>,
    has_pre: bool,
) -> Vec<&'a symbi_grid::Field<f64, D, Mem>> {
    let mut v: Vec<&symbi_grid::Field<f64, D, Mem>> =
        Vec::with_capacity(1 + DOF + has_pre as usize);
    v.push(&p.rho);
    for kk in 0..DOF {
        v.push(&p.vel[kk]);
    }
    if has_pre {
        v.push(p.pre_field().expect("pre present when has_pre"));
    }
    v
}

/// the parent range of `region` grown by the stencil width: floor_div(lo, 2)
/// - w .. floor_div(hi - 1, 2) + 1 + w per axis. euclidean division — ghost
/// slab indices are negative.
fn coarse_parents<const D: usize>(region: &Domain<D>, w: isize) -> Domain<D> {
    Domain::new(std::array::from_fn(|a| {
        let s = &region.spaces[a];
        symbi_algebra::Space {
            name: s.name,
            lo: s.lo.div_euclid(2) - w,
            hi: (s.hi - 1).div_euclid(2) + 1 + w,
        }
    }))
}

/// the two mixed-resolution intermediate lattices of the axis-split
/// prolongation of `region`: pass 0's output A is fine along
/// axis 0 and coarse (parents + stencil halo) elsewhere; pass 1's output B is
/// fine along axes 0..=1 and coarse along axis 2.
fn sweep_domains<const D: usize>(region: &Domain<D>, w: isize) -> (Domain<D>, Domain<D>) {
    let parents = coarse_parents(region, w);
    let mix = |fine_axes: usize| -> Domain<D> {
        Domain::new(std::array::from_fn(|a| {
            if a < fine_axes {
                region.spaces[a].clone()
            } else {
                parents.spaces[a].clone()
            }
        }))
    };
    (mix(1), mix(2))
}

/// the per-slab intermediates of the axis-split prolongation: one prim batch
/// per pass, shaped exactly to the slab's mixed lattices. SMR slabs are
/// static, so these allocate once (the fine level's lazy init) and are reused
/// every call — the step loop allocates nothing.
pub struct ProlongSweepScratch<const D: usize, const DOF: usize, Mem: MemorySpace> {
    pub a: PrimFieldsGeneric<D, DOF, Mem>,
    pub b: PrimFieldsGeneric<D, DOF, Mem>,
}

impl<const D: usize, const DOF: usize, Mem: MemorySpace> ProlongSweepScratch<D, DOF, Mem> {
    pub fn for_slab(slab: &Domain<D>, order: ProlongOrder, has_pre: bool) -> Self {
        let (a_dom, b_dom) = sweep_domains(slab, order.ghost_width() as isize);
        Self {
            a: PrimFieldsGeneric::zeros_with_pressure(&a_dom, has_pre)
                .expect("prolong sweep scratch A"),
            b: PrimFieldsGeneric::zeros_with_pressure(&b_dom, has_pre)
                .expect("prolong sweep scratch B"),
        }
    }
}

/// prolong the prim batch as THREE axis-split sweep passes over the lerped
/// coarse scratch: interp along axis 0 into A (fine-x,
/// coarse-yz), along axis 1 into B (fine-xy, coarse-z), along axis 2 into
/// `dst` — bit-identical to the fused tensor-product kernel at ~1/17 the
/// interp evaluations and ~1/14 the loads. `scratch` must have been built
/// with `for_slab(region, order, ..)` — a shape mismatch panics. plm/ppm
/// 3d ncomp 4/5 only; everything else falls back to the lerp+1t path (pcm's
/// single-load kernel cannot be beaten by three passes).
#[allow(clippy::too_many_arguments)]
pub fn prolong_prims_swept<const D: usize, const DOF: usize, Mem: MemorySpace>(
    scratch: &ProlongSweepScratch<D, DOF, Mem>,
    lerp: &PrimFieldsGeneric<D, DOF, Mem>,
    old: &PrimFieldsGeneric<D, DOF, Mem>,
    new: &PrimFieldsGeneric<D, DOF, Mem>,
    dst: &PrimFieldsGeneric<D, DOF, Mem>,
    region: &Domain<D>,
    order: ProlongOrder,
    alpha: f64,
) {
    let has_pre = old.pre_field().is_some();
    let ncomp = 1 + DOF + has_pre as usize;
    if !(D == 3 && (ncomp == 4 || ncomp == 5) && order != ProlongOrder::Pcm) {
        prof("refine_prolong_1t", || {
            prolong_prims_lerped(lerp, old, new, dst, region, order, alpha)
        });
        return;
    }
    let w = order.ghost_width() as isize;
    let (a_dom, b_dom) = sweep_domains(region, w);
    for a in 0..D {
        let (sa, ea) = (&scratch.a.rho.domain().spaces[a], &a_dom.spaces[a]);
        assert!(
            sa.lo == ea.lo && sa.hi == ea.hi,
            "prolong sweep scratch A does not match this region/order on axis {a}",
        );
    }

    // pass 0 feed: the time-lerped coarse snapshots over the parent region.
    let coarse = coarse_parents(region, w);
    let (old_c, new_c, lerp_c) = (
        prim_comps(old, has_pre),
        prim_comps(new, has_pre),
        prim_comps(lerp, has_pre),
    );
    let mut lerp_in: Vec<&symbi_grid::Field<f64, D, Mem>> = Vec::with_capacity(2 * ncomp);
    for k in 0..ncomp {
        lerp_in.push(old_c[k]);
        lerp_in.push(new_c[k]);
    }
    let name = KernelId::FieldLerpMulti {
        ncomp: ncomp as u8,
        ndim: D as u8,
    }
    .name();
    prof("refine_prolong_lerp", || {
        dispatch_fields_each::<f64, Mem, D>(name, &coarse, &lerp_in, &lerp_c, &[], &[alpha]);
    });

    // the three sweeps: lerp -> A -> B -> dst, axis 0 innermost first (the
    // inlined kernel's nesting order — the bit-identity requirement).
    let (a_c, b_c, dst_c) = (
        prim_comps(&scratch.a, has_pre),
        prim_comps(&scratch.b, has_pre),
        prim_comps(dst, has_pre),
    );
    let sweep = |axis: u8| {
        KernelId::RefineProlongSweep {
            order: prolong_tag(order),
            axis,
            ncomp: ncomp as u8,
            ndim: D as u8,
        }
        .name()
    };
    prof("refine_prolong_sw0", || {
        dispatch_fields_each::<f64, Mem, D>(sweep(0), &a_dom, &lerp_c, &a_c, &[], &[]);
    });
    prof("refine_prolong_sw1", || {
        dispatch_fields_each::<f64, Mem, D>(sweep(1), &b_dom, &a_c, &b_c, &[], &[]);
    });
    prof("refine_prolong_sw2", || {
        dispatch_fields_each::<f64, Mem, D>(sweep(2), region, &b_c, &dst_c, &[], &[]);
    });
}

/// prolong the prim batch through a pre-lerped coarse scratch: one `field_lerp`
/// pass time-interpolates the coarse snapshots ONCE PER COARSE CELL over the
/// parent region of `region` (+ stencil halo), then the single-snapshot prolong
/// reads the lerped buffer — half the gather traffic of the fused time-pair
/// kernel (which re-lerps the whole stencil neighbourhood per FINE cell), with
/// bit-identical output (the lerp expression and its consumption are unchanged;
/// only where the intermediate lives moves). `lerp` is a caller-owned coarse
/// scratch (allocated once — the step loop allocates nothing per call). falls
/// back to `prolong_prims` when the batched 3d kernels are not generated.
pub fn prolong_prims_lerped<const D: usize, const DOF: usize, Mem: MemorySpace>(
    lerp: &PrimFieldsGeneric<D, DOF, Mem>,
    old: &PrimFieldsGeneric<D, DOF, Mem>,
    new: &PrimFieldsGeneric<D, DOF, Mem>,
    dst: &PrimFieldsGeneric<D, DOF, Mem>,
    region: &Domain<D>,
    order: ProlongOrder,
    alpha: f64,
) {
    let has_pre = old.pre_field().is_some();
    let ncomp = 1 + DOF + has_pre as usize;
    if !(D == 3 && (ncomp == 4 || ncomp == 5)) {
        prolong_prims(old, new, dst, region, order, alpha);
        return;
    }

    // the coarse cells the prolong stencil reads for this fine region.
    let coarse = coarse_parents(region, order.ghost_width() as isize);

    // component order everywhere: rho, vel[0..DOF], pre (when present).
    let (old_c, new_c, lerp_c, dst_c) = (
        prim_comps(old, has_pre),
        prim_comps(new, has_pre),
        prim_comps(lerp, has_pre),
        prim_comps(dst, has_pre),
    );

    // pass 1: lerp the coarse snapshots — inputs interleaved (old_k, new_k),
    // outputs lerp_k, the field_lerp_multi_gv buffer order.
    let mut lerp_in: Vec<&symbi_grid::Field<f64, D, Mem>> = Vec::with_capacity(2 * ncomp);
    for k in 0..ncomp {
        lerp_in.push(old_c[k]);
        lerp_in.push(new_c[k]);
    }
    let name = KernelId::FieldLerpMulti {
        ncomp: ncomp as u8,
        ndim: D as u8,
    }
    .name();
    dispatch_fields_each::<f64, Mem, D>(name, &coarse, &lerp_in, &lerp_c, &[], &[alpha]);

    // pass 2: single-snapshot prolong from the lerped coarse buffer (no scalars).
    let name = KernelId::RefineProlongMulti1t {
        order: prolong_tag(order),
        ncomp: ncomp as u8,
        ndim: D as u8,
    }
    .name();
    dispatch_fields_each::<f64, Mem, D>(name, region, &lerp_c, &dst_c, &[], &[]);
}

/// restrict one cell-centered scalar field (volume-weighted child average)
/// from the fine interior onto the covered coarse cells. `coverage` is the
/// covered region in absolute coarse indices; the kernel reads the `2^D` fine
/// children at `2*c + o`.
pub fn restrict_cell_field<const D: usize, Mem: MemorySpace>(
    fine: &symbi_grid::Field<f64, D, Mem>,
    coarse: &symbi_grid::Field<f64, D, Mem>,
    coverage: &Domain<D>,
) {
    let name = KernelId::RefineRestrict { ndim: D as u8 }.name();
    dispatch_fields_each::<f64, Mem, D>(name, coverage, &[fine], &[coarse], &[], &[]);
}

/// restrict the conserved components (den, mom[0..DOF], nrg when present).
pub fn restrict_cons<const D: usize, const DOF: usize, Mem: MemorySpace>(
    fine: &ConsFieldsGeneric<D, DOF, Mem>,
    coarse: &ConsFieldsGeneric<D, DOF, Mem>,
    coverage: &Domain<D>,
) {
    restrict_cell_field(&fine.den, &coarse.den, coverage);
    for kk in 0..DOF {
        restrict_cell_field(&fine.mom[kk], &coarse.mom[kk], coverage);
    }
    if let (Some(fnrg), Some(cnrg)) = (fine.nrg_field(), coarse.nrg_field()) {
        restrict_cell_field(fnrg, cnrg, coverage);
    }
}

/// the coarse-fine halo slabs of the staggered face field bface[d] in absolute
/// fine FACE indices: one single-row slab per CF transverse side (the +/-1
/// transverse halo the flux sweep's Gardiner-Stone override reads). spans the
/// full owned face extent on the normal axis; the other transverse axis
/// extends into ITS halo where that side is also CF (corner rows) and clips
/// to the interior on physical sides (the scalar ghost fill owns those).
pub fn bface_cf_halo_slabs<const D: usize>(
    interior: &Domain<D>,
    boundaries: &Boundaries<D>,
    dd: usize,
) -> Vec<Domain<D>> {
    let mut slabs = Vec::new();
    for tt in 0..D {
        if tt == dd {
            continue;
        }
        for side in 0..2usize {
            let bc_tt = if side == 0 {
                boundaries.lo(tt)
            } else {
                boundaries.hi(tt)
            };
            if bc_tt != BoundaryType::CoarseFine {
                continue;
            }
            slabs.push(Domain::new(std::array::from_fn(|aa| {
                let s = &interior.spaces[aa];
                let (lo, hi) = if aa == tt {
                    if side == 0 {
                        (s.lo - 1, s.lo)
                    } else {
                        (s.hi, s.hi + 1)
                    }
                } else if aa == dd {
                    (s.lo, s.hi + 1)
                } else {
                    let lo = if boundaries.lo(aa) == BoundaryType::CoarseFine {
                        s.lo - 1
                    } else {
                        s.lo
                    };
                    let hi = if boundaries.hi(aa) == BoundaryType::CoarseFine {
                        s.hi + 1
                    } else {
                        s.hi
                    };
                    (lo, hi)
                };
                Space {
                    name: s.name,
                    lo,
                    hi,
                }
            })));
        }
    }
    slabs
}

/// prolong one staggered face field (normal axis `axis`) from the
/// time-interpolated coarse face lattice into a fine face region: the normal
/// axis pair-averages the coincident/midpoint coarse faces, transverse axes
/// interpolate with plm (the coarse face halo is one deep — see
/// refine_prolong_face_gv).
pub fn prolong_face_field<const D: usize, Mem: MemorySpace>(
    axis: usize,
    old: &symbi_grid::Field<f64, D, Mem>,
    new: &symbi_grid::Field<f64, D, Mem>,
    dst: &symbi_grid::Field<f64, D, Mem>,
    region: &Domain<D>,
    alpha: f64,
) {
    let name = KernelId::RefineProlongFace {
        axis: axis as u8,
        ndim: D as u8,
    }
    .name();
    dispatch_fields_each::<f64, Mem, D>(name, region, &[old, new], &[dst], &[], &[alpha]);
}

/// restrict the staggered face field bface[axis] (area-weighted average of the
/// `2^(D-1)` coincident fine faces) over the coverage FACE domain — the
/// coverage extended by one face index on the normal axis, interface faces
/// included.
pub fn restrict_bface<const D: usize, Mem: MemorySpace>(
    fine: &symbi_sim::state::BfaceFields<D, Mem>,
    coarse: &symbi_sim::state::BfaceFields<D, Mem>,
    coverage: &Domain<D>,
) {
    for aa in 0..D {
        let name = KernelId::RefineRestrictFace {
            axis: aa as u8,
            ndim: D as u8,
        }
        .name();
        let face_dom = coverage.extend(aa, 0, 1);
        dispatch_fields_each::<f64, Mem, D>(
            name,
            &face_dom,
            &[&fine[aa]],
            &[&coarse[aa]],
            &[],
            &[],
        );
    }
}

/// re-derive cell-centered B from the (restricted + reflux-corrected) face
/// field over `region`, applying the 1/2|B|^2 magnetic-energy correction to
/// cons.nrg in place when the regime carries energy — the same kernel the
/// single-level CT corrector runs, on an arbitrary exec domain.
pub fn bcell_from_bface_region<const D: usize, const DOF: usize, Mem: MemorySpace>(
    mhd: &symbi_sim::state::MhdStaggeredFields<D, DOF, Mem>,
    cons_nrg: Option<&symbi_grid::Field<f64, D, Mem>>,
    region: &Domain<D>,
) {
    let name = if cons_nrg.is_some() {
        format!("rmhd_bcell_from_bface_{}d", D)
    } else {
        format!("imhd_bcell_from_bface_{}d", D)
    };
    let inputs: Vec<&symbi_grid::Field<f64, D, Mem>> = (0..D).map(|aa| &mhd.bface[aa]).collect();
    let mut outputs: Vec<&symbi_grid::Field<f64, D, Mem>> =
        (0..D).map(|aa| &mhd.bcell[aa]).collect();
    if let Some(nrg) = cons_nrg {
        outputs.push(nrg);
    }
    dispatch_fields_each::<f64, Mem, D>(&name, region, &inputs, &outputs, &[], &[]);
}

// =============================================================================
// tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // ─── reference oracle ────────────────────────────────────────────────────
    // the ORIGINAL overlapping per-face slab construction. the disjoint
    // `cf_ghost_slabs` must cover the EXACT same cell set — this is the
    // union-equivalence contract that lets the refactor be numerically a no-op.
    fn old_overlapping_slabs<const D: usize>(
        allocated: &Domain<D>,
        interior: &Domain<D>,
        boundaries: &Boundaries<D>,
    ) -> Vec<Domain<D>> {
        let mut slabs = Vec::new();
        for ax in 0..D {
            for side in 0..2usize {
                let bc_ax = if side == 0 {
                    boundaries.lo(ax)
                } else {
                    boundaries.hi(ax)
                };
                if bc_ax != BoundaryType::CoarseFine {
                    continue;
                }
                slabs.push(Domain::new(std::array::from_fn(|aa| {
                    let (lo, hi) = if aa == ax {
                        if side == 0 {
                            (allocated.spaces[aa].lo, interior.spaces[aa].lo)
                        } else {
                            (interior.spaces[aa].hi, allocated.spaces[aa].hi)
                        }
                    } else {
                        let t_lo = if boundaries.lo(aa) == BoundaryType::CoarseFine {
                            allocated.spaces[aa].lo
                        } else {
                            interior.spaces[aa].lo
                        };
                        let t_hi = if boundaries.hi(aa) == BoundaryType::CoarseFine {
                            allocated.spaces[aa].hi
                        } else {
                            interior.spaces[aa].hi
                        };
                        (t_lo, t_hi)
                    };
                    Space {
                        name: allocated.spaces[aa].name,
                        lo,
                        hi,
                    }
                })));
            }
        }
        slabs
    }

    fn cell_set<const D: usize>(slabs: &[Domain<D>]) -> HashSet<[isize; D]> {
        slabs.iter().flat_map(|d| d.iter()).collect()
    }

    // LAW: a partition is disjoint — no two boxes share a cell.
    fn assert_disjoint<const D: usize>(slabs: &[Domain<D>]) {
        for ii in 0..slabs.len() {
            for jj in (ii + 1)..slabs.len() {
                assert!(
                    !slabs[ii].overlaps(&slabs[jj]),
                    "cf_ghost_slabs not disjoint: box {ii} overlaps box {jj}"
                );
            }
        }
    }

    // LAW: disjointness + union-equivalence to the reference oracle, over a
    // sweep of boundary configurations (every subset of the 6 faces being CF).
    fn check_laws<const D: usize>(allocated: &Domain<D>, interior: &Domain<D>) {
        for mask in 0u32..(1 << (2 * D)) {
            let mut b = Boundaries::<D>::uniform(BoundaryType::Outflow);
            for s in 0..(2 * D) {
                if mask & (1 << s) != 0 {
                    b.0[s / 2][s % 2] = BoundaryType::CoarseFine;
                }
            }
            let new = cf_ghost_slabs(allocated, interior, &b);
            let old = old_overlapping_slabs(allocated, interior, &b);

            assert_disjoint(&new);

            // disjoint => union volume is exactly the sum of box volumes.
            let new_vol: usize = new.iter().map(|d| d.volume()).sum();
            let new_cells = cell_set(&new);
            assert_eq!(
                new_vol,
                new_cells.len(),
                "mask {mask:#b}: boxes not disjoint by volume"
            );

            // union-equivalence: the disjoint partition covers the SAME cells
            // the overlapping construction did.
            assert_eq!(
                new_cells,
                cell_set(&old),
                "mask {mask:#b}: disjoint slabs cover a different cell set than the oracle"
            );
        }
    }

    #[test]
    fn cf_slabs_1d_cf_lo_face() {
        // interior [6, 14), 3 ghosts each side; lo face CF, hi face physical.
        let allocated = Domain::new([Space {
            name: "i",
            lo: 3,
            hi: 17,
        }]);
        let interior = Domain::new([Space {
            name: "i",
            lo: 6,
            hi: 14,
        }]);
        let mut b = Boundaries::<1>::uniform(BoundaryType::Outflow);
        b.0[0][0] = BoundaryType::CoarseFine;
        let slabs = cf_ghost_slabs(&allocated, &interior, &b);
        assert_eq!(slabs.len(), 1);
        assert_eq!((slabs[0].spaces[0].lo, slabs[0].spaces[0].hi), (3, 6));
    }

    #[test]
    fn cf_slabs_laws_2d() {
        let allocated = Domain::new([
            Space {
                name: "i",
                lo: -3,
                hi: 17,
            },
            Space {
                name: "j",
                lo: -3,
                hi: 17,
            },
        ]);
        let interior = Domain::new([
            Space {
                name: "i",
                lo: 0,
                hi: 14,
            },
            Space {
                name: "j",
                lo: 0,
                hi: 14,
            },
        ]);
        check_laws(&allocated, &interior);
    }

    #[test]
    fn cf_slabs_laws_3d() {
        let allocated = Domain::new([
            Space {
                name: "i",
                lo: -2,
                hi: 10,
            },
            Space {
                name: "j",
                lo: -2,
                hi: 10,
            },
            Space {
                name: "k",
                lo: -2,
                hi: 10,
            },
        ]);
        let interior = Domain::new([
            Space {
                name: "i",
                lo: 0,
                hi: 8,
            },
            Space {
                name: "j",
                lo: 0,
                hi: 8,
            },
            Space {
                name: "k",
                lo: 0,
                hi: 8,
            },
        ]);
        check_laws(&allocated, &interior);
    }

    #[test]
    fn cf_slabs_fully_embedded_is_the_disjoint_shell_and_strictly_smaller() {
        // every face coarse-fine: the shell is exactly allocated\interior,
        // tiled by 3^D-1 disjoint boxes, and STRICTLY fewer cells than an
        // overlapping construction (the measured ~19% win on binary_disk).
        let ng = 3isize;
        let n = 24isize;
        let allocated = Domain::new(std::array::from_fn::<_, 3, _>(|a| Space {
            name: ["i", "j", "k"][a],
            lo: -ng,
            hi: n + ng,
        }));
        let interior = Domain::new(std::array::from_fn::<_, 3, _>(|a| Space {
            name: ["i", "j", "k"][a],
            lo: 0,
            hi: n,
        }));
        let b = Boundaries::<3>::uniform(BoundaryType::CoarseFine);

        let new = cf_ghost_slabs(&allocated, &interior, &b);
        assert_disjoint(&new);
        // minimal partition: one box per coarse-fine face (2*D), fewer dispatches than
        // the 3^D-1 maximal split for the same disjoint shell.
        assert_eq!(new.len(), 2 * 3, "shell should tile into 2*D = 6 boxes");

        let new_vol: usize = new.iter().map(|d| d.volume()).sum();
        assert_eq!(new_vol, allocated.volume() - interior.volume());

        // the overlapping-slab decomposition sums to MORE cells than the disjoint shell —
        // that gap is the redundant edge/corner prolongation the disjoint shell avoids.
        let old_vol: usize = old_overlapping_slabs(&allocated, &interior, &b)
            .iter()
            .map(|d| d.volume())
            .sum();
        assert!(
            old_vol > new_vol,
            "expected overlap: old {old_vol} > new {new_vol}"
        );
        // 30^3 - 24^3 = 13176 disjoint, 6*3*30^2 = 16200 overlapping.
        assert_eq!(new_vol, 13176);
        assert_eq!(old_vol, 16200);
    }
}
