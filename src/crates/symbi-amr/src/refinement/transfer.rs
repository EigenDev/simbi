// =============================================================================
// transfer.rs
//
// the inter-level field-transfer driver: selects the
// regions and dispatches the aot amr kernels (refine_restrict_{D}d /
// refine_prolong_{order}_{D}d, built in symbi-discretize gv_refinement.rs) per field
// component through `dispatch_fields_each` — each buffer resolves the absolute
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
    /// exact degree-4 fit to the 5-cell stencil (order 4), monotonized-parabolic
    /// fallback at non-smooth cells. halfwidth 2 — the ppm evolution partner.
    Quartic,
}

impl ProlongOrder {
    /// coarse cells the 1d stencil reads per side beyond the parent.
    pub fn ghost_width(self) -> usize {
        match self {
            ProlongOrder::Pcm => 0,
            ProlongOrder::Plm => 1,
            ProlongOrder::Ppm => 2,
            ProlongOrder::Quartic => 2,
        }
    }

    /// polynomial degree of the smooth-data interpolant. the coarse-fine
    /// invariant is expressed against this: the prolongation degree must
    /// exceed the evolution reconstruction's stencil reach minus one (pcm
    /// evolution -> plm prolong, plm -> ppm, ppm -> quartic), or the ghost
    /// averages feed the fine reconstruction a lower-order error than its
    /// own interior truncation and the boundary caps the interior order.
    pub fn degree(self) -> u8 {
        match self {
            ProlongOrder::Pcm => 0,
            ProlongOrder::Plm => 1,
            ProlongOrder::Ppm => 2,
            ProlongOrder::Quartic => 4,
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
        ProlongOrder::Quartic => ProlongTag::Quartic,
    }
}

/// the coarse-fine ghost slabs of a fine level, in absolute fine indices: one
/// slab per CoarseFine face. transverse extents include the allocated ghosts
/// on sides that are themselves CoarseFine (prolongation owns those corners)
/// and clip to the interior on physical sides (the physical ghost fill owns
/// those). cf-cf corners appear in both axis slabs — the double write is the
/// same value.
pub fn cf_ghost_slabs<const D: usize>(
    allocated: &Domain<D>,
    interior: &Domain<D>,
    boundaries: &Boundaries<D>,
) -> Vec<Domain<D>> {
    // policy: which ghosts come from the coarser level. `cf_region` is the
    // interior grown out to `allocated` only on coarse-fine sides — physical
    // boundary ghosts are filled by the bc kernel, so those sides
    // stay clamped to `interior`. the cells to prolong are then exactly
    // `cf_region \ interior`.
    //
    // geometry: `guillotine_difference` (symbi-algebra) returns that set as the
    // minimal disjoint cover — `2*D` boxes, no overlap. this is union-equivalent
    // to overlapping per-face slabs (identical cell set; prolongation is a pure
    // function of (coarse state, fine coord), so the 2-3x edge/corner
    // double-writes of an overlapping cover are redundant yet still correct) but writes each cell once: the
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
    // multi-field batch: one dispatch (one rayon launch) over the whole prim set
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

/// prolong the prim batch as three axis-split sweep passes over the lerped
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
    if !sweep_eligible::<D>(ncomp, order) {
        prof("refine_prolong_1t", || {
            prolong_prims_lerped(lerp, old, new, dst, region, order, alpha)
        });
        return;
    }
    let w = order.ghost_width() as isize;

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
    sweep_from_lerped(scratch, lerp, dst, region, order);
}

/// whether the axis-split sweep kernels exist for this prim batch: 3d, plm/ppm
/// (pcm's single-load kernel cannot be beaten by three passes), ncomp 4 or 5.
fn sweep_eligible<const D: usize>(ncomp: usize, order: ProlongOrder) -> bool {
    D == 3 && (ncomp == 4 || ncomp == 5) && order != ProlongOrder::Pcm
}

/// the three sweeps of an already-lerped coarse scratch: lerp -> A -> B ->
/// dst, axis 0 innermost first (the inlined kernel's nesting order — the
/// bit-identity requirement). `scratch` must have been built with
/// `for_slab(region, order, ..)` — a shape mismatch panics.
fn sweep_from_lerped<const D: usize, const DOF: usize, Mem: MemorySpace>(
    scratch: &ProlongSweepScratch<D, DOF, Mem>,
    lerp: &PrimFieldsGeneric<D, DOF, Mem>,
    dst: &PrimFieldsGeneric<D, DOF, Mem>,
    region: &Domain<D>,
    order: ProlongOrder,
) {
    let has_pre = lerp.pre_field().is_some();
    let ncomp = 1 + DOF + has_pre as usize;
    let w = order.ghost_width() as isize;
    let (a_dom, b_dom) = sweep_domains(region, w);
    for a in 0..D {
        let (sa, ea) = (&scratch.a.rho.domain().spaces[a], &a_dom.spaces[a]);
        assert!(
            sa.lo == ea.lo && sa.hi == ea.hi,
            "prolong sweep scratch A does not match this region/order on axis {a}",
        );
    }
    let (lerp_c, a_c, b_c, dst_c) = (
        prim_comps(lerp, has_pre),
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

/// prolong the prim batch through the hydrostatic-equilibrium decomposition:
///
///   encode  — the time-lerped coarse pressure over the stencil's parent region
///             becomes its departure from the mechanical equilibrium chained, on
///             the coarse lattice, from the coarse cell under the nearest fine
///             interior cell; density and velocities are lerped unchanged.
///   prolong — the existing kernels, unchanged, act on (rho, vel.., d_pre).
///   decode  — each fine ghost rebuilds pre as the equilibrium chained, on the
///             fine lattice, from its nearest fine interior cell through the
///             prolonged fine densities, plus the prolonged departure.
///
/// the chain is the piecewise-constant-density integral of `-rho dphi` with each
/// cell's own density on its own segment (Kaeppeli & Mishra, A&A 587, A94,
/// 2016). a coarse stencil in its discrete class encodes to departures that
/// vanish identically, so at any prolongation order and any limiter the fine
/// ghosts land exactly on the fine lattice's own recursion against the interior,
/// for whatever density the prolongation hands them — the transfer's polynomial
/// bias has nothing to act on. prolonging the raw state instead deposits an
/// O(dx^2) one-signed entropy drain at the first uncovered coarse cell every
/// subcycle. the reference lives on the fine interior, so the round trip is exact
/// however restriction averages the covered coarse cells.
///
/// the time-lerp commutes with the encode (the equilibrium chain is evaluated on
/// the lerped densities), so lerping first and encoding the lerped slab once is
/// exact.
///
/// both passes are baked kernels (`wb_cf_lerp_encode` / `wb_cf_decode`), so the
/// same transfer runs on host and device memory. `scratch` is a caller-owned
/// coarse buffer covering the parent stencil region (the lerp scratch) and
/// `sweep` the slab's axis-split intermediates. the
/// decode reads the fine interior edge cell, which lies outside every slab, and
/// the fine densities, which it leaves untouched, so its in-place pressure write
/// is race-free. the mechanical equilibrium commits to no thermal structure, so
/// the transfer serves any eos.
#[allow(clippy::too_many_arguments)]
pub fn prolong_prims_balanced<const D: usize, const DOF: usize, Mem: MemorySpace>(
    sweep: &ProlongSweepScratch<D, DOF, Mem>,
    old: &PrimFieldsGeneric<D, DOF, Mem>,
    new: &PrimFieldsGeneric<D, DOF, Mem>,
    scratch: &PrimFieldsGeneric<D, DOF, Mem>,
    dst: &PrimFieldsGeneric<D, DOF, Mem>,
    region: &Domain<D>,
    fine_interior: &Domain<D>,
    order: ProlongOrder,
    alpha: f64,
    coarse_x_lo: &[f64; D],
    coarse_dx: &[f64; D],
    fine_x_lo: &[f64; D],
    fine_dx: &[f64; D],
    coarse_bodies: &symbi_ib::BodyCollection<f64, D>,
    fine_bodies: &symbi_ib::BodyCollection<f64, D>,
) {
    assert!(
        DOF == D,
        "the balance-aware coarse-fine transfer is baked for ncomp = D + 2 \
         (rho + D velocities + pre); DOF = {DOF} on a {D}d grid has no kernel"
    );
    let has_pre = old.pre_field().is_some();
    assert!(
        has_pre,
        "the balance-aware transfer requires an energy-carrying prim set"
    );
    let ncomp = 1 + DOF + 1;
    let parents = coarse_parents(region, order.ghost_width() as isize);

    // the interior the chains start from, inclusive per axis: the fine interior
    // for the decode, the coarse cells under it for the encode (euclidean
    // division — ghost slab indices are negative).
    let fine_lo: Vec<i32> = (0..D).map(|a| fine_interior.spaces[a].lo as i32).collect();
    let fine_hi: Vec<i32> = (0..D)
        .map(|a| fine_interior.spaces[a].hi as i32 - 1)
        .collect();
    let coarse_lo: Vec<i32> = fine_lo.iter().map(|&v| v.div_euclid(2)).collect();
    let coarse_hi: Vec<i32> = fine_hi.iter().map(|&v| v.div_euclid(2)).collect();
    // the kernels unroll a bounded chain; both regions must sit within it.
    let max_reach = |dom: &Domain<D>, lo: &[i32], hi: &[i32]| -> i64 {
        (0..D)
            .map(|a| {
                let (dlo, dhi) = (dom.spaces[a].lo as i64, dom.spaces[a].hi as i64 - 1);
                (lo[a] as i64 - dlo).max(dhi - hi[a] as i64).max(0)
            })
            .max()
            .unwrap_or(0)
    };
    let bound = symbi_discretize::WB_CF_CHAIN_MAX;
    assert!(
        max_reach(region, &fine_lo, &fine_hi) <= bound
            && max_reach(&parents, &coarse_lo, &coarse_hi) <= bound,
        "the coarse-fine transfer chain reaches past its unrolled bound of {bound} cells"
    );
    let ints: Vec<i32> = |lo: &[i32], hi: &[i32]| -> Vec<i32> {
        lo.iter().chain(hi.iter()).copied().collect()
    }(&coarse_lo, &coarse_hi);

    // fused lerp + encode into the coarse scratch: every component time-lerped,
    // pre written as its departure from the coarse chain.
    let (old_c, new_c, scratch_c) = (
        prim_comps(old, has_pre),
        prim_comps(new, has_pre),
        prim_comps(scratch, has_pre),
    );
    let mut inputs: Vec<&symbi_grid::Field<f64, D, Mem>> = Vec::with_capacity(2 * ncomp);
    for k in 0..ncomp {
        inputs.push(old_c[k]);
        inputs.push(new_c[k]);
    }
    let mut scalars = vec![alpha];
    scalars.extend_from_slice(coarse_x_lo);
    scalars.extend_from_slice(coarse_dx);
    push_body_slot_scalars(coarse_bodies, &mut scalars);
    prof("wb_cf_encode", || {
        dispatch_fields_each::<f64, Mem, D>(
            KernelId::WbCfLerpEncode { ndim: D as u8 }.name(),
            &parents,
            &inputs,
            &scratch_c,
            &ints,
            &scalars,
        );
    });

    // prolong the departures with the unchanged kernels: the axis-split sweeps
    // where they exist (3d plm/ppm — bit-identical to the fused kernel at a
    // fraction of its interpolation work), the fused kernel otherwise with the
    // scratch standing as both time endpoints. the lerp already happened in the
    // encode.
    if sweep_eligible::<D>(ncomp, order) {
        sweep_from_lerped(sweep, scratch, dst, region, order);
    } else {
        prolong_prims(scratch, scratch, dst, region, order, 0.0);
    }

    // decode: fine ghost pre += the fine chain from the nearest interior cell.
    let pre = "the balance-aware transfer requires an energy-carrying prim set";
    let b_inputs = [&dst.rho];
    let b_outputs = [dst.pre_field().expect(pre)];
    let fine_ints: Vec<i32> = fine_lo.iter().chain(fine_hi.iter()).copied().collect();
    let mut scalars = Vec::new();
    scalars.extend_from_slice(fine_x_lo);
    scalars.extend_from_slice(fine_dx);
    push_body_slot_scalars(fine_bodies, &mut scalars);
    let departures = census_snapshot(dst.pre_field().expect(pre), region);
    prof("wb_cf_decode", || {
        dispatch_fields_each::<f64, Mem, D>(
            KernelId::WbCfDecode { ndim: D as u8 }.name(),
            region,
            &b_inputs,
            &b_outputs,
            &fine_ints,
            &scalars,
        );
    });
    census_decoded_pressure("cf ghost decode", departures, dst.pre_field().expect(pre), region);
    census_positive_field("cf ghost rho", &dst.rho, region);
}


/// admissibility census over a balanced decode, from a snapshot of the
/// departure field the decode consumes in place. the pressure slot holds the
/// transported departures immediately before the decode and the guarded
/// pressure immediately after, so the pair reconstructs the decode per cell:
/// where the guard passed, `p_eq = after - dep` and the margin `|dep| / p_eq`
/// measures how far the cell sat from the admissibility boundary; where the
/// guard abstained onto the equilibrium, `after = p_eq` and the rejected value
/// is `dep + after <= 0`. an abstention onto the anchor (the equilibrium
/// itself inadmissible) evades this reconstruction, so the census reports
/// every first-level abstention and the anchor tier separately through the
/// non-finite scan. reads through host-visible views; enabled by
/// SYMBI_WB_CENSUS=1.
pub static WB_DECODE_BAD_CELLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static WB_DECODE_ABSTAIN_CELLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static WB_DECODE_TOTAL_CELLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

fn wb_census_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SYMBI_WB_CENSUS").is_ok_and(|v| v == "1"))
}

/// positivity and finiteness census over a raw-transferred field: the
/// prolongation passes density through unencoded, and a non-positive or
/// non-finite ghost density poisons the sound speed exactly as an inadmissible
/// pressure does. enabled by SYMBI_WB_CENSUS=1.
fn census_positive_field<const D: usize, Mem: MemorySpace>(
    site: &str,
    field: &symbi_grid::Field<f64, D, Mem>,
    dom: &Domain<D>,
) {
    if !wb_census_enabled() {
        return;
    }
    let view = field.view();
    for coord in dom.iter() {
        let v = *view.at(coord);
        if !(v > 0.0) || !v.is_finite() {
            let n = WB_DECODE_BAD_CELLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 32 {
                eprintln!("[wb-census] {site}: inadmissible value {v:.6e} at {coord:?}");
            }
        }
    }
}

/// admissibility census over the band decode with the conservative fallback:
/// the pressure field enters holding the conservative restriction and leaves
/// holding the decoded value where it was admissible and the conservative
/// value where it was not, so an abstention reads as an output equal to the
/// input in a cell carrying a nonzero departure. margins come from the passing
/// cells, where `p_eq = after - dep`. enabled by SYMBI_WB_CENSUS=1.
fn census_band_decode<const D: usize, Mem: MemorySpace>(
    site: &str,
    cons_before: Option<Vec<f64>>,
    departures: &symbi_grid::Field<f64, D, Mem>,
    pre: &symbi_grid::Field<f64, D, Mem>,
    dom: &Domain<D>,
) {
    let Some(cons_before) = cons_before else {
        return;
    };
    let dep_view = departures.view();
    let view = pre.view();
    let mut abstained = 0u64;
    let mut cells = 0u64;
    let mut max_ratio = 0.0_f64;
    let mut first_abstain = None;
    for (coord, cons) in dom.iter().zip(cons_before) {
        cells += 1;
        let p = *view.at(coord);
        let dep = *dep_view.at(coord);
        if !(p > 0.0) || !p.is_finite() {
            let n = WB_DECODE_BAD_CELLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 32 {
                eprintln!("[wb-census] {site}: inadmissible written pressure {p:.6e} at {coord:?}");
            }
            continue;
        }
        if p == cons && dep != 0.0 {
            abstained += 1;
            if first_abstain.is_none() {
                first_abstain = Some((coord, dep, cons));
            }
        } else {
            let p_eq = p - dep;
            if p_eq > 0.0 {
                max_ratio = max_ratio.max((dep / p_eq).abs());
            }
        }
    }
    WB_DECODE_TOTAL_CELLS.fetch_add(cells, std::sync::atomic::Ordering::Relaxed);
    if abstained > 0 {
        WB_DECODE_ABSTAIN_CELLS.fetch_add(abstained, std::sync::atomic::Ordering::Relaxed);
        let (coord, dep, cons) = first_abstain.unwrap();
        eprintln!(
            "[wb-abstain] {site}: {abstained}/{cells} cells kept the conservative pressure; \
             first at {coord:?} held {cons:.4e} against dep {dep:.4e}; \
             passing-cell max |dep|/p_eq {max_ratio:.3e}"
        );
    }
}

/// the departure field over `dom`, copied out before the decode overwrites it.
fn census_snapshot<const D: usize, Mem: MemorySpace>(
    pre: &symbi_grid::Field<f64, D, Mem>,
    dom: &Domain<D>,
) -> Option<Vec<f64>> {
    if !wb_census_enabled() {
        return None;
    }
    let view = pre.view();
    Some(dom.iter().map(|coord| *view.at(coord)).collect())
}

fn census_decoded_pressure<const D: usize, Mem: MemorySpace>(
    site: &str,
    departures: Option<Vec<f64>>,
    pre: &symbi_grid::Field<f64, D, Mem>,
    dom: &Domain<D>,
) {
    let Some(departures) = departures else {
        return;
    };
    let view = pre.view();
    let mut abstained = 0u64;
    let mut cells = 0u64;
    let mut max_ratio = 0.0_f64;
    let mut first_abstain = None;
    for (coord, dep) in dom.iter().zip(departures) {
        cells += 1;
        let p = *view.at(coord);
        if !(p > 0.0) || !p.is_finite() {
            let n = WB_DECODE_BAD_CELLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 16 {
                eprintln!(
                    "[wb-census] {site}: inadmissible written pressure {p:.6e} at {coord:?}"
                );
            }
            continue;
        }
        if dep + p <= 0.0 {
            abstained += 1;
            if first_abstain.is_none() {
                first_abstain = Some((coord, dep, p));
            }
        } else {
            let p_eq = p - dep;
            if p_eq > 0.0 {
                max_ratio = max_ratio.max((dep / p_eq).abs());
            }
        }
    }
    WB_DECODE_TOTAL_CELLS.fetch_add(cells, std::sync::atomic::Ordering::Relaxed);
    if abstained > 0 {
        WB_DECODE_ABSTAIN_CELLS.fetch_add(abstained, std::sync::atomic::Ordering::Relaxed);
        let (coord, dep, p_eq) = first_abstain.unwrap();
        eprintln!(
            "[wb-abstain] {site}: {abstained}/{cells} cells took the equilibrium fallback; \
             first at {coord:?} rejected {:.4e} (dep {dep:.4e} on p_eq {p_eq:.4e}); \
             passing-cell max |dep|/p_eq {max_ratio:.3e}",
            dep + p_eq
        );
    } else if max_ratio > 0.25 {
        eprintln!(
            "[wb-margin] {site}: {cells} cells all passed; max |dep|/p_eq {max_ratio:.3e}"
        );
    }
}

/// the balanced restriction of a coarse seam band: after the conservative
/// restriction and the coarse c2p, rewrite the pressure of the covered coarse
/// cells within `band` cells of each coarse-fine seam so that they sit on the
/// coarse mechanical chain from the uncovered cell beyond the seam, carrying the
/// fine solution's departure from the composite-lattice equilibrium:
///
///   encode   — each fine cell under the band writes its pressure's departure
///              from the equilibrium chained from the uncovered coarse cell,
///              across the seam face, through the fine densities (`WbBandEncode`
///              into the fine `departure` scratch).
///   restrict — the departures average into the coarse band (`RefineRestrict`
///              into the coarse pressure field, which the decode reads back).
///   decode   — each band cell adds the coarse chain from the uncovered cell
///              through the restricted coarse densities (`WbCfDecode`).
///   energy   — the band's conserved energy follows the rewritten pressure
///              under the gamma law (`BandEnergy`).
///
/// conservative averaging alone leaves every covered cell below its class
/// pressure by the jensen gap `rho phi'' h^2 / 8` while the uncovered neighbor
/// sits on the class, a standing pressure step the coarse stage fluxes kick
/// every step. the band is the only covered data the uncovered stencils read
/// (the evolution reach), so the deeper covered cells keep the conservative
/// average; inside the band the energy differs from the fine average by that
/// O(h^2) gap. the departures are measured against the uncovered cell continued
/// across the seam, so a column balanced across the seam encodes to zero and a
/// wave standing at the seam encodes to its full amplitude.
#[allow(clippy::too_many_arguments)]
pub fn restrict_band_balanced<const D: usize, const DOF: usize, Mem: MemorySpace>(
    fine: &PrimFieldsGeneric<D, DOF, Mem>,
    fine_interior: &Domain<D>,
    departure: &symbi_grid::Field<f64, D, Mem>,
    coarse_departure: &symbi_grid::Field<f64, D, Mem>,
    coarse_prim: &PrimFieldsGeneric<D, DOF, Mem>,
    coarse_nrg: &symbi_grid::Field<f64, D, Mem>,
    coarse_interior: &Domain<D>,
    coverage: &Domain<D>,
    band: usize,
    gamma: f64,
    fine_x_lo: &[f64; D],
    fine_dx: &[f64; D],
    coarse_x_lo: &[f64; D],
    coarse_dx: &[f64; D],
    bodies: &symbi_ib::BodyCollection<f64, D>,
) {
    assert!(
        DOF == D,
        "the balanced restriction is baked for ncomp = D + 2; DOF = {DOF} on a {D}d grid has no kernel"
    );
    let pre = "the balanced restriction requires an energy-carrying prim set";
    let fine_pre = fine.pre_field().expect(pre);
    let coarse_pre = coarse_prim.pre_field().expect(pre);
    let band = band as isize;
    assert!(
        band as i64 <= symbi_discretize::WB_CF_CHAIN_MAX
            && (2 * band - 1) as i64 <= symbi_discretize::WB_BAND_CHAIN_MAX,
        "the balanced restriction band of {band} cells reaches past the unrolled chain bounds"
    );

    let mut body_scalars = Vec::new();
    push_body_slot_scalars(bodies, &mut body_scalars);
    let fine_lo: Vec<i32> = (0..D).map(|a| fine_interior.spaces[a].lo as i32).collect();
    let fine_hi: Vec<i32> = (0..D)
        .map(|a| fine_interior.spaces[a].hi as i32 - 1)
        .collect();
    let coarse_lo: Vec<i32> = (0..D).map(|a| coarse_interior.spaces[a].lo as i32).collect();
    let coarse_hi: Vec<i32> = (0..D)
        .map(|a| coarse_interior.spaces[a].hi as i32 - 1)
        .collect();

    // one band per coarse-fine seam face: the coverage edge on axis `ax`, low or
    // high side, where an uncovered coarse cell lies beyond it inside the coarse
    // interior. a coverage edge flush with the coarse interior is a physical
    // boundary of the whole hierarchy and carries no seam.
    for ax in 0..D {
        for high in [false, true] {
            let (c_lo, c_hi) = (coverage.spaces[ax].lo, coverage.spaces[ax].hi);
            let uncovered = if high { c_hi } else { c_lo - 1 };
            if uncovered < coarse_interior.spaces[ax].lo || uncovered >= coarse_interior.spaces[ax].hi {
                continue;
            }
            // the coarse band along `ax`, full coverage extent elsewhere, and the
            // fine cells under it.
            let (b_lo, b_hi) = if high {
                ((c_hi - band).max(c_lo), c_hi)
            } else {
                (c_lo, (c_lo + band).min(c_hi))
            };
            let mut band_dom = coverage.clone();
            band_dom.spaces[ax].lo = b_lo;
            band_dom.spaces[ax].hi = b_hi;
            let mut fine_band = Domain::new(std::array::from_fn(|a| Space {
                name: coverage.spaces[a].name,
                lo: 2 * coverage.spaces[a].lo,
                hi: 2 * coverage.spaces[a].hi,
            }));
            fine_band.spaces[ax].lo = 2 * b_lo;
            fine_band.spaces[ax].hi = 2 * b_hi;
            // the regions a dispatch covers must lie inside the lattices they index,
            // and the chain's reference must lie outside the region the decode writes:
            // every band cell runs concurrently on a device, so a reference inside the
            // written set would read pressures its neighbours are overwriting.
            for a in 0..D {
                assert!(
                    band_dom.spaces[a].lo >= coarse_interior.spaces[a].lo
                        && band_dom.spaces[a].hi <= coarse_interior.spaces[a].hi,
                    "band {:?} leaves the coarse interior {:?} on axis {a}",
                    band_dom.spaces, coarse_interior.spaces
                );
                assert!(
                    fine_band.spaces[a].lo >= fine_interior.spaces[a].lo
                        && fine_band.spaces[a].hi <= fine_interior.spaces[a].hi,
                    "fine band {:?} leaves the fine interior {:?} on axis {a}",
                    fine_band.spaces, fine_interior.spaces
                );
            }
            assert!(
                uncovered < band_dom.spaces[ax].lo || uncovered >= band_dom.spaces[ax].hi,
                "the decode's reference cell {uncovered} lies inside the band it writes, \
                 {:?}: concurrent threads would read pressures being overwritten",
                band_dom.spaces[ax]
            );

            // the fine edge cell and the seam face along `ax`.
            let edge = if high { 2 * c_hi - 1 } else { 2 * c_lo };
            let face = fine_x_lo[ax] + (if high { 2 * c_hi } else { 2 * c_lo }) as f64 * fine_dx[ax];

            // encode: fine departures from the uncovered cell continued across the seam.
            let mut ints: Vec<i32> = Vec::new();
            let mut lo = fine_lo.clone();
            let mut hi = fine_hi.clone();
            lo[ax] = edge as i32;
            hi[ax] = edge as i32;
            ints.extend(lo.iter());
            ints.extend(hi.iter());
            ints.push(uncovered as i32);
            let mut scalars = vec![face];
            scalars.extend_from_slice(fine_x_lo);
            scalars.extend_from_slice(fine_dx);
            scalars.extend_from_slice(coarse_x_lo);
            scalars.extend_from_slice(coarse_dx);
            scalars.extend_from_slice(&body_scalars);
            prof("wb_band_encode", || {
                dispatch_fields_each::<f64, Mem, D>(
                    KernelId::WbBandEncode {
                        ndim: D as u8,
                        axis: ax as u8,
                    }
                    .name(),
                    &fine_band,
                    &[&fine.rho, fine_pre, &coarse_prim.rho, coarse_pre],
                    &[departure],
                    &ints,
                    &scalars,
                );
            });

            // restrict the departures into the coarse-side scratch; the pressure
            // field keeps the conservative restriction, which the decode reads
            // per cell as its abstention value.
            prof("wb_band_restrict", || {
                restrict_cell_field(departure, coarse_departure, &band_dom)
            });

            // decode: the coarse chain from the uncovered cell through the band.
            let mut lo = coarse_lo.clone();
            let mut hi = coarse_hi.clone();
            lo[ax] = uncovered as i32;
            hi[ax] = uncovered as i32;
            let ints: Vec<i32> = lo.iter().chain(hi.iter()).copied().collect();
            let mut scalars = Vec::new();
            scalars.extend_from_slice(coarse_x_lo);
            scalars.extend_from_slice(coarse_dx);
            scalars.extend_from_slice(&body_scalars);
            let cons_before = census_snapshot(coarse_pre, &band_dom);
            prof("wb_band_decode", || {
                dispatch_fields_each::<f64, Mem, D>(
                    KernelId::WbBandDecode { ndim: D as u8 }.name(),
                    &band_dom,
                    &[&coarse_prim.rho, coarse_departure],
                    &[coarse_pre],
                    &ints,
                    &scalars,
                );
            });
            census_band_decode("band decode", cons_before, coarse_departure, coarse_pre, &band_dom);

            // the band's conserved energy follows its rewritten pressure.
            let mut inputs: Vec<&symbi_grid::Field<f64, D, Mem>> = vec![&coarse_prim.rho];
            for k in 0..D {
                inputs.push(&coarse_prim.vel[k]);
            }
            inputs.push(coarse_pre);
            prof("wb_band_energy", || {
                dispatch_fields_each::<f64, Mem, D>(
                    KernelId::BandEnergy { ndim: D as u8 }.name(),
                    &band_dom,
                    &inputs,
                    &[coarse_nrg],
                    &[],
                    &[gamma],
                );
            });
        }
    }
}

/// pack the balance-aware transfer's per-slot body scalars in the kernels'
/// declared order: `pos[0..D], mass, soft, softkind` per slot, MAX_SOURCE_BODIES
/// slots. an absent or non-gravitating slot carries mass = 0, which zeroes its
/// potential identically (soft = 1 keeps the softened form regular at the slot's
/// zero position).
fn push_body_slot_scalars<const D: usize>(
    bodies: &symbi_ib::BodyCollection<f64, D>,
    scalars: &mut Vec<f64>,
) {
    for b in 0..symbi_ib::collection::MAX_SOURCE_BODIES {
        if b < bodies.len() {
            let body = bodies.get(b);
            for ax in 0..D {
                scalars.push(body.position[ax]);
            }
            scalars.push(if body.has_gravity() { body.mass } else { 0.0 });
            scalars.push(body.softening().unwrap_or(1.0));
            scalars.push(body.softening_kind().unwrap_or(0.0));
        } else {
            for _ in 0..D {
                scalars.push(0.0);
            }
            scalars.push(0.0);
            scalars.push(1.0);
            scalars.push(0.0);
        }
    }
}

/// prolong the prim batch through a pre-lerped coarse scratch: one `field_lerp`
/// pass time-interpolates the coarse snapshots once per coarse cell over the
/// parent region of `region` (+ stencil halo), then the single-snapshot prolong
/// reads the lerped buffer — half the gather traffic of the fused time-pair
/// kernel (which re-lerps the whole stencil neighborhood per fine cell), with
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
    // the conserved dye `D_chi = rho chi` is a volume-extensive density like the others, so the
    // same volume-weighted average carries it. omitting it would leave the covered coarse cells
    // holding the dye they had before the fine level ran.
    if let (Some(fchi), Some(cchi)) = (fine.chi_field(), coarse.chi_field()) {
        restrict_cell_field(fchi, cchi, coverage);
    }
}

/// the coarse-fine halo slabs of the staggered face field bface[d] in absolute
/// fine face indices: one single-row slab per CF transverse side (the +/-1
/// transverse halo the flux sweep's Gardiner-Stone override reads). spans the
/// full owned face extent on the normal axis; the other transverse axis
/// extends into its halo where that side is also CF (corner rows) and clips
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
/// `2^(D-1)` coincident fine faces) over the coverage face domain — the
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

    // ─── reference construction ──────────────────────────────────────────────
    // the overlapping per-face slab construction. the disjoint `cf_ghost_slabs`
    // must cover the exact same cell set — the union-equivalence contract that
    // makes the disjoint form numerically indistinguishable.
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

    // law: a partition is disjoint — no two boxes share a cell.
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

    // law: disjointness + union-equivalence to the reference construction, over a
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

            // union-equivalence: the disjoint partition covers the same cells
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
        // tiled by 3^D-1 disjoint boxes, and strictly fewer cells than an
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

        // the overlapping-slab decomposition sums to more cells than the disjoint shell —
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
