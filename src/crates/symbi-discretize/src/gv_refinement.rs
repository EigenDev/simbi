// =============================================================================
// gv_refinement.rs
//
// the amr field-transfer kernels: restriction
// (fine -> coarse conservative child average) and prolongation (coarse -> fine
// limited interpolation, time-interpolated between two coarse snapshots) as
// gv-traced pullbacks over the refinement lattice maps (lattice.rs
// Refine / Coarsen). levels share absolute index space: fine cell f covers
// coarse cell floor_div(f, ratio), so every index stays level-global — the
// destination thread coordinate is the level-global index, and each field
// buffer resolves it against its own lo.
//
// the straightforward formulation of this transfer (symbi-amr prolong_nd /
// restrict_nd) is an axis-by-axis sweep over scratch buffers — host-only by
// construction. here the
// sweep is inlined per destination cell: pass order is axis 0 innermost, the
// per-pass 1d operators (pcm / van-leer plm / monotonized ppm sub-cell average)
// use the identical arithmetic, so the traced expression per output cell is
// bit-identical to the reference sweep at f64. limiters are carrier-generic in
// the cmp/select dialect; both select arms are NaN-free (the van leer
// denominator is guarded before the select).
//
// usage:
//  let (k, w) = refine_restrict_gv(ndim, 2);
//  let (k, w) = refine_prolong_gv(ndim, 2, ProlongOrder::Ppm);
// =============================================================================

use symbi_algebra::algebra::Numeric;
use symbi_carrier::Scalar;
use symbi_ir::graph::{ConstValue, ElementWiseOp, NodeId};
use symbi_ir::{Gv, GvKernel, KernelWrite, KernelWrites, TraceCx, trace};

use super::gv::gv_load_at;

/// prolongation order for the coarse-fine transfer. one order higher than the
/// evolution reconstruction (pcm evolution -> plm prolong, plm -> ppm) so the
/// coarse-fine boundary preserves the interior order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProlongOrder {
    /// piecewise constant (order 0): parent injection. stencil halfwidth 0.
    Pcm,
    /// piecewise linear with van leer limiting (order 1). stencil halfwidth 1.
    Plm,
    /// piecewise parabolic with monotonicity (order 2), exact sub-cell averages
    /// (conservative by construction). stencil halfwidth 2.
    Ppm,
    /// the exact degree-4 polynomial matching all five stencil cell averages
    /// (order 4, conservative by construction — its cell-0 integral is the
    /// parent average), with the monotonized parabolic form as the fallback
    /// wherever the undivided second differences change sign across the
    /// stencil (a discontinuity; the raw quartic overshoots there). stencil
    /// halfwidth 2, identical to ppm. serves an evolution reconstruction of
    /// parabolic order: the coarse-fine ghost averages are O(h^5), so the
    /// interface layer's flux-divergence order loss lands at O(h^4) locally
    /// and the boundary preserves the interior order.
    Quartic,
}

impl ProlongOrder {
    /// coarse ghost cells required per side by the 1d stencil.
    pub fn ghost_width(self) -> usize {
        match self {
            ProlongOrder::Pcm => 0,
            ProlongOrder::Plm => 1,
            ProlongOrder::Ppm => 2,
            ProlongOrder::Quartic => 2,
        }
    }
}

// =============================================================================
// carrier-generic stencil math (identical arithmetic to the host sweep,
// branches rewritten as cmp/select — same value at f64, traceable at Gv)
// =============================================================================

/// van leer (harmonic) limited slope: `2*dl*dr/(dl+dr)` when the one-sided
/// differences share a strict sign, zero otherwise. the denominator is guarded
/// before the select (Gv evaluates both arms; a same-signed pair has a nonzero
/// sum, so the guard only replaces the discarded arm).
fn van_leer<S: Scalar>(dl: S, dr: S) -> S {
    let prod = dl * dr;
    let pos = prod.cmp_gt(S::ZERO);
    let denom = S::select(pos, dl + dr, S::ONE);
    let two = S::ONE + S::ONE;
    S::select(pos, two * prod / denom, S::ZERO)
}

/// the 1d plm prolongation sample: the parent's limited linear profile
/// evaluated at the sub-cell position `frac` in [-1/2, 1/2).
fn plm_interp<S: Scalar>(vm: S, vc: S, vp: S, frac: S) -> S {
    vc + van_leer(vc - vm, vp - vc) * frac
}

/// the 1d ppm prolongation sub-cell average over [xi_lo, xi_hi] (xi in [0,1]
/// across the parent): 4th-order interface values clamped to the neighbor
/// range, monotonized (the select form preserves the reference's sequential
/// left-then-right overshoot correction), then the exact parabola
/// antiderivative difference times `ratio` — so the children average back to
/// the parent exactly (conservation by construction).
fn ppm_interp<S: Scalar>(vm2: S, vm1: S, vc: S, vp1: S, vp2: S, xi_lo: S, xi_hi: S, ratio: S) -> S {
    let (a_l, a_r) = super::gv::ppm_cell_interfaces(vm2, vm1, vc, vp1, vp2);
    let two = S::ONE + S::ONE;
    let half = S::ONE / two;
    let three = S::THREE;
    let six = S::from_f64(6.0);

    // parabola u(xi) = a_l + xi*(a_r - a_l + (1-xi)*u6); the sub-cell average
    // is the antiderivative difference scaled by the refinement ratio.
    let u6 = six * (vc - half * (a_l + a_r));
    let c1 = a_l;
    let c2 = (a_r - a_l + u6) * half;
    let c3 = u6 / three;
    let a_hi = xi_hi * (c1 + xi_hi * (c2 - xi_hi * c3));
    let a_lo = xi_lo * (c1 + xi_lo * (c2 - xi_lo * c3));
    (a_hi - a_lo) * ratio
}

/// the 1d quartic prolongation sub-cell average over [xi_lo, xi_hi] (xi in [0,1]
/// across the parent): the unique degree-4 polynomial whose cell averages match
/// all five stencil values (the derivative of the degree-5 interpolant of the
/// primitive function), integrated exactly — so the children average back to the
/// parent exactly and the sub-cell averages are O(h^5) on smooth data, including
/// at extrema, where the raw polynomial stands unclamped. wherever the undivided
/// second differences at cells -1, 0, +1 differ in sign — a discontinuity inside the
/// stencil, where the raw quartic rings — the value falls back to the
/// monotonized parabolic average, so a shock crossing a refinement boundary sees
/// the same bounded transfer as the ppm order.
fn quartic_interp<S: Scalar>(
    vm2: S,
    vm1: S,
    vc: S,
    vp1: S,
    vp2: S,
    xi_lo: S,
    xi_hi: S,
    ratio: S,
) -> S {
    let two = S::ONE + S::ONE;
    // p(xi) = c0 + c1 xi + c2 xi^2 + c3 xi^3 + c4 xi^4 on xi in [0, 1]:
    // int_0^1 p = vc exactly; the other four averages integrate to zero weight.
    let c0 = S::from_f64(-1.0 / 20.0) * vm2
        + S::from_f64(9.0 / 20.0) * vm1
        + S::from_f64(47.0 / 60.0) * vc
        + S::from_f64(-13.0 / 60.0) * vp1
        + S::from_f64(1.0 / 30.0) * vp2;
    let c1 = S::from_f64(1.0 / 12.0) * vm2
        + S::from_f64(-5.0 / 4.0) * vm1
        + S::from_f64(5.0 / 4.0) * vc
        + S::from_f64(-1.0 / 12.0) * vp1;
    let c2 = S::from_f64(1.0 / 8.0) * vm2 + S::from_f64(1.0 / 4.0) * vm1 - vc
        + S::from_f64(3.0 / 4.0) * vp1
        + S::from_f64(-1.0 / 8.0) * vp2;
    let c3 = S::from_f64(-1.0 / 6.0) * vm2
        + S::from_f64(1.0 / 2.0) * vm1
        + S::from_f64(-1.0 / 2.0) * vc
        + S::from_f64(1.0 / 6.0) * vp1;
    let c4 = S::from_f64(1.0 / 24.0) * vm2
        + S::from_f64(-1.0 / 6.0) * vm1
        + S::from_f64(1.0 / 4.0) * vc
        + S::from_f64(-1.0 / 6.0) * vp1
        + S::from_f64(1.0 / 24.0) * vp2;
    // antiderivative difference over the sub-cell, scaled to an average.
    let anti = |x: S| {
        x * (c0
            + x * (c1 / two + x * (c2 / S::THREE + x * (c3 / S::FOUR + x * c4 / S::from_f64(5.0)))))
    };
    let quartic = (anti(xi_hi) - anti(xi_lo)) * ratio;

    let d2_l = vm2 - two * vm1 + vc;
    let d2_c = vm1 - two * vc + vp1;
    let d2_r = vc - two * vp1 + vp2;
    let d2_min = d2_l.min(d2_c).min(d2_r);
    let d2_max = d2_l.max(d2_c).max(d2_r);
    let limited = ppm_interp(vm2, vm1, vc, vp1, vp2, xi_lo, xi_hi, ratio);
    S::select(
        d2_min.cmp_gt(S::ZERO),
        quartic,
        S::select(d2_max.cmp_lt(S::ZERO), quartic, limited),
    )
}

// =============================================================================
// restriction: thread over the coarse coverage, read the ratio^D fine children
// =============================================================================

/// trace the fine -> coarse restriction: `dst[c] = sweep-average of
/// src[ratio*c + o]` over the child offsets, axis 0 innermost (the reference
/// restrict_nd pass order). input "src" is the fine field, output "dst" the
/// coarse field; the dispatch domain is the coarse coverage in absolute coarse
/// indices.
pub fn refine_restrict_gv(ndim: usize, ratio: i64) -> (GvKernel, KernelWrites) {
    assert!(
        (1..=3).contains(&ndim),
        "refine_restrict_gv: ndim must be 1..=3"
    );
    assert!(ratio >= 2, "refine_restrict_gv: ratio must be >= 2");
    trace(|cx| {
    // the per-axis Refine map base: source[ax] = coord[ax] * ratio (+ child offset).
    let scaled: Vec<NodeId> = cx.with_trace(|t| {
        let coords: Vec<NodeId> = (0..ndim).map(|ax| t.coord(ax as u8)).collect();
        let g = t.graph();
        coords
            .into_iter()
            .map(|c| {
                let r = g.add_const(ConstValue::I32(ratio as i32), None);
                g.element_wise(ElementWiseOp::Mul, vec![c, r], None)
            })
            .collect()
    });
    let inv_r = Gv::from_f64(1.0 / ratio as f64);
    let val = restrict_eval(cx, &scaled, ratio, inv_r, ndim as isize - 1, &mut [0; 3]);
    let writes = vec![KernelWrite::new("dst", "dst", val.node())];
    writes
    })
}

/// the inlined restrict sweep: axis `ax` averages `ratio` child values, each
/// recursively restricted through the lower axes (axis 0 innermost = the
/// reference's first pass). leaf = the fine child read at `scaled + off`.
fn restrict_eval<'t>(
    cx: TraceCx<'t>,
    scaled: &[NodeId],
    ratio: i64,
    inv_r: Gv<'t>,
    ax: isize,
    off: &mut [i64; 3],
) -> Gv<'t> {
    if ax < 0 {
        let coords: Vec<NodeId> = cx.with_trace(|t| {
            let g = t.graph();
            scaled
                .iter()
                .enumerate()
                .map(|(kk, &s)| {
                    let o = g.add_const(ConstValue::I32(off[kk] as i32), None);
                    g.element_wise(ElementWiseOp::Add, vec![s, o], None)
                })
                .collect()
        });
        return gv_load_at(cx, "src", "src", &coords);
    }
    let aa = ax as usize;
    off[aa] = 0;
    let mut sum = restrict_eval(cx, scaled, ratio, inv_r, ax - 1, off);
    for kk in 1..ratio {
        off[aa] = kk;
        sum = sum + restrict_eval(cx, scaled, ratio, inv_r, ax - 1, off);
    }
    off[aa] = 0;
    sum * inv_r
}

// =============================================================================
// register / snapshot utility kernels (the amr gpu pass): the flux-register
// accumulate/apply/zero and the *_old snapshots as substrate kernels, so the
// berger-oliger driver touches no field from the host on a device backend.
// =============================================================================

/// dst = src, pointwise — the device field snapshot (prim_old / bcell_old /
/// bface_old). each buffer resolves the thread coord against its own lo, so
/// staggered fields copy with the same kernel.
pub fn field_copy_gv(ndim: usize) -> (GvKernel, KernelWrites) {
    assert!((1..=3).contains(&ndim), "field_copy_gv: ndim must be 1..=3");
    trace(|cx| {
    let v = cx.field("src", "src");
    let _ = ndim;
    let writes = vec![KernelWrite::new("dst", "dst", v.node())];
    writes
    })
}

/// dst = value (a runtime scalar), pointwise — the register zero.
pub fn field_fill_gv(ndim: usize) -> (GvKernel, KernelWrites) {
    assert!((1..=3).contains(&ndim), "field_fill_gv: ndim must be 1..=3");
    trace(|cx| {
    let v = cx.scalar("value");
    let _ = ndim;
    let writes = vec![KernelWrite::new("dst", "dst", v.node())];
    writes
    })
}

/// dst(c) += scale * src(c + arg), with per-axis runtime integer offsets (the
/// lattice-style args) and a runtime scalar `scale`. serves the register's
/// coarse-flux accumulation (arg = 0, scale = -A*w) and the reflux apply
/// (src = the register face read at the cell's adjacent face, scale = sign/V).
pub fn field_axpy_shift_gv(ndim: usize) -> (GvKernel, KernelWrites) {
    assert!(
        (1..=3).contains(&ndim),
        "field_axpy_shift_gv: ndim must be 1..=3"
    );
    trace(|cx| {
    let src_coords: Vec<NodeId> = cx.with_trace(|t| {
        let coords: Vec<NodeId> = (0..ndim).map(|ax| t.coord(ax as u8)).collect();
        let args: Vec<NodeId> = (0..ndim)
            .map(|ax| t.scalar_int(&format!("arg_{ax}")))
            .collect();
        let g = t.graph();
        coords
            .into_iter()
            .zip(args)
            .map(|(c, a)| g.element_wise(ElementWiseOp::Add, vec![c, a], None))
            .collect()
    });
    let scale = cx.scalar("scale");
    let dst = cx.field("dst", "dst");
    let src = gv_load_at(cx, "src", "src", &src_coords);
    let v = dst + scale * src;
    let writes = vec![KernelWrite::new("dst", "dst", v.node())];
    writes
    })
}

/// dst(c) += scale * sum over the `ratio^(D-1)` fine faces coincident with
/// coarse face c (transverse child offsets, the normal index scaling exactly)
/// — the register's fine-flux accumulation. `scale` carries the fine face
/// area times the stage weight.
pub fn refine_acc_face_gv(ndim: usize, ratio: i64, axis: usize) -> (GvKernel, KernelWrites) {
    assert!(
        (1..=3).contains(&ndim),
        "refine_acc_face_gv: ndim must be 1..=3"
    );
    assert!(
        axis < ndim,
        "refine_acc_face_gv: axis {axis} out of range for ndim {ndim}"
    );
    assert!(ratio >= 2, "refine_acc_face_gv: ratio must be >= 2");
    trace(|cx| {
    let scaled: Vec<NodeId> = cx.with_trace(|t| {
        let coords: Vec<NodeId> = (0..ndim).map(|ax| t.coord(ax as u8)).collect();
        let g = t.graph();
        coords
            .into_iter()
            .map(|c| {
                let r = g.add_const(ConstValue::I32(ratio as i32), None);
                g.element_wise(ElementWiseOp::Mul, vec![c, r], None)
            })
            .collect()
    });
    let scale = cx.scalar("scale");
    let dst = cx.field("dst", "dst");
    let sum = acc_face_sum(cx, &scaled, ratio, axis, ndim as isize - 1, &mut [0; 3]);
    let v = dst + scale * sum;
    let writes = vec![KernelWrite::new("dst", "dst", v.node())];
    writes
    })
}

/// dst(g) += scale * sum over the `ratio` fine sub-edges of coarse edge g
/// (child offsets along the edge axis only; every other index scales exactly)
/// — the emf register's fine accumulation. `scale` carries the fine dt times
/// the length-average factor 1/ratio.
pub fn refine_acc_edge_gv(ndim: usize, ratio: i64, axis: usize) -> (GvKernel, KernelWrites) {
    assert!(
        (1..=3).contains(&ndim),
        "refine_acc_edge_gv: ndim must be 1..=3"
    );
    assert!(
        axis < ndim,
        "refine_acc_edge_gv: axis {axis} out of range for ndim {ndim}"
    );
    assert!(ratio >= 2, "refine_acc_edge_gv: ratio must be >= 2");
    trace(|cx| {
    let scaled: Vec<NodeId> = cx.with_trace(|t| {
        let coords: Vec<NodeId> = (0..ndim).map(|ax| t.coord(ax as u8)).collect();
        let g = t.graph();
        coords
            .into_iter()
            .map(|c| {
                let r = g.add_const(ConstValue::I32(ratio as i32), None);
                g.element_wise(ElementWiseOp::Mul, vec![c, r], None)
            })
            .collect()
    });
    let scale = cx.scalar("scale");
    let dst = cx.field("dst", "dst");
    let load = |off_axis: i64| {
        let coords: Vec<NodeId> = cx.with_trace(|t| {
            let g = t.graph();
            scaled
                .iter()
                .enumerate()
                .map(|(kk, &s)| {
                    let o = if kk == axis { off_axis } else { 0 };
                    let oc = g.add_const(ConstValue::I32(o as i32), None);
                    g.element_wise(ElementWiseOp::Add, vec![s, oc], None)
                })
                .collect()
        });
        gv_load_at(cx, "src", "src", &coords)
    };
    let mut sum = load(0);
    for kk in 1..ratio {
        sum = sum + load(kk);
    }
    let v = dst + scale * sum;
    let writes = vec![KernelWrite::new("dst", "dst", v.node())];
    writes
    })
}

/// the raw transverse child sum (the caller's scale carries the weights,
/// averaging included), normal axis passing through.
fn acc_face_sum<'t>(
    cx: TraceCx<'t>,
    scaled: &[NodeId],
    ratio: i64,
    axis: usize,
    ax: isize,
    off: &mut [i64; 3],
) -> Gv<'t> {
    if ax < 0 {
        let coords: Vec<NodeId> = cx.with_trace(|t| {
            let g = t.graph();
            scaled
                .iter()
                .enumerate()
                .map(|(kk, &s)| {
                    let o = g.add_const(ConstValue::I32(off[kk] as i32), None);
                    g.element_wise(ElementWiseOp::Add, vec![s, o], None)
                })
                .collect()
        });
        return gv_load_at(cx, "src", "src", &coords);
    }
    let aa = ax as usize;
    if aa == axis {
        off[aa] = 0;
        return acc_face_sum(cx, scaled, ratio, axis, ax - 1, off);
    }
    off[aa] = 0;
    let mut sum = acc_face_sum(cx, scaled, ratio, axis, ax - 1, off);
    for kk in 1..ratio {
        off[aa] = kk;
        sum = sum + acc_face_sum(cx, scaled, ratio, axis, ax - 1, off);
    }
    off[aa] = 0;
    sum
}

// =============================================================================
// face restriction: thread over the coarse coverage face domain, read the
// ratio^(D-1) coincident fine faces (staggered fields)
// =============================================================================

/// trace the fine -> coarse face restriction for a face-normal staggered field
/// (bface[axis]): `dst[c] = transverse-sweep-average of src[ratio*c + o]` where
/// the child offsets `o` run over the transverse axes only — the normal index
/// scales exactly (a coarse face is the union of its ratio^(D-1) fine faces,
/// area-weighted average = plain average on a uniform cartesian grid). input
/// "src" is the fine face field, output "dst" the coarse one; the dispatch
/// domain is the coverage face domain in absolute coarse indices.
pub fn refine_restrict_face_gv(ndim: usize, ratio: i64, axis: usize) -> (GvKernel, KernelWrites) {
    assert!(
        (1..=3).contains(&ndim),
        "refine_restrict_face_gv: ndim must be 1..=3"
    );
    assert!(
        axis < ndim,
        "refine_restrict_face_gv: axis {axis} out of range for ndim {ndim}"
    );
    assert!(ratio >= 2, "refine_restrict_face_gv: ratio must be >= 2");
    trace(|cx| {
    let scaled: Vec<NodeId> = cx.with_trace(|t| {
        let coords: Vec<NodeId> = (0..ndim).map(|ax| t.coord(ax as u8)).collect();
        let g = t.graph();
        coords
            .into_iter()
            .map(|c| {
                let r = g.add_const(ConstValue::I32(ratio as i32), None);
                g.element_wise(ElementWiseOp::Mul, vec![c, r], None)
            })
            .collect()
    });
    let inv_r = Gv::from_f64(1.0 / ratio as f64);
    let val = restrict_face_eval(cx, &scaled, ratio, inv_r, axis, ndim as isize - 1, &mut [0; 3]);
    let writes = vec![KernelWrite::new("dst", "dst", val.node())];
    writes
    })
}

/// the inlined transverse restrict sweep: identical to `restrict_eval` except
/// the face-normal axis passes through unaveraged (its fine index is exactly
/// `ratio * coord`).
fn restrict_face_eval<'t>(
    cx: TraceCx<'t>,
    scaled: &[NodeId],
    ratio: i64,
    inv_r: Gv<'t>,
    axis: usize,
    ax: isize,
    off: &mut [i64; 3],
) -> Gv<'t> {
    if ax < 0 {
        let coords: Vec<NodeId> = cx.with_trace(|t| {
            let g = t.graph();
            scaled
                .iter()
                .enumerate()
                .map(|(kk, &s)| {
                    let o = g.add_const(ConstValue::I32(off[kk] as i32), None);
                    g.element_wise(ElementWiseOp::Add, vec![s, o], None)
                })
                .collect()
        });
        return gv_load_at(cx, "src", "src", &coords);
    }
    let aa = ax as usize;
    if aa == axis {
        off[aa] = 0;
        return restrict_face_eval(cx, scaled, ratio, inv_r, axis, ax - 1, off);
    }
    off[aa] = 0;
    let mut sum = restrict_face_eval(cx, scaled, ratio, inv_r, axis, ax - 1, off);
    for kk in 1..ratio {
        off[aa] = kk;
        sum = sum + restrict_face_eval(cx, scaled, ratio, inv_r, axis, ax - 1, off);
    }
    off[aa] = 0;
    sum * inv_r
}

// =============================================================================
// prolongation: thread over the fine destination region, read the coarse
// parent neighborhood time-interpolated between two snapshots
// =============================================================================

/// trace the coarse -> fine prolongation: each fine cell reads its coarse
/// parent neighborhood (`floor_div(f, ratio)` — the Coarsen map, absolute
/// indices, ghost-safe for negatives) from the time-interpolated coarse state
/// `(1 - alpha)*src_old + alpha*src_new`, then applies the inlined per-axis
/// sweep at `order`. inputs "src_old"/"src_new" are the coarse field snapshots
/// (bind the same buffer twice to skip the time interpolation), scalar
/// "alpha" the interpolation fraction; output "dst" is the fine field. the
/// dispatch domain is the fine destination region (a coarse-fine ghost slab,
/// or a freshly nested patch interior) in absolute fine indices.
pub fn refine_prolong_gv(ndim: usize, ratio: i64, order: ProlongOrder) -> (GvKernel, KernelWrites) {
    assert!(
        (1..=3).contains(&ndim),
        "refine_prolong_gv: ndim must be 1..=3"
    );
    assert!(ratio >= 2, "refine_prolong_gv: ratio must be >= 2");
    trace(|cx| {
    let alpha = cx.scalar("alpha");
    let geom = prolong_geometry(cx, ndim, ratio);
    let src = ProlongSrc::TimePair {
        old: "src_old",
        new: "src_new",
        alpha,
    };
    let ctx = geom.ctx(ndim, order, &src);
    let val = prolong_eval(cx, &ctx, ndim as isize - 1, &mut [0; 3]);
    let writes = vec![KernelWrite::new("dst", "dst", val.node())];
    writes
    })
}

/// trace the multi-field (prim batch) cell prolongation: one kernel that sweeps
/// the shared coarse->fine stencil over `ncomp` co-located fields, reading
/// `src_old_{k}`/`src_new_{k}` and writing `dst_{k}` for k in 0..ncomp. the
/// per-cell geometry (parent index, parity, plm/ppm weights) is computed once
/// and reused across all components (graph CSE), and the host issues a single
/// dispatch (one rayon launch) covering the whole prim set. bit-identical to
/// `ncomp` single-field prolongs.
/// buffers in signature order: src_old_0, src_new_0, .., src_old_{n-1},
/// src_new_{n-1} (inputs) then dst_0..dst_{n-1} (outputs); scalar "alpha".
pub fn refine_prolong_multi_gv(
    ndim: usize,
    ratio: i64,
    order: ProlongOrder,
    ncomp: usize,
) -> (GvKernel, KernelWrites) {
    assert!(
        (1..=3).contains(&ndim),
        "refine_prolong_multi_gv: ndim must be 1..=3"
    );
    assert!(ratio >= 2, "refine_prolong_multi_gv: ratio must be >= 2");
    assert!(ncomp >= 1, "refine_prolong_multi_gv: ncomp must be >= 1");
    trace(|cx| {
    let alpha = cx.scalar("alpha");
    let geom = prolong_geometry(cx, ndim, ratio);
    let mut writes = Vec::with_capacity(ncomp);
    for k in 0..ncomp {
        let (old_name, new_name, dst_name) = (
            format!("src_old_{k}"),
            format!("src_new_{k}"),
            format!("dst_{k}"),
        );
        let src = ProlongSrc::TimePair {
            old: &old_name,
            new: &new_name,
            alpha,
        };
        let ctx = geom.ctx(ndim, order, &src);
        let val = prolong_eval(cx, &ctx, ndim as isize - 1, &mut [0; 3]);
        writes.push(KernelWrite::new(dst_name.clone(), dst_name, val.node()));
    }
    writes
    })
}

/// the single-snapshot multi-field prolongation: `refine_prolong_multi_gv`
/// with the leaf reading one coarse buffer per component ("src_{k}"), the time
/// interpolation already folded in. a `field_lerp` pass time-interpolates the
/// coarse snapshots once per coarse cell, halving the gather traffic (a time pair reads 2x the
/// loads of a 5^3 ppm neighborhood, recomputed per fine cell). buffers in
/// signature order: src_0..src_{n-1} (inputs) then dst_0..dst_{n-1} (outputs);
/// no scalars.
pub fn refine_prolong_multi_1t_gv(
    ndim: usize,
    ratio: i64,
    order: ProlongOrder,
    ncomp: usize,
) -> (GvKernel, KernelWrites) {
    assert!(
        (1..=3).contains(&ndim),
        "refine_prolong_multi_1t_gv: ndim must be 1..=3"
    );
    assert!(ratio >= 2, "refine_prolong_multi_1t_gv: ratio must be >= 2");
    assert!(ncomp >= 1, "refine_prolong_multi_1t_gv: ncomp must be >= 1");
    trace(|cx| {
    let geom = prolong_geometry(cx, ndim, ratio);
    let mut writes = Vec::with_capacity(ncomp);
    for k in 0..ncomp {
        let (src_name, dst_name) = (format!("src_{k}"), format!("dst_{k}"));
        let src = ProlongSrc::Single { name: &src_name };
        let ctx = geom.ctx(ndim, order, &src);
        let val = prolong_eval(cx, &ctx, ndim as isize - 1, &mut [0; 3]);
        writes.push(KernelWrite::new(dst_name.clone(), dst_name, val.node()));
    }
    writes
    })
}

/// trace one pass of the axis-split prolongation: the 1d
/// interpolation operator applied along `sweep_axis` only, every other axis
/// passing the thread coordinate through to the input load. the swept axis of
/// the output lattice is fine-indexed (parent = floor_div(c, r), parity = the
/// child sub-position), the unswept axes keep the input's indexing — chaining
/// the passes axis 0 -> 1 -> 2 reproduces the inlined tensor product bit for
/// bit (same 1d operators, same operand order, f64 intermediates). inputs
/// "src_{k}" (the lerped coarse for pass 0, the previous intermediate after),
/// outputs "dst_{k}"; scalars: none.
pub fn refine_prolong_sweep_multi_gv(
    ndim: usize,
    ratio: i64,
    order: ProlongOrder,
    sweep_axis: usize,
    ncomp: usize,
) -> (GvKernel, KernelWrites) {
    assert!(
        (1..=3).contains(&ndim),
        "refine_prolong_sweep_multi_gv: ndim must be 1..=3"
    );
    assert!(
        ratio >= 2,
        "refine_prolong_sweep_multi_gv: ratio must be >= 2"
    );
    assert!(
        sweep_axis < ndim,
        "refine_prolong_sweep_multi_gv: sweep_axis out of range"
    );
    assert!(
        ncomp >= 1,
        "refine_prolong_sweep_multi_gv: ncomp must be >= 1"
    );
    trace(|cx| {
    // parent + parity on the swept axis only (the same arithmetic
    // prolong_geometry builds per axis).
    let (parent, parity): (NodeId, NodeId) = cx.with_trace(|t| {
        let c = t.coord(sweep_axis as u8);
        let g = t.graph();
        let r = g.add_const(ConstValue::I32(ratio as i32), None);
        let p = g.element_wise(ElementWiseOp::FloorDiv, vec![c, r], None);
        let pr = g.element_wise(ElementWiseOp::Mul, vec![p, r], None);
        let q = g.element_wise(ElementWiseOp::Sub, vec![c, pr], None);
        (p, q)
    });
    let half = Gv::from_f64(0.5);
    let inv_ratio = Gv::from_f64(1.0 / ratio as f64);
    let ratio_f = Gv::from_f64(ratio as f64);
    let one = Gv::from_f64(1.0);
    let parity_f = cx.gv(parity);
    let frac = (parity_f + half) * inv_ratio - half;
    let xi_lo = parity_f * inv_ratio;
    let xi_hi = (parity_f + one) * inv_ratio;

    let w = order.ghost_width() as i64;
    let mut writes = Vec::with_capacity(ncomp);
    for k in 0..ncomp {
        let src_name = format!("src_{k}");
        // the stencil loads: swept axis at parent + dd, unswept axes at the
        // raw thread coordinate.
        let load = |dd: i64| {
            let coords: Vec<NodeId> = cx.with_trace(|t| {
                let raw: Vec<NodeId> = (0..ndim).map(|ax| t.coord(ax as u8)).collect();
                let g = t.graph();
                raw.into_iter()
                    .enumerate()
                    .map(|(ax, c)| {
                        if ax == sweep_axis {
                            let o = g.add_const(ConstValue::I32(dd as i32), None);
                            g.element_wise(ElementWiseOp::Add, vec![parent, o], None)
                        } else {
                            c
                        }
                    })
                    .collect()
            });
            gv_load_at(cx, &src_name, src_name.as_str(), &coords)
        };
        let vals: Vec<Gv> = (-w..=w).map(load).collect();
        let val = match order {
            ProlongOrder::Pcm => vals[0],
            ProlongOrder::Plm => plm_interp(vals[0], vals[1], vals[2], frac),
            ProlongOrder::Ppm => ppm_interp(
                vals[0], vals[1], vals[2], vals[3], vals[4], xi_lo, xi_hi, ratio_f,
            ),
            ProlongOrder::Quartic => quartic_interp(
                vals[0], vals[1], vals[2], vals[3], vals[4], xi_lo, xi_hi, ratio_f,
            ),
        };
        let dst_name = format!("dst_{k}");
        writes.push(KernelWrite::new(dst_name.clone(), dst_name, val.node()));
    }
    writes
    })
}

/// the pointwise time interpolation `dst_k = (1 - alpha)*src_old_k +
/// alpha*src_new_k` over `ncomp` co-located fields in one dispatch — the pass
/// that hoists the prolong leaf's per-fine-cell lerp to once per coarse cell.
/// the expression is spelled exactly as the prolong leaf spelled it, so the
/// lerp-then-prolong-1t chain is bit-identical to the fused time-pair kernel.
/// buffers: src_old_0, src_new_0, .., interleaved (inputs) then dst_0..
/// (outputs); scalar "alpha".
pub fn field_lerp_multi_gv(ndim: usize, ncomp: usize) -> (GvKernel, KernelWrites) {
    assert!(
        (1..=3).contains(&ndim),
        "field_lerp_multi_gv: ndim must be 1..=3"
    );
    assert!(ncomp >= 1, "field_lerp_multi_gv: ncomp must be >= 1");
    trace(|cx| {
    let alpha = cx.scalar("alpha");
    let one = Gv::from_f64(1.0);
    let _ = ndim;
    let mut writes = Vec::with_capacity(ncomp);
    for k in 0..ncomp {
        let old_key = format!("src_old_{k}");
        let new_key = format!("src_new_{k}");
        let v_old = cx.field(&old_key, old_key.as_str());
        let v_new = cx.field(&new_key, new_key.as_str());
        let v = (one - alpha) * v_old + alpha * v_new;
        let dst_name = format!("dst_{k}");
        writes.push(KernelWrite::new(dst_name.clone(), dst_name, v.node()));
    }
    writes
    })
}

// =============================================================================
// balance-aware coarse-fine transfer: the fused lerp+encode over the coarse
// parent region and the decode over the fine ghost slab. between them the
// unchanged prolong kernels act on pressure departures from the mechanical
// equilibrium chained out of the interior (Kaeppeli & Mishra, A&A 587, A94,
// 2016): pressure is the piecewise-constant-density path integral of
// `-rho dphi`, with each cell's own density carrying its segment and the
// segments meeting at the faces. the encode chains on the coarse lattice from
// the coarse cell under the nearest fine interior cell; the decode chains on the
// fine lattice from that interior cell itself, through the ghosts' own
// densities. a coarse stencil in its discrete class encodes to departures that
// vanish identically, and the decoded ghosts then satisfy the fine lattice's
// own recursion against the interior for whatever density the prolongation
// hands them — the seam is a fixed point of the balanced hierarchy, with no
// thermal structure assumed and no dependence on how restriction averages.
// density and velocity pass through the transfer unchanged. cartesian.
// =============================================================================

/// the longest chain the coarse-fine transfer kernels unroll, in cells along one
/// axis: the fine ghost slab is the evolution halo (at most three cells for the
/// parabolic stencil), and the coarse parent region reaches the prolongation's
/// own ghost width past the slab's parents. the balanced restriction's coarse
/// band (the evolution reach, at most three) decodes through the same kernel.
/// the host dispatch asserts every extent against this bound.
pub const WB_CF_CHAIN_MAX: i64 = 4;

/// the longest chain the balanced restriction's fine encode unrolls: the fine
/// cells under a coarse band of up to `WB_CF_CHAIN_MAX` cells, chained from the
/// fine interior edge — twice the band less one.
pub const WB_BAND_CHAIN_MAX: i64 = 2 * WB_CF_CHAIN_MAX;

/// the per-slot body scalars a balance-aware transfer kernel declares:
/// `body_{b}_pos_{g}` per grid axis, then `body_{b}_mass` / `_soft` /
/// `_softkind` — the same slot layout the wb ghost fill and body source bind
/// (an inert slot carries mass = 0 and contributes exactly zero potential).
/// declared eagerly so the scalar manifest order follows the declaration order
/// here, independent of the trace's evaluation order.
fn declare_body_slots<'t>(
    cx: TraceCx<'t>,
    ndim: usize,
    n_bodies: usize,
) -> Vec<([Gv<'t>; 3], Gv<'t>, Gv<'t>, Gv<'t>)> {
    (0..n_bodies)
        .map(|b| {
            let mut pos = [Gv::from_f64(0.0), Gv::from_f64(0.0), Gv::from_f64(0.0)];
            for (g, p) in pos.iter_mut().enumerate().take(ndim) {
                *p = cx.scalar(&format!("body_{b}_pos_{g}"));
            }
            (
                pos,
                cx.scalar(&format!("body_{b}_mass")),
                cx.scalar(&format!("body_{b}_soft")),
                cx.scalar(&format!("body_{b}_softkind")),
            )
        })
        .collect()
}

/// the total body potential at a cartesian position (grid axes are the
/// coordinates; ungridded components zero), summed over every slot.
fn body_slots_potential<'t>(
    pos: &[Gv<'t>; 3],
    slots: &[([Gv<'t>; 3], Gv<'t>, Gv<'t>, Gv<'t>)],
) -> Gv<'t> {
    slots
        .iter()
        .map(|(bpos, mass, soft, softkind)| {
            let rvec: [Gv; 3] = std::array::from_fn(|i| pos[i] - bpos[i]);
            crate::ibm::body_potential(rvec, *mass, *soft, *softkind)
        })
        .sum::<Gv>()
}

/// the cell centroid on a uniform cartesian lattice: `x_lo + (i + 1/2) dx` per
/// grid axis — the same text the host `stagger_coord(Center)` computes, so the
/// kernel potential agrees with a host evaluation bit for bit.
fn centroid_position<'t>(
    cx: TraceCx<'t>,
    ndim: usize,
    coords: &[NodeId],
    x_lo: &[Gv<'t>],
    dx: &[Gv<'t>],
) -> [Gv<'t>; 3] {
    let half = Gv::from_f64(0.5);
    let mut pos = [Gv::from_f64(0.0), Gv::from_f64(0.0), Gv::from_f64(0.0)];
    for g in 0..ndim {
        pos[g] = x_lo[g] + (cx.gv(coords[g]) + half) * dx[g];
    }
    pos
}

/// integer-index helpers on the trace graph: constants, arithmetic and the
/// clamp-by-select the chain walk needs to keep every load inside the cells it
/// actually visits.
fn int_const(cx: TraceCx<'_>, v: i64) -> NodeId {
    cx.with_trace(|t| t.graph().add_const(ConstValue::I32(v as i32), None))
}
fn int_op(cx: TraceCx<'_>, op: ElementWiseOp, a: NodeId, b: NodeId) -> NodeId {
    cx.with_trace(|t| t.graph().element_wise(op, vec![a, b], None))
}
fn int_select(cx: TraceCx<'_>, cond: NodeId, yes: NodeId, no: NodeId) -> NodeId {
    cx.with_trace(|t| t.graph().select(cond, yes, no, None))
}
/// `coord` clamped into the inclusive interior range `[lo, hi]` along one axis.
fn int_clamp(cx: TraceCx<'_>, coord: NodeId, lo: NodeId, hi: NodeId) -> NodeId {
    let below = int_op(cx, ElementWiseOp::Lt, coord, lo);
    let above = int_op(cx, ElementWiseOp::Gt, coord, hi);
    int_select(cx, below, lo, int_select(cx, above, hi, coord))
}

/// the mechanical equilibrium pressure at the thread's cell, chained from the
/// reference cell `ref_idx` through the lattice's own cells along an
/// axis-ordered staircase (axis 0 first): `p_ref` plus, for every step between
/// adjacent cells, `rho_from (phi(c_from) - phi(F)) + rho_to (phi(F) - phi(c_to))`
/// with `F` the face the two share. each leg unrolls `max_steps` steps, carrying
/// its direction as data; steps past the leg's length are masked to zero and their
/// loads clamp onto the leg's last cell, so every read lands on a visited cell.
/// a thread sitting on the reference cell takes every mask and returns `p_ref`
/// bit for bit. the chain carries no floor and no fade: it is a reference the
/// departure is measured against on one lattice and added back on the other,
/// and that round trip is exact linear algebra wherever the two lattices hold
/// the same densities.
fn chained_pressure<'t>(
    cx: TraceCx<'t>,
    ndim: usize,
    ref_idx: &[NodeId],
    p_ref: Gv<'t>,
    rho_at: &dyn Fn(&[NodeId]) -> Gv<'t>,
    x_lo: &[Gv<'t>],
    dx: &[Gv<'t>],
    slots: &[([Gv<'t>; 3], Gv<'t>, Gv<'t>, Gv<'t>)],
    max_steps: i64,
    vary: &[bool],
) -> Gv<'t> {
    let coords: Vec<NodeId> = cx.with_trace(|t| (0..ndim).map(|ax| t.coord(ax as u8)).collect());
    let phi_center = |idx: &[NodeId]| {
        body_slots_potential(&centroid_position(cx, ndim, idx, x_lo, dx), slots)
    };
    // the face of cell `idx` on its lower (`side = 0`) or upper (`side = 1`) edge along
    // `ax`. the side rides as a value, since a walk carries its direction as data.
    let phi_face = |idx: &[NodeId], ax: usize, side: NodeId| {
        let mut pos = centroid_position(cx, ndim, idx, x_lo, dx);
        let face_index = int_op(cx, ElementWiseOp::Add, idx[ax], side);
        pos[ax] = x_lo[ax] + cx.gv(face_index) * dx[ax];
        body_slots_potential(&pos, slots)
    };
    let zero = Gv::from_f64(0.0);
    let zero_i = int_const(cx, 0);
    let mut p = p_ref;
    // the staircase: leg `ax` walks from the point reached by the lower legs
    // (thread coords on axes < ax, reference coords on axes >= ax) to the
    // thread's coordinate on `ax`.
    let mut base: Vec<NodeId> = ref_idx.to_vec();
    for ax in 0..ndim {
        // a leg whose reference shares the thread's coordinate on this axis walks
        // zero cells for every thread in the region. its steps would each mask to
        // zero, and a select evaluates both arms, so an unrolled leg costs its
        // potential evaluations and its loads to add an exact zero. the caller
        // knows which axes its region can leave the reference on, so the graph
        // carries only those.
        if !vary[ax] {
            base[ax] = coords[ax];
            continue;
        }
        let d = int_op(cx, ElementWiseOp::Sub, coords[ax], ref_idx[ax]);
        // the leg is one-sided. the reference is the thread's own coordinate clamped
        // into a range, so a thread sits below that range, above it, or inside — never
        // on both sides at once. the direction therefore rides as data and the graph
        // unrolls a single leg, where a leg per direction would spend half its steps
        // masked to zero for every thread.
        let below = int_op(cx, ElementWiseOp::Lt, d, zero_i);
        let above = int_op(cx, ElementWiseOp::Gt, d, zero_i);
        let sgn = int_select(
            cx,
            below,
            int_const(cx, -1),
            int_select(cx, above, int_const(cx, 1), zero_i),
        );
        let span = int_select(cx, below, int_op(cx, ElementWiseOp::Sub, zero_i, d), d);
        let span_f = cx.gv(span);
        // the face a step crosses is the lower face of the cell it arrives at when
        // walking up, and that cell's upper face when walking down.
        let face_side = int_select(cx, below, int_const(cx, 1), zero_i);

        let at = |t: i64| -> Vec<NodeId> {
            let offset = int_op(cx, ElementWiseOp::Mul, sgn, int_const(cx, t));
            let reached = int_op(cx, ElementWiseOp::Add, base[ax], offset);
            let inside = int_op(cx, ElementWiseOp::Ge, span, int_const(cx, t));
            let mut idx = base.clone();
            idx[ax] = int_select(cx, inside, reached, coords[ax]);
            idx
        };

        // a visited cell serves the step that arrives at it and the step that leaves
        // it, so its center potential and its density are formed once and carried.
        let origin = at(0);
        let (mut phi_prev, mut rho_prev) = (phi_center(&origin), rho_at(&origin));
        for t in 1..=max_steps {
            let cur = at(t);
            let (phi_cur, rho_cur) = (phi_center(&cur), rho_at(&cur));
            let phi_f = phi_face(&cur, ax, face_side);
            let step = rho_prev * (phi_prev - phi_f) + rho_cur * (phi_f - phi_cur);
            let live = span_f.cmp_ge(Gv::from_f64(t as f64));
            p = p + Gv::select(live, step, zero);
            phi_prev = phi_cur;
            rho_prev = rho_cur;
        }
        base[ax] = coords[ax];
    }
    p
}

/// trace the fused lerp + hydrostatic encode over the coarse parent region of a
/// coarse-fine ghost slab: every component is time-interpolated
/// `(1 - alpha)*src_old_k + alpha*src_new_k`, and pre (component ncomp-1)
/// additionally subtracts the mechanical equilibrium chained from the coarse
/// cell under the nearest fine interior cell — the thread's own coordinates
/// clamped into `[lo_{ax}, hi_{ax}]` — through the lerped coarse densities. the
/// prolong kernels then act on the resulting departures unchanged; density and
/// velocities carry their lerped values.
///
/// buffers: src_old_0, src_new_0, .., interleaved (inputs) then dst_0..
/// (outputs). ints: lo_{ax}, hi_{ax} (the coarse cells under the fine interior,
/// inclusive). scalars: alpha, x_lo_{ax}, dx_{ax} (the coarse lattice), then
/// the body slots.
pub fn wb_cf_lerp_encode_gv(
    ndim: usize,
    ncomp: usize,
    n_bodies: usize,
) -> (GvKernel, KernelWrites) {
    assert!(
        (1..=3).contains(&ndim),
        "wb_cf_lerp_encode_gv: ndim must be 1..=3"
    );
    assert!(
        ncomp == ndim + 2,
        "wb_cf_lerp_encode_gv: the balance-aware transfer carries rho + ndim velocities + pre"
    );
    trace(|cx| {
    // scalar manifest in declaration order: alpha; the interior-bound int lanes;
    // grid origin/step; body slots.
    let alpha = cx.scalar("alpha");
    let (lo, hi) = declare_interior_bounds(cx, ndim);
    let x_lo: Vec<Gv> = (0..ndim)
        .map(|ax| cx.scalar(&format!("x_lo_{ax}")))
        .collect();
    let dx: Vec<Gv> = (0..ndim)
        .map(|ax| cx.scalar(&format!("dx_{ax}")))
        .collect();
    let slots = declare_body_slots(cx, ndim, n_bodies);

    let one = Gv::from_f64(1.0);
    // cell loads in interleaved (old_k, new_k) registration order — the buffer
    // order the dispatch binds.
    let lerp_cell: Vec<Gv> = (0..ncomp)
        .map(|k| {
            let old_key = format!("src_old_{k}");
            let new_key = format!("src_new_{k}");
            let v_old = cx.field(&old_key, old_key.as_str());
            let v_new = cx.field(&new_key, new_key.as_str());
            (one - alpha) * v_old + alpha * v_new
        })
        .collect();
    // the lerped state at an arbitrary coarse index, from the same registered
    // buffers.
    let lerp_at = |k: usize, idx: &[NodeId]| {
        let old_key = format!("src_old_{k}");
        let new_key = format!("src_new_{k}");
        let v_old = gv_load_at(cx, &old_key, old_key.as_str(), idx);
        let v_new = gv_load_at(cx, &new_key, new_key.as_str(), idx);
        (one - alpha) * v_old + alpha * v_new
    };

    let coords: Vec<NodeId> = cx.with_trace(|t| (0..ndim).map(|ax| t.coord(ax as u8)).collect());
    let ref_idx: Vec<NodeId> = (0..ndim)
        .map(|ax| int_clamp(cx, coords[ax], lo[ax], hi[ax]))
        .collect();
    let p_ref = lerp_at(ncomp - 1, &ref_idx);
    let rho_at = |idx: &[NodeId]| lerp_at(0, idx);
    let all = vec![true; ndim];
    let p_eq = chained_pressure(
        cx,
        ndim,
        &ref_idx,
        p_ref,
        &rho_at,
        &x_lo,
        &dx,
        &slots,
        WB_CF_CHAIN_MAX,
        &all,
    );

    let mut writes = Vec::with_capacity(ncomp);
    for k in 0..ncomp {
        let val = if k == ncomp - 1 {
            lerp_cell[k] - p_eq
        } else {
            lerp_cell[k]
        };
        let dst_name = format!("dst_{k}");
        writes.push(KernelWrite::new(dst_name.clone(), dst_name, val.node()));
    }
    writes
    })
}

/// the inclusive interior bounds per axis as int scalar lanes `lo_{ax}` / `hi_{ax}`.
fn declare_interior_bounds(cx: TraceCx<'_>, ndim: usize) -> (Vec<NodeId>, Vec<NodeId>) {
    cx.with_trace(|t| {
        let lo = (0..ndim)
            .map(|ax| t.scalar_int(&format!("lo_{ax}")))
            .collect();
        let hi = (0..ndim)
            .map(|ax| t.scalar_int(&format!("hi_{ax}")))
            .collect();
        (lo, hi)
    })
}

/// trace the hydrostatic decode over the fine coarse-fine ghost slab: the
/// prolonged pressure departure already sits in the fine `dst_pre`, and each
/// ghost adds back the mechanical equilibrium chained from the nearest fine
/// interior cell — its own coordinates clamped into `[lo_{ax}, hi_{ax}]` —
/// through the fine densities the prolongation wrote. the interior cell is
/// outside the slab, so its pressure is never a write of this pass, and the
/// densities are read only, so the in-place pressure write is race-free.
///
/// buffers: dst_rho (the fine density, input), dst_pre (the fine ghosts, in-place
/// output). ints: lo_{ax}, hi_{ax} (the fine interior, inclusive). scalars:
/// x_lo_{ax}, dx_{ax} (the fine lattice), then the body slots.
pub fn wb_cf_decode_gv(ndim: usize, n_bodies: usize) -> (GvKernel, KernelWrites) {
    assert!(
        (1..=3).contains(&ndim),
        "wb_cf_decode_gv: ndim must be 1..=3"
    );
    trace(|cx| {
    let (lo, hi) = declare_interior_bounds(cx, ndim);
    let x_lo: Vec<Gv> = (0..ndim)
        .map(|ax| cx.scalar(&format!("x_lo_{ax}")))
        .collect();
    let dx: Vec<Gv> = (0..ndim)
        .map(|ax| cx.scalar(&format!("dx_{ax}")))
        .collect();
    let slots = declare_body_slots(cx, ndim, n_bodies);

    // registration order pins the buffer order: density first (input), then the
    // pressure (in-place).
    let rho_at = |idx: &[NodeId]| gv_load_at(cx, "dst_rho", "dst_rho", idx);
    let coords: Vec<NodeId> = cx.with_trace(|t| (0..ndim).map(|ax| t.coord(ax as u8)).collect());
    let _ = rho_at(&coords);
    let departure = cx.field("dst_pre", "dst_pre");
    let ref_idx: Vec<NodeId> = (0..ndim)
        .map(|ax| int_clamp(cx, coords[ax], lo[ax], hi[ax]))
        .collect();
    let p_ref = gv_load_at(cx, "dst_pre", "dst_pre", &ref_idx);
    let all = vec![true; ndim];
    let p_eq = chained_pressure(
        cx,
        ndim,
        &ref_idx,
        p_ref,
        &rho_at,
        &x_lo,
        &dx,
        &slots,
        WB_CF_CHAIN_MAX,
        &all,
    );

    // the decoded pressure is the local equilibrium plus the transported
    // departure, and the decomposition holds only inside the physical regime.
    // where the sum leaves it the cell takes the equilibrium alone -- the
    // zero-departure decode, which keeps the cell on the mechanical recursion
    // -- and where the chain itself has left the regime, the reference cell's
    // own pressure, positive by construction as the anchor the chain starts
    // from. a positive decoded value passes through bit for bit, so a balanced
    // column's ghosts and the machine-exact seam gates carry the identical
    // graph values.
    let zero = Gv::from_f64(0.0);
    let decoded = departure + p_eq;
    let fallback = Gv::select(p_eq.cmp_gt(zero), p_eq, p_ref);
    let pre_g = Gv::select(decoded.cmp_gt(zero), decoded, fallback);
    let writes = vec![KernelWrite::new("dst_pre", "dst_pre", pre_g.node())];
    writes
    })
}

/// add a fixed fine-level target to prolonged primitive departures. the
/// candidate is accepted only when density and pressure are positive and every
/// primitive component is finite; otherwise the entire target state is written.
/// selecting one coherent fallback avoids combining a target density with a
/// departed velocity or pressure.
///
/// buffers: target components (inputs), then departure/candidate components
/// (in-place outputs), ordered as rho, velocities, optional pressure.
pub fn wb_target_decode_gv(ndim: usize, ncomp: usize) -> (GvKernel, KernelWrites) {
    assert!(
        (1..=3).contains(&ndim),
        "wb_target_decode_gv: ndim must be 1..=3"
    );
    assert!(
        ncomp == ndim + 1 || ncomp == ndim + 2,
        "wb_target_decode_gv: expected rho + ndim velocities + optional pressure"
    );
    trace(|cx| {
    let mut target = Vec::with_capacity(ncomp);
    let mut candidate = Vec::with_capacity(ncomp);
    for kk in 0..ncomp {
        let eq_key = format!("eq_{kk}");
        let dst_key = format!("dst_{kk}");
        let eq = cx.field(&eq_key, eq_key.as_str());
        let departure = cx.field(&dst_key, dst_key.as_str());
        target.push(eq);
        candidate.push(eq + departure);
    }
    // finiteness probes as named-brand fns: a local closure cannot name the trace
    // brand, and annotating the elided lifetime mints regions invariance rejects.
    fn finite<'t>(value: Gv<'t>) -> symbi_ir::GvMask<'t> {
        (value - value).cmp_eq(Gv::ZERO)
    }
    fn finite_pos<'t>(value: Gv<'t>) -> symbi_ir::GvMask<'t> {
        finite(value) & value.cmp_gt(Gv::ZERO)
    }
    let mut physical = finite_pos(candidate[0]);
    for kk in 1..=ndim {
        physical = physical & finite(candidate[kk]);
    }
    if ncomp == ndim + 2 {
        physical = physical & finite_pos(candidate[ncomp - 1]);
    }
    let writes = (0..ncomp)
        .map(|kk| {
            let dst_key = format!("dst_{kk}");
            KernelWrite::new(
                dst_key.clone(),
                dst_key,
                Gv::select(physical, candidate[kk], target[kk]).node(),
            )
        })
        .collect();
    writes
    })
}

/// trace the balanced band decode over the covered coarse band at a seam: each
/// band cell adds the coarse mechanical chain from the uncovered cell beyond
/// the seam to its restricted departure, and a sum outside the physical regime
/// falls back to the cell's own conservative pressure -- the fine average the
/// restriction wrote, which is the covered cell's honest value whenever the
/// departure decomposition breaks. a drain-evacuated fine region under a dense
/// exterior is the breaking configuration: the class continuation of the
/// exterior overstates the evacuated gas by an order of magnitude, so the
/// abstention must return the conservative average rather than the class.
/// in-class the departure vanishes, the decoded value is the positive class
/// pressure, and the guard passes it bit for bit.
///
/// the departures live in their own buffer, so the pressure field holds the
/// conservative restriction until this pass overwrites it. each thread reads
/// the pressure at the uncovered reference (outside the written band) and at
/// its own cell, so the in-place write is race-free.
///
/// buffers: dst_rho (the restricted coarse density, input), band_dep (the
/// restricted departures, input), dst_pre (conservative pressure in, decoded
/// pressure out). ints: lo_{ax}, hi_{ax} (the uncovered reference row,
/// inclusive). scalars: x_lo_{ax}, dx_{ax} (the coarse lattice), then the body
/// slots.
pub fn wb_band_decode_gv(ndim: usize, n_bodies: usize) -> (GvKernel, KernelWrites) {
    assert!(
        (1..=3).contains(&ndim),
        "wb_band_decode_gv: ndim must be 1..=3"
    );
    trace(|cx| {
    let (lo, hi) = declare_interior_bounds(cx, ndim);
    let x_lo: Vec<Gv> = (0..ndim)
        .map(|ax| cx.scalar(&format!("x_lo_{ax}")))
        .collect();
    let dx: Vec<Gv> = (0..ndim)
        .map(|ax| cx.scalar(&format!("dx_{ax}")))
        .collect();
    let slots = declare_body_slots(cx, ndim, n_bodies);

    // registration order pins the buffer order: density, departures, pressure.
    let rho_at = |idx: &[NodeId]| gv_load_at(cx, "dst_rho", "dst_rho", idx);
    let coords: Vec<NodeId> = cx.with_trace(|t| (0..ndim).map(|ax| t.coord(ax as u8)).collect());
    let _ = rho_at(&coords);
    let departure = cx.field("band_dep", "band_dep");
    let p_cons = cx.field("dst_pre", "dst_pre");
    let ref_idx: Vec<NodeId> = (0..ndim)
        .map(|ax| int_clamp(cx, coords[ax], lo[ax], hi[ax]))
        .collect();
    let p_ref = gv_load_at(cx, "dst_pre", "dst_pre", &ref_idx);
    let all = vec![true; ndim];
    let p_eq = chained_pressure(
        cx,
        ndim,
        &ref_idx,
        p_ref,
        &rho_at,
        &x_lo,
        &dx,
        &slots,
        WB_CF_CHAIN_MAX,
        &all,
    );

    let zero = Gv::from_f64(0.0);
    let decoded = departure + p_eq;
    let pre_g = Gv::select(decoded.cmp_gt(zero), decoded, p_cons);
    let writes = vec![KernelWrite::new("dst_pre", "dst_pre", pre_g.node())];
    writes
    })
}

/// trace the balanced restriction's fine encode over the fine cells under a
/// coarse band at a coarse-fine seam: each fine cell's pressure becomes its
/// departure from the mechanical equilibrium chained from the uncovered coarse
/// cell beyond the seam, continued across the seam face and through the fine
/// cells' own densities. the reference is the mechanical class of the
/// composite lattice (the uncovered coarse cells together with the fine cells),
/// which is the grid the solution lives on: the coarse cell `a` carries its own
/// segment from its center to the seam face, the fine edge cell `e` carries the
/// segment from that face point to its own center, and the fine chain runs on
/// from `e`. a fine column balanced against its coarse neighbor across the seam
/// encodes to departures that vanish identically; a wave standing at the seam
/// encodes to its full amplitude, so the coarse side sees it.
///
/// buffers: src_rho, src_pre (the fine primitives, inputs), crs_rho, crs_pre
/// (the coarse primitives, inputs) then dst (the fine departure, output). ints:
/// lo_{ax}, hi_{ax} (the fine reference clamp: the fine interior edge on the
/// seam's normal axis, the interior elsewhere), a (the uncovered coarse index on
/// the normal axis). the normal itself is a bake-time axis, so the chain carries
/// one leg. scalars: face (the seam coordinate along the normal), x_lo_{ax},
/// dx_{ax} (fine), crs_x_lo_{ax}, crs_dx_{ax} (coarse), then the body slots.
pub fn wb_band_encode_gv(ndim: usize, n_bodies: usize, normal: usize) -> (GvKernel, KernelWrites) {
    assert!(
        (1..=3).contains(&ndim),
        "wb_band_encode_gv: ndim must be 1..=3"
    );
    assert!(
        normal < ndim,
        "wb_band_encode_gv: the seam normal must be a grid axis"
    );
    trace(|cx| {
    let (lo, hi) = declare_interior_bounds(cx, ndim);
    let a_idx: NodeId = cx.with_trace(|t| t.scalar_int("a"));
    let face = cx.scalar("face");
    let x_lo: Vec<Gv> = (0..ndim)
        .map(|ax| cx.scalar(&format!("x_lo_{ax}")))
        .collect();
    let dx: Vec<Gv> = (0..ndim)
        .map(|ax| cx.scalar(&format!("dx_{ax}")))
        .collect();
    let crs_x_lo: Vec<Gv> = (0..ndim)
        .map(|ax| cx.scalar(&format!("crs_x_lo_{ax}")))
        .collect();
    let crs_dx: Vec<Gv> = (0..ndim)
        .map(|ax| cx.scalar(&format!("crs_dx_{ax}")))
        .collect();
    let slots = declare_body_slots(cx, ndim, n_bodies);

    // registration order pins the buffer order: fine rho, fine pre, coarse rho,
    // coarse pre, then the departure output.
    let coords: Vec<NodeId> = cx.with_trace(|t| (0..ndim).map(|ax| t.coord(ax as u8)).collect());
    let rho_at = |idx: &[NodeId]| gv_load_at(cx, "src_rho", "src_rho", idx);
    let _ = rho_at(&coords);
    let pre_cell = cx.field("src_pre", "src_pre");
    let ref_idx: Vec<NodeId> = (0..ndim)
        .map(|ax| int_clamp(cx, coords[ax], lo[ax], hi[ax]))
        .collect();
    // the uncovered coarse cell: `a` on the normal axis, the thread's own parent
    // elsewhere.
    let two_i = int_const(cx, 2);
    let crs_idx: Vec<NodeId> = (0..ndim)
        .map(|ax| {
            if ax == normal {
                a_idx
            } else {
                int_op(cx, ElementWiseOp::FloorDiv, coords[ax], two_i)
            }
        })
        .collect();
    let rho_a = gv_load_at(cx, "crs_rho", "crs_rho", &crs_idx);
    let pre_a = gv_load_at(cx, "crs_pre", "crs_pre", &crs_idx);

    // the coarse cell's center, the seam face point on its transverse center
    // line, and the fine edge cell's center.
    let c_a = centroid_position(cx, ndim, &crs_idx, &crs_x_lo, &crs_dx);
    let mut f_pt = c_a;
    f_pt[normal] = face;
    let c_e = centroid_position(cx, ndim, &ref_idx, &x_lo, &dx);
    let phi_a = body_slots_potential(&c_a, &slots);
    let phi_f = body_slots_potential(&f_pt, &slots);
    let phi_e = body_slots_potential(&c_e, &slots);
    let rho_e = rho_at(&ref_idx);
    let p_ref = pre_a + rho_a * (phi_a - phi_f) + rho_e * (phi_f - phi_e);
    // the band spans the coverage transversely and the reference is its own edge
    // cell, so a thread leaves the reference on the seam normal alone.
    let mut vary = vec![false; ndim];
    vary[normal] = true;
    let p_eq = chained_pressure(
        cx,
        ndim,
        &ref_idx,
        p_ref,
        &rho_at,
        &x_lo,
        &dx,
        &slots,
        WB_BAND_CHAIN_MAX,
        &vary,
    );
    let departure = pre_cell - p_eq;
    let writes = vec![KernelWrite::new("dst", "dst", departure.node())];
    writes
    })
}

/// trace the gamma-law energy rebuild over a region whose pressure was rewritten
/// in primitive space: `nrg = pre / (gamma - 1) + rho |v|^2 / 2`.
///
/// buffers: prim_rho, prim_vel_0.., prim_pre (inputs) then cons_nrg (output).
/// scalars: gamma.
pub fn band_energy_gv(ndim: usize) -> (GvKernel, KernelWrites) {
    assert!(
        (1..=3).contains(&ndim),
        "band_energy_gv: ndim must be 1..=3"
    );
    trace(|cx| {
    let gamma = cx.scalar("gamma");
    let rho = cx.field("prim_rho", "prim_rho");
    let vel: Vec<Gv> = (0..ndim)
        .map(|k| {
            let key = format!("prim_vel_{k}");
            cx.field(&key, key.as_str())
        })
        .collect();
    let pre = cx.field("prim_pre", "prim_pre");
    let half = Gv::from_f64(0.5);
    let v2 = vel.iter().map(|&v| v * v).sum::<Gv>();
    let nrg = pre / (gamma - Gv::ONE) + half * rho * v2;
    let writes = vec![KernelWrite::new("cons_nrg", "cons_nrg", nrg.node())];
    writes
    })
}

// =============================================================================
// face prolongation: thread over a fine face region (a bface transverse-halo
// slab at a coarse-fine boundary), read the time-interpolated coarse face
// field — the fine boundary-edge EMF at the coarse-fine interface
// =============================================================================

/// trace the coarse -> fine face prolongation for a face-normal staggered
/// field (bface[axis]). along the normal axis the fine face lattice
/// interleaves the coarse one: an even fine face (2c) coincides with coarse
/// face c, an odd one (2c+1) sits at the midpoint of faces c and c+1 — the
/// pair `floor_div(f, 2)` / `floor_div(f+1, 2)` collapses to (c, c) on even
/// faces (the half-sum is then exact) and (c, c+1) on odd ones, so every read
/// stays inside the coarse face domain. transverse axes use the van-leer plm
/// sweep (axis 0 innermost among them): the coarse bface carries a +/-1
/// transverse halo, exactly the reach the plm stencil needs, so plm is the
/// maximum order here, one above the pcm a plain copy would be.
/// inputs "src_old"/"src_new" + scalar "alpha" as in `refine_prolong_gv`.
pub fn refine_prolong_face_gv(ndim: usize, ratio: i64, axis: usize) -> (GvKernel, KernelWrites) {
    assert!(
        (1..=3).contains(&ndim),
        "refine_prolong_face_gv: ndim must be 1..=3"
    );
    assert!(
        axis < ndim,
        "refine_prolong_face_gv: axis {axis} out of range for ndim {ndim}"
    );
    assert!(
        ratio == 2,
        "refine_prolong_face_gv: the face-lattice midpoint pair is ratio-2"
    );
    trace(|cx| {
    let alpha = cx.scalar("alpha");
    let one = Gv::from_f64(1.0);

    // per-axis maps: the normal axis gets the face pair (lo, hi); transverse
    // axes the cell parent + plm frac, exactly as the cell prolongation.
    let (pair_lo, pair_hi, parent, parity): (Vec<NodeId>, Vec<NodeId>, Vec<NodeId>, Vec<NodeId>) =
        cx.with_trace(|t| {
            let coords: Vec<NodeId> = (0..ndim).map(|ax| t.coord(ax as u8)).collect();
            let g = t.graph();
            let mut lo = Vec::with_capacity(ndim);
            let mut hi = Vec::with_capacity(ndim);
            let mut ps = Vec::with_capacity(ndim);
            let mut qs = Vec::with_capacity(ndim);
            for &c in &coords {
                let r = g.add_const(ConstValue::I32(ratio as i32), None);
                let one_i = g.add_const(ConstValue::I32(1), None);
                let p = g.element_wise(ElementWiseOp::FloorDiv, vec![c, r], None);
                let c1 = g.element_wise(ElementWiseOp::Add, vec![c, one_i], None);
                let ph = g.element_wise(ElementWiseOp::FloorDiv, vec![c1, r], None);
                let pr = g.element_wise(ElementWiseOp::Mul, vec![p, r], None);
                let q = g.element_wise(ElementWiseOp::Sub, vec![c, pr], None);
                lo.push(p);
                hi.push(ph);
                ps.push(p);
                qs.push(q);
            }
            (lo, hi, ps, qs)
        });

    let half = Gv::from_f64(0.5);
    let inv_ratio = Gv::from_f64(1.0 / ratio as f64);
    let frac: Vec<Gv> = parity
        .iter()
        .map(|&q| (cx.gv(q) + half) * inv_ratio - half)
        .collect();

    let ctx = FaceProlongCtx {
        ndim,
        axis,
        pair_lo: &pair_lo,
        pair_hi: &pair_hi,
        parent: &parent,
        frac: &frac,
        one,
        alpha,
    };
    let val = face_prolong_eval(cx, &ctx, ndim as isize - 1, &mut [0; 3]);
    let writes = vec![KernelWrite::new("dst", "dst", val.node())];
    writes
    })
}

struct FaceProlongCtx<'a, 't> {
    ndim: usize,
    axis: usize,
    pair_lo: &'a [NodeId],
    pair_hi: &'a [NodeId],
    parent: &'a [NodeId],
    frac: &'a [Gv<'t>],
    one: Gv<'t>,
    alpha: Gv<'t>,
}

/// the inlined face-prolong sweep: transverse axes interpolate with plm (axis
/// 0 innermost among them), the normal axis passes through to the leaf, which
/// averages the time-interpolated coarse face pair.
fn face_prolong_eval<'t>(
    cx: TraceCx<'t>,
    ctx: &FaceProlongCtx<'_, 't>,
    ax: isize,
    off: &mut [i64; 3],
) -> Gv<'t> {
    if ax < 0 {
        let load = |normals: &[NodeId]| {
            let coords: Vec<NodeId> = cx.with_trace(|t| {
                let g = t.graph();
                (0..ctx.ndim)
                    .map(|kk| {
                        let base = if kk == ctx.axis {
                            normals[kk]
                        } else {
                            ctx.parent[kk]
                        };
                        let o = g.add_const(ConstValue::I32(off[kk] as i32), None);
                        g.element_wise(ElementWiseOp::Add, vec![base, o], None)
                    })
                    .collect()
            });
            let v_old = gv_load_at(cx, "src_old", "src_old", &coords);
            let v_new = gv_load_at(cx, "src_new", "src_new", &coords);
            (ctx.one - ctx.alpha) * v_old + ctx.alpha * v_new
        };
        let half = Gv::from_f64(0.5);
        return half * (load(ctx.pair_lo) + load(ctx.pair_hi));
    }
    let aa = ax as usize;
    if aa == ctx.axis {
        return face_prolong_eval(cx, ctx, ax - 1, off);
    }
    let vals: Vec<Gv> = (-1..=1i64)
        .map(|dd| {
            off[aa] = dd;
            face_prolong_eval(cx, ctx, ax - 1, off)
        })
        .collect();
    off[aa] = 0;
    plm_interp(vals[0], vals[1], vals[2], ctx.frac[aa])
}

/// the coarse-field read the prolong leaf performs: the classic time pair
/// (`(1 - alpha)*old + alpha*new` per coarse cell, recomputed per fine cell)
/// or a single pre-interpolated buffer (a `field_lerp` pass hoisted the time
/// interpolation to once per coarse cell — half the gather traffic).
enum ProlongSrc<'a, 't> {
    TimePair {
        old: &'a str,
        new: &'a str,
        alpha: Gv<'t>,
    },
    Single {
        name: &'a str,
    },
}

/// the shared per-cell prolong geometry: the per-axis Coarsen map (parent =
/// floor_div(c, r), parity = c - parent*r in 0..r) and the reference's
/// per-pass sub-cell positions from the parity kk (int promoted to f64 by the
/// graph): plm midpoint frac = (kk + 1/2)/r - 1/2, ppm sub-cell average
/// bounds xi_lo = kk/r, xi_hi = (kk + 1)/r.
struct ProlongGeom<'t> {
    parent: Vec<NodeId>,
    frac: Vec<Gv<'t>>,
    xi_lo: Vec<Gv<'t>>,
    xi_hi: Vec<Gv<'t>>,
    ratio_f: Gv<'t>,
    one: Gv<'t>,
}

fn prolong_geometry<'t>(cx: TraceCx<'t>, ndim: usize, ratio: i64) -> ProlongGeom<'t> {
    let one = Gv::from_f64(1.0);
    let (parent, parity): (Vec<NodeId>, Vec<NodeId>) = cx.with_trace(|t| {
        let coords: Vec<NodeId> = (0..ndim).map(|ax| t.coord(ax as u8)).collect();
        let g = t.graph();
        let mut ps = Vec::with_capacity(ndim);
        let mut qs = Vec::with_capacity(ndim);
        for &c in &coords {
            let r = g.add_const(ConstValue::I32(ratio as i32), None);
            let p = g.element_wise(ElementWiseOp::FloorDiv, vec![c, r], None);
            let pr = g.element_wise(ElementWiseOp::Mul, vec![p, r], None);
            let q = g.element_wise(ElementWiseOp::Sub, vec![c, pr], None);
            ps.push(p);
            qs.push(q);
        }
        (ps, qs)
    });
    let half = Gv::from_f64(0.5);
    let inv_ratio = Gv::from_f64(1.0 / ratio as f64);
    let ratio_f = Gv::from_f64(ratio as f64);
    let parity_f: Vec<Gv> = parity.iter().map(|&q| cx.gv(q)).collect();
    let frac: Vec<Gv> = parity_f
        .iter()
        .map(|&q| (q + half) * inv_ratio - half)
        .collect();
    let xi_lo: Vec<Gv> = parity_f.iter().map(|&q| q * inv_ratio).collect();
    let xi_hi: Vec<Gv> = parity_f.iter().map(|&q| (q + one) * inv_ratio).collect();
    ProlongGeom {
        parent,
        frac,
        xi_lo,
        xi_hi,
        ratio_f,
        one,
    }
}

impl<'t> ProlongGeom<'t> {
    fn ctx<'a>(
        &'a self,
        ndim: usize,
        order: ProlongOrder,
        src: &'a ProlongSrc<'a, 't>,
    ) -> ProlongCtx<'a, 't> {
        ProlongCtx {
            ndim,
            order,
            parent: &self.parent,
            frac: &self.frac,
            xi_lo: &self.xi_lo,
            xi_hi: &self.xi_hi,
            ratio_f: self.ratio_f,
            one: self.one,
            src,
        }
    }
}

struct ProlongCtx<'a, 't> {
    ndim: usize,
    order: ProlongOrder,
    parent: &'a [NodeId],
    frac: &'a [Gv<'t>],
    xi_lo: &'a [Gv<'t>],
    xi_hi: &'a [Gv<'t>],
    ratio_f: Gv<'t>,
    one: Gv<'t>,
    src: &'a ProlongSrc<'a, 't>,
}

/// the inlined prolong sweep: axis `ax` interpolates the values recursively
/// prolonged through the lower axes at the stencil offsets (axis 0 innermost =
/// the reference's first pass). leaf = the coarse read at `parent + off`
/// (time-interpolated for a TimePair source, plain for a pre-lerped Single).
fn prolong_eval<'t>(
    cx: TraceCx<'t>,
    ctx: &ProlongCtx<'_, 't>,
    ax: isize,
    off: &mut [i64; 3],
) -> Gv<'t> {
    if ax < 0 {
        let coords: Vec<NodeId> = cx.with_trace(|t| {
            let g = t.graph();
            ctx.parent
                .iter()
                .take(ctx.ndim)
                .enumerate()
                .map(|(kk, &p)| {
                    let o = g.add_const(ConstValue::I32(off[kk] as i32), None);
                    g.element_wise(ElementWiseOp::Add, vec![p, o], None)
                })
                .collect()
        });
        return match ctx.src {
            ProlongSrc::TimePair { old, new, alpha } => {
                let v_old = gv_load_at(cx, old, *old, &coords);
                let v_new = gv_load_at(cx, new, *new, &coords);
                (ctx.one - *alpha) * v_old + *alpha * v_new
            }
            ProlongSrc::Single { name } => gv_load_at(cx, name, *name, &coords),
        };
    }
    let aa = ax as usize;
    let w = ctx.order.ghost_width() as i64;
    let vals: Vec<Gv> = (-w..=w)
        .map(|dd| {
            off[aa] = dd;
            prolong_eval(cx, ctx, ax - 1, off)
        })
        .collect();
    off[aa] = 0;
    match ctx.order {
        ProlongOrder::Pcm => vals[0],
        ProlongOrder::Plm => plm_interp(vals[0], vals[1], vals[2], ctx.frac[aa]),
        ProlongOrder::Ppm => ppm_interp(
            vals[0],
            vals[1],
            vals[2],
            vals[3],
            vals[4],
            ctx.xi_lo[aa],
            ctx.xi_hi[aa],
            ctx.ratio_f,
        ),
        ProlongOrder::Quartic => quartic_interp(
            vals[0],
            vals[1],
            vals[2],
            vals[3],
            vals[4],
            ctx.xi_lo[aa],
            ctx.xi_hi[aa],
            ctx.ratio_f,
        ),
    }
}

#[cfg(test)]
mod quartic_interp_tests {
    use super::quartic_interp;

    // exact cell average of sin(tau x) over a width-h cell centered at x
    fn avg(x: f64, h: f64) -> f64 {
        let tau = std::f64::consts::TAU;
        (tau * x).sin() * (tau * h / 2.0).sin() / (tau * h / 2.0)
    }

    /// smooth data: away from the sine's inflection cells — where the undivided
    /// second differences change sign and the hybrid deliberately takes the
    /// monotonized O(h^3) fallback (a per-parent switch, so sibling children
    /// stay conservative) — the half-cell child averages converge at the
    /// degree-4 fit's O(h^5), halving h cuts the error ~32x. the fallback set
    /// must stay confined to the sign-change cells: a detector that widened
    /// would silently drag the whole transfer to third order.
    #[test]
    fn children_are_fifth_order_on_smooth_averages_away_from_inflections() {
        let sweep = |h: f64| -> (f64, usize) {
            let n = (1.0 / h) as usize;
            let mut worst = 0.0_f64;
            let mut fallback_cells = 0usize;
            for kk in 0..n {
                let xc = (kk as f64 + 0.5) * h;
                let v: Vec<f64> = (-2..=2).map(|o| avg(xc + o as f64 * h, h)).collect();
                let d2 = [
                    v[0] - 2.0 * v[1] + v[2],
                    v[1] - 2.0 * v[2] + v[3],
                    v[2] - 2.0 * v[3] + v[4],
                ];
                let d2_lo = d2.iter().cloned().fold(f64::INFINITY, f64::min);
                let d2_hi = d2.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                if !(d2_lo > 0.0 || d2_hi < 0.0) {
                    fallback_cells += 1;
                    continue;
                }
                for (lo, hi) in [(0.0, 0.5), (0.5, 1.0)] {
                    let got: f64 = quartic_interp(v[0], v[1], v[2], v[3], v[4], lo, hi, 2.0);
                    let xa = xc - h / 2.0 + lo * h;
                    let xb = xc - h / 2.0 + hi * h;
                    let exact = avg((xa + xb) / 2.0, xb - xa);
                    worst = worst.max((got - exact).abs());
                }
            }
            (worst, fallback_cells)
        };
        let (e1, f1) = sweep(1.0 / 32.0);
        let (e2, f2) = sweep(1.0 / 64.0);
        // the sine has two inflections; each sign change straddles at most two cells.
        assert!(
            f1 <= 4 && f2 <= 4,
            "the fallback detector fired on {f1}/{f2} cells; it must stay confined to \
             the sign-change neighborhood of the two inflections"
        );
        let ratio = e1 / e2;
        assert!(
            ratio > 24.0,
            "quartic children are not ~5th order away from inflections: \
             err(1/32)={e1:.3e}, err(1/64)={e2:.3e}, ratio={ratio:.2} (expect ~32)"
        );
    }

    /// conservation: the two half-cell children average back to the parent to
    /// roundoff, on data arbitrary enough that nothing cancels by symmetry.
    #[test]
    fn children_conserve_the_parent_average() {
        let v: [f64; 5] = [0.3, 1.7, 0.9, -0.4, 2.2];
        let lo: f64 = quartic_interp(v[0], v[1], v[2], v[3], v[4], 0.0, 0.5, 2.0);
        let hi: f64 = quartic_interp(v[0], v[1], v[2], v[3], v[4], 0.5, 1.0, 2.0);
        let resid = (0.5 * (lo + hi) - v[2]).abs();
        assert!(
            resid < 1e-14,
            "children do not conserve the parent: resid {resid:e}"
        );
    }

    /// a jump inside the stencil mixes the second-difference signs: the value
    /// must equal the monotonized parabolic fallback exactly (the raw quartic
    /// rings by ~13% here, so the fallback is what the kernel emits).
    #[test]
    fn falls_back_to_the_monotonized_parabola_at_a_jump() {
        let v: [f64; 5] = [1.0, 1.0, 1.0, 0.1, 0.1];
        for (lo, hi) in [(0.0, 0.5), (0.5, 1.0)] {
            let got: f64 = quartic_interp(v[0], v[1], v[2], v[3], v[4], lo, hi, 2.0);
            let ppm: f64 = super::ppm_interp(v[0], v[1], v[2], v[3], v[4], lo, hi, 2.0);
            assert_eq!(
                got.to_bits(),
                ppm.to_bits(),
                "jump stencil did not take the monotonized fallback"
            );
        }
    }
}
