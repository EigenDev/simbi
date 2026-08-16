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
use symbi_ir::FieldBind;
use symbi_ir::algebra::Scalar;
use symbi_ir::graph::{ConstValue, ElementWiseOp, NodeId};
use symbi_ir::{Gv, GvKernel, begin_trace, end_trace, with_trace};

use super::gv::gv_load_at;

type Writes = Vec<(String, FieldBind, NodeId)>;

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
    let three = S::from_f64(3.0);
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
    let c1 = S::from_f64(1.0 / 12.0) * vm2 + S::from_f64(-5.0 / 4.0) * vm1
        + S::from_f64(5.0 / 4.0) * vc
        + S::from_f64(-1.0 / 12.0) * vp1;
    let c2 = S::from_f64(1.0 / 8.0) * vm2 + S::from_f64(1.0 / 4.0) * vm1 - vc
        + S::from_f64(3.0 / 4.0) * vp1
        + S::from_f64(-1.0 / 8.0) * vp2;
    let c3 = S::from_f64(-1.0 / 6.0) * vm2 + S::from_f64(1.0 / 2.0) * vm1
        + S::from_f64(-1.0 / 2.0) * vc
        + S::from_f64(1.0 / 6.0) * vp1;
    let c4 = S::from_f64(1.0 / 24.0) * vm2 + S::from_f64(-1.0 / 6.0) * vm1
        + S::from_f64(1.0 / 4.0) * vc
        + S::from_f64(-1.0 / 6.0) * vp1
        + S::from_f64(1.0 / 24.0) * vp2;
    // antiderivative difference over the sub-cell, scaled to an average.
    let anti = |x: S| {
        x * (c0
            + x * (c1 / two
                + x * (c2 / S::from_f64(3.0)
                    + x * (c3 / S::from_f64(4.0) + x * c4 / S::from_f64(5.0)))))
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
pub fn refine_restrict_gv(ndim: usize, ratio: i64) -> (GvKernel, Writes) {
    assert!(
        (1..=3).contains(&ndim),
        "refine_restrict_gv: ndim must be 1..=3"
    );
    assert!(ratio >= 2, "refine_restrict_gv: ratio must be >= 2");
    begin_trace();
    // the per-axis Refine map base: source[ax] = coord[ax] * ratio (+ child offset).
    let scaled: Vec<NodeId> = with_trace(|t| {
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
    let val = restrict_eval(&scaled, ratio, inv_r, ndim as isize - 1, &mut [0; 3]);
    let writes = vec![("dst".to_string(), "dst".into(), val.node())];
    (end_trace(), writes)
}

/// the inlined restrict sweep: axis `ax` averages `ratio` child values, each
/// recursively restricted through the lower axes (axis 0 innermost = the
/// reference's first pass). leaf = the fine child read at `scaled + off`.
fn restrict_eval(scaled: &[NodeId], ratio: i64, inv_r: Gv, ax: isize, off: &mut [i64; 3]) -> Gv {
    if ax < 0 {
        let coords: Vec<NodeId> = with_trace(|t| {
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
        return gv_load_at("src", "src", &coords);
    }
    let aa = ax as usize;
    off[aa] = 0;
    let mut sum = restrict_eval(scaled, ratio, inv_r, ax - 1, off);
    for kk in 1..ratio {
        off[aa] = kk;
        sum = sum + restrict_eval(scaled, ratio, inv_r, ax - 1, off);
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
pub fn field_copy_gv(ndim: usize) -> (GvKernel, Writes) {
    assert!((1..=3).contains(&ndim), "field_copy_gv: ndim must be 1..=3");
    begin_trace();
    let v = Gv::field("src", "src");
    let _ = ndim;
    let writes = vec![("dst".to_string(), "dst".into(), v.node())];
    (end_trace(), writes)
}

/// dst = value (a runtime scalar), pointwise — the register zero.
pub fn field_fill_gv(ndim: usize) -> (GvKernel, Writes) {
    assert!((1..=3).contains(&ndim), "field_fill_gv: ndim must be 1..=3");
    begin_trace();
    let v = Gv::scalar("value");
    let _ = ndim;
    let writes = vec![("dst".to_string(), "dst".into(), v.node())];
    (end_trace(), writes)
}

/// dst(c) += scale * src(c + arg), with per-axis runtime integer offsets (the
/// lattice-style args) and a runtime scalar `scale`. serves the register's
/// coarse-flux accumulation (arg = 0, scale = -A*w) and the reflux apply
/// (src = the register face read at the cell's adjacent face, scale = sign/V).
pub fn field_axpy_shift_gv(ndim: usize) -> (GvKernel, Writes) {
    assert!(
        (1..=3).contains(&ndim),
        "field_axpy_shift_gv: ndim must be 1..=3"
    );
    begin_trace();
    let src_coords: Vec<NodeId> = with_trace(|t| {
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
    let scale = Gv::scalar("scale");
    let dst = Gv::field("dst", "dst");
    let src = gv_load_at("src", "src", &src_coords);
    let v = dst + scale * src;
    let writes = vec![("dst".to_string(), "dst".into(), v.node())];
    (end_trace(), writes)
}

/// dst(c) += scale * sum over the `ratio^(D-1)` fine faces coincident with
/// coarse face c (transverse child offsets, the normal index scaling exactly)
/// — the register's fine-flux accumulation. `scale` carries the fine face
/// area times the stage weight.
pub fn refine_acc_face_gv(ndim: usize, ratio: i64, axis: usize) -> (GvKernel, Writes) {
    assert!(
        (1..=3).contains(&ndim),
        "refine_acc_face_gv: ndim must be 1..=3"
    );
    assert!(
        axis < ndim,
        "refine_acc_face_gv: axis {axis} out of range for ndim {ndim}"
    );
    assert!(ratio >= 2, "refine_acc_face_gv: ratio must be >= 2");
    begin_trace();
    let scaled: Vec<NodeId> = with_trace(|t| {
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
    let scale = Gv::scalar("scale");
    let dst = Gv::field("dst", "dst");
    let sum = acc_face_sum(&scaled, ratio, axis, ndim as isize - 1, &mut [0; 3]);
    let v = dst + scale * sum;
    let writes = vec![("dst".to_string(), "dst".into(), v.node())];
    (end_trace(), writes)
}

/// dst(g) += scale * sum over the `ratio` fine sub-edges of coarse edge g
/// (child offsets along the edge axis only; every other index scales exactly)
/// — the emf register's fine accumulation. `scale` carries the fine dt times
/// the length-average factor 1/ratio.
pub fn refine_acc_edge_gv(ndim: usize, ratio: i64, axis: usize) -> (GvKernel, Writes) {
    assert!(
        (1..=3).contains(&ndim),
        "refine_acc_edge_gv: ndim must be 1..=3"
    );
    assert!(
        axis < ndim,
        "refine_acc_edge_gv: axis {axis} out of range for ndim {ndim}"
    );
    assert!(ratio >= 2, "refine_acc_edge_gv: ratio must be >= 2");
    begin_trace();
    let scaled: Vec<NodeId> = with_trace(|t| {
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
    let scale = Gv::scalar("scale");
    let dst = Gv::field("dst", "dst");
    let load = |off_axis: i64| -> Gv {
        let coords: Vec<NodeId> = with_trace(|t| {
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
        gv_load_at("src", "src", &coords)
    };
    let mut sum = load(0);
    for kk in 1..ratio {
        sum = sum + load(kk);
    }
    let v = dst + scale * sum;
    let writes = vec![("dst".to_string(), "dst".into(), v.node())];
    (end_trace(), writes)
}

/// the raw transverse child sum (the caller's scale carries the weights,
/// averaging included), normal axis passing through.
fn acc_face_sum(scaled: &[NodeId], ratio: i64, axis: usize, ax: isize, off: &mut [i64; 3]) -> Gv {
    if ax < 0 {
        let coords: Vec<NodeId> = with_trace(|t| {
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
        return gv_load_at("src", "src", &coords);
    }
    let aa = ax as usize;
    if aa == axis {
        off[aa] = 0;
        return acc_face_sum(scaled, ratio, axis, ax - 1, off);
    }
    off[aa] = 0;
    let mut sum = acc_face_sum(scaled, ratio, axis, ax - 1, off);
    for kk in 1..ratio {
        off[aa] = kk;
        sum = sum + acc_face_sum(scaled, ratio, axis, ax - 1, off);
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
pub fn refine_restrict_face_gv(ndim: usize, ratio: i64, axis: usize) -> (GvKernel, Writes) {
    assert!(
        (1..=3).contains(&ndim),
        "refine_restrict_face_gv: ndim must be 1..=3"
    );
    assert!(
        axis < ndim,
        "refine_restrict_face_gv: axis {axis} out of range for ndim {ndim}"
    );
    assert!(ratio >= 2, "refine_restrict_face_gv: ratio must be >= 2");
    begin_trace();
    let scaled: Vec<NodeId> = with_trace(|t| {
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
    let val = restrict_face_eval(&scaled, ratio, inv_r, axis, ndim as isize - 1, &mut [0; 3]);
    let writes = vec![("dst".to_string(), "dst".into(), val.node())];
    (end_trace(), writes)
}

/// the inlined transverse restrict sweep: identical to `restrict_eval` except
/// the face-normal axis passes through unaveraged (its fine index is exactly
/// `ratio * coord`).
fn restrict_face_eval(
    scaled: &[NodeId],
    ratio: i64,
    inv_r: Gv,
    axis: usize,
    ax: isize,
    off: &mut [i64; 3],
) -> Gv {
    if ax < 0 {
        let coords: Vec<NodeId> = with_trace(|t| {
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
        return gv_load_at("src", "src", &coords);
    }
    let aa = ax as usize;
    if aa == axis {
        off[aa] = 0;
        return restrict_face_eval(scaled, ratio, inv_r, axis, ax - 1, off);
    }
    off[aa] = 0;
    let mut sum = restrict_face_eval(scaled, ratio, inv_r, axis, ax - 1, off);
    for kk in 1..ratio {
        off[aa] = kk;
        sum = sum + restrict_face_eval(scaled, ratio, inv_r, axis, ax - 1, off);
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
pub fn refine_prolong_gv(ndim: usize, ratio: i64, order: ProlongOrder) -> (GvKernel, Writes) {
    assert!(
        (1..=3).contains(&ndim),
        "refine_prolong_gv: ndim must be 1..=3"
    );
    assert!(ratio >= 2, "refine_prolong_gv: ratio must be >= 2");
    begin_trace();
    let alpha = Gv::scalar("alpha");
    let geom = prolong_geometry(ndim, ratio);
    let src = ProlongSrc::TimePair {
        old: "src_old",
        new: "src_new",
        alpha,
    };
    let ctx = geom.ctx(ndim, order, &src);
    let val = prolong_eval(&ctx, ndim as isize - 1, &mut [0; 3]);
    let writes = vec![("dst".to_string(), "dst".into(), val.node())];
    (end_trace(), writes)
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
) -> (GvKernel, Writes) {
    assert!(
        (1..=3).contains(&ndim),
        "refine_prolong_multi_gv: ndim must be 1..=3"
    );
    assert!(ratio >= 2, "refine_prolong_multi_gv: ratio must be >= 2");
    assert!(ncomp >= 1, "refine_prolong_multi_gv: ncomp must be >= 1");
    begin_trace();
    let alpha = Gv::scalar("alpha");
    let geom = prolong_geometry(ndim, ratio);
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
        let val = prolong_eval(&ctx, ndim as isize - 1, &mut [0; 3]);
        writes.push((dst_name.clone(), dst_name.into(), val.node()));
    }
    (end_trace(), writes)
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
) -> (GvKernel, Writes) {
    assert!(
        (1..=3).contains(&ndim),
        "refine_prolong_multi_1t_gv: ndim must be 1..=3"
    );
    assert!(ratio >= 2, "refine_prolong_multi_1t_gv: ratio must be >= 2");
    assert!(ncomp >= 1, "refine_prolong_multi_1t_gv: ncomp must be >= 1");
    begin_trace();
    let geom = prolong_geometry(ndim, ratio);
    let mut writes = Vec::with_capacity(ncomp);
    for k in 0..ncomp {
        let (src_name, dst_name) = (format!("src_{k}"), format!("dst_{k}"));
        let src = ProlongSrc::Single { name: &src_name };
        let ctx = geom.ctx(ndim, order, &src);
        let val = prolong_eval(&ctx, ndim as isize - 1, &mut [0; 3]);
        writes.push((dst_name.clone(), dst_name.into(), val.node()));
    }
    (end_trace(), writes)
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
) -> (GvKernel, Writes) {
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
    begin_trace();
    // parent + parity on the swept axis only (the same arithmetic
    // prolong_geometry builds per axis).
    let (parent, parity): (NodeId, NodeId) = with_trace(|t| {
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
    let parity_f = Gv::of(parity);
    let frac = (parity_f + half) * inv_ratio - half;
    let xi_lo = parity_f * inv_ratio;
    let xi_hi = (parity_f + one) * inv_ratio;

    let w = order.ghost_width() as i64;
    let mut writes = Vec::with_capacity(ncomp);
    for k in 0..ncomp {
        let src_name = format!("src_{k}");
        // the stencil loads: swept axis at parent + dd, unswept axes at the
        // raw thread coordinate.
        let load = |dd: i64| -> Gv {
            let coords: Vec<NodeId> = with_trace(|t| {
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
            gv_load_at(&src_name, src_name.as_str(), &coords)
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
        writes.push((dst_name.clone(), dst_name.into(), val.node()));
    }
    (end_trace(), writes)
}

/// the pointwise time interpolation `dst_k = (1 - alpha)*src_old_k +
/// alpha*src_new_k` over `ncomp` co-located fields in one dispatch — the pass
/// that hoists the prolong leaf's per-fine-cell lerp to once per coarse cell.
/// the expression is spelled exactly as the prolong leaf spelled it, so the
/// lerp-then-prolong-1t chain is bit-identical to the fused time-pair kernel.
/// buffers: src_old_0, src_new_0, .., interleaved (inputs) then dst_0..
/// (outputs); scalar "alpha".
pub fn field_lerp_multi_gv(ndim: usize, ncomp: usize) -> (GvKernel, Writes) {
    assert!(
        (1..=3).contains(&ndim),
        "field_lerp_multi_gv: ndim must be 1..=3"
    );
    assert!(ncomp >= 1, "field_lerp_multi_gv: ncomp must be >= 1");
    begin_trace();
    let alpha = Gv::scalar("alpha");
    let one = Gv::from_f64(1.0);
    let _ = ndim;
    let mut writes = Vec::with_capacity(ncomp);
    for k in 0..ncomp {
        let old_key = format!("src_old_{k}");
        let new_key = format!("src_new_{k}");
        let v_old = Gv::field(&old_key, old_key.as_str());
        let v_new = Gv::field(&new_key, new_key.as_str());
        let v = (one - alpha) * v_old + alpha * v_new;
        let dst_name = format!("dst_{k}");
        writes.push((dst_name.clone(), dst_name.into(), v.node()));
    }
    (end_trace(), writes)
}

// =============================================================================
// balance-aware coarse-fine transfer: the fused lerp+encode over the coarse
// parent region and the decode over the fine ghost slab. between them the
// unchanged prolong kernels act on departures from one hydrostatic anchor, so
// coarse stencil data on one isentrope land the fine ghosts exactly back on it
// at any prolongation order and any limiter. cartesian, gamma-law only, like
// every balance-carrying kernel.
// =============================================================================

/// the per-slot body scalars a balance-aware transfer kernel declares:
/// `body_{b}_pos_{g}` per grid axis, then `body_{b}_mass` / `_soft` /
/// `_softkind` — the same slot layout the wb ghost fill and body source bind
/// (an inert slot carries mass = 0 and contributes exactly zero potential).
/// declared eagerly so the scalar manifest order follows the declaration order
/// here, independent of the trace's evaluation order.
fn declare_body_slots(ndim: usize, n_bodies: usize) -> Vec<([Gv; 3], Gv, Gv, Gv)> {
    (0..n_bodies)
        .map(|b| {
            let mut pos = [Gv::from_f64(0.0), Gv::from_f64(0.0), Gv::from_f64(0.0)];
            for (g, p) in pos.iter_mut().enumerate().take(ndim) {
                *p = Gv::scalar(&format!("body_{b}_pos_{g}"));
            }
            (
                pos,
                Gv::scalar(&format!("body_{b}_mass")),
                Gv::scalar(&format!("body_{b}_soft")),
                Gv::scalar(&format!("body_{b}_softkind")),
            )
        })
        .collect()
}

/// the total body potential at a cartesian position (grid axes are the
/// coordinates; ungridded components zero), summed over every slot.
fn body_slots_potential(pos: &[Gv; 3], slots: &[([Gv; 3], Gv, Gv, Gv)]) -> Gv {
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
fn centroid_position(ndim: usize, coords: &[NodeId], x_lo: &[Gv], dx: &[Gv]) -> [Gv; 3] {
    let half = Gv::from_f64(0.5);
    let mut pos = [Gv::from_f64(0.0), Gv::from_f64(0.0), Gv::from_f64(0.0)];
    for g in 0..ndim {
        pos[g] = x_lo[g] + (Gv::of(coords[g]) + half) * dx[g];
    }
    pos
}

/// trace the fused lerp + hydrostatic encode over the coarse parent region of a
/// coarse-fine ghost slab: every component is time-interpolated
/// `(1 - alpha)*src_old_k + alpha*src_new_k`, and rho (component 0) and pre
/// (component ncomp-1) additionally subtract the anchor equilibrium evaluated
/// at the cell's own potential. the anchor (rho, pre) is re-lerped in-thread
/// from the raw inputs at the `anchor_{ax}` index scalars, so each thread builds
/// the anchor from immutable input, independent of the anchor cell's encode. the
/// prolong kernels then act on the resulting departures unchanged.
///
/// buffers: src_old_0, src_new_0, .., interleaved (inputs) then dst_0..
/// (outputs). ints: anchor_{ax}. scalars: alpha, gamma, x_lo_{ax}, dx_{ax}
/// (the coarse lattice), then the body slots.
pub fn wb_cf_lerp_encode_gv(ndim: usize, ncomp: usize, n_bodies: usize) -> (GvKernel, Writes) {
    use symbi_hydro::hydrostatic::LocalEquilibrium;
    assert!(
        (1..=3).contains(&ndim),
        "wb_cf_lerp_encode_gv: ndim must be 1..=3"
    );
    assert!(
        ncomp == ndim + 2,
        "wb_cf_lerp_encode_gv: the balance-aware transfer carries rho + ndim velocities + pre"
    );
    begin_trace();
    // scalar manifest in declaration order: alpha, gamma float; anchor int lane;
    // grid origin/step; body slots.
    let alpha = Gv::scalar("alpha");
    let gamma = Gv::scalar("gamma");
    let anchor: Vec<NodeId> = with_trace(|t| {
        (0..ndim)
            .map(|ax| t.scalar_int(&format!("anchor_{ax}")))
            .collect()
    });
    let x_lo: Vec<Gv> = (0..ndim)
        .map(|ax| Gv::scalar(&format!("x_lo_{ax}")))
        .collect();
    let dx: Vec<Gv> = (0..ndim).map(|ax| Gv::scalar(&format!("dx_{ax}"))).collect();
    let slots = declare_body_slots(ndim, n_bodies);

    let one = Gv::from_f64(1.0);
    // cell loads in interleaved (old_k, new_k) registration order — the buffer
    // order the dispatch binds.
    let lerp_cell: Vec<Gv> = (0..ncomp)
        .map(|k| {
            let old_key = format!("src_old_{k}");
            let new_key = format!("src_new_{k}");
            let v_old = Gv::field(&old_key, old_key.as_str());
            let v_new = Gv::field(&new_key, new_key.as_str());
            (one - alpha) * v_old + alpha * v_new
        })
        .collect();

    // the anchor state, re-lerped in-thread from the same inputs (keys already
    // registered — no new buffers).
    let lerp_anchor = |k: usize| -> Gv {
        let old_key = format!("src_old_{k}");
        let new_key = format!("src_new_{k}");
        let v_old = gv_load_at(&old_key, old_key.as_str(), &anchor);
        let v_new = gv_load_at(&new_key, new_key.as_str(), &anchor);
        (one - alpha) * v_old + alpha * v_new
    };
    let rho_a = lerp_anchor(0);
    let pre_a = lerp_anchor(ncomp - 1);

    let cell_coords: Vec<NodeId> =
        with_trace(|t| (0..ndim).map(|ax| t.coord(ax as u8)).collect());
    let phi_anchor = body_slots_potential(&centroid_position(ndim, &anchor, &x_lo, &dx), &slots);
    let phi_cell = body_slots_potential(&centroid_position(ndim, &cell_coords, &x_lo, &dx), &slots);
    let eq = LocalEquilibrium::through(rho_a, pre_a, phi_anchor, gamma);
    let (r_eq, p_eq) = eq.state_at(phi_cell);

    let mut writes = Vec::with_capacity(ncomp);
    for k in 0..ncomp {
        let val = if k == 0 {
            lerp_cell[k] - r_eq
        } else if k == ncomp - 1 {
            lerp_cell[k] - p_eq
        } else {
            lerp_cell[k]
        };
        let dst_name = format!("dst_{k}");
        writes.push((dst_name.clone(), dst_name.into(), val.node()));
    }
    (end_trace(), writes)
}

/// trace the hydrostatic decode over the fine coarse-fine ghost slab: the
/// prolonged departures already sit in the fine (rho, pre), and each ghost adds
/// back the anchor equilibrium evaluated at its own potential. the encoded
/// scratch holds a zero departure at the anchor, so the anchor state comes from
/// the raw coarse (rho, pre) snapshots, bound as inputs and re-lerped in-thread
/// with the same alpha the encode used.
///
/// buffers: src_old_rho, src_new_rho, src_old_pre, src_new_pre (the coarse
/// snapshots, inputs) then dst_rho, dst_pre (the fine ghosts, in-place
/// outputs). ints: anchor_{ax}. scalars: alpha, gamma, x_lo_{ax}, dx_{ax} (the
/// fine lattice), src_x_lo_{ax}, src_dx_{ax} (the coarse lattice), then the
/// body slots.
pub fn wb_cf_decode_gv(ndim: usize, n_bodies: usize) -> (GvKernel, Writes) {
    use symbi_hydro::hydrostatic::LocalEquilibrium;
    assert!((1..=3).contains(&ndim), "wb_cf_decode_gv: ndim must be 1..=3");
    begin_trace();
    let alpha = Gv::scalar("alpha");
    let gamma = Gv::scalar("gamma");
    let anchor: Vec<NodeId> = with_trace(|t| {
        (0..ndim)
            .map(|ax| t.scalar_int(&format!("anchor_{ax}")))
            .collect()
    });
    let fine_x_lo: Vec<Gv> = (0..ndim)
        .map(|ax| Gv::scalar(&format!("x_lo_{ax}")))
        .collect();
    let fine_dx: Vec<Gv> = (0..ndim).map(|ax| Gv::scalar(&format!("dx_{ax}"))).collect();
    let src_x_lo: Vec<Gv> = (0..ndim)
        .map(|ax| Gv::scalar(&format!("src_x_lo_{ax}")))
        .collect();
    let src_dx: Vec<Gv> = (0..ndim)
        .map(|ax| Gv::scalar(&format!("src_dx_{ax}")))
        .collect();
    let slots = declare_body_slots(ndim, n_bodies);

    let one = Gv::from_f64(1.0);
    // the coarse snapshots register first, in the dispatch's input order.
    let lerp_anchor = |name: &str| -> Gv {
        let old_key = format!("src_old_{name}");
        let new_key = format!("src_new_{name}");
        let v_old = gv_load_at(&old_key, old_key.as_str(), &anchor);
        let v_new = gv_load_at(&new_key, new_key.as_str(), &anchor);
        (one - alpha) * v_old + alpha * v_new
    };
    let rho_a = lerp_anchor("rho");
    let pre_a = lerp_anchor("pre");

    let cell_coords: Vec<NodeId> =
        with_trace(|t| (0..ndim).map(|ax| t.coord(ax as u8)).collect());
    let phi_anchor =
        body_slots_potential(&centroid_position(ndim, &anchor, &src_x_lo, &src_dx), &slots);
    let phi_fine = body_slots_potential(
        &centroid_position(ndim, &cell_coords, &fine_x_lo, &fine_dx),
        &slots,
    );
    let eq = LocalEquilibrium::through(rho_a, pre_a, phi_anchor, gamma);
    let (r_eq, p_eq) = eq.state_at(phi_fine);

    // in-place: the prolonged departures already sit in the ghosts.
    let rho_g = Gv::field("dst_rho", "dst_rho") + r_eq;
    let pre_g = Gv::field("dst_pre", "dst_pre") + p_eq;
    let writes = vec![
        ("dst_rho".to_string(), "dst_rho".into(), rho_g.node()),
        ("dst_pre".to_string(), "dst_pre".into(), pre_g.node()),
    ];
    (end_trace(), writes)
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
pub fn refine_prolong_face_gv(ndim: usize, ratio: i64, axis: usize) -> (GvKernel, Writes) {
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
    begin_trace();
    let alpha = Gv::scalar("alpha");
    let one = Gv::from_f64(1.0);

    // per-axis maps: the normal axis gets the face pair (lo, hi); transverse
    // axes the cell parent + plm frac, exactly as the cell prolongation.
    let (pair_lo, pair_hi, parent, parity): (Vec<NodeId>, Vec<NodeId>, Vec<NodeId>, Vec<NodeId>) =
        with_trace(|t| {
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
        .map(|&q| (Gv::of(q) + half) * inv_ratio - half)
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
    let val = face_prolong_eval(&ctx, ndim as isize - 1, &mut [0; 3]);
    let writes = vec![("dst".to_string(), "dst".into(), val.node())];
    (end_trace(), writes)
}

struct FaceProlongCtx<'a> {
    ndim: usize,
    axis: usize,
    pair_lo: &'a [NodeId],
    pair_hi: &'a [NodeId],
    parent: &'a [NodeId],
    frac: &'a [Gv],
    one: Gv,
    alpha: Gv,
}

/// the inlined face-prolong sweep: transverse axes interpolate with plm (axis
/// 0 innermost among them), the normal axis passes through to the leaf, which
/// averages the time-interpolated coarse face pair.
fn face_prolong_eval(ctx: &FaceProlongCtx, ax: isize, off: &mut [i64; 3]) -> Gv {
    if ax < 0 {
        let load = |normals: &[NodeId]| -> Gv {
            let coords: Vec<NodeId> = with_trace(|t| {
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
            let v_old = gv_load_at("src_old", "src_old", &coords);
            let v_new = gv_load_at("src_new", "src_new", &coords);
            (ctx.one - ctx.alpha) * v_old + ctx.alpha * v_new
        };
        let half = Gv::from_f64(0.5);
        return half * (load(ctx.pair_lo) + load(ctx.pair_hi));
    }
    let aa = ax as usize;
    if aa == ctx.axis {
        return face_prolong_eval(ctx, ax - 1, off);
    }
    let vals: Vec<Gv> = (-1..=1i64)
        .map(|dd| {
            off[aa] = dd;
            face_prolong_eval(ctx, ax - 1, off)
        })
        .collect();
    off[aa] = 0;
    plm_interp(vals[0], vals[1], vals[2], ctx.frac[aa])
}

/// the coarse-field read the prolong leaf performs: the classic time pair
/// (`(1 - alpha)*old + alpha*new` per coarse cell, recomputed per fine cell)
/// or a single pre-interpolated buffer (a `field_lerp` pass hoisted the time
/// interpolation to once per coarse cell — half the gather traffic).
enum ProlongSrc<'a> {
    TimePair {
        old: &'a str,
        new: &'a str,
        alpha: Gv,
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
struct ProlongGeom {
    parent: Vec<NodeId>,
    frac: Vec<Gv>,
    xi_lo: Vec<Gv>,
    xi_hi: Vec<Gv>,
    ratio_f: Gv,
    one: Gv,
}

fn prolong_geometry(ndim: usize, ratio: i64) -> ProlongGeom {
    let one = Gv::from_f64(1.0);
    let (parent, parity): (Vec<NodeId>, Vec<NodeId>) = with_trace(|t| {
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
    let parity_f: Vec<Gv> = parity.iter().map(|&q| Gv::of(q)).collect();
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

impl ProlongGeom {
    fn ctx<'a>(
        &'a self,
        ndim: usize,
        order: ProlongOrder,
        src: &'a ProlongSrc<'a>,
    ) -> ProlongCtx<'a> {
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

struct ProlongCtx<'a> {
    ndim: usize,
    order: ProlongOrder,
    parent: &'a [NodeId],
    frac: &'a [Gv],
    xi_lo: &'a [Gv],
    xi_hi: &'a [Gv],
    ratio_f: Gv,
    one: Gv,
    src: &'a ProlongSrc<'a>,
}

/// the inlined prolong sweep: axis `ax` interpolates the values recursively
/// prolonged through the lower axes at the stencil offsets (axis 0 innermost =
/// the reference's first pass). leaf = the coarse read at `parent + off`
/// (time-interpolated for a TimePair source, plain for a pre-lerped Single).
fn prolong_eval(ctx: &ProlongCtx, ax: isize, off: &mut [i64; 3]) -> Gv {
    if ax < 0 {
        let coords: Vec<NodeId> = with_trace(|t| {
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
                let v_old = gv_load_at(old, *old, &coords);
                let v_new = gv_load_at(new, *new, &coords);
                (ctx.one - *alpha) * v_old + *alpha * v_new
            }
            ProlongSrc::Single { name } => gv_load_at(name, *name, &coords),
        };
    }
    let aa = ax as usize;
    let w = ctx.order.ghost_width() as i64;
    let vals: Vec<Gv> = (-w..=w)
        .map(|dd| {
            off[aa] = dd;
            prolong_eval(ctx, ax - 1, off)
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
        assert!(resid < 1e-14, "children do not conserve the parent: resid {resid:e}");
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
