// =============================================================================
// gv_refinement.rs
//
// the amr field-transfer kernels (docs/design/21, 22 phase 1): restriction
// (fine -> coarse conservative child average) and prolongation (coarse -> fine
// limited interpolation, time-interpolated between two coarse snapshots) as
// gv-traced pullbacks over the refinement lattice maps (lattice.rs
// Refine / Coarsen). levels share ABSOLUTE index space: fine cell f covers
// coarse cell floor_div(f, ratio), so no coverage-relative translation exists
// anywhere — the destination thread coordinate IS the level-global index, and
// each field buffer resolves it against its own lo.
//
// the gen-1 reference (symbi-amr prolong_nd / restrict_nd at git 3bfc5b9) is an
// axis-by-axis sweep with scratch buffers — host-only by construction. here the
// sweep is INLINED per destination cell: pass order is axis 0 innermost, the
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
use symbi_ir::algebra::Scalar;
use symbi_ir::FieldBind;
use symbi_ir::graph::{ConstValue, ElementWiseOp, NodeId};
use symbi_ir::{begin_trace, end_trace, with_trace, Gv, GvKernel};

use super::gv::gv_load_at;

type Writes = Vec<(String, FieldBind, NodeId)>;

/// prolongation order for the coarse-fine transfer. one order higher than the
/// evolution reconstruction (pcm evolution -> plm prolong, plm -> ppm) so the
/// coarse-fine boundary does not degrade the interior order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProlongOrder {
    /// piecewise constant (order 0): parent injection. stencil halfwidth 0.
    Pcm,
    /// piecewise linear with van leer limiting (order 1). stencil halfwidth 1.
    Plm,
    /// piecewise parabolic with monotonicity (order 2), exact sub-cell averages
    /// (conservative by construction). stencil halfwidth 2.
    Ppm,
}

impl ProlongOrder {
    /// coarse ghost cells required per side by the 1d stencil.
    pub fn ghost_width(self) -> usize {
        match self {
            ProlongOrder::Pcm => 0,
            ProlongOrder::Plm => 1,
            ProlongOrder::Ppm => 2,
        }
    }
}

// =============================================================================
// carrier-generic stencil math (identical arithmetic to the gen-1 reference,
// branches rewritten as cmp/select — same value at f64, traceable at Gv)
// =============================================================================

/// van leer (harmonic) limited slope: `2*dl*dr/(dl+dr)` when the one-sided
/// differences share a strict sign, zero otherwise. the denominator is guarded
/// BEFORE the select (Gv evaluates both arms; a same-signed pair has a nonzero
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

/// the 1d ppm prolongation sub-cell AVERAGE over [xi_lo, xi_hi] (xi in [0,1]
/// across the parent): 4th-order interface values clamped to the neighbour
/// range, monotonized (the select form preserves the reference's sequential
/// left-then-right overshoot correction), then the exact parabola
/// antiderivative difference times `ratio` — so the children average back to
/// the parent exactly (conservation by construction).
fn ppm_interp<S: Scalar>(
    vm2: S, vm1: S, vc: S, vp1: S, vp2: S,
    xi_lo: S, xi_hi: S, ratio: S,
) -> S {
    let two = S::ONE + S::ONE;
    let half = S::ONE / two;
    let three = S::from_f64(3.0);
    let six = S::from_f64(6.0);
    let seven = S::from_f64(7.0);
    let twelve_inv = S::ONE / S::from_f64(12.0);

    let u_l = (seven * (vm1 + vc) - (vm2 + vp1)) * twelve_inv;
    let u_r = (seven * (vc + vp1) - (vm1 + vp2)) * twelve_inv;

    // clamp interface values to the neighbour range before monotonizing — the
    // 4th-order stencil overshoots at discontinuities.
    let u_l = u_l.max(vm1.min(vc)).min(vm1.max(vc));
    let u_r = u_r.max(vc.min(vp1)).min(vc.max(vp1));

    // monotonicity: flatten at a local extremum, else correct overshoots. the
    // right correction reads the corrected left value (sequential dependency,
    // matching the reference); diff/curv read the pre-correction interfaces.
    let extremum = ((u_r - vc) * (vc - u_l)).cmp_le(S::ZERO);
    let diff = u_r - u_l;
    let curv = six * (vc - (u_l + u_r) / two);
    let a_l = S::select((diff * curv).cmp_gt(diff * diff), three * vc - two * u_r, u_l);
    let a_r = S::select(
        (diff * curv).cmp_lt(S::ZERO - diff * diff),
        three * vc - two * a_l,
        u_r,
    );
    let a_l = S::select(extremum, vc, a_l);
    let a_r = S::select(extremum, vc, a_r);

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

// =============================================================================
// restriction: thread over the coarse coverage, read the ratio^D fine children
// =============================================================================

/// trace the fine -> coarse restriction: `dst[c] = sweep-average of
/// src[ratio*c + o]` over the child offsets, axis 0 innermost (the reference
/// restrict_nd pass order). input "src" is the FINE field, output "dst" the
/// COARSE field; the dispatch domain is the coarse coverage in absolute coarse
/// indices.
pub fn refine_restrict_gv(ndim: usize, ratio: i64) -> (GvKernel, Writes) {
    assert!((1..=3).contains(&ndim), "refine_restrict_gv: ndim must be 1..=3");
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

/// dst(c) += scale * src(c + arg), with per-axis runtime INTEGER offsets (the
/// lattice-style args) and a runtime scalar `scale`. serves the register's
/// coarse-flux accumulation (arg = 0, scale = -A*w) and the reflux apply
/// (src = the register face read at the cell's adjacent face, scale = sign/V).
pub fn field_axpy_shift_gv(ndim: usize) -> (GvKernel, Writes) {
    assert!((1..=3).contains(&ndim), "field_axpy_shift_gv: ndim must be 1..=3");
    begin_trace();
    let src_coords: Vec<NodeId> = with_trace(|t| {
        let coords: Vec<NodeId> = (0..ndim).map(|ax| t.coord(ax as u8)).collect();
        let args: Vec<NodeId> =
            (0..ndim).map(|ax| t.scalar_int(&format!("arg_{ax}"))).collect();
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
    assert!((1..=3).contains(&ndim), "refine_acc_face_gv: ndim must be 1..=3");
    assert!(axis < ndim, "refine_acc_face_gv: axis {axis} out of range for ndim {ndim}");
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
/// (child offsets along the EDGE axis only; every other index scales exactly)
/// — the emf register's fine accumulation. `scale` carries the fine dt times
/// the length-average factor 1/ratio.
pub fn refine_acc_edge_gv(ndim: usize, ratio: i64, axis: usize) -> (GvKernel, Writes) {
    assert!((1..=3).contains(&ndim), "refine_acc_edge_gv: ndim must be 1..=3");
    assert!(axis < ndim, "refine_acc_edge_gv: axis {axis} out of range for ndim {ndim}");
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

/// the raw transverse child SUM (no averaging — the caller's scale carries
/// the weights), normal axis passing through.
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
// face restriction: thread over the coarse coverage FACE domain, read the
// ratio^(D-1) coincident fine faces (staggered fields — amr phase 3)
// =============================================================================

/// trace the fine -> coarse FACE restriction for a face-normal staggered field
/// (bface[axis]): `dst[c] = transverse-sweep-average of src[ratio*c + o]` where
/// the child offsets `o` run over the transverse axes only — the normal index
/// scales exactly (a coarse face is the union of its ratio^(D-1) fine faces,
/// area-weighted average = plain average on a uniform cartesian grid). input
/// "src" is the FINE face field, output "dst" the COARSE one; the dispatch
/// domain is the coverage face domain in absolute coarse indices.
pub fn refine_restrict_face_gv(ndim: usize, ratio: i64, axis: usize) -> (GvKernel, Writes) {
    assert!((1..=3).contains(&ndim), "refine_restrict_face_gv: ndim must be 1..=3");
    assert!(axis < ndim, "refine_restrict_face_gv: axis {axis} out of range for ndim {ndim}");
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
// parent neighbourhood time-interpolated between two snapshots
// =============================================================================

/// trace the coarse -> fine prolongation: each fine cell reads its coarse
/// parent neighbourhood (`floor_div(f, ratio)` — the Coarsen map, absolute
/// indices, ghost-safe for negatives) from the time-interpolated coarse state
/// `(1 - alpha)*src_old + alpha*src_new`, then applies the inlined per-axis
/// sweep at `order`. inputs "src_old"/"src_new" are the COARSE field snapshots
/// (bind the same buffer twice when no time interpolation is wanted), scalar
/// "alpha" the interpolation fraction; output "dst" is the FINE field. the
/// dispatch domain is the fine destination region (a coarse-fine ghost slab,
/// or a freshly nested patch interior) in absolute fine indices.
pub fn refine_prolong_gv(ndim: usize, ratio: i64, order: ProlongOrder) -> (GvKernel, Writes) {
    assert!((1..=3).contains(&ndim), "refine_prolong_gv: ndim must be 1..=3");
    assert!(ratio >= 2, "refine_prolong_gv: ratio must be >= 2");
    begin_trace();
    let alpha = Gv::scalar("alpha");
    let one = Gv::from_f64(1.0);

    // the per-axis Coarsen map + the child parity: parent = floor_div(c, r),
    // parity = c - parent*r in 0..r.
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

    // the reference's per-pass sub-cell positions from the parity kk (int
    // promoted to f64 by the graph): plm midpoint frac = (kk + 1/2)/r - 1/2,
    // ppm sub-cell average bounds xi_lo = kk/r, xi_hi = (kk + 1)/r.
    let half = Gv::from_f64(0.5);
    let inv_ratio = Gv::from_f64(1.0 / ratio as f64);
    let ratio_f = Gv::from_f64(ratio as f64);
    let parity_f: Vec<Gv> = parity.iter().map(|&q| Gv::of(q)).collect();
    let frac: Vec<Gv> = parity_f.iter().map(|&q| (q + half) * inv_ratio - half).collect();
    let xi_lo: Vec<Gv> = parity_f.iter().map(|&q| q * inv_ratio).collect();
    let xi_hi: Vec<Gv> = parity_f.iter().map(|&q| (q + one) * inv_ratio).collect();

    let ctx = ProlongCtx {
        ndim, order, parent: &parent, frac: &frac, xi_lo: &xi_lo, xi_hi: &xi_hi,
        ratio_f, one, alpha, old_name: "src_old", new_name: "src_new",
    };
    let val = prolong_eval(&ctx, ndim as isize - 1, &mut [0; 3]);
    let writes = vec![("dst".to_string(), "dst".into(), val.node())];
    (end_trace(), writes)
}

/// trace the MULTI-FIELD (prim batch) cell prolongation: ONE kernel that sweeps
/// the shared coarse->fine stencil over `ncomp` co-located fields, reading
/// `src_old_{k}`/`src_new_{k}` and writing `dst_{k}` for k in 0..ncomp. the
/// per-cell geometry (parent index, parity, plm/ppm weights) is computed ONCE
/// and reused across all components (graph CSE), and — the point — the host
/// issues ONE dispatch (one rayon launch) for the whole prim set instead of
/// `ncomp` separate launches. bit-identical to `ncomp` single-field prolongs.
/// buffers in signature order: src_old_0, src_new_0, .., src_old_{n-1},
/// src_new_{n-1} (inputs) then dst_0..dst_{n-1} (outputs); scalar "alpha".
pub fn refine_prolong_multi_gv(
    ndim: usize,
    ratio: i64,
    order: ProlongOrder,
    ncomp: usize,
) -> (GvKernel, Writes) {
    assert!((1..=3).contains(&ndim), "refine_prolong_multi_gv: ndim must be 1..=3");
    assert!(ratio >= 2, "refine_prolong_multi_gv: ratio must be >= 2");
    assert!(ncomp >= 1, "refine_prolong_multi_gv: ncomp must be >= 1");
    begin_trace();
    let alpha = Gv::scalar("alpha");
    let one = Gv::from_f64(1.0);

    // shared per-cell geometry: parent (coarse) index + child parity per axis —
    // identical for every component, CSE'd across them.
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
    let frac: Vec<Gv> = parity_f.iter().map(|&q| (q + half) * inv_ratio - half).collect();
    let xi_lo: Vec<Gv> = parity_f.iter().map(|&q| q * inv_ratio).collect();
    let xi_hi: Vec<Gv> = parity_f.iter().map(|&q| (q + one) * inv_ratio).collect();

    let mut writes = Vec::with_capacity(ncomp);
    for k in 0..ncomp {
        let (old_name, new_name, dst_name) =
            (format!("src_old_{k}"), format!("src_new_{k}"), format!("dst_{k}"));
        let ctx = ProlongCtx {
            ndim, order, parent: &parent, frac: &frac, xi_lo: &xi_lo, xi_hi: &xi_hi,
            ratio_f, one, alpha, old_name: &old_name, new_name: &new_name,
        };
        let val = prolong_eval(&ctx, ndim as isize - 1, &mut [0; 3]);
        writes.push((dst_name.clone(), dst_name.into(), val.node()));
    }
    (end_trace(), writes)
}

// =============================================================================
// face prolongation: thread over a fine FACE region (a bface transverse-halo
// slab at a coarse-fine boundary), read the time-interpolated coarse face
// field — amr phase 3 follow-up (the fine boundary-edge EMF quality fix)
// =============================================================================

/// trace the coarse -> fine FACE prolongation for a face-normal staggered
/// field (bface[axis]). along the NORMAL axis the fine face lattice
/// interleaves the coarse one: an even fine face (2c) coincides with coarse
/// face c, an odd one (2c+1) sits at the midpoint of faces c and c+1 — the
/// pair `floor_div(f, 2)` / `floor_div(f+1, 2)` collapses to (c, c) on even
/// faces (the half-sum is then exact) and (c, c+1) on odd ones, so no read
/// ever leaves the coarse face domain. TRANSVERSE axes use the van-leer plm
/// sweep (axis 0 innermost among them): the coarse bface carries only a +/-1
/// transverse halo, so the ppm stencil's reach is structurally unavailable —
/// plm is the maximum order here, one above the pcm a plain copy would be.
/// inputs "src_old"/"src_new" + scalar "alpha" as in `refine_prolong_gv`.
pub fn refine_prolong_face_gv(ndim: usize, ratio: i64, axis: usize) -> (GvKernel, Writes) {
    assert!((1..=3).contains(&ndim), "refine_prolong_face_gv: ndim must be 1..=3");
    assert!(axis < ndim, "refine_prolong_face_gv: axis {axis} out of range for ndim {ndim}");
    assert!(ratio == 2, "refine_prolong_face_gv: the face-lattice midpoint pair is ratio-2");
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
                        let base = if kk == ctx.axis { normals[kk] } else { ctx.parent[kk] };
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

struct ProlongCtx<'a> {
    ndim: usize,
    order: ProlongOrder,
    parent: &'a [NodeId],
    frac: &'a [Gv],
    xi_lo: &'a [Gv],
    xi_hi: &'a [Gv],
    ratio_f: Gv,
    one: Gv,
    alpha: Gv,
    // the coarse-field buffer names this sweep reads. single-field prolong uses
    // "src_old"/"src_new"; the multi-field (prim batch) kernel sweeps the SAME
    // geometry over "src_old_{k}"/"src_new_{k}" per component.
    old_name: &'a str,
    new_name: &'a str,
}

/// the inlined prolong sweep: axis `ax` interpolates the values recursively
/// prolonged through the lower axes at the stencil offsets (axis 0 innermost =
/// the reference's first pass). leaf = the time-interpolated coarse read at
/// `parent + off`.
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
        let v_old = gv_load_at(ctx.old_name, ctx.old_name, &coords);
        let v_new = gv_load_at(ctx.new_name, ctx.new_name, &coords);
        return (ctx.one - ctx.alpha) * v_old + ctx.alpha * v_new;
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
            vals[0], vals[1], vals[2], vals[3], vals[4],
            ctx.xi_lo[aa], ctx.xi_hi[aa], ctx.ratio_f,
        ),
    }
}
