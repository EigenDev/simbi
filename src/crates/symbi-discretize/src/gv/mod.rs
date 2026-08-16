// =============================================================================
// gv/mod.rs
//
// the Gv kernel builders: each `*_gv` fn instantiates a carrier-generic
// symbi-hydro physics function (written over `S: Scalar`) at `S = Gv` and traces
// it into a stencil DAG — the dispatchable kernel (graph + ABI manifest). the
// `Gv` carrier + the trace itself live in `symbi-ir`; this module is the
// discretization layer that drives it: it picks coords/spacing/reconstruction
// (the numerical choices) and builds c2p / flux / godunov / wave-speed / CT /
// ghost-fill / geometry kernels. `S = f64`
// gives the host body; `S = Gv` gives the kernel graph — one physics source.
//
// raw index/stencil IR (integer coord arithmetic, lattice-map boundary source,
// multi-axis load_at) is built directly against `symbi_ir::with_trace`; the
// f64 `Gv` carrier stays a floating-point carrier and leaves integer addressing there.
// =============================================================================

use symbi_algebra::Tensor;
use symbi_algebra::algebra::Numeric;
use symbi_hydro::ShockwaveLimiter;
use symbi_hydro::energy::Zero;
use symbi_hydro::eos::{IdealGas, Isothermal, TaubMathews};
use symbi_hydro::isothermal_mhd::{IsothermalMhd, imhd_recover};
use symbi_hydro::mhd_state::{IsoMhdCons, IsoMhdPrim, MhdCons, MhdPrim};
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::newtonian_mhd::{NewtonianMhd, nmhd_recover};
use symbi_hydro::regime::Regime;
use symbi_hydro::rhd::{Rhd, rhd_recover};
use symbi_hydro::riemann::{
    HlldStates, hllc, hllc_newtonian, hllc_rhd, hllc_rmhd, hlld_isothermal, hlld_isothermal_coeffs,
    hlld_newtonian, hlld_newtonian_coeffs, hlld_rmhd, hlld_rmhd_gr_ortho, hlld_rmhd_states,
    hlld_rmhd_states_gr_ortho, hlle, hlle_with_speeds,
};
use symbi_hydro::rmhd::{
    Rmhd, RmhdGr, rmhd_magnetosonic_cfl_speeds, rmhd_recover, rmhd_source_quantities,
};
use symbi_hydro::spatial_metric::{Gamma, GammaInv, SpatialMetric};
use symbi_hydro::state::{Cons, ConsG, Prim, PrimG};
use symbi_ir::Symbol;
use symbi_ir::algebra::Scalar;
use symbi_ir::graph::{ConstValue, ElementWiseOp, NodeId};
use symbi_ir::{FieldBind, FieldRef};

// the carrier + trace live alongside Op + Graph in symbi-ir. the builders below
// instantiate carrier-generic symbi-hydro physics at S = Gv and trace it into the IR.
use symbi_ir::{Gv, GvKernel, MeshScalar, TileSpec, begin_trace, end_trace, with_trace};

use super::coords::{Coords, EosArm, Recon, Spacetime, Spacing};

// submodule declarations: each category is its own file; the glob re-exports below preserve
// the byte-identical public path `gv::NAME` for every builder lib.rs + downstream crates reach
// (the split is purely organizational — no api change).
mod c2p;
pub mod census_map;
mod ct_emf;
mod flux;
pub(crate) mod geometry;
mod ghost;
mod godunov;
mod sources;
mod wavespeed;

pub use c2p::*;
pub use ct_emf::*;
pub use flux::*;
pub use geometry::*;
pub use ghost::*;
pub use godunov::*;
pub use sources::*;
pub use wavespeed::*;

// shared low-level helpers used across 2+ category submodules (or with no single category home)
// — kept here so every submodule sees them through `use super::*`.

/// the 3-way minmod for the theta-MC (generalized minmod) limiter, carrier-generic
/// (matches the substrate `minmod3`): the common-signed minimum-magnitude argument iff
/// x,y,z share a strict sign, else 0.
fn minmod3<S: Scalar>(x: S, y: S, z: S) -> S {
    let mn = x.min(y).min(z);
    let mx = x.max(y).max(z);
    let all_pos = mn.cmp_gt(S::ZERO);
    let all_neg = mx.cmp_lt(S::ZERO);
    S::select(all_pos, mn, S::select(all_neg, mx, S::ZERO))
}

/// van leer harmonic slope limiter: `2 dl dr/(dl+dr)` for same-sign slopes, `0` otherwise. SMOOTH
/// (C^1, no kink at the origin — unlike minmod), so the reconstructed staggered field stays clean.
/// the MHD-friendly limiter: keeps the L/R jumps small enough that the HLLD EMF's intermediate-field
/// overshoot (-> anti-diffusive `d`, see the d-sign analysis) stays SUBCRITICAL without a clamp.
/// selected in `plm_theta_gv` when `theta < 0`.
fn van_leer<S: Scalar>(dl: S, dr: S) -> S {
    let prod = dl * dr;
    let pos = prod.cmp_gt(S::ZERO);
    let denom = S::select(pos, dl + dr, S::ONE);
    let two = S::ONE + S::ONE;
    S::select(pos, two * prod / denom, S::ZERO)
}

/// PLM reconstruct with a runtime-selectable limiter, keyed on the SIGN of the `theta` scalar param:
///   theta >= 0 -> theta-MC minmod: `minmod3((vc-vl)*theta, (vr-vl)*0.5, (vr-vc)*theta)`, theta in
///                 [1,2] tuning compression (1 == plain minmod, 0 == pcm/first-order).
///   theta <  0 -> van leer (the smooth, MHD-friendly limiter; the magnitude is unused).
/// overloading theta's sign avoids a second ABI scalar — switch limiters at runtime via `--plm-theta`
/// (>=0 minmod-MC, e.g., -1 for van leer). both branches are traced; `select` keeps it NaN-safe.
fn plm_theta_gv(
    key: &str,
    runtime: impl Into<FieldBind>,
    ndim: u8,
    dir: u8,
    theta: Gv,
) -> (Gv, Gv) {
    let runtime = runtime.into();
    let qm2 = Gv::field_shifted(key, runtime.clone(), ndim, dir, -2);
    let qm1 = Gv::field_shifted(key, runtime.clone(), ndim, dir, -1);
    let q0 = Gv::field_shifted(key, runtime.clone(), ndim, dir, 0);
    let qp1 = Gv::field_shifted(key, runtime, ndim, dir, 1);
    plm_theta_from_stencil(qm2, qm1, q0, qp1, theta)
}

/// the theta-MC / van-leer PLM face pair from FOUR stencil VALUES (offsets -2..+1 along the
/// sweep) — the limiter core of `plm_theta_gv`, exposed so a reconstruction can run in a
/// TRANSFORMED variable (values built from several fields + analytic coefficients) rather
/// than a raw field.
pub(crate) fn plm_theta_from_stencil(qm2: Gv, qm1: Gv, q0: Gv, qp1: Gv, theta: Gv) -> (Gv, Gv) {
    let half = Gv::from_f64(0.5);
    let slope = |vl: Gv, vc: Gv, vr: Gv| {
        let a = vc - vl;
        let b = vr - vc;
        let minmod = minmod3(a * theta, half * (a + b), b * theta); // theta-MC (theta >= 0)
        let vleer = van_leer(a, b); // smooth harmonic limiter (theta < 0)
        Gv::select(theta.cmp_lt(Gv::ZERO), vleer, minmod)
    };
    let left = qm1 + half * slope(qm2, qm1, q0);
    let right = q0 - half * slope(qm1, q0, qp1);
    (left, right)
}

/// monotonized ppm interface pair `(a_l, a_r)` for the cell holding `vc` (colella &
/// woodward 1984): 4th-order interface values from the 5-cell stencil, clamped to the
/// neighbor range (the raw stencil overshoots at discontinuities), flattened at a local
/// extremum, with the sequential left-then-right overshoot correction — the right
/// correction reads the corrected left value; diff/curv read the pre-correction
/// interfaces. shared core of the ppm prolongation sub-cell average and the ppm
/// evolution face pair, so coarse-fine transfer and evolution monotonize identically.
pub(crate) fn ppm_cell_interfaces<S: Scalar>(vm2: S, vm1: S, vc: S, vp1: S, vp2: S) -> (S, S) {
    let two = S::ONE + S::ONE;
    // half belongs to the prolongation's sub-cell average; constructing it here holds the
    // traced node order of the prolongation kernel fixed (hash-consed ids are assigned at
    // first construction).
    let _half = S::ONE / two;
    let three = S::from_f64(3.0);
    let six = S::from_f64(6.0);
    let seven = S::from_f64(7.0);
    let twelve_inv = S::ONE / S::from_f64(12.0);

    let u_l = (seven * (vm1 + vc) - (vm2 + vp1)) * twelve_inv;
    let u_r = (seven * (vc + vp1) - (vm1 + vp2)) * twelve_inv;

    // clamp interface values to the neighbor range before monotonizing — the
    // 4th-order stencil overshoots at discontinuities.
    let u_l = u_l.max(vm1.min(vc)).min(vm1.max(vc));
    let u_r = u_r.max(vc.min(vp1)).min(vc.max(vp1));

    // monotonicity: flatten at a local extremum, else correct overshoots.
    let extremum = ((u_r - vc) * (vc - u_l)).cmp_le(S::ZERO);
    let diff = u_r - u_l;
    let curv = six * (vc - (u_l + u_r) / two);
    let a_l = S::select(
        (diff * curv).cmp_gt(diff * diff),
        three * vc - two * u_r,
        u_l,
    );
    let a_r = S::select(
        (diff * curv).cmp_lt(S::ZERO - diff * diff),
        three * vc - two * a_l,
        u_r,
    );
    let a_l = S::select(extremum, vc, a_l);
    let a_r = S::select(extremum, vc, a_r);
    (a_l, a_r)
}

/// extremum-preserving ppm interface pair (colella & sekora 2008): identical to
/// `ppm_cell_interfaces` away from extrema (4th-order interfaces, neighbor-range clamp,
/// sequential overshoot correction), but the cell holding a smooth extremum keeps a
/// limited parabola instead of flattening to a constant. flattening every smooth
/// extremum caps the scheme's L1 convergence at second order — the crest cell injects
/// an O(h^2) error stream each step — and in sustained turbulence, where velocity
/// extrema fill the box, it is a first-order dissipation mechanism. the extremum
/// parabola is rescaled by the limited second derivative: same-signed over the
/// {left, centered, right, interface-implied} candidates -> scale by
/// `min(|d2_faces|, 1.25 min(|d2_l|, |d2_c|, |d2_r|)) / |d2_faces|` (in [0, 1]);
/// mixed signs (a genuine discontinuity or odd-even noise) -> zero, which reproduces the
/// flatten, so shocked cells behave exactly as the cw84 form.
fn ppm_cell_interfaces_ep<S: Scalar>(vm2: S, vm1: S, vc: S, vp1: S, vp2: S) -> (S, S) {
    let two = S::ONE + S::ONE;
    let half = S::ONE / two;
    let three = S::from_f64(3.0);
    let six = S::from_f64(6.0);
    let seven = S::from_f64(7.0);
    let twelve_inv = S::ONE / S::from_f64(12.0);
    let mag = |x: S| x.max(S::ZERO - x);
    let c_ep = S::from_f64(1.25);

    let u_l = (seven * (vm1 + vc) - (vm2 + vp1)) * twelve_inv;
    let u_r = (seven * (vc + vp1) - (vm1 + vp2)) * twelve_inv;

    // an interface value OUTSIDE its neighbor-average range is legitimate on smooth
    // data whenever the profile's extremum sits near the interface — the point value
    // there exceeds both adjacent AVERAGES by O(h^2), so the plain neighbor-range
    // clamp is itself a second-order error source. instead rebuild the out-of-range
    // value from `u = (a_j + a_{j+1})/2 - d2/6` with the second derivative limited
    // over the same-signed {interface-implied, left, right} candidates; mixed signs
    // (a genuine jump) zero the curvature and leave the neighbor mean.
    let iface = |am1: S, a0: S, ap1: S, ap2: S, u_raw: S| -> S {
        let out = ((u_raw - a0) * (ap1 - u_raw)).cmp_lt(S::ZERO);
        let d2 = three * (a0 - two * u_raw + ap1);
        let d2_l = am1 - two * a0 + ap1;
        let d2_r = a0 - two * ap1 + ap2;
        let lim_mag = mag(d2).min(c_ep * mag(d2_l).min(mag(d2_r)));
        let signed = S::select(d2.cmp_lt(S::ZERO), S::ZERO - lim_mag, lim_mag);
        let lo = d2.min(d2_l).min(d2_r);
        let hi = d2.max(d2_l).max(d2_r);
        let lim = S::select(
            lo.cmp_gt(S::ZERO),
            signed,
            S::select(hi.cmp_lt(S::ZERO), signed, S::ZERO),
        );
        let corrected = (a0 + ap1) * half - lim / six;
        S::select(out, corrected, u_raw)
    };
    let u_l = iface(vm2, vm1, vc, vp1, u_l);
    let u_r = iface(vm1, vc, vp1, vp2, u_r);

    // the two-clause extremum test: face-implied ((u_r - a)(a - u_l) <= 0) OR
    // cell-average ((a_{j+1} - a)(a - a_{j-1}) <= 0) — the second clause catches the
    // cell whose extremum sits at an interface, where the face-implied product is
    // merely zero-adjacent and the cw84 overshoot arm would inject an O(h^2) face.
    let ext_faces = ((u_r - vc) * (vc - u_l)).cmp_le(S::ZERO);
    let ext_cells = ((vp1 - vc) * (vc - vm1)).cmp_le(S::ZERO);
    let one_if = |m: S::Mask| S::select(m, S::ONE, S::ZERO);
    let extremum = (one_if(ext_faces) + one_if(ext_cells)).cmp_gt(S::ZERO);

    // the non-extremum arm: the cw84 sequential overshoot correction.
    let diff = u_r - u_l;
    let curv = six * (vc - (u_l + u_r) / two);
    let a_l = S::select(
        (diff * curv).cmp_gt(diff * diff),
        three * vc - two * u_r,
        u_l,
    );
    let a_r = S::select(
        (diff * curv).cmp_lt(S::ZERO - diff * diff),
        three * vc - two * a_l,
        u_r,
    );

    // the extremum arm: limited-second-derivative rescale of the face-implied parabola.
    // the h^2 factors cancel in the ratio, so every d2 is an undivided difference.
    let d2 = six * (u_l + u_r - two * vc);
    let d2_c = vm1 - two * vc + vp1;
    let d2_l = vm2 - two * vm1 + vc;
    let d2_r = vc - two * vp1 + vp2;
    let d2_min = d2.min(d2_c).min(d2_l).min(d2_r);
    let d2_max = d2.max(d2_c).max(d2_l).max(d2_r);
    let mag = |x: S| x.max(S::ZERO - x);
    let c_ep = S::from_f64(1.25);
    let d2_lim_mag = mag(d2).min(c_ep * mag(d2_l).min(mag(d2_c)).min(mag(d2_r)));
    // |d2_lim| <= |d2| by construction, so the ratio lies in [0, 1]. a vanishing d2
    // satisfies neither strict-sign test (d2_min <= 0 <= d2_max) and takes the zero
    // arm; the denominator guard only keeps the unselected arm finite.
    let denom = S::select(mag(d2).cmp_gt(S::ZERO), mag(d2), S::ONE);
    let ratio = d2_lim_mag / denom;
    let scale = S::select(
        d2_min.cmp_gt(S::ZERO),
        ratio,
        S::select(d2_max.cmp_lt(S::ZERO), ratio, S::ZERO),
    );
    let e_l = vc + (u_l - vc) * scale;
    let e_r = vc + (u_r - vc) * scale;

    (
        S::select(extremum, e_l, a_l),
        S::select(extremum, e_r, a_r),
    )
}

/// PPM reconstruct: the monotonized-parabola face pair from SIX stencil VALUES (offsets
/// -3..+2 along the sweep). the face sits between cells -1 and 0: `left` is cell -1's
/// corrected right-interface value, `right` is cell 0's corrected left-interface value.
/// method-of-lines form — the parabola supplies face STATES for the riemann solver
/// under an ssp-rk update; characteristic tracing belongs to the lagrangian-remap
/// formulation and has no role here. values in, face pair out, so a reconstruction can
/// run in a transformed variable exactly as `plm_theta_from_stencil` does. evolution
/// uses the EXTREMUM-PRESERVING monotonization; the coarse-fine prolongation keeps the
/// flatten-at-extremum form (`ppm_cell_interfaces`), whose conservative sub-cell
/// averages are the safer transfer.
pub(crate) fn ppm_from_stencil(qm3: Gv, qm2: Gv, qm1: Gv, q0: Gv, qp1: Gv, qp2: Gv) -> (Gv, Gv) {
    let (_, left) = ppm_cell_interfaces_ep(qm3, qm2, qm1, q0, qp1);
    let (right, _) = ppm_cell_interfaces_ep(qm2, qm1, q0, qp1, qp2);
    (left, right)
}

/// the raw-field PPM wrapper mirroring `plm_theta_gv`: six shifted loads of `key`
/// along the sweep, monotonized-parabola face pair out. no limiter parameter — ppm
/// carries its own monotonicity constraint.
pub(crate) fn ppm_gv(key: &str, runtime: impl Into<FieldBind>, ndim: u8, dir: u8) -> (Gv, Gv) {
    let runtime = runtime.into();
    let qm3 = Gv::field_shifted(key, runtime.clone(), ndim, dir, -3);
    let qm2 = Gv::field_shifted(key, runtime.clone(), ndim, dir, -2);
    let qm1 = Gv::field_shifted(key, runtime.clone(), ndim, dir, -1);
    let q0 = Gv::field_shifted(key, runtime.clone(), ndim, dir, 0);
    let qp1 = Gv::field_shifted(key, runtime.clone(), ndim, dir, 1);
    let qp2 = Gv::field_shifted(key, runtime, ndim, dir, 2);
    ppm_from_stencil(qm3, qm2, qm1, q0, qp1, qp2)
}

/// the traced eos at a bake-time closure arm: gamma-law reads the `gamma` scalar,
/// taub-mathews ignores it (the scalar stays bound so the kernel ABI is uniform
/// across arms). the selection resolves at trace time — the emitted graph carries
/// only the chosen closure's operations.
pub(crate) fn gv_eos(arm: EosArm, gamma: Gv) -> symbi_hydro::eos::EosSelect<Gv> {
    match arm {
        EosArm::IdealGamma => symbi_hydro::eos::EosSelect::Ideal(IdealGas { gamma }),
        EosArm::TaubMathews => symbi_hydro::eos::EosSelect::Tm(TaubMathews),
    }
}

/// raw-field reconstruction dispatch at the bake-time `Recon` choice: the plm family
/// (runtime theta selects theta-MC / van leer / pcm-at-zero) or the ppm monotonized
/// parabola (theta unused). identical contract across arms: shifted loads of `key`
/// along the sweep in, face pair out.
pub(crate) fn recon_gv(
    key: &str,
    runtime: impl Into<FieldBind>,
    ndim: u8,
    dir: u8,
    theta: Gv,
    recon: Recon,
) -> (Gv, Gv) {
    match recon {
        Recon::Plm => plm_theta_gv(key, runtime, ndim, dir, theta),
        Recon::Ppm => ppm_gv(key, runtime, ndim, dir),
    }
}

// =============================================================================
// the lattice-map GHOST FILL in Gv — the boundary pullback: read the
// primitives at the per-axis integer SOURCE coord (periodic shift / reflect pivot / outflow
// clamp on a runtime `map_type`), write at the cell (in place), with the grade-1 jacobian
// `vel_sign` flip on the velocity (and B for RMHD). the source coord is PURE INTEGER (the
// `_coord_N` + the I32 `map_type`/`arg` params), so the read is an ordinary multi-axis
// `load_at` — no gather, no float->int cast. the gv multi-axis stencil cap (the integer
// `field_at`) that ghost + CT share, mirroring `pullback::{source_axis, iso_ghost_fill}`.
// =============================================================================

/// the per-axis lattice-map source coord, a pure-integer select on `map_type` (mirror of
/// `pullback::source_axis`): `0` skip -> `c`; `1` periodic -> `c+arg`; `2` reflect -> `arg-c`;
/// `3` outflow -> `arg`. registers `_coord_N` + the I32 `map_type_{ax}`/`arg_{ax}` params.
fn gv_lattice_source(ndim: usize) -> Vec<NodeId> {
    use ElementWiseOp::*;
    with_trace(|t| {
        // register coords, then every map_type, then every arg (grouped — matching the positional
        // rmhd ghost-fill dispatch ints [map_type_0..D, arg_0..D]).
        let coords: Vec<NodeId> = (0..ndim).map(|ax| t.coord(ax as u8)).collect();
        let map_type: Vec<NodeId> = (0..ndim)
            .map(|ax| t.scalar_int(&format!("map_type_{ax}")))
            .collect();
        let arg: Vec<NodeId> = (0..ndim)
            .map(|ax| t.scalar_int(&format!("arg_{ax}")))
            .collect();
        (0..ndim)
            .map(|ax| {
                let (c, mt, ag) = (coords[ax], map_type[ax], arg[ax]);
                let g = t.graph();
                let zero = g.add_const(ConstValue::I32(0), None);
                let one = g.add_const(ConstValue::I32(1), None);
                let two = g.add_const(ConstValue::I32(2), None);
                let is_skip = g.element_wise(Eq, vec![mt, zero], None);
                let is_periodic = g.element_wise(Eq, vec![mt, one], None);
                let is_reflect = g.element_wise(Eq, vec![mt, two], None);
                let periodic = g.element_wise(Add, vec![c, ag], None); // c + arg
                let reflect = g.element_wise(Sub, vec![ag, c], None); // arg - c
                let pick_reflect = g.select(is_reflect, reflect, ag, None); // else outflow
                let pick_periodic = g.select(is_periodic, periodic, pick_reflect, None);
                g.select(is_skip, c, pick_periodic, None)
            })
            .collect()
    })
}

/// load field `key` at the integer source coord vector `src` (deduped manifest registration) —
/// the gv multi-axis `load_at`, the pullback read. returns the loaded value as a `Gv`.
/// pub(crate): the amr transfer builders (gv_refinement.rs) share this pullback read.
pub(crate) fn gv_load_at(key: &str, runtime: impl Into<FieldBind>, src: &[NodeId]) -> Gv {
    let runtime = runtime.into();
    Gv::of(with_trace(|t| {
        t.register_field(key, runtime);
        t.graph().load_at(Symbol::intern(key), src.to_vec(), None)
    }))
}

// =============================================================================
// the RMHD constrained-transport stack in Gv — the staggered curl / edge-EMF / face->cell B /
// cell-B flux-predictor / EMF save+average. all built on the gv multi-axis offset stencil
// `gv_field_at` (the staggered gather: read field at `coord + offsets`). div(B)=0 to machine
// precision is preserved by the stencil (the discrete curl + divergence telescope the shared
// h-weighted edge EMFs to exactly 0); gated by the rmhd_ct_curl*_divb tests. the input/write
// order matches the hand-built staggered runtime dispatch the RMHD regime binds.
// =============================================================================

/// register a field in the manifest as a bare entry, emitting no node, to pin the buffer order
/// (the staggered runtime dispatch is positional) ahead of the stencil reads that follow.
fn gv_register_field(key: &str, runtime: &str) {
    with_trace(|t| t.register_field(key, runtime));
}

/// load field `key` at `coord + offsets` (per-axis integer offset; all-zero = the cell coord) —
/// the gv multi-axis offset stencil (the CT staggered gather). registers the field (deduped),
/// builds the integer coord arithmetic + `load_at`. like `field_shifted` but a full offset vector.
pub(crate) fn gv_field_at(key: &str, runtime: &str, ndim: usize, offsets: &[i32]) -> Gv {
    Gv::of(with_trace(|t| {
        t.register_field(key, runtime);
        let comps: Vec<NodeId> = (0..ndim)
            .map(|ax| {
                let c = t.coord(ax as u8);
                if offsets[ax] == 0 {
                    c
                } else {
                    let off = t.graph().add_const(ConstValue::I32(offsets[ax]), None);
                    t.graph()
                        .element_wise(ElementWiseOp::Add, vec![c, off], None)
                }
            })
            .collect();
        t.graph().load_at(Symbol::intern(key), comps, None)
    }))
}

/// clamp a denominator's magnitude to at least `eps` while preserving its sign. the HLLC/HLLD
/// coefficients divide by wave-speed differences that approach zero in a degenerate riemann fan (the
/// fast speed meeting the contact speed requires both the sound and alfven speeds to vanish), and by
/// an HLL-averaged density that approaches zero in vacuum.
///
/// clamping the magnitude is what makes the guard safe: every argument with `|x| >= eps` comes back
/// bit-for-bit, and only the arguments that would otherwise be unsafe move. the tempting
/// `x + eps * sgn(x)` fails on both counts — signum is zero at zero, so that form adds nothing
/// exactly where the guard is needed and still divides by zero, and for a well-resolved denominator
/// it perturbs a value that needed no help.
/// an exact zero takes the positive branch — at zero either direction is equally valid, so the
/// choice is arbitrary and only the magnitude matters.
fn guard_denominator(x: Gv, eps: Gv) -> Gv {
    let threshold = Gv::select(eps.cmp_gt(Gv::ZERO), eps, Gv::ONE);
    Gv::select(
        x.abs().cmp_gt(eps),
        x,
        Gv::select(x.cmp_ge(Gv::ZERO), threshold, Gv::ZERO - threshold),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbi_ir::graph::{Graph, Op};

    #[test]
    fn infer_tile_spec_stencil_vs_pointwise() {
        // the rmhd flux builder declares an explicit per-axis slab tile
        // (halo on the reconstruction axis `dir`, 0 transverse). a pointwise
        // kernel (same-cell reads only) leaves the spec absent -> infers None.
        let (flux, _) = rmhd_flux_gv(1, 0, 0);
        assert!(
            !flux.coord_components.is_empty(),
            "flux must be a stencil kernel"
        );
        let ts = flux.infer_tile_spec().expect("rmhd flux -> Some(TileSpec)");
        assert_eq!(
            ts.halo,
            vec![2],
            "PLM reconstruction radius on the single (dir=0) axis"
        );
        assert!(!ts.tiled_field_keys.is_empty(), "tiled fields populated");

        let (c2p, _) = rmhd_c2p_gv(100);
        assert!(
            c2p.coord_components.is_empty(),
            "c2p must be pointwise (same-cell)"
        );
        assert!(
            c2p.infer_tile_spec().is_none(),
            "pointwise c2p -> no smem tile"
        );
    }

    #[test]
    fn adiabatic_c2p_traces_the_real_physics_to_a_kernel() {
        // the payoff: symbi-hydro's adiabatic c2p, run at S=Gv, yields a dispatchable
        // kernel — the right ABI manifest + the right writes — entirely from the traced physics.
        let (k, writes) = adiabatic_c2p_gv::<1>();
        assert_eq!(
            k.field_inputs
                .iter()
                .map(|(k, b)| (k.clone(), b.name()))
                .collect::<Vec<_>>(),
            vec![
                ("cons_den".to_string(), FieldRef::cons_den().name()),
                ("cons_mom_0".to_string(), "cons.mom_0".to_string()),
                ("cons_nrg".to_string(), FieldRef::cons_nrg().name()),
            ]
        );
        assert_eq!(k.scalar_params, vec!["gamma".to_string()]);
        let write_paths: Vec<String> = writes.iter().map(|(_, rt, _)| rt.name()).collect();
        assert_eq!(write_paths, vec!["prim.rho", "prim.vel[0]", "prim.pre"]);
        assert!(
            !k.graph.has_errors(),
            "graph errors: {:?}",
            k.graph.errors()
        );
    }

    #[test]
    fn boundary_fill_is_the_coord_assign_instance() {
        // the same operator (apply_dag_core_gv) as the source pass, at the
        // (Coord, Assign) coordinate. proves the abstraction is general — two instances, one builder.
        // a prim prescription: rho=2, vel=0.5, pre=1 (consts; a real boundary reads x/t, same path).
        use symbi_ir::graph::ConstValue;
        let mk = |v: f64| {
            let mut g = Graph::new();
            let c = g.add_const(ConstValue::F64(v), None);
            symbi_hydro::source_spec::BuiltSource {
                graph: g,
                params: vec![],
                outputs: vec![c],
            }
        };
        let (rho, vel, pre) = (mk(2.0), mk(0.5), mk(1.0));
        let sources = [("den", &rho), ("mom", &vel), ("nrg", &pre)];
        let (k, writes) = boundary_fill_from_built_gv(
            Coords::Cartesian,
            &[Spacing::Uniform],
            &[0],
            1,
            1,
            true,
            &sources,
        );
        assert!(
            !k.graph.has_errors(),
            "graph errors: {:?}",
            k.graph.errors()
        );
        // Assign writes the prim state, one DAG per slot.
        let paths: Vec<String> = writes.iter().map(|(_, p, _)| p.name()).collect();
        assert_eq!(paths, vec!["prim.rho", "prim.vel[0]", "prim.pre"]);
        // Assign is a prescription, so it carries a bare value with no `dt` weight.
        assert!(
            !k.scalar_params.contains(&"dt".to_string()),
            "Assign carries no dt weight"
        );
        // Coord binds position alone, so the kernel's inputs are free of `u_stage.*` / `cons.*`
        // (a pure coordinate prescription). the const prims read nothing at all here.
        assert!(
            !k.field_inputs
                .iter()
                .any(|(_, p)| p.name().starts_with("u_stage") || p.name().starts_with("cons")),
            "Coord/Assign reads no interior state, got inputs {:?}",
            k.field_inputs,
        );
    }

    #[test]
    fn boundary_fill_prescribes_cell_b_for_mhd() {
        // the toroidal driven boundary: an MHD prescription (ncomp=3) with a `bcell` slot must
        // emit prim.mag[k] writes alongside rho/vel/pre. a purely toroidal injection sets the
        // in-plane B (mag[0],mag[1]) to 0 and the out-of-plane B_phi (mag[2]) to a value.
        use symbi_ir::graph::ConstValue;
        let mk = |vals: &[f64]| {
            let mut g = Graph::new();
            let outs = vals
                .iter()
                .map(|&v| g.add_const(ConstValue::F64(v), None))
                .collect();
            symbi_hydro::source_spec::BuiltSource {
                graph: g,
                params: vec![],
                outputs: outs,
            }
        };
        let den = mk(&[1.0]);
        let mom = mk(&[0.1, 0.0, 0.0]);
        let nrg = mk(&[1.0]);
        let bcell = mk(&[0.0, 0.0, 0.5]); // B_r=0, B_theta=0, B_phi=0.5 (purely toroidal)
        let sources = [
            ("den", &den),
            ("mom", &mom),
            ("nrg", &nrg),
            ("bcell", &bcell),
        ];
        let (k, writes) = boundary_fill_from_built_gv(
            Coords::Spherical,
            &[Spacing::Log, Spacing::Uniform],
            &[0, 1],
            2,
            3,
            true,
            &sources,
        );
        assert!(
            !k.graph.has_errors(),
            "graph errors: {:?}",
            k.graph.errors()
        );
        let paths: Vec<String> = writes.iter().map(|(_, p, _)| p.name()).collect();
        assert_eq!(
            paths,
            vec![
                "prim.rho",
                "prim.vel[0]",
                "prim.vel[1]",
                "prim.vel[2]",
                "prim.pre",
                "prim.mag[0]",
                "prim.mag[1]",
                "prim.mag[2]",
            ],
        );
    }

    #[test]
    fn traces_axpy_to_elementwise_ir() {
        begin_trace();
        let a = Gv::param("a");
        let b = Gv::param("b");
        let c = Gv::param("c");
        let r = a * b + c;
        let root = r.node();
        let g = end_trace().graph;
        // root is Add over [Mul(a, b), c].
        match &g.node(root).op {
            Op::ElementWise(ElementWiseOp::Add, ins) => {
                assert_eq!(ins.len(), 2);
                assert!(matches!(
                    &g.node(ins[0]).op,
                    Op::ElementWise(ElementWiseOp::Mul, _)
                ));
            }
            other => panic!("expected Add, got {other:?}"),
        }
    }

    #[test]
    fn const_literal_materializes_to_const_node_on_use() {
        begin_trace();
        let two = Gv::from_f64(2.0);
        let x = Gv::param("x");
        let r = two * x; // the 2.0 literal materializes to a Const node here
        let root = r.node();
        let g = end_trace().graph;
        match &g.node(root).op {
            Op::ElementWise(ElementWiseOp::Mul, ins) => {
                let has_two = ins
                    .iter()
                    .any(|&i| matches!(&g.node(i).op, Op::Const(ConstValue::F64(v)) if *v == 2.0));
                assert!(has_two, "the 2.0 literal should be a Const(F64(2.0)) node");
            }
            other => panic!("expected Mul, got {other:?}"),
        }
    }

    #[test]
    fn field_reads_build_the_kernel_abi_manifest() {
        // the input binding (1): the cons fields a c2p reads become field-read nodes
        // whose (ir_key, runtime_path) land — first-seen, deduped — in the manifest the
        // dispatch binds buffers by; declared scalars (gamma) land in the signature.
        begin_trace();
        let _den = Gv::field("cons_den", FieldRef::cons_den());
        let _mx = Gv::field("cons_mom_0", FieldRef::cons_mom(0));
        let _nrg = Gv::field("cons_nrg", FieldRef::cons_nrg());
        let _g = Gv::scalar("gamma");
        let _reread = Gv::field("cons_den", FieldRef::cons_den()); // a re-read dedups
        let k = end_trace();
        assert_eq!(
            k.field_inputs
                .iter()
                .map(|(k, b)| (k.clone(), b.name()))
                .collect::<Vec<_>>(),
            vec![
                ("cons_den".to_string(), FieldRef::cons_den().name()),
                ("cons_mom_0".to_string(), "cons.mom_0".to_string()),
                ("cons_nrg".to_string(), FieldRef::cons_nrg().name()),
            ]
        );
        assert_eq!(k.scalar_params, vec!["gamma".to_string()]);
    }

    #[test]
    fn iso_c2p_traces_the_real_physics_to_a_kernel() {
        // symbi-hydro's locally-isothermal recovery (`Cons::to_primitive` + Isothermal eos,
        // reading cs^2 from the nrg slot) at S = Gv: rho = den, vel = mom/den, pre = cs2*rho.
        // cs2 is a per-cell field (the prescribed temperature), which is what makes the run
        // locally isothermal (cs varies per cell). a global run supplies a uniform cs2.
        let (k, writes) = iso_c2p_gv::<1>();
        assert_eq!(
            k.field_inputs
                .iter()
                .map(|(k, b)| (k.clone(), b.name()))
                .collect::<Vec<_>>(),
            vec![
                ("cons_den".to_string(), FieldRef::cons_den().name()),
                ("cons_mom_0".to_string(), "cons.mom_0".to_string()),
                ("cs2".to_string(), "cs2".to_string()),
            ]
        );
        assert!(
            k.scalar_params.is_empty(),
            "cs2 is a field, not a scalar: {:?}",
            k.scalar_params
        );
        let write_paths: Vec<String> = writes.iter().map(|(_, rt, _)| rt.name()).collect();
        assert_eq!(write_paths, vec!["prim.rho", "prim.vel[0]", "prim.pre"]);
        assert!(
            !k.graph.has_errors(),
            "graph errors: {:?}",
            k.graph.errors()
        );

        // pre = recover_pressure = cs2 * rho (the per-cell sound-speed-squared times density).
        let pre_id = writes
            .iter()
            .find(|(_, rt, _)| rt.name() == "prim.pre")
            .unwrap()
            .2;
        assert!(
            matches!(
                &k.graph.node(pre_id).op,
                Op::ElementWise(ElementWiseOp::Mul, _)
            ),
            "expected pre = Mul(cs2, rho), got {:?}",
            k.graph.node(pre_id).op
        );
    }

    #[test]
    fn rhd_c2p_traces_the_real_iterative_physics_to_a_kernel() {
        // the iterative payoff: symbi-hydro's branch-free `rhd_recover` (a carrier-generic
        // newton on the pressure root) run at S=Gv yields a dispatchable kernel whose pressure
        // is a single Op::IterateInline (body traced once), so the deep newton stays folded
        // instead of unfolding into an exponential tree. the manifest + writes match the retired
        // `rhd_c2p` Expr builder.
        let (k, writes) = rhd_c2p_gv::<1>(20, EosArm::IdealGamma);
        assert_eq!(
            k.field_inputs
                .iter()
                .map(|(k, b)| (k.clone(), b.name()))
                .collect::<Vec<_>>(),
            vec![
                ("cons_den".to_string(), FieldRef::cons_den().name()),
                ("cons_mom_0".to_string(), "cons.mom_0".to_string()),
                ("cons_nrg".to_string(), FieldRef::cons_nrg().name()),
            ]
        );
        assert_eq!(k.scalar_params, vec!["gamma".to_string()]);
        let write_paths: Vec<String> = writes.iter().map(|(_, rt, _)| rt.name()).collect();
        assert_eq!(write_paths, vec!["prim.rho", "prim.vel[0]", "prim.pre"]);
        assert!(
            !k.graph.has_errors(),
            "graph errors: {:?}",
            k.graph.errors()
        );

        // the recovered pressure is the wu-2017 cone select over the fixed-count inline newton
        // loop: `pre = select(q(U)/D > 0, newton_p, cone_fail_sentinel)`. the select's then-branch
        // is the single Op::IterateInline (the deep newton stays folded); the else-branch is the
        // shared out-of-cone sentinel, which is non-positive (see c2p_result).
        let pre_id = writes
            .iter()
            .find(|(_, rt, _)| rt.name() == "prim.pre")
            .unwrap()
            .2;
        let newton_id = match &k.graph.node(pre_id).op {
            Op::Select(_, then_branch, _) => *then_branch,
            other => panic!("expected prim.pre = Select(cone, newton, sentinel), got {other:?}"),
        };
        assert!(
            matches!(
                &k.graph.node(newton_id).op,
                Op::IterateInline { count: 20, .. }
            ),
            "expected the cone select's then-branch = IterateInline(count=20), got {:?}",
            k.graph.node(newton_id).op
        );
    }

    #[test]
    fn rmhd_c2p_traces_the_real_bracketed_physics_to_a_kernel() {
        // the last + hardest c2p: symbi-hydro's `rmhd_recover` (KKC false-position) at
        // S=Gv yields a dispatchable kernel — 8 conserved reads + gamma, the 4 prim writes,
        // and the bracketed solve as a MULTI-accumulator IterateInline (the false-position's
        // 6-state bracket). proves iterate_vec carries the carrier-generic RMHD c2p.
        let (k, writes) = rmhd_c2p_gv(100);
        assert_eq!(
            k.field_inputs
                .iter()
                .map(|(k, b)| (k.clone(), b.name()))
                .collect::<Vec<_>>(),
            vec![
                ("cons_den".to_string(), FieldRef::cons_den().name()),
                ("cons_mom_0".to_string(), "cons.mom_0".to_string()),
                ("cons_mom_1".to_string(), "cons.mom_1".to_string()),
                ("cons_mom_2".to_string(), "cons.mom_2".to_string()),
                ("cons_nrg".to_string(), FieldRef::cons_nrg().name()),
                ("cons_mag_0".to_string(), "cons.mag_0".to_string()),
                ("cons_mag_1".to_string(), "cons.mag_1".to_string()),
                ("cons_mag_2".to_string(), "cons.mag_2".to_string()),
            ]
        );
        assert_eq!(k.scalar_params, vec!["gamma".to_string()]);
        let write_paths: Vec<String> = writes.iter().map(|(_, rt, _)| rt.name()).collect();
        assert_eq!(
            write_paths,
            vec![
                "prim.rho",
                "prim.vel[0]",
                "prim.vel[1]",
                "prim.vel[2]",
                "prim.pre"
            ]
        );
        assert!(
            !k.graph.has_errors(),
            "graph errors: {:?}",
            k.graph.errors()
        );

        // the false-position is a 6-accumulator IterateInline (count=100).
        let has_multi_iter = (0..k.graph.len()).any(|i| {
            matches!(&k.graph.node(NodeId(i as u32)).op,
                Op::IterateInline { accs, count: 100, .. } if accs.len() == 6)
        });
        assert!(
            has_multi_iter,
            "expected a 6-accumulator IterateInline(count=100) for the false-position"
        );
    }

    #[test]
    fn adiabatic_flux_traces_recon_plus_hlle_to_a_kernel() {
        // the first gv FLUX: PLM reconstruction (a stencil -> LoadAt) composed with the
        // carrier-generic riemann::hlle (-> Select branches). proves Gv::field_shifted +
        // symbi-hydro's hlle build a dispatchable face-flux kernel — no rhd_side-style
        // hand-written per-component U/F. manifest + writes match the substrate hlle_flux.
        let (k, writes) = adiabatic_flux_gv::<1>(0, Recon::Plm);
        assert_eq!(
            k.field_inputs
                .iter()
                .map(|(k, b)| (k.clone(), b.name()))
                .collect::<Vec<_>>(),
            vec![
                ("prim_rho".to_string(), FieldRef::PrimRho.name()),
                ("prim_v0".to_string(), "prim.vel[0]".to_string()),
                ("prim_pre".to_string(), FieldRef::PrimPre.name()),
            ]
        );
        assert_eq!(
            k.scalar_params,
            vec![
                "gamma".to_string(),
                "theta".to_string(),
                "mesh_adot_0".to_string(),
                "x_lo_0".to_string(),
                "dx_0".to_string(),
                "map_kind_0".to_string(),
                "map_param_0".to_string(),
                "mesh_vtrans_0".to_string(),
            ]
        );
        assert_eq!(k.coord_components, vec![0]);
        let write_paths: Vec<String> = writes.iter().map(|(_, rt, _)| rt.name()).collect();
        assert_eq!(write_paths, vec!["flux.den", "flux.mom_0", "flux.nrg"]);
        assert!(
            !k.graph.has_errors(),
            "graph errors: {:?}",
            k.graph.errors()
        );

        let has_load_at = (0..k.graph.len())
            .any(|i| matches!(&k.graph.node(NodeId(i as u32)).op, Op::LoadAt(..)));
        let has_select = (0..k.graph.len())
            .any(|i| matches!(&k.graph.node(NodeId(i as u32)).op, Op::Select(..)));
        assert!(
            has_load_at,
            "reconstruction should emit LoadAt stencil nodes"
        );
        assert!(has_select, "HLLE should emit Select branches");
    }

    #[test]
    fn rhd_flux_traces_the_relativistic_hlle_to_a_kernel() {
        // same PLM + riemann::hlle pattern at the Rhd regime (relativistic U/F/wave speeds).
        // the only change from adiabatic is the regime — one HLLE source, two physics.
        let (k, writes) = rhd_flux_gv::<1>(0, EosArm::IdealGamma);
        assert_eq!(
            k.field_inputs
                .iter()
                .map(|(k, b)| (k.clone(), b.name()))
                .collect::<Vec<_>>(),
            vec![
                ("prim_rho".to_string(), FieldRef::PrimRho.name()),
                ("prim_v0".to_string(), "prim.vel[0]".to_string()),
                ("prim_pre".to_string(), FieldRef::PrimPre.name()),
            ]
        );
        assert_eq!(
            k.scalar_params,
            vec![
                "gamma".to_string(),
                "theta".to_string(),
                "mesh_adot_0".to_string(),
                "x_lo_0".to_string(),
                "dx_0".to_string(),
                "map_kind_0".to_string(),
                "map_param_0".to_string(),
                "mesh_vtrans_0".to_string(),
            ]
        );
        let write_paths: Vec<String> = writes.iter().map(|(_, rt, _)| rt.name()).collect();
        assert_eq!(write_paths, vec!["flux.den", "flux.mom_0", "flux.nrg"]);
        assert!(
            !k.graph.has_errors(),
            "graph errors: {:?}",
            k.graph.errors()
        );
    }

    #[test]
    fn iso_flux_traces_the_newtonian_hlle_minus_energy() {
        // the iso flux is the newtonian flux at gamma->1 (sound speed sqrt(p/rho) from the
        // reconstructed prim.pre = cs^2(x)*rho — locally isothermal) MINUS the energy flux.
        // so it reconstructs prim.pre and writes only den + mom. it is gamma-INDEPENDENT (the
        // sound speed comes from the reconstructed pressure), so the only scalar is
        // the PLM limiter `theta`.
        let (k, writes) = iso_flux_gv::<1>(0);
        assert_eq!(
            k.field_inputs
                .iter()
                .map(|(k, b)| (k.clone(), b.name()))
                .collect::<Vec<_>>(),
            vec![
                ("prim_rho".to_string(), FieldRef::PrimRho.name()),
                ("prim_v0".to_string(), "prim.vel[0]".to_string()),
                ("prim_pre".to_string(), FieldRef::PrimPre.name()),
            ]
        );
        assert_eq!(
            k.scalar_params,
            vec![
                "theta".to_string(),
                "mesh_adot_0".to_string(),
                "x_lo_0".to_string(),
                "dx_0".to_string(),
                "map_kind_0".to_string(),
                "map_param_0".to_string(),
                "mesh_vtrans_0".to_string(),
            ]
        );
        let write_paths: Vec<String> = writes.iter().map(|(_, rt, _)| rt.name()).collect();
        assert_eq!(
            write_paths,
            vec!["flux.den", "flux.mom_0"],
            "iso has no energy flux"
        );
        assert!(
            !k.graph.has_errors(),
            "graph errors: {:?}",
            k.graph.errors()
        );
    }

    #[test]
    fn rmhd_flux_traces_the_mhd_hlle_to_a_kernel() {
        // RMHD flux: theta-MC PLM (the free-theta limiter) over rho/vel(3)/pre/mag(3),
        // composed with riemann::hlle_with_speeds at the Rmhd regime. the flux READS the
        // per-cell quartic wave_speed_l/r (ws_l/ws_r, bound after the 8 prim) and forms the
        // davis fan. 8 conserved fluxes (D, S_k, tau, B_k).
        let (k, writes) = rmhd_flux_gv(1, 0, 0);
        assert_eq!(
            k.scalar_params,
            vec!["gamma".to_string(), "theta".to_string()]
        );
        assert_eq!(
            k.field_inputs
                .iter()
                .map(|(key, _)| key.as_str())
                .collect::<Vec<_>>(),
            vec![
                "prim_rho", "prim_v0", "prim_v1", "prim_v2", "prim_pre", "prim_b0", "prim_b1",
                "prim_b2",
                "bface_n", // <- the staggered normal-B face field (Gardiner-Stone CT coupling)
                "ws_l", "ws_r", // <- the materialized per-cell wave speeds, read for the fan
            ]
        );
        let write_paths: Vec<String> = writes.iter().map(|(_, rt, _)| rt.name()).collect();
        assert_eq!(
            write_paths,
            vec![
                "flux.den",
                "flux.mom_0",
                "flux.mom_1",
                "flux.mom_2",
                "flux.nrg",
                "flux.mag_0",
                "flux.mag_1",
                "flux.mag_2",
            ]
        );
        assert!(
            !k.graph.has_errors(),
            "graph errors: {:?}",
            k.graph.errors()
        );
        // the quartic's resolvent-cubic transcendentals are ABSENT from the flux — they live
        // only in rmhd_wave_speeds_cell_gv, computed once per cell.
        use symbi_ir::graph::ElementWiseOp as E;
        let has_transcendental = (0..k.graph.len()).any(|i| {
            matches!(
                &k.graph.node(NodeId(i as u32)).op,
                Op::ElementWise(E::Asinh | E::Acosh | E::Cosh | E::Cos | E::Sin | E::Pow, _)
            )
        });
        assert!(
            !has_transcendental,
            "flux must NOT carry the quartic's transcendentals anymore"
        );
    }

    #[test]
    fn field_shifted_traces_a_stencil_load_at() {
        // the stencil cap (foundation for the gv flux / PLM reconstruction): a shifted field
        // read builds a LoadAt at `_coord + offset` and records the field + the coord axis in
        // the manifest; offset 0 dedups to the direct cell read of the same buffer.
        begin_trace();
        let _q0 = Gv::field_shifted("prim_rho", FieldRef::PrimRho, 1, 0, 0); // direct cell read
        let qm1 = Gv::field_shifted("prim_rho", FieldRef::PrimRho, 1, 0, -1); // left neighbor
        let qm1_id = qm1.node();
        let k = end_trace();
        assert!(
            matches!(&k.graph.node(qm1_id).op, Op::LoadAt(..)),
            "shifted read should be a LoadAt, got {:?}",
            k.graph.node(qm1_id).op
        );
        assert_eq!(
            k.field_inputs
                .iter()
                .map(|(k, b)| (k.clone(), b.name()))
                .collect::<Vec<_>>(),
            vec![("prim_rho".to_string(), FieldRef::PrimRho.name())],
        );
        assert_eq!(k.coord_components, vec![0], "axis 0's _coord recorded once");
    }

    #[test]
    fn iterate_vec_host_runs_a_coupled_recurrence() {
        // f64 multi-state iterate: a 2-state bracketed mean — converge x to the average
        // of the bounds. state=[lo, hi]; body=[(lo+hi)/2, hi]; converged when hi-lo small.
        // checks the host loop advances the whole state + early-breaks on convergence.
        let r = f64::iterate_vec(
            [0.0, 1.0],
            100,
            |[lo, hi]| [(lo + hi) * 0.5, hi],
            |[lo, _], [lo_n, _]| (lo_n - lo).abs().cmp_lt(1e-15),
            0,
        );
        assert!(
            (r - 1.0).abs() < 1e-12,
            "lo should climb toward hi=1.0, got {r}"
        );
    }

    #[test]
    fn iterate_vec_traces_to_multi_acc_iterate_inline() {
        // the bracketed-iterate cap: a 2-component coupled step traces to a single
        // multi-accumulator IterateInline (body + per-component freeze recorded once).
        begin_trace();
        let a0 = Gv::param("a0");
        let b0 = Gv::param("b0");
        let r = Gv::iterate_vec(
            [a0, b0],
            7,
            |[a, b]| [b, a + b],              // fibonacci-style coupling
            |_, _| Gv::ZERO.cmp_lt(Gv::ZERO), // false mask: run the fixed count
            0,
        );
        let root = r.node();
        let g = end_trace().graph;
        match &g.node(root).op {
            Op::IterateInline {
                accs,
                inits,
                steps,
                count,
                result,
                ..
            } => {
                assert_eq!(*count, 7);
                assert_eq!(*result, 0);
                assert_eq!(accs.len(), 2, "two accumulators");
                assert_eq!(inits.len(), 2);
                assert_eq!(steps.len(), 2);
            }
            other => panic!("expected multi-acc IterateInline, got {other:?}"),
        }
    }

    #[test]
    fn iterate_traces_to_iterate_inline() {
        begin_trace();
        let x0 = Gv::param("x0");
        // x_{n+1} = x_n * 0.5, a fixed 3 steps. the convergence predicate traces on Gv as well:
        // scalar iterate lowers to a single-accumulator IterateInline whose step is a
        // Select(converged, old, new) — the keep-old freeze (carrier equivalence with the
        // host early-break).
        let r = x0.iterate(
            3,
            |x| x * Gv::from_f64(0.5),
            |prev, cur| (cur - prev).cmp_lt(Gv::from_f64(1e-9)),
        );
        let root = r.node();
        let g = end_trace().graph;
        match &g.node(root).op {
            Op::IterateInline {
                accs,
                steps,
                count,
                result,
                ..
            } => {
                assert_eq!(*count, 3, "fixed bound preserved");
                assert_eq!(accs.len(), 1, "single value accumulator");
                assert_eq!(*result, 0, "result is the value component");
                assert!(
                    matches!(&g.node(steps[0]).op, Op::Select(..)),
                    "the step must be the keep-OLD freeze Select, got {:?}",
                    g.node(steps[0]).op
                );
            }
            other => panic!("expected IterateInline(count=3, 1 acc, Select step), got {other:?}"),
        }
    }

    #[test]
    fn iterate_freezes_on_convergence_carrier_equivalent() {
        // the carrier-equivalence regression for Gv::iterate. a deliberately non-idempotent
        // body (each step +1, converge at the threshold): the host early-break returns the
        // value at convergence, and the traced loop matches it only when it freezes there.
        // a trace that runs to the full count overshoots, giving a value that is CPU-correct
        // and GPU-wrong.
        use symbi_ir::emit::{Precision, Target, TargetConfig};
        use symbi_ir::{Cpu, CpuField, CpuFieldMut, KernelEmitInputs, emit_kernel_cpu};

        fn ramp<S: Scalar>(start: S, count: usize, threshold: f64) -> S {
            start.iterate(
                count,
                |x| x + S::ONE,
                move |_prev, cur| cur.cmp_ge(S::from_f64(threshold)),
            )
        }

        // evaluate ramp::<Gv> on a single cell via the CPU interpreter (no nvcc needed).
        fn run_gv(count: usize, threshold: f64, start: f64) -> f64 {
            begin_trace();
            let s = Gv::field("start", "start");
            let root = ramp::<Gv>(s, count, threshold).node();
            let writes = vec![("out".to_string(), "out".into(), root)];
            let k = end_trace();
            assert!(
                !k.graph.has_errors(),
                "ramp graph errors: {:?}",
                k.graph.errors()
            );
            let spec = KernelEmitInputs {
                kernel_name: "ramp",
                coalesce_layout: false,
                ndim: 1,
                target: TargetConfig {
                    target: Target::Cuda,
                    precision: Precision::F64,
                },
                field_inputs: &k.field_inputs,
                scalar_params: &k.scalar_params,
                field_writes: &writes,
                coord_components: &k.coord_components,
                device_preamble: &[],
                tile_spec: None,
            };
            // also exercise emission (the lowering the AOT path renders).
            let _ = emit_kernel_cpu(&k.graph, &spec);
            let (lo, extent) = ([0i32], [1u32]);
            let start_data = [start];
            let inputs = [CpuField {
                data: &start_data,
                lo: &lo,
                extent: &extent,
            }];
            let mut out_data = [0.0f64];
            let mut outputs = [CpuFieldMut {
                data: &mut out_data,
                lo: &lo,
                extent: &extent,
            }];
            Cpu.run_kernel(
                &k.graph,
                &spec,
                &inputs,
                &mut outputs,
                &[],
                &[1u32],
                &[0i32],
            );
            out_data[0]
        }

        // converges within the count: keep-OLD freezes at the last pre-threshold value
        // (4: at prev=4, cur=5 trips `cur >= 5`, so the OLD 4 is kept). the traced loop
        // must freeze there at 4, short of the full count=20.
        let host = ramp::<f64>(0.0, 20, 5.0);
        let gv = run_gv(20, 5.0, 0.0);
        assert_eq!(
            host, 4.0,
            "host keep-OLD freeze returns the last pre-convergence value"
        );
        assert!(
            (gv - host).abs() < 1e-12,
            "carrier divergence: host={host}, gv={gv}"
        );
        assert!(
            (gv - 20.0).abs() > 0.5,
            "no freeze: traced loop ran to count ({gv})"
        );

        // convergence lies beyond the count: both carriers run the full bound and agree.
        let host_nc = ramp::<f64>(0.0, 3, 5.0);
        let gv_nc = run_gv(3, 5.0, 0.0);
        assert_eq!(host_nc, 3.0);
        assert!(
            (gv_nc - host_nc).abs() < 1e-12,
            "non-converged divergence: host={host_nc}, gv={gv_nc}"
        );
    }

    #[test]
    fn cond_is_a_lazy_branch_carrier_equivalent_and_renders_if_else() {
        // the dual of iterate: `S::cond` is a real data-dependent branch. the
        // untaken arm computes acosh(x) (NaN for x < 1); with `cond` it traces
        // into the `if` block and runs where x > 1 alone — a carrier-portable
        // early-`if`.
        use symbi_ir::emit::{Precision, Target, TargetConfig};
        use symbi_ir::{Cpu, CpuField, CpuFieldMut, KernelEmitInputs, emit_kernel_from_lowering};

        fn pick<S: Scalar>(x: S) -> S {
            S::cond(x.cmp_gt(S::ONE), || x.acosh(), || x * x)
        }

        // trace structure: the root is Op::IfElse, the lazy branch form.
        begin_trace();
        let xp = Gv::param("x");
        let root = pick::<Gv>(xp).node();
        let g = end_trace().graph;
        match &g.node(root).op {
            Op::IfElse {
                then_results,
                else_results,
                ..
            } => {
                assert_eq!(then_results.len(), 1, "scalar cond -> 1 then-result");
                assert_eq!(else_results.len(), 1, "scalar cond -> 1 else-result");
            }
            other => panic!("expected Op::IfElse, got {other:?}"),
        }

        // run pick::<Gv> on one cell via the CPU interpreter; return value + the
        // emitted (CUDA) source for the structural check.
        fn run_gv(x: f64) -> (f64, String) {
            begin_trace();
            let xf = Gv::field("x", "x");
            let root = pick::<Gv>(xf).node();
            let writes = vec![("out".to_string(), "out".into(), root)];
            let k = end_trace();
            assert!(
                !k.graph.has_errors(),
                "pick graph errors: {:?}",
                k.graph.errors()
            );
            let spec = KernelEmitInputs {
                kernel_name: "pick",
                coalesce_layout: false,
                ndim: 1,
                target: TargetConfig {
                    target: Target::Cuda,
                    precision: Precision::F64,
                },
                field_inputs: &k.field_inputs,
                scalar_params: &k.scalar_params,
                field_writes: &writes,
                coord_components: &k.coord_components,
                device_preamble: &[],
                tile_spec: None,
            };
            let src = emit_kernel_from_lowering(&k.graph, &spec).source;
            let (lo, extent) = ([0i32], [1u32]);
            let xdata = [x];
            let inputs = [CpuField {
                data: &xdata,
                lo: &lo,
                extent: &extent,
            }];
            let mut out = [0.0f64];
            let mut outputs = [CpuFieldMut {
                data: &mut out,
                lo: &lo,
                extent: &extent,
            }];
            Cpu.run_kernel(
                &k.graph,
                &spec,
                &inputs,
                &mut outputs,
                &[],
                &[1u32],
                &[0i32],
            );
            (out[0], src)
        }

        // carrier equivalence: f64 host == Gv interp, bit-identical, on each of the two
        // arms (x<1 takes else=x*x; x>1 takes then=acosh; near the boundary).
        for &x in &[0.5_f64, 2.0, 1.5, 0.999, 1.0, 3.7] {
            let host = pick::<f64>(x);
            let (gv, _) = run_gv(x);
            assert!(
                gv.to_bits() == host.to_bits() || (gv.is_nan() && host.is_nan()),
                "carrier divergence at x={x}: host={host} gv={gv}",
            );
        }

        // the emitted source is a genuine `if (...) { ... } else { ... }`, with the
        // expensive `acosh` inside the branch (after `if (`) and the higher-order
        // placeholder resolved away: the branch is structurally lazy.
        let (_, src) = run_gv(2.0);
        assert!(
            src.contains("if ("),
            "no real branch in emitted source:\n{src}"
        );
        assert!(
            src.contains("} else {"),
            "no else arm in emitted source:\n{src}"
        );
        assert!(
            !src.contains("HIGHER_ORDER"),
            "IfElse not intercepted by emit:\n{src}"
        );
        let if_pos = src.find("if (").expect("if");
        let acosh_pos = src.find("acosh").expect("acosh in then-arm");
        assert!(
            acosh_pos > if_pos,
            "acosh computed BEFORE the branch (not lazy):\n{src}",
        );
    }

    #[test]
    fn cond_vec_is_an_n_output_lazy_branch_carrier_equivalent() {
        // the dual of iterate_vec: one branch, two outputs from the same taken
        // arm. the else arm computes a shared expensive value (acosh) feeding
        // both outputs — proving the arm runs once and both outputs project
        // from it: the (sl, sr) wave-speed fast-path shape.
        use symbi_ir::emit::{Precision, Target, TargetConfig};
        use symbi_ir::{Cpu, CpuField, CpuFieldMut, KernelEmitInputs, emit_kernel_from_lowering};

        fn pick2<S: Scalar>(x: S) -> [S; 2] {
            // x > 1 -> (acosh(x), 2*acosh(x)) sharing acosh; else -> (x, -x).
            S::cond_vec(
                x.cmp_gt(S::ONE),
                || {
                    let a = x.acosh();
                    [a, a + a]
                },
                || [x, S::ZERO - x],
            )
        }

        // TRACE STRUCTURE: two Op::Proj over one Op::IfElse with 2 results.
        begin_trace();
        let xp = Gv::param("x");
        let out = pick2::<Gv>(xp);
        let g = end_trace().graph;
        for (j, gv) in out.iter().enumerate() {
            match &g.node(gv.node()).op {
                Op::Proj { source, index } => {
                    assert_eq!(*index as usize, j, "proj index");
                    match &g.node(*source).op {
                        Op::IfElse {
                            then_results,
                            else_results,
                            ..
                        } => {
                            assert_eq!(then_results.len(), 2, "2 then-results");
                            assert_eq!(else_results.len(), 2, "2 else-results");
                        }
                        other => panic!("proj source not IfElse: {other:?}"),
                    }
                }
                other => panic!("output {j} not a Proj: {other:?}"),
            }
        }

        // CARRIER EQUIVALENCE + shared-arm: run pick2::<Gv> (both outputs)
        // via the CPU interp, compare bit-for-bit to pick2::<f64> on both
        // arms; assert the emitted source computes acosh exactly ONCE.
        fn run_gv(x: f64) -> [f64; 2] {
            begin_trace();
            let xf = Gv::field("x", "x");
            let out = pick2::<Gv>(xf);
            let writes = vec![
                ("o0".to_string(), "o0".into(), out[0].node()),
                ("o1".to_string(), "o1".into(), out[1].node()),
            ];
            let k = end_trace();
            assert!(
                !k.graph.has_errors(),
                "pick2 graph errors: {:?}",
                k.graph.errors()
            );
            let spec = KernelEmitInputs {
                kernel_name: "pick2",
                coalesce_layout: false,
                ndim: 1,
                target: TargetConfig {
                    target: Target::Cuda,
                    precision: Precision::F64,
                },
                field_inputs: &k.field_inputs,
                scalar_params: &k.scalar_params,
                field_writes: &writes,
                coord_components: &k.coord_components,
                device_preamble: &[],
                tile_spec: None,
            };
            let src = emit_kernel_from_lowering(&k.graph, &spec).source;
            assert!(
                !src.contains("HIGHER_ORDER"),
                "IfElse/Proj not intercepted:\n{src}"
            );
            assert_eq!(
                src.matches("acosh").count(),
                1,
                "acosh must be SHARED (computed once):\n{src}"
            );
            let (lo, extent) = ([0i32], [1u32]);
            let xdata = [x];
            let inputs = [CpuField {
                data: &xdata,
                lo: &lo,
                extent: &extent,
            }];
            let mut o0 = [0.0f64];
            let mut o1 = [0.0f64];
            let mut outputs = [
                CpuFieldMut {
                    data: &mut o0,
                    lo: &lo,
                    extent: &extent,
                },
                CpuFieldMut {
                    data: &mut o1,
                    lo: &lo,
                    extent: &extent,
                },
            ];
            Cpu.run_kernel(
                &k.graph,
                &spec,
                &inputs,
                &mut outputs,
                &[],
                &[1u32],
                &[0i32],
            );
            [o0[0], o1[0]]
        }

        for &x in &[0.5_f64, 2.0, 1.5, 0.999, 1.0, 3.7] {
            let host = pick2::<f64>(x);
            let gv = run_gv(x);
            for j in 0..2 {
                assert!(
                    gv[j].to_bits() == host[j].to_bits() || (gv[j].is_nan() && host[j].is_nan()),
                    "carrier divergence at x={x} out{j}: host={} gv={}",
                    host[j],
                    gv[j],
                );
            }
        }
    }

    #[test]
    fn rhd_wave_speed_map_traces_the_real_physics() {
        // symbi-hydro's Rhd::wave_speeds_axis (mignone-bodo, normal velocity only) at S=Gv,
        // folded with the in-kernel cartesian-uniform widths into one timestep kernel — the same
        // physics the RHD flux's HLLE uses. cartesian 2D: reads rho + the gridded normal
        // velocities (v0, v1) + pre; the dead v2 stays a zero constant outside the graph.
        let (k, writes) = rhd_wave_speed_map_gv(
            Coords::Cartesian,
            Spacetime::Minkowski,
            &[Spacing::Uniform; 2],
            &[0, 1],
            2,
            EosArm::IdealGamma,
        );
        assert_eq!(writes.len(), 1, "one scratch lambda write");
        assert_eq!(writes[0].1.name(), "scratch");
        assert_eq!(
            k.scalar_params,
            vec![
                "gamma".to_string(),
                "map_kind_0".to_string(),
                "x_lo_0".to_string(),
                "dx_0".to_string(),
                "map_param_0".to_string(),
                "inv_dx_0".to_string(),
                "map_kind_1".to_string(),
                "x_lo_1".to_string(),
                "dx_1".to_string(),
                "map_param_1".to_string(),
                "inv_dx_1".to_string(),
                "mesh_adot_0".to_string(),
                "mesh_vtrans_0".to_string(),
                "mesh_adot_1".to_string(),
                "mesh_vtrans_1".to_string(),
            ]
        );
        let keys: Vec<&str> = k.field_inputs.iter().map(|(key, _)| key.as_str()).collect();
        assert_eq!(
            keys,
            vec!["prim_rho", "prim_v0", "prim_v1", "prim_pre"],
            "RHD CFL reads rho + the gridded normal velocities + pre (no dead v2)"
        );
        assert!(
            !k.graph.has_errors(),
            "graph errors: {:?}",
            k.graph.errors()
        );
    }

    #[test]
    fn coord_is_the_index_to_physical_bridge() {
        // Gv::coord(ax) is the integer cell index, usable in arithmetic against the f64 grid
        // scalars (auto-promotes) — the foundation for in-kernel geometry. a physical position
        // x = x_lo + coord*dx traces cleanly and records the axis + scalars in the manifest.
        begin_trace();
        let _x = Gv::coord(0) * Gv::scalar("dx_0") + Gv::scalar("x_lo_0");
        let k = end_trace();
        assert_eq!(k.coord_components, vec![0], "axis 0's _coord recorded once");
        assert!(k.scalar_params.contains(&"dx_0".to_string()));
        assert!(k.scalar_params.contains(&"x_lo_0".to_string()));
        assert!(
            !k.graph.has_errors(),
            "graph errors: {:?}",
            k.graph.errors()
        );
    }

    #[test]
    fn cell_inv_phys_widths_gv_match_the_geometry_per_coords() {
        // the gv metric: cartesian has scale factor 1 (no angular term); spherical's phi axis
        // carries h = r*sin(theta) -> a Sin node. proves the geometry traces in Gv from the
        // cell index, the foundation the curvilinear CFL / divergence / sources will use.
        begin_trace();
        let inv = cell_inv_phys_widths_gv(
            Coords::Cartesian,
            &[Spacing::Uniform, Spacing::Uniform],
            &[0, 1],
            2,
        );
        let _r: Vec<NodeId> = inv.iter().map(|g| g.node()).collect();
        let kc = end_trace();
        assert_eq!(inv.len(), 2);
        assert!(
            !kc.graph.has_errors(),
            "graph errors: {:?}",
            kc.graph.errors()
        );
        let has_sin = |g: &Graph| {
            (0..g.len()).any(|i| {
                matches!(
                    &g.node(NodeId(i as u32)).op,
                    Op::ElementWise(ElementWiseOp::Sin, _)
                )
            })
        };
        assert!(!has_sin(&kc.graph), "cartesian has no angular scale factor");

        begin_trace();
        let inv = cell_inv_phys_widths_gv(
            Coords::Spherical,
            &[Spacing::Uniform, Spacing::Uniform, Spacing::Uniform],
            &[0, 1, 2],
            3,
        );
        let _r: Vec<NodeId> = inv.iter().map(|g| g.node()).collect();
        let ks = end_trace();
        assert!(
            !ks.graph.has_errors(),
            "graph errors: {:?}",
            ks.graph.errors()
        );
        assert!(
            has_sin(&ks.graph),
            "spherical phi axis needs h = r*sin(theta)"
        );
    }

    #[test]
    fn rmhd_wave_speed_map_traces_the_magnetosonic_bound() {
        // symbi-hydro's rmhd_magnetosonic_cfl_speeds (the cheap c_f^2 = c_s^2 + c_A^2 -
        // c_s^2 c_A^2 upper bound) at S=Gv, folded into one timestep kernel. it reads the full
        // 3-vector prim + gamma (vsq/bsq), and is ~25x cheaper than the exact quartic. proves
        // the CFL pays no resolvent cubic (asinh/acosh/cos/cosh) — the mignone & del zanna
        // quartic's transcendentals stay on the riemann/flux path only.
        let (k, writes) =
            rmhd_wave_speed_map_gv(Coords::Cartesian, &[Spacing::Uniform; 3], &[0, 1, 2], 3);
        assert_eq!(writes.len(), 1);
        assert_eq!(
            k.scalar_params,
            vec![
                "gamma".to_string(),
                "map_kind_0".to_string(),
                "x_lo_0".to_string(),
                "dx_0".to_string(),
                "map_param_0".to_string(),
                "inv_dx_0".to_string(),
                "map_kind_1".to_string(),
                "x_lo_1".to_string(),
                "dx_1".to_string(),
                "map_param_1".to_string(),
                "inv_dx_1".to_string(),
                "map_kind_2".to_string(),
                "x_lo_2".to_string(),
                "dx_2".to_string(),
                "map_param_2".to_string(),
                "inv_dx_2".to_string(),
            ]
        );
        let keys: Vec<&str> = k.field_inputs.iter().map(|(key, _)| key.as_str()).collect();
        assert_eq!(
            keys,
            vec![
                "prim_rho", "prim_v0", "prim_v1", "prim_v2", "prim_pre", "prim_b0", "prim_b1",
                "prim_b2"
            ]
        );
        assert!(
            !k.graph.has_errors(),
            "graph errors: {:?}",
            k.graph.errors()
        );
        // the magnetosonic bound is free of resolvent-cubic transcendentals.
        //
        // the trigonometric and hyperbolic family alone is a complete signature of the resolvent,
        // even though the resolvent's depressed-cubic branch reaches its root through a cube root:
        // tracing materializes each arm of every conditional, so a resolvent in this graph would
        // carry its `acos`/`cos` and `acosh`/`cosh` branches alongside that cube root. `Pow` is
        // therefore redundant here, and it is separately emitted by the axis map -- a logarithmic
        // axis places its faces at `start*10^(i*slope)` -- so forbidding it would flag the geometry
        // rather than the wave-speed physics.
        use symbi_ir::graph::ElementWiseOp as E;
        let expensive = [
            E::Sin,
            E::Cos,
            E::Acos,
            E::Sinh,
            E::Cosh,
            E::Asinh,
            E::Acosh,
        ];
        let has_transcendental = (0..k.graph.len()).any(|i| {
            matches!(&k.graph.node(NodeId(i as u32)).op,
                Op::ElementWise(op, _) if expensive.contains(op))
        });
        assert!(
            !has_transcendental,
            "CFL bound must not emit the quartic's transcendentals"
        );
        // it still computes a single sqrt (the relativistic-addition discriminant).
        let n_sqrt = (0..k.graph.len())
            .filter(|&i| {
                matches!(
                    &k.graph.node(NodeId(i as u32)).op,
                    Op::ElementWise(E::Sqrt, _)
                )
            })
            .count();
        assert!(
            n_sqrt >= 1,
            "magnetosonic bound needs the discriminant sqrt"
        );
    }

    #[test]
    fn rmhd_wave_speeds_cell_traces_the_exact_quartic() {
        // the per-cell wave-speed kernel: the exact mignone & del zanna quartic per cell, one
        // (lambda_min, lambda_max) pair per direction -> wave_speed_l[d] / wave_speed_r[d].
        // proves it reads the full prim + gamma, writes 6, and carries the resolvent-cubic
        // transcendentals, being the exact quartic — the cost lifted off the flux.
        let (k, writes) = rmhd_wave_speeds_cell_gv(3);
        assert_eq!(writes.len(), 6, "lambda_min/max per 3 directions");
        let out_paths: Vec<String> = writes.iter().map(|(_, p, _)| p.name()).collect();
        assert_eq!(
            out_paths,
            vec![
                "wave_speed_l[0]",
                "wave_speed_r[0]",
                "wave_speed_l[1]",
                "wave_speed_r[1]",
                "wave_speed_l[2]",
                "wave_speed_r[2]",
            ]
        );
        assert_eq!(k.scalar_params, vec!["gamma".to_string()]);
        let keys: Vec<&str> = k.field_inputs.iter().map(|(key, _)| key.as_str()).collect();
        assert_eq!(
            keys,
            vec![
                "prim_rho", "prim_v0", "prim_v1", "prim_v2", "prim_pre", "prim_b0", "prim_b1",
                "prim_b2"
            ]
        );
        assert!(
            !k.graph.has_errors(),
            "graph errors: {:?}",
            k.graph.errors()
        );
        // being the exact quartic, this graph carries the resolvent cubic's transcendentals
        // (the whole point: this kernel pays them once per cell, the flux pays none).
        use symbi_ir::graph::ElementWiseOp as E;
        let has_resolvent = (0..k.graph.len()).any(|i| {
            matches!(
                &k.graph.node(NodeId(i as u32)).op,
                Op::ElementWise(E::Acosh, _)
            )
        });
        assert!(
            has_resolvent,
            "per-cell kernel must carry the exact quartic (resolvent cubic)"
        );
    }

    #[test]
    fn snapshot_gv_traces_a_pure_copy() {
        // u_n = cons: each write root is the read field param (a direct buffer copy), scalar-free
        // and geometry-free. ncomp=2 + energy -> cons den/mom_0/mom_1/nrg -> u_n.*.
        let (k, writes) = snapshot_gv(2, true);
        assert!(k.scalar_params.is_empty(), "snapshot takes no scalars");
        assert!(
            k.coord_components.is_empty(),
            "snapshot is pointwise (no stencil)"
        );
        let in_rt: Vec<String> = k.field_inputs.iter().map(|(_, rt)| rt.name()).collect();
        assert_eq!(
            in_rt,
            vec!["cons.den", "cons.mom_0", "cons.mom_1", "cons.nrg"]
        );
        let out_rt: Vec<String> = writes.iter().map(|(_, rt, _)| rt.name()).collect();
        assert_eq!(out_rt, vec!["u_n.den", "u_n.mom_0", "u_n.mom_1", "u_n.nrg"]);
        assert!(
            !k.graph.has_errors(),
            "graph errors: {:?}",
            k.graph.errors()
        );
    }

    #[test]
    fn schwarzschild_wave_speed_map_wires_the_lapse_mass_scalar() {
        // the schwarzschild wave-speed map threads the banyuls-font coordinate correction
        // (lapse + radial proper-width -> the `schwarzschild_mass` scalar) into the DAG. the flat
        // spherical map omits that scalar and stays bit-identical to the lapse-free form.
        let (k_gr, _) = rhd_wave_speed_map_gv(
            Coords::Spherical,
            Spacetime::SchwarzschildKS,
            &[Spacing::Uniform],
            &[0],
            1,
            EosArm::IdealGamma,
        );
        assert!(
            k_gr.scalar_params.iter().any(|s| s == "schwarzschild_mass"),
            "Schwarzschild wave-speed map must carry the lapse mass scalar; got {:?}",
            k_gr.scalar_params,
        );
        let (k_flat, _) = rhd_wave_speed_map_gv(
            Coords::Spherical,
            Spacetime::Minkowski,
            &[Spacing::Uniform],
            &[0],
            1,
            EosArm::IdealGamma,
        );
        assert!(
            !k_flat
                .scalar_params
                .iter()
                .any(|s| s == "schwarzschild_mass"),
            "flat spherical wave-speed map must NOT carry the lapse mass scalar",
        );
    }

    #[test]
    fn schwarzschild_stage_wires_the_lapse_mass_scalar() {
        // the GR lapse wiring: a schwarzschild-spacetime stage threads the lapse
        // alpha = sqrt(1 - 2M/r) into the DAG, so the host scalar `schwarzschild_mass` appears in
        // the kernel manifest. the flat (minkowski) stage on the same spherical grid omits it and
        // stays bit-identical to the lapse-free flat result. proves the (spherical, schwarzschild) -> metric.lapse path.
        let (k_gr, _) = godunov_stage_gv(
            Coords::Spherical,
            Spacetime::SchwarzschildKS,
            &[Spacing::Uniform],
            &[0],
            1,
            1,
            true,
            GeoSource::Hydro { inertial: false },
        );
        assert!(
            k_gr.scalar_params.iter().any(|s| s == "schwarzschild_mass"),
            "Schwarzschild stage must carry the lapse mass scalar; got {:?}",
            k_gr.scalar_params,
        );

        let (k_flat, _) = godunov_stage_gv(
            Coords::Spherical,
            Spacetime::Minkowski,
            &[Spacing::Uniform],
            &[0],
            1,
            1,
            true,
            GeoSource::Hydro { inertial: false },
        );
        assert!(
            !k_flat
                .scalar_params
                .iter()
                .any(|s| s == "schwarzschild_mass"),
            "flat stage must NOT carry the lapse mass scalar (densitization is a no-op)",
        );
    }

    #[test]
    fn godunov_stage_gv_traces_the_ssp_combine() {
        // in-place `cons = a0*u_n + ac*(u - dt*div(F))`: cartesian-uniform 2D, ncomp=2 + energy
        // (no geometric source). declares dt + the SSP coefficients a0/ac + the per-axis dx; reads
        // the snapshot `u_n` + the conserved fields + the per-direction fluxes (a +e_i stencil, so
        // coord axes recorded); writes the conserved set in place.
        let (k, writes) = godunov_stage_gv(
            Coords::Cartesian,
            Spacetime::Minkowski,
            &[Spacing::Uniform; 2],
            &[0, 1],
            2,
            2,
            true,
            GeoSource::Hydro { inertial: false },
        );
        assert_eq!(
            k.scalar_params,
            vec![
                "dt".to_string(),
                "a0".to_string(),
                "ac".to_string(),
                "mesh_hdil".to_string(),
                "map_kind_0".to_string(),
                "x_lo_0".to_string(),
                "dx_0".to_string(),
                "map_param_0".to_string(),
                "map_kind_1".to_string(),
                "x_lo_1".to_string(),
                "dx_1".to_string(),
                "map_param_1".to_string(),
            ],
        );
        assert_eq!(
            k.coord_components,
            vec![0, 1],
            "the +e_i divergence stencil records both axes"
        );
        let out_rt: Vec<String> = writes.iter().map(|(_, rt, _)| rt.name()).collect();
        assert_eq!(
            out_rt,
            vec!["cons.den", "cons.mom_0", "cons.mom_1", "cons.nrg"],
            "in place"
        );
        let in_rt: Vec<String> = k.field_inputs.iter().map(|(_, rt)| rt.name()).collect();
        // the snapshot reads the SSP `a0*u_n` term needs (held by `snapshot_gv`).
        for rt in ["u_n.den", "u_n.mom_0", "u_n.mom_1", "u_n.nrg"] {
            assert!(
                in_rt.iter().any(|x| x == rt),
                "missing snapshot input {rt}; got {in_rt:?}"
            );
        }
        // the flux components the divergence reads (mass + per-momentum + energy, both axes).
        for rt in [
            "mass_flux[0]",
            "mass_flux[1]",
            "mom_flux_0[0]",
            "mom_flux_1[1]",
            "nrg_flux[0]",
        ] {
            assert!(
                in_rt.iter().any(|x| x == rt),
                "missing flux input {rt}; got {in_rt:?}"
            );
        }
        assert!(
            !k.graph.has_errors(),
            "graph errors: {:?}",
            k.graph.errors()
        );
    }

    #[test]
    fn fused_built_core_matches_spec_adapter_trace() {
        // the SourceSpec entry (`godunov_stage_gv_with_fused_sources`) and the BuiltSource core
        // (`godunov_stage_gv_with_fused_built`) emit an identical godunov+source kernel — same
        // ABI manifest, same writes, same lowered source. the AOT spec path and the runtime
        // BuiltSource path share a single trace, so they stay in lockstep. a family depending on
        // both position and energy (mom + nrg) exercises the centroid `x_k` binding and the energy
        // overlay, the parts most likely to diverge under a bad split.
        use symbi_ir::emit::{Precision, Target, TargetConfig};
        use symbi_ir::{KernelEmitInputs, emit_kernel_from_lowering};

        let specs = symbi_hydro::source_spec::point_mass_gravity_sources(2, true);
        let spec_refs: Vec<&symbi_hydro::source_spec::SourceSpec> = specs.iter().collect();
        let (coords, spacing, axes) = (Coords::Cartesian, [Spacing::Uniform; 2], [0usize, 1]);
        let geo = GeoSource::Hydro { inertial: false };

        // the compile-time spec path.
        let (k_spec, w_spec) = godunov_stage_gv_with_fused_sources(
            coords,
            Spacetime::Minkowski,
            &spacing,
            &axes,
            2,
            2,
            true,
            geo,
            &spec_refs,
            false,
        );

        // the runtime BuiltSource-value path (what `RuntimeSource` feeds).
        let builts: Vec<(&str, symbi_hydro::source_spec::BuiltSource)> = specs
            .iter()
            .map(|s| (s.target_field, (s.build_source)(2)))
            .collect();
        let src_refs: Vec<(&str, &symbi_hydro::source_spec::BuiltSource)> =
            builts.iter().map(|(t, b)| (*t, b)).collect();
        let (k_built, w_built) = godunov_stage_gv_with_fused_built(
            coords,
            Spacetime::Minkowski,
            &spacing,
            &axes,
            2,
            2,
            true,
            geo,
            &src_refs,
            false,
            0,
        );

        // the ABI manifest + writes are identical (NodeIds match because both trace the same op
        // sequence — building the BuiltSource values outside the trace leaves the node allocation
        // untouched).
        assert_eq!(
            k_spec.field_inputs, k_built.field_inputs,
            "field_inputs drift"
        );
        assert_eq!(
            k_spec.scalar_params, k_built.scalar_params,
            "scalar_params drift"
        );
        assert_eq!(
            k_spec.coord_components, k_built.coord_components,
            "coord_components drift"
        );
        assert_eq!(w_spec, w_built, "writes drift");

        // the lowered source is byte-identical — the strongest structural equality available
        // (`Graph` has no `PartialEq`; the emitted source captures the full computation).
        let emit = |k: &GvKernel, w: &[(String, FieldBind, NodeId)]| {
            let spec = KernelEmitInputs {
                kernel_name: "fused_eq",
                coalesce_layout: false,
                ndim: 2,
                target: TargetConfig {
                    target: Target::Cuda,
                    precision: Precision::F64,
                },
                field_inputs: &k.field_inputs,
                scalar_params: &k.scalar_params,
                field_writes: w,
                coord_components: &k.coord_components,
                device_preamble: &[],
                tile_spec: None,
            };
            emit_kernel_from_lowering(&k.graph, &spec).source
        };
        assert_eq!(
            emit(&k_spec, &w_spec),
            emit(&k_built, &w_built),
            "lowered source drift"
        );
    }

    /// the extremum-preserving interface pair at S = f64 on smooth data: fed exact cell
    /// AVERAGES of sin(2 pi x), the corrected interfaces must converge to the profile's
    /// interface point values at 4th order (the unlimited stencil's order), i.e. halving
    /// h cuts the error ~16x. a limiter arm that engages spuriously on smooth data (or a
    /// mis-centered stencil) shows up as a collapsed ratio here, independent of any
    /// kernel emission or dispatch machinery.
    #[test]
    fn ppm_ep_interfaces_are_fourth_order_on_smooth_averages() {
        let tau = std::f64::consts::TAU;
        // exact cell average of sin over [x-h/2, x+h/2]
        let avg = |x: f64, h: f64| (tau * x).sin() * (tau * h / 2.0).sin() / (tau * h / 2.0);
        // max interface error over cells centered at x0 + k h, k in a period
        let max_err = |h: f64| -> f64 {
            let mut worst = 0.0_f64;
            let n = (1.0 / h) as usize;
            for kk in 0..n {
                let xc = (kk as f64 + 0.5) * h;
                let v: Vec<f64> = (-2..=2).map(|o| avg(xc + o as f64 * h, h)).collect();
                let (a_l, a_r) = super::ppm_cell_interfaces_ep(v[0], v[1], v[2], v[3], v[4]);
                worst = worst
                    .max((a_l - (tau * (xc - h / 2.0)).sin()).abs())
                    .max((a_r - (tau * (xc + h / 2.0)).sin()).abs());
            }
            worst
        };
        let e1 = max_err(1.0 / 64.0);
        let e2 = max_err(1.0 / 128.0);
        let ratio = e1 / e2;
        assert!(
            ratio > 12.0,
            "extremum-preserving ppm interfaces are not ~4th order on smooth averages: \
             err(1/64)={e1:.3e}, err(1/128)={e2:.3e}, ratio={ratio:.2} (expect ~16)"
        );
    }
}
