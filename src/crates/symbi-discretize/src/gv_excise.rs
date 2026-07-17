// =============================================================================
// gv_excise.rs
//
// the traced horizon-excision fill for the cartesian kerr-schild charts: three
// kernels that together overwrite every excised cell with a zero-gradient copy
// of its outward neighbor's gas primitive state, then rebuild the conserved
// state with the cell's own metric. the excised region is the sublevel set of
// the kerr-schild radius r_ks(x; a) < r_exc — the sphere about the chart
// origin at a = 0, the oblate spheroid (x^2 + y^2)/(r_exc^2 + a^2) +
// z^2/r_exc^2 < 1 at spin a about z (the r = const surfaces of the chart).
// inside the horizon every characteristic points inward, so the filled values
// are numerical padding the exterior never sees.
//
//   excise_fill      prim (own + diagonals) -> scratch = one onion sweep
//   excise_writeback scratch -> prim                    (the sweep commit)
//   excise_p2c       prim -> cons, valencia to_conserved at the cell centroid,
//                    excised cells only (live cells pass their cons through)
//
// the fill carries the GAS primitives only (rho, v, p). the magnetized p2c
// reads the cell's OWN B (the face average the constrained transport owns) and
// rebuilds (D, S_i, tau) with it — the staggered faces are never written, so
// the densitized div(B) invariant survives excision by construction.
//
// the fill/writeback pair runs onion_pass_count times (values propagate one
// diagonal cell inward per sweep); p2c runs once after the last sweep. the
// dispatch box is the excision region's index bbox — computed host-side, so
// no output-support declaration rides the artifacts.
//
// usage (build.rs):
//   let (k, writes) = excise_fill_gv();
//   emit_gv(out, "excise_fill_2d", 2, &k, &writes);
// =============================================================================

use symbi_algebra::algebra::Numeric;
use symbi_algebra::Tensor;
use symbi_geometry::{KerrKSCartesian, Metric};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::regime::Regime;
use symbi_hydro::spatial_metric::{Gamma, GammaInv, SpatialMetric};
use symbi_hydro::state::Prim;
use symbi_hydro::{MhdPrim, RhdGr, RmhdGr};
use symbi_ib::excise::{ks_excised, onion_fill_cell, onion_fill_cell_3d};
use symbi_ir::algebra::Scalar;
use symbi_ir::gv::Writes;
use symbi_ir::{begin_trace, end_trace, FieldRef, Gv, GvKernel};

use crate::coords::{Coords, Spacing};
use crate::gv::cell_geometry_gv;

/// the gas-primitive FieldRefs in storage order: rho, vel_0..dof-1, pre.
fn prim_refs(dof: usize) -> Vec<FieldRef> {
    let mut refs = vec![FieldRef::PrimRho];
    for kk in 0..dof {
        refs.push(FieldRef::PrimVel(kk as u8));
    }
    refs.push(FieldRef::PrimPre);
    refs
}

fn prim_names(dof: usize) -> Vec<String> {
    let mut names = vec!["rho".to_string()];
    for kk in 0..dof {
        names.push(format!("vel_{kk}"));
    }
    names.push("pre".to_string());
    names
}

/// the cell centroid of the uniform cartesian grid, as traced coordinates.
fn centroid(ndim: usize) -> Vec<Gv> {
    let spacing = vec![Spacing::Uniform; ndim];
    let axes: Vec<usize> = (0..ndim).collect();
    let geo = cell_geometry_gv(Coords::Cartesian, &spacing, &axes, ndim);
    geo.centroid
}

/// the excision mask at the traced centroid: r_ks(x; a) < r_exc with the
/// host-filled `kerr_spin` / `excision_radius` scalars (spin = 0 for the
/// schwarzschild chart). ONE definition shared by the fill and the rebuild.
fn excised_mask_2d(x: &[Gv; 2]) -> <Gv as Scalar>::Mask {
    ks_excised(x, Gv::scalar("kerr_spin"), Gv::scalar("excision_radius"))
}

fn excised_mask_3d(x: &[Gv; 3]) -> <Gv as Scalar>::Mask {
    ks_excised(x, Gv::scalar("kerr_spin"), Gv::scalar("excision_radius"))
}

/// one 2d onion sweep over `1 + dof + 1` gas primitives: every excised cell
/// takes the primitive state of its diagonal-outward neighbor; live cells copy
/// their own state. writes the swept state to the exc_0.. scratch (the commit
/// is `excise_writeback`, so the parallel stencil never reads a value written
/// by the same sweep). `dof = 2` is the in-plane GR-hydro state; `dof = 3`
/// carries the out-of-plane momentum of the 2.5d MHD state.
fn excise_fill_2d_dof_gv(dof: usize) -> (GvKernel, Writes) {
    begin_trace();
    let names = prim_names(dof);
    let refs = prim_refs(dof);
    let nf = refs.len();

    let read_at = |suffix: &str, off: [i32; 2]| -> Vec<Gv> {
        (0..nf)
            .map(|kk| Gv::field_offset(&format!("{}_{suffix}", names[kk]), refs[kk], 2, &off))
            .collect()
    };
    let own: Vec<Gv> = (0..nf).map(|kk| Gv::field(&names[kk], refs[kk])).collect();
    let pp = read_at("pp", [1, 1]);
    let pm = read_at("pm", [1, -1]);
    let mp = read_at("mp", [-1, 1]);
    let mm = read_at("mm", [-1, -1]);

    let c = centroid(2);
    let x = [c[0], c[1]];
    let excised = excised_mask_2d(&x);
    // component-wise sweep through the shared fixed-arity core (one mask, NF selects).
    let filled: Vec<Gv> = (0..nf)
        .map(|kk| onion_fill_cell([own[kk]], [pp[kk]], [pm[kk]], [mp[kk]], [mm[kk]], x, excised)[0])
        .collect();

    let mut writes: Writes = Vec::new();
    for (kk, val) in filled.iter().enumerate() {
        writes.push((format!("exc_out_{kk}"), format!("exc_{kk}").into(), val.node()));
    }
    (end_trace(), writes)
}

pub fn excise_fill_gv() -> (GvKernel, Writes) {
    excise_fill_2d_dof_gv(2)
}

/// the 2.5d (dof = 3) gas fill: rho, vel_0..2, pre on the 2d grid — the
/// magnetized equatorial slice's momentum set.
pub fn excise_fill_dof3_gv() -> (GvKernel, Writes) {
    excise_fill_2d_dof_gv(3)
}

/// the sweep commit: copy the exc scratch back into the primitive fields.
/// unmasked over the dispatch box — the fill wrote live cells' own values,
/// so the copy is the bitwise identity there.
fn excise_writeback_dof_gv(dof: usize) -> (GvKernel, Writes) {
    begin_trace();
    let names = prim_names(dof);
    let refs = prim_refs(dof);
    let vals: Vec<Gv> = (0..refs.len())
        .map(|kk| Gv::field(&format!("exc_{kk}"), format!("exc_{kk}")))
        .collect();
    let mut writes: Writes = Vec::new();
    for kk in 0..refs.len() {
        writes.push((format!("{}_out", names[kk]), refs[kk].into(), vals[kk].node()));
    }
    (end_trace(), writes)
}

pub fn excise_writeback_gv() -> (GvKernel, Writes) {
    excise_writeback_dof_gv(2)
}

pub fn excise_writeback_dof3_gv() -> (GvKernel, Writes) {
    excise_writeback_dof_gv(3)
}

/// rebuild the conserved state of every excised cell from its (just-filled)
/// primitives: the valencia `to_conserved` (covariant S_i = rho h W^2 gamma_ij v^j)
/// with the cartesian kerr-schild spatial metric at the cell's own centroid — a
/// donor cell's conserved state carries the donor's metric factors, so only the
/// primitive copy + local rebuild is exact. live cells pass their conserved
/// state through untouched (in-place select). the metric is the spinning-kerr
/// rank-1 form with the host-filled `kerr_spin` (zero for the a = 0 chart).
pub fn excise_p2c_gv() -> (GvKernel, Writes) {
    begin_trace();
    let gamma = Gv::scalar("gamma");
    let mass = Gv::scalar("schwarzschild_mass");
    let spin = Gv::scalar("kerr_spin");

    let rho = Gv::field("rho", FieldRef::PrimRho);
    let vel: [Gv; 2] =
        std::array::from_fn(|kk| Gv::field(&format!("vel_{kk}"), FieldRef::PrimVel(kk as u8)));
    let pre = Gv::field("pre", FieldRef::PrimPre);
    let den = Gv::field("den", FieldRef::cons_den());
    let mom: [Gv; 2] =
        std::array::from_fn(|kk| Gv::field(&format!("mom_{kk}"), FieldRef::cons_mom(kk as u8)));
    let nrg = Gv::field("nrg", FieldRef::cons_nrg());

    let c = centroid(2);
    let x = [c[0], c[1]];
    let excised = excised_mask_2d(&x);

    let xt = Tensor::<Gv, 2>::new(x);
    let m = KerrKSCartesian { mass, spin };
    let metric = SpatialMetric::<Gv, 2>::new(
        Gamma::new(m.spatial_metric(xt)),
        GammaInv::new(m.spatial_metric_inv(xt)),
    );
    // the lapse never enters to_conserved; the regime carries it for the fluxes only.
    let regime = RhdGr { metric, alpha: Gv::ONE };
    let prim = Prim::<Gv, 2> { rho, vel: Tensor::new(vel), pre };
    let cons = regime.to_conserved(&IdealGas { gamma }, &prim);

    let mut writes: Writes = Vec::new();
    writes.push((
        "den_out".to_string(),
        FieldRef::cons_den().into(),
        Gv::select(excised, cons.den, den).node(),
    ));
    for kk in 0..2 {
        writes.push((
            format!("mom_out_{kk}"),
            FieldRef::cons_mom(kk as u8).into(),
            Gv::select(excised, cons.mom[kk], mom[kk]).node(),
        ));
    }
    writes.push((
        "nrg_out".to_string(),
        FieldRef::cons_nrg().into(),
        Gv::select(excised, cons.nrg, nrg).node(),
    ));
    (end_trace(), writes)
}

/// the 3d gas fill: every excised cell takes the primitive state of its outward
/// CORNER-diagonal neighbor (sign-selected on all three axes); live cells copy
/// their own state. the rho/vel_0..2/pre set serves BOTH the 3d GR-hydro state
/// and the 3d magnetized gas state (the field rides the staggered faces, never
/// the fill).
pub fn excise_fill_3d_gv() -> (GvKernel, Writes) {
    begin_trace();
    let names = prim_names(3);
    let refs = prim_refs(3);

    let read_at = |suffix: &str, off: [i32; 3]| -> [Gv; 5] {
        std::array::from_fn(|kk| {
            Gv::field_offset(&format!("{}_{suffix}", names[kk]), refs[kk], 3, &off)
        })
    };
    let own: [Gv; 5] = std::array::from_fn(|kk| Gv::field(&names[kk], refs[kk]));
    // the 8 corner diagonals in z-fastest sign order (the order onion_fill_cell_3d selects by).
    let signs = [
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1],
    ];
    let tags = ["mmm", "mmp", "mpm", "mpp", "pmm", "pmp", "ppm", "ppp"];
    let diags: [[Gv; 5]; 8] = std::array::from_fn(|dd| read_at(tags[dd], signs[dd]));

    let c = centroid(3);
    let x = [c[0], c[1], c[2]];
    let excised = excised_mask_3d(&x);
    let filled = onion_fill_cell_3d(own, &diags, x, excised);

    let mut writes: Writes = Vec::new();
    for (kk, val) in filled.iter().enumerate() {
        writes.push((format!("exc_out_{kk}"), format!("exc_{kk}").into(), val.node()));
    }
    (end_trace(), writes)
}

/// the 3d sweep commit: copy the exc scratch back into the primitive fields.
pub fn excise_writeback_3d_gv() -> (GvKernel, Writes) {
    excise_writeback_dof_gv(3)
}

/// the 3d conserved rebuild of every excised cell from its (just-filled) primitives:
/// the valencia `to_conserved` with the (spin-generic) cartesian kerr-schild spatial
/// metric at the cell's own centroid; live cells pass their conserved state through
/// untouched.
pub fn excise_p2c_3d_gv() -> (GvKernel, Writes) {
    begin_trace();
    let gamma = Gv::scalar("gamma");
    let mass = Gv::scalar("schwarzschild_mass");
    let spin = Gv::scalar("kerr_spin");

    let rho = Gv::field("rho", FieldRef::PrimRho);
    let vel: [Gv; 3] =
        std::array::from_fn(|kk| Gv::field(&format!("vel_{kk}"), FieldRef::PrimVel(kk as u8)));
    let pre = Gv::field("pre", FieldRef::PrimPre);
    let den = Gv::field("den", FieldRef::cons_den());
    let mom: [Gv; 3] =
        std::array::from_fn(|kk| Gv::field(&format!("mom_{kk}"), FieldRef::cons_mom(kk as u8)));
    let nrg = Gv::field("nrg", FieldRef::cons_nrg());

    let c = centroid(3);
    let x = [c[0], c[1], c[2]];
    let excised = excised_mask_3d(&x);

    let xt = Tensor::<Gv, 3>::new(x);
    let m = KerrKSCartesian { mass, spin };
    let metric = SpatialMetric::<Gv, 3>::new(
        Gamma::new(m.spatial_metric(xt)),
        GammaInv::new(m.spatial_metric_inv(xt)),
    );
    // the lapse never enters to_conserved; the regime carries it for the fluxes only.
    let regime = RhdGr { metric, alpha: Gv::ONE };
    let prim = Prim::<Gv, 3> { rho, vel: Tensor::new(vel), pre };
    let cons = regime.to_conserved(&IdealGas { gamma }, &prim);

    let mut writes: Writes = Vec::new();
    writes.push((
        "den_out".to_string(),
        FieldRef::cons_den().into(),
        Gv::select(excised, cons.den, den).node(),
    ));
    for kk in 0..3 {
        writes.push((
            format!("mom_out_{kk}"),
            FieldRef::cons_mom(kk as u8).into(),
            Gv::select(excised, cons.mom[kk], mom[kk]).node(),
        ));
    }
    writes.push((
        "nrg_out".to_string(),
        FieldRef::cons_nrg().into(),
        Gv::select(excised, cons.nrg, nrg).node(),
    ));
    (end_trace(), writes)
}

/// the MAGNETIZED conserved rebuild of every excised cell: the ideal-GRMHD
/// valencia `to_conserved` — S_i = (rho h W^2 + B^2) v_i - (v.B) B_i and
/// tau = rho h W^2 + B^2 - (p + b^2/2) - D, all contractions through the
/// cell-centroid spatial metric — from the just-filled GAS primitives and the
/// cell's OWN B (the face average the constrained transport owns). the
/// staggered faces are never written, so d_i(sqrt(gamma) B^i) = 0 survives
/// excision identically; the conserved B slots alias the cell B and pass
/// through untouched. MHD momentum/velocity vectors are always 3-component
/// (the 2d grid instance is the equatorial slice with z = 0 in the metric
/// position), so one builder serves both grid dimensions.
fn excise_p2c_mhd_dim_gv(ndim: usize) -> (GvKernel, Writes) {
    begin_trace();
    let gamma = Gv::scalar("gamma");
    let mass = Gv::scalar("schwarzschild_mass");
    let spin = Gv::scalar("kerr_spin");

    let rho = Gv::field("rho", FieldRef::PrimRho);
    let vel: [Gv; 3] =
        std::array::from_fn(|kk| Gv::field(&format!("vel_{kk}"), FieldRef::PrimVel(kk as u8)));
    let pre = Gv::field("pre", FieldRef::PrimPre);
    let mag: [Gv; 3] =
        std::array::from_fn(|kk| Gv::field(&format!("bc_{kk}"), FieldRef::BCell(kk as u8)));
    let den = Gv::field("den", FieldRef::cons_den());
    let mom: [Gv; 3] =
        std::array::from_fn(|kk| Gv::field(&format!("mom_{kk}"), FieldRef::cons_mom(kk as u8)));
    let nrg = Gv::field("nrg", FieldRef::cons_nrg());

    let c = centroid(ndim);
    // the metric position padded to 3 slots (the 2d grid is the z = 0 equatorial slice).
    let x3: [Gv; 3] = std::array::from_fn(|kk| c.get(kk).copied().unwrap_or(Gv::ZERO));
    let excised = if ndim == 2 {
        excised_mask_2d(&[x3[0], x3[1]])
    } else {
        excised_mask_3d(&x3)
    };

    let xt = Tensor::<Gv, 3>::new(x3);
    let m = KerrKSCartesian { mass, spin };
    let metric = SpatialMetric::<Gv, 3>::new(
        Gamma::new(m.spatial_metric(xt)),
        GammaInv::new(m.spatial_metric_inv(xt)),
    );
    let regime = RmhdGr { metric, alpha: Gv::ONE };
    let prim = MhdPrim::<Gv, 3> {
        hydro: Prim { rho, vel: Tensor::new(vel), pre },
        mag: Tensor::new(mag),
    };
    let cons = regime.to_conserved(&IdealGas { gamma }, &prim);

    let mut writes: Writes = Vec::new();
    writes.push((
        "den_out".to_string(),
        FieldRef::cons_den().into(),
        Gv::select(excised, cons.hydro.den, den).node(),
    ));
    for kk in 0..3 {
        writes.push((
            format!("mom_out_{kk}"),
            FieldRef::cons_mom(kk as u8).into(),
            Gv::select(excised, cons.hydro.mom[kk], mom[kk]).node(),
        ));
    }
    writes.push((
        "nrg_out".to_string(),
        FieldRef::cons_nrg().into(),
        Gv::select(excised, cons.hydro.nrg, nrg).node(),
    ));
    (end_trace(), writes)
}

pub fn excise_p2c_mhd_gv() -> (GvKernel, Writes) {
    excise_p2c_mhd_dim_gv(2)
}

pub fn excise_p2c_mhd_3d_gv() -> (GvKernel, Writes) {
    excise_p2c_mhd_dim_gv(3)
}
