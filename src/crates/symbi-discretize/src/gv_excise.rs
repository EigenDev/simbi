// =============================================================================
// gv_excise.rs
//
// the traced horizon-excision fill for the cartesian kerr-schild chart: three
// kernels that together overwrite every excised cell (|x| < r_exc, the sphere
// about the chart origin where the black hole sits) with a zero-gradient copy
// of its outward neighbor's primitive state, then rebuild the conserved state
// with the cell's own metric. inside the horizon every characteristic points
// inward, so the filled values are numerical padding the exterior never sees.
//
//   excise_fill      prim (own + 4 diagonals) -> scratch = one onion sweep
//   excise_writeback scratch -> prim                     (the sweep commit)
//   excise_p2c       prim -> cons, valencia to_conserved at the cell centroid,
//                    excised cells only (live cells pass their cons through)
//
// the fill/writeback pair runs onion_pass_count times (values propagate one
// diagonal cell inward per sweep); p2c runs once after the last sweep. the
// dispatch box is the excision sphere's index bbox — computed host-side, so
// no output-support declaration rides the artifacts.
//
// usage (build.rs):
//   let (k, writes) = excise_fill_gv();
//   emit_gv(out, "excise_fill_2d", 2, &k, &writes);
// =============================================================================

use symbi_algebra::algebra::Numeric;
use symbi_algebra::Tensor;
use symbi_geometry::{Metric, SchwarzschildKSCartesian};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::regime::Regime;
use symbi_hydro::spatial_metric::{Gamma, GammaInv, SpatialMetric};
use symbi_hydro::state::Prim;
use symbi_hydro::RhdGr;
use symbi_ib::excise::onion_fill_cell;
use symbi_ir::algebra::Scalar;
use symbi_ir::gv::Writes;
use symbi_ir::{begin_trace, end_trace, FieldRef, Gv, GvKernel};

use crate::coords::{Coords, Spacing};
use crate::gv::cell_geometry_gv;

/// the primitive component count of the 2d GR-hydro state: rho, vel_0, vel_1, pre.
const NF: usize = 4;

/// the primitive FieldRefs in storage order.
fn prim_refs() -> [FieldRef; NF] {
    [FieldRef::PrimRho, FieldRef::PrimVel(0), FieldRef::PrimVel(1), FieldRef::PrimPre]
}

/// the cell centroid of the uniform 2d cartesian grid, as traced coordinates.
fn centroid_2d() -> [Gv; 2] {
    let geo = cell_geometry_gv(Coords::Cartesian, &[Spacing::Uniform, Spacing::Uniform], &[0, 1], 2);
    [geo.centroid[0], geo.centroid[1]]
}

/// one onion sweep: every excised cell takes the primitive state of its
/// diagonal-outward neighbor; live cells copy their own state. writes the
/// swept state to the exc_0..exc_3 scratch (the commit is `excise_writeback`,
/// so the parallel stencil never reads a value written by the same sweep).
pub fn excise_fill_gv() -> (GvKernel, Writes) {
    begin_trace();
    let r_exc = Gv::scalar("excision_radius");
    let names = ["rho", "vel_0", "vel_1", "pre"];
    let refs = prim_refs();

    let read_at = |suffix: &str, off: [i32; 2]| -> [Gv; NF] {
        std::array::from_fn(|kk| {
            Gv::field_offset(&format!("{}_{suffix}", names[kk]), refs[kk], 2, &off)
        })
    };
    let own: [Gv; NF] = std::array::from_fn(|kk| Gv::field(names[kk], refs[kk]));
    let pp = read_at("pp", [1, 1]);
    let pm = read_at("pm", [1, -1]);
    let mp = read_at("mp", [-1, 1]);
    let mm = read_at("mm", [-1, -1]);

    let x = centroid_2d();
    let filled = onion_fill_cell(own, pp, pm, mp, mm, x, r_exc);

    let mut writes: Writes = Vec::new();
    for (kk, val) in filled.iter().enumerate() {
        writes.push((format!("exc_out_{kk}"), format!("exc_{kk}").into(), val.node()));
    }
    (end_trace(), writes)
}

/// the sweep commit: copy the exc scratch back into the primitive fields.
/// unmasked over the dispatch box — the fill wrote live cells' own values,
/// so the copy is the bitwise identity there.
pub fn excise_writeback_gv() -> (GvKernel, Writes) {
    begin_trace();
    let names = ["rho", "vel_0", "vel_1", "pre"];
    let refs = prim_refs();
    let vals: [Gv; NF] =
        std::array::from_fn(|kk| Gv::field(&format!("exc_{kk}"), format!("exc_{kk}")));
    let mut writes: Writes = Vec::new();
    for kk in 0..NF {
        writes.push((format!("{}_out", names[kk]), refs[kk].into(), vals[kk].node()));
    }
    (end_trace(), writes)
}

/// rebuild the conserved state of every excised cell from its (just-filled)
/// primitives: the valencia `to_conserved` (covariant S_i = rho h W^2 gamma_ij v^j)
/// with the cartesian kerr-schild spatial metric at the cell's own centroid — a
/// donor cell's conserved state carries the donor's metric factors, so only the
/// primitive copy + local rebuild is exact. live cells pass their conserved
/// state through untouched (in-place select).
pub fn excise_p2c_gv() -> (GvKernel, Writes) {
    begin_trace();
    let gamma = Gv::scalar("gamma");
    let mass = Gv::scalar("schwarzschild_mass");
    let r_exc = Gv::scalar("excision_radius");

    let rho = Gv::field("rho", FieldRef::PrimRho);
    let vel: [Gv; 2] =
        std::array::from_fn(|kk| Gv::field(&format!("vel_{kk}"), FieldRef::PrimVel(kk as u8)));
    let pre = Gv::field("pre", FieldRef::PrimPre);
    let den = Gv::field("den", FieldRef::cons_den());
    let mom: [Gv; 2] =
        std::array::from_fn(|kk| Gv::field(&format!("mom_{kk}"), FieldRef::cons_mom(kk as u8)));
    let nrg = Gv::field("nrg", FieldRef::cons_nrg());

    let x = centroid_2d();
    // the excision mask, spelled EXACTLY as the fill spells it: one shared
    // definition of "excised" across the sweep and the rebuild.
    let r = (x[0] * x[0] + x[1] * x[1]).sqrt();
    let excised = r.cmp_lt(r_exc);

    let xt = Tensor::<Gv, 2>::new(x);
    let m = SchwarzschildKSCartesian { mass };
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

/// the primitive component count of the 3d GR-hydro state: rho, vel_0..2, pre.
const NF3: usize = 5;

fn prim_refs_3() -> [FieldRef; NF3] {
    [
        FieldRef::PrimRho,
        FieldRef::PrimVel(0),
        FieldRef::PrimVel(1),
        FieldRef::PrimVel(2),
        FieldRef::PrimPre,
    ]
}

/// the cell centroid of the uniform 3d cartesian grid, as traced coordinates.
fn centroid_3d() -> [Gv; 3] {
    let geo = cell_geometry_gv(
        Coords::Cartesian,
        &[Spacing::Uniform, Spacing::Uniform, Spacing::Uniform],
        &[0, 1, 2],
        3,
    );
    [geo.centroid[0], geo.centroid[1], geo.centroid[2]]
}

/// one 3d onion sweep: every excised cell takes the primitive state of its outward
/// CORNER-diagonal neighbor (sign-selected on all three axes); live cells copy their
/// own state. writes the swept state to the exc_0..exc_4 scratch.
pub fn excise_fill_3d_gv() -> (GvKernel, Writes) {
    begin_trace();
    let r_exc = Gv::scalar("excision_radius");
    let names = ["rho", "vel_0", "vel_1", "vel_2", "pre"];
    let refs = prim_refs_3();

    let read_at = |suffix: &str, off: [i32; 3]| -> [Gv; NF3] {
        std::array::from_fn(|kk| {
            Gv::field_offset(&format!("{}_{suffix}", names[kk]), refs[kk], 3, &off)
        })
    };
    let own: [Gv; NF3] = std::array::from_fn(|kk| Gv::field(names[kk], refs[kk]));
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
    let diags: [[Gv; NF3]; 8] = std::array::from_fn(|dd| read_at(tags[dd], signs[dd]));

    let x = centroid_3d();
    let filled = symbi_ib::excise::onion_fill_cell_3d(own, &diags, x, r_exc);

    let mut writes: Writes = Vec::new();
    for (kk, val) in filled.iter().enumerate() {
        writes.push((format!("exc_out_{kk}"), format!("exc_{kk}").into(), val.node()));
    }
    (end_trace(), writes)
}

/// the 3d sweep commit: copy the exc scratch back into the primitive fields.
pub fn excise_writeback_3d_gv() -> (GvKernel, Writes) {
    begin_trace();
    let names = ["rho", "vel_0", "vel_1", "vel_2", "pre"];
    let refs = prim_refs_3();
    let vals: [Gv; NF3] =
        std::array::from_fn(|kk| Gv::field(&format!("exc_{kk}"), format!("exc_{kk}")));
    let mut writes: Writes = Vec::new();
    for kk in 0..NF3 {
        writes.push((format!("{}_out", names[kk]), refs[kk].into(), vals[kk].node()));
    }
    (end_trace(), writes)
}

/// the 3d conserved rebuild of every excised cell from its (just-filled) primitives:
/// the valencia `to_conserved` with the cartesian kerr-schild spatial metric at the
/// cell's own centroid; live cells pass their conserved state through untouched.
pub fn excise_p2c_3d_gv() -> (GvKernel, Writes) {
    begin_trace();
    let gamma = Gv::scalar("gamma");
    let mass = Gv::scalar("schwarzschild_mass");
    let r_exc = Gv::scalar("excision_radius");

    let rho = Gv::field("rho", FieldRef::PrimRho);
    let vel: [Gv; 3] =
        std::array::from_fn(|kk| Gv::field(&format!("vel_{kk}"), FieldRef::PrimVel(kk as u8)));
    let pre = Gv::field("pre", FieldRef::PrimPre);
    let den = Gv::field("den", FieldRef::cons_den());
    let mom: [Gv; 3] =
        std::array::from_fn(|kk| Gv::field(&format!("mom_{kk}"), FieldRef::cons_mom(kk as u8)));
    let nrg = Gv::field("nrg", FieldRef::cons_nrg());

    let x = centroid_3d();
    // the excision mask, spelled EXACTLY as the fill spells it: one shared definition
    // of "excised" across the sweep and the rebuild.
    let r = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
    let excised = r.cmp_lt(r_exc);

    let xt = Tensor::<Gv, 3>::new(x);
    let m = SchwarzschildKSCartesian { mass };
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
