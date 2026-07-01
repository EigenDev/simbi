// =============================================================================
// ghost.rs
//
// lattice-map ghost-fill kernel builders (the boundary pullback).
// =============================================================================

use super::*;


/// the isothermal lattice-map ghost fill — pull back rho/vel/pre at the per-axis source coord,
/// write IN PLACE; the velocity component whose coordinate is a GRID axis picks up that axis's
/// wall-normal `vel_sign` (an ungridded swirl coordinate has no wall map -> unflipped). rho/pre
/// are grade-0 copies. `ncomp` velocity components, `ndim` gridded axes; `axes[d]` = the coord
/// of grid axis d. the EOS-generic 3-field pullback the iso/newton/rhd ghost fill share.
pub fn iso_ghost_fill_gv(
    ndim: usize,
    ncomp: usize,
    axes: &[usize],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let src = gv_lattice_source(ndim);
    let vel_sign: Vec<Gv> = (0..ndim).map(|ax| Gv::scalar(&format!("vel_sign_{ax}"))).collect();
    let rho = gv_load_at("prim_rho", "prim.rho", &src);
    let mut writes = vec![("prim_rho".to_string(), FieldRef::PrimRho.into(), rho.node())];
    for k in 0..ncomp {
        let v = gv_load_at(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8), &src);
        // grade-1 wall flip on the grid axis whose coordinate IS k; ungridded -> unflipped.
        let v = match axes.iter().position(|&c| c == k) {
            Some(ax) => v * vel_sign[ax],
            None => v,
        };
        writes.push((format!("prim_v{k}"), FieldRef::PrimVel(k as u8).into(), v.node()));
    }
    let pre = gv_load_at("prim_pre", "prim.pre", &src);
    writes.push(("prim_pre".to_string(), FieldRef::PrimPre.into(), pre.node()));
    (end_trace(), writes)
}


/// the SINGLE-SCALAR lattice-map ghost fill: pull back one field "f" at the per-axis
/// integer source coord, times the runtime grade `sign` (+1 for a scalar copy or a
/// tangential staggered component; -1 for a wall-normal component under a reflect
/// map). the staggered `bface` transverse-halo fill dispatches this per component —
/// the field resolves the region's absolute coords against its OWN staggered lo, so
/// the same kernel serves any cell- or face-anchored scalar.
pub fn scalar_ghost_fill_gv(ndim: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let src = gv_lattice_source(ndim);
    let sign = Gv::scalar("sign");
    let v = gv_load_at("f", "f", &src) * sign;
    let writes = vec![("f".to_string(), "f".into(), v.node())];
    (end_trace(), writes)
}


// the per-vector-component wall-map sign: the in-plane components (k < ndim) pick up the
// boundary axis's reflect sign (B/vel are grade-1 vectors under the wall map); the out-of-
// plane components (k >= ndim, e.g., Bz/vz in 1.5D/2.5D) are tangential to every grid-axis
// wall, so they copy unchanged (sign = +1). this is why ghost fill loops 0..ncomp (DOF),
// NOT 0..ndim — else the out-of-plane ghosts stay zero and drain the boundary.
fn gv_ghost_sign(k: usize, ndim: usize, vel_sign: &[Gv]) -> Gv {
    if k < ndim { vel_sign[k] } else { Gv::ONE }
}


/// the RMHD lattice-map ghost fill — `iso_ghost_fill_gv` plus the cell-centered B: pull back
/// rho/vel/pre + `mhd.bcell[k]`, the velocity AND B (DOF-vectors) picking up the per-axis
/// `vel_sign` for in-plane components and copying the out-of-plane ones. `ndim` = grid axes
/// (the lattice source + reflect signs), `ncomp` = vector components (DOF).
pub fn rmhd_ghost_fill_gv(ndim: usize, ncomp: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let src = gv_lattice_source(ndim);
    let vel_sign: Vec<Gv> = (0..ndim).map(|k| Gv::scalar(&format!("vel_sign_{k}"))).collect();
    let rho = gv_load_at("prim_rho", "prim.rho", &src);
    let mut writes = vec![("prim_rho".to_string(), FieldRef::PrimRho.into(), rho.node())];
    for k in 0..ncomp {
        let v = gv_load_at(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8), &src) * gv_ghost_sign(k, ndim, &vel_sign);
        writes.push((format!("prim_v{k}"), FieldRef::PrimVel(k as u8).into(), v.node()));
    }
    let pre = gv_load_at("prim_pre", "prim.pre", &src);
    writes.push(("prim_pre".to_string(), FieldRef::PrimPre.into(), pre.node()));
    for k in 0..ncomp {
        let b = gv_load_at(&format!("bcell_{k}"), &format!("mhd.bcell[{k}]"), &src) * gv_ghost_sign(k, ndim, &vel_sign);
        writes.push((format!("bcell_{k}"), format!("mhd.bcell[{k}]").into(), b.node()));
    }
    (end_trace(), writes)
}


/// the ISOTHERMAL lattice-map ghost fill — `rmhd_ghost_fill_gv` minus the `pre` field
/// (isothermal MHD has no pressure to fill). rho + vel + bcell only.
pub fn imhd_ghost_fill_gv(ndim: usize, ncomp: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let src = gv_lattice_source(ndim);
    let vel_sign: Vec<Gv> = (0..ndim).map(|k| Gv::scalar(&format!("vel_sign_{k}"))).collect();
    let rho = gv_load_at("prim_rho", "prim.rho", &src);
    let mut writes = vec![("prim_rho".to_string(), FieldRef::PrimRho.into(), rho.node())];
    for k in 0..ncomp {
        let v = gv_load_at(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8), &src) * gv_ghost_sign(k, ndim, &vel_sign);
        writes.push((format!("prim_v{k}"), FieldRef::PrimVel(k as u8).into(), v.node()));
    }
    for k in 0..ncomp {
        let b = gv_load_at(&format!("bcell_{k}"), &format!("mhd.bcell[{k}]"), &src) * gv_ghost_sign(k, ndim, &vel_sign);
        writes.push((format!("bcell_{k}"), format!("mhd.bcell[{k}]").into(), b.node()));
    }
    (end_trace(), writes)
}
