// =============================================================================
// penalize_shaped_jit.rs
//
// the make-or-break gate for the runtime-JIT arbitrary-shape rigid wall: the
// setup-built shaped penalization kernel MUST cranelift-compile on host. its
// surface normal is the SDF gradient (Dual-derived CSG min/max branches), so
// this pins that the shaped kernel's op set lies inside the JIT subset — else
// the moving arbitrary-shape body would have no host execution path.
// =============================================================================
use symbi_discretize::coords::Coords;
use symbi_discretize::{
    penalize_porous_gv_shaped, penalize_porous_gv_spinning, penalize_porous_iso_gv_shaped,
    penalize_porous_iso_gv_spinning,
};
use symbi_ib::sdf::SdfExpr;

#[test]
fn shaped_porous_penalize_jit_compiles_on_host() {
    // a box unioned with an offset sphere — a genuine CSG with min/max kinks and a
    // Dual-autodiff normal, in the body-local frame.
    let shape = SdfExpr::<f64, 3>::cuboid([0.0, 0.0, 0.0], [0.5, 0.3, 0.2])
        .union(SdfExpr::sphere([0.6, 0.0, 0.0], 0.25));
    let (kernel, writes) = penalize_porous_gv_shaped(Coords::Cartesian, 3, &shape);
    let compiled = symbi_jit::compile_gv_kernel(&kernel, &writes, 3);
    assert!(
        compiled.is_ok(),
        "the shaped penalize kernel is outside the cranelift JIT subset: {:?}",
        compiled.err(),
    );
}

#[test]
fn shaped_porous_penalize_jit_compiles_2d() {
    // a 2d run (r-z / x-y): the kernel is 3D internally but the active dims are 2.
    let shape = SdfExpr::<f64, 3>::cuboid([0.0, 0.0, 0.0], [0.4, 0.6, 1.0]);
    let (kernel, writes) = penalize_porous_gv_shaped(Coords::Cartesian, 2, &shape);
    assert!(symbi_jit::compile_gv_kernel(&kernel, &writes, 2).is_ok());
}

#[test]
fn shaped_rotated_penalize_jit_compiles() {
    // a ROTATED box (static tilt): the Rotated node is affine (multiply/add), so it must lie in
    // the JIT subset — the same runtime kernel bakes the orientation matrix as constants.
    let tilt = [[0.7071, -0.7071, 0.0], [0.7071, 0.7071, 0.0], [0.0, 0.0, 1.0]];
    let shape = SdfExpr::<f64, 3>::cuboid([0.0, 0.0, 0.0], [0.5, 0.2, 0.3]).rotated(tilt);
    let (kernel, writes) = penalize_porous_gv_shaped(Coords::Cartesian, 2, &shape);
    assert!(
        symbi_jit::compile_gv_kernel(&kernel, &writes, 2).is_ok(),
        "the rotated shaped penalize kernel is outside the JIT subset",
    );
}

#[test]
fn shaped_penalize_jit_compiles_curvilinear() {
    // the mask distance is physical: on a spherical / cylindrical grid the kernel maps the
    // coordinate centroid to Cartesian first. that path (centroid_to_cartesian +
    // vector_from_cartesian) must lie in the JIT subset for a shaped body.
    let shape = SdfExpr::<f64, 3>::cuboid([1.5, 0.0, 0.0], [0.3, 0.3, 0.3]);
    let (k, w) = penalize_porous_gv_shaped(Coords::Spherical, 2, &shape);
    assert!(symbi_jit::compile_gv_kernel(&k, &w, 2).is_ok(), "spherical shaped kernel not JIT-able");
    let (k, w) = penalize_porous_gv_shaped(Coords::Cylindrical, 2, &shape);
    assert!(symbi_jit::compile_gv_kernel(&k, &w, 2).is_ok(), "cylindrical shaped kernel not JIT-able");
    let (k, w) = penalize_porous_iso_gv_shaped(Coords::Cylindrical, 3, &shape);
    assert!(symbi_jit::compile_gv_kernel(&k, &w, 3).is_ok(), "cylindrical iso shaped kernel not JIT-able");
}

#[test]
fn spinning_penalize_jit_compiles() {
    // the SPINNING wall: the mask is rotated by R(body_0_angle) built from Gv cos/sin (runtime
    // angle), and the surface velocity carries omega x r. both the adiabatic and iso kernels must
    // JIT — cos/sin are in the cranelift subset.
    let shape = SdfExpr::<f64, 3>::cuboid([0.0, 0.0, 0.0], [0.5, 0.2, 0.3]);
    let (k, w) = penalize_porous_gv_spinning(Coords::Cartesian, 2, &shape);
    assert!(symbi_jit::compile_gv_kernel(&k, &w, 2).is_ok(), "adiabatic spinning kernel not JIT-able");
    let (k, w) = penalize_porous_iso_gv_spinning(Coords::Cartesian, 2, &shape);
    assert!(symbi_jit::compile_gv_kernel(&k, &w, 2).is_ok(), "iso spinning kernel not JIT-able");
}

#[test]
fn shaped_iso_porous_penalize_jit_compiles() {
    // the energy-free shaped wall (iso obstacle flows): same CSG normal, no nrg channel.
    let shape = SdfExpr::<f64, 3>::cuboid([0.0, 0.0, 0.0], [0.5, 0.3, 0.2])
        .union(SdfExpr::sphere([0.6, 0.0, 0.0], 0.25));
    let (kernel, writes) = penalize_porous_iso_gv_shaped(Coords::Cartesian, 2, &shape);
    assert!(
        symbi_jit::compile_gv_kernel(&kernel, &writes, 2).is_ok(),
        "the iso shaped penalize kernel is outside the JIT subset",
    );
}
