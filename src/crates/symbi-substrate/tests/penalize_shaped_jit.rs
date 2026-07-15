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
use symbi_discretize::penalize_porous_gv_shaped;
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
