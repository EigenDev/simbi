// =============================================================================
// ghost_fill.rs
//
// the lattice-map PULLBACK scheme (docs/design/11) emits correct integer code.
// `iso_ghost_fill` reads the primitives at a per-axis lattice-map source coord
// (periodic / reflect / outflow, branched on a runtime integer `map_type`) and
// writes them at the cell, with the per-component `vel_sign` (the map's Jacobian).
//
// these are SOURCE-shape checks: the source coord is pure-integer (the select
// branches render as `ii + arg_0`, `arg_0 - ii`, no float, no cast); the params
// are typed integer; no dead base cell read is emitted (the field is read only at
// the source). the bit-identical run vs production lives in the AOT crate (the
// path production actually compiles).
// =============================================================================

mod harness;
use harness::KernelRun;

use symbi_discretize::iso_ghost_fill_gv;

fn emit(ndim: u8) -> (String, Vec<String>) {
    let nd = ndim as usize;
    let axes: Vec<usize> = (0..nd).collect();
    let e = KernelRun::new(iso_ghost_fill_gv(nd, nd, &axes)).grid(vec![1usize; nd]).emit_cpu();
    (e.source, e.scalar_params)
}

#[test]
fn ghost_fill_1d_emits_integer_source_coord_and_vel_sign() {
    let (src, scalar_params) = emit(1);

    // params, in first-use order: the lattice-map encoding then the grade sign.
    assert_eq!(scalar_params, vec!["map_type_0", "arg_0", "vel_sign_0"]);
    // the lattice-map source-coord args are INTEGER; vel_sign is a float, typed `S`
    // (the kernels are precision-generic `fn k<S: Scalar>(..)`).
    assert!(src.contains("map_type_0: i32"), "map_type must be i32:\n{src}");
    assert!(src.contains("arg_0: i32"), "arg must be i32:\n{src}");
    assert!(src.contains("vel_sign_0: S"), "vel_sign must be the scalar S:\n{src}");

    // the source coord branches on map_type in PURE INTEGER space: periodic
    // `_coord_0 + arg_0`, reflect `arg_0 - _coord_0`, identity `_coord_0`,
    // outflow `arg_0`. shared across rho + every vel read, it is CSE'd — and the
    // CSE'd local stays `i32`, so the buffer index built from it is integer.
    assert!(src.contains(": i32 = (if (map_type_0 == 0_i32)"),
        "the shared source coord must be a CSE'd i32 select:\n{src}");
    assert!(src.contains("(_coord_0 + arg_0)"), "periodic source coord missing:\n{src}");
    assert!(src.contains("(arg_0 - _coord_0)"), "reflect source coord missing:\n{src}");
    assert!(src.contains("map_type_0 == 1_i32"), "periodic branch test missing:\n{src}");
    assert!(!src.contains("as f64"), "an index must never route through f64:\n{src}");
    assert!(!src.contains("as i64"), "indices are i32, no i64:\n{src}");

    // density is a straight pullback; velocity multiplies by vel_sign_0.
    assert!(src.contains("* vel_sign_0)") || src.contains("vel_sign_0)"),
        "velocity must pick up vel_sign_0:\n{src}");

    // NO dead base cell read: prim is read only at the source coord, so the
    // emitter must NOT emit `let prim_rho: f64 = buf0[(ii - buf_lo_0_0) ...]`.
    assert!(!src.contains("let prim_rho: f64 = buf0[(ii - buf_lo_0_0) as usize];"),
        "a dead base cell read leaked in:\n{src}");
}

#[test]
fn ghost_fill_scheme_is_dimension_generic() {
    // the SAME builder produces a 2D fill: per-axis map params (map_type_0/1,
    // arg_0/1) and one vel_sign per momentum component. no per-dimension variant.
    let (src, scalar_params) = emit(2);
    assert_eq!(
        scalar_params,
        vec!["map_type_0", "map_type_1", "arg_0", "arg_1", "vel_sign_0", "vel_sign_1"],
    );
    assert!(src.contains("map_type_1: i32") && src.contains("arg_1: i32"));
    assert!(src.contains("vel_sign_1: S"));
    // two velocity components, each with its own sign.
    assert!(src.contains("vel_sign_0") && src.contains("vel_sign_1"));
}
