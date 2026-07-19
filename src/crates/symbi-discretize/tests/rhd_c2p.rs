// =============================================================================
// rhd_c2p.rs
//
// the RHD iterative cons->prim scheme LOWERS + EMITS valid CPU code. this is a
// BUILD + EMIT test only — it does NOT run the kernel, because the masked Newton
// unroll is exponential in the #4 interpreter (see feedback). the numerical
// validation (vs rhd_to_primitive) runs through the COMPILED path in the AOT
// crate, where the unroll CSE-collapses to linear.
//
// what this checks: the scheme builds without graph errors; the emitted Rust has
// the conserved reads, the gamma param, the Newton unroll (nested if/else from
// branch), sqrt (Lorentz + the body), and the prim writes.
// =============================================================================

mod harness;
use harness::KernelRun;

use symbi_discretize::{rhd_c2p_gv, rhd_flux_gv};
use symbi_ir::graph::NodeId;

// emit the gv-built RHD c2p (`rhd_c2p_gv` = symbi-hydro's `rhd_recover` at S=Gv) as
// CPU source. the const-generic <D> instance is selected by ndim (build.rs does the same).
fn emit(ndim: u8, max_iters: usize) -> (String, Vec<(String, String)>, Vec<String>) {
    let built = match ndim {
        1 => rhd_c2p_gv::<1>(max_iters),
        2 => rhd_c2p_gv::<2>(max_iters),
        3 => rhd_c2p_gv::<3>(max_iters),
        _ => panic!("unsupported ndim {ndim}"),
    };
    let e = KernelRun::new(built)
        .grid(vec![1usize; ndim as usize])
        .emit_cpu();
    (e.source, e.field_inputs, e.scalar_params)
}

#[test]
fn rhd_c2p_1d_lowers_and_emits() {
    let (src, field_inputs, scalar_params) = emit(1, 8);

    // conserved inputs + gamma.
    assert!(
        field_inputs.iter().any(|(k, _)| k == "cons_den"),
        "missing cons_den"
    );
    assert!(
        field_inputs.iter().any(|(k, _)| k == "cons_mom_0"),
        "missing cons_mom_0"
    );
    assert!(
        field_inputs.iter().any(|(k, _)| k == "cons_nrg"),
        "missing cons_nrg"
    );
    assert_eq!(
        scalar_params,
        vec!["gamma"],
        "the RHD c2p takes one scalar: gamma"
    );
    assert!(
        src.contains("gamma: S"),
        "gamma must be a scalar param (precision-generic S):\n{src}"
    );

    // the Newton emits as ONE loop (body emitted once), not an 8x
    // unroll — identified by its literal `0..8` bound. the cache-tiled default CPU
    // emit ALSO emits per-axis cell loops (`for _d in 0.._ts`) for the outer
    // iteration, so count the Newton specifically rather than all `in 0..` loops.
    let n_newton = src.matches(" in 0..8").count();
    assert_eq!(
        n_newton, 1,
        "expected exactly ONE Newton loop (0..8, no 8x unroll), got {n_newton}:\n{src}"
    );

    // sqrt appears (Lorentz factor in the body + recovery).
    assert!(src.contains(".sqrt()"), "missing sqrt:\n{src}");
    // the three prim writes are present.
    assert!(
        src.contains("] = ") && src.contains("buf"),
        "missing writes:\n{src}"
    );
    // pure-integer indices (pointwise: no stencil, but the loop index is integer).
    assert!(!src.contains("as f64"), "no float-routed index:\n{src}");
}

#[test]
fn rhd_c2p_is_dimension_generic() {
    // the SAME gv builder yields D velocity components — 1D: 1, 2D: 2, 3D: 3.
    let n_vel = |writes: &[(String, symbi_ir::FieldBind, NodeId)]| {
        writes
            .iter()
            .filter(|(_, rt, _)| rt.name().starts_with("prim.vel["))
            .count()
    };
    let (k1, w1) = rhd_c2p_gv::<1>(4);
    let (k2, w2) = rhd_c2p_gv::<2>(4);
    let (k3, w3) = rhd_c2p_gv::<3>(4);
    for (k, w, want) in [(&k1, &w1, 1usize), (&k2, &w2, 2), (&k3, &w3, 3)] {
        assert!(
            !k.graph.has_errors(),
            "rhd_c2p errors: {:?}",
            k.graph.errors()
        );
        assert_eq!(n_vel(w), want, "expected {want} velocity components");
    }
}

#[test]
fn rhd_face_flux_1d_lowers_and_emits() {
    // the gv RHD flux (PLM stencil + riemann::hlle at the Rhd regime) emits valid CPU
    // code: the conserved reads + gamma, the sqrt (Lorentz / relativistic wave speeds),
    // the stencil reads/writes, integer-only indices.
    let e = KernelRun::new(rhd_flux_gv::<1>(0))
        .grid([1usize])
        .emit_cpu();
    let src = e.source;

    assert!(
        e.field_inputs.iter().any(|(key, _)| key == "prim_rho"),
        "missing prim_rho"
    );
    assert!(
        e.field_inputs.iter().any(|(key, _)| key == "prim_v0"),
        "missing prim_v0"
    );
    assert!(
        e.field_inputs.iter().any(|(key, _)| key == "prim_pre"),
        "missing prim_pre"
    );
    assert_eq!(
        e.scalar_params,
        vec![
            "gamma",
            "theta",
            "mesh_adot_0",
            "x_lo_0",
            "dx_0",
            "mesh_vtrans_0"
        ],
        "the RHD flux takes gamma + the theta-MC limiter + the per-axis mesh-motion/geometry scalars",
    );
    assert!(
        src.contains(".sqrt()"),
        "missing sqrt (Lorentz / wave speeds):\n{src}"
    );
    // a face flux is a STENCIL — PLM reconstruction reads shifted neighbours.
    assert!(
        src.contains("] = ") && src.contains("buf"),
        "missing reads/writes:\n{src}"
    );
    // no float-routed BUFFER index. the only sanctioned `as f64` is the index-to-physical
    // coordinate bridge `S::from_f64((_coord_n) as f64)` (the moving-mesh geometry term reads the
    // cell's physical position x = x_lo + coord*dx). EVERY `as f64` in the emitted source must be
    // exactly that bridge cast — i.e., immediately preceded by a `(_coord_n)` token.
    for (off, _) in src.match_indices("as f64") {
        // the token immediately before the cast must be a `(_coord_<n>)` group.
        let before = src[..off].trim_end();
        let is_bridge = before.ends_with(')')
            && before
                .rfind("(_coord_")
                .map_or(false, |start| !before[start..].contains(' '));
        assert!(
            is_bridge,
            "float-routed index (an `as f64` not from the coord->physical bridge):\n{src}",
        );
    }
}
