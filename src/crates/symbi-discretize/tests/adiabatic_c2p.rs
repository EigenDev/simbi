// =============================================================================
// adiabatic_c2p.rs
//
// the adiabatic (ideal-gas) c2p is the gv single-source physics (symbi-hydro's
// `Cons::to_primitive` at S=Gv). it is ncomp-generic: a single `adiabatic_c2p_gv::<NCOMP>`
// instance for NCOMP in {1,2,3}, the kinetic energy `v_sq = sum_k vel_k^2` folding all
// NCOMP velocity components (NCOMP=3 covers the cyl r-z swirl on a 2D grid). the
// bit-identical-vs-production run is in the symbi crate (substrate_adiabatic_*).
// =============================================================================

use symbi_discretize::adiabatic_c2p_gv;

fn n_components<const NCOMP: usize>() -> usize {
    let (k, writes) = adiabatic_c2p_gv::<NCOMP>();
    assert!(
        !k.graph.has_errors(),
        "adiabatic_c2p graph errors: {:?}",
        k.graph.errors()
    );
    writes
        .iter()
        .filter(|write| write.destination.name().starts_with("prim.vel["))
        .count()
}

#[test]
fn adiabatic_c2p_is_ncomp_generic() {
    // the same gv builder yields NCOMP velocity components, free of any 1D or ndim lock;
    // codegen instantiates it per velocity-component count (NCOMP=3 on a 2D grid is the
    // cyl r-z swirl).
    assert_eq!(n_components::<1>(), 1);
    assert_eq!(n_components::<2>(), 2);
    assert_eq!(n_components::<3>(), 3);
}
