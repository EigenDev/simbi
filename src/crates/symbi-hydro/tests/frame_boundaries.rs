// =============================================================================
// frame_boundaries.rs
//
// structural gate: the migrated frame boundaries keep their witnesses. the
// compile-fail doctests on `Indexed`, `Normalized`, `Regime::Normal`, and
// `Valencia` prove the illegal operations are rejected; the scans here pin the
// signatures themselves, so a bare tensor cannot quietly return at a boundary
// the witness migration typed.
// =============================================================================

use std::path::Path;

fn source(relative: &str) -> String {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(root.join(relative)).expect("gate source must exist")
}

/// the regime flux and wave-speed doors consume the regime-associated normal
/// witness; a bare tensor normal stays out of the trait surface. signatures
/// are matched with all whitespace stripped, so the pin holds the types in
/// the signature and is indifferent to line wrapping.
#[test]
fn regime_normal_doors_take_the_witness() {
    let regime = source("src/regime.rs");
    let stripped: String = regime.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        regime.contains("type Normal: FaceNormal<S, D>;"),
        "the Regime trait must declare its face-normal witness"
    );
    for door in [
        "fnto_flux(&self,prim:&Self::Prim,nhat:&Self::Normal",
        "fnwave_speeds(&self,eos:&implEosFor<S,Self::Energy>,prim:&Self::Prim,nhat:&Self::Normal",
    ] {
        assert!(stripped.contains(door), "witness door lost: {door}");
    }
    assert!(
        !stripped.contains("nhat:&Tensor"),
        "a bare tensor normal returned to the Regime trait"
    );
}

/// the valencia regimes state both associated frames in their types: the
/// primitive's velocity is the coordinate contravariant v^i and the conserved
/// momentum the covariant densitized S_i, witnessed by the wrapper at every
/// conversion, recovery, and flux boundary.
#[test]
fn valencia_regimes_witness_both_associated_states() {
    for (file, prim, cons) in [
        (
            "src/rhd/gr.rs",
            "type Prim = Valencia<Prim<S, D>>;",
            "type Cons = Valencia<Cons<S, D>>;",
        ),
        (
            "src/rmhd/gr.rs",
            "type Prim = Valencia<MhdPrim<S, D>>;",
            "type Cons = Valencia<MhdCons<S, D>>;",
        ),
    ] {
        let text = source(file);
        assert!(
            text.contains(prim),
            "{file} must declare its primitive through the Valencia witness"
        );
        assert!(
            text.contains(cons),
            "{file} must declare its conserved state through the Valencia witness"
        );
    }
}

/// each regime family declares the frame its normal is lawful in: the
/// locally-flat solvers the orthonormal witness, the valencia regimes the
/// coordinate-covariant one.
#[test]
fn regimes_declare_their_lawful_normal_frame() {
    for file in [
        "src/newtonian.rs",
        "src/isothermal.rs",
        "src/rhd.rs",
        "src/rmhd.rs",
        "src/newtonian_mhd.rs",
        "src/isothermal_mhd.rs",
    ] {
        assert!(
            source(file).contains("type Normal = Normalized<Physical<S, D>>;"),
            "{file}: flat solver must use the physical-frame witness"
        );
    }
    for file in ["src/rhd/gr.rs", "src/rmhd/gr.rs"] {
        assert!(
            source(file).contains("type Normal = Normalized<Covariant<S, D>>;"),
            "{file}: valencia regime must use the coordinate-covariant witness"
        );
    }
}
