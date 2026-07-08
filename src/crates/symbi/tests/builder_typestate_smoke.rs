// =============================================================================
// builder_typestate_smoke.rs
//
// the safe-path-only frontend: the typestate SimBuilder reaches a
// usable SimState only through the NeedsGrid -> NeedsCells -> Ready path, with
// `build()` callable solely at Ready. these smokes prove the happy paths
// compile and produce a seeded sim:
// - a hydro regime: allocate -> set_initial -> build (set_initial lands at Ready)
// - an MHD regime: allocate -> set_initial -> seed_faces -> build (faces owed,
//   so set_initial lands at NeedsCells; seed_faces reaches Ready + sets the
//   bface_initialized flag)
// also pins the new fallible allocate() validation surface.
// =============================================================================

use symbi::prelude::*;
use symbi::sim::state::Boundaries;

type Hydro = SimCpu<Newtonian, 2, Cartesian, IdealGas<f64>>;
type Mhd = SimCpuGeneric<Rmhd, 2, 3, Cartesian, IdealGas<f64>>;

// hydro: set_initial reaches Ready in one call (no staggered faces owed), then build().
#[test]
fn hydro_builder_reaches_ready_and_builds() {
    let sim = Hydro::build(Newtonian, IdealGas { gamma: 1.4 }, Cartesian)
        .cells([16, 16])
        .bounds([0.0, 0.0], [1.0, 1.0])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .allocate()
        .expect("valid grid config allocates")
        .set_initial(|_x| Prim { rho: 1.0, vel: Tensor::new([0.0, 0.0]), pre: 1.0 })
        .build();

    assert_eq!(sim.geom.interior.volume(), 16 * 16);
    // seeded: every interior cell recovers the prim it was set to.
    for c in sim.geom.interior.iter() {
        let p = sim.prim_at(c);
        assert!((p.rho - 1.0).abs() < 1e-14, "cell {c:?}: rho={}", p.rho);
        assert!((p.pre - 1.0).abs() < 1e-14, "cell {c:?}: p={}", p.pre);
    }
}

// mhd: set_initial lands at NeedsCells (faces owed); seed_faces reaches Ready and arms the CT.
#[test]
fn mhd_builder_requires_faces_then_builds() {
    let sim = Mhd::build(Rmhd, IdealGas { gamma: 5.0 / 3.0 }, Cartesian)
        .cells([16, 16])
        .bounds([0.0, 0.0], [1.0, 1.0])
        .boundaries(BoundaryType::Periodic)
        .allocate()
        .expect("valid grid config allocates")
        .set_initial(|_x| MhdPrim {
            hydro: Prim { rho: 1.0, vel: Tensor::new([0.1, 0.0, 0.0]), pre: 1.0 },
            mag: Tensor::new([0.2, 0.2, 0.0]),
        })
        .seed_faces_uniform([0.2, 0.2])
        .build();

    assert_eq!(sim.geom.interior.volume(), 16 * 16);
    // the staggered faces are seeded -> the CT guard would pass (bface_initialized set).
    let mhd = sim.fields.mhd.as_ref().expect("rmhd allocates mhd fields");
    assert!(
        mhd.bface_initialized.load(std::sync::atomic::Ordering::Relaxed),
        "seed_faces must arm bface_initialized for the CT ground truth"
    );
    for c in sim.geom.interior.iter() {
        let p = sim.prim_at(c);
        assert!(p.rho.is_finite() && p.rho > 0.0, "cell {c:?}: rho={}", p.rho);
    }
}

// the fallible config surface: a zero cell count is rejected BEFORE allocation.
#[test]
fn allocate_rejects_nonpositive_cells() {
    let r = Hydro::build(Newtonian, IdealGas { gamma: 1.4 }, Cartesian)
        .cells([0, 16])
        .bounds([0.0, 0.0], [1.0, 1.0])
        .allocate();
    assert!(r.is_err(), "zero cell count must be a config error");
}

// missing spacing (no .spacing / .bounds) is rejected.
#[test]
fn allocate_rejects_missing_spacing() {
    let r = Hydro::build(Newtonian, IdealGas { gamma: 1.4 }, Cartesian)
        .cells([16, 16])
        .allocate();
    assert!(r.is_err(), "missing spacing/bounds must be a config error");
}
