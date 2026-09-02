// =============================================================================
// stage_input_alias_equals_snapshot.rs
//
// proves the stage-0 `u_stage` elision is a pure removal of redundant work.
//
// at the first stage of a multi-stage SSP scheme, `snapshot` has just copied `cons -> u_n` and
// nothing has touched cons since — so `u_n` already holds the stage input, and the separate
// `cons -> u_stage` copy moves a full-grid conserved set for no information. the driver therefore
// sets `stage_input_is_un` and skips `snapshot_stage`, and `FieldStore::stage_input()` binds `u_n`
// for that stage.
//
// the risk this gates: `snapshot` fills the allocated domain while `snapshot_stage` fills only the
// interior, so the two buffers differ in the ghost band. if any consumer of the stage input ever
// read a ghost cell, the alias would silently change physics. it must not — so the same initial
// state, evolved with the elision on and with it forced off (`elide_stage_snapshot = false`, the
// reference path), must produce a bit-for-bit identical trajectory.
//
// a runtime user source is attached because `source_apply` is the consumer that reads the stage
// input every substage (`apply_runtime_source` reads it per cell); without a source the elision is
// never exercised on the uni-grid driver.
//
// run: cargo test -p symbi --test stage_input_alias_equals_snapshot
// =============================================================================

use std::sync::atomic::Ordering;

use symbi::prelude::*;
use symbi_algebra::Domain;
use symbi_grid::Field;
use symbi_source_compile::expr_bridge::build_user_source;
use symbi_hydro::NEWTONIAN_SPEC;
use symbi_source_compile::SourceConfig;
use symbi_xpu::HostMemory;

fn assert_cons_bit_identical<const D: usize>(
    interior: &Domain<D>,
    a: &Field<f64, D, HostMemory>,
    b: &Field<f64, D, HostMemory>,
    label: &str,
) {
    for c in interior.iter() {
        let (va, vb) = (*a.view().at(c), *b.view().at(c));
        assert_eq!(
            va.to_bits(),
            vb.to_bits(),
            "{label} differs at {c:?}: elided={va:?} reference={vb:?} (delta={:?})",
            va - vb,
        );
    }
}

#[test]
fn stage0_un_alias_matches_the_snapshot_stage_copy() {
    type Sim = SimCpu<Newtonian, 2, Cartesian, IdealGas<f64>>;
    const GAMMA: f64 = 1.4;
    let n = 32usize;
    let t_final = 0.05;

    // an external acceleration; build_user_source wraps it into the momentum and energy overlays,
    // both of which evaluate S at the stage input — the buffer under test.
    let json = r#"{
        "kind": "force", "dim": 2, "outputs": [0, 1], "params": [0.7, -0.4],
        "nodes": [ {"op": "PARAMETER", "param_idx": 0}, {"op": "PARAMETER", "param_idx": 1} ]
    }"#;
    let cfg = SourceConfig::from_json(json).expect("parse config");

    let build = || -> Sim {
        let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
            .cells([n, n])
            .bounds([0.0, 0.0], [1.0, 1.0])
            .boundaries(BoundaryType::Periodic)
            .finish()
            .unwrap();
        // non-uniform density (live flux divergence) + nonzero velocity (live v.a energy term), so a
        // stale or wrong stage-input buffer diverges immediately, where a uniform state might let the error cancel.
        sim.seed_cells(|p| {
            let (x, y) = (p[0], p[1]);
            let rho =
                1.0 + 0.3 * (std::f64::consts::TAU * x).sin() * (std::f64::consts::TAU * y).cos();
            Prim {
                rho,
                vel: Tensor::new([0.15, -0.08]),
                pre: 1.0,
            }
        });
        sim
    };

    // reference: force the `cons -> u_stage` copy at every stage, eliding nothing.
    let mut sim_ref = build();
    sim_ref
        .workspace
        .elide_stage_snapshot
        .store(false, Ordering::Relaxed);
    let sub_ref = sim_ref.substrate().with_runtime_source(
        build_user_source(&cfg, &NEWTONIAN_SPEC).unwrap(),
        cfg.params.clone(),
    );
    evolve(&mut sim_ref, &sub_ref, t_final).expect("reference evolve");

    // elided: production default — stage 0 binds u_n and skips the copy.
    let mut sim_elide = build();
    assert!(
        sim_elide
            .workspace
            .elide_stage_snapshot
            .load(Ordering::Relaxed),
        "the elision must be the production default, else this oracle compares two reference runs",
    );
    let sub_elide = sim_elide.substrate().with_runtime_source(
        build_user_source(&cfg, &NEWTONIAN_SPEC).unwrap(),
        cfg.params.clone(),
    );
    evolve(&mut sim_elide, &sub_elide, t_final).expect("elided evolve");

    // guard: the alias really engaged. RK2 leaves the flag set from its last stage, which is the
    // corrector (ii = 1) -> false. so the assertion targets the invariant that makes the elision legal:
    // multi-stage, and the reference genuinely took the other branch.
    assert!(
        sim_elide.timestepping.stages().len() > 1,
        "the elision only applies to multi-stage schemes"
    );
    assert!(
        !sim_ref
            .workspace
            .elide_stage_snapshot
            .load(Ordering::Relaxed),
        "reference run must have the elision disabled",
    );

    let interior = &sim_elide.geom.interior;
    assert_cons_bit_identical(
        interior,
        &sim_elide.fields.cons.den,
        &sim_ref.fields.cons.den,
        "cons.den",
    );
    for k in 0..2 {
        assert_cons_bit_identical(
            interior,
            &sim_elide.fields.cons.mom[k],
            &sim_ref.fields.cons.mom[k],
            "cons.mom",
        );
    }
    let (ne, nr) = (
        sim_elide.fields.cons.nrg_field().unwrap(),
        sim_ref.fields.cons.nrg_field().unwrap(),
    );
    assert_cons_bit_identical(interior, ne, nr, "cons.nrg");

    // and the runs must be non-trivial: the source + divergence actually moved the state.
    let moved = interior.iter().any(|c| {
        (*sim_elide.fields.cons.mom[0].view().at(c)
            - 0.15 * *sim_elide.fields.cons.den.view().at(c))
        .abs()
            > 1e-9
    });
    assert!(
        moved,
        "state never evolved — the comparison would be vacuous"
    );
}
