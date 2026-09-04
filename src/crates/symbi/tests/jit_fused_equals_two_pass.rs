// =============================================================================
// jit_fused_equals_two_pass.rs
//
// the evolve equivalence check for the fused runtime-source host path. it is the
// runtime-user-source twin of
// `additive_source_equals_fused_evolve` (which proves it for AOT-baked sources):
// the same runtime-loaded user source (python -> json -> build_user_source DAG) is
// run two ways through the production `evolve()` loop —
//   - two-pass: `with_runtime_source(..)`        -> plain AOT godunov + the per-cell
//                                                   `apply_runtime_source` pass
//   - fused:    `with_fused_runtime_source(..)`  -> one Cranelift-JIT'd godunov+source
//                                                   host kernel (run_parallel_raw)
// and must produce bit-for-bit identical conserved state at every interior cell.
//
// it stresses the FP-sensitive seams: SSP-RK2 (a0=ac=0.5 corrector every step), a
// non-uniform density (nonzero flux divergence — a uniform state would hide an
// ordering bug behind a zero divergence) and a nonzero velocity (so the energy-side
// `v . a` source term is live from step one). a single-ULP drift in the godunov
// arithmetic, the snapshot, the `ac*dt` weight, or the source eval makes a
// trajectory diverge and `assert_eq!` on the bits fails.
//
// run: cargo test -p symbi --test jit_fused_equals_two_pass
// =============================================================================

use symbi::prelude::*;
use symbi_algebra::Domain;
use symbi_grid::Field;
use symbi_hydro::energy::IsoModel;
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::state::PrimG;
use symbi_hydro::{ISO_NEWTONIAN_SPEC, NEWTONIAN_SPEC};
use symbi_source_compile::SourceConfig;
use symbi_source_compile::expr_bridge::build_user_source;
use symbi_xpu::HostMemory;

// assert two conserved fields are bit-for-bit equal over the interior.
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
            "{label} differs at {c:?}: fused={va:?} two_pass={vb:?} (delta={:?})",
            va - vb,
        );
    }
}

#[test]
fn adiabatic_runtime_force_fused_equals_two_pass_rk2() {
    // the two-pass is the default; this test pins the fused kernel as live, so
    // opt in before the policy OnceLock latches.
    unsafe { std::env::set_var("SYMBI_FUSE", "1") };
    type Sim = SimCpu<Newtonian, 2, Cartesian, IdealGas<f64>>;
    const GAMMA: f64 = 1.4;
    let n = 24usize;
    let t_final = 0.04;

    // the runtime user source: external acceleration a = [p0, p1] (force kind). for an
    // energy regime build_user_source wraps it into both the momentum overlay (S_mom = rho*a)
    // and the energy overlay (S_nrg = rho*v.a) — so this exercises the in-place mom_k and nrg
    // writes of the fused godunov.
    let json = r#"{
        "kind": "force", "dim": 2, "outputs": [0, 1], "params": [0.5, -0.3],
        "vocabulary":{"reads":[],"params":[0,1]},
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
        // non-uniform density (nonzero divergence) + nonzero velocity (live v.a energy term).
        sim.seed_cells(|p| {
            let (x, y) = (p[0], p[1]);
            let rho =
                1.0 + 0.2 * (std::f64::consts::TAU * x).sin() * (std::f64::consts::TAU * y).cos();
            Prim::adiabatic(Density(rho), Tensor::new([0.1, -0.05]), Pressure(1.0))
        });
        sim
    };

    // two-pass: plain AOT godunov + the per-cell apply_runtime_source pass.
    let mut sim_two = build();
    let sub_two = sim_two.substrate().with_runtime_source(
        build_user_source(&cfg, &NEWTONIAN_SPEC).unwrap(),
        cfg.params.clone(),
    );
    evolve(&mut sim_two, &sub_two, t_final).expect("two-pass evolve");

    // fused: one Cranelift-JIT'd godunov+source launch.
    let mut sim_fused = build();
    let sub_fused = sim_fused.substrate().with_fused_runtime_source(
        build_user_source(&cfg, &NEWTONIAN_SPEC).unwrap(),
        cfg.params.clone(),
    );
    evolve(&mut sim_fused, &sub_fused, t_final).expect("fused evolve");

    // guard: the fused kernel actually JIT-compiled + ran (else this would compare two-pass vs
    // two-pass and pass vacuously — the exact trap this test exists to avoid).
    assert_eq!(
        sub_fused.runtime_source.as_ref().unwrap().fused_cpu_state(),
        Some(true),
        "fused godunov+source kernel did not compile — fused path silently fell back to two-pass",
    );

    // same source, two execution strategies -> identical trajectory, bit-for-bit.
    let interior = &sim_fused.geom.interior;
    assert_cons_bit_identical(
        interior,
        &sim_fused.fields.cons.den,
        &sim_two.fields.cons.den,
        "cons.den",
    );
    for k in 0..2 {
        assert_cons_bit_identical(
            interior,
            &sim_fused.fields.cons.mom[k],
            &sim_two.fields.cons.mom[k],
            "cons.mom",
        );
    }
    let (nf, nt) = (
        sim_fused.fields.cons.nrg_field().unwrap(),
        sim_two.fields.cons.nrg_field().unwrap(),
    );
    assert_cons_bit_identical(interior, nf, nt, "cons.nrg");

    // sanity: the run actually moved (else the test exercised nothing).
    let moved = sim_fused
        .geom
        .interior
        .iter()
        .any(|c| (*sim_fused.fields.prim.vel[0].view().at(c) - 0.1).abs() > 1e-9);
    assert!(moved, "gas never accelerated — the test exercised nothing");
}

#[test]
fn iso_runtime_force_fused_equals_two_pass_rk2() {
    // the two-pass is the default; this test pins the fused kernel as live, so
    // opt in before the policy OnceLock latches.
    unsafe { std::env::set_var("SYMBI_FUSE", "1") };
    // the iso analogue: no energy law, so the fused kernel writes only den + mom_k (no nrg).
    // proves the has_energy=false fused path matches the iso two-pass bit-for-bit.
    type Sim = SimCpu<IsoNewtonian, 2, Cartesian, Isothermal<f64>>;
    let n = 24usize;
    let cs = 0.05;
    let t_final = 0.04;

    let json = r#"{
        "kind": "force", "dim": 2, "outputs": [0, 1], "params": [0.4, -0.25],
        "vocabulary":{"reads":[],"params":[0,1]},
        "nodes": [ {"op": "PARAMETER", "param_idx": 0}, {"op": "PARAMETER", "param_idx": 1} ]
    }"#;
    let cfg = SourceConfig::from_json(json).expect("parse config");

    let build = || -> Sim {
        let sim = Sim::build(IsoNewtonian, Isothermal { cs }, Cartesian)
            .cells([n, n])
            .bounds([0.0, 0.0], [1.0, 1.0])
            .boundaries(BoundaryType::Periodic)
            .finish()
            .unwrap();
        sim.seed_cells(|p| {
            let (x, y) = (p[0], p[1]);
            let rho =
                1.0 + 0.2 * (std::f64::consts::TAU * x).sin() * (std::f64::consts::TAU * y).cos();
            PrimG::<f64, 2, IsoModel>::isothermal(Density(rho), Tensor::new([0.1, -0.05]))
        });
        sim
    };

    let mut sim_two = build();
    let sub_two = sim_two.substrate().with_runtime_source(
        build_user_source(&cfg, &ISO_NEWTONIAN_SPEC).unwrap(),
        cfg.params.clone(),
    );
    evolve(&mut sim_two, &sub_two, t_final).expect("iso two-pass evolve");

    let mut sim_fused = build();
    let sub_fused = sim_fused.substrate().with_fused_runtime_source(
        build_user_source(&cfg, &ISO_NEWTONIAN_SPEC).unwrap(),
        cfg.params.clone(),
    );
    evolve(&mut sim_fused, &sub_fused, t_final).expect("iso fused evolve");

    assert_eq!(
        sub_fused.runtime_source.as_ref().unwrap().fused_cpu_state(),
        Some(true),
        "iso fused godunov+source kernel did not compile — fell back to two-pass",
    );

    let interior = &sim_fused.geom.interior;
    assert_cons_bit_identical(
        interior,
        &sim_fused.fields.cons.den,
        &sim_two.fields.cons.den,
        "cons.den",
    );
    for k in 0..2 {
        assert_cons_bit_identical(
            interior,
            &sim_fused.fields.cons.mom[k],
            &sim_two.fields.cons.mom[k],
            "cons.mom",
        );
    }
}

#[test]
fn adiabatic_fused_equals_two_pass_on_the_cache_tiled_cover() {
    // the two-pass is the default; this test pins the fused kernel as live, so
    // opt in before the policy OnceLock latches.
    unsafe { std::env::set_var("SYMBI_FUSE", "1") };
    // the other force tests run 24^2 = 576 interior cells, below `WHOLE_BELOW_CELLS` — so their
    // fused dispatch takes `ExecPolicy::Whole` (the flat driver) and the cache-tiled cover is never
    // exercised. this runs a domain large enough that `policy_for` returns `Cover`, so the fused
    // godunov executes through `run_cover_raw` (blocks fanned out, serial axis-0-innermost within).
    // the cover must be a pure reordering: bit-for-bit equal to the two-pass trajectory.
    use symbi_exec::policy::{ExecPolicy, policy_for};

    type Sim = SimCpu<Newtonian, 2, Cartesian, IdealGas<f64>>;
    const GAMMA: f64 = 1.4;
    let n = 192usize; // 192^2 = 36864 >= WHOLE_BELOW_CELLS (32768)
    let t_final = 0.001;

    let json = r#"{
        "kind": "force", "dim": 2, "outputs": [0, 1], "params": [0.5, -0.3],
        "vocabulary":{"reads":[],"params":[0,1]},
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
        sim.seed_cells(|p| {
            let (x, y) = (p[0], p[1]);
            let rho =
                1.0 + 0.2 * (std::f64::consts::TAU * x).sin() * (std::f64::consts::TAU * y).cos();
            Prim::adiabatic(Density(rho), Tensor::new([0.1, -0.05]), Pressure(1.0))
        });
        sim
    };

    let mut sim_two = build();
    let sub_two = sim_two.substrate().with_runtime_source(
        build_user_source(&cfg, &NEWTONIAN_SPEC).unwrap(),
        cfg.params.clone(),
    );
    evolve(&mut sim_two, &sub_two, t_final).expect("two-pass evolve");

    let mut sim_fused = build();
    let sub_fused = sim_fused.substrate().with_fused_runtime_source(
        build_user_source(&cfg, &NEWTONIAN_SPEC).unwrap(),
        cfg.params.clone(),
    );
    evolve(&mut sim_fused, &sub_fused, t_final).expect("fused evolve");

    // guard 1: the fused kernel compiled (else two-pass vs two-pass).
    assert_eq!(
        sub_fused.runtime_source.as_ref().unwrap().fused_cpu_state(),
        Some(true),
        "fused godunov+source kernel did not compile",
    );
    // guard 2: the domain really does select the tiled cover (else this duplicates the Whole-path
    // oracles and proves nothing about `run_cover_raw`).
    assert!(
        matches!(
            policy_for(&sim_fused.geom.interior, false),
            ExecPolicy::Cover(_)
        ),
        "domain did not select ExecPolicy::Cover — the cache-tiled fused path was not exercised",
    );

    let interior = &sim_fused.geom.interior;
    assert_cons_bit_identical(
        interior,
        &sim_fused.fields.cons.den,
        &sim_two.fields.cons.den,
        "cons.den",
    );
    for k in 0..2 {
        assert_cons_bit_identical(
            interior,
            &sim_fused.fields.cons.mom[k],
            &sim_two.fields.cons.mom[k],
            "cons.mom",
        );
    }
    let (nf, nt) = (
        sim_fused.fields.cons.nrg_field().unwrap(),
        sim_two.fields.cons.nrg_field().unwrap(),
    );
    assert_cons_bit_identical(interior, nf, nt, "cons.nrg");
}
