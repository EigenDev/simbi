// =============================================================================
// source_term_carrier.rs
//
// the source-term carrier + user-expression gates. two layers:
//   1. built-in carrier-generic sources (`UniformAccel`, `PointMassGravity`) traced at S=Gv and
//      rendered/evaluated, matching their analytical form (`S_mom_k = rho*g_ext_k`, etc.) — the
//      f64 half lives in symbi-hydro (`source_term::tests`), so f64 == Gv == analytical from ONE
//      definition, no separate eval path, no graph-divergence bug class.
//   2. USER expressions bridged from a DAG (`expr_bridge`) + rendered through the SAME splice path
//      (`splice_user_source_gv`): a raw user field codegens correctly, and the AXIOMATIC wrappers
//      (`user_force_*` / `user_cooling_source`) enforce the work-energy coupling `S_nrg = rho*(a.v)`
//      BY CONSTRUCTION — a user cannot desync energy from the force.
// =============================================================================

mod harness;

use harness::KernelRun;
use symbi_discretize::{
    point_mass_gravity_probe_gv, splice_user_source_gv, uniform_accel_probe_gv,
};

#[test]
fn carrier_generic_uniform_accel_traces_to_analytical_source() {
    let rho = 1.5_f64;
    let vel = [0.3_f64, -0.2, 0.4];
    let g = [-0.1_f64, -0.2, -9.81];

    let out = KernelRun::new(uniform_accel_probe_gv::<3>())
        .grid([1usize, 1, 1])
        .fields(&[
            ("rho", rho),
            ("prim_v0", vel[0]),
            ("prim_v1", vel[1]),
            ("prim_v2", vel[2]),
        ])
        .scalars(&[("g_ext_0", g[0]), ("g_ext_1", g[1]), ("g_ext_2", g[2])])
        .run();

    let v_dot_g = vel[0] * g[0] + vel[1] * g[1] + vel[2] * g[2];
    out.expect(
        [0usize, 0, 0],
        &[
            ("s_mom_0", rho * g[0]),
            ("s_mom_1", rho * g[1]),
            ("s_mom_2", rho * g[2]),
            ("s_nrg", rho * v_dot_g),
        ],
        1e-13,
    );
}

#[test]
fn carrier_generic_point_mass_gravity_traces_to_analytical_source() {
    let rho = 1.5_f64;
    let vel = [0.3_f64, -0.2, 0.4];
    let x = [1.0_f64, 0.5, -0.4];
    let xm = [0.1_f64, -0.3, 0.2];
    let gm = 2.0_f64;
    let eps = 0.05_f64;

    let out = KernelRun::new(point_mass_gravity_probe_gv::<3>())
        .grid([1usize, 1, 1])
        .fields(&[
            ("rho", rho),
            ("prim_v0", vel[0]),
            ("prim_v1", vel[1]),
            ("prim_v2", vel[2]),
        ])
        .scalars(&[
            ("gm", gm), ("eps", eps),
            ("x_0", x[0]), ("x_1", x[1]), ("x_2", x[2]),
            ("xm_0", xm[0]), ("xm_1", xm[1]), ("xm_2", xm[2]),
        ])
        .run();

    // softened reference: f = rho*GM / (|x-xm|^2 + eps^2)^{3/2}.
    let dx = [x[0] - xm[0], x[1] - xm[1], x[2] - xm[2]];
    let r_sq = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2] + eps * eps;
    let f = rho * gm / (r_sq * r_sq.sqrt());
    let v_dot_dx = vel[0] * dx[0] + vel[1] * dx[1] + vel[2] * dx[2];
    out.expect(
        [0usize, 0, 0],
        &[
            ("s_mom_0", -f * dx[0]),
            ("s_mom_1", -f * dx[1]),
            ("s_mom_2", -f * dx[2]),
            ("s_nrg", -f * v_dot_dx),
        ],
        1e-12,
    );
}

#[test]
fn user_expression_codegens_through_the_source_path() {
    // a user "script": accel = p0 * sin(x_0) — a spatially varying source field, the kind a
    // config-driven user would pass. parse-shaped here as a symbi-expr Dag, bridged into the
    // symbi-ir Graph, then spliced into a Gv kernel and RENDERED (the harness emits CUDA +
    // evaluates on the CPU interp). proves a user expression compiles through the SAME path a
    // built-in source uses, producing compiled kernel code; there is no per-cell interpreted VM.
    use symbi_expr::dag::Dag;
    use symbi_hydro::expr_bridge::lower_dag_to_builtsource;

    let mut dag = Dag::new();
    let x0 = dag.var_x1();
    let p0 = dag.param(0);
    let sinx = dag.sin(x0);
    let root = dag.mul(p0, sinx);
    let nodes = dag.nodes().to_vec();
    let built = lower_dag_to_builtsource(&nodes, &[root]).expect("bridge lowers");

    let (x0v, p0v) = (0.7_f64, 2.5_f64);
    let out = KernelRun::new(splice_user_source_gv(&built))
        .grid([1usize])
        .scalars(&[("x_0", x0v), ("p0", p0v)])
        .run();

    out.expect([0usize], &[("s_0", p0v * x0v.sin())], 1e-12);
}

#[test]
fn user_force_source_cannot_desync_energy_from_work() {
    // a user "force script": acceleration a(x) = [p0 * x_0, 0.4] (D=2) — a spatially varying body
    // force. the framework wraps it in the conservation law via `user_force_*_source`:
    //   S_mom_k = rho * a_k        (momentum source, D outputs)
    //   S_nrg   = rho * (a . v)     (energy source, DERIVED from the SAME a)
    // the user never writes S_nrg, so it cannot desync from the force. proven two ways below:
    // analytically, and structurally — S_nrg == S_mom . v using ONLY the rendered outputs.
    use symbi_expr::dag::Dag;
    use symbi_hydro::expr_bridge::lower_dag_to_builtsource;
    use symbi_hydro::source_spec::{user_force_energy_source, user_force_momentum_source};

    let mut dag = Dag::new();
    let x0 = dag.var_x1();
    let p0 = dag.param(0);
    let a0 = dag.mul(p0, x0);
    let a1 = dag.constant(0.4);
    let nodes = dag.nodes().to_vec();
    // ONE acceleration field, embedded into BOTH wrappers — the structural reason they can't desync.
    let accel = lower_dag_to_builtsource(&nodes, &[a0, a1]).expect("bridge lowers accel");
    let mom = user_force_momentum_source(&accel, 2);
    let nrg = user_force_energy_source(&accel, 2);

    let (rho, x0v, p0v) = (1.5_f64, 0.7_f64, 2.5_f64);
    let vel = [0.3_f64, -0.2_f64];

    let out_mom = KernelRun::new(splice_user_source_gv(&mom))
        .grid([1usize])
        .scalars(&[("rho", rho), ("x_0", x0v), ("p0", p0v)])
        .run();
    let out_nrg = KernelRun::new(splice_user_source_gv(&nrg))
        .grid([1usize])
        .scalars(&[("rho", rho), ("vel_0", vel[0]), ("vel_1", vel[1]), ("x_0", x0v), ("p0", p0v)])
        .run();

    // analytical: a = [p0*x0, 0.4]; S_mom_k = rho*a_k; S_nrg = rho*(a.v).
    let a = [p0v * x0v, 0.4_f64];
    out_mom.expect([0usize], &[("s_0", rho * a[0]), ("s_1", rho * a[1])], 1e-12);
    out_nrg.expect([0usize], &[("s_0", rho * (a[0] * vel[0] + a[1] * vel[1]))], 1e-12);

    // structural: the energy source equals the momentum source dotted with velocity — the
    // work-energy coupling, in the RENDERED outputs alone (no reference to `a`).
    let s_mom = [out_mom.get([0usize], "s_0"), out_mom.get([0usize], "s_1")];
    let s_nrg = out_nrg.get([0usize], "s_0");
    let work = s_mom[0] * vel[0] + s_mom[1] * vel[1];
    assert!(
        (s_nrg - work).abs() < 1e-12,
        "work-energy coupling violated: S_nrg={s_nrg}, S_mom.v={work}",
    );
}

#[test]
fn user_cooling_source_is_energy_sink_only() {
    // a user cooling rate Lambda(x) = p0 * x_0^2. the framework wraps it as S_nrg = -Lambda,
    // reaching ONLY the energy slot — a cooling kind cannot touch momentum or mass.
    use symbi_expr::dag::Dag;
    use symbi_hydro::expr_bridge::lower_dag_to_builtsource;
    use symbi_hydro::source_spec::user_cooling_source;

    let mut dag = Dag::new();
    let x0 = dag.var_x1();
    let p0 = dag.param(0);
    let x0sq = dag.mul(x0, x0);
    let lam = dag.mul(p0, x0sq);
    let nodes = dag.nodes().to_vec();
    let rate = lower_dag_to_builtsource(&nodes, &[lam]).expect("bridge lowers rate");
    let cool = user_cooling_source(&rate, 2);

    let (x0v, p0v) = (0.7_f64, 2.5_f64);
    let out = KernelRun::new(splice_user_source_gv(&cool))
        .grid([1usize])
        .scalars(&[("x_0", x0v), ("p0", p0v)])
        .run();
    out.expect([0usize], &[("s_0", -(p0v * x0v * x0v))], 1e-12);
}

#[test]
fn front_door_json_force_config_renders_axiomatic_source() {
    // THE FULL FRONT DOOR: a force SourceConfig — exactly what python `Dag.force_source` emits as
    // json — parsed, then `build_user_source` lowers + wraps it in the conservation law, and the
    // mom + nrg BuiltSources render. accel = [p0*x_0, 0.4]; the coupling S_nrg == S_mom.v holds,
    // driven entirely from the serialized config (python -> json -> rust -> kernel, no recompile).
    use symbi_expr::SourceConfig;
    use symbi_hydro::expr_bridge::build_user_source;
    use symbi_hydro::NEWTONIAN_SPEC;

    let json = r#"{
        "kind": "force", "dim": 2, "outputs": [2, 3], "params": [],
        "nodes": [
            {"op": "VARIABLE_X1"},
            {"op": "PARAMETER", "param_idx": 0},
            {"op": "MULTIPLY", "left": 1, "right": 0},
            {"op": "CONSTANT", "value": 0.4}
        ]
    }"#;
    let cfg = SourceConfig::from_json(json).expect("parse config");
    let built = build_user_source(&cfg, &NEWTONIAN_SPEC).expect("build axiomatic source");
    assert_eq!(built.len(), 2);
    assert_eq!(built[0].0, "mom");
    assert_eq!(built[1].0, "nrg");

    let (rho, x0v, p0v) = (1.5_f64, 0.7_f64, 2.5_f64);
    let vel = [0.3_f64, -0.2_f64];

    let out_mom = KernelRun::new(splice_user_source_gv(&built[0].1))
        .grid([1usize])
        .scalars(&[("rho", rho), ("x_0", x0v), ("p0", p0v)])
        .run();
    let out_nrg = KernelRun::new(splice_user_source_gv(&built[1].1))
        .grid([1usize])
        .scalars(&[("rho", rho), ("vel_0", vel[0]), ("vel_1", vel[1]), ("x_0", x0v), ("p0", p0v)])
        .run();

    let a = [p0v * x0v, 0.4_f64];
    out_mom.expect([0usize], &[("s_0", rho * a[0]), ("s_1", rho * a[1])], 1e-12);
    out_nrg.expect([0usize], &[("s_0", rho * (a[0] * vel[0] + a[1] * vel[1]))], 1e-12);

    let s_mom = [out_mom.get([0usize], "s_0"), out_mom.get([0usize], "s_1")];
    let work = s_mom[0] * vel[0] + s_mom[1] * vel[1];
    assert!(
        (out_nrg.get([0usize], "s_0") - work).abs() < 1e-12,
        "work-energy coupling violated from json config",
    );
}
