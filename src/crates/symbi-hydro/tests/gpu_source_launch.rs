// =============================================================================
// gpu_source_launch.rs
//
// **end-to-end GPU launch test for spec-driven sources** — the
// load-bearing claim:
//
//   the same spec data, lowered through `Homomorphism<Cuda>`, NVRTC-
//   compiled, and launched on actual GPU hardware, produces ULP-equivalent
//   results to the CPU `SourceEvaluator` path on the same SimulationLaws.
//
// proves the A1 homomorphism law for the GPU lowering target by witnessing
// it numerically: spec data -> CPU eval == spec data -> GPU launch (modulo
// nvcc/NVRTC FMA fusion drift, which is ULP-bounded).
//
// run (in the cuda-enabled environment, e.g., symbi-cuda distrobox):
//   cargo test -p symbi-hydro --features cuda --test gpu_source_launch
//
// the test is gated behind `--features cuda`; without the feature this
// file compiles to nothing.
// =============================================================================

#![cfg(feature = "gpu")]

use symbi_hydro::regime_spec::{NEWTONIAN_SPEC, law_params};
use symbi_hydro::source_spec::{gravity_params, point_mass_gravity_sources, source_params};
use symbi_hydro::{GpuSourceKernel, SimulationLaws, SourceEvaluator, launch_source_kernel};

/// helper: build a small grid of per-cell field values for one named
/// parameter (e.g., "rho", "vel_0", "xm_0", "gm"). uniform-across-cells
/// for scalars (gm, xm_*); per-cell for position (x_*) etc.
fn build_input_buffers(
    params: &[String],
    cell_count: usize,
    cell_vals: &dyn Fn(usize) -> Vec<(String, f64)>,
) -> Vec<Vec<f64>> {
    let mut buffers: Vec<Vec<f64>> = params
        .iter()
        .map(|_| Vec::with_capacity(cell_count))
        .collect();
    for i in 0..cell_count {
        let vals = cell_vals(i);
        for (p_idx, pname) in params.iter().enumerate() {
            let v = vals
                .iter()
                .find(|(n, _)| n == pname)
                .map(|(_, v)| *v)
                .unwrap_or_else(|| panic!("cell {i}: missing param '{pname}'"));
            buffers[p_idx].push(v);
        }
    }
    buffers
}

#[test]
fn gpu_launch_gravity_mom_matches_cpu_evaluator_per_cell() {
    // **the load-bearing GPU witness**: build a SimulationLaws + a
    // GpuSourceKernel, launch the resulting CUDA kernel over N cells,
    // and compare each cell's output to the CPU `SourceEvaluator`'s
    // value for that cell.

    const N: usize = 8;
    let sim =
        SimulationLaws::new(&NEWTONIAN_SPEC).with_gravity(point_mass_gravity_sources(3, false));
    let cpu_eval = SourceEvaluator::new(&sim, 3).expect("cpu evaluator");
    let gpu_kern = GpuSourceKernel::new(&sim, 3).expect("gpu kernel");

    // per-cell state. positions sweep over a small 3D test grid; mass
    // params are uniform (one body at origin, gm=1.0).
    let cell_vals = |i: usize| -> Vec<(String, f64)> {
        let f = i as f64 + 1.0; // 1.0 ..= N
        vec![
            (law_params::RHO.to_string(), 1.5),
            (law_params::vel(0), 0.1 * f),
            (law_params::vel(1), -0.05 * f),
            (law_params::vel(2), 0.02 * f),
            (source_params::x(0), 1.0 + 0.3 * f),
            (source_params::x(1), -0.5 + 0.1 * f),
            (source_params::x(2), 0.2 * f),
            (gravity_params::xm(0), 0.0),
            (gravity_params::xm(1), 0.0),
            (gravity_params::xm(2), 0.0),
            (gravity_params::GM.to_string(), 1.0),
            (gravity_params::EPS.to_string(), 0.0),
        ]
    };

    let params = gpu_kern.params_for("mom").expect("params declared");
    let input_buffers = build_input_buffers(params, N, &cell_vals);

    // launch the GPU kernel.
    let gpu_out: Vec<Vec<f64>> = launch_source_kernel(&gpu_kern, "mom", &input_buffers, N);
    assert_eq!(gpu_out.len(), 3, "3D momentum source emits 3 components");
    for (k, buf) in gpu_out.iter().enumerate() {
        assert_eq!(
            buf.len(),
            N,
            "output buffer {k} has length {} != {N}",
            buf.len()
        );
    }

    // CPU per-cell ground truth.
    for i in 0..N {
        let vals_owned = cell_vals(i);
        let vals_ref: Vec<(&str, f64)> = vals_owned.iter().map(|(n, v)| (n.as_str(), *v)).collect();
        let cpu = cpu_eval.eval("mom", &vals_ref).expect("cpu eval");

        for k in 0..3 {
            // FMA fusion: NVRTC fuses, the CPU interpreter doesn't. expect
            // ULP-bounded drift; the existing GPU<->CPU validation suite
            // (`gpu_regimes.rs`) uses 1e-9 relative. apply the same.
            let g = gpu_out[k][i];
            let c = cpu[k];
            let rel = (g - c).abs() / c.abs().max(1.0);
            assert!(
                rel < 1e-9,
                "cell {i} component {k}: GPU {g} != CPU {c} (rel {rel:e})",
            );
        }
    }
}

#[test]
fn gpu_launch_ib_localization_holds_on_real_hardware() {
    // **the clause-3 canary on actual GPU**: the IB region mask
    // (`S::select` on `cmp_lt`) emits as a C ternary on Cuda and survives
    // through NVRTC compilation. cells outside the body must produce
    // exactly 0.0 — proves the branchless conditional discipline holds
    // all the way from spec data to compiled GPU machine code.
    use symbi_hydro::source_spec::{ib_params, rigid_body_penalty_sources};

    const N: usize = 8;
    let sim = SimulationLaws::new(&NEWTONIAN_SPEC).with_ib(rigid_body_penalty_sources(3));
    let gpu_kern = GpuSourceKernel::new(&sim, 3).expect("ib kernel");

    // every cell is outside the body (body at origin, radius 1.0; cells
    // at distance > 2.0 along the x-axis). every output must be 0.0.
    let cell_vals = |i: usize| -> Vec<(String, f64)> {
        let f = i as f64 + 1.0;
        vec![
            (law_params::RHO.to_string(), 1.0),
            (law_params::vel(0), 0.5),
            (law_params::vel(1), 0.0),
            (law_params::vel(2), 0.0),
            (source_params::x(0), 3.0 + 0.5 * f), // far outside
            (source_params::x(1), 0.0),
            (source_params::x(2), 0.0),
            (ib_params::body_xm(0), 0.0),
            (ib_params::body_xm(1), 0.0),
            (ib_params::body_xm(2), 0.0),
            (ib_params::BODY_RADIUS.to_string(), 1.0),
            (ib_params::vbody(0), 0.0),
            (ib_params::vbody(1), 0.0),
            (ib_params::vbody(2), 0.0),
            (ib_params::PENALTY_STRENGTH.to_string(), 100.0),
        ]
    };

    let params = gpu_kern.params_for("mom").expect("params declared");
    let input_buffers = build_input_buffers(params, N, &cell_vals);
    let gpu_out = launch_source_kernel(&gpu_kern, "mom", &input_buffers, N);

    for i in 0..N {
        for k in 0..3 {
            assert_eq!(
                gpu_out[k][i], 0.0,
                "cell {i} component {k}: outside-body source must be EXACTLY 0.0 \
                 on GPU (clause-3 canary at the compiled-kernel level)",
            );
        }
    }
}

#[test]
fn gpu_launch_composed_overlays_sums_additively_on_hardware() {
    // **the composition x GPU witness**: two overlay sources, summed
    // additively via `build_total_source`, lowered through
    // `Homomorphism<Cuda>`, NVRTC-compiled, launched. the GPU output
    // matches the CPU evaluator's per-cell sum to ULP-bounded drift.
    // proves the additive-composition contract survives all the way to
    // the GPU machine code.
    use symbi_hydro::source_spec::spherical_geometric_sources;

    const N: usize = 8;
    let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
        .with_geometric(spherical_geometric_sources(3))
        .with_gravity(point_mass_gravity_sources(3, false));
    let cpu_eval = SourceEvaluator::new(&sim, 3).expect("cpu");
    let gpu_kern = GpuSourceKernel::new(&sim, 3).expect("gpu");

    let cell_vals = |i: usize| -> Vec<(String, f64)> {
        // spherical coords: r = x_0 (well above 0 to avoid singularities),
        // theta = x_1 (well away from 0/pi), phi = x_2.
        let f = i as f64 + 1.0;
        vec![
            (law_params::RHO.to_string(), 1.2),
            (law_params::vel(0), 0.1 * f),
            (law_params::vel(1), 0.15),
            (law_params::vel(2), 0.05),
            (law_params::PRE.to_string(), 0.8),
            (source_params::x(0), 2.0 + 0.2 * f),  // r > 0
            (source_params::x(1), 0.8 + 0.05 * f), // theta safely positive
            (source_params::x(2), 0.3), // phi (unused by 3D sph mom source's expr but a declared param)
            (gravity_params::xm(0), 0.5),
            (gravity_params::xm(1), 0.5),
            (gravity_params::xm(2), 0.5),
            (gravity_params::GM.to_string(), 1.0),
            (gravity_params::EPS.to_string(), 0.0),
        ]
    };

    let params = gpu_kern.params_for("mom").expect("params");
    let input_buffers = build_input_buffers(params, N, &cell_vals);
    let gpu_out = launch_source_kernel(&gpu_kern, "mom", &input_buffers, N);

    for i in 0..N {
        let vals_owned = cell_vals(i);
        let vals_ref: Vec<(&str, f64)> = vals_owned.iter().map(|(n, v)| (n.as_str(), *v)).collect();
        let cpu = cpu_eval.eval("mom", &vals_ref).expect("cpu eval");

        for k in 0..3 {
            let g = gpu_out[k][i];
            let c = cpu[k];
            let rel = (g - c).abs() / c.abs().max(1.0);
            assert!(
                rel < 1e-9,
                "composed cell {i} comp {k}: GPU {g} != CPU {c} (rel {rel:e})",
            );
        }
    }
}
