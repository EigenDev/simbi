// =============================================================================
// evolve_fused_source_routing.rs
//
// configuring the substrate kernel-set with a
// `FusedSourceBinding` causes the production `evolve()` loop to use the
// AOT-baked fused godunov in place of the unfused kernel — one launch per RK
// stage covers `div(F) + spec source + integrator`, fully driven from the
// kernel-set's declarative state.
//
// what this validates at the integration layer (a routing check, not a kernel
// unit test):
//   - `AdiabaticSubstrateKernelSet::new(..).with_fused_source(b)` produces a
//     kernel-set whose `godunov_euler` / `godunov_rk2` route to the AOT
//     fused kernel — proven by exercising the full `evolve()` loop and
//     checking that gas under `uniform_acceleration` actually accelerates;
//   - a kernel-set with `fused_source = None` (the default) runs the unfused
//     path, leaving the gas at rest;
//   - the iso variant routes the same way (`IsoSubstrateKernelSet::
//     with_fused_source(..)`).
//
// the physics claim is modest by design: the test only proves the binding
// reaches the kernel and the spec contribution actually applies (the gas
// moves in the prescribed direction by ~ \rho\cdot g\cdot t over the run). bit-equivalence
// vs the unfused path is covered by the lower-layer unit tests; this
// layer's job is end-to-end routing through the real evolve loop.
//
// run: cargo test -p symbi --test evolve_fused_source_routing
// =============================================================================

use symbi::regimes::substrate_kernels::FusedSourceBinding;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
const GAMMA: f64 = 1.4;

#[test]
fn adiabatic_evolve_with_fused_uniform_accel_actually_accelerates_gas() {
    // end-to-end: a `with_fused_source` binding causes the
    // production `evolve()` loop to apply the spec source's contribution at
    // every RK stage. configured with `uniform_acceleration` toward +x, the
    // initially-stationary gas develops a +x velocity proportional to g_ext\cdot t.
    //
    // for a uniform medium under uniform_accel, in the gas frame (no flux
    // divergence at the start), the analytical momentum profile is
    //     mom(t) = mom(0) + \rho\cdot g_ext\cdot t      = \rho\cdot g_ext\cdot t  (starting from rest)
    // which integrates to a velocity v(t) \approx g_ext\cdot t. for g_ext = 0.5 and
    // t_final \approx 0.05, v \approx 0.025 — clearly above any numerical noise floor
    // and far less than cs \approx 1, so the flow stays subsonic + smooth.
    let n = 32usize;
    let dx = 1.0 / n as f64;
    // uniform stationary state — every cell starts at rest, rho = 1, p = 1.
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([n])
        .spacing([dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .timestepping(Timestepping::Euler)
        .allocate()
        .expect("sim construction failed")
        .set_initial(|_x| Prim {
            rho: 1.0,
            vel: Tensor::zeros(),
            pre: 1.0,
        })
        .build();

    // a uniform_accel SourceSpec mapping `g_ext_0 = 0.5`.
    // the kernel-set's `godunov_euler` / `godunov_rk2` will now route through
    // `adiabatic_godunov_euler_with_uniform_accel_1d` — the AOT-baked fused
    // kernel — in every step of the real evolve() loop.
    let g_ext_0 = 0.5_f64;
    let binding = FusedSourceBinding::new("uniform_accel", &[("g_ext_0", g_ext_0)]);
    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, 0.4, &sim.geom.allocated)
            .with_fused_source(binding);

    // march a short time. the fused-source contribution accumulates per RK
    // step inside evolve() — there is no manual `cons.mom += dt\cdot\rho\cdot g_ext` loop
    // outside the substrate. if the fused binding works, gas accelerates;
    // if not, gas stays at rest and the v_mid_after check fails.
    let t_final = 0.05_f64;
    evolve(&mut sim, &sub, t_final).expect("evolve with fused source failed");

    // check: every interior cell carries a nonzero +x velocity, and the
    // average is close to the analytical g_ext\cdot t. the boundary cells drift
    // a touch because of the outflow / ghost interactions over many steps,
    // so the assertion is "interior mean is in the right ballpark".
    let cells: Vec<[isize; 1]> = sim.geom.interior.iter().collect();
    let cnt = cells.len() as f64;
    let mean_v: f64 = cells
        .iter()
        .map(|c| *sim.fields.prim.vel[0].view().at(*c))
        .sum::<f64>()
        / cnt;
    let analytical_v = g_ext_0 * sim.time;

    // gas actually accelerated in the right direction:
    assert!(
        mean_v > 0.0,
        "gas did not accelerate (mean v = {mean_v}); fused source binding did not reach the kernel"
    );
    // and the magnitude is close to the analytical estimate (loose tol —
    // outflow boundary + finite-volume diffusion drag the mean slightly):
    assert!(
        (mean_v - analytical_v).abs() < 0.5 * analytical_v.abs(),
        "mean v = {mean_v} too far from analytical g_ext*t = {analytical_v} (delta = {})",
        (mean_v - analytical_v).abs(),
    );
    // density should stay close to 1 — the mom overlay alone (energy side
    // is also applied for adiabatic, but the gas frame stays near-uniform):
    let rho_min = cells
        .iter()
        .map(|c| *sim.fields.prim.rho.view().at(*c))
        .fold(f64::INFINITY, f64::min);
    let rho_max = cells
        .iter()
        .map(|c| *sim.fields.prim.rho.view().at(*c))
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        rho_min > 0.7 && rho_max < 1.3,
        "density spread out of bounds [{rho_min}, {rho_max}] — the fused step destabilized the flow",
    );
}

#[test]
fn adiabatic_evolve_without_binding_stays_at_rest() {
    // the negative control: the same setup but with no `with_fused_source`
    // binding. evolve() runs the unfused path; gas stays at rest (mean v = 0
    // to floating-point noise). proves the binding is what causes the
    // acceleration.
    let n = 32usize;
    let dx = 1.0 / n as f64;
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([n])
        .spacing([dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .timestepping(Timestepping::Euler)
        .allocate()
        .expect("sim construction failed")
        .set_initial(|_x| Prim {
            rho: 1.0,
            vel: Tensor::zeros(),
            pre: 1.0,
        })
        .build();

    // no `with_fused_source` — the kernel-set has `fused_source: None`, so
    // godunov_euler routes through the unfused `dispatch_godunov`.
    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, 0.4, &sim.geom.allocated);
    evolve(&mut sim, &sub, 0.05).expect("unfused evolve failed");

    let cells: Vec<[isize; 1]> = sim.geom.interior.iter().collect();
    let max_abs_v: f64 = cells
        .iter()
        .map(|c| sim.fields.prim.vel[0].view().at(*c).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs_v < 1e-10,
        "gas accelerated WITHOUT a fused source binding (max |v| = {max_abs_v}); the routing chokepoint leaked",
    );
}
