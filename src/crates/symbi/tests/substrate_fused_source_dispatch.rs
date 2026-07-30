// =============================================================================
// substrate_fused_source_dispatch.rs
//
// the substrate dispatch entry
// `dispatch_godunov_with_sources` selects the AOT-baked FUSED kernel
// (`{prefix}_godunov_euler_with_{source_id}_{ndim}d`) by composing the
// runtime name + packing the spec source's scalar params alongside `dt` + the
// per-axis grid scalars. one substrate call runs `div(F) + spec source +
// integrator` in ONE kernel launch — replacing the two-kernel (godunov +
// body_source) pattern.
//
// **what this validates**:
//   - the runtime composes the right fused-kernel name and the AOT registry
//     resolves it (no missing kernel panic);
//   - `source_scalars` correctly routes spec params (`g_ext_0`, ...) through
//     `scalars_for` into the kernel's declared scalar tail (the substrate's
//     type-sorted scalar manifest accepts them);
//   - invoked on a SimState with constant flux buffers (zero divergence) and
//     the `uniform_accel` overlay, the per-cell cons.mom is updated exactly by
//     the analytical `mom + dt\cdot\rho\cdot g_ext_0` — bit-close at f64;
//   - cons.nrg picks up the energy-overlay contribution too (`v\cdot g_ext`); cons.den
//     stays invariant — proves the multi-source binding routes to the right
//     conservation laws.
//
// run: cargo test -p symbi --test substrate_fused_source_dispatch
// =============================================================================

use std::collections::HashMap;

use symbi::regimes::substrate_kernels::dispatch_godunov_with_sources;
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
fn substrate_routes_to_adiabatic_fused_uniform_accel() {
    // end-to-end: the substrate dispatch composes the fused
    // kernel name, packs the spec source's `g_ext_0` scalar, AOT-registry
    // resolves the kernel, and one launch applies BOTH mom + nrg overlays of
    // `uniform_acceleration` (multi-source) to the SimState in-place.
    let n = 8usize;
    let dx = 1.0 / n as f64;
    // trivial seed to reach a Ready sim; the EXACT conserved literals this test asserts are
    // written raw below (the source step is applied to these raw gauge values, which need not form a physical prim).
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([n])
        .spacing([dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim construction failed")
        .set_initial(|_x| Prim {
            rho: 1.0,
            vel: Tensor::zeros(),
            pre: 1.0,
        })
        .build();

    // uniform interior state — gauge readings the source is supposed to update.
    let rho_v = 1.5_f64;
    let mom_v = 0.3_f64;
    let nrg_v = 5.0_f64;
    let g_ext_0 = -9.81_f64;
    let dt = 0.01_f64;
    let cnrg = sim
        .fields
        .cons
        .nrg_field()
        .expect("Newtonian cons.nrg")
        .clone();
    for c in sim.geom.interior.iter() {
        sim.fields.cons.den.view_mut().set(c, rho_v);
        sim.fields.cons.mom[0].view_mut().set(c, mom_v);
        cnrg.view_mut().set(c, nrg_v);
    }

    // pre-populate the per-axis flux buffers as CONSTANTS — uniform fluxes give
    // zero divergence on the interior, so the output IS the source contribution.
    // (the production evolve() path computes these via the flux kernel; stubbed
    // here so the test isolates the godunov + fused-source step.)
    let face = sim.fields.flux[0].den.domain().clone();
    for c in face.iter() {
        sim.fields.flux[0].den.view_mut().set(c, 0.7);
        sim.fields.flux[0].mom[0].view_mut().set(c, 0.4);
        sim.fields.flux[0]
            .nrg
            .as_ref()
            .expect("flux.nrg")
            .view_mut()
            .set(c, 1.1);
    }

    // the substrate's adiabatic primitive-pressure field — the unused 4th input
    // of geometric godunov; here Cartesian, the kernel doesn't read it. pass
    // `prim.pre` (the runtime's pressure field) to match the existing dispatch
    // convention.
    let pre = sim.fields.prim.pre_field().expect("prim.pre").clone();

    // the fused-dispatch entry point: one substrate call binds buffers + scalars
    // by name through the AOT-registered fused kernel.
    let mut source_scalars: HashMap<String, f64> = HashMap::new();
    source_scalars.insert("g_ext_0".to_string(), g_ext_0);
    dispatch_godunov_with_sources::<1, 1, _, _>(
        &sim,
        &pre,
        "adiabatic",
        dt,
        /* a0, ac = forward-Euler */ 0.0,
        1.0,
        "uniform_accel",
        &source_scalars,
    );

    // verify the analytical update applied per interior cell.
    let cells: Vec<[isize; 1]> = sim.geom.interior.iter().collect();
    let tol = 1e-12;
    for c in &cells {
        let rho_out = *sim.fields.cons.den.view().at(*c);
        let mom_out = *sim.fields.cons.mom[0].view().at(*c);
        let nrg_out = *cnrg.view().at(*c);

        let mom_expected = mom_v + dt * rho_v * g_ext_0;
        // v = mom / rho (Newtonian); S_nrg = \rho \cdot v \cdot g_ext_0
        let v_in = mom_v / rho_v;
        let nrg_expected = nrg_v + dt * rho_v * v_in * g_ext_0;

        assert!(
            (rho_out - rho_v).abs() < tol,
            "cell {c:?}: rho_new {rho_out} != rho_in {rho_v} (mass should be invariant under the momentum/energy overlays)",
        );
        assert!(
            (mom_out - mom_expected).abs() < tol,
            "cell {c:?}: mom_new {mom_out} != analytical {mom_expected}",
        );
        assert!(
            (nrg_out - nrg_expected).abs() < tol,
            "cell {c:?}: nrg_new {nrg_out} != analytical {nrg_expected} (the AOT-fused nrg overlay must apply)",
        );
    }
}

#[test]
fn substrate_routes_to_iso_fused_uniform_accel() {
    // **iso variant**: the iso fused kernel has NO energy law (mass + mom only).
    // proves the per-regime name routing — `iso_godunov_euler_with_uniform_accel_1d`
    // resolves correctly with the same fused-source plumbing.
    use symbi_algebra::Tensor;
    use symbi_hydro::IsoNewtonian;
    use symbi_hydro::energy::IsoModel;
    use symbi_hydro::eos::Isothermal;
    use symbi_hydro::state::PrimG;
    type IsoSim = SimState<IsoNewtonian, 1, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;

    let n = 8usize;
    let dx = 1.0 / n as f64;
    let cs = 1.0_f64;
    // trivial seed to reach a Ready iso sim; the EXACT conserved literals are written raw below.
    let sim = IsoSim::build(IsoNewtonian, Isothermal { cs }, Cartesian)
        .cells([n])
        .spacing([dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("iso sim construction failed")
        .set_initial(|_x| PrimG::<f64, 1, IsoModel> {
            rho: 1.0,
            vel: Tensor::zeros(),
            pre: Default::default(),
        })
        .build();

    let rho_v = 1.5_f64;
    let mom_v = 0.3_f64;
    let g_ext_0 = -9.81_f64;
    let dt = 0.01_f64;
    for c in sim.geom.interior.iter() {
        sim.fields.cons.den.view_mut().set(c, rho_v);
        sim.fields.cons.mom[0].view_mut().set(c, mom_v);
    }
    let face = sim.fields.flux[0].den.domain().clone();
    for c in face.iter() {
        sim.fields.flux[0].den.view_mut().set(c, 0.7);
        sim.fields.flux[0].mom[0].view_mut().set(c, 0.4);
    }
    // iso has no cons.nrg; the iso fused kernel has no nrg overlay either.

    // iso has no prim.pre field — the iso fused godunov doesn't read pressure
    // (no energy term). pass any in-scope field; dispatch_named's resolve_path
    // never looks it up because the iso kernel's manifest has no "prim.pre".
    let pre = sim.fields.cons.den.clone();
    let mut source_scalars: HashMap<String, f64> = HashMap::new();
    source_scalars.insert("g_ext_0".to_string(), g_ext_0);
    dispatch_godunov_with_sources::<1, 1, _, _>(
        &sim,
        &pre,
        "iso",
        dt,
        /* a0, ac = forward-Euler */ 0.0,
        1.0,
        "uniform_accel",
        &source_scalars,
    );

    let cells: Vec<[isize; 1]> = sim.geom.interior.iter().collect();
    let tol = 1e-12;
    for c in &cells {
        let rho_out = *sim.fields.cons.den.view().at(*c);
        let mom_out = *sim.fields.cons.mom[0].view().at(*c);
        let mom_expected = mom_v + dt * rho_v * g_ext_0;
        assert!(
            (rho_out - rho_v).abs() < tol,
            "iso cell {c:?}: rho_new {rho_out} != rho_in {rho_v}",
        );
        assert!(
            (mom_out - mom_expected).abs() < tol,
            "iso cell {c:?}: mom_new {mom_out} != analytical {mom_expected}",
        );
    }
}

#[test]
#[should_panic(expected = "unexpected spec scalar")]
fn missing_source_scalar_panics_loudly() {
    // **discipline**: a `source_scalars` map missing a param the AOT kernel
    // declares (e.g., forgetting `g_ext_0` for uniform_accel) must panic at the
    // dispatch's resolver; silently defaulting the missing scalar to 0.0 would
    // corrupt the source term. surface vocabulary mismatches at the call site.
    let n = 8usize;
    let dx = 1.0 / n as f64;
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([n])
        .spacing([dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim construction failed")
        .set_initial(|_x| Prim {
            rho: 1.0,
            vel: Tensor::zeros(),
            pre: 1.0,
        })
        .build();
    let pre = sim.fields.prim.pre_field().expect("prim.pre").clone();
    // intentionally empty — missing g_ext_0 should surface as a panic.
    let source_scalars: HashMap<String, f64> = HashMap::new();
    dispatch_godunov_with_sources::<1, 1, _, _>(
        &sim,
        &pre,
        "adiabatic",
        0.01,
        /* a0, ac = forward-Euler */ 0.0,
        1.0,
        "uniform_accel",
        &source_scalars,
    );
}
