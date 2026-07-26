// =============================================================================
// additive_source_equals_fused_evolve.rs
//
// **S3b proof — the evolve-loop lift of S2**. the kernel-level S2 proof
// (`symbi-discretize::godunov_with_fused_source::fused_stage_equals_plain_plus_
// additive_pass`) asserts a SINGLE stage of fused-godunov is bit-identical to
// plain-godunov + the standalone `source_apply` pass. this test lifts that to
// the production `evolve()` loop: a full multi-step march with the SAME source
// run two ways —
//   - FUSED:    `with_fused_source(b)`    -> source folded into the godunov kernel
//   - ADDITIVE: `with_additive_source(b)` -> plain godunov + per-stage source_apply
// must produce bit-for-bit identical conserved state at every interior cell,
// every step. SSP-RK2 is used so the FP-sensitive corrector stage (a0=ac=0.5,
// the exact reason the fused builder was restructured to add the source as a
// separate `+ ac*dt*S` term) is exercised every step.
//
// if the snapshot (u_stage), the weight (ac*dt), or the source evaluation drift
// by a single ULP, the trajectories diverge and `assert_eq!` fails.
//
// run: cargo test -p symbi --test additive_source_equals_fused_evolve
// =============================================================================

use symbi::regimes::substrate::IsoSubstrateKernelSet;
use symbi::regimes::substrate_kernels::FusedSourceBinding;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Domain;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_grid::Field;
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::{IdealGas, Isothermal};
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::PrimG;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;

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
            "{label} differs at {c:?}: fused={va:?} additive={vb:?} (Δ={:?})",
            va - vb,
        );
    }
}

#[test]
fn adiabatic_uniform_accel_additive_equals_fused_rk2() {
    // 1D ideal-gas Euler under uniform external acceleration, SSP-RK2. fused vs
    // additive must agree bit-for-bit including the energy-side overlay.
    type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let n = 48usize;
    let dx = 1.0 / n as f64;
    let g_ext_0 = 0.5_f64;
    let t_final = 0.06_f64;

    let build = || -> Sim {
        let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
            .cells([n])
            .spacing([dx])
            .boundaries(Boundaries::uniform(BoundaryType::Outflow))
            .finish()
            .expect("sim construction failed");
        let cnrg = sim
            .fields
            .cons
            .nrg_field()
            .expect("Newtonian cons.nrg")
            .clone();
        // a mildly non-uniform profile so flux divergence is nonzero (a uniform
        // state hides any ordering bug behind a zero divergence).
        for c in sim.geom.interior.iter() {
            let x = (c[0] as f64 + 0.5) * dx;
            let rho = 1.0 + 0.2 * (std::f64::consts::TAU * x).sin();
            sim.fields.cons.den.view_mut().set(c, rho);
            sim.fields.cons.mom[0].view_mut().set(c, 0.0);
            cnrg.view_mut().set(c, 1.0 / (GAMMA - 1.0));
        }
        sim
    };

    let binding = || FusedSourceBinding::new("uniform_accel", &[("g_ext_0", g_ext_0)]);

    let mut sim_fused = build();
    let sub_fused = AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(
        GAMMA,
        0.4,
        &sim_fused.geom.allocated,
    )
    .with_fused_source(binding());
    evolve(&mut sim_fused, &sub_fused, t_final).expect("fused evolve failed");

    let mut sim_add = build();
    let sub_add =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, 0.4, &sim_add.geom.allocated)
            .with_additive_source(binding());
    evolve(&mut sim_add, &sub_add, t_final).expect("additive evolve failed");

    // same source, two execution strategies -> identical trajectory, bit-for-bit.
    let interior = &sim_fused.geom.interior;
    assert_cons_bit_identical(
        interior,
        &sim_fused.fields.cons.den,
        &sim_add.fields.cons.den,
        "cons.den",
    );
    assert_cons_bit_identical(
        interior,
        &sim_fused.fields.cons.mom[0],
        &sim_add.fields.cons.mom[0],
        "cons.mom_0",
    );
    let (nf, na) = (
        sim_fused.fields.cons.nrg_field().unwrap(),
        sim_add.fields.cons.nrg_field().unwrap(),
    );
    assert_cons_bit_identical(interior, nf, na, "cons.nrg");
    // sanity: the run actually did something (gas accelerated).
    let moved = sim_fused
        .geom
        .interior
        .iter()
        .any(|c| sim_fused.fields.prim.vel[0].view().at(c).abs() > 1e-6);
    assert!(moved, "gas never accelerated — the test exercised nothing");
}

#[test]
fn adiabatic_point_mass_additive_equals_fused_rk2() {
    // 2D ideal-gas Euler under point-mass gravity, SSP-RK2. exercises the lazily-
    // declared centroid scalars (x_lo_k, dx_k) AND the energy overlay in 2D.
    type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let n = 24usize;
    let bound = 2.0_f64;
    let dx = 2.0 * bound / n as f64;
    let gm = 1.0_f64;
    let t_final = 0.04_f64;

    let build = || -> Sim {
        let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
            .cells([n, n])
            .origin([-bound, -bound])
            .spacing([dx, dx])
            .boundaries(Boundaries::uniform(BoundaryType::Outflow))
            .finish()
            .expect("sim construction failed");
        let cnrg = sim
            .fields
            .cons
            .nrg_field()
            .expect("Newtonian cons.nrg")
            .clone();
        for c in sim.geom.interior.iter() {
            let x = -bound + (c[0] as f64 + 0.5) * dx;
            let y = -bound + (c[1] as f64 + 0.5) * dx;
            let r = (x * x + y * y).sqrt().max(0.3);
            let rho = 1.0 + 0.5 * (-(r - 1.0).powi(2) / 0.2).exp();
            sim.fields.cons.den.view_mut().set(c, rho);
            sim.fields.cons.mom[0].view_mut().set(c, 0.0);
            sim.fields.cons.mom[1].view_mut().set(c, 0.0);
            cnrg.view_mut().set(c, 1.0 / (GAMMA - 1.0));
        }
        sim
    };

    let binding = || {
        FusedSourceBinding::new(
            "point_mass_grav",
            &[("gm", gm), ("xm_0", 0.0), ("xm_1", 0.0), ("eps", 0.0)],
        )
    };

    let mut sim_fused = build();
    let sub_fused = AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(
        GAMMA,
        0.4,
        &sim_fused.geom.allocated,
    )
    .with_fused_source(binding());
    evolve(&mut sim_fused, &sub_fused, t_final).expect("fused evolve failed");

    let mut sim_add = build();
    let sub_add =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.4, &sim_add.geom.allocated)
            .with_additive_source(binding());
    evolve(&mut sim_add, &sub_add, t_final).expect("additive evolve failed");

    let interior = &sim_fused.geom.interior;
    assert_cons_bit_identical(
        interior,
        &sim_fused.fields.cons.den,
        &sim_add.fields.cons.den,
        "cons.den",
    );
    for k in 0..2 {
        assert_cons_bit_identical(
            interior,
            &sim_fused.fields.cons.mom[k],
            &sim_add.fields.cons.mom[k],
            "cons.mom",
        );
    }
    let (nf, na) = (
        sim_fused.fields.cons.nrg_field().unwrap(),
        sim_add.fields.cons.nrg_field().unwrap(),
    );
    assert_cons_bit_identical(interior, nf, na, "cons.nrg");
}

#[test]
fn iso_point_mass_additive_equals_fused_rk2() {
    // kepler's exact regime: 2D isothermal under point-mass gravity, SSP-RK2,
    // no energy law. proves the iso source_apply path matches the iso fused path.
    type Sim = SimState<IsoNewtonian, 2, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
    let n = 24usize;
    let bound = 2.0_f64;
    let dx = 2.0 * bound / n as f64;
    let gm = 1.0_f64;
    let cs = 0.05_f64;
    let t_final = 0.04_f64;

    let build = || -> Sim {
        let sim = Sim::build(IsoNewtonian, Isothermal { cs }, Cartesian)
            .cells([n, n])
            .origin([-bound, -bound])
            .spacing([dx, dx])
            .boundaries(Boundaries::uniform(BoundaryType::Outflow))
            .finish()
            .expect("sim construction failed");
        sim.seed_cells(|p| {
            let (x, y) = (p[0], p[1]);
            let r = (x * x + y * y).sqrt().max(0.3);
            let sigma = 1.0 + 0.5 * (-(r - 1.0).powi(2) / 0.2).exp();
            let v_kep = (gm / r).sqrt();
            PrimG::<f64, 2, IsoModel> {
                rho: sigma,
                vel: Tensor::new([-v_kep * (y / r), v_kep * (x / r)]),
                pre: Default::default(),
            }
        });
        sim
    };

    let binding = || {
        FusedSourceBinding::new(
            "point_mass_grav",
            &[("gm", gm), ("xm_0", 0.0), ("xm_1", 0.0), ("eps", 0.0)],
        )
    };

    let mut sim_fused = build();
    let sub_fused =
        IsoSubstrateKernelSet::<HostMemory, f64, 2>::new(cs, 0.4, &sim_fused.geom.allocated)
            .with_fused_source(binding());
    evolve(&mut sim_fused, &sub_fused, t_final).expect("fused evolve failed");

    let mut sim_add = build();
    let sub_add =
        IsoSubstrateKernelSet::<HostMemory, f64, 2>::new(cs, 0.4, &sim_add.geom.allocated)
            .with_additive_source(binding());
    evolve(&mut sim_add, &sub_add, t_final).expect("additive evolve failed");

    let interior = &sim_fused.geom.interior;
    assert_cons_bit_identical(
        interior,
        &sim_fused.fields.cons.den,
        &sim_add.fields.cons.den,
        "cons.den",
    );
    for k in 0..2 {
        assert_cons_bit_identical(
            interior,
            &sim_fused.fields.cons.mom[k],
            &sim_add.fields.cons.mom[k],
            "cons.mom",
        );
    }
}
