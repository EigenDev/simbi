// =============================================================================
// dye_gpu.rs
//
// the two passive-scalar write paths that only exist on device.
//
// the dye reaches the GPU through code that the CPU tests never execute: the driven-boundary
// prescription writes `prim.chi` through the NVRTC boundary kernel rather than the host
// interpreter, and the dyed penalize kernels write the drained `cons.chi` through the device
// dispatch. both were compiled and reviewed but unexercised, which is the failure class this repo
// has been bitten by before — an excision path that was dead on one of its two entry points.
//
// each gate asserts the device result against the value the CPU path is separately gated on, so a
// silently wrong GPU write fails here rather than producing a plausible-looking field.
//
// run: cargo test -p symbi --features cuda --test dye_gpu
// =============================================================================
#![cfg(feature = "cuda")]

use symbi::regimes::substrate_kernels::dispatch_penalize;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::KernelSet;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::NEWTONIAN_SPEC;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, SurfaceSpec};
use symbi_source_compile::SourceConfig;
use symbi_source_compile::expr_bridge::build_boundary_dag;
use symbi_xpu::cuda::{CudaSpace, UnifiedMemory};

type DevSim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CudaSpace, UnifiedMemory>;

const CHI_IN: f64 = 0.6;
const POISON: f64 = -7.0;

// the driven-boundary dye slot, through the NVRTC boundary kernel. mirrors the CPU gate
// `driven_inflow_prescribes_the_injected_dye`: the interior is undyed and the ghost band is
// poisoned, so the prescribed concentration can come from neither a copy nor a leftover.
#[test]
fn a_driven_face_prescribes_the_injected_dye_on_gpu() {
    let boundaries = Boundaries::<2>::per_axis([
        [BoundaryType::Driven(0), BoundaryType::Outflow],
        [BoundaryType::Periodic, BoundaryType::Periodic],
    ]);
    let sim = DevSim::build(Newtonian, IdealGas { gamma: 1.4 }, Cartesian)
        .cells([8, 8])
        .bounds([0.0, 0.0], [1.0, 1.0])
        .boundaries(boundaries)
        .finish()
        .unwrap()
        .with_passive_scalar()
        .expect("chi alloc");
    sim.seed_cells(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0]), Pressure(1.0)));
    let chi_f = sim.fields.prim.chi_field().expect("prim chi");
    for c in sim.geom.allocated.iter() {
        let v = if sim.geom.interior.contains(c) {
            0.0
        } else {
            POISON
        };
        chi_f.view_mut().set(c, v);
    }

    // [rho, vel_0, vel_1, pre, chi] — the prim state plus the trailing dye.
    let json = r#"{
        "kind": "dirichlet", "dim": 2, "outputs": [0, 1, 2, 3, 4], "params": [],
        "nodes": [ {"op":"CONSTANT","value":2.0}, {"op":"CONSTANT","value":1.0},
                   {"op":"CONSTANT","value":0.0}, {"op":"CONSTANT","value":3.0},
                   {"op":"CONSTANT","value":0.6} ]
    }"#;
    let cfg = SourceConfig::from_json(json).expect("parse");
    let built = build_boundary_dag(&cfg, &NEWTONIAN_SPEC).expect("driven boundary with dye");
    assert!(
        built.iter().any(|(slot, _)| slot == "chi"),
        "the trailing output must lower to a chi prescription"
    );

    let base =
        AdiabaticSubstrateKernelSet::<UnifiedMemory, f64, 2>::new(1.4, 0.4, &sim.geom.allocated);
    let (sub, _) = base.with_driven_boundary(built, cfg.params.clone());
    sub.ghost_fill(&sim);
    symbi_xpu::cuda::ctx_sync();

    let mut checked = 0usize;
    for c in sim.geom.allocated.iter() {
        let x = sim.geom.cell_coord(c);
        if x[0] >= 0.0 || x[1] <= 0.0 || x[1] >= 1.0 {
            continue;
        }
        let got = *chi_f.view().at(c);
        assert!(
            (got - POISON).abs() > 1e-12,
            "gpu x_lo dye ghost at {c:?} was never written (still poisoned)"
        );
        assert!(
            (got - CHI_IN).abs() < 1e-12,
            "gpu x_lo dye ghost at {c:?}: {got} != prescribed {CHI_IN}"
        );
        checked += 1;
    }
    assert!(checked > 0, "no x_lo ghost-band cells found on device");
}

// the dyed penalize kernel's `cons.chi` write, on device. mirrors the CPU gate
// `a_sink_removes_mass_and_its_dye_together`: a uniform dye must survive the drain, because the
// sink removes gas and its dye together and leaves the concentration untouched.
#[test]
fn a_sink_carries_the_dye_with_the_mass_on_gpu() {
    const CHI: f64 = 0.7;
    const RADIUS: f64 = 0.3;
    let n = 48usize;
    let l = 1.0;
    let dx = 2.0 * l / n as f64;
    let sim = DevSim::build(Newtonian, IdealGas { gamma: 1.4 }, Cartesian)
        .cells([n, n])
        .origin([-l, -l])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(0.3)
        .allocate()
        .expect("sim")
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0]), Pressure(1.0)))
        .build()
        .with_bodies(
            BodyCollection::new().add(
                // gravitating because the spherical drain carries a free-fall-rate arm;
                // the pull never acts here, since the gravity source is not dispatched.
                Body::black_hole(
                    0,
                    Tensor::new([0.0, 0.0]),
                    Tensor::new([0.0, 0.0]),
                    1.0,
                    RADIUS,
                    RADIUS,
                    0.0,
                    1.0,
                    RADIUS,
                )
                .with_surface(SurfaceSpec::Drain),
            ),
        )
        .with_passive_scalar()
        .expect("chi alloc");

    let cons_chi = sim.fields.cons.chi_field().expect("cons chi");
    let prim_chi = sim.fields.prim.chi_field().expect("prim chi");
    for c in sim.geom.allocated.iter() {
        let rho = *sim.fields.cons.den.view().at(c);
        cons_chi.view_mut().set(c, rho * CHI);
        prim_chi.view_mut().set(c, CHI);
    }

    let mass = || -> f64 {
        sim.geom
            .interior
            .iter()
            .map(|c| *sim.fields.cons.den.view().at(c))
            .sum()
    };
    let mass0 = mass();
    for _ in 0..40 {
        dispatch_penalize(&sim, 1e-3, 1.4, 1.0);
    }
    symbi_xpu::cuda::ctx_sync();
    let swallowed = (mass0 - mass()) / mass0;
    assert!(
        swallowed > 1e-3,
        "the gpu sink drained only {swallowed:e} of the mass; the gate is vacuous"
    );

    let (mut worst, mut worst_at) = (0.0_f64, [0isize; 2]);
    for c in sim.geom.interior.iter() {
        let rho = *sim.fields.cons.den.view().at(c);
        if rho <= 0.0 {
            continue;
        }
        let err = (*cons_chi.view().at(c) / rho - CHI).abs();
        if err > worst {
            worst = err;
            worst_at = c;
        }
    }
    assert!(
        worst < 1e-12,
        "the gpu sink changed the dye concentration by {worst:e} at {worst_at:?} (seeded {CHI}); \
         the device drain is not carrying the dye with the mass"
    );
}
