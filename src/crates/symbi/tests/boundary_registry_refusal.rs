// =============================================================================
// boundary_registry_refusal.rs
//
// a driven or gradient face names its prescription by id in the kernel set's
// registry. the standard ghost pullback skips these faces, and the registered
// prescription writes their ghost band on every fill.
//   - a registered driven face: one step of a constant density source from gas at
//     rest raises every interior cell, the column beside the driven face included,
//     by the source times the timestep (the flux divergence vanishes at rest);
//   - a driven face whose id is unregistered is refused by name at the first
//     ghost fill, on an empty registry as on a short one, in the adiabatic,
//     isothermal and relativistic gas regimes;
//   - a gradient face whose id is unregistered is refused by name in each of them;
//   - a magnetized run refuses a gradient face by name (the magnetized regimes
//     carry no gradient-wall registry).
//
// run: cargo test -p symbi --test boundary_registry_refusal
// =============================================================================

use symbi::prelude::SimSubstrate;
use symbi::sim::evolve::step_once;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::NEWTONIAN_SPEC;
use symbi::regimes::substrate::IsoSubstrateKernelSet;
use symbi::regimes::substrate_rhd::RhdSubstrateKernelSet;
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::{IdealGas, Isothermal};
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::rhd::Rhd;
use symbi_hydro::state::PrimG;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::substrate_seam::KernelSet;
use symbi_source_compile::SourceConfig;
use symbi_source_compile::expr_bridge::{build_boundary_dag, build_user_source};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const N: usize = 8;

type GasSim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type IsoSim = SimState<IsoNewtonian, 2, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
type RhdSim = SimState<Rhd, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type MhdSim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;
type MhdKernels = symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet<HostMemory, f64, 2>;

/// a unit box of gas at rest whose lower x face is `inner`
fn gas(inner: BoundaryType) -> GasSim {
    GasSim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([0.0, 0.0])
        .spacing([1.0 / N as f64; 2])
        .boundaries(Boundaries::per_axis([[inner, BoundaryType::Outflow], [BoundaryType::Outflow; 2]]))
        .allocate()
        .expect("gas sim")
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0]), Pressure(1.0)))
        .build()
}

/// isothermal gas at rest whose lower x face is `inner`, filled once by its kernel set
fn iso_fill(inner: BoundaryType) {
    let s = IsoSim::build(IsoNewtonian, Isothermal { cs: 1.0 }, Cartesian)
        .cells([N, N])
        .origin([0.0, 0.0])
        .spacing([1.0 / N as f64; 2])
        .boundaries(Boundaries::per_axis([[inner, BoundaryType::Outflow], [BoundaryType::Outflow; 2]]))
        .allocate()
        .expect("iso sim")
        .set_initial(|_| PrimG::<f64, 2, IsoModel>::isothermal(Density(1.0), Tensor::new([0.0, 0.0])))
        .build();
    let kernels = IsoSubstrateKernelSet::<HostMemory, f64, 2>::new(1.0, 0.4, &s.geom.allocated);
    kernels.c2p(&s);
    kernels.ghost_fill(&s);
}

/// relativistic gas at rest whose lower x face is `inner`, filled once by its kernel set
fn rhd_fill(inner: BoundaryType) {
    let s = RhdSim::build(Rhd, IdealGas { gamma: 4.0 / 3.0 }, Cartesian)
        .cells([N, N])
        .origin([0.0, 0.0])
        .spacing([1.0 / N as f64; 2])
        .boundaries(Boundaries::per_axis([[inner, BoundaryType::Outflow], [BoundaryType::Outflow; 2]]))
        .allocate()
        .expect("rhd sim")
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0]), Pressure(1.0)))
        .build();
    let kernels = RhdSubstrateKernelSet::<HostMemory, f64, 2>::new(4.0 / 3.0, 0.4, &s.geom.allocated);
    kernels.c2p(&s);
    kernels.ghost_fill(&s);
}

/// the at-rest state as a driven prescription: rho = 1, v = 0, p = 1
fn rest_prescription() -> (symbi_source_compile::expr_bridge::BoundaryPrescription, Vec<f64>) {
    let cfg = SourceConfig::from_json(
        r#"{"kind": "dirichlet", "dim": 2, "outputs": [0, 1, 2, 3], "params": [],
            "nodes": [{"op": "CONSTANT", "value": 1.0}, {"op": "CONSTANT", "value": 0.0},
                      {"op": "CONSTANT", "value": 0.0}, {"op": "CONSTANT", "value": 1.0}]}"#,
    )
    .expect("boundary config");
    (build_boundary_dag(&cfg, &NEWTONIAN_SPEC).expect("prescription"), cfg.params)
}

/// a constant density source d(rho)/dt = 1
fn unit_density_source() -> SourceConfig {
    SourceConfig::from_json(
        r#"{"kind": "raw", "dim": 2, "outputs": [0], "params": [], "vocabulary": {"reads": [], "params": []},
            "target": "den", "nodes": [{"op": "CONSTANT", "value": 1.0}]}"#,
    )
    .expect("source config")
}

#[test]
fn a_registered_driven_face_lets_every_cell_beside_it_evolve() {
    let mut s = gas(BoundaryType::Driven(0));
    s.timestepping = Timestepping::Euler;
    let source = unit_density_source();
    let built = build_user_source(&source, &NEWTONIAN_SPEC).expect("source");
    let (prescription, bparams) = rest_prescription();
    let (kernels, id) = s.substrate().with_runtime_source(built, source.params).with_driven_boundary(prescription, bparams);
    assert_eq!(id, 0);
    kernels.c2p(&s);
    kernels.ghost_fill(&s);
    let dt = 1e-3;
    let before: Vec<f64> = s.geom.interior.iter().map(|c| *s.fields.cons.den.view().at(c)).collect();
    step_once(&mut s, &kernels, dt);
    let lo = s.geom.interior.spaces[0].lo;
    let mut beside = 0;
    for (c, b) in s.geom.interior.iter().zip(before) {
        let gain = (*s.fields.cons.den.view().at(c) - b) / dt;
        assert!((gain - 1.0).abs() < 1e-10, "cell {c:?} gained {gain} per unit time against the source's 1");
        beside += usize::from(c[0] == lo);
    }
    assert_eq!(beside, N, "the check covered {beside} cells beside the driven face");
}

#[test]
#[should_panic(expected = "references unregistered id 0")]
fn a_driven_face_on_an_empty_registry_is_refused() {
    let s = gas(BoundaryType::Driven(0));
    let source = unit_density_source();
    let built = build_user_source(&source, &NEWTONIAN_SPEC).expect("source");
    let kernels = s.substrate().with_runtime_source(built, source.params);
    kernels.c2p(&s);
    kernels.ghost_fill(&s);
}

#[test]
#[should_panic(expected = "references unregistered id 1")]
fn a_driven_face_past_the_registry_is_refused() {
    let s = gas(BoundaryType::Driven(1));
    let (prescription, bparams) = rest_prescription();
    let (kernels, _) = s.substrate().with_driven_boundary(prescription, bparams);
    kernels.c2p(&s);
    kernels.ghost_fill(&s);
}

#[test]
#[should_panic(expected = "gradient boundary on axis 0 side 0 references unregistered id 0")]
fn a_gradient_face_on_an_empty_registry_is_refused() {
    let s = gas(BoundaryType::Neumann(0));
    let kernels = s.substrate();
    kernels.c2p(&s);
    kernels.ghost_fill(&s);
}

#[test]
#[should_panic(expected = "references unregistered id 0")]
fn an_isothermal_driven_face_on_an_empty_registry_is_refused() {
    iso_fill(BoundaryType::Driven(0));
}

#[test]
#[should_panic(expected = "gradient boundary on axis 0 side 0 references unregistered id 0")]
fn an_isothermal_gradient_face_on_an_empty_registry_is_refused() {
    iso_fill(BoundaryType::Neumann(0));
}

#[test]
#[should_panic(expected = "references unregistered id 0")]
fn a_relativistic_driven_face_on_an_empty_registry_is_refused() {
    rhd_fill(BoundaryType::Driven(0));
}

#[test]
#[should_panic(expected = "gradient boundary on axis 0 side 0 references unregistered id 0")]
fn a_relativistic_gradient_face_on_an_empty_registry_is_refused() {
    rhd_fill(BoundaryType::Robin(0));
}

#[test]
#[should_panic(expected = "magnetized regimes accept no Neumann or Robin faces")]
fn a_magnetized_run_refuses_a_gradient_face() {
    let s = MhdSim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([0.0, 0.0])
        .spacing([1.0 / N as f64; 2])
        .boundaries(Boundaries::per_axis([[BoundaryType::Robin(0), BoundaryType::Outflow], [BoundaryType::Outflow; 2]]))
        .allocate()
        .expect("mhd sim")
        .set_initial(|_| {
            MhdPrim::new(Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)), Tensor::new([1.0, 0.0, 0.0]))
        })
        .seed_faces_uniform([1.0, 0.0])
        .build();
    let kernels = MhdKernels::new(GAMMA, 0.4, 1.0, &s.geom.allocated);
    kernels.c2p(&s);
    kernels.ghost_fill(&s);
}
