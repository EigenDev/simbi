// =============================================================================
// viscous_capability.rs
//
// the post-step recipe runs a kernel set's viscous transport only where the set
// declares itself viscous, so every set that carries a viscous operator must
// declare it exactly when a viscosity is configured:
// - the newtonian, isothermal and newtonian-MHD sets are inviscid by default
//   and viscous with a constant nu or an alpha prescription;
// - an isothermal shear layer v_x = V sin(k y) loses far more kinetic energy
//   with a constant nu than the inviscid scheme's own numerical floor, through
//   the uni-grid driver the recipe runs in;
// - the relativistic sets carry no viscous transport and refuse a nonzero
//   viscosity or alpha by name when it is configured, before any evolution.
//
// run: cargo test -p symbi --test viscous_capability
// =============================================================================

use symbi::regimes::substrate::IsoSubstrateKernelSet;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::regimes::substrate_rhd::RhdSubstrateKernelSet;
use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet;
use symbi::sim::evolve::{KernelSet, evolve_with_callback};
use symbi::sim::state::*;
use symbi::sim::substrate_seam::WithViscosity;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::Isothermal;
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::quantity::Density;
use symbi_hydro::state::PrimG;
use symbi_xpu::{CpuSpace, HostMemory};

type IsoSim = SimState<IsoNewtonian, 2, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;

const N: usize = 32;
const CS: f64 = 1.0;
const V0: f64 = 0.05;
const T_FINAL: f64 = 0.1;

fn domain() -> symbi_algebra::Domain<2> {
    let sim = iso_sim();
    sim.geom.allocated.clone()
}

/// the viscous declaration of a kernel set built by `make`: inviscid, with a constant nu, with an
/// alpha prescription
fn declarations<const DOF: usize, K>(make: impl Fn() -> K) -> [bool; 3]
where
    K: KernelSet<2, DOF, HostMemory, f64> + WithViscosity,
{
    [make().is_viscous(), make().with_viscosity(0.02).is_viscous(), make().with_alpha(0.1).is_viscous()]
}

#[test]
fn every_set_with_a_viscous_operator_declares_it_exactly_when_configured() {
    let d = domain();
    let expected = [false, true, true];
    assert_eq!(declarations::<2, _>(|| AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(5.0 / 3.0, 0.3, &d)), expected, "newtonian");
    assert_eq!(declarations::<2, _>(|| IsoSubstrateKernelSet::<HostMemory, f64, 2>::new(CS, 0.3, &d)), expected, "isothermal");
    assert_eq!(
        declarations::<3, _>(|| NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(5.0 / 3.0, 0.3, 1.0, &d)),
        expected,
        "newtonian MHD"
    );
}

fn iso_sim() -> IsoSim {
    let dx = 1.0 / N as f64;
    let k = 2.0 * std::f64::consts::PI;
    IsoSim::build(IsoNewtonian, Isothermal { cs: CS }, Cartesian)
        .cells([N, N])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .allocate()
        .expect("isothermal sim")
        .set_initial(|[_x, y]| PrimG::<f64, 2, IsoModel>::isothermal(Density(1.0), Tensor::new([V0 * (k * y).sin(), 0.0])))
        .build()
}

fn kinetic_energy(s: &IsoSim) -> f64 {
    s.geom
        .interior
        .iter()
        .map(|c| {
            let rho = *s.fields.cons.den.view().at(c);
            let mx = *s.fields.cons.mom[0].view().at(c);
            let my = *s.fields.cons.mom[1].view().at(c);
            0.5 * (mx * mx + my * my) / rho
        })
        .sum()
}

fn kinetic_loss(nu: f64) -> f64 {
    let mut sim = iso_sim();
    let k0 = kinetic_energy(&sim);
    assert!(k0 > 0.0, "degenerate seed");
    let sub = IsoSubstrateKernelSet::<HostMemory, f64, 2>::new(CS, 0.3, &sim.geom.allocated).with_viscosity(nu);
    evolve_with_callback(&mut sim, &sub, T_FINAL, u64::MAX, |_| {}).expect("isothermal evolve");
    k0 - kinetic_energy(&sim)
}

#[test]
fn isothermal_viscosity_dissipates_far_more_than_the_numerical_floor() {
    let viscous = kinetic_loss(0.02);
    let ideal = kinetic_loss(0.0);
    assert!(
        viscous > 5.0 * ideal,
        "isothermal viscosity did not act: kinetic energy loss {viscous:.3e} viscous vs {ideal:.3e} inviscid"
    );
}

/// the panic message of `f`, or none when it returns
fn refusal(f: impl FnOnce() + std::panic::UnwindSafe) -> Option<String> {
    std::panic::catch_unwind(f).err().map(|p| {
        p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
    })
}

#[test]
fn the_relativistic_sets_refuse_a_configured_viscosity_by_name() {
    let d = domain();
    for (label, nu, alpha) in [("nu", 0.02, 0.0), ("alpha", 0.0, 0.1)] {
        let rhd = refusal(|| {
            let _ = RhdSubstrateKernelSet::<HostMemory, f64, 2>::new(4.0 / 3.0, 0.3, &d).with_viscosity(nu).with_alpha(alpha);
        });
        assert!(
            rhd.as_deref().is_some_and(|m| m.contains("viscous transport is not built for the rhd regime")),
            "RHD accepted {label}: {rhd:?}"
        );
        let rmhd = refusal(|| {
            let _ = RmhdSubstrateKernelSet::<HostMemory, f64, 2>::new(4.0 / 3.0, 0.3, 1.0, &d).with_viscosity(nu).with_alpha(alpha);
        });
        assert!(
            rmhd.as_deref().is_some_and(|m| m.contains("viscous transport is not built for the rmhd regime")),
            "RMHD accepted {label}: {rmhd:?}"
        );
    }
    // an inviscid configuration builds
    let rhd = RhdSubstrateKernelSet::<HostMemory, f64, 2>::new(4.0 / 3.0, 0.3, &d).with_viscosity(0.0).with_alpha(0.0);
    let rmhd = RmhdSubstrateKernelSet::<HostMemory, f64, 2>::new(4.0 / 3.0, 0.3, 1.0, &d).with_viscosity(0.0);
    assert!(!KernelSet::<2, 2, HostMemory, f64>::is_viscous(&rhd) && !KernelSet::<2, 3, HostMemory, f64>::is_viscous(&rmhd));
}
