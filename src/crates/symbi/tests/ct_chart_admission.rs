// =============================================================================
// ct_chart_admission.rs
//
// a constrained-transport method is admitted on a chart only where every edge
// of the chart has an emitted EMF kernel; the kernel set asks the kernel
// registry at its first CFL, before any stage runs, and panics with the
// refusal otherwise. gates:
// - the refusal names the method, the chart and the missing kernel: with the
//   registry withholding the r-z E_phi UCT kernel, UCT on the flat cylindrical
//   r-z plane is refused for newtonian and relativistic MHD, and Contact there
//   is admitted;
// - under the production registry every flat 2.5D chart (x-y, r-phi, r-z,
//   r-theta) admits both methods for every magnetized regime and solver
//   family, and every curved 2.5D chart that bakes GRMHD constrained transport
//   admits both for relativistic MHD;
// - UCT and Contact evolve on the r-z plane, and UCT on the x-y and r-phi
//   planes, completing steps.
//
// run: cargo test -p symbi --test ct_chart_admission
// =============================================================================

use std::panic::AssertUnwindSafe;
use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{Cartesian, Cylindrical};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const N: usize = 16;

/// a smooth magnetized flow on a 16 x 16 grid of `$regime` over `$metric`, the r-phi plane when
/// `$rphi` holds
macro_rules! sim {
    ($regime:expr, $metric:expr, $rphi:expr) => {{
        let b = SimStateGeneric::<_, 2, 3, _, IdealGas<f64>, CpuSpace, HostMemory>::build(
            $regime,
            IdealGas { gamma: 4.0 / 3.0 },
            $metric,
        );
        let b = if $rphi {
            b.cyl_plane(CylPlane::RPhi)
        } else {
            b
        };
        b.cells([N, N])
            .bounds([1.0, 0.0], [2.0, 1.0])
            .boundaries(Boundaries::per_axis([
                [BoundaryType::Outflow, BoundaryType::Outflow],
                [BoundaryType::Periodic, BoundaryType::Periodic],
            ]))
            .cfl(0.3)
            .timestepping(Timestepping::Rk2)
            .allocate()
            .expect("sim")
            .set_initial(|[x, y]: [f64; 2]| {
                MhdPrim::new(
                    Prim::adiabatic(
                        Density(1.0),
                        Tensor::new([0.1 * (6.0 * y).sin(), 0.05 * x, 0.1]),
                        Pressure(1.0),
                    ),
                    Tensor::new([0.0, 0.0, 0.0]),
                )
            })
            .seed_faces(|axis, [_, _]| if axis == 1 { 0.2 } else { 0.0 })
            .build()
    }};
}

/// the conserved state and the face field over the interior, as bits
fn bits(s: &FieldStore<2, 3, HostMemory, f64>) -> Vec<u64> {
    let c = &s.fields.cons;
    let m = s.fields.mhd.as_ref().unwrap();
    let mut fields = vec![&c.den, c.nrg_field().unwrap()];
    fields.extend(c.mom.iter());
    fields.extend(m.bface.b.iter());
    fields
        .iter()
        .flat_map(|f| {
            s.geom
                .interior
                .iter()
                .map(|x| f.view().at(x).to_bits())
                .collect::<Vec<_>>()
        })
        .collect()
}

/// the evolve's panic message, the steps it completed, and whether the state kept its bits
macro_rules! attempt {
    ($sim:expr, $k:expr) => {{
        let mut s = $sim;
        let before = bits(&s);
        let mut steps = 0u64;
        let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
            evolve_with_callback(&mut s, &$k, 0.05, 1, |_| steps += 1).expect("evolve");
        }));
        let message = outcome.err().map(|p| {
            p.downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|m| m.to_string()))
                .unwrap_or_default()
        });
        (message, steps, bits(&s) == before)
    }};
}

use symbi::regimes::mhd_substrate::unbuilt_edge_emf;
use symbi::sim::substrate_seam::Solver;
use symbi_geometry::Geometry;

/// the production registry: whether the kernel of this name was emitted
fn registry(name: &str) -> bool {
    symbi_aot::kernel_by_name::<f64>(name).is_some()
}

#[test]
fn a_chart_with_an_unbuilt_edge_is_refused_by_name() {
    let withheld = "rmhd_edge_emf_uct_2d_1";
    let without = |name: &str| name != withheld && registry(name);
    for prefix in ["nmhd", "rmhd"] {
        let refusal = unbuilt_edge_emf::<2>(Geometry::Cylindrical, &[0, 2], "", CtMethod::Uct, Solver::Hlle, prefix, without)
            .unwrap_or_else(|| panic!("{prefix}: UCT admitted on r-z with its E_phi kernel withheld"));
        assert!(
            refusal.contains("Uct constrained transport is not built for the Cylindrical chart on grid axes [0, 2]")
                && refusal.contains(withheld),
            "{prefix}: {refusal}"
        );
        assert_eq!(
            unbuilt_edge_emf::<2>(Geometry::Cylindrical, &[0, 2], "", CtMethod::Contact, Solver::Hlle, prefix, without),
            None,
            "{prefix}: Contact refused on r-z"
        );
    }
}

#[test]
fn every_flat_chart_admits_both_methods_for_every_regime_and_solver() {
    let charts = [
        (Geometry::Cartesian, [0usize, 1]),
        (Geometry::Cylindrical, [0, 1]),
        (Geometry::Cylindrical, [0, 2]),
        (Geometry::Spherical, [0, 1]),
    ];
    for (coords, axes) in charts {
        for prefix in ["nmhd", "imhd", "rmhd"] {
            for ct in [CtMethod::Contact, CtMethod::Uct] {
                for solver in [Solver::Hlle, Solver::Hlld] {
                    assert_eq!(
                        unbuilt_edge_emf::<2>(coords, &axes, "", ct, solver, prefix, registry),
                        None,
                        "{prefix} {ct:?} {solver:?} on {coords:?} {axes:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn every_curved_chart_admits_both_methods_for_every_solver() {
    // the curved 2.5D charts that bake GRMHD constrained transport: the kerr-schild x-y plane, the
    // spinning-kerr x-y plane, the kerr-schild r-phi disk and r-z plane, and the spinning-kerr
    // r-theta plane
    let charts = [
        (Geometry::Cartesian, [0usize, 1], "_ks"),
        (Geometry::Cartesian, [0, 1], "_kerr"),
        (Geometry::Cylindrical, [0, 1], "_ks"),
        (Geometry::Cylindrical, [0, 2], "_ks"),
        (Geometry::Spherical, [0, 1], "_kerr"),
    ];
    for (coords, axes, st) in charts {
        for ct in [CtMethod::Contact, CtMethod::Uct] {
            for solver in [Solver::Hlle, Solver::Hlld] {
                assert_eq!(
                    unbuilt_edge_emf::<2>(coords, &axes, st, ct, solver, "rmhd", registry),
                    None,
                    "{ct:?} {solver:?} on {coords:?} {axes:?} {st}"
                );
            }
        }
    }
}

#[test]
fn the_charts_whose_edges_are_built_run() {
    let s = sim!(NewtonianMhd, Cylindrical, false);
    let k = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(4.0 / 3.0, 0.3, 1.5, &s.geom.allocated)
        .ct_method(CtMethod::Uct);
    let (message, steps, _) = attempt!(s, k);
    assert!(message.is_none() && steps > 0, "newtonian UCT on r-z: {message:?} after {steps} steps");

    let s = sim!(Rmhd, Cylindrical, false);
    let k = RmhdSubstrateKernelSet::<HostMemory, f64, 2>::new(4.0 / 3.0, 0.3, 1.5, &s.geom.allocated)
        .ct_method(CtMethod::Uct);
    let (message, steps, _) = attempt!(s, k);
    assert!(message.is_none() && steps > 0, "relativistic UCT on r-z: {message:?} after {steps} steps");

    let s = sim!(NewtonianMhd, Cylindrical, false);
    let k = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(
        4.0 / 3.0,
        0.3,
        1.5,
        &s.geom.allocated,
    )
    .ct_method(CtMethod::Contact);
    let (message, steps, _) = attempt!(s, k);
    assert!(
        message.is_none() && steps > 0,
        "Contact on r-z: {message:?} after {steps} steps"
    );

    let s = sim!(NewtonianMhd, Cartesian, false);
    let k = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(
        4.0 / 3.0,
        0.3,
        1.5,
        &s.geom.allocated,
    )
    .ct_method(CtMethod::Uct);
    let (message, steps, _) = attempt!(s, k);
    assert!(
        message.is_none() && steps > 0,
        "UCT on x-y: {message:?} after {steps} steps"
    );

    let s = sim!(NewtonianMhd, Cylindrical, true);
    let k = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(
        4.0 / 3.0,
        0.3,
        1.5,
        &s.geom.allocated,
    )
    .ct_method(CtMethod::Uct);
    let (message, steps, _) = attempt!(s, k);
    assert!(
        message.is_none() && steps > 0,
        "UCT on r-phi: {message:?} after {steps} steps"
    );
}
