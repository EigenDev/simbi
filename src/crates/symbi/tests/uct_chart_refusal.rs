// =============================================================================
// uct_chart_refusal.rs
//
// a constrained-transport method is admitted on a chart only where every edge
// of the chart has an emitted EMF kernel. the flat cylindrical r-z plane has
// the single edge E_phi, built for Contact alone, so UCT there is refused by
// name at the kernel set's first CFL: the evolve panics naming the method and
// the chart, completes no step, and leaves the conserved state and the face
// field bitwise as seeded. the neighbors run: Contact on the r-z plane, UCT on
// the cartesian x-y and the cylindrical r-phi planes, whose single edge is the
// out-of-plane E_z.
//
// run: cargo test -p symbi --test uct_chart_refusal
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

const REFUSAL: &str =
    "Uct constrained transport is not built for the Cylindrical chart on grid axes [0, 2]";

#[test]
fn uct_on_the_flat_cylindrical_rz_plane_is_refused_before_any_step() {
    let s = sim!(NewtonianMhd, Cylindrical, false);
    let k = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(
        4.0 / 3.0,
        0.3,
        1.5,
        &s.geom.allocated,
    )
    .ct_method(CtMethod::Uct);
    let (message, steps, kept) = attempt!(s, k);
    assert!(
        message.as_deref().is_some_and(|m| m.contains(REFUSAL)),
        "newtonian MHD: {message:?}"
    );
    assert_eq!(
        steps, 0,
        "newtonian MHD completed {steps} steps before the refusal"
    );
    assert!(
        kept,
        "newtonian MHD: the refused run changed the conserved state or the face field"
    );

    let s = sim!(Rmhd, Cylindrical, false);
    let k =
        RmhdSubstrateKernelSet::<HostMemory, f64, 2>::new(4.0 / 3.0, 0.3, 1.5, &s.geom.allocated)
            .ct_method(CtMethod::Uct);
    let (message, steps, kept) = attempt!(s, k);
    assert!(
        message.as_deref().is_some_and(|m| m.contains(REFUSAL)),
        "relativistic MHD: {message:?}"
    );
    assert_eq!(
        steps, 0,
        "relativistic MHD completed {steps} steps before the refusal"
    );
    assert!(
        kept,
        "relativistic MHD: the refused run changed the conserved state or the face field"
    );
}

#[test]
fn the_charts_whose_edges_are_built_run() {
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
