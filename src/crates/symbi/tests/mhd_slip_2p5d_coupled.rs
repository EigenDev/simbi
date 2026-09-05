// =============================================================================
// mhd_slip_2p5d_coupled.rs
//
// the palindromic coupled step D(dt/2) M(dt/2) H(dt) M(dt/2) D(dt/2) for a slip-enabled draining
// body on a 2.5D cartesian grid: the evolve loop routes such a body through the coupled step, a body
// without the slip keeps the ordinary advance, the event-split drain composes to third order, and
// the coupled step self-converges at second order in each storage complex.
// =============================================================================

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::refinement::hierarchy::{slip_schedule_arm, slip_schedule_take};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, MagneticSpec, SurfaceSpec};
use symbi::regimes::substrate_kernels::Solver;
use symbi_sim::state::CtMethod;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;
type Kernels = NewtonianMhdSubstrateKernelSet<HostMemory, f64, 2>;

const N: usize = 16;
const GAMMA: f64 = 5.0 / 3.0;
const DT: f64 = 2.0e-3;
const BODY: [f64; 2] = [0.5, 0.5];
const R_BODY: f64 = 0.22;

fn slip_spec() -> MagneticSpec {
    MagneticSpec::Slip {
        diffusivity_ratio: 2.0,
        shell_width: 0.12,
        slip_length_ratio: 1.0,
        field_regularization: 0.1,
        placement: 0.0,
    }
}

// a smooth divergence-free in-plane field with a smooth out-of-plane component around a draining
// body, seeded through the two-representation contract.
fn build(n: usize, magnetic: MagneticSpec) -> Sim {
    let dx = 1.0 / n as f64;
    let k = 2.0 * std::f64::consts::PI;
    let (a0, b0) = (0.3, 0.2);
    let sim = SimStateGeneric::<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>::build(
        NewtonianMhd,
        IdealGas { gamma: GAMMA },
        Cartesian,
    )
    .cells([n, n])
    .origin([0.0, 0.0])
    .spacing([dx, dx])
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .cfl(0.3)
    .allocate()
    .expect("2.5D sim construction")
    .set_initial(move |[x, y]| {
        let bx = |xf: f64| -a0 * (k * xf).cos() * (k * y).sin();
        let by = |yf: f64| a0 * (k * x).sin() * (k * yf).cos();
        MhdPrim::new(
            Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)),
            Tensor::new([
                0.5 * (bx(x - 0.5 * dx) + bx(x + 0.5 * dx)),
                0.5 * (by(y - 0.5 * dx) + by(y + 0.5 * dx)),
                b0 * (k * x).cos() * (k * y).cos(),
            ]),
        )
    })
    .seed_faces(move |axis, [x, y]| match axis {
        0 => -a0 * (k * x).cos() * (k * y).sin(),
        _ => a0 * (k * x).sin() * (k * y).cos(),
    })
    .build();
    sim.with_bodies(
        BodyCollection::new().add(
            Body::black_hole(0, Tensor::new(BODY), Tensor::zeros(), 1.0, R_BODY, 0.05, 1.0, 1.0, R_BODY)
                .with_surface(SurfaceSpec::Drain)
                .with_magnetic(magnetic),
        ),
    )
}

// the ideal-MHD step runs under the UCT edge EMF with the two-wave HLLE solver, the transport the
// 2.5D vertical-field model uses: the Contact EMF's mass-flux soft-sign upwinding holds the
// in-plane field at first order in time on any flow whose mass flux changes sign, and the HLLD fan
// switches branches where the normal field crosses zero under a vertical component, so a
// temporal-order gate on the coupled step reads the transport rather than the splitting unless the
// transport is itself second order on this regime.
fn primed(n: usize, magnetic: MagneticSpec) -> Hierarchy<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kernels> {
    let sim = build(n, magnetic);
    let sub = Kernels::new(GAMMA, 0.3, 1.0, &sim.geom.allocated)
        .with_solver(Solver::Hlle)
        .expect("hlle")
        .ct_method(CtMethod::Uct);
    let mut hier = Hierarchy::single(sim, sub);
    hier.prime();
    hier
}

#[test]
fn the_evolve_loop_drives_the_coupled_schedule_for_a_2p5d_slip_body() {
    let mut hier = primed(N, slip_spec());
    slip_schedule_arm();
    hier.step_root_with_dt(DT);
    let trace = slip_schedule_take().expect("schedule was armed");
    let ops: Vec<&str> = trace.iter().map(|(op, _)| *op).collect();
    assert_eq!(ops, ["D", "M", "H", "M", "D"], "the root advance did not run the coupled step");
    let sum = |name: &str| -> f64 { trace.iter().filter(|(op, _)| *op == name).map(|(_, d)| *d).sum() };
    for name in ["D", "M", "H"] {
        assert!((sum(name) - DT).abs() < 1e-15, "{name} advances {} instead of dt", sum(name));
    }
}

#[test]
fn a_2p5d_body_without_the_slip_keeps_the_ordinary_advance() {
    for (label, magnetic) in [("transparent", MagneticSpec::None), ("resistive", MagneticSpec::Resistive { eta: 0.1 })] {
        let mut hier = primed(N, magnetic);
        slip_schedule_arm();
        hier.step_root_with_dt(DT);
        let trace = slip_schedule_take().expect("schedule was armed");
        assert!(trace.is_empty(), "{label}: a body without the slip entered the coupled step: {trace:?}");
    }
}

// D_{h/2} o D_{h/2} against D_h on the event-split drain: the rate stiffens with the local Alfven term
// as the density drains, the midpoint integration is second order, so the composition differs at
// O(h^3) and halving h shrinks the difference eightfold.
#[test]
fn the_2p5d_drain_composes_to_third_order() {
    let run = |dt: f64| -> f64 {
        let two = primed(N, slip_spec());
        let one = primed(N, slip_spec());
        two.drain_and_rebuild(0, 0.5 * dt);
        two.drain_and_rebuild(0, 0.5 * dt);
        one.drain_and_rebuild(0, dt);
        let (a, b) = (two.density_snapshot(0), one.density_snapshot(0));
        a.iter().zip(&b).map(|(x, y)| ((x - y) / x.abs().max(1e-12)).abs()).fold(0.0_f64, f64::max)
    };
    let h = 8.0 * DT;
    let (e_full, e_half) = (run(h), run(0.5 * h));
    assert!(e_full > 1e-9, "vacuous semigroup test (diff {e_full})");
    let ratio = e_full / e_half.max(1e-300);
    assert!(ratio > 6.0, "the drain composition error is not O(h^3): ratio {ratio:.2} (expect ~8)");
}

// the coupled step to a fixed final time under timestep refinement, per storage complex.
#[test]
fn the_2p5d_coupled_step_is_second_order_in_each_storage_complex() {
    let l2_rel = |a: &[f64], b: &[f64]| -> f64 {
        let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        let den: f64 = b.iter().map(|y| y * y).sum();
        (num / den.max(1e-300)).sqrt()
    };
    let run = |dt: f64, nsteps: usize| -> [Vec<f64>; 4] {
        let mut hier = primed(N, slip_spec());
        for _ in 0..nsteps {
            hier.step_root_with_dt(dt);
        }
        let (bf, bc, nrg, _pre) = hier.slip_state_snapshots(0);
        let ncells = nrg.len();
        [hier.density_snapshot(0), bf, bc[2 * ncells..3 * ncells].to_vec(), nrg]
    };
    let dt = 1.0e-3;
    let (u1, u2, u3) = (run(dt, 8), run(dt / 2.0, 16), run(dt / 4.0, 32));
    for (name, i) in [("density", 0), ("bface", 1), ("bz", 2), ("energy", 3)] {
        let (e1, e2) = (l2_rel(&u1[i], &u2[i]), l2_rel(&u2[i], &u3[i]));
        let ratio = e1 / e2.max(1e-300);
        println!("2.5D DMHMD {name:>8}: ratio {ratio:.2}");
        assert!(e2 > 1e-14, "vacuous coupled-order measurement in {name}");
        assert!(ratio > 3.4, "the 2.5D coupled step is not second order in {name}: ratio {ratio:.2}");
    }
}
