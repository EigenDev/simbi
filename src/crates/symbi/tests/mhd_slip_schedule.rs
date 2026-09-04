// =============================================================================
// mhd_slip_schedule.rs
//
// the palindromic coupled-step schedule for a slip-enabled draining body. the spy records each
// sub-operation's name and duration, proving the composition
//   D(dt/2) M(dt/2) H(dt) M(dt/2) D(dt/2)
// runs as one named operation with the drain and the magnetic slip each advancing a total dt, the
// ideal-MHD RK step H once, and no magnetic solve inside an RK stage (the RK stages are internal to
// the single H entry). numerical correctness is a separate test; this pins the schedule.
// =============================================================================

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet3D;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, MagneticSpec, SurfaceSpec};
use symbi_refinement::refinement::Hierarchy;
use symbi_refinement::refinement::hierarchy::{slip_schedule_arm, slip_schedule_take};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;

const N: usize = 12;
const GAMMA: f64 = 5.0 / 3.0;
const DT: f64 = 2.0e-3;
const BODY: [f64; 3] = [0.5, 0.5, 0.5];
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

fn build_slip_sim() -> Sim {
    let dx = 1.0 / N as f64;
    let k = 2.0 * std::f64::consts::PI;
    let a0 = 0.3;
    let sim = SimStateGeneric::<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>::build(
        NewtonianMhd,
        IdealGas { gamma: GAMMA },
        Cartesian,
    )
    .cells([N, N, N])
    .origin([0.0, 0.0, 0.0])
    .spacing([dx, dx, dx])
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .cfl(0.3)
    .allocate()
    .expect("schedule sim construction failed")
    .set_initial(|_| {
        MhdPrim::new(
            Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)),
            Tensor::new([0.0, 0.0, 0.0]),
        )
    })
    .seed_faces(move |axis, [x, y, _z]| match axis {
        0 => -a0 * (k * x).cos() * (k * y).sin(),
        1 => a0 * (k * x).sin() * (k * y).cos(),
        _ => 0.0,
    })
    .build();
    sim.with_bodies(
        BodyCollection::new().add(
            Body::black_hole(0, Tensor::new(BODY), Tensor::zeros(), 1.0, R_BODY, 0.05, 1.0, 1.0, R_BODY)
                .with_surface(SurfaceSpec::Drain)
                .with_magnetic(slip_spec()),
        ),
    )
}

fn build_drain_sim() -> Sim {
    // the same field/gas but a pure-drain body with no magnetic coupling, to isolate D's semigroup.
    let dx = 1.0 / N as f64;
    let k = 2.0 * std::f64::consts::PI;
    let a0 = 0.3;
    let sim = SimStateGeneric::<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>::build(
        NewtonianMhd,
        IdealGas { gamma: GAMMA },
        Cartesian,
    )
    .cells([N, N, N])
    .origin([0.0, 0.0, 0.0])
    .spacing([dx, dx, dx])
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .cfl(0.3)
    .allocate()
    .expect("drain sim construction failed")
    .set_initial(|_| {
        MhdPrim::new(
            Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)),
            Tensor::new([0.0, 0.0, 0.0]),
        )
    })
    .seed_faces(move |axis, [x, y, _z]| match axis {
        0 => -a0 * (k * x).cos() * (k * y).sin(),
        1 => a0 * (k * x).sin() * (k * y).cos(),
        _ => 0.0,
    })
    .build();
    sim.with_bodies(
        BodyCollection::new().add(
            Body::black_hole(0, Tensor::new(BODY), Tensor::zeros(), 1.0, R_BODY, 0.05, 1.0, 1.0, R_BODY)
                .with_surface(SurfaceSpec::Drain),
        ),
    )
}

fn max_rel_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / x.abs().max(1e-12))
        .fold(0.0_f64, f64::max)
}

#[test]
fn drain_half_steps_compose_to_second_order() {
    // D_{dt/2} o D_{dt/2} vs D_{dt}. the uniform drain preserves e_int and cs, so a frozen rate would
    // give an exact semigroup; the alfven term c_a2 = |B|^2/rho is recomputed from the drained state
    // and rises as rho drains, so the composition differs at O(dt^2). pins the pure-draining case and
    // that broadening beyond it needs a rate-freezing decision.
    let run = |dt: f64| -> f64 {
        // two half-drains with a rebuild between, vs one full drain.
        let sim_two = build_drain_sim();
        let sub_two = NewtonianMhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, 0.3, 1.0, &sim_two.geom.allocated);
        let mut two = Hierarchy::single(sim_two, sub_two);
        two.evolve(1.0e-9).expect("prime");

        let sim_one = build_drain_sim();
        let sub_one = NewtonianMhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, 0.3, 1.0, &sim_one.geom.allocated);
        let mut one = Hierarchy::single(sim_one, sub_one);
        one.evolve(1.0e-9).expect("prime");

        two.drain_and_rebuild(0, 0.5 * dt);
        two.drain_and_rebuild(0, 0.5 * dt);
        one.drain_and_rebuild(0, dt);
        max_rel_diff(&two.density_snapshot(0), &one.density_snapshot(0))
    };
    let e_full = run(DT);
    let e_half = run(0.5 * DT);
    assert!(e_full > 1e-9, "vacuous semigroup test (diff {e_full})");
    let ratio = e_full / e_half.max(1e-300);
    println!(
        "\ndrain semigroup:  |D_h/2 o D_h/2 - D_h| = {e_full:.3e} (dt)  {e_half:.3e} (dt/2)  ratio = {ratio:.2} (O(dt^2) -> ~4)\n"
    );
    assert!(ratio > 3.0, "drain composition error is not O(dt^2): ratio {ratio:.2}");
}

#[test]
fn slip_coupled_step_schedule_is_palindromic() {
    let sim = build_slip_sim();
    let sub = NewtonianMhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, 0.3, 1.0, &sim.geom.allocated);
    let mut hier = Hierarchy::single(sim, sub);
    // prime the state (ghosts, primitives, staggered B) with one warmup step of the normal path.
    hier.evolve(1.0e-9).expect("warmup prime");

    slip_schedule_arm();
    let failed = hier.advance_slip_coupled_step(0, DT, 1.0);
    let trace = slip_schedule_take().expect("schedule was armed");
    assert!(!failed, "the coupled step reported a failure (nonconverged solve?)");

    let ops: Vec<&str> = trace.iter().map(|(op, _)| *op).collect();
    assert_eq!(ops, ["D", "M", "H", "M", "D"], "coupled step is not palindromic D M H M D");

    let sum = |name: &str| -> f64 { trace.iter().filter(|(op, _)| *op == name).map(|(_, d)| *d).sum() };
    assert!((sum("D") - DT).abs() < 1e-15, "total drain duration != dt: {}", sum("D"));
    assert!((sum("M") - DT).abs() < 1e-15, "total magnetic-slip duration != dt: {}", sum("M"));
    assert!((sum("H") - DT).abs() < 1e-15, "ideal-MHD H duration != dt: {}", sum("H"));

    // no magnetic solve inside an RK stage: H is a single trace entry (its RK stages are internal),
    // and every M lies strictly before or after it -- there is no M between H's start and end.
    let h_pos = ops.iter().position(|o| *o == "H").unwrap();
    assert!(ops.iter().filter(|o| **o == "H").count() == 1, "H must appear once (one RK step)");
    assert!(
        ops[..h_pos].contains(&"M") && ops[h_pos + 1..].contains(&"M"),
        "each RK step must be bracketed by magnetic slip, never contain it"
    );
}
