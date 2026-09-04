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
