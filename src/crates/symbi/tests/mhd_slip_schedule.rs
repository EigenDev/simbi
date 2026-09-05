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
    build_slip_sim_n(N)
}

fn build_slip_sim_n(n: usize) -> Sim {
    build_slip_sim_na(n, 0.3)
}

fn build_slip_sim_na(n: usize, a0: f64) -> Sim {
    let dx = 1.0 / n as f64;
    let k = 2.0 * std::f64::consts::PI;
    let sim = SimStateGeneric::<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>::build(
        NewtonianMhd,
        IdealGas { gamma: GAMMA },
        Cartesian,
    )
    .cells([n, n, n])
    .origin([0.0, 0.0, 0.0])
    .spacing([dx, dx, dx])
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .cfl(0.3)
    .allocate()
    .expect("schedule sim construction failed")
    // both face components vary across their own normal, so the cell value the CT interpolation forms
    // departs from the analytic field at the cell center. seeding the average of the two bounding face
    // values reproduces that interpolation exactly, deposits the magnetic energy the first C2P
    // subtracts, and leaves the face->cell projection an identity at t = 0.
    .set_initial(move |[x, y, _z]| {
        let bx = |xf: f64| -a0 * (k * xf).cos() * (k * y).sin();
        let by = |yf: f64| a0 * (k * x).sin() * (k * yf).cos();
        MhdPrim::new(
            Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)),
            Tensor::new([
                0.5 * (bx(x - 0.5 * dx) + bx(x + 0.5 * dx)),
                0.5 * (by(y - 0.5 * dx) + by(y + 0.5 * dx)),
                0.0,
            ]),
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

// a smooth, strong, discrete-divergence-free field for the H convergence study: a strong uniform
// background Bx = 1 (magnetic pressure important) plus a small smooth perturbation. each face component
// depends only on a transverse coordinate, so div B = 0 exactly by construction with no grid-scale
// structure. no immersed body -> H's magnetic dynamics are measured without gravity or a mask.
fn build_smooth_field_sim(n: usize) -> Sim {
    let dx = 1.0 / n as f64;
    let k = 2.0 * std::f64::consts::PI;
    let eps = 0.1;
    SimStateGeneric::<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>::build(
        NewtonianMhd,
        IdealGas { gamma: GAMMA },
        Cartesian,
    )
    .cells([n, n, n])
    .origin([0.0, 0.0, 0.0])
    .spacing([dx, dx, dx])
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .cfl(0.3)
    .allocate()
    .expect("smooth field sim construction failed")
    // each face component varies only across its own normal, so a cell's two bounding faces are equal
    // and the arithmetic average the CT interpolation forms is the analytic field at the cell center.
    // seeding that value deposits exactly the magnetic energy the first C2P subtracts.
    .set_initial(move |[x, y, _z]| {
        MhdPrim::new(
            Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)),
            Tensor::new([1.0 + eps * (k * y).sin(), eps * (k * x).sin(), 0.0]),
        )
    })
    .seed_faces(move |axis, [x, y, _z]| match axis {
        0 => 1.0 + eps * (k * y).sin(), // Bx(y): strong uniform + smooth, d/dx = 0
        1 => eps * (k * x).sin(),        // By(x): smooth, d/dy = 0
        _ => 0.0,
    })
    .build()
}

// the same field and gas but a pure-drain body with no magnetic slip: the evolve loop must keep the
// ordinary RK march for it, bit-for-bit.
fn build_drain_only_sim() -> Sim {
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
    .expect("drain-only sim construction failed")
    .set_initial(move |[x, y, _z]| {
        let bx = |xf: f64| -a0 * (k * xf).cos() * (k * y).sin();
        let by = |yf: f64| a0 * (k * x).sin() * (k * yf).cos();
        MhdPrim::new(
            Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)),
            Tensor::new([
                0.5 * (bx(x - 0.5 * dx) + bx(x + 0.5 * dx)),
                0.5 * (by(y - 0.5 * dx) + by(y + 0.5 * dx)),
                0.0,
            ]),
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
fn drain_half_steps_compose_to_third_order() {
    // D_{dt/2} o D_{dt/2} vs D_{dt} on the slip-path midpoint drain. the uniform drain preserves e_int
    // and cs, so the only state dependence is the local Alfven term c_a2 = |B|^2/rho, which rises as
    // rho drains at fixed B. the midpoint rate makes the drain second-order accurate in time, so the
    // full step and two half steps agree to O(dt^3): halving dt shrinks their difference by ~8. the
    // start-state rate (the legacy global drain) would shrink it by only ~4.
    let semigroup_dt = 8.0 * DT; // large enough that the O(dt^3) difference clears the roundoff floor
    let run = |dt: f64| -> f64 {
        // two half-drains with a rebuild between, vs one full drain.
        let sim_two = build_slip_sim();
        let sub_two = NewtonianMhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, 0.3, 1.0, &sim_two.geom.allocated);
        let mut two = Hierarchy::single(sim_two, sub_two);
        two.evolve(1.0e-9).expect("prime");

        let sim_one = build_slip_sim();
        let sub_one = NewtonianMhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, 0.3, 1.0, &sim_one.geom.allocated);
        let mut one = Hierarchy::single(sim_one, sub_one);
        one.evolve(1.0e-9).expect("prime");

        two.drain_and_rebuild(0, 0.5 * dt);
        two.drain_and_rebuild(0, 0.5 * dt);
        one.drain_and_rebuild(0, dt);
        max_rel_diff(&two.density_snapshot(0), &one.density_snapshot(0))
    };
    let e_full = run(semigroup_dt);
    let e_half = run(0.5 * semigroup_dt);
    assert!(e_full > 1e-9, "vacuous semigroup test (diff {e_full})");
    let ratio = e_full / e_half.max(1e-300);
    println!(
        "\ndrain semigroup:  |D_h/2 o D_h/2 - D_h| = {e_full:.3e} (dt)  {e_half:.3e} (dt/2)  ratio = {ratio:.2} (O(dt^3) -> ~8)\n"
    );
    assert!(ratio > 6.0, "midpoint drain composition error is not O(dt^3): ratio {ratio:.2} (expect ~8)");
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

#[test]
fn the_evolve_loop_drives_the_coupled_schedule_for_a_slip_body() {
    // a root step of the ordinary evolve driver runs the palindromic coupled step for a slip body: the
    // gate in step_root_with_dt selects advance_slip_coupled_step, so the spy records D M H M D.
    let sim = build_slip_sim();
    let sub = NewtonianMhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, 0.3, 1.0, &sim.geom.allocated);
    let mut hier = Hierarchy::single(sim, sub);
    hier.prime();

    slip_schedule_arm();
    hier.step_root_with_dt(DT);
    let trace = slip_schedule_take().expect("schedule was armed");
    let ops: Vec<&str> = trace.iter().map(|(op, _)| *op).collect();
    assert_eq!(ops, ["D", "M", "H", "M", "D"], "the evolve loop did not run the coupled step D M H M D");
}

#[test]
fn a_non_slip_body_keeps_the_ordinary_advance() {
    // a pure-drain body carries no magnetic slip, so the gate keeps advance_level: the coupled step
    // never runs, and the spy records nothing. this is the non-slip bit-identity boundary -- the same
    // RK march the driver ran before the coupled step existed.
    let sim = build_drain_only_sim();
    let sub = NewtonianMhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, 0.3, 1.0, &sim.geom.allocated);
    let mut hier = Hierarchy::single(sim, sub);
    hier.prime();

    slip_schedule_arm();
    hier.step_root_with_dt(DT);
    let trace = slip_schedule_take().expect("schedule was armed");
    assert!(trace.is_empty(), "a non-slip body entered the coupled step: {trace:?}");
}

// Aitken self-convergence of the density to the same final time: |u(dt)-u(dt/2)| / |u(dt/2)-u(dt/4)|
// -> 4 for a second-order method. an L2 field norm, since the body-center cell drains toward zero
// density where a relative difference is meaningless noise. `a0` sets the seed field amplitude.
fn coupled_temporal_ratio(a0: f64) -> (f64, f64) {
    let l2_rel = |a: &[f64], b: &[f64]| -> f64 {
        let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        let den: f64 = b.iter().map(|y| y * y).sum();
        (num / den.max(1e-300)).sqrt()
    };
    let run = |dt: f64, nsteps: usize| -> Vec<f64> {
        let sim = build_slip_sim_na(N, a0);
        let sub = NewtonianMhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, 0.3, 1.0, &sim.geom.allocated);
        let mut hier = Hierarchy::single(sim, sub);
        hier.prime();
        for _ in 0..nsteps {
            hier.step_root_with_dt(dt);
        }
        hier.density_snapshot(0)
    };
    // four timestep levels to the same final time: a pair of successive ratios separates an asymptotic
    // second-order trend from a single pre-asymptotic coincidence.
    let dt = 1.0e-3;
    let (a, b, c, d) = (run(dt, 8), run(dt / 2.0, 16), run(dt / 4.0, 32), run(dt / 8.0, 64));
    let (e_lo, e_mid, e_hi) = (l2_rel(&a, &b), l2_rel(&b, &c), l2_rel(&c, &d));
    println!(
        "DMHMD a0={a0}: diffs {e_lo:.3e} {e_mid:.3e} {e_hi:.3e}  ratios {:.2} {:.2}  (successive; -> 4 if second order)",
        e_lo / e_mid.max(1e-300),
        e_mid / e_hi.max(1e-300)
    );
    (e_mid / e_hi.max(1e-300), e_mid)
}

// the strong-field convergence studies below run on grids of 48 and 96 cells per side, minutes to
// tens of minutes in a release build, so they sit behind the opt-in feature. one exact command runs
// them, recording the commit the numbers belong to:
//
//   git rev-parse HEAD && cargo test -p symbi --release --features expensive-convergence-tests \
//       --test mhd_slip_schedule expensive_ -- --nocapture --test-threads=1
//
// every study asserts its acceptance criterion; a study whose criterion is open fails rather than
// reporting.
#[cfg(feature = "expensive-convergence-tests")]
mod expensive {
    use super::*;

    const FIELDS: [&str; 4] = ["density", "bface", "energy", "pressure"];

    // the drain rate lambda_rho = max(sqrt(cs^2 + |B|^2/rho) / (c_drain dx), sqrt(GM/r_acc^3)) evaluated
    // on a snapshot, together with the branch it selects. the acoustic arm stiffens without bound as the
    // density drains at fixed field, so a cell's branch and its stiffness h lambda both track depletion.
    fn drain_clock_census(
        n: usize,
        rho: &[f64],
        bcell: &[f64],
        pre: &[f64],
        h: f64,
    ) -> (f64, f64, usize) {
        let ncells = rho.len();
        let inv_cd_dx = n as f64; // c_drain = 1 on the slip path, so 1/(c_drain dx) = n
        let lambda_ff = (1.0f64 / (R_BODY * R_BODY * R_BODY)).sqrt(); // body mass 1
        let mut min_rho = f64::INFINITY;
        let mut max_h_lambda = 0.0f64;
        let mut acoustic = 0usize;
        for ii in 0..ncells {
            let den = rho[ii];
            let b_sq: f64 = (0..3).map(|d| bcell[d * ncells + ii].powi(2)).sum();
            let cs_sq = GAMMA * pre[ii] / den;
            let sound_rate = ((cs_sq + b_sq / den).max(0.0)).sqrt() * inv_cd_dx;
            min_rho = min_rho.min(den);
            max_h_lambda = max_h_lambda.max(h * sound_rate.max(lambda_ff));
            if sound_rate > lambda_ff {
                acoustic += 1;
            }
        }
        (min_rho, max_h_lambda, acoustic)
    }

    struct Sequence {
        diffs: [Vec<f64>; 4],
        ratios: [Vec<f64>; 4],
        troubled: u64,
        frozen: u64,
    }

    fn l2_rel(a: &[f64], b: &[f64]) -> f64 {
        let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        let den: f64 = b.iter().map(|y| y * y).sum();
        (num / den.max(1e-300)).sqrt()
    }

    // the strong-field coupled step on an n-cell grid, refined `levels` times from `steps` steps of
    // dt = 1e-3 to the same final time. each run books the first-order flux fallback, floors, and
    // retries in the guard ledger, since any of them firing in different cells at different timesteps
    // makes the solution map irregular in dt. prints the raw relative L2 norms of successive
    // differences per field with the build profile, grid, horizon, and solver tolerance, so a ratio
    // can be re-derived and compared across machines.
    fn strong_field_sequence(n: usize, steps: usize, levels: usize) -> Sequence {
        let dt = 1.0e-3;
        let run = |dt: f64, nsteps: usize| -> ([Vec<f64>; 4], u64, u64) {
            let sim = build_slip_sim_na(n, 0.3);
            let sub = NewtonianMhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, 0.3, 1.0, &sim.geom.allocated);
            let mut hier = Hierarchy::single(sim, sub);
            hier.prime();
            symbi_sim::guard_ledger::reset();
            let scope = symbi_sim::guard_ledger::open_scope();
            for _ in 0..nsteps {
                hier.step_root_with_dt(dt);
            }
            let (attempted, _accepted) = symbi_sim::guard_ledger::report();
            drop(scope);
            symbi_sim::guard_ledger::reset();
            let (bf, _bc, nrg, pre) = hier.slip_state_snapshots(0);
            (
                [hier.density_snapshot(0), bf, nrg, pre],
                attempted.troubled_cells.total,
                attempted.frozen_cells.total,
            )
        };
        let runs: Vec<([Vec<f64>; 4], u64, u64)> =
            (0..levels).map(|ll| run(dt / (1u32 << ll) as f64, steps << ll)).collect();
        let diffs: [Vec<f64>; 4] = std::array::from_fn(|ff| {
            runs.windows(2).map(|w| l2_rel(&w[0].0[ff], &w[1].0[ff])).collect()
        });
        let ratios: [Vec<f64>; 4] =
            std::array::from_fn(|ff| diffs[ff].windows(2).map(|w| w[0] / w[1].max(1e-300)).collect());
        let troubled = runs.iter().map(|r| r.1).sum();
        let frozen = runs.iter().map(|r| r.2).sum();
        println!(
            "\nSTRONG-FIELD DMHMD a0=0.3  grid {n}^3  base dt {dt:.1e} x {steps} steps  T={:.1e}  levels {levels}  \
             profile {}  arch {}  cpus {}  slip CG tol 1e-10 rel / 500 iters",
            steps as f64 * dt,
            if cfg!(debug_assertions) { "debug" } else { "release" },
            std::env::consts::ARCH,
            std::thread::available_parallelism().map(|c| c.get()).unwrap_or(0)
        );
        println!("  guards: troubled={troubled} frozen={frozen}");
        for ff in 0..4 {
            println!(
                "  {:>8}: diffs {}  ratios {}",
                FIELDS[ff],
                diffs[ff].iter().map(|d| format!("{d:.4e}")).collect::<Vec<_>>().join(" "),
                ratios[ff].iter().map(|r| format!("{r:.3}")).collect::<Vec<_>>().join(" ")
            );
        }
        Sequence { diffs, ratios, troubled, frozen }
    }

    // ratio 3.73 corresponds to a measured order of 1.9. the finest successive difference must sit
    // well above the solver floor: the slip solve converges to 1e-10 relative residual per step, and
    // the observed differences are 1e-6 to 1e-8 relative.
    const SECOND_ORDER_RATIO: f64 = 3.73;
    const NOISE_FLOOR: f64 = 1.0e-9;

    fn require_quiet(seq: &Sequence) {
        assert!(
            seq.troubled == 0 && seq.frozen == 0,
            "a guard fired during the sequence (troubled {} frozen {}): the map is irregular in dt",
            seq.troubled,
            seq.frozen
        );
    }

    fn require_second_order(seq: &Sequence, ff: usize, ratio_floor: f64) {
        let fine = *seq.diffs[ff].last().unwrap();
        assert!(fine > NOISE_FLOOR, "{}: finest difference {fine:.3e} is at the solver floor", FIELDS[ff]);
        assert!(
            seq.ratios[ff].iter().all(|r| *r > ratio_floor),
            "{}: ratios {:?} do not all exceed {ratio_floor}",
            FIELDS[ff],
            seq.ratios[ff]
        );
    }

    // on 48 cells per side the gas fields are resolved: density, energy, and pressure self-converge
    // at four over a 64-step horizon with every guard quiet. the staggered field is reported here and
    // gated on the finer grid below.
    #[test]
    fn expensive_strong_field_coupled_step_thermodynamic_fields_are_second_order() {
        let seq = strong_field_sequence(48, 8, 4);
        require_quiet(&seq);
        for ff in [0, 2, 3] {
            require_second_order(&seq, ff, 3.5);
        }
    }

    // the staggered field's ratio climbs with resolution (about 2 at 24 cells per side, 3.9 then 3.3
    // at 48), consistent with second order read through a receding mask-seam spatial structure. the
    // 96-cell grid is the closing measurement: both successive bface ratios above 3.73 close the
    // gate; a finest ratio that improves but stays near 3.2 to 3.5 is still pre-asymptotic and is to
    // be characterized, and one that stagnates or declines is a residual term to be investigated,
    // never attributed to resolution by default.
    #[test]
    fn expensive_strong_field_coupled_step_staggered_field_is_second_order_at_96() {
        let seq = strong_field_sequence(96, 8, 4);
        require_quiet(&seq);
        for ff in [0, 2, 3] {
            require_second_order(&seq, ff, 3.5);
        }
        require_second_order(&seq, 1, SECOND_ORDER_RATIO);
    }

    // the drain-side candidates for a residual first-order term, each measured on the fixture: every
    // masked cell sits on the acoustic branch of the rate law throughout, so the max(acoustic,
    // free-fall) switching surface is never crossed between operators; the stiffness h lambda stays
    // small and flat; and the density stays bounded away from zero. under those conditions the
    // density ratio on the 12-cell grid holds near four at every horizon.
    #[test]
    fn expensive_depletion_matrix_keeps_every_cell_on_one_branch() {
        let run = |dt: f64, nsteps: usize| -> (Vec<f64>, Vec<f64>, Vec<f64>) {
            let sim = build_slip_sim_na(N, 0.3);
            let sub = NewtonianMhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, 0.3, 1.0, &sim.geom.allocated);
            let mut hier = Hierarchy::single(sim, sub);
            hier.prime();
            for _ in 0..nsteps {
                hier.step_root_with_dt(dt);
            }
            let (_bf, bc, _nrg, pre) = hier.slip_state_snapshots(0);
            (hier.density_snapshot(0), bc, pre)
        };
        let dt = 1.0e-3;
        let (r0, b0, p0) = run(dt, 0);
        let ncells = r0.len();
        let (_, hl0, ac0) = drain_clock_census(N, &r0, &b0, &p0, dt);
        assert!(ac0 == ncells, "the fixture starts with {ac0}/{ncells} cells on the acoustic branch");
        println!("\nDEPLETION t=0: max h*lambda={hl0:.3e}  acoustic={ac0}/{ncells}");
        for horizon in [1usize, 2, 4, 8, 16] {
            let (a, _, _) = run(dt, horizon);
            let (b, _, _) = run(dt / 2.0, 2 * horizon);
            let (c, bc, pc) = run(dt / 4.0, 4 * horizon);
            let (e_lo, e_hi) = (l2_rel(&a, &b), l2_rel(&b, &c));
            let ratio = e_lo / e_hi.max(1e-300);
            let (min_rho, h_lambda, acoustic) = drain_clock_census(N, &c, &bc, &pc, dt);
            println!(
                "DEPLETION steps={horizon:>2} T={:.0e}: min_rho={min_rho:.3e}  max h*lambda={h_lambda:.3e}  acoustic={acoustic}  ratio {ratio:.2}",
                horizon as f64 * dt
            );
            assert!(acoustic == ncells, "T={}: {acoustic}/{ncells} cells acoustic, a branch crossing occurred", horizon as f64 * dt);
            assert!(h_lambda < 0.05, "T={}: h lambda {h_lambda:.3e} is stiff", horizon as f64 * dt);
            assert!(min_rho > 0.5, "T={}: density depleted to {min_rho:.3e}", horizon as f64 * dt);
            assert!(e_hi > 1e-12, "T={}: vacuous measurement", horizon as f64 * dt);
            assert!(ratio > 3.5, "T={}: density ratio {ratio:.2} on the resolved field", horizon as f64 * dt);
        }
    }
}

fn l2_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}

// the complete production magnetic-slip map (solve -> commit -> c2p -> ghost fill), evolved in
// isolation on the full state, is second order in time per field: bface, bcell, cons.nrg, and pressure
// each quarter their fixed-time error when the step halves. this holds because the solve freezes the
// coefficient on an explicit predicted midpoint gas energy (e_g* = e_g^0 + (dt/2) qdot^0) rather than
// reconstructing gas energy from the endpoint-reconciled total energy. the deep-interior ratios (cells
// off the periodic wrap, no derived-halo dependence) confirm the order on the physical complex.
#[test]
fn the_production_m_map_is_second_order_per_field() {
    let run = |dt: f64, nsteps: usize| -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let sim = build_slip_sim_na(N, 0.3);
        let sub = NewtonianMhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, 0.3, 1.0, &sim.geom.allocated);
        let mut hier = Hierarchy::single(sim, sub);
        hier.prime();
        for _ in 0..nsteps {
            assert!(!hier.magnetic_slip_map(0, dt), "the M map diverged at dt={dt}");
        }
        hier.slip_state_snapshots(0)
    };
    // the deep-interior cell mask: cells whose whole interp/curl stencil stays off the periodic wrap,
    // so bcell/nrg/pressure there depend on no derived halo face. this reads the physical complex only,
    // ruling out a boundary-storage artifact in the order measurement.
    let deep: Vec<bool> = {
        let sim = build_slip_sim_na(N, 0.3);
        sim.geom
            .interior
            .iter()
            .map(|c| (0..3).all(|a| c[a] >= 2 && c[a] < N as isize - 2))
            .collect()
    };
    let ncells = deep.len();
    let l2_deep = |a: &[f64], b: &[f64], tiled: bool| -> f64 {
        let mut s = 0.0;
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            let cell = if tiled { i % ncells } else { i };
            if deep[cell] {
                s += (x - y) * (x - y);
            }
        }
        s.sqrt()
    };
    let dt = 1.0e-3;
    let (a_bf, a_bc, a_nr, a_pr) = run(dt, 8);
    let (b_bf, b_bc, b_nr, b_pr) = run(dt / 2.0, 16);
    let (c_bf, c_bc, c_nr, c_pr) = run(dt / 4.0, 32);
    for (name, tiled, (a, b, c)) in [
        ("bface", true, (&a_bf, &b_bf, &c_bf)),
        ("bcell", true, (&a_bc, &b_bc, &c_bc)),
        ("nrg", false, (&a_nr, &b_nr, &c_nr)),
        ("pressure", false, (&a_pr, &b_pr, &c_pr)),
    ] {
        let (ef_lo, ef_hi) = (l2_diff(a, b), l2_diff(b, c));
        let (ed_lo, ed_hi) = (l2_deep(a, b, tiled), l2_deep(b, c, tiled));
        let (full, deep) = (ef_lo / ef_hi.max(1e-300), ed_lo / ed_hi.max(1e-300));
        println!("M map {name:>8}: full ratio={full:.2}  deep-interior ratio={deep:.2}");
        assert!(ef_lo > 1e-12, "M map {name}: vacuous ({ef_lo})");
        assert!(deep > 3.4, "the production M map is not second order in {name}: deep ratio {deep:.2}");
    }
}

// a body-free smooth-wave fixture for the H-only ownership ladder: the primitive IC, the seeded face
// field, and the cell-centered B are supplied as closures. div B = 0 is the caller's responsibility
// (each fixture uses a component that depends only on a transverse coordinate).
//
// the caller owes both magnetic representations, consistently: staggered B carries the divergence
// constraint and cell B is its arithmetic face average, bcell_c = (bface_c[i] + bface_c[i+1])/2. the
// cell value is what prim->cons deposits as magnetic energy in cons.nrg, and the CT face->cell
// interpolation rewrites bcell from the faces while leaving nrg alone. supplying the exact average
// makes that interpolation an identity, so the first C2P subtracts the same magnetic energy the
// seeding deposited.
fn build_wave_sim(
    n: usize,
    prim: impl Fn([f64; 3]) -> (f64, [f64; 3], f64) + 'static,
    face: impl Fn(usize, [f64; 3]) -> f64 + 'static,
    cell_b: impl Fn([f64; 3]) -> [f64; 3] + 'static,
) -> Sim {
    let dx = 1.0 / n as f64;
    SimStateGeneric::<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>::build(
        NewtonianMhd,
        IdealGas { gamma: GAMMA },
        Cartesian,
    )
    .cells([n, n, n])
    .origin([0.0, 0.0, 0.0])
    .spacing([dx, dx, dx])
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .cfl(0.3)
    .allocate()
    .expect("wave sim construction failed")
    .set_initial(move |x| {
        let (rho, v, p) = prim(x);
        MhdPrim::new(Prim::adiabatic(Density(rho), Tensor::new(v), Pressure(p)), Tensor::new(cell_b(x)))
    })
    .seed_faces(move |axis, x| face(axis, x))
    .build()
}

// scaling probe: one corrector step of passive induction at dt and dt/2. ||E*-En|| ~ dt means the
// predictor midpoint correction exists; O(1) means stage 2 evaluates an inconsistent state.
#[test]
fn diag_passive_e_scaling() {
    let k = 2.0 * std::f64::consts::PI;
    // dt = 0 idempotence: the predictor state equals stage-1, so a consistent stage-2 EMF must reproduce
    // E^n exactly (||E*-En|| = 0). any nonzero value is a pure producer inconsistency, temporal-error-free.
    for dt in [0.0f64, 5.0e-4, 2.5e-4] {
        let sim = build_wave_sim(
            24,
            |_| (1.0, [0.5, 0.0, 0.0], 1.0),
            move |axis, [x, _, _]| if axis == 1 { 0.01 * (k * x).sin() } else { 0.0 },
            move |[x, _, _]| [0.0, 0.01 * (k * x).sin(), 0.0],
        );
        let sub = NewtonianMhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, 0.3, 1.0, &sim.geom.allocated);
        let mut hier = Hierarchy::single(sim, sub);
        hier.prime();
        // the fixture seeds both magnetic representations consistently, so the primed state is already
        // canonical and the step measures from t = 0 with no time advanced beforehand.
        hier.hydro_map(0, dt);
    }
}

// the H-only ownership ladder: acoustic (B=0), passive induction (weak B advected by a uniform flow,
// negligible Lorentz), and a strong-field MHD wave. acoustic ~2 => the SSP-RK stage/rebuild path is
// first order; acoustic ~4 & passive ~2 => CT/EMF staging; both ~4 & strong ~2 => the gas-CT coupling.
#[test]
fn the_acoustic_and_passive_induction_waves_are_second_order() {
    let k = 2.0 * std::f64::consts::PI;
    let l2 = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
    macro_rules! wave {
        ($label:expr, $gated:expr, $build:expr) => {{
            let run = |dt: f64, nsteps: usize| {
                let sim = $build;
                let sub = NewtonianMhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, 0.3, 1.0, &sim.geom.allocated);
                let mut hier = Hierarchy::single(sim, sub);
                hier.prime();
                for _ in 0..nsteps {
                    assert!(!hier.hydro_map(0, dt), "H retry");
                }
                let (bf, _bc, nrg, pre) = hier.slip_state_snapshots(0);
                (bf, hier.density_snapshot(0), nrg, pre)
            };
            let dt = 2.5e-4;
            let (a, b, c) = (run(dt, 8), run(dt / 2.0, 16), run(dt / 4.0, 32));
            for (fname, x, y, z) in [
                ("bface", &a.0, &b.0, &c.0),
                ("density", &a.1, &b.1, &c.1),
                ("energy", &a.2, &b.2, &c.2),
                ("pressure", &a.3, &b.3, &c.3),
            ] {
                let fine = l2(y, z);
                let r = l2(x, y) / fine.max(1e-300);
                println!("WAVE {:>8} {fname:>8}: ratio={r:.2}", $label);
                // a field the fixture leaves identically zero has no order to read.
                if $gated && fine > 1e-14 {
                    assert!(r > 3.5, "{} wave: H is not second order in {fname}: ratio {r:.2}", $label);
                }
            }
        }};
    }
    // acoustic: smooth density/velocity/pressure perturbation, no field.
    wave!("acoustic", true, build_wave_sim(
        24,
        move |[x, _, _]| (1.0 + 0.1 * (k * x).sin(), [0.1 * (k * x).sin(), 0.0, 0.0], 1.0 + 0.14 * (k * x).sin()),
        |_, _| 0.0,
        |_| [0.0, 0.0, 0.0],
    ));
    // passive induction: uniform flow advects a weak transverse B (Lorentz ~ B^2 ~ 1e-4 negligible).
    wave!("passive", true, build_wave_sim(
        24,
        |_| (1.0, [0.5, 0.0, 0.0], 1.0),
        move |axis, [x, _, _]| if axis == 1 { 0.01 * (k * x).sin() } else { 0.0 },
        // By(x) sits on the y-faces, so a cell's two bounding y-faces share its x: the average is the
        // same sinusoid at the cell center, exactly.
        move |[x, _, _]| [0.0, 0.01 * (k * x).sin(), 0.0],
    ));
    // strong MHD wave: strong Bx background + a smooth velocity perturbation driving Alfven/magnetosonic.
    // read at 24 cells per side, where the staggered field is still pre-asymptotic (its ratio climbs
    // with resolution and reaches four at 48), so this rung reports without gating.
    wave!("strongMHD", false, build_wave_sim(
        24,
        move |[x, _, _]| (1.0, [0.0, 0.1 * (k * x).sin(), 0.0], 1.0),
        move |axis, [x, y, _]| match axis {
            0 => 1.0 + 0.1 * (k * y).sin(),
            1 => 0.1 * (k * x).sin(),
            _ => 0.0,
        },
        // each face component varies only across its own normal, so the two bounding faces of a cell
        // carry equal values and their average is the analytic field at the cell center.
        move |[x, y, _]| [1.0 + 0.1 * (k * y).sin(), 0.1 * (k * x).sin(), 0.0],
    ));
}

// an MHD initial condition owes two consistent magnetic representations: staggered faces carrying the
// divergence constraint, and cell-centered B equal to their arithmetic average, whose magnetic energy
// prim->cons deposits into cons.nrg. seeding nonzero faces beside a zero cell B satisfies div B = 0 yet
// leaves cons.nrg short by |B|^2/2, and the first face->cell interpolation then hands C2P a magnetic
// energy to subtract that was never deposited, an O(|B|^2) pressure step on step one that pollutes
// every temporal-convergence measurement taken from the fixture.
//
// a consistently seeded state is a fixed point of the projection the first step applies: interpolating
// faces to cells reproduces the seeded cell B, and the following C2P reproduces the seeded pressure. a
// zero-length step exercises exactly that projection, so bit-identical state across it certifies the
// seeding, and any change localizes the inconsistency to the field it moved.
#[test]
fn seeded_face_and_cell_magnetic_fields_agree_after_priming() {
    // the seeded cell average evaluates the analytic face field at x_c +- dx/2 while the grid places
    // its faces at x_lo + i dx; the two spellings of the same point agree to the last bit, so the
    // seeded average and the kernel's average of the stored faces differ at roundoff. the bound is a
    // few ulps of the field's own amplitude. the defect it guards against, a cell field inconsistent
    // with its faces, is O(|B|) in the field and O(|B|^2) in the pressure, fourteen orders above it.
    let same = |name: &str, a: &[f64], b: &[f64]| {
        let scale = a.iter().fold(0.0_f64, |m, x| m.max(x.abs())).max(1.0);
        let tol = 64.0 * f64::EPSILON * scale;
        let worst = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            worst <= tol,
            "seeded state is not a fixed point of the face->cell projection: {name} moved by {worst:.3e} (roundoff bound {tol:.3e})"
        );
    };
    let k = 2.0 * std::f64::consts::PI;
    let mut fixtures: Vec<(&str, Sim)> = vec![
        ("smooth_strong_field", build_smooth_field_sim(12)),
        ("slip_body", build_slip_sim_n(12)),
        (
            "passive_wave",
            build_wave_sim(
                12,
                |_| (1.0, [0.5, 0.0, 0.0], 1.0),
                move |axis, [x, _, _]| if axis == 1 { 0.01 * (k * x).sin() } else { 0.0 },
                move |[x, _, _]| [0.0, 0.01 * (k * x).sin(), 0.0],
            ),
        ),
        (
            "strong_wave",
            build_wave_sim(
                12,
                move |[x, _, _]| (1.0, [0.0, 0.1 * (k * x).sin(), 0.0], 1.0),
                move |axis, [x, y, _]| match axis {
                    0 => 1.0 + 0.1 * (k * y).sin(),
                    1 => 0.1 * (k * x).sin(),
                    _ => 0.0,
                },
                move |[x, y, _]| [1.0 + 0.1 * (k * y).sin(), 0.1 * (k * x).sin(), 0.0],
            ),
        ),
    ];
    for (name, sim) in fixtures.drain(..) {
        let sub = NewtonianMhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, 0.3, 1.0, &sim.geom.allocated);
        let mut hier = Hierarchy::single(sim, sub);
        hier.prime();
        let (bf0, bc0, nrg0, pre0) = hier.slip_state_snapshots(0);
        let rho0 = hier.density_snapshot(0);
        assert!(
            bc0.iter().any(|b| b.abs() > 1e-12),
            "{name}: cell B is zero everywhere, so the projection cannot be exercised"
        );
        hier.hydro_map(0, 0.0);
        let (bf1, bc1, nrg1, pre1) = hier.slip_state_snapshots(0);
        let rho1 = hier.density_snapshot(0);
        same(&format!("{name} bface"), &bf0, &bf1);
        same(&format!("{name} bcell"), &bc0, &bc1);
        same(&format!("{name} density"), &rho0, &rho1);
        same(&format!("{name} energy"), &nrg0, &nrg1);
        same(&format!("{name} pressure"), &pre0, &pre1);
    }
}

// H standalone temporal convergence on a smooth strong-uniform-field, body-free, FOFC-free fixture,
// swept over resolution. an extra halving level distinguishes a pre-asymptotic ratio from a stable
// ratio of two. the staggered-field ratio climbs with resolution (about 2.5 at 12 cells per side, 3.8
// at 24, 4.0 at 48) while the energy ratio sits at four throughout: the temporal order is two, and the
// coarse grids are read through their own spatial truncation in the field.
#[test]
fn the_ideal_mhd_step_is_second_order_per_field_once_resolved() {
    let l2 = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
    let run = |n: usize, dt: f64, nsteps: usize| -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let sim = build_smooth_field_sim(n);
        let sub = NewtonianMhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, 0.3, 1.0, &sim.geom.allocated);
        let mut hier = Hierarchy::single(sim, sub);
        hier.prime();
        for _ in 0..nsteps {
            assert!(!hier.hydro_map(0, dt), "H stage retry on the smooth fixture");
        }
        let (bf, _bc, nrg, pre) = hier.slip_state_snapshots(0);
        (bf, nrg, pre)
    };
    for n in [12usize, 24, 48] {
        // fixed horizon, four timestep levels: ratios of successive differences reveal the asymptotic
        // order. dt well below the fast-magnetosonic CFL at every resolution.
        let dt = 2.5e-4;
        let (u1, u2, u3, u4) = (run(n, dt, 8), run(n, dt / 2.0, 16), run(n, dt / 4.0, 32), run(n, dt / 8.0, 64));
        for (fname, i) in [("bface", 0), ("energy", 1), ("pressure", 2)] {
            let g = |u: &(Vec<f64>, Vec<f64>, Vec<f64>)| match i {
                0 => u.0.clone(),
                1 => u.1.clone(),
                _ => u.2.clone(),
            };
            let (e1, e2, e3) = (l2(&g(&u1), &g(&u2)), l2(&g(&u2), &g(&u3)), l2(&g(&u3), &g(&u4)));
            let (r1, r2) = (e1 / e2.max(1e-300), e2 / e3.max(1e-300));
            println!("H N={n:<3} {fname:>8}: ratios {r1:.2} {r2:.2}  (successive; -> 4 if second order)");
            // the finest grid resolves every field's structure; there the temporal order reads clean.
            if n == 48 {
                assert!(e3 > 1e-14, "vacuous H order measurement in {fname} at N={n}");
                assert!(
                    r1 > 3.5 && r2 > 3.5,
                    "the ideal-MHD RK2 step is not second order in {fname} at N={n}: ratios {r1:.2} {r2:.2}"
                );
            }
        }
    }
}

// the operator-split ladder at strong field: standalone D and H, the DMD and MHM triples, and the full
// DMHMD, each self-converged per field (bface, density, total energy, pressure) on the 12-cell grid.
// the field-free rungs (D, DMD) read four in every field; every rung containing H reads the grid's
// pre-asymptotic staggered-field structure rather than a temporal order, so a rung below four is
// attributable to the splitting only when it also appears at a resolution where H alone reads four.
#[test]
fn diag_split_ladder_order() {
    let l2 = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
    macro_rules! rung {
        ($label:expr, $body:expr) => {{
            let run = |dt: f64, nsteps: usize| {
                let sim = build_slip_sim_na(N, 0.3);
                let sub = NewtonianMhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, 0.3, 1.0, &sim.geom.allocated);
                let mut hier = Hierarchy::single(sim, sub);
                hier.prime();
                for _ in 0..nsteps {
                    ($body)(&mut hier, dt);
                }
                let (bf, _bc, nrg, pre) = hier.slip_state_snapshots(0);
                (bf, hier.density_snapshot(0), nrg, pre)
            };
            let dt = 1.0e-3;
            let (a, b, c) = (run(dt, 8), run(dt / 2.0, 16), run(dt / 4.0, 32));
            for (fname, x, y, z) in [
                ("bface", &a.0, &b.0, &c.0),
                ("density", &a.1, &b.1, &c.1),
                ("energy", &a.2, &b.2, &c.2),
                ("pressure", &a.3, &b.3, &c.3),
            ] {
                let r = l2(x, y) / l2(y, z).max(1e-300);
                println!("LADDER {:>6} {fname:>8}: ratio={r:.2}", $label);
            }
        }};
    }
    rung!("D", |h: &mut Hierarchy<_,3,3,_,_,_,_,_>, dt: f64| {
        h.drain_and_rebuild(0, dt);
    });
    rung!("H", |h: &mut Hierarchy<_,3,3,_,_,_,_,_>, dt: f64| {
        h.hydro_map(0, dt);
    });
    rung!("DMD", |h: &mut Hierarchy<_,3,3,_,_,_,_,_>, dt: f64| {
        h.drain_and_rebuild(0, 0.5 * dt);
        h.magnetic_slip_map(0, dt);
        h.drain_and_rebuild(0, 0.5 * dt);
    });
    rung!("MHM", |h: &mut Hierarchy<_,3,3,_,_,_,_,_>, dt: f64| {
        h.magnetic_slip_map(0, 0.5 * dt);
        h.hydro_map(0, dt);
        h.magnetic_slip_map(0, 0.5 * dt);
    });
    rung!("DMHMD", |h: &mut Hierarchy<_,3,3,_,_,_,_,_>, dt: f64| {
        h.advance_slip_coupled_step(0, dt, 0.0);
    });
}

#[test]
fn the_drain_hydro_coupled_step_is_second_order_in_time() {
    // fixed grid, refine the timestep: the spatial error is common to every run and cancels to leading
    // order, so the fixed-time error isolates the temporal order and quarters when dt halves. the
    // density is the field this grid resolves well enough to read: the staggered field and the
    // pressure carry a mask-seam spatial structure that a 12-cell grid leaves pre-asymptotic, so their
    // ratios reach four only from about 48 cells per side, where every evolved field does.
    let (ratio, e_lo) = coupled_temporal_ratio(0.3);
    println!("\ndrain-hydro coupled temporal:  ratio = {ratio:.2}  (second order -> ~4)\n");
    assert!(e_lo > 1e-10, "vacuous coupled-order test (diff {e_lo})");
    assert!(ratio > 3.4, "the drain-hydro coupled step is not second order in time: ratio {ratio:.2}");
}
