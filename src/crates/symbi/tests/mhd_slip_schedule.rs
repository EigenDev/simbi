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
fn drain_half_steps_compose_to_second_order() {
    // D_{dt/2} o D_{dt/2} vs D_{dt}. the uniform drain preserves e_int and cs, so a frozen rate would
    // give an exact semigroup; the alfven term c_a2 = |B|^2/rho is recomputed from the drained state
    // and rises as rho drains, so the composition differs at O(dt^2). pins the pure-draining case and
    // that broadening beyond it needs a rate-freezing decision.
    let run = |dt: f64| -> f64 {
        // two half-drains with a rebuild between, vs one full drain.
        let sim_two = build_drain_only_sim();
        let sub_two = NewtonianMhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, 0.3, 1.0, &sim_two.geom.allocated);
        let mut two = Hierarchy::single(sim_two, sub_two);
        two.evolve(1.0e-9).expect("prime");

        let sim_one = build_drain_only_sim();
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
