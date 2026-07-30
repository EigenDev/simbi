// =============================================================================
// constraint_projection_oracle.rs
//
// the trace-carrier oracle for the state-constraint projection.
//
// WHAT THIS DOES NOT TEST, and why that matters for reading it: the projection is NOT reimplemented
// in the kernel. `constraint_projection_gv` calls the same `symbi_hydro::constraints` functions the
// host unit gates exercise, instantiated at `S = Gv` — one implementation, two carriers, exactly as
// `admissible_theta` is already shared. so "does the traced algebra agree with the host algebra" is
// true by construction and is not a claim worth a test.
//
// WHAT THIS TESTS is the WIRING around the algebra, which is genuinely a second implementation and
// is reachable only through a trace-and-execute round trip:
//   - the anchor and the candidate are not swapped (an inverted segment still produces a plausible
//     theta in [0, 1], so a magnitude-only check on a symmetric case would pass);
//   - each field reaches the slot its residual expects;
//   - the stored covariant energy is mapped to the eulerian energy the admissible set is defined on
//     (`E = (ehat + D + beta^i S_i) / alpha`), not passed through raw;
//   - the ATTRIBUTION survives: `binding` names the member that actually bound. two different
//     members can produce the same joint theta, so a scalar comparison alone can pass while the
//     ledger silently charges the wrong constraint.
//
// SCOPE: the members are exercised on a flat background, which is where the wiring is isolated from
// metric evaluation (itself covered by the carrier oracles for the metric kernels). every member of
// the family applies on a magnetized energy regime, so the structurally-inapplicable `None` path is
// not reachable through THIS kernel and is gated host-side instead; an isothermal or unmagnetized
// projection kernel would need its own case here.
//
// usage:
//   cargo test -p symbi-discretize --test constraint_projection_oracle
// =============================================================================

mod harness;

use harness::KernelRun;
use symbi_algebra::{Matrix, Tensor};
use symbi_discretize::gv::constraint_projection_gv;
use symbi_discretize::{Coords, Spacetime, Spacing};
use symbi_hydro::admissible::{ADMISSIBLE_REL_FLOOR, rmhd_state_scale};
use symbi_hydro::constraints::{
    ConstraintState, DensityFloor, MagnetizationCeiling, StateConstraint, TemperatureFloor,
    WuTangAdmissibility, constraint_thetas, joint_theta,
};

/// one cell's scenario: an admissible anchor and a candidate that some member should reject.
#[derive(Clone, Copy)]
struct Case {
    what: &'static str,
    a_den: f64,
    a_mom: [f64; 3],
    a_nrg: f64,
    x_den: f64,
    x_mom: [f64; 3],
    x_nrg: f64,
    b: [f64; 3],
}

const F_MIN: f64 = 0.1;
const SIGMA_MAX: f64 = 4.0;
const DEN_MIN: f64 = 0.05;

/// the family indices, in the order `constraint_projection_gv` declares them. the checks below
/// assert against these, so a reordering in the kernel that silently remapped the ledger fails.
const IDX_ADMISSIBILITY: f64 = 0.0;
const IDX_TEMPERATURE: f64 = 1.0;
const IDX_MAGNETIZATION: f64 = 2.0;
const IDX_DENSITY: f64 = 3.0;

fn cases() -> Vec<Case> {
    vec![
        // nothing binds: every member slack. theta must be exactly 1 and nothing may be charged.
        Case {
            what: "all slack",
            a_den: 1.0,
            a_mom: [0.0; 3],
            a_nrg: 2.0,
            x_den: 1.0,
            x_mom: [0.01, 0.0, 0.0],
            x_nrg: 2.0,
            b: [0.1, 0.0, 0.0],
        },
        // the TEMPERATURE floor binds, and on a flat background with zero momentum the crossing is
        // analytic, so this case checks the traced kernel against the closed form rather than
        // merely against the host — a bug shared by both would survive a host-vs-trace comparison.
        Case {
            what: "temperature floor binds",
            a_den: 1.0,
            a_mom: [0.0; 3],
            a_nrg: 2.0,
            x_den: 1.0,
            x_mom: [0.0; 3],
            x_nrg: -0.5,
            b: [0.0; 3],
        },
        // the DENSITY floor binds: a different member, so the attribution must move with it.
        Case {
            what: "density floor binds",
            a_den: 1.0,
            a_mom: [0.0; 3],
            a_nrg: 3.0,
            x_den: 0.0,
            x_mom: [0.0; 3],
            x_nrg: 3.0,
            b: [0.0; 3],
        },
        // the MAGNETIZATION ceiling binds: density falls at fixed field, so sigma rises.
        Case {
            what: "magnetization ceiling binds",
            a_den: 1.0,
            a_mom: [0.0; 3],
            a_nrg: 4.0,
            x_den: 0.2,
            x_mom: [0.0; 3],
            x_nrg: 4.0,
            b: [1.0, 0.0, 0.0],
        },
        // ADMISSIBILITY binds — the one member whose crossing is BISECTED rather than closed-form,
        // so it is the path with the most room for the trace to diverge. the candidate drives the
        // momentum past what the energy can carry, which no affine member notices.
        Case {
            what: "admissibility binds",
            a_den: 1.0,
            a_mom: [0.0; 3],
            a_nrg: 3.0,
            x_den: 1.0,
            x_mom: [8.0, 0.0, 0.0],
            x_nrg: 3.0,
            b: [0.05, 0.0, 0.0],
        },
    ]
}

/// the HOST evaluation of the same family, on the same flat background the kernel traces.
fn host(case: &Case) -> (f64, f64) {
    let (gm, gi) = (Matrix::<f64, 3>::identity(), Matrix::<f64, 3>::identity());
    let a_mom = Tensor::new(case.a_mom);
    let x_mom = Tensor::new(case.x_mom);
    let b = Tensor::new(case.b);
    let blend = |t: f64| ConstraintState {
        den: case.a_den + t * (case.x_den - case.a_den),
        mom: Tensor::new(std::array::from_fn(|k| {
            case.a_mom[k] + t * (case.x_mom[k] - case.a_mom[k])
        })),
        // flat: alpha = 1, beta = 0, so the eulerian energy is ehat + D.
        nrg: Some(
            (case.a_nrg + t * (case.x_nrg - case.a_nrg))
                + (case.a_den + t * (case.x_den - case.a_den)),
        ),
        mag: Some(b),
        gm: &gm,
        gm_inv: &gi,
    };
    let scale = rmhd_state_scale(case.a_den, &a_mom, case.a_nrg + case.a_den, &b, &gi, &gm);
    let _ = x_mom;
    let g = WuTangAdmissibility {
        eps_d: 1e-12 * scale,
        eps_q: ADMISSIBLE_REL_FLOOR * scale,
        eps_psi: ADMISSIBLE_REL_FLOOR * scale * scale.sqrt(),
    };
    let temperature = TemperatureFloor { f_min: F_MIN };
    let magnetization = MagnetizationCeiling {
        sigma_max: SIGMA_MAX,
    };
    let density = DensityFloor { den_min: DEN_MIN };
    let family: Vec<&dyn StateConstraint<f64>> = vec![&g, &temperature, &magnetization, &density];
    let thetas = constraint_thetas(&family, &blend, 20);
    let theta = joint_theta(&thetas);
    let mut binding = -1.0;
    let mut best = 1.0;
    for (index, candidate) in thetas.iter().enumerate() {
        if let Some(tk) = candidate
            && *tk < best
        {
            binding = index as f64;
            best = *tk;
        }
    }
    (theta, binding)
}

fn run_kernel(cases: &[Case]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = cases.len();
    let owned: Vec<Case> = cases.to_vec();
    let pick = |f: fn(&Case) -> f64| {
        let owned = owned.clone();
        move |c: &[usize]| f(&owned[c[0]])
    };
    let pick_k = |f: fn(&Case, usize) -> f64, k: usize| {
        let owned = owned.clone();
        move |c: &[usize]| f(&owned[c[0]], k)
    };
    let mut k = KernelRun::new(constraint_projection_gv(
        Coords::Cartesian,
        Spacetime::Minkowski,
        &[Spacing::Uniform; 3],
        &[0, 1, 2],
    ))
    .grid([n, 1, 1])
    .scalars(&[
        ("floor_temperature", F_MIN),
        ("ceiling_magnetization", SIGMA_MAX),
        ("floor_density", DEN_MIN),
        // declared by the shared metric prologue; the flat branch ignores it.
        ("schwarzschild_mass", 0.0),
        // the cell position is traced even on a flat chart, where the identity metric ignores it.
        ("x_lo_0", 0.0),
        ("x_lo_1", 0.0),
        ("x_lo_2", 0.0),
        ("dx_0", 1.0),
        ("dx_1", 1.0),
        ("dx_2", 1.0),
    ])
    .field_with("a_den", pick(|c| c.a_den))
    .field_with("a_nrg", pick(|c| c.a_nrg))
    .field_with("x_den", pick(|c| c.x_den))
    .field_with("x_nrg", pick(|c| c.x_nrg));
    for idx in 0..3 {
        k = k
            .field_with(&format!("a_mom_{idx}"), pick_k(|c, k| c.a_mom[k], idx))
            .field_with(&format!("x_mom_{idx}"), pick_k(|c, k| c.x_mom[k], idx))
            .field_with(&format!("bc_{idx}"), pick_k(|c, k| c.b[k], idx));
    }
    let out = k.run();
    (
        out.values("theta").to_vec(),
        out.values("binding").to_vec(),
        out.values("x_den").to_vec(),
        out.values("x_nrg").to_vec(),
    )
}

#[test]
fn the_traced_projection_matches_the_host_family_in_magnitude_and_attribution() {
    let cases = cases();
    let (theta, binding, _, _) = run_kernel(&cases);

    // TOLERANCE, calibrated rather than guessed: the affine members cross in closed form (exact),
    // and the bisected member converges to 2^-20 of the trial interval. anything tighter than that
    // would false-positive on the bisection's own stopping point; anything looser would admit a
    // genuinely wrong crossing.
    let tol = 2.0_f64.powi(-20);

    for (ii, case) in cases.iter().enumerate() {
        let (want_theta, want_binding) = host(case);
        assert!(
            (theta[ii] - want_theta).abs() <= tol,
            "{}: traced theta {} vs host {want_theta}",
            case.what,
            theta[ii]
        );
        // ATTRIBUTION: two members can bind at the same theta, so the ledger's key is a separate
        // claim from the magnitude and is checked separately.
        assert_eq!(
            binding[ii], want_binding,
            "{}: traced binding member {} vs host {want_binding}",
            case.what, binding[ii]
        );
    }
}

#[test]
fn each_case_binds_the_member_it_was_built_to_bind() {
    // PREMISE: without this, the traced-vs-host comparison could pass with every case slack — two
    // implementations that both correctly do nothing. it also pins the family's DECLARATION ORDER,
    // so a reordering in the kernel that silently remapped the ledger's keys fails here.
    let cases = cases();
    let (theta, binding, _, _) = run_kernel(&cases);
    let expect = [
        (-1.0, "all slack"),
        (IDX_TEMPERATURE, "temperature floor binds"),
        (IDX_DENSITY, "density floor binds"),
        (IDX_MAGNETIZATION, "magnetization ceiling binds"),
        (IDX_ADMISSIBILITY, "admissibility binds"),
    ];
    for (ii, (want, what)) in expect.iter().enumerate() {
        assert_eq!(
            binding[ii], *want,
            "case '{what}' bound member {} but was built to exercise {want}",
            binding[ii]
        );
        if *want < 0.0 {
            assert_eq!(theta[ii], 1.0, "'{what}' must pass through untouched");
        } else {
            assert!(
                theta[ii] < 1.0,
                "'{what}' did not actually project (theta = {})",
                theta[ii]
            );
        }
    }
}

#[test]
fn the_temperature_crossing_matches_the_analytic_value() {
    // checking the trace against the HOST would pass on a bug the two share. this case has a
    // closed-form answer independent of both: with zero momentum and no field on a flat
    // background, the residual is (ehat + D) - D - f_min D = ehat - f_min D, so with D fixed at 1
    // the crossing sits where ehat = f_min.
    let cases = cases();
    let (theta, _, _, x_nrg) = run_kernel(&cases);
    let c = cases[1];
    // residual(t) = ehat(t) - f_min D, with D fixed, so the crossing is where ehat = f_min D.
    let want = (c.a_nrg - F_MIN * c.a_den) / (c.a_nrg - c.x_nrg);
    assert!(
        (theta[1] - want).abs() < 1e-12,
        "traced theta {} vs analytic {want}",
        theta[1]
    );
    // and the PROJECTED state sits on the boundary: ehat = f_min * D exactly.
    assert!(
        (x_nrg[1] - F_MIN * c.a_den).abs() < 1e-12,
        "projected ehat {} is not on the floor {}",
        x_nrg[1],
        F_MIN * c.a_den
    );
}

#[test]
fn swapping_the_anchor_and_candidate_is_detectable() {
    // the orientation bug this design is most exposed to: an inverted segment still yields a theta
    // in [0, 1], so nothing about the magnitude alone reveals it. running the family with the roles
    // reversed must NOT reproduce the forward answer.
    let forward = cases()[1];
    let reversed = Case {
        what: "reversed",
        a_den: forward.x_den,
        a_mom: forward.x_mom,
        a_nrg: forward.x_nrg,
        x_den: forward.a_den,
        x_mom: forward.a_mom,
        x_nrg: forward.a_nrg,
        b: forward.b,
    };
    let (theta, _, _, _) = run_kernel(&[forward, reversed]);
    assert!(
        (theta[0] - theta[1]).abs() > 1e-6,
        "the projection is insensitive to which endpoint is the anchor; an orientation error \
         in the wiring would be invisible (forward {}, reversed {})",
        theta[0],
        theta[1]
    );
}
