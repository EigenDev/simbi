// =============================================================================
// spherical_excision_evolve.rs
//
// horizon excision end to end on a spherical kerr-schild chart, through the production driver
// rather than the kernels alone: a uniform gas at rest on a radial grid that spans r_+ = 2M
// accretes, and the region inside the excision surface must be held at the cold vacuum floor
// while the exterior evolves normally.
//
// the excision is the causal statement, not a numerical convenience: inside r_+ every
// characteristic points inward, so those cells cannot influence the exterior and holding them at a
// floor makes the surface a one-way absorber. the failure this gate catches is the substrate
// silently not running the pass — the interior then keeps accreting gas forever, which nothing
// else notices precisely because it is causally disconnected and never protests.
// =============================================================================

use symbi::regimes::substrate_rhd::RhdSubstrateKernelSet;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi::sim::substrate_seam::WithExcision;
use symbi_algebra::Tensor;
use symbi_geometry::SchwarzschildKS;
use symbi_hydro::Rhd;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const N: usize = 64;
const MASS: f64 = 1.0;
const GAMMA: f64 = 4.0 / 3.0;
const CFL: f64 = 0.3;
const R_MIN: f64 = 1.0;
const R_MAX: f64 = 40.0;
const R_EXC: f64 = 1.4;
const R_PLUS: f64 = 2.0 * MASS;
const RHO_AMBIENT: f64 = 1.0;
const P_AMBIENT: f64 = 1.0e-2;
const T_FINAL: f64 = 6.0;

type Sim = SimState<Rhd, 1, SchwarzschildKS<f64>, IdealGas<f64>, CpuSpace, HostMemory>;

/// cell-center radii of the uniform radial grid the sim is built on.
fn radii() -> Vec<f64> {
    let dr = (R_MAX - R_MIN) / N as f64;
    (0..N).map(|i| R_MIN + (i as f64 + 0.5) * dr).collect()
}

fn run(excision_radius: f64) -> (Vec<f64>, Vec<f64>) {
    run_on(excision_radius, false).0
}

/// `log_radial` selects the geometric radial map the accretion problems actually use. it belongs in
/// this gate because the kernel's face map reads the axis map's own parameter — the log slope, not
/// a linear width — and a dispatch that hands it the wrong one puts every cell center outside the
/// excision surface, masking nothing. a uniform-only gate cannot see that: the conversion between
/// the two is the identity there.
fn run_on(excision_radius: f64, log_radial: bool) -> ((Vec<f64>, Vec<f64>), Vec<f64>) {
    let dr = (R_MAX - R_MIN) / N as f64;
    let builder = Sim::build(
        Rhd,
        IdealGas { gamma: GAMMA },
        SchwarzschildKS { mass: MASS },
    )
    .cells([N])
    .origin([R_MIN]);
    let log_slope = (R_MAX / R_MIN).log10() / N as f64;
    // note the deliberate mismatch: `spacing` carries the linear cell width while the axis map is
    // logarithmic. that is the state the config front end actually produces — the map is the
    // authority and every dispatch is required to derive the kernel's face-map parameter from it
    // via `kernel_geom`. building the two consistently here would hide exactly the defect this
    // gate exists to catch.
    let builder = if log_radial {
        builder
            .spacing([dr])
            .coord_maps(Some([symbi_geometry::AxisMap::Log {
                start: R_MIN,
                log_slope,
            }]))
    } else {
        builder.spacing([dr])
    };
    let mut sim = builder
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .expect("spherical KS sim")
        .set_initial(|_r| Prim {
            rho: RHO_AMBIENT,
            vel: Tensor::zeros(),
            pre: P_AMBIENT,
        })
        .build();

    let sub = RhdSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, CFL, &sim.geom.allocated)
        .with_excision(excision_radius, 1.0, 1.0);
    evolve_with_callback(&mut sim, &sub, T_FINAL, 1, |_| {}).expect("evolve");

    let mut rho = Vec::with_capacity(N);
    let mut pre = Vec::with_capacity(N);
    for c in sim.geom.interior.iter() {
        rho.push(*sim.fields.prim.rho.view().at(c));
        pre.push(*sim.fields.prim.pre_field().expect("prim.pre").view().at(c));
    }
    let r_used: Vec<f64> = if log_radial {
        (0..N)
            .map(|i| {
                let f0 = R_MIN * 10.0f64.powf(i as f64 * log_slope);
                let f1 = R_MIN * 10.0f64.powf((i + 1) as f64 * log_slope);
                (f0 + f1) * 0.5
            })
            .collect()
    } else {
        radii()
    };
    ((rho, pre), r_used)
}

#[test]
fn the_excised_interior_is_held_at_the_vacuum_floor() {
    let r = radii();

    // the premise: the grid must straddle both the excision surface and the horizon, with live
    // cells on either side. a grid that stopped outside r_+ would be testing an inner wall.
    let inside: Vec<usize> = (0..N).filter(|&i| r[i] < R_EXC).collect();
    assert!(
        !inside.is_empty() && r[0] < R_EXC && R_EXC < R_PLUS && R_PLUS < r[N - 1],
        "the grid does not span the excision surface and the horizon \
         (r = [{:.3}, {:.3}], r_exc = {R_EXC}, r_+ = {R_PLUS})",
        r[0],
        r[N - 1]
    );

    let (rho_exc, pre_exc) = run(R_EXC);

    for &i in &inside {
        assert!(
            rho_exc[i] < 1.0e-6 * RHO_AMBIENT,
            "excised cell {i} (r = {:.3}) carries gas: rho = {:.3e}. the excision pass is not \
             running, or it is transmitting rather than absorbing",
            r[i],
            rho_exc[i]
        );
        assert!(
            pre_exc[i] < 1.0e-6 * P_AMBIENT,
            "excised cell {i} (r = {:.3}) carries pressure: p = {:.3e}",
            r[i],
            pre_exc[i]
        );
    }

    // the exterior must be untouched by the excision in the sense that it still holds physical
    // gas: an excision that swallowed the whole grid would satisfy the assertions above.
    let outer = N - 1;
    assert!(
        rho_exc[outer] > 0.5 * RHO_AMBIENT && pre_exc[outer] > 0.0,
        "the exterior did not survive: rho = {:.3e}, p = {:.3e}",
        rho_exc[outer],
        pre_exc[outer]
    );
}

#[test]
fn the_excised_interior_is_held_on_a_log_radial_grid() {
    // the same contract on the geometric radial map the accretion problems use. the excision
    // surface is a coordinate surface on either map, so the physics is identical; what differs is
    // the parameter the kernel's face map consumes.
    let ((rho, pre), r) = run_on(R_EXC, true);
    let inside: Vec<usize> = (0..N).filter(|&i| r[i] < R_EXC).collect();
    assert!(
        !inside.is_empty() && R_PLUS < r[N - 1],
        "the log grid does not straddle the excision surface and the horizon \
         (r = [{:.3}, {:.3}])",
        r[0],
        r[N - 1]
    );
    for &i in &inside {
        assert!(
            rho[i] < 1.0e-6 * RHO_AMBIENT && pre[i] < 1.0e-6 * P_AMBIENT,
            "log-grid excised cell {i} (r = {:.3}) carries gas: rho = {:.3e}, p = {:.3e}. the \
             kernel's face map is being fed a linear width where it expects the log slope",
            r[i],
            rho[i],
            pre[i]
        );
    }
    assert!(
        rho[N - 1] > 0.5 * RHO_AMBIENT,
        "the log-grid exterior did not survive: rho = {:.3e}",
        rho[N - 1]
    );
}

#[test]
fn without_excision_the_interior_fills_with_gas() {
    // the companion that makes the gate above non-vacuous. at zero radius the pass is inert, so
    // the same cells accrete and hold gas: the difference between the two runs is the excision.
    // without this, a run that somehow evacuated the interior for an unrelated reason would let
    // the excision gate pass while the pass did nothing.
    let r = radii();
    let inside: Vec<usize> = (0..N).filter(|&i| r[i] < R_EXC).collect();
    let (rho_plain, _) = run(0.0);
    let worst = inside
        .iter()
        .map(|&i| rho_plain[i])
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        worst > 0.5 * RHO_AMBIENT,
        "the un-excised interior did not retain gas (max rho = {worst:.3e}), so the excision \
         gate is comparing against an already-empty region and proves nothing"
    );
}
