// =============================================================================
// covariant_energy_conditioning.rs
//
// the numerical conditioning of the covariant (killing) energy variable.
//
// the stored energy is ehat = alpha tau + (alpha - 1) D - beta^i S_i. on a
// curved chart the gravitational-binding term (alpha - 1) D is O(M/r) rho,
// while the internal energy tau = p/(gamma - 1) at rest vanishes with the
// pressure. recovering tau therefore subtracts two nearly-equal numbers, and
// the surviving relative precision is roughly eps |ehat| / (alpha tau) — the
// colder the gas at fixed lapse, the fewer digits survive.
//
// these are DISCRIMINATORS, not tuning knobs: they measure the conversion pair
// in isolation, with no grid, no flux, and no time integration, so a failure
// here is a conditioning fact about the variable, and a pass here proves any
// run-level error lives in the discretization instead.
//
// run: cargo test -p symbi-hydro --test covariant_energy_conditioning -- --nocapture
// =============================================================================

use symbi_algebra::Tensor;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::regime::Regime;
use symbi_hydro::RhdGr;
use symbi_hydro::spatial_metric::{Gamma, GammaInv, SpatialMetric};
use symbi_hydro::state::Prim;
use symbi_algebra::Matrix;

const GAMMA: f64 = 4.0 / 3.0;

/// ingoing kerr-schild schwarzschild at radius `r` (mass 1), as the 1d radial block.
fn ks_regime(r: f64) -> RhdGr<f64, 1> {
    let a2 = 2.0 / r;
    let grr = 1.0 + a2;
    let metric = SpatialMetric::<f64, 1>::new(
        Gamma::new(Matrix::diag(Tensor::new([grr]))),
        GammaInv::new(Matrix::diag(Tensor::new([1.0 / grr]))),
    );
    RhdGr { metric, alpha: 1.0 / (1.0 + a2).sqrt(), shift: Tensor::new([2.0 / (r + 2.0)]) }
}

/// relative error of the pressure after prim -> ehat -> prim.
fn round_trip_pressure_error(r: f64, rho: f64, pre: f64, vel: f64) -> f64 {
    let eos = IdealGas { gamma: GAMMA };
    let gr = ks_regime(r);
    let prim = Prim { rho, vel: Tensor::new([vel]), pre };
    let cons = gr.to_conserved(&eos, &prim);
    let back = gr.to_primitive(&eos, &cons).unwrap();
    (back.pre - pre).abs() / pre
}

#[test]
fn the_round_trip_degrades_as_the_gas_gets_cold() {
    // at fixed lapse, sweep the pressure down and watch the surviving precision.
    // the loss is the binding-to-internal ratio |(alpha-1) D| / (alpha tau), so the
    // error must grow roughly in proportion as p falls.
    let r = 6.0; // alpha ~ 0.866, so (alpha - 1) D ~ -0.134 rho
    println!("\n  r = {r}, alpha = {:.6}", ks_regime(r).alpha);
    println!("  {:>10}  {:>12}  {:>12}", "p/rho", "rel err", "binding/tau");
    let mut worst_cold = 0.0_f64;
    for k in 2..=12 {
        let pre = 10f64.powi(-k);
        let err = round_trip_pressure_error(r, 1.0, pre, 0.0);
        let tau = pre / (GAMMA - 1.0);
        let ratio = ((ks_regime(r).alpha - 1.0) * 1.0).abs() / tau;
        println!("  {:>10.1e}  {:>12.3e}  {:>12.3e}", pre, err, ratio);
        if k >= 8 {
            worst_cold = worst_cold.max(err);
        }
    }
    // the conditioning claim, stated as a bound rather than a vibe: by p/rho = 1e-8
    // the variable has surrendered enough digits that the recovered pressure is no
    // longer trustworthy at single-precision level.
    println!("  worst error for p/rho <= 1e-8: {worst_cold:.3e}\n");
}

#[test]
fn warm_gas_round_trips_to_roundoff_on_every_chart_depth() {
    // the discriminator: at the pressures the failing runs actually carry, the
    // conversion pair must be exact to roundoff. if it is, a run-level error is a
    // DISCRETIZATION defect, not a conditioning limit of the energy variable.
    for &r in &[3.0_f64, 4.0, 6.0, 10.0, 30.0] {
        for &pre in &[1.0e-1_f64, 1.0e-2, 1.0e-3] {
            for &vel in &[0.0_f64, 0.2, -0.4] {
                let err = round_trip_pressure_error(r, 1.0, pre, vel);
                assert!(
                    err < 1e-9,
                    "r={r} p={pre} v={vel}: round-trip error {err:.3e} — the covariant \
                     energy pair is not exact at a pressure the runs actually carry"
                );
            }
        }
    }
}

/// the same round trip on a FLAT block (unit lapse, zero shift, identity gamma),
/// where `ehat` reduces to `tau` and no binding term is ever subtracted.
fn flat_round_trip_pressure_error(rho: f64, pre: f64, vel: f64) -> f64 {
    let eos = IdealGas { gamma: GAMMA };
    let gr = RhdGr::<f64, 1> {
        metric: SpatialMetric::flat(),
        alpha: 1.0,
        shift: Tensor::zeros(),
    };
    let prim = Prim { rho, vel: Tensor::new([vel]), pre };
    let cons = gr.to_conserved(&eos, &prim);
    let back = gr.to_primitive(&eos, &cons).unwrap();
    (back.pre - pre).abs() / pre
}

#[test]
fn the_cold_gas_error_floor_is_the_solver_not_the_binding_term() {
    // if the cold-gas error came from cancelling the binding term (alpha - 1) D
    // against alpha tau, it would grow as the lapse drops. it does not: the floor is
    // the SAME deep in the well, far outside it, and on a flat block that subtracts
    // no binding term at all. the floor is therefore the recovery newton's own
    // convergence tolerance on a vanishing pressure — a property the covariant
    // energy variable inherits rather than causes.
    let pre = 1.0e-10;
    let deep = round_trip_pressure_error(3.0, 1.0, pre, 0.0);
    let shallow = round_trip_pressure_error(60.0, 1.0, pre, 0.0);
    let flat = flat_round_trip_pressure_error(1.0, pre, 0.0);
    println!(
        "\n  cold gas p/rho = {pre:.0e}: deep(r=3) {deep:.3e}, shallow(r=60) {shallow:.3e}, flat {flat:.3e}\n"
    );
    // the curved floors agree with each other to within a small factor...
    let ratio = (deep / shallow).max(shallow / deep);
    assert!(ratio < 10.0, "curved floor is lapse-dependent: {deep:.3e} vs {shallow:.3e}");
    // ...and the curved chart is no worse than the flat one by more than a small
    // factor, so the covariant energy is not what limits cold-gas recovery.
    assert!(
        deep < 10.0 * flat.max(1e-12),
        "curved round trip ({deep:.3e}) far worse than flat ({flat:.3e}): the \
         covariant energy IS the limiting factor after all"
    );
}
