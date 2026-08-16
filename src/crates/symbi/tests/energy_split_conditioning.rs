// =============================================================================
// energy_split_conditioning.rs
//
// `ConservationDiag::max_ke_over_eint` must report the conditioning of the energy split.
//
// recovering internal energy from a conserved state is the subtraction
// `e = E - |m|^2 / 2 rho`. at a kinetic-to-internal ratio R the result is a `1/(1 + R)`
// fraction of its operands, so the inversion sheds about `log10(R)` significant digits
// each time it runs. the associated failure is silent and self-reinforcing -- an
// under-recovered internal energy cools the gas, which raises R, which worsens the next
// inversion -- and it hides from the timestep, because a cooling gas keeps the sound
// speed, and therefore the CFL, comfortable. nothing else in the run reports it.
//
// the law: the reported value is the analytic ratio of the state it is given, it is the
// maximum over the interior rather than a mean (one badly conditioned cell is the
// problem, and an average over a large quiescent domain buries it), and it is absent for
// the regimes whose c2p takes another route entirely.
// =============================================================================

use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::rhd::Rhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const N: usize = 32;

type SimNewt = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type SimRhd = SimState<Rhd, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

/// a uniform newtonian state at density 1, speed `v`, pressure `p`.
fn uniform_newtonian(v: f64, p: f64) -> SimNewt {
    SimNewt::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N])
        .spacing([1.0 / N as f64])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim construction failed")
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([v]),
            pre: p,
        })
        .build()
}

/// the analytic ratio for a uniform ideal gas: `(rho v^2 / 2) / (p / (gamma - 1))`,
/// equivalently `gamma (gamma - 1) M^2 / 2` with `M = v / sqrt(gamma p / rho)`.
fn analytic_ratio(rho: f64, v: f64, p: f64) -> f64 {
    (0.5 * rho * v * v) / (p / (GAMMA - 1.0))
}

#[test]
fn the_reported_ratio_is_the_analytic_one() {
    // three decades of conditioning, from comfortably subsonic to the cold kinetically
    // dominated regime the guard exists to name.
    for (v, p) in [(0.1_f64, 1.0_f64), (1.0, 1.0), (10.0, 1.0), (100.0, 1.0)] {
        let sim = uniform_newtonian(v, p);
        let diag = sim.conservation_diag().expect("host-accessible fields");
        let got = diag
            .max_ke_over_eint
            .expect("an adiabatic newtonian regime carries the energy split");
        let want = analytic_ratio(1.0, v, p);
        let rel = (got - want).abs() / want.max(1.0);
        assert!(
            rel < 1e-12,
            "v = {v}, p = {p}: reported KE/e_int {got:e} against the analytic {want:e} \
             (rel {rel:e})"
        );
    }
}

#[test]
fn a_single_ill_conditioned_cell_is_reported_over_a_quiescent_domain() {
    // the statistic is the maximum over the interior. the failure is local -- one cell whose
    // internal energy is a vanishing fraction of its total -- and a domain average over a large
    // quiet region reports it as healthy, which is exactly the silence this diagnostic removes.
    let sim = uniform_newtonian(0.1, 1.0);
    let quiet = sim
        .conservation_diag()
        .expect("host-accessible fields")
        .max_ke_over_eint
        .expect("energy split present");

    // spike one interior cell to a large momentum at fixed total energy: internal energy
    // becomes a small remainder there and nowhere else.
    let target = sim
        .geom
        .interior
        .iter()
        .nth(N / 2)
        .expect("an interior cell");
    let e_tot = *sim.fields.cons.nrg.as_ref().expect("nrg").view().at(target);
    let rho = *sim.fields.cons.den.view().at(target);
    // choose |m| so the recovered internal energy is 1e-4 of the total -> ratio 9999.
    let ke = 0.9999 * e_tot;
    sim.fields.cons.mom[0]
        .view_mut()
        .set(target, (2.0 * ke * rho).sqrt());

    let spiked = sim
        .conservation_diag()
        .expect("host-accessible fields")
        .max_ke_over_eint
        .expect("energy split present");

    // non-vacuity: the quiescent domain must be well conditioned, or "the max rose" says
    // nothing about whether the max is what is being reported.
    assert!(
        quiet < 1.0,
        "the unspiked domain is already ill-conditioned at {quiet:e}; this test cannot \
         distinguish a maximum from a mean"
    );
    assert!(
        spiked > 1.0e3,
        "one cell at 1e-4 of its total energy internal must surface: reported {spiked:e} \
         against a quiescent {quiet:e}"
    );
}

#[test]
fn the_ratio_is_absent_where_the_subtraction_does_not_happen() {
    // relativistic c2p is a bracketed root-find on a master function, leaving the E - KE
    // subtraction outside its algorithm entirely; reporting a newtonian-shaped ratio there
    // would be a category error, well past a merely conservative over-report.
    let sim = SimRhd::build(Rhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N])
        .spacing([1.0 / N as f64])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim construction failed")
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([0.5]),
            pre: 1.0,
        })
        .build();
    let diag = sim.conservation_diag().expect("host-accessible fields");
    assert!(
        diag.max_ke_over_eint.is_none(),
        "a relativistic regime reported {:?}, but its c2p performs no E - KE subtraction",
        diag.max_ke_over_eint
    );
    // the relativistic diagnostic that is defined stays populated, which makes this a scoping
    // statement about one field and leaves the diagnostic as a whole live.
    assert!(
        diag.max_w.is_some(),
        "max_w must remain populated for a relativistic regime"
    );
}
