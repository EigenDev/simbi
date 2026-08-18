// =============================================================================
// balanced_drain_vacuum_extrapolation.rs
//
// a spherical drain pulling supersonic inflow out of gas that sits below the
// ambient isentrope, reconstructed on the departures from the local hydrostatic
// profile.
//
// the profile a balanced reconstruction extrapolates along is the isentrope
// through the anchor cell, rho_eq ~ [1 + (gamma-1)(phi_anchor - phi)/cs^2]^
// (1/(gamma-1)). the bracket reaches zero — the isentrope's own vacuum boundary
// — once the potential climbs by more than cs^2/(gamma-1) across the
// reconstruction footprint. cold gas in a deep potential reaches that boundary
// within three cells: the free-fall mach number and the enthalpy the footprint
// spends are the same number, and the boundary lands inside the stencil at
// roughly mach 3. the equilibrium then collapses toward zero at the outer
// stencil points while growing without bound at the inner ones, the limiter
// sees departures spanning that whole range, and the face states it builds go
// negative — which is what the first-order flux correction spends its budget
// repairing.
//
// the balancing carries a per-cell weight for exactly this: where the isentrope
// terminates inside the footprint, the weight scales the potential variation
// down and the profile degrades continuously to a constant, whose departures are
// the plain differences of the state. this gate measures the flux correction's
// firing count on a configuration that puts 500 cells past the vacuum boundary
// while leaving the surrounding atmosphere balanced.
//
// run: cargo test -p symbi --test balanced_drain_vacuum_extrapolation -- --nocapture
// =============================================================================

use symbi::prelude::Solver;
use symbi::regimes::fofc::{fofc_reset_stats, fofc_stats};
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_discretize::Recon;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::hydrostatic::{BALANCE_FADE_FULL, BALANCE_STENCIL_REACH};
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, SurfaceSpec};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.3;
const N: usize = 32;
const DX: f64 = 1.0 / N as f64;
/// the ambient adiabat the atmosphere is built on.
const K0: f64 = 0.6;
const GM: f64 = 3.0;
/// four cells across the accretion radius, the production accretor geometry.
const R_ACC: f64 = 4.0 / N as f64;
/// one cell of plummer softening, so the field is within a percent of the bare
/// point mass everywhere outside the mask.
const SOFT: f64 = R_ACC / 4.0;
/// the entropy the core carries as a fraction of the ambient adiabat, and the
/// radius the deficit tapers out over. a quarter of the adiabat is a quarter of
/// the enthalpy, which puts the isentrope's vacuum boundary a factor four closer
/// and lands it inside the three-cell footprint.
const COLD: f64 = 0.25;
const R_COLD: f64 = 0.3;
const STEPS: u64 = 40;

type Sim = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 3>;
type Hier = Hierarchy<Newtonian, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset>;

/// the plummer-softened potential of the accretor, the field the body itself applies.
fn phi(r: f64) -> f64 {
    -GM / (r * r + SOFT * SOFT).sqrt()
}

/// the entropy deficit profile: `COLD` inside the accretion radius, the ambient adiabat
/// beyond `R_COLD`, linear between. the taper keeps the deficit's outer edge from standing
/// as a contact discontinuity, so what the flux correction reports comes from the drain.
fn adiabat(r: f64) -> f64 {
    let t = ((r - R_ACC) / (R_COLD - R_ACC)).clamp(0.0, 1.0);
    K0 * (COLD + (1.0 - COLD) * t)
}

/// the isentropic atmosphere in hydrostatic balance against the body's own softened
/// potential, normalized to rho = 1 at the domain corner, carrying the entropy deficit in
/// its core. the density profile is the balanced one throughout, so the atmosphere outside
/// `R_COLD` is on the ambient isentrope and the deficit shows up as a pressure the local
/// gravity cannot hold.
fn atmosphere(x: [f64; 3]) -> Prim<f64, 3> {
    let r = x.iter().map(|c| c * c).sum::<f64>().sqrt();
    let r_ref = 3.0_f64.sqrt() * 0.5;
    let a = (GAMMA - 1.0) / (GAMMA * K0);
    let rho = (1.0 + a * (phi(r_ref) - phi(r))).powf(1.0 / (GAMMA - 1.0));
    Prim {
        rho,
        vel: Tensor::new([0.0; 3]),
        pre: adiabat(r) * rho.powf(GAMMA),
    }
}

fn build() -> Hier {
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N, N])
        .origin([-0.5, -0.5, -0.5])
        .spacing([DX, DX, DX])
        // the parabola loads -3..+2 along the sweep.
        .ghosts(3)
        // outflow lets the atmosphere feed the drain; the balance-aware fill extends the
        // column hydrostatically through it.
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim construction failed")
        .set_initial(atmosphere)
        .build();
    let kernels = Kset::new(GAMMA, CFL, &sim.geom.allocated)
        .with_solver(Solver::HllcPlus)
        .expect("solver/regime mismatch")
        .reconstruction(Recon::Ppm)
        .well_balanced_reconstruction(true);
    Hierarchy::single(sim, kernels).with_bodies(
        BodyCollection::new().add(
            // porosity 1 is the pure drain channel: mass and energy leave through the mask
            // at the porous kernel's drain rate and no wall is imposed, so the gas crosses
            // the accretion radius in free fall.
            Body::black_hole(
                0,
                Tensor::new([0.0; 3]),
                Tensor::zeros(),
                GM,
                R_ACC,
                SOFT,
                // the pointwise body-source sink stays inert; the surface penalization is
                // the whole drain.
                0.0,
                1.0,
                R_ACC,
            )
            .with_surface(SurfaceSpec::Porous {
                porosity: 1.0,
                k_eta_n: 50.0,
                k_eta_t: 0.0,
            }),
        ),
    )
}

/// the largest share of a cell's enthalpy that the reconstruction footprint through it
/// spends climbing the potential, over the interior, and the number of cells past the
/// isentrope's vacuum boundary. this is the quantity the per-cell weight reads, evaluated
/// here on the state the run actually reaches.
fn footprint_enthalpy_spend(hier: &Hier) -> (f64, usize) {
    let st = &hier.levels[0].state;
    let rho = st.fields.prim.rho.view();
    let pre = st.fields.prim.pre_field().expect("adiabatic pre").view();
    let lo: [isize; 3] = std::array::from_fn(|a| st.geom.interior.spaces[a].lo as isize);
    let reach = BALANCE_STENCIL_REACH * DX;
    let (mut worst, mut past) = (0.0_f64, 0usize);
    for c in st.geom.interior.iter() {
        let x: [f64; 3] =
            std::array::from_fn(|a| st.geom.x_lo[a] + ((c[a] as isize - lo[a]) as f64 + 0.5) * DX);
        let phi_c = phi(x.iter().map(|v| v * v).sum::<f64>().sqrt());
        let mut rise = 0.0_f64;
        for ax in 0..3 {
            for step in [-reach, reach] {
                let mut probe = x;
                probe[ax] += step;
                let r = probe.iter().map(|v| v * v).sum::<f64>().sqrt();
                rise = rise.max(phi(r) - phi_c);
            }
        }
        let cs2 = GAMMA * pre.at(c) / rho.at(c);
        let spend = (GAMMA - 1.0) * rise / cs2;
        worst = worst.max(spend);
        if spend >= BALANCE_FADE_FULL {
            past += 1;
        }
    }
    (worst, past)
}

/// the drain runs without spending the first-order flux correction on its own reconstruction.
///
/// measured, N = 32, four cells across the accretion radius, ppm + hllc-lm + balanced
/// reconstruction, on the two arms that differ only in whether the local profile is weighted
/// where its vacuum boundary lands inside the reconstruction footprint:
///
///   weighted by the isentrope's validity   40 steps, 0 fallback cell-substages, 0 frozen
///   at full strength everywhere            first firing on step 4, 1058 fallback and 708
///                                          frozen by step 11, then a halt on 16 consecutive
///                                          unrecoverable freeze substages carrying
///                                          rho = -83.1, p = -310 three cells off the accretor
///
/// at full strength the equilibrium collapses to the floor at the outer stencil points and
/// grows without bound at the inner ones, so the face states the limiter builds on the
/// resulting departures leave the admissible set across the draining core; the first-order
/// redo reads the same poisoned reconstruction and cannot recover them, which is what turns
/// the fallback into a freeze and the freeze streak into a halted run. weighted, the profile
/// stays inside its own domain and the correction has nothing to repair.
///
/// the flow itself carries no shock — an atmosphere falling smoothly through a drain — so the
/// admissible-set redo has no legitimate work in this domain and the count belongs entirely to
/// the reconstruction.
#[test]
fn a_supersonic_drain_does_not_reconstruct_past_the_isentropes_vacuum_boundary() {
    fofc_reset_stats();
    let mut hier = build();
    // the footprint spend is a property of the state the run passes through, and the drain
    // heats its core as it empties it, so the deepest state is sampled step by step rather
    // than read off the end.
    let (mut spend, mut past) = footprint_enthalpy_spend(&hier);
    for _ in 0..STEPS {
        hier.evolve_steps(1).unwrap();
        let (s, p) = footprint_enthalpy_spend(&hier);
        spend = spend.max(s);
        past = past.max(p);
    }
    let (fired, froze) = fofc_stats();
    println!(
        "\nsupersonic drain, {STEPS} steps, balanced ppm reconstruction\n\
         fofc fallback cell-substages: {fired}\n\
         fofc frozen cell-substages:   {froze}\n\
         worst footprint enthalpy spend: {spend:.3}\n\
         cells past the vacuum boundary: {past}"
    );

    // the premise: the configuration has to drive cells past the isentrope's vacuum boundary,
    // or the reconstruction never reaches the state this gate is about and the firing count
    // measures nothing.
    assert!(
        past >= 100,
        "only {past} cells carry a footprint that spends the whole enthalpy of the gas in \
         them; the balanced reconstruction never reaches its profile's vacuum boundary here \
         and the flux-correction count below is a measurement of an untested path. deepen the \
         potential, cool the core, or coarsen the grid"
    );
    assert!(
        spend > 2.0,
        "the deepest footprint spends {spend:.3} of its cell's enthalpy; the defect this gate \
         reproduces needs the vacuum boundary well inside the stencil rather than at its edge"
    );

    assert_eq!(
        fired, 0,
        "the first-order flux correction fired on {fired} cell-substages. a draining accretor \
         is a smooth supersonic inflow with no shock in the domain, so the admissible-set redo \
         has no legitimate work here: every firing is the balanced reconstruction extrapolating \
         its local isentrope past the point where the isentrope describes a gas"
    );
    assert_eq!(
        froze, 0,
        "the flux correction froze {froze} cell-substages, so the first-order redo could not \
         recover the state either — the inadmissibility is being carried into the redo by the \
         stage input rather than produced by the reconstruction"
    );
}
