// =============================================================================
// balanced_drain_vacuum_extrapolation.rs
//
// a spherical drain pulling supersonic inflow out of gas that sits below the
// ambient adiabat, reconstructed on the pressure departures from the local
// mechanical equilibrium (Kaeppeli & Mishra, A&A 587, A94, 2016).
//
// the profile a balanced reconstruction extrapolates along is the linear segment
// p_eq = p_anchor + rho_anchor (phi_anchor - phi). cold gas in a deep potential
// drives that line past its positive domain within the reconstruction footprint:
// the segment crosses zero once the potential climbs by p/rho across the stencil,
// and a face state built on a negative equilibrium leaves the admissible set --
// which is what the first-order flux correction would spend its budget repairing.
//
// the profile carries a positivity floor for exactly this: past the crossing the
// evaluation returns a floor rather than a negative pressure, the departure
// carries the difference, and the limiter works on finite numbers. this gate
// measures the flux correction's firing count on a configuration that puts
// hundreds of cells past the segment's positive domain while leaving the
// surrounding atmosphere balanced: the correction must find nothing to repair.
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
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
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
    Prim::adiabatic(
        Density(rho),
        Tensor::new([0.0; 3]),
        Pressure(adiabat(r) * rho.powf(GAMMA)),
    )
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

/// the reconstruction footprint in cell widths: the farthest offset the parabola's
/// six-point window evaluates the equilibrium at.
const FOOTPRINT_REACH: f64 = 3.0;

/// the largest share of its segment's positive domain that a cell's reconstruction
/// footprint spends climbing the potential, over the interior, and the number of cells
/// whose footprint crosses the segment's zero (`rho * rise > p`, where the positivity
/// floor engages). evaluated on the state the run actually reaches.
fn footprint_overreach(hier: &Hier) -> (f64, usize) {
    let st = &hier.levels[0].state;
    let rho = st.fields.prim.rho.view();
    let pre = st.fields.prim.pre_field().expect("adiabatic pre").view();
    let lo: [isize; 3] = std::array::from_fn(|a| st.geom.interior.spaces[a].lo as isize);
    let reach = FOOTPRINT_REACH * DX;
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
        let spend = rho.at(c) * rise / pre.at(c);
        worst = worst.max(spend);
        if spend >= 1.0 {
            past += 1;
        }
    }
    (worst, past)
}

/// the drain runs without spending the first-order flux correction on its own
/// reconstruction. the flow itself carries no shock — an atmosphere falling smoothly
/// through a drain — so the admissible-set redo has no legitimate work in this domain,
/// and every firing would be the balanced reconstruction extrapolating its equilibrium
/// past the point where the segment describes a gas. the positivity floor and the
/// departure that carries the clamped difference are what keep every face state
/// admissible while hundreds of cells sit past the segment's zero crossing.
#[test]
fn a_supersonic_drain_reconstructs_admissible_faces_past_the_segments_positive_domain() {
    fofc_reset_stats();
    let mut hier = build();
    // the footprint spend is a property of the state the run passes through, and the drain
    // heats its core as it empties it, so the deepest state is sampled step by step rather
    // than read off the end.
    let (mut spend, mut past) = footprint_overreach(&hier);
    for _ in 0..STEPS {
        hier.evolve_steps(1).unwrap();
        let (s, p) = footprint_overreach(&hier);
        spend = spend.max(s);
        past = past.max(p);
    }
    let (fired, froze) = fofc_stats();
    println!(
        "\nsupersonic drain, {STEPS} steps, balanced ppm reconstruction\n\
         fofc fallback cell-substages: {fired}\n\
         fofc frozen cell-substages:   {froze}\n\
         worst footprint overreach: {spend:.3}\n\
         cells past the segment's positive domain: {past}"
    );

    // the premise: the configuration has to drive cells past the segment's positive
    // domain, or the floor never engages, the reconstruction never reaches the state this
    // gate is about, and the firing count measures nothing.
    assert!(
        past >= 100,
        "only {past} cells carry a footprint that drives the equilibrium segment past its \
         zero; the positivity floor never engages here and the flux-correction count below \
         is a measurement of an untested path. deepen the potential, cool the core, or \
         coarsen the grid"
    );
    assert!(
        spend > 2.0,
        "the deepest footprint spends {spend:.3} of its segment's positive domain; the \
         hazard this gate reproduces needs the zero crossing well inside the stencil rather \
         than at its edge"
    );

    assert_eq!(
        fired, 0,
        "the first-order flux correction fired on {fired} cell-substages. a draining accretor \
         is a smooth supersonic inflow with no shock in the domain, so the admissible-set redo \
         has no legitimate work here: every firing is the balanced reconstruction building \
         a face state its own floor and departures failed to keep admissible"
    );
    assert_eq!(
        froze, 0,
        "the flux correction froze {froze} cell-substages, so the first-order redo could not \
         recover the state either — the inadmissibility is being carried into the redo by the \
         stage input rather than produced by the reconstruction"
    );
}
