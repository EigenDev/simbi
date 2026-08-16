// =============================================================================
// refinement_well_balanced_isothermal.rs
//
// the same well-balancing on an energy-free regime.
//
// an isothermal run stores no pressure: its equation of state supplies `p = cs^2 rho` from
// the density alone. every piece of the machinery therefore carries one fewer component —
// the declared target has no pressure slot, the captured imbalance has no energy component,
// and the flux register has no energy face — and none of that is exercised by an adiabatic
// test. a wire that reads its components positionally will misalign here and nowhere else.
//
// the physics differs too, not just the bookkeeping. with `p = cs^2 rho` hydrostatic balance
// is `grad(ln rho) = -grad phi / cs^2`, so the atmosphere is exponential in the potential
// rather than a power of it, and the density contrast is set by the potential depth measured
// in units of `cs^2`.
//
// run: cargo test -p symbi --test refinement_well_balanced_isothermal -- --nocapture
// =============================================================================

use symbi::regimes::substrate::IsoSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::Isothermal;
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::state::PrimG;
use symbi_xpu::{CpuSpace, HostMemory};

const N: usize = 128;
const CFL: f64 = 0.4;
const CS: f64 = 1.0;
/// the gravitating mass sits one domain width left of x = 0, so the gas at x feels a bare
/// point mass at radius x + 1 and the domain covers r in [1, 2] with no singularity.
const G_OFFSET: f64 = 1.0;
/// deep enough in units of `cs^2` that the atmosphere is genuinely stratified: the potential
/// difference across the domain is `GM/2`, so the density contrast is `exp(GM/(2 cs^2))`.
const GM: f64 = 4.0;
const STEPS: u64 = 20;

type Sim = SimState<IsoNewtonian, 1, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
type Kset = IsoSubstrateKernelSet<HostMemory, f64, 1>;
type Hier = Hierarchy<IsoNewtonian, 1, 1, Cartesian, Isothermal<f64>, CpuSpace, HostMemory, Kset>;

/// the isothermal atmosphere in balance against `phi = -GM/r`: `rho = exp(-(phi - phi_ref)/cs^2)`,
/// normalized to `rho = 1` at the outer edge.
fn isothermal_atmosphere(x: [f64; 1]) -> PrimG<f64, 1, IsoModel> {
    let r = x[0] + G_OFFSET;
    let phi = -GM / r;
    let phi_reference = -GM / (1.0 + G_OFFSET);
    PrimG {
        rho: ((phi_reference - phi) / (CS * CS)).exp(),
        vel: Tensor::new([0.0]),
        pre: Default::default(),
    }
}

fn nested(levels: usize) -> Vec<RefinementRegion<1>> {
    (0..levels.saturating_sub(1))
        .map(|ii| {
            let half = 0.2 / 2f64.powi(ii as i32);
            RefinementRegion {
                x_lo: [0.5 - half],
                x_hi: [0.5 + half],
            }
        })
        .collect()
}

fn build(regions: &[RefinementRegion<1>]) -> Hier {
    let coarse = Sim::build(IsoNewtonian, Isothermal { cs: CS }, Cartesian)
        .cells([N])
        .spacing([1.0 / N as f64])
        // a reflecting wall exerts no work on gas at rest, so the atmosphere is a fixed point
        // of the boundary as well as of the interior.
        .boundaries(Boundaries::uniform(BoundaryType::Reflect))
        .cfl(CFL)
        .allocate()
        .expect("sim construction failed")
        .set_initial(isothermal_atmosphere)
        .build();
    let ck = Kset::new(CS, CFL, &coarse.geom.allocated);
    let hier = Hierarchy::with_refinement(coarse, ck, regions, ProlongOrder::Ppm, |s| {
        Kset::new(CS, CFL, &s.geom.allocated)
    })
    .unwrap()
    .with_bodies(symbi_ib::BodyCollection::new().add(symbi_ib::Body::gravitational(
        0,
        Tensor::new([-G_OFFSET]),
        Tensor::zeros(),
        GM,
        1.0e-3,
        0.0,
    )));
    for lvl in 1..hier.levels.len() {
        hier.levels[lvl].state.seed_cells(isothermal_atmosphere);
    }
    hier
}

fn worst_speed(hier: &Hier, level: usize) -> f64 {
    let st = &hier.levels[level].state;
    let vel = st.fields.prim.vel[0].view();
    st.geom
        .interior
        .iter()
        .map(|c| vel.at(c).abs())
        .fold(0.0_f64, f64::max)
}

fn composite_mass(hier: &Hier) -> f64 {
    let mut mass = 0.0;
    for lvl in hier.levels.iter() {
        let vol: f64 = lvl.state.geom.dx.iter().product();
        let den = lvl.state.fields.cons.den.view();
        for c in lvl.state.geom.interior.iter() {
            if lvl.coverage.as_ref().is_some_and(|cov| cov.contains(c)) {
                continue;
            }
            mass += *den.at(c) * vol;
        }
    }
    mass
}

#[test]
fn an_energy_free_regime_holds_its_declared_target() {
    // non-vacuity of the setup itself: an isothermal atmosphere shallow enough to be nearly
    // uniform would be interpolated exactly by the transfer and would exercise nothing.
    let contrast = isothermal_atmosphere([0.0]).rho / isothermal_atmosphere([1.0]).rho;
    println!("\nisothermal atmosphere, density contrast across the domain: {contrast:.2}x");
    assert!(
        contrast > 5.0,
        "the atmosphere spans only {contrast:.2}x in density; a near-constant profile is \
         interpolated exactly at any order and the transfer would have nothing to get wrong"
    );

    for levels in 1..=3usize {
        let mut control = build(&nested(levels));
        control.evolve_steps(STEPS).unwrap();
        let drift: Vec<f64> = (0..levels).map(|ll| worst_speed(&control, ll)).collect();

        let mut hier = build(&nested(levels)).with_equilibrium(isothermal_atmosphere).unwrap();
        hier.seed_equilibrium();
        let m0 = composite_mass(&hier);
        hier.evolve_steps(STEPS).unwrap();
        let m1 = composite_mass(&hier);
        let held: Vec<f64> = (0..levels).map(|ll| worst_speed(&hier, ll)).collect();

        let shown: Vec<String> = drift
            .iter()
            .zip(&held)
            .map(|(d, h)| format!("{d:.3e} -> {h:.3e}"))
            .collect();
        println!(
            "levels={levels}  max|v| undeclared -> declared: {shown:?}   mass drift {:.3e}",
            ((m1 - m0) / m0).abs()
        );

        // the undeclared run has to genuinely fail to hold the atmosphere, or holding it
        // still proves nothing.
        for (ll, speed) in drift.iter().enumerate() {
            assert!(
                *speed > 1.0e-6,
                "level {ll} of the UNDECLARED {levels}-level isothermal run drifted only \
                 {speed:.3e} in {STEPS} steps; the setup no longer exercises the imbalance"
            );
        }
        for (ll, speed) in held.iter().enumerate() {
            assert!(
                *speed < 1.0e-12,
                "level {ll} of the {levels}-level isothermal run moved at {speed:.3e} while \
                 sitting on its declared target; a fixed point may only lose roundoff"
            );
        }
        let relative = ((m1 - m0) / m0).abs();
        assert!(
            relative < 1.0e-14,
            "composite mass moved by {relative:.3e} over {STEPS} steps of the {levels}-level \
             isothermal run; the coarse-fine transfer is no longer conservative"
        );
    }
}

#[test]
fn the_captured_target_carries_no_energy_component() {
    // the positional wire is what this pins. an energy-free regime allocates no energy field,
    // so the captured imbalance and the target flux must both be one component shorter than
    // the adiabatic case — a reader that assumed a fixed layout would write the pressure into
    // the momentum slot and produce a moving atmosphere that still looked like an equilibrium.
    let hier = build(&nested(2)).with_equilibrium(isothermal_atmosphere).unwrap();
    for (ll, level) in hier.levels.iter().enumerate() {
        assert!(
            level.residual_eq.as_ref().unwrap().nrg_field().is_none(),
            "level {ll} captured an energy component on a regime that carries no energy"
        );
        assert!(
            level.flux_eq.as_ref().unwrap()[0].nrg_field().is_none(),
            "level {ll} captured an energy flux on a regime that carries no energy"
        );
        assert!(
            level.cons_eq.as_ref().unwrap().nrg_field().is_none(),
            "level {ll} stored an energy component of the target on an energy-free regime"
        );
    }

    // and the imbalance that is carried has to be real, or the absence above is trivial.
    let measured = hier
        .target_imbalance_convergence(0)
        .expect("levels 0 and 1 share an interior to compare over");
    let peak = measured.scale.iter().fold(0.0_f64, |m, s| m.max(*s));
    let shown: Vec<String> = measured.ratio.iter().map(|r| format!("{r:.2}")).collect();
    println!("isothermal per-cell median imbalance ratio by component: {shown:?}");
    assert!(
        peak > 1.0e-9,
        "the captured imbalance peaks at {peak:.3e}, indistinguishable from zero; there is \
         nothing being corrected and the component check above is vacuous"
    );
    // density and momentum only: two components, not three.
    assert_eq!(
        measured.ratio.len(),
        2,
        "an energy-free 1d regime has a density and one momentum component, got {}",
        measured.ratio.len()
    );
}
