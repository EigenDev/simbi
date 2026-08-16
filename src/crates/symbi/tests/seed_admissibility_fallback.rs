// =============================================================================
// seed_admissibility_fallback.rs
//
// fine-level IC seeding must deliver an admissible conserved state. the seed
// path prolongs each conserved component independently at high order, and no
// component-wise prolongation preserves E >= |m|^2/(2 rho): across a steep
// velocity ramp the limited slope of `m` and the limited slope of `E` are cut
// at different places (E carries the kinetic energy, which is quadratic in m),
// so a child cell can inherit the momentum of the ramp with the energy of the
// plateau and land at negative internal energy. the failure needs kinetic
// energy density comparable to internal — a near-sonic velocity field on a
// cold background — which is why a weak subsonic seed never trips it.
//
// gates:
// - a transonic ramp on a cold background forces the parent-injection fallback
//   (count > 0, so the gate cannot pass vacuously), and every fine cell is
//   admissible afterward;
// - a deeply subsonic ramp takes zero fallbacks, so admissible initial data is
//   seeded bit-identically to the path without the audit.
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.3;
const N: usize = 16;
const PRE: f64 = 0.006;

type Sim = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 3>;
type Hier = Hierarchy<Newtonian, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset>;

/// a 2-level hierarchy on [0,1)^3, middle half refined, holding a cold uniform
/// gas with an x-velocity wave of amplitude `amp` at 4 coarse cells per
/// wavelength. the kinetic energy density rides at double the wavenumber — two
/// cells per wavelength, the grid nyquist — so the energy's reconstruction and
/// the momentum's cannot agree near the wave's extrema. internal energy density
/// is pre / (gamma - 1) = 0.009, so an amplitude of order one puts the wave's
/// kinetic energy density far above it. a linear or quadratic profile would be
/// reproduced exactly by the parabolic stencil; the oscillation is what makes
/// the component-wise prolongation inconsistent.
fn ramp_hierarchy(amp: f64) -> Hier {
    let dx = 1.0 / N as f64;
    let kx = 2.0 * std::f64::consts::PI * (N as f64 / 4.0);
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 3])
        .origin([0.0; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .unwrap()
        .set_initial(move |x: [f64; 3]| Prim {
            rho: 1.0,
            vel: Tensor::new([
                amp * (kx * x[0]).sin(),
                amp * (kx * x[1] + 1.0).sin(),
                amp * (kx * x[2] + 2.0).sin(),
            ]),
            pre: PRE,
        })
        .build();
    let ck = Kset::new(GAMMA, CFL, &coarse.geom.allocated);
    // two nested levels: the inadmissibility compounds down the cascade (each
    // level prolongs the previous level's already-prolonged data), so the
    // deeper level is where the overshoot first crosses the margin.
    let regions = [
        RefinementRegion {
            x_lo: [0.25; 3],
            x_hi: [0.75; 3],
        },
        RefinementRegion {
            x_lo: [0.375; 3],
            x_hi: [0.625; 3],
        },
    ];
    // the order plm-reconstruction hierarchies seed with (recon order + 1): a
    // parabolic stencil, non-monotone at extrema, which is where the momentum
    // overshoot that breaks admissibility comes from. the strictly monotone
    // linear prolong survives this ramp; the parabolic one does not.
    Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, |s| {
        Kset::new(GAMMA, CFL, &s.geom.allocated)
    })
    .unwrap()
}

/// smallest internal energy density E - |m|^2/(2 rho) over a level's interior.
fn min_internal_energy(hier: &Hier, level: usize) -> f64 {
    let state = &hier.levels[level].state;
    let cons = &state.fields.cons;
    let nrg = cons.nrg_field().expect("adiabatic state carries energy");
    state
        .geom
        .interior
        .iter()
        .map(|c| {
            let den = *cons.den.view().at(c);
            let ke: f64 = (0..3)
                .map(|k| {
                    let m = *cons.mom[k].view().at(c);
                    m * m
                })
                .sum::<f64>()
                / (2.0 * den);
            *nrg.view().at(c) - ke
        })
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn transonic_seed_forces_the_fallback_and_stays_admissible() {
    let hier = ramp_hierarchy(1.0);
    let reseeded = hier.seed_fine_from_coarse().unwrap();
    assert!(
        reseeded > 0,
        "the transonic ramp never produced an inadmissible prolonged cell; the \
         gate is vacuous — sharpen the ramp or drop the background pressure"
    );
    for level in 0..hier.levels.len() {
        let e_int = min_internal_energy(&hier, level);
        assert!(
            e_int > 0.0,
            "level {level} holds a cell with non-positive internal energy \
             ({e_int:.3e}) after the admissibility fallback"
        );
    }
}

#[test]
fn subsonic_seed_takes_no_fallback() {
    // kinetic energy density at amplitude 0.05 is 1.25e-3 against an internal
    // energy density of 9e-3: prolongation overshoot is orders below the
    // admissibility margin, so the audit must not touch a single cell and the
    // seeded state is identical to the unaudited path.
    let hier = ramp_hierarchy(0.05);
    let reseeded = hier.seed_fine_from_coarse().unwrap();
    assert_eq!(
        reseeded, 0,
        "a deeply subsonic seed took {reseeded} fallback cell(s); admissible \
         data must seed identically to the path without the audit"
    );
}
