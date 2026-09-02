// =============================================================================
// body_dye_drain.rs
//
// an accreting immersed body and the passive scalar.
//
// a sink is a mass drain: it swallows gas parcels whole, and the dye dissolved in them goes with
// the gas. so the concentration of what remains is untouched — `D_chi = rho chi` scales by exactly
// the factor the density does. the failure this gate exists to catch is the drain removing mass
// while leaving `cons.chi` alone, which reads downstream as the sink concentrating the dye in the
// surviving gas: smooth, bounded, positive, and wrong.
//
// a uniform dye is the sharpest probe. under the correct rule it stays uniform at its seeded value
// no matter how much mass the sink removes; under the broken one it rises as 1/f exactly where the
// drain bit, so the error is largest in the cells nearest the body.
//
// run: cargo test -p symbi --test body_dye_drain
// =============================================================================

use symbi::regimes::substrate_kernels::dispatch_penalize;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, SurfaceSpec};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const N: usize = 48;
const L: f64 = 1.0;
const RADIUS: f64 = 0.3;
const CHI: f64 = 0.7;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

fn build() -> Sim {
    let dx = 2.0 * L / N as f64;
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([-L, -L])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(0.3)
        .allocate()
        .expect("sim")
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0]), Pressure(1.0)))
        .build()
        .with_bodies(
            BodyCollection::new().add(
                // a bare drain: no wall channels, so the only thing acting is the mass sink and any
                // dye error cannot be attributed to a momentum or thermal channel. the body
                // gravitates because the spherical drain rate is k_drain*sqrt(GM/r_acc^3);
                // its pull never acts here, since the gravity source is not dispatched.
                Body::black_hole(
                    0,
                    Tensor::new([0.0, 0.0]),
                    Tensor::new([0.0, 0.0]),
                    1.0,
                    RADIUS,
                    RADIUS,
                    0.0,
                    1.0,
                    RADIUS,
                )
                .with_surface(SurfaceSpec::Drain),
            ),
        )
        .with_passive_scalar()
        .expect("chi alloc");
    // no explicit shape: a bare sphere drain uses the analytic mask. attaching an SdfExpr would
    // route the dispatch to the shaped porous kernel, whose porosity dials a Drain surface has.
    // uniform concentration over the whole allocated grid, consistent with the seeded density.
    let cons_chi = sim.fields.cons.chi_field().expect("cons chi");
    let prim_chi = sim.fields.prim.chi_field().expect("prim chi");
    for c in sim.geom.allocated.clone().iter() {
        let rho = *sim.fields.cons.den.view().at(c);
        cons_chi.view_mut().set(c, rho * CHI);
        prim_chi.view_mut().set(c, CHI);
    }
    sim
}

fn total_mass(sim: &Sim) -> f64 {
    sim.geom
        .interior
        .iter()
        .map(|c| *sim.fields.cons.den.view().at(c))
        .sum()
}

#[test]
fn a_sink_removes_mass_and_its_dye_together() {
    let sim = build();
    let mass0 = total_mass(&sim);
    let dt = 1e-3;
    for _ in 0..40 {
        dispatch_penalize(&sim, dt, GAMMA, 1.0, 3.0);
    }
    let mass1 = total_mass(&sim);

    // the premise: the sink actually swallowed something. with no drain there is nothing to get
    // wrong and a uniform dye would stay uniform for free.
    let swallowed = (mass0 - mass1) / mass0;
    assert!(
        swallowed > 1e-3,
        "the sink drained only a fraction {swallowed:e} of the mass; the gate is vacuous"
    );

    // the concentration of the surviving gas is unchanged. checked on the conserved dye against the
    // conserved density, so this is independent of when the dye c2p last ran.
    let cons_chi = sim.fields.cons.chi_field().expect("cons chi");
    let (mut worst, mut worst_at) = (0.0_f64, [0isize; 2]);
    for c in sim.geom.interior.iter() {
        let rho = *sim.fields.cons.den.view().at(c);
        if rho <= 0.0 {
            continue;
        }
        let chi = *cons_chi.view().at(c) / rho;
        let err = (chi - CHI).abs();
        if err > worst {
            worst = err;
            worst_at = c;
        }
    }
    assert!(
        worst < 1e-12,
        "the sink changed the dye concentration by {worst:e} at {worst_at:?} \
         (seeded {CHI}); a mass drain must carry the dye with the mass it removes"
    );
}

// the isothermal twin of the drain gate. the dye is a slot on the conserved state, orthogonal to
// the energy slot: `D_chi = rho chi` involves no energy, so an isothermal sink must carry the dye
// with the mass exactly as an adiabatic one does. this is the gate that would have caught an
// isothermal run silently concentrating its dye while the adiabatic path was correct.
#[cfg(test)]
mod isothermal {
    use symbi::regimes::substrate_kernels::dispatch_penalize;
    use symbi::sim::state::*;
    use symbi_algebra::Tensor;
    use symbi_geometry::Cartesian;
    use symbi_hydro::energy::IsoModel;
    use symbi_hydro::quantity::Density;
    use symbi_hydro::state::PrimG;
    use symbi_hydro::{IsoNewtonian, Isothermal};
    use symbi_ib::{Body, BodyCollection, SurfaceSpec};
    use symbi_xpu::{CpuSpace, HostMemory};

    const N: usize = 48;
    const L: f64 = 1.0;
    const RADIUS: f64 = 0.3;
    const CHI: f64 = 0.7;
    const CS: f64 = 0.5;

    type SimIso = SimState<IsoNewtonian, 2, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;

    #[test]
    fn an_isothermal_sink_removes_mass_and_its_dye_together() {
        let dx = 2.0 * L / N as f64;
        let sim = SimIso::build(IsoNewtonian, Isothermal { cs: CS }, Cartesian)
            .cells([N, N])
            .origin([-L, -L])
            .spacing([dx, dx])
            .boundaries(Boundaries::uniform(BoundaryType::Outflow))
            .cfl(0.3)
            .allocate()
            .expect("sim")
            .set_initial(|_| {
                PrimG::<f64, 2, IsoModel>::isothermal(Density(1.0), Tensor::new([0.0, 0.0]))
            })
            .build()
            .with_bodies(
                BodyCollection::new().add(
                    // gravitating for the same reason as the adiabatic arm: the spherical
                    // drain rate is k_drain*sqrt(GM/r_acc^3), and the pull never acts here.
                    Body::black_hole(
                        0,
                        Tensor::new([0.0, 0.0]),
                        Tensor::new([0.0, 0.0]),
                        1.0,
                        RADIUS,
                        RADIUS,
                        0.0,
                        1.0,
                        RADIUS,
                    )
                    .with_surface(SurfaceSpec::Drain),
                ),
            )
            .with_passive_scalar()
            .expect("chi alloc");

        let cons_chi = sim.fields.cons.chi_field().expect("cons chi");
        let prim_chi = sim.fields.prim.chi_field().expect("prim chi");
        for c in sim.geom.allocated.clone().iter() {
            let rho = *sim.fields.cons.den.view().at(c);
            cons_chi.view_mut().set(c, rho * CHI);
            prim_chi.view_mut().set(c, CHI);
        }

        let mass = |s: &SimIso| -> f64 {
            s.geom
                .interior
                .iter()
                .map(|c| *s.fields.cons.den.view().at(c))
                .sum()
        };
        let mass0 = mass(&sim);
        for _ in 0..40 {
            dispatch_penalize(&sim, 1e-3, 1.0, 1.0, 3.0);
        }
        let swallowed = (mass0 - mass(&sim)) / mass0;
        assert!(
            swallowed > 1e-3,
            "the isothermal sink drained only {swallowed:e} of the mass; the gate is vacuous"
        );

        let (mut worst, mut worst_at) = (0.0_f64, [0isize; 2]);
        for c in sim.geom.interior.iter() {
            let rho = *sim.fields.cons.den.view().at(c);
            if rho <= 0.0 {
                continue;
            }
            let err = (*cons_chi.view().at(c) / rho - CHI).abs();
            if err > worst {
                worst = err;
                worst_at = c;
            }
        }
        assert!(
            worst < 1e-12,
            "the isothermal sink changed the dye concentration by {worst:e} at {worst_at:?} \
             (seeded {CHI})"
        );
    }
}
