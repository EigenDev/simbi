// =============================================================================
// drain_leaves_primitives_current.rs
//
// the immersed-body drain runs once per step after the full RK combination, and the stored
// primitive state is recovered from the drained conserved state before the step ends: the next
// step's reconstruction and a checkpoint written at the step boundary read primitives that match
// the conserved fields. a magnetized sink pins it: after one step every interior cell's stored
// primitive density equals its conserved density bitwise, sink cells included.
// =============================================================================

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet3D;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, MagneticSpec, SurfaceSpec};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;

const N: usize = 16;
const GAMMA: f64 = 5.0 / 3.0;
const BODY: [f64; 3] = [0.5, 0.5, 0.5];
const R_BODY: f64 = 0.2;
const B0: f64 = 0.5;

fn build() -> Sim {
    let dx = 1.0 / N as f64;
    let sim = SimStateGeneric::<
        NewtonianMhd,
        3,
        3,
        Cartesian,
        IdealGas<f64>,
        CpuSpace,
        HostMemory,
        f64,
    >::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
    .cells([N, N, N])
    .origin([0.0, 0.0, 0.0])
    .spacing([dx, dx, dx])
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .cfl(0.3)
    .allocate()
    .expect("sim construction")
    .set_initial(move |_p| {
        MhdPrim::new(
            Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)),
            Tensor::new([B0, 0.0, 0.0]),
        )
    })
    .seed_faces(move |axis, _p| if axis == 0 { B0 } else { 0.0 })
    .build();
    sim.with_bodies(
        BodyCollection::new().add(
            Body::black_hole(
                0,
                Tensor::new(BODY),
                Tensor::zeros(),
                1.0,
                R_BODY,
                0.05,
                1.0,
                1.0,
                R_BODY,
            )
            .with_surface(SurfaceSpec::Drain)
            .with_magnetic(MagneticSpec::None),
        ),
    )
}

#[test]
fn the_stored_primitives_match_the_drained_conserved_state_after_a_step() {
    let mut sim = build();
    let sub = NewtonianMhdSubstrateKernelSet3D::<HostMemory, f64>::new(
        GAMMA,
        0.3,
        1.0,
        &sim.geom.allocated,
    );
    evolve_with_callback(&mut sim, &sub, 1e-3, u64::MAX, |_| {})
        .expect("the drained step went inadmissible");
    assert!(sim.iteration >= 1, "the evolve loop took no step");
    let den = sim.fields.cons.den.view();
    let rho = sim.fields.prim.rho.view();
    let drained = sim.geom.interior.iter().any(|c| *den.at(c) < 1.0);
    assert!(
        drained,
        "the sink removed nothing; the primitive check is vacuous"
    );
    let mut stale = Vec::new();
    for c in sim.geom.interior.iter() {
        if den.at(c).to_bits() != rho.at(c).to_bits() {
            stale.push((c, *rho.at(c), *den.at(c)));
        }
    }
    assert!(
        stale.is_empty(),
        "{} interior cells end the step with a stored primitive density behind the conserved one; first: cell {:?} rho_prim = {:.17e} rho_cons = {:.17e}",
        stale.len(),
        stale[0].0,
        stale[0].1,
        stale[0].2
    );
}
