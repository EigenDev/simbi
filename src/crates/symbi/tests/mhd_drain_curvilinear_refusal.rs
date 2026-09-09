// =============================================================================
// mhd_drain_curvilinear_refusal.rs
//
// a magnetized drain relaxes each cell at its local Alfven rate through a kernel baked for the
// cartesian 3D grid and the 2.5D x-y grid. on any other chart the dispatch refuses the body
// loudly instead of relaxing it at a domain-wide rate; the cylindrical r-z section with an
// on-axis draining point mass pins the refusal.
// =============================================================================

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cylindrical;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, MagneticSpec, SurfaceSpec};
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory>;

const GAMMA: f64 = 5.0 / 3.0;
const N: usize = 16;

#[test]
#[should_panic(expected = "a magnetized drain relaxes at its local Alfven rate")]
fn a_magnetized_drain_on_the_cylindrical_section_is_refused() {
    let dx = 1.0 / N as f64;
    let sim = Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cylindrical)
        .cells([N, N])
        .origin([0.0, -0.5])
        .spacing([dx, dx])
        .cfl(0.3)
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("cyl r-z sim construction failed")
        .set_initial(|_| {
            MhdPrim::new(
                Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)),
                Tensor::new([0.0, 0.0, 0.3]),
            )
        })
        .seed_faces_uniform([0.0, 0.3])
        .build()
        .with_bodies(
            BodyCollection::new().add(
                Body::black_hole(
                    0,
                    Tensor::new([0.0, 0.0]),
                    Tensor::zeros(),
                    1.0,
                    0.2,
                    0.05,
                    1.0,
                    1.0,
                    0.2,
                )
                .with_surface(SurfaceSpec::Drain)
                .with_magnetic(MagneticSpec::None),
            ),
        );
    let sub = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(
        GAMMA,
        0.3,
        1.0,
        &sim.geom.allocated,
    );
    sub.penalize(&sim, 1e-3);
}
