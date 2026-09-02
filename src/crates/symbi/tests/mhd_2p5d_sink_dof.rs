// =============================================================================
// mhd_2p5d_sink_dof.rs
//
// the DOF-aware immersed-body drain under 2.5D MHD (D=2 grid, DOF=3 momentum). the penalize kernel
// drains every momentum component, the out-of-plane one included (mom[2], the v_z the 2.5D plane
// carries). draining the in-plane pair alone would leave the out-of-plane momentum standing
// while the density is evacuated, so its velocity v_z = mom[2]/den runs away at the sink. the sink is
// selected via the `_dof3` kernel (the dispatch appends `_dof{DOF}` when DOF != D).
// =============================================================================

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, SurfaceSpec};
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;

const N: usize = 32;
const GAMMA: f64 = 5.0 / 3.0;
const VZ: f64 = 0.3; // uniform out-of-plane velocity (the 2.5D swirl component)
const BODY: [f64; 2] = [0.5, 0.5];
const R_BODY: f64 = 0.2;

fn make() -> Sim {
    let dx = 1.0 / N as f64;
    let sim = SimStateGeneric::<
        NewtonianMhd,
        2,
        3,
        Cartesian,
        IdealGas<f64>,
        CpuSpace,
        HostMemory,
        f64,
    >::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
    .cells([N, N])
    .origin([0.0, 0.0])
    .spacing([dx, dx])
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .cfl(0.3)
    .allocate()
    .expect("2.5D MHD sink sim construction failed")
    // uniform still gas with an out-of-plane velocity v_z; tiny uniform in-plane B (div-free).
    .set_initial(|_| {
        MhdPrim::new(
            Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, VZ]), Pressure(1.0)),
            Tensor::new([1e-3, 0.0, 0.0]),
        )
    })
    .seed_faces(|axis, _| if axis == 0 { 1e-3 } else { 0.0 })
    .build();
    sim.with_bodies(
        BodyCollection::new().add(
            // gravitating because the spherical drain rate is k_drain*sqrt(GM/r_acc^3);
            // the pull never acts here, since the gravity source is not dispatched.
            Body::black_hole(
                0,
                Tensor::new(BODY),
                Tensor::zeros(),
                1.0,
                R_BODY,
                R_BODY,
                0.0,
                1.0,
                R_BODY,
            )
            .with_surface(SurfaceSpec::Drain),
        ),
    )
}

// (inside-mask, far-field) sums of the out-of-plane momentum |mom[2]| over the interior. "far" is
// several cells beyond the mask, clear of the mollified tanh tail (width ~one cell).
fn out_of_plane_momentum(s: &Sim) -> (f64, f64) {
    let dx = s.geom.dx[0];
    let momz = &s.fields.cons.mom[2];
    s.geom.interior.iter().fold((0.0, 0.0), |(inside, far), c| {
        let px = s.geom.x_lo[0] + (c[0] as f64 + 0.5) * dx;
        let py = s.geom.x_lo[1] + (c[1] as f64 + 0.5) * dx;
        let r = ((px - BODY[0]).powi(2) + (py - BODY[1]).powi(2)).sqrt();
        let m = momz.view().at(c).abs();
        if r < R_BODY {
            (inside + m, far)
        } else if r > R_BODY + 8.0 * dx {
            (inside, far + m)
        } else {
            (inside, far)
        }
    })
}

#[test]
fn drain_removes_out_of_plane_momentum() {
    let sim = make();
    let (in0, out0) = out_of_plane_momentum(&sim);
    assert!(
        in0 > 0.0,
        "the seed must carry out-of-plane momentum for the test to bite"
    );

    let sub = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(
        GAMMA,
        0.3,
        1.0,
        &sim.geom.allocated,
    );
    // a handful of drain steps at a modest dt so the mask accumulates a visible evacuation.
    for _ in 0..20 {
        sub.penalize(&sim, 1e-3);
    }
    let (in1, out1) = out_of_plane_momentum(&sim);

    // the out-of-plane momentum inside the mask is drained substantially...
    assert!(
        in1 < 0.9 * in0,
        "the 2.5D MHD drain did not remove the out-of-plane momentum: inside {in0:.5} -> {in1:.5}"
    );
    // ...while outside the mask it is untouched (the drain is local, and mom[2] is a real conserved
    // channel).
    assert!(
        (out1 - out0).abs() < 1e-4 * out0,
        "the drain perturbed the out-of-plane momentum far from the mask: {out0:.5} -> {out1:.5}"
    );
}
