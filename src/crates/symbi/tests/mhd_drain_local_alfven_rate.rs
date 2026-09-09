// =============================================================================
// mhd_drain_local_alfven_rate.rs
//
// the material drain of a magnetized sink relaxes each cell at its own fast-magnetosonic
// crossing rate `sqrt(c_s^2 + |B|^2/rho) / dx` (lifted by the free-fall rate). two properties pin
// it against a domain-wide stiffness:
//   - locality: the drain factor inside the sink depends on the sink cells' own field alone, so a
//     strong field patch far outside the mask leaves the sink's drain bitwise unchanged.
//   - bounded evacuation: a uniform field threading the sink is left in place by the drain, so a
//     rate that grows with the least density anywhere makes every sink cell drain at the emptiest
//     cell's rate and the whole interior evacuates in the finite time ~ 2 dx sqrt(rho_0) / |B|
//     while the field stays. the local rate lets the inflow feed the interior and the density
//     inside the sink settles far above the evacuation floor.
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
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;

const N: usize = 16;
const GAMMA: f64 = 5.0 / 3.0;
const BODY: [f64; 3] = [0.5, 0.5, 0.5];
const R_BODY: f64 = 0.2;
const B0: f64 = 1.0;
// the far patch: the corner block of PATCH cells per axis, at least 0.3 from the sink surface.
const PATCH: usize = 3;
const B_PATCH: f64 = 40.0 * B0;

fn in_patch(dx: f64, [x, y, z]: [f64; 3]) -> bool {
    let edge = PATCH as f64 * dx;
    x < edge && y < edge && z < edge
}

// a uniform field along x threading a draining point mass at the box center; `patched` adds the
// strong far-corner block to the cell and face fields.
fn build(patched: bool) -> Sim {
    let dx = 1.0 / N as f64;
    let bx = move |p: [f64; 3]| {
        if patched && in_patch(dx, p) {
            B_PATCH
        } else {
            B0
        }
    };
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
    .set_initial(move |p| {
        MhdPrim::new(
            Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)),
            Tensor::new([bx(p), 0.0, 0.0]),
        )
    })
    .seed_faces(move |axis, p| if axis == 0 { bx(p) } else { 0.0 })
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

fn kernels(sim: &Sim) -> NewtonianMhdSubstrateKernelSet3D<HostMemory, f64> {
    NewtonianMhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, 0.3, 1.0, &sim.geom.allocated)
}

// (cell, density) over the interior cells inside the sink radius.
fn sink_density(sim: &Sim) -> Vec<([isize; 3], f64)> {
    let dx = sim.geom.dx[0];
    let den = sim.fields.cons.den.view();
    sim.geom
        .interior
        .iter()
        .filter_map(|c| {
            let r2: f64 = (0..3)
                .map(|a| (sim.geom.x_lo[a] + (c[a] as f64 + 0.5) * dx - BODY[a]).powi(2))
                .sum();
            (r2.sqrt() < R_BODY).then(|| (c, *den.at(c)))
        })
        .collect()
}

fn min_internal_energy(sim: &Sim) -> f64 {
    let den = sim.fields.cons.den.view();
    let mom: Vec<_> = sim.fields.cons.mom.iter().map(|m| m.view()).collect();
    let nrg = sim
        .fields
        .cons
        .nrg_field()
        .expect("adiabatic energy slot")
        .view();
    let bcell: Vec<_> = sim
        .fields
        .mhd
        .as_ref()
        .expect("mhd fields")
        .bcell
        .b
        .iter()
        .map(|b| b.view())
        .collect();
    sim.geom
        .interior
        .iter()
        .map(|c| {
            let rho = *den.at(c);
            let ke = 0.5 * mom.iter().map(|m| m.at(c).powi(2)).sum::<f64>() / rho;
            let mag = 0.5 * bcell.iter().map(|b| b.at(c).powi(2)).sum::<f64>();
            *nrg.at(c) - ke - mag
        })
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn a_far_field_patch_leaves_the_sink_drain_bitwise_unchanged() {
    let plain = build(false);
    let patched = build(true);
    kernels(&plain).penalize(&plain, 1e-3);
    kernels(&patched).penalize(&patched, 1e-3);
    let a = sink_density(&plain);
    let b = sink_density(&patched);
    assert!(!a.is_empty(), "the sink radius covers no interior cell");
    assert!(
        a.iter().any(|(_, d)| *d < 1.0),
        "the drain removed nothing inside the sink; the locality check is vacuous"
    );
    for ((c, da), (_, db)) in a.iter().zip(&b) {
        assert!(
            da.to_bits() == db.to_bits(),
            "sink cell {c:?} drains at a rate set by the far field patch: {da:.17e} (plain) vs {db:.17e} (patched)"
        );
    }
}

#[test]
fn a_threaded_sink_settles_instead_of_evacuating() {
    let mut sim = build(false);
    let sub = kernels(&sim);
    // the domain-wide rate empties the sink by t ~ 2 dx sqrt(rho_0) / B0 = 0.125; the horizon is
    // several of those.
    evolve_with_callback(&mut sim, &sub, 0.6, u64::MAX, |_| {})
        .expect("the threaded sink went inadmissible");
    let least = sink_density(&sim)
        .iter()
        .map(|(_, d)| *d)
        .fold(f64::INFINITY, f64::min);
    let e_int = min_internal_energy(&sim);
    eprintln!(
        "least sink density {least:.3e}, least internal energy {e_int:.3e}, t = {:.3}",
        sim.time
    );
    assert!(
        e_int > 0.0,
        "the least internal energy density after the run is {e_int:.3e}"
    );
    // the domain-wide rate leaves the sink interior near 1e-12 of the ambient density with the
    // field intact; the bound sits five decades above that floor and three below the ambient gas.
    assert!(
        least >= 1e-3,
        "the sink interior evacuated to {least:.3e} of the ambient density under a field it leaves in place"
    );
}
