// =============================================================================
// resistive_cfl_curvilinear.rs
//
// the resistive CFL must bound dt by the smallest physical cell, `dt <= min(h_a dx_a)^2 / (2 D eta)`,
// not the coordinate spacing `dx`. on a near-pole spherical grid the physical azimuthal cell
// `h_phi dphi = r sin(theta) dphi` is much smaller than `dphi`, so the coordinate-width bound would
// pick a dt many times too large -> the explicit resistive diffusion amplifies the pole modes -> NaN.
// this pins that the fold uses the physical width: the returned dt sits far below the coordinate bound.
// =============================================================================

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Spherical;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::state::Prim;
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<NewtonianMhd, 3, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;

const N: usize = 12;
const GAMMA: f64 = 5.0 / 3.0;
const ETA: f64 = 1.0; // strong, so the resistive rate dominates the CFL (isolates it from the waves)
const CFL: f64 = 0.3;

#[test]
fn resistive_cfl_uses_the_physical_cell_width() {
    // r in [1,2], theta in [0.1, 0.4] (near the pole -> sin theta ~ 0.1), phi in [0, 0.5].
    let sim = Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Spherical)
        .cells([N, N, N])
        .bounds([1.0, 0.1, 0.0], [2.0, 0.4, 0.5])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .expect("3d spherical sim")
        .set_initial(|_| MhdPrim {
            hydro: Prim {
                rho: 1.0,
                vel: Tensor::new([0.0, 0.0, 0.0]),
                pre: 1.0,
            },
            mag: Tensor::new([0.0, 0.0, 0.0]),
        })
        .seed_faces(|_, _| 0.0)
        .build();

    let sub = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 3>::new(
        GAMMA,
        CFL,
        1.0,
        &sim.geom.allocated,
    )
    .with_resistivity(ETA);
    let dt = sub.cfl(&sim);

    // the resistive dt bound formed from the coordinate width, which the physical bound must beat.
    let dx_min = sim.geom.dx.iter().copied().fold(f64::INFINITY, f64::min);
    let dt_coord = CFL * dx_min * dx_min / (2.0 * 3.0 * ETA);

    assert!(
        dt.is_finite() && dt > 0.0,
        "resistive cfl returned a non-physical dt = {dt}"
    );
    // the near-pole physical cell is ~5x smaller than dtheta, so the physical bound is >20x tighter.
    assert!(
        dt < 0.1 * dt_coord,
        "resistive cfl did not use the physical cell width: dt = {dt:.3e}, coordinate-width bound \
         dt_coord = {dt_coord:.3e} (expected dt << dt_coord on a near-pole spherical grid)"
    );
}
