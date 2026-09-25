// =============================================================================
// worker_refined_refusals.rs
//
// the launched refinement's scope for register transport: a flux-register unit
// crossing a cut rides the constant-area register kernels of a uniform
// cartesian root, so a curvilinear root whose patch face lies on a cut is
// refused by name at entry, while the same layout on a cartesian root is
// admitted. each gate builds the tile holding the fine faces of a patch face
// on a cut and asks the driver's scope check directly.
//
// run: cargo test -p symbi --test worker_refined_refusals
// =============================================================================
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::launched::refuse_transported_registers;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_sim::worker::WorkerError;
use symbi_algebra::Tensor;
use symbi_geometry::{Cartesian, Cylindrical};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
/// the root halo depth that holds the PPM prolongation stencil beyond a real face on a cut
const NG: usize = 5;
/// the tile: 32 cells on axis 0 ending at the cut, 32 on axis 1
const CELLS: [usize; 2] = [32, 32];
const DX: f64 = 1.0 / 32.0;

fn tile_boundaries() -> Boundaries<2> {
    // the hi face on axis 0 is the cut; every other face is physical
    Boundaries(std::array::from_fn(|a| [BoundaryType::Outflow, if a == 0 { BoundaryType::CoarseFine } else { BoundaryType::Outflow }]))
}

/// the patch on this tile, whose hi face on axis 0 coincides with the cut
fn regions(x_lo: [f64; 2]) -> (RefinementRegion<2>, RefinementRegion<2>) {
    let lo = [x_lo[0] + 12.0 * DX, x_lo[1] + 12.0 * DX];
    let hi = [x_lo[0] + 32.0 * DX, x_lo[1] + 20.0 * DX];
    let clipped = RefinementRegion { x_lo: lo, x_hi: hi };
    (clipped.clone(), clipped)
}

type CartSim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type CylSim = SimStateGeneric<Newtonian, 2, 3, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

fn cartesian_tile() -> Hierarchy<Newtonian, 2, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kern> {
    let root = CartSim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(CELLS)
        .spacing([DX; 2])
        .origin([0.0, 0.0])
        .boundaries(tile_boundaries())
        .cfl(CFL)
        .ghosts(NG)
        .allocate()
        .expect("cartesian root")
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0]), Pressure(1.0)))
        .build();
    let (clipped, whole) = regions([0.0, 0.0]);
    let k = Kern::new(GAMMA, CFL, &root.geom.allocated);
    Hierarchy::with_refinement_of_contributing(root, k, &[clipped], &[whole], ProlongOrder::Ppm, |s| {
        Kern::new(GAMMA, CFL, &s.geom.allocated)
    })
    .expect("cartesian tile")
}

fn cylindrical_tile() -> Hierarchy<Newtonian, 2, 3, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory, Kern> {
    let origin = [1.0, 0.0];
    let root = CylSim::build(Newtonian, IdealGas { gamma: GAMMA }, Cylindrical)
        .cells(CELLS)
        .spacing([DX; 2])
        .origin(origin)
        .boundaries(tile_boundaries())
        .cfl(CFL)
        .ghosts(NG)
        .allocate()
        .expect("cylindrical root")
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)))
        .build();
    let (clipped, whole) = regions(origin);
    let k = Kern::new(GAMMA, CFL, &root.geom.allocated);
    Hierarchy::with_refinement_of_contributing(root, k, &[clipped], &[whole], ProlongOrder::Ppm, |s| {
        Kern::new(GAMMA, CFL, &s.geom.allocated)
    })
    .expect("cylindrical tile")
}

#[test]
fn a_transported_register_on_a_curvilinear_chart_is_refused_by_name() {
    let tile = cylindrical_tile();
    assert!(tile.transports_registers(), "the patch face on the cut yields no contributing face; the gate is vacuous");
    match refuse_transported_registers(std::slice::from_ref(&tile)) {
        Err(WorkerError::Refused(reason)) => {
            assert!(reason.contains("curvilinear chart"), "the refusal names the chart: {reason}");
        }
        other => panic!("the curvilinear transport was admitted: {other:?}"),
    }
}

#[test]
fn a_transported_register_on_a_cartesian_root_is_admitted() {
    let tile = cartesian_tile();
    assert!(tile.transports_registers(), "the patch face on the cut yields no contributing face; the gate is vacuous");
    refuse_transported_registers(std::slice::from_ref(&tile)).expect("the cartesian transport is admitted");
}
