// =============================================================================
// tracer_owner_addressing.rs
//
// a discrete tracer's owner is a container address, and the transport reads that
// address back to find the cell: `container_cell` inverts it, `face_destination`
// steps it to a neighbor, and the derived position is the addressed cell's
// centroid. so the address a seeded tracer carries must name the cell its
// position sits in.
//
// the addresses run the first axis fastest. a domain iterator walks the last axis
// fastest, so the two orders are transposes of each other, and on a square grid a
// transposed address is still a legal cell -- it names the wrong one silently.
// these grids are deliberately unequal along every axis, where a transposed
// address is out of range or lands on a demonstrably different cell.
//
// run: cargo test -p symbi --test tracer_owner_addressing
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::state::*;
use symbi::sim::tracers::{cell_container_address, seed_mass_weighted};
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;

type Sim2 = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern2 = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
type Sim3 = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern3 = AdiabaticSubstrateKernelSet<HostMemory, f64, 3>;

/// the container address of a global cell: first axis fastest over the cell counts.
fn address<const D: usize>(cell: [usize; D], cells: [usize; D]) -> usize {
    let mut linear = 0;
    let mut stride = 1;
    for dd in 0..D {
        linear += cell[dd] * stride;
        stride *= cells[dd];
    }
    linear
}

/// the cell a position sits in, from the grid origin and widths.
fn cell_at<const D: usize>(x: [f64; D], dx: [f64; D]) -> [usize; D] {
    std::array::from_fn(|dd| (x[dd] / dx[dd]).floor() as usize)
}

/// a density that varies along every axis, so the mass-weighted seeding spreads
/// tracers over the whole grid instead of concentrating them on one slab.
fn ramp<const D: usize>(x: [f64; D]) -> f64 {
    1.0 + x.iter().enumerate().map(|(dd, v)| (dd + 1) as f64 * v).sum::<f64>()
}

#[test]
fn a_seeded_tracer_in_2d_is_addressed_to_the_cell_it_sits_in() {
    const CELLS: [usize; 2] = [8, 4];
    let dx = [1.0 / CELLS[0] as f64, 1.0 / CELLS[1] as f64];
    let sim = Sim2::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(CELLS)
        .spacing(dx)
        .origin([0.0, 0.0])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .timestepping(Timestepping::Euler)
        .allocate()
        .expect("sim construction failed")
        .set_initial(|x| Prim {
            rho: ramp(x),
            vel: Tensor::new([0.0, 0.0]),
            pre: 1.0,
        })
        .build();
    let _k = Kern2::new(GAMMA, 0.4, &sim.geom.allocated);

    let set = seed_mass_weighted(&sim, 64);
    assert!(!set.id.is_empty(), "the seeding produced no tracers");

    // the seeding must reach both halves of the long axis, or a transposed address
    // could still fall inside the occupied range and the check would be vacuous.
    let spread = set
        .x
        .iter()
        .filter(|x| cell_at(**x, dx)[0] >= CELLS[0] / 2)
        .count();
    assert!(
        spread > 0 && spread < set.x.len(),
        "every tracer landed on one side of the grid; the addressing check is vacuous"
    );

    for ii in 0..set.id.len() {
        let cell = cell_at(set.x[ii], dx);
        let (level, linear) =
            cell_container_address(set.owner[ii]).expect("a seeded tracer owns a cell");
        assert_eq!(level, 0, "tracer {ii} was seeded onto a refined level");
        assert_eq!(
            linear,
            address(cell, CELLS),
            "tracer {ii} at {:?} sits in cell {cell:?} but is addressed to cell {:?}",
            set.x[ii],
            {
                let mut rest = linear;
                let decoded: [usize; 2] = std::array::from_fn(|dd| {
                    let v = rest % CELLS[dd];
                    rest /= CELLS[dd];
                    v
                });
                decoded
            }
        );
    }
}

#[test]
fn a_seeded_tracer_in_3d_is_addressed_to_the_cell_it_sits_in() {
    const CELLS: [usize; 3] = [4, 3, 2];
    let dx = [
        1.0 / CELLS[0] as f64,
        1.0 / CELLS[1] as f64,
        1.0 / CELLS[2] as f64,
    ];
    let sim = Sim3::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(CELLS)
        .spacing(dx)
        .origin([0.0; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .timestepping(Timestepping::Euler)
        .allocate()
        .expect("sim construction failed")
        .set_initial(|x| Prim {
            rho: ramp(x),
            vel: Tensor::new([0.0, 0.0, 0.0]),
            pre: 1.0,
        })
        .build();
    let _k = Kern3::new(GAMMA, 0.4, &sim.geom.allocated);

    let set = seed_mass_weighted(&sim, 96);
    assert!(!set.id.is_empty(), "the seeding produced no tracers");

    // unequal counts on all three axes: any rotation of the stride order lands elsewhere.
    for ii in 0..set.id.len() {
        let cell = cell_at(set.x[ii], dx);
        let (level, linear) =
            cell_container_address(set.owner[ii]).expect("a seeded tracer owns a cell");
        assert_eq!(level, 0, "tracer {ii} was seeded onto a refined level");
        assert_eq!(
            linear,
            address(cell, CELLS),
            "tracer {ii} at {:?} sits in cell {cell:?} but is addressed to linear {linear}",
            set.x[ii]
        );
    }
}
