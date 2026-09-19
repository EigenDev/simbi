// =============================================================================
// decomp_fofc_one_sided.rs
//
// the first-order flux correction when trouble sits against a tile cut on one
// side only. the correction keeps the first-order flux on every face of a
// troubled cell, and a face on a cut belongs to two tiles: both must take the
// same flux there, so the clean neighbor has to learn of the trouble and redo
// its side of the face. a neighbor that keeps its high-order flux leaves the
// pair with two different fluxes on one face, which creates or destroys mass
// on the cut and departs from the one-tile run.
//
// two cold streams part at mach 50 about a line two cells left of the cut. the
// evacuated cells between them fail the high-order recovery; the first of them
// lie in the left tile's column against the cut while the right tile's
// recovery is clean everywhere. the flow is uniform along y, so every row of
// the cut carries the same one-sided event.
//
// the gate asserts its premise (a stage with flags on exactly one side of the
// cut, and the freeze tier silent so the corrected update is conservative),
// then bitwise equality with the one-tile run and conservation of the total
// mass under periodic boundaries.
//
// run: cargo test -p symbi --test decomp_fofc_one_sided
// =============================================================================

use std::ops::ControlFlow;
use symbi::regimes::fofc::{fofc_reset_stats, fofc_stats};
use symbi::regimes::substrate_rhd::RhdSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, Partition, Schedule, Topology, evolve_scheduled, flatten, unflatten};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::Rhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 4.0 / 3.0;
const CFL: f64 = 0.4;
const NX: usize = 64;
const NY: usize = 4;
const DX: f64 = 1.0 / NX as f64;
const CUT: usize = NX / 2;
// the parting line, two cells left of the cut.
const X_PART: f64 = (NX - 1) as f64 * DX;
const STEPS: u64 = 16;
// the step by which the left tile's cut column has been corrected twice with the right tile
// clean, and before the freeze tier has acted in either arm.
const EARLY: u64 = 7;

type Sim = SimState<Rhd, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = RhdSubstrateKernelSet<HostMemory, f64, 2>;

fn make(cells: [usize; 2], origin: [f64; 2], bnd: Boundaries<2>) -> (Sim, Kern) {
    let sim = Sim::build(Rhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 2])
        .origin(origin)
        .boundaries(bnd)
        .timestepping(Timestepping::Euler)
        .allocate()
        .expect("sim construction failed")
        .set_initial(|[x, _]| {
            // the streams meet on X_PART and part half a domain away.
            let toward = if (x - X_PART).rem_euclid(1.0) >= 0.5 { 0.99 } else { -0.96 };
            Prim::adiabatic(Density(1.0), Tensor::new([toward, 0.0]), Pressure(1e-3))
        })
        .build();
    let mut k = Kern::new(GAMMA, CFL, &sim.geom.allocated);
    k.theta = 1.5;
    (sim, k)
}

fn partition_tiles(partition: &Partition<2>) -> Vec<(Sim, Kern)> {
    let counts = partition.counts();
    (0..partition.n_tiles())
        .map(|flat| {
            let tc = unflatten(flat, counts);
            let ext = partition.tile_extents(tc);
            let bnd = Boundaries(std::array::from_fn(|a| {
                let kind = if counts[a] > 1 {
                    BoundaryType::CoarseFine
                } else {
                    BoundaryType::Periodic
                };
                [kind, kind]
            }));
            make(
                [ext[0].1, ext[1].1],
                [ext[0].0 as f64 * DX, ext[1].0 as f64 * DX],
                bnd,
            )
        })
        .collect()
}

/// the flags in a tile's interior column `offset` cells in from its `side` on axis 0.
fn column_flags(sim: &FieldStore<2, 2, HostMemory, f64>, hi_side: bool) -> u64 {
    let interior = &sim.geom.interior;
    let (xlo, xhi) = (interior.spaces[0].lo, interior.spaces[0].hi);
    let col = if hi_side { xhi - 1 } else { xlo };
    let view = sim.workspace.fofc_flag.view();
    interior
        .iter()
        .filter(|c| c[0] == col)
        .filter(|c| *view.at(*c) != 0.0)
        .count() as u64
}

struct Run {
    den: Vec<f64>,
    fallbacks: u64,
    freezes: u64,
    /// steps whose stage flagged the left tile's cut column and left the right tile's clean
    left_only_steps: u64,
    /// the total mass and the freeze count at the end of step `EARLY`
    early_mass: f64,
    early_freezes: u64,
}

fn run(partition: &Partition<2>) -> Run {
    let mut tiles = partition_tiles(partition);
    let counts = partition.counts();
    let devices = vec![0i32; tiles.len()];
    let mut left_only_steps = 0u64;
    let (mut early_mass, mut early_freezes) = (f64::NAN, u64::MAX);
    fofc_reset_stats();
    {
        let mut stores = Vec::new();
        let mut kernels = Vec::new();
        for (s, k) in tiles.iter_mut() {
            stores.push(&mut **s);
            kernels.push(&*k);
        }
        let schedule = Schedule::derive(counts, stores[0].geom.ng, &Topology::wrapping([true; 2]));
        evolve_scheduled(
            &mut stores,
            &kernels,
            &schedule,
            &devices,
            Timestepping::Euler,
            0.0,
            f64::MAX,
            1,
            &LocalCopy,
            |step, _, stores| {
                if stores.len() == 2 {
                    let left = column_flags(stores[0], true);
                    let right = column_flags(stores[1], false);
                    if left > 0 && right == 0 {
                        left_only_steps += 1;
                    }
                }
                if step == EARLY {
                    early_mass = stores
                        .iter()
                        .map(|s| {
                            let den = s.fields.cons.den.view();
                            s.geom.interior.iter().map(|c| *den.at(c)).sum::<f64>()
                        })
                        .sum();
                    early_freezes = fofc_stats().1;
                }
                if step >= STEPS {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            },
        );
    }
    let (fallbacks, freezes) = fofc_stats();
    let mut den = vec![f64::NAN; NX * NY];
    for (flat_tile, (sim, _)) in tiles.iter().enumerate() {
        let ext = partition.tile_extents(unflatten(flat_tile, counts));
        let ilo: [isize; 2] = std::array::from_fn(|a| sim.geom.interior.spaces[a].lo);
        for c in sim.geom.interior.iter() {
            let g: [usize; 2] = std::array::from_fn(|a| ext[a].0 + (c[a] - ilo[a]) as usize);
            den[flatten(g, [NX, NY])] = *sim.fields.cons.den.view().at(c);
        }
    }
    Run {
        den,
        fallbacks,
        freezes,
        left_only_steps,
        early_mass,
        early_freezes,
    }
}

#[test]
fn trouble_on_one_side_of_a_cut_matches_the_one_tile_run() {
    let whole = Partition::explicit([NX, NY], [Vec::new(), Vec::new()]).expect("one tile");
    let split = Partition::explicit([NX, NY], [vec![CUT], Vec::new()]).expect("one cut on x");

    let mono = run(&whole);
    let dec = run(&split);

    assert!(mono.fallbacks > 0, "the one-tile run never tripped the correction");
    assert!(
        dec.left_only_steps >= 2,
        "{} steps flagged the left of the cut with the right clean; the gate is vacuous",
        dec.left_only_steps
    );
    // the corrected update telescopes only while the freeze tier is silent, so conservation
    // is asserted at the early step, where neither arm has frozen a cell.
    assert_eq!(
        (mono.early_freezes, dec.early_freezes),
        (0, 0),
        "the freeze tier acted before the early step"
    );
    let initial_mass = mono_initial_mass();
    let drift = |m: f64| ((m - initial_mass) / initial_mass).abs();
    assert!(drift(mono.early_mass) < 1e-13, "one-tile mass drift {:e}", drift(mono.early_mass));
    assert!(
        drift(dec.early_mass) < 1e-13,
        "two-tile mass drift {:e}: the cut face carried two different fluxes",
        drift(dec.early_mass)
    );
    let differing = mono.den.iter().zip(&dec.den).filter(|(a, b)| a != b).count();
    assert_eq!(differing, 0, "{differing} cells differ from the one-tile run");
    assert_eq!(
        (mono.fallbacks, mono.freezes),
        (dec.fallbacks, dec.freezes),
        "the two arms booked different correction censuses"
    );
}

/// the total rest-mass density D of the initial data, summed over the one-tile grid.
fn mono_initial_mass() -> f64 {
    let whole = Partition::explicit([NX, NY], [Vec::new(), Vec::new()]).expect("one tile");
    let tiles = partition_tiles(&whole);
    let sim = &tiles[0].0;
    let den = sim.fields.cons.den.view();
    sim.geom.interior.iter().map(|c| *den.at(c)).sum()
}

/// the flag exchange writes the ghost layer next to the interior and rewrites it on every
/// call: a flag a neighbor raised in one stage is gone from the cut ghosts once the neighbor
/// is clean, and the outer ghost layers of the wider allocation stay untouched.
#[test]
fn flags_cross_into_the_adjacent_ghost_layer_and_are_replaced_when_the_neighbor_clears() {
    use symbi::sim::decomp::exchange_trouble;
    let split = Partition::explicit([NX, NY], [vec![CUT], Vec::new()]).expect("one cut on x");
    let tiles = partition_tiles(&split);
    let counts = split.counts();
    let stores: Vec<&FieldStore<2, 2, HostMemory, f64>> = tiles.iter().map(|(s, _)| &**s).collect();
    let schedule = Schedule::derive(counts, stores[0].geom.ng, &Topology::wrapping([true; 2]));
    assert!(stores[0].geom.ng >= 2, "one ghost layer leaves no outer layer to distinguish");

    let left = &stores[0].geom.interior;
    let right = &stores[1].geom.interior;
    let row = left.spaces[1].lo + 1;
    let source = [left.spaces[0].hi - 1, row];
    let adjacent = [right.spaces[0].lo - 1, row];
    let outer = [right.spaces[0].lo - 2, row];
    let flag_at = |tile: usize, c: [isize; 2]| *stores[tile].workspace.fofc_flag.view().at(c);

    *stores[0].workspace.fofc_flag.view_mut().at_mut(source) = 1.0;
    let received = exchange_trouble(&stores, &schedule, &[0, 0], &LocalCopy);
    assert_eq!(flag_at(1, adjacent), 1.0, "the neighbor's flag never reached the adjacent ghost");
    assert_eq!(flag_at(1, outer), 0.0, "the flag landed in the outer ghost layer");
    assert!(received[1].iter().any(|r| r.contains(adjacent)));

    *stores[0].workspace.fofc_flag.view_mut().at_mut(source) = 0.0;
    exchange_trouble(&stores, &schedule, &[0, 0], &LocalCopy);
    assert_eq!(flag_at(1, adjacent), 0.0, "a cleared flag survived in the cut ghost");
}
