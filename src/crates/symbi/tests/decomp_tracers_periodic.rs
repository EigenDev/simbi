// =============================================================================
// decomp_tracers_periodic.rs
//
// discrete tracers on a periodic axis that carries more than one tile. the two
// end tiles are neighbors across the domain seam, so a tracer leaving the last
// cell re-enters at the first: the schedule's wrap legs fill the ghosts, and the
// transport addresses the far-side cell through the layout's own wrap flag rather
// than through a tile's declared faces, which name a cut there.
//
// the contract is exact: the halo exchange reproduces the monolithic stencil bit
// for bit, so the fluid state is identical and tracer transport -- a deterministic
// function of the fluxes, the ids and the keyed hashes -- must place every tracer
// on the same cell as a single-tile periodic run.
//
// run: cargo test -p symbi --test decomp_tracers_periodic
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, Partition, Schedule, Topology, evolve_scheduled, unflatten};
use symbi::sim::state::*;
use symbi::sim::tracers::seed_and_partition;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const N: usize = 64;
const DX: f64 = 1.0 / N as f64;
// long enough at |vx| = 1 for the population to cross the domain seam.
const T_FINAL: f64 = 0.35;
const VX: f64 = 1.0;
const VY: f64 = 0.5;
const N_TRACERS: usize = 512;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

/// the shortest distance to a center along a periodic axis of unit length.
fn wrapped(x: f64, center: f64) -> f64 {
    let d = (x - center).abs();
    d.min(1.0 - d)
}

/// a smooth blob straddling the seam at x = 0, so the mass-weighted seeding puts
/// tracers on both sides of it.
fn blob(x: f64, y: f64) -> f64 {
    let r2 = (wrapped(x, 0.0) / 0.12).powi(2) + (wrapped(y, 0.0) / 0.12).powi(2);
    0.5 * (-r2).exp()
}

fn make(cells: [usize; 2], origin: [f64; 2], bnd: Boundaries<2>) -> (Sim, Kern) {
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 2])
        .origin(origin)
        .boundaries(bnd)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim construction failed")
        .set_initial(|[x, y]| {
            Prim::adiabatic(
                Density(1.0 + blob(x, y)),
                Tensor::new([VX, VY]),
                Pressure(1.0),
            )
        })
        .build();
    let k = Kern::new(GAMMA, CFL, &sim.geom.allocated);
    (sim, k)
}

/// on a cut wrapping axis every face is a cut, the two at the domain seam included:
/// the schedule's wrap legs carry those. an uncut axis keeps its periodic faces,
/// which wrap the single tile onto itself over the whole domain.
fn partition_tiles(partition: &Partition<2>) -> Vec<(Sim, Kern)> {
    let counts = partition.counts();
    (0..partition.n_tiles())
        .map(|flat| {
            let tc = unflatten(flat, counts);
            let ext = partition.tile_extents(tc);
            let bnd = Boundaries(std::array::from_fn(|a| {
                let seam = if counts[a] > 1 {
                    BoundaryType::CoarseFine
                } else {
                    BoundaryType::Periodic
                };
                let lo = if tc[a] == 0 {
                    seam
                } else {
                    BoundaryType::CoarseFine
                };
                let hi = if tc[a] == counts[a] - 1 {
                    seam
                } else {
                    BoundaryType::CoarseFine
                };
                [lo, hi]
            }));
            make(
                [ext[0].1, ext[1].1],
                [ext[0].0 as f64 * DX, ext[1].0 as f64 * DX],
                bnd,
            )
        })
        .collect()
}

/// seed the population once from a whole-domain reference and split it by position,
/// so both arms start from the identical particles.
fn seed(tiles: &mut [(Sim, Kern)], partition: &Partition<2>) {
    let (reference, _) = make(
        [N, N],
        [0.0, 0.0],
        Boundaries::uniform(BoundaryType::Periodic),
    );
    let sets = seed_and_partition(&reference, N_TRACERS, partition);
    for ((sim, _), set) in tiles.iter_mut().zip(sets) {
        sim.tracers = Some(set);
    }
}

fn run(tiles: &mut [(Sim, Kern)], counts: [usize; 2]) {
    let devices = vec![0i32; tiles.len()];
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
        Timestepping::Rk2,
        0.0,
        T_FINAL,
        u64::MAX,
        &LocalCopy,
        |_, _, _| std::ops::ControlFlow::Continue(()),
    );
}

/// every tracer as (owner, position), indexed by id.
fn gather(tiles: &[(Sim, Kern)]) -> Vec<Option<(u64, [f64; 2])>> {
    let mut out = vec![None; N_TRACERS];
    for (sim, _) in tiles {
        let t = sim
            .tracers
            .as_ref()
            .expect("every tile carries a tracer set");
        for ii in 0..t.id.len() {
            out[t.id[ii] as usize] = Some((t.owner[ii].0, t.x[ii]));
        }
    }
    out
}

fn global_density(tiles: &[(Sim, Kern)], partition: &Partition<2>) -> Vec<f64> {
    let counts = partition.counts();
    let mut out = vec![f64::NAN; N * N];
    for (flat, (sim, _)) in tiles.iter().enumerate() {
        let ext = partition.tile_extents(unflatten(flat, counts));
        let ilo: [isize; 2] = std::array::from_fn(|a| sim.geom.interior.spaces[a].lo);
        for c in sim.geom.interior.iter() {
            let g: [usize; 2] = std::array::from_fn(|a| ext[a].0 + (c[a] - ilo[a]) as usize);
            out[g[0] + N * g[1]] = *sim.fields.cons.den.view().at(c);
        }
    }
    out
}

fn assert_matches(cuts: [Vec<usize>; 2]) {
    let split = Partition::explicit([N, N], cuts).expect("interior cuts lie inside the grid");
    let whole = Partition::explicit([N, N], [Vec::new(), Vec::new()])
        .expect("the uncut partition is one tile");

    let mut one = partition_tiles(&whole);
    seed(&mut one, &whole);
    let seeded = gather(&one);
    run(&mut one, whole.counts());
    let mono = gather(&one);

    let mut dec = partition_tiles(&split);
    seed(&mut dec, &split);
    run(&mut dec, split.counts());
    let decomposed = gather(&dec);

    // the exactness claim rests on the two arms seeing the same fluxes.
    let mono_den = global_density(&one, &whole);
    let dec_den = global_density(&dec, &split);
    let differing = mono_den
        .iter()
        .zip(&dec_den)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        differing,
        0,
        "the wrap exchange changed the density field in {differing} of {} cells; the tracer \
         comparison below assumes an identical fluid state",
        N * N
    );

    // a tracer that never reaches the seam proves nothing about the wrap, so require both
    // that the population moved and that some of it crossed the domain edge.
    let cell_x = |owner: u64| (owner & ((1u64 << 56) - 1)) as usize % N;
    let mut moved = 0usize;
    let mut wrapped_seam = 0usize;
    for ii in 0..N_TRACERS {
        let (Some(start), Some(end)) = (seeded[ii], mono[ii]) else {
            continue;
        };
        if start.0 != end.0 {
            moved += 1;
        }
        // at vx > 0 a tracer that ends up left of where it started came back around.
        if cell_x(end.0) < cell_x(start.0) {
            wrapped_seam += 1;
        }
    }
    assert!(moved > 0, "no tracer changed cell; transport is vacuous");
    assert!(
        wrapped_seam > 0,
        "no tracer re-entered at the far side; the wrap seam was never exercised"
    );

    let mut diverged = Vec::new();
    for ii in 0..N_TRACERS {
        let m = mono[ii].unwrap_or_else(|| panic!("tracer {ii} was lost by the single-tile run"));
        let d = decomposed[ii].unwrap_or_else(|| panic!("tracer {ii} was lost in tile migration"));
        if m != d {
            diverged.push((ii, m, d));
        }
    }
    assert!(
        diverged.is_empty(),
        "the cut wrapping axis {:?} moved {} of {N_TRACERS} tracers elsewhere than the \
         single-tile periodic run ({wrapped_seam} crossed the seam); first is id {} at \
         cell {} {:?} instead of cell {} {:?}",
        split.counts(),
        diverged.len(),
        diverged[0].0,
        diverged[0].2.0,
        diverged[0].2.1,
        diverged[0].1.0,
        diverged[0].1.1
    );
}

/// an even cut on each periodic axis: four tiles, two interior seams and two wrap seams.
#[test]
fn tracers_cross_a_cut_wrapping_axis() {
    assert_matches([vec![32], vec![32]]);
}

/// the same with unequal cuts, so the wrap pairs tiles of different sizes.
#[test]
fn tracers_cross_ragged_cuts_on_a_wrapping_axis() {
    assert_matches([vec![19, 37], vec![27]]);
}
