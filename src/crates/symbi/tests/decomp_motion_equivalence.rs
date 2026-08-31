// =============================================================================
// decomp_motion_equivalence.rs
//
// mesh motion x decomposition: tiles evolving a homologously expanding (or uniformly
// translating) mesh must reproduce the monolithic single-grid run to round-off. every tile
// carries the identical motion state and the decomposed loop advances all tile clocks and
// scale factors in lockstep with the shared global dt — identical dt sequence, identical
// a(t) sequence, bit-for-bit. the cuts sit at fixed comoving indices, so the halo exchange
// is untouched by the expansion. the monolithic reference is the production single-grid
// `evolve` loop (its own per-step motion advance), so this oracle pins the decomposed
// advance semantics against the canonical ones — a tile that failed to advance its clock or
// scale factor diverges immediately (the fluxes carry a(t)).
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, evolve_decomposed, flatten, unflatten};
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi::sim::tracers::{cell_container_address, seed_and_partition, seed_mass_weighted};
use symbi_algebra::Tensor;
use symbi_geometry::{Cartesian, MotionState};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const N: usize = 32;
const DX: f64 = 1.0 / N as f64;
const T_FINAL: f64 = 0.04;
const ADOT: f64 = 0.5;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

fn bump(x: f64, y: f64) -> f64 {
    0.3 * (-(((x - 0.5) / 0.1).powi(2) + ((y - 0.5) / 0.1).powi(2))).exp()
}

fn make(
    cells: [usize; 2],
    origin: [f64; 2],
    bnd: Boundaries<2>,
    motion: MotionState<f64>,
) -> (Sim, Kern) {
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 2])
        .origin(origin)
        .boundaries(bnd)
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim construction failed")
        .set_initial(|[x, y]| {
            let b = bump(x, y);
            Prim {
                rho: 1.0 + b,
                vel: Tensor::new([0.0, 0.0]),
                pre: 1.0 + b,
            }
        })
        .build();
    sim.motion = motion;
    let k = Kern::new(GAMMA, CFL, &sim.geom.allocated);
    (sim, k)
}

fn grid_tiles(counts: [usize; 2], motion: MotionState<f64>) -> Vec<(Sim, Kern)> {
    let m: [usize; 2] = std::array::from_fn(|a| N / counts[a]);
    (0..counts.iter().product::<usize>())
        .map(|flat| {
            let tc = unflatten(flat, counts);
            let origin = std::array::from_fn(|a| tc[a] as f64 * m[a] as f64 * DX);
            let bnd = Boundaries(std::array::from_fn(|a| {
                let lo = if tc[a] == 0 {
                    BoundaryType::Outflow
                } else {
                    BoundaryType::CoarseFine
                };
                let hi = if tc[a] == counts[a] - 1 {
                    BoundaryType::Outflow
                } else {
                    BoundaryType::CoarseFine
                };
                [lo, hi]
            }));
            make(m, origin, bnd, motion)
        })
        .collect()
}

fn run_decomposed(tiles: &mut [(Sim, Kern)], counts: [usize; 2]) {
    let devices: Vec<i32> = vec![0; tiles.len()];
    let mut stores = Vec::new();
    let mut kernels = Vec::new();
    for (s, k) in tiles.iter_mut() {
        stores.push(&mut **s);
        kernels.push(&*k);
    }
    evolve_decomposed(
        &mut stores,
        &kernels,
        counts,
        &devices,
        Timestepping::Rk2,
        0.0,
        T_FINAL,
        u64::MAX,
        &LocalCopy,
        |_, _, _| std::ops::ControlFlow::Continue(()),
    );
}

fn global_field(
    tiles: &[(Sim, Kern)],
    counts: [usize; 2],
    pick: impl Fn(&Sim, [isize; 2]) -> f64,
) -> Vec<f64> {
    let m: [usize; 2] = std::array::from_fn(|a| N / counts[a]);
    let mut out = vec![f64::NAN; N * N];
    for (flat_tile, (sim, _)) in tiles.iter().enumerate() {
        let tc = unflatten(flat_tile, counts);
        let ilo: [isize; 2] = std::array::from_fn(|a| sim.geom.interior.spaces[a].lo);
        for c in sim.geom.interior.iter() {
            let g: [usize; 2] = std::array::from_fn(|a| tc[a] * m[a] + (c[a] - ilo[a]) as usize);
            out[flatten(g, [N; 2])] = pick(sim, c);
        }
    }
    out
}

fn max_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn assert_motion_matches(counts: [usize; 2], motion: MotionState<f64>) {
    let den = |s: &Sim, c: [isize; 2]| *s.fields.cons.den.view().at(c);
    let momx = |s: &Sim, c: [isize; 2]| *s.fields.cons.mom[0].view().at(c);

    let (mut mono, mk) = make(
        [N, N],
        [0.0, 0.0],
        Boundaries::uniform(BoundaryType::Outflow),
        motion,
    );
        let partition = symbi::sim::decomp::Partition::uniform(
        std::array::from_fn(|a| mono.geom.interior.spaces[a].size()),
        counts,
    )
    .expect("even tile counts divide the grid");
    let per_tile = seed_and_partition(&mono, 2048, &partition);
    mono.tracers = Some(seed_mass_weighted(&mono, 2048));
    let initial_owner = mono.tracers.as_ref().unwrap().owner.clone();
    evolve(&mut mono, &mk, T_FINAL).expect("mono evolve");
    let mono_wrap = [(mono, mk)];
    let mono_den = global_field(&mono_wrap, [1, 1], den);
    let mono_momx = global_field(&mono_wrap, [1, 1], momx);
    let mono_sim = &mono_wrap[0].0;

    let mut tiles = grid_tiles(counts, motion);
    for ((tile, _), tracers) in tiles.iter_mut().zip(per_tile) {
        tile.tracers = Some(tracers);
    }
    run_decomposed(&mut tiles, counts);
    let dec_den = global_field(&tiles, counts, den);
    let dec_momx = global_field(&tiles, counts, momx);

    assert!(
        mono_den.iter().all(|v| v.is_finite()) && dec_den.iter().all(|v| v.is_finite()),
        "some global cells were never written"
    );
    // every tile's clock and scale factor track the monolithic run exactly: a stale tile
    // clock (or a never-advanced scale factor) is the failure mode under test.
    for (i, (s, _)) in tiles.iter().enumerate() {
        assert!(
            (s.time - mono_sim.time).abs() < 1e-12,
            "{counts:?} tile {i}: clock {:.6e} != mono {:.6e}",
            s.time,
            mono_sim.time
        );
        assert!(
            (s.motion.a - mono_sim.motion.a).abs() < 1e-14,
            "{counts:?} tile {i}: scale factor {} != mono {}",
            s.motion.a,
            mono_sim.motion.a
        );
    }
    // the mesh genuinely moved (non-vacuous): an expanding mesh grows a past 1.
    if motion.homologous {
        assert!(
            mono_sim.motion.a > 1.0 + 0.5 * ADOT * T_FINAL,
            "the scale factor never advanced (a = {}); test is vacuous",
            mono_sim.motion.a
        );
    }
    let de = max_err(&mono_den, &dec_den);
    let me = max_err(&mono_momx, &dec_momx);
    assert!(
        de < 1e-12,
        "{counts:?}: density diverged under mesh motion: {de:e}"
    );
    assert!(
        me < 1e-12,
        "{counts:?}: mom_x diverged under mesh motion: {me:e}"
    );

    let mono_tracers = mono_sim.tracers.as_ref().unwrap();
    let mut decomposed = Vec::new();
    for (sim, _) in &tiles {
        let tracers = sim.tracers.as_ref().unwrap();
        decomposed.extend((0..tracers.len()).map(|ii| {
            (
                tracers.id[ii],
                tracers.owner[ii],
                tracers.x[ii],
                tracers.flags[ii],
            )
        }));
    }
    decomposed.sort_unstable_by_key(|record| record.0);
    assert_eq!(decomposed.len(), mono_tracers.len());
    let histogram = |owners: &[symbi_sim::mass_transport::ContainerId]| {
        let mut counts = std::collections::BTreeMap::new();
        for owner in owners {
            *counts.entry(*owner).or_insert(0isize) += 1;
        }
        counts
    };
    let decomposed_owner: Vec<_> = decomposed.iter().map(|record| record.1).collect();
    let mono_histogram = histogram(&mono_tracers.owner);
    let decomposed_histogram = histogram(&decomposed_owner);
    let l1: usize = mono_histogram
        .keys()
        .chain(decomposed_histogram.keys())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .map(|owner| {
            (mono_histogram.get(owner).copied().unwrap_or(0)
                - decomposed_histogram.get(owner).copied().unwrap_or(0))
            .unsigned_abs()
        })
        .sum();
    assert!(
        l1 <= mono_tracers.len() / 5,
        "{counts:?}: moving-mesh decomposition changed the ownership histogram by l1={l1}"
    );
    let crossed_cut = decomposed_owner
        .iter()
        .zip(&initial_owner)
        .filter(|(owner, initial)| {
            let owner_x = owner.0 as usize % N;
            let initial_x = initial.0 as usize % N;
            owner_x / (N / counts[0]) != initial_x / (N / counts[0])
        })
        .count();
    if counts[0] > 1 {
        assert!(
            crossed_cut > 0,
            "{counts:?}: no tracer crossed an x tile cut; migration is vacuous"
        );
    }
    for (ii, &(id, owner, position, _flags)) in decomposed.iter().enumerate() {
        assert_eq!(id, mono_tracers.id[ii]);
        let Some((level, linear)) = cell_container_address(owner) else {
            continue;
        };
        assert_eq!(level, 0);
        let cell = [linear % N, linear / N];
        let mut expected = std::array::from_fn(|dd| (cell[dd] as f64 + 0.5) * DX);
        if motion.homologous {
            expected = expected.map(|value| value * mono_sim.motion.a);
        } else {
            expected[0] += motion.a_dot * mono_sim.time;
        }
        assert_eq!(
            position, expected,
            "{counts:?}: tracer {id} position does not match its accepted owner"
        );
    }
}

#[test]
fn homologous_expansion_two_tiles_x() {
    assert_motion_matches([2, 1], MotionState::homologous(1.0, ADOT));
}

#[test]
fn homologous_expansion_two_tiles_y() {
    assert_motion_matches([1, 2], MotionState::homologous(1.0, ADOT));
}

#[test]
fn homologous_expansion_four_tiles() {
    assert_motion_matches([2, 2], MotionState::homologous(1.0, ADOT));
}

#[test]
fn uniform_translation_two_tiles() {
    // translation: a stays 1, a_dot is the frame velocity along x; the cut rides the
    // comoving indices while the flux carries the frame advection.
    assert_motion_matches([2, 1], MotionState::uniform(1.0, 0.25));
}

#[test]
fn homologous_single_tile_matches_raw_evolve() {
    // discriminator: a 1x1 "decomposition" runs the decomposed pipeline on the whole grid —
    // with no tile cuts present, any divergence here isolates a pipeline difference under motion.
    assert_motion_matches([1, 1], MotionState::homologous(1.0, ADOT));
}
