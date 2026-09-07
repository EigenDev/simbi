// =============================================================================
// axis_boundary_decomposed.rs
//
// the polar half-turn across an azimuth cut into tiles. a tile's ghost band beyond the pole
// is the interior half a period away, which lives on other tiles, so the schedule's antipodal
// legs fill it and the band's odd components take their sign afterwards. the gates hold a
// decomposed 3D spherical shell, azimuth cut into equal slabs and into uneven ones whose
// half-turn image straddles tiles, to the single-tile run of the same shell: after the seed
// exchange every ghost band beyond the pole matches the single tile's band, and after
// evolution every interior cell and staggered face agrees.
// =============================================================================
use std::f64::consts::PI;
use symbi::prelude::KernelSet;
use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet3D;
use symbi::sim::decomp::{
    LocalCopy, PolarSeam, Schedule, Topology, evolve_scheduled, exchange_grid, flatten, unflatten,
};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Spherical;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Rmhd, 3, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = RmhdSubstrateKernelSet3D<HostMemory, f64>;

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.3;
const R_LO: f64 = 1.0;
const R_HI: f64 = 2.0;
const N: [usize; 3] = [6, 8, 8];
const T_FINAL: f64 = 0.03;
const V0: [f64; 3] = [0.03, 0.02, 0.01];
const B0: [f64; 3] = [0.1, 0.07, 0.05];

fn spherical_of(v: [f64; 3], th: f64, ph: f64) -> [f64; 3] {
    let (st, ct, sp, cp) = (th.sin(), th.cos(), ph.sin(), ph.cos());
    [
        v[0] * st * cp + v[1] * st * sp + v[2] * ct,
        v[0] * ct * cp + v[1] * ct * sp - v[2] * st,
        -v[0] * sp + v[1] * cp,
    ]
}

fn probe(r: f64, th: f64, ph: f64) -> MhdPrim<f64, 3> {
    let v = spherical_of(V0, th, ph);
    MhdPrim::new(
        Prim::adiabatic(
            Density(1.0 + 0.1 * r + 0.05 * th.cos() + 0.02 * th.sin() * ph.cos()),
            Tensor::new(v),
            Pressure(1.0 + 0.03 * th.sin() * ph.sin()),
        ),
        Tensor::new(spherical_of(B0, th, ph)),
    )
}

/// one tile: `cells` on each axis, its azimuth slab starting at global cell `phi0`, with the
/// face kinds given. the physical spacing is the global shell's on every tile.
fn tile(cells: [usize; 3], phi0: usize, bnd: Boundaries<3>) -> (Sim, Kern) {
    let dphi = 2.0 * PI / N[2] as f64;
    let sim = Sim::build(Rmhd, IdealGas { gamma: GAMMA }, Spherical)
        .cells(cells)
        .spacing([(R_HI - R_LO) / N[0] as f64, PI / N[1] as f64, dphi])
        .origin([R_LO, 0.0, phi0 as f64 * dphi])
        .boundaries(bnd)
        .cfl(CFL)
        .allocate()
        .expect("spherical 3D tile")
        .set_initial(|[r, th, ph]| probe(r, th, ph))
        .seed_faces(|axis, [_r, th, ph]| spherical_of(B0, th, ph)[axis])
        .build();
    let k = Kern::new(GAMMA, CFL, 1.0, &sim.geom.allocated);
    (sim, k)
}

fn single() -> (Sim, Kern) {
    tile(
        N,
        0,
        Boundaries::per_axis([
            [BoundaryType::Outflow, BoundaryType::Outflow],
            [BoundaryType::Axis, BoundaryType::Axis],
            [BoundaryType::Periodic, BoundaryType::Periodic],
        ]),
    )
}

/// the azimuth cut at `cuts` (global cell indices): every tile spans the full r and theta
/// extents, its pole faces handed to the antipodal legs and its azimuth faces to the wrap
/// legs, exactly as a decomposed run declares them.
fn slabs(cuts: &[usize]) -> (Vec<(Sim, Kern)>, [usize; 3], Schedule<3>) {
    let mut bounds = vec![0usize];
    bounds.extend_from_slice(cuts);
    bounds.push(N[2]);
    let widths: Vec<usize> = bounds.windows(2).map(|w| w[1] - w[0]).collect();
    let counts = [1usize, 1, widths.len()];
    let tiles: Vec<(Sim, Kern)> = widths
        .iter()
        .zip(&bounds)
        .map(|(&w, &phi0)| {
            tile(
                [N[0], N[1], w],
                phi0,
                Boundaries::per_axis([
                    [BoundaryType::Outflow, BoundaryType::Outflow],
                    [BoundaryType::CoarseFine, BoundaryType::CoarseFine],
                    [BoundaryType::CoarseFine, BoundaryType::CoarseFine],
                ]),
            )
        })
        .collect();
    let seam = PolarSeam {
        mirror: 1,
        sides: [true, true],
        azimuth: 2,
        slabs: widths,
    };
    let schedule = Schedule::derive_with_seam(
        counts,
        tiles[0].0.geom.ng,
        &Topology::wrapping([false, false, true]),
        Some(seam),
    );
    (tiles, counts, schedule)
}

fn cell(sim: &Sim, c: [isize; 3]) -> [f64; 8] {
    let p = &sim.fields.prim;
    let m = sim.fields.mhd.as_ref().expect("mhd fields");
    [
        *p.rho.view().at(c),
        *p.vel[0].view().at(c),
        *p.vel[1].view().at(c),
        *p.vel[2].view().at(c),
        *p.pre.as_ref().expect("pressure").view().at(c),
        *m.bcell[0].view().at(c),
        *m.bcell[1].view().at(c),
        *m.bcell[2].view().at(c),
    ]
}

const NAMES: [&str; 8] = [
    "rho", "v_r", "v_theta", "v_phi", "pre", "B_r", "B_theta", "B_phi",
];

/// the global azimuth cell index of a tile-local index, and the tile's local index of a
/// global one, for the tile whose slab starts at `phi0`.
fn to_global(phi0: usize, k: isize) -> isize {
    k + phi0 as isize
}

/// every cell of every tile's ghost band beyond either pole (the interior azimuth range, the
/// full r extent) against the single tile's cell at the same global position, bit for bit;
/// then every interior cell, then every owned face of the staggered field.
fn assert_tiles_match(
    single: &Sim,
    tiles: &[(Sim, Kern)],
    counts: [usize; 3],
    label: &str,
    bands: bool,
) {
    let sg = &single.geom;
    let mut phi0 = 0usize;
    let mut checked = 0usize;
    for flat in 0..tiles.len() {
        let tc = unflatten(flat, counts);
        let (sim, _) = &tiles[flatten(tc, counts)];
        let g = &sim.geom;
        let width = (g.interior.spaces[2].hi - g.interior.spaces[2].lo) as usize;
        let ng = g.ng as isize;
        let (jlo, jhi) = (g.interior.spaces[1].lo, g.interior.spaces[1].hi);
        let rows: Vec<isize> = if bands {
            (jlo - ng..jlo).chain(jhi..jhi + ng).collect()
        } else {
            (jlo..jhi).collect()
        };
        let r_range = if bands {
            g.allocated.spaces[0].lo..g.allocated.spaces[0].hi
        } else {
            g.interior.spaces[0].lo..g.interior.spaces[0].hi
        };
        for i in r_range.clone() {
            for &j in &rows {
                for k in g.interior.spaces[2].lo..g.interior.spaces[2].hi {
                    let gc = [i, j, to_global(phi0, k)];
                    let a = cell(sim, [i, j, k]);
                    let b = cell(single, gc);
                    for q in 0..8 {
                        if a[q] == 0.0 && b[q] == 0.0 {
                            continue;
                        }
                        assert_eq!(
                            a[q].to_bits(),
                            b[q].to_bits(),
                            "{label}: tile {flat} {} at local {:?} (global {gc:?}) = {} vs single {}",
                            NAMES[q],
                            [i, j, k],
                            a[q],
                            b[q]
                        );
                    }
                    checked += 1;
                }
            }
        }
        if !bands {
            let sm = single.fields.mhd.as_ref().expect("mhd");
            let tm = sim.fields.mhd.as_ref().expect("mhd");
            for d in 0..3 {
                let owned = g.interior.extend(d, 0, 1);
                for f in owned.iter() {
                    let gf = [f[0], f[1], to_global(phi0, f[2])];
                    let a = *tm.bface[d].view().at(f);
                    let b = *sm.bface[d].view().at(gf);
                    if a == 0.0 && b == 0.0 {
                        continue;
                    }
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{label}: tile {flat} bface[{d}] at local {f:?} (global {gf:?}) = {a} vs single {b}"
                    );
                }
            }
        }
        phi0 += width;
    }
    let _ = sg;
    assert!(checked > 0, "{label}: nothing compared");
}

/// the seed-time state with every halo filled: the single tile through its own fill, the
/// tiles through fill, exchange (antipodal legs included), sign flip and refill, which is the
/// priming sequence of the decomposed march.
fn prime(single: &mut (Sim, Kern), tiles: &mut [(Sim, Kern)], schedule: &Schedule<3>) {
    single.1.c2p(&single.0);
    single.1.ghost_fill(&single.0);
    for (s, k) in tiles.iter() {
        k.c2p(s);
        k.ghost_fill(s);
    }
    let devices = vec![0i32; tiles.len()];
    let sh: Vec<_> = tiles.iter().map(|(s, _)| &**s).collect();
    exchange_grid(&sh, schedule, &devices, &LocalCopy);
    for (tile, side) in schedule.polar_bands() {
        tiles[tile]
            .1
            .flip_polar_band(&tiles[tile].0, 1, side, schedule.reach());
    }
    for (s, k) in tiles.iter() {
        k.ghost_fill(s);
    }
}

fn evolve_tiles(tiles: &mut [(Sim, Kern)], schedule: &Schedule<3>) {
    let devices = vec![0i32; tiles.len()];
    let mut stores = Vec::new();
    let mut kernels = Vec::new();
    for (s, k) in tiles.iter_mut() {
        stores.push(&mut **s);
        kernels.push(&*k);
    }
    evolve_scheduled(
        &mut stores,
        &kernels,
        schedule,
        &devices,
        Timestepping::Rk2,
        0.0,
        T_FINAL,
        u64::MAX,
        &LocalCopy,
        |_, _, _| std::ops::ControlFlow::Continue(()),
    );
}

fn check(cuts: &[usize], label: &str) {
    let mut one = single();
    let (mut tiles, counts, schedule) = slabs(cuts);
    assert!(
        !schedule.polar_legs().is_empty(),
        "{label}: the seam produced antipodal legs"
    );
    prime(&mut one, &mut tiles, &schedule);
    assert_tiles_match(&one.0, &tiles, counts, &format!("{label} seed bands"), true);
    assert_tiles_match(
        &one.0,
        &tiles,
        counts,
        &format!("{label} seed interior"),
        false,
    );
    let mut one = single();
    let (mut tiles, counts, schedule) = slabs(cuts);
    symbi::sim::evolve::evolve(&mut one.0, &one.1, T_FINAL).expect("single evolve");
    evolve_tiles(&mut tiles, &schedule);
    assert_tiles_match(&one.0, &tiles, counts, &format!("{label} evolved"), false);
}

#[test]
fn two_equal_slabs_match_the_single_tile() {
    check(&[4], "two slabs");
}

#[test]
fn four_equal_slabs_match_the_single_tile() {
    check(&[2, 4, 6], "four slabs");
}

#[test]
fn uneven_slabs_whose_antipodes_straddle_tiles_match_the_single_tile() {
    check(&[3, 5], "slabs 3/2/3");
}

/// the antipodal legs through the one-sided message seam: the tiles split into two owner
/// groups, each group posts then completes, axis by axis, and the polar bands must match the
/// whole-ownership exchange cell for cell.
#[test]
fn the_antipodal_legs_travel_as_messages_like_any_other_leg() {
    use symbi::sim::decomp::{MessageQueue, Ownership, Phase, exchange_grid_phase};
    let (mut direct, _, schedule) = slabs(&[3, 5]);
    let (mut ranked, _, _) = slabs(&[3, 5]);
    for tiles in [&mut direct, &mut ranked] {
        for (s, k) in tiles.iter() {
            k.c2p(s);
            k.ghost_fill(s);
        }
    }
    let devices = vec![0i32; 3];
    {
        let sh: Vec<_> = direct.iter().map(|(s, _)| &**s).collect();
        exchange_grid(&sh, &schedule, &devices, &LocalCopy);
    }
    {
        let sh: Vec<_> = ranked.iter().map(|(s, _)| &**s).collect();
        let owner = [0usize, 1, 0];
        let queue = MessageQueue::new();
        for axis in 0..3 {
            for me in 0..2 {
                exchange_grid_phase(
                    &sh,
                    &schedule,
                    &devices,
                    Ownership::Ranked { owner: &owner, me },
                    &LocalCopy,
                    &queue,
                    axis,
                    Phase::Post,
                );
            }
            for me in 0..2 {
                exchange_grid_phase(
                    &sh,
                    &schedule,
                    &devices,
                    Ownership::Ranked { owner: &owner, me },
                    &LocalCopy,
                    &queue,
                    axis,
                    Phase::Complete,
                );
            }
        }
    }
    let mut checked = 0;
    for (flat, ((a, _), (b, _))) in direct.iter().zip(&ranked).enumerate() {
        let g = &a.geom;
        let ng = g.ng as isize;
        let (jlo, jhi) = (g.interior.spaces[1].lo, g.interior.spaces[1].hi);
        for i in g.allocated.spaces[0].lo..g.allocated.spaces[0].hi {
            for j in (jlo - ng..jlo).chain(jhi..jhi + ng) {
                for k in g.allocated.spaces[2].lo..g.allocated.spaces[2].hi {
                    let (x, y) = (cell(a, [i, j, k]), cell(b, [i, j, k]));
                    for q in 0..8 {
                        assert!(
                            x[q].to_bits() == y[q].to_bits() || (x[q] == 0.0 && y[q] == 0.0),
                            "tile {flat} {} at {:?}: direct {} vs messaged {}",
                            NAMES[q],
                            [i, j, k],
                            x[q],
                            y[q]
                        );
                    }
                    checked += 1;
                }
            }
        }
        let (am, bm) = (
            a.fields.mhd.as_ref().unwrap(),
            b.fields.mhd.as_ref().unwrap(),
        );
        // the staggered faces of the band itself, over the azimuth window the polar legs
        // move; the band's azimuth halo faces come from ordinary legs, which the message seam
        // carries for cell fields alone.
        for d in [0usize, 2] {
            let extra = if d == 2 { 1 } else { 0 };
            let window = am.bface[d]
                .domain()
                .boundary(1, symbi_algebra::Side::Lo, ng)
                .slab(
                    2,
                    (g.interior.spaces[2].lo, g.interior.spaces[2].hi + extra),
                );
            for f in window.iter() {
                let (x, y) = (*am.bface[d].view().at(f), *bm.bface[d].view().at(f));
                assert!(
                    x.to_bits() == y.to_bits() || (x == 0.0 && y == 0.0),
                    "tile {flat} bface[{d}] at {f:?}: direct {x} vs messaged {y}"
                );
            }
        }
    }
    assert!(checked > 0);
}
