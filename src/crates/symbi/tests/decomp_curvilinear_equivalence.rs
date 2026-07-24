// =============================================================================
// decomp_curvilinear_equivalence.rs
//
// the correctness contract for multi-gpu domain decomposition on a CURVILINEAR
// chart, validated in-process on the cpu. decomp_equivalence.rs proves the
// Cartesian case; this proves the case that actually matters for disks and
// accretors, where the metric varies with radius.
//
// on a cylindrical (r, z) or spherical (r, theta) grid the wave speeds, the
// geometric source, and the face areas all depend on r. so a radial cut hands
// each tile a DIFFERENT r-range: a tile is not a translate of its neighbor the
// way a Cartesian tile is. the only thing that makes the decomposed run
// reproduce the monolithic one is a per-tile radial origin placed so the tile's
// local cell i sits at the same physical r as the undecomposed grid's global
// cell tc*m + i, plus the halo exchange feeding each cut face the neighbor's
// interior.
//
// a smooth radial pulse (density + pressure) launches waves that cross the radial
// cut; the decomposed density must match the monolithic density to round-off, and
// a non-vacuity guard requires the pulse to have actually moved (else a broken
// exchange would match a frozen field trivially).
//
// r starts at R_LO > 0: the axis r = 0 is a coordinate singularity and no cut is
// placed on it. the harness drives the PRODUCTION `evolve_decomposed` loop, the
// same one the multi-gpu python entry runs.
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{evolve_decomposed, flatten, unflatten, LocalCopy};
use symbi::regimes::substrate_gpu::device_sync;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{AxisMap, Cylindrical, Spherical};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{with_device, CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const NR: usize = 48; // radial cells (split across radial tiles)
const NT: usize = 24; // transverse cells (z or theta)
const R_LO: f64 = 1.0; // inner radius, away from the r = 0 singularity
const DR: f64 = 1.0 / NR as f64; // r spans [1.0, 2.0]
const DT_AX: f64 = 0.5 / NT as f64; // transverse extent 0.5 (z, or theta in radians)
const R_HI: f64 = R_LO + 1.0; // outer radius (r spans [1, 2])
const T_FINAL: f64 = 0.03; // waves from the pulse stay well inside [R_LO, R_HI].

// a smooth pulse centered mid-radius -> radial waves that cross a radial cut placed
// anywhere in the interior. away from both radial ends so the outflow boundaries never
// activate and mono == decomposed is exact.
fn pulse(r: f64) -> f64 {
    0.2 * (-((r - 1.5) / 0.08).powi(2)).exp()
}

// the log radial slope spanning [R_LO, R_HI] over NR cells: face(i) = R_LO * 10^(i*slope).
fn log_slope() -> f64 {
    (R_HI / R_LO).log10() / NR as f64
}

// the per-tile radial origin and coordinate maps for a tile whose first radial cell is the
// global cell `tc0*m0`. UNIFORM: the radial axis starts at R_LO + tc0*m0*DR and the builder's
// origin+spacing define the geometry (maps = None). LOG: the start is advanced
// MULTIPLICATIVELY (R_LO * 10^(tile_lo*slope)) and a shifted Log map with the SAME slope carries
// the geometry -- mirroring the production per-tile map shift, so a tile's local cell i sits at
// the identical physical r as the undecomposed grid's global cell tc0*m0 + i.
fn tile_radial(logr: bool, tc0: usize, m0: usize, z0: f64) -> (f64, Option<[AxisMap; 2]>) {
    if logr {
        let slope = log_slope();
        let r_start = R_LO * 10.0_f64.powf((tc0 * m0) as f64 * slope);
        (
            r_start,
            Some([
                AxisMap::Log { start: r_start, log_slope: slope },
                AxisMap::Uniform { start: z0, dx: DT_AX },
            ]),
        )
    } else {
        (R_LO + (tc0 * m0) as f64 * DR, None)
    }
}

macro_rules! curvilinear_harness {
    ($modname:ident, $chart:ty, $chart_val:expr, $logr:expr) => {
        mod $modname {
            use super::*;

            const LOGR: bool = $logr;
            type Sim = SimState<Newtonian, 2, $chart, IdealGas<f64>, CpuSpace, HostMemory>;
            type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

            const NDEV: i32 = 2;
            fn tile_device(flat: usize) -> i32 {
                (flat as i32) % NDEV
            }
            fn sync_devices() {
                for dd in 0..NDEV {
                    with_device(dd, || device_sync::<HostMemory>());
                }
            }

            fn make(
                cells: [usize; 2],
                origin: [f64; 2],
                maps: Option<[AxisMap; 2]>,
                bnd: Boundaries<2>,
                ts: Timestepping,
            ) -> (Sim, Kern) {
                let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, $chart_val)
                    .cells(cells)
                    .origin(origin)
                    .spacing([DR, DT_AX])
                    .coord_maps(maps)
                    .boundaries(bnd)
                    .timestepping(ts)
                    .allocate()
                    .expect("curvilinear sim construction failed")
                    .set_initial(|x| {
                        // x[0] is the physical radius of this cell (origin + i*dr), identical
                        // for the monolithic grid and the tile that owns the cell.
                        let b = pulse(x[0]);
                        Prim { rho: 1.0 + b, vel: Tensor::new([0.0, 0.0]), pre: 1.0 + b }
                    })
                    .build();
                let k = Kern::new(GAMMA, CFL, &sim.geom.allocated);
                (sim, k)
            }

            // tile grid `counts` = [radial tiles, transverse tiles]. a tile gets a CoarseFine
            // face wherever it borders a neighbor; the radial ends are outflow and the transverse
            // ends are periodic (a z-uniform / theta-uniform pulse never reaches them, so periodic
            // and outflow agree, and periodic avoids any ghost-extrapolation asymmetry at a cut).
            fn grid_tiles(counts: [usize; 2], ts: Timestepping) -> Vec<(Sim, Kern)> {
                let m: [usize; 2] = [NR / counts[0], NT / counts[1]];
                let total = counts[0] * counts[1];
                (0..total)
                    .map(|flat| {
                        let tc = unflatten(flat, counts);
                        let z0 = (tc[1] * m[1]) as f64 * DT_AX;
                        let (r0, maps) = tile_radial(LOGR, tc[0], m[0], z0);
                        let origin = [r0, z0];
                        let bnd = Boundaries::per_axis([
                            [
                                if tc[0] == 0 { BoundaryType::Outflow } else { BoundaryType::CoarseFine },
                                if tc[0] == counts[0] - 1 { BoundaryType::Outflow } else { BoundaryType::CoarseFine },
                            ],
                            [
                                if tc[1] == 0 { BoundaryType::Periodic } else { BoundaryType::CoarseFine },
                                if tc[1] == counts[1] - 1 { BoundaryType::Periodic } else { BoundaryType::CoarseFine },
                            ],
                        ]);
                        with_device(tile_device(flat), || make(m, origin, maps, bnd, ts))
                    })
                    .collect()
            }

            fn run(tiles: &mut [(Sim, Kern)], counts: [usize; 2], ts: Timestepping) {
                let devices: Vec<i32> = (0..tiles.len()).map(tile_device).collect();
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
                    ts,
                    0.0,
                    T_FINAL,
                    u64::MAX,
                    &LocalCopy,
                    |_, _, _| std::ops::ControlFlow::Continue(()),
                );
            }

            fn global_den(tiles: &[(Sim, Kern)], counts: [usize; 2]) -> Vec<f64> {
                sync_devices();
                let m: [usize; 2] = [NR / counts[0], NT / counts[1]];
                let mut out = vec![f64::NAN; NR * NT];
                for (flat_tile, (sim, _)) in tiles.iter().enumerate() {
                    let tc = unflatten(flat_tile, counts);
                    let ilo: [isize; 2] = std::array::from_fn(|a| sim.geom.interior.spaces[a].lo);
                    for c in sim.geom.interior.iter() {
                        let g: [usize; 2] =
                            std::array::from_fn(|a| tc[a] * m[a] + (c[a] - ilo[a]) as usize);
                        out[flatten(g, [NR, NT])] = *sim.fields.cons.den.view().at(c);
                    }
                }
                out
            }

            // run mono (counts = [1, 1]) and the requested decomposition, then assert the global
            // density grids agree to round-off -- the whole point on a curvilinear chart, where
            // each tile carries its own r-range and metric.
            pub fn assert_matches(counts: [usize; 2], ts: Timestepping) {
                let mut mono = grid_tiles([1, 1], ts);
                // the initial density, read before evolving -- the non-vacuity baseline, taken from
                // the grid itself so it holds for both uniform and log radial cell centers.
                let ic = global_den(&mono, [1, 1]);
                run(&mut mono, [1, 1], ts);
                let mono_vals = global_den(&mono, [1, 1]);

                let mut dec = grid_tiles(counts, ts);
                run(&mut dec, counts, ts);
                let dec_vals = global_den(&dec, counts);

                assert!(
                    mono_vals.iter().all(|v| v.is_finite()) && dec_vals.iter().all(|v| v.is_finite()),
                    "some global cells were never written (gather bug)"
                );

                // NON-VACUITY: the pulse must have actually moved off its IC, or a broken exchange
                // would match a frozen field and the gate would prove nothing.
                let max_move = mono_vals
                    .iter()
                    .zip(&ic)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                assert!(
                    max_move > 1e-4,
                    "the radial pulse never moved (max {max_move:e}); the evolution is vacuous"
                );

                let max_err = mono_vals
                    .iter()
                    .zip(&dec_vals)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                assert!(
                    max_err < 1e-12,
                    "curvilinear decomposition {counts:?} ({ts:?}) vs monolithic density max err \
                     {max_err:e} (pulse moved {max_move:e})"
                );
            }
        }
    };
}

curvilinear_harness!(cyl, Cylindrical, Cylindrical, false);
curvilinear_harness!(sph, Spherical, Spherical, false);
// log radial: the disk/accretor case, where the per-tile origin is advanced multiplicatively
// and the coordinate-map arithmetic (not just a shifted linear extent) has to be right.
curvilinear_harness!(cyl_log, Cylindrical, Cylindrical, true);
curvilinear_harness!(sph_log, Spherical, Spherical, true);

// a radial cut is the load-bearing case: the two tiles carry different r-ranges. euler
// isolates the exchange; rk2 adds the between-stage exchange. the 2x2 grid cuts the
// transverse axis too (corner ghosts).
#[test]
fn cyl_radial_two_tile_euler() {
    cyl::assert_matches([2, 1], Timestepping::Euler);
}

#[test]
fn cyl_radial_four_tile_rk2() {
    cyl::assert_matches([4, 1], Timestepping::Rk2);
}

#[test]
fn cyl_quad_tile_rk2() {
    cyl::assert_matches([2, 2], Timestepping::Rk2);
}

#[test]
fn sph_radial_two_tile_euler() {
    sph::assert_matches([2, 1], Timestepping::Euler);
}

#[test]
fn sph_radial_four_tile_rk2() {
    sph::assert_matches([4, 1], Timestepping::Rk2);
}

// log radial cuts: the per-tile map start is multiplicative, so a wrong shift lands the tile's
// cells at the wrong radii and the metric diverges from the monolithic grid at the cut.
#[test]
fn cyl_log_radial_four_tile_rk2() {
    cyl_log::assert_matches([4, 1], Timestepping::Rk2);
}

#[test]
fn cyl_log_quad_tile_rk2() {
    cyl_log::assert_matches([2, 2], Timestepping::Rk2);
}

#[test]
fn sph_log_radial_four_tile_rk2() {
    sph_log::assert_matches([4, 1], Timestepping::Rk2);
}

// =============================================================================
// swirl (DOF != NDIM): the out-of-plane azimuthal momentum under decomposition
// =============================================================================
//
// a cylindrical (r, z) grid with DOF = 3 lifts the azimuthal momentum v_phi (slot 1,
// scale factor h3 = r) onto the 2D grid. it is advected by the in-plane radial flow and
// carries an angular-momentum geometric source; the decomposition must exchange it across
// a radial cut like any other momentum component (the transport ranges over mom[0..DOF]).
// this is the piece the swirl build-macro refusal was guarding: only the BUILD hardcoded
// DOF = D, never the transport.
mod swirl {
    use super::*;

    type Sim = SimStateGeneric<Newtonian, 2, 3, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory>;
    type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

    const NDEV: i32 = 2;
    fn tile_device(flat: usize) -> i32 {
        (flat as i32) % NDEV
    }
    fn sync_devices() {
        for dd in 0..NDEV {
            with_device(dd, || device_sync::<HostMemory>());
        }
    }

    // a localized azimuthal-velocity blob, NOT rigid rotation (which would be a discrete
    // null and never move); the radial pulse's flow advects it across the cut.
    fn vphi(r: f64) -> f64 {
        0.3 * (-((r - 1.5) / 0.1).powi(2)).exp()
    }

    fn make(origin: [f64; 2], bnd: Boundaries<2>, cells: [usize; 2], ts: Timestepping) -> (Sim, Kern) {
        let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cylindrical)
            .cells(cells)
            .origin(origin)
            .spacing([DR, DT_AX])
            .boundaries(bnd)
            .timestepping(ts)
            .allocate()
            .expect("swirl sim construction failed")
            .set_initial(|x| {
                let b = pulse(x[0]);
                // vel = (v_r, v_phi, v_z); the swirl rides slot 1.
                Prim { rho: 1.0 + b, vel: Tensor::new([0.0, vphi(x[0]), 0.0]), pre: 1.0 + b }
            })
            .build();
        let k = Kern::new(GAMMA, CFL, &sim.geom.allocated);
        (sim, k)
    }

    fn grid_tiles(counts: [usize; 2], ts: Timestepping) -> Vec<(Sim, Kern)> {
        let m: [usize; 2] = [NR / counts[0], NT / counts[1]];
        let total = counts[0] * counts[1];
        (0..total)
            .map(|flat| {
                let tc = unflatten(flat, counts);
                let origin = [R_LO + (tc[0] * m[0]) as f64 * DR, (tc[1] * m[1]) as f64 * DT_AX];
                let bnd = Boundaries::per_axis([
                    [
                        if tc[0] == 0 { BoundaryType::Outflow } else { BoundaryType::CoarseFine },
                        if tc[0] == counts[0] - 1 { BoundaryType::Outflow } else { BoundaryType::CoarseFine },
                    ],
                    [
                        if tc[1] == 0 { BoundaryType::Periodic } else { BoundaryType::CoarseFine },
                        if tc[1] == counts[1] - 1 { BoundaryType::Periodic } else { BoundaryType::CoarseFine },
                    ],
                ]);
                with_device(tile_device(flat), || make(origin, bnd, m, ts))
            })
            .collect()
    }

    fn run(tiles: &mut [(Sim, Kern)], counts: [usize; 2], ts: Timestepping) {
        let devices: Vec<i32> = (0..tiles.len()).map(tile_device).collect();
        let mut stores = Vec::new();
        let mut kernels = Vec::new();
        for (s, k) in tiles.iter_mut() {
            stores.push(&mut **s);
            kernels.push(&*k);
        }
        evolve_decomposed(
            &mut stores, &kernels, counts, &devices, ts, 0.0, T_FINAL, u64::MAX, &LocalCopy,
            |_, _, _| std::ops::ControlFlow::Continue(()),
        );
    }

    // scatter the azimuthal momentum (slot 1) into one global grid.
    fn global_mphi(tiles: &[(Sim, Kern)], counts: [usize; 2]) -> Vec<f64> {
        sync_devices();
        let m: [usize; 2] = [NR / counts[0], NT / counts[1]];
        let mut out = vec![f64::NAN; NR * NT];
        for (flat_tile, (sim, _)) in tiles.iter().enumerate() {
            let tc = unflatten(flat_tile, counts);
            let ilo: [isize; 2] = std::array::from_fn(|a| sim.geom.interior.spaces[a].lo);
            for c in sim.geom.interior.iter() {
                let g: [usize; 2] = std::array::from_fn(|a| tc[a] * m[a] + (c[a] - ilo[a]) as usize);
                out[flatten(g, [NR, NT])] = *sim.fields.cons.mom[1].view().at(c);
            }
        }
        out
    }

    pub fn assert_matches(counts: [usize; 2], ts: Timestepping) {
        let mut mono = grid_tiles([1, 1], ts);
        let ic = global_mphi(&mono, [1, 1]);
        run(&mut mono, [1, 1], ts);
        let mono_vals = global_mphi(&mono, [1, 1]);

        let mut dec = grid_tiles(counts, ts);
        run(&mut dec, counts, ts);
        let dec_vals = global_mphi(&dec, counts);

        assert!(
            mono_vals.iter().all(|v| v.is_finite()) && dec_vals.iter().all(|v| v.is_finite()),
            "some global cells were never written"
        );

        // NON-VACUITY: the azimuthal momentum must have moved off its IC, or a broken
        // exchange of the out-of-plane slot would match a frozen field trivially.
        let max_move = mono_vals
            .iter()
            .zip(&ic)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_move > 1e-4,
            "the azimuthal momentum never moved (max {max_move:e}); the gate is vacuous"
        );

        let max_err = mono_vals
            .iter()
            .zip(&dec_vals)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err < 1e-12,
            "swirl decomposition {counts:?} ({ts:?}) vs monolithic S_phi max err {max_err:e} \
             (moved {max_move:e})"
        );
    }
}

// the radial cut is load-bearing: the two tiles carry different r-ranges AND the azimuthal
// momentum must cross the cut. 4-tile rk2 puts an interior tile between two cuts.
#[test]
fn swirl_radial_two_tile_euler() {
    swirl::assert_matches([2, 1], Timestepping::Euler);
}

#[test]
fn swirl_radial_four_tile_rk2() {
    swirl::assert_matches([4, 1], Timestepping::Rk2);
}
