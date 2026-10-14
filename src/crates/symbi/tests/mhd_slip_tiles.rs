// =============================================================================
// mhd_slip_tiles.rs
//
// the implicit magnetic-slip midpoint solve and its commit over a level cut
// into tiles, in lockstep, reproduce the level held whole bit for bit. on
// periodic adiabatic grids whose slip body's shell the cuts cross, 2.5D on two
// and six tiles and 3D on two and eight, the tiles take the whole grid's
// iteration count and residual, and after the commit every face each tile
// owns, every closing face it shares at a cut, the cell field and the energy
// of every interior cell equal the whole grid's, and the heat receipt booked
// once equals the whole grid's.
//
// run: cargo test -p symbi --test mhd_slip_tiles
// =============================================================================

use symbi::sim::decomp::{LocalCopy, Partition, Schedule, TileCuts, Topology, exchange_grid, unflatten};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, MagneticSpec, SurfaceSpec};
use symbi_substrate::regimes::mhd_substrate::{
    CutHalo, magnetic_slip_commit, magnetic_slip_commit_tiles, magnetic_slip_solve, magnetic_slip_solve_tiles,
};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;
type Store<const D: usize> = FieldStore<D, 3, HostMemory, f64>;
type Sim3 = SimStateGeneric<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;
const N3: usize = 16;

const N: usize = 32;
const GAMMA: f64 = 5.0 / 3.0;
const BODY: [f64; 2] = [0.5, 0.5];
const R_BODY: f64 = 0.22;
// four times the fixture's explicit step, so the solve takes several iterations
const DT: f64 = 4e-3;

fn slip_spec() -> MagneticSpec {
    MagneticSpec::Slip { diffusivity_ratio: 2.0, shell_width: 0.12, slip_length_ratio: 1.5, field_regularization: 0.1, placement: 0.0 }
}

// the grid, or the tile of it whose first cell sits at `offset`, placed as production places a
// tile: the whole grid's origin and an index offset
fn build(cells: [usize; 2], offset: [usize; 2], boundaries: Boundaries<2>) -> Sim {
    let dx = 1.0 / N as f64;
    let k = 2.0 * std::f64::consts::PI;
    let (a0, b0) = (0.3, 0.2);
    let sim = Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .origin([0.0, 0.0])
        .spacing([dx, dx])
        .index_offset([offset[0] as isize, offset[1] as isize])
        .boundaries(boundaries)
        .cfl(0.3)
        .allocate()
        .expect("2.5D sim")
        .set_initial(move |[x, y]| {
            let bx = |xf: f64| -a0 * (k * xf).cos() * (k * y).sin();
            let by = |yf: f64| a0 * (k * x).sin() * (k * yf).cos();
            MhdPrim::new(
                Prim::adiabatic(Density(1.0 + 0.2 * (k * x).sin()), Tensor::new([0.1, -0.05, 0.0]), Pressure(1.0)),
                Tensor::new([0.5 * (bx(x - 0.5 * dx) + bx(x + 0.5 * dx)), 0.5 * (by(y - 0.5 * dx) + by(y + 0.5 * dx)), b0 * (k * x).cos() * (k * y).cos()]),
            )
        })
        .seed_faces(move |axis, [x, y]| match axis {
            0 => -a0 * (k * x).cos() * (k * y).sin(),
            _ => a0 * (k * x).sin() * (k * y).cos(),
        })
        .build();
    sim.with_bodies(BodyCollection::new().add(
        Body::black_hole(0, Tensor::new(BODY), Tensor::zeros(), 1.0, R_BODY, 0.05, 1.0, 1.0, R_BODY)
            .with_surface(SurfaceSpec::Drain)
            .with_magnetic(slip_spec()),
    ))
}

fn heat<const D: usize>(sim: &Store<D>) -> Vec<u64> {
    sim.immersed.as_ref().unwrap().diagnostics.consolidate().iter().map(|d| d.slip_heat_delta.to_bits()).collect()
}

// a 3D grid, or the tile of it whose first cell sits at `offset`, with smooth faces and cells
fn build3(cells: [usize; 3], offset: [usize; 3], boundaries: Boundaries<3>) -> Sim3 {
    let dx = 1.0 / N3 as f64;
    let k = 2.0 * std::f64::consts::PI;
    let sim = Sim3::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .origin([0.0; 3])
        .spacing([dx; 3])
        .index_offset(std::array::from_fn(|a| offset[a] as isize))
        .boundaries(boundaries)
        .cfl(0.3)
        .allocate()
        .expect("3D sim")
        .set_initial(move |[x, y, z]| {
            MhdPrim::new(
                Prim::adiabatic(Density(1.0 + 0.2 * (k * x).sin()), Tensor::new([0.1, -0.05, 0.02]), Pressure(1.0)),
                Tensor::new([0.3 * (k * y).sin(), 0.3 * (k * z).sin(), 0.2 * (k * x).sin()]),
            )
        })
        .seed_faces(move |axis, [x, y, z]| match axis {
            0 => 0.3 * (k * y).sin(),
            1 => 0.3 * (k * z).sin(),
            _ => 0.2 * (k * x).sin(),
        })
        .build();
    sim.with_bodies(BodyCollection::new().add(
        Body::black_hole(0, Tensor::new([0.5; 3]), Tensor::zeros(), 1.0, R_BODY, 0.05, 1.0, 1.0, R_BODY)
            .with_surface(SurfaceSpec::Drain)
            .with_magnetic(slip_spec()),
    ))
}

// the slip midpoint on the whole grid and on its tiles; every tile's interior state against the
// whole grid's at the same cells
fn assert_tiles_match_whole<const D: usize>(n: usize, cuts: [Vec<usize>; D], whole: &Store<D>, tile: impl Fn([usize; D], [usize; D], Boundaries<D>) -> Store<D>) {
    let receipt = magnetic_slip_solve(whole, DT, GAMMA, 1e-10, 500);
    assert!(receipt.converged && receipt.iterations > 2, "the whole solve took {} iterations; the gate is weak", receipt.iterations);
    magnetic_slip_commit(whole, DT, GAMMA);
    let whole_heat = heat(whole);
    assert!(whole_heat.len() == 1 && f64::from_bits(whole_heat[0]) > 0.0, "the whole grid booked no heat");

    let partition = Partition::explicit([n; D], cuts.clone()).expect("cuts inside the grid");
    let counts = partition.counts();
    let n_tiles = partition.n_tiles();
    let extents: Vec<[(usize, usize); D]> = (0..n_tiles).map(|f| partition.tile_extents(unflatten(f, counts))).collect();
    let tiles: Vec<Store<D>> = extents
        .iter()
        .map(|ext| {
            let bnd = Boundaries(std::array::from_fn(|a| {
                let kind = if counts[a] > 1 { BoundaryType::CoarseFine } else { BoundaryType::Periodic };
                [kind, kind]
            }));
            tile(std::array::from_fn(|a| ext[a].1), std::array::from_fn(|a| ext[a].0), bnd)
        })
        .collect();
    let stores: Vec<&Store<D>> = tiles.iter().collect();
    let schedule = Schedule::derive(counts, stores[0].geom.ng, &Topology::wrapping([true; D]));
    let devices = vec![0i32; n_tiles];
    exchange_grid(&stores, &schedule, &devices, &LocalCopy);
    let tile_cuts = TileCuts { schedule: &schedule, devices: &devices, transport: &LocalCopy };
    let halo = CutHalo { cuts: &tile_cuts };
    let tiled = magnetic_slip_solve_tiles(&stores, DT, GAMMA, 1e-10, 500, &halo);
    assert_eq!(tiled.iterations, receipt.iterations, "{cuts:?}: iterations");
    assert_eq!(tiled.final_residual_norm.to_bits(), receipt.final_residual_norm.to_bits(), "{cuts:?}: final residual");
    magnetic_slip_commit_tiles(&stores, DT, GAMMA, &halo, true);
    assert_eq!(heat(stores[0]), whole_heat, "{cuts:?}: the booked heat");
    for s in &stores[1..] {
        assert!(heat(*s).iter().all(|h| *h == 0.0f64.to_bits()), "{cuts:?}: a second tile booked heat");
    }

    let wm = whole.fields.mhd.as_ref().unwrap();
    let wlo: [isize; D] = std::array::from_fn(|a| whole.geom.interior.spaces[a].lo);
    let mut compared = 0usize;
    for (s, ext) in stores.iter().zip(&extents) {
        let m = s.fields.mhd.as_ref().unwrap();
        let tlo: [isize; D] = std::array::from_fn(|a| s.geom.interior.spaces[a].lo);
        for c in s.geom.interior.iter() {
            let g: [isize; D] = std::array::from_fn(|a| c[a] - tlo[a] + ext[a].0 as isize + wlo[a]);
            for k in 0..3 {
                assert_eq!(m.bcell[k].at(c).to_bits(), wm.bcell[k].at(g).to_bits(), "{cuts:?}: bcell {k} at {g:?}");
            }
            assert_eq!(s.fields.cons.nrg_field().unwrap().at(c).to_bits(), whole.fields.cons.nrg_field().unwrap().at(g).to_bits(), "{cuts:?}: energy at {g:?}");
            for d in 0..D {
                assert_eq!(m.bface[d].at(c).to_bits(), wm.bface[d].at(g).to_bits(), "{cuts:?}: face {d} at {g:?}");
                // the closing face of the tile's last cell along d, shared at a cut
                if c[d] == s.geom.interior.spaces[d].hi - 1 {
                    let (mut cc, mut gg) = (c, g);
                    cc[d] += 1;
                    gg[d] += 1;
                    assert_eq!(m.bface[d].at(cc).to_bits(), wm.bface[d].at(gg).to_bits(), "{cuts:?}: closing face {d} at {gg:?}");
                }
            }
            compared += 1;
        }
    }
    assert_eq!(compared, n.pow(D as u32), "{cuts:?}: the tiles cover {compared} cells");
}

#[test]
fn two_tiles_cut_through_the_slip_shell_solve_and_commit_as_the_whole_grid() {
    let whole = build([N; 2], [0; 2], Boundaries::uniform(BoundaryType::Periodic));
    assert_tiles_match_whole::<2>(N, [vec![16], vec![]], &whole.store, |c, o, b| build(c, o, b).store);
}

#[test]
fn six_uneven_tiles_cut_through_the_slip_shell_solve_and_commit_as_the_whole_grid() {
    let whole = build([N; 2], [0; 2], Boundaries::uniform(BoundaryType::Periodic));
    assert_tiles_match_whole::<2>(N, [vec![11, 21], vec![13]], &whole.store, |c, o, b| build(c, o, b).store);
}

#[test]
fn three_dimensional_tiles_cut_through_the_slip_shell_solve_and_commit_as_the_whole_grid() {
    for cuts in [[vec![8], vec![], vec![]], [vec![7], vec![9], vec![6]]] {
        let whole = build3([N3; 3], [0; 3], Boundaries::uniform(BoundaryType::Periodic));
        assert_tiles_match_whole::<3>(N3, cuts, &whole.store, |c, o, b| build3(c, o, b).store);
    }
}
