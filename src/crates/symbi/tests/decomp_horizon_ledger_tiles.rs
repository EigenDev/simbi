// =============================================================================
// decomp_horizon_ledger_tiles.rs
//
// the horizon ledger of an excised kerr-schild run on four tiles meeting at the
// horizon. every tile holds a quarter of the shell the horizon diagnostic
// measures, so the body's per-step deltas are the sum of four tile partials:
// - the fluid, the clock and the timestep equal the single grid's bit for bit
//   (the tiles carry the whole grid's origin and the index of their first cell);
// - the accreted mass and the accreted energy equal the single grid's bit for
//   bit: each tile adds its shell terms into an exact sum, and the four
//   partials merge into the integer the whole grid's sum holds, rounded once.
// the horizon shell spans all four tiles, so the ledger each tile books is the
// cross-tile sum of the four shell partials.
// =============================================================================

use symbi::prelude::*;
use symbi::regimes::substrate_rhd::RhdSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, evolve_decomposed, unflatten};
use symbi::sim::substrate_seam::WithExcision;
use symbi_geometry::SchwarzschildKSCartesian;
use symbi_hydro::Rhd;
use symbi_ib::{Body, BodyCollection, BodyKind};

const GAMMA: f64 = 4.0 / 3.0;
const MASS: f64 = 0.3;
const R_EXC: f64 = 0.35;
const R_DIAG: f64 = 0.9;
const T_FINAL: f64 = 0.05;
const N: usize = 24;
const LO: f64 = -1.2;
const DX: f64 = 2.4 / N as f64;

type Sim = SimState<Rhd, 2, SchwarzschildKSCartesian<f64>, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = RhdSubstrateKernelSet<HostMemory, f64, 2>;

/// a tile of `cells` whose first cell is whole-grid cell `lo`, at the whole grid's origin
fn make(cells: [usize; 2], lo: [usize; 2], bnd: Boundaries<2>) -> (Sim, Kern) {
    let sim = Sim::build(Rhd, IdealGas { gamma: GAMMA }, SchwarzschildKSCartesian { mass: MASS })
        .cells(cells)
        .spacing([DX; 2])
        .origin([LO; 2])
        .index_offset(std::array::from_fn(|a| lo[a] as isize))
        .boundaries(bnd)
        .cfl(0.3)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("kerr-schild simulation construction")
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0]), Pressure(0.1)))
        .build()
        .with_bodies(BodyCollection::new().add(Body::horizon(0, R_EXC, R_DIAG)));
    let kernels = Kern::new(GAMMA, 0.3, &sim.geom.allocated).with_excision(R_EXC, 1.0, 1.0);
    (sim, kernels)
}

fn ledger(sim: &Sim) -> (f64, f64) {
    match sim.immersed.as_ref().expect("horizon body").bodies.get(0).kind {
        BodyKind::Horizon { total_accreted_mass, total_accreted_energy, .. } => (total_accreted_mass, total_accreted_energy),
        _ => panic!("test body is not a horizon"),
    }
}

#[test]
fn four_tiles_meeting_at_the_horizon_book_the_single_grid_ledger() {
    let (mut single, single_kernels) = make([N, N], [0, 0], Boundaries::uniform(BoundaryType::Outflow));
    evolve(&mut single, &single_kernels, T_FINAL).expect("single-grid evolve");

    let counts = [2usize, 2];
    let half = N / 2;
    let mut tiles: Vec<(Sim, Kern)> = (0..4)
        .map(|flat| {
            let tc = unflatten(flat, counts);
            let bnd = Boundaries(std::array::from_fn(|a| {
                let face = |edge: bool| if edge { BoundaryType::Outflow } else { BoundaryType::CoarseFine };
                [face(tc[a] == 0), face(tc[a] == 1)]
            }));
            make([half, half], [tc[0] * half, tc[1] * half], bnd)
        })
        .collect();
    {
        let mut stores: Vec<&mut Sim> = Vec::new();
        let mut kernels: Vec<&Kern> = Vec::new();
        for (s, k) in tiles.iter_mut() {
            stores.push(s);
            kernels.push(k);
        }
        let mut stores: Vec<&mut _> = stores.into_iter().map(|s| &mut **s).collect();
        evolve_decomposed(
            &mut stores,
            &kernels,
            counts,
            &[0, 0, 0, 0],
            Timestepping::Rk2,
            0.0,
            T_FINAL,
            u64::MAX,
            &LocalCopy,
            |_, _, _| std::ops::ControlFlow::Continue(()),
        );
    }

    for (flat, (tile, _)) in tiles.iter().enumerate() {
        let tc = unflatten(flat, counts);
        assert_eq!(tile.time.to_bits(), single.time.to_bits(), "tile {flat}: time differs");
        assert_eq!(tile.dt.to_bits(), single.dt.to_bits(), "tile {flat}: dt differs");
        for c in tile.geom.interior.iter() {
            let g = [c[0] + (tc[0] * half) as isize - tile.geom.interior.spaces[0].lo, c[1] + (tc[1] * half) as isize - tile.geom.interior.spaces[1].lo];
            let g = [g[0] + single.geom.interior.spaces[0].lo, g[1] + single.geom.interior.spaces[1].lo];
            for (name, a, b) in [
                ("cons.den", &tile.fields.cons.den, &single.fields.cons.den),
                ("cons.mom[0]", &tile.fields.cons.mom[0], &single.fields.cons.mom[0]),
                ("cons.mom[1]", &tile.fields.cons.mom[1], &single.fields.cons.mom[1]),
                ("cons.nrg", tile.fields.cons.nrg_field().unwrap(), single.fields.cons.nrg_field().unwrap()),
            ] {
                assert_eq!(a.view().at(c).to_bits(), b.view().at(g).to_bits(), "tile {flat}: {name} differs at {g:?}");
            }
        }
    }

    let (mass, energy) = ledger(&single);
    assert!(mass.abs() > 1e-12 && energy.abs() > 1e-12, "the horizon booked nothing; the gate is vacuous");
    for (flat, (tile, _)) in tiles.iter().enumerate() {
        let (tile_mass, tile_energy) = ledger(tile);
        assert_eq!(tile_mass.to_bits(), mass.to_bits(), "tile {flat}: accreted mass {tile_mass:e} against {mass:e}");
        assert_eq!(tile_energy.to_bits(), energy.to_bits(), "tile {flat}: accreted energy {tile_energy:e} against {energy:e}");
    }
}
