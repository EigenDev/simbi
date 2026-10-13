// =============================================================================
// refinement_horizon_ledger.rs
//
// the horizon ledger of a refined excised kerr-schild run is the root's. the
// root books the flux through the diagnostic shell once per accepted root step,
// and the refined patch keeps clear of the shell and its stencil band, so the
// root's shell faces carry the flux the root applied. a run whose patch sits in
// a corner, away from the shell:
// - books a nonzero ledger on every level's copy of the horizon body, the same
//   on each, and the same as the unrefined run's within the patch's influence on
//   the root flow (the patch is far from the shell);
// - on two host tiles cut through the shell and three uneven tiles cut through
//   the shell and the patch, matches the single hierarchy bit for bit: state,
//   clock and ledger, the tiles' exact shell partials merging into the whole
//   root's.
//
// run: cargo test -p symbi --test refinement_horizon_ledger
// =============================================================================

use symbi::prelude::*;
use symbi::regimes::substrate_rhd::RhdSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, Partition, unflatten};
use symbi::sim::refinement::launched::receiving_faces_of_tile;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion, evolve_hierarchy_decomposed, seed_decomposed_fine_from_coarse};
use symbi::sim::substrate_seam::WithExcision;
use symbi_geometry::SchwarzschildKSCartesian;
use symbi_hydro::Rhd;
use symbi_ib::{Body, BodyCollection, BodyKind};
use symbi_sim::atlas::TileId;

const GAMMA: f64 = 4.0 / 3.0;
const MASS: f64 = 0.3;
const R_EXC: f64 = 0.35;
const R_DIAG: f64 = 0.9;
const N: usize = 32;
const LO: f64 = -1.2;
const DX: f64 = 2.4 / N as f64;
const T_FINAL: f64 = 0.06;
// the patch over root cells 26..30 on both axes: its nearest point lies at r = 1.47, outside the
// shell band of r = 0.9 +- 4 root diagonals (0.42)
const PATCH: [(usize, usize); 2] = [(26, 30), (26, 30)];

type Sim = SimState<Rhd, 2, SchwarzschildKSCartesian<f64>, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = RhdSubstrateKernelSet<HostMemory, f64, 2>;
type Hier = Hierarchy<Rhd, 2, 2, SchwarzschildKSCartesian<f64>, IdealGas<f64>, CpuSpace, HostMemory, Kern>;

fn region() -> RefinementRegion<2> {
    RefinementRegion {
        x_lo: std::array::from_fn(|a| LO + PATCH[a].0 as f64 * DX),
        x_hi: std::array::from_fn(|a| LO + PATCH[a].1 as f64 * DX),
    }
}

/// a gas falling onto the hole from a uniform rest state, with a density gradient so the flow
/// across the shell is not symmetric
fn root(cells: [usize; 2], lo: [usize; 2], bnd: Boundaries<2>) -> Sim {
    Sim::build(Rhd, IdealGas { gamma: GAMMA }, SchwarzschildKSCartesian { mass: MASS })
        .cells(cells)
        .spacing([DX; 2])
        .origin([LO; 2])
        .index_offset(std::array::from_fn(|a| lo[a] as isize))
        .boundaries(bnd)
        .cfl(0.3)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("kerr-schild simulation construction")
        .set_initial(|x| Prim::adiabatic(Density(1.0 + 0.2 * x[0]), Tensor::new([0.0, 0.0]), Pressure(0.1)))
        .build()
}

fn kset(s: &Sim) -> Kern {
    Kern::new(GAMMA, 0.3, &s.geom.allocated).with_excision(R_EXC, 1.0, 1.0)
}

fn horizon() -> BodyCollection<f64, 2> {
    BodyCollection::new().add(Body::horizon(0, R_EXC, R_DIAG))
}

fn single(refined: bool) -> Hier {
    let r = root([N, N], [0, 0], Boundaries::uniform(BoundaryType::Outflow));
    let k = kset(&r);
    let h = if refined {
        let h = Hier::with_refinement(r, k, &[region()], ProlongOrder::Ppm, kset).expect("refined hierarchy");
        h.seed_fine_from_coarse().expect("fine seed");
        h
    } else {
        Hier::single(r, k)
    };
    let mut h = h.with_bodies(horizon());
    h.prime();
    h
}

fn tiles(partition: &Partition<2>) -> Vec<Hier> {
    let counts = partition.counts();
    let whole = region();
    let mut tiles = Vec::new();
    for flat in 0..partition.n_tiles() {
        let tc = unflatten(flat, counts);
        let ext = partition.tile_extents(tc);
        let bnd = Boundaries(std::array::from_fn(|a| {
            let face = |edge: bool| if edge { BoundaryType::Outflow } else { BoundaryType::CoarseFine };
            [face(tc[a] == 0), face(tc[a] == counts[a] - 1)]
        }));
        let r = root([ext[0].1, ext[1].1], [ext[0].0, ext[1].0], bnd);
        let k = kset(&r);
        let lo: [f64; 2] = std::array::from_fn(|a| LO + ext[a].0 as f64 * DX);
        let hi: [f64; 2] = std::array::from_fn(|a| LO + (ext[a].0 + ext[a].1) as f64 * DX);
        let clip: [(f64, f64); 2] = std::array::from_fn(|a| (whole.x_lo[a].max(lo[a]), whole.x_hi[a].min(hi[a])));
        let h = if (0..2).all(|a| clip[a].0 < clip[a].1) {
            let c = RefinementRegion { x_lo: [clip[0].0, clip[1].0], x_hi: [clip[0].1, clip[1].1] };
            Hier::with_refinement_of_contributing(r, k, &[c], &[whole], ProlongOrder::Ppm, kset).expect("refined tile")
        } else {
            match receiving_faces_of_tile(partition, PATCH, TileId(flat as u32)) {
                Some((p, f)) => Hier::single(r, k).with_receiving_register(&p, f).expect("receiving tile"),
                None => Hier::single(r, k),
            }
        };
        tiles.push(h);
    }
    seed_decomposed_fine_from_coarse(&tiles, counts, &vec![0; tiles.len()], &LocalCopy).expect("decomposed seed");
    tiles
        .into_iter()
        .map(|h| {
            let mut h = h.with_bodies(horizon());
            h.prime();
            h
        })
        .collect()
}

/// the horizon body's (accreted mass, accreted energy) on every level, coarsest first
fn ledgers(h: &Hier) -> Vec<(f64, f64)> {
    h.levels
        .iter()
        .map(|l| match l.state.immersed.as_ref().expect("bodies").bodies.get(0).kind {
            BodyKind::Horizon { total_accreted_mass, total_accreted_energy, .. } => (total_accreted_mass, total_accreted_energy),
            _ => panic!("body 0 is no horizon"),
        })
        .collect()
}

fn densities(h: &Hier, level: usize, origin: [isize; 2]) -> Vec<([isize; 2], u64)> {
    let s = &h.levels[level].state;
    let lo = [s.geom.interior.spaces[0].lo, s.geom.interior.spaces[1].lo];
    s.geom
        .interior
        .iter()
        .map(|c| ([origin[0] + c[0] - lo[0], origin[1] + c[1] - lo[1]], s.fields.cons.den.view().at(c).to_bits()))
        .collect()
}

#[test]
fn a_refined_run_books_the_roots_shell_flux_on_every_level() {
    let mut refined = single(true);
    refined.evolve(T_FINAL).expect("refined evolve");
    let mut plain = single(false);
    plain.evolve(T_FINAL).expect("unrefined evolve");
    assert!(refined.crash.is_none() && plain.crash.is_none());
    let levels = ledgers(&refined);
    assert_eq!(levels.len(), 2, "the run holds a refined level");
    let (mass, energy) = levels[0];
    assert!(mass.abs() > 1e-6 && energy.abs() > 1e-6, "the refined run booked ({mass:e}, {energy:e}); its ledger is empty");
    for (level, &(m, e)) in levels.iter().enumerate() {
        assert_eq!((m.to_bits(), e.to_bits()), (mass.to_bits(), energy.to_bits()), "level {level} holds another ledger");
    }
    // the patch sits outside the shell band, far from the shell: the root flow across the shell
    // stays close to the unrefined one, whose ledger the refined one reproduces to the patch's
    // reach. the measured gap is recorded below; it bounds nothing about the method
    let (plain_mass, plain_energy) = ledgers(&plain)[0];
    let gap = ((mass - plain_mass) / plain_mass).abs().max(((energy - plain_energy) / plain_energy).abs());
    assert!(gap < 1e-2, "the refined ledger ({mass:e}, {energy:e}) departs from the unrefined ({plain_mass:e}, {plain_energy:e}) by {gap:e}");
}

#[test]
fn refined_host_tiles_cut_through_the_shell_book_the_single_hierarchy_ledger() {
    let mut want = single(true);
    want.evolve(T_FINAL).expect("single evolve");
    let (mass, energy) = ledgers(&want)[0];
    let root: std::collections::HashMap<_, _> = densities(&want, 0, [0, 0]).into_iter().collect();
    // x = 16 cuts the shell; 9 and 28 cut the shell and the patch
    for cuts in [[vec![16], vec![]], [vec![9, 28], vec![]]] {
        let partition = Partition::explicit([N, N], cuts.clone()).expect("cuts inside the grid");
        let mut t = tiles(&partition);
        let devices = vec![0; t.len()];
        evolve_hierarchy_decomposed(&mut t, partition.counts(), &devices, &LocalCopy, Timestepping::Rk2, 0.0, T_FINAL, u64::MAX, |_, _, _| {
            std::ops::ControlFlow::Continue(())
        });
        for (flat, h) in t.iter().enumerate() {
            assert!(h.crash.is_none(), "cuts {cuts:?}: tile {flat} crashed");
            assert_eq!(h.levels[0].state.iteration, want.levels[0].state.iteration, "cuts {cuts:?}: tile {flat} stopped short");
            let ext = partition.tile_extents(unflatten(flat, partition.counts()));
            for (g, bits) in densities(h, 0, [ext[0].0 as isize, ext[1].0 as isize]) {
                assert_eq!(bits, root[&g], "cuts {cuts:?}: root cell {g:?}");
            }
            for (level, (m, e)) in ledgers(h).into_iter().enumerate() {
                assert_eq!((m.to_bits(), e.to_bits()), (mass.to_bits(), energy.to_bits()), "cuts {cuts:?}: tile {flat} level {level} ledger ({m:e}, {e:e}) against ({mass:e}, {energy:e})");
            }
        }
    }
}
