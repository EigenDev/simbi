// =============================================================================
// decomp_refine_fofc_one_sided.rs
//
// the first-order flux correction on a refined, decomposed grid when trouble
// sits against a tile cut on one side only. a refined patch spans the cut, so
// the fine level carries the cut too; two cold streams part about a line two
// root cells left of the cut, inside the patch, and the evacuated fine cells
// between them fail the high-order recovery in the left tile's column against
// the fine cut while the right tile's recovery is clean; the streams leave
// through outflow ends, so the total mass falls the same way in both arms. the correction keeps
// the first-order flux on every face of a troubled cell, and a face on a cut
// belongs to two tiles: the clean neighbor has to learn of the trouble and
// redo its side, or the pair carries two fluxes on one face and creates or
// destroys mass on the cut.
//
// the gate asserts its premise (a step whose fine stage flagged the left
// tile's cut column and left the right tile's clean, the freeze tier silent),
// then bitwise equality of the composite grid with the one-tile hierarchy
// driven through the same loop, the total mass included.
//
// run: cargo test -p symbi --test decomp_refine_fofc_one_sided
// =============================================================================
use std::ops::ControlFlow;
use symbi::regimes::fofc::{fofc_reset_stats, fofc_stats};
use symbi::regimes::substrate_rhd::RhdSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, Partition, unflatten};
use symbi::sim::refinement::{
    Hierarchy, ProlongOrder, RefinementRegion, evolve_hierarchy_decomposed,
    seed_decomposed_fine_from_coarse,
};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::Rhd;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 4.0 / 3.0;
const CFL: f64 = 0.4;
const NX: usize = 64;
const NY: usize = 8;
const DX: f64 = 1.0 / NX as f64;
const RATIO: usize = 2;
const CUT: usize = NX / 2;
/// the parting line, two root cells left of the cut, inside the refined patch
const X_PART: f64 = (CUT - 2) as f64 * DX;
/// the refined patch spanning the cut and the parting line, over the full height
const PX_LO: f64 = 0.375;
const PX_HI: f64 = 0.625;
const STEPS: u64 = 12;
/// the step by which the fine cut column has been corrected with the right tile clean, and
/// before the freeze tier has acted in either arm
const EARLY: u64 = 5;

type Sim = SimState<Rhd, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = RhdSubstrateKernelSet<HostMemory, f64, 2>;
type Hier = Hierarchy<Rhd, 2, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kern>;

fn kset(sim: &Sim) -> Kern {
    let mut k = Kern::new(GAMMA, CFL, &sim.geom.allocated);
    k.theta = 1.5;
    k
}

/// a root tile of `cells` placed as production places it: the whole grid's origin and the index
/// offset of its first cell
fn build_root(cells: [usize; 2], offset: [usize; 2], bnd: Boundaries<2>) -> Sim {
    Sim::build(Rhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 2])
        .origin([0.0; 2])
        .index_offset([offset[0] as isize, offset[1] as isize])
        .boundaries(bnd)
        .cfl(CFL)
        .timestepping(Timestepping::Euler)
        .allocate()
        .expect("root sim construction failed")
        .set_initial(|[x, _]| {
            // the streams part on X_PART and leave through the outflow ends
            let toward = if x >= X_PART { 0.99 } else { -0.96 };
            Prim::adiabatic(Density(1.0), Tensor::new([toward, 0.0]), Pressure(1e-3))
        })
        .build()
}

/// the tiles of a partition, each refining the global patch clipped to its own extent, so the
/// fine grids meet at the cut the patch crosses; a cut face is filled by the exchange, the x
/// ends flow out and y wraps on its own
fn build_tiles(partition: &Partition<2>) -> Vec<Hier> {
    let counts = partition.counts();
    let mut tiles = Vec::with_capacity(partition.n_tiles());
    for flat in 0..partition.n_tiles() {
        let tc = unflatten(flat, counts);
        let ext = partition.tile_extents(tc);
        let m: [usize; 2] = [ext[0].1, ext[1].1];
        let offset: [usize; 2] = [ext[0].0, ext[1].0];
        // the tile's physical x extent, against which the patch is clipped
        let lo = offset[0] as f64 * DX;
        let hi = lo + m[0] as f64 * DX;
        let bnd = Boundaries(std::array::from_fn(|a| {
            let edge = if a == 0 { BoundaryType::Outflow } else { BoundaryType::Periodic };
            [
                if tc[a] == 0 { edge } else { BoundaryType::CoarseFine },
                if tc[a] + 1 == counts[a] { edge } else { BoundaryType::CoarseFine },
            ]
        }));
        let root = build_root(m, offset, bnd);
        let cx = [PX_LO.max(lo), PX_HI.min(hi)];
        let k = kset(&root);
        let h = if cx[0] < cx[1] {
            let region = RefinementRegion { x_lo: [cx[0], 0.0], x_hi: [cx[1], NY as f64 * DX] };
            let whole = RefinementRegion { x_lo: [PX_LO, 0.0], x_hi: [PX_HI, NY as f64 * DX] };
            Hier::with_refinement_of(root, k, &[region], &[whole], ProlongOrder::Ppm, kset).expect("tile hierarchy")
        } else {
            Hier::single(root, k)
        };
        tiles.push(h);
    }
    seed_decomposed_fine_from_coarse(&tiles, counts, &vec![0; tiles.len()], &LocalCopy)
        .expect("decomposed fine seed");
    for h in tiles.iter_mut() {
        h.prime();
    }
    tiles
}

/// the flags in a level's interior column against its `hi_side` on axis 0
fn column_flags(sim: &FieldStore<2, 2, HostMemory, f64>, hi_side: bool) -> u64 {
    let interior = &sim.geom.interior;
    let (xlo, xhi) = (interior.spaces[0].lo, interior.spaces[0].hi);
    let col = if hi_side { xhi - 1 } else { xlo };
    let view = sim.workspace.fofc_flag.view();
    interior.iter().filter(|c| c[0] == col).filter(|c| *view.at(*c) != 0.0).count() as u64
}

/// the composite grid at the fine resolution: root cells outside the coverage as their fine
/// blocks, fine cells at their global fine index
fn composite_fine(tiles: &[Hier], partition: &Partition<2>) -> Vec<f64> {
    let (fnx, fny) = (RATIO * NX, RATIO * NY);
    let counts = partition.counts();
    let mut out = vec![f64::NAN; fnx * fny];
    for (flat, h) in tiles.iter().enumerate() {
        let ext = partition.tile_extents(unflatten(flat, counts));
        let root = &h.levels[0].state;
        let cov = h.levels[0].coverage.as_ref();
        let rlo: [isize; 2] = std::array::from_fn(|a| root.geom.interior.spaces[a].lo);
        for c in root.geom.interior.iter() {
            if cov.is_some_and(|cov| cov.contains(c)) {
                continue;
            }
            let g: [usize; 2] = std::array::from_fn(|a| ext[a].0 + (c[a] - rlo[a]) as usize);
            let d = *root.fields.cons.den.view().at(c);
            for sy in 0..RATIO {
                for sx in 0..RATIO {
                    out[(RATIO * g[1] + sy) * fnx + (RATIO * g[0] + sx)] = d;
                }
            }
        }
        if h.levels.len() > 1 {
            let fine = &h.levels[1].state;
            let flo: [isize; 2] = std::array::from_fn(|a| fine.geom.interior.spaces[a].lo);
            let cov = h.levels[0].coverage.as_ref().unwrap();
            let clo: [isize; 2] = std::array::from_fn(|a| cov.spaces[a].lo);
            for c in fine.geom.interior.iter() {
                let gf: [usize; 2] = std::array::from_fn(|a| {
                    (ext[a].0 + (clo[a] - rlo[a]) as usize) * RATIO + (c[a] - flo[a]) as usize
                });
                out[gf[1] * fnx + gf[0]] = *fine.fields.cons.den.view().at(c);
            }
        }
    }
    assert!(out.iter().all(|v| v.is_finite()), "a composite cell no tile wrote");
    out
}

/// the total mass over the composite, every fine cell weighing a fine cell's area
fn total_mass(tiles: &[Hier], partition: &Partition<2>) -> f64 {
    let area = (DX / RATIO as f64) * (DX / RATIO as f64);
    composite_fine(tiles, partition).iter().sum::<f64>() * area
}

struct Run {
    den: Vec<f64>,
    fallbacks: u64,
    freezes: u64,
    /// steps whose fine stage flagged the left tile's cut column and left the right tile's clean
    left_only_steps: u64,
    early_mass: f64,
    early_freezes: u64,
    /// (fallbacks, freezes) after every step
    census: Vec<(u64, u64)>,
}

fn run(partition: &Partition<2>) -> Run {
    let counts = partition.counts();
    let mut tiles = build_tiles(partition);
    let devices = vec![0i32; tiles.len()];
    let initial_mass = total_mass(&tiles, partition);
    let mut left_only_steps = 0u64;
    let (mut early_mass, mut early_freezes) = (f64::NAN, u64::MAX);
    let mut census = Vec::new();
    fofc_reset_stats();
    evolve_hierarchy_decomposed(
        &mut tiles,
        counts,
        &devices,
        &LocalCopy,
        Timestepping::Euler,
        0.0,
        f64::MAX,
        1,
        |step, _, tiles| {
            if tiles.len() == 2 && tiles[0].levels.len() > 1 && tiles[1].levels.len() > 1 {
                let left = column_flags(&tiles[0].levels[1].state, true);
                let right = column_flags(&tiles[1].levels[1].state, false);
                if left > 0 && right == 0 {
                    left_only_steps += 1;
                }
            }
            census.push(fofc_stats());
            if step == EARLY {
                early_mass = total_mass(tiles, partition);
                early_freezes = fofc_stats().1;
            }
            if step >= STEPS { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
        },
    );
    let (fallbacks, freezes) = fofc_stats();
    Run {
        den: composite_fine(&tiles, partition),
        fallbacks,
        freezes,
        left_only_steps,
        early_mass: early_mass / initial_mass,
        early_freezes,
        census,
    }
}

#[test]
fn trouble_on_one_side_of_a_fine_cut_matches_the_one_tile_hierarchy() {
    let whole = Partition::explicit([NX, NY], [Vec::new(), Vec::new()]).expect("one tile");
    let split = Partition::explicit([NX, NY], [vec![CUT], Vec::new()]).expect("one cut on x");

    let mono = run(&whole);
    let dec = run(&split);

    assert!(mono.fallbacks > 0, "the one-tile hierarchy never tripped the correction");
    assert!(
        dec.left_only_steps > 0,
        "no step flagged the left tile's fine cut column with the right tile clean; the gate is vacuous"
    );
    assert_eq!(
        (mono.early_freezes, dec.early_freezes),
        (0, 0),
        "the freeze tier acted before step {EARLY}; the corrected update is no longer conservative. \
         census per step (fallbacks, freezes): one tile {:?}, two tiles {:?}",
        mono.census,
        dec.census
    );
    assert_eq!(
        mono.early_mass.to_bits(),
        dec.early_mass.to_bits(),
        "the total mass at step {EARLY} differs between the arms ({:e} against {:e}): the cut carries two fluxes on one face",
        mono.early_mass,
        dec.early_mass
    );
    let differing = mono.den.iter().zip(&dec.den).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    assert_eq!(differing, 0, "{differing} composite cells differ from the one-tile hierarchy");
    assert_eq!(mono.freezes, dec.freezes, "the arms froze different cell counts");
}
