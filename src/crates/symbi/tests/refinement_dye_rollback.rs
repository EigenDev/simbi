// =============================================================================
// refinement_dye_rollback.rs
//
// a rejection in the second fine substep restores the fine level's passive
// scalar to the root attempt's entry along with the gas: the fine level's own
// step snapshot is rewritten by every substep, so the attempt entry lives in a
// buffer of its own, and that buffer carries the complete conserved schema.
// the dye starts nonuniform so a stale concentration is visible. two arms,
// each against a clean hierarchy driven with the accepted timesteps: the
// single hierarchy, and two tiles cut inside the patch under the in-process
// decomposed driver.
//
// run: cargo test -p symbi --test refinement_dye_rollback
// =============================================================================
use std::collections::HashMap;
use std::ops::ControlFlow;

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, Partition, unflatten};
use symbi::sim::refinement::{
    Hierarchy, ProlongOrder, RefinementRegion, evolve_hierarchy_decomposed, seed_decomposed_fine_from_coarse,
};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_grid::Field;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
type Hier = Hierarchy<Newtonian, 2, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kern>;

const N: usize = 32;
const DX: f64 = 1.0 / N as f64;
const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const NG: usize = 2;
const RATIO: usize = 2;
const STEPS: u64 = 8;
const PATCH: [(usize, usize); 2] = [(12, 20), (12, 20)];
/// the cut of the decomposed arm, inside the patch
const CUT: usize = 17;
/// the forced rejection: the last stage of the second fine substep of step 3
const REJECT_AT: (u64, u8, usize) = (3, 1, 1);

fn bump(x: f64, y: f64) -> f64 {
    0.3 * (-(((x - 0.5) / 0.08).powi(2) + ((y - 0.5) / 0.08).powi(2))).exp()
}

/// the dye concentration at a position, a smooth nonuniform field
fn dye(x: f64, y: f64) -> f64 {
    0.5 + 0.4 * (std::f64::consts::TAU * x).sin() * (std::f64::consts::TAU * y).cos()
}

/// a root tile of `cells` placed as production places it: the whole grid's origin and the index
/// offset of its first cell
fn build_root(cells: [usize; 2], offset: [usize; 2], bnd: Boundaries<2>) -> Sim {
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 2])
        .origin([0.0; 2])
        .index_offset([offset[0] as isize, offset[1] as isize])
        .boundaries(bnd)
        .cfl(CFL)
        .ghosts(NG)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("root sim")
        .set_initial(|[x, y]| {
            let b = bump(x, y);
            Prim::adiabatic(Density(1.0 + b), Tensor::new([0.0, 0.0]), Pressure(1.0 + b))
        })
        .build()
        .with_passive_scalar()
        .expect("dye slots")
}

fn kset(sim: &Sim) -> Kern {
    Kern::new(GAMMA, CFL, &sim.geom.allocated)
}

fn patch_region() -> RefinementRegion<2> {
    RefinementRegion {
        x_lo: [PATCH[0].0 as f64 * DX, PATCH[1].0 as f64 * DX],
        x_hi: [PATCH[0].1 as f64 * DX, PATCH[1].1 as f64 * DX],
    }
}

/// seed the dye of one level over its interior from the cell positions: the conserved
/// `rho chi` from the conserved density and the concentration
fn seed_dye(state: &FieldStore<2, 2, HostMemory, f64>) {
    let rho = state.fields.cons.den.view();
    let cons_chi = state.fields.cons.chi_field().expect("cons chi");
    let prim_chi = state.fields.prim.chi_field().expect("prim chi");
    for c in state.geom.interior.iter() {
        let [x, y] = state.geom.cell_coord(c);
        let chi = dye(x, y);
        cons_chi.view_mut().set(c, *rho.at(c) * chi);
        prim_chi.view_mut().set(c, chi);
    }
}

fn build_mono() -> Hier {
    let root = build_root([N, N], [0, 0], Boundaries::uniform(BoundaryType::Outflow));
    let k = kset(&root);
    let mut h = Hier::with_refinement(root, k, &[patch_region()], ProlongOrder::Ppm, kset).expect("mono hier");
    h.seed_fine_from_coarse().expect("seed fine");
    seed_dye(&h.levels[0].state);
    seed_dye(&h.levels[1].state);
    h.prime();
    h
}

/// the two tiles of the cut at `CUT`, each refining the patch clipped to its extent
fn build_tiles() -> (Vec<Hier>, Partition<2>) {
    let partition = Partition::explicit([N, N], [vec![CUT], Vec::new()]).expect("partition");
    let counts = partition.counts();
    let mut tiles = Vec::new();
    for flat in 0..partition.n_tiles() {
        let tc = unflatten(flat, counts);
        let ext = partition.tile_extents(tc);
        let m: [usize; 2] = [ext[0].1, ext[1].1];
        let offset: [usize; 2] = [ext[0].0, ext[1].0];
        let lo: [f64; 2] = std::array::from_fn(|a| offset[a] as f64 * DX);
        let hi: [f64; 2] = std::array::from_fn(|a| lo[a] + m[a] as f64 * DX);
        let bnd = Boundaries(std::array::from_fn(|a| {
            let face = |edge: bool| if edge { BoundaryType::Outflow } else { BoundaryType::CoarseFine };
            [face(tc[a] == 0), face(tc[a] == counts[a] - 1)]
        }));
        let root = build_root(m, offset, bnd);
        let whole = patch_region();
        let clipped = RefinementRegion {
            x_lo: [whole.x_lo[0].max(lo[0]), whole.x_lo[1].max(lo[1])],
            x_hi: [whole.x_hi[0].min(hi[0]), whole.x_hi[1].min(hi[1])],
        };
        let k = kset(&root);
        tiles.push(Hier::with_refinement_of(root, k, &[clipped], &[whole], ProlongOrder::Ppm, kset).expect("tile hier"));
    }
    seed_decomposed_fine_from_coarse(&tiles, counts, &vec![0; tiles.len()], &LocalCopy).expect("decomposed seed");
    for h in tiles.iter_mut() {
        seed_dye(&h.levels[0].state);
        seed_dye(&h.levels[1].state);
        h.prime();
    }
    (tiles, partition)
}

/// (level, global i, global j, field, bits) of every conserved and primitive component of
/// every interior cell on both levels of a hierarchy whose root tile starts at `ext`
fn dump(h: &Hier, ext: [(usize, usize); 2], out: &mut HashMap<(u8, usize, usize, u8), u64>) {
    let root = &h.levels[0].state;
    let fine = &h.levels[1].state;
    let rlo: [isize; 2] = std::array::from_fn(|a| root.geom.interior.spaces[a].lo);
    let flo: [isize; 2] = std::array::from_fn(|a| fine.geom.interior.spaces[a].lo);
    let clo: [isize; 2] = std::array::from_fn(|a| h.levels[0].coverage.as_ref().unwrap().spaces[a].lo);
    for (level, state) in [(0u8, root), (1u8, fine)] {
        let cons = &state.fields.cons;
        let prim = &state.fields.prim;
        let fields: Vec<&Field<f64, 2, HostMemory>> = vec![
            &cons.den,
            &cons.mom[0],
            &cons.mom[1],
            cons.nrg_field().unwrap(),
            cons.chi_field().unwrap(),
            &prim.rho,
            &prim.vel[0],
            &prim.vel[1],
            prim.pre_field().unwrap(),
            prim.chi_field().unwrap(),
        ];
        for c in state.geom.interior.iter() {
            let g: [usize; 2] = std::array::from_fn(|a| {
                if level == 0 {
                    (ext[a].0 as isize + c[a] - rlo[a]) as usize
                } else {
                    ((ext[a].0 as isize + clo[a] - rlo[a]) * RATIO as isize + c[a] - flo[a]) as usize
                }
            });
            for (f, field) in fields.iter().enumerate() {
                let stale = out.insert((level, g[0], g[1], f as u8), field.view().at(c).to_bits());
                assert!(stale.is_none(), "cell held twice");
            }
        }
    }
}

/// the single hierarchy driven with the accepted timesteps
fn clean_reference(dts: &[f64]) -> HashMap<(u8, usize, usize, u8), u64> {
    let mut mono = build_mono();
    for &dt in dts {
        mono.step_root_with_dt(dt);
    }
    let mut out = HashMap::new();
    dump(&mono, [(0, N), (0, N)], &mut out);
    out
}

/// the premise that the gate has something to see: the fine dye is nonuniform at the end
fn assert_dye_nonuniform(values: &HashMap<(u8, usize, usize, u8), u64>) {
    let fine_chi: Vec<f64> = values.iter().filter(|(k, _)| k.0 == 1 && k.3 == 9).map(|(_, &b)| f64::from_bits(b)).collect();
    let (lo, hi) = fine_chi.iter().fold((f64::MAX, f64::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    assert!(hi - lo > 0.1, "the fine dye is uniform ({lo}..{hi}); a stale concentration would be invisible");
}

fn assert_equal(
    got: &HashMap<(u8, usize, usize, u8), u64>,
    want: &HashMap<(u8, usize, usize, u8), u64>,
    label: &str,
) {
    assert_eq!(got.len(), want.len(), "{label}: cell counts differ");
    let mut differing: HashMap<(u8, u8), usize> = HashMap::new();
    for (k, v) in got {
        if want[k] != *v {
            *differing.entry((k.0, k.3)).or_default() += 1;
        }
    }
    assert!(differing.is_empty(), "{label}: cells differing from the clean run by (level, field): {differing:?}");
}

/// the single hierarchy rejects the last stage of the second fine substep of step 3 and
/// replays the step at the halved timestep; the run equals a clean hierarchy driven with
/// the accepted timesteps in every component, the dye included
#[test]
fn a_fine_rejection_restores_the_dye_on_the_single_hierarchy() {
    let mut h = build_mono();
    h.reject_fine_once_at(REJECT_AT.0, REJECT_AT.1, REJECT_AT.2);
    let mut dts = Vec::new();
    for _ in 0..STEPS {
        let dt = h.root_cfl_dt();
        h.step_root_with_dt(dt);
        dts.push(h.levels[0].state.dt);
    }
    assert!(!h.forced_fine_rejection_pending(), "the forced rejection never fired; the gate is vacuous");
    assert!(dts[3] < 0.75 * dts[2], "step 3 was accepted at the full timestep: {dts:?}");
    let mut got = HashMap::new();
    dump(&h, [(0, N), (0, N)], &mut got);
    assert_dye_nonuniform(&got);
    assert_equal(&got, &clean_reference(&dts), "the rejecting single hierarchy");
}

/// the refined tile on the left of a cut inside the patch rejects in the second fine
/// substep; both tiles roll the root step back and the composite equals the clean single
/// hierarchy driven with the accepted timesteps in every component, the dye included
#[test]
fn a_fine_rejection_restores_the_dye_on_decomposed_tiles() {
    let (mut tiles, partition) = build_tiles();
    tiles[0].reject_fine_once_at(REJECT_AT.0, REJECT_AT.1, REJECT_AT.2);
    let mut dts = Vec::new();
    evolve_hierarchy_decomposed(
        &mut tiles,
        partition.counts(),
        &[0, 0],
        &LocalCopy,
        Timestepping::Rk2,
        0.0,
        f64::MAX,
        1,
        |iter, _, tiles| {
            dts.push(tiles[0].levels[0].state.dt);
            if iter >= STEPS { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
        },
    );
    assert_eq!(dts.len(), STEPS as usize);
    assert!(!tiles[0].forced_fine_rejection_pending(), "the forced rejection never fired; the gate is vacuous");
    assert!(dts[3] < 0.75 * dts[2], "step 3 was accepted at the full timestep: {dts:?}");
    let mut got = HashMap::new();
    for (flat, h) in tiles.iter().enumerate() {
        dump(h, partition.tile_extents(unflatten(flat, partition.counts())), &mut got);
    }
    assert_dye_nonuniform(&got);
    assert_equal(&got, &clean_reference(&dts), "the rejecting decomposed tiles");
}

