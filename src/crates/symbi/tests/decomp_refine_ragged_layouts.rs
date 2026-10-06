// =============================================================================
// decomp_refine_ragged_layouts.rs
//
// a refined patch spanning two cuts whose positions sit off the patch's
// center: the root tiles and the fine tiles are unequal on both axes. the
// decomposed hierarchy must equal the single hierarchy interior cell for
// interior cell after one root step, under a PLM root and under a PPM root
// with quartic prolongation. both runs feed the phase probe, so a failure
// names the first operation (per level and ordinal), tile, cell and field
// after which the two hierarchies differ, before the composite is compared.
//
// run: cargo test -p symbi --test decomp_refine_ragged_layouts
// =============================================================================
use std::collections::HashMap;

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, Partition, unflatten};
use symbi::sim::refinement::{
    Hierarchy, ProlongOrder, RefinementRegion, evolve_hierarchy_decomposed,
    seed_decomposed_fine_from_coarse,
};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_discretize::Recon;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_refinement::refinement::phase_probe::{self, PhaseSample};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
type Hier = Hierarchy<Newtonian, 2, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kern>;

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const N: usize = 32;
const DX: f64 = 1.0 / N as f64;
// one root step (the cfl step is about 0.0057)
const T_ONE_STEP: f64 = 0.003;
const RATIO: usize = 2;
// the global refined patch: root cells 12..20 on both axes
const PX_LO: f64 = 0.375;
const PX_HI: f64 = 0.625;
const PY_LO: f64 = 0.375;
const PY_HI: f64 = 0.625;
const PATCH_LO_FINE: isize = 24;

/// the evolution reconstruction with its halo width, and the prolongation into the patch
#[derive(Clone, Copy)]
struct Scheme {
    recon: Recon,
    ng: usize,
    prolong: ProlongOrder,
}

const PLM: Scheme = Scheme { recon: Recon::Plm, ng: 2, prolong: ProlongOrder::Ppm };
const PPM: Scheme = Scheme { recon: Recon::Ppm, ng: 3, prolong: ProlongOrder::Quartic };

fn bump(x: f64, y: f64) -> f64 {
    0.3 * (-(((x - 0.5) / 0.08).powi(2) + ((y - 0.5) / 0.08).powi(2))).exp()
}

/// a root tile placed as production places it: the whole grid's origin and the index offset of
/// its first cell
fn build_root(cells: [usize; 2], offset: [usize; 2], bnd: Boundaries<2>, ng: usize) -> Sim {
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 2])
        .origin([0.0; 2])
        .index_offset(std::array::from_fn(|a| offset[a] as isize))
        .boundaries(bnd)
        .cfl(CFL)
        .ghosts(ng)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("root sim construction failed")
        .set_initial(|[x, y]| {
            let b = bump(x, y);
            Prim::adiabatic(Density(1.0 + b), Tensor::new([0.0, 0.0]), Pressure(1.0 + b))
        })
        .build()
}

fn kset_plm(sim: &Sim) -> Kern {
    Kern::new(GAMMA, CFL, &sim.geom.allocated).reconstruction(Recon::Plm)
}

fn kset_ppm(sim: &Sim) -> Kern {
    Kern::new(GAMMA, CFL, &sim.geom.allocated).reconstruction(Recon::Ppm)
}

fn kset_of(scheme: Scheme) -> fn(&Sim) -> Kern {
    match scheme.recon {
        Recon::Plm => kset_plm,
        Recon::Ppm => kset_ppm,
    }
}

fn build_mono(scheme: Scheme) -> Hier {
    let make = kset_of(scheme);
    let root = build_root([N, N], [0, 0], Boundaries::uniform(BoundaryType::Outflow), scheme.ng);
    let k = make(&root);
    let region = RefinementRegion { x_lo: [PX_LO, PY_LO], x_hi: [PX_HI, PY_HI] };
    let mut h = Hier::with_refinement(root, k, &[region], scheme.prolong, make).expect("mono hier");
    h.seed_fine_from_coarse().expect("seed fine");
    h.prime();
    h
}

/// root tiles from the partition's extents, each refining the global patch clipped to its
/// physical range; the fine interiors are seeded after the root conserved halos are exchanged
fn build_tiles(partition: &Partition<2>, scheme: Scheme) -> Vec<Hier> {
    let make = kset_of(scheme);
    let counts = partition.counts();
    let mut tiles = Vec::with_capacity(partition.n_tiles());
    for flat in 0..partition.n_tiles() {
        let tc = unflatten(flat, counts);
        let ext = partition.tile_extents(tc);
        let m: [usize; 2] = [ext[0].1, ext[1].1];
        let offset: [usize; 2] = [ext[0].0, ext[1].0];
        let lo: [f64; 2] = std::array::from_fn(|a| offset[a] as f64 * DX);
        let hi: [f64; 2] = std::array::from_fn(|a| (offset[a] + m[a]) as f64 * DX);
        let bnd = Boundaries(std::array::from_fn(|a| {
            let face = |edge: bool| if edge { BoundaryType::Outflow } else { BoundaryType::CoarseFine };
            [face(tc[a] == 0), face(tc[a] == counts[a] - 1)]
        }));
        let root = build_root(m, offset, bnd, scheme.ng);
        let cx = [PX_LO.max(lo[0]), PX_HI.min(hi[0])];
        let cy = [PY_LO.max(lo[1]), PY_HI.min(hi[1])];
        let k = make(&root);
        let h = if cx[0] < cx[1] && cy[0] < cy[1] {
            let region = RefinementRegion { x_lo: [cx[0], cy[0]], x_hi: [cx[1], cy[1]] };
            let whole = RefinementRegion { x_lo: [PX_LO, PY_LO], x_hi: [PX_HI, PY_HI] };
            Hier::with_refinement_of(root, k, &[region], &[whole], scheme.prolong, make).expect("tile hier")
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

fn run_decomposed(tiles: &mut [Hier], counts: [usize; 2], t_final: f64) {
    let devices: Vec<i32> = vec![0; tiles.len()];
    evolve_hierarchy_decomposed(
        tiles,
        counts,
        &devices,
        &LocalCopy,
        Timestepping::Rk2,
        0.0,
        t_final,
        u64::MAX,
        |_, _, _| std::ops::ControlFlow::Continue(()),
    );
}

/// the global cell index of a sample's cell: root cells by the tile's extent, fine cells by
/// the coverage's global origin, both from the tile's interior origin
struct TileFrame {
    root_origin: [isize; 2],
    fine_origin: Option<[isize; 2]>,
}

fn frames(tiles: &[Hier], partition: &Partition<2>) -> Vec<TileFrame> {
    let counts = partition.counts();
    tiles
        .iter()
        .enumerate()
        .map(|(flat, h)| {
            let ext = partition.tile_extents(unflatten(flat, counts));
            let root = &h.levels[0].state;
            let rlo: [isize; 2] = std::array::from_fn(|a| root.geom.interior.spaces[a].lo);
            let root_origin = [ext[0].0 as isize, ext[1].0 as isize];
            let fine_origin = h.levels[0].coverage.as_ref().map(|cov| {
                std::array::from_fn(|a| (root_origin[a] + (cov.spaces[a].lo - rlo[a])) * RATIO as isize)
            });
            TileFrame { root_origin, fine_origin }
        })
        .collect()
}

fn global_of(sample: &PhaseSample, origin: [isize; 2], cell: [isize; 3]) -> [isize; 2] {
    std::array::from_fn(|a| origin[a] + (cell[a] - sample.interior_lo[a]))
}

/// the mono sample's cells keyed by global index
fn index_mono(sample: &PhaseSample) -> HashMap<[isize; 2], usize> {
    let origin = if sample.level == 0 { [0, 0] } else { [PATCH_LO_FINE; 2] };
    sample.cells.iter().enumerate().map(|(k, c)| (global_of(sample, origin, *c), k)).collect()
}

/// the first phase, in the order the drivers run them, after which any tile's field differs
/// from the single hierarchy at a shared global cell; the report names the worst cell of it
fn first_divergence(mono: &[PhaseSample], dec: &[PhaseSample], frames: &[TileFrame]) -> Option<String> {
    // per level, the mono samples in order; per (level, tile), the decomposed samples in order
    let mut mono_by_level: HashMap<usize, Vec<&PhaseSample>> = HashMap::new();
    for s in mono {
        mono_by_level.entry(s.level).or_default().push(s);
    }
    let mut ordinal: HashMap<(usize, usize), usize> = HashMap::new();
    for d in dec {
        let k = ordinal.entry((d.level, d.tile)).or_insert(0);
        let m = mono_by_level
            .get(&d.level)
            .and_then(|v| v.get(*k))
            .unwrap_or_else(|| panic!("level {} tile {} has more samples than the single hierarchy", d.level, d.tile));
        assert_eq!(
            m.phase, d.phase,
            "level {} tile {} ordinal {}: the decomposed driver ran {} where the single hierarchy ran {}",
            d.level, d.tile, k, d.phase, m.phase
        );
        *k += 1;
        let index = index_mono(m);
        let frame = &frames[d.tile];
        let origin = if d.level == 0 { frame.root_origin } else { frame.fine_origin.expect("a fine sample on a refined tile") };
        let mut worst: Option<(f64, String)> = None;
        let mut extent: Option<([isize; 2], [isize; 2], usize, usize)> = None;
        let interior = |s: &PhaseSample, c: &[isize; 3]| -> bool {
            let lo = s.interior_lo;
            let hi = s.interior_hi;
            (0..2).all(|a| c[a] >= lo[a] && c[a] < hi[a])
        };
        for (c_index, c) in d.cells.iter().enumerate() {
            let g = global_of(d, origin, *c);
            let Some(&m_index) = index.get(&g) else { continue };
            let is_interior = interior(d, c);
            for ((name, dv), (_, mv)) in d.fields.iter().zip(&m.fields) {
                // the conserved halos are never exchanged and no stage reads them
                if *name == "den" && !is_interior {
                    continue;
                }
                let (x, y) = (dv[c_index], mv[m_index]);
                let diff = (x - y).abs();
                if diff > 1e-13 {
                    let e = extent.get_or_insert((g, g, 0, 0));
                    e.0 = [e.0[0].min(g[0]), e.0[1].min(g[1])];
                    e.1 = [e.1[0].max(g[0]), e.1[1].max(g[1])];
                    e.2 += 1;
                    e.3 += usize::from(is_interior);
                }
                if diff > 1e-13 && worst.as_ref().is_none_or(|w| diff > w.0) {
                    // the same global cell on the tile whose interior holds it
                    let owner = dec
                        .iter()
                        .filter(|o| o.level == d.level && o.phase == d.phase && o.tile != d.tile)
                        .find_map(|o| {
                            let oo = if o.level == 0 { frames[o.tile].root_origin } else { frames[o.tile].fine_origin? };
                            o.cells.iter().enumerate().find(|(_, oc)| global_of(o, oo, **oc) == g && interior(o, oc)).map(|(oi, _)| {
                                let ov = o.fields.iter().find(|(n, _)| n == name).map(|(_, v)| v[oi]).unwrap_or(f64::NAN);
                                format!("tile {} holds {:.9e} in its interior", o.tile, ov)
                            })
                        })
                        .unwrap_or_else(|| "no other tile holds it in its interior".into());
                    worst = Some((diff, format!(
                        "after {} on level {} (ordinal {} of that level), tile {}: global cell {:?} ({}) {} = {:.9e} against {:.9e} (diff {:.3e}); {}",
                        d.phase, d.level, *k - 1, d.tile, g, if is_interior { "interior" } else { "halo" }, name, x, y, diff, owner
                    )));
                }
            }
        }
        if let Some((_, report)) = worst {
            let (lo, hi, n, n_interior) = extent.unwrap();
            return Some(format!("{report}; {n} differing samples ({n_interior} interior) on this tile within global {lo:?}..={hi:?}"));
        }
    }
    None
}

fn assert_one_step_matches(cuts: [usize; 2], scheme: Scheme) {
    let partition = Partition::explicit([N, N], [vec![cuts[0]], vec![cuts[1]]])
        .expect("interior cuts are strictly increasing and inside the grid");
    let counts = partition.counts();
    use symbi::regimes::fofc::{fofc_reset_stats, fofc_stats};
    let mut mono = build_mono(scheme);
    fofc_reset_stats();
    phase_probe::start();
    mono.evolve(T_ONE_STEP).expect("mono evolve");
    let mono_log = phase_probe::take();
    let mono_fofc = fofc_stats();
    assert!(!mono_log.is_empty(), "the single hierarchy fed no phase samples; the probe is vacuous");

    let mut dec = build_tiles(&partition, scheme);
    fofc_reset_stats();
    phase_probe::start();
    run_decomposed(&mut dec, counts, T_ONE_STEP);
    let dec_log = phase_probe::take();
    let dec_fofc = fofc_stats();
    let frames = frames(&dec, &partition);
    if let Some(report) = first_divergence(&mono_log, &dec_log, &frames) {
        panic!(
            "cuts {cuts:?} ({:?}): the hierarchies first differ {report}; fofc (fallback, freeze) events: single {mono_fofc:?}, decomposed {dec_fofc:?}",
            scheme.recon
        );
    }
    // every fine interior cell of every tile against the single hierarchy at the end
    let mono_fine = mono_log.iter().rev().find(|s| s.level == 1).expect("a fine sample");
    let index = index_mono(mono_fine);
    let mut last_per_tile: HashMap<usize, &PhaseSample> = HashMap::new();
    for d in dec_log.iter().filter(|s| s.level == 1) {
        last_per_tile.insert(d.tile, d);
    }
    for d in last_per_tile.values() {
        let frame = &frames[d.tile];
        let origin = frame.fine_origin.expect("fine origin");
        for (c_index, c) in d.cells.iter().enumerate() {
            let g = global_of(d, origin, *c);
            if let Some(&m_index) = index.get(&g) {
                let (x, y) = (d.fields[0].1[c_index], mono_fine.fields[0].1[m_index]);
                assert!((x - y).abs() == 0.0, "cuts {cuts:?}: fine cell {g:?} rho {x} against {y}");
            }
        }
    }
}

#[test]
fn plm_root_cuts_18_14_match_after_one_root_step() {
    assert_one_step_matches([18, 14], PLM);
}

#[test]
fn plm_root_cuts_17_13_match_after_one_root_step() {
    assert_one_step_matches([17, 13], PLM);
}

#[test]
fn ppm_root_cuts_17_17_match_after_one_root_step() {
    assert_one_step_matches([17, 17], PPM);
}

/// a patch face lying on a root cut is refused by name: the fine faces on it and the coarse
/// cell their reflux corrects would live on different tiles, and no register contribution
/// crosses a cut yet
#[test]
fn a_patch_face_on_a_cut_is_refused_by_name() {
    let partition = Partition::explicit([N, N], [vec![12], vec![16]]).expect("cuts inside the grid");
    let counts = partition.counts();
    let flat = 0;
    let ext = partition.tile_extents(unflatten(flat, counts));
    let m: [usize; 2] = [ext[0].1, ext[1].1];
    let offset: [usize; 2] = [ext[0].0, ext[1].0];
    let bnd = Boundaries([
        [BoundaryType::Outflow, BoundaryType::CoarseFine],
        [BoundaryType::Outflow, BoundaryType::CoarseFine],
    ]);
    let root = build_root(m, offset, bnd, PLM.ng);
    // the patch's low x face at 0.375 = root cell 12 is the cut: this tile holds no patch cell,
    // the tile beyond the cut holds the patch's face on its own boundary
    let beyond = partition.tile_extents(unflatten(2, counts));
    let offset_beyond: [usize; 2] = [beyond[0].0, beyond[1].0];
    let lo_beyond: [f64; 2] = std::array::from_fn(|a| offset_beyond[a] as f64 * DX);
    let m_beyond: [usize; 2] = [beyond[0].1, beyond[1].1];
    let bnd_beyond = Boundaries([
        [BoundaryType::CoarseFine, BoundaryType::Outflow],
        [BoundaryType::Outflow, BoundaryType::CoarseFine],
    ]);
    let root_beyond = build_root(m_beyond, offset_beyond, bnd_beyond, PLM.ng);
    let hi: [f64; 2] = std::array::from_fn(|a| (offset_beyond[a] + m_beyond[a]) as f64 * DX);
    let clipped = RefinementRegion { x_lo: [PX_LO.max(lo_beyond[0]), PY_LO], x_hi: [PX_HI.min(hi[0]), PY_HI.min(hi[1])] };
    let whole = RefinementRegion { x_lo: [PX_LO, PY_LO], x_hi: [PX_HI, PY_HI] };
    drop(root);
    let k = kset_plm(&root_beyond);
    let err = Hier::with_refinement_of(root_beyond, k, &[clipped], &[whole], PLM.prolong, kset_plm)
        .err()
        .expect("a patch face on a cut was admitted");
    assert!(
        err.detail.contains("lies on a decomposition cut"),
        "the refusal names another reason: {}",
        err.detail
    );
}

/// a patch face one root cell from a cut is refused by name: the prolongation stencil of the
/// fine ghosts reaches beyond the root halo across the cut
#[test]
fn a_patch_face_one_cell_from_a_cut_is_refused_by_name() {
    let partition = Partition::explicit([N, N], [vec![11], vec![16]]).expect("cuts inside the grid");
    let counts = partition.counts();
    let beyond = partition.tile_extents(unflatten(2, counts));
    let offset: [usize; 2] = [beyond[0].0, beyond[1].0];
    let m: [usize; 2] = [beyond[0].1, beyond[1].1];
    let hi: [f64; 2] = std::array::from_fn(|a| (offset[a] + m[a]) as f64 * DX);
    let bnd = Boundaries([
        [BoundaryType::CoarseFine, BoundaryType::Outflow],
        [BoundaryType::Outflow, BoundaryType::CoarseFine],
    ]);
    let root = build_root(m, offset, bnd, PLM.ng);
    // the patch's low x face at root cell 12, one cell past this tile's cut at 11
    let clipped = RefinementRegion { x_lo: [PX_LO, PY_LO], x_hi: [PX_HI.min(hi[0]), PY_HI.min(hi[1])] };
    let whole = RefinementRegion { x_lo: [PX_LO, PY_LO], x_hi: [PX_HI, PY_HI] };
    let k = kset_plm(&root);
    let err = Hier::with_refinement_of(root, k, &[clipped], &[whole], PLM.prolong, kset_plm)
        .err()
        .expect("a patch face one cell from the cut was admitted");
    assert!(
        err.detail.contains("prolongation stencil reaches") && err.detail.contains("move the patch face"),
        "the refusal names another reason: {}",
        err.detail
    );
}

/// the centered layout under both schemes: the probe's alignment itself, on a case known to
/// hold
#[test]
fn centered_cuts_match_after_one_root_step() {
    assert_one_step_matches([16, 16], PLM);
    assert_one_step_matches([16, 16], PPM);
}

#[test]
fn ppm_root_cuts_18_14_match_after_one_root_step() {
    assert_one_step_matches([18, 14], PPM);
}

#[test]
fn ppm_root_cuts_17_13_match_after_one_root_step() {
    assert_one_step_matches([17, 13], PPM);
}
