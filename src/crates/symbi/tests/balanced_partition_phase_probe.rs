// =============================================================================
// balanced_partition_phase_probe.rs
//
// a balanced refined root step on the single hierarchy and on three uneven host
// tiles, sampled after every named operation (phase_probe) and compared bit for
// bit, on the build's default memory (host memory in a cpu build, managed
// device memory in a gpu build). the problem: a 24^3 root on [-5, 5]^3 of gas at
// rest (rho 1, p 0.6) about a point mass at the origin, under the balanced
// reconstruction and the hllc flux, one fine box over root cells 6..18, the
// tiles cut at root cells 7 and 15 on the x axis. the cut at 15 lies inside the
// level-0 restriction band of the box (root cells 14..18).
// the gate holds every sample of every tile equal, bit for bit at every shared
// global cell, to the single hierarchy's sample of the same operation. a
// difference reports the first operation, level and cell after which a field
// differs, the differing bits, and both drivers' bits of every probed field at
// that cell in the sample before it (the operation's input). a difference after a
// restriction also reports the restriction's other operands: both drivers' last
// sample of the finer level before it at the cell's 2^3 children, and the coarse
// neighbors along the cut axis (x - 2 .. x + 2) in the input sample, and every
// band rewrite either driver ran on that level between the input sample and the
// restriction whose coarse cells share the cell's row along the seam normal: the
// seam (side, uncovered cell, reference rows), the geometry, the gamma and the
// body scalars the kernels read, the coarse chain order from the reference row to
// the cell, and the bits of every coarse operand on the row (density, velocity,
// pressure, energy, departure, anchor, reference) and every fine operand on the
// cell's children rows (density, velocity, pressure, departure, anchor,
// reference), before and after the rewrite.
//
// run: cargo test -p symbi --test balanced_partition_phase_probe
//      (with --features hip on a gpu node: the device operators)
// =============================================================================
#![allow(dead_code)]
use std::collections::HashMap;

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, Partition, unflatten};
use symbi::sim::refinement::launched::receiving_faces_of_tile;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion, evolve_hierarchy_decomposed, seed_decomposed_fine_from_coarse};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_discretize::Recon;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_refinement::refinement::phase_probe::{self, BandSample, PhaseSample};
use symbi_sim::atlas::TileId;
use symbi_xpu::{DefaultMemory, DefaultSpace};

type Sim = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, DefaultSpace, DefaultMemory>;
type Kern = AdiabaticSubstrateKernelSet<DefaultMemory, f64, 3>;
type Hier = Hierarchy<Newtonian, 3, 3, Cartesian, IdealGas<f64>, DefaultSpace, DefaultMemory, Kern>;

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const N: usize = 24;
const LO: f64 = -5.0;
const DX: f64 = 10.0 / N as f64;
const GM: f64 = 1.0;
const SOFTENING: f64 = 0.5;
const RATIO: usize = 2;
const PATCH: [(usize, usize); 3] = [(6, 18), (6, 18), (6, 18)];
const CUTS: [usize; 2] = [7, 15];
const STEPS: u64 = 2;

fn kset(s: &Sim) -> Kern {
    Kern::new(GAMMA, CFL, &s.geom.allocated)
        .reconstruction(Recon::Plm)
        .well_balanced_reconstruction(true)
        .with_solver(symbi::regimes::substrate_kernels::Solver::Hllc)
        .unwrap()
}

fn root(cells: [usize; 3], offset: [usize; 3], bnd: Boundaries<3>) -> Sim {
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 3])
        .origin([LO; 3])
        .index_offset(std::array::from_fn(|a| offset[a] as isize))
        .boundaries(bnd)
        .cfl(CFL)
        .ghosts(2)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .unwrap()
        .set_initial(|_x| Prim::adiabatic(Density(1.0), Tensor::new([0.0; 3]), Pressure(0.6)))
        .build()
}

fn bodies() -> symbi_ib::BodyCollection<f64, 3> {
    symbi_ib::BodyCollection::new().add(symbi_ib::Body::gravitational(0, Tensor::zeros(), Tensor::zeros(), GM, 0.0, SOFTENING))
}

fn region() -> RefinementRegion<3> {
    let x = |c: usize| LO + c as f64 * DX;
    RefinementRegion { x_lo: [x(PATCH[0].0); 3], x_hi: [x(PATCH[0].1); 3] }
}

fn mono() -> Hier {
    let r = root([N; 3], [0; 3], Boundaries::uniform(BoundaryType::Outflow));
    let k = kset(&r);
    let h = Hier::with_refinement(r, k, &[region()], ProlongOrder::Ppm, kset).unwrap();
    h.seed_fine_from_coarse().unwrap();
    let mut h = h.with_bodies(bodies());
    h.prime();
    h
}

fn tiles(partition: &Partition<3>) -> Vec<Hier> {
    let counts = partition.counts();
    let mut tiles = Vec::new();
    for flat in 0..partition.n_tiles() {
        let tc = unflatten(flat, counts);
        let ext = partition.tile_extents(tc);
        let m: [usize; 3] = std::array::from_fn(|a| ext[a].1);
        let off: [usize; 3] = std::array::from_fn(|a| ext[a].0);
        let bnd = Boundaries(std::array::from_fn(|a| {
            let face = |edge: bool| if edge { BoundaryType::Outflow } else { BoundaryType::CoarseFine };
            [face(tc[a] == 0), face(tc[a] == counts[a] - 1)]
        }));
        let r = root(m, off, bnd);
        let lo: [f64; 3] = std::array::from_fn(|a| LO + off[a] as f64 * DX);
        let hi: [f64; 3] = std::array::from_fn(|a| LO + (off[a] + m[a]) as f64 * DX);
        let w = region();
        let c_lo: [f64; 3] = std::array::from_fn(|a| w.x_lo[a].max(lo[a]));
        let c_hi: [f64; 3] = std::array::from_fn(|a| w.x_hi[a].min(hi[a]));
        let k = kset(&r);
        let h = if (0..3).all(|a| c_lo[a] < c_hi[a]) {
            Hier::with_refinement_of_contributing(r, k, &[RefinementRegion { x_lo: c_lo, x_hi: c_hi }], &[w], ProlongOrder::Ppm, kset).unwrap()
        } else {
            match receiving_faces_of_tile(partition, PATCH, TileId(flat as u32)) {
                Some((p, f)) => Hier::single(r, k).with_receiving_register(&p, f).unwrap(),
                None => Hier::single(r, k),
            }
        };
        tiles.push(h);
    }
    seed_decomposed_fine_from_coarse(&tiles, counts, &vec![0; tiles.len()], &LocalCopy).unwrap();
    tiles.into_iter().map(|h| { let mut h = h.with_bodies(bodies()); h.prime(); h }).collect()
}

/// each tile's root origin on the whole grid and its fine level's origin on the whole fine lattice
fn origins(dec: &[Hier], partition: &Partition<3>) -> Vec<([isize; 3], Option<[isize; 3]>)> {
    let counts = partition.counts();
    dec.iter()
        .enumerate()
        .map(|(flat, h)| {
            let ext = partition.tile_extents(unflatten(flat, counts));
            let root = &h.levels[0].state;
            let rlo: [isize; 3] = std::array::from_fn(|a| root.geom.interior.spaces[a].lo);
            let root_origin: [isize; 3] = std::array::from_fn(|a| ext[a].0 as isize);
            let fine = h.levels[0].coverage.as_ref().map(|cov| std::array::from_fn(|a| (root_origin[a] + (cov.spaces[a].lo - rlo[a])) * RATIO as isize));
            (root_origin, fine)
        })
        .collect()
}

fn global(sample: &PhaseSample, origin: [isize; 3], cell: [isize; 3]) -> [isize; 3] {
    std::array::from_fn(|a| origin[a] + (cell[a] - sample.interior_lo[a]))
}

fn interior(sample: &PhaseSample, cell: &[isize; 3]) -> bool {
    (0..3).all(|a| cell[a] >= sample.interior_lo[a] && cell[a] < sample.interior_hi[a])
}

/// every probed field of a sample at one global cell, as (name, value, bits)
fn values_at(sample: &PhaseSample, origin: [isize; 3], g: [isize; 3]) -> Vec<(&'static str, f64, u64)> {
    match sample.cells.iter().position(|c| global(sample, origin, *c) == g) {
        Some(k) => sample.fields.iter().map(|(n, v)| (*n, v[k], v[k].to_bits())).collect(),
        None => Vec::new(),
    }
}

/// both drivers' bits of every probed field at each listed global cell of two samples
fn operands(label: &str, a: &PhaseSample, a_origin: [isize; 3], b: &PhaseSample, b_origin: [isize; 3], cells: &[[isize; 3]]) -> String {
    let mut out = format!("\n  {label}, after {} (tiles) / {} (single):", a.phase, b.phase);
    for g in cells {
        let (x, y) = (values_at(a, a_origin, *g), values_at(b, b_origin, *g));
        if x.is_empty() || y.is_empty() {
            out += &format!("\n    cell {g:?}: held by {}", if x.is_empty() { "the single hierarchy alone" } else { "the tile alone" });
            continue;
        }
        for ((n, va, ab), (_, vb, bb)) in x.into_iter().zip(y) {
            out += &format!("\n    cell {g:?} {n}: {va:e} ({ab:#018x}) / {vb:e} ({bb:#018x}){}", if ab == bb { "" } else { "  differs" });
        }
    }
    out
}

/// every band rewrite of `level` logged in `window` (phase-log positions) on the seam whose band
/// holds whole-lattice cell `w`, every tile's part of it, with the coarse operands on `w`'s row along
/// the seam normal and the fine operands on the rows of `w`'s children (`fine_lo`: the whole fine
/// index of the child at offset 0 on every axis)
fn band_report(label: &str, bands: &[BandSample], level: usize, window: (usize, usize), w: [isize; 3], fine_lo: [isize; 3]) -> String {
    let logged: Vec<&BandSample> = bands.iter().filter(|b| b.level == level && b.logged > window.0 && b.logged <= window.1).collect();
    let row = |b: &BandSample, c: &[isize; 3]| (0..3).all(|a| a == b.axis || c[a] == w[a]);
    let holds = |b: &BandSample| (b.band.0..b.band.1).contains(&w[b.axis]) && b.coarse.cells.iter().any(|c| row(b, c));
    let Some(seam) = logged.iter().find(|b| holds(b)).map(|b| (b.axis, b.high)) else {
        return format!("\n  {label}: no band rewrite of level {level} in the window holds {w:?}");
    };
    let mut out = String::new();
    for b in logged.into_iter().filter(|b| (b.axis, b.high) == seam) {
        let ax = b.axis;
        let fine_row = |c: &[isize; 3]| (0..3).all(|a| a == ax || (c[a] - fine_lo[a] == 0 || c[a] - fine_lo[a] == 1));
        out += &format!("\n  {label} {} tile {} level {}: {}", b.stage, b.tile, b.level, b.seam);
        // the decode walks from the reference row through the band, away from it, one cell a step
        let mut chain = vec![b.reference];
        if b.reference < b.band.0 {
            chain.extend(b.band.0..b.band.1);
        } else {
            chain.extend((b.band.0..b.band.1).rev());
        }
        out += &format!("\n    coarse chain order along axis {ax} (whole index, reference first): {chain:?}");
        for (n, v) in &b.scalars {
            out += &format!("\n    {n} = {v:e} ({:#018x})", v.to_bits());
        }
        for (k, c) in b.coarse.cells.iter().enumerate().filter(|(_, c)| row(b, c)) {
            let values: Vec<String> = b.coarse.fields.iter().map(|(n, v)| format!("{n} {:e} ({:#018x})", v[k], v[k].to_bits())).collect();
            out += &format!("\n    coarse {c:?}: {}", values.join(", "));
        }
        for (k, c) in b.fine.cells.iter().enumerate().filter(|(_, c)| fine_row(c)) {
            let values: Vec<String> = b.fine.fields.iter().map(|(n, v)| format!("{n} {:e} ({:#018x})", v[k], v[k].to_bits())).collect();
            out += &format!("\n    fine {c:?}: {}", values.join(", "));
        }
    }
    out
}

/// the first operation after which a tile's interior field differs from the single hierarchy's
/// at a shared global cell, bit for bit, with the operation's input from the sample before it
fn first_difference(
    (mono, mono_bands): (&[PhaseSample], &[BandSample]),
    (dec, dec_bands): (&[PhaseSample], &[BandSample]),
    frames: &[([isize; 3], Option<[isize; 3]>)],
) -> Option<String> {
    let mut mono_by_level: HashMap<usize, Vec<&PhaseSample>> = HashMap::new();
    for s in mono {
        mono_by_level.entry(s.level).or_default().push(s);
    }
    let fine_origin = [PATCH[0].0 as isize * RATIO as isize; 3];
    let mut ordinal: HashMap<(usize, usize), usize> = HashMap::new();
    let mut previous: HashMap<(usize, usize), &PhaseSample> = HashMap::new();
    // the latest sample of each (level, tile) so far, and of each level on the single hierarchy at
    // the matching ordinal, for a restriction's finer operands
    let mut latest: HashMap<(usize, usize), &PhaseSample> = HashMap::new();
    for d in dec {
        let k = ordinal.entry((d.level, d.tile)).or_insert(0);
        let m = mono_by_level.get(&d.level).and_then(|v| v.get(*k)).unwrap_or_else(|| panic!("level {} tile {}: more samples than the single hierarchy", d.level, d.tile));
        assert_eq!(m.phase, d.phase, "level {} tile {} ordinal {}: the tiles ran {} where the single hierarchy ran {}", d.level, d.tile, k, d.phase, m.phase);
        let m_prev = if *k > 0 { mono_by_level[&d.level].get(*k - 1).copied() } else { None };
        *k += 1;
        let origin = if d.level == 0 { frames[d.tile].0 } else { frames[d.tile].1.expect("a fine sample on a refined tile") };
        let m_origin = if d.level == 0 { [0; 3] } else { fine_origin };
        let index: HashMap<[isize; 3], usize> = m.cells.iter().enumerate().map(|(i, c)| (global(m, m_origin, *c), i)).collect();
        for (ci, c) in d.cells.iter().enumerate() {
            if !interior(d, c) {
                continue;
            }
            let g = global(d, origin, *c);
            let Some(&mi) = index.get(&g) else { continue };
            for ((name, dv), (_, mv)) in d.fields.iter().zip(&m.fields) {
                if dv[ci].to_bits() == mv[mi].to_bits() {
                    continue;
                }
                let mut report = format!(
                    "after {} on level {} (ordinal {}), tile {}: global cell {g:?} {name} = {:e} ({:#018x}) against the single hierarchy's {:e} ({:#018x})",
                    d.phase, d.level, *k - 1, d.tile, dv[ci], dv[ci].to_bits(), mv[mi], mv[mi].to_bits()
                );
                let d_prev = previous.get(&(d.level, d.tile)).copied();
                if let (Some(dp), Some(mp)) = (d_prev, m_prev) {
                    report += &format!("\n  input, after {} (tiles) / {} (single):", dp.phase, mp.phase);
                    for ((n, a, ab), (_, b, bb)) in values_at(dp, origin, g).into_iter().zip(values_at(mp, m_origin, g)) {
                        report += &format!("\n    {n}: {a:e} ({ab:#018x}) / {b:e} ({bb:#018x}){}", if ab == bb { "" } else { "  differs" });
                    }
                    if d.phase == "restrict" {
                        let line: Vec<[isize; 3]> = (-2..=2).map(|o| [g[0] + o, g[1], g[2]]).filter(|c| *c != g).collect();
                        report += &operands("coarse neighbors along x", dp, origin, mp, m_origin, &line);
                    }
                }
                if d.phase == "restrict" {
                    let fine = d.level + 1;
                    // the single hierarchy's sample of the finer level at the ordinal the tile's latest one holds
                    let k_fine = ordinal.get(&(fine, d.tile)).copied().unwrap_or(0);
                    let mono_fine = k_fine.checked_sub(1).and_then(|k| mono_by_level.get(&fine).and_then(|v| v.get(k).copied()));
                    if let (Some(df), Some(mf), Some(f_origin)) = (latest.get(&(fine, d.tile)).copied(), mono_fine, frames[d.tile].1) {
                        let children: Vec<[isize; 3]> = (0..8)
                            .map(|k| std::array::from_fn(|a| g[a] * RATIO as isize + ((k >> a) & 1) as isize))
                            .collect();
                        report += &operands("finer level children", df, f_origin, mf, fine_origin, &children);
                    }
                    // the bands rewritten between the input sample and this restriction, by phase-log
                    // position; whole-lattice indices are local + index offset, the single hierarchy's
                    // offsets are zero, so its interior lower corner turns a global cell into one
                    let at = |log: &[PhaseSample], s: Option<&PhaseSample>| s.and_then(|s| log.iter().position(|x| std::ptr::eq(x, s))).unwrap_or(0);
                    let d_window = (at(dec, d_prev), at(dec, Some(d)));
                    let m_window = (at(mono, m_prev), at(mono, Some(m)));
                    let whole: [isize; 3] = std::array::from_fn(|a| g[a] + m.interior_lo[a]);
                    let fine_ng = mono.iter().find(|s| s.level == d.level + 1).map_or([0; 3], |s| s.interior_lo);
                    let fine_lo: [isize; 3] = std::array::from_fn(|a| g[a] * RATIO as isize - fine_origin[a] + fine_ng[a]);
                    report += &band_report("tiles", dec_bands, d.level, d_window, whole, fine_lo);
                    report += &band_report("single", mono_bands, d.level, m_window, whole, fine_lo);
                }
                return Some(report);
            }
        }
        previous.insert((d.level, d.tile), d);
        latest.insert((d.level, d.tile), d);
    }
    None
}

#[test]
fn a_balanced_step_on_three_uneven_tiles_equals_the_single_hierarchy_after_every_operation() {
    let partition = Partition::explicit([N; 3], [CUTS.to_vec(), vec![], vec![]]).expect("cuts inside the grid");
    assert!((PATCH[0].1 - 4..=PATCH[0].1).contains(&CUTS[1]), "the cut at {} lies outside the box's level-0 restriction band", CUTS[1]);

    let mut single = mono();
    phase_probe::start();
    single.evolve_steps(STEPS).expect("single hierarchy evolve");
    let mono_log = phase_probe::take();
    let mono_bands = phase_probe::take_bands();
    assert!(mono_log.iter().any(|s| s.level == 0 && s.phase == "restrict"), "the single hierarchy restricted nothing; the bands are never rewritten");

    let mut dec = tiles(&partition);
    let frames = origins(&dec, &partition);
    phase_probe::start();
    let devices = vec![0; dec.len()];
    let t0 = dec[0].levels[0].state.time;
    evolve_hierarchy_decomposed(&mut dec, partition.counts(), &devices, &LocalCopy, Timestepping::Rk2, t0, 1.0e3, 1, |it, _, _| {
        if it >= STEPS { std::ops::ControlFlow::Break(()) } else { std::ops::ControlFlow::Continue(()) }
    });
    let dec_log = phase_probe::take();
    let dec_bands = phase_probe::take_bands();
    for (i, h) in dec.iter().enumerate() {
        assert!(h.crash.is_none(), "tile {i} crashed: {:?}", h.crash);
        assert_eq!(h.levels[0].state.iteration, single.levels[0].state.iteration, "tile {i} stopped short of the single hierarchy");
    }
    let moved = mono_log.iter().filter(|s| s.level == 0).any(|s| s.fields.iter().any(|(n, v)| *n == "pre" && v.iter().any(|p| p.to_bits() != 0.6f64.to_bits())));
    assert!(moved, "the pressure never left its initial value; the comparison is vacuous");
    assert!(mono_bands.iter().any(|b| b.level == 0), "the single hierarchy logged no band rewrite; the operand report would be empty");
    assert!(dec_bands.iter().any(|b| b.level == 0), "the tiles logged no band rewrite; the operand report would be empty");
    if let Some(report) = first_difference((&mono_log, &mono_bands), (&dec_log, &dec_bands), &frames) {
        panic!("three uneven tiles first differ from the single hierarchy {report}");
    }
}
