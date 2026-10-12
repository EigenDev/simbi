// =============================================================================
// decomp_refine_gr_mhd_equivalence.rs
//
// a refined magnetized run on a curved spacetime across host tiles. a swirling
// gas under a field seeded free of densitized divergence on a cartesian
// kerr-schild box, one refined box crossing the cuts with its faces clear of
// them: every tile carries the densitized flux sqrt(gamma) B through its face
// transfers with the sqrt(gamma) of the same global face coordinates the
// single hierarchy evaluates, so two tiles and four uneven tiles equal the
// single hierarchy in every conserved and cell field value and every staggered
// face of both levels, bit for bit, and every tile holds the densitized
// divergence at machine zero. the gate asserts its premises: the field moves
// on the refined level, and the refined box spans the cuts.
//
// run: cargo test -p symbi --test decomp_refine_gr_mhd_equivalence
// =============================================================================
use std::collections::HashMap;

use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet3D;
use symbi::sim::decomp::{LocalCopy, Partition, unflatten};
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion, evolve_hierarchy_decomposed, seed_decomposed_fine_from_coarse};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{KerrKSCartesian, Metric};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Rmhd, 3, KerrKSCartesian<f64>, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = RmhdSubstrateKernelSet3D<HostMemory, f64>;
type Hier = Hierarchy<Rmhd, 3, 3, KerrKSCartesian<f64>, IdealGas<f64>, CpuSpace, HostMemory, Kern>;

const N: usize = 12;
const DX: f64 = 1.0 / N as f64;
const GAMMA: f64 = 4.0 / 3.0;
const CFL: f64 = 0.3;
const MASS: f64 = 0.2;
const SPIN: f64 = 0.5;
const B0: f64 = 0.1;
// the box [1.2, 2.2]^3 lies outside the horizon: every cell sits at r >= sqrt(3) * 1.2
const X_LO: f64 = 1.2;
const DIVB_TOL: f64 = 1e-12;
const STEPS: u64 = 3;
const METRIC: KerrKSCartesian<f64> = KerrKSCartesian { mass: MASS, spin: SPIN };

fn kernels(sim: &Sim) -> Kern {
    Kern::new(GAMMA, CFL, 1.0, &sim.geom.allocated)
}

/// the refined box over root cells 3..9 on every axis
fn region() -> RefinementRegion<3> {
    RefinementRegion { x_lo: [X_LO + 3.0 * DX; 3], x_hi: [X_LO + 9.0 * DX; 3] }
}

fn swirl([x, y, z]: [f64; 3]) -> MhdPrim<f64, 3> {
    let s = 2.0 * std::f64::consts::PI;
    MhdPrim::new(
        Prim::adiabatic(Density(1.0), Tensor::new([0.1 * (s * y).sin(), 0.1 * (s * z).sin(), 0.1 * (s * x).sin()]), Pressure(0.5)),
        Tensor::new([0.0; 3]),
    )
}

/// a root tile placed as production places it, B^x = B0 / sqrt(gamma) on the faces of its own lattice
fn root(cells: [usize; 3], offset: [usize; 3], bnd: Boundaries<3>) -> Sim {
    Sim::build(Rmhd, IdealGas { gamma: GAMMA }, METRIC)
        .cells(cells)
        .origin([X_LO; 3])
        .spacing([DX; 3])
        .index_offset(std::array::from_fn(|a| offset[a] as isize))
        .boundaries(bnd)
        .cfl(CFL)
        .allocate()
        .expect("root")
        .set_initial(swirl)
        .seed_faces(|d, x| {
            let inside = (0..3).all(|a| {
                let (lo, hi) = (X_LO + offset[a] as f64 * DX, X_LO + (offset[a] + cells[a]) as f64 * DX);
                let eps = 1e-9 * DX;
                if a == d { x[a] >= lo - eps && x[a] <= hi + eps } else { x[a] > lo && x[a] < hi }
            });
            if inside && d == 0 { B0 / METRIC.sqrt_det_gamma(Tensor::new(x)) } else { 0.0 }
        })
        .build()
}

fn single() -> Hier {
    let r = root([N; 3], [0; 3], Boundaries::uniform(BoundaryType::Outflow));
    let k = kernels(&r);
    let mut h = Hier::with_refinement(r, k, &[region()], ProlongOrder::Ppm, kernels).expect("single hierarchy");
    h.seed_fine_from_coarse().expect("seed fine");
    h.prime();
    h
}

fn tiles(partition: &Partition<3>) -> Vec<Hier> {
    let counts = partition.counts();
    let whole = region();
    let mut tiles = Vec::new();
    for flat in 0..partition.n_tiles() {
        let tc = unflatten(flat, counts);
        let ext = partition.tile_extents(tc);
        let m = [ext[0].1, ext[1].1, ext[2].1];
        let off = [ext[0].0, ext[1].0, ext[2].0];
        let bnd = Boundaries(std::array::from_fn(|a| {
            let face = |edge: bool| if edge { BoundaryType::Outflow } else { BoundaryType::CoarseFine };
            [face(tc[a] == 0), face(tc[a] == counts[a] - 1)]
        }));
        let r = root(m, off, bnd);
        let lo: [f64; 3] = std::array::from_fn(|a| X_LO + off[a] as f64 * DX);
        let hi: [f64; 3] = std::array::from_fn(|a| X_LO + (off[a] + m[a]) as f64 * DX);
        let clipped = RefinementRegion {
            x_lo: std::array::from_fn(|a| whole.x_lo[a].max(lo[a])),
            x_hi: std::array::from_fn(|a| whole.x_hi[a].min(hi[a])),
        };
        let k = kernels(&r);
        let h = if (0..3).all(|a| clipped.x_lo[a] < clipped.x_hi[a]) {
            Hier::with_refinement_of(r, k, &[clipped], &[whole], ProlongOrder::Ppm, kernels).expect("tile hierarchy")
        } else {
            Hier::single(r, k)
        };
        tiles.push(h);
    }
    seed_decomposed_fine_from_coarse(&tiles, counts, &vec![0; tiles.len()], &LocalCopy).expect("decomposed seed");
    for h in tiles.iter_mut() {
        h.prime();
    }
    tiles
}

type Store = FieldStore<3, 3, HostMemory, f64>;

/// (component, level, centroid on twice the level's lattice) -> bits, over every tile's interiors
fn cell_state(tiles: &[Hier]) -> HashMap<(&'static str, usize, [i64; 3]), u64> {
    let picks: [(&'static str, fn(&Store, [isize; 3]) -> f64); 8] = [
        ("den", |s, c| *s.fields.cons.den.view().at(c)),
        ("mom0", |s, c| *s.fields.cons.mom[0].view().at(c)),
        ("mom1", |s, c| *s.fields.cons.mom[1].view().at(c)),
        ("mom2", |s, c| *s.fields.cons.mom[2].view().at(c)),
        ("nrg", |s, c| *s.fields.cons.nrg_field().unwrap().view().at(c)),
        ("bcell0", |s, c| *s.fields.mhd.as_ref().unwrap().bcell[0].view().at(c)),
        ("bcell1", |s, c| *s.fields.mhd.as_ref().unwrap().bcell[1].view().at(c)),
        ("bcell2", |s, c| *s.fields.mhd.as_ref().unwrap().bcell[2].view().at(c)),
    ];
    let mut out = HashMap::new();
    for h in tiles {
        for (ll, level) in h.levels.iter().enumerate() {
            let s = &level.state;
            let scale = (N << ll) as f64;
            for c in s.geom.interior.iter() {
                let x = s.geom.centroid(c);
                let at: [i64; 3] = std::array::from_fn(|a| ((x[a] - X_LO) * scale * 2.0).round() as i64);
                for (name, pick) in &picks {
                    let previous = out.insert((*name, ll, at), pick(s, c).to_bits());
                    assert!(previous.is_none(), "{name} on level {ll} at {at:?} is held by two tiles");
                }
            }
        }
    }
    out
}

/// (axis, level, face position on twice the level's lattice) -> bits, over every face bounding an
/// interior cell; a face two tiles hold carries the same bits on both
fn face_state(tiles: &[Hier]) -> HashMap<(usize, usize, [i64; 3]), u64> {
    let mut out = HashMap::new();
    for h in tiles {
        for (ll, level) in h.levels.iter().enumerate() {
            let s = &level.state;
            let mhd = s.fields.mhd.as_ref().unwrap();
            let scale = (N << ll) as f64;
            for c in s.geom.interior.iter() {
                for d in 0..3 {
                    let mut hi = c;
                    hi[d] += 1;
                    for f in [c, hi] {
                        let x = s.geom.face_coord(f, d);
                        let at: [i64; 3] = std::array::from_fn(|a| ((x[a] - X_LO) * scale * 2.0).round() as i64);
                        let bits = mhd.bface[d].view().at(f).to_bits();
                        if let Some(previous) = out.insert((d, ll, at), bits) {
                            assert_eq!(previous, bits, "face {d} on level {ll} at {at:?} differs between the tiles holding it");
                        }
                    }
                }
            }
        }
    }
    out
}

/// the largest densitized divergence over a level's interior, sqrt(gamma) at each face's center
fn densitized_divergence(sim: &Sim) -> f64 {
    let mhd = sim.fields.mhd.as_ref().unwrap();
    sim.geom
        .interior
        .iter()
        .map(|c| {
            (0..3)
                .map(|a| {
                    let mut n = c;
                    n[a] += 1;
                    let flux = |f: [isize; 3]| METRIC.sqrt_det_gamma(Tensor::new(sim.geom.face_coord(f, a))) * *mhd.bface[a].view().at(f);
                    (flux(n) - flux(c)) / sim.geom.dx[a]
                })
                .sum::<f64>()
                .abs()
        })
        .fold(0.0, f64::max)
}

fn assert_tiles_match(cuts: [Vec<usize>; 3], want: &Hier) {
    let partition = Partition::explicit([N; 3], cuts.clone()).expect("cuts inside the grid");
    let mut t = tiles(&partition);
    let refined = t.iter().filter(|h| h.levels.len() > 1).count();
    assert!(refined >= 2, "cuts {cuts:?}: the refined box lies on {refined} tile(s); no cut crosses it");
    let devices = vec![0; t.len()];
    evolve_hierarchy_decomposed(&mut t, partition.counts(), &devices, &LocalCopy, Timestepping::Rk2, 0.0, 10.0, 1, |it, _, _| {
        if it >= STEPS { std::ops::ControlFlow::Break(()) } else { std::ops::ControlFlow::Continue(()) }
    });
    for (i, h) in t.iter().enumerate() {
        assert!(h.crash.is_none(), "cuts {cuts:?}: tile {i} crashed: {:?}", h.crash);
        assert_eq!(h.levels[0].state.iteration, want.levels[0].state.iteration, "cuts {cuts:?}: tile {i} stopped short");
        for (ll, level) in h.levels.iter().enumerate() {
            let div = densitized_divergence(&level.state);
            assert!(div < DIVB_TOL, "cuts {cuts:?}: tile {i} level {ll} holds a densitized divergence of {div:e}");
        }
    }
    let (want_cells, got_cells) = (cell_state(std::slice::from_ref(want)), cell_state(&t));
    assert_eq!(got_cells.len(), want_cells.len(), "cuts {cuts:?}: the tiles hold a different set of cells");
    let mut differing: Vec<_> = want_cells.iter().filter(|(k, v)| got_cells.get(k) != Some(v)).map(|(k, _)| *k).collect();
    differing.sort();
    assert!(differing.is_empty(), "cuts {cuts:?}: {} values differ, first {:?}", differing.len(), differing.first());
    let (want_faces, got_faces) = (face_state(std::slice::from_ref(want)), face_state(&t));
    assert_eq!(got_faces.len(), want_faces.len(), "cuts {cuts:?}: the tiles hold a different set of faces");
    let mut differing: Vec<_> = want_faces.iter().filter(|(k, v)| got_faces.get(k) != Some(v)).map(|(k, _)| *k).collect();
    differing.sort();
    assert!(differing.is_empty(), "cuts {cuts:?}: {} face values differ, first {:?}", differing.len(), differing.first());
}

#[test]
fn refined_kerr_schild_mhd_tiles_match_the_single_hierarchy() {
    let initial = single();
    let mut want = single();
    want.evolve_steps(STEPS).expect("single evolve");
    assert!(want.crash.is_none(), "the single hierarchy crashed: {:?}", want.crash);
    // premise: the field moves on the refined level
    let fine = |h: &Hier| -> Vec<f64> {
        let m = h.levels[1].state.fields.mhd.as_ref().unwrap();
        m.bface[1].domain().clone().iter().map(|f| *m.bface[1].view().at(f)).collect()
    };
    let moved = fine(&want).iter().zip(&fine(&initial)).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(moved > 1e-6, "the refined y-face field moved by {moved:e}; the swirl transports nothing");
    for (ll, level) in want.levels.iter().enumerate() {
        let div = densitized_divergence(&level.state);
        assert!(div < DIVB_TOL, "the single hierarchy's level {ll} holds a densitized divergence of {div:e}");
    }
    // the box spans root cells 3..9; every patch face lies two or more cells from a cut
    for cuts in [[vec![6], vec![], vec![]], [vec![6], vec![5], vec![]]] {
        assert_tiles_match(cuts, &want);
    }
}
