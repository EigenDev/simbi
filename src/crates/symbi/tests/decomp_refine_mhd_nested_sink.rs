// =============================================================================
// decomp_refine_mhd_nested_sink.rs
//
// a magnetized sink under two nested refined levels on host tiles. a draining
// point mass inside the finest box of a 2.5d newtonian MHD state carrying a
// sheared, divergence-free field at plasma beta ~ 1e4, the finest box holding
// the sink's operator support (the accretion sphere and the drain mask to its
// f64 support); the first tile, cut at root cell 14, holds the root and the
// first refined level alone, outside the support the sink would have on that
// level. the body couples to the run's finest level on every tile, so the
// tiles equal the single hierarchy in every conserved component and every cell
// field component of every level, in the accreted mass and in the force
// history, bit for bit, and in every staggered face. a pressure pulse centered
// on the cut drives gas across it on the first refined level. the gate asserts
// its premises: the single hierarchy completes, the sink drains, the first
// refined level spans the cut, gas moves across the cut on that level, the
// field evolves on the finest level, and the same run without a field leaves a
// measurably different gas.
//
// run: cargo test -p symbi --test decomp_refine_mhd_nested_sink
// =============================================================================
use std::collections::HashMap;

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, Partition, unflatten};
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion, evolve_hierarchy_decomposed, seed_decomposed_fine_from_coarse};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, MagneticSpec, SurfaceSpec};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;
type Kern = NewtonianMhdSubstrateKernelSet<HostMemory, f64, 2>;
type Hier = Hierarchy<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kern>;

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const N: usize = 64;
const DX: f64 = 1.0 / N as f64;
const T_FINAL: f64 = 0.05;
/// the field strength relative to the unit gas pressure: beta ~ 1e4 outside the sink. the drain
/// lowers the finest density inside the sink to ~1e-5 under this field and to ~3e-4 without one
const B_SCALE: f64 = 0.01;
/// the root cell the cut between the two tiles lies on
const CUT: usize = 14;

fn regions() -> [RefinementRegion<2>; 2] {
    [
        RefinementRegion { x_lo: [0.0625, 0.0625], x_hi: [0.9375, 0.9375] },
        RefinementRegion { x_lo: [0.3125, 0.3125], x_hi: [0.6875, 0.6875] },
    ]
}

// face-normal components each independent of their own axis: discretely divergence-free
fn bx(scale: f64, y: f64) -> f64 {
    scale * (0.4 + 0.2 * (2.0 * std::f64::consts::PI * y + 0.3).sin())
}

fn by(scale: f64, x: f64) -> f64 {
    scale * (0.2 + 0.2 * (2.0 * std::f64::consts::PI * x + 0.7).cos())
}

fn bump(x: f64, y: f64) -> f64 {
    0.3 * (-(((x - 0.5) / 0.07).powi(2) + ((y - 0.48) / 0.09).powi(2))).exp()
}

/// a pressure pulse centered on the cut, whose waves cross it on the first refined level
fn pulse(x: f64, y: f64) -> f64 {
    0.5 * (-(((x - CUT as f64 * DX) / 0.04).powi(2) + ((y - 0.5) / 0.1).powi(2))).exp()
}

fn kset(sim: &Sim) -> Kern {
    Kern::new(GAMMA, CFL, 1.0, &sim.geom.allocated)
}

/// a root tile placed as production places it, seeding the faces of its own lattice
fn build_root(cells: [usize; 2], offset: [usize; 2], bnd: Boundaries<2>, scale: f64) -> Sim {
    Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 2])
        .origin([0.0; 2])
        .index_offset([offset[0] as isize, offset[1] as isize])
        .boundaries(bnd)
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("root sim construction failed")
        .set_initial(|x| {
            let b = bump(x[0], x[1]);
            MhdPrim::new(
                Prim::adiabatic(Density(1.0 + b), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0 + b + pulse(x[0], x[1]))),
                Tensor::new([bx(scale, x[1]), by(scale, x[0]), 0.1 * scale]),
            )
        })
        .seed_faces(|d, x| {
            let inside = (0..2).all(|a| {
                let (lo, hi) = (offset[a] as f64 * DX, (offset[a] + cells[a]) as f64 * DX);
                let eps = 1e-9 * DX;
                if a == d { x[a] >= lo - eps && x[a] <= hi + eps } else { x[a] > lo && x[a] < hi }
            });
            match (inside, d) {
                (false, _) => 0.0,
                (true, 0) => bx(scale, x[1]),
                (true, _) => by(scale, x[0]),
            }
        })
        .build()
}

fn sink() -> BodyCollection<f64, 2> {
    BodyCollection::new().add(
        Body::black_hole(0, Tensor::new([0.5, 0.5]), Tensor::zeros(), 0.01, 0.08, 0.05, 0.0, 1.0, 0.08)
            .with_surface(SurfaceSpec::Drain)
            .with_magnetic(MagneticSpec::None),
    )
}

fn build_single(scale: f64) -> Hier {
    let root = build_root([N, N], [0, 0], Boundaries::uniform(BoundaryType::Outflow), scale);
    let k = kset(&root);
    let h = Hier::with_refinement(root, k, &regions(), ProlongOrder::Ppm, kset).expect("single hierarchy");
    h.seed_fine_from_coarse().expect("seed fine");
    let mut h = h.with_bodies(sink());
    h.prime();
    h
}

fn build_tiles(partition: &Partition<2>) -> Vec<Hier> {
    let counts = partition.counts();
    let whole = regions();
    let mut tiles = Vec::with_capacity(partition.n_tiles());
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
        let root = build_root(m, offset, bnd, B_SCALE);
        let clipped: Vec<RefinementRegion<2>> = whole
            .iter()
            .map(|r| RefinementRegion { x_lo: std::array::from_fn(|a| r.x_lo[a].max(lo[a])), x_hi: std::array::from_fn(|a| r.x_hi[a].min(hi[a])) })
            .take_while(|r| (0..2).all(|a| r.x_lo[a] < r.x_hi[a]))
            .collect();
        let k = kset(&root);
        let h = if clipped.is_empty() {
            Hier::single(root, k)
        } else {
            Hier::with_refinement_of(root, k, &clipped, &whole[..clipped.len()], ProlongOrder::Ppm, kset).expect("tile hierarchy")
        };
        tiles.push(h);
    }
    seed_decomposed_fine_from_coarse(&tiles, counts, &vec![0; tiles.len()], &LocalCopy).expect("decomposed fine seed");
    tiles
        .into_iter()
        .map(|h| {
            let mut h = h.with_bodies(sink());
            h.prime();
            h
        })
        .collect()
}

type Store = FieldStore<2, 3, HostMemory, f64>;

/// (component, level, centroid on that level's lattice) -> bits, over every tile's interiors
fn interior_state(tiles: &[Hier]) -> HashMap<(&'static str, usize, [i64; 2]), u64> {
    let picks: [(&'static str, fn(&Store, [isize; 2]) -> f64); 8] = [
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
                let at: [i64; 2] = std::array::from_fn(|a| (x[a] * scale * 2.0).round() as i64);
                for (name, pick) in &picks {
                    let previous = out.insert((*name, ll, at), pick(s, [c[0], c[1]]).to_bits());
                    assert!(previous.is_none(), "{name} on level {ll} at {at:?} is held by two tiles");
                }
            }
        }
    }
    out
}

/// (axis, level, face position on twice that level's lattice) -> bits, over every face bounding an
/// interior cell of every tile; a face two tiles hold carries the same bits on both
fn face_state(tiles: &[Hier]) -> HashMap<(usize, usize, [i64; 2]), u64> {
    let mut out = HashMap::new();
    for h in tiles {
        for (ll, level) in h.levels.iter().enumerate() {
            let s = &level.state;
            let mhd = s.fields.mhd.as_ref().unwrap();
            let scale = (N << ll) as f64;
            for c in s.geom.interior.iter() {
                for d in 0..2 {
                    let mut hi = [c[0], c[1]];
                    hi[d] += 1;
                    for f in [[c[0], c[1]], hi] {
                        let x = s.geom.face_coord(f, d);
                        let at: [i64; 2] = std::array::from_fn(|a| (x[a] * scale * 2.0).round() as i64);
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

/// the largest magnitude of `pick` over the finest level's interior
fn finest_max(h: &Hier, pick: impl Fn(&Store, [isize; 2]) -> f64) -> f64 {
    let s = &h.levels.last().unwrap().state;
    s.geom.interior.iter().map(|c| pick(s, [c[0], c[1]]).abs()).fold(0.0, f64::max)
}

fn body_record(h: &Hier) -> (f64, Vec<f64>) {
    let im = h.levels.last().unwrap().state.immersed.as_ref().expect("bodies attached");
    let mut accreted = 0.0;
    im.bodies.visit_accretion(|b| {
        if let symbi_ib::BodyKind::BlackHole { total_accreted_mass, .. } = &b.kind {
            accreted += *total_accreted_mass;
        }
    });
    (accreted, im.history.force().to_vec())
}

#[test]
fn a_tile_short_of_the_finest_level_couples_a_magnetized_nested_sink_as_the_single_hierarchy_does() {
    let partition = Partition::explicit([N, N], [vec![CUT], vec![]]).expect("cuts inside the grid");
    let mut single = build_single(B_SCALE);
    let mut tiles = build_tiles(&partition);
    assert_eq!((tiles[0].levels.len(), tiles[1].levels.len()), (2, 3), "premise: the first tile holds levels 0 and 1 alone");
    single.evolve(T_FINAL).expect("single evolve");
    assert!(single.crash.is_none(), "the single hierarchy crashed: {:?}", single.crash);
    assert!(single.levels[0].state.time >= T_FINAL, "the single hierarchy stopped at {}", single.levels[0].state.time);
    // gas crosses the cut on the first refined level: the momentum normal to the cut in the two
    // level-1 columns beside it. at the final time the pulse leaves ~5e-3 there; the same run
    // without the pulse leaves ~1e-5
    let fine = &single.levels[1].state;
    let cut_fine = 2 * CUT as isize;
    let crossing = fine
        .geom
        .interior
        .iter()
        .filter(|c| c[0] == cut_fine - 1 || c[0] == cut_fine)
        .map(|c| fine.fields.cons.mom[0].view().at(c).abs())
        .fold(0.0, f64::max);
    assert!(crossing > 1.0e-3, "the level-1 momentum across the cut peaks at {crossing:e}; the cut couples nothing");
    let devices = vec![0; tiles.len()];
    evolve_hierarchy_decomposed(&mut tiles, partition.counts(), &devices, &LocalCopy, Timestepping::Rk2, 0.0, T_FINAL, u64::MAX, |_, _, _| std::ops::ControlFlow::Continue(()));
    for (k, tile) in tiles.iter().enumerate() {
        assert!(tile.crash.is_none(), "tile {k} crashed: {:?}", tile.crash);
        assert_eq!(tile.levels[0].state.iteration, single.levels[0].state.iteration, "tile {k} stopped short");
    }
    // the field acts: it evolves on the finest level, and the same run without a field leaves a
    // measurably different gas. measured: the finest x-face field moves by 0.027 of a 0.030 peak,
    // and the finest density without the field differs by up to 96% inside the sink
    let initial = build_single(B_SCALE);
    let field_change = finest_max(&single, |s, c| {
        let b = s.fields.mhd.as_ref().unwrap().bface[0].view();
        let b0 = initial.levels.last().unwrap().state.fields.mhd.as_ref().unwrap().bface[0].view();
        *b.at(c) - *b0.at(c)
    });
    let field = finest_max(&single, |s, c| *s.fields.mhd.as_ref().unwrap().bface[0].view().at(c));
    assert!(field_change > 0.25 * field, "the finest x-face field moved by {field_change:e} out of {field:e}; the field is static");
    let mut unmagnetized = build_single(0.0);
    unmagnetized.evolve(T_FINAL).expect("unmagnetized evolve");
    assert!(unmagnetized.crash.is_none(), "the zero-field control crashed");
    let gas_change = finest_max(&single, |s, c| {
        let other = &unmagnetized.levels.last().unwrap().state;
        (*s.fields.cons.den.view().at(c) - *other.fields.cons.den.view().at(c)) / *other.fields.cons.den.view().at(c)
    });
    assert!(gas_change > 0.25, "the field changes the finest density by at most {gas_change:e} relative; it couples nothing");
    let want = interior_state(std::slice::from_ref(&single));
    let got = interior_state(&tiles);
    assert_eq!(got.len(), want.len(), "the tiles hold a different set of level cells than the single hierarchy");
    let differing: Vec<_> = want.iter().filter(|(k, v)| got.get(k) != Some(v)).map(|(k, _)| k).collect();
    assert!(differing.is_empty(), "{} values differ, first {:?}", differing.len(), differing.first());
    let (want_faces, got_faces) = (face_state(std::slice::from_ref(&single)), face_state(&tiles));
    assert_eq!(got_faces.len(), want_faces.len(), "the tiles hold a different set of faces than the single hierarchy");
    let differing_faces: Vec<_> = want_faces.iter().filter(|(k, v)| got_faces.get(k) != Some(v)).map(|(k, _)| k).collect();
    assert!(differing_faces.is_empty(), "{} face values differ, first {:?}", differing_faces.len(), differing_faces.first());
    let (want_mass, want_force) = body_record(&single);
    assert!(want_mass > 1.0e-4, "the sink accreted {want_mass:e}; the drain is idle and the gate is vacuous");
    for (k, tile) in tiles.iter().enumerate() {
        let (mass, force) = body_record(tile);
        assert_eq!(mass.to_bits(), want_mass.to_bits(), "tile {k}: accreted {mass:e} against {want_mass:e}");
        assert_eq!(force.len(), want_force.len(), "tile {k}: force history length");
        for (ii, (a, b)) in force.iter().zip(&want_force).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "tile {k}: force entry {ii} is {a:e} against {b:e}");
        }
    }
}
