// =============================================================================
// decomp_refine_mhd_sink_equivalence.rs
//
// a magnetized sink x refinement x decomposition: per-tile 2-level hierarchies carrying the same
// draining point mass at its global position, driven through `evolve_hierarchy_decomposed`,
// reproduce the monolithic refined run to round-off in the composite density, the composite
// stored primitive density, and the cell field. the finest level owns the drain, which relaxes
// each cell at its local Alfven rate and is followed by the primitive recovery in the level step
// tail; the fine cut halos are refreshed before the next fine substep. two placements: the sink
// inside one tile's fine patch, and the sink straddling a root cut with the patch spanning both
// tiles.
// =============================================================================

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, unflatten};
use symbi::sim::refinement::{
    Hierarchy, ProlongOrder, RefinementRegion, evolve_hierarchy_decomposed,
};
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

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const N: usize = 96;
const DX: f64 = 3.0 / N as f64;
const T_FINAL: f64 = 0.04;
const B0: [f64; 3] = [0.5, 0.25, 0.2];

type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;
type Kern = NewtonianMhdSubstrateKernelSet<HostMemory, f64, 2>;
type Hier = Hierarchy<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kern>;

fn kset(sim: &Sim) -> Kern {
    Kern::new(GAMMA, CFL, 1.0, &sim.geom.allocated)
}

// a transparent draining point mass at `pos`; a fresh collection per attach.
fn sink_at(pos: [f64; 2]) -> BodyCollection<f64, 2> {
    BodyCollection::new().add(
        Body::black_hole(
            0,
            Tensor::new(pos),
            Tensor::zeros(),
            1.0,
            0.08,
            0.05,
            1.0,
            1.0,
            0.08,
        )
        .with_surface(SurfaceSpec::Drain)
        .with_magnetic(MagneticSpec::None),
    )
}

fn build_root(cells: [usize; 2], origin: [f64; 2], bnd: Boundaries<2>) -> Sim {
    Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 2])
        .origin(origin)
        .boundaries(bnd)
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("root sim construction failed")
        .set_initial(|_| {
            MhdPrim::new(
                Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)),
                Tensor::new(B0),
            )
        })
        .seed_faces_uniform([B0[0], B0[1]])
        .build()
}

// clip a region to a tile's physical slab; None when the overlap is empty or degenerate.
fn clip(
    region: &RefinementRegion<2>,
    origin: [f64; 2],
    m: [usize; 2],
) -> Option<RefinementRegion<2>> {
    let mut lo = [0.0; 2];
    let mut hi = [0.0; 2];
    for a in 0..2 {
        let tlo = origin[a];
        let thi = origin[a] + m[a] as f64 * DX;
        lo[a] = region.x_lo[a].max(tlo);
        hi[a] = region.x_hi[a].min(thi);
        if hi[a] - lo[a] < DX {
            return None;
        }
    }
    Some(RefinementRegion { x_lo: lo, x_hi: hi })
}

fn build_mono(region: &RefinementRegion<2>, sink_pos: Option<[f64; 2]>) -> Hier {
    let root = build_root(
        [N, N],
        [0.0, 0.0],
        Boundaries::uniform(BoundaryType::Outflow),
    );
    let k = kset(&root);
    let h = Hier::with_refinement(
        root,
        k,
        std::slice::from_ref(region),
        ProlongOrder::Ppm,
        kset,
    )
    .expect("mono hierarchy");
    h.seed_fine_from_coarse().expect("seed fine");
    let mut h = match sink_pos {
        Some(pos) => h.with_bodies(sink_at(pos)),
        None => h,
    };
    h.prime();
    h
}

fn build_tiles(
    counts: [usize; 2],
    region: &RefinementRegion<2>,
    sink_pos: Option<[f64; 2]>,
) -> Vec<Hier> {
    let m: [usize; 2] = std::array::from_fn(|a| N / counts[a]);
    let total: usize = counts.iter().product();
    let mut tiles = Vec::with_capacity(total);
    for flat in 0..total {
        let tc = unflatten(flat, counts);
        let origin = std::array::from_fn(|a| tc[a] as f64 * m[a] as f64 * DX);
        let bnd = Boundaries(std::array::from_fn(|a| {
            let lo = if tc[a] == 0 {
                BoundaryType::Outflow
            } else {
                BoundaryType::CoarseFine
            };
            let hi = if tc[a] == counts[a] - 1 {
                BoundaryType::Outflow
            } else {
                BoundaryType::CoarseFine
            };
            [lo, hi]
        }));
        let root = build_root(m, origin, bnd);
        let h = match clip(region, origin, m) {
            Some(r) => {
                let k = kset(&root);
                let h = Hier::with_refinement(root, k, &[r], ProlongOrder::Ppm, kset)
                    .expect("tile hierarchy");
                h.seed_fine_from_coarse().expect("seed fine");
                h
            }
            None => {
                let k = kset(&root);
                Hier::single(root, k)
            }
        };
        let mut h = match sink_pos {
            Some(pos) => h.with_bodies(sink_at(pos)),
            None => h,
        };
        h.prime();
        tiles.push(h);
    }
    tiles
}

fn run_decomposed(tiles: &mut [Hier], counts: [usize; 2]) {
    let devices: Vec<i32> = vec![0; tiles.len()];
    evolve_hierarchy_decomposed(
        tiles,
        counts,
        &devices,
        &LocalCopy,
        Timestepping::Rk2,
        0.0,
        T_FINAL,
        u64::MAX,
        |_, _, _| std::ops::ControlFlow::Continue(()),
    );
}

// the composite fine-resolution picture of a cell field: uncovered root cells replicated 2x2,
// fine interiors at their global fine index.
fn composite(
    tiles: &[Hier],
    counts: [usize; 2],
    pick: &dyn Fn(&Store, [isize; 2]) -> f64,
) -> Vec<f64> {
    let fn_n = 2 * N;
    let mut out = vec![f64::NAN; fn_n * fn_n];
    let m: [usize; 2] = std::array::from_fn(|a| N / counts[a]);
    for (flat, h) in tiles.iter().enumerate() {
        let tc = unflatten(flat, counts);
        let root = &h.levels[0].state;
        let cov = h.levels[0].coverage.as_ref();
        let rlo: [isize; 2] = std::array::from_fn(|a| root.geom.interior.spaces[a].lo);
        for c in root.geom.interior.iter() {
            if let Some(cov) = cov {
                if cov.contains(c) {
                    continue;
                }
            }
            let g: [usize; 2] = std::array::from_fn(|a| tc[a] * m[a] + (c[a] - rlo[a]) as usize);
            let d = pick(root, c);
            for sy in 0..2 {
                for sx in 0..2 {
                    out[(2 * g[1] + sy) * fn_n + (2 * g[0] + sx)] = d;
                }
            }
        }
        if h.levels.len() > 1 {
            let fine = &h.levels[1].state;
            for c in fine.geom.interior.iter() {
                let fx = 2 * tc[0] * m[0] + c[0] as usize;
                let fy = 2 * tc[1] * m[1] + c[1] as usize;
                out[fy * fn_n + fx] = pick(fine, c);
            }
        }
    }
    out
}

fn accreted(h: &Hier) -> f64 {
    let finest = &h.levels[h.levels.len() - 1].state;
    let Some(im) = finest.immersed.as_ref() else {
        return 0.0;
    };
    let mut total = 0.0;
    im.bodies.visit_accretion(|b| {
        if let symbi_ib::BodyKind::BlackHole {
            total_accreted_mass,
            ..
        } = &b.kind
        {
            total += *total_accreted_mass;
        }
    });
    total
}

type Store = FieldStore<2, 3, HostMemory, f64>;

fn assert_matches(counts: [usize; 2], region: RefinementRegion<2>, sink_pos: Option<[f64; 2]>) {
    let mut mono = build_mono(&region, sink_pos);
    mono.evolve(T_FINAL).expect("mono evolve");
    let mono_acc = accreted(&mono);
    if sink_pos.is_some() {
        assert!(
            mono_acc > 1e-6,
            "the sink accreted nothing ({mono_acc:e}); the equivalence is vacuous"
        );
    }
    assert!(
        mono.levels[0].state.iteration >= 2,
        "the run took one root step; the cut halo state carried across steps goes untested"
    );

    let mut tiles = build_tiles(counts, &region, sink_pos);
    run_decomposed(&mut tiles, counts);
    // every tile carries the global accreted-mass tally; the tally sums the tiles' partials, so
    // it agrees with the monolithic reduction to the summation-order roundoff.
    for (i, h) in tiles.iter().enumerate() {
        let a = accreted(h);
        assert!(
            (a - mono_acc).abs() <= 1e-14 * mono_acc.max(1.0),
            "{counts:?} tile {i}: accreted mass {a:e} != mono {mono_acc:e}"
        );
    }

    // the shared cut face of the fine level: both tiles evolve it from the same inputs, so their
    // copies agree bitwise whenever the fine patch spans the cut.
    if tiles.len() == 2 && tiles.iter().all(|h| h.levels.len() > 1) {
        let l0 = &tiles[0].levels[1].state;
        let l1 = &tiles[1].levels[1].state;
        let f0 = &l0.fields.mhd.as_ref().unwrap().bface[0];
        let f1 = &l1.fields.mhd.as_ref().unwrap().bface[0];
        let x0 = l0.geom.interior.spaces[0].hi;
        let x1 = l1.geom.interior.spaces[0].lo;
        for y in l0.geom.interior.spaces[1].lo..l0.geom.interior.spaces[1].hi {
            let a = *f0.view().at([x0, y]);
            let b = *f1.view().at([x1, y]);
            assert!(
                a.to_bits() == b.to_bits(),
                "{counts:?}: the tiles' copies of the shared fine cut face differ at row {y}: {a:e} vs {b:e}"
            );
        }
    }

    let picks: Vec<(&str, Box<dyn Fn(&Store, [isize; 2]) -> f64>)> = vec![
        ("den", Box::new(|s, c| *s.fields.cons.den.view().at(c))),
        ("momx", Box::new(|s, c| *s.fields.cons.mom[0].view().at(c))),
        ("momy", Box::new(|s, c| *s.fields.cons.mom[1].view().at(c))),
        ("momz", Box::new(|s, c| *s.fields.cons.mom[2].view().at(c))),
        (
            "nrg",
            Box::new(|s, c| *s.fields.cons.nrg_field().unwrap().view().at(c)),
        ),
        (
            "bcell0",
            Box::new(|s, c| *s.fields.mhd.as_ref().unwrap().bcell[0].view().at(c)),
        ),
        (
            "bcell1",
            Box::new(|s, c| *s.fields.mhd.as_ref().unwrap().bcell[1].view().at(c)),
        ),
        (
            "bcell2",
            Box::new(|s, c| *s.fields.mhd.as_ref().unwrap().bcell[2].view().at(c)),
        ),
        ("prim rho", Box::new(|s, c| *s.fields.prim.rho.view().at(c))),
        (
            "prim pre",
            Box::new(|s, c| *s.fields.prim.pre_field().unwrap().view().at(c)),
        ),
    ];
    for (name, pick) in &picks {
        let mv = composite(std::slice::from_ref(&mono), [1, 1], pick);
        let dv = composite(&tiles, counts, pick);
        assert!(
            mv.iter().all(|v| v.is_finite()) && dv.iter().all(|v| v.is_finite()),
            "unwritten composite cells"
        );
        let worst = mv
            .iter()
            .zip(&dv)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            mv.iter().zip(&dv).all(|(a, b)| a.to_bits() == b.to_bits()),
            "{counts:?} {name}: refined+decomposed magnetized sink differs from mono, worst {worst:e}"
        );
    }
}

#[test]
fn magnetized_gas_without_a_body_matches() {
    assert_matches(
        [2, 1],
        RefinementRegion {
            x_lo: [1.03, 1.03],
            x_hi: [1.97, 1.97],
        },
        None,
    );
}

#[test]
fn magnetized_sink_inside_one_tiles_patch() {
    // the sink support (accretion sphere, mask tail and stencil reach) spans 0.44, so the fine
    // patch reaches 0.47 from the sink on every side, stays inside the left tile, and keeps the
    // prolongation stencil inside the domain.
    assert_matches(
        [2, 1],
        RefinementRegion {
            x_lo: [0.28, 0.28],
            x_hi: [1.22, 1.22],
        },
        Some([0.75, 0.75]),
    );
}

#[test]
fn magnetized_sink_straddling_a_cut_with_the_patch_spanning_tiles() {
    assert_matches(
        [2, 1],
        RefinementRegion {
            x_lo: [1.03, 1.03],
            x_hi: [1.97, 1.97],
        },
        Some([1.5, 1.5]),
    );
}
