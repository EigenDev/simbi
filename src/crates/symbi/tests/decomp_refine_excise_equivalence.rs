// =============================================================================
// decomp_refine_excise_equivalence.rs
//
// REFINEMENT x DECOMPOSITION x EXCISION on a kerr-schild chart. the excised core is owned by the
// ROOT level (the refinement request gate forbids a fine patch overlapping it), so a decomposed
// refined run has to excise its root exactly as the monolithic one does.
//
// the failure this exists to catch is silent: the decomposed driver builds its own step, and a root
// tail that runs only the emf part never fires the root excise. un-excised gas then evolves forever
// inside the horizon, and because that region is causally disconnected nothing downstream ever
// complains -- the exterior looks fine while the interior is fiction.
//
// gates:
//   - the decomposed refined excised composite matches the monolithic one to roundoff, for every
//     tile topology (x-cut, y-cut, 2x2 -- the 2x2 cut point is the chart origin, so every tile owns
//     one quadrant of the excised sphere and the excised rim straddles all four cuts);
//   - the excise DEMONSTRABLY fires: the excised core differs from the same run with excision off.
//     without this the equivalence above would pass just as happily if both sides excised nothing.
// =============================================================================
use symbi::regimes::substrate_rhd::RhdSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, unflatten};
use symbi::sim::refinement::{
    Hierarchy, ProlongOrder, RefinementRegion, evolve_hierarchy_decomposed,
};
use symbi::sim::state::*;
use symbi::sim::substrate_seam::WithExcision;
use symbi_algebra::Tensor;
use symbi_geometry::SchwarzschildKSCartesian;
use symbi_hydro::Rhd;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 4.0 / 3.0;
const CFL: f64 = 0.3;
const N: usize = 48;
const L: f64 = 1.2;
const DX: f64 = 2.0 * L / N as f64;
const MASS: f64 = 0.3; // r_+ = 2M = 0.6
const R_EXC: f64 = 0.35; // inside r_+, above the metric guard M/2 = 0.15
const T_FINAL: f64 = 0.08;

type Sim = SimState<Rhd, 2, SchwarzschildKSCartesian<f64>, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = RhdSubstrateKernelSet<HostMemory, f64, 2>;
type Hier =
    Hierarchy<Rhd, 2, 2, SchwarzschildKSCartesian<f64>, IdealGas<f64>, CpuSpace, HostMemory, Kern>;

fn kset(sim: &Sim) -> Kern {
    Kern::new(GAMMA, CFL, &sim.geom.allocated).with_excision(R_EXC, 1.0, 1.0)
}

fn kset_unexcised(sim: &Sim) -> Kern {
    Kern::new(GAMMA, CFL, &sim.geom.allocated)
}

// the fine patch sits in the far field, clear of the excised sphere: the excise pass runs on the
// level that OWNS the excised region, and a fine patch overlapping it would evolve its own copy and
// restrict un-excised values back over the fill. the request gate enforces this separation, so the
// oracle honors it.
fn patch() -> RefinementRegion<2> {
    RefinementRegion {
        x_lo: [0.55 * L, 0.55 * L],
        x_hi: [0.90 * L, 0.90 * L],
    }
}

fn build_root(cells: [usize; 2], origin: [f64; 2], bnd: Boundaries<2>) -> Sim {
    Sim::build(
        Rhd,
        IdealGas { gamma: GAMMA },
        SchwarzschildKSCartesian { mass: MASS },
    )
    .cells(cells)
    .spacing([DX; 2])
    .origin(origin)
    .boundaries(bnd)
    .cfl(CFL)
    .timestepping(Timestepping::Rk2)
    .allocate()
    .expect("root sim construction failed")
    .set_initial(|_| Prim {
        rho: 1.0,
        vel: Tensor::new([0.0; 2]),
        pre: 0.1,
    })
    .build()
}

fn build_mono(mk: fn(&Sim) -> Kern) -> Hier {
    let root = build_root([N, N], [-L, -L], Boundaries::uniform(BoundaryType::Outflow));
    let k = mk(&root);
    let mut h =
        Hier::with_refinement(root, k, &[patch()], ProlongOrder::Plm, mk).expect("mono hierarchy");
    h.seed_fine_from_coarse().expect("seed fine");
    h.prime();
    h
}

fn build_tiles(counts: [usize; 2], mk: fn(&Sim) -> Kern) -> Vec<Hier> {
    let m: [usize; 2] = std::array::from_fn(|a| N / counts[a]);
    let total: usize = counts.iter().product();
    let mut tiles = Vec::with_capacity(total);
    for flat in 0..total {
        let tc = unflatten(flat, counts);
        let origin: [f64; 2] = std::array::from_fn(|a| -L + tc[a] as f64 * m[a] as f64 * DX);
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
        let p = patch();
        let owns_patch =
            (0..2).all(|a| origin[a] <= p.x_lo[a] && origin[a] + m[a] as f64 * DX >= p.x_hi[a]);
        let mut h = if owns_patch {
            let k = mk(&root);
            let h = Hier::with_refinement(root, k, &[p], ProlongOrder::Plm, mk)
                .expect("tile hierarchy");
            h.seed_fine_from_coarse().expect("seed fine");
            h
        } else {
            let k = mk(&root);
            Hier::single(root, k)
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

// the ROOT density over the whole box, in global root-index order. the excised core lives on the
// root, so this is the field the excise pass actually writes.
fn root_density(tiles: &[Hier], counts: [usize; 2]) -> Vec<f64> {
    let m: [usize; 2] = std::array::from_fn(|a| N / counts[a]);
    let mut out = vec![f64::NAN; N * N];
    for (flat, h) in tiles.iter().enumerate() {
        let tc = unflatten(flat, counts);
        let root = &h.levels[0].state;
        let rlo: [isize; 2] = std::array::from_fn(|a| root.geom.interior.spaces[a].lo);
        for c in root.geom.interior.iter() {
            let g: [usize; 2] = std::array::from_fn(|a| tc[a] * m[a] + (c[a] - rlo[a]) as usize);
            out[g[1] * N + g[0]] = *root.fields.cons.den.view().at(c);
        }
    }
    assert!(
        out.iter().all(|v| v.is_finite()),
        "root density has unwritten cells"
    );
    out
}

fn max_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

// the cells strictly inside the excised surface, in global root-index order.
fn excised_mask() -> Vec<bool> {
    let mut m = vec![false; N * N];
    for j in 0..N {
        for i in 0..N {
            let x = -L + (i as f64 + 0.5) * DX;
            let y = -L + (j as f64 + 0.5) * DX;
            m[j * N + i] = (x * x + y * y).sqrt() < R_EXC;
        }
    }
    m
}

fn assert_matches(counts: [usize; 2]) {
    let mut mono = vec![build_mono(kset)];
    run_decomposed(&mut mono, [1, 1]);
    let want = root_density(&mono, [1, 1]);

    let mut tiles = build_tiles(counts, kset);
    run_decomposed(&mut tiles, counts);
    let got = root_density(&tiles, counts);

    let e = max_err(&want, &got);
    assert!(
        e < 1e-12,
        "{counts:?} refined+excised decomposed vs monolithic: err {e:e}"
    );

    // NON-VACUITY: the excise must actually have fired. compare the excised core against the same
    // run with excision OFF -- if the pass never ran, the two are identical and the equivalence
    // above proves nothing about excision.
    let mut plain = vec![build_mono(kset_unexcised)];
    run_decomposed(&mut plain, [1, 1]);
    let bare = root_density(&plain, [1, 1]);
    let mask = excised_mask();
    let ncore = mask.iter().filter(|b| **b).count();
    assert!(
        ncore > 8,
        "the excised sphere covers only {ncore} root cells; setup is too coarse"
    );
    let core_diff = mask
        .iter()
        .enumerate()
        .filter(|(_, inside)| **inside)
        .map(|(k, _)| (got[k] - bare[k]).abs())
        .fold(0.0, f64::max);
    assert!(
        core_diff > 1e-9,
        "the excised core is identical to an UNEXCISED run (max diff {core_diff:e} over {ncore} \
         cells): the root excise never fired in the decomposed refined driver, so this equivalence \
         is vacuous"
    );
}

#[test]
fn refined_excised_decomposed_two_tile_x_cut() {
    assert_matches([2, 1]);
}

#[test]
fn refined_excised_decomposed_two_tile_y_cut() {
    assert_matches([1, 2]);
}

// the 2x2 cut point is EXACTLY the chart origin, so every tile owns one quadrant of the excised
// sphere and the excised rim straddles all four cuts.
#[test]
fn refined_excised_decomposed_quad_tile() {
    assert_matches([2, 2]);
}
