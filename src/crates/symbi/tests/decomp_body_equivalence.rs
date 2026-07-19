// =============================================================================
// decomp_body_equivalence.rs
//
// the IMMERSED-BODY correctness contract for multi-gpu domain decomposition: a domain split into
// tiles, each carrying the SAME body (global position), evolved through `evolve_decomposed`, must
// reproduce the monolithic run to round-off for the FLUID (den/mom/nrg).
//
// the body is a fixed accreting BLACK HOLE at the domain center, placed exactly on the 2x2 tile
// CORNER -- the worst case: its gravity AND its mass-removing sink straddle all four tiles. for a
// cartesian grid the body source is FUSED into `godunov_stage` (gravity on mom/nrg + the sink on
// den), so `evolve_decomposed` needs NO change: each tile's godunov self-applies the body to its
// own cells from the body's GLOBAL position. body motion is prescribed (here: fixed) and the
// backward force/accreted-mass feedback is DIAGNOSTICS-only (it never re-enters the fluid or the
// body's gravitating mass), so the per-tile fluid is correct with no cross-tile reduction. the
// accreted-mass DIAGNOSTIC sum across tiles is a separate concern from the fluid correctness this test checks.
//
// cpu-only + 2d: same exchange index math as the gpu path; the fast iteration loop. a tile that
// applied the body at a wrong (local) coordinate, or that failed to remove sink mass at the cut,
// would diverge here at the corner cells.
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{evolve_decomposed, flatten, gather_interiors, unflatten, LocalCopy};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_ib::{BinaryParams, Body, BodyCollection, BodyKind, ReferenceFrame};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const N: usize = 32; // cells per axis
const L: f64 = 1.0; // domain [-L, L]^2, body at the origin (the 2x2 corner)
const DX: f64 = 2.0 * L / N as f64;
const T_FINAL: f64 = 0.03;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

// a fixed accreting black hole at the origin: gravity (mom/nrg) + a mass-removing sink (den). the
// sink + accretion radii are generous so the removal straddles the tile corner. a FRESH collection
// per tile (with_bodies takes ownership), all at the same GLOBAL position.
fn central_bh() -> BodyCollection<f64, 2> {
    BodyCollection::new().add(Body::black_hole(
        0,
        Tensor::new([0.0, 0.0]),
        Tensor::zeros(),
        1.0, // gravitating mass (held fixed)
        0.1, // radius
        0.2, // softening
        10.0, // sink_rate
        0.5, // sink_delta
        0.5, // accretion_radius
    ))
}

// a MOVING binary: two gravitating bodies orbiting the origin (prescribed Keplerian motion via
// advance_binary). this exercises the decomposed body bookkeeping's prescribed-orbit advance: each
// tile must advance the orbit identically so all tiles' body positions stay in lockstep and the
// rotating gravity the fluid feels is consistent across cuts.
fn binary() -> BodyCollection<f64, 2> {
    BodyCollection::new()
        .add(Body::gravitational(0, Tensor::new([0.25, 0.0]), Tensor::new([0.0, 0.3]), 0.5, 0.1, 0.2))
        .add(Body::gravitational(1, Tensor::new([-0.25, 0.0]), Tensor::new([0.0, -0.3]), 0.5, 0.1, 0.2))
        .as_binary()
        .with_frame(ReferenceFrame::Inertial)
        .with_binary_params(BinaryParams::new(1.0, 0.5, 0.0, 1.0))
}

// dense fluid at rest: no pressure gradient, so the ONLY dynamics are the body gravity (+ the BH
// sink). outflow boundaries + short time keep the flow interior so mono == decomposed is exact.
// `bodies` builds a FRESH collection per tile (with_bodies takes ownership); all tiles get the same
// body state at the same GLOBAL position.
fn make(
    cells: [usize; 2],
    origin: [f64; 2],
    bnd: Boundaries<2>,
    ts: Timestepping,
    bodies: fn() -> BodyCollection<f64, 2>,
) -> (Sim, Kern) {
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 2])
        .origin(origin)
        .boundaries(bnd)
        .cfl(CFL)
        .timestepping(ts)
        .allocate()
        .expect("sim construction failed")
        .set_initial(|_| Prim { rho: 2.0, vel: Tensor::new([0.0, 0.0]), pre: 1.0 })
        .build()
        .with_bodies(bodies());
    let k = Kern::new(GAMMA, CFL, &sim.geom.allocated);
    (sim, k)
}

// build the tile grid: each tile an equal slice with its own global origin (so the body's global
// position maps to the right local cells), CoarseFine on internal faces, Outflow on the boundary.
fn grid_tiles(counts: [usize; 2], ts: Timestepping, bodies: fn() -> BodyCollection<f64, 2>) -> Vec<(Sim, Kern)> {
    let m: [usize; 2] = std::array::from_fn(|a| {
        assert!(N % counts[a] == 0, "N must split evenly into counts[{a}]");
        N / counts[a]
    });
    let total: usize = counts.iter().product();
    (0..total)
        .map(|flat| {
            let tc = unflatten(flat, counts);
            let origin = std::array::from_fn(|a| -L + tc[a] as f64 * m[a] as f64 * DX);
            let bnd = Boundaries(std::array::from_fn(|a| {
                let lo = if tc[a] == 0 { BoundaryType::Outflow } else { BoundaryType::CoarseFine };
                let hi = if tc[a] == counts[a] - 1 { BoundaryType::Outflow } else { BoundaryType::CoarseFine };
                [lo, hi]
            }));
            make(m, origin, bnd, ts, bodies)
        })
        .collect()
}

fn run(tiles: &mut [(Sim, Kern)], counts: [usize; 2], ts: Timestepping) {
    let devices: Vec<i32> = vec![0; tiles.len()];
    // evolve_decomposed takes the tiles by &mut (per-step body bookkeeping); build the &mut store
    // handles + & kernels from the same tiles.
    let mut stores = Vec::new();
    let mut kernels = Vec::new();
    for (s, k) in tiles.iter_mut() {
        stores.push(&mut **s);
        kernels.push(&*k);
    }
    evolve_decomposed(
        &mut stores,
        &kernels,
        counts,
        &devices,
        ts,
        0.0,
        T_FINAL,
        u64::MAX,
        &LocalCopy,
        |_, _, _| std::ops::ControlFlow::Continue(()),
    );
}

fn global_field(tiles: &[(Sim, Kern)], counts: [usize; 2], pick: impl Fn(&Sim, [isize; 2]) -> f64) -> Vec<f64> {
    let m: [usize; 2] = std::array::from_fn(|a| N / counts[a]);
    let mut out = vec![f64::NAN; N * N];
    for (flat_tile, (sim, _)) in tiles.iter().enumerate() {
        let tc = unflatten(flat_tile, counts);
        let ilo: [isize; 2] = std::array::from_fn(|a| sim.geom.interior.spaces[a].lo);
        for c in sim.geom.interior.iter() {
            let g: [usize; 2] = std::array::from_fn(|a| tc[a] * m[a] + (c[a] - ilo[a]) as usize);
            out[flatten(g, [N; 2])] = pick(sim, c);
        }
    }
    out
}

fn max_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max)
}

// read a body's (position, total_accreted_mass). gravitational bodies report 0 accretion.
fn body_state(sim: &Sim, idx: usize) -> ([f64; 2], f64) {
    let b = sim.immersed.as_ref().unwrap().bodies.get(idx);
    let acc = match b.kind {
        BodyKind::BlackHole { total_accreted_mass, .. } => total_accreted_mass,
        _ => 0.0,
    };
    ([b.position[0], b.position[1]], acc)
}

// `expect_accretion`: BH runs must remove sink mass + record accretion (non-vacuous); binary runs
// (gravitational, no sink) instead require the bodies to have actually MOVED.
fn assert_body_matches(
    counts: [usize; 2],
    ts: Timestepping,
    bodies: fn() -> BodyCollection<f64, 2>,
    expect_accretion: bool,
) {
    let den = |s: &Sim, c| *s.fields.cons.den.view().at(c);
    let momx = |s: &Sim, c| *s.fields.cons.mom[0].view().at(c);
    let momy = |s: &Sim, c| *s.fields.cons.mom[1].view().at(c);
    let nrg = |s: &Sim, c| *s.fields.cons.nrg_field().unwrap().view().at(c);

    let mut mono = grid_tiles([1, 1], ts, bodies);
    run(&mut mono, [1, 1], ts);
    let mono_den = global_field(&mono, [1, 1], den);

    let mut dec = grid_tiles(counts, ts, bodies);
    run(&mut dec, counts, ts);
    let dec_den = global_field(&dec, counts, den);

    assert!(
        mono_den.iter().all(|v| v.is_finite()) && dec_den.iter().all(|v| v.is_finite()),
        "some global cells were never written"
    );

    // FLUID equivalence: decomposed == monolithic to round-off (den/mom/nrg).
    for (name, pick) in [
        ("den", &den as &dyn Fn(&Sim, [isize; 2]) -> f64),
        ("momx", &momx),
        ("momy", &momy),
        ("nrg", &nrg),
    ] {
        let mv = global_field(&mono, [1, 1], &pick);
        let dv = global_field(&dec, counts, &pick);
        let e = max_err(&mv, &dv);
        assert!(e < 1e-12, "{counts:?} {ts:?} {name} err {e:e} under immersed body");
    }

    // BODY-STATE equivalence: every decomposed tile's body must match the monolithic body's
    // position AND total_accreted_mass. accreted-mass tests the cross-tile DIAGNOSTIC SUM
    // (step_bodies_decomposed); position tests the prescribed-orbit advance staying in lockstep.
    let nbodies = mono[0].0.immersed.as_ref().unwrap().bodies.len();
    let mut moved = 0.0_f64;
    let mut accreted = 0.0_f64;
    for bi in 0..nbodies {
        let (mono_pos, mono_acc) = body_state(&mono[0].0, bi);
        accreted = accreted.max(mono_acc);
        for (s, _) in dec.iter() {
            let (p, a) = body_state(s, bi);
            let dp = ((p[0] - mono_pos[0]).powi(2) + (p[1] - mono_pos[1]).powi(2)).sqrt();
            assert!(dp < 1e-12, "{counts:?} {ts:?} body {bi} position desync/mismatch: {dp:e}");
            assert!((a - mono_acc).abs() < 1e-10, "{counts:?} {ts:?} body {bi} accreted-mass mismatch: {a} vs {mono_acc}");
        }
        // distance the body moved from its initial position (binary non-vacuous check).
        let r0 = (mono_pos[0].powi(2) + mono_pos[1].powi(2)).sqrt();
        moved = moved.max((r0 - 0.25).abs().max(mono_pos[1].abs()));
    }
    if expect_accretion {
        let max_removed = mono_den.iter().map(|d| 2.0 - d).fold(0.0_f64, f64::max);
        assert!(max_removed > 1e-3, "sink removed no mass ({max_removed:e}); test is vacuous");
        assert!(accreted > 1e-6, "body recorded no accretion ({accreted:e}); diagnostic test vacuous");
    } else {
        assert!(moved > 1e-4, "binary did not move ({moved:e}); prescribed-orbit test vacuous");
    }

    // the production gather path (the python checkpoint output).
    let bnd = Boundaries(std::array::from_fn(|_| [BoundaryType::Outflow; 2]));
    let global = make([N, N], [-L, -L], bnd, ts, bodies);
    let stores: Vec<_> = dec.iter().map(|(s, _)| &**s).collect();
    gather_interiors(&*global.0, &stores, counts);
    let gathered = global_field(std::slice::from_ref(&global), [1, 1], den);
    let ge = max_err(&gathered, &dec_den);
    assert!(ge < 1e-12, "{counts:?} {ts:?} gather density err {ge:e}");
}

#[test]
fn body_euler_two_tile_x_cut() {
    assert_body_matches([2, 1], Timestepping::Euler, central_bh, true);
}

#[test]
fn body_euler_two_tile_y_cut() {
    assert_body_matches([1, 2], Timestepping::Euler, central_bh, true);
}

#[test]
fn body_rk2_two_tile_x_cut() {
    assert_body_matches([2, 1], Timestepping::Rk2, central_bh, true);
}

// the 2x2 corner under rk2 -- the body sits exactly on the four-tile corner, so its gravity and
// sink straddle every cut. the hardest case. also asserts the cross-tile accreted-mass diagnostic
// SUM (step_bodies_decomposed) equals the monolithic value and all tiles stay in lockstep.
#[test]
fn body_rk2_quad_tile_2d_grid() {
    assert_body_matches([2, 2], Timestepping::Rk2, central_bh, true);
}

// a MOVING binary across the 2x2 corner: the two bodies orbit (prescribed Keplerian advance run per
// tile in step_bodies_decomposed). the fluid feels the rotating gravity; decomposed == monolithic
// requires every tile to advance the orbit identically (lockstep body positions). asserts the
// bodies actually moved (non-vacuous) and stayed synced across tiles.
#[test]
fn binary_rk2_quad_tile_2d_grid() {
    assert_body_matches([2, 2], Timestepping::Rk2, binary, false);
}

#[test]
fn binary_euler_two_tile_x_cut() {
    assert_body_matches([2, 1], Timestepping::Euler, binary, false);
}
