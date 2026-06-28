// =============================================================================
// decomp_mhd_equivalence.rs
//
// the MHD correctness contract for multi-gpu domain decomposition (docs/design/37 M4): a 2d
// RMHD grid split into a 2x2 tile grid, evolved in lockstep with the same-level halo exchange,
// must reproduce the monolithic run to round-off AND keep div(B) at machine zero across the
// tile cuts. the second check is the MHD-specific one: a wrong staggered `bface` exchange
// compiles fine and runs, but silently creates a magnetic monopole at the seam.
//
// cpu-only + 2d on purpose: 2d is the minimal constrained-transport case (one E_z edge), and a
// host run exercises the SAME exchange index math (`exchange_faces`/`face_ghost_strip`) as the
// gpu path -- so this is the fast iteration loop for the staggered exchange. a gpu variant rides
// on the same logic once this is green.
//
// the design hypothesis under test: only the TRANSVERSE bface halos need exchanging; the normal
// (shared interface) face stays bit-identical by construction. if div(B) drifts, the hypothesis
// is wrong and the exchange needs the shared face synced.
// =============================================================================

use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet;
use symbi::sim::decomp::{evolve_decomposed, exchange_grid, flatten, unflatten, LocalCopy};
use symbi::sim::evolve::KernelSet;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const N: usize = 32; // cells per axis
const DX: f64 = 1.0 / N as f64;
const T_FINAL: f64 = 0.02;
const B0: [f64; 3] = [0.3, 0.2, 0.1];

type Sim = SimStateGeneric<Rmhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = RmhdSubstrateKernelSet<HostMemory, f64, 2>;

// a smooth centered pressure+density bump -> the flow develops velocity, the v x B EMF evolves
// B through the CT update, so the staggered exchange is genuinely exercised. zero initial
// velocity + outflow boundaries keep mono == decomposed exact (waves stay inside).
fn ic(x: f64, y: f64) -> MhdPrim<f64, 3> {
    let r2 = (x - 0.5).powi(2) + (y - 0.5).powi(2);
    let b = 0.2 * (-(r2 / 0.01)).exp();
    MhdPrim {
        hydro: Prim {
            rho: 1.0 + b,
            vel: Tensor::new([0.0, 0.0, 0.0]),
            pre: 1.0 + b,
        },
        mag: Tensor::new(B0),
    }
}

fn make(cells: [usize; 2], origin: [f64; 2], bnd: Boundaries<2>, ts: Timestepping) -> (Sim, Kern) {
    let sim = Sim::build(Rmhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 2])
        .origin(origin)
        .boundaries(bnd)
        .cfl(CFL)
        .timestepping(ts)
        .allocate()
        .expect("mhd sim construction failed")
        .set_initial(|[x, y]| ic(x, y))
        .seed_faces_uniform([B0[0], B0[1]]) // uniform staggered B (div-free); CT evolves it
        .build();
    let k = Kern::new(GAMMA, CFL, 1.0, &sim.geom.allocated);
    (sim, k)
}

// build the 2x2 (or counts) tile grid: each tile an equal slice, CoarseFine on internal faces,
// Outflow on the outer domain boundary. mirrors the hydro harness.
fn grid_tiles(counts: [usize; 2], ts: Timestepping) -> Vec<(Sim, Kern)> {
    let m: [usize; 2] = std::array::from_fn(|a| {
        assert!(N % counts[a] == 0, "N must split evenly into counts[{a}]");
        N / counts[a]
    });
    let total: usize = counts.iter().product();
    (0..total)
        .map(|flat| {
            let tc = unflatten(flat, counts);
            let origin = std::array::from_fn(|a| tc[a] as f64 * m[a] as f64 * DX);
            let bnd = Boundaries(std::array::from_fn(|a| {
                let lo = if tc[a] == 0 { BoundaryType::Outflow } else { BoundaryType::CoarseFine };
                let hi = if tc[a] == counts[a] - 1 { BoundaryType::Outflow } else { BoundaryType::CoarseFine };
                [lo, hi]
            }));
            make(m, origin, bnd, ts)
        })
        .collect()
}

fn run(tiles: &mut [(Sim, Kern)], counts: [usize; 2], ts: Timestepping) {
    let stores: Vec<_> = tiles.iter().map(|(s, _)| &**s).collect();
    let kernels: Vec<_> = tiles.iter().map(|(_, k)| k).collect();
    let devices: Vec<i32> = vec![0; tiles.len()]; // host: all "device 0" (no-op with_device)
    evolve_decomposed(
        &stores,
        &kernels,
        counts,
        &devices,
        ts,
        0.0,
        T_FINAL,
        u64::MAX,
        &LocalCopy,
        |_, _| std::ops::ControlFlow::Continue(()),
    );
}

// scatter every tile's interior density into one global N^2 grid by global cell coordinate.
fn global_den(tiles: &[(Sim, Kern)], counts: [usize; 2]) -> Vec<f64> {
    let m: [usize; 2] = std::array::from_fn(|a| N / counts[a]);
    let mut out = vec![f64::NAN; N * N];
    for (flat_tile, (sim, _)) in tiles.iter().enumerate() {
        let tc = unflatten(flat_tile, counts);
        let ilo: [isize; 2] = std::array::from_fn(|a| sim.geom.interior.spaces[a].lo);
        for c in sim.geom.interior.iter() {
            let g: [usize; 2] = std::array::from_fn(|a| tc[a] * m[a] + (c[a] - ilo[a]) as usize);
            out[flatten(g, [N; 2])] = *sim.fields.cons.den.view().at(c);
        }
    }
    out
}

// max |div(B)| over a tile's interior cells: div(B)[c] = sum_d (bface[d](c + e_d) - bface[d](c)) / dx.
// the staggered CT update keeps this at machine zero; a broken seam exchange makes it spike at
// the cut-adjacent cells.
fn div_b_max(sim: &Sim) -> f64 {
    let mhd = sim.fields.mhd.as_ref().expect("rmhd has mhd fields");
    let mut worst = 0.0_f64;
    for c in sim.geom.interior.iter() {
        let mut div = 0.0;
        for d in 0..2 {
            let mut chi = c;
            chi[d] += 1;
            let lo = *mhd.bface[d].view().at(c);
            let hi = *mhd.bface[d].view().at(chi);
            div += (hi - lo) / DX;
        }
        worst = worst.max(div.abs());
    }
    worst
}

fn tiles_div_b_max(tiles: &[(Sim, Kern)]) -> f64 {
    tiles.iter().map(|(s, _)| div_b_max(s)).fold(0.0_f64, f64::max)
}

// scatter cell-centered B component `comp` (0..3) into one global N^2 grid. the REAL MHD check:
// the CT curl preserves div(B) for ANY emf, so div(B)==0 does not prove the field is right --
// only `bcell` value equivalence vs the monolithic run does.
fn global_bcell(tiles: &[(Sim, Kern)], counts: [usize; 2], comp: usize) -> Vec<f64> {
    let m: [usize; 2] = std::array::from_fn(|a| N / counts[a]);
    let mut out = vec![f64::NAN; N * N];
    for (flat_tile, (sim, _)) in tiles.iter().enumerate() {
        let tc = unflatten(flat_tile, counts);
        let ilo: [isize; 2] = std::array::from_fn(|a| sim.geom.interior.spaces[a].lo);
        let mhd = sim.fields.mhd.as_ref().unwrap();
        for c in sim.geom.interior.iter() {
            let g: [usize; 2] = std::array::from_fn(|a| tc[a] * m[a] + (c[a] - ilo[a]) as usize);
            out[flatten(g, [N; 2])] = *mhd.bcell.b[comp].view().at(c);
        }
    }
    out
}

// scatter the CURRENT edge EMF (efield.e[slot]) -- the freshly recomputed stage-2 emf, before
// post_godunov averages it with efield_n and curls it into bface.
fn global_ef(tiles: &[(Sim, Kern)], counts: [usize; 2], slot: usize) -> Vec<f64> {
    let m: [usize; 2] = std::array::from_fn(|a| N / counts[a]);
    let mut out = vec![f64::NAN; N * N];
    for (flat_tile, (sim, _)) in tiles.iter().enumerate() {
        let tc = unflatten(flat_tile, counts);
        let ilo: [isize; 2] = std::array::from_fn(|a| sim.geom.interior.spaces[a].lo);
        let mhd = sim.fields.mhd.as_ref().unwrap();
        for c in sim.geom.interior.iter() {
            let g: [usize; 2] = std::array::from_fn(|a| tc[a] * m[a] + (c[a] - ilo[a]) as usize);
            out[flatten(g, [N; 2])] = *mhd.efield.e[slot].view().at(c);
        }
    }
    out
}

// scatter the SAVED stage-1 edge EMF (efield_n.e[slot]) read at each interior cell's corner.
// stage 2 averages this into the CT curl; if it diverges, the corrector's emf is inconsistent.
fn global_efn(tiles: &[(Sim, Kern)], counts: [usize; 2], slot: usize) -> Vec<f64> {
    let m: [usize; 2] = std::array::from_fn(|a| N / counts[a]);
    let mut out = vec![f64::NAN; N * N];
    for (flat_tile, (sim, _)) in tiles.iter().enumerate() {
        let tc = unflatten(flat_tile, counts);
        let ilo: [isize; 2] = std::array::from_fn(|a| sim.geom.interior.spaces[a].lo);
        let mhd = sim.fields.mhd.as_ref().unwrap();
        for c in sim.geom.interior.iter() {
            let g: [usize; 2] = std::array::from_fn(|a| tc[a] * m[a] + (c[a] - ilo[a]) as usize);
            out[flatten(g, [N; 2])] = *mhd.efield_n.e[slot].view().at(c);
        }
    }
    out
}

// argmax abs-difference between two global grids: (flat index, error).
fn argmax_diff(a: &[f64], b: &[f64]) -> (usize, f64) {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .enumerate()
        .fold((0usize, 0.0_f64), |(bi, be), (i, e)| if e > be { (i, e) } else { (bi, be) })
}

fn assert_matches(counts: [usize; 2], ts: Timestepping) {
    let mut mono = grid_tiles([1; 2], ts);
    run(&mut mono, [1; 2], ts);
    let mono_vals = global_den(&mono, [1; 2]);
    let mono_divb = tiles_div_b_max(&mono);

    let mut dec = grid_tiles(counts, ts);
    run(&mut dec, counts, ts);
    let dec_vals = global_den(&dec, counts);
    let dec_divb = tiles_div_b_max(&dec);

    assert!(
        mono_vals.iter().all(|v| v.is_finite()) && dec_vals.iter().all(|v| v.is_finite()),
        "some global cells were never written (gather bug)"
    );
    // 1. density equivalence: decomposed == monolithic to round-off.
    // diagnostics FIRST (B-field, the source; then density, downstream), then assert. the curl
    // preserves div(B) for any emf, so div(B)==0 is necessary but not sufficient -- the bcell
    // VALUE equivalence is the real field check.
    let (di, de) = argmax_diff(&mono_vals, &dec_vals);
    for comp in 0..3 {
        let mb = global_bcell(&mono, [1; 2], comp);
        let db = global_bcell(&dec, counts, comp);
        let (bi, be) = argmax_diff(&mb, &db);
        if be >= 1e-11 {
            eprintln!(
                "[B{comp}] {counts:?} {ts:?}: max bcell[{comp}] err {be:e} at ({},{}) [center=({},{})] mono={} dec={}",
                bi % N, bi / N, N / 2, N / 2, mb[bi], db[bi]
            );
        }
    }
    if de >= 1e-11 {
        eprintln!(
            "[rho] {counts:?} {ts:?}: max density err {de:e} at ({},{}); div(B) mono={mono_divb:e} dec={dec_divb:e}",
            di % N, di / N
        );
    }
    assert!(dec_divb < 1e-10, "div(B) {dec_divb:e} (mono {mono_divb:e})");
    for comp in 0..3 {
        let mb = global_bcell(&mono, [1; 2], comp);
        let db = global_bcell(&dec, counts, comp);
        let (_, be) = argmax_diff(&mb, &db);
        assert!(be < 1e-11, "{counts:?} {ts:?} bcell[{comp}] err {be:e} (div-free but WRONG)");
    }
    assert!(de < 1e-11, "{counts:?} {ts:?} density err {de:e}");
}

// DEBUG: step mono and the 2x2 decomposition in lockstep through ONE RK2 step, comparing the
// cell-centered B after prime, stage 1, and stage 2. RK2 stage 1 (a0=0,ac=1) is an Euler step and
// corner-Euler passes, so the divergence must appear in stage 2 -- this pins which operation.
#[test]
#[ignore = "diagnostic, run with --ignored --nocapture"]
fn mhd_debug_rk2_stages() {
    let ts = Timestepping::Rk2;
    let mono = grid_tiles([1, 1], ts);
    let dec = grid_tiles([2, 2], ts);
    let exch = |tiles: &[(Sim, Kern)], counts: [usize; 2]| {
        let stores: Vec<_> = tiles.iter().map(|(s, _)| &**s).collect();
        let devs = vec![0i32; tiles.len()];
        exchange_grid(&stores, counts, &devs, &LocalCopy);
    };
    // prime BEFORE cfl: cfl reads prim, which c2p populates (otherwise it returns inf).
    let prime = |tiles: &[(Sim, Kern)], counts: [usize; 2]| {
        for (s, k) in tiles {
            k.c2p(&**s);
            k.ghost_fill(&**s);
        }
        exch(tiles, counts);
    };
    let stage = |tiles: &[(Sim, Kern)], counts: [usize; 2], si: usize, a0: f64, ac: f64, dt: f64| {
        for (s, k) in tiles {
            k.wave_speeds(&**s);
            for d in 0..2 {
                k.flux(&**s, d);
            }
            k.efield(&**s);
            k.godunov_stage(&**s, dt, a0, ac);
            k.post_godunov(&**s, dt, (si + 1) as u8);
            k.c2p(&**s);
            k.ghost_fill(&**s);
        }
        exch(tiles, counts);
    };
    let cmp = |when: &str| {
        for comp in 0..3 {
            let mb = global_bcell(&mono, [1, 1], comp);
            let db = global_bcell(&dec, [2, 2], comp);
            let (bi, be) = argmax_diff(&mb, &db);
            eprintln!("[{when}] bcell[{comp}] err {be:e} at ({},{})", bi % N, bi / N);
        }
    };

    prime(&mono, [1, 1]);
    prime(&dec, [2, 2]);
    cmp("prime ");
    // cfl AFTER prime (prim is now valid). per-run dt, exactly as evolve_decomposed does.
    let dt_mono = mono.iter().map(|(s, k)| k.cfl(&**s)).fold(f64::INFINITY, f64::min);
    let dt_dec = dec.iter().map(|(s, k)| k.cfl(&**s)).fold(f64::INFINITY, f64::min);
    eprintln!(
        "[cfl] dt_mono={dt_mono:.17e} dt_dec={dt_dec:.17e} diff={:.3e}",
        (dt_mono - dt_dec).abs()
    );
    for (s, k) in &mono {
        k.snapshot(&**s);
    }
    for (s, k) in &dec {
        k.snapshot(&**s);
    }
    let stages = ts.stages();
    stage(&mono, [1, 1], 0, stages[0].0, stages[0].1, dt_mono);
    stage(&dec, [2, 2], 0, stages[0].0, stages[0].1, dt_dec);
    cmp("stage1");
    // efield_n was just saved during stage 1's post_godunov. is it consistent across the cut?
    for slot in 0..2 {
        let me = global_efn(&mono, [1, 1], slot);
        let de = global_efn(&dec, [2, 2], slot);
        let (bi, be) = argmax_diff(&me, &de);
        // flatten = gx*N + gy, so gx = bi/N, gy = bi%N.
        eprintln!("[efn{slot}] after stage1: err {be:e} at (gx={},gy={})", bi / N, bi % N);
    }
    // dump ACTUAL field values around the REAL corner (x-cut x=16, bottom outflow y=0). INDEX
    // LOCALLY: global (gx, gy) -> owning tile (gx/16, gy/16) at interior-lo + the in-tile offset.
    let dump = |label: &str, sim: &Sim, gx: usize, gy: usize, tx: usize, ty: usize| {
        let i = &sim.geom.interior;
        let c = [
            i.spaces[0].lo + (gx - tx * 16) as isize,
            i.spaces[1].lo + (gy - ty * 16) as isize,
        ];
        let mhd = sim.fields.mhd.as_ref().unwrap();
        eprintln!(
            "  {label} g({gx:>2},{gy:>2}): efn0={:+.4e} bc0={:+.4e} bc1={:+.4e} bf0={:+.4e} vx={:+.4e} vy={:+.4e} wsl0={:+.4e} wsr0={:+.4e}",
            mhd.efield_n.e[0].view().at(c),
            mhd.bcell.b[0].view().at(c),
            mhd.bcell.b[1].view().at(c),
            mhd.bface[0].view().at(c),
            sim.fields.prim.vel[0].view().at(c),
            sim.fields.prim.vel[1].view().at(c),
            mhd.wave_speed_l[0].view().at(c),
            mhd.wave_speed_r[0].view().at(c),
        );
    };
    for gx in 14..18 {
        let gy = 0;
        let tx = gx / 16;
        dump("mono", &mono[0].0, gx, gy, 0, 0);
        dump("dec ", &dec[flatten([tx, 0], [2, 2])].0, gx, gy, tx, 0);
    }
    // stage 2 SPLIT: run wave_speeds+flux+efield (recompute the corrector emf), capture it, then
    // finish (godunov, post_godunov averages+curls, c2p, ghost_fill). this isolates whether the
    // recomputed stage-2 efield (vs efield_n) is the inconsistent input.
    for (s, k) in &mono {
        k.wave_speeds(&**s);
        for d in 0..2 {
            k.flux(&**s, d);
        }
        k.efield(&**s);
    }
    for (s, k) in &dec {
        k.wave_speeds(&**s);
        for d in 0..2 {
            k.flux(&**s, d);
        }
        k.efield(&**s);
    }
    for slot in 0..2 {
        let me = global_ef(&mono, [1, 1], slot);
        let de = global_ef(&dec, [2, 2], slot);
        let (bi, be) = argmax_diff(&me, &de);
        eprintln!("[ef{slot}] stage2 recomputed (pre-curl): err {be:e} at ({},{})", bi % N, bi / N);
    }
    for (s, k) in &mono {
        k.godunov_stage(&**s, dt_mono, stages[1].0, stages[1].1);
        k.post_godunov(&**s, dt_mono, 2);
        k.c2p(&**s);
        k.ghost_fill(&**s);
    }
    for (s, k) in &dec {
        k.godunov_stage(&**s, dt_dec, stages[1].0, stages[1].1);
        k.post_godunov(&**s, dt_dec, 2);
        k.c2p(&**s);
        k.ghost_fill(&**s);
    }
    exch(&dec, [2, 2]);
    cmp("stage2");
}

#[test]
fn mhd_euler_two_tile_x_cut() {
    assert_matches([2, 1], Timestepping::Euler);
}

// isolation: single y-cut (symmetric to x) + single-cut RK2 + the 2x2 corner under Euler.
// together with the x-cut + 2x2-rk2 these localize a failure to corner-vs-RK2.
#[test]
fn mhd_euler_two_tile_y_cut() {
    assert_matches([1, 2], Timestepping::Euler);
}

#[test]
fn mhd_rk2_two_tile_x_cut() {
    assert_matches([2, 1], Timestepping::Rk2);
}

// the y-cut counterpart of the x-cut RK2 test. if this passes but the 2x2 fails, the bug needs
// BOTH cuts (the corner); if it fails, it's single-cut + RK2 + the outflow boundary.
#[test]
fn mhd_rk2_two_tile_y_cut() {
    assert_matches([1, 2], Timestepping::Rk2);
}

#[test]
fn mhd_euler_quad_tile_2d_grid() {
    assert_matches([2, 2], Timestepping::Euler);
}

// the 2x2 corner under RK2 -- the hard case. FIXED: the bug was a ghost_fill/exchange ORDERING
// defect in evolve_decomposed (ghost_fill ran before the cut exchange, so a domain-boundary ghost
// at a boundary-meets-cut corner read a stale unexchanged cut cell -> spurious edge-EMF, only
// exposed by the RK2 corrector averaging the saved stage-1 emf). moving ghost_fill AFTER the
// exchange fixed it; div(B) exact and decomposed == monolithic to round-off here too.
#[test]
fn mhd_rk2_quad_tile_2d_grid() {
    assert_matches([2, 2], Timestepping::Rk2);
}
