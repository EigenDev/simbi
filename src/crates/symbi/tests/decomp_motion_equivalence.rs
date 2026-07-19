// =============================================================================
// decomp_motion_equivalence.rs
//
// MESH MOTION x DECOMPOSITION: tiles evolving a homologously expanding (or uniformly
// translating) mesh must reproduce the monolithic single-grid run to round-off. every tile
// carries the IDENTICAL motion state and the decomposed loop advances all tile clocks and
// scale factors in lockstep with the shared global dt — identical dt sequence, identical
// a(t) sequence, bit-for-bit. the cuts sit at fixed COMOVING indices, so the halo exchange
// is untouched by the expansion. the monolithic reference is the production single-grid
// `evolve` loop (its own per-step motion advance), so this oracle pins the decomposed
// advance semantics against the canonical ones — a tile that failed to advance its clock or
// scale factor diverges immediately (the fluxes carry a(t)).
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{evolve_decomposed, flatten, unflatten, LocalCopy};
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{Cartesian, MotionState};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const N: usize = 32;
const DX: f64 = 1.0 / N as f64;
const T_FINAL: f64 = 0.04;
const ADOT: f64 = 0.5;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

fn bump(x: f64, y: f64) -> f64 {
    0.3 * (-(((x - 0.5) / 0.1).powi(2) + ((y - 0.5) / 0.1).powi(2))).exp()
}

fn make(cells: [usize; 2], origin: [f64; 2], bnd: Boundaries<2>, motion: MotionState<f64>) -> (Sim, Kern) {
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 2])
        .origin(origin)
        .boundaries(bnd)
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim construction failed")
        .set_initial(|[x, y]| {
            let b = bump(x, y);
            Prim { rho: 1.0 + b, vel: Tensor::new([0.0, 0.0]), pre: 1.0 + b }
        })
        .build();
    sim.motion = motion;
    let k = Kern::new(GAMMA, CFL, &sim.geom.allocated);
    (sim, k)
}

fn grid_tiles(counts: [usize; 2], motion: MotionState<f64>) -> Vec<(Sim, Kern)> {
    let m: [usize; 2] = std::array::from_fn(|a| N / counts[a]);
    (0..counts.iter().product::<usize>())
        .map(|flat| {
            let tc = unflatten(flat, counts);
            let origin = std::array::from_fn(|a| tc[a] as f64 * m[a] as f64 * DX);
            let bnd = Boundaries(std::array::from_fn(|a| {
                let lo = if tc[a] == 0 { BoundaryType::Outflow } else { BoundaryType::CoarseFine };
                let hi = if tc[a] == counts[a] - 1 { BoundaryType::Outflow } else { BoundaryType::CoarseFine };
                [lo, hi]
            }));
            make(m, origin, bnd, motion)
        })
        .collect()
}

fn run_decomposed(tiles: &mut [(Sim, Kern)], counts: [usize; 2]) {
    let devices: Vec<i32> = vec![0; tiles.len()];
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
        Timestepping::Rk2,
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

fn assert_motion_matches(counts: [usize; 2], motion: MotionState<f64>) {
    let den = |s: &Sim, c: [isize; 2]| *s.fields.cons.den.view().at(c);
    let momx = |s: &Sim, c: [isize; 2]| *s.fields.cons.mom[0].view().at(c);

    let (mut mono, mk) = make([N, N], [0.0, 0.0], Boundaries::uniform(BoundaryType::Outflow), motion);
    evolve(&mut mono, &mk, T_FINAL).expect("mono evolve");
    let mono_wrap = [(mono, mk)];
    let mono_den = global_field(&mono_wrap, [1, 1], den);
    let mono_momx = global_field(&mono_wrap, [1, 1], momx);
    let mono_sim = &mono_wrap[0].0;

    let mut tiles = grid_tiles(counts, motion);
    run_decomposed(&mut tiles, counts);
    let dec_den = global_field(&tiles, counts, den);
    let dec_momx = global_field(&tiles, counts, momx);

    assert!(
        mono_den.iter().all(|v| v.is_finite()) && dec_den.iter().all(|v| v.is_finite()),
        "some global cells were never written"
    );
    // every tile's clock and scale factor track the monolithic run exactly: a stale tile
    // clock (or a never-advanced scale factor) is the failure mode under test.
    for (i, (s, _)) in tiles.iter().enumerate() {
        assert!(
            (s.time - mono_sim.time).abs() < 1e-12,
            "{counts:?} tile {i}: clock {:.6e} != mono {:.6e}",
            s.time,
            mono_sim.time
        );
        assert!(
            (s.motion.a - mono_sim.motion.a).abs() < 1e-14,
            "{counts:?} tile {i}: scale factor {} != mono {}",
            s.motion.a,
            mono_sim.motion.a
        );
    }
    // the mesh genuinely moved (non-vacuous): an expanding mesh grows a past 1.
    if motion.homologous {
        assert!(
            mono_sim.motion.a > 1.0 + 0.5 * ADOT * T_FINAL,
            "the scale factor never advanced (a = {}); test is vacuous",
            mono_sim.motion.a
        );
    }
    let de = max_err(&mono_den, &dec_den);
    let me = max_err(&mono_momx, &dec_momx);
    assert!(de < 1e-12, "{counts:?}: density diverged under mesh motion: {de:e}");
    assert!(me < 1e-12, "{counts:?}: mom_x diverged under mesh motion: {me:e}");
}

#[test]
fn homologous_expansion_two_tiles_x() {
    assert_motion_matches([2, 1], MotionState::homologous(1.0, ADOT));
}

#[test]
fn homologous_expansion_two_tiles_y() {
    assert_motion_matches([1, 2], MotionState::homologous(1.0, ADOT));
}

#[test]
fn homologous_expansion_four_tiles() {
    assert_motion_matches([2, 2], MotionState::homologous(1.0, ADOT));
}

#[test]
fn uniform_translation_two_tiles() {
    // translation: a stays 1, a_dot is the frame velocity along x; the cut rides the
    // comoving indices while the flux carries the frame advection.
    assert_motion_matches([2, 1], MotionState::uniform(1.0, 0.25));
}

#[test]
fn homologous_single_tile_matches_raw_evolve() {
    // discriminator: a 1x1 "decomposition" runs the decomposed pipeline on the whole grid —
    // with no tile cuts present, any divergence here isolates a pipeline difference under motion.
    assert_motion_matches([1, 1], MotionState::homologous(1.0, ADOT));
}
