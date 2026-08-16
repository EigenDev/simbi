// =============================================================================
// decomp_shaped_wall_equivalence.rs
//
// the shaped rigid-wall correctness contract under domain decomposition: a domain
// split into tiles, each carrying the same shaped (CSG) rigid wall at the global
// position, evolved through `evolve_decomposed`, must reproduce the monolithic run
// to round-off for the fluid (den/mom/nrg).
//
// the shaped penalization is pointwise (each cell relaxes its momentum toward the
// wall from the body's global position via the mask at that cell), and the per-body
// force/torque receipts are reduced per-tile then summed across tiles
// (step_bodies_decomposed) -- the tile interiors partition the global interior, so
// the sum is the monolithic reduction. hence decomposed == monolithic by
// construction; this pins it. the wall sits on the 2x2 tile corner so its support
// straddles every cut -- the worst case. a tile that dispatched the shaped bbox at a
// wrong local coordinate, or clipped the support wrong at a cut, would diverge here.
//
// cpu tiles + 2d: the same exchange index math as the gpu path (the decomposition is
// memory-space agnostic); combined with the monolithic device parity gate
// (rigid_no_penetration_gpu) this establishes the decomposed device path too.
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, evolve_decomposed, flatten, unflatten};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_ib::sdf::SdfExpr;
use symbi_ib::{Body, BodyCollection, SurfaceSpec};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const N: usize = 32;
const L: f64 = 1.0; // domain [-L, L]^2, wall at the origin (the 2x2 corner)
const DX: f64 = 2.0 * L / N as f64;
const R_BODY: f64 = 0.3;
const V_INF: f64 = 0.3;
const T_FINAL: f64 = 0.03;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

// a fresh sealed capped-cylinder rigid wall at the global origin, per tile.
fn make(cells: [usize; 2], origin: [f64; 2], bnd: Boundaries<2>, ts: Timestepping) -> (Sim, Kern) {
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 2])
        .origin(origin)
        .boundaries(bnd)
        .cfl(CFL)
        .timestepping(ts)
        .allocate()
        .expect("sim")
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([V_INF, 0.0]),
            pre: 1.0,
        })
        .build()
        .with_bodies(
            BodyCollection::new().add(
                Body::rigid_sphere(
                    0,
                    Tensor::new([0.0, 0.0]),
                    Tensor::zeros(),
                    1.0,
                    R_BODY,
                    0.1,
                    false,
                )
                .with_surface(SurfaceSpec::Porous {
                    porosity: 0.0,
                    k_eta_n: 1.0e3,
                    k_eta_t: 0.0,
                }),
            ),
        );
    // the CSG shape routes dispatch to the runtime shaped kernel.
    sim.immersed.as_mut().unwrap().shapes[0] = Some(SdfExpr::<f64, 3>::capped_cylinder(
        [0.0, 0.0, 0.0],
        R_BODY,
        1.0,
    ));
    let k = Kern::new(GAMMA, CFL, &sim.geom.allocated);
    (sim, k)
}

fn grid_tiles(counts: [usize; 2], ts: Timestepping) -> Vec<(Sim, Kern)> {
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
            make(m, origin, bnd, ts)
        })
        .collect()
}

fn run(tiles: &mut [(Sim, Kern)], counts: [usize; 2], ts: Timestepping) {
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
        ts,
        0.0,
        T_FINAL,
        u64::MAX,
        &LocalCopy,
        |_, _, _| std::ops::ControlFlow::Continue(()),
    );
}

fn global_field(
    tiles: &[(Sim, Kern)],
    counts: [usize; 2],
    pick: impl Fn(&Sim, [isize; 2]) -> f64,
) -> Vec<f64> {
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
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn assert_decomposed_matches(counts: [usize; 2], ts: Timestepping) {
    let den = |s: &Sim, c| *s.fields.cons.den.view().at(c);
    let momx = |s: &Sim, c| *s.fields.cons.mom[0].view().at(c);
    let momy = |s: &Sim, c| *s.fields.cons.mom[1].view().at(c);
    let nrg = |s: &Sim, c| *s.fields.cons.nrg_field().unwrap().view().at(c);

    let mut mono = grid_tiles([1, 1], ts);
    run(&mut mono, [1, 1], ts);
    let mut dec = grid_tiles(counts, ts);
    run(&mut dec, counts, ts);

    for (name, pick) in [
        ("den", &den as &dyn Fn(&Sim, [isize; 2]) -> f64),
        ("momx", &momx),
        ("momy", &momy),
        ("nrg", &nrg),
    ] {
        let mv = global_field(&mono, [1, 1], &pick);
        let dv = global_field(&dec, counts, &pick);
        assert!(
            mv.iter().all(|v| v.is_finite()) && dv.iter().all(|v| v.is_finite()),
            "unwritten cells"
        );
        let e = max_err(&mv, &dv);
        assert!(
            e < 1e-12,
            "{counts:?} {ts:?} {name} decomposed!=mono under shaped wall: err {e:e}"
        );
    }

    // non-vacuous: the sealed wall must have suppressed the wall-normal momentum somewhere, so the
    // fluid genuinely deviates from the free stream (else the test passes on an untouched flow).
    let mono_momy = global_field(&mono, [1, 1], &momy);
    let dev_from_stream = mono_momy.iter().map(|m| m.abs()).fold(0.0_f64, f64::max);
    assert!(
        dev_from_stream > 1e-3,
        "the shaped wall never perturbed the flow ({dev_from_stream:e}); test vacuous"
    );
}

#[test]
fn shaped_wall_euler_two_tile_x_cut() {
    assert_decomposed_matches([2, 1], Timestepping::Euler);
}

#[test]
fn shaped_wall_euler_two_tile_y_cut() {
    assert_decomposed_matches([1, 2], Timestepping::Euler);
}

#[test]
fn shaped_wall_rk2_two_tile_x_cut() {
    assert_decomposed_matches([2, 1], Timestepping::Rk2);
}

// the 2x2 corner under rk2: the wall sits exactly on the four-tile corner, so its support straddles
// every cut -- the hardest decomposition case for the shaped bbox dispatch + the receipt reduction.
#[test]
fn shaped_wall_rk2_quad_tile_2d_grid() {
    assert_decomposed_matches([2, 2], Timestepping::Rk2);
}
