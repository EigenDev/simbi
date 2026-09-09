// =============================================================================
// decomp_mhd_sink_equivalence.rs
//
// the magnetized-sink decomposition contract: a draining point mass placed on the tile corner of
// a magnetized gas, evolved through `evolve_decomposed`, reproduces the monolithic run to
// round-off in the gas, the cell field, and the stored primitives. the drain runs once per step
// after the last stage and relaxes each cell at its local Alfven rate, then every tile recovers
// its primitives and refreshes its cut halos, so the next step's reconstruction reads the
// drained gas across cuts exactly as the single grid does. the transparent and the resistive
// couplings are both pinned; the resistive body adds a masked EMF that straddles the cuts.
// =============================================================================

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, evolve_decomposed, flatten, unflatten};
use symbi::sim::evolve::evolve_with_callback;
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
const N: usize = 32;
const DX: f64 = 1.0 / N as f64;
const T_FINAL: f64 = 0.06;
const B0: [f64; 3] = [0.6, 0.3, 0.2];
const BODY: [f64; 2] = [0.5, 0.5];
const R_BODY: f64 = 0.15;

type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;
type Kern = NewtonianMhdSubstrateKernelSet<HostMemory, f64, 2>;

fn make(
    cells: [usize; 2],
    origin: [f64; 2],
    bnd: Boundaries<2>,
    magnetic: MagneticSpec,
) -> (Sim, Kern) {
    let sim = Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 2])
        .origin(origin)
        .boundaries(bnd)
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("mhd sim")
        .set_initial(|_| {
            MhdPrim::new(
                Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)),
                Tensor::new(B0),
            )
        })
        .seed_faces_uniform([B0[0], B0[1]])
        .build()
        .with_bodies(
            BodyCollection::new().add(
                Body::black_hole(
                    0,
                    Tensor::new(BODY),
                    Tensor::zeros(),
                    1.0,
                    R_BODY,
                    0.05,
                    1.0,
                    1.0,
                    R_BODY,
                )
                .with_surface(SurfaceSpec::Drain)
                .with_magnetic(magnetic),
            ),
        );
    let k = Kern::new(GAMMA, CFL, 1.0, &sim.geom.allocated);
    (sim, k)
}

fn grid_tiles(counts: [usize; 2], magnetic: MagneticSpec) -> Vec<(Sim, Kern)> {
    let m: [usize; 2] = std::array::from_fn(|a| {
        assert!(N % counts[a] == 0, "N must split into counts[{a}]");
        N / counts[a]
    });
    let total: usize = counts.iter().product();
    (0..total)
        .map(|flat| {
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
            make(m, origin, bnd, magnetic)
        })
        .collect()
}

fn run(tiles: &mut [(Sim, Kern)], counts: [usize; 2]) {
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

fn assert_decomposed_matches(counts: [usize; 2], magnetic: MagneticSpec) {
    let den = |s: &Sim, c| *s.fields.cons.den.view().at(c);
    let momx = |s: &Sim, c| *s.fields.cons.mom[0].view().at(c);
    let momy = |s: &Sim, c| *s.fields.cons.mom[1].view().at(c);
    let momz = |s: &Sim, c| *s.fields.cons.mom[2].view().at(c);
    let nrg = |s: &Sim, c| *s.fields.cons.nrg_field().unwrap().view().at(c);
    let bcell0 = |s: &Sim, c| *s.fields.mhd.as_ref().unwrap().bcell[0].view().at(c);
    let bcell1 = |s: &Sim, c| *s.fields.mhd.as_ref().unwrap().bcell[1].view().at(c);
    let rho = |s: &Sim, c| *s.fields.prim.rho.view().at(c);
    let pre = |s: &Sim, c| *s.fields.prim.pre_field().unwrap().view().at(c);

    // the reference is the single-grid driver, so the decomposed loop's own step tail (the drain,
    // the recovery, the halo refresh) is measured against the monolithic step.
    let mut mono = vec![make(
        [N, N],
        [0.0, 0.0],
        Boundaries::uniform(BoundaryType::Outflow),
        magnetic,
    )];
    {
        let (sim, k) = &mut mono[0];
        evolve_with_callback(sim, k, T_FINAL, u64::MAX, |_| {})
            .expect("the monolithic run went inadmissible");
    }
    let mut dec = grid_tiles(counts, magnetic);
    run(&mut dec, counts);

    let mono_den = global_field(&mono, [1, 1], den);
    let removed = mono_den.iter().map(|d| 1.0 - d).fold(0.0_f64, f64::max);
    assert!(
        removed > 1e-3,
        "the sink removed no mass ({removed:e}); the equivalence is vacuous"
    );
    assert!(
        mono[0].0.iteration >= 3,
        "the run took {} steps; the post-drain halo refresh needs several",
        mono[0].0.iteration
    );

    for (name, pick) in [
        ("den", &den as &dyn Fn(&Sim, [isize; 2]) -> f64),
        ("momx", &momx),
        ("momy", &momy),
        ("momz", &momz),
        ("nrg", &nrg),
        ("bcell0", &bcell0),
        ("bcell1", &bcell1),
        ("prim rho", &rho),
        ("prim pre", &pre),
    ] {
        let mv = global_field(&mono, [1, 1], &pick);
        let dv = global_field(&dec, counts, &pick);
        assert!(
            mv.iter().all(|v| v.is_finite()) && dv.iter().all(|v| v.is_finite()),
            "unwritten cells"
        );
        let e = max_err(&mv, &dv);
        assert!(
            e < 1e-11,
            "{counts:?} {magnetic:?} {name} decomposed != mono under a magnetized sink: err {e:e}"
        );
    }
}

#[test]
fn transparent_sink_two_tile_x_cut() {
    assert_decomposed_matches([2, 1], MagneticSpec::None);
}

#[test]
fn transparent_sink_quad_tile() {
    assert_decomposed_matches([2, 2], MagneticSpec::None);
}

#[test]
fn resistive_sink_two_tile_y_cut() {
    assert_decomposed_matches([1, 2], MagneticSpec::Resistive { eta: 0.05 });
}

#[test]
fn resistive_sink_quad_tile() {
    assert_decomposed_matches([2, 2], MagneticSpec::Resistive { eta: 0.05 });
}
