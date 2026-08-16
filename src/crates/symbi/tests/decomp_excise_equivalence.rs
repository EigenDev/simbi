// =============================================================================
// decomp_excise_equivalence.rs
//
// the tiling-invariance contract for horizon excision on the decomposed path:
// a cartesian kerr-schild box with the black hole at the origin, excised, must
// evolve identically whether it runs as one tile or as a 2x2 grid whose cuts
// pass THROUGH the excised sphere. the fill is pointwise — every cell inside
// the excision radius is frozen at the cold vacuum floor from its own state —
// so tiling invariance rests on each tile classifying its own cells against the
// same global level set, and on the rim gas, which rarefies INTO the vacuum,
// reading exchange-fresh halos. the uniform infall
// develops real dynamics (the KS chart accretes from rest), so the excised
// rim, the fill, and the conserved rebuild are all genuinely exercised.
// =============================================================================

use symbi::regimes::substrate_rhd::RhdSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, evolve_decomposed, flatten, unflatten};
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
const MASS: f64 = 0.3; // r_+ = 0.6, well inside the box
const R_EXC: f64 = 0.35; // inside r_+, above the metric guard M/2 = 0.15
const T_FINAL: f64 = 0.2;

type Sim = SimState<Rhd, 2, SchwarzschildKSCartesian<f64>, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = RhdSubstrateKernelSet<HostMemory, f64, 2>;

fn make(cells: [usize; 2], origin: [f64; 2], bnd: Boundaries<2>) -> (Sim, Kern) {
    let sim = Sim::build(
        Rhd,
        IdealGas { gamma: GAMMA },
        SchwarzschildKSCartesian { mass: MASS },
    )
    .cells(cells)
    .spacing([DX; 2])
    .origin(origin)
    .boundaries(bnd)
    .timestepping(Timestepping::Rk2)
    .allocate()
    .expect("sim construction failed")
    .set_initial(|_| Prim {
        rho: 1.0,
        vel: Tensor::new([0.0; 2]),
        pre: 0.1,
    })
    .build();
    let k = Kern::new(GAMMA, CFL, &sim.geom.allocated).with_excision(R_EXC, 1.0, 1.0);
    (sim, k)
}

// the tile grid over the origin-centered box: the 2x2 cut point lands exactly on the
// chart origin, so every tile owns one quadrant of the excised sphere and the
// excised rim straddles all four cuts.
fn grid_tiles(counts: [usize; 2]) -> Vec<(Sim, Kern)> {
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
            make(m, origin, bnd)
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
        |_, _, stores| {
            // dt-collapse tripwire: the vacuum-sink rim can drive dt into the
            // floor on cold configurations; a run whose dt has collapsed fails
            // loud here instead of spinning to the time limit.
            assert!(
                stores[0].dt > 1.0e-9,
                "dt collapsed to {:.3e} at t = {:.6e}",
                stores[0].dt,
                stores[0].time,
            );
            std::ops::ControlFlow::Continue(())
        },
    );
}

// scatter every tile's interior (den, nrg) into global grids: den is the accretion
// canary; nrg carries the excised cells' rebuilt tau, so a wrong fill or a stale
// halo at a cut shows up in either.
fn global_fields(tiles: &[(Sim, Kern)], counts: [usize; 2]) -> (Vec<f64>, Vec<f64>) {
    let m: [usize; 2] = std::array::from_fn(|a| N / counts[a]);
    let mut den = vec![f64::NAN; N * N];
    let mut nrg = vec![f64::NAN; N * N];
    for (flat_tile, (sim, _)) in tiles.iter().enumerate() {
        let tc = unflatten(flat_tile, counts);
        let ilo: [isize; 2] = std::array::from_fn(|a| sim.geom.interior.spaces[a].lo);
        let nv = sim.fields.cons.nrg_field().expect("GR cons.nrg");
        for c in sim.geom.interior.iter() {
            let g: [usize; 2] = std::array::from_fn(|a| tc[a] * m[a] + (c[a] - ilo[a]) as usize);
            den[flatten(g, [N; 2])] = *sim.fields.cons.den.view().at(c);
            nrg[flatten(g, [N; 2])] = *nv.view().at(c);
        }
    }
    (den, nrg)
}

fn assert_matches(counts: [usize; 2]) {
    let mut mono = grid_tiles([1, 1]);
    run(&mut mono, [1, 1]);
    let (mden, mnrg) = global_fields(&mono, [1, 1]);

    let mut dec = grid_tiles(counts);
    run(&mut dec, counts);
    let (dden, dnrg) = global_fields(&dec, counts);

    assert!(
        mden.iter().all(|v| v.is_finite()) && dden.iter().all(|v| v.is_finite()),
        "some global cells were never written (gather bug)"
    );
    // non-vacuous: the infall genuinely developed and the excision genuinely acted
    // (the deep excised cells hold rebuilt, non-initial values).
    let den_max = mden.iter().cloned().fold(0.0_f64, f64::max);
    assert!(
        den_max > 1.05,
        "no accretion developed (max den {den_max:.3})"
    );

    for (name, a, b) in [("den", &mden, &dden), ("nrg", &mnrg, &dnrg)] {
        let max_err = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err < 1e-12,
            "excised decomposition {counts:?} vs monolithic {name} max err {max_err:e}"
        );
    }
}

#[test]
fn excised_two_tile_cut_through_the_sphere() {
    assert_matches([2, 1]);
}

#[test]
fn excised_2x2_grid_origin_on_the_corner() {
    assert_matches([2, 2]);
}
