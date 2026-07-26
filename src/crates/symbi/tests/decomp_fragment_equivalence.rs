// =============================================================================
// decomp_fragment_equivalence.rs
//
// bonded rigid fragments (a rubble-pile cluster) under multi-gpu domain
// decomposition. a fragment feels a fluid drag force booked by the penalization
// over ITS cells; the bonded DEM subcycle (bonds + contact + gravity + drag) then
// moves the cluster. when a bond spans a tile cut, one fragment's cells sit in one
// tile and the other's in the neighbor, so each tile books only its own fragment's
// drag -- the decomposed body step must SUM those per-fragment loads across tiles
// and run the fragment subcycle on the total, replicated identically on every tile.
//
// the contract: decomposed fragment trajectories == monolithic to round-off. the
// gate is self-non-vacuous: a decomposed step WITHOUT the fragment subcycle leaves
// the fragments frozen at their seed while the monolithic cluster drifts downstream,
// so the position comparison fails hard. a bond spanning the cut is asserted so the
// cross-tile reduction is genuinely exercised.
// =============================================================================

use symbi::regimes::substrate_gpu::device_sync;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, evolve_decomposed, unflatten};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, Bond, BondMaterial, FragmentPhysics, SurfaceSpec};
use symbi_xpu::{CpuSpace, HostMemory, with_device};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const N: usize = 48;
const L: f64 = 1.0; // domain [-1, 1]^2
const DX: f64 = 2.0 * L / N as f64;
const WIND: f64 = 0.5;
const FRAG_MASS: f64 = 0.6;
const FRAG_RADIUS: f64 = 0.15;
const T_FINAL: f64 = 0.12;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

fn fragment(x: f64, y: f64) -> Body<f64, 2> {
    Body::rigid_sphere(
        0,
        Tensor::new([x, y]),
        Tensor::zeros(),
        FRAG_MASS,
        FRAG_RADIUS,
        1e-3,
        true,
    )
    .with_surface(SurfaceSpec::Porous {
        porosity: 0.0,
        k_eta_n: 50.0,
        k_eta_t: 50.0,
    })
    .with_two_way_coupling(true)
}

// a bonded pair straddling y = 0: fragment 0 at y = -0.35, fragment 1 at y = +0.35. each fragment
// (radius 0.15) lies entirely on its side of the cut, but the BOND crosses it. a FRESH collection +
// bonds per tile (with_bodies / attach_fragment_physics take ownership), identical everywhere.
fn attach_fragments(sim: Sim) -> Sim {
    let coll = BodyCollection::new()
        .add_fragment(fragment(0.0, -0.35))
        .add_fragment(fragment(0.0, 0.35));
    let mat = BondMaterial {
        k_n: 50.0,
        gamma: 0.5,
        ..BondMaterial::rigid()
    };
    let bonds = vec![Bond::form(0, 1, coll.get(0), coll.get(1), mat)];
    let mut sim = sim.with_bodies(coll);
    sim.attach_fragment_physics(FragmentPhysics {
        bonds,
        contacts: None,
        gravity: None,
    });
    sim
}

fn make(cells: [usize; 2], origin: [f64; 2], bnd: Boundaries<2>, ts: Timestepping) -> (Sim, Kern) {
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 2])
        .origin(origin)
        .boundaries(bnd)
        .cfl(CFL)
        .timestepping(ts)
        .allocate()
        .expect("sim construction failed")
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([WIND, 0.0]),
            pre: 1.0,
        })
        .build();
    let sim = attach_fragments(sim);
    let k = Kern::new(GAMMA, CFL, &sim.geom.allocated);
    (sim, k)
}

fn grid_tiles(counts: [usize; 2], ts: Timestepping) -> Vec<(Sim, Kern)> {
    let m: [usize; 2] = std::array::from_fn(|a| N / counts[a]);
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

// the two fragments' (position, velocity) read from any tile (bodies are replicated in lockstep).
fn fragment_state(tiles: &[(Sim, Kern)]) -> [([f64; 2], [f64; 2]); 2] {
    for dd in 0..2 {
        with_device(dd, || device_sync::<HostMemory>());
    }
    let im = tiles[0].0.immersed.as_ref().expect("fragments attached");
    std::array::from_fn(|i| {
        let b = im.bodies.get(i);
        (
            [b.position[0], b.position[1]],
            [b.velocity[0], b.velocity[1]],
        )
    })
}

fn assert_matches(counts: [usize; 2], ts: Timestepping) {
    let mut mono = grid_tiles([1, 1], ts);
    run(&mut mono, [1, 1], ts);
    let mono_st = fragment_state(&mono);

    let mut dec = grid_tiles(counts, ts);
    run(&mut dec, counts, ts);
    let dec_st = fragment_state(&dec);

    // NON-VACUITY: the wind drag must have moved the cluster downstream, or a frozen-fragment
    // decomposed run would match a frozen monolithic run and the subcycle would be untested.
    let drift = mono_st.iter().map(|(p, _)| p[0]).fold(0.0_f64, f64::max);
    assert!(
        drift > 1e-3,
        "the fragment cluster never drifted (max x = {drift:e}); the subcycle is untested"
    );

    let mut max_err = 0.0_f64;
    for (i, ((mp, mv), (dp, dv))) in mono_st.iter().zip(&dec_st).enumerate() {
        for a in 0..2 {
            max_err = max_err
                .max((mp[a] - dp[a]).abs())
                .max((mv[a] - dv[a]).abs());
        }
        let _ = i;
    }
    assert!(
        max_err < 1e-12,
        "fragment decomposition {counts:?} ({ts:?}) vs monolithic max err {max_err:e} \
         (drifted {drift:e})"
    );
}

// the y-cut is the load-bearing case: the bond spans it, so each fragment's drag is booked by a
// DIFFERENT tile and the decomposed body step must sum them before the subcycle. the 2x2 grid adds
// the transverse x-cut (the wind direction).
#[test]
fn fragment_bond_across_y_cut_rk2() {
    assert_matches([1, 2], Timestepping::Rk2);
}

#[test]
fn fragment_cluster_quad_tile_rk2() {
    assert_matches([2, 2], Timestepping::Rk2);
}
