// =============================================================================
// decomp_mhd_wall_ca2.rs
//
// the Alfven-stiffness (c_a2) decomposition contract for an MHD immersed wall: the
// wall relaxation is lifted to the fast-magnetosonic speed via
// c_a2 = max_interior |B|^2/rho, and that max is a GLOBAL property of the domain.
// under domain decomposition each tile must see the SAME global c_a2, else the same
// wall cell relaxes at a different rate in a tile than in the monolithic run.
//
// the setup isolates c_a2: uniform (div-free) B, so the base MHD decomposition is
// already bit-equivalent (decomp_mhd_equivalence), plus a sharp DENSITY DIP in the
// far (right) tile that drives c_a2 = |B|^2/rho to its global max THERE -- away from
// the sealed shaped wall on the tile cut. a per-tile c_a2 makes the near-wall tile
// use its small local max while the monolithic run uses the far dip's global max, so
// the wall penalization diverges at the cut. with a correct global c_a2 the
// decomposed run reproduces the monolithic one to round-off.
// =============================================================================

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::decomp::{evolve_decomposed, flatten, unflatten, LocalCopy};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::state::Prim;
use symbi_ib::sdf::SdfExpr;
use symbi_ib::{Body, BodyCollection, SurfaceSpec};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const N: usize = 32;
const DX: f64 = 1.0 / N as f64;
const T_FINAL: f64 = 0.02;
const B0: [f64; 3] = [1.0, 0.5, 0.3]; // uniform, |B|^2 = 1.34
const V_INF: f64 = 0.3;
const R_BODY: f64 = 0.2;

type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;
type Kern = NewtonianMhdSubstrateKernelSet<HostMemory, f64, 2>;

// a sharp density dip centered at (0.75, 0.5) -- the RIGHT half -- so |B|^2/rho peaks there
// (rho -> 0.1, c_a2 -> ~13.4), far from the wall on the x = 0.5 cut where rho ~ 1 (c_a2 ~ 1.34).
fn ic(x: f64, y: f64) -> MhdPrim<f64, 3> {
    let r2 = (x - 0.75).powi(2) + (y - 0.5).powi(2);
    let dip = 0.9 * (-(r2 / 0.01)).exp();
    MhdPrim {
        hydro: Prim { rho: 1.0 - dip, vel: Tensor::new([V_INF, 0.0, 0.0]), pre: 1.0 },
        mag: Tensor::new(B0),
    }
}

fn make(cells: [usize; 2], origin: [f64; 2], bnd: Boundaries<2>) -> (Sim, Kern) {
    let mut sim = Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 2])
        .origin(origin)
        .boundaries(bnd)
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("mhd sim")
        .set_initial(|[x, y]| ic(x, y))
        .seed_faces_uniform([B0[0], B0[1]])
        .build()
        .with_bodies(BodyCollection::new().add(
            // a sealed shaped wall on the x = 0.5 cut; moderate k_eta so the c_a2-dependent
            // relaxation rate (not a saturated stiff limit) drives the near-wall momentum.
            Body::rigid_sphere(0, Tensor::new([0.5, 0.5]), Tensor::zeros(), 1.0, R_BODY, 1.0, true)
                .with_surface(SurfaceSpec::Porous { porosity: 0.0, k_eta_n: 20.0, k_eta_t: 20.0 }),
        ));
    sim.immersed.as_mut().unwrap().shapes[0] = Some(SdfExpr::<f64, 3>::sphere([0.0, 0.0, 0.0], R_BODY));
    let k = Kern::new(GAMMA, CFL, 1.0, &sim.geom.allocated);
    (sim, k)
}

fn grid_tiles(counts: [usize; 2]) -> Vec<(Sim, Kern)> {
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
                let lo = if tc[a] == 0 { BoundaryType::Outflow } else { BoundaryType::CoarseFine };
                let hi = if tc[a] == counts[a] - 1 { BoundaryType::Outflow } else { BoundaryType::CoarseFine };
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
        &mut stores, &kernels, counts, &devices, Timestepping::Rk2, 0.0, T_FINAL, u64::MAX, &LocalCopy,
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

fn assert_decomposed_matches(counts: [usize; 2]) {
    let den = |s: &Sim, c| *s.fields.cons.den.view().at(c);
    let momx = |s: &Sim, c| *s.fields.cons.mom[0].view().at(c);
    let momy = |s: &Sim, c| *s.fields.cons.mom[1].view().at(c);
    let bcell0 = |s: &Sim, c| *s.fields.mhd.as_ref().unwrap().bcell[0].view().at(c);

    let mut mono = grid_tiles([1, 1]);
    run(&mut mono, [1, 1]);
    let mut dec = grid_tiles(counts);
    run(&mut dec, counts);

    for (name, pick) in [
        ("den", &den as &dyn Fn(&Sim, [isize; 2]) -> f64),
        ("momx", &momx),
        ("momy", &momy),
        ("bcell0", &bcell0),
    ] {
        let mv = global_field(&mono, [1, 1], &pick);
        let dv = global_field(&dec, counts, &pick);
        assert!(mv.iter().all(|v| v.is_finite()) && dv.iter().all(|v| v.is_finite()), "unwritten cells");
        let e = max_err(&mv, &dv);
        assert!(e < 1e-11, "{counts:?} {name} decomposed!=mono under mhd wall (c_a2 divergence): err {e:e}");
    }
}

// the x-cut splits the near-wall tile from the far density dip: the near tile's local c_a2 is
// small, the monolithic global c_a2 is the dip's large value -> the wall on the cut diverges
// unless c_a2 is reduced globally.
#[test]
fn mhd_wall_ca2_two_tile_x_cut() {
    assert_decomposed_matches([2, 1]);
}

#[test]
fn mhd_wall_ca2_quad_tile() {
    assert_decomposed_matches([2, 2]);
}
