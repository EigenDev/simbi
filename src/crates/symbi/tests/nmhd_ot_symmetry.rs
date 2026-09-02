// diagnostic: the canonical Newtonian Orszag-Tang vortex has a 180-degree point
// symmetry about the domain center — v -> -v, B -> -B, rho/p invariant — so the
// scalar fields obey rho(x,y,t) = rho(1-x,1-y,t) for all t. on the grid this is the
// exact cell map i -> N-1-i (no interpolation). the relative-L1 of that reflection
// is the discriminator the bisection-in-time argument needs:
//   ~roundoff at t=0.3, finite at t=1.0  => physical instability seeded by roundoff.
//   already finite at t=0.3              => a scheme asymmetry (sweep order / CT EMF).
// run: cargo test -p symbi --release --test nmhd_ot_symmetry -- --ignored --nocapture

use std::f64::consts::PI;
use symbi_hydro::quantity::{Density, Pressure};

use symbi::regimes::substrate_kernels::Solver;
use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet3D;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<NewtonianMhd, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

const NX: usize = 256;
const NY: usize = 256;
const GAMMA: f64 = 5.0 / 3.0;
const V0: f64 = 1.0;

fn make_sim() -> Sim {
    let dx = 1.0 / NX as f64;
    let dy = 1.0 / NY as f64;
    let rho0 = 25.0 / (36.0 * PI);
    let p0 = 5.0 / (12.0 * PI);
    let b0 = 1.0 / (4.0 * PI).sqrt();
    // staggered B via face_coord() (in seed_faces): Bx on the x-face is exact in x,
    // cell-centered in y/z. the accessor owns the half-cell offset that this whole test
    // exists to police.
    Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([NX, NY, 1])
        .spacing([dx, dy, 1.0])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .allocate()
        .unwrap()
        .set_initial(|[x, y, _z]| {
            let vx = -V0 * (2.0 * PI * y).sin();
            let vy = V0 * (2.0 * PI * x).sin();
            let bx = -b0 * (2.0 * PI * y).sin();
            let by = b0 * (4.0 * PI * x).sin();
            MhdPrim::new(
                Prim::adiabatic(Density(rho0), Tensor::new([vx, vy, 0.0]), Pressure(p0)),
                Tensor::new([bx, by, 0.0]),
            )
        })
        .seed_faces(|axis, [x, y, _z]| match axis {
            0 => -b0 * (2.0 * PI * y).sin(),
            1 => b0 * (4.0 * PI * x).sin(),
            _ => 0.0,
        })
        .build()
}

// relative L1 of the center point-reflection on density: mean|rho(i,j) - rho(N-1-i,N-1-j)| / mean(rho).
fn reflection_l1(sim: &Sim) -> f64 {
    let interior = &sim.geom.interior;
    let nx = interior.spaces[0].size() as usize;
    let ny = interior.spaces[1].size() as usize;
    let i0 = interior.spaces[0].lo as i64;
    let j0 = interior.spaces[1].lo as i64;
    let rho = sim.fields.prim.rho.view();
    let mut grid = vec![0.0f64; nx * ny];
    for c in interior.iter() {
        let li = (c[0] as i64 - i0) as usize;
        let lj = (c[1] as i64 - j0) as usize;
        grid[li * ny + lj] = *rho.at(c);
    }
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for li in 0..nx {
        for lj in 0..ny {
            let r = grid[li * ny + lj];
            let rr = grid[(nx - 1 - li) * ny + (ny - 1 - lj)];
            num += (r - rr).abs();
            den += r;
        }
    }
    num / den // mean-normalized (the 1/N cancels)
}

#[test]
#[ignore = "diagnostic: ~60s at 256^2; run explicitly with --ignored"]
fn nmhd_ot_reflection_symmetry_t03_vs_t10() {
    let mut sim = make_sim();
    let sub =
        NewtonianMhdSubstrateKernelSet3D::<HostMemory>::new(GAMMA, 0.4, 1.0, &sim.geom.allocated)
            .with_solver(Solver::Hlld)
            .expect("valid solver/regime pair");

    eprintln!("[ot-sym] {NX}x{NY} HLLD — center point-reflection relative-L1 of density:");
    eprintln!("[ot-sym]   t=0     : 0 (uniform IC; prim not yet recovered, so not measured)");

    evolve_with_callback(&mut sim, &sub, 0.3, 1, |_| {}).ok();
    let s03 = reflection_l1(&sim);
    eprintln!("[ot-sym]   t=0.3   : {s03:.3e}  (iter {})", sim.iteration);

    evolve_with_callback(&mut sim, &sub, 1.0, 1, |_| {}).ok();
    let s10 = reflection_l1(&sim);
    eprintln!("[ot-sym]   t=1.0   : {s10:.3e}  (iter {})", sim.iteration);

    // the scheme preserves the OT point symmetry: with a correctly-sampled (face-midpoint)
    // IC, any asymmetry at t=0.3 is roundoff. a directional bias in the Riemann solver / CT EMF
    // / the bface coupling would show up here as a finite (>> roundoff) value. late growth to
    // ~1e-10 is the physical tearing/kh instability amplifying roundoff — expected.
    assert!(
        s03 < 1e-10,
        "scheme broke OT point symmetry at t=0.3: rel-L1 {s03:.3e} >> roundoff -> directional bias",
    );
}
