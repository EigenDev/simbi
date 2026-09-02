// =============================================================================
// substrate_rmhd_f32.rs
//
// f32 smoke for the RMHD substrate KernelSet: proves RmhdSubstrateKernelSet3D is
// generic over the trailing Sc and runs end-to-end at single precision. mirrors
// the f64 make_sim init (uniform div-free B + smooth periodic hydro) but with f32
// fields, then drives the full per-step kernel chain (c2p -> ghost_fill -> flux
// per dir -> cfl -> snapshot -> godunov_euler -> post_godunov) and asserts every
// resulting prim / cons field and the cfl dt are finite over the interior.
//
// this is not a physics-accuracy test (no parity vs f64); it only proves the f32
// instantiation type-checks and the substrate kernels execute without NaN/inf.
// =============================================================================

use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet3D;
use symbi::sim::evolve::KernelSet;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_grid::Field;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Rmhd, 3, Cartesian, IdealGas<f32>, CpuSpace, HostMemory, f32>;

const GAMMA: f32 = 5.0 / 3.0;
const CFL: f32 = 0.4;
// uniform B (div-free, trivially staggered) — exercises the magnetic terms.
const B0: [f32; 3] = [0.3, 0.4, 0.2];

fn make_sim() -> Sim {
    let n = 8usize;
    let dx = 1.0 / n as f64;
    let pi = std::f32::consts::PI;
    let amp = 0.1_f32;
    // smooth periodic hydro (|v| small) + uniform staggered B0; the prim -> cons forward map and
    // the cell-centered B are folded in by set_initial, the staggered faces by seed_faces_uniform.
    Sim::build(Rmhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([n, n, n])
        .spacing([dx, dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(CFL as f64)
        .allocate()
        .expect("sim construction failed")
        .set_initial(|[x, y, z]| {
            let (x, y, z) = (x as f32, y as f32, z as f32);
            let rho = 1.0 + amp * (2.0 * pi * x).sin();
            let vx = 0.1 * (2.0 * pi * y).cos();
            let vy = 0.1 * (2.0 * pi * z).sin();
            let vz = 0.05 * (2.0 * pi * x).cos();
            let p = 1.0 + amp * (2.0 * pi * y).sin();
            MhdPrim::new(
                Prim::adiabatic(Density(rho), Tensor::new([vx, vy, vz]), Pressure(p)),
                Tensor::new(B0),
            )
        })
        .seed_faces_uniform(B0)
        .build()
}

// assert every value over the interior is finite.
fn assert_finite(f: &Field<f32, 3, HostMemory>, interior: &[[isize; 3]], what: &str) {
    for c in interior {
        let v = *f.view().at(*c);
        assert!(v.is_finite(), "{what} cell {c:?} not finite: {v}");
    }
}

#[test]
fn substrate_rmhd_f32_smoke() {
    let sim = make_sim();
    let sub =
        RmhdSubstrateKernelSet3D::<HostMemory, f32>::new(5.0 / 3.0, 0.4, 1.0, &sim.geom.allocated);
    let interior: Vec<[isize; 3]> = sim.geom.interior.iter().collect();
    let pre = sim.fields.prim.pre_field().expect("prim.pre");
    let cnrg = sim.fields.cons.nrg_field().expect("cons.nrg");

    // full per-step kernel chain at f32. proves the substrate set runs end-to-end.
    sub.c2p(&sim);
    assert_finite(&sim.fields.prim.rho, &interior, "c2p rho");
    assert_finite(&sim.fields.prim.vel[0], &interior, "c2p v0");
    assert_finite(&sim.fields.prim.vel[1], &interior, "c2p v1");
    assert_finite(&sim.fields.prim.vel[2], &interior, "c2p v2");
    assert_finite(pre, &interior, "c2p pre");

    sub.ghost_fill(&sim);

    for dir in 0..3 {
        sub.flux(&sim, dir);
        let fnrg = sim.fields.flux[dir].nrg_field().expect("flux.nrg");
        assert_finite(&sim.fields.flux[dir].den, &interior, "flux den");
        assert_finite(fnrg, &interior, "flux nrg");
    }

    let dt = sub.cfl(&sim);
    assert!(
        dt > 0.0 && dt.is_finite(),
        "cfl dt not finite/positive: {dt}"
    );

    sub.snapshot(&sim);
    sub.godunov_stage(&sim, dt, 0.0, 1.0);
    assert_finite(&sim.fields.cons.den, &interior, "godunov den");
    assert_finite(cnrg, &interior, "godunov nrg");

    sub.post_godunov(&sim, dt, 0);
    let mhd = sim.fields.mhd.as_ref().unwrap();
    assert_finite(&mhd.bcell[0], &interior, "ct bcell0");
    assert_finite(&mhd.bcell[1], &interior, "ct bcell1");
    assert_finite(&mhd.bcell[2], &interior, "ct bcell2");
    assert_finite(cnrg, &interior, "ct cons.nrg");

    // c2p the updated state — proves the round-trip stays finite.
    sub.c2p(&sim);
    assert_finite(&sim.fields.prim.rho, &interior, "post c2p rho");
    assert_finite(pre, &interior, "post c2p pre");
}
