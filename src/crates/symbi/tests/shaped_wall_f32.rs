// =============================================================================
// shaped_wall_f32.rs
//
// the reduced-precision (f32) runtime-JIT shaped rigid wall on host: the
// generic-precision cranelift codegen must produce f32 results physically close
// to the f64 reference. builds the SAME sealed shaped-sphere wall in a uniform
// stream at f64 and f32, runs one penalization on each, and asserts the f32 cons
// + force receipt track the f64 run at f32 tolerance (the physics is correct at
// reduced precision). this exercises the f32 field
// loads/stores/strides, f32 consts, and the f32 transcendental (tanh) shim.
// =============================================================================

use symbi::regimes::substrate_kernels::dispatch_penalize;
use symbi::sim::state::*;
use symbi_algebra::{OrderedNumeric, Tensor};
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_ib::sdf::SdfExpr;
use symbi_ib::{Body, BodyCollection, SurfaceSpec};
use symbi_ir::algebra::Scalar;
use symbi_xpu::{CpuSpace, HostMemory};

const N: usize = 48;
const L: f64 = 1.0;
const R_BODY: f64 = 0.25;
const V_INF: f64 = 0.3;
const GAMMA: f64 = 1.4;

fn build<Sc: Scalar + OrderedNumeric>(
) -> SimState<Newtonian, 2, Cartesian, IdealGas<Sc>, CpuSpace, HostMemory, Sc> {
    let dx = 2.0 * L / N as f64;
    let s = |v: f64| Sc::from_f64(v);
    let mut sim =
        SimState::<Newtonian, 2, Cartesian, IdealGas<Sc>, CpuSpace, HostMemory, Sc>::build(
            Newtonian,
            IdealGas { gamma: s(GAMMA) },
            Cartesian,
        )
        .cells([N, N])
        .origin([-L, -L])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(0.3)
        .allocate()
        .expect("sim")
        .set_initial(|_| Prim { rho: s(1.0), vel: Tensor::new([s(V_INF), s(0.0)]), pre: s(1.0) })
        .build()
        .with_bodies(BodyCollection::new().add(
            Body::rigid_sphere(0, Tensor::new([0.0, 0.0]), Tensor::zeros(), 1.0, R_BODY, 0.1, false)
                .with_surface(SurfaceSpec::Porous { porosity: 0.0, k_eta_n: 1.0e3, k_eta_t: 0.0 }),
        ));
    // the CSG shape routes the runtime cranelift kernel, built at the sim's precision.
    sim.immersed.as_mut().unwrap().shapes[0] =
        Some(SdfExpr::<f64, 3>::sphere([0.0, 0.0, 0.0], R_BODY));
    sim
}

#[test]
fn f32_shaped_wall_tracks_f64() {
    let h64 = build::<f64>();
    let h32 = build::<f32>();
    dispatch_penalize(&h64, 1e-3, GAMMA, 1.0);
    dispatch_penalize(&h32, 1e-3, GAMMA, 1.0);

    // cons agree at f32 tolerance (the f32 codegen is physically correct).
    let mut gap = 0.0_f64;
    for c in h64.geom.interior.iter() {
        let a = (*h64.fields.cons.den.view().at(c)).to_f64();
        let b = (*h32.fields.cons.den.view().at(c)).to_f64();
        assert!(b.is_finite(), "non-finite f32 den at {c:?}");
        gap = gap.max((a - b).abs() / a.abs().max(1.0));
        for k in 0..2 {
            let a = (*h64.fields.cons.mom[k].view().at(c)).to_f64();
            let b = (*h32.fields.cons.mom[k].view().at(c)).to_f64();
            gap = gap.max((a - b).abs() / a.abs().max(1.0));
        }
    }
    assert!(gap < 1e-4, "f32 shaped-wall cons diverges from f64: rel gap {gap:e}");

    // the wall actually penalized (non-vacuous) and the f32 force receipt tracks the f64 one.
    let f64f = h64.immersed.as_ref().unwrap().diagnostics.consolidate()[0].force_delta;
    let f32f = h32.immersed.as_ref().unwrap().diagnostics.consolidate()[0].force_delta;
    let mag = (f64f[0] * f64f[0] + f64f[1] * f64f[1]).sqrt();
    assert!(mag > 1e-6, "the shaped wall never penalized ({mag:e}); test vacuous");
    for k in 0..2 {
        assert!(f32f[k].is_finite(), "non-finite f32 force[{k}]");
        assert!(
            (f64f[k] - f32f[k]).abs() < 1e-4 * mag + 1e-6,
            "f32 force[{k}] {} diverges from f64 {}",
            f32f[k],
            f64f[k],
        );
    }
}
