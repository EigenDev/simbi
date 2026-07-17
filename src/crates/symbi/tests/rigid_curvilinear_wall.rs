// =============================================================================
// rigid_curvilinear_wall.rs
//
// shaped rigid walls on the cylindrical (R, phi) chart, through the production
// dispatch. gas momentum is stored in PHYSICAL (orthonormal) components whose
// basis rotates with phi, while the body lives in the cartesian world frame —
// the penalization must bridge the frames per cell. gates:
// - NO-PENETRATION: uniform cartesian flow past a shaped drain-off porous wall
//   suppresses the wall-normal speed in the surface band by an order of
//   magnitude versus the body-free run (the same contract the cartesian
//   rigid-sphere gate enforces);
// - RECEIPT FRAME: the body force receipt is a cartesian world vector. a wall
//   at phi = pi/2 in an x-directed stream absorbs x-momentum, so the receipt
//   points along +x with the y component cancelling by symmetry — summing the
//   raw physical-frame components instead would rotate the receipt into the
//   local (r, phi) basis and swap the axes at that position;
// - TARGET FRAME: a wall translating through still gas drags the gas along its
//   own (cartesian) velocity. at phi = pi/2 an unrotated velocity target would
//   read the x-velocity as a RADIAL (locally y-directed) target and push the
//   gas outward instead of along x.
// =============================================================================

use std::f64::consts::PI;

use symbi::regimes::substrate_kernels::dispatch_penalize;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::CylindricalRPhi;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_ib::sdf::SdfExpr;
use symbi_ib::{Body, BodyCollection, SurfaceSpec};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.3;
const NR: usize = 48;
const NP: usize = 96;
const R_LO: f64 = 1.0;
const R_HI: f64 = 3.0;
const DR: f64 = (R_HI - R_LO) / NR as f64;
const DP: f64 = 2.0 * PI / NP as f64;
const R_BODY: f64 = 0.35;
const V_INF: f64 = 0.3;
// the body center in cartesian: phi = pi/2 at physical radius 2, where the
// local (r, phi) basis is rotated a quarter turn from (x, y) — the position
// that maximally distinguishes world-frame from local-frame vectors.
const BODY_X: f64 = 0.0;
const BODY_Y: f64 = 2.0;

type Sim = SimState<Newtonian, 2, CylindricalRPhi, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

// a uniform cartesian x-directed stream in the local physical basis:
// xhat = cos(phi) rhat - sin(phi) phihat.
fn stream_prim(phi: f64, v: f64) -> Prim<f64, 2> {
    Prim { rho: 1.0, vel: Tensor::new([v * phi.cos(), -v * phi.sin()]), pre: 1.0 }
}

fn build(vel_x: f64, with_body: bool, body_vel: [f64; 2], surface: SurfaceSpec) -> Sim {
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, CylindricalRPhi)
        .cells([NR, NP])
        .origin([R_LO, 0.0])
        .spacing([DR, DP])
        .boundaries(Boundaries(std::array::from_fn(|a| {
            if a == 1 { [BoundaryType::Periodic; 2] } else { [BoundaryType::Outflow; 2] }
        })))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("cyl sim")
        .set_initial(|x| stream_prim(x[1], vel_x))
        .build();
    if !with_body {
        return sim;
    }
    let mut sim = sim.with_bodies(BodyCollection::new().add(
        Body::rigid_sphere(
            0,
            Tensor::new([BODY_X, BODY_Y]),
            Tensor::new(body_vel),
            1.0,
            R_BODY,
            0.1,
            false,
        )
        .with_surface(surface),
    ));
    // the CSG shape (a sphere expressed as a shape, body-local) routes the
    // runtime-JIT shaped kernel instead of the AOT sphere path.
    sim.immersed.as_mut().unwrap().shapes[0] =
        Some(SdfExpr::<f64, 3>::sphere([0.0, 0.0, 0.0], R_BODY));
    sim
}

// the max wall-normal speed |v_cart . n| over cells within one radial cell
// width of the body surface, with the velocity rotated from the local physical
// basis to cartesian and n the outward normal of the sphere.
fn band_normal_speed(sim: &Sim) -> f64 {
    let ilo: [isize; 2] = std::array::from_fn(|a| sim.geom.interior.spaces[a].lo);
    let mut vmax = 0.0_f64;
    for c in sim.geom.interior.iter() {
        let r = R_LO + ((c[0] - ilo[0]) as f64 + 0.5) * DR;
        let phi = ((c[1] - ilo[1]) as f64 + 0.5) * DP;
        let (x, y) = (r * phi.cos(), r * phi.sin());
        let (dx_b, dy_b) = (x - BODY_X, y - BODY_Y);
        let dist = (dx_b * dx_b + dy_b * dy_b).sqrt();
        if (dist - R_BODY).abs() > DR {
            continue;
        }
        let vr = *sim.fields.prim.vel[0].view().at(c);
        let vp = *sim.fields.prim.vel[1].view().at(c);
        let vx = vr * phi.cos() - vp * phi.sin();
        let vy = vr * phi.sin() + vp * phi.cos();
        vmax = vmax.max(((vx * dx_b + vy * dy_b) / dist.max(1e-30)).abs());
    }
    vmax
}

#[test]
fn shaped_wall_enforces_no_penetration_on_the_cylindrical_chart() {
    let no_pen = SurfaceSpec::Porous { porosity: 0.0, k_eta_n: 1.0e3, k_eta_t: 0.0 };
    let mut with = build(V_INF, true, [0.0, 0.0], no_pen);
    let kw = Kern::new(GAMMA, CFL, &with.geom.allocated);
    evolve(&mut with, &kw, 1.0).expect("shaped-wall run");

    let mut without = build(V_INF, false, [0.0, 0.0], SurfaceSpec::Porous { porosity: 0.0, k_eta_n: 0.0, k_eta_t: 0.0 });
    let ko = Kern::new(GAMMA, CFL, &without.geom.allocated);
    evolve(&mut without, &ko, 1.0).expect("free-stream run");

    let vn_wall = band_normal_speed(&with);
    let vn_free = band_normal_speed(&without);
    assert!(
        vn_free > 0.5 * V_INF,
        "free-stream normal speed unexpectedly small ({vn_free:e}); the band probe is broken"
    );
    assert!(
        vn_wall < 0.1 * vn_free,
        "no-penetration violated on the cylindrical chart: wall-band |v.n| = {vn_wall:e} \
         vs free-stream {vn_free:e}"
    );
}

#[test]
fn force_receipt_is_a_cartesian_world_vector() {
    // one penalization of the x-directed stream: the sealed wall absorbs
    // x-momentum, and by the mirror symmetry of the mask about phi = pi/2 the
    // y receipt cancels. the raw local-frame sum would land the receipt in the
    // (r, phi) slots — at this position mostly slot 1, with slot 0 near zero.
    let sealed = SurfaceSpec::Porous { porosity: 0.0, k_eta_n: 50.0, k_eta_t: 50.0 };
    let sim = build(V_INF, true, [0.0, 0.0], sealed);
    dispatch_penalize(&sim, 1e-3, GAMMA, 1.0);
    let d = sim.immersed.as_ref().unwrap().diagnostics.consolidate();
    let f = d[0].force_delta;
    assert!(f[0] > 0.0, "an x-stream must push the wall along +x: {f:?}");
    assert!(
        f[0] > 3.0 * f[1].abs(),
        "the force receipt is not a world-frame vector (axes swapped at phi = pi/2): {f:?}"
    );
}

#[test]
fn moving_wall_target_is_rotated_into_the_local_frame() {
    // a sealed no-slip wall translating along +x through STILL gas drags the
    // gas toward +x: the gas gains x-momentum, so the receipt on the body
    // points along -x. an unrotated velocity target reads the x-velocity as a
    // radial (locally +y) push at phi = pi/2 and the receipt lands in -y.
    let sealed = SurfaceSpec::Porous { porosity: 0.0, k_eta_n: 50.0, k_eta_t: 50.0 };
    let sim = build(0.0, true, [V_INF, 0.0], sealed);
    dispatch_penalize(&sim, 1e-3, GAMMA, 1.0);
    let d = sim.immersed.as_ref().unwrap().diagnostics.consolidate();
    let f = d[0].force_delta;
    assert!(f[0] < 0.0, "dragging still gas along +x must react along -x: {f:?}");
    assert!(
        f[0].abs() > 3.0 * f[1].abs(),
        "the moving-wall velocity target is not frame-rotated (push landed on y): {f:?}"
    );
}
