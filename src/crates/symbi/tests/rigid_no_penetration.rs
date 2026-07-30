// =============================================================================
// rigid_no_penetration.rs
//
// the rigid-wall contract, live: uniform flow past a rigid sphere whose
// drain-off porous surface (porosity 0, stiff normal channel) must enforce
// no-penetration — the wall-normal gas velocity in the surface band collapses
// far below the free stream, while the same band in a body-free run carries
// the full free-stream normal component.
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, SurfaceSpec};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.3;
const N: usize = 64;
const L: f64 = 1.0;
const DX: f64 = 2.0 * L / N as f64;
const R_BODY: f64 = 0.25;
const V_INF: f64 = 0.3;
const T_FINAL: f64 = 1.0;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

fn build(with_body: bool) -> Sim {
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([-L, -L])
        .spacing([DX; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim")
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([V_INF, 0.0]),
            pre: 1.0,
        })
        .build();
    if !with_body {
        return sim;
    }
    sim.with_bodies(
        BodyCollection::new().add(
            Body::rigid_sphere(
                0,
                Tensor::new([0.0, 0.0]),
                Tensor::new([0.0, 0.0]),
                1.0,
                R_BODY,
                0.1,
                false, // free slip: the no-penetration (normal) channel alone is under test
            )
            .with_surface(SurfaceSpec::Porous {
                porosity: 0.0,
                k_eta_n: 1.0e3,
                k_eta_t: 0.0,
            }),
        ),
    )
}

// the max wall-normal speed |v . n| over the band of cells within one cell width
// of the body surface (n = the outward radial unit vector of the sphere).
fn band_normal_speed(sim: &Sim) -> f64 {
    let ilo: [isize; 2] = std::array::from_fn(|a| sim.geom.interior.spaces[a].lo);
    let mut vmax = 0.0_f64;
    for c in sim.geom.interior.iter() {
        let x = -L + ((c[0] - ilo[0]) as f64 + 0.5) * DX;
        let y = -L + ((c[1] - ilo[1]) as f64 + 0.5) * DX;
        let r = (x * x + y * y).sqrt();
        if (r - R_BODY).abs() > DX {
            continue;
        }
        let vx = *sim.fields.prim.vel[0].view().at(c);
        let vy = *sim.fields.prim.vel[1].view().at(c);
        vmax = vmax.max(((vx * x + vy * y) / r.max(1e-30)).abs());
    }
    vmax
}

#[test]
fn rigid_sphere_enforces_no_penetration() {
    let mut with = build(true);
    let kw = Kern::new(GAMMA, CFL, &with.geom.allocated);
    evolve(&mut with, &kw, T_FINAL).expect("rigid-body run");

    let mut without = build(false);
    let ko = Kern::new(GAMMA, CFL, &without.geom.allocated);
    evolve(&mut without, &ko, T_FINAL).expect("free-stream run");

    let vn_wall = band_normal_speed(&with);
    let vn_free = band_normal_speed(&without);
    // the free stream carries its full normal component through the band...
    assert!(
        vn_free > 0.5 * V_INF,
        "free-stream normal speed unexpectedly small ({vn_free:e}); the band probe is broken"
    );
    // ...and the rigid wall suppresses it by an order of magnitude or more.
    assert!(
        vn_wall < 0.1 * vn_free,
        "no-penetration violated: wall-band |v.n| = {vn_wall:e} vs free-stream {vn_free:e}"
    );
}
