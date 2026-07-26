// =============================================================================
// rigid_spin_axis.rs
//
// arbitrary-AXIS rigid spin, end to end: the whole chain — the spec's spin
// axis, the Body's omega vector, the Rodrigues orientation roll, the runtime
// 3x3 mask rotation, and the omega x r wall velocity — is axis-general in the
// code, but nothing exercised a non-z axis. gates:
// - the orientation roll about x maps y -> z (the isotropic torque-free body
//   rolls exactly by Rodrigues about its omega axis);
// - a 3D shaped wall spinning about +x drags the gas into a CIRCULATION about
//   x (nonzero y-z swirl) while the swirl about z stays at roundoff — a
//   z-hardcoded mask or wall velocity would produce exactly the opposite.
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve;
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
const CFL: f64 = 0.3;
const N: usize = 32;
const L: f64 = 1.0;
const DX: f64 = 2.0 * L / N as f64;
const OMEGA: f64 = 2.0;
const T_FINAL: f64 = 0.5;

type Sim = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 3>;

#[test]
fn orientation_rolls_about_the_x_axis() {
    // an isotropic torque-free body spinning about +x: omega is conserved and the
    // orientation is exactly Rodrigues(x, omega t) — y rolls into z.
    let mut b = Body::<f64, 3>::rigid_sphere(
        0,
        Tensor::new([0.0; 3]),
        Tensor::new([0.0; 3]),
        1.0,
        0.3,
        0.1,
        false,
    )
    .with_spin_about(OMEGA, Tensor::new([1.0, 0.0, 0.0]));
    let quarter = std::f64::consts::FRAC_PI_2 / OMEGA;
    let steps = 1000;
    for _ in 0..steps {
        b.advance_rotation(Tensor::new([0.0; 3]), quarter / steps as f64);
    }
    let r = b.orientation;
    // R maps body-frame y to world z (a quarter turn about x): column 1 = (0, 0, 1).
    assert!(
        (r[0][1]).abs() < 1e-9 && (r[1][1]).abs() < 1e-6 && (r[2][1] - 1.0).abs() < 1e-6,
        "quarter turn about x did not map y -> z: column 1 = ({}, {}, {})",
        r[0][1],
        r[1][1],
        r[2][1]
    );
    // omega unchanged (isotropic, torque-free).
    assert!(
        (b.omega[0] - OMEGA).abs() < 1e-12 && b.omega[1].abs() < 1e-12 && b.omega[2].abs() < 1e-12
    );
}

#[test]
fn shaped_wall_spinning_about_x_swirls_the_gas_about_x() {
    // a shaped (SDF sphere) rigid wall spinning about +x in quiescent gas: the
    // omega x r drag must set up a circulation about the X axis. the swirl
    // moments discriminate the axis: L_x = sum rho (y v_z - z v_y) grows, while
    // L_z = sum rho (x v_y - y v_x) stays at roundoff.
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 3])
        .origin([-L; 3])
        .spacing([DX; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim")
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([0.0; 3]),
            pre: 1.0,
        })
        .build();
    let mut sim = sim.with_bodies(
        BodyCollection::new().add(
            Body::rigid_sphere(
                0,
                Tensor::new([0.0; 3]),
                Tensor::new([0.0; 3]),
                1.0,
                0.3,
                0.1,
                true, // no-slip: the tangential drag is the whole point here
            )
            .with_surface(SurfaceSpec::Porous {
                porosity: 0.0,
                k_eta_n: 1.0e3,
                k_eta_t: 1.0e3,
            })
            .with_spin_about(OMEGA, Tensor::new([1.0, 0.0, 0.0])),
        ),
    );
    sim.attach_body_shapes(vec![Some(SdfExpr::sphere([0.0; 3], 0.3))]);
    let k = Kern::new(GAMMA, CFL, &sim.geom.allocated);
    evolve(&mut sim, &k, T_FINAL).expect("spinning-shaped run");

    let ilo: [isize; 3] = std::array::from_fn(|a| sim.geom.interior.spaces[a].lo);
    let (mut lx, mut lz) = (0.0_f64, 0.0_f64);
    for c in sim.geom.interior.iter() {
        let x = -L + ((c[0] - ilo[0]) as f64 + 0.5) * DX;
        let y = -L + ((c[1] - ilo[1]) as f64 + 0.5) * DX;
        let z = -L + ((c[2] - ilo[2]) as f64 + 0.5) * DX;
        let rho = *sim.fields.prim.rho.view().at(c);
        let vx = *sim.fields.prim.vel[0].view().at(c);
        let vy = *sim.fields.prim.vel[1].view().at(c);
        let vz = *sim.fields.prim.vel[2].view().at(c);
        lx += rho * (y * vz - z * vy);
        lz += rho * (x * vy - y * vx);
    }
    assert!(
        lx.abs() > 1e-4,
        "no circulation about the spin axis developed (L_x = {lx:e}); the x-spin never acted"
    );
    assert!(
        lz.abs() < 1e-2 * lx.abs(),
        "spurious z-circulation: L_z = {lz:e} vs L_x = {lx:e} — a z-hardcoded mask or wall velocity"
    );
}
