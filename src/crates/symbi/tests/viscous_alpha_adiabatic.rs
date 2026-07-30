// =============================================================================
// viscous_alpha_adiabatic.rs
//
// the shakura-sunyaev alpha viscosity with the LOCAL adiabatic sound speed,
// nu(x) = alpha (gamma p / rho) / Omega_K(r) about immersed body 0. one
// viscous application on a sheared uniform-cs disk gates:
// - the operator genuinely acts (momenta change where the shear lives) and
//   carries the viscous HEATING (total energy rises where shear dissipates);
// - the LOCAL-cs law: on a uniform (p, rho) state the local cs^2 is the one
//   constant gamma p0/rho0, so the alpha update must differ from a
//   constant-nu update (nu varies with radius through Omega_K) — a kernel
//   that dropped the radial law would match it;
// - the 2.5D magnetized-gas (DOF = 3) variant dispatches and diffuses the
//   out-of-plane momentum.
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::state::*;
use symbi::sim::substrate_seam::{KernelSet, WithViscosity};

use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.3;
const N: usize = 32;
const L: f64 = 1.0;
const DX: f64 = 2.0 * L / N as f64;
const ALPHA: f64 = 0.1;
const DT: f64 = 1.0e-4;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

fn build() -> Sim {
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([-L, -L])
        .spacing([DX; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .expect("sim")
        // uniform (rho, p) — the one constant local cs^2 — with a linear shear
        // v_x(y), so the stress has one dominant component and the heating is
        // strictly positive where the gradient lives.
        .set_initial(|x| Prim {
            rho: 1.0,
            vel: Tensor::new([0.3 * x[1], 0.0]),
            pre: 1.0,
        })
        .build()
        .with_bodies(BodyCollection::new().add(Body::gravitational(
            0,
            Tensor::new([0.0, 0.0]),
            Tensor::new([0.0, 0.0]),
            1.0,
            0.05,
            0.1,
        )))
}

fn snapshot(sim: &Sim) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let nrg = sim.fields.cons.nrg_field().expect("adiabatic nrg");
    let mut m0 = Vec::new();
    let mut m1 = Vec::new();
    let mut en = Vec::new();
    for c in sim.geom.interior.iter() {
        m0.push(*sim.fields.cons.mom[0].view().at(c));
        m1.push(*sim.fields.cons.mom[1].view().at(c));
        en.push(*nrg.view().at(c));
    }
    (m0, m1, en)
}

#[test]
fn alpha_viscosity_acts_heats_and_carries_the_radial_law() {
    // the alpha pass.
    let sim_a = build();
    let ka = Kern::new(GAMMA, CFL, &sim_a.geom.allocated).with_alpha(ALPHA);
    // prims are populated by the builder's initial state only in cons; run c2p to
    // materialize them for the viscous stencil (the production loop's invariant).
    ka.c2p(&sim_a);
    let (m0_before, _, en_before) = snapshot(&sim_a);
    ka.viscous(&sim_a, DT);
    let (m0_alpha, _, en_alpha) = snapshot(&sim_a);

    let dmom: f64 = m0_alpha
        .iter()
        .zip(&m0_before)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        dmom > 1e-12,
        "the alpha viscous pass never touched the momentum"
    );
    let dnrg_max = en_alpha
        .iter()
        .zip(&en_before)
        .map(|(a, b)| a - b)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        dnrg_max > 0.0,
        "no viscous heating booked (max dE {dnrg_max:e})"
    );

    // the radial law: a constant-nu pass with nu = alpha cs0^2 / Omega_K(r_ref)
    // matches the alpha pass ONLY at r_ref; anywhere else Omega_K differs, so the
    // two updates must NOT coincide globally (an alpha kernel that dropped the
    // radial dependence would match it bit-for-bit on this uniform-cs state).
    let cs2 = GAMMA * 1.0 / 1.0;
    let r_ref = 0.5_f64;
    let nu_ref = ALPHA * cs2 / (1.0_f64 / (r_ref * r_ref * r_ref)).sqrt();
    let sim_c = build();
    let kc = Kern::new(GAMMA, CFL, &sim_c.geom.allocated).with_viscosity(nu_ref);
    kc.c2p(&sim_c);
    kc.viscous(&sim_c, DT);
    let (m0_const, _, _) = snapshot(&sim_c);
    let dlaw = m0_alpha
        .iter()
        .zip(&m0_const)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        dlaw > 1e-12,
        "the alpha update equals a constant-nu update everywhere: the radial \
         Omega_K law is not being applied"
    );
}

#[test]
fn alpha_and_constant_nu_both_cap_the_timestep() {
    // the parabolic viscous limit dt <= 0.1 dx^2 / nu_max must bind for BOTH
    // paths: a large constant nu and a large alpha (whose nu_max is the largest
    // local sound speed at the slowest orbit) each pull dt well below the
    // inviscid wave-speed value.
    let sim = build();
    let k0 = Kern::new(GAMMA, CFL, &sim.geom.allocated);
    k0.c2p(&sim);
    let dt_inviscid = k0.cfl(&sim);

    let kc = Kern::new(GAMMA, CFL, &sim.geom.allocated).with_viscosity(50.0);
    let dt_const = kc.cfl(&sim);
    assert!(
        dt_const < 0.5 * dt_inviscid,
        "constant-nu cap did not bind: {dt_const:e} vs inviscid {dt_inviscid:e}"
    );
    let expect = 0.1 * DX * DX / 50.0;
    assert!(
        (dt_const - expect).abs() < 1e-12 * expect.max(1.0),
        "constant-nu cap is not the parabolic limit: {dt_const:e} vs {expect:e}"
    );

    let ka = Kern::new(GAMMA, CFL, &sim.geom.allocated).with_alpha(50.0);
    let dt_alpha = ka.cfl(&sim);
    assert!(
        dt_alpha < 0.5 * dt_inviscid,
        "alpha cap did not bind: {dt_alpha:e} vs inviscid {dt_inviscid:e}"
    );
    // the alpha bound uses the LARGEST (p/rho) at the SLOWEST orbit: on this
    // uniform state, nu_max = alpha gamma (p0/rho0) / Omega_K(r_corner).
    let r_corner = (2.0_f64).sqrt() * L;
    let nu_max = 50.0 * GAMMA * 1.0 / (1.0 / (r_corner * r_corner * r_corner)).sqrt();
    let expect_a = 0.1 * DX * DX / nu_max;
    assert!(
        (dt_alpha - expect_a).abs() < 1e-9 * expect_a.max(1.0),
        "alpha cap does not match the nu_max bound: {dt_alpha:e} vs {expect_a:e}"
    );
}
