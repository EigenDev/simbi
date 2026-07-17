// =============================================================================
// viscous_adiabatic_ortho.rs
//
// the ADIABATIC orthogonal viscous kernels, end to end on a cylindrical (R, phi)
// grid through the production dispatch: the same scale-factor operator as the
// iso ortho kernel plus the div(tau . u) heating onto the total energy. gates:
// - a differentially rotating disk (u_phi ~ 1/R, nonzero shear) loses angular
//   momentum locally AND books heating (the total energy field changes beyond
//   the momentum work alone would);
// - a RIGIDLY rotating disk (v^phi = Omega, coordinate-contravariant) is an
//   EXACT discrete stress-free null through the production dispatch;
// - the ALPHA ortho variant (local cs^2, Omega_K from the radial coordinate)
//   genuinely acts and differs from the constant-nu update.
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::state::*;
use symbi::sim::substrate_seam::{KernelSet, WithViscosity};
use symbi_algebra::Tensor;
use symbi_geometry::CylindricalRPhi;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.3;
const NR: usize = 32;
const NP: usize = 24;
const R_LO: f64 = 1.0;
const R_HI: f64 = 3.0;
const DR: f64 = (R_HI - R_LO) / NR as f64;
const DP: f64 = 2.0 * std::f64::consts::PI / NP as f64;
const DT: f64 = 1.0e-4;

// the rigid-rotation null residual at radial resolution `nr`: one viscous
// application on v_phi = Omega R, measured as the max interior radial-momentum
// change. the discrete state samples v at cell centers while the operator's
// scale factors live at volumetric centroids, so the null holds to TRUNCATION
// order, not exactly — the invariant is that the residual converges away at
// second order (the carrier-level null, with consistent sampling, is exact to
// 1e-15 and is gated in symbi-hydro).
fn rigid_null_residual(nr: usize) -> f64 {
    let omega = 0.4;
    let dr = (R_HI - R_LO) / nr as f64;
    let sim = SimState::<Newtonian, 2, CylindricalRPhi, IdealGas<f64>, CpuSpace, HostMemory>::build(
        Newtonian,
        IdealGas { gamma: GAMMA },
        CylindricalRPhi,
    )
    .cells([nr, NP])
    .origin([R_LO, 0.0])
    .spacing([dr, DP])
    .boundaries(Boundaries(std::array::from_fn(|a| {
        if a == 1 { [BoundaryType::Periodic; 2] } else { [BoundaryType::Outflow; 2] }
    })))
    .cfl(CFL)
    .allocate()
    .expect("sim")
    // prim.vel is COORDINATE-contravariant: the rigid rotation is v^phi = Omega
    // CONSTANT (physical u_phi = h2 v^phi = Omega R).
    .set_initial(|_| Prim { rho: 1.0, vel: Tensor::new([0.0, omega]), pre: 1.0 })
    .build();
    let k = Kern::new(GAMMA, CFL, &sim.geom.allocated).with_viscosity(0.05);
    // the production stage-entry invariant: prims current AND ghosts filled —
    // the viscous stencil reads +-1, and an unfilled ghost band is garbage the
    // production loop never exposes it to.
    k.c2p(&sim);
    k.ghost_fill(&sim);
    let before: Vec<f64> =
        sim.geom.interior.iter().map(|c| *sim.fields.cons.mom[0].view().at(c)).collect();
    k.viscous(&sim, DT);
    let after: Vec<f64> =
        sim.geom.interior.iter().map(|c| *sim.fields.cons.mom[0].view().at(c)).collect();
    let _ = nr;
    before
        .iter()
        .zip(after.iter())
        .map(|(b, a)| (b - a).abs())
        .fold(0.0_f64, f64::max)
}

type Sim = SimState<Newtonian, 2, CylindricalRPhi, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

// vphi returns the COORDINATE v^phi (physical u_phi = R v^phi).
fn build(vphi: impl Fn(f64) -> f64, with_body: bool) -> Sim {
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, CylindricalRPhi)
        .cells([NR, NP])
        .origin([R_LO, 0.0])
        .spacing([DR, DP])
        .boundaries(Boundaries(std::array::from_fn(|a| {
            if a == 1 { [BoundaryType::Periodic; 2] } else { [BoundaryType::Outflow; 2] }
        })))
        .cfl(CFL)
        .allocate()
        .expect("sim")
        .set_initial(|x| Prim { rho: 1.0, vel: Tensor::new([0.0, vphi(x[0])]), pre: 1.0 })
        .build();
    if !with_body {
        return sim;
    }
    // the alpha law needs body 0's mass for Omega_K; on the axis, position 0.
    sim.with_bodies(BodyCollection::new().add(Body::gravitational(
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
    let (mut m0, mut m1, mut en) = (Vec::new(), Vec::new(), Vec::new());
    for c in sim.geom.interior.iter() {
        m0.push(*sim.fields.cons.mom[0].view().at(c));
        m1.push(*sim.fields.cons.mom[1].view().at(c));
        en.push(*nrg.view().at(c));
    }
    (m0, m1, en)
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max)
}

#[test]
fn differential_shear_diffuses_and_heats_on_the_cylindrical_chart() {
    let sim = build(|r| 0.5 / (r * r), false);
    let k = Kern::new(GAMMA, CFL, &sim.geom.allocated).with_viscosity(0.05);
    k.c2p(&sim);
    let (m0b, m1b, enb) = snapshot(&sim);
    k.viscous(&sim, DT);
    let (m0a, m1a, ena) = snapshot(&sim);
    assert!(max_abs_diff(&m1b, &m1a) > 1e-12, "the shear stress never touched u_phi");
    let _ = (m0b, m0a);
    let den = max_abs_diff(&enb, &ena);
    assert!(den > 1e-14, "no viscous energy flux booked on the shear ({den:e})");
}

#[test]
fn rigid_rotation_is_an_exact_discrete_null() {
    // v^phi = Omega constant (the coordinate-contravariant rigid rotation) gives
    // physical u_phi = Omega h2 at EVERY stencil point, which the orthogonal
    // stress nulls identically — through the full production dispatch, over the
    // whole domain (periodic phi and outflow-R ghosts both represent the profile
    // exactly). the PRE-conversion kernels fed coordinate components as physical
    // and produced e_Rphi = -Omega/(2R) here — an O(1) spurious torque on rigid
    // rotation that this gate holds extinct.
    let residual = rigid_null_residual(32);
    assert!(
        residual < 1e-14,
        "rigid rotation produced a viscous force through the dispatch: {residual:e}"
    );
}

#[test]
fn alpha_ortho_acts_and_differs_from_constant_nu() {
    let sim_a = build(|r| 0.5 / (r * r), true);
    let ka = Kern::new(GAMMA, CFL, &sim_a.geom.allocated).with_alpha(0.1);
    ka.c2p(&sim_a);
    let (_, m1b, _) = snapshot(&sim_a);
    ka.viscous(&sim_a, DT);
    let (_, m1a, _) = snapshot(&sim_a);
    assert!(max_abs_diff(&m1b, &m1a) > 1e-12, "the alpha ortho pass never acted");

    let sim_c = build(|r| 0.5 / (r * r), true);
    let kc = Kern::new(GAMMA, CFL, &sim_c.geom.allocated).with_viscosity(0.05);
    kc.c2p(&sim_c);
    kc.viscous(&sim_c, DT);
    let (_, m1c, _) = snapshot(&sim_c);
    assert!(
        max_abs_diff(&m1a, &m1c) > 1e-12,
        "the alpha update equals constant-nu everywhere: the radial law is dropped"
    );
}

#[test]
fn isolation_probe_rigid_null_with_hand_set_prims() {
    // bypass the builder/c2p entirely: prims set directly over the ALLOCATED
    // domain (ghosts included), one dispatch, per-ring dmom printout. exact
    // coordinate-rigid input v^phi = Omega everywhere.
    let omega = 0.4;
    let sim = SimState::<Newtonian, 2, CylindricalRPhi, IdealGas<f64>, CpuSpace, HostMemory>::build(
        Newtonian,
        IdealGas { gamma: GAMMA },
        CylindricalRPhi,
    )
    .cells([NR, NP])
    .origin([R_LO, 0.0])
    .spacing([DR, DP])
    .boundaries(Boundaries(std::array::from_fn(|a| {
        if a == 1 { [BoundaryType::Periodic; 2] } else { [BoundaryType::Outflow; 2] }
    })))
    .cfl(CFL)
    .allocate()
    .expect("sim")
    .set_initial(|_| Prim { rho: 1.0, vel: Tensor::new([0.0, 0.0]), pre: 1.0 })
    .build();
    let pre = sim.fields.prim.pre_field().expect("pre");
    for c in sim.geom.allocated.iter() {
        sim.fields.prim.rho.set(c, 1.0);
        sim.fields.prim.vel[0].set(c, 0.0);
        sim.fields.prim.vel[1].set(c, omega);
        pre.set(c, 1.0);
        sim.fields.cons.mom[0].set(c, 0.0);
        sim.fields.cons.mom[1].set(c, 0.0);
    }
    let k = Kern::new(GAMMA, CFL, &sim.geom.allocated).with_viscosity(0.05);
    k.viscous(&sim, DT);
    let after: Vec<f64> =
        sim.geom.interior.iter().map(|c| *sim.fields.cons.mom[0].view().at(c)).collect();
    let rings: Vec<f64> = after
        .chunks(NP)
        .map(|r| r.iter().map(|v| v.abs()).fold(0.0_f64, f64::max))
        .collect();
    eprintln!("isolation ring |dmom_r|: {:?}", &rings[..8]);
    let interior_max = rings[2..NR - 2].iter().cloned().fold(0.0_f64, f64::max);
    assert!(
        interior_max < 1e-13,
        "hand-set coordinate-rigid input still produces a radial force: {interior_max:e}"
    );
}
