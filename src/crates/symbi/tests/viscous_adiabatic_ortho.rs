// =============================================================================
// viscous_adiabatic_ortho.rs
//
// the ADIABATIC orthogonal viscous kernels, end to end on a cylindrical (R, phi)
// grid through the production dispatch: the scale-factor stress operator plus the
// div(tau . u) heating onto the total energy. prim.vel stores PHYSICAL
// (orthonormal) components on every newtonian chart — the r-phi inertial source
// (m_phi v_phi / r), the CFL's physical-width crossing rate, and the keplerian
// disk gate (v_phi = sqrt(M/r) holds against central gravity) all pin that
// convention — so rigid rotation is v_phi = Omega * r, and the stress operator
// must consume the stored components directly. gates:
// - a constant-specific-angular-momentum disk (v_phi = ell / r, nonzero shear)
//   loses angular momentum locally AND books heating;
// - a RIGIDLY rotating disk (v_phi = Omega r) is stress-free through the
//   production dispatch to truncation order: prims are point samples at
//   arithmetic cell centers while the operator's scale factors live at
//   volumetric centroids, so the discrete residual is O(dx^2) and must
//   CONVERGE at second order (the consistent-sampling carrier null is exact to
//   1e-15 and gated in symbi-hydro). scaling the stored components by the
//   metric h (as if they were coordinate-contravariant) shifts the stress null
//   to v_phi = const, leaving a resolution-INDEPENDENT O(1) torque on rigid
//   rotation — caught by both the magnitude and the convergence assert;
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
// application on the physical profile v_phi = Omega r, measured as the max
// change over BOTH momentum components on the interior with the two rings
// nearest each radial boundary excluded (the zero-gradient outflow ghosts hold
// the constant edge value; because that departs from the smooth v_phi = Omega r
// continuation, the boundary-adjacent stencils see genuine shear). the AZIMUTHAL
// component is the discriminating one: for
// any axisymmetric v_phi(r) the radial force vanishes identically (t11 = t22 =
// 0 and t12 only enters the radial force through phi-differences), so a
// radial-only probe is vacuous — the shear signal lives in the angular-momentum
// flux (1/h2^2) d1(h2^2 t12).
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
    .set_initial(|x| Prim { rho: 1.0, vel: Tensor::new([0.0, omega * x[0]]), pre: 1.0 })
    .build();
    let k = Kern::new(GAMMA, CFL, &sim.geom.allocated).with_viscosity(0.05);
    // the production stage-entry invariant: prims current AND ghosts filled —
    // the viscous stencil reads +-1, and an unfilled ghost band is garbage the
    // production loop never exposes it to.
    k.c2p(&sim);
    k.ghost_fill(&sim);
    let grab = |c_ax: usize| -> Vec<f64> {
        sim.geom.interior.iter().map(|c| *sim.fields.cons.mom[c_ax].view().at(c)).collect()
    };
    let (b0, b1) = (grab(0), grab(1));
    k.viscous(&sim, DT);
    let (a0, a1) = (grab(0), grab(1));
    // ring-major layout (phi fastest): ring i occupies [i*NP, (i+1)*NP). trim the
    // outflow-contaminated radial edge rings from the max.
    let trimmed_max = |b: &[f64], a: &[f64]| -> f64 {
        b.iter()
            .zip(a)
            .enumerate()
            .filter(|(idx, _)| {
                let ring = idx / NP;
                (2..nr - 2).contains(&ring)
            })
            .map(|(_, (x, y))| (x - y).abs())
            .fold(0.0_f64, f64::max)
    };
    trimmed_max(&b0, &a0).max(trimmed_max(&b1, &a1))
}

type Sim = SimState<Newtonian, 2, CylindricalRPhi, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

// vphi returns the PHYSICAL azimuthal speed v_phi(r).
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
    // constant specific angular momentum: v_phi = ell / r has shear rate
    // e_Rphi = (r/2) d(v_phi/r)/dr = -ell/r^2 != 0 everywhere.
    let sim = build(|r| 0.5 / r, false);
    let k = Kern::new(GAMMA, CFL, &sim.geom.allocated).with_viscosity(0.05);
    k.c2p(&sim);
    k.ghost_fill(&sim);
    let (m0b, m1b, enb) = snapshot(&sim);
    k.viscous(&sim, DT);
    let (m0a, m1a, ena) = snapshot(&sim);
    assert!(max_abs_diff(&m1b, &m1a) > 1e-12, "the shear stress never touched u_phi");
    let _ = (m0b, m0a);
    let den = max_abs_diff(&enb, &ena);
    assert!(den > 1e-14, "no viscous energy flux booked on the shear ({den:e})");
}

#[test]
fn rigid_rotation_null_converges_at_second_order() {
    // v_phi = Omega r (the physical rigid rotation) is stress-free. the discrete
    // residual is the arithmetic-center vs volumetric-centroid sampling gap —
    // O(dx^2), tiny, and it must CONVERGE at second order. a kernel that
    // h-scales the stored components (reading them as coordinate-contravariant)
    // shifts the stress null to v_phi = const, so rigid rotation carries a
    // genuine resolution-independent torque: magnitude and ratio both fail.
    let r32 = rigid_null_residual(32);
    let r64 = rigid_null_residual(64);
    assert!(
        r32 < 1e-8,
        "rigid rotation produced a viscous torque through the dispatch: {r32:e}"
    );
    assert!(
        r64 < 0.35 * r32,
        "the rigid-null residual does not converge at second order: {r32:e} -> {r64:e}"
    );
}

#[test]
fn alpha_ortho_acts_and_differs_from_constant_nu() {
    let sim_a = build(|r| 0.5 / r, true);
    let ka = Kern::new(GAMMA, CFL, &sim_a.geom.allocated).with_alpha(0.1);
    ka.c2p(&sim_a);
    ka.ghost_fill(&sim_a);
    let (_, m1b, _) = snapshot(&sim_a);
    ka.viscous(&sim_a, DT);
    let (_, m1a, _) = snapshot(&sim_a);
    assert!(max_abs_diff(&m1b, &m1a) > 1e-12, "the alpha ortho pass never acted");

    let sim_c = build(|r| 0.5 / r, true);
    let kc = Kern::new(GAMMA, CFL, &sim_c.geom.allocated).with_viscosity(0.05);
    kc.c2p(&sim_c);
    kc.ghost_fill(&sim_c);
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
    // domain (ghosts included), one dispatch, per-ring dmom printout. the
    // physical rigid profile v_phi = Omega r evaluated at every allocated cell
    // center (the formula extends smoothly through the ghost bands).
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
    let ilo = sim.geom.interior.spaces[0].lo;
    let pre = sim.fields.prim.pre_field().expect("pre");
    for c in sim.geom.allocated.iter() {
        let r = R_LO + ((c[0] - ilo) as f64 + 0.5) * DR;
        sim.fields.prim.rho.set(c, 1.0);
        sim.fields.prim.vel[0].set(c, 0.0);
        sim.fields.prim.vel[1].set(c, omega * r);
        pre.set(c, 1.0);
        sim.fields.cons.mom[0].set(c, 0.0);
        sim.fields.cons.mom[1].set(c, 0.0);
    }
    let k = Kern::new(GAMMA, CFL, &sim.geom.allocated).with_viscosity(0.05);
    k.viscous(&sim, DT);
    // the azimuthal momentum carries the discriminating signal (the radial force
    // is identically zero for any axisymmetric v_phi profile).
    let rings = |c_ax: usize| -> Vec<f64> {
        sim.geom
            .interior
            .iter()
            .map(|c| (*sim.fields.cons.mom[c_ax].view().at(c)).abs())
            .collect::<Vec<f64>>()
            .chunks(NP)
            .map(|r| r.iter().cloned().fold(0.0_f64, f64::max))
            .collect()
    };
    let (r0, r1) = (rings(0), rings(1));
    eprintln!("isolation ring |dmom_r|: {:?}", &r0[..8]);
    eprintln!("isolation ring |dmom_phi|: {:?}", &r1[..8]);
    let interior_max = r0[2..NR - 2]
        .iter()
        .chain(&r1[2..NR - 2])
        .cloned()
        .fold(0.0_f64, f64::max);
    // the residual floor is the arithmetic-center vs volumetric-centroid
    // sampling gap (~1e-10 at this resolution); the h-scaling failure mode
    // sits four orders above it.
    assert!(
        interior_max < 1e-8,
        "hand-set physical rigid rotation still produces a viscous force: {interior_max:e}"
    );
}
