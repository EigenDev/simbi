// =============================================================================
// nmhd_rotor_cyl_rphi.rs
//
// the magnetized rotor in the cylindrical r-phi disk plane — the
// curvilinear sibling of nmhd_rotor_cyl_rz, validating the r-phi constrained-transport
// stack: the single out-of-plane corner EMF E_z (reused from cartesian, identity axes),
// the metric in-plane curl
//   dB_r/dt   = -(1/r) d_phi E_z   (the 1/r metric on the phi-derivative),
//   dB_phi/dt = +d_r E_z           (flat, no metric — mirror of r-z),
// the gas + geometric source on the (r, phi) disk, and the `with_cyl_plane(RPhi)` selector
// (grid axes [0,1], out-of-plane vertical B_z). a dense disk spins in the (r, phi) disk
// plane inside a uniform toroidal B_phi, winding B_r out of the advected
// azimuthal field. asserts:
//   (a) the staggered cylindrical div(B) = (1/r) d_r(r B_r) + (1/r) d_phi B_phi stays at
//       machine zero (the discrete d-of-d the metric curl must preserve),
//   (b) the state stays physical (rho>0, p>0, finite),
//   (c) the field winds — B_r (zero in the IC) develops from the in-plane advection of B_phi,
//       proving the E_z edge binds the right in-plane components (v_r, v_phi, B_r, B_phi) and
//       the (1/r) d_phi metric curl actually evolves the in-plane field.
// =============================================================================

use symbi::prelude::*;

type Sim = SimCpuGeneric<NewtonianMhd, 2, 3, Cylindrical, IdealGas<f64>>;

const NR: usize = 128;
const NPHI: usize = 128;
const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
// an annulus r in [1, 2] (r_min > 0 avoids the r=0 axis singularity), phi in [0, 1] (radians).
const R_LO: f64 = 1.0;
const R_HI: f64 = 2.0;
const PHI_LO: f64 = 0.0;
const PHI_HI: f64 = 1.0;
// rotor center + radii (in-plane spin), in physical (r, r*phi) distance.
const RC: f64 = 1.5;
const PHIC: f64 = 0.5;
const R0: f64 = 0.15;
const R1: f64 = 0.18;
const OMEGA: f64 = 2.0; // in-plane angular rate (core edge speed omega*R0 ~ 0.3).
const B0: f64 = 1.0; // uniform toroidal field B_phi.
const T_FINAL: f64 = 0.04;

// the rotor primitive at physical (r, phi): a dense core in solid-body rotation in the disk
// plane about (rc, phic). the physical displacement is (dr, r_c*dphi) (the azimuthal arc-length
// uses the metric), and v = omega x displacement = (-omega*r_c*dphi, omega*dr) in (r, phi).
// v_z (the out-of-plane) stays zero — this exercises the in-plane CT (E_z edge). returns
// (rho, v_r, v_phi).
fn rotor_state(r: f64, phi: f64) -> (f64, f64, f64) {
    let dr = r - RC;
    let arc = RC * (phi - PHIC); // physical azimuthal offset at the rotor radius.
    let rad = (dr * dr + arc * arc).sqrt();
    if rad < R0 {
        (10.0, -OMEGA * arc, OMEGA * dr)
    } else if rad < R1 {
        let f = (R1 - rad) / (R1 - R0);
        (1.0 + 9.0 * f, -f * OMEGA * arc, f * OMEGA * dr)
    } else {
        (1.0, 0.0, 0.0)
    }
}

fn make_sim() -> Sim {
    // fluent builder: grid + physical box (dx derived) + plane; ng/RK2/outflow/device defaulted.
    let sim = Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cylindrical)
        .cells([NR, NPHI])
        .bounds([R_LO, PHI_LO], [R_HI, PHI_HI])
        .cfl(CFL)
        .cyl_plane(CylPlane::RPhi) // grid the disk plane (axes [0,1], out-of-plane B_z).
        .finish()
        .expect("cyl r-phi disk sim");
    // uniform toroidal B_phi on the phi-faces (bface[1]); B_r stays zero — the CT ground truth.
    sim.seed_face(1, B0);
    // the rotor IC, from physical (r, phi). velocity is coordinate-indexed (0=r, 1=phi, 2=z);
    // B = (B_r, B_phi, B_z) = (0, B0, 0).
    sim.seed_cells(|[r, phi]| {
        let (rho, vr, vphi) = rotor_state(r, phi);
        MhdPrim::new(
            Prim::adiabatic(Density(rho), Tensor::new([vr, vphi, 0.0]), Pressure(1.0)),
            Tensor::new([0.0, B0, 0.0]),
        )
    });
    sim
}

// the staggered cylindrical div(B), normalized by |B|: the metric divergence the CT curl must
// hold at machine zero. (1/(r_c dr))(r_hi B_r_hi - r_lo B_r_lo) + (1/(r_c dphi))(B_phi_hi - B_phi_lo).
fn rel_divb(s: &Sim) -> f64 {
    let mhd = s.fields.mhd.as_ref().unwrap();
    let dr = (R_HI - R_LO) / NR as f64;
    let dphi = (PHI_HI - PHI_LO) / NPHI as f64;
    let mut md = 0.0_f64;
    let mut mb = B0;
    for c in s.geom.interior.iter() {
        let r_lo = R_LO + c[0] as f64 * dr;
        let r_hi = R_LO + (c[0] + 1) as f64 * dr;
        let r_c = R_LO + (c[0] as f64 + 0.5) * dr;
        let br_lo = *mhd.bface[0].view().at(c);
        let br_hi = *mhd.bface[0].view().at([c[0] + 1, c[1]]);
        let bphi_lo = *mhd.bface[1].view().at(c);
        let bphi_hi = *mhd.bface[1].view().at([c[0], c[1] + 1]);
        let div = (r_hi * br_hi - r_lo * br_lo) / (r_c * dr) + (bphi_hi - bphi_lo) / (r_c * dphi);
        md = md.max(div.abs());
        mb = mb.max((br_lo * br_lo + bphi_lo * bphi_lo).sqrt());
    }
    md / mb
}

#[test]
fn nmhd_rotor_cyl_rphi_preserves_divb_winds_field_stays_physical() {
    let mut sim = make_sim();
    // matched KernelSet straight off the sim — gamma/cfl/alloc pulled from it; tune theta + solver.
    let sub = sim
        .substrate()
        .theta(1.5)
        .with_solver(Solver::Hlld)
        .expect("valid solver/regime pair");

    // the curvilinear metric curl carries the r_hi/r_lo + 1/r factors (O(1) cancellation), so its
    // div(B) roundoff floor sits an order above cartesian ~1e-13. assert div(B) stays at a bounded
    // machine-zero (< 1e-11) every step — a sign/metric bug would leak secularly.
    let mut steps = 0u64;
    let mut max_rel = 0.0_f64;
    evolve_with_callback(&mut sim, &sub, T_FINAL, 1, |s| {
        let rel = rel_divb(s);
        assert!(
            rel < 1e-11,
            "cyl r-phi rotor div(B) grew to rel={rel:e} at iter {}",
            s.iteration
        );
        max_rel = max_rel.max(rel);
        steps = s.iteration;
    })
    .expect("cyl r-phi rotor evolve failed");
    assert!(
        steps >= 10,
        "cyl r-phi rotor produced only {steps} steps — gate barely exercised"
    );

    // physicality through the spun-up core + the field has wound (B_r developed from 0).
    let mhd = sim.fields.mhd.as_ref().unwrap();
    let mut max_br = 0.0_f64;
    for c in sim.geom.interior.iter() {
        let prim = sim.prim_at(c); // c2p recover — no hand-built MhdCons
        assert!(
            prim.rho().is_finite() && prim.rho() > 0.0,
            "cell {c:?}: rho={}",
            prim.rho()
        );
        assert!(
            prim.pre().is_finite() && prim.pre() > 0.0,
            "cell {c:?}: p={}",
            prim.pre()
        );
        max_br = max_br.max(mhd.bcell[0].view().at(c).abs());
    }
    assert!(
        max_br > 0.01,
        "field did not wind: max|B_r|={max_br:e} (in-plane rotor should generate B_r)"
    );

    eprintln!(
        "[nmhd_rotor cyl r-phi] DONE iter={} t={:.4e} max|B_r|={:.4} max rel div(B)={:e}",
        sim.iteration, sim.time, max_br, max_rel
    );
}
