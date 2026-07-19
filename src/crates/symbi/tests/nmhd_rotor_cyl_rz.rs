// =============================================================================
// nmhd_rotor_cyl_rz.rs
//
// the magnetized ROTOR in CYLINDRICAL r-z AXISYMMETRIC newtonian MHD —
// the curvilinear sibling of nmhd_rotor_2p5d, validating the cyl r-z constrained-transport
// stack: the SINGLE out-of-plane corner EMF E_phi, the metric in-plane curl
//   dB_r/dt = +d_z E_phi          (flat, h_z = 1),
//   dB_z/dt = -(1/r) d_r(r E_phi) (the cylindrical metric on the radial derivative),
// plus the DOF-lifted (DOF=3, grid D=2) gas + geometric source. a dense disk spins in the
// MERIDIONAL (r, z) plane inside a uniform vertical B_z, winding B_r out of the advected
// vertical field. asserts:
//   (a) the staggered CYLINDRICAL div(B) = (1/r) d_r(r B_r) + d_z B_z stays at machine zero
//       (the discrete d-of-d the metric curl must preserve — the whole point of CT),
//   (b) the state stays physical (rho>0, p>0, finite),
//   (c) the field WINDS — B_r (zero in the IC) develops from the poloidal advection of B_z,
//       proving the E_phi edge binds the right in-plane components (v_r, v_z, B_r, B_z) and
//       the metric curl actually evolves the in-plane field.
// =============================================================================

use symbi::regimes::substrate_kernels::Solver;
use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cylindrical;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::{MhdCons, MhdPrim};
use symbi_hydro::newtonian_mhd::{nmhd_recover, NewtonianMhd};
use symbi_hydro::state::{Cons, Prim};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory, f64>;

const NR: usize = 128;
const NZ: usize = 128;
const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
// an annulus r in [1, 2] (r_min > 0 avoids the r=0 axis singularity), z in [0, 1].
const R_LO: f64 = 1.0;
const R_HI: f64 = 2.0;
const Z_LO: f64 = 0.0;
const Z_HI: f64 = 1.0;
// rotor centre + radii (meridional-plane spin), in physical (r, z) coordinates.
const RC: f64 = 1.5;
const ZC: f64 = 0.5;
const R0: f64 = 0.15;
const R1: f64 = 0.18;
const OMEGA: f64 = 2.0; // poloidal angular rate (core edge speed OMEGA*R0 ~ 0.3).
const B0: f64 = 1.0; // uniform vertical field B_z.
const T_FINAL: f64 = 0.04;

// the rotor primitive at physical (r, z): a dense core in solid-body POLOIDAL rotation
// v = OMEGA * (e_phi x displacement) = (-OMEGA*(z-zc), OMEGA*(r-rc)) in the (r, z) plane,
// tapered to the ambient over [R0, R1]. v_phi (the swirl) stays zero — this test exercises
// the IN-PLANE CT (E_phi edge), not the out-of-plane induction-flux path. returns (rho, v_r, v_z).
fn rotor_state(r: f64, z: f64) -> (f64, f64, f64) {
    let (dr, dz) = (r - RC, z - ZC);
    let rad = (dr * dr + dz * dz).sqrt();
    if rad < R0 {
        (10.0, -OMEGA * dz, OMEGA * dr)
    } else if rad < R1 {
        let f = (R1 - rad) / (R1 - R0);
        (1.0 + 9.0 * f, -f * OMEGA * dz, f * OMEGA * dr)
    } else {
        (1.0, 0.0, 0.0)
    }
}

fn make_sim() -> Sim {
    let dr = (R_HI - R_LO) / NR as f64;
    let dz = (Z_HI - Z_LO) / NZ as f64;
    Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cylindrical)
        .cells([NR, NZ]).origin([R_LO, Z_LO]).spacing([dr, dz])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("cyl r-z rotor sim")
        .set_initial(|[r, z]| {
            let (rho, vr, vz) = rotor_state(r, z);
            // velocity is COORDINATE-indexed (0 = r, 1 = phi, 2 = z); B = (B_r, B_phi, B_z) = (0, 0, B0).
            MhdPrim {
                hydro: Prim { rho, vel: Tensor::new([vr, 0.0, vz]), pre: 1.0 },
                mag: Tensor::new([0.0, 0.0, B0]),
            }
        })
        // uniform vertical B_z on the z-faces (bface[1]); B_r on the r-faces (bface[0]) stays
        // zero — the staggered CT ground truth (cyl div(B) = 0 since r_hi*0 - r_lo*0 = 0).
        .seed_faces(|axis, _x| if axis == 1 { B0 } else { 0.0 })
        .build()
}

// the staggered CYLINDRICAL div(B), normalized by |B|: the metric divergence the CT curl
// must hold at machine zero. (1/r_c)*(r_hi*B_r_hi - r_lo*B_r_lo)/dr + (B_z_hi - B_z_lo)/dz.
fn rel_divb(s: &Sim) -> f64 {
    let mhd = s.fields.mhd.as_ref().unwrap();
    let dr = (R_HI - R_LO) / NR as f64;
    let dz = (Z_HI - Z_LO) / NZ as f64;
    let mut md = 0.0_f64;
    let mut mb = B0;
    for c in s.geom.interior.iter() {
        let r_lo = R_LO + c[0] as f64 * dr;
        let r_hi = R_LO + (c[0] + 1) as f64 * dr;
        let r_c = R_LO + (c[0] as f64 + 0.5) * dr;
        let br_lo = *mhd.bface[0].view().at(c);
        let br_hi = *mhd.bface[0].view().at([c[0] + 1, c[1]]);
        let bz_lo = *mhd.bface[1].view().at(c);
        let bz_hi = *mhd.bface[1].view().at([c[0], c[1] + 1]);
        let div = (r_hi * br_hi - r_lo * br_lo) / (r_c * dr) + (bz_hi - bz_lo) / dz;
        md = md.max(div.abs());
        mb = mb.max((br_lo * br_lo + bz_lo * bz_lo).sqrt());
    }
    md / mb
}

#[test]
fn nmhd_rotor_cyl_rz_preserves_divb_winds_field_stays_physical() {
    let mut sim = make_sim();
    let sub = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, CFL, 1.5, &sim.geom.allocated)
        .with_solver(Solver::Hlld).expect("valid solver/regime pair");

    // the curvilinear metric curl carries the r_hi/r_lo + 1/r_c factors (O(1) cancellation), so
    // its div(B) roundoff floor sits an order above the cartesian ~1e-13. assert div(B) stays at
    // a BOUNDED machine-zero (< 1e-11) every step — a sign/metric bug would leak SECULARLY.
    let mut steps = 0u64;
    let mut max_rel = 0.0_f64;
    evolve_with_callback(&mut sim, &sub, T_FINAL, 1, |s| {
        let rel = rel_divb(s);
        assert!(rel < 1e-11, "cyl r-z rotor div(B) grew to rel={rel:e} at iter {}", s.iteration);
        max_rel = max_rel.max(rel);
        steps = s.iteration;
    })
    .expect("cyl r-z rotor evolve failed");
    assert!(steps >= 10, "cyl r-z rotor produced only {steps} steps — gate barely exercised");

    // physicality through the spun-up core + the field has WOUND (B_r developed from 0).
    let eos = IdealGas { gamma: GAMMA };
    let mhd = sim.fields.mhd.as_ref().unwrap();
    let cnrg = sim.fields.cons.nrg_field().unwrap();
    let mut max_br = 0.0_f64;
    for c in sim.geom.interior.iter() {
        let cons = MhdCons::<f64, 3> {
            hydro: Cons {
                den: *sim.fields.cons.den.view().at(c),
                mom: Tensor::new([
                    *sim.fields.cons.mom[0].view().at(c),
                    *sim.fields.cons.mom[1].view().at(c),
                    *sim.fields.cons.mom[2].view().at(c),
                ]),
                nrg: *cnrg.view().at(c),
            },
            mag: Tensor::new([
                *mhd.bcell[0].view().at(c),
                *mhd.bcell[1].view().at(c),
                *mhd.bcell[2].view().at(c),
            ]),
        };
        let prim = nmhd_recover(&eos, &cons);
        assert!(prim.rho.is_finite() && prim.rho > 0.0, "cell {c:?}: rho={}", prim.rho);
        assert!(prim.pre.is_finite() && prim.pre > 0.0, "cell {c:?}: p={}", prim.pre);
        max_br = max_br.max(mhd.bcell[0].view().at(c).abs());
    }
    assert!(max_br > 0.01, "field did not wind: max|B_r|={max_br:e} (poloidal rotor should generate B_r)");

    eprintln!(
        "[nmhd_rotor cyl r-z] DONE iter={} t={:.4e} max|B_r|={:.4} max rel div(B)={:e}",
        sim.iteration, sim.time, max_br, max_rel
    );
}
