// =============================================================================
// mhd_cyl_rphi_gpu.rs
//
// GPU<->CPU parity for the CYLINDRICAL r-phi DISK MHD path — the carrier gate for the
// r-phi substrate kernels (r-z parity lives in
// mhd_cyl_rz_gpu.rs). builds the SAME in-plane-rotor-in-B_phi disk sim on host and
// device via `with_cyl_plane(RPhi)`, evolves a handful of RK2 steps through the
// production loop, and asserts conserved state + cell B + the staggered FACE B agree
// over the interior. exercises the device r-phi metric curl (-(1/r) d_phi E_z), the
// reused E_z corner EMF, and the manifest-driven curvilinear (unfused) godunov.
// =============================================================================

#![cfg(feature = "cuda")]

use symbi::regimes::substrate_kernels::Solver;
use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cylindrical;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::state::Prim;
use symbi_xpu::cuda::{CudaSpace, UnifiedMemory};
use symbi_xpu::{CpuSpace, ExecutionSpace, HostMemory, MemorySpace};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const NR: usize = 48;
const NPHI: usize = 48;
const R_LO: f64 = 1.0;
const R_HI: f64 = 2.0;
const PHI_LO: f64 = 0.0;
const PHI_HI: f64 = 1.0;
const RC: f64 = 1.5;
const PHIC: f64 = 0.5;
const R0: f64 = 0.15;
const R1: f64 = 0.18;
const OMEGA: f64 = 2.0;
const B0: f64 = 1.0;

// in-plane disk rotor (r, phi) at physical (r, phi): (rho, v_r, v_phi); v_z = 0.
fn rotor_state(r: f64, phi: f64) -> (f64, f64, f64) {
    let dr = r - RC;
    let arc = RC * (phi - PHIC);
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

fn build_sim<S: ExecutionSpace, Mem: MemorySpace>()
-> SimStateGeneric<NewtonianMhd, 2, 3, Cylindrical, IdealGas<f64>, S, Mem, f64> {
    let dr = (R_HI - R_LO) / NR as f64;
    let dphi = (PHI_HI - PHI_LO) / NPHI as f64;
    SimStateGeneric::<NewtonianMhd, 2, 3, Cylindrical, IdealGas<f64>, S, Mem, f64>::build(
        NewtonianMhd,
        IdealGas { gamma: GAMMA },
        Cylindrical,
    )
    .cells([NR, NPHI])
    .origin([R_LO, PHI_LO])
    .spacing([dr, dphi])
    .cfl(CFL)
    .boundaries(Boundaries::uniform(BoundaryType::Outflow))
    // the cyl plane MUST be set before seeding (it picks the _cyl_rphi axis set the CT reads).
    .cyl_plane(CylPlane::RPhi)
    .allocate()
    .expect("cyl r-phi sim")
    .set_initial(|[r, phi]| {
        let (rho, vr, vphi) = rotor_state(r, phi);
        // velocity COORDINATE-indexed (0=r, 1=phi, 2=z); B = (B_r, B_phi, B_z) = (0, B0, 0).
        MhdPrim {
            hydro: Prim {
                rho,
                vel: Tensor::new([vr, vphi, 0.0]),
                pre: 1.0,
            },
            mag: Tensor::new([0.0, B0, 0.0]),
        }
    })
    // uniform toroidal B_phi on the phi-faces (bface[1]); B_r (bface[0]) stays zero.
    .seed_faces_uniform([0.0, B0])
    .build()
}

#[test]
fn nmhd_cyl_rphi_evolve_gpu_matches_cpu() {
    let mut host = build_sim::<CpuSpace, HostMemory>();
    let mut dev = build_sim::<CudaSpace, UnifiedMemory>();
    let hset = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(
        GAMMA,
        CFL,
        1.5,
        &host.geom.allocated,
    )
    .with_solver(Solver::Hlld)
    .expect("valid solver/regime pair");
    let dset = NewtonianMhdSubstrateKernelSet::<UnifiedMemory, f64, 2>::new(
        GAMMA,
        CFL,
        1.5,
        &dev.geom.allocated,
    )
    .with_solver(Solver::Hlld)
    .expect("valid solver/regime pair");

    let t_final = 0.04_f64;
    evolve(&mut host, &hset, t_final).expect("cpu evolve");
    evolve(&mut dev, &dset, t_final).expect("gpu evolve");
    symbi_xpu::cuda::ctx_sync();

    assert!(host.iteration >= 3, "too few steps ({})", host.iteration);
    assert_eq!(
        host.iteration, dev.iteration,
        "step count diverged: cpu {} gpu {}",
        host.iteration, dev.iteration
    );

    let hmhd = host.fields.mhd.as_ref().unwrap();
    let dmhd = dev.fields.mhd.as_ref().unwrap();
    let hnrg = host.fields.cons.nrg_field().unwrap();
    let dnrg = dev.fields.cons.nrg_field().unwrap();
    let close = |g: f64, c: f64, what: &str, coord: [isize; 2]| {
        assert!(g.is_finite(), "{what} at {coord:?} non-finite on GPU: {g}");
        let rel = (g - c).abs() / c.abs().max(1.0);
        assert!(
            rel < 1e-9,
            "{what} at {coord:?}: gpu {g} != cpu {c} (rel {rel:e})"
        );
    };
    for coord in host.geom.interior.iter() {
        close(
            *dev.fields.cons.den.view().at(coord),
            *host.fields.cons.den.view().at(coord),
            "cons.den",
            coord,
        );
        close(
            *dnrg.view().at(coord),
            *hnrg.view().at(coord),
            "cons.nrg",
            coord,
        );
        for k in 0..3 {
            close(
                *dev.fields.cons.mom[k].view().at(coord),
                *host.fields.cons.mom[k].view().at(coord),
                "cons.mom",
                coord,
            );
            close(
                *dmhd.bcell[k].view().at(coord),
                *hmhd.bcell[k].view().at(coord),
                "bcell",
                coord,
            );
        }
    }
    // the STAGGERED face B — the r-phi metric curl writes these on device (B_r on r-faces,
    // B_phi on phi-faces). diff over each face domain to gate the device curl directly.
    for d in 0..2 {
        for fc in hmhd.bface[d].domain().clone().iter() {
            close(
                *dmhd.bface[d].view().at(fc),
                *hmhd.bface[d].view().at(fc),
                "bface",
                fc,
            );
        }
    }
}
