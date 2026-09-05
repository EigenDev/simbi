// =============================================================================
// mhd_slip_device_equivalence.rs
//
// the implicit magnetic-slip solve and its coupled step on device memory against the same run on
// host memory. every field operation of the solve is a baked kernel or the field reduction, so the
// two runs execute the same arithmetic in the same order up to the reduction's block fold; the
// receipts, the committed field, the deposited energy, and one full D-M-H-M-D root advance agree to
// roundoff, and the frozen operator is symmetric on device. gpu-feature builds only.
// =============================================================================
#![cfg(feature = "gpu")]

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet3D;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, MagneticSpec, SurfaceSpec};
use symbi_substrate::regimes::mhd_substrate::{
    magnetic_slip_apply_operator, magnetic_slip_commit, magnetic_slip_solve,
};
use symbi_substrate::regimes::substrate_gpu::device_sync;
use symbi_xpu::{CpuSpace, DeviceMemory, DeviceSpace, ExecutionSpace, HostMemory, MemorySpace};

const N: usize = 12;
const GAMMA: f64 = 5.0 / 3.0;
const BODY: [f64; 3] = [0.5, 0.5, 0.5];
const R_BODY: f64 = 0.22;
const DT: f64 = 2.0e-3;
const TOL: f64 = 1e-11;

fn build<S: ExecutionSpace, Mem: MemorySpace>(
    boundary: BoundaryType,
) -> SimStateGeneric<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, S, Mem, f64> {
    let dx = 1.0 / N as f64;
    let k = 2.0 * std::f64::consts::PI;
    let a0 = 0.3;
    let sim = SimStateGeneric::<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, S, Mem, f64>::build(
        NewtonianMhd,
        IdealGas { gamma: GAMMA },
        Cartesian,
    )
    .cells([N, N, N])
    .origin([0.0, 0.0, 0.0])
    .spacing([dx, dx, dx])
    .boundaries(Boundaries::uniform(boundary))
    .cfl(0.3)
    .allocate()
    .expect("sim construction")
    .set_initial(move |[x, y, _z]| {
        let bx = |xf: f64| -a0 * (k * xf).cos() * (k * y).sin();
        let by = |yf: f64| a0 * (k * x).sin() * (k * yf).cos();
        MhdPrim::new(
            Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)),
            Tensor::new([
                0.5 * (bx(x - 0.5 * dx) + bx(x + 0.5 * dx)),
                0.5 * (by(y - 0.5 * dx) + by(y + 0.5 * dx)),
                0.0,
            ]),
        )
    })
    .seed_faces(move |axis, [x, y, _z]| match axis {
        0 => -a0 * (k * x).cos() * (k * y).sin(),
        1 => a0 * (k * x).sin() * (k * y).cos(),
        _ => 0.0,
    })
    .build();
    sim.with_bodies(
        BodyCollection::new().add(
            Body::black_hole(0, Tensor::new(BODY), Tensor::zeros(), 1.0, R_BODY, 0.05, 1.0, 1.0, R_BODY)
                .with_surface(SurfaceSpec::Drain)
                .with_magnetic(MagneticSpec::Slip {
                    diffusivity_ratio: 2.0,
                    shell_width: 0.12,
                    slip_length_ratio: 1.0,
                    field_regularization: 0.1,
                    placement: 0.0,
                }),
        ),
    )
}

// every stored face of every component, the interior total energy, and the interior cell field.
fn snapshot<S: ExecutionSpace, Mem: MemorySpace>(
    sim: &SimStateGeneric<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, S, Mem, f64>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    device_sync::<Mem>();
    let m = sim.fields.mhd.as_ref().unwrap();
    let mut faces = Vec::new();
    for d in 0..3 {
        for c in m.bface[d].domain().iter() {
            faces.push(*m.bface[d].at(c));
        }
    }
    let nrg = sim.fields.cons.nrg_field().unwrap();
    let energy: Vec<f64> = sim.geom.interior.iter().map(|c| *nrg.at(c)).collect();
    let mut bcell = Vec::new();
    for d in 0..3 {
        for c in sim.geom.interior.iter() {
            bcell.push(*m.bcell[d].at(c));
        }
    }
    (faces, energy, bcell)
}

fn assert_close(label: &str, host: &[f64], dev: &[f64]) {
    assert_eq!(host.len(), dev.len(), "{label}: length mismatch");
    let scale = host.iter().fold(0.0_f64, |m, x| m.max(x.abs())).max(1.0);
    let worst = host
        .iter()
        .zip(dev)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        worst <= TOL * scale,
        "{label}: host and device disagree by {worst:.3e} (scale {scale:.3e})"
    );
}

// one solve and commit on each memory space from the same state: the receipts and the committed
// state agree to roundoff, on a periodic and on an outflow domain.
#[test]
fn the_solve_and_commit_on_device_match_the_host() {
    for boundary in [BoundaryType::Periodic, BoundaryType::Outflow] {
        let host = build::<CpuSpace, HostMemory>(boundary);
        let dev = build::<DeviceSpace, DeviceMemory>(boundary);
        let rh = magnetic_slip_solve::<3, 3, HostMemory, f64>(&host, DT, GAMMA, 1e-12, 500);
        let rd = magnetic_slip_solve::<3, 3, DeviceMemory, f64>(&dev, DT, GAMMA, 1e-12, 500);
        assert!(rh.converged && rd.converged, "{boundary:?}: a solve did not converge");
        assert_eq!(rh.iterations, rd.iterations, "{boundary:?}: iteration counts differ");
        assert!(
            (rh.final_residual_norm - rd.final_residual_norm).abs() <= TOL * rh.initial_residual_norm,
            "{boundary:?}: final residuals differ"
        );
        magnetic_slip_commit::<3, 3, HostMemory, f64>(&host, DT, GAMMA);
        magnetic_slip_commit::<3, 3, DeviceMemory, f64>(&dev, DT, GAMMA);
        let (fh, eh, bh) = snapshot(&host);
        let (fd, ed, bd) = snapshot(&dev);
        assert_close(&format!("{boundary:?} bface"), &fh, &fd);
        assert_close(&format!("{boundary:?} energy"), &eh, &ed);
        assert_close(&format!("{boundary:?} bcell"), &bh, &bd);
    }
}

// the frozen operator on device is symmetric in the face inner product: <p, L q> = <L p, q> for
// two independent face fields, with the products read back after the device barrier.
#[test]
fn the_frozen_operator_is_symmetric_on_device() {
    let dev = build::<DeviceSpace, DeviceMemory>(BoundaryType::Periodic);
    // the predictor freeze: one solve leaves bcell = interp(B^0) and the workspace's frozen field
    // set; the apply reads production bcell as the frozen state.
    let receipt = magnetic_slip_solve::<3, 3, DeviceMemory, f64>(&dev, DT, GAMMA, 1e-12, 500);
    assert!(receipt.converged);
    let m = dev.fields.mhd.as_ref().unwrap();
    let ws = m.magnetic_slip.as_ref().unwrap();
    device_sync::<DeviceMemory>();
    // two deterministic pseudo-random face fields in the workspace's spare vectors.
    let hash = |d: usize, c: [isize; 3], salt: u64| -> f64 {
        let mut h = salt ^ (d as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        for v in c {
            h ^= (v as u64).wrapping_add(0x1234_5678);
            h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        }
        ((h >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    for d in 0..3 {
        for c in ws.rhs.b[d].domain().iter() {
            ws.rhs.b[d].set(c, hash(d, c, 1));
            ws.residual.b[d].set(c, hash(d, c, 2));
        }
    }
    magnetic_slip_apply_operator::<3, 3, DeviceMemory, f64>(&dev, GAMMA, true, &ws.rhs, &ws.direction);
    magnetic_slip_apply_operator::<3, 3, DeviceMemory, f64>(&dev, GAMMA, true, &ws.residual, &ws.operator_direction);
    device_sync::<DeviceMemory>();
    let dot = |x: &symbi_sim::state::BfaceFields<3, DeviceMemory, f64>,
               y: &symbi_sim::state::BfaceFields<3, DeviceMemory, f64>|
     -> f64 {
        let mut s = 0.0;
        for d in 0..3 {
            for c in dev.geom.interior.iter() {
                s += *x.b[d].at(c) * *y.b[d].at(c);
            }
        }
        s
    };
    let p_lq = dot(&ws.rhs, &ws.operator_direction);
    let lp_q = dot(&ws.direction, &ws.residual);
    let scale = p_lq.abs().max(lp_q.abs()).max(1e-300);
    assert!(
        (p_lq - lp_q).abs() <= 1e-10 * scale,
        "the frozen operator is not symmetric on device: <p, L q> = {p_lq:.6e}, <L p, q> = {lp_q:.6e}"
    );
    let p_lp = dot(&ws.rhs, &ws.direction);
    assert!(p_lp >= -1e-10 * scale, "the frozen operator is not positive semidefinite on device: {p_lp:.3e}");
}

// one root advance of the coupled step on each memory space: the same D-M-H-M-D composition
// through the production driver, agreeing to roundoff in every evolved field.
#[test]
fn one_coupled_root_step_on_device_matches_the_host() {
    let host = build::<CpuSpace, HostMemory>(BoundaryType::Periodic);
    let dev = build::<DeviceSpace, DeviceMemory>(BoundaryType::Periodic);
    let hset = NewtonianMhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, 0.3, 1.0, &host.geom.allocated);
    let dset = NewtonianMhdSubstrateKernelSet3D::<DeviceMemory, f64>::new(GAMMA, 0.3, 1.0, &dev.geom.allocated);
    let mut hh = Hierarchy::single(host, hset);
    let mut hd = Hierarchy::single(dev, dset);
    hh.prime();
    hd.prime();
    for _ in 0..3 {
        hh.step_root_with_dt(DT);
        hd.step_root_with_dt(DT);
    }
    device_sync::<DeviceMemory>();
    let (bfh, _, nh, ph) = hh.slip_state_snapshots(0);
    let (bfd, _, nd, pd) = hd.slip_state_snapshots(0);
    assert_close("bface", &bfh, &bfd);
    assert_close("energy", &nh, &nd);
    assert_close("pressure", &ph, &pd);
    assert_close("density", &hh.density_snapshot(0), &hd.density_snapshot(0));
}
