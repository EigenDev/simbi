// =============================================================================
// mhd_drain_device_equivalence.rs
//
// the immersed-body drain on a magnetized gas on device memory against the same run on host
// memory: the magnetic-energy sandwich, the Alfven stiffness reduction, and the drain itself are
// baked kernels and field reductions in the store's memory space, so a transparent sink and a
// magnetic-slip sink advanced over several coupled root steps agree with the host run to
// roundoff in the gas, the staggered field, the accreted mass, and the slip-heat receipt, and the
// device sink accretes. gpu-feature builds only.
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
use symbi_ib::{Body, BodyCollection, BodyKind, MagneticSpec, SurfaceSpec};
use symbi_substrate::regimes::substrate_gpu::device_sync;
use symbi_xpu::{CpuSpace, DeviceMemory, DeviceSpace, ExecutionSpace, HostMemory, MemorySpace};

const N: usize = 16;
const GAMMA: f64 = 5.0 / 3.0;
const DT: f64 = 1.0e-3;
const BODY: [f64; 3] = [0.5, 0.5, 0.5];
const R_BODY: f64 = 0.22;
const TOL: f64 = 1e-11;

fn build<S: ExecutionSpace, Mem: MemorySpace>(
    magnetic: MagneticSpec,
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
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
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
                .with_magnetic(magnetic),
        ),
    )
}

// the interior gas and cell field, every stored face, and the body's receipts.
fn snapshot<S: ExecutionSpace, Mem: MemorySpace>(
    sim: &SimStateGeneric<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, S, Mem, f64>,
) -> Vec<(&'static str, Vec<f64>)> {
    device_sync::<Mem>();
    let m = sim.fields.mhd.as_ref().unwrap();
    let interior = &sim.geom.interior;
    let den: Vec<f64> = interior.iter().map(|c| *sim.fields.cons.den.at(c)).collect();
    let nrg: Vec<f64> = interior.iter().map(|c| *sim.fields.cons.nrg_field().unwrap().at(c)).collect();
    let mut mom = Vec::new();
    let mut bcell = Vec::new();
    for d in 0..3 {
        for c in interior.iter() {
            mom.push(*sim.fields.cons.mom[d].at(c));
            bcell.push(*m.bcell[d].at(c));
        }
    }
    let mut faces = Vec::new();
    for d in 0..3 {
        for c in m.bface[d].domain().iter() {
            faces.push(*m.bface[d].at(c));
        }
    }
    let b = sim.immersed.as_ref().unwrap().bodies.get(0);
    let accreted = match b.kind {
        BodyKind::BlackHole { total_accreted_mass, .. } => total_accreted_mass,
        _ => 0.0,
    };
    vec![
        ("density", den),
        ("momentum", mom),
        ("energy", nrg),
        ("cell field", bcell),
        ("faces", faces),
        ("receipts", vec![accreted, b.slip_heat_total]),
    ]
}

fn assert_close(label: &str, host: &[f64], dev: &[f64]) {
    assert_eq!(host.len(), dev.len(), "{label}: length mismatch");
    let scale = host.iter().fold(0.0_f64, |m, x| m.max(x.abs())).max(1.0);
    let worst = host.iter().zip(dev).map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
    assert!(worst <= TOL * scale, "{label}: host and device disagree by {worst:.3e} (scale {scale:.3e})");
}

fn advance<S: ExecutionSpace, Mem: MemorySpace>(
    sim: SimStateGeneric<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, S, Mem, f64>,
    steps: usize,
) -> Vec<(&'static str, Vec<f64>)> {
    let kset = NewtonianMhdSubstrateKernelSet3D::<Mem, f64>::new(GAMMA, 0.3, 1.0, &sim.geom.allocated);
    let mut hier = Hierarchy::single(sim, kset);
    hier.prime();
    for _ in 0..steps {
        hier.step_root_with_dt(DT);
    }
    snapshot(&hier.levels[0].state)
}

#[test]
fn a_transparent_magnetized_sink_drains_on_device_as_on_the_host() {
    let host = advance(build::<CpuSpace, HostMemory>(MagneticSpec::None), 4);
    let dev = advance(build::<DeviceSpace, DeviceMemory>(MagneticSpec::None), 4);
    assert!(dev[5].1[0] > 0.0, "the device sink accreted nothing; the equivalence is vacuous");
    for ((label, h), (_, d)) in host.iter().zip(&dev) {
        assert_close(label, h, d);
    }
}

#[test]
fn a_magnetic_slip_sink_drains_and_slips_on_device_as_on_the_host() {
    let slip = MagneticSpec::Slip {
        diffusivity_ratio: 2.0,
        shell_width: 0.12,
        slip_length_ratio: 1.0,
        field_regularization: 0.1,
        placement: 0.0,
    };
    let host = advance(build::<CpuSpace, HostMemory>(slip), 4);
    let dev = advance(build::<DeviceSpace, DeviceMemory>(slip), 4);
    assert!(dev[5].1[0] > 0.0 && dev[5].1[1] > 0.0, "the device sink accreted or heated nothing: {:?}", dev[5].1);
    for ((label, h), (_, d)) in host.iter().zip(&dev) {
        assert_close(label, h, d);
    }
}
