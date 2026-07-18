// =============================================================================
// rigid_spin_axis_gpu.rs
//
// the device twin of rigid_spin_axis: a 3D shaped (SDF sphere) rigid wall
// spinning about the +x axis, run on device memory (NVRTC render of the
// SPINNING shaped GvKernel — the runtime 3x3 mask rotation + omega x r wall
// velocity) and asserted BIT-CLOSE to the CPU run. this closes the arbitrary-
// axis spin front on device: the spinning kernel path (selected by a nonzero
// omega) was compile-verified but never run for physics on a device.
// gates:
// - PARITY: the evolved cons state matches the CPU run;
// - AXIS: the circulation the drag sets up is about X (L_x grows, L_z at
//   roundoff) on device too — a z-hardcoded mask or wall velocity would swap it.
//
// runs on the host GPU (NVRTC needs no nvcc). run:
//   cargo test -p symbi --features cuda --test rigid_spin_axis_gpu
// =============================================================================
#![cfg(feature = "cuda")]

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
use symbi_xpu::cuda::{CudaSpace, UnifiedMemory};
use symbi_xpu::{CpuSpace, ExecutionSpace, HostMemory, MemorySpace};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.3;
const N: usize = 32;
const L: f64 = 1.0;
const DX: f64 = 2.0 * L / N as f64;
const OMEGA: f64 = 2.0;
const T_FINAL: f64 = 0.5;

type Sim<S, Mem> = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, S, Mem>;

fn build<S: ExecutionSpace, Mem: MemorySpace>() -> Sim<S, Mem> {
    let sim = Sim::<S, Mem>::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 3])
        .origin([-L; 3])
        .spacing([DX; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim")
        .set_initial(|_| Prim { rho: 1.0, vel: Tensor::new([0.0; 3]), pre: 1.0 })
        .build();
    let mut sim = sim.with_bodies(BodyCollection::new().add(
        Body::rigid_sphere(
            0,
            Tensor::new([0.0; 3]),
            Tensor::new([0.0; 3]),
            1.0,
            0.3,
            0.1,
            true, // no-slip: the tangential drag is the whole point here
        )
        .with_surface(SurfaceSpec::Porous { porosity: 0.0, k_eta_n: 1.0e3, k_eta_t: 1.0e3 })
        .with_spin_about(OMEGA, Tensor::new([1.0, 0.0, 0.0])),
    ));
    // the CSG shape routes the runtime spinning shaped kernel (host cranelift / device NVRTC).
    sim.attach_body_shapes(vec![Some(SdfExpr::sphere([0.0; 3], 0.3))]);
    sim
}

type HostSim = Sim<CpuSpace, HostMemory>;
type DevSim = Sim<CudaSpace, UnifiedMemory>;

fn cons_rel_gap(h: &HostSim, d: &DevSim) -> f64 {
    let mut gap = 0.0_f64;
    let cmp = |hf: &symbi_grid::Field<f64, 3, HostMemory>,
               df: &symbi_grid::Field<f64, 3, UnifiedMemory>,
               gap: &mut f64| {
        for c in h.geom.interior.iter() {
            let (a, b) = (*hf.view().at(c), *df.view().at(c));
            assert!(b.is_finite(), "non-finite device cons at {c:?}: {b}");
            *gap = gap.max((a - b).abs() / a.abs().max(1.0));
        }
    };
    cmp(&h.fields.cons.den, &d.fields.cons.den, &mut gap);
    for k in 0..3 {
        cmp(&h.fields.cons.mom[k], &d.fields.cons.mom[k], &mut gap);
    }
    let (hn, dn) = (h.fields.cons.nrg_field().unwrap(), d.fields.cons.nrg_field().unwrap());
    cmp(hn, dn, &mut gap);
    gap
}

// the swirl moments L_x = sum rho (y v_z - z v_y) and L_z = sum rho (x v_y - y v_x).
fn swirl(sim: &DevSim) -> (f64, f64) {
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
    (lx, lz)
}

// a free (two-way) shaped spinner started about an off-axis direction: the gas reaction torque
// decelerates it (drag) while the runtime orientation MATRIX advances each step (the mask rotates),
// so the device kernel reads a fresh R + omega every step and the body EOM (feedback + step_bodies
// + advance_rotation) closes the two-way loop. the evolving-mask two-way path, on device.
fn build_twoway<S: ExecutionSpace, Mem: MemorySpace>() -> Sim<S, Mem> {
    let sim = Sim::<S, Mem>::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 3])
        .origin([-L; 3])
        .spacing([DX; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim")
        .set_initial(|_| Prim { rho: 1.0, vel: Tensor::new([0.0; 3]), pre: 1.0 })
        .build();
    let mut sim = sim.with_bodies(BodyCollection::new().add(
        Body::rigid_sphere(0, Tensor::new([0.0; 3]), Tensor::new([0.0; 3]), 1.0, 0.3, 5.0, true)
            .with_surface(SurfaceSpec::Porous { porosity: 0.0, k_eta_n: 1.0e3, k_eta_t: 1.0e3 })
            // spin about an off-axis direction so the orientation matrix (all 9 entries) evolves.
            .with_spin_about(OMEGA, Tensor::new([1.0, 0.7, 0.3]))
            .with_two_way_coupling(true),
    ));
    sim.attach_body_shapes(vec![Some(SdfExpr::sphere([0.0; 3], 0.3))]);
    sim
}

#[test]
fn evolving_two_way_spinner_matches_cpu_on_device() {
    let mut h = build_twoway::<CpuSpace, HostMemory>();
    let mut d = build_twoway::<CudaSpace, UnifiedMemory>();
    let hk = AdiabaticSubstrateKernelSet::<HostMemory, f64, 3>::new(GAMMA, CFL, &h.geom.allocated);
    let dk = AdiabaticSubstrateKernelSet::<UnifiedMemory, f64, 3>::new(GAMMA, CFL, &d.geom.allocated);
    evolve(&mut h, &hk, 0.3).expect("host two-way spin run");
    evolve(&mut d, &dk, 0.3).expect("device two-way spin run");
    symbi::regimes::substrate_gpu::device_sync::<UnifiedMemory>();

    // the full evolving-mask two-way loop composes identically on device.
    let gap = cons_rel_gap(&h, &d);
    assert!(gap < 1e-6, "evolving two-way spinner cons host!=device: rel gap {gap:e}");
    // the body's evolved omega vector + orientation matrix agree host==device.
    let ho = h.immersed.as_ref().unwrap().bodies.get(0).omega;
    let dvo = d.immersed.as_ref().unwrap().bodies.get(0).omega;
    for k in 0..3 {
        assert!((ho[k] - dvo[k]).abs() < 1e-6 * OMEGA + 1e-9, "omega[{k}] host {} != device {}", ho[k], dvo[k]);
    }
    let hr = h.immersed.as_ref().unwrap().bodies.get(0).orientation;
    let dr = d.immersed.as_ref().unwrap().bodies.get(0).orientation;
    let mut rgap = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            rgap = rgap.max((hr[i][j] - dr[i][j]).abs());
        }
    }
    assert!(rgap < 1e-6, "orientation matrix host!=device: {rgap:e}");
    // non-vacuous: the two-way spinner was dragged (omega magnitude decreased from OMEGA) and the
    // orientation actually advanced off identity.
    let omag = (ho[0] * ho[0] + ho[1] * ho[1] + ho[2] * ho[2]).sqrt();
    assert!(omag < OMEGA * 0.9999, "two-way drag did not decelerate the spinner ({omag} vs {OMEGA})");
    assert!((hr[0][0] - 1.0).abs() > 1e-6 || hr[0][1].abs() > 1e-6, "orientation never advanced (vacuous)");
}

#[test]
fn spinning_shaped_wall_matches_cpu_on_device() {
    let mut h = build::<CpuSpace, HostMemory>();
    let mut d = build::<CudaSpace, UnifiedMemory>();
    let hk = AdiabaticSubstrateKernelSet::<HostMemory, f64, 3>::new(GAMMA, CFL, &h.geom.allocated);
    let dk = AdiabaticSubstrateKernelSet::<UnifiedMemory, f64, 3>::new(GAMMA, CFL, &d.geom.allocated);
    evolve(&mut h, &hk, T_FINAL).expect("host spinning-shaped run");
    evolve(&mut d, &dk, T_FINAL).expect("device spinning-shaped run");
    symbi::regimes::substrate_gpu::device_sync::<UnifiedMemory>();

    let gap = cons_rel_gap(&h, &d);
    assert!(gap < 1e-7, "spinning shaped-wall cons host!=device: rel gap {gap:e}");

    // the physics discriminator holds through the device path: circulation about the spin
    // axis (X) develops while the swirl about Z stays at roundoff.
    let (lx, lz) = swirl(&d);
    assert!(lx.abs() > 1e-4, "device: no circulation about the spin axis (L_x = {lx:e})");
    assert!(lz.abs() < 1e-2 * lx.abs(), "device: spurious z-circulation L_z = {lz:e} vs L_x = {lx:e}");
}
