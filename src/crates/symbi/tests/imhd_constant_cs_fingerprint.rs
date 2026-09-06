// =============================================================================
// imhd_constant_cs_fingerprint.rs
//
// the constant-sound-speed isothermal MHD path, pinned bit for bit: a 2.5D periodic vortex with an
// in-plane field and a 3D isothermal box with a magnetic-slip sink, each advanced a few root steps
// and hashed over every interior conserved value, cell field, and stored face. the isothermal
// closure carries a prescribed sound-speed field; a run whose field is uniform is this run, so
// the hashes hold across that field's introduction and any later change to the closure.
// =============================================================================

use symbi::regimes::substrate_isothermal_mhd::IsothermalMhdSubstrateKernelSet;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::Isothermal;
use symbi_hydro::isothermal_mhd::IsothermalMhd;
use symbi_hydro::mhd_state::MhdPrimG;
use symbi_hydro::quantity::Density;
use symbi_hydro::state::PrimG;
use symbi_ib::{Body, BodyCollection, MagneticSpec, SurfaceSpec};
use symbi_sim::state::FieldStore;
use symbi_xpu::{CpuSpace, HostMemory};

const CS: f64 = 1.0;

fn fnv(bits: impl Iterator<Item = u64>) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bits {
        for byte in b.to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn fingerprint<const D: usize, const DOF: usize>(sim: &FieldStore<D, DOF, HostMemory, f64>) -> u64 {
    let m = sim.fields.mhd.as_ref().unwrap();
    let mut bits = Vec::new();
    for c in sim.geom.interior.iter() {
        bits.push(sim.fields.cons.den.at(c).to_bits());
        for k in 0..DOF {
            bits.push(sim.fields.cons.mom[k].at(c).to_bits());
            bits.push(m.bcell[k].at(c).to_bits());
        }
    }
    for d in 0..D {
        for c in m.bface[d].domain().iter() {
            bits.push(m.bface[d].at(c).to_bits());
        }
    }
    fnv(bits.into_iter())
}

#[test]
fn the_2p5d_isothermal_vortex_fingerprint_holds() {
    const N: usize = 32;
    let k = 2.0 * std::f64::consts::PI;
    let dx = 1.0 / N as f64;
    let sim = SimStateGeneric::<IsothermalMhd, 2, 3, Cartesian, Isothermal<f64>, CpuSpace, HostMemory, f64>::build(IsothermalMhd, Isothermal { cs: CS }, Cartesian)
        .cells([N; 2])
        .origin([0.0; 2])
        .spacing([dx; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .allocate()
        .expect("2.5D iso construction")
        .set_initial(move |[x, y]| {
            let vel = Tensor::new([-(k * y).sin(), (k * x).sin(), 0.0]);
            let bx = |xf: f64| -0.3 * (k * y).sin() * (k * xf).cos();
            let by = |yf: f64| 0.3 * (k * x).sin() * (k * yf).cos();
            let mag = Tensor::new([0.5 * (bx(x - 0.5 * dx) + bx(x + 0.5 * dx)), 0.5 * (by(y - 0.5 * dx) + by(y + 0.5 * dx)), 0.2]);
            MhdPrimG::<f64, 3, IsoModel>::new(PrimG::isothermal(Density(1.0), vel), mag)
        })
        .seed_faces(move |axis, [x, y]| match axis {
            0 => -0.3 * (k * y).sin() * (k * x).cos(),
            _ => 0.3 * (k * x).sin() * (k * y).cos(),
        })
        .build();
    let kset = IsothermalMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(CS, 0.3, 1.0, &sim.geom.allocated);
    let mut hier = Hierarchy::single(sim, kset);
    hier.prime();
    for _ in 0..8 {
        hier.step_root_with_dt(2.0e-3);
    }
    let h = fingerprint(&hier.levels[0].state);
    println!("2.5D isothermal vortex fingerprint: {h:#018x}");
    assert_eq!(h, FINGERPRINT_2P5D, "the constant-sound-speed 2.5D isothermal MHD path changed");
}

#[test]
fn the_3d_isothermal_slip_sink_fingerprint_holds() {
    const N: usize = 16;
    let k = 2.0 * std::f64::consts::PI;
    let dx = 1.0 / N as f64;
    let az = move |x: f64, y: f64| 0.3 / k * (k * x + 0.25 * std::f64::consts::PI).sin() * (k * y + 0.25 * std::f64::consts::PI).sin();
    let face = move |axis: usize, [x, y, _z]: [f64; 3]| match axis {
        0 => 0.3 + (az(x, y + 0.5 * dx) - az(x, y - 0.5 * dx)) / dx,
        1 => -(az(x + 0.5 * dx, y) - az(x - 0.5 * dx, y)) / dx,
        _ => 0.0,
    };
    let sim = SimStateGeneric::<IsothermalMhd, 3, 3, Cartesian, Isothermal<f64>, CpuSpace, HostMemory, f64>::build(IsothermalMhd, Isothermal { cs: CS }, Cartesian)
        .cells([N; 3])
        .origin([0.0; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .allocate()
        .expect("3D iso construction")
        .set_initial(move |[x, y, z]| {
            let bx = 0.5 * (face(0, [x - 0.5 * dx, y, z]) + face(0, [x + 0.5 * dx, y, z]));
            let by = 0.5 * (face(1, [x, y - 0.5 * dx, z]) + face(1, [x, y + 0.5 * dx, z]));
            MhdPrimG::<f64, 3, IsoModel>::new(PrimG::isothermal(Density(1.0), Tensor::new([0.0; 3])), Tensor::new([bx, by, 0.0]))
        })
        .seed_faces(face)
        .build();
    let sim = sim.with_bodies(BodyCollection::new().add(
        Body::black_hole(0, Tensor::new([0.5; 3]), Tensor::zeros(), 1.0, 0.22, 0.05, 1.0, 1.0, 0.22)
            .with_surface(SurfaceSpec::Drain)
            .with_magnetic(MagneticSpec::Slip { diffusivity_ratio: 2.0, shell_width: 0.12, slip_length_ratio: 1.5, field_regularization: 0.1, placement: 0.0 }),
    ));
    let kset = IsothermalMhdSubstrateKernelSet::<HostMemory, f64, 3>::new(CS, 0.3, 1.0, &sim.geom.allocated);
    let mut hier = Hierarchy::single(sim, kset);
    hier.prime();
    for _ in 0..4 {
        hier.step_root_with_dt(2.0e-3);
    }
    let h = fingerprint(&hier.levels[0].state);
    println!("3D isothermal slip sink fingerprint: {h:#018x}");
    assert_eq!(h, FINGERPRINT_3D, "the constant-sound-speed 3D isothermal slip path changed");
}

const FINGERPRINT_2P5D: u64 = 0x6dfa4ab720c0cd33;
const FINGERPRINT_3D: u64 = 0xfde53997de2c34df;
