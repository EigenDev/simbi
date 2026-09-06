// =============================================================================
// nmhd_2p5d_fingerprint.rs
//
// the single-grid 2.5D adiabatic MHD path with a magnetic-slip sink, pinned bit for bit: a
// periodic vortex threaded by an in-plane field and a sheared vertical field, draining onto a
// slip sink, advanced a few root steps and hashed over every interior conserved value, cell field,
// and stored face. the refined 2.5D hierarchy composes this step; the single grid is unchanged
// by that composition.
// =============================================================================

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
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
use symbi_sim::state::FieldStore;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;

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

fn fingerprint(sim: &FieldStore<2, 3, HostMemory, f64>) -> u64 {
    let m = sim.fields.mhd.as_ref().unwrap();
    let mut bits = Vec::new();
    for c in sim.geom.interior.iter() {
        bits.push(sim.fields.cons.den.at(c).to_bits());
        bits.push(sim.fields.cons.nrg_field().unwrap().at(c).to_bits());
        for k in 0..3 {
            bits.push(sim.fields.cons.mom[k].at(c).to_bits());
            bits.push(m.bcell[k].at(c).to_bits());
        }
    }
    for d in 0..2 {
        for c in m.bface[d].domain().iter() {
            bits.push(m.bface[d].at(c).to_bits());
        }
    }
    fnv(bits.into_iter())
}

#[test]
fn the_2p5d_adiabatic_slip_sink_fingerprint_holds() {
    const N: usize = 32;
    let k = 2.0 * std::f64::consts::PI;
    let dx = 1.0 / N as f64;
    let sim = SimStateGeneric::<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 2])
        .origin([0.0; 2])
        .spacing([dx; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .allocate()
        .expect("2.5D construction")
        .set_initial(move |[x, y]| {
            let vel = Tensor::new([-(k * y).sin(), (k * x).sin(), 0.0]);
            let bx = |xf: f64| -0.3 * (k * y).sin() * (k * xf).cos();
            let by = |yf: f64| 0.3 * (k * x).sin() * (k * yf).cos();
            let bz = 0.2 * (1.0 + 0.3 * (k * x).sin() * (k * y).cos());
            let mag = Tensor::new([0.5 * (bx(x - 0.5 * dx) + bx(x + 0.5 * dx)), 0.5 * (by(y - 0.5 * dx) + by(y + 0.5 * dx)), bz]);
            MhdPrim::new(Prim::adiabatic(Density(1.0), vel, Pressure(1.0)), mag)
        })
        .seed_faces(move |axis, [x, y]| match axis {
            0 => -0.3 * (k * y).sin() * (k * x).cos(),
            _ => 0.3 * (k * x).sin() * (k * y).cos(),
        })
        .build();
    let sim = sim.with_bodies(BodyCollection::new().add(
        Body::black_hole(0, Tensor::new([0.5, 0.5]), Tensor::zeros(), 1.0, 0.15, 0.05, 1.0, 1.0, 0.15)
            .with_surface(SurfaceSpec::Drain)
            .with_magnetic(MagneticSpec::Slip { diffusivity_ratio: 2.0, shell_width: 0.08, slip_length_ratio: 1.0, field_regularization: 0.1, placement: 0.0 }),
    ));
    let kset = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.3, 1.0, &sim.geom.allocated);
    let mut hier = Hierarchy::single(sim, kset);
    hier.prime();
    for _ in 0..6 {
        hier.step_root_with_dt(2.0e-3);
    }
    let h = fingerprint(&hier.levels[0].state);
    println!("2.5D adiabatic slip sink fingerprint: {h:#018x}");
    assert_eq!(h, FINGERPRINT, "the single-grid 2.5D adiabatic slip path changed");
}

const FINGERPRINT: u64 = 0x9a4cb779bb2193e3;
