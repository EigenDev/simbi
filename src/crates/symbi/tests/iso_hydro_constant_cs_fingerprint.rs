// =============================================================================
// iso_hydro_constant_cs_fingerprint.rs
//
// the constant-sound-speed isothermal hydrodynamic path with a draining sink, pinned bit for bit:
// a 2D periodic vortex falling onto a drain, advanced a few root steps and hashed over every
// interior conserved value. the isothermal closure carries a prescribed sound-speed field the
// drain reads; a run whose field is uniform is this run, so the hash holds across that field's
// introduction.
// =============================================================================

use symbi::regimes::substrate::IsoSubstrateKernelSet;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::Isothermal;
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::quantity::Density;
use symbi_hydro::state::PrimG;
use symbi_ib::{Body, BodyCollection, SurfaceSpec};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<IsoNewtonian, 2, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;

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

#[test]
fn the_isothermal_hydro_drain_fingerprint_holds() {
    const N: usize = 32;
    let k = 2.0 * std::f64::consts::PI;
    let dx = 1.0 / N as f64;
    let sim = Sim::build(IsoNewtonian, Isothermal { cs: 1.0 }, Cartesian)
        .cells([N; 2])
        .origin([0.0; 2])
        .spacing([dx; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .allocate()
        .expect("iso hydro construction")
        .set_initial(move |[x, y]| PrimG::isothermal(Density(1.0 + 0.1 * (k * x).sin()), Tensor::new([-(k * y).sin(), (k * x).sin()])))
        .build();
    let sim = sim.with_bodies(BodyCollection::new().add(
        Body::black_hole(0, Tensor::new([0.5, 0.5]), Tensor::zeros(), 1.0, 0.15, 0.05, 1.0, 1.0, 0.15).with_surface(SurfaceSpec::Drain),
    ));
    let kset = IsoSubstrateKernelSet::<HostMemory, f64, 2>::new(1.0, 0.3, &sim.geom.allocated);
    let mut hier = Hierarchy::single(sim, kset);
    hier.prime();
    for _ in 0..8 {
        hier.step_root_with_dt(2.0e-3);
    }
    let s = &hier.levels[0].state;
    let mut bits = Vec::new();
    for c in s.geom.interior.iter() {
        bits.push(s.fields.cons.den.at(c).to_bits());
        for k in 0..2 {
            bits.push(s.fields.cons.mom[k].at(c).to_bits());
        }
    }
    let h = fnv(bits.into_iter());
    println!("isothermal hydro drain fingerprint: {h:#018x}");
    assert_eq!(h, FINGERPRINT, "the constant-sound-speed isothermal hydro drain path changed");
}

const FINGERPRINT: u64 = 0x7c2b8334e9b4f5fb;
