// =============================================================================
// fragment_restart_refusal.rs
//
// a restart of a bonded assembly checks the file's fragment records before it
// writes anything. a newtonian wind past three bonded fragments is
// checkpointed after a few steps (bonds 0-1 and 1-2); a run configured with a
// different bond list (0-1 alone) refuses the file and keeps its clock, gas,
// bodies and bonds as they were; a run configured with the recorded bonds
// restores the file (the control).
//
// run: cargo test -p symbi --test fragment_restart_refusal
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, Bond, BondMaterial, FragmentPhysics, SurfaceSpec};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const N: usize = 32;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

fn fragment(x: f64, y: f64) -> Body<f64, 2> {
    Body::rigid_sphere(0, Tensor::new([x, y]), Tensor::zeros(), 0.6, 0.12, 1e-3, true)
        .with_surface(SurfaceSpec::Porous { porosity: 0.0, k_eta_n: 50.0, k_eta_t: 50.0 })
        .with_two_way_coupling(true)
}

/// the assembly with the bonds `pairs`
fn assembly(pairs: &[(usize, usize)]) -> Sim {
    let coll = BodyCollection::new()
        .add_fragment(fragment(0.0, -0.4))
        .add_fragment(fragment(0.0, 0.0))
        .add_fragment(fragment(0.0, 0.4));
    let mat = BondMaterial { k_n: 50.0, gamma: 0.5, ..BondMaterial::rigid() };
    let bonds = pairs.iter().map(|&(i, j)| Bond::form(i, j, coll.get(i), coll.get(j), mat)).collect();
    let dx = 2.0 / N as f64;
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([-1.0, -1.0])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim")
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.5, 0.0]), Pressure(1.0)))
        .build()
        .with_bodies(coll);
    sim.attach_fragment_physics(FragmentPhysics { bonds, contacts: None, gravity: None });
    sim
}

/// the clock, conserved gas, body kinematics and bond state of a run, bits
fn state_bits(sim: &Sim) -> Vec<u64> {
    let mut bits = vec![sim.time.to_bits(), sim.dt.to_bits(), sim.iteration];
    let cons = &sim.fields.cons;
    for f in [&cons.den, &cons.mom[0], &cons.mom[1], cons.nrg_field().unwrap()] {
        bits.extend(sim.geom.interior.iter().map(|c| f.view().at(c).to_bits()));
    }
    let im = sim.immersed.as_ref().unwrap();
    for b in 0..im.bodies.len() {
        let body = im.bodies.get(b);
        bits.extend((0..2).flat_map(|a| [body.position[a].to_bits(), body.velocity[a].to_bits()]));
        bits.push(body.mass.to_bits());
    }
    for bond in &im.fragment_physics.as_ref().unwrap().bonds {
        bits.push(bond.rest_length.to_bits());
        bits.extend((0..3).map(|a| bond.slip[a].to_bits()));
        bits.push(u64::from(bond.intact));
    }
    bits
}

fn written_checkpoint(name: &str) -> (std::path::PathBuf, Sim) {
    let dir = std::env::temp_dir().join(format!("symbi_fragment_restart_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut source = assembly(&[(0, 1), (1, 2)]);
    let kernels = AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.4, &source.geom.allocated);
    evolve(&mut source, &kernels, 0.02).expect("the source run");
    let path = dir.join("bonded.h5");
    symbi_sim::checkpoint::write_checkpoint(&source, path.to_str().unwrap(), &symbi_io::Metadata::new()).expect("write");
    (path, source)
}

#[test]
fn a_restart_refused_for_its_bond_list_leaves_the_run_as_it_found_it() {
    let (path, _) = written_checkpoint("refused");
    let mut other = assembly(&[(0, 1)]);
    let before = state_bits(&other);
    let refused = symbi_sim::checkpoint::load_checkpoint(&mut other, path.to_str().unwrap());
    assert!(
        matches!(&refused, Err(e) if format!("{e}").contains("bond")),
        "the restart did not refuse the bond list: {refused:?}"
    );
    assert!(state_bits(&other) == before, "the refused restart wrote the run");
}

#[test]
fn a_restart_onto_the_recorded_bond_list_restores_the_assembly() {
    let (path, source) = written_checkpoint("restored");
    let mut fresh = assembly(&[(0, 1), (1, 2)]);
    assert!(state_bits(&fresh) != state_bits(&source), "the source never moved; the control is vacuous");
    symbi_sim::checkpoint::load_checkpoint(&mut fresh, path.to_str().unwrap()).expect("the matching restart");
    let (im, src) = (fresh.immersed.as_ref().unwrap(), source.immersed.as_ref().unwrap());
    for b in 0..3 {
        assert_eq!(im.bodies.get(b).position, src.bodies.get(b).position, "body {b} position");
        assert_eq!(im.bodies.get(b).velocity, src.bodies.get(b).velocity, "body {b} velocity");
    }
    assert_eq!((fresh.time, fresh.iteration), (source.time, source.iteration), "the clock");
}
