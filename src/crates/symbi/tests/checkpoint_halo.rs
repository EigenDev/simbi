// =============================================================================
// checkpoint_halo.rs
//
// a checkpoint's conserved halo is initialized the same way on every driver:
// each ghost cell holds the conserved state of the primitives the boundary and
// exchange machinery filled there. no stage reads the conserved halo, so before
// this a single grid published zeros on its periodic seam while tiles published
// their prime's exchange. the halo is prepared for the write and restored after
// it, so:
// - the file's conserved halo on a periodic single grid is the conversion of
//   its halo primitives, nonzero;
// - the state the run holds after the write is the state it held before, bit
//   for bit, interior and halo;
// - a run that checkpoints part way evolves exactly as one that does not.
//
// run: cargo test -p symbi --test checkpoint_halo
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_io::{Hdf5Backend, IoBackend};
use symbi_refinement::refinement::Hierarchy;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
type Hier = Hierarchy<Newtonian, 2, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kern>;

const N: usize = 16;

fn hier() -> Hier {
    let sim = Sim::build(Newtonian, IdealGas { gamma: 1.4 }, Cartesian)
        .cells([N, N])
        .origin([0.0; 2])
        .spacing([1.0 / N as f64; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.4)
        .allocate()
        .expect("sim")
        .set_initial(|x| {
            let s = (2.0 * std::f64::consts::PI * x[0]).sin();
            Prim::adiabatic(Density(1.0 + 0.3 * s), Tensor::new([0.5, 0.2 * s]), Pressure(1.0))
        })
        .build();
    let k = Kern::new(1.4, 0.4, &sim.geom.allocated);
    let mut h = Hier::single(sim, k);
    h.prime();
    h
}

/// every conserved component's bits over the allocated cells
fn conserved(h: &Hier) -> Vec<u64> {
    let s = &h.levels[0].state;
    let mut fields = vec![&s.fields.cons.den];
    fields.extend(s.fields.cons.mom.iter());
    fields.extend(s.fields.cons.nrg_field());
    fields.iter().flat_map(|f| s.geom.allocated.iter().map(|c| f.view().at(c).to_bits()).collect::<Vec<_>>()).collect()
}

fn write(h: &Hier, path: &std::path::Path) {
    let levels: Vec<&Sim> = h.levels.iter().map(|l| &l.state).collect();
    symbi_sim::checkpoint::write_hierarchy_checkpoint(&levels, path.to_str().unwrap(), &Default::default()).expect("checkpoint");
}

#[test]
fn the_conserved_halo_is_the_conversion_of_the_halo_primitives() {
    let mut h = hier();
    h.evolve_steps(3).expect("evolve");
    let dir = std::env::temp_dir().join(format!("checkpoint_halo_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("chkpt.h5");
    let before = conserved(&h);
    write(&h, &path);
    assert_eq!(conserved(&h), before, "the write changed the state the run holds");

    let tree = Hdf5Backend.read(&path).expect("read back");
    let _ = std::fs::remove_dir_all(&dir);
    let den = tree.find_group("level_0").and_then(|g| g.find_group("conserved")).and_then(|g| g.find_dataset("den")).expect("conserved den");
    let den = den.data.as_f64().expect("f64");
    // the file holds the allocated grid, halo included, in storage order; its ghost cells hold
    // the conversion of the halo primitives, which on the periodic seam is the image's state
    let s = &h.levels[0].state;
    let interior = &s.geom.interior;
    let rho = &s.fields.prim.rho;
    let (mut ghosts, mut matched) = (0, 0);
    // storage order: axis 0 fastest
    let a = &s.geom.allocated;
    let cells = (a.spaces[1].lo..a.spaces[1].hi).flat_map(|j| (a.spaces[0].lo..a.spaces[0].hi).map(move |i| [i, j]));
    for (n, c) in cells.enumerate() {
        if interior.contains(c) {
            continue;
        }
        ghosts += 1;
        assert!(den[n] != 0.0, "ghost {c:?} holds zero density in the file");
        // the newtonian conserved density is the primitive density
        if den[n].to_bits() == rho.view().at(c).to_bits() {
            matched += 1;
        }
    }
    assert!(ghosts > 0, "the grid has no halo");
    assert_eq!(matched, ghosts, "the file's conserved halo differs from its halo primitives");
}

#[test]
fn a_run_that_checkpoints_evolves_as_one_that_does_not() {
    let mut plain = hier();
    plain.evolve_steps(6).expect("evolve");
    let mut written = hier();
    written.evolve_steps(3).expect("evolve");
    let dir = std::env::temp_dir().join(format!("checkpoint_halo_run_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    write(&written, &dir.join("chkpt.h5"));
    let _ = std::fs::remove_dir_all(&dir);
    written.evolve_steps(3).expect("evolve");
    assert_eq!(conserved(&written), conserved(&plain), "checkpointing changed the run");
}
