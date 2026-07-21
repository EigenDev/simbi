// =============================================================================
// stage_pipeline_law.rs
//
// the STAGE-SEQUENCE EQUALITY LAW: the uni-grid driver (`sim::evolve::step`,
// folding STAGE_PIPELINE) and the hierarchy driver (`Hierarchy::level_stage`,
// a hand-maintained copy) must issue the IDENTICAL per-stage kernel-call
// sequence under every gate combination. the two sequences are hand-copies of
// one another, and a phase added to one but not the other runs silently
// short on whichever driver a run happens to take — the python frontend
// always drives the hierarchy, so such drift freezes the missing physics in
// every python run while uni-grid tests stay green.
//
// mechanism: a recording kernel set logs every trait-method call; both
// drivers advance the same tiny sim for the same steps; the logs must be
// EQUAL (not merely same-set: ordering and per-stage multiplicity count) for
// all 2^3 combinations of (additive source, fofc, passive scalar). immersed
// bodies are excluded: their per-STEP machinery (penalize/feedback/motion)
// legitimately differs between drivers and is gated by its own equivalence
// suite (decomp_body_equivalence).
//
// run: cargo test -p symbi --test stage_pipeline_law
// =============================================================================

use std::sync::{Arc, Mutex};

use symbi::sim::evolve::{evolve, KernelSet};
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const N: usize = 8;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Store = FieldStore<2, 2, HostMemory, f64>;

/// records every kernel-set call in order; touches no fields. `dt` is fixed so
/// both drivers take identical steps.
#[derive(Clone)]
struct Recorder {
    log: Arc<Mutex<Vec<String>>>,
    additive: bool,
    fofc: bool,
}

impl Recorder {
    fn new(additive: bool, fofc: bool) -> Self {
        Self { log: Arc::new(Mutex::new(Vec::new())), additive, fofc }
    }
    fn push(&self, s: &str) {
        self.log.lock().unwrap().push(s.to_string());
    }
    fn take(&self) -> Vec<String> {
        std::mem::take(&mut *self.log.lock().unwrap())
    }
}

impl KernelSet<2, 2, HostMemory, f64> for Recorder {
    fn flux(&self, _s: &Store, dir: usize) {
        self.push(&format!("flux{dir}"));
    }
    fn c2p(&self, _s: &Store) {
        self.push("c2p");
    }
    fn godunov_stage(&self, _s: &Store, _dt: f64, _a0: f64, _ac: f64) {
        self.push("godunov_stage");
    }
    fn cfl(&self, _s: &Store) -> f64 {
        1e-3
    }
    fn ghost_fill(&self, _s: &Store) {
        self.push("ghost_fill");
    }
    fn snapshot(&self, _s: &Store) {
        self.push("snapshot");
    }
    fn fofc(&self, _s: &Store, _dt: f64, _a0: f64, _ac: f64, _stage: u8) {
        self.push("fofc");
    }
    fn fofc_active(&self) -> bool {
        self.fofc
    }
    fn post_godunov(&self, _s: &Store, _dt: f64, _stage: u8) {
        self.push("post_godunov");
    }
    fn efield(&self, _s: &Store) {
        self.push("efield");
    }
    fn wave_speeds(&self, _s: &Store) {
        self.push("wave_speeds");
    }
    fn body_source(&self, _s: &Store, _dt: f64) {
        self.push("body_source");
    }
    fn has_additive_source(&self) -> bool {
        self.additive
    }
    fn snapshot_stage(&self, _s: &Store) {
        self.push("snapshot_stage");
    }
    fn source_apply(&self, _s: &Store, _w: f64) {
        self.push("source_apply");
    }
    fn chi_update(&self, _s: &Store, _dt: f64, _a0: f64, _ac: f64) {
        self.push("chi_update");
    }
}

fn tiny_sim(with_chi: bool) -> Sim {
    let dx = 2.0 / N as f64;
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([-1.0, -1.0])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim")
        .set_initial(|_| Prim { rho: 1.0, vel: Tensor::new([0.0, 0.0]), pre: 1.0 })
        .build();
    if with_chi {
        sim.with_passive_scalar().expect("chi alloc")
    } else {
        sim
    }
}

#[test]
fn both_drivers_issue_the_identical_stage_sequence() {
    // fixed dt = 1e-3, t_final = 2.5e-3 -> the same 3 steps on both drivers.
    const T: f64 = 2.5e-3;
    for additive in [false, true] {
        for fofc in [false, true] {
            for chi in [false, true] {
                let rec_a = Recorder::new(additive, fofc);
                let mut sim_a = tiny_sim(chi);
                evolve(&mut sim_a, &rec_a, T).expect("uni-grid drive");
                let uni = rec_a.take();

                let rec_b = Recorder::new(additive, fofc);
                let sim_b = tiny_sim(chi);
                let mut hier = Hierarchy::single(sim_b, rec_b.clone());
                hier.evolve(T).expect("hierarchy drive");
                let hi = rec_b.take();

                assert_eq!(
                    uni, hi,
                    "stage sequences diverged at (additive={additive}, fofc={fofc}, \
                     chi={chi}):\nuni-grid:  {uni:?}\nhierarchy: {hi:?}"
                );
                // and the sequence must actually contain the gated phases when
                // their gates are on — equality of two empty logs proves nothing.
                assert!(uni.contains(&"godunov_stage".to_string()));
                if chi {
                    assert!(
                        uni.contains(&"chi_update".to_string()),
                        "chi gate on but chi_update never issued"
                    );
                }
                if fofc {
                    assert!(uni.contains(&"fofc".to_string()));
                }
                if additive {
                    assert!(uni.contains(&"source_apply".to_string()));
                }
            }
        }
    }
}
