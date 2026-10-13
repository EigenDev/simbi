// =============================================================================
// stage_pipeline_law.rs
//
// the stage-sequence equality law: the uni-grid driver (`sim::evolve::step`,
// folding STAGE_PIPELINE) and the hierarchy driver (`Hierarchy::level_stage`,
// a hand-maintained copy) must issue the identical per-stage kernel-call
// sequence under every gate combination. the two sequences are hand-copies of
// one another, and a phase added to one but not the other runs silently
// short on whichever driver a run happens to take — the python frontend
// always drives the hierarchy, so such drift freezes the missing physics in
// every python run while uni-grid tests stay green.
//
// mechanism: a recording kernel set logs every trait-method call; both
// drivers advance the same tiny sim for the same steps; the logs must be
// equal (not merely same-set: ordering and per-stage multiplicity count) for
// all 2^3 combinations of (additive source, fofc, passive scalar). immersed
// bodies are excluded: their per-step machinery (penalize/feedback/motion)
// legitimately differs between drivers and is gated by its own equivalence
// suite (decomp_body_equivalence).
//
// run: cargo test -p symbi --test stage_pipeline_law
// =============================================================================

use std::sync::{Arc, Mutex};
use symbi_hydro::quantity::{Density, Pressure};

use symbi::sim::evolve::{KernelSet, evolve};
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
/// both drivers take identical steps. the post-step gates (a viscous transport,
/// an excision region needing `excise_passes` fill passes) are off unless
/// `with_post_step` turns them on.
#[derive(Clone)]
struct Recorder {
    log: Arc<Mutex<Vec<String>>>,
    additive: bool,
    fofc: bool,
    viscous: bool,
    excise_passes: usize,
}

/// the fill passes the recorded excision region needs; more than one, so the
/// per-pass multiplicity is part of the compared sequence
const EXCISE_PASSES: usize = 2;

impl Recorder {
    fn new(additive: bool, fofc: bool) -> Self {
        Self {
            log: Arc::new(Mutex::new(Vec::new())),
            additive,
            fofc,
            viscous: false,
            excise_passes: 0,
        }
    }
    /// the post-step operators gated on: the viscous transport and the excision passes
    fn with_post_step(mut self) -> Self {
        self.viscous = true;
        self.excise_passes = EXCISE_PASSES;
        self
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
    fn fofc(
        &self,
        _s: &Store,
        _dt: f64,
        _a0: f64,
        _ac: f64,
        _stage: u8,
        _redo_update: &mut dyn FnMut(),
    ) -> symbi_sim::substrate_seam::FofcReport {
        self.push("fofc");
        symbi_sim::substrate_seam::FofcReport::inactive()
    }
    fn fofc_active(&self) -> bool {
        self.fofc
    }
    fn stage_emf(&self, _s: &Store, _dt: f64, _stage: u8) {
        self.push("stage_emf");
    }
    fn axis_emf(&self, _s: &Store) {
        self.push("axis_emf");
    }
    fn stage_curl(&self, _s: &Store, _dt: f64, _stage: u8) {
        self.push("stage_curl");
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
    fn viscous(&self, _s: &Store, _dt: f64) {
        self.push("viscous");
    }
    fn is_viscous(&self) -> bool {
        self.viscous
    }
    fn excise_pass_count(&self, _s: &Store) -> usize {
        self.excise_passes
    }
    fn penalize(&self, _s: &Store, _dt: f64) {
        self.push("penalize");
    }
    fn body_feedback(&self, _s: &Store, _dt: f64) {
        self.push("body_feedback");
    }
    fn horizon_accretion(&self, _s: &Store, _r: f64) -> Option<symbi_sim::driver::HorizonPartial> {
        self.push("horizon_accretion");
        Some(symbi_sim::driver::HorizonPartial::from_rates(0.0, 0.0))
    }
    fn excise_sweep(&self, _s: &Store) {
        self.push("excise_sweep");
    }
    fn excise_finalize(&self, _s: &Store) {
        self.push("excise_finalize");
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
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0]), Pressure(1.0)))
        .build();
    if with_chi {
        sim.with_passive_scalar().expect("chi alloc")
    } else {
        sim
    }
}

// the per-step extras (viscous, excise, horizon ledger, penalize, feedback)
// under the same equality demand. an extra present in one driver only — a
// horizon-ledger booking, say — reports zero on every run that takes the other
// driver, and only a per-step sequence comparison catches it.
#[test]
fn both_drivers_issue_the_identical_per_step_sequence_with_bodies() {
    const T: f64 = 2.5e-3;
    // three body configurations: none; a one-way gravitational mass (feedback
    // reduction skipped by the needs_feedback gate on both drivers); a two-way
    // rigid body plus a GR horizon diagnostic body (penalize + feedback +
    // shell-flux ledger all active).
    let configs: Vec<(&str, Option<symbi_ib::BodyCollection<f64, 2>>)> = vec![
        ("none", None),
        (
            "one-way-grav",
            Some(
                symbi_ib::BodyCollection::new().add(symbi_ib::Body::gravitational(
                    0,
                    Tensor::new([0.0, 0.0]),
                    Tensor::zeros(),
                    1.0,
                    0.1,
                    0.05,
                )),
            ),
        ),
        (
            "two-way-rigid+horizon",
            Some(
                symbi_ib::BodyCollection::new()
                    .add(
                        symbi_ib::Body::rigid_sphere(
                            0,
                            Tensor::new([0.0, 0.0]),
                            Tensor::zeros(),
                            1.0,
                            0.2,
                            0.05,
                            true,
                        )
                        .with_two_way_coupling(true),
                    )
                    .add(symbi_ib::Body::horizon(1, 0.3, 0.6)),
            ),
        ),
    ];
    for (tag, bodies) in configs {
        let rec_a = Recorder::new(false, false).with_post_step();
        let mut sim_a = tiny_sim(false);
        if let Some(b) = bodies.clone() {
            sim_a = sim_a.with_bodies(b);
        }
        evolve(&mut sim_a, &rec_a, T).expect("uni-grid drive");
        let uni = rec_a.take();

        let rec_b = Recorder::new(false, false).with_post_step();
        let mut sim_b = tiny_sim(false);
        if let Some(b) = bodies.clone() {
            sim_b = sim_b.with_bodies(b);
        }
        let mut hier = Hierarchy::single(sim_b, rec_b.clone());
        hier.evolve(T).expect("hierarchy drive");
        let hi = rec_b.take();

        assert_eq!(
            uni, hi,
            "per-step sequences diverged for bodies = {tag}:\nuni-grid:  {uni:?}\nhierarchy: {hi:?}"
        );
        // both post-step operators issued, the excision with every one of its passes
        let count = |name: &str| uni.iter().filter(|p| *p == name).count();
        assert!(count("viscous") > 0, "the viscous gate is on but no viscous transport was issued");
        assert!(count("excise_finalize") > 0, "the excision gate is on but no excision was issued");
        assert_eq!(count("excise_sweep"), EXCISE_PASSES * count("excise_finalize"), "an excision ran short of its passes");
        // the excision closes every stage and the step: two stages and the post-step fold per
        // rk2 step, the per-step snapshot marking each step
        assert_eq!(count("excise_finalize"), 3 * count("snapshot"), "an excision is missing from a stage end or the step end");
        match tag {
            "none" => assert!(!uni.contains(&"penalize".to_string())),
            "one-way-grav" => {
                assert!(uni.contains(&"penalize".to_string()));
                assert!(
                    !uni.contains(&"body_feedback".to_string()),
                    "one-way mass must skip the feedback reduction"
                );
            }
            _ => {
                assert!(uni.contains(&"penalize".to_string()));
                assert!(uni.contains(&"body_feedback".to_string()));
                assert!(
                    uni.contains(&"horizon_accretion".to_string()),
                    "the horizon ledger must be booked on both drivers"
                );
            }
        }
    }
}

// the decomposed (gpus > 1) driver against the canonical sequence, modulo its
// three documented structural deltas — anything else is drift:
//   - no stage-input elision: snapshot_stage runs at every gated stage,
//     including stage 0 of a multi-stage scheme (the uni-grid driver elides
//     that copy into the per-step snapshot);
//   - one more ghost_fill closing every run of ghost_fills: the physical
//     refill after the halo exchange (cut-corner consistency);
//   - the excise protocol is sweep-based with its own equivalence oracle
//     (decomp_excise_equivalence), so excise-family calls are normalized out
//     of both logs here, with the halo refresh the decomposed driver issues
//     after each sweep and after the finalize.
// the passive scalar is config-fenced off this driver, so chi stays false.
#[test]
fn decomposed_driver_matches_the_canonical_sequence_modulo_documented_deltas() {
    use symbi::sim::decomp::{LocalCopy, evolve_decomposed};
    const T: f64 = 2.5e-3;
    for additive in [false, true] {
        for fofc in [false, true] {
            let rec_a = Recorder::new(additive, fofc);
            let mut sim_a = tiny_sim(false);
            evolve(&mut sim_a, &rec_a, T).expect("uni-grid drive");
            let uni = rec_a.take();

            // transform the canonical log by the documented deltas: drop the
            // excise family, close every ghost_fill run with the refill after the
            // exchange, and materialize the elided stage-0 snapshot_stage (it
            // directly follows the per-step snapshot in the canonical order).
            let gated = additive || fofc;
            let mut expected: Vec<String> = Vec::new();
            for ph in with_exchange_refills(&uni) {
                let snapshot = ph == "snapshot";
                expected.push(ph);
                if snapshot && gated {
                    expected.push("snapshot_stage".to_string());
                }
            }

            let rec_c = Recorder::new(additive, fofc);
            let mut sim_c = tiny_sim(false);
            let stores: &mut [&mut Store] = &mut [&mut sim_c];
            let kernels: Vec<&Recorder> = vec![&rec_c];
            evolve_decomposed(
                stores,
                &kernels,
                [1, 1],
                &[0],
                Timestepping::Rk2,
                0.0,
                T,
                u64::MAX,
                &LocalCopy,
                |_, _, _| std::ops::ControlFlow::Continue(()),
            );
            let dec: Vec<String> = rec_c
                .take()
                .into_iter()
                .filter(|p| !p.starts_with("excise"))
                .collect();

            assert_eq!(
                dec, expected,
                "decomposed sequence drifted from the canonical (additive={additive}, \
                 fofc={fofc}):\ndecomposed: {dec:?}\nexpected:   {expected:?}"
            );
        }
    }
}

// the decomposed driver's per-step extras (viscous / penalize / gated
// feedback) against the canonical order, with a two-way body active — the
// same transform as the stage law (excise family normalized: the decomposed
// sweep protocol has its own oracle; body motion is driver-level shared code,
// invisible to the recorder on every driver).
#[test]
fn decomposed_per_step_extras_match_the_canonical_order() {
    use symbi::sim::decomp::{LocalCopy, evolve_decomposed};
    const T: f64 = 2.5e-3;
    let bodies = || {
        symbi_ib::BodyCollection::new().add(
            symbi_ib::Body::rigid_sphere(
                0,
                Tensor::new([0.0, 0.0]),
                Tensor::zeros(),
                1.0,
                0.2,
                0.05,
                true,
            )
            .with_two_way_coupling(true),
        )
    };

    let rec_a = Recorder::new(false, false).with_post_step();
    let mut sim_a = tiny_sim(false).with_bodies(bodies());
    evolve(&mut sim_a, &rec_a, T).expect("uni-grid body drive");
    let mut expected: Vec<String> = Vec::new();
    for ph in with_exchange_refills(&rec_a.take()) {
        let snapshot = ph == "snapshot";
        expected.push(ph);
        // this sim carries bodies, and the immersed-body pass evaluates its contribution at
        // the stage input, so the snapshot is gated on. the decomposed driver does not elide
        // the stage-0 copy, so it materializes right after the per-step snapshot.
        if snapshot {
            expected.push("snapshot_stage".to_string());
        }
    }

    let rec_c = Recorder::new(false, false).with_post_step();
    let mut sim_c = tiny_sim(false).with_bodies(bodies());
    let stores: &mut [&mut Store] = &mut [&mut sim_c];
    let kernels: Vec<&Recorder> = vec![&rec_c];
    evolve_decomposed(
        stores,
        &kernels,
        [1, 1],
        &[0],
        Timestepping::Rk2,
        0.0,
        T,
        u64::MAX,
        &LocalCopy,
        |_, _, _| std::ops::ControlFlow::Continue(()),
    );
    let raw = rec_c.take();
    // the decomposed driver ran every excision pass the region asks for
    let count = |name: &str| raw.iter().filter(|p| *p == name).count();
    assert!(count("excise_finalize") > 0, "the excision gate is on but the decomposed driver issued no excision");
    assert_eq!(count("excise_sweep"), EXCISE_PASSES * count("excise_finalize"), "a decomposed excision ran short of its passes");
    let dec = without_excise_protocol(&raw);

    assert_eq!(
        dec, expected,
        "decomposed per-step extras drifted:\ndecomposed: {dec:?}\nexpected:   {expected:?}"
    );
    assert!(expected.contains(&"penalize".to_string()));
    assert!(expected.contains(&"body_feedback".to_string()));
    assert!(expected.contains(&"viscous".to_string()));
}

/// a decomposed log with the excise protocol normalized out: every excise-family call and
/// the one halo refresh (a ghost_fill) the protocol issues after each sweep and after the
/// finalize. the protocol is gated by its own equivalence oracle (decomp_excise_equivalence)
fn without_excise_protocol(log: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(log.len());
    let mut after_excise = false;
    for p in log {
        if p.starts_with("excise") {
            after_excise = true;
            continue;
        }
        if after_excise && p == "ghost_fill" {
            after_excise = false;
            continue;
        }
        after_excise = false;
        out.push(p.clone());
    }
    out
}

// law completeness: every phase in the canonical table must actually appear in
// the recorded logs when its gate is on. this closes the recorder-coupling
// gap: a phase added to STAGE_PIPELINE whose recorder method is missing would
// no-op identically on every driver — the equality laws would still pass while
// their coverage silently shrank. iterating the exported table makes that
// impossible: the new phase's name never appears and this test names it.
#[test]
fn every_canonical_phase_appears_in_the_recorded_union() {
    const T: f64 = 2.5e-3;
    let mut union: Vec<String> = Vec::new();

    // all scalar gates on at once (additive + fofc + chi)...
    let rec = Recorder::new(true, true);
    let mut sim = tiny_sim(true);
    evolve(&mut sim, &rec, T).expect("gated drive");
    union.extend(rec.take());

    // ...plus a body-bearing run for the Bodies gate, with the post-step gates on.
    let rec = Recorder::new(false, false).with_post_step();
    let mut sim = tiny_sim(false).with_bodies(symbi_ib::BodyCollection::new().add(
        symbi_ib::Body::gravitational(0, Tensor::new([0.0, 0.0]), Tensor::zeros(), 1.0, 0.1, 0.05),
    ));
    evolve(&mut sim, &rec, T).expect("body drive");
    union.extend(rec.take());

    for ph in symbi_sim::stage::STAGE_PIPELINE {
        // the flux phase logs per direction ("flux0", "flux1"); every other
        // phase logs its table name verbatim.
        let present = union.iter().any(|e| e == ph.name || e.starts_with(ph.name));
        assert!(
            present,
            "canonical phase '{}' never appeared in any recorded log — the law's \
             recorder is missing its method, so sequence equality no longer covers it",
            ph.name,
        );
    }
    // the post-step table under the same demand: the excise phase logs its passes and
    // its finalize ("excise_sweep", "excise_finalize"); the viscous phase its table name
    for ph in symbi_sim::stage::POST_STEP_PIPELINE {
        let present = union.iter().any(|e| e.starts_with(ph.name));
        assert!(
            present,
            "post-step phase '{}' never appeared in any recorded log — its gate is off in \
             the recorder or its method is missing, so sequence equality no longer covers it",
            ph.name,
        );
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

/// a canonical log with the excise family dropped and every run of consecutive ghost_fills
/// closed by one more: the physical refill the decomposed driver runs after each halo exchange
fn with_exchange_refills(log: &[String]) -> Vec<String> {
    let kept: Vec<&String> = log.iter().filter(|p| !p.starts_with("excise")).collect();
    let mut out = Vec::with_capacity(kept.len() * 2);
    for (ii, p) in kept.iter().enumerate() {
        out.push((*p).clone());
        let run_ends = kept.get(ii + 1).is_none_or(|next| *next != "ghost_fill");
        if *p == "ghost_fill" && run_ends {
            out.push("ghost_fill".to_string());
        }
    }
    out
}
