// =============================================================================
// torus_ledger_gates.rs
//
// the torus ledger driver (examples/torus_ledger.rs) from the captured cold surface state:
// - the production driver reaches the same bits, so the hooks add no numerics
// - the ledger closes mass, angular momentum and E + rho Phi against its boundary transport, with
//   transport through every surface
// - a launch stopped by a step or wall budget is incomplete, and its resumption is bit-identical to
//   the uninterrupted run, rows and ledger included
// - the failing update ends as a failure with the failing cell, separately from a budget
// - rows land on exact multiples of the sample interval, and every-stage minima bound the samples
// - the extremes scan locates planted values and rejects non-finite ones
//
// run: cargo test -p symbi --test torus_ledger_gates
// =============================================================================

#[allow(dead_code)]
#[path = "../examples/torus_ledger.rs"]
mod driver;

use std::path::{Path, PathBuf};

use driver::*;
use symbi::sim::evolve::evolve;
use symbi_sim::checkpoint::write_checkpoint;
use symbi_sim::substrate_seam::KernelSet;

const FIXTURE: &[u8] = include_bytes!("fixtures/cold_surface_torus_84x44_step2000.bin");
const NI: usize = 84;
const NJ: usize = 44;
const STEP0: u64 = 2000;
const SPAN: f64 = 2e-4;

fn fixture_time() -> f64 {
    f64::from_le_bytes(FIXTURE[..8].try_into().unwrap())
}

fn captured() -> (Sim, Kern) {
    let (mut sim, kernels) = build(NI, NJ);
    let cells: Vec<[isize; 2]> = (0..NJ as isize).flat_map(|j| (0..NI as isize).map(move |i| [i, j])).collect();
    let fill = |f: &symbi_grid::Field<f64, 2, symbi_xpu::HostMemory>, index: usize| {
        let start = 8 + index * NI * NJ * 8;
        let values = FIXTURE[start..start + NI * NJ * 8].chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap()));
        for (c, v) in cells.iter().zip(values) {
            f.view_mut().set(*c, v);
        }
    };
    fill(&sim.fields.cons.den, 0);
    for k in 0..3 {
        fill(&sim.fields.cons.mom[k], k + 1);
    }
    fill(sim.fields.cons.nrg_field().expect("cons.nrg"), 4);
    sim.time = fixture_time();
    sim.iteration = STEP0;
    KernelSet::c2p(&kernels, &sim);
    KernelSet::ghost_fill(&kernels, &sim);
    (sim, kernels)
}

/// a directory unique to this test, emptied, holding the seed checkpoint
fn workspace(name: &str) -> (PathBuf, PathBuf) {
    let root = std::env::temp_dir().join(format!("torus_ledger_gates_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("the test directory");
    let seed = root.join("seed.h5");
    let (sim, _) = captured();
    write_checkpoint(&sim, seed.to_str().unwrap(), &symbi_io::Metadata::new()).expect("the seed checkpoint");
    (root, seed)
}

fn args(root: &Path, seed: &Path, run: &str, t_end: f64, sample_dt: f64, max_steps: u64, max_wall_seconds: f64) -> Args {
    Args {
        seed: Some(seed.to_path_buf()),
        out: root.join(run),
        ni: NI,
        nj: NJ,
        bounds: Bounds { t_end, sample_dt, snapshot_every: 1, max_steps, max_wall_seconds },
        revision: "gate".to_string(),
        resume: false,
        cell_work: false,
    }
}

fn on_a_big_stack<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new().stack_size(512 * 1024 * 1024).spawn(work).unwrap().join().unwrap()
}

/// the interior conserved state of a checkpoint
fn state_of(path: &Path) -> Vec<Vec<f64>> {
    let (mut sim, _) = build(NI, NJ);
    symbi_sim::checkpoint::load_checkpoint(&mut sim, path.to_str().unwrap()).expect("a checkpoint");
    interior(&sim)
}

fn interior(sim: &Sim) -> Vec<Vec<f64>> {
    let read = |f: &symbi_grid::Field<f64, 2, symbi_xpu::HostMemory>| -> Vec<f64> {
        let view = f.view();
        (0..NJ as isize).flat_map(|j| (0..NI as isize).map(move |i| [i, j])).map(|c| *view.at(c)).collect()
    };
    let c = &sim.fields.cons;
    vec![read(&c.den), read(&c.mom[0]), read(&c.mom[1]), read(&c.mom[2]), read(c.nrg_field().unwrap())]
}

fn scalar_after(line: &str, key: &str) -> f64 {
    let at = line.find(&format!("\"{key}\":")).unwrap_or_else(|| panic!("no {key} in {line}")) + key.len() + 3;
    let rest = &line[at..];
    let end = rest.find([',', '}']).unwrap();
    rest[..end].parse().unwrap_or_else(|e| panic!("{key}: {e}: {}", &rest[..end]))
}

#[test]
fn the_ledger_driver_reaches_the_production_drivers_bits() {
    on_a_big_stack(|| {
        let (root, seed) = workspace("production");
        let t_end = fixture_time() + SPAN;
        let ending = run(&args(&root, &seed, "run", t_end, 1.0, u64::MAX, 1e9));
        assert_eq!(ending, Ending::Complete);
        let ledgered = state_of(&root.join("run/restart.h5"));
        let (mut sim, kernels) = captured();
        evolve(&mut sim, &kernels, t_end).expect("the production driver");
        assert!(sim.iteration > STEP0 + 10, "the comparison window held {} steps", sim.iteration - STEP0);
        let produced = interior(&sim);
        for (field, (a, b)) in ledgered.iter().zip(&produced).enumerate() {
            assert_eq!(a, b, "field {field} left the production driver's bits");
        }
        std::fs::remove_dir_all(root).unwrap();
    });
}

#[test]
fn the_ledger_closes_against_its_boundary_transport() {
    on_a_big_stack(|| {
        let (root, _) = workspace("closure");
        let t_end = fixture_time() + 10.0 * SPAN;
        let (mut sim, kernels) = captured();
        let mesh = Mesh::new(NI, NJ);
        let totals = totals_of(&sim.fields, &mesh);
        let mut ledger = Ledger { start: [totals[0], totals[1], totals[2]], sample: (sim.time / 1e-4).floor() as u64, ..Ledger::default() };
        let bounds = Bounds { t_end, sample_dt: 1e-4, snapshot_every: 1000, max_steps: u64::MAX, max_wall_seconds: 1e9 };
        let out = root.join("run");
        std::fs::create_dir_all(&out).unwrap();
        let launch = Launch { out: &out, bounds: &bounds, revision: "gate" };
        assert_eq!(march(&mut sim, &kernels, &mesh, &mut ledger, &launch), Ending::Complete);
        let closure = closure_of(&sim.fields, &mesh, &ledger);
        eprintln!("closure over {} steps: {closure:?}; shells at {:?}", sim.iteration - STEP0, mesh.shells);
        // every total closes at roundoff of its accumulated sums: at most N * 2.2e-16 relative over N steps
        let bound = (sim.iteration - STEP0) as f64 * 2.2e-16;
        assert!(closure.iter().all(|c| c.abs() < bound), "closure {closure:?} against {bound:e}");
        // the premise: gas crosses every surface, in the two directions the gross columns separate
        for (slot, row) in ledger.transport.iter().enumerate() {
            assert!(row[1] + row[2] > 0.0, "surface {slot}: no mass crossed");
            assert!((row[0] - (row[2] - row[1])).abs() <= 1e-12 * (row[1] + row[2]), "surface {slot}: the gross columns do not sum to the net");
        }
        assert!(ledger.transport[1][1] > 0.0 && ledger.transport[1][2] > 0.0, "the shell at r = 1 saw one direction only");
        // the shell radii are faces of every nested grid
        let fine = Mesh::new(2 * NI, 2 * NJ);
        for (coarse, nested) in mesh.shells.iter().zip(&fine.shells) {
            assert_eq!(nested.0, 2 * coarse.0);
            assert!((nested.1 - coarse.1).abs() < 1e-13, "the shell radii differ: {} against {}", coarse.1, nested.1);
        }
        let rows = std::fs::read_to_string(out.join("rows.jsonl")).unwrap();
        for line in rows.lines() {
            let t = scalar_after(line, "t");
            let sample = scalar_after(line, "sample");
            if t < t_end {
                assert_eq!(t, sample * 1e-4, "a row left its multiple of the sample interval");
            }
            let every_stage = line.find("\"every_stage\"").unwrap();
            let sampled = scalar_after(&line[..every_stage], "value");
            let bounded = scalar_after(&line[every_stage..], "value");
            assert!(bounded <= sampled, "the every-stage rho minimum {bounded:e} exceeds the sampled {sampled:e}");
        }
        assert!(rows.lines().count() >= 10, "{} rows", rows.lines().count());
        std::fs::remove_dir_all(root).unwrap();
    });
}

#[test]
fn a_budget_is_incomplete_and_its_resumption_is_bit_identical() {
    on_a_big_stack(|| {
        let (root, seed) = workspace("resume");
        let t_end = fixture_time() + SPAN;
        let sample = 2e-5;
        let straight = args(&root, &seed, "straight", t_end, sample, u64::MAX, 1e9);
        assert_eq!(run(&straight), Ending::Complete);
        // a step budget mid-sample, then the rest
        let mut split = args(&root, &seed, "split", t_end, sample, STEP0 + 17, 1e9);
        let stopped = run(&split);
        assert_eq!(stopped, Ending::Budget("step"));
        assert_eq!(stopped.exit_code(), EXIT_BUDGET);
        let status = std::fs::read_to_string(split.out.join("status.json")).unwrap();
        assert!(status.contains("\"state\":\"incomplete: step budget\""), "{status}");
        assert!(scalar_after(&status, "time") < t_end, "the budget stopped after the target: {status}");
        split.resume = true;
        // a resumption with its wall budget already spent takes no step and changes nothing
        split.bounds.max_wall_seconds = 0.0;
        split.bounds.max_steps = u64::MAX;
        let held = state_of(&split.out.join("restart.h5"));
        assert_eq!(run(&split), Ending::Budget("wall"));
        assert_eq!(state_of(&split.out.join("restart.h5")), held, "a spent wall budget moved the state");
        split.bounds.max_wall_seconds = 1e9;
        assert_eq!(run(&split), Ending::Complete);
        assert_eq!(state_of(&split.out.join("restart.h5")), state_of(&straight.out.join("restart.h5")), "the resumed run left the straight run's bits");
        // the rows differ only in their wall clock
        let wall_free = |path: &Path| -> Vec<String> {
            std::fs::read_to_string(path).unwrap().lines().map(|line| {
                let at = line.find("\"wall\":").unwrap();
                let end = at + line[at..].find(',').unwrap();
                format!("{}{}", &line[..at], &line[end..])
            }).collect()
        };
        let (a, b) = (wall_free(&split.out.join("rows.jsonl")), wall_free(&straight.out.join("rows.jsonl")));
        assert!(a.len() >= 5, "{} rows", a.len());
        assert_eq!(a, b, "the resumed run's rows left the straight run's");
        std::fs::remove_dir_all(root).unwrap();
    });
}

#[test]
fn the_failing_update_is_a_failure_with_its_cell_and_budgets_are_not() {
    on_a_big_stack(|| {
        let (root, seed) = workspace("failure");
        let mut failing = args(&root, &seed, "failing", 0.0115, 1.0, u64::MAX, 1e9);
        failing.cell_work = true;
        let ending = run(&failing);
        let Ending::Failed(reason) = &ending else { panic!("the failing update ended as {ending:?}") };
        assert_eq!(ending.exit_code(), EXIT_FAILED);
        assert!(reason.contains("step 2239") && reason.contains("[61, "), "{reason}");
        let status = std::fs::read_to_string(failing.out.join("status.json")).unwrap();
        assert!(status.contains("\"state\":\"failed: "), "{status}");
        assert!(failing.out.join("failed_state.h5").exists(), "the failed state was not kept");
        std::fs::remove_dir_all(root).unwrap();
    });
}

#[test]
fn the_scan_locates_planted_extremes_and_rejects_non_finite_values() {
    on_a_big_stack(|| {
        let (sim, _) = captured();
        let mesh = Mesh::new(NI, NJ);
        let clean = extremes_of(&sim.fields, &mesh);
        assert_eq!(clean.rejected, None, "the captured state is admissible");
        let den = &sim.fields.cons.den;
        let inner = [40usize, 22usize];
        let held = *den.view().at([40, 22]);
        den.view_mut().set([40, 22], 1e-30);
        let planted = extremes_of(&sim.fields, &mesh);
        assert_eq!(planted.rho_min.cell, inner, "the planted density minimum");
        den.view_mut().set([40, 22], held);
        let nrg = sim.fields.cons.nrg_field().unwrap();
        let held = *nrg.view().at([75, 30]);
        let cell = [75usize, 30usize];
        assert!(mesh.r_c[75] >= SPLIT_RADIUS, "the planted cell lies outside the split radius");
        nrg.view_mut().set([75, 30], f64::NAN);
        assert_eq!(extremes_of(&sim.fields, &mesh).rejected, Some(cell), "NaN energy");
        nrg.view_mut().set([75, 30], f64::INFINITY);
        assert_eq!(extremes_of(&sim.fields, &mesh).rejected, Some(cell), "infinite energy");
        nrg.view_mut().set([75, 30], -1.0);
        assert_eq!(extremes_of(&sim.fields, &mesh).rejected, Some(cell), "negative internal energy");
        nrg.view_mut().set([75, 30], held);
        den.view_mut().set([40, 22], 0.0);
        assert_eq!(extremes_of(&sim.fields, &mesh).rejected, Some(inner), "zero density");
    });
}
