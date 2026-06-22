// =============================================================================
// examples/common.rs
//
// shared scaffolding for the validation-problem examples (kepler, sod, sedov,
// mignone_bodo, marti_muller). owns:
//   - `BaseCli` — flexible parser for the knobs every example accepts (grid
//     size, end time, checkpoint count, CFL, EOS gamma, timestepping, out
//     dir); each example layers problem-specific args on top via `--key val`
//     pass-through retrievable as `cli.get("key")`.
//   - `run_simulation` — generic over the kernel-set type. takes the SimState
//     + kernel set + cadence + output dir, runs the production evolve loop
//     writing HDF5 checkpoints, prints a one-line progress log per snapshot.
//
// each example is then ~80 focused lines: parse CLI extras, build initial
// state, declare the kernel set, hand off to `run_simulation`. zero copy-paste
// of the evolve+checkpoint loop across problems.
// =============================================================================

#![allow(dead_code)] // some helpers are used by only a subset of examples

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use symbi::sim::refinement::Hierarchy;
use symbi::sim::checkpoint::write_checkpoint;
use symbi::sim::evolve::{evolve_with_callback, KernelSet};
use symbi::sim::state::*;
use symbi_geometry::Metric;
use symbi_hydro::eos::Eos;
pub use symbi_io::{display_tree_buf, IoBackend, Metadata};
use symbi_hydro::regime::Regime;
use symbi_xpu::{ExecutionSpace, MemorySpace};

pub use symbi::regimes::substrate_kernels::Solver;

/// the display label for the chosen Riemann solver — the single source for the live
/// progress table's "solver" row (replaces the retired `SYMBI_RIEMANN` env read).
pub fn solver_label(solver: Solver) -> &'static str {
    match solver {
        Solver::Hlle => "HLLE",
        Solver::Hllc => "HLLC",
        Solver::Hlld => "HLLD",
    }
}

pub mod progress;
use progress::{build_problem_setup_rows, build_system_info_rows};

// =============================================================================
// CLI parsing — flexible, parser combinators not needed for this many knobs.
// =============================================================================

/// the shared CLI knobs every example accepts. problem-specific args ride on
/// `extras` (a HashMap parsed from any unrecognized `--key val` pair); the
/// example pulls them out by name with the typed `extra_*` accessors.
#[derive(Debug)]
pub struct BaseCli {
    /// per-axis grid resolution. 1D problems use `cli.n1()`, 2D `cli.n2()`,
    /// 3D `cli.n3()`. `--n 256` sets every axis; `--n 256,128` sets per-axis.
    pub n:             Vec<usize>,
    /// end time in PROBLEM units. for kepler, multiplied by TAU inside the
    /// example; for hydro shocks, used directly.
    pub end_time:      f64,
    /// total number of checkpoints to write (uniformly spaced over t_final).
    /// 0 = no intermediate, only initial + final. defaults to 10.
    pub n_checkpoints: usize,
    /// CFL number. defaults to 0.4.
    pub cfl:           f64,
    /// EOS gamma (adiabatic / SRHD); ignored for iso. defaults to per-problem.
    pub gamma:         Option<f64>,
    /// `--timestepping rk2` (default) or `--timestepping euler`.
    pub timestepping:  Timestepping,
    /// PLM slope-limiter steepness `theta` in `[1, 2]`: 1 = minmod (most
    /// diffusive), 2 = MC (sharpest, least diffusive). `--theta`, default 1.5.
    /// the example forwards it via `.theta(cli.theta)` on its kernel set.
    pub theta:         f64,
    /// `--solver hlle|hllc|hlld`. defaults to HLLE. iso is HLLE-only by
    /// physics (no contact wave); HLLD is RMHD-only (5-wave). the example
    /// is responsible for forwarding `cli.solver` into its kernel set via
    /// `.with_solver(cli.solver)` and for rejecting invalid combinations.
    pub solver:        Solver,
    /// output directory for HDF5 snapshots; `--out path/`. defaults to
    /// `output/{problem}/data` (the example supplies `{problem}`).
    pub out_dir:       PathBuf,
    /// problem-specific args; populated from any unrecognized `--key val`
    /// pair so each example can declare its own without parser changes here.
    /// access via `cli.extra_f64("rho_left")` / `cli.extra_int(..)` / etc.
    pub extras:        HashMap<String, String>,
}

impl BaseCli {
    /// parse argv with defaults; the `default_problem_dir` argument names
    /// the subfolder under `output/` if the user does not pass `--out`.
    pub fn parse(default_problem_dir: &str) -> Self {
        let mut n: Vec<usize> = vec![64];
        let mut end_time: f64 = 1.0;
        let mut n_checkpoints: usize = 10;
        let mut cfl: f64 = 0.4;
        let mut gamma: Option<f64> = None;
        let mut timestepping = Timestepping::Rk2;
        let mut theta: f64 = 1.5;
        let mut solver = Solver::Hlle;
        let mut out_dir = PathBuf::from(format!("output/{default_problem_dir}/data"));
        let mut extras: HashMap<String, String> = HashMap::new();
        let mut args = std::env::args().skip(1);
        while let Some(a) = args.next() {
            match a.as_str() {
                "--n" => {
                    let v = args.next().expect("--n requires value");
                    n = v.split(',').map(|s| s.trim().parse().expect("--n: usize"))
                        .collect();
                }
                "--end-time" | "--t-final" => {
                    end_time = args.next().expect("--end-time requires value")
                        .parse().expect("--end-time: f64");
                }
                "--n-checkpoints" => {
                    n_checkpoints = args.next().expect("--n-checkpoints requires value")
                        .parse().expect("--n-checkpoints: usize");
                }
                "--cfl" => cfl = args.next().expect("--cfl requires value").parse().unwrap(),
                "--theta" => theta = args.next().expect("--theta requires value").parse().expect("--theta: f64"),
                "--gamma" => gamma = Some(args.next().expect("--gamma requires value").parse().unwrap()),
                "--timestepping" => {
                    let v = args.next().expect("--timestepping requires euler|rk2|rk3");
                    timestepping = match v.as_str() {
                        "euler" | "Euler" => Timestepping::Euler,
                        "rk2"   | "Rk2"   => Timestepping::Rk2,
                        "rk3"   | "Rk3"   => Timestepping::Rk3,
                        o => panic!("--timestepping: unknown '{o}' (expect euler|rk2|rk3)"),
                    };
                }
                "--solver" => {
                    let v = args.next().expect("--solver requires hlle|hllc|hlld");
                    solver = match v.as_str() {
                        "hlle" | "Hlle" | "HLLE" => Solver::Hlle,
                        "hllc" | "Hllc" | "HLLC" => Solver::Hllc,
                        "hlld" | "Hlld" | "HLLD" => Solver::Hlld,
                        o => panic!("--solver: unknown '{o}' (expect hlle|hllc|hlld)"),
                    };
                }
                "--out" => out_dir = PathBuf::from(args.next().expect("--out requires value")),
                "-h" | "--help" => {
                    eprintln!("usage: <example> [base + extra args]");
                    eprintln!("  --n <N | Nx,Ny,Nz>           grid resolution (per axis)");
                    eprintln!("  --end-time <t>               simulation end time (problem units)");
                    eprintln!("  --n-checkpoints <count>      number of snapshots to write");
                    eprintln!("  --cfl <c>                    CFL number (default 0.4)");
                    eprintln!("  --gamma <g>                  EOS gamma (problem default)");
                    eprintln!("  --timestepping <euler|rk2>   integrator (default rk2)");
                    eprintln!("  --theta <t>                  PLM limiter steepness, [1,2] (default 1.5)");
                    eprintln!("  --solver <hlle|hllc|hlld>    Riemann solver (default hlle)");
                    eprintln!("                                  iso: HLLE only; HLLD: RMHD only");
                    eprintln!("  --out <path>                 HDF5 output directory");
                    eprintln!("  --<key> <value>              problem-specific extras");
                    std::process::exit(0);
                }
                _ => {
                    // unknown `--key val` ride on `extras` so each example
                    // can declare its own knobs without rewriting this parser.
                    if let Some(key) = a.strip_prefix("--") {
                        let val = args.next().unwrap_or_else(
                            || panic!("--{key} requires a value"));
                        extras.insert(key.to_string(), val);
                    } else {
                        panic!("unknown positional arg '{a}' (expected --key value)");
                    }
                }
            }
        }
        Self { n, end_time, n_checkpoints, cfl, gamma, timestepping, theta, solver, out_dir, extras }
    }

    pub fn n1(&self) -> usize { self.n[0] }
    pub fn n2(&self) -> [usize; 2] {
        [self.n[0], self.n.get(1).copied().unwrap_or(self.n[0])]
    }
    pub fn n3(&self) -> [usize; 3] {
        [self.n[0], self.n.get(1).copied().unwrap_or(self.n[0]),
            self.n.get(2).copied().unwrap_or(self.n[0])]
    }

    pub fn extra_f64(&self, key: &str, default: f64) -> f64 {
        self.extras.get(key)
            .map(|s| s.parse::<f64>().unwrap_or_else(|_| panic!("--{key}: f64 parse failed on '{s}'")))
            .unwrap_or(default)
    }
    pub fn extra_int(&self, key: &str, default: i64) -> i64 {
        self.extras.get(key)
            .map(|s| s.parse::<i64>().unwrap_or_else(|_| panic!("--{key}: int parse failed on '{s}'")))
            .unwrap_or(default)
    }
    pub fn extra_str<'a>(&'a self, key: &str, default: &'a str) -> &'a str {
        self.extras.get(key).map(String::as_str).unwrap_or(default)
    }
}

// =============================================================================
// the evolve+checkpoint loop — one chokepoint for every problem.
// =============================================================================

/// drive `evolve_with_callback` writing HDF5 checkpoints at uniform `t_final /
/// n_checkpoints` intervals. writes a `{name}_0000.h5` baseline frame BEFORE
/// the loop and a `{name}_final.h5` snapshot AFTER, so the plot scripts always
/// have an initial + terminal frame. `t_unit` is the "natural" time scale
/// (TAU for kepler-like orbital problems; 1.0 for everything else) used in
/// the progress log: `t/T_unit`.
#[allow(clippy::too_many_arguments)]
pub fn run_simulation<R, const D: usize, const DOF: usize, M, E, S, Mem, K>(
    sim:           &mut SimStateGeneric<R, D, DOF, M, E, S, Mem, f64>,
    kernels:       &K,
    t_final:       f64,
    n_checkpoints: usize,
    out_dir:       &Path,
    name:          &str,
    metadata:      &Metadata,
    t_unit:        f64,
    t_unit_label:  &str,
    solver_label:  &str,
) -> Result<(), Box<dyn std::error::Error>>
where
    R:   Regime<f64, D>,
    M:   Metric<f64, D> + Copy + Send + Sync,
    E:   Eos<f64> + Send + Sync,
    S:   ExecutionSpace,
    Mem: MemorySpace + Sync,
    K:   KernelSet<D, DOF, Mem, f64>,
{
    fs::create_dir_all(out_dir)?;

    // install the cursor guard FIRST — before the Table hides the cursor.
    // ensures Ctrl-C / kill / SIGHUP / panic-with-abort all restore the
    // terminal cursor; Table::Drop covers the normal-exit + unwind path.
    symbi_display::term_guard::install();

    // ---- the LIVE PROGRESS WIDGET. one Table per example run carries the
    //      system info, problem setup, benchmark row, progress bar, and the
    //      message board (checkpoint events). every refresh is in-place when
    //      stdout is a TTY; falls back to non-clearing prints under redirect.
    let total_cells = sim.geom.interior.volume() as u64;
    let recon_label = std::env::var("SYMBI_RECON").unwrap_or_else(|_| "θ-MC PLM".into());
    let mut table = symbi_display::Table::new(
        &format!("symbi · {name}"), /*dynamic=*/ true,
    );
    {
        let sys_rows = build_system_info_rows();
        let sys_ref: Vec<[&str; 3]> = sys_rows.iter()
            .map(|r| [r[0].as_str(), r[1].as_str(), r[2].as_str()]).collect();
        table.set_system_info(&sys_ref);

        let setup_rows = build_problem_setup_rows(sim, solver_label, &recon_label);
        let setup_ref: Vec<[&str; 3]> = setup_rows.iter()
            .map(|r| [r[0].as_str(), r[1].as_str(), r[2].as_str()]).collect();
        table.set_problem_setup(&setup_ref);

        table.set_header(&["Iter", "t", &format!("t/{t_unit_label}"),
            "dt", "MZCS"]);
        table.update_row(&["0", "0.000000e0", "0.000", "0.000000e0", "—"]);
        table.set_progress(0);
        if let Ok(log) = out_dir.join(format!("{name}.log")).into_os_string().into_string() {
            let _ = table.set_log_file(std::path::Path::new(&log));
        }
    }

    // baseline frame at t = 0.
    let path0 = out_dir.join(format!("{name}_0000.h5"));
    write_checkpoint(sim, path0.to_str().unwrap(), metadata)?;
    table.post_info(&format!("wrote {} (t = 0)", path0.display()));
    table.refresh();

    // optional one-shot schema/metadata dump (opt-in via env var). does NOT
    // interfere with the live widget — both go to stdout, the table dump
    // simply appears once below the widget.
    let want_schema = std::env::var("SYMBI_SCHEMA").as_deref() == Ok("1");
    let want_table  = std::env::var("SYMBI_TABLE").as_deref()  == Ok("1");
    if (want_schema || want_table)
        && let Ok(tree) = symbi_io::Hdf5Backend.read(&path0)
    {
        eprintln!("\n[{name}] schema:");
        if want_table {
            eprintln!("{}", symbi_display::render_metadata("example extras", metadata));
            eprintln!("{}", symbi_display::render_tree_buf(&tree));
        } else {
            eprintln!("{}", display_tree_buf(&tree, ""));
        }
    }

    // refresh cadence — `SYMBI_REFRESH` overrides the default 100-iter interval.
    let refresh_interval: u64 = std::env::var("SYMBI_REFRESH").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(100);

    let start_time = std::time::Instant::now();
    let mut last_refresh_iter: u64 = 0;
    let mut last_refresh_time = start_time;
    // checkpoint-write (I/O) time, accumulated separately so the MZCS metric
    // measures COMPUTE throughput only — disk time does not belong in the bench.
    let mut io_time = std::time::Duration::ZERO;
    let mut last_refresh_io = std::time::Duration::ZERO;

    {
        let out = out_dir.to_path_buf();
        let md = metadata.clone();
        let name_owned = name.to_string();
        let cp_interval: f64 = if n_checkpoints == 0 {
            f64::INFINITY
        } else { t_final / n_checkpoints as f64 };
        let mut next_cp = cp_interval;
        let mut cp_index: u32 = 1;

        evolve_with_callback(sim, kernels, t_final, refresh_interval, |s| {
            // checkpoint events first — they get a post_success in the log.
            while s.time + 1e-12 >= next_cp && next_cp.is_finite() {
                let path = out.join(format!("{name_owned}_{cp_index:04}.h5"));
                // I/O is NOT compute: time the checkpoint write and accumulate it
                // so it can be subtracted from the MZCS timer.
                let _io_t0 = std::time::Instant::now();
                let cp_result = write_checkpoint(s, path.to_str().unwrap(), &md);
                io_time += _io_t0.elapsed();
                match cp_result {
                    Ok(_)  => table.post_success(&format!(
                        "checkpoint {cp_index:04}: {} (t/{t_unit_label} = {:.4})",
                        path.display(), s.time / t_unit,
                    )),
                    Err(e) => table.post_error(&format!(
                        "checkpoint {cp_index:04} failed at t={:.6}: {e}", s.time,
                    )),
                }
                cp_index += 1;
                next_cp += cp_interval;
            }
            // refresh the live benchmark row. MZCS is COMPUTE-only: subtract the
            // checkpoint-write (I/O) time accrued in this interval from the wall delta.
            let now = std::time::Instant::now();
            let iter_delta = s.iteration.saturating_sub(last_refresh_iter);
            let io_delta = (io_time - last_refresh_io).as_secs_f64();
            let time_delta =
                (now.duration_since(last_refresh_time).as_secs_f64() - io_delta).max(0.0);
            let mzcs = if time_delta > 0.0 && iter_delta > 0 {
                (iter_delta as f64 * total_cells as f64) / (time_delta * 1.0e6)
            } else { 0.0 };
            let progress_pct = ((s.time / t_final).clamp(0.0, 1.0) * 100.0) as usize;
            let row_strs = [
                format!("{}", s.iteration),
                format!("{:.6e}", s.time),
                format!("{:.4}", s.time / t_unit),
                format!("{:.6e}", s.dt),
                if mzcs > 0.0 { format!("{mzcs:.1}") } else { "—".to_string() },
            ];
            table.update_row(&[
                row_strs[0].as_str(), row_strs[1].as_str(), row_strs[2].as_str(),
                row_strs[3].as_str(), row_strs[4].as_str(),
            ]);
            table.set_progress(progress_pct);
            table.refresh();
            last_refresh_iter = s.iteration;
            last_refresh_time = now;
            last_refresh_io = io_time;
        })?;
    }

    // compute-only timing: total wall MINUS accumulated checkpoint-write (I/O)
    // time. captured BEFORE the final checkpoint below, so that write — also
    // I/O — never counts against the compute throughput either.
    let wall = start_time.elapsed().as_secs_f64();
    let compute_elapsed = (wall - io_time.as_secs_f64()).max(0.0);
    let avg_mzcs = if compute_elapsed > 0.0 {
        (sim.iteration as f64 * total_cells as f64) / (compute_elapsed * 1.0e6)
    } else { 0.0 };

    let path_final = out_dir.join(format!("{name}_final.h5"));
    write_checkpoint(sim, path_final.to_str().unwrap(), metadata)?;
    table.post_success(&format!(
        "final: {} (t/{t_unit_label} = {:.4}, iter = {}, compute = {:.1}s, I/O = {:.1}s, avg MZCS = {:.1})",
        path_final.display(), sim.time / t_unit, sim.iteration,
        compute_elapsed, io_time.as_secs_f64(), avg_mzcs,
    ));
    table.set_progress(100);
    table.refresh();

    // clean stdout line (the TUI table truncates the message) + the per-phase
    // breakdown under SYMBI_PROFILE — for roofline / fusion analysis of the
    // uniform single-grid path.
    println!("\nUNIFORM final: avg MZCS = {avg_mzcs:.2}  (iter = {}, compute = {:.2}s, {total_cells} cells)",
        sim.iteration, compute_elapsed);
    let phases = symbi::sim::evolve::report_profile();
    if !phases.is_empty() {
        let total: f64 = phases.iter().map(|(_, ms)| ms).sum();
        let zone_cycles = sim.iteration as f64 * total_cells as f64;
        println!("--- per-phase wall over {} steps (SYMBI_PROFILE) ---", sim.iteration);
        let mut rows = phases.clone();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (n, ms) in rows {
            println!("  {n:<16} {ms:>8.1} ms  ({:>4.1}%)   {:.1} ns/zone-cycle",
                100.0 * ms / total, ms * 1e6 / zone_cycles.max(1.0));
        }
    }
    Ok(())
}

/// a named-field config for [`run_simulation`] — the 9 positional args grouped, with sane defaults
/// (10 checkpoints, `t_unit = 1.0` labeled "t", no extra metadata). build it fluently and `.run`:
///   `RunConfig::new("nmhd_rotor", &cli.out_dir, t_final).checkpoints(15).metadata(&md).run(&mut sim, &sub)?;`
/// purely additive over `run_simulation` (which it wraps); the positional fn keeps working.
pub struct RunConfig<'a> {
    /// problem name — the checkpoint file stem + the live-widget title.
    pub name: &'a str,
    /// output directory for the HDF5 checkpoints.
    pub out_dir: &'a Path,
    /// end time in problem units.
    pub t_final: f64,
    /// uniformly-spaced checkpoints (0 = only the initial + final frame). default 10.
    pub n_checkpoints: usize,
    /// optional run metadata embedded in every checkpoint. default none.
    pub metadata: Option<&'a Metadata>,
    /// the "natural" time scale for the progress `t/T` column (e.g. an orbital period). default 1.0.
    pub t_unit: f64,
    /// the label for that column. default "t".
    pub t_unit_label: &'a str,
    /// the Riemann-solver label for the live setup table. default "HLLE".
    pub solver_label: &'a str,
}

impl<'a> RunConfig<'a> {
    /// the three required fields; everything else defaults.
    pub fn new(name: &'a str, out_dir: &'a Path, t_final: f64) -> Self {
        Self { name, out_dir, t_final, n_checkpoints: 10, metadata: None, t_unit: 1.0, t_unit_label: "t", solver_label: "HLLE" }
    }

    /// the Riemann-solver label shown in the live setup table (derived from the chosen `Solver`).
    pub fn solver(mut self, solver: Solver) -> Self {
        self.solver_label = solver_label(solver);
        self
    }
    /// number of uniformly-spaced checkpoints (0 = initial + final only).
    pub fn checkpoints(mut self, n: usize) -> Self {
        self.n_checkpoints = n;
        self
    }
    /// run metadata embedded in every checkpoint.
    pub fn metadata(mut self, m: &'a Metadata) -> Self {
        self.metadata = Some(m);
        self
    }
    /// the natural time scale + label for the progress `t/T` column.
    pub fn time_unit(mut self, unit: f64, label: &'a str) -> Self {
        self.t_unit = unit;
        self.t_unit_label = label;
        self
    }

    /// drive the run — the fluent equivalent of calling [`run_simulation`] with these fields.
    pub fn run<R, const D: usize, const DOF: usize, M, E, S, Mem, K>(
        &self,
        sim: &mut SimStateGeneric<R, D, DOF, M, E, S, Mem, f64>,
        kernels: &K,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        R: Regime<f64, D>,
        M: Metric<f64, D> + Copy + Send + Sync,
        E: Eos<f64> + Send + Sync,
        S: ExecutionSpace,
        Mem: MemorySpace + Sync,
        K: KernelSet<D, DOF, Mem, f64>,
    {
        let empty = Metadata::new();
        run_simulation(
            sim, kernels, self.t_final, self.n_checkpoints, self.out_dir, self.name,
            self.metadata.unwrap_or(&empty), self.t_unit, self.t_unit_label, self.solver_label,
        )
    }
}

// =============================================================================
// the AMR evolve+checkpoint loop — run_simulation's sibling for a Hierarchy.
// =============================================================================

/// drive `Hierarchy::evolve_with_callback` with the same live progress widget
/// as `run_simulation`, writing per-level checkpoint files side by side
/// (`{name}_L{ll}_####.h5`, plus `{name}_L{ll}_0000.h5` baselines and
/// `{name}_L{ll}_final.h5` terminals). `metadata` is re-evaluated per
/// checkpoint (accretion totals and other diagnostics move); `status` returns
/// a one-line summary appended to each checkpoint message. MZCS counts the
/// COMPOSITE zone cycles of one root step: level ll subcycles 2^ll times, so
/// a root iteration updates sum over levels of cells_ll * 2^ll zones.
#[allow(clippy::too_many_arguments)]
pub fn run_hierarchy<R, const D: usize, M, E, S, Mem, K>(
    hier:          &mut Hierarchy<R, D, D, M, E, S, Mem, K>,
    t_final:       f64,
    n_checkpoints: usize,
    out_dir:       &Path,
    name:          &str,
    t_unit:        f64,
    t_unit_label:  &str,
    solver_label:  &str,
    extra_setup:   &[[String; 3]],
    metadata:      impl Fn(&Hierarchy<R, D, D, M, E, S, Mem, K>) -> Metadata,
    status:        impl Fn(&Hierarchy<R, D, D, M, E, S, Mem, K>) -> String,
) -> Result<(), Box<dyn std::error::Error>>
where
    R:   Regime<f64, D> + Copy,
    M:   Metric<f64, D> + Copy + Send + Sync,
    E:   Eos<f64> + Copy + Send + Sync,
    S:   ExecutionSpace,
    Mem: MemorySpace + Sync,
    K:   KernelSet<D, D, Mem, f64>,
{
    fs::create_dir_all(out_dir)?;
    symbi_display::term_guard::install();

    let zc_per_iter: u64 = hier.levels.iter().enumerate()
        .map(|(ll, lvl)| (lvl.state.geom.interior.volume() as u64) << ll)
        .sum();
    let recon_label = std::env::var("SYMBI_RECON").unwrap_or_else(|_| "θ-MC PLM".into());
    let mut table = symbi_display::Table::new(&format!("symbi · {name}"), /*dynamic=*/ true);
    {
        let sys_rows = build_system_info_rows();
        let sys_ref: Vec<[&str; 3]> = sys_rows.iter()
            .map(|r| [r[0].as_str(), r[1].as_str(), r[2].as_str()]).collect();
        table.set_system_info(&sys_ref);

        let mut setup_rows =
            build_problem_setup_rows(&hier.levels[0].state, solver_label, &recon_label);
        setup_rows.push(["AMR".to_string(), "Levels".to_string(),
            hier.levels.len().to_string()]);
        for (ll, lvl) in hier.levels.iter().enumerate().skip(1) {
            let dims: Vec<String> = (0..D)
                .map(|ax| lvl.state.geom.interior.spaces[ax].size().to_string()).collect();
            setup_rows.push(["".to_string(), format!("L{ll} resolution"), dims.join(" × ")]);
        }
        setup_rows.extend_from_slice(extra_setup);
        let setup_ref: Vec<[&str; 3]> = setup_rows.iter()
            .map(|r| [r[0].as_str(), r[1].as_str(), r[2].as_str()]).collect();
        table.set_problem_setup(&setup_ref);

        table.set_header(&["Iter", "t", &format!("t/{t_unit_label}"), "dt", "MZCS"]);
        table.update_row(&["0", "0.000000e0", "0.000", "0.000000e0", "—"]);
        table.set_progress(0);
        if let Ok(log) = out_dir.join(format!("{name}.log")).into_os_string().into_string() {
            let _ = table.set_log_file(std::path::Path::new(&log));
        }
    }

    // per-level baseline frames at t = 0.
    let write_levels = |h: &Hierarchy<R, D, D, M, E, S, Mem, K>, tag: &str|
        -> Result<PathBuf, Box<dyn std::error::Error>> {
        let md = metadata(h);
        let mut first = PathBuf::new();
        for (ll, lvl) in h.levels.iter().enumerate() {
            let path = out_dir.join(format!("{name}_L{ll}_{tag}.h5"));
            write_checkpoint(&lvl.state, path.to_str().unwrap(), &md)?;
            if ll == 0 {
                first = path;
            }
        }
        Ok(first)
    };
    let path0 = write_levels(hier, "0000")?;
    table.post_info(&format!("wrote {} + L1.. siblings (t = 0)", path0.display()));
    table.refresh();

    // refresh every root step: a 3d composite root step dwarfs a table
    // refresh, and the checkpoint trigger lives in the callback — a coarser
    // cadence would overshoot the checkpoint times by whole root steps.
    let refresh_interval: u64 = std::env::var("SYMBI_REFRESH").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(1);

    symbi::sim::evolve::reset_profile();
    symbi::regimes::substrate_kernels::reset_dispatch_profile();
    let start_time = std::time::Instant::now();
    let mut last_refresh_iter: u64 = 0;
    let mut last_refresh_time = start_time;
    let mut io_time = std::time::Duration::ZERO;
    let mut last_refresh_io = std::time::Duration::ZERO;

    {
        let cp_interval: f64 = if n_checkpoints == 0 {
            f64::INFINITY
        } else { t_final / n_checkpoints as f64 };
        let mut next_cp = cp_interval;
        let mut cp_index: u32 = 1;

        hier.evolve_with_callback(t_final, refresh_interval, |h| {
            let root = &h.levels[0].state;
            while root.time + 1e-12 >= next_cp && next_cp.is_finite() {
                let io_t0 = std::time::Instant::now();
                let cp_result = write_levels(h, &format!("{cp_index:04}"));
                io_time += io_t0.elapsed();
                match cp_result {
                    Ok(path) => table.post_success(&format!(
                        "checkpoint {cp_index:04}: {} (t/{t_unit_label} = {:.4})  {}",
                        path.display(), root.time / t_unit, status(h),
                    )),
                    Err(e) => table.post_error(&format!(
                        "checkpoint {cp_index:04} failed at t={:.6}: {e}", root.time,
                    )),
                }
                cp_index += 1;
                next_cp += cp_interval;
            }
            let now = std::time::Instant::now();
            let iter_delta = root.iteration.saturating_sub(last_refresh_iter);
            let io_delta = (io_time - last_refresh_io).as_secs_f64();
            let time_delta =
                (now.duration_since(last_refresh_time).as_secs_f64() - io_delta).max(0.0);
            let mzcs = if time_delta > 0.0 && iter_delta > 0 {
                (iter_delta as f64 * zc_per_iter as f64) / (time_delta * 1.0e6)
            } else { 0.0 };
            let progress_pct = ((root.time / t_final).clamp(0.0, 1.0) * 100.0) as usize;
            let row_strs = [
                format!("{}", root.iteration),
                format!("{:.6e}", root.time),
                format!("{:.4}", root.time / t_unit),
                format!("{:.6e}", root.dt),
                if mzcs > 0.0 { format!("{mzcs:.1}") } else { "—".to_string() },
            ];
            table.update_row(&[
                row_strs[0].as_str(), row_strs[1].as_str(), row_strs[2].as_str(),
                row_strs[3].as_str(), row_strs[4].as_str(),
            ]);
            table.set_progress(progress_pct);
            table.refresh();
            last_refresh_iter = root.iteration;
            last_refresh_time = now;
            last_refresh_io = io_time;
        })?;
    }

    let wall = start_time.elapsed().as_secs_f64();
    let compute_elapsed = (wall - io_time.as_secs_f64()).max(0.0);
    let root = &hier.levels[0].state;
    let avg_mzcs = if compute_elapsed > 0.0 {
        (root.iteration as f64 * zc_per_iter as f64) / (compute_elapsed * 1.0e6)
    } else { 0.0 };

    let path_final = write_levels(hier, "final")?;
    let root = &hier.levels[0].state;
    table.post_success(&format!(
        "final: {} (t/{t_unit_label} = {:.4}, iter = {}, compute = {:.1}s, I/O = {:.1}s, avg MZCS = {:.1})  {}",
        path_final.display(), root.time / t_unit, root.iteration,
        compute_elapsed, io_time.as_secs_f64(), avg_mzcs, status(hier),
    ));
    table.set_progress(100);
    table.refresh();

    // per-phase wall breakdown across the composite hierarchy step (SYMBI_PROFILE).
    // physics phases (flux/c2p/godunov_stage/...) share the labels emitted by the
    // uniform sim/evolve::step; refine_* phases are the refinement bookkeeping
    // (reflux registers, restrict, prolong, coarse-fine ghosts) the uniform path
    // never pays — the whole point of this breakdown.
    let phases = symbi::sim::evolve::report_profile();
    if !phases.is_empty() {
        let total: f64 = phases.iter().map(|(_, ms)| ms).sum();
        let zone_cycles = root.iteration as f64 * zc_per_iter as f64;
        println!("\n--- per-phase wall time over {} root steps (SYMBI_PROFILE) ---", root.iteration);
        let mut rows = phases.clone();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let refine_total: f64 = phases.iter().filter(|(n, _)| n.starts_with("refine_")).map(|(_, ms)| ms).sum();
        for (name, ms) in rows {
            println!("  {name:<18} {ms:>8.1} ms  ({:>4.1}%)   {:.0} ns/zone-cycle",
                100.0 * ms / total, ms * 1e6 / zone_cycles.max(1.0));
        }
        println!("  {:<18} {total:>8.1} ms  (sum of instrumented phases)", "TOTAL");
        println!("  {:<18} {refine_total:>8.1} ms  ({:>4.1}% of instrumented — refinement bookkeeping)",
            "AMR-BOOKKEEPING", 100.0 * refine_total / total);
    }

    // per-dispatch attribution of the amr-transfer / register path (SYMBI_DISPATCH_PROF):
    // splits the registry name lookup from the kernel execution (rayon launch + work),
    // and reports the dispatch count — to pin which of (lookup, launch, granularity)
    // drives the prolong cost.
    let (dcount, dlookup_ns, dexec_ns) = symbi::regimes::substrate_kernels::report_dispatch_profile();
    if dcount > 0 {
        let total_ns = (dlookup_ns + dexec_ns) as f64;
        println!("\n--- dispatch_fields_each attribution (SYMBI_DISPATCH_PROF) ---");
        println!("  dispatches          : {dcount}  ({:.0} per root step)", dcount as f64 / root.iteration.max(1) as f64);
        println!("  registry lookup     : {:>8.1} ms  ({:>4.1}%)   {:.0} ns/call",
            dlookup_ns as f64 / 1e6, 100.0 * dlookup_ns as f64 / total_ns, dlookup_ns as f64 / dcount as f64);
        println!("  kernel execution    : {:>8.1} ms  ({:>4.1}%)   {:.0} ns/call",
            dexec_ns as f64 / 1e6, 100.0 * dexec_ns as f64 / total_ns, dexec_ns as f64 / dcount as f64);
    }
    Ok(())
}
