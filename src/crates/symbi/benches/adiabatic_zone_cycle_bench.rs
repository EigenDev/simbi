// =============================================================================
// adiabatic_zone_cycle_bench.rs
//
// full-timestep ("zone-cycle") throughput of the production Newtonian (ideal-gas)
// substrate, in Mzcps (million zone-cycles per second). one zone-cycle = one full
// RK step of a cell: c2p -> ghost_fill -> flux x2 dirs -> cfl -> godunov -> rk2.
//
// this drives the RAW substrate via `evolve_with_callback` — NO AMR hierarchy, NO
// FOFC pass, NO TUI. so it isolates the kernel cost from the driver overhead, the
// instrument for separating "the flux kernel got slow" (kernel-bound) from
// "driver overhead accreted" (FOFC probe/reduce, snapshot_stage, AMR wrapper).
//
// 2D Kelvin-Helmholtz shear layer, the smooth developing phase. adiabatic c2p is
// algebraic (no iteration), so nothing NaNs. measures interior-cell updates only.
//
// usage:
//   cargo bench -p symbi --bench adiabatic_zone_cycle_bench                      # all cores, HLLE
//   SYMBI_SOLVER=hllc cargo bench -p symbi --bench adiabatic_zone_cycle_bench    # HLLC
//   SYMBI_BENCH_N=512 SYMBI_SOLVER=hllc cargo bench -p symbi --bench adiabatic_zone_cycle_bench
//   RAYON_NUM_THREADS=1 cargo bench -p symbi --bench adiabatic_zone_cycle_bench  # per-core
//   SYMBI_PROFILE=1 cargo bench -p symbi --bench adiabatic_zone_cycle_bench      # per-phase breakdown
//   SYMBI_PHASE=flux cargo bench -p symbi --bench adiabatic_zone_cycle_bench     # one phase in isolation
// =============================================================================

use std::f64::consts::PI;
use std::time::Instant;

use symbi::prelude::Solver;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

// square grid N^2; override with SYMBI_BENCH_N (KH configs run 256^2 / 512^2).
fn grid_n() -> usize {
    std::env::var("SYMBI_BENCH_N").ok().and_then(|s| s.parse().ok()).unwrap_or(512)
}
fn solver() -> Solver {
    match std::env::var("SYMBI_SOLVER").ok().as_deref().map(|s| s.to_ascii_lowercase()) {
        Some(ref s) if s == "hllc" => Solver::Hllc,
        _ => Solver::Hlle,
    }
}
const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;

fn make_sim(n: usize) -> Sim {
    let dx = 1.0 / n as f64;
    // Kelvin-Helmholtz: a central band (|y - 0.5| < 0.25) shears against the outer
    // fluid, tanh-smoothed so the layer is resolved; a small vy seed grows the roll-up.
    // uniform pressure. this is a benchmark IC for throughput measurement.
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([n, n])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("adiabatic sim construction failed")
        .set_initial(|[x, y]| {
            let smooth = 0.02;
            // tanh window ~1 inside the central band (0.25 < y < 0.75), ~0 outside.
            let inside = 0.5 * (1.0 + ((y - 0.25) / smooth).tanh() * ((0.75 - y) / smooth).tanh());
            let vx = -0.5 + inside; // -0.5 outside, +0.5 inside
            let rho = 1.0 + inside; // 1 outside, 2 inside
            let vy = 0.01 * (4.0 * PI * x).sin();
            Prim { rho, vel: Tensor::new([vx, vy]), pre: 2.5 }
        })
        .build()
}

fn main() {
    let n = grid_n();
    let n_cells = n * n;
    let sol = solver();
    let mut sim = make_sim(n);
    let sub = AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, CFL, &sim.geom.allocated)
        .with_solver(sol)
        .expect("solver/regime mismatch");

    // warm up: settle caches / branch predictors.
    evolve_with_callback(&mut sim, &sub, 0.02, 1, |_| {}).expect("warmup failed");
    let warm_iter = sim.iteration;

    // one phase in isolation over the whole grid (SYMBI_PHASE=flux|godunov|c2p|ghost):
    // separates memory-bound (poor isolated scaling) from step-context effects.
    if let Ok(phase) = std::env::var("SYMBI_PHASE") {
        use symbi::sim::evolve::KernelSet;
        let dt = sub.cfl(&sim);
        let call = || match phase.as_str() {
            "flux" => { for d in 0..2 { sub.flux(&sim, d); } }
            "godunov" => sub.godunov_stage(&sim, dt, 0.0, 1.0),
            "c2p" => sub.c2p(&sim),
            "ghost" => sub.ghost_fill(&sim),
            other => panic!("SYMBI_PHASE: unknown phase '{other}'"),
        };
        // SYMBI_POLLUTE=1: run snapshot_stage (the ~2.6 MB full-cons copy the AMR level_stage runs
        // before flux every substage) UNTIMED before each timed phase call — same timed kernel, but
        // cold cache. if the timed flux jumps to the driver's ~40 ns/zc, the level_stage snapshot copy
        // is the cache polluter inflating flux, confirming the non-destructive-godunov fix.
        let pollute = std::env::var("SYMBI_POLLUTE").is_ok();
        for _ in 0..5 { call(); }
        let mut best = f64::INFINITY;
        for _ in 0..100 {
            if pollute { sub.snapshot_stage(&sim); }
            let t = Instant::now();
            call();
            best = best.min(t.elapsed().as_secs_f64());
        }
        let sweeps = if phase == "flux" { 2.0 } else { 1.0 };
        let ns_cell = best * 1e9 / (n_cells as f64 * sweeps);
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
        let tag = if pollute { " [snapshot-polluted]" } else { "" };
        println!("PHASE={phase}{tag}  N={n}^2  {ns_cell:.2} ns/cell-sweep  (threads avail {threads})");
        return;
    }

    // timed window: evolve a span, count steps, measure wall time.
    symbi::sim::evolve::reset_profile();
    let t0 = Instant::now();
    let mut last_iter = warm_iter;
    evolve_with_callback(&mut sim, &sub, 0.12, 1, |s| {
        last_iter = s.iteration;
    })
    .expect("timed evolve failed");
    let elapsed = t0.elapsed().as_secs_f64();

    let steps = (last_iter - warm_iter) as usize;
    let zone_cycles = (n_cells as f64) * (steps as f64);
    let mzcps = zone_cycles / elapsed / 1e6;
    let ns_per_zc = elapsed * 1e9 / zone_cycles;
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
    let solver_name = match sol { Solver::Hllc => "HLLC", _ => "HLLE" };

    println!("=== Newtonian full zone-cycle (2D Kelvin-Helmholtz, {n}^2 = {n_cells} cells, {solver_name}) ===");
    println!("steps timed     : {steps}");
    println!("wall time       : {:.4} s", elapsed);
    println!("ns / zone-cycle : {:.1}", ns_per_zc);
    println!("THROUGHPUT      : {:.1} Mzcps   (rayon threads available: {threads})", mzcps);

    let prof = symbi::sim::evolve::report_profile();
    if !prof.is_empty() {
        let total: f64 = prof.iter().map(|(_, ms)| ms).sum();
        println!("\n--- per-phase wall time over {steps} steps (SYMBI_PROFILE) ---");
        let mut rows = prof.clone();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (name, ms) in rows {
            println!("  {name:<16} {ms:>8.1} ms  ({:>4.1}%)   {:.0} ns/zone-cycle",
                100.0 * ms / total, ms * 1e6 / (steps as f64 * n_cells as f64));
        }
        println!("  {:<16} {total:>8.1} ms  (sum of instrumented phases)", "TOTAL");
    }
}
