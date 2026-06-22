// =============================================================================
// rmhd_zone_cycle_bench.rs
//
// full-timestep ("zone-cycle") throughput of the production RMHD substrate, in
// Mzcps (million zone-cycles per second) — the SAME unit AthenaK reports
// (Stone et al. 2024, Tables 1-2). one zone-cycle = one full RK step of a cell:
// c2p -> ghost_fill -> flux x3 dirs -> cfl -> snapshot -> godunov -> CT -> rk2.
//
// 3D Orszag-Tang vortex (extruded in z), the smooth-and-developing phase, so c2p
// converges and nothing NaNs. measures interior-cell updates only.
//
// usage:
//   cargo run -p symbi --release --example rmhd_zone_cycle_bench           # all cores
//   RAYON_NUM_THREADS=1 cargo run -p symbi --release --example rmhd_zone_cycle_bench  # per-core
// =============================================================================

use std::f64::consts::PI;
use std::time::Instant;

use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet3D;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::state::Prim;
use symbi_hydro::rmhd::Rmhd;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Rmhd, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

// cubic grid N^3; override with SYMBI_BENCH_N to probe parallelization granularity.
fn grid_n() -> usize {
    std::env::var("SYMBI_BENCH_N").ok().and_then(|s| s.parse().ok()).unwrap_or(64)
}
const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.3;
const V0: f64 = 0.5;
const B0: f64 = 1.0;

fn make_sim(n: usize) -> Sim {
    let dx = 1.0 / n as f64;
    let rho0 = GAMMA * GAMMA;
    let p0 = GAMMA;
    // div-free staggered B (Orszag-Tang), z-independent so it extrudes cleanly; seed_faces routes
    // through face_coord (exact staggered position) and seed_cell folds the cell-centered B in.
    Sim::build(Rmhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([n, n, n])
        .spacing([dx, dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("rmhd sim construction failed")
        .set_initial(|[x, y, _z]| {
            let vx = -V0 * (2.0 * PI * y).sin();
            let vy = V0 * (2.0 * PI * x).sin();
            let bx_c = -B0 * (2.0 * PI * y).sin();
            let by_c = B0 * (4.0 * PI * x).sin();
            MhdPrim {
                hydro: Prim { rho: rho0, vel: Tensor::new([vx, vy, 0.0]), pre: p0 },
                mag: Tensor::new([bx_c, by_c, 0.0]),
            }
        })
        .seed_faces(|axis, [x, y, _z]| match axis {
            0 => -B0 * (2.0 * PI * y).sin(),
            1 => B0 * (4.0 * PI * x).sin(),
            _ => 0.0,
        })
        .build()
}

fn main() {
    let n = grid_n();
    let n_cells = n * n * n;
    let mut sim = make_sim(n);
    let sub = RmhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, CFL, 1.0, &sim.geom.allocated);

    // warm up: a few steps to settle caches / branch predictors.
    evolve_with_callback(&mut sim, &sub, 0.02, 1, |_| {}).expect("warmup failed");
    let warm_iter = sim.iteration;

    // Step-1.0 diagnostic: time ONE kernel phase in isolation over the whole grid
    // at the current RAYON_NUM_THREADS, to separate memory-bound (poor isolated
    // scaling) from step-context/barrier effects (good isolated scaling).
    // SYMBI_PHASE=flux|godunov|c2p|ghost. min-of-reps ns per cell-sweep.
    if let Ok(phase) = std::env::var("SYMBI_PHASE") {
        use symbi::sim::evolve::KernelSet;
        let dt = sub.cfl(&sim);
        let call = || match phase.as_str() {
            "flux"    => { for d in 0..3 { sub.flux(&sim, d); } }
            "godunov" => sub.godunov_stage(&sim, dt, 0.0, 1.0),
            "c2p"     => sub.c2p(&sim),
            "ghost"   => sub.ghost_fill(&sim),
            other     => panic!("SYMBI_PHASE: unknown phase '{other}'"),
        };
        for _ in 0..5 { call(); } // warmup
        let mut best = f64::INFINITY;
        for _ in 0..100 {
            let t = Instant::now();
            call();
            best = best.min(t.elapsed().as_secs_f64());
        }
        let sweeps = if phase == "flux" { 3.0 } else { 1.0 }; // flux = 3 directional sweeps
        let ns_cell = best * 1e9 / (n_cells as f64 * sweeps);
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
        println!("PHASE={phase}  N={n}^3  {ns_cell:.2} ns/cell-sweep  (threads avail {threads})");
        return;
    }

    // timed window: evolve a further span, count steps, measure wall time.
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

    println!("=== RMHD full zone-cycle (3D Orszag-Tang, {n}^3 = {n_cells} cells) ===");
    println!("steps timed     : {steps}");
    println!("wall time       : {:.4} s", elapsed);
    println!("ns / zone-cycle : {:.1}", ns_per_zc);
    println!("THROUGHPUT      : {:.1} Mzcps   (rayon threads available: {threads})", mzcps);
    println!("  (AthenaK SR-MHD ref: 20 Mzcps Xeon-6326/32c, 178 A100, 297 GraceHopper)");

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
