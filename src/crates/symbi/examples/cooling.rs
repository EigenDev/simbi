// =============================================================================
// examples/cooling.rs
//
// radiative cooling as a USER-DEFINED source — thermal condensation. a uniform
// 2D adiabatic box is seeded with small density fluctuations and given an
// optically-thin-style cooling rate
//
//     Lambda(rho, p) = C * rho * p          ->   S_nrg = -Lambda
//
// loaded at RUNTIME from a json `SourceConfig` (the python -> json -> bridge
// path, no recompile), interpreted per cell each SSP stage via
// `with_runtime_source`. the source reads the PER-CELL state (rho AND p) — the
// capability the carrier user-source surface exists for.
//
// physics: with Lambda ~ rho*p the pressure decays as exp(-(gamma-1)*C*rho*t),
// so it stays strictly positive (no negative-pressure blow-up), but DENSER gas
// cools FASTER. a slightly over-dense fluctuation loses pressure support, gets
// squeezed by its warmer surroundings, grows denser, and cools faster still —
// the medium condenses into a network of cold, dense filaments and clumps.
//
// usage:
//   cargo run --release -p symbi --example cooling -- --n 256 --end-time 4 \
//       --n-checkpoints 40 --out output/cooling/data
//   uv run python3 scripts/plot_cooling.py output/cooling/data/cooling_*.h5
//
// problem-specific knobs (via `--key val`):
//   --cool <1.0>     cooling coefficient C (larger = faster cooling)
//   --rho0 <1.0>     background density
//   --p0 <1.0>       background pressure
//   --amp <0.1>      density-fluctuation amplitude (fraction of rho0)
//   --seed <12345>   hash-based RNG seed (deterministic)
//
// numerics (base knobs, see --help): --solver hlle|hllc, --theta <1..2> PLM
//   limiter, --timestepping euler|rk2, --cfl, --gamma, --end-time, --n.
// =============================================================================

mod common;
use common::{BaseCli, Metadata, RunConfig};

use symbi::prelude::*;
use symbi_hydro::expr_bridge::build_user_source;
use symbi_hydro::{SourceConfig, NEWTONIAN_SPEC};

type Sim = SimDefault<Newtonian, 2, Cartesian, IdealGas<f64>>;

/// deterministic per-cell noise on [-1, 1] via a SplitMix64-style hash — the same
/// seeding idiom as `kh.rs`, so a run is reproducible without an RNG byte stream.
fn cell_noise(i: usize, j: usize, seed: u64) -> f64 {
    let mix = |x: u64| -> u64 {
        let mut y = x.wrapping_add(0x9E3779B97F4A7C15);
        y = (y ^ (y >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        y = (y ^ (y >> 27)).wrapping_mul(0x94D049BB133111EB);
        y ^ (y >> 31)
    };
    let h = mix(seed
        ^ (i as u64).wrapping_mul(0xA0761D6478BD642F)
        ^ (j as u64).wrapping_mul(0xE7037ED1A0B428DB));
    2.0 * ((h >> 11) as f64 / ((1u64 << 53) as f64)) - 1.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = BaseCli::parse("cooling");

    let gamma = cli.gamma.unwrap_or(5.0 / 3.0);
    let cool  = cli.extra_f64("cool", 1.0);
    let rho0  = cli.extra_f64("rho0", 1.0);
    let p0    = cli.extra_f64("p0", 1.0);
    let amp   = cli.extra_f64("amp", 0.1);
    let seed  = cli.extra_int("seed", 12345) as u64;
    let [nx, ny] = cli.n2();

    let dx = 1.0 / nx as f64;
    let dy = 1.0 / ny as f64;

    eprintln!("[cooling] grid = {nx}x{ny}, gamma = {gamma}, C = {cool}, end_time = {}", cli.end_time);
    eprintln!("[cooling] Lambda = C*rho*p (S_nrg = -Lambda); rho0 = {rho0}, p0 = {p0}, amp = {amp}");

    // the cooling rate as a user expression, EXACTLY what `Dag.cooling_source` would emit as json:
    //   Lambda = p0 * rho * pre        (node 4), kind = "cooling" -> S_nrg = -Lambda.
    // nodes: 0 = PARAM C, 1 = rho, 2 = pre, 3 = C*rho, 4 = (C*rho)*pre.
    let json = format!(
        r#"{{
            "kind": "cooling", "dim": 2, "outputs": [4], "params": [{cool}],
            "nodes": [
                {{"op": "PARAMETER", "param_idx": 0}},
                {{"op": "VARIABLE_RHO"}},
                {{"op": "VARIABLE_PRESSURE"}},
                {{"op": "MULTIPLY", "left": 0, "right": 1}},
                {{"op": "MULTIPLY", "left": 3, "right": 2}}
            ]
        }}"#
    );
    let cfg = SourceConfig::from_json(&json).expect("parse cooling config");
    let built = build_user_source(&cfg, &NEWTONIAN_SPEC).expect("wrap cooling source");

    let mut sim = Sim::build(Newtonian, IdealGas { gamma }, Cartesian)
        .cells([nx, ny])
        .origin([0.0, 0.0])
        .spacing([dx, dy])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(cli.cfl)
        .timestepping(cli.timestepping)
        .allocate()?
        .set_initial_indexed(|coord, _x| {
            // uniform pressure, density seeded with small fluctuations — the seeds the
            // cooling instability feeds on. velocity starts at rest.
            let (i, j) = (coord[0] as usize, coord[1] as usize);
            let rho = rho0 * (1.0 + amp * cell_noise(i, j, seed));
            Prim { rho, vel: Tensor::zeros(), pre: p0 }
        })
        .build();

    // attach the runtime-loaded cooling source. `--fused 1` routes it through the FUSED host path
    // (one Cranelift-JIT'd godunov+source launch per stage); default 0 = the two-pass (AOT godunov
    // + the per-cell JIT source pass). both are bit-for-bit identical (jit_fused_equals_two_pass) —
    // the flag is a perf A/B. solver + PLM theta are set on the kernel set first.
    let fused = cli.extra_int("fused", 0) != 0;
    eprintln!("[cooling] source path = {}", if fused { "FUSED (JIT godunov+source)" } else { "two-pass" });
    let sub = sim.substrate().with_solver(cli.solver)?.theta(cli.theta);
    let sub = if fused {
        sub.with_fused_runtime_source(built, cfg.params.clone())
    } else {
        sub.with_runtime_source(built, cfg.params.clone())
    };

    let metadata = Metadata::new()
        .with("problem", "radiative_cooling")
        .with("gamma", gamma)
        .with("cool", cool)
        .with("rho0", rho0)
        .with("p0", p0)
        .with("amp", amp)
        .with("seed", seed as i64);

    RunConfig::new("cooling", &cli.out_dir, cli.end_time)
        .checkpoints(cli.n_checkpoints)
        .metadata(&metadata)
        .solver(cli.solver)
        .run(&mut sim, &sub)
}
