// =============================================================================
// examples/kh.rs
//
// Kelvin-Helmholtz instability — 2D adiabatic Newtonian shear flow with a
// density contrast. matches legacy_examples/kh.py's initial conditions:
//   • central layer  |y| <  0.25 : ρ = 2.0, vx = +0.5, vy = 0, p = 2.5
//   • outer regions  |y| >= 0.25 : ρ = 1.0, vx = -0.5, vy = 0, p = 2.5
//   • velocity perturbations: 0.01 * sin(2π * U(-1,1)) per cell (deterministic
//     hash-based RNG; Python uses np.random.default_rng(12345). identical
//     SHAPE of noise — uniformly seeded — but not cell-for-cell identical).
//   • domain [-0.5, 0.5] × [-0.5, 0.5], periodic BC on all sides
//   • γ = 5/3, HLLE Riemann (our default; Python used HLLC)
//
// usage:
//   cargo run --release -p symbi --example kh -- --n 256 --end-time 20 \
//       --n-checkpoints 40 --out output/kh/data
//
// GPU:
//   cargo run --release -p symbi --example kh --features cuda -- ...
//
// problem-specific knobs (via `--key val`):
//   --rho-l  / --rho-r       central / outer density (defaults 2.0 / 1.0)
//   --vx-t   / --vx-b        central / outer x-velocity (+0.5 / -0.5)
//   --p-l    / --p-r         central / outer pressure (both 2.5)
//   --noise <0.01>           velocity perturbation amplitude
//   --y-shear <0.25>         half-width of the central layer
//   --seed <12345>           hash-based RNG seed (deterministic)
// =============================================================================

mod common;
use common::{BaseCli, Metadata, RunConfig};

use symbi::prelude::*;

type Sim = SimDefault<Newtonian, 2, Cartesian, IdealGas<f64>>;

/// deterministic per-cell noise pair on [-1, 1] × [-1, 1] via SplitMix64-style
/// hash. independent of Python's RNG byte stream but produces statistically
/// equivalent seeding for the KH instability.
fn cell_noise(i: usize, j: usize, seed: u64) -> (f64, f64) {
    let mix = |x: u64| -> u64 {
        let mut y = x.wrapping_add(0x9E3779B97F4A7C15);
        y = (y ^ (y >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        y = (y ^ (y >> 27)).wrapping_mul(0x94D049BB133111EB);
        y ^ (y >> 31)
    };
    let h1 = mix(seed
        ^ (i as u64).wrapping_mul(0xA0761D6478BD642F)
        ^ (j as u64).wrapping_mul(0xE7037ED1A0B428DB));
    let h2 = mix(h1 ^ 0xCBF29CE484222325);
    // uniform on [0, 1)
    let u1 = (h1 >> 11) as f64 / ((1u64 << 53) as f64);
    let u2 = (h2 >> 11) as f64 / ((1u64 << 53) as f64);
    // shift to [-1, 1) — matches the SHAPE of Python's `2*rng.normal()-modulo-uniform`
    (2.0 * u1 - 1.0, 2.0 * u2 - 1.0)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = BaseCli::parse("kh");

    let gamma   = cli.gamma.unwrap_or(5.0 / 3.0);
    let rho_l   = cli.extra_f64("rho-l", 2.0);
    let rho_r   = cli.extra_f64("rho-r", 1.0);
    let vx_t    = cli.extra_f64("vx-t",  0.5);
    let vx_b    = cli.extra_f64("vx-b", -0.5);
    let p_l     = cli.extra_f64("p-l",   2.5);
    let p_r     = cli.extra_f64("p-r",   2.5);
    let noise_amp = cli.extra_f64("noise", 0.01);
    let y_shear = cli.extra_f64("y-shear", 0.25);
    let seed    = cli.extra_int("seed", 12345) as u64;
    let [nx, ny] = cli.n2();

    // domain [-0.5, 0.5] × [-0.5, 0.5]
    let x_lo = -0.5_f64;
    let x_hi =  0.5_f64;
    let y_lo = -0.5_f64;
    let y_hi =  0.5_f64;
    let dx = (x_hi - x_lo) / nx as f64;
    let dy = (y_hi - y_lo) / ny as f64;

    eprintln!("[kh] grid = {nx}×{ny}, γ = {gamma}, end_time = {}", cli.end_time);
    eprintln!("[kh] central layer: ρ={rho_l}, vx={vx_t}, p={p_l}, |y|<{y_shear}");
    eprintln!("[kh] outer regions: ρ={rho_r}, vx={vx_b}, p={p_r}");
    eprintln!("[kh] velocity perturbation amplitude: {noise_amp} (hash seed {seed})");

    // ---- KH initial conditions ----
    // central layer: |y| < y_shear -> (ρ_L, vx_T, vy=0, p_L)
    // outer regions:                 (ρ_R, vx_B, vy=0, p_R)
    // velocity perturbations: vx += noise * sin(2π * U(-1,1))
    //                         vy += noise * sin(2π * U(-1,1))
    // index-based IC: Python uses the cell ORIGIN `y = ymin + jj*dy` (not the centroid), so the
    // closure reads the integer index, not the passed center coordinate.
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut sim = Sim::build(Newtonian, IdealGas { gamma }, Cartesian)
        .cells([nx, ny])
        .origin([x_lo, y_lo])
        .spacing([dx, dy])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(cli.cfl)
        .timestepping(cli.timestepping)
        .allocate()?
        .set_initial_indexed(|coord, _x| {
            let i = coord[0] as usize;
            let j = coord[1] as usize;
            let y = y_lo + j as f64 * dy;
            let in_layer = y.abs() < y_shear;
            let (rho, vx0, p) = if in_layer { (rho_l, vx_t, p_l) } else { (rho_r, vx_b, p_r) };
            let (r1, r2) = cell_noise(i, j, seed);
            let vx = vx0 + noise_amp * (two_pi * r1).sin();
            let vy = noise_amp * (two_pi * r2).sin();
            Prim { rho, vel: Tensor::new([vx, vy]), pre: p }
        })
        .build();

    let sub = sim.substrate().with_solver(cli.solver)?;

    let metadata = Metadata::new()
        .with("problem",  "kh")
        .with("gamma",    gamma)
        .with("rho_l",    rho_l)
        .with("rho_r",    rho_r)
        .with("vx_t",     vx_t)
        .with("vx_b",     vx_b)
        .with("p_l",      p_l)
        .with("p_r",      p_r)
        .with("y_shear",  y_shear)
        .with("noise_amp", noise_amp)
        .with("seed",      seed as i64);

    RunConfig::new("kh", &cli.out_dir, cli.end_time)
        .checkpoints(cli.n_checkpoints)
        .metadata(&metadata)
        .solver(cli.solver)
        .run(&mut sim, &sub)
}
