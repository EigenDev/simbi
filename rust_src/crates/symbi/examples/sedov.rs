// =============================================================================
// examples/sedov.rs
//
// Sedov–Taylor blast wave on a 2D Cartesian grid. high-pressure thermal energy
// deposited inside a small disk at the center expands into ambient gas,
// producing the canonical self-similar blast wave. validates the substrate's
// adiabatic-Euler shock-capturing on a strong shock with radial symmetry —
// the Cartesian grid should preserve circular symmetry to numerical accuracy.
//
// (this is the Cartesian analog of the legacy log-spherical Sedov; the spherical
// `Spacing::Log` path isn't yet wired through SimState. Cartesian is a valid
// shock-physics test in its own right and reproduces the same self-similar
// scaling once the blast leaves the deposition region.)
//
// usage:
//   cargo run --release -p symbi --example sedov -- --n 256 --end-time 0.1 \
//       --n-checkpoints 10 --out output/sedov/data
//
// problem-specific knobs (via `--key val`):
//   --e0 <1.0>          total deposited energy
//   --rho-amb <1.0>     ambient density
//   --p-amb <1e-5>      ambient pressure (~ cold)
//   --r-blast <0.05>    blast deposition radius
//   --bound <0.5>       domain half-width (extent = ±bound)
// =============================================================================

mod common;
use common::{BaseCli, Metadata, RunConfig};

use symbi::prelude::*;

type Sim = SimDefault<Newtonian, 2, Cartesian, IdealGas<f64>>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = BaseCli::parse("sedov");

    let gamma   = cli.gamma.unwrap_or(5.0 / 3.0);
    let bound   = cli.extra_f64("bound", 0.5);
    let e0      = cli.extra_f64("e0", 1.0);
    let rho_amb = cli.extra_f64("rho-amb", 1.0);
    let p_amb   = cli.extra_f64("p-amb", 1e-5);
    let r_blast = cli.extra_f64("r-blast", 0.05);
    let [nx, ny] = cli.n2();
    let dx = 2.0 * bound / nx as f64;
    let dy = 2.0 * bound / ny as f64;

    eprintln!("[sedov] grid = {nx}x{ny}, γ = {gamma}, E_0 = {e0}, blast radius = {r_blast}");

    // ---- blast deposition: spread E_0 uniformly over a circular disk ----
    // pressure inside disk: (γ-1) E_0 / (π r_blast²)   (2D analog).
    let p_blast = (gamma - 1.0) * e0 / (std::f64::consts::PI * r_blast * r_blast);
    let mut sim = Sim::build(Newtonian, IdealGas { gamma }, Cartesian)
        .cells([nx, ny])
        .origin([-bound, -bound])
        .spacing([dx, dy])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(cli.cfl)
        .timestepping(cli.timestepping)
        .allocate()?
        .set_initial(|p| {
            let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
            let pre = if r <= r_blast { p_blast } else { p_amb };
            Prim { rho: rho_amb, vel: Tensor::new([0.0, 0.0]), pre }
        })
        .build();

    let sub = sim.substrate().with_solver(cli.solver)?;

    let metadata = Metadata::new()
        .with("problem", "sedov")
        .with("gamma",   gamma)
        .with("e0",      e0)
        .with("r_blast", r_blast);

    RunConfig::new("sedov", &cli.out_dir, cli.end_time)
        .checkpoints(cli.n_checkpoints)
        .metadata(&metadata)
        .solver(cli.solver)
        .run(&mut sim, &sub)
}
