// =============================================================================
// examples/marti_muller.rs
//
// Martí & Müller (2003) relativistic shock tube on a 1D Cartesian grid. STRONG
// shock with two orders of magnitude pressure jump — the canonical SRHD
// stress test for the substrate's quartic wave-speed + Newton c2p. left
// state (ρ=10, v=0, p=13.33), right state (ρ=1, v=0, p=1e-10), γ=4/3.
//
// usage:
//   cargo run --release -p symbi --example marti_muller -- --n 1000 \
//       --end-time 0.4 --n-checkpoints 5 --out output/marti_muller/data
//
// problem-specific knobs (via `--key val`):
//   --rho-left, --p-left      (defaults: 10.0, 13.33)
//   --rho-right, --p-right    (defaults: 1.0, 1e-10)
//   --discontinuity-x <0.5>
//   --bound <1.0>             domain extent [0, bound]
// =============================================================================

mod common;
use common::{BaseCli, Metadata, RunConfig};

use symbi::prelude::*;

type Sim = SimDefault<Srhd, 1, Cartesian, IdealGas<f64>>;


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = BaseCli::parse("marti_muller");

    let gamma     = cli.gamma.unwrap_or(4.0 / 3.0);
    let bound     = cli.extra_f64("bound", 1.0);
    let x_disc    = cli.extra_f64("discontinuity-x", 0.5);
    let rho_left  = cli.extra_f64("rho-left",  10.0);
    let p_left    = cli.extra_f64("p-left",    13.33);
    let rho_right = cli.extra_f64("rho-right", 1.0);
    let p_right   = cli.extra_f64("p-right",   1e-10);
    let n = cli.n1();
    let dx = bound / n as f64;

    eprintln!("[marti_muller] N = {n}, γ = {gamma}, end_time = {}", cli.end_time);

    // rest states (v = 0); the Srhd regime does the relativistic prim -> cons (D, S, tau).
    let mut sim = Sim::build(Srhd, IdealGas { gamma }, Cartesian)
        .cells([n])
        .origin([0.0])
        .spacing([dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(cli.cfl)
        .timestepping(cli.timestepping)
        .allocate()?
        .set_initial(|x| {
            let (rho, pre) = if x[0] < x_disc { (rho_left, p_left) } else { (rho_right, p_right) };
            Prim { rho, vel: Tensor::new([0.0]), pre }
        })
        .build();

    let sub = sim.substrate().with_solver(cli.solver)?;

    let metadata = Metadata::new()
        .with("problem", "marti_muller")
        .with("gamma",   gamma);

    RunConfig::new("marti_muller", &cli.out_dir, cli.end_time)
        .checkpoints(cli.n_checkpoints)
        .metadata(&metadata)
        .solver(cli.solver)
        .run(&mut sim, &sub)
}
