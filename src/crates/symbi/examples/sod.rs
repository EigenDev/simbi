// =============================================================================
// examples/sod.rs
//
// Sod (1978) shock tube on a 1D Cartesian grid — the canonical adiabatic-Euler
// Riemann test. left state (ρ=1, v=0, p=1), right state (ρ=0.125, v=0, p=0.1),
// γ=1.4, transmissive walls. produces an exact rarefaction + contact + shock
// triple that the substrate's adiabatic godunov should reproduce.
//
// usage:
//   cargo run --release -p symbi --example sod -- --n 256 --end-time 0.2 \
//       --n-checkpoints 5 --out output/sod/data
//
// problem-specific knobs (via `--key val`):
//   --rho-left  / --p-left  / --v-left
//   --rho-right / --p-right / --v-right
//   --discontinuity-x <0.5>  position of the initial discontinuity
//   --bound <1.0>            domain extent [0, bound]
// =============================================================================

mod common;
use common::{BaseCli, Metadata, RunConfig};

use symbi::prelude::*;

type Sim = SimDefault<Newtonian, 1, Cartesian, IdealGas<f64>>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = BaseCli::parse("sod");

    let gamma     = cli.gamma.unwrap_or(1.4);
    let bound     = cli.extra_f64("bound", 1.0);
    let x_disc    = cli.extra_f64("discontinuity-x", 0.5);
    let rho_left  = cli.extra_f64("rho-left",  1.0);
    let v_left    = cli.extra_f64("v-left",    0.0);
    let p_left    = cli.extra_f64("p-left",    1.0);
    let rho_right = cli.extra_f64("rho-right", 0.125);
    let v_right   = cli.extra_f64("v-right",   0.0);
    let p_right   = cli.extra_f64("p-right",   0.1);
    let n = cli.n1();
    let dx = bound / n as f64;

    eprintln!("[sod] N = {n}, γ = {gamma}, end_time = {}, discontinuity at x = {x_disc}",
        cli.end_time);

    let mut sim = Sim::build(Newtonian, IdealGas { gamma }, Cartesian)
        .cells([n])
        .origin([0.0])
        .spacing([dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(cli.cfl)
        .timestepping(cli.timestepping)
        .allocate()?
        .set_initial(|x| {
            let (rho, v, pre) = if x[0] < x_disc {
                (rho_left, v_left, p_left)
            } else {
                (rho_right, v_right, p_right)
            };
            Prim { rho, vel: Tensor::new([v]), pre }
        })
        .build();

    let sub = sim.substrate().with_solver(cli.solver)?;

    let metadata = Metadata::new()
        .with("problem", "sod")
        .with("gamma",   gamma);

    RunConfig::new("sod", &cli.out_dir, cli.end_time)
        .checkpoints(cli.n_checkpoints)
        .metadata(&metadata)
        .solver(cli.solver)
        .run(&mut sim, &sub)
}
