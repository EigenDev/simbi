// =============================================================================
// examples/sedov3d.rs
//
// 3D Cartesian Sedov–Taylor blast — the 3D Newtonian uniform-grid throughput
// reference (for apples-to-apples MZCS vs AthenaK Newtonian 3D). same scheme as
// the 2D sedov: adiabatic Euler, HLLC + theta-MC PLM, SSP-RK2, outflow BCs.
//
// usage:
//   cargo run --release -p symbi --example sedov3d -- --n 128 --end-time 0.05 \
//       --n-checkpoints 0 --out output/sedov3d/data
// =============================================================================

mod common;
use common::{BaseCli, Metadata, RunConfig};

use symbi::prelude::*;

type Sim = SimDefault<Newtonian, 3, Cartesian, IdealGas<f64>>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = BaseCli::parse("sedov3d");

    let gamma   = cli.gamma.unwrap_or(5.0 / 3.0);
    let bound   = cli.extra_f64("bound", 0.5);
    let e0      = cli.extra_f64("e0", 1.0);
    let rho_amb = cli.extra_f64("rho-amb", 1.0);
    let p_amb   = cli.extra_f64("p-amb", 1e-5);
    let r_blast = cli.extra_f64("r-blast", 0.1);
    let [nx, ny, nz] = cli.n3();
    let dx = 2.0 * bound / nx as f64;
    let dy = 2.0 * bound / ny as f64;
    let dz = 2.0 * bound / nz as f64;

    eprintln!("[sedov3d] grid = {nx}x{ny}x{nz}, γ = {gamma}, E_0 = {e0}, blast radius = {r_blast}");

    // pressure inside the sphere: (γ-1) E_0 / (4/3 π r_blast³) (3D analog).
    let vol = (4.0 / 3.0) * std::f64::consts::PI * r_blast.powi(3);
    let p_blast = (gamma - 1.0) * e0 / vol;
    let mut sim = Sim::build(Newtonian, IdealGas { gamma }, Cartesian)
        .cells([nx, ny, nz])
        .origin([-bound, -bound, -bound])
        .spacing([dx, dy, dz])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(cli.cfl)
        .timestepping(cli.timestepping)
        .allocate()?
        .set_initial(|p| {
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            let pre = if r <= r_blast { p_blast } else { p_amb };
            Prim { rho: rho_amb, vel: Tensor::new([0.0, 0.0, 0.0]), pre }
        })
        .build();

    let sub = sim.substrate().with_solver(cli.solver)?;

    let metadata = Metadata::new()
        .with("problem", "sedov3d")
        .with("gamma",   gamma)
        .with("e0",      e0)
        .with("r_blast", r_blast);

    RunConfig::new("sedov3d", &cli.out_dir, cli.end_time)
        .checkpoints(cli.n_checkpoints)
        .metadata(&metadata)
        .solver(cli.solver)
        .run(&mut sim, &sub)
}
