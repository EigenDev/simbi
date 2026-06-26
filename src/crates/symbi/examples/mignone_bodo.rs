// =============================================================================
// examples/mignone_bodo.rs
//
// Mignone & Bodo (2005) relativistic shock tube on a 1D Cartesian grid. two
// problems selectable via `--problem`:
//   1 (default): mild — ρL=1, vL=0, pL=1; ρR=0.1, vR=0, pR=0.125 (γ=4/3)
//   2:           moving — ρL=1, vL=-0.2, pL=0.4; ρR=1, vR=+0.2, pR=0.4 (γ=4/3)
// problem 2 is the canonical "v ≠ 0" SRHD test — exercises the full
// prim→cons transform via the Lorentz factor.
//
// usage:
//   cargo run --release -p symbi --example mignone_bodo -- --problem 2 \
//       --n 1000 --end-time 0.4 --n-checkpoints 5 \
//       --out output/mignone_bodo/data
//
// problem-specific knobs (via `--key val`):
//   --problem <1|2>             which test (default 1)
//   --discontinuity-x <0.5>
//   --bound <1.0>               domain extent [0, bound]
// =============================================================================

mod common;
use common::{BaseCli, Metadata, RunConfig};

use symbi::prelude::*;

type Sim = SimDefault<Srhd, 1, Cartesian, IdealGas<f64>>;

/// SRHD prim (ρ, v, p) → cons (D, S, τ) via the Lorentz factor.
/// D = ρW, S = ρhW²v, τ = ρhW² - p - D, where W = 1/√(1-v²),
/// h = 1 + γp/(ρ(γ-1)) (ideal-gas relativistic enthalpy).

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = BaseCli::parse("mignone_bodo");

    let gamma     = cli.gamma.unwrap_or(4.0 / 3.0);
    let problem   = cli.extra_int("problem", 1);
    let bound     = cli.extra_f64("bound", 1.0);
    let x_disc    = cli.extra_f64("discontinuity-x", 0.5);
    let n = cli.n1();
    let dx = bound / n as f64;

    let (prim_l, prim_r): ((f64, f64, f64), (f64, f64, f64)) = match problem {
        1 => ((1.0, 0.0, 1.0), (0.1, 0.0, 0.125)),
        2 => ((1.0, -0.2, 0.4), (1.0, 0.2, 0.4)),
        other => panic!("--problem must be 1 or 2 (got {other})"),
    };

    eprintln!("[mignone_bodo] problem {problem}, N = {n}, γ = {gamma}, end_time = {}",
        cli.end_time);
    eprintln!("[mignone_bodo] left  prim (ρ, v, p) = ({}, {}, {})", prim_l.0, prim_l.1, prim_l.2);
    eprintln!("[mignone_bodo] right prim (ρ, v, p) = ({}, {}, {})", prim_r.0, prim_r.1, prim_r.2);

    // boosted relativistic states; the Srhd regime does the prim -> cons (D, S, tau).
    let mut sim = Sim::build(Srhd, IdealGas { gamma }, Cartesian)
        .cells([n])
        .origin([0.0])
        .spacing([dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(cli.cfl)
        .timestepping(cli.timestepping)
        .allocate()?
        .set_initial(|x| {
            let (rho, v, pre) = if x[0] < x_disc { prim_l } else { prim_r };
            Prim { rho, vel: Tensor::new([v]), pre }
        })
        .build();

    let sub = sim.substrate().with_solver(cli.solver)?;

    let metadata = Metadata::new()
        .with("problem",         "mignone_bodo")
        .with("gamma",           gamma)
        .with("problem_variant", problem);

    RunConfig::new("mignone_bodo", &cli.out_dir, cli.end_time)
        .checkpoints(cli.n_checkpoints)
        .metadata(&metadata)
        .solver(cli.solver)
        .run(&mut sim, &sub)
}
