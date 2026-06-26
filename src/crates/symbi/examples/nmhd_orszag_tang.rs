// =============================================================================
// examples/nmhd_orszag_tang.rs
//
// the Orszag-Tang vortex in NEWTONIAN (non-relativistic) ideal MHD, run with the
// HLLD five-wave solver by default — the robustness payoff of the NMHD regime: the
// ALGEBRAIC c2p cannot fail in the current sheets that form, and closed-form HLLD
// resolves the contact/alfven structure HLLE smears.
//
// runs on a GENUINE 2.5D grid (spatial D=2, vector DOF=3 — docs/design/30): no z
// axis, no z-ghosts, no z-sweep. ~3-4x faster than the old 3D-with-nz=1 hack (and
// correct on GPU). the out-of-plane Bz/vz are carried (cell-centered) but zero here.
//
// usage:
//   cargo run --release -p symbi --example nmhd_orszag_tang -- --n 256 \
//       --end-time 0 --n-checkpoints 10 --out output/nmhd_orszag_tang/data
//
//   defaults to HLLD; override with --solver hlle|hllc|hlld.
//
// problem knobs (via `--key val`): --v0 <1.0>  --b0 <1/sqrt(4pi)>
// =============================================================================

mod common;
use common::{BaseCli, Metadata, RunConfig};

use std::f64::consts::PI;

use symbi::prelude::*;

// genuine 2.5D MHD: D=2 spatial axes, DOF=3 vector components.
type Sim = SimDefaultGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>>;

const XMIN: f64 = 0.0;
const XMAX: f64 = 1.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cli = BaseCli::parse("nmhd_orszag_tang");

    let gamma = cli.gamma.unwrap_or(5.0 / 3.0);
    // canonical Athena Newtonian OT: v amplitude 1, B0 = 1/sqrt(4pi).
    let v0 = cli.extra_f64("v0", 1.0);
    let b0 = cli.extra_f64("b0", 1.0 / (4.0 * PI).sqrt());
    let [nx, ny] = cli.n2();

    // HLLD is the headline here — default to it (so the dashboard + the run agree),
    // but respect an explicit --solver hlle|hllc|hlld.
    if !std::env::args().any(|a| a == "--solver") {
        cli.solver = Solver::Hlld;
    }
    let solver = cli.solver;

    let dx = (XMAX - XMIN) / nx as f64;
    let dy = (XMAX - XMIN) / ny as f64;

    // canonical Athena Newtonian OT: rho = 25/(36 pi), p = 5/(12 pi) -> cs^2 = gamma p/rho = 1.
    let rho_0 = 25.0 / (36.0 * PI);
    let p_0 = 5.0 / (12.0 * PI);
    let cs = (gamma * p_0 / rho_0).sqrt(); // = 1
    let t_final = if cli.end_time > 0.0 { cli.end_time } else { (XMAX - XMIN) / cs };

    eprintln!("[nmhd_orszag_tang] 2.5D grid = {nx}x{ny}, gamma = {gamma}, solver = {solver:?}");
    eprintln!("[nmhd_orszag_tang] rho0 = {rho_0:.4}, p0 = {p_0:.4}, cs = {cs:.4}, v0 = {v0}, B0 = {b0:.4}");
    eprintln!("[nmhd_orszag_tang] end_time = {t_final}");

    // cell state + staggered in-plane B (CT ground truth). face_coord (via seed_faces) owns the
    // half-cell offset so Bx on the x-face is exact in x but cell-centered in y (and vice-versa
    // for By) — sampling the transverse coord at the edge would break OT 180-degree point symmetry.
    let mut sim = Sim::build(NewtonianMhd, IdealGas { gamma }, Cartesian)
        .cells([nx, ny])
        .origin([XMIN, XMIN])
        .spacing([dx, dy])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(cli.cfl)
        .timestepping(cli.timestepping)
        .allocate()?
        .set_initial(|[x, y]| {
            let vel = Tensor::new([-v0 * (2.0 * PI * y).sin(), v0 * (2.0 * PI * x).sin(), 0.0]);
            let mag = Tensor::new([-b0 * (2.0 * PI * y).sin(), b0 * (4.0 * PI * x).sin(), 0.0]);
            MhdPrim { hydro: Prim { rho: rho_0, vel, pre: p_0 }, mag }
        })
        .seed_faces(|axis, [x, y]| match axis {
            0 => -b0 * (2.0 * PI * y).sin(),
            _ => b0 * (4.0 * PI * x).sin(),
        })
        .build();

    let sub = sim.substrate().with_solver(solver)?;

    let metadata = Metadata::new()
        .with("problem", "nmhd_orszag_tang")
        .with("gamma", gamma)
        .with("solver", format!("{solver:?}"))
        .with("v0", v0)
        .with("b0", b0)
        .with("rho_0", rho_0)
        .with("p_0", p_0);

    RunConfig::new("nmhd_orszag_tang", &cli.out_dir, t_final)
        .checkpoints(cli.n_checkpoints)
        .metadata(&metadata)
        .solver(solver)
        .run(&mut sim, &sub)
}
