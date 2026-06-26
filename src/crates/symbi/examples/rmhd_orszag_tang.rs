// =============================================================================
// examples/rmhd_orszag_tang.rs
//
// the Orszag-Tang vortex — 2D MHD turbulence test, RMHD-relativistic. matches
// legacy_examples/orszag_tang.py:
//
//   • γ = 5/3
//   • ρ₀ = γ² = 25/9
//   • p₀ = γ = 5/3
//   • v₀ = 0.5, B₀ = 1.0
//   • vx = -v₀·sin(2π y),  vy = +v₀·sin(2π x),  vz = 0
//   • face-staggered B (Constrained-Transport ground truth):
//        Bx(face_x, y) = -B₀·sin(2π y)
//        By(x, face_y) = +B₀·sin(4π x)
//        Bz            = 0
//   • domain [0,1]×[0,1], periodic
//   • end_time defaults to (XMAX-XMIN)/cs  where  cs = (γ-1)/γ
//
// runs on a GENUINE 2.5D grid (spatial D=2, vector DOF=3 — docs/design/30): no z
// axis / z-ghosts / z-sweep, matching the 2D physics of the problem.
//
// usage:
//   cargo run --release -p symbi --example rmhd_orszag_tang -- --n 256 \
//       --end-time 0 --n-checkpoints 10 --out output/rmhd_orszag_tang/data
//
// GPU:
//   cargo run --release -p symbi --example rmhd_orszag_tang --features cuda -- ...
//
// problem-specific knobs (via `--key val`):
//   --v0 <0.5>   velocity scale
//   --b0 <1.0>   magnetic field scale
// =============================================================================

mod common;
use common::{BaseCli, Metadata, RunConfig};

use std::f64::consts::PI;

use symbi::prelude::*;

// genuine 2.5D MHD: D=2 spatial axes, DOF=3 vector components.
type Sim = SimDefaultGeneric<Rmhd, 2, 3, Cartesian, IdealGas<f64>>;

const XMIN: f64 = 0.0;
const XMAX: f64 = 1.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = BaseCli::parse("rmhd_orszag_tang");

    let gamma = cli.gamma.unwrap_or(5.0 / 3.0);
    let v0    = cli.extra_f64("v0", 0.5);
    let b0    = cli.extra_f64("b0", 1.0);
    let [nx, ny] = cli.n2();

    // Python's `cs = (γ-1)/γ`, auto end_time = (XMAX - XMIN) / cs.
    let cs = (gamma - 1.0) / gamma;
    let t_final = if cli.end_time > 0.0 { cli.end_time } else { (XMAX - XMIN) / cs };

    let dx = (XMAX - XMIN) / nx as f64;
    let dy = (XMAX - XMIN) / ny as f64;

    let rho_0 = gamma * gamma;
    let p_0   = gamma;

    eprintln!("[rmhd_orszag_tang] 2.5D grid = {nx}×{ny}, γ = {gamma}");
    eprintln!("[rmhd_orszag_tang] ρ₀ = γ² = {rho_0:.4}, p₀ = γ = {p_0:.4}, v₀ = {v0}, B₀ = {b0}");
    eprintln!("[rmhd_orszag_tang] end_time = {t_final} (cs = (γ-1)/γ = {cs:.4})");

    // ---- cell state + staggered B-field (CT ground truth) ----
    // Bx lives on the lower-x face of cell i, cell-CENTERED in y/z (and By on the y-face,
    // cell-centered in x). seed_faces routes through face_coord() so the staggered position
    // is exact — sampling the transverse coord at the edge would break OT point symmetry. the
    // CT bcell-from-bface kernel keeps the in-plane bcell consistent with the faces in evolution.
    let mut sim = Sim::build(Rmhd, IdealGas { gamma }, Cartesian)
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

    let sub = sim.substrate().with_solver(cli.solver)?;

    let metadata = Metadata::new()
        .with("problem", "rmhd_orszag_tang")
        .with("gamma",   gamma)
        .with("v0",      v0)
        .with("b0",      b0)
        .with("rho_0",   rho_0)
        .with("p_0",     p_0);

    RunConfig::new("rmhd_orszag_tang", &cli.out_dir, t_final).checkpoints(cli.n_checkpoints).metadata(&metadata).solver(cli.solver).run(&mut sim, &sub)
}
