// =============================================================================
// examples/rt.rs
//
// Rayleigh-Taylor instability — 2D adiabatic Newtonian gas with a heavy fluid
// resting on top of a light fluid in a uniform downward gravity. the unstable
// interface, given a small perturbation, develops the classic rising-bubble /
// falling-spike mushrooms.
//
// a direct port of legacy_examples/rt.py:
//   domain  x in [-0.25, 0.25], y in [-0.75, 0.75]   (a 1:3 box; square cells)
//   rho     2.0 above the y=0 interface, 1.0 below
//   p       p0 - g0*rho*y                              (hydrostatic equilibrium)
//   vy      vamp * 1/4 * (1 + cos(4 pi x)) * (1 + cos(3 pi y))   (the seed)
//   BC      periodic in x, reflecting (solid wall) in y
//   gravity uniform [0, -g0]  -> the fused body force S_mom = rho*g
//
// the gravity is a FUSED source: `uniform_accel([0, -g0])` routed through
// `configure_source` into the AOT-baked `adiabatic_godunov_*_with_uniform_accel_2d`
// kernel (or the proven-equal additive pass) — applied every RK stage inside the
// real evolve loop.
//
// usage (the python-equivalent run — HLLC, t=10, 200x600):
//   cargo run --release -p symbi --example rt -- --n 200,600 --solver hllc \
//       --end-time 10 --n-checkpoints 50 --out output/rt/data
//   uv run python3 scripts/plot_rt.py output/rt/data/rt_*.h5
//
// a single `--n N` gives an N x 3N box (square cells); the mushrooms need
// t ~ 5..10 to grow (RT e-folding ~ 1.5 at g0 = 0.1).
//
// problem-specific knobs (via `--key val`):
//   --g0 <0.1>          gravitational acceleration (downward)
//   --rho-u <2.0>       density of the upper (heavy) fluid
//   --rho-d <1.0>       density of the lower (light) fluid
//   --p0 <2.5>          reference pressure (at the y = 0 interface)
//   --vamp <0.01>       vy perturbation amplitude
//
// numerics (base knobs, see --help): --solver hlle|hllc (python uses hllc),
//   --theta <1..2> PLM limiter, --timestepping euler|rk2, --cfl, --gamma,
//   --end-time, --n.
// =============================================================================

mod common;
use common::{BaseCli, Metadata, RunConfig};

use symbi::prelude::*;
use symbi::regimes::source_config::configure_source;
use symbi_hydro::uniform_accel;

const PI: f64 = std::f64::consts::PI;

type Sim = SimDefault<Newtonian, 2, Cartesian, IdealGas<f64>>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = BaseCli::parse("rt");

    let gamma = cli.gamma.unwrap_or(7.0 / 5.0);
    let g0    = cli.extra_f64("g0", 0.1);
    let rho_u = cli.extra_f64("rho-u", 2.0);
    let rho_d = cli.extra_f64("rho-d", 1.0);
    let p0    = cli.extra_f64("p0", 2.5);
    let vamp  = cli.extra_f64("vamp", 0.01);

    // domain x in [-0.25, 0.25] (width 0.5), y in [-0.75, 0.75] (height 1.5) — the
    // python box. resolution: `--n Nx,Ny`, or a single `--n N` -> N x 3N (square cells).
    let nx = cli.n[0];
    let ny = cli.n.get(1).copied().unwrap_or(3 * cli.n[0]);
    let x_lo = -0.25;
    let y_lo = -0.75;
    let lx = 0.5;
    let ly = 1.5;
    let dx = lx / nx as f64;
    let dy = ly / ny as f64;

    eprintln!("[rt] grid = {nx}x{ny}, gamma = {gamma}, g0 = {g0}, solver = {:?}, end_time = {}",
        cli.solver, cli.end_time);
    eprintln!("[rt] heavy rho = {rho_u} above y=0, light rho = {rho_d} below, p0 = {p0}");

    let mut sim = Sim::build(Newtonian, IdealGas { gamma }, Cartesian)
        .cells([nx, ny])
        .origin([x_lo, y_lo])
        .spacing([dx, dy])
        // periodic in x; reflecting (solid wall) top and bottom — the stratified column
        // is held in place so the gravity source can't drain through the y faces.
        .boundaries(
            Boundaries::uniform(BoundaryType::Periodic)
                .axis(1, BoundaryType::Reflect, BoundaryType::Reflect),
        )
        .cfl(cli.cfl)
        .timestepping(cli.timestepping)
        .allocate()?
        .set_initial(|[x, y]| {
            // heavy fluid above the y = 0 interface, light below.
            let rho = if y > 0.0 { rho_u } else { rho_d };
            // hydrostatic balance dp/dy = rho*g_ext_y = -rho*g0, continuous at y = 0.
            let pre = p0 - g0 * rho * y;
            // the python seed: fixed-wavenumber single mode, strongest at the interface.
            let vy = vamp * 0.25 * (1.0 + (4.0 * PI * x).cos()) * (1.0 + (3.0 * PI * y).cos());
            Prim { rho, vel: Tensor::new([0.0, vy]), pre }
        })
        .build();

    // gravity as a fused body-force overlay; solver + PLM theta set on the kernel set first.
    let sub = configure_source(
        sim.substrate().with_solver(cli.solver)?.theta(cli.theta),
        &uniform_accel(vec![0.0, -g0]),
        "adiabatic",
        2,
    );

    let metadata = Metadata::new()
        .with("problem", "rayleigh_taylor")
        .with("gamma", gamma)
        .with("g0", g0)
        .with("rho_u", rho_u)
        .with("rho_d", rho_d)
        .with("p0", p0)
        .with("vamp", vamp);

    RunConfig::new("rt", &cli.out_dir, cli.end_time)
        .checkpoints(cli.n_checkpoints)
        .metadata(&metadata)
        .solver(cli.solver)
        .run(&mut sim, &sub)
}
