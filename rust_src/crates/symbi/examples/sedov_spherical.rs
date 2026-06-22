// =============================================================================
// examples/sedov_spherical.rs
//
// Sedov-Taylor blast wave on a 2D SPHERICAL (r, theta) grid. high-pressure thermal
// energy deposited in the innermost radial shells expands outward as a spherical blast.
// the initial condition is theta-UNIFORM (a radial shell), so a correct run stays
// theta-symmetric — an expanding spherical shell, a horizontal band in (r, theta).
//
// this is the geometric-source demonstrator: on the spherical grid the evolution runs
// through the area-weighted curvilinear divergence + the metric momentum source
// (S_r = (rho v_t^2 + 2p)/r, well-balanced against the r^2-area pressure flux). a
// Cartesian run on the same index grid would NOT expand spherically — the difference IS
// the geometric source. (the substrate's `substrate_spherical_sod` test asserts exactly
// this: spherical != cartesian on the identical grid.)
//
// the theta range is a WEDGE around the equator, kept away from the poles (theta = 0, pi)
// where the metric's cot(theta) term is singular — the standard 2D (r, theta) practice.
//
// usage:
//   cargo run --release -p symbi --example sedov_spherical -- --n 256 --end-time 0.4 \
//       --n-checkpoints 12 --out output/sedov_sph/data
//   uv run scripts/plot_sedov_spherical.py output/sedov_sph/data/sedov_spherical_*.h5
//
// problem-specific knobs (via `--key val`):
//   --e0 <1.0>            total deposited energy (over the inner shell, axisymmetric)
//   --rho-amb <1.0>       ambient density
//   --p-amb <1e-5>        ambient pressure (~ cold)
//   --r-min <0.1>         inner radius (spherical is singular at r=0)
//   --r-max <1.0>         outer radius
//   --r-blast <0.15>      blast deposition radius (energy in r <= r_blast)
//   --theta-lo <pi/2-0.5> wedge lower theta (kept off the poles)
//   --theta-span <1.0>    wedge angular width
// =============================================================================

mod common;
use common::{BaseCli, Metadata, RunConfig};

use std::f64::consts::PI;
use symbi::prelude::*;

type Sim = SimDefault<Newtonian, 2, Spherical, IdealGas<f64>>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = BaseCli::parse("sedov_spherical");

    let gamma     = cli.gamma.unwrap_or(5.0 / 3.0);
    let e0        = cli.extra_f64("e0", 1.0);
    let rho_amb   = cli.extra_f64("rho-amb", 1.0);
    let p_amb     = cli.extra_f64("p-amb", 1e-5);
    let r_min     = cli.extra_f64("r-min", 0.1);
    let r_max     = cli.extra_f64("r-max", 1.0);
    let r_blast   = cli.extra_f64("r-blast", 0.15);
    let theta_lo  = cli.extra_f64("theta-lo", PI / 2.0 - 0.5);
    let theta_span = cli.extra_f64("theta-span", 1.0);

    let [nr, nt] = cli.n2();
    let dr = (r_max - r_min) / nr as f64;
    let dt = theta_span / nt as f64;
    let theta_hi = theta_lo + theta_span;

    // deposit E_0 over the inner shell r <= r_blast, uniform in theta. axisymmetric volume of
    // the wedge shell: V = 2*pi * (cos(theta_lo) - cos(theta_hi)) * (r_blast^3 - r_min^3)/3.
    // p_blast follows from E_0 = p_blast/(gamma-1) * V (internal energy, v = 0).
    let v_blast = 2.0 * PI * (theta_lo.cos() - theta_hi.cos())
        * (r_blast.powi(3) - r_min.powi(3)) / 3.0;
    let p_blast = (gamma - 1.0) * e0 / v_blast.max(1e-30);

    eprintln!(
        "[sedov_spherical] grid = {nr}x{nt} (r,theta), r in [{r_min}, {r_max}], \
         theta in [{theta_lo:.3}, {theta_hi:.3}], gamma = {gamma}, E_0 = {e0}, p_blast = {p_blast:.3e}"
    );

    // theta-uniform radial-shell deposition: a hot over-pressured inner shell, v = 0.
    // the closure gets the native (r, theta) coordinate; the regime converts prim -> cons.
    let mut sim = Sim::build(Newtonian, IdealGas { gamma }, Spherical)
        .cells([nr, nt])
        .origin([r_min, theta_lo])
        .spacing([dr, dt])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(cli.cfl)
        .timestepping(cli.timestepping)
        .allocate()?
        .set_initial(|x| {
            let r = x[0];
            let pre = if r <= r_blast { p_blast } else { p_amb };
            Prim { rho: rho_amb, vel: Tensor::new([0.0, 0.0]), pre }
        })
        .build();

    let sub = sim.substrate().with_solver(cli.solver)?;

    let metadata = Metadata::new()
        .with("problem", "sedov_spherical")
        .with("gamma",   gamma)
        .with("e0",      e0)
        .with("r_blast", r_blast);

    RunConfig::new("sedov_spherical", &cli.out_dir, cli.end_time)
        .checkpoints(cli.n_checkpoints)
        .metadata(&metadata)
        .solver(cli.solver)
        .run(&mut sim, &sub)
}
