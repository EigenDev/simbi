// =============================================================================
// examples/kepler.rs
//
// thin Gaussian ring of matter on a Keplerian orbit around a softened
// point-mass at the origin, integrated on a 2D Cartesian grid by the AOT-baked
// `iso_godunov_*_with_point_mass_grav_2d` fused kernel — gravity FUSED into
// the integrator (Phase 2c position binding, Phase 3b AOT bake, Phase 4c
// SimulationLaws derivation, Phase 4b kernel-set dispatch, Phase 6 GPU-
// validated on RTX 2070). produces HDF5 checkpoints `scripts/plot_kepler.py`
// reads directly.
//
// usage (smoke — ~30 s on CPU at N=64, 0.5 orbit):
//   cargo run --release -p symbi --example kepler -- --n 64 --end-time 0.5 \
//       --n-checkpoints 10 --out output/kepler/data
//   uv run python3 scripts/plot_kepler.py output/kepler/data/kepler_final.h5
//
// usage (production — N=512, 10 orbits — ~30 min CPU):
//   cargo run --release -p symbi --example kepler -- --n 512 --end-time 10 \
//       --n-checkpoints 20 --out output/kepler/data
//   uv run python3 scripts/plot_kepler.py --1d output/kepler/data/kepler_*.h5
//
// problem-specific knobs (via `--key val`):
//   --gm <1.0>     central mass × G
//   --eps <r-inner-mask>  plummer softening length for the central mass
//   --r0 <1.0>     ring center radius
//   --dr <0.1>     ring Gaussian half-thickness
//   --bound <2.0>  domain half-width (extent = ±bound on each axis)
//   --cs <0.01>    isothermal sound speed
//   --r-inner-mask <0.2>  below this radius, gas starts at rest (avoids
//                         the v_kep ~ 1/√r singularity at r→0)
//
// --end-time is interpreted in ORBITS (one orbit = 2π at GM=1, r=1).
// =============================================================================

mod common;
use common::{BaseCli, Metadata, RunConfig};

use symbi::prelude::*;
use symbi::regimes::source_config::configure_source;
use symbi_hydro::energy::IsoModel;
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::point_mass;
use symbi_hydro::state::PrimG;

const TAU: f64 = 2.0 * std::f64::consts::PI;

type Sim = SimDefault<IsoNewtonian, 2, Cartesian, Isothermal<f64>>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = BaseCli::parse("kepler");

    let bound = cli.extra_f64("bound", 2.0);
    let gm    = cli.extra_f64("gm", 1.0);
    let r0    = cli.extra_f64("r0", 1.0);
    let dr    = cli.extra_f64("dr", 0.1);
    let cs    = cli.extra_f64("cs", 0.01);
    let r_inner_mask = cli.extra_f64("r-inner-mask", 0.2);
    // plummer softening for the central mass at the origin: keeps the gravitational
    // acceleration finite in the innermost cells. defaults to the inner-mask radius.
    let eps   = cli.extra_f64("eps", r_inner_mask);
    let [nx, ny] = cli.n2();
    let dx = 2.0 * bound / nx as f64;
    let dy = 2.0 * bound / ny as f64;
    let t_final = cli.end_time * TAU;

    eprintln!("[kepler] grid = {nx}x{ny}, end_time = {} orbits, bound = ±{bound}, GM = {gm}",
        cli.end_time);

    // ---- Gaussian ring + Keplerian rotation, masked inner core ----
    let mut sim = Sim::build(IsoNewtonian, Isothermal { cs }, Cartesian)
        .cells([nx, ny])
        .origin([-bound, -bound])
        .spacing([dx, dy])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(cli.cfl)
        .timestepping(cli.timestepping)
        .allocate()?
        .set_initial(|[x, y]| {
            let r = (x * x + y * y).sqrt();
            let sigma_min = 1e-6;
            let sigma = sigma_min + 1.0 * (-((r - r0).powi(2)) / (2.0 * dr.powi(2))).exp();
            let (vx, vy) = if r >= r_inner_mask {
                let v_kep = (gm / r).sqrt();
                (-v_kep * (y / r), v_kep * (x / r))
            } else {
                (0.0, 0.0)
            };
            PrimG::<f64, 2, IsoModel> { rho: sigma, vel: Tensor::new([vx, vy]), pre: Default::default() }
        })
        .build();

    // ---- declare the source overlay → kernel set (fused-when-baked auto-choice) ----
    // `configure_source` consults the AOT registry: point_mass IS baked for iso 2d,
    // so this routes the fused godunov kernel (else the proven-equal additive pass).
    let sub = configure_source(
        sim.substrate(),
        &point_mass(gm, vec![0.0, 0.0], eps), "iso", 2,
    );

    let metadata = Metadata::new()
        .with("problem", "kepler")
        .with("ring_r0", r0)
        .with("ring_dr", dr)
        .with("gm",      gm)
        .with("cs",      cs);

    RunConfig::new("kepler", &cli.out_dir, t_final)
        .checkpoints(cli.n_checkpoints)
        .metadata(&metadata)
        .time_unit(TAU, "T_orb")
        .run(&mut sim, &sub)
}
