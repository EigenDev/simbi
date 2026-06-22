// =============================================================================
// examples/imhd_orszag_tang.rs
//
// the Orszag-Tang vortex in ISOTHERMAL ideal MHD (Mignone 2007), run with the
// 3-state HLLD solver by default. the isothermal closure p = cs^2 rho drops the
// energy equation entirely — the conserved state is just (rho, rho v, B), the c2p
// is trivial (and cannot fail), and the CT stack is shared with NMHD/RMHD.
//
// runs on a GENUINE 2.5D grid (spatial D=2, vector DOF=3 — docs/design/30): no z
// axis / z-ghosts / z-sweep.
//
// usage:
//   cargo run --release -p symbi --example imhd_orszag_tang -- --n 256 \
//       --end-time 1 --n-checkpoints 10 --out output/imhd_orszag_tang/data
//
//   defaults to HLLD; override with --solver hlle|hlld.
// problem knobs (via `--key val`): --cs <1.0>  --v0 <1.0>  --b0 <1/sqrt(4pi)>
// =============================================================================

mod common;
use common::{BaseCli, Metadata, RunConfig};

use std::f64::consts::PI;

use symbi::prelude::*;
// the energy-model-generic primitive variants — the iso IC seeds an IsoModel prim (no pressure
// slot), which the prelude's concrete MhdPrim/Prim (energy model) don't cover.
use symbi_hydro::energy::IsoModel;
use symbi_hydro::mhd_state::MhdPrimG;
use symbi_hydro::state::PrimG;

// genuine 2.5D MHD: D=2 spatial axes, DOF=3 vector components.
type Sim = SimDefaultGeneric<IsothermalMhd, 2, 3, Cartesian, Isothermal<f64>>;

const XMIN: f64 = 0.0;
const XMAX: f64 = 1.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cli = BaseCli::parse("imhd_orszag_tang");

    let cs = cli.extra_f64("cs", 1.0);
    let v0 = cli.extra_f64("v0", 1.0);
    let b0 = cli.extra_f64("b0", 1.0 / (4.0 * PI).sqrt());
    let [nx, ny] = cli.n2();

    if !std::env::args().any(|a| a == "--solver") {
        cli.solver = Solver::Hlld;
    }
    let solver = cli.solver;

    let dx = (XMAX - XMIN) / nx as f64;
    let dy = (XMAX - XMIN) / ny as f64;
    let rho_0 = 1.0;
    let t_final = if cli.end_time > 0.0 { cli.end_time } else { (XMAX - XMIN) / cs };

    eprintln!("[imhd_orszag_tang] 2.5D grid = {nx}x{ny}, cs = {cs}, solver = {solver:?}");
    eprintln!("[imhd_orszag_tang] rho0 = {rho_0}, v0 = {v0}, B0 = {b0:.4}, end_time = {t_final}");

    let mut sim = Sim::build(IsothermalMhd, Isothermal { cs }, Cartesian)
        .cells([nx, ny])
        .origin([XMIN, XMIN])
        .spacing([dx, dy])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(cli.cfl)
        .timestepping(cli.timestepping)
        .allocate()?
        // cell state — the iso primitive has no pressure slot (ZST).
        .set_initial(|[x, y]| {
            let vel = Tensor::new([-v0 * (2.0 * PI * y).sin(), v0 * (2.0 * PI * x).sin(), 0.0]);
            let mag = Tensor::new([-b0 * (2.0 * PI * y).sin(), b0 * (4.0 * PI * x).sin(), 0.0]);
            MhdPrimG::<f64, 3, IsoModel> {
                hydro: PrimG { rho: rho_0, vel, pre: Default::default() },
                mag,
            }
        })
        // staggered in-plane B (CT ground truth): x is the per-axis face-midpoint coordinate.
        .seed_faces(|axis, [x, y]| match axis {
            0 => -b0 * (2.0 * PI * y).sin(),
            _ => b0 * (4.0 * PI * x).sin(),
        })
        .build();

    let sub = sim.substrate().with_solver(solver)?;

    let metadata = Metadata::new()
        .with("problem", "imhd_orszag_tang")
        .with("cs", cs)
        .with("solver", format!("{solver:?}"))
        .with("v0", v0)
        .with("b0", b0)
        .with("rho_0", rho_0);

    RunConfig::new("imhd_orszag_tang", &cli.out_dir, t_final)
        .checkpoints(cli.n_checkpoints)
        .metadata(&metadata)
        .solver(solver)
        .run(&mut sim, &sub)
}
