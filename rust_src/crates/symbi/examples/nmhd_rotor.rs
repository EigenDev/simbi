// =============================================================================
// examples/nmhd_rotor.rs
//
// the magnetized ROTOR test (Balsara & Spicer 1999; Tóth 2000 test 1) in NEWTONIAN
// ideal MHD on a GENUINE 2.5D grid (spatial D=2, vector DOF=3 — docs/design/30). a
// dense disk spins inside a uniform Bx field: the rotation winds the field into
// torsional Alfvén waves that launch outward, and the low-beta core is a brutal
// robustness + div(B) stress test. run with HLLD by default.
//
// IC (Tóth test 1, gamma=1.4) on [0,1]^2, centre (0.5,0.5), r0=0.1, r1=0.115:
//   r<r0      : rho=10, v = v0/r0 * (-(y-yc), (x-xc)),  v0=2
//   r0<r<r1   : f=(r1-r)/(r1-r0); rho=1+9f; v = f*v0/r * (-(y-yc), (x-xc))
//   r>r1      : rho=1, v=0
//   p=1 everywhere; B=(5/sqrt(4pi), 0, 0); vz=Bz=0.
//
// usage:
//   cargo run --release -p symbi --example nmhd_rotor -- --n 256 \
//       --end-time 0.15 --n-checkpoints 15 --out output/nmhd_rotor/data
// =============================================================================

mod common;
use common::{BaseCli, Metadata, RunConfig};

use std::f64::consts::PI;

use symbi::prelude::*;

// genuine 2.5D MHD: D=2 spatial axes, DOF=3 vector components; feature-selected backend
// (GPU under `--features cuda`, else CPU) — no per-file `#[cfg(feature="cuda")]` block.
type Sim = SimDefaultGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>>;

const XC: f64 = 0.5;
const R0: f64 = 0.1;
const R1: f64 = 0.115;
const V0: f64 = 2.0;

// the rotor primitive (rho, vx, vy) at physical position (x,y); p=1, vz=0 elsewhere.
fn rotor_state(x: f64, y: f64) -> (f64, f64, f64) {
    let (dx, dy) = (x - XC, y - XC);
    let r = (dx * dx + dy * dy).sqrt();
    if r < R0 {
        (10.0, -V0 * dy / R0, V0 * dx / R0)
    } else if r < R1 {
        let f = (R1 - r) / (R1 - R0);
        (1.0 + 9.0 * f, -f * V0 * dy / r, f * V0 * dx / r)
    } else {
        (1.0, 0.0, 0.0)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cli = BaseCli::parse("nmhd_rotor");

    let gamma = cli.gamma.unwrap_or(1.4);
    let b0 = cli.extra_f64("b0", 5.0 / (4.0 * PI).sqrt());
    let p0 = cli.extra_f64("p0", 1.0);
    let [nx, ny] = cli.n2();

    if !std::env::args().any(|a| a == "--solver") {
        cli.solver = Solver::Hlld;
    }
    let solver = cli.solver;

    let dx = 1.0 / nx as f64;
    let dy = 1.0 / ny as f64;
    let t_final = if cli.end_time > 0.0 { cli.end_time } else { 0.15 };

    eprintln!("[nmhd_rotor] 2.5D grid = {nx}x{ny}, gamma = {gamma}, solver = {solver:?}");
    eprintln!("[nmhd_rotor] B0 = {b0:.4}, p0 = {p0}, end_time = {t_final}");

    // uniform Bx on the x-faces (the staggered CT ground truth); By stays 0 on the y-faces.
    let mut sim = Sim::build(NewtonianMhd, IdealGas { gamma }, Cartesian)
        .cells([nx, ny])
        .spacing([dx, dy])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(cli.cfl)
        .timestepping(cli.timestepping)
        .allocate()?
        .set_initial(|[x, y]| {
            let (rho, vx, vy) = rotor_state(x, y);
            MhdPrim {
                hydro: Prim { rho, vel: Tensor::new([vx, vy, 0.0]), pre: p0 },
                mag: Tensor::new([b0, 0.0, 0.0]),
            }
        })
        .seed_faces_uniform([b0, 0.0])
        .build();

    // matched KernelSet straight off the sim (gamma/cfl/alloc pulled from it); tune theta + solver.
    let sub = sim.substrate().theta(1.5).with_solver(solver)?;

    let metadata = Metadata::new()
        .with("problem", "nmhd_rotor")
        .with("gamma", gamma)
        .with("solver", format!("{solver:?}"))
        .with("b0", b0)
        .with("p0", p0);

    RunConfig::new("nmhd_rotor", &cli.out_dir, t_final)
        .checkpoints(cli.n_checkpoints)
        .metadata(&metadata)
        .solver(solver)
        .run(&mut sim, &sub)
}
