// =============================================================================
// flux_timing.rs
//
// single-core wall-time of the emitted 3d adiabatic flux kernel on a 128^3
// interior with ghost padding, via the name-keyed invocation — the exact AOT
// symbol production dispatch runs, at production compile flags. prints
// ns/zone for the minmod branch (theta = 1.5) and the van-leer branch
// (theta = -1.0) of the unswitched kernel.
//
// usage:
//   cargo run --release -p symbi-aot --example flux_timing
// =============================================================================

use std::hint::black_box;
use std::time::Instant;
use symbi_aot::NamedKernel;

fn main() {
    let gx: i32 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(128);
    let gy: i32 = std::env::args()
        .nth(3)
        .and_then(|a| a.parse().ok())
        .unwrap_or(gx);
    let gz: i32 = std::env::args()
        .nth(4)
        .and_then(|a| a.parse().ok())
        .unwrap_or(gy);
    let (px, py, pz) = (gx + 4, gy + 4, gz + 4);
    let n = (px * py * pz) as usize;
    let prim: Vec<f64> = (0..n)
        .map(|ii| 1.0 + 0.3 * ((ii % 97) as f64) / 97.0)
        .collect();
    let mut fden = vec![0.0f64; n];
    let mut fm0 = vec![0.0f64; n];
    let mut fm1 = vec![0.0f64; n];
    let mut fm2 = vec![0.0f64; n];
    let mut fnrg = vec![0.0f64; n];

    let lo = [0i32, 0, 0];
    let extent = [px as u32, py as u32, pz as u32];
    let grid = [gx as u32, gy as u32, gz as u32];
    let dom_lo = [2i32, 2, 2];
    let zones = (gx as f64) * (gy as f64) * (gz as f64);

    for (label, theta) in [
        ("minmod (theta=1.5)", 1.5f64),
        ("vanleer (theta=-1)", -1.0f64),
    ] {
        let mut run = || {
            NamedKernel::new(
                std::env::args()
                    .nth(2)
                    .map(|s| s.leak() as &str)
                    .unwrap_or("adiabatic_face_flux_3d_0_serial"),
            )
            .input_at("prim.rho", &prim, &lo, &extent)
            .input_at("prim.vel[0]", &prim, &lo, &extent)
            .input_at("prim.vel[1]", &prim, &lo, &extent)
            .input_at("prim.vel[2]", &prim, &lo, &extent)
            .input_at("prim.pre", &prim, &lo, &extent)
            .output_at("flux.den", &mut fden, &lo, &extent)
            .output_at("flux.mom_0", &mut fm0, &lo, &extent)
            .output_at("flux.mom_1", &mut fm1, &lo, &extent)
            .output_at("flux.mom_2", &mut fm2, &lo, &extent)
            .output_at("flux.nrg", &mut fnrg, &lo, &extent)
            .grid(&grid)
            .dom_lo(&dom_lo)
            .scalar("gamma", black_box(1.4))
            .scalar("theta", black_box(theta))
            .scalar("x_lo_0", black_box(0.0))
            .scalar("dx_0", black_box(0.01))
            .run();
        };
        run();
        let reps: u32 = std::env::args()
            .nth(5)
            .and_then(|a| a.parse().ok())
            .unwrap_or(10);
        let t0 = Instant::now();
        for _ in 0..reps {
            run();
        }
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "{label:20} {:6.2} ns/zone   sink {:.6e}",
            dt * 1e9 / (reps as f64 * zones),
            fden[(2 + 2 * px + 2 * px * py) as usize],
        );
    }
}
