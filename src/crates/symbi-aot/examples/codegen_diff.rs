// =============================================================================
// codegen_diff.rs
//
// apples-to-apples timing of two compilation paths for the SAME algorithm:
//
//   path A (native): `rmhd_magnetosonic_cfl_speeds(eos, prim, nhat)` — the SAME
//     function `rmhd_wave_speed_map_gv` traces — called directly in a tight
//     per-cell loop, compiled by rustc with full LLVM optimisation over the
//     source as written. (NOT the exact quartic `rmhd_wave_speeds`: the CFL map
//     uses the cheap magnetosonic UPPER BOUND, so path A must too — else the two
//     sides compute different physics and the comparison is meaningless.)
//
//   path B (IR-emitted): the same physics traced through `Gv`, scalarized,
//     CSE'd, emitted as Rust source via `emit_kernel_cpu`, then compiled by
//     rustc into the symbi-aot crate. invoked via the public
//     `rmhd_wave_speed_map_3d` registry entry.
//
// both paths run the SAME algorithm, so they must agree to ULP; the only
// difference is the codegen path. wall-time ratio = pure codegen
// overhead. if A ~= B, the IR pipeline preserves rustc's optimisation
// opportunities and any speed-up has to come from the algorithm. if B >> A,
// the IR is leaking ops (e.g., unfolded `x * 0.0`, dropped strength
// reductions) and the codegen pipeline is the lever.
//
// usage:
//   cargo run --release -p symbi-aot --example codegen_diff -- [--n N] [--repeats R]
// =============================================================================

use std::time::Instant;

use symbi_algebra::Tensor;
use symbi_aot::{Buf, BufHandle, KernelInvocation, rmhd_wave_speed_map_3d};
use symbi_hydro::{IdealGas, MhdPrim, Prim, rmhd::rmhd_magnetosonic_cfl_speeds};

fn parse_arg<T: std::str::FromStr>(flag: &str, default: T) -> T {
    let args: Vec<String> = std::env::args().collect();
    for w in args.windows(2) {
        if w[0] == flag {
            if let Ok(v) = w[1].parse::<T>() {
                return v;
            }
        }
    }
    default
}

// generate a million-cell field of physically-realistic RMHD prims. uses a
// deterministic LCG so both paths see EXACTLY the same input — any disagreement
// is then pure floating-point pattern from the shared input.
fn make_fields(
    n: usize,
) -> (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
) {
    // small subluminal velocities, modest B field, modest pressure.
    let mut rho = vec![0.0; n];
    let mut v0 = vec![0.0; n];
    let mut v1 = vec![0.0; n];
    let mut v2 = vec![0.0; n];
    let mut pre = vec![0.0; n];
    let mut b0 = vec![0.0; n];
    let mut b1 = vec![0.0; n];
    let mut b2 = vec![0.0; n];
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut nxt = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as f64) / (u32::MAX as f64)
    };
    for ii in 0..n {
        rho[ii] = 1.0 + 0.5 * nxt();
        v0[ii] = 0.4 * (nxt() - 0.5);
        v1[ii] = 0.4 * (nxt() - 0.5);
        v2[ii] = 0.4 * (nxt() - 0.5);
        pre[ii] = 1.0 + 0.5 * nxt();
        b0[ii] = 0.6 * (nxt() - 0.5);
        b1[ii] = 0.6 * (nxt() - 0.5);
        b2[ii] = 0.6 * (nxt() - 0.5);
    }
    (rho, v0, v1, v2, pre, b0, b1, b2)
}

fn main() {
    let n_side: usize = parse_arg("--n", 100);
    let repeats: usize = parse_arg("--repeats", 5);
    let n = n_side * n_side * n_side;
    let gamma = 5.0 / 3.0;
    let eos = IdealGas { gamma };
    let inv_dx = [128.0_f64, 128.0, 128.0];

    eprintln!(
        "[codegen_diff] cells = {}, repeats = {}, gamma = {}",
        n, repeats, gamma
    );

    let (rho, v0, v1, v2, pre, b0, b1, b2) = make_fields(n);
    let mut lambda_a = vec![0.0_f64; n];
    let mut lambda_b = vec![0.0_f64; n];

    // ---------- path A: direct native call ----------
    // mirrors rmhd_wave_speed_map_gv's loop body exactly: per-axis magnetosonic
    // cfl speeds, max |s_l|/|s_r| * inv_dx_d, max over axes. nothing else.
    //
    // PARALLELISM: path B uses `into_par_iter` (emitted by the renderer); to
    // compare codegen and not threading models, path A must use rayon over the
    // SAME outer iteration. with_min_len(16) matches the emitted kernel.
    use rayon::prelude::*;
    let mut best_a = f64::INFINITY;
    for _ in 0..repeats {
        let t0 = Instant::now();
        lambda_a
            .par_iter_mut()
            .enumerate()
            .with_min_len(16)
            .for_each(|(ii, lam)| {
                let prim: MhdPrim<f64, 3> = MhdPrim {
                    hydro: Prim {
                        rho: rho[ii],
                        vel: Tensor::new([v0[ii], v1[ii], v2[ii]]),
                        pre: pre[ii],
                    },
                    mag: Tensor::new([b0[ii], b1[ii], b2[ii]]),
                };
                let mut lambda = 0.0_f64;
                for dd in 0..3 {
                    let nhat = Tensor::<f64, 3>::unit(dd);
                    let (sl, sr) = rmhd_magnetosonic_cfl_speeds(&eos, &prim, &nhat);
                    let s = sl.abs().max(sr.abs()) * inv_dx[dd];
                    if s > lambda {
                        lambda = s;
                    }
                }
                *lam = lambda;
            });
        let dt = t0.elapsed().as_secs_f64();
        if dt < best_a {
            best_a = dt;
        }
    }

    // ---------- path B: IR-emitted via AOT registry ----------
    // the same loop, but the inner per-cell work runs through the
    // build.rs-generated `rmhd_wave_speed_map_3d` function (emitted Rust).
    // grid is a flat n x 1 x 1 strip — equivalent work, same cell count.
    let mut best_b = f64::INFINITY;
    let alo: [i32; 3] = [0, 0, 0];
    let aext: [u32; 3] = [n as u32, 1, 1];
    let grid: [u32; 3] = [n as u32, 1, 1];
    let dom_lo: [i32; 3] = [0, 0, 0];
    let scalars: [f64; 4] = [gamma, inv_dx[0], inv_dx[1], inv_dx[2]];
    for _ in 0..repeats {
        let t0 = Instant::now();
        {
            let inv = KernelInvocation {
                buffers: vec![
                    Buf {
                        handle: BufHandle::Host(&rho),
                        lo: &alo,
                        extent: &aext,
                    },
                    Buf {
                        handle: BufHandle::Host(&v0),
                        lo: &alo,
                        extent: &aext,
                    },
                    Buf {
                        handle: BufHandle::Host(&v1),
                        lo: &alo,
                        extent: &aext,
                    },
                    Buf {
                        handle: BufHandle::Host(&v2),
                        lo: &alo,
                        extent: &aext,
                    },
                    Buf {
                        handle: BufHandle::Host(&pre),
                        lo: &alo,
                        extent: &aext,
                    },
                    Buf {
                        handle: BufHandle::Host(&b0),
                        lo: &alo,
                        extent: &aext,
                    },
                    Buf {
                        handle: BufHandle::Host(&b1),
                        lo: &alo,
                        extent: &aext,
                    },
                    Buf {
                        handle: BufHandle::Host(&b2),
                        lo: &alo,
                        extent: &aext,
                    },
                    Buf {
                        handle: BufHandle::HostMut(&mut lambda_b),
                        lo: &alo,
                        extent: &aext,
                    },
                ],
                grid: &grid,
                dom_lo: &dom_lo,
                ints: &[],
                scalars: &scalars,
            };
            inv.run_cpu(rmhd_wave_speed_map_3d::<f64>);
        }
        let dt = t0.elapsed().as_secs_f64();
        if dt < best_b {
            best_b = dt;
        }
    }

    // ---------- numerical sanity ----------
    // path A and path B must agree on every cell (carrier-equivalence). a
    // small ULP gap is OK (rustc may re-associate); a wide gap means the two
    // paths aren't comparing the same algorithm.
    let mut max_rel = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut n_disagree = 0;
    for ii in 0..n {
        let a = lambda_a[ii];
        let b = lambda_b[ii];
        let abs_diff = (a - b).abs();
        let scale = a.abs().max(b.abs()).max(1e-30);
        let rel = abs_diff / scale;
        if abs_diff > max_abs {
            max_abs = abs_diff;
        }
        if rel > max_rel {
            max_rel = rel;
        }
        if rel > 1e-10 {
            n_disagree += 1;
        }
    }

    let ns_per_cell_a = best_a * 1e9 / n as f64;
    let ns_per_cell_b = best_b * 1e9 / n as f64;
    let ratio = best_b / best_a;

    println!("\n=== codegen_diff: rmhd_wave_speeds — native vs IR-emitted ===\n");
    println!("  cells:     {}", n);
    println!("  repeats:   {} (best-of)\n", repeats);
    println!("  path A (native, rustc on hand source):");
    println!("    wall:        {:.3} ms", best_a * 1e3);
    println!("    ns/cell:     {:.1}", ns_per_cell_a);
    println!("  path B (IR-emitted, rustc on Gv-traced source):");
    println!("    wall:        {:.3} ms", best_b * 1e3);
    println!("    ns/cell:     {:.1}", ns_per_cell_b);
    println!("\n  B/A ratio:   {:.2}x", ratio);
    println!(
        "  numerical agreement: max |a-b| = {:.3e}, max rel = {:.3e}, disagreeing cells (rel > 1e-10) = {}/{}\n",
        max_abs, max_rel, n_disagree, n
    );
}
