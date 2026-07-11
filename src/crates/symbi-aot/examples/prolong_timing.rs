// =============================================================================
// prolong_timing.rs
//
// single-core wall-time of the amr prolongation on a coarse-fine ghost slab:
// the fused TIME-PAIR kernel (refine_prolong_{order}_5c_3d — the design-47
// act-8 gather census: 2 snapshots x 5^3 x 5 comps loads per fine cell)
// against the LERP-THEN-PROLONG split (field_lerp_5c_3d over the coarse
// parent region + refine_prolong_1t_{order}_5c_3d — one snapshot). serial
// twins, so the number is per-core compute without rayon scheduling noise.
// prints ns/fine-zone for both paths; the split includes the lerp pass.
//
// usage:
//   cargo run --release -p symbi-aot --example prolong_timing [reps] [order]
// =============================================================================

use std::hint::black_box;
use std::time::Instant;
use symbi_aot::{kernel_by_name, CpuField, CpuFieldMut};

// the fine ghost slab: thickness 4 along x, 192 span transverse (a realistic
// coarse-fine shell face of a refined bondi patch).
const SLAB: [usize; 3] = [4, 192, 192];
const LO: [i32; 3] = [-4, -4, -4];
// the coarse parent region + ppm stencil halo: parents of x in -4..0 are
// -2..0, +-2 -> -4..2; transverse parents -2..94, +-2 -> -4..96.
const CLAB: [usize; 3] = [6, 100, 100];

fn main() {
    let reps: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(30);
    let order: String = std::env::args().nth(2).unwrap_or_else(|| "ppm".to_string());
    let ncoarse = CLAB.iter().product::<usize>();
    let nfine = SLAB.iter().product::<usize>();
    let zones = nfine as f64;

    let coarse: Vec<Vec<f64>> = (0..10)
        .map(|k| (0..ncoarse).map(|ii| 1.0 + 0.3 * (((ii * 7 + k * 13) % 97) as f64) / 97.0).collect())
        .collect();
    let mut fine: Vec<Vec<f64>> = (0..5).map(|_| vec![0.0f64; nfine]).collect();
    let mut lerped: Vec<Vec<f64>> = (0..5).map(|_| vec![0.0f64; ncoarse]).collect();

    let c_ext = [CLAB[0] as u32, CLAB[1] as u32, CLAB[2] as u32];
    let f_ext = [SLAB[0] as u32, SLAB[1] as u32, SLAB[2] as u32];
    let alpha = black_box(0.37f64);

    let (pair, _) = kernel_by_name::<f64>(&format!("refine_prolong_{order}_5c_3d_serial"))
        .expect("time-pair prolong kernel");
    let (lerp, _) = kernel_by_name::<f64>("field_lerp_5c_3d_serial").expect("lerp kernel");
    let (single, _) = kernel_by_name::<f64>(&format!("refine_prolong_1t_{order}_5c_3d_serial"))
        .expect("1t prolong kernel");

    // pair: inputs (src_old_k, src_new_k) interleaved = the same coarse data
    // bound per slot; outputs dst_k over the fine slab.
    let run_pair = |fine: &mut [Vec<f64>]| {
        let inputs: Vec<CpuField> = (0..10).map(|k| CpuField::from_layout(&coarse[k], &LO, &c_ext)).collect();
        let mut outs: Vec<CpuFieldMut> =
            fine.iter_mut().map(|o| CpuFieldMut::from_layout(o, &LO, &f_ext)).collect();
        pair(&inputs, &mut outs, &f_ext, &LO, &[], &[alpha]);
    };
    // split: lerp the coarse region once, then the single-snapshot prolong.
    let run_split = |lerped: &mut [Vec<f64>], fine: &mut [Vec<f64>]| {
        {
            let inputs: Vec<CpuField> = (0..10).map(|k| CpuField::from_layout(&coarse[k], &LO, &c_ext)).collect();
            let mut louts: Vec<CpuFieldMut> =
                lerped.iter_mut().map(|o| CpuFieldMut::from_layout(o, &LO, &c_ext)).collect();
            lerp(&inputs, &mut louts, &c_ext, &LO, &[], &[alpha]);
        }
        let inputs: Vec<CpuField> =
            lerped.iter().map(|l| CpuField::from_layout(l, &LO, &c_ext)).collect();
        let mut outs: Vec<CpuFieldMut> =
            fine.iter_mut().map(|o| CpuFieldMut::from_layout(o, &LO, &f_ext)).collect();
        single(&inputs, &mut outs, &f_ext, &LO, &[], &[]);
    };

    // the axis-split sweep chain (design 49): lerp -> sw0 (A: fine-x,
    // coarse-yz) -> sw1 (B: fine-xy, coarse-z) -> sw2 (dst). w = the order's
    // stencil halfwidth; the intermediate lattices mirror transfer.rs.
    let w: isize = match order.as_str() {
        "pcm" => 0,
        "plm" => 1,
        _ => 2,
    };
    let parents = |lo: isize, hi: isize| (lo.div_euclid(2) - w, (hi - 1).div_euclid(2) + 1 + w);
    let f_hi = [LO[0] as isize + SLAB[0] as isize, LO[1] as isize + SLAB[1] as isize, LO[2] as isize + SLAB[2] as isize];
    let (p1_lo, p1_hi) = parents(LO[1] as isize, f_hi[1]);
    let (p2_lo, p2_hi) = parents(LO[2] as isize, f_hi[2]);
    let a_lo = [LO[0], p1_lo as i32, p2_lo as i32];
    let a_ext = [SLAB[0] as u32, (p1_hi - p1_lo) as u32, (p2_hi - p2_lo) as u32];
    let b_lo = [LO[0], LO[1], p2_lo as i32];
    let b_ext = [SLAB[0] as u32, SLAB[1] as u32, (p2_hi - p2_lo) as u32];
    let na = (a_ext[0] * a_ext[1] * a_ext[2]) as usize;
    let nb = (b_ext[0] * b_ext[1] * b_ext[2]) as usize;
    let mut mid_a: Vec<Vec<f64>> = (0..5).map(|_| vec![0.0f64; na]).collect();
    let mut mid_b: Vec<Vec<f64>> = (0..5).map(|_| vec![0.0f64; nb]).collect();
    let (sw0, _) = kernel_by_name::<f64>(&format!("refine_prolong_sw0_{order}_5c_3d_serial")).expect("sw0");
    let (sw1, _) = kernel_by_name::<f64>(&format!("refine_prolong_sw1_{order}_5c_3d_serial")).expect("sw1");
    let (sw2, _) = kernel_by_name::<f64>(&format!("refine_prolong_sw2_{order}_5c_3d_serial")).expect("sw2");
    let run_sweep = |lerped: &mut [Vec<f64>], mid_a: &mut [Vec<f64>], mid_b: &mut [Vec<f64>], fine: &mut [Vec<f64>]| {
        {
            let inputs: Vec<CpuField> = (0..10).map(|k| CpuField::from_layout(&coarse[k], &LO, &c_ext)).collect();
            let mut louts: Vec<CpuFieldMut> =
                lerped.iter_mut().map(|o| CpuFieldMut::from_layout(o, &LO, &c_ext)).collect();
            lerp(&inputs, &mut louts, &c_ext, &LO, &[], &[alpha]);
        }
        {
            let inputs: Vec<CpuField> = lerped.iter().map(|l| CpuField::from_layout(l, &LO, &c_ext)).collect();
            let mut outs: Vec<CpuFieldMut> =
                mid_a.iter_mut().map(|o| CpuFieldMut::from_layout(o, &a_lo, &a_ext)).collect();
            sw0(&inputs, &mut outs, &a_ext, &a_lo, &[], &[]);
        }
        {
            let inputs: Vec<CpuField> = mid_a.iter().map(|l| CpuField::from_layout(l, &a_lo, &a_ext)).collect();
            let mut outs: Vec<CpuFieldMut> =
                mid_b.iter_mut().map(|o| CpuFieldMut::from_layout(o, &b_lo, &b_ext)).collect();
            sw1(&inputs, &mut outs, &b_ext, &b_lo, &[], &[]);
        }
        let inputs: Vec<CpuField> = mid_b.iter().map(|l| CpuField::from_layout(l, &b_lo, &b_ext)).collect();
        let mut outs: Vec<CpuFieldMut> =
            fine.iter_mut().map(|o| CpuFieldMut::from_layout(o, &LO, &f_ext)).collect();
        sw2(&inputs, &mut outs, &f_ext, &LO, &[], &[]);
    };

    // warmup + verify every path agrees bitwise before timing anything.
    run_pair(&mut fine);
    let reference: Vec<Vec<f64>> = fine.clone();
    for f in fine.iter_mut() {
        f.fill(0.0);
    }
    run_split(&mut lerped, &mut fine);
    for (k, (a, b)) in reference.iter().zip(&fine).enumerate() {
        assert!(
            a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "comp {k}: split path diverged from the time-pair kernel — timing would be meaningless",
        );
    }
    for f in fine.iter_mut() {
        f.fill(0.0);
    }
    run_sweep(&mut lerped, &mut mid_a, &mut mid_b, &mut fine);
    for (k, (a, b)) in reference.iter().zip(&fine).enumerate() {
        assert!(
            a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "comp {k}: sweep chain diverged from the time-pair kernel — timing would be meaningless",
        );
    }

    let t0 = Instant::now();
    for _ in 0..reps {
        run_pair(&mut fine);
        black_box(&fine);
    }
    let pair_ns = t0.elapsed().as_secs_f64() * 1e9 / (reps as f64 * zones);

    let t0 = Instant::now();
    for _ in 0..reps {
        run_split(&mut lerped, &mut fine);
        black_box(&fine);
        black_box(&lerped);
    }
    let split_ns = t0.elapsed().as_secs_f64() * 1e9 / (reps as f64 * zones);

    let t0 = Instant::now();
    for _ in 0..reps {
        run_sweep(&mut lerped, &mut mid_a, &mut mid_b, &mut fine);
        black_box(&fine);
        black_box(&mid_a);
        black_box(&mid_b);
    }
    let sweep_ns = t0.elapsed().as_secs_f64() * 1e9 / (reps as f64 * zones);

    println!("{order} on {SLAB:?} slab ({} fine zones, {reps} reps, serial):", nfine);
    println!("  time-pair kernel:       {pair_ns:8.1} ns/zone");
    println!("  lerp + single-snapshot: {split_ns:8.1} ns/zone  ({:.2}x)", pair_ns / split_ns);
    println!("  lerp + axis sweeps:     {sweep_ns:8.1} ns/zone  ({:.2}x)", pair_ns / sweep_ns);
}
