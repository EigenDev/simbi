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

    // warmup + verify the two paths agree bitwise before timing anything.
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

    println!("{order} on {SLAB:?} slab ({} fine zones, {reps} reps, serial):", nfine);
    println!("  time-pair kernel:       {pair_ns:8.1} ns/zone");
    println!("  lerp + single-snapshot: {split_ns:8.1} ns/zone  ({:.2}x)", pair_ns / split_ns);
}
