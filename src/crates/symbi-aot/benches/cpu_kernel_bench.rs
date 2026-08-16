// =============================================================================
// cpu_kernel_bench.rs
//
// real wall-clock timing of the AOT-compiled rmhd cpu kernels (c2p + face flux)
// over a grid. the only valid perf metric available on this host (no cuda/ptxas)
// — actual compiled native code exercised end to end.
//
// usage: cargo run -p symbi-aot --release --example cpu_kernel_bench
//
// A/B protocol: to measure a CODEGEN change, edit the emitter, `cargo clean -p
// symbi-aot` (force kernel regen), rerun. physics/kernel definitions stay fixed;
// only the generated cpu code changes. report min-of-reps ns/cell.
// =============================================================================

use std::hint::black_box;
use std::time::Instant;
// the bench resolves the kernels at COMPILE time (the slice-form `pub fn k<S>`),
// not the per-call name-keyed NamedKernel — its IR parse would dominate the hot
// loop and poison the timing. the slice ABI is still drift-stable (the signature
// stays fixed as a builder adds a field; the buffer slice just grows).
use symbi_aot::{
    CpuField, CpuFieldMut, rhd_c2p_1d, rhd_face_flux_1d_0, rmhd_c2p_1d, rmhd_face_flux_1d,
};

const GAMMA: f64 = 5.0 / 3.0;
const THETA: f64 = 1.5;

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// rmhd p2c (3-velocity) — valid conserved state from analytic primitives.
fn p2c(rho: f64, v: [f64; 3], p: f64, b: [f64; 3]) -> (f64, [f64; 3], f64, [f64; 3]) {
    let v2 = dot(&v, &v);
    let wsq = 1.0 / (1.0 - v2);
    let w = wsq.sqrt();
    let h = 1.0 + GAMMA / (GAMMA - 1.0) * p / rho;
    let bsq = dot(&b, &b);
    let vdb = dot(&v, &b);
    let ed = rho * h * wsq;
    let s = [
        (ed + bsq) * v[0] - vdb * b[0],
        (ed + bsq) * v[1] - vdb * b[1],
        (ed + bsq) * v[2] - vdb * b[2],
    ];
    let tau = ed - p - rho * w + 0.5 * (bsq + bsq * v2 - vdb * vdb);
    (rho * w, s, tau, b)
}

// a smooth physical primitive profile at cell ii (keeps |v|<1, rho>0, p>0).
fn prim_at(ii: usize, n: usize) -> (f64, [f64; 3], f64, [f64; 3]) {
    let x = ii as f64 / n as f64;
    let s = (x * std::f64::consts::TAU).sin();
    let c = (x * std::f64::consts::TAU).cos();
    let rho = 1.0 + 0.3 * s;
    let v = [0.2 * c, -0.15 * s, 0.1 * c];
    let p = 1.0 + 0.4 * c;
    let b = [0.5 + 0.2 * s, 0.3 * c, -0.1 * s];
    (rho, v, p, b)
}

fn bench<F: FnMut()>(label: &str, n_cells: usize, reps: usize, mut run: F) {
    // warm up.
    for _ in 0..10 {
        run();
    }
    let mut samples: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        run();
        samples.push(t0.elapsed().as_secs_f64());
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = samples[0];
    let median = samples[reps / 2];
    let p90 = samples[(reps * 9) / 10];
    let nc = n_cells as f64;
    // min = best-case compute throughput; median/p90 expose run-to-run noise.
    println!(
        "{label:<18} min {:>7.2}  median {:>7.2}  p90 {:>7.2} ns/cell   (noise: {:+.0}% p90 vs min)",
        min * 1e9 / nc,
        median * 1e9 / nc,
        p90 * 1e9 / nc,
        (p90 / min - 1.0) * 100.0,
    );
}

fn main() {
    const N: usize = 1 << 14; // 16384 cells
    const REPS: usize = 200;

    // ---- c2p: conserved -> primitive ----
    let (mut cden, mut cm0, mut cm1, mut cm2, mut cnrg, mut cb0, mut cb1, mut cb2) = (
        vec![0.0; N],
        vec![0.0; N],
        vec![0.0; N],
        vec![0.0; N],
        vec![0.0; N],
        vec![0.0; N],
        vec![0.0; N],
        vec![0.0; N],
    );
    for ii in 0..N {
        let (rho, v, p, b) = prim_at(ii, N);
        let (d, s, tau, bb) = p2c(rho, v, p, b);
        cden[ii] = d;
        cm0[ii] = s[0];
        cm1[ii] = s[1];
        cm2[ii] = s[2];
        cnrg[ii] = tau;
        cb0[ii] = bb[0];
        cb1[ii] = bb[1];
        cb2[ii] = bb[2];
    }
    let (mut prho, mut pv0, mut pv1, mut pv2, mut ppre) = (
        vec![0.0; N],
        vec![0.0; N],
        vec![0.0; N],
        vec![0.0; N],
        vec![0.0; N],
    );

    bench("rmhd c2p", N, REPS, || {
        let cd = CpuField::from_layout(&cden, &[0], &[N as u32]);
        let m0 = CpuField::from_layout(&cm0, &[0], &[N as u32]);
        let m1 = CpuField::from_layout(&cm1, &[0], &[N as u32]);
        let m2 = CpuField::from_layout(&cm2, &[0], &[N as u32]);
        let cn = CpuField::from_layout(&cnrg, &[0], &[N as u32]);
        let b0 = CpuField::from_layout(&cb0, &[0], &[N as u32]);
        let b1 = CpuField::from_layout(&cb1, &[0], &[N as u32]);
        let b2 = CpuField::from_layout(&cb2, &[0], &[N as u32]);
        let pr = CpuFieldMut::from_layout(&mut prho, &[0], &[N as u32]);
        let p0 = CpuFieldMut::from_layout(&mut pv0, &[0], &[N as u32]);
        let p1 = CpuFieldMut::from_layout(&mut pv1, &[0], &[N as u32]);
        let p2v = CpuFieldMut::from_layout(&mut pv2, &[0], &[N as u32]);
        let pp = CpuFieldMut::from_layout(&mut ppre, &[0], &[N as u32]);
        rmhd_c2p_1d(
            &[cd, m0, m1, m2, cn, b0, b1, b2],
            &mut [pr, p0, p1, p2v, pp],
            &[N as u32],
            &[0],
            &[],
            &[GAMMA],
        );
        black_box(&prho);
    });

    // ---- face flux: primitive -> 8 flux components (PLM stencil) ----
    let (mut rho, mut vx, mut vy, mut vz, mut pre, mut bx, mut by, mut bz) = (
        vec![0.0; N],
        vec![0.0; N],
        vec![0.0; N],
        vec![0.0; N],
        vec![0.0; N],
        vec![0.0; N],
        vec![0.0; N],
        vec![0.0; N],
    );
    for ii in 0..N {
        let (r, v, p, b) = prim_at(ii, N);
        rho[ii] = r;
        vx[ii] = v[0];
        vy[ii] = v[1];
        vz[ii] = v[2];
        pre[ii] = p;
        bx[ii] = b[0];
        by[ii] = b[1];
        bz[ii] = b[2];
    }
    let (mut fden, mut fsx, mut fsy, mut fsz, mut fnrg, mut fbx, mut fby, mut fbz) = (
        vec![0.0; N],
        vec![0.0; N],
        vec![0.0; N],
        vec![0.0; N],
        vec![0.0; N],
        vec![0.0; N],
        vec![0.0; N],
        vec![0.0; N],
    );
    // the refactored rmhd flux reads the per-cell Davis fan speeds (ws_l/ws_r); the bench
    // binds the relativistic light-speed bound (-1/+1) — the Rusanov member of the HLLE
    // family — so the kernel's flux work is exercised without a separate wave-speed pass.
    let ws_neg = vec![-1.0f64; N];
    let ws_pos = vec![1.0f64; N];
    bench("rmhd face flux", N - 4, REPS, || {
        let rf = CpuField::from_layout(&rho, &[0], &[N as u32]);
        let vxf = CpuField::from_layout(&vx, &[0], &[N as u32]);
        let vyf = CpuField::from_layout(&vy, &[0], &[N as u32]);
        let vzf = CpuField::from_layout(&vz, &[0], &[N as u32]);
        let pf = CpuField::from_layout(&pre, &[0], &[N as u32]);
        let bxf = CpuField::from_layout(&bx, &[0], &[N as u32]);
        let byf = CpuField::from_layout(&by, &[0], &[N as u32]);
        let bzf = CpuField::from_layout(&bz, &[0], &[N as u32]);
        let fdf = CpuFieldMut::from_layout(&mut fden, &[0], &[N as u32]);
        let fsxf = CpuFieldMut::from_layout(&mut fsx, &[0], &[N as u32]);
        let fsyf = CpuFieldMut::from_layout(&mut fsy, &[0], &[N as u32]);
        let fszf = CpuFieldMut::from_layout(&mut fsz, &[0], &[N as u32]);
        let fnf = CpuFieldMut::from_layout(&mut fnrg, &[0], &[N as u32]);
        let fbxf = CpuFieldMut::from_layout(&mut fbx, &[0], &[N as u32]);
        let fbyf = CpuFieldMut::from_layout(&mut fby, &[0], &[N as u32]);
        let fbzf = CpuFieldMut::from_layout(&mut fbz, &[0], &[N as u32]);
        let wslf = CpuField::from_layout(&ws_neg, &[0], &[N as u32]);
        let wsrf = CpuField::from_layout(&ws_pos, &[0], &[N as u32]);
        // PLM stencil reads coord +/- 2, so sweep the interior with a 2-cell
        // ghost margin on each side (dom_lo_0 = 2, grid_size_0 = N - 4).
        rmhd_face_flux_1d(
            // bface_n (normal face field) at manifest position 8; constant Bx in 1D -> bxf.
            &[rf, vxf, vyf, vzf, pf, bxf, byf, bzf, bxf, wslf, wsrf],
            &mut [fdf, fsxf, fsyf, fszf, fnf, fbxf, fbyf, fbzf],
            &[(N - 4) as u32],
            &[2],
            &[],
            &[GAMMA, THETA],
        );
        black_box(&fden);
    });

    // ---- RHD comparison: same relativistic physics, hydrodynamic sector only
    // (magnetic field and quartic solve drop out) ----
    // rhd p2c (1-velocity): D = rho*W, S = rho*h*W^2*v, tau = rho*h*W^2 - p - D.
    let (mut sden, mut smom, mut snrg) = (vec![0.0; N], vec![0.0; N], vec![0.0; N]);
    for ii in 0..N {
        let (rho, v, p, _b) = prim_at(ii, N);
        let vv = v[0];
        let w = 1.0 / (1.0 - vv * vv).sqrt();
        let h = 1.0 + GAMMA / (GAMMA - 1.0) * p / rho;
        let rhw2 = rho * h * w * w;
        sden[ii] = rho * w;
        smom[ii] = rhw2 * vv;
        snrg[ii] = rhw2 - p - rho * w;
    }
    let (mut sprho, mut spvel, mut sppre) = (vec![0.0; N], vec![0.0; N], vec![0.0; N]);
    bench("rhd c2p", N, REPS, || {
        let cd = CpuField::from_layout(&sden, &[0], &[N as u32]);
        let cm = CpuField::from_layout(&smom, &[0], &[N as u32]);
        let cn = CpuField::from_layout(&snrg, &[0], &[N as u32]);
        let pr = CpuFieldMut::from_layout(&mut sprho, &[0], &[N as u32]);
        let pv = CpuFieldMut::from_layout(&mut spvel, &[0], &[N as u32]);
        let pp = CpuFieldMut::from_layout(&mut sppre, &[0], &[N as u32]);
        rhd_c2p_1d(
            &[cd, cm, cn],
            &mut [pr, pv, pp],
            &[N as u32],
            &[0],
            &[],
            &[GAMMA],
        );
        black_box(&sprho);
    });

    // rhd face flux (1-velocity prim -> 3 flux components, PLM stencil).
    let (mut srho, mut svel, mut spre) = (vec![0.0; N], vec![0.0; N], vec![0.0; N]);
    for ii in 0..N {
        let (rho, v, p, _b) = prim_at(ii, N);
        srho[ii] = rho;
        svel[ii] = v[0];
        spre[ii] = p;
    }
    let (mut sfden, mut sfmom, mut sfnrg) = (vec![0.0; N], vec![0.0; N], vec![0.0; N]);
    bench("rhd face flux", N - 4, REPS, || {
        let a = CpuField::from_layout(&srho, &[0], &[N as u32]);
        let b = CpuField::from_layout(&svel, &[0], &[N as u32]);
        let c = CpuField::from_layout(&spre, &[0], &[N as u32]);
        let fd = CpuFieldMut::from_layout(&mut sfden, &[0], &[N as u32]);
        let fm = CpuFieldMut::from_layout(&mut sfmom, &[0], &[N as u32]);
        let fn_ = CpuFieldMut::from_layout(&mut sfnrg, &[0], &[N as u32]);
        rhd_face_flux_1d_0(
            &[a, b, c],
            &mut [fd, fm, fn_],
            &[(N - 4) as u32],
            &[2],
            &[],
            &[GAMMA, THETA],
        );
        black_box(&sfden);
    });

    // ---- RMHD with B = 0: does the lazy `cond_bn` branch elide the quartic? ----
    // c2p with zero magnetic field: regenerate conserved from p2c(b=0).
    let (mut zden, mut zm0, mut zm1, mut zm2, mut znrg) = (
        vec![0.0; N],
        vec![0.0; N],
        vec![0.0; N],
        vec![0.0; N],
        vec![0.0; N],
    );
    let zb = vec![0.0; N]; // shared zero B buffer for all 3 components
    for ii in 0..N {
        let (rho, v, p, _b) = prim_at(ii, N);
        let (d, s, tau, _bb) = p2c(rho, v, p, [0.0, 0.0, 0.0]);
        zden[ii] = d;
        zm0[ii] = s[0];
        zm1[ii] = s[1];
        zm2[ii] = s[2];
        znrg[ii] = tau;
    }
    bench("rmhd c2p (B=0)", N, REPS, || {
        let cd = CpuField::from_layout(&zden, &[0], &[N as u32]);
        let m0 = CpuField::from_layout(&zm0, &[0], &[N as u32]);
        let m1 = CpuField::from_layout(&zm1, &[0], &[N as u32]);
        let m2 = CpuField::from_layout(&zm2, &[0], &[N as u32]);
        let cn = CpuField::from_layout(&znrg, &[0], &[N as u32]);
        let b0 = CpuField::from_layout(&zb, &[0], &[N as u32]);
        let pr = CpuFieldMut::from_layout(&mut prho, &[0], &[N as u32]);
        let p0 = CpuFieldMut::from_layout(&mut pv0, &[0], &[N as u32]);
        let p1 = CpuFieldMut::from_layout(&mut pv1, &[0], &[N as u32]);
        let p2v = CpuFieldMut::from_layout(&mut pv2, &[0], &[N as u32]);
        let pp = CpuFieldMut::from_layout(&mut ppre, &[0], &[N as u32]);
        rmhd_c2p_1d(
            &[cd, m0, m1, m2, cn, b0, b0, b0],
            &mut [pr, p0, p1, p2v, pp],
            &[N as u32],
            &[0],
            &[],
            &[GAMMA],
        );
        black_box(&prho);
    });

    // flux with zero magnetic field: prim B components all zero.
    bench("rmhd face flux (B=0)", N - 4, REPS, || {
        let rf = CpuField::from_layout(&rho, &[0], &[N as u32]);
        let vxf = CpuField::from_layout(&vx, &[0], &[N as u32]);
        let vyf = CpuField::from_layout(&vy, &[0], &[N as u32]);
        let vzf = CpuField::from_layout(&vz, &[0], &[N as u32]);
        let pf = CpuField::from_layout(&pre, &[0], &[N as u32]);
        let zbf = CpuField::from_layout(&zb, &[0], &[N as u32]);
        let fdf = CpuFieldMut::from_layout(&mut fden, &[0], &[N as u32]);
        let fsxf = CpuFieldMut::from_layout(&mut fsx, &[0], &[N as u32]);
        let fsyf = CpuFieldMut::from_layout(&mut fsy, &[0], &[N as u32]);
        let fszf = CpuFieldMut::from_layout(&mut fsz, &[0], &[N as u32]);
        let fnf = CpuFieldMut::from_layout(&mut fnrg, &[0], &[N as u32]);
        let fbxf = CpuFieldMut::from_layout(&mut fbx, &[0], &[N as u32]);
        let fbyf = CpuFieldMut::from_layout(&mut fby, &[0], &[N as u32]);
        let fbzf = CpuFieldMut::from_layout(&mut fbz, &[0], &[N as u32]);
        let wslf = CpuField::from_layout(&ws_neg, &[0], &[N as u32]);
        let wsrf = CpuField::from_layout(&ws_pos, &[0], &[N as u32]);
        rmhd_face_flux_1d(
            // bface_n at position 8; B=0 case -> zbf.
            &[rf, vxf, vyf, vzf, pf, zbf, zbf, zbf, zbf, wslf, wsrf],
            &mut [fdf, fsxf, fsyf, fszf, fnf, fbxf, fbyf, fbzf],
            &[(N - 4) as u32],
            &[2],
            &[],
            &[GAMMA, THETA],
        );
        black_box(&fden);
    });
}
