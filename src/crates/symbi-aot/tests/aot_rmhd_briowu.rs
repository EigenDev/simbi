// =============================================================================
// aot_rmhd_briowu.rs
//
// the first END-TO-END substrate RMHD evolution: a 1D relativistic Brio-Wu MHD
// shock tube driven entirely by the build-time-compiled substrate kernels
// (rmhd_c2p_1d + rmhd_face_flux_1d), with a host godunov flux-difference and
// outflow ghosts. one full step is:
//   ghost-fill (outflow) -> c2p (cons->prim) -> face HLLE flux (all 8 conserved)
//   -> godunov update cons[i] -= dt/dx*(F[i+1] - F[i])
// B evolves through the induction flux (F(Bx)=0 keeps Bx constant; By/Bz advect).
// dt = cfl*dx is CFL-safe (relativistic |lambda| <= 1).
//
// this is a SHORT SMOKE (a handful of steps): assert the evolution stays physical
// (rho>0, p>0, |v|<1, finite) and that the shock actually develops. the
// quantitative Brio-Wu profile vs the reference is a longer run, left to the
// user.
// =============================================================================

use symbi_aot::NamedKernel;

// thin shims binding the emitted kernels BY FIELD NAME (NamedKernel) — order-
// independent, and a missing/renamed field panics with the manifest's expected
// names rather than drifting silently. all buffers here are 1D (lo = 0).
#[allow(clippy::too_many_arguments, dead_code)]
fn rmhd_c2p_1d(
    den: &[f64], sx: &[f64], sy: &[f64], sz: &[f64], nrg: &[f64],
    bx: &[f64], by: &[f64], bz: &[f64],
    rho: &mut [f64], vx: &mut [f64], vy: &mut [f64], vz: &mut [f64], pre: &mut [f64],
    grid_size_0: i32, dom_lo_0: i32,
    _: i32,_: i32,_: i32,_: i32,_: i32,_: i32,_: i32,_: i32,_: i32,_: i32,_: i32,_: i32,_: i32,
    gamma: f64,
) {
    let grid = [grid_size_0 as u32];
    let dom = [dom_lo_0];
    NamedKernel::new("rmhd_c2p_1d")
        .input("cons.den", den).input("cons.mom_0", sx).input("cons.mom_1", sy).input("cons.mom_2", sz)
        .input("cons.nrg", nrg).input("cons.mag_0", bx).input("cons.mag_1", by).input("cons.mag_2", bz)
        .output("prim.rho", rho).output("prim.vel_0", vx).output("prim.vel_1", vy)
        .output("prim.vel_2", vz).output("prim.pre", pre)
        .grid(&grid).dom_lo(&dom)
        .scalar("gamma", gamma)
        .run();
}

#[allow(clippy::too_many_arguments, dead_code)]
fn rmhd_face_flux_1d(
    rho: &[f64], vx: &[f64], vy: &[f64], vz: &[f64], pre: &[f64],
    bx: &[f64], by: &[f64], bz: &[f64],
    fden: &mut [f64], fsx: &mut [f64], fsy: &mut [f64], fsz: &mut [f64], fnrg: &mut [f64],
    fbx: &mut [f64], fby: &mut [f64], fbz: &mut [f64],
    grid_size_0: i32, dom_lo_0: i32,
    _: i32,_: i32,_: i32,_: i32,_: i32,_: i32,_: i32,_: i32,_: i32,_: i32,_: i32,_: i32,_: i32,_: i32,_: i32,_: i32,
    gamma: f64, theta: f64,
) {
    // the refactored flux reads the per-cell Davis fan speeds (ws_l/ws_r), produced in
    // the live solver by rmhd_wave_speeds_cell (the exact quartic). this test binds the
    // global relativistic LIGHT-SPEED bound ws_l = -1, ws_r = +1 — valid because |lambda|
    // <= c = 1, giving the (more diffusive) Rusanov/LLF member of the HLLE family. still
    // physical, still develops the shock; the exact-quartic profile is the longer run.
    let wsl = vec![-1.0f64; rho.len()];
    let wsr = vec![1.0f64; rho.len()];
    let grid = [grid_size_0 as u32];
    let dom = [dom_lo_0];
    NamedKernel::new("rmhd_face_flux_1d")
        .input("prim.rho", rho).input("prim.vel[0]", vx).input("prim.vel[1]", vy).input("prim.vel[2]", vz)
        .input("prim.pre", pre).input("prim.mag[0]", bx).input("prim.mag[1]", by).input("prim.mag[2]", bz)
        // the flux now reads the NORMAL field from the staggered FACE field (Gardiner-Stone CT
        // coupling); in 1D Brio-Wu Bx is constant, so the cell bx array IS the face field.
        .input("bface_n", bx)
        .input("wave_speed_l[0]", &wsl).input("wave_speed_r[0]", &wsr)
        .output("flux.den", fden).output("flux.mom_0", fsx).output("flux.mom_1", fsy)
        .output("flux.mom_2", fsz).output("flux.nrg", fnrg)
        .output("flux.mag_0", fbx).output("flux.mag_1", fby).output("flux.mag_2", fbz)
        .grid(&grid).dom_lo(&dom)
        .scalar("gamma", gamma).scalar("theta", theta)
        .run();
}

const GAMMA: f64 = 2.0; // relativistic Brio-Wu (Balsara test 1)
const THETA: f64 = 1.5; // theta-MC limiter compression (default plm_theta)
const N: usize = 200; // interior cells
const NG: usize = 2; // ghost cells each side (PLM reads coord-2)
const TOT: usize = N + 2 * NG;

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// RMHD p2c (3-velocity): primitives -> conserved (D, S, tau, B).
fn p2c(rho: f64, v: [f64; 3], p: f64, b: [f64; 3]) -> (f64, [f64; 3], f64, [f64; 3]) {
    let v2 = dot(&v, &v);
    let wsq = 1.0 / (1.0 - v2);
    let w = wsq.sqrt();
    let h = 1.0 + GAMMA / (GAMMA - 1.0) * p / rho;
    let bsq = dot(&b, &b);
    let vdb = dot(&v, &b);
    let ed = rho * h * wsq;
    let s = [(ed + bsq) * v[0] - vdb * b[0], (ed + bsq) * v[1] - vdb * b[1], (ed + bsq) * v[2] - vdb * b[2]];
    let tau = ed - p - rho * w + 0.5 * (bsq + bsq * v2 - vdb * vdb);
    (rho * w, s, tau, b)
}

// outflow (zero-gradient): copy the nearest interior cell into the ghosts.
fn ghost_outflow(f: &mut [f64]) {
    for g in 0..NG {
        f[g] = f[NG];
        f[TOT - 1 - g] = f[TOT - 1 - NG];
    }
}

// godunov flux difference on the interior: cons[i] -= lam*(flux[i+1] - flux[i]).
fn godunov(cons: &mut [f64], flux: &[f64], lam: f64) {
    for i in NG..NG + N {
        cons[i] -= lam * (flux[i + 1] - flux[i]);
    }
}

#[test]
fn rmhd_briowu_1d_evolves_physically() {
    // conserved (scalar buffers): D, Sx/Sy/Sz, tau, Bx/By/Bz. Bx is constant.
    let mut den = vec![0.0; TOT];
    let mut sx = vec![0.0; TOT];
    let mut sy = vec![0.0; TOT];
    let mut sz = vec![0.0; TOT];
    let mut nrg = vec![0.0; TOT];
    let mut bx = vec![0.0; TOT];
    let mut by = vec![0.0; TOT];
    let mut bz = vec![0.0; TOT];

    // Brio-Wu IC: split at the middle interior cell.
    let left = (1.0_f64, [0.0_f64; 3], 1.0_f64, [0.5_f64, 1.0, 0.0]);
    let right = (0.125_f64, [0.0_f64; 3], 0.1_f64, [0.5_f64, -1.0, 0.0]);
    for i in 0..TOT {
        let (rho0, v0, p0, b0) = if i < TOT / 2 { left } else { right };
        let (d, s, tau, bb) = p2c(rho0, v0, p0, b0);
        den[i] = d;
        sx[i] = s[0];
        sy[i] = s[1];
        sz[i] = s[2];
        nrg[i] = tau;
        bx[i] = bb[0];
        by[i] = bb[1];
        bz[i] = bb[2];
    }

    // primitive scratch.
    let mut rho = vec![0.0; TOT];
    let mut vx = vec![0.0; TOT];
    let mut vy = vec![0.0; TOT];
    let mut vz = vec![0.0; TOT];
    let mut pre = vec![0.0; TOT];
    // flux scratch (per conserved component).
    let mut fden = vec![0.0; TOT];
    let mut fsx = vec![0.0; TOT];
    let mut fsy = vec![0.0; TOT];
    let mut fsz = vec![0.0; TOT];
    let mut fnrg = vec![0.0; TOT];
    let mut fbx = vec![0.0; TOT];
    let mut fby = vec![0.0; TOT];
    let mut fbz = vec![0.0; TOT];

    let cfl = 0.25;
    let lam = cfl; // dt/dx = cfl (dt = cfl*dx, relativistic max |lambda| <= 1)

    let c2p = |den: &[f64], sx: &[f64], sy: &[f64], sz: &[f64], nrg: &[f64], bx: &[f64], by: &[f64], bz: &[f64],
               rho: &mut [f64], vx: &mut [f64], vy: &mut [f64], vz: &mut [f64], pre: &mut [f64]| {
        rmhd_c2p_1d(
            den, sx, sy, sz, nrg, bx, by, bz, rho, vx, vy, vz, pre,
            TOT as i32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, GAMMA,
        );
    };

    let steps = 30;
    for _ in 0..steps {
        for f in [&mut den, &mut sx, &mut sy, &mut sz, &mut nrg, &mut bx, &mut by, &mut bz] {
            ghost_outflow(f);
        }
        c2p(&den, &sx, &sy, &sz, &nrg, &bx, &by, &bz, &mut rho, &mut vx, &mut vy, &mut vz, &mut pre);
        rmhd_face_flux_1d(
            &rho, &vx, &vy, &vz, &pre, &bx, &by, &bz,
            &mut fden, &mut fsx, &mut fsy, &mut fsz, &mut fnrg, &mut fbx, &mut fby, &mut fbz,
            (N + 1) as i32, NG as i32,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, GAMMA, THETA,
        );
        godunov(&mut den, &fden, lam);
        godunov(&mut sx, &fsx, lam);
        godunov(&mut sy, &fsy, lam);
        godunov(&mut sz, &fsz, lam);
        godunov(&mut nrg, &fnrg, lam);
        godunov(&mut bx, &fbx, lam);
        godunov(&mut by, &fby, lam);
        godunov(&mut bz, &fbz, lam);
    }

    for f in [&mut den, &mut sx, &mut sy, &mut sz, &mut nrg, &mut bx, &mut by, &mut bz] {
        ghost_outflow(f);
    }
    c2p(&den, &sx, &sy, &sz, &nrg, &bx, &by, &bz, &mut rho, &mut vx, &mut vy, &mut vz, &mut pre);

    for i in NG..NG + N {
        assert!(rho[i].is_finite() && rho[i] > 0.0, "cell {i}: rho = {}", rho[i]);
        assert!(pre[i].is_finite() && pre[i] > 0.0, "cell {i}: p = {}", pre[i]);
        let vsq = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i];
        assert!(vsq < 1.0, "cell {i}: |v|^2 = {vsq} (superluminal)");
        // Bx stays exactly constant (induction normal flux is 0).
        assert!((bx[i] - 0.5).abs() < 1e-12, "cell {i}: Bx drifted to {}", bx[i]);
    }

    // the shock develops: x-velocity becomes nonzero (it started at 0), and the
    // interface density is no longer a pure step (intermediate states formed).
    let max_vx = (NG..NG + N).map(|i| vx[i].abs()).fold(0.0_f64, f64::max);
    assert!(max_vx > 1e-3, "no flow developed (max |vx| = {max_vx}); shock did not form");
    let rho_mid = rho[TOT / 2];
    assert!(rho_mid > 0.125 + 1e-3 && rho_mid < 1.0 - 1e-3,
        "interface density {rho_mid} still a pure step — no intermediate states formed");
}
