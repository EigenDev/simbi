// =============================================================================
// aot_rmhd_c2p.rs
//
// numerical validation of the BUILD-TIME-GENERATED RMHD cons->prim kernel — the
// deepest substrate regime (a KKC vector-state bracketed false-position over
// kkc_fmu44, find_mu_plus + Illinois half-damp, 100-step bound baked at codegen,
// lowered + emitted to compiled Rust via the DAG-preserving lowering). this is the
// proof the COMPILED kernel produces correct numbers at run time
// (the IR graph itself is validated against the reference c2p in
// symbi-discretize/tests/rmhd_c2p.rs).
//
// ROUND-TRIP: pick analytic primitives (rho, 3-velocity v, p, lab B), forward-map
// to the conserved (D, S_k, tau, B_k) via the RMHD relations, run the compiled
// c2p, assert it recovers the originals (3-velocity; use_four_velocity = false).
//
// generated signature (OUT_DIR/rmhd_c2p_generated.rs):
//   rmhd_c2p_1d(cons_den, cons_mom_0..2, cons_nrg, cons_mag_0..2 : &[f64],   // 8 in
//               prim_rho, prim_vel_0..2, prim_pre : &mut [f64],              // 5 out
//               grid_size_0, dom_lo_0, buf_lo_0_0..buf_lo_12_0 : i32, gamma: f64)
// =============================================================================

use symbi_aot::NamedKernel;

// shim binding the emitted c2p BY FIELD NAME (NamedKernel) — order-independent,
// loud + named on manifest drift. all buffers here are 1D (lo = 0).
#[allow(non_snake_case, clippy::too_many_arguments)]
fn rmhd_c2p_1d(
    cons_den: &[f64],
    cm0: &[f64],
    cm1: &[f64],
    cm2: &[f64],
    cnrg: &[f64],
    cb0: &[f64],
    cb1: &[f64],
    cb2: &[f64],
    prim_rho: &mut [f64],
    pv0: &mut [f64],
    pv1: &mut [f64],
    pv2: &mut [f64],
    ppre: &mut [f64],
    grid_size_0: i32,
    dom_lo_0: i32,
    _l0: i32,
    _l1: i32,
    _l2: i32,
    _l3: i32,
    _l4: i32,
    _l5: i32,
    _l6: i32,
    _l7: i32,
    _l8: i32,
    _l9: i32,
    _l10: i32,
    _l11: i32,
    _l12: i32,
    gamma: f64,
) {
    let grid = [grid_size_0 as u32];
    let dom = [dom_lo_0];
    NamedKernel::new("rmhd_c2p_1d")
        .input("cons.den", cons_den)
        .input("cons.mom_0", cm0)
        .input("cons.mom_1", cm1)
        .input("cons.mom_2", cm2)
        .input("cons.nrg", cnrg)
        .input("cons.mag_0", cb0)
        .input("cons.mag_1", cb1)
        .input("cons.mag_2", cb2)
        .output("prim.rho", prim_rho)
        .output("prim.vel_0", pv0)
        .output("prim.vel_1", pv1)
        .output("prim.vel_2", pv2)
        .output("prim.pre", ppre)
        .grid(&grid)
        .dom_lo(&dom)
        .scalar("gamma", gamma)
        .run();
}

const GAMMA: f64 = 5.0 / 3.0;

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// RMHD p2c (3-velocity): primitives -> conserved (D, S, tau, B).
//   W = 1/sqrt(1-v^2);  h = 1 + gamma/(gamma-1)*p/rho;  ed = rho*h*W^2
//   D = rho*W;  S_k = (ed + B^2) v_k - (v.B) B_k
//   tau = ed - p - D + 0.5*(B^2 + B^2 v^2 - (v.B)^2)
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

#[test]
fn aot_rmhd_c2p_round_trips() {
    // physical primitives: rho>0, p>0, |v|<1, parallel + oblique B.
    let prims = [
        (1.0_f64, [0.1, 0.0, 0.0], 1.0_f64, [0.5, 0.0, 0.0]),
        (2.0, [0.3, 0.1, 0.0], 0.5, [0.2, 0.3, 0.0]),
        (0.5, [0.0, 0.0, 0.4], 2.0, [0.1, 0.1, 0.6]),
        (1.0, [0.5, 0.2, 0.1], 3.0, [1.0, 0.5, 0.2]),
        (1.5, [0.6, 0.0, 0.0], 0.8, [0.0, 0.4, 0.0]),
    ];
    let n = prims.len();

    // pack the conserved into the kernel's per-field input buffers.
    let (mut den, mut nrg) = (vec![0.0; n], vec![0.0; n]);
    let mut mom = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    let mut mag = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    for (i, &(rho, v, p, b)) in prims.iter().enumerate() {
        let (d, s, tau, bb) = p2c(rho, v, p, b);
        den[i] = d;
        nrg[i] = tau;
        for k in 0..3 {
            mom[k][i] = s[k];
            mag[k][i] = bb[k];
        }
    }

    // separate output buffers (the kernel takes 5 distinct &mut [f64]).
    let (mut rho_o, mut p_o) = (vec![0.0; n], vec![0.0; n]);
    let (mut v0_o, mut v1_o, mut v2_o) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);

    rmhd_c2p_1d(
        &den, &mom[0], &mom[1], &mom[2], &nrg, &mag[0], &mag[1], &mag[2], &mut rho_o, &mut v0_o,
        &mut v1_o, &mut v2_o, &mut p_o, n as i32, 0, // grid_size_0, dom_lo_0
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 13 buf_lo
        GAMMA,
    );

    let vel_o = [&v0_o, &v1_o, &v2_o];
    for (i, &(rho, v, p, _b)) in prims.iter().enumerate() {
        let close = |a: f64, want: f64, what: &str| {
            assert!(
                (a - want).abs() <= 1e-9 * (1.0 + want.abs()),
                "state {i} {what}: {a} != {want}"
            );
        };
        close(rho_o[i], rho, "rho");
        close(p_o[i], p, "p");
        for k in 0..3 {
            close(vel_o[k][i], v[k], &format!("v[{k}]"));
        }
    }
}
