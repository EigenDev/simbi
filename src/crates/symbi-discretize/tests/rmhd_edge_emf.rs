// =============================================================================
// rmhd_edge_emf.rs
//
// validates the substrate rmhd_edge_emf (the CT edge-EMF gather + soft-sign
// gardiner-stone blend) against an INDEPENDENT straight-Rust transcription of the
// production ct_edge_emf (kernels_shared.rs) + soft_upwind. this is the gather
// geometry — the error-prone 12-input staggered stencil — checked against the
// working production reference, on a NON-uniform field (so the offsets + the
// soft-upwind sign actually matter), for every edge axis dir=0/1/2.
//
// p1=(dir+1)%3, p2=(dir+2)%3. corners (cell E_dir = v_p2*b_p1 - v_p1*b_p2) at
// coord / -e_p1 / -e_p2 / -e_p1-e_p2; faces en=-bflux_a, es=-bflux_a[-e_p2],
// ee=+bflux_b, ew=+bflux_b[-e_p1]; density fn/fs=fden_p1, fe/fw=fden_p2.
// =============================================================================

mod harness;
use harness::KernelRun;

use symbi_discretize::rmhd_edge_emf_gv;

const M: usize = 6;

fn idx3(i: usize, j: usize, k: usize) -> usize {
    i + (j + k * M) * M // axis-0-fastest, matching the harness/interp/Field convention
}
fn at(b: &[f64], c: [usize; 3]) -> f64 {
    b[idx3(c[0], c[1], c[2])]
}
fn dec(c: [usize; 3], ax: usize) -> [usize; 3] {
    let mut c = c;
    c[ax] -= 1;
    c
}

// soft-sign upwind (the production soft_upwind).
// the soft-sign switch. `eps` is RELATIVE to the local density-flux magnitude: an absolute
// floor lets a near-static edge (every |f| far below it) drive `s` from roundoff, so two
// mirror-symmetric edges pick opposite upwind sides. a fully zero flux stencil leaves the
// denominator at 0 and the blend degenerates to the plain average.
fn soft(f: f64, a: f64, b: f64, flux_scale: f64) -> f64 {
    let denominator = f.abs() + 32.0 * f64::EPSILON * flux_scale;
    let s = if denominator > 0.0 {
        f / denominator
    } else {
        0.0
    };
    0.5 * ((a + b) + s * (a - b))
}

// independent transcription of ct_edge_emf + rmhd_ct_contact_edge.
#[allow(clippy::too_many_arguments)]
fn ref_edge_emf(
    dir: usize,
    c: [usize; 3],
    vp1: &[f64],
    vp2: &[f64],
    bp1: &[f64],
    bp2: &[f64],
    fa: &[f64],
    fb: &[f64],
    gp1: &[f64],
    gp2: &[f64],
) -> f64 {
    let p1 = (dir + 1) % 3;
    let p2 = (dir + 2) % 3;
    let cell = |c: [usize; 3]| at(vp2, c) * at(bp1, c) - at(vp1, c) * at(bp2, c);
    let ene = cell(c);
    let enw = cell(dec(c, p1));
    let ese = cell(dec(c, p2));
    let esw = cell(dec(dec(c, p1), p2));
    let en = -at(fa, c);
    let es = -at(fa, dec(c, p2));
    let ee = at(fb, c);
    let ew = at(fb, dec(c, p1));
    let fn_ = at(gp1, c);
    let fs = at(gp1, dec(c, p2));
    let fe = at(gp2, c);
    let fw = at(gp2, dec(c, p1));
    let eavg = 0.25 * (es + en + ew + ee);
    let flux_scale = fn_.abs().max(fs.abs()).max(fe.abs()).max(fw.abs());
    let de_jl = soft(fw, 2.0 * (es - esw), 2.0 * (en - enw), flux_scale);
    let de_jr = soft(fe, 2.0 * (ese - es), 2.0 * (ene - en), flux_scale);
    let de_kl = soft(fs, 2.0 * (ew - esw), 2.0 * (ee - ese), flux_scale);
    let de_kr = soft(fn_, 2.0 * (enw - ew), 2.0 * (ene - ee), flux_scale);
    eavg + 0.125 * (de_jl - de_jr + de_kl - de_kr)
}

// bind one staggered input field from its pre-built buffer (per-cell pullback into the
// harness; the generator math lives once in `mk`, reused for the reference diff).
fn from_buf(buf: &[f64]) -> impl Fn(&[usize]) -> f64 + 'static {
    let owned = buf.to_vec();
    move |c| owned[idx3(c[0], c[1], c[2])]
}

// run the edge-EMF kernel over the interior [1, M)^3 (the -1 offsets stay in bounds).
fn run(dir: usize, bufs: &[(&str, &[f64])]) -> Vec<f64> {
    let mut k = KernelRun::new(rmhd_edge_emf_gv(3, (dir + 1) % 3, (dir + 2) % 3))
        .grid([M, M, M])
        .compute_window([1, 1, 1], [M - 1, M - 1, M - 1]);
    for &(key, buf) in bufs {
        k = k.field_with(key, from_buf(buf));
    }
    k.run().values("emf").to_vec()
}

#[test]
fn rmhd_edge_emf_matches_production_gather() {
    let n = M * M * M;
    // non-uniform smooth fields; fden swings sign so the soft upwind is exercised.
    let mk = |g: &dyn Fn(usize, usize, usize) -> f64| -> Vec<f64> {
        let mut v = vec![0.0; n];
        for i in 0..M {
            for j in 0..M {
                for k in 0..M {
                    v[idx3(i, j, k)] = g(i, j, k);
                }
            }
        }
        v
    };
    let vp1 = mk(&|i, j, k| 0.1 + 0.2 * (0.3 * i as f64 + 0.1 * j as f64).sin() - 0.05 * k as f64);
    let vp2 = mk(&|i, j, k| 0.2 * (0.2 * j as f64).cos() + 0.1 * (i as f64 - k as f64) * 0.1);
    let bp1 = mk(&|i, _j, k| 0.5 + 0.1 * (0.25 * k as f64 + 0.2 * i as f64).sin());
    let bp2 = mk(&|_i, j, k| -0.3 + 0.15 * (0.3 * j as f64 - 0.1 * k as f64).cos());
    let fa = mk(&|i, j, k| 0.4 * (0.2 * i as f64).sin() * (0.3 * j as f64 + 0.1 * k as f64).cos());
    let fb = mk(&|_i, j, k| 0.3 * (0.15 * k as f64 - 0.2 * j as f64).sin() + 0.1);
    let gp1 = mk(&|i, j, k| (0.4 * i as f64 - 0.3 * j as f64).sin() + 0.2 * (k as f64 - 2.5)); // swings +/-
    let gp2 = mk(&|i, j, k| (0.3 * j as f64 - 0.4 * k as f64).cos() * (i as f64 - 2.0) * 0.5); // swings +/-

    for dir in 0..3 {
        let bufs: Vec<(&str, &[f64])> = vec![
            ("edge_vp1", &vp1),
            ("edge_vp2", &vp2),
            ("edge_bp1", &bp1),
            ("edge_bp2", &bp2),
            ("edge_bflux_a", &fa),
            ("edge_bflux_b", &fb),
            ("edge_fden_p1", &gp1),
            ("edge_fden_p2", &gp2),
        ];
        let got = run(dir, &bufs);
        // compare on [1, M)^3 (the -1 offsets stay in bounds).
        for i in 1..M {
            for j in 1..M {
                for k in 1..M {
                    let want =
                        ref_edge_emf(dir, [i, j, k], &vp1, &vp2, &bp1, &bp2, &fa, &fb, &gp1, &gp2);
                    let g = got[idx3(i, j, k)];
                    assert!(
                        (g - want).abs() < 1e-13,
                        "dir {dir} cell {i},{j},{k}: {g} != {want}"
                    );
                }
            }
        }
    }
}
