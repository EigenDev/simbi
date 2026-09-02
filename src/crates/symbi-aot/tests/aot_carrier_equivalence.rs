// =============================================================================
// aot_carrier_equivalence.rs
//
// the carrier-equivalence regression. a carrier-generic physics fn `f<S: Scalar>`
// must compute the same thing at `S = f64` (the host, run here) and at `S = Gv`
// (traced -> compiled into the build-time kernel) — for any input, at the same
// baked iteration count. a compiled iterate that stops at a different count than the
// host is invisible to the round-trip suites: they only sample states that converge
// well within the baked count, and the divergence appears only on slow /
// non-convergent inputs.
//
// so these tests deliberately include hard inputs (ultra-relativistic, strong
// contrast, near-vacuum) and compare the compiled kernel against the real host
// source (`rhd_recover` / `rmhd_recover` at `S = f64`) — not a re-implemented
// reference Newton — at the kernel's baked count: same source, two carriers,
// identical result.
// =============================================================================

use symbi_algebra::FaceNormal;
use symbi_aot::NamedKernel;
use symbi_hydro::quantity::{Density, EnergyDensity, Pressure};

// shims binding the emitted kernels by field name (NamedKernel) — order-
// independent, loud + named on manifest drift. every buffer is 1D (lo = 0); the
// silenced ignores absorb the scattered `buf_lo_*` zeros each call site passes.
#[allow(clippy::too_many_arguments)]
fn rhd_c2p_1d(
    cd: &[f64],
    cm: &[f64],
    cn: &[f64],
    pr: &mut [f64],
    pv: &mut [f64],
    pp: &mut [f64],
    g: i32,
    dl: i32,
    _: i32,
    _: i32,
    _: i32,
    _: i32,
    _: i32,
    _: i32,
    gamma: f64,
) {
    NamedKernel::new("rhd_c2p_1d")
        .input("cons.den", cd)
        .input("cons.mom_0", cm)
        .input("cons.nrg", cn)
        .output("prim.rho", pr)
        .output("prim.vel_0", pv)
        .output("prim.pre", pp)
        .grid(&[g as u32])
        .dom_lo(&[dl])
        .scalar("gamma", gamma)
        .run();
}

#[allow(clippy::too_many_arguments)]
fn rmhd_c2p_1d(
    cd: &[f64],
    cm0: &[f64],
    cm1: &[f64],
    cm2: &[f64],
    cn: &[f64],
    cb0: &[f64],
    cb1: &[f64],
    cb2: &[f64],
    pr: &mut [f64],
    pv0: &mut [f64],
    pv1: &mut [f64],
    pv2: &mut [f64],
    pp: &mut [f64],
    g: i32,
    dl: i32,
    _: i32,
    _: i32,
    _: i32,
    _: i32,
    _: i32,
    _: i32,
    _: i32,
    _: i32,
    _: i32,
    _: i32,
    _: i32,
    _: i32,
    _: i32,
    gamma: f64,
) {
    NamedKernel::new("rmhd_c2p_1d")
        .input("cons.den", cd)
        .input("cons.mom_0", cm0)
        .input("cons.mom_1", cm1)
        .input("cons.mom_2", cm2)
        .input("cons.nrg", cn)
        .input("cons.mag_0", cb0)
        .input("cons.mag_1", cb1)
        .input("cons.mag_2", cb2)
        .output("prim.rho", pr)
        .output("prim.vel_0", pv0)
        .output("prim.vel_1", pv1)
        .output("prim.vel_2", pv2)
        .output("prim.pre", pp)
        .grid(&[g as u32])
        .dom_lo(&[dl])
        .scalar("gamma", gamma)
        .run();
}

#[allow(clippy::too_many_arguments)]
fn adiabatic_face_flux_1d(
    den: &[f64],
    v0: &[f64],
    pre: &[f64],
    fden: &mut [f64],
    fmom: &mut [f64],
    fnrg: &mut [f64],
    g: i32,
    dl: i32,
    _: i32,
    _: i32,
    _: i32,
    _: i32,
    _: i32,
    _: i32,
    gamma: f64,
    theta: f64,
) {
    // the flux body traces a moving-mesh face velocity
    // `vface = mesh_adot_0*(x_lo_0 + coord*dx_0) + mesh_vtrans_0` (mesh_face_velocity_gv).
    // the static-mesh / identity-geometry binding (all rates zero, dx_0 = 1) makes vface
    // exactly 0 — bit-identical to the host `hlle(.., vface = 0)` reference.
    NamedKernel::new("adiabatic_face_flux_1d_0")
        .input("prim.rho", den)
        .input("prim.vel[0]", v0)
        .input("prim.pre", pre)
        .output("flux.den", fden)
        .output("flux.mom_0", fmom)
        .output("flux.nrg", fnrg)
        .grid(&[g as u32])
        .dom_lo(&[dl])
        .scalar("gamma", gamma)
        .scalar("theta", theta)
        .scalar("mesh_adot_0", 0.0)
        .scalar("mesh_vtrans_0", 0.0)
        .scalar("x_lo_0", 0.0)
        .scalar("dx_0", 1.0)
        .run();
}

#[allow(clippy::too_many_arguments)]
fn rhd_face_flux_1d(
    den: &[f64],
    v0: &[f64],
    pre: &[f64],
    fden: &mut [f64],
    fmom: &mut [f64],
    fnrg: &mut [f64],
    g: i32,
    dl: i32,
    _: i32,
    _: i32,
    _: i32,
    _: i32,
    _: i32,
    _: i32,
    gamma: f64,
    theta: f64,
) {
    // same moving-mesh face velocity as the adiabatic flux; static-mesh binding -> vface = 0.
    NamedKernel::new("rhd_face_flux_1d_0")
        .input("prim.rho", den)
        .input("prim.vel[0]", v0)
        .input("prim.pre", pre)
        .output("flux.den", fden)
        .output("flux.mom_0", fmom)
        .output("flux.nrg", fnrg)
        .grid(&[g as u32])
        .dom_lo(&[dl])
        .scalar("gamma", gamma)
        .scalar("theta", theta)
        .scalar("mesh_adot_0", 0.0)
        .scalar("mesh_vtrans_0", 0.0)
        .scalar("x_lo_0", 0.0)
        .scalar("dx_0", 1.0)
        .run();
}

#[allow(clippy::too_many_arguments)]
fn iso_wave_speed_map_1d(
    rho: &[f64],
    v0: &[f64],
    pre: &[f64],
    lambda: &mut [f64],
    g: i32,
    dl: i32,
    _: i32,
    _: i32,
    _: i32,
    _: i32,
    gamma: f64,
    inv_dx: f64,
) {
    // the map folds the grid-relative cfl speed `|s - v_g|` with the per-axis grid velocity
    // `v_g = mesh_adot_0*x_centroid + mesh_vtrans_0` (euler_wave_speed_map_gv). the static-mesh /
    // identity-geometry binding (all rates zero, dx_0 = 1) makes v_g exactly 0 — bit-identical
    // to the host `max(|sl|,|sr|)*inv_dx` reference.
    NamedKernel::new("iso_wave_speed_map_1d")
        .input("prim.rho", rho)
        .input("prim.vel[0]", v0)
        .input("prim.pre", pre)
        .output("scratch", lambda)
        .grid(&[g as u32])
        .dom_lo(&[dl])
        .scalar("gamma", gamma)
        .scalar("inv_dx_0", inv_dx)
        .scalar("mesh_adot_0", 0.0)
        .scalar("mesh_vtrans_0", 0.0)
        .scalar("x_lo_0", 0.0)
        .scalar("dx_0", 1.0)
        .run();
}

#[allow(clippy::too_many_arguments)]
fn rhd_wave_speed_map_1d(
    rho: &[f64],
    v0: &[f64],
    pre: &[f64],
    lambda: &mut [f64],
    g: i32,
    dl: i32,
    _: i32,
    _: i32,
    _: i32,
    _: i32,
    gamma: f64,
    inv_dx: f64,
) {
    // same grid-relative cfl map as the iso one; static-mesh binding -> v_g = 0.
    NamedKernel::new("rhd_wave_speed_map_1d")
        .input("prim.rho", rho)
        .input("prim.vel[0]", v0)
        .input("prim.pre", pre)
        .output("scratch", lambda)
        .grid(&[g as u32])
        .dom_lo(&[dl])
        .scalar("gamma", gamma)
        .scalar("inv_dx_0", inv_dx)
        .scalar("mesh_adot_0", 0.0)
        .scalar("mesh_vtrans_0", 0.0)
        .scalar("x_lo_0", 0.0)
        .scalar("dx_0", 1.0)
        .run();
}
use symbi_algebra::Tensor;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdCons;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::regime::Regime;
use symbi_hydro::rhd::{Rhd, rhd_recover};
use symbi_hydro::riemann::hlle;
use symbi_hydro::rmhd::rmhd_recover;
use symbi_hydro::spatial_metric::SpatialMetric;
use symbi_hydro::state::{Cons, Prim};

const GAMMA: f64 = 5.0 / 3.0;
// the iteration counts build.rs bakes into each compiled c2p kernel — the host
// must run the same count for the carriers to be comparable (the production host
// MAX_ITER differs; that is a separate count-adequacy question, while this test checks carrier equivalence).
const RHD_ITERS: usize = 20;
const RMHD_ITERS: usize = 100;

// carrier agreement: bit-equal (covers both-Inf), or both-NaN, or tightly close.
// the tolerance is loose enough to absorb benign codegen reassociation over the
// Newton unroll, but a freeze regression diverges by >> this on the slow inputs.
fn agree(kernel: f64, host: f64) -> bool {
    if kernel == host {
        return true;
    }
    if kernel.is_nan() && host.is_nan() {
        return true;
    }
    (kernel - host).abs() / host.abs().max(1.0) < 1e-10
}

// forward map (1D ideal-gas RHD): primitives -> conserved (D, S, tau).
fn rhd_prim_to_cons(rho: f64, v: f64, p: f64) -> (f64, f64, f64) {
    let w = 1.0 / (1.0 - v * v).sqrt();
    let eps = p / ((GAMMA - 1.0) * rho);
    let h = 1.0 + eps + p / rho;
    let rhw2 = rho * h * w * w;
    let d = rho * w;
    (d, rhw2 * v, rhw2 - p - d)
}

#[test]
fn rhd_c2p_kernel_equals_host_at_baked_count() {
    // (rho, v, p). first row converges within 20 (the round-trip regime); the rest
    // are hard — ultra-relativistic (slow Newton) + strong contrast / near-vacuum.
    // a no-freeze kernel runs all 20 steps past convergence and drifts from the
    // host early-break on exactly these, so they are the freeze regression guard.
    let cases: &[(f64, f64, f64)] = &[
        (1.0, 0.0, 1.0),
        (2.0, 0.5, 1.0),
        (1.0, -0.7, 1.0),
        (1.0, 0.9, 1.0),
        (1.0, 0.95, 5.0),
        (1.0, 0.99, 10.0),
        (10.0, 0.0, 1.0e-3),
        (1.0e-2, 0.0, 1.0e-4),
    ];
    let eos = IdealGas { gamma: GAMMA };

    for &(rho, v, p) in cases {
        let (d, s, tau) = rhd_prim_to_cons(rho, v, p);

        // host: the same rhd_recover source at S = f64, at the kernel's baked count.
        let host = rhd_recover::<f64, 1>(
            &eos,
            &Cons::adiabatic(Density(d), Tensor::new([s]), EnergyDensity(tau)),
            &SpatialMetric::flat(),
            RHD_ITERS,
        );

        // kernel: rhd_recover traced at S = Gv, compiled (bakes RHD_ITERS).
        let (mut kr, mut kv, mut kp) = (vec![0.0_f64], vec![0.0_f64], vec![0.0_f64]);
        rhd_c2p_1d(
            &[d],
            &[s],
            &[tau],
            &mut kr,
            &mut kv,
            &mut kp,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            GAMMA,
        );

        let ctx = format!("(rho={rho}, v={v}, p={p})");
        assert!(
            agree(kr[0], host.rho()),
            "{ctx} rho carrier divergence: kernel {} vs host {}",
            kr[0],
            host.rho()
        );
        assert!(
            agree(kv[0], host.vel()[0]),
            "{ctx} vel carrier divergence: kernel {} vs host {}",
            kv[0],
            host.vel()[0]
        );
        assert!(
            agree(kp[0], host.pre()),
            "{ctx} pre carrier divergence: kernel {} vs host {}",
            kp[0],
            host.pre()
        );
    }
}

// RMHD p2c (3-velocity): primitives -> conserved (D, S, tau, B).
fn rmhd_p2c(rho: f64, v: [f64; 3], p: f64, b: [f64; 3]) -> (f64, [f64; 3], f64, [f64; 3]) {
    let dot = |a: &[f64; 3], c: &[f64; 3]| a[0] * c[0] + a[1] * c[1] + a[2] * c[2];
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
fn rmhd_c2p_kernel_equals_host_at_baked_count() {
    // (rho, v, p, B). nice states + hard ones (relativistic + strongly magnetized).
    let cases: &[(f64, [f64; 3], f64, [f64; 3])] = &[
        (1.0, [0.1, 0.0, 0.0], 1.0, [0.5, 0.0, 0.0]),
        (1.0, [0.5, 0.2, 0.1], 3.0, [1.0, 0.5, 0.2]),
        (1.0, [0.9, 0.0, 0.0], 1.0, [0.0, 2.0, 0.0]),
        (0.5, [0.6, 0.5, 0.0], 0.1, [3.0, 0.0, 1.0]),
        (1.0e-2, [0.0, 0.0, 0.0], 1.0e-3, [0.0, 0.0, 0.5]),
    ];
    let eos = IdealGas { gamma: GAMMA };

    for &(rho, v, p, b) in cases {
        let (d, s, tau, bb) = rmhd_p2c(rho, v, p, b);

        // host: rmhd_recover at S = f64, at the kernel's baked count.
        let cons = MhdCons::<f64, 3>::new(
            Cons::adiabatic(Density(d), Tensor::new(s), EnergyDensity(tau)),
            Tensor::new(bb),
        );
        let host = rmhd_recover::<f64, 3>(&eos, &cons, &SpatialMetric::flat(), RMHD_ITERS);

        // kernel: rmhd_recover traced at S = Gv, compiled (bakes RMHD_ITERS).
        let (mut kr, mut kp) = (vec![0.0_f64], vec![0.0_f64]);
        let (mut kv0, mut kv1, mut kv2) = (vec![0.0_f64], vec![0.0_f64], vec![0.0_f64]);
        rmhd_c2p_1d(
            &[d],
            &[s[0]],
            &[s[1]],
            &[s[2]],
            &[tau],
            &[bb[0]],
            &[bb[1]],
            &[bb[2]],
            &mut kr,
            &mut kv0,
            &mut kv1,
            &mut kv2,
            &mut kp,
            1,
            0, // grid_size_0, dom_lo_0
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0, // 13 buf_lo
            GAMMA,
        );

        let ctx = format!("(rho={rho}, v={v:?}, p={p}, B={b:?})");
        assert!(
            agree(kr[0], host.hydro().rho()),
            "{ctx} rho carrier divergence: kernel {} vs host {}",
            kr[0],
            host.hydro().rho()
        );
        assert!(
            agree(kp[0], host.hydro().pre()),
            "{ctx} pre carrier divergence: kernel {} vs host {}",
            kp[0],
            host.hydro().pre()
        );
        let kv = [kv0[0], kv1[0], kv2[0]];
        for k in 0..3 {
            assert!(
                agree(kv[k], host.hydro().vel()[k]),
                "{ctx} vel[{k}] carrier divergence: kernel {} vs host {}",
                kv[k],
                host.hydro().vel()[k]
            );
        }
    }
}

#[test]
fn flux_kernel_equals_host_hlle_on_uniform_field() {
    // a face flux bundles PLM reconstruction + the HLLE Riemann solver. on a uniform
    // field PLM gives L == R, which isolates HLLE: the compiled flux kernel must equal
    // the host `hlle::<f64>` source on that state. (non-uniform L != R would require
    // replicating the theta-MC reconstruction by hand — the re-implemented-reference
    // anti-pattern; the c2p tests already cover the iterative carrier-divergence class,
    // and HLLE for L == R collapses to F(U) in every wave-speed regime.)
    let eos = IdealGas { gamma: GAMMA };
    let nhat = symbi_algebra::Normalized::axis(0); // +x face, dir = 0
    let theta = 1.0_f64; // plain minmod
    let n = 8usize; // interior 2..6 so the PLM stencil stays in-bounds

    // (rho, v, p): static, forward, reversed, and fast flow.
    let cases: &[(f64, f64, f64)] = &[
        (1.5, 0.4, 0.8),
        (1.0, 0.0, 1.0),
        (2.0, -0.6, 1.5),
        (1.0, 0.9, 0.5),
    ];

    for &(rho, v, p) in cases {
        let prim = Prim::adiabatic(Density(rho), Tensor::new([v]), Pressure(p));
        let (den, v0, pre) = (vec![rho; n], vec![v; n], vec![p; n]);

        // adiabatic (Newtonian Euler)
        let host = hlle::<f64, 1, Newtonian>(&Newtonian, &eos, &prim, &prim, &nhat, 0.0);
        let (mut fden, mut fmom, mut fnrg) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        adiabatic_face_flux_1d(
            &den, &v0, &pre, &mut fden, &mut fmom, &mut fnrg, 4, 2, 0, 0, 0, 0, 0, 0, GAMMA, theta,
        );
        let ctx = format!("adiabatic flux (rho={rho}, v={v}, p={p})");
        for i in 2..6 {
            assert!(
                agree(fden[i], host.den()),
                "{ctx} F_den: kernel {} vs host {}",
                fden[i],
                host.den()
            );
            assert!(
                agree(fmom[i], host.mom()[0]),
                "{ctx} F_mom: kernel {} vs host {}",
                fmom[i],
                host.mom()[0]
            );
            assert!(
                agree(fnrg[i], host.nrg()),
                "{ctx} F_nrg: kernel {} vs host {}",
                fnrg[i],
                host.nrg()
            );
        }

        // rhd
        let host = hlle::<f64, 1, Rhd>(&Rhd, &eos, &prim, &prim, &nhat, 0.0);
        let (mut fden, mut fmom, mut fnrg) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        rhd_face_flux_1d(
            &den, &v0, &pre, &mut fden, &mut fmom, &mut fnrg, 4, 2, 0, 0, 0, 0, 0, 0, GAMMA, theta,
        );
        let ctx = format!("rhd flux (rho={rho}, v={v}, p={p})");
        for i in 2..6 {
            assert!(
                agree(fden[i], host.den()),
                "{ctx} F_den: kernel {} vs host {}",
                fden[i],
                host.den()
            );
            assert!(
                agree(fmom[i], host.mom()[0]),
                "{ctx} F_mom: kernel {} vs host {}",
                fmom[i],
                host.mom()[0]
            );
            assert!(
                agree(fnrg[i], host.nrg()),
                "{ctx} F_nrg: kernel {} vs host {}",
                fnrg[i],
                host.nrg()
            );
        }
    }
}

#[test]
fn wave_speed_map_kernel_equals_host() {
    // the CFL wave-speed map is pointwise. 1D cartesian:
    //   lambda = max(0, max(|sl|,|sr|) * inv_dx_0),  (sl,sr) = regime.wave_speeds_axis(prim, 0).
    // this validates the wave-speed math itself (|v|+cs for Newtonian, Mignone-Bodo for RHD)
    // is carrier-equivalent — the surface c2p doesn't touch and the uniform flux collapses
    // (L==R makes HLLE = F(U) regardless of the wave speeds).
    let eos = IdealGas { gamma: GAMMA };
    let inv_dx = 2.0_f64; // = 1/dx, dx = 0.5 (exercises the inv_dx multiply)
    let n = 4usize;
    // (rho, v, p): static, subsonic, fast, reversed.
    let cases: &[(f64, f64, f64)] = &[
        (1.0, 0.0, 1.0),
        (1.5, 0.4, 0.8),
        (1.0, 0.9, 0.5),
        (2.0, -0.6, 1.5),
    ];

    for &(rho, v, p) in cases {
        // host prim mirrors the kernel's internal Prim<.,3> (v[0] gridded, v[1..2] zero).
        let prim = Prim::<f64, 3>::adiabatic(Density(rho), Tensor::new([v, 0.0, 0.0]), Pressure(p));
        let (rho_b, v0_b, pre_b) = (vec![rho; n], vec![v; n], vec![p; n]);

        // adiabatic / Newtonian map (iso_wave_speed_map = the Newtonian CFL map).
        let (sl, sr) = Newtonian.wave_speeds_axis(&eos, &prim, 0);
        let host = 0.0_f64.max(sl.abs().max(sr.abs()) * inv_dx);
        let mut lambda = vec![0.0; n];
        iso_wave_speed_map_1d(
            &rho_b,
            &v0_b,
            &pre_b,
            &mut lambda,
            n as i32,
            0,
            0,
            0,
            0,
            0,
            GAMMA,
            inv_dx,
        );
        let ctx = format!("Newtonian wave-speed map (rho={rho}, v={v}, p={p})");
        for i in 0..n {
            assert!(
                agree(lambda[i], host),
                "{ctx}: kernel {} vs host {}",
                lambda[i],
                host
            );
        }

        // rhd map (Mignone-Bodo per-axis speed).
        let (sl, sr) = Rhd.wave_speeds_axis(&eos, &prim, 0);
        let host = 0.0_f64.max(sl.abs().max(sr.abs()) * inv_dx);
        let mut lambda = vec![0.0; n];
        rhd_wave_speed_map_1d(
            &rho_b,
            &v0_b,
            &pre_b,
            &mut lambda,
            n as i32,
            0,
            0,
            0,
            0,
            0,
            GAMMA,
            inv_dx,
        );
        let ctx = format!("RHD wave-speed map (rho={rho}, v={v}, p={p})");
        for i in 0..n {
            assert!(
                agree(lambda[i], host),
                "{ctx}: kernel {} vs host {}",
                lambda[i],
                host
            );
        }
    }
}
