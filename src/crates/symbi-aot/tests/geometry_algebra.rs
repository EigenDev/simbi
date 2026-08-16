// =============================================================================
// geometry_algebra.rs
//
// validates the substrate geometry algebra: the build-time-generated
// `geom_*` probe kernels compute per-cell finite-volume geometric factors
// (inverse volume, dir-0 face areas, dir-0 volume-weighted centroid) from the
// cell index, and must match the analytic formulas — for Cartesian + Spherical,
// uniform + log radial spacing. proves the in-kernel factor-from-index path:
// analytic exact-integral factors + volume-weighted centroids (not the
// coordinate center), with log zones via the axis map.
// =============================================================================

use symbi_aot::NamedKernel;

// all kernels bind by field name (NamedKernel) — order-independent, loud + named
// on manifest drift. buffers are 1D flat; the 2D/3D probes here run on grids whose
// trailing axes are length 1 (nr x 1, nr x 1 x 1), so the default 1D buffer layout
// gives the identical flat index — only `grid`/`dom_lo` carry the kernel's rank.

const PI: f64 = std::f64::consts::PI;

fn close(got: f64, want: f64, what: &str, i: usize) {
    let rel = (got - want).abs() / want.abs().max(1.0);
    assert!(
        rel < 1e-12,
        "{what} cell {i}: got {got} want {want} (rel {rel:e})"
    );
}

// the analytic 1D spherical factors for a radial shell [r_lo, r_hi]: the
// suppressed theta/phi measures are 2 and 2*pi (full sphere) — they cancel in the
// divergence, kept here to match the kernel's absolute volume.
fn sph_1d(r_lo: f64, r_hi: f64) -> (f64, f64, f64, f64) {
    let ir1 = (r_hi.powi(3) - r_lo.powi(3)) / 3.0; // int r^2 dr
    let omega = 2.0 * (2.0 * PI); // theta(=2) * phi(=2pi) = 4pi
    let inv_vol = 1.0 / (ir1 * omega);
    let area_lo = r_lo.powi(2) * omega;
    let area_hi = r_hi.powi(2) * omega;
    // volume-weighted radial centroid: (3/4)(r_hi^4 - r_lo^4)/(r_hi^3 - r_lo^3).
    let centroid = 0.75 * (r_hi.powi(4) - r_lo.powi(4)) / (r_hi.powi(3) - r_lo.powi(3));
    (inv_vol, area_lo, area_hi, centroid)
}

#[test]
fn cartesian_uniform_1d_matches_analytic() {
    let n = 8usize;
    let (start, dx) = (0.5_f64, 0.1_f64);
    let (mut iv, mut al, mut ah, mut ct) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    NamedKernel::new("geom_cart_unif_1d")
        .output("inv_volume", &mut iv)
        .output("area_lo_0", &mut al)
        .output("area_hi_0", &mut ah)
        .output("centroid_0", &mut ct)
        .grid(&[n as u32])
        .dom_lo(&[0])
        .scalar("x_lo_0", start)
        .scalar("dx_0", dx)
        .run();
    for i in 0..n {
        let lo = start + i as f64 * dx;
        let hi = lo + dx;
        close(iv[i], 1.0 / dx, "inv_vol", i); // flat cell: V = dx
        close(al[i], 1.0, "area_lo", i); // 1D perpendicular face = unit
        close(ah[i], 1.0, "area_hi", i);
        close(ct[i], 0.5 * (lo + hi), "centroid", i); // arithmetic mid
    }
}

#[test]
fn spherical_uniform_1d_matches_analytic() {
    let n = 8usize;
    let (start, dx) = (1.0_f64, 0.1_f64); // r in [1.0, 1.8], away from r=0
    let (mut iv, mut al, mut ah, mut ct) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    NamedKernel::new("geom_sph_unif_1d")
        .output("inv_volume", &mut iv)
        .output("area_lo_0", &mut al)
        .output("area_hi_0", &mut ah)
        .output("centroid_0", &mut ct)
        .grid(&[n as u32])
        .dom_lo(&[0])
        .scalar("x_lo_0", start)
        .scalar("dx_0", dx)
        .run();
    for i in 0..n {
        let r_lo = start + i as f64 * dx;
        let r_hi = r_lo + dx;
        let (eiv, eal, eah, ect) = sph_1d(r_lo, r_hi);
        close(iv[i], eiv, "inv_vol", i);
        close(al[i], eal, "area_lo", i);
        close(ah[i], eah, "area_hi", i);
        close(ct[i], ect, "centroid", i);
        // the volume-weighted centroid sits above the arithmetic midpoint (more
        // volume at larger r) — the whole point of distinguishing them.
        assert!(
            ct[i] > 0.5 * (r_lo + r_hi),
            "centroid not volume-weighted at cell {i}"
        );
    }
}

#[test]
fn spherical_log_1d_matches_analytic() {
    let n = 8usize;
    // log radial: r_face(i) = start * 10^(i * log_slope). param = log_slope.
    let (start, slope) = (1.0_f64, 0.05_f64);
    let (mut iv, mut al, mut ah, mut ct) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    NamedKernel::new("geom_sph_log_1d")
        .output("inv_volume", &mut iv)
        .output("area_lo_0", &mut al)
        .output("area_hi_0", &mut ah)
        .output("centroid_0", &mut ct)
        .grid(&[n as u32])
        .dom_lo(&[0])
        .scalar("x_lo_0", start)
        .scalar("dx_0", slope)
        .scalar("map_kind_0", 1.0)
        .run();
    for i in 0..n {
        let r_lo = start * 10f64.powf(i as f64 * slope);
        let r_hi = start * 10f64.powf((i as f64 + 1.0) * slope);
        let (eiv, eal, eah, ect) = sph_1d(r_lo, r_hi);
        close(iv[i], eiv, "inv_vol", i);
        close(al[i], eal, "area_lo", i);
        close(ah[i], eah, "area_hi", i);
        close(ct[i], ect, "centroid", i);
    }
    // log zones grow: widths increase with i (geometric spacing actually applied).
    let w0 = (start * 10f64.powf(slope)) - start;
    let w_last =
        start * 10f64.powf(n as f64 * slope) - start * 10f64.powf((n as f64 - 1.0) * slope);
    assert!(
        w_last > w0 * 1.5,
        "log spacing not applied: w0 {w0} w_last {w_last}"
    );
}

// the analytic area-weighted divergence. the spherical mass-law godunov computes
// `rho_new = rho - dt*div(F)` with `div = (1/V)(F_hi*A_hi - F_lo*A_lo)`. with rho=0,
// dt=1, F=1 everywhere: rho_new = -div = -(A_hi - A_lo)/V = -3(r_hi^2-r_lo^2)/(r_hi^3-r_lo^3)
// — nonzero (a Cartesian flat divergence of a constant flux would be exactly 0; the
// spherical one is the geometric divergence from the radially growing face area).
#[test]
fn spherical_weighted_divergence_matches_analytic() {
    let n = 8usize;
    let (x_lo, dx) = (1.0_f64, 0.1_f64);
    let sz = n + 1; // the divergence at cell n-1 reads the flux at face n.
    let rho = vec![0.0_f64; sz];
    let mass_flux = vec![1.0_f64; sz]; // uniform flux F = 1
    let mut rho_new = vec![0.0_f64; sz];
    NamedKernel::new("godunov_mass_sph_1d")
        .input("cons.den", &rho)
        .input("mass_flux[0]", &mass_flux)
        .output("cons.den_new", &mut rho_new)
        .grid(&[n as u32])
        .dom_lo(&[0])
        .scalar("dt", 1.0)
        .scalar("x_lo_0", x_lo)
        .scalar("dx_0", dx)
        .run();
    let mut max_div = 0.0_f64;
    for i in 0..n {
        let r_lo = x_lo + i as f64 * dx;
        let r_hi = r_lo + dx;
        let div = 3.0 * (r_hi.powi(2) - r_lo.powi(2)) / (r_hi.powi(3) - r_lo.powi(3));
        close(rho_new[i], -div, "rho_new = -div", i);
        assert!(
            div > 0.0,
            "spherical div of uniform flux must be positive (geometric) at {i}"
        );
        max_div = max_div.max(div);
    }
    assert!(
        max_div > 0.1,
        "divergence implausibly small — area weighting not applied"
    );
}

// exact discrete hydrostatic balance. the spherical adiabatic godunov_euler with
// the well-balanced geometric momentum source. set a v=0 uniform-pressure state — the
// momentum face flux is then p, the mass/energy fluxes are 0 — and assert the momentum
// stays exactly 0: the source `(p*A_hi - p*A_lo)/V` bit-cancels the pressure flux
// divergence `(p*A_hi - p*A_lo)/V`. (without the source, momentum would spuriously
// gain -dt*(p*A_hi - p*A_lo)/V each step — a curvilinear code's classic failure mode.)
#[test]
fn spherical_hydrostatic_balance_exact() {
    let n = 8usize;
    let (x_lo, dx) = (1.0_f64, 0.1_f64);
    let sz = n + 1; // divergence at cell n-1 reads the flux at face n.
    let p = 2.0_f64;
    let pre = vec![p; sz]; // prim.pre (cell pressure)
    let mut den = vec![1.0_f64; sz]; // cons.den (in-place)
    let mass_flux = vec![0.0_f64; sz]; // v=0 -> 0
    let mut mom = vec![0.0_f64; sz]; // cons.mom_0 (in-place) — must stay 0
    let mom_flux = vec![p; sz]; // v=0 -> rho*v^2 + p = p
    let mut nrg = vec![5.0_f64; sz]; // cons.nrg (in-place)
    let nrg_flux = vec![0.0_f64; sz]; // v=0 -> 0
    let dt = 0.01_f64;

    // the godunov-stage kernel reads the u_n snapshot per law + the SSP (a0, ac) coefficients.
    // hse is a forward-Euler step (a0=0, ac=1): u_n is the initial state (multiplied by 0, so it
    // drops out of the result). buffer order (per the stage ABI): prim.pre, then per law the
    // (u_n, flux) pair, then the in-place cons outputs. scalar order [dt, a0, ac, x_lo, dx].
    let (u_n_den, u_n_mom, u_n_nrg) = (den.clone(), mom.clone(), nrg.clone());
    NamedKernel::new("adiabatic_godunov_stage_sph_1d")
        .input("prim.pre", &pre)
        .input("u_n.den", &u_n_den)
        .input("mass_flux[0]", &mass_flux)
        .input("u_n.mom_0", &u_n_mom)
        .input("mom_flux_0[0]", &mom_flux)
        .input("u_n.nrg", &u_n_nrg)
        .input("nrg_flux[0]", &nrg_flux)
        .output("cons.den", &mut den)
        .output("cons.mom_0", &mut mom)
        .output("cons.nrg", &mut nrg)
        .grid(&[n as u32])
        .dom_lo(&[0])
        .scalar("dt", dt)
        .scalar("a0", 0.0)
        .scalar("ac", 1.0)
        .scalar("x_lo_0", x_lo)
        .scalar("dx_0", dx)
        .run();

    for i in 0..n {
        // bit-exact: the geometric source cancels the geometric pressure divergence.
        assert_eq!(
            mom[i], 0.0,
            "HSE broken: cons.mom_0[{i}] = {} (should stay exactly 0)",
            mom[i]
        );
        assert_eq!(
            den[i], 1.0,
            "density drifted at {i}: {} (mass flux is 0)",
            den[i]
        );
        assert_eq!(
            nrg[i], 5.0,
            "energy drifted at {i}: {} (nrg flux is 0)",
            nrg[i]
        );
    }
}

// per-cell physical CFL widths. the iso wave-speed map on a log-spaced spherical
// radial grid (v=0): lambda[i] = cs / (h_r * dr_i) = cs / dr_i (h_r=1). the log zones
// grow, so lambda must shrink cell-to-cell — a single uniform inv_dx would give a
// constant value and mis-estimate the CFL. proves the metric-correct per-cell width.
#[test]
fn log_spherical_cfl_uses_per_cell_widths() {
    let n = 8usize;
    let (start, slope) = (1.0_f64, 0.05_f64); // x_lo_0 = start, dx_0 = log_slope
    let (gamma, p, rho) = (1.4_f64, 1.0_f64, 1.0_f64);
    let cs = (gamma * p / rho).sqrt();
    let rho_v = vec![rho; n];
    let vel = vec![0.0_f64; n];
    let pre = vec![p; n];
    let mut lambda = vec![0.0_f64; n];
    NamedKernel::new("iso_wave_speed_map_sph_log_1d")
        .input("prim.rho", &rho_v)
        .input("prim.vel[0]", &vel)
        .input("prim.pre", &pre)
        .output("scratch", &mut lambda)
        .grid(&[n as u32])
        .dom_lo(&[0])
        .scalar("gamma", gamma)
        .scalar("x_lo_0", start)
        .scalar("dx_0", slope)
        .scalar("map_kind_0", 1.0)
        .run();
    let mut prev = f64::INFINITY;
    for i in 0..n {
        let r_lo = start * 10f64.powf(i as f64 * slope);
        let r_hi = start * 10f64.powf((i as f64 + 1.0) * slope);
        let dr_i = r_hi - r_lo;
        close(lambda[i], cs / dr_i, "lambda = cs/dr_i", i); // h_r = 1
        // log zones grow with i -> the per-cell lambda strictly decreases.
        assert!(
            lambda[i] < prev,
            "lambda not per-cell (should shrink as dr grows) at {i}"
        );
        prev = lambda[i];
    }
}

// the centrifugal/coriolis inertial geometric source, regime-agnostic via the
// conserved momentum. a 2D spherical grid with a single theta cell isolates the radial
// centrifugal: with cons.mom = (density factor)*v and v_r, v_theta set, the source per
// cell is s_0 = mom_theta*v_theta/r_c (centrifugal, outward) and s_1 = -mom_r*v_theta/r_c
// (coriolis), r_c the volume-weighted radial centroid. the density factor is rho
// (Newtonian) or rho*h*W^2 (RHD/relativistic) — both exercised: RHD just feeds the
// relativistic momentum density as cons.mom, one source path serves both.
#[test]
fn spherical_inertial_source_matches_analytic() {
    let nr = 8usize;
    let (x_lo_r, dr) = (1.0_f64, 0.1_f64);
    let (vr, vt) = (0.3_f64, 0.5_f64);

    // run the probe with cons.mom = factor*v (per the regime's momentum density).
    let run = |factor: f64| -> (Vec<f64>, Vec<f64>) {
        let mom0 = vec![factor * vr; nr]; // cons.mom_r
        let mom1 = vec![factor * vt; nr]; // cons.mom_theta
        let v0 = vec![vr; nr];
        let v1 = vec![vt; nr];
        let mut s0 = vec![0.0_f64; nr];
        let mut s1 = vec![0.0_f64; nr];
        // 2D grid nr x 1 (single theta cell); buffers are 1D-flat (trailing axis = 1).
        NamedKernel::new("inertial_momentum_sph_2d")
            .input("cons.mom_0", &mom0)
            .input("cons.mom_1", &mom1)
            .input("prim.vel[0]", &v0)
            .input("prim.vel[1]", &v1)
            .output("s_0", &mut s0)
            .output("s_1", &mut s1)
            .grid(&[nr as u32, 1])
            .dom_lo(&[0, 0])
            .scalar("x_lo_0", x_lo_r)
            .scalar("dx_0", dr)
            .scalar("x_lo_1", 0.0)
            .scalar("dx_1", std::f64::consts::PI)
            .run();
        (s0, s1)
    };

    // Newtonian density factor rho, and the RHD relativistic momentum density rho*h*W^2.
    let rho = 1.0_f64;
    let (gamma, p) = (5.0_f64 / 3.0, 0.4_f64);
    let w2 = 1.0 / (1.0 - (vr * vr + vt * vt)); // lorentz factor squared
    let h = 1.0 + gamma / (gamma - 1.0) * p / rho; // relativistic specific enthalpy
    let wgam2 = rho * h * w2; // rho h W^2 — the RHD conserved-momentum density

    for (label, factor) in [("newtonian rho", rho), ("rhd rho h W^2", wgam2)] {
        let (s0, s1) = run(factor);
        for i in 0..nr {
            let r_l = x_lo_r + i as f64 * dr;
            let r_h = r_l + dr;
            // volume-weighted radial centroid.
            let r_c = 0.75 * (r_h.powi(4) - r_l.powi(4)) / (r_h.powi(3) - r_l.powi(3));
            close(
                s0[i],
                factor * vt * vt / r_c,
                &format!("{label}: s_0 centrifugal"),
                i,
            );
            close(
                s1[i],
                -factor * vr * vt / r_c,
                &format!("{label}: s_1 coriolis"),
                i,
            );
            assert!(
                s0[i] > 0.0,
                "{label}: centrifugal must push outward (+) at {i}"
            );
        }
    }
}

// the full relativistic-MHD geometric source via the RMHD adapter onto the
// same regime-generic builder — total-pressure source + gas inertial (wgam2 v^2) +
// magnetic tension (-bmu^2). validates the radial source s_0 (no cot term; the angular
// extent cancels in (A_hi-A_lo)*inv_V): exercises all three RMHD-specific quantities
// (ptot, the gas momentum density wgam2 = rho h W^2, and the magnetic four-vector bmu).
#[test]
fn rmhd_spherical_source_radial_matches_analytic() {
    let nr = 6usize;
    let (r_lo, dr) = (1.0_f64, 0.1_f64);
    let (rho, vr, vt, vp) = (1.5_f64, 0.2_f64, 0.3_f64, 0.1_f64);
    let (p, br, bt, bp) = (0.8_f64, 0.4_f64, 0.5_f64, 0.2_f64);
    let gamma = 5.0_f64 / 3.0;

    let den = vec![rho; nr];
    let (v0, v1, v2) = (vec![vr; nr], vec![vt; nr], vec![vp; nr]);
    let pre = vec![p; nr];
    let (b0, b1, b2) = (vec![br; nr], vec![bt; nr], vec![bp; nr]);
    let (mut s0, mut s1, mut s2) = (vec![0.0_f64; nr], vec![0.0_f64; nr], vec![0.0_f64; nr]);
    // 3D grid nr x 1 x 1; buffers 1D-flat (theta/phi axes length 1). scalar order
    // [x_lo_0, dx_0, x_lo_1, dx_1, x_lo_2, dx_2, gamma] — all routed by name.
    NamedKernel::new("rmhd_geometric_source_sph_3d")
        .input("prim.rho", &den)
        .input("prim.vel[0]", &v0)
        .input("prim.vel[1]", &v1)
        .input("prim.vel[2]", &v2)
        .input("prim.pre", &pre)
        .input("prim.mag[0]", &b0)
        .input("prim.mag[1]", &b1)
        .input("prim.mag[2]", &b2)
        .output("s_0", &mut s0)
        .output("s_1", &mut s1)
        .output("s_2", &mut s2)
        .grid(&[nr as u32, 1, 1])
        .dom_lo(&[0, 0, 0])
        .scalar("x_lo_0", r_lo)
        .scalar("dx_0", dr)
        .scalar("x_lo_1", 0.5)
        .scalar("dx_1", 0.3)
        .scalar("x_lo_2", 0.0)
        .scalar("dx_2", 0.4)
        .scalar("gamma", gamma)
        .run();

    // the RMHD source quantities (rmhd_side closed forms).
    let vsq = vr * vr + vt * vt + vp * vp;
    let w = 1.0 / (1.0 - vsq).sqrt();
    let wsq = w * w;
    let h = 1.0 + gamma / (gamma - 1.0) * p / rho;
    let wgam2 = rho * h * wsq; // gas momentum density rho h W^2
    let bsq = br * br + bt * bt + bp * bp;
    let vdb = vr * br + vt * bt + vp * bp;
    let ptot = p + 0.5 * (bsq / wsq + vdb * vdb); // total pressure
    let bmu_t = bt / w + vt * w * vdb;
    let bmu_p = bp / w + vp * w * vdb;

    for i in 0..nr {
        let r_l = r_lo + i as f64 * dr;
        let r_h = r_l + dr;
        // radial pressure source ptot*(A_hi-A_lo)*inv_V = ptot*3(r_h^2-r_l^2)/(r_h^3-r_l^3).
        let pre_src = ptot * 3.0 * (r_h * r_h - r_l * r_l) / (r_h.powi(3) - r_l.powi(3));
        let r_c = 0.75 * (r_h.powi(4) - r_l.powi(4)) / (r_h.powi(3) - r_l.powi(3));
        // gas inertial wgam2(vt^2+vp^2)/r_c  -  magnetic tension (bmu_t^2+bmu_p^2)/r_c.
        let want = pre_src + (wgam2 * (vt * vt + vp * vp) - (bmu_t * bmu_t + bmu_p * bmu_p)) / r_c;
        close(
            s0[i],
            want,
            "RMHD radial source = pressure + gas inertial + magnetic tension",
            i,
        );
    }
}

// the full relativistic-MHD geometric source (cylindrical) on a (r, phi, z) grid via
// the same coord-generic builder as spherical — only the Christoffel changes (cylindrical has the
// r-phi pair; z carries no inertial/tension and, with uniform ptot, no pressure source). validates
// all three components against the analytic centrifugal + magnetic-tension forms. proves the
// metric-generic source machinery reuses existing physics for cylindrical RMHD — only a bake
// entry is new.
#[test]
fn rmhd_cylindrical_source_matches_analytic() {
    let nr = 6usize;
    let (r_lo, dr) = (1.0_f64, 0.1_f64);
    // v0=vr, v1=vphi, v2=vz; b0=br, b1=bphi, b2=bz.
    let (rho, vr, vphi, vz) = (1.5_f64, 0.2_f64, 0.3_f64, 0.15_f64);
    let (p, br, bphi, bz) = (0.8_f64, 0.4_f64, 0.5_f64, 0.25_f64);
    let gamma = 5.0_f64 / 3.0;

    let den = vec![rho; nr];
    let (v0, v1, v2) = (vec![vr; nr], vec![vphi; nr], vec![vz; nr]);
    let pre = vec![p; nr];
    let (b0, b1, b2) = (vec![br; nr], vec![bphi; nr], vec![bz; nr]);
    let (mut s0, mut s1, mut s2) = (vec![0.0_f64; nr], vec![0.0_f64; nr], vec![0.0_f64; nr]);
    // scalar order [x_lo_0, dx_0, x_lo_1, dx_1, x_lo_2, dx_2, gamma]; phi/z axes length 1
    // (their extents cancel in the source area ratios) — all bound by name.
    NamedKernel::new("rmhd_geometric_source_cyl_3d")
        .input("prim.rho", &den)
        .input("prim.vel[0]", &v0)
        .input("prim.vel[1]", &v1)
        .input("prim.vel[2]", &v2)
        .input("prim.pre", &pre)
        .input("prim.mag[0]", &b0)
        .input("prim.mag[1]", &b1)
        .input("prim.mag[2]", &b2)
        .output("s_0", &mut s0)
        .output("s_1", &mut s1)
        .output("s_2", &mut s2)
        .grid(&[nr as u32, 1, 1])
        .dom_lo(&[0, 0, 0])
        .scalar("x_lo_0", r_lo)
        .scalar("dx_0", dr)
        .scalar("x_lo_1", 0.5)
        .scalar("dx_1", 0.3)
        .scalar("x_lo_2", 0.0)
        .scalar("dx_2", 0.4)
        .scalar("gamma", gamma)
        .run();

    // RMHD source quantities (same closed forms as spherical — they're coordinate-independent).
    let vsq = vr * vr + vphi * vphi + vz * vz;
    let w = 1.0 / (1.0 - vsq).sqrt();
    let wsq = w * w;
    let h = 1.0 + gamma / (gamma - 1.0) * p / rho;
    let wgam2 = rho * h * wsq;
    let bsq = br * br + bphi * bphi + bz * bz;
    let vdb = vr * br + vphi * bphi + vz * bz;
    let ptot = p + 0.5 * (bsq / wsq + vdb * vdb);
    let bmu_r = br / w + vr * w * vdb;
    let bmu_phi = bphi / w + vphi * w * vdb;

    for i in 0..nr {
        let r_l = r_lo + i as f64 * dr;
        let r_h = r_l + dr;
        // cylindrical radial pressure source = ptot*(A_hi - A_lo)*inv_V = ptot*(r_h - r_l)/Ir2,
        // Ir2 = (r_h^2 - r_l^2)/2; volume-weighted r_c = (2/3)(r_h^3 - r_l^3)/(r_h^2 - r_l^2).
        let ir2 = (r_h * r_h - r_l * r_l) / 2.0;
        let pre_src = ptot * (r_h - r_l) / ir2;
        let r_c = (2.0 / 3.0) * (r_h.powi(3) - r_l.powi(3)) / (r_h * r_h - r_l * r_l);
        // S_r = pressure + (gas centrifugal wgam2 v_phi^2 - magnetic tension bmu_phi^2)/r_c.
        let want_r = pre_src + (wgam2 * vphi * vphi - bmu_phi * bmu_phi) / r_c;
        // S_phi = (-gas coriolis wgam2 v_r v_phi + magnetic bmu_r bmu_phi)/r_c (no phi pressure: uniform).
        let want_phi = (-wgam2 * vr * vphi + bmu_r * bmu_phi) / r_c;
        close(
            s0[i],
            want_r,
            "RMHD cyl S_r = pressure + centrifugal - magnetic tension",
            i,
        );
        close(s1[i], want_phi, "RMHD cyl S_phi = -coriolis + magnetic", i);
        // S_z: z carries no inertial/tension; with uniform ptot the z pressure faces cancel.
        assert!(
            s2[i].abs() < 1e-13,
            "RMHD cyl S_z must vanish (no z geometric source), got {}",
            s2[i]
        );
    }
}
