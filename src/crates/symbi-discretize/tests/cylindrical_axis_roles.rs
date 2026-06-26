// =============================================================================
// cylindrical_axis_roles.rs
//
// validates the AXIS-ROLE physics (docs/design/18) for the cylindrical (r, z)
// axisymmetric swirl plane: grid axes [r, z] = coordinate map [0, 2], with phi
// (coord 1) a SYMMETRY axis but a CARRIED swirl component (ncomp=3 > ndim=2). two
// gv probes pin the (r, z) case the fixed-order convention could not express:
//   - the geometric momentum SOURCE: pressure-curvature + centrifugal/coriolis,
//     with the angular-momentum swirl law S_phi = -mom_r*v_phi/r.
//   - the cyl r-z FLUX: HLLE along the r-sweep — F_mom_phi is advective (no
//     pressure, since phi is not the sweep coordinate).
// the underlying cell_geometry_gv r-weighting (sqrt(g)=r, phi's 2*pi folded into
// volume + face areas) is exercised through both. interpreter run via KernelRun.
// =============================================================================

mod harness;
use harness::KernelRun;

use symbi_discretize::{adiabatic_flux_cyl_rz_gv, geometric_momentum_source_probe_gv, Coords, GeoSource, Spacing};

const NR: usize = 4;
const NZ: usize = 3;
const R0: f64 = 1.0; // shell away from r=0
const DR: f64 = 0.1;
const Z0: f64 = 0.0;
const DZ: f64 = 0.2;

fn close(got: f64, want: f64, what: &str, i: usize, j: usize) {
    let rel = (got - want).abs() / want.abs().max(1.0);
    assert!(rel < 1e-12, "{what} cell ({i},{j}): got {got} want {want} (rel {rel:e})");
}

#[test]
fn cylindrical_rz_swirl_source_matches_analytic() {
    // the axisymmetric swirl source (docs/design/18 stage 2): cylindrical (r,z) grid [0,2]
    // with ncomp=3 (carry v_phi though phi isn't gridded). per coordinate the source is:
    //   S_r   = p*(A_r,hi - A_r,lo)/V  +  mom_phi*v_phi/r   (pressure-curvature + centrifugal)
    //   S_phi = -mom_r*v_phi/r                              (angular-momentum, the swirl law)
    //   S_z   = 0                                           (z-faces z-invariant; no curvature)
    // uniform state: rho=1 (cons.mom = rho*v = v), p=1, v = (v_r, v_phi, v_z).
    let (vr, vphi, _vz, p) = (0.1_f64, 0.3_f64, 0.05_f64, 1.0_f64);

    let out = KernelRun::new(geometric_momentum_source_probe_gv(
        Coords::Cylindrical, &[Spacing::Uniform; 2], &[0, 2], 2, 3,
        GeoSource::Hydro { inertial: true },
    ))
    .grid([NR, NZ])
    .fields(&[("pre", p), ("mom_0", vr), ("mom_1", vphi), ("prim_v0", vr), ("prim_v1", vphi)])
    .scalars(&[("x_lo_0", R0), ("dx_0", DR), ("x_lo_1", Z0), ("dx_1", DZ)])
    .run();

    for i in 0..NR {
        for j in 0..NZ {
            let rl = R0 + i as f64 * DR;
            let rh = rl + DR;
            let ir2 = (rh * rh - rl * rl) / 2.0;
            let centroid_r = (2.0 / 3.0) * (rh.powi(3) - rl.powi(3)) / (rh.powi(2) - rl.powi(2));
            // pressure-curvature: p*(A_r,hi - A_r,lo)/V = p*(rh-rl)/Ir2 (i_phi, i_z cancel).
            let want_sr = p * (rh - rl) / ir2 + vphi * vphi / centroid_r; // mom_phi*v_phi = v_phi^2
            let want_sphi = -(vr * vphi) / centroid_r;
            close(out.get([i, j], "s_0"), want_sr, "S_r", i, j);
            close(out.get([i, j], "s_1"), want_sphi, "S_phi", i, j);
            close(out.get([i, j], "s_2"), 0.0, "S_z", i, j);
        }
    }
    // the swirl angular-momentum source is genuinely nonzero (the new physics).
    assert!(out.values("s_1")[0].abs() > 1e-6, "swirl S_phi vanished — angular-momentum source idle");
}

#[test]
fn cylindrical_rz_swirl_flux_is_advective() {
    // ncomp>ndim hydro flux (docs/design/18 stage 3): cylindrical (r,z) [0,2], ncomp=3.
    // the gv cyl r-z flux (riemann::hlle at the Newtonian regime, coord_n = axes[0] = r)
    // builds a momentum flux for ALL THREE coordinates; for a uniform state HLLE returns the
    // physical flux, so along the r-sweep (coord_n=r):
    //   F_den = rho*v_r;  F_mom_r = rho*v_r^2 + p;  F_mom_phi = rho*v_phi*v_r (advection, NO
    //   pressure — phi != sweep coord);  F_mom_z = rho*v_z*v_r.
    let (rho, vr, vphi, vz, p, gamma) = (1.3_f64, 0.2_f64, 0.4_f64, 0.1_f64, 0.7_f64, 1.4_f64);
    let (nr, nz) = (6usize, 3usize);

    // recon reads i-2..i+1, so compute only the interior (i in 2..5) of a full nr x nz buffer.
    let out = KernelRun::new(adiabatic_flux_cyl_rz_gv(0)) // sweep r
        .grid([nr, nz])
        .compute_window([2, 0], [3, nz])
        .fields(&[("prim_rho", rho), ("prim_v0", vr), ("prim_v1", vphi), ("prim_v2", vz), ("prim_pre", p)])
        // theta=1 == plain minmod; static mesh -> zero mesh motion; uniform geometry (the
        // advective flux of a uniform state is geometry-independent so x_lo_0/dx_0 are arbitrary).
        .scalars(&[
            ("gamma", gamma),
            ("theta", 1.0),
            ("mesh_adot_0", 0.0),
            ("x_lo_0", 1.0),
            ("dx_0", 1.0),
            ("mesh_vtrans_0", 0.0),
        ])
        .run();

    // interior cell (i=3, j=1): recon reads i-2..i+1, all uniform -> HLLE returns the physical flux.
    let cell = [3usize, 1];
    let vsq = vr * vr + vphi * vphi + vz * vz;
    let e = p / (gamma - 1.0) + 0.5 * rho * vsq;
    let close1 = |got: f64, want: f64, what: &str| {
        let rel = (got - want).abs() / want.abs().max(1.0);
        assert!(rel < 1e-12, "{what}: got {got} want {want} (rel {rel:e})");
    };
    close1(out.get(cell, "flux_den"), rho * vr, "F_den");
    close1(out.get(cell, "flux_mom_0"), rho * vr * vr + p, "F_mom_r (carries pressure)");
    close1(out.get(cell, "flux_mom_1"), rho * vphi * vr, "F_mom_phi (advection, no pressure)");
    close1(out.get(cell, "flux_mom_2"), rho * vz * vr, "F_mom_z (advection)");
    close1(out.get(cell, "flux_nrg"), (e + p) * vr, "F_energy");
}
