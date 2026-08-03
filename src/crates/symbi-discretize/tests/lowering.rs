// =============================================================================
// lowering.rs
//
// the LOWERABILITY contract (the third leg of the kernel trio, alongside the
// carrier-equivalence oracle + its numeric tolerance): every production kernel's
// traced graph must render to clean CPU (rust) AND GPU (CUDA) source. an op the
// renderer can't lower PANICS in `assert_lowers`, on the CPU test path, catching a
// GPU-codegen regression long before the nvcc / on-device gate. this is the fast,
// device-free "runs everywhere" check.
//
// the canonical kernel list is symbi-aot/build.rs: every `gen_*` there constructs
// a builder and emits it. each builder call below MIRRORS a build.rs call EXACTLY
// (same args), so this suite covers every distinct (regime, geometry, ndim, dir)
// kernel shape build.rs generates. the grid ndim MUST equal the builder's
// construction ndim — stencil/curvilinear kernels reference coord axes 0..ndim, so
// a mismatch panics or mis-emits.
// =============================================================================

mod harness;
use harness::KernelRun;

use symbi_discretize::gv::{
    adiabatic_flux_cyl_rz_gv, adiabatic_flux_gv, iso_flux_gv, rhd_flux_gv, rmhd_flux_gv,
};
use symbi_discretize::{
    Recon,
    Coords, GeoSource, Spacetime, Spacing, adiabatic_c2p_gv, body_feedback_gv, body_source_gv,
    geometric_momentum_source_probe_gv, geometry_probe_gv, godunov_mass_gv, godunov_stage_gv,
    inertial_momentum_probe_gv, iso_c2p_gv, iso_ghost_fill_gv, iso_wave_speed_map_gv, rhd_c2p_gv,
    rhd_wave_speed_map_gv, rmhd_average_efield_gv, rmhd_bcell_from_bface_gv,
    rmhd_bcell_godunov_euler_gv, rmhd_bcell_godunov_rk2_gv, rmhd_c2p_gv, rmhd_ct_curl_2d_dir_gv,
    rmhd_ct_curl_3d_dir_gv, rmhd_edge_emf_gv, rmhd_ghost_fill_gv, rmhd_save_efield_gv,
    rmhd_wave_speed_map_gv, snapshot_gv,
};

// MAX_SOURCE_BODIES is owned by the runtime (symbi_ib::collection::MAX_SOURCE_BODIES = 2); mirrored
// here as a literal since symbi-discretize does not depend on symbi-ib. the lowering
// only cares that the per-body-unrolled graph builds + renders, regardless of the exact count.
const MAX_SOURCE_BODIES: usize = 2;

// the curvilinear momentum source the godunov binds, by regime prefix — mirrors
// build.rs::geo_source. rmhd uses the magnetic-tension source; the hydro regimes
// (iso/adiabatic/rhd) the inertial pressure+centrifugal source.
fn geo_source(prefix: &str) -> GeoSource {
    match prefix {
        "rmhd" => GeoSource::Rmhd,
        _ => GeoSource::Hydro {
            inertial: matches!(prefix, "iso" | "adiabatic" | "rhd"),
        },
    }
}

// -----------------------------------------------------------------------------
// c2p (cons -> prim): pointwise, geometry-independent. iso/adiabatic closed-form,
// rhd/rmhd iterative. any ndim works (build.rs emits 1..=3); one per regime here.
// -----------------------------------------------------------------------------
#[test]
fn c2p_kernels_lower_to_every_backend() {
    KernelRun::new(iso_c2p_gv::<1>()).grid([8]).assert_lowers();
    KernelRun::new(adiabatic_c2p_gv::<1>())
        .grid([8])
        .assert_lowers();
    // cyl r-z adiabatic c2p folds a 3-component velocity on a 2-axis grid (ncomp=3).
    KernelRun::new(adiabatic_c2p_gv::<3>())
        .grid([8])
        .assert_lowers();
    KernelRun::new(rhd_c2p_gv::<1>(20))
        .grid([8])
        .assert_lowers();
    KernelRun::new(rmhd_c2p_gv(100)).grid([8]).assert_lowers();
}

// -----------------------------------------------------------------------------
// face flux: PLM reconstruction + the canonical HLLE per regime. the sweep dir is
// baked per kernel instance, so the grid ndim == the builder's D and dir < ndim.
// cartesian iso/adiabatic/rhd at 1D dir-0; cyl-rz adiabatic (3-comp on 2D grid);
// rmhd 1D + per-dir 3D.
// -----------------------------------------------------------------------------
#[test]
fn flux_kernels_lower() {
    // cartesian, dir 0, ndim 1 (build.rs emits all (ndim,dir); one of each family here).
    KernelRun::new(iso_flux_gv::<1>(0))
        .grid([8])
        .assert_lowers();
    KernelRun::new(adiabatic_flux_gv::<1>(0, Recon::Plm))
        .grid([8])
        .assert_lowers();
    KernelRun::new(rhd_flux_gv::<1>(0))
        .grid([8])
        .assert_lowers();
    // a 2D cartesian instance per family to exercise the transverse stencil axis.
    KernelRun::new(iso_flux_gv::<2>(1))
        .grid([8, 8])
        .assert_lowers();
    KernelRun::new(adiabatic_flux_gv::<2>(1, Recon::Plm))
        .grid([8, 8])
        .assert_lowers();
    KernelRun::new(rhd_flux_gv::<2>(1))
        .grid([8, 8])
        .assert_lowers();
    // cyl r-z adiabatic flux: 3-component swirl on a 2-axis (r,z) grid, both sweep dirs.
    KernelRun::new(adiabatic_flux_cyl_rz_gv(0))
        .grid([8, 8])
        .assert_lowers();
    KernelRun::new(adiabatic_flux_cyl_rz_gv(1))
        .grid([8, 8])
        .assert_lowers();
    // rmhd flux: 8 conserved fluxes (D,S,tau,B). 1D dir-0 + the per-dir 3D instances.
    KernelRun::new(rmhd_flux_gv(1, 0, 0))
        .grid([8])
        .assert_lowers();
    for dir in 0..3 {
        KernelRun::new(rmhd_flux_gv(3, dir, dir as usize))
            .grid([8, 8, 8])
            .assert_lowers();
    }
}

// -----------------------------------------------------------------------------
// wave-speed maps (per-cell CFL lambda): iso (shared by adiabatic), rhd, rmhd.
// cartesian + a curvilinear instance each. curvilinear folds per-cell physical
// widths (log-radial spherical), so its grid ndim must match the axes.
// -----------------------------------------------------------------------------
#[test]
fn wave_speed_kernels_lower() {
    // iso: cartesian 1D + log-radial spherical 1D (per-cell physical width).
    KernelRun::new(iso_wave_speed_map_gv(
        Coords::Cartesian,
        &[Spacing::Uniform],
        &[0],
        1,
    ))
    .grid([8])
    .assert_lowers();
    KernelRun::new(iso_wave_speed_map_gv(
        Coords::Spherical,
        &[Spacing::Log],
        &[0],
        1,
    ))
    .grid([8])
    .assert_lowers();
    // iso spherical 3D (the curvilinear hydro family) to exercise multi-axis widths.
    KernelRun::new(iso_wave_speed_map_gv(
        Coords::Spherical,
        &[Spacing::Uniform; 3],
        &[0, 1, 2],
        3,
    ))
    .grid([8, 8, 8])
    .assert_lowers();
    // rhd: cartesian 1D + spherical 3D.
    KernelRun::new(rhd_wave_speed_map_gv(
        Coords::Cartesian,
        Spacetime::Minkowski,
        &[Spacing::Uniform],
        &[0],
        1,
    ))
    .grid([8])
    .assert_lowers();
    KernelRun::new(rhd_wave_speed_map_gv(
        Coords::Spherical,
        Spacetime::Minkowski,
        &[Spacing::Uniform; 3],
        &[0, 1, 2],
        3,
    ))
    .grid([8, 8, 8])
    .assert_lowers();
    // rmhd: the quartic-wave-speed map, cartesian 3D + spherical 3D.
    KernelRun::new(rmhd_wave_speed_map_gv(
        Coords::Cartesian,
        &[Spacing::Uniform; 3],
        &[0, 1, 2],
        3,
    ))
    .grid([8, 8, 8])
    .assert_lowers();
    KernelRun::new(rmhd_wave_speed_map_gv(
        Coords::Spherical,
        &[Spacing::Uniform; 3],
        &[0, 1, 2],
        3,
    ))
    .grid([8, 8, 8])
    .assert_lowers();
}

// -----------------------------------------------------------------------------
// godunov euler/rk2/mass + snapshot: the EOS-generic conserved update. cartesian
// (no geometric source) + spherical + cylindrical (area-weighted divergence + the
// regime-shared geometric momentum source). the grid ndim == the builder's ndim.
// -----------------------------------------------------------------------------
#[test]
fn godunov_kernels_lower() {
    // single mass law, separate output buffer. cartesian + spherical area-weighted.
    KernelRun::new(godunov_mass_gv(
        Coords::Cartesian,
        &[Spacing::Uniform],
        &[0],
        1,
    ))
    .grid([8])
    .assert_lowers();
    KernelRun::new(godunov_mass_gv(
        Coords::Spherical,
        &[Spacing::Uniform],
        &[0],
        1,
    ))
    .grid([8])
    .assert_lowers();

    // cartesian godunov-stage/snapshot for each EOS regime, 1D (iso no-energy, adiabatic/rhd
    // energy). the one `godunov_stage_gv` kernel serves every SSP scheme (euler/rk2/rk3) via the
    // runtime (a0, ac) coefficients, so one lowering check per geometry replaces the euler+rk2 pair.
    for (prefix, has_energy) in [("iso", false), ("adiabatic", true), ("rhd", true)] {
        KernelRun::new(godunov_stage_gv(
            Coords::Cartesian,
            Spacetime::Minkowski,
            &[Spacing::Uniform],
            &[0],
            1,
            1,
            has_energy,
            geo_source(prefix),
        ))
        .grid([8])
        .assert_lowers();
        KernelRun::new(snapshot_gv(1, has_energy))
            .grid([8])
            .assert_lowers();
    }

    // spherical curvilinear hydro (adiabatic, energy) at 2D — area-weighted + inertial source.
    KernelRun::new(godunov_stage_gv(
        Coords::Spherical,
        Spacetime::Minkowski,
        &[Spacing::Uniform; 2],
        &[0, 1],
        2,
        2,
        true,
        geo_source("adiabatic"),
    ))
    .grid([8, 8])
    .assert_lowers();

    // cylindrical r-z axisymmetric adiabatic (ncomp=3 swirl on a 2-axis (r,z) grid).
    KernelRun::new(godunov_stage_gv(
        Coords::Cylindrical,
        Spacetime::Minkowski,
        &[Spacing::Uniform; 2],
        &[0, 2],
        2,
        3,
        true,
        geo_source("adiabatic"),
    ))
    .grid([8, 8])
    .assert_lowers();
    KernelRun::new(snapshot_gv(3, true))
        .grid([8, 8])
        .assert_lowers();

    // cylindrical r-phi disk (ncomp == ndim == 2, natural plane).
    KernelRun::new(godunov_stage_gv(
        Coords::Cylindrical,
        Spacetime::Minkowski,
        &[Spacing::Uniform; 2],
        &[0, 1],
        2,
        2,
        true,
        geo_source("adiabatic"),
    ))
    .grid([8, 8])
    .assert_lowers();

    // rmhd hydro godunov-stage (D/S/tau), 3D cartesian + spherical.
    KernelRun::new(godunov_stage_gv(
        Coords::Cartesian,
        Spacetime::Minkowski,
        &[Spacing::Uniform; 3],
        &[0, 1, 2],
        3,
        3,
        true,
        geo_source("rmhd"),
    ))
    .grid([8, 8, 8])
    .assert_lowers();
    KernelRun::new(godunov_stage_gv(
        Coords::Spherical,
        Spacetime::Minkowski,
        &[Spacing::Uniform; 3],
        &[0, 1, 2],
        3,
        3,
        true,
        geo_source("rmhd"),
    ))
    .grid([8, 8, 8])
    .assert_lowers();
    KernelRun::new(snapshot_gv(3, true))
        .grid([8, 8, 8])
        .assert_lowers();
}

// -----------------------------------------------------------------------------
// ghost fill: the lattice-map pullback stencil (read at the per-axis source coord).
// in-place; the grid ndim == the builder's ndim. iso (cartesian + cyl r-z) + rmhd 3D.
// -----------------------------------------------------------------------------
#[test]
fn ghost_kernels_lower() {
    // iso cartesian 1D (identity axes) + cyl r-z 2D (3-component, axes [0,2]).
    KernelRun::new(iso_ghost_fill_gv(1, 1, &[0]))
        .grid([8])
        .assert_lowers();
    KernelRun::new(iso_ghost_fill_gv(2, 3, &[0, 2]))
        .grid([8, 8])
        .assert_lowers();
    // rmhd 3D pullback: prim rho/vel/pre + mhd.bcell with vel/B sign flips.
    KernelRun::new(rmhd_ghost_fill_gv(3, 3))
        .grid([8, 8, 8])
        .assert_lowers();
}

// -----------------------------------------------------------------------------
// constrained transport (rmhd B-field): 2D curl, per-dir 3D curl, per-dir edge EMF,
// face->cell + magnetic-energy correction, bcell godunov euler/rk2, save/average
// efield. 2D at ndim=2, everything else at ndim=3 (build.rs emits them at 3).
// -----------------------------------------------------------------------------
#[test]
fn ct_kernels_lower() {
    // 2D in-plane curl from the out-of-plane edge EMF (built at ndim=2); the combined 2d curl was
    // split per in-plane direction (dir=0 -> B_x, dir=1 -> B_y, both from the corner E_z).
    for dir in 0..2 {
        KernelRun::new(rmhd_ct_curl_2d_dir_gv(dir))
            .grid([8, 8])
            .assert_lowers();
    }
    // per-dir 3D curl, cartesian + spherical + cylindrical (the orthogonal-curl scale-factor weights).
    for dir in 0..3 {
        for coords in [Coords::Cartesian, Coords::Spherical, Coords::Cylindrical] {
            KernelRun::new(rmhd_ct_curl_3d_dir_gv(coords, &[Spacing::Uniform; 3], dir))
                .grid([8, 8, 8])
                .assert_lowers();
        }
    }
    // per-dir 3D edge EMF.
    for dir in 0..3 {
        KernelRun::new(rmhd_edge_emf_gv(3, (dir + 1) % 3, (dir + 2) % 3))
            .grid([8, 8, 8])
            .assert_lowers();
    }
    // face->cell B interpolation.
    KernelRun::new(rmhd_bcell_from_bface_gv(3))
        .grid([8, 8, 8])
        .assert_lowers();
    // cell-B out-of-plane flux predictor (euler + rk2) on a 2D reduced plane (axes [0,1] -> the
    // predictor writes the single out-of-plane component); cartesian + spherical + cylindrical.
    for coords in [Coords::Cartesian, Coords::Spherical, Coords::Cylindrical] {
        KernelRun::new(rmhd_bcell_godunov_euler_gv(
            coords,
            Spacetime::Minkowski,
            &[Spacing::Uniform; 2],
            2,
            3,
            &[0, 1],
        ))
        .grid([8, 8])
        .assert_lowers();
        KernelRun::new(rmhd_bcell_godunov_rk2_gv(
            coords,
            Spacetime::Minkowski,
            &[Spacing::Uniform; 2],
            2,
            3,
            &[0, 1],
        ))
        .grid([8, 8])
        .assert_lowers();
    }
    // edge-EMF save (out-of-place copy) + time-average (in-place).
    KernelRun::new(rmhd_save_efield_gv())
        .grid([8, 8, 8])
        .assert_lowers();
    KernelRun::new(rmhd_average_efield_gv())
        .grid([8, 8, 8])
        .assert_lowers();
}

// -----------------------------------------------------------------------------
// immersed bodies: forward source (gravity + accretion, cons->cons) + backward
// feedback (force/torque/mass scratch). per geometry; the grid ndim == ndim.
// -----------------------------------------------------------------------------
#[test]
fn immersed_kernels_lower() {
    // body source: cartesian 1..=3, curvilinear cyl (r-phi 2D, r-phi-z 3D) + spherical (2D, 3D).
    for ndim in 1usize..=3 {
        let axes: Vec<usize> = (0..ndim).collect();
        KernelRun::new(body_source_gv(
            MAX_SOURCE_BODIES,
            Coords::Cartesian,
            ndim,
            ndim,
            &axes,
            false,
        ))
        .grid(vec![8usize; ndim])
        .assert_lowers();
    }
    for &(coords, ndim) in &[
        (Coords::Cylindrical, 2usize),
        (Coords::Cylindrical, 3),
        (Coords::Spherical, 2),
        (Coords::Spherical, 3),
    ] {
        let axes: Vec<usize> = (0..ndim).collect();
        KernelRun::new(body_source_gv(
            MAX_SOURCE_BODIES,
            coords,
            ndim,
            ndim,
            &axes,
            false,
        ))
        .grid(vec![8usize; ndim])
        .assert_lowers();
    }
    // body feedback: every geometry, ndim 2 + 3.
    for &cc in &[Coords::Cartesian, Coords::Cylindrical, Coords::Spherical] {
        for ndim in 2usize..=3 {
            let axes: Vec<usize> = (0..ndim).collect();
            KernelRun::new(body_feedback_gv(MAX_SOURCE_BODIES, cc, ndim, ndim, &axes))
                .grid(vec![8usize; ndim])
                .assert_lowers();
        }
    }
}

// -----------------------------------------------------------------------------
// geometry-algebra probes: the in-kernel metric (inverse volume, face areas,
// volume-weighted centroid) + the inertial + geometric momentum source probes.
// the grid ndim == the probe's ndim.
// -----------------------------------------------------------------------------
#[test]
fn geometry_probe_kernels_lower() {
    // geometry probe: cartesian + spherical (uniform + log), 1D + 2D.
    KernelRun::new(geometry_probe_gv(Coords::Cartesian, &[Spacing::Uniform], 1))
        .grid([8])
        .assert_lowers();
    KernelRun::new(geometry_probe_gv(Coords::Spherical, &[Spacing::Uniform], 1))
        .grid([8])
        .assert_lowers();
    KernelRun::new(geometry_probe_gv(Coords::Spherical, &[Spacing::Log], 1))
        .grid([8])
        .assert_lowers();
    KernelRun::new(geometry_probe_gv(
        Coords::Spherical,
        &[Spacing::Log, Spacing::Uniform],
        2,
    ))
    .grid([8, 8])
    .assert_lowers();
    // newtonian inertial (centrifugal/coriolis) momentum source, 2D spherical.
    KernelRun::new(inertial_momentum_probe_gv(
        Coords::Spherical,
        &[Spacing::Uniform; 2],
        2,
    ))
    .grid([8, 8])
    .assert_lowers();
    // full rmhd geometric momentum source (total pressure + gas inertial + magnetic tension), 3D
    // spherical AND cylindrical — the coord-generic christoffel covers both (cyl: r-phi pair).
    for coords in [Coords::Spherical, Coords::Cylindrical] {
        KernelRun::new(geometric_momentum_source_probe_gv(
            coords,
            &[Spacing::Uniform; 3],
            &[0, 1, 2],
            3,
            3,
            GeoSource::Rmhd,
        ))
        .grid([8, 8, 8])
        .assert_lowers();
    }
}
