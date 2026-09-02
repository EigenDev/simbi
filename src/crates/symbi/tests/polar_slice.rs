// =============================================================================
// polar_slice.rs
//
// the polar (physical-shape) live-heatmap decimation: a 2D spherical (r, theta)
// grid renders as the meridional half-plane and a cylindrical (R, phi) disk
// plane as the disk — by inverse pixel -> cell sampling, with NaN outside the
// annulus/wedge. a constant field must sample constant everywhere inside, the
// outside must be NaN in the right proportion, and a theta-step field must land
// in the right half — the orientation gate a constant field cannot catch.
// =============================================================================

use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{CylindricalRPhi, Spherical};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;

fn seed_rho<R, M2>(
    sim: &SimState<R, 2, M2, IdealGas<f64>, CpuSpace, HostMemory>,
    f: impl Fn(f64, f64) -> f64,
) where
    R: symbi_hydro::regime::Regime<f64, 2, Prim = Prim<f64, 2>>,
    M2: symbi_geometry::Metric<f64, 2> + Copy,
{
    // the live heatmap reads prim.rho; a freshly built sim stores cons only, so
    // seed the primitive directly (the post-c2p state the heatmap sees in flight).
    for c in sim.geom.interior.iter() {
        let x0 = sim.geom.x_lo[0] + (c[0] as f64 + 0.5) * sim.geom.dx[0];
        let x1 = sim.geom.x_lo[1] + (c[1] as f64 + 0.5) * sim.geom.dx[1];
        sim.fields.prim.rho.set(c, f(x0, x1));
    }
}

fn spherical_sim() -> SimState<Newtonian, 2, Spherical, IdealGas<f64>, CpuSpace, HostMemory> {
    // r in [1, 5], theta in [0, pi]: the full meridional half-plane.
    let (nr, nt) = (64usize, 96usize);
    let (dr, dt) = (4.0 / nr as f64, std::f64::consts::PI / nt as f64);
    SimState::build(Newtonian, IdealGas { gamma: GAMMA }, Spherical)
        .cells([nr, nt])
        .origin([1.0, 0.0])
        .spacing([dr, dt])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim")
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::zeros(), Pressure(1.0)))
        .build()
}

#[test]
fn spherical_slice_is_a_meridional_half_annulus() {
    let sim = spherical_sim();
    seed_rho(&sim, |_, _| 3.25);
    let fd = sim.field_slice(120, 0).expect("slice");

    // meridional: x in [0, 1] (half the y span), pole up.
    assert_eq!(
        (fd.width, fd.height),
        (60, 120),
        "meridional half-plane dims"
    );
    assert!(fd.name.contains("meridional"), "label: {}", fd.name);

    let n_inside = fd.data.iter().filter(|v| !v.is_nan()).count();
    // every inside pixel holds the constant; min == max == the constant.
    assert!(fd.data.iter().all(|v| v.is_nan() || *v == 3.25));
    assert_eq!((fd.vmin, fd.vmax), (3.25, 3.25));
    // the half-annulus fills ~ pi (1 - s0^2) / 4 of the [0,1] x [-1,1] box
    // (s0 = 0.08 hole): ~0.78. bounds catch a broken mask or a full rectangle.
    let frac = n_inside as f64 / (fd.width * fd.height) as f64;
    assert!((0.6..0.95).contains(&frac), "inside fraction {frac}");
    // the four corners are outside the half-disk.
    for (px, py) in [(fd.width - 1, 0), (fd.width - 1, fd.height - 1)] {
        assert!(
            fd.data[py * fd.width + px].is_nan(),
            "corner ({px},{py}) not NaN"
        );
    }
}

#[test]
fn meridional_orientation_puts_small_theta_at_the_top() {
    // rho = 1 in the northern half (theta < pi/2), 2 in the southern: the top rows
    // of the display (y > 0, near the pole) must read 1 and the bottom rows 2 —
    // the orientation gate a constant field cannot catch.
    let sim = spherical_sim();
    seed_rho(&sim, |_, theta| {
        if theta < std::f64::consts::FRAC_PI_2 {
            1.0
        } else {
            2.0
        }
    });
    let fd = sim.field_slice(120, 0).expect("slice");
    let row_val = |py: usize| -> Vec<f32> {
        (0..fd.width)
            .filter_map(|px| {
                let v = fd.data[py * fd.width + px];
                (!v.is_nan()).then_some(v)
            })
            .collect()
    };
    let top = row_val(2);
    let bot = row_val(fd.height - 3);
    assert!(
        !top.is_empty() && top.iter().all(|&v| v == 1.0),
        "top rows must be northern"
    );
    assert!(
        !bot.is_empty() && bot.iter().all(|&v| v == 2.0),
        "bottom rows must be southern"
    );
}

#[test]
fn quadrant_wedge_gets_a_tight_square_box() {
    // theta in [0, pi/2] (the thermal-bomb quadrant): the display box must hug the
    // wedge — x, y in [0, 1], a square slice with the pole up the left edge and the
    // equator along the bottom — not float in a mostly-NaN full half-plane.
    let (nr, nt) = (64usize, 48usize);
    let sim = SimState::<Newtonian, 2, Spherical, IdealGas<f64>, CpuSpace, HostMemory>::build(
        Newtonian,
        IdealGas { gamma: GAMMA },
        Spherical,
    )
    .cells([nr, nt])
    .origin([1.0, 0.0])
    .spacing([4.0 / nr as f64, std::f64::consts::FRAC_PI_2 / nt as f64])
    .boundaries(Boundaries::uniform(BoundaryType::Outflow))
    .allocate()
    .expect("sim")
    .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::zeros(), Pressure(1.0)))
    .build();
    seed_rho(&sim, |_, _| 5.5);
    let fd = sim.field_slice(120, 0).expect("slice");

    assert_eq!(
        (fd.width, fd.height),
        (120, 120),
        "quadrant wedge box must be square"
    );
    // the quarter annulus fills ~ pi/4 (1 - s0^2) ~ 0.78 of its tight box.
    let frac =
        fd.data.iter().filter(|v| !v.is_nan()).count() as f64 / (fd.width * fd.height) as f64;
    assert!((0.6..0.95).contains(&frac), "inside fraction {frac}");
    // the origin corner (bottom-left, inside the hole) and the far corner
    // (top-right, outside the arc) are NaN; the mid-radius diagonal is inside.
    assert!(
        fd.data[(fd.height - 1) * fd.width].is_nan(),
        "origin corner not NaN"
    );
    assert!(fd.data[fd.width - 1].is_nan(), "far corner not NaN");
    let mid = (fd.height / 2) * fd.width + fd.width / 2;
    assert_eq!(fd.data[mid], 5.5, "mid-wedge sample");
}

#[test]
fn cylindrical_rphi_slice_is_a_disk_with_wraparound() {
    // R in [0.5, 2], phi in [0, 2 pi]: the full disk with a central hole.
    let (n_r, n_p) = (48usize, 128usize);
    let two_pi = 2.0 * std::f64::consts::PI;
    let sim =
        SimState::<Newtonian, 2, CylindricalRPhi, IdealGas<f64>, CpuSpace, HostMemory>::build(
            Newtonian,
            IdealGas { gamma: GAMMA },
            CylindricalRPhi,
        )
        .cells([n_r, n_p])
        .origin([0.5, 0.0])
        .spacing([1.5 / n_r as f64, two_pi / n_p as f64])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim")
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::zeros(), Pressure(1.0)))
        .build()
        // the 2d cylindrical default is the (R, z) plane; the disk view needs (R, phi).
        .with_cyl_plane(CylPlane::RPhi);
    seed_rho(&sim, |_, _| 7.5);
    let fd = sim.field_slice(120, 0).expect("slice");

    assert_eq!((fd.width, fd.height), (120, 120), "disk dims");
    assert!(fd.name.contains("disk"), "label: {}", fd.name);
    assert!(fd.data.iter().all(|v| v.is_nan() || *v == 7.5));
    // the annulus fills ~ pi (1 - s0^2) / 4 ~ 0.78 of the square; the center
    // (the hole) and the corners are NaN.
    let frac =
        fd.data.iter().filter(|v| !v.is_nan()).count() as f64 / (fd.width * fd.height) as f64;
    assert!((0.6..0.95).contains(&frac), "inside fraction {frac}");
    let mid = fd.height / 2 * fd.width + fd.width / 2;
    assert!(fd.data[mid].is_nan(), "central hole missing");
    assert!(fd.data[0].is_nan(), "corner not NaN");
}

#[test]
fn thin_shell_on_a_fine_grid_renders_as_a_continuous_arc() {
    // a 2-cell-thick bright shell on a 1024-cell radial grid: each display pixel
    // spans ~8 radial cells, so an undersampled inverse map aliases the shell into
    // a dotted arc (some rays hit it, some miss). footprint-matched supersampling
    // must catch the shell on every ray: each display row crossing the shell's
    // radius carries at least one shell-brightened pixel.
    let (nr, nt) = (1024usize, 256usize);
    let sim = SimState::<Newtonian, 2, Spherical, IdealGas<f64>, CpuSpace, HostMemory>::build(
        Newtonian,
        IdealGas { gamma: GAMMA },
        Spherical,
    )
    .cells([nr, nt])
    .origin([1.0, 0.0])
    .spacing([4.0 / nr as f64, std::f64::consts::PI / nt as f64])
    .boundaries(Boundaries::uniform(BoundaryType::Outflow))
    .allocate()
    .expect("sim")
    .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::zeros(), Pressure(1.0)))
    .build();
    // the shell: radial cells [510, 512) at 10x ambient, at every angle.
    let dr = 4.0 / nr as f64;
    let r_shell_lo = 1.0 + 510.0 * dr;
    let r_shell_hi = 1.0 + 512.0 * dr;
    seed_rho(&sim, |r, _| {
        if (r_shell_lo..r_shell_hi).contains(&r) {
            10.0
        } else {
            1.0
        }
    });
    let fd = sim.field_slice(120, 0).expect("slice");

    // the shell's display radius (index fraction, s0 hole = 0.08): rows crossing it
    // satisfy |y_row| < s_shell. every such row must contain a pixel visibly above
    // ambient — the averaged shell (~2 bright of ~8 cells) clears 1.5 easily.
    let s_shell = 0.08 + (1.0 - 0.08) * (511.0 / nr as f64);
    let mut rows_checked = 0usize;
    for jj in 0..fd.height {
        let y_row = 1.0 - (jj as f64 + 0.5) / fd.height as f64 * 2.0;
        if y_row.abs() < s_shell - 0.02 {
            rows_checked += 1;
            let hit = (0..fd.width).any(|ii| {
                let v = fd.data[jj * fd.width + ii];
                !v.is_nan() && v > 1.5
            });
            assert!(
                hit,
                "row {jj} (y={y_row:.3}) crosses the shell but shows no arc pixel"
            );
        }
    }
    assert!(
        rows_checked > 50,
        "the shell must cross many rows (got {rows_checked})"
    );
}

#[test]
fn cartesian_slice_is_unchanged_by_the_polar_path() {
    // a 2D cartesian grid keeps the plain index-space decimation: full rectangle,
    // no NaN, unlabeled projection.
    let n = 32usize;
    let sim = SimState::<
        Newtonian,
        2,
        symbi_geometry::Cartesian,
        IdealGas<f64>,
        CpuSpace,
        HostMemory,
    >::build(
        Newtonian,
        IdealGas { gamma: GAMMA },
        symbi_geometry::Cartesian,
    )
    .cells([n, n])
    .origin([-1.0, -1.0])
    .spacing([2.0 / n as f64, 2.0 / n as f64])
    .boundaries(Boundaries::uniform(BoundaryType::Outflow))
    .allocate()
    .expect("sim")
    .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::zeros(), Pressure(1.0)))
    .build();
    seed_rho(&sim, |_, _| 2.0);
    let fd = sim.field_slice(120, 0).expect("slice");
    assert!(
        fd.data.iter().all(|v| !v.is_nan()),
        "cartesian slice must be dense"
    );
    assert!(!fd.name.contains("meridional") && !fd.name.contains("disk"));
}
