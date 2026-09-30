// =============================================================================
// gr_meridional_diagnostics.rs
//
// the run-reported totals of relativistic states on the schwarzschild
// kerr-schild meridional (R, z) plane, against independent sums:
// - GR MHD stores the undensitized Valencia state, so its rest mass is
//   sum D sqrt(gamma)(R, 0, z) dV with sqrt(gamma) = R sqrt(1 + 2M/r),
//   r = sqrt(R^2 + z^2), integrated here by two-point gauss over each cell times
//   the ungridded 2 pi;
// - GR hydro stores the densitized state sqrt(-g) rho u^t, so its rest mass is
//   the conserved density over the plain coordinate volume 2 pi dR dz;
// - the largest lorentz factor is 1/sqrt(1 - gamma_ij v^i v^j) of the
//   contravariant velocity at the cell center.
// off the midplane the equatorial metric reads r = R, so each sum fails there.
// =============================================================================
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::SchwarzschildKSCylindricalRz;
use symbi_hydro::Rhd;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type GrmhdSim = SimStateGeneric<Rmhd, 2, 3, SchwarzschildKSCylindricalRz<f64>, IdealGas<f64>, CpuSpace, HostMemory, f64>;
type GrhdSim = SimStateGeneric<Rhd, 2, 3, SchwarzschildKSCylindricalRz<f64>, IdealGas<f64>, CpuSpace, HostMemory, f64>;

const MASS: f64 = 0.8;
const N: [usize; 2] = [12, 16];
const R_LO: f64 = 2.5;
const R_HI: f64 = 6.0;
const Z_HALF: f64 = 4.0;
const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

fn spacing() -> [f64; 2] {
    [(R_HI - R_LO) / N[0] as f64, 2.0 * Z_HALF / N[1] as f64]
}

fn sqrt_gamma(big_r: f64, z: f64) -> f64 {
    big_r * (1.0 + 2.0 * MASS / (big_r * big_r + z * z).sqrt()).sqrt()
}

/// the cell's proper volume: two-point gauss of sqrt(gamma) over (R, z), times 2 pi
fn proper_volume(i: usize, j: usize) -> f64 {
    let [dr, dz] = spacing();
    let (rc, zc) = (R_LO + (i as f64 + 0.5) * dr, -Z_HALF + (j as f64 + 0.5) * dz);
    let o = 0.5 / 3.0_f64.sqrt();
    let mut sum = 0.0;
    for sr in [-1.0, 1.0] {
        for sz in [-1.0, 1.0] {
            sum += sqrt_gamma(rc + sr * o * dr, zc + sz * o * dz);
        }
    }
    sum / 4.0 * dr * dz * TWO_PI
}

fn velocity([big_r, z]: [f64; 2]) -> [f64; 3] {
    [0.05 * z / Z_HALF, 0.02 / big_r, -0.03 * big_r / R_HI]
}

fn boundaries() -> Boundaries<2> {
    Boundaries([[BoundaryType::Outflow; 2], [BoundaryType::Outflow; 2]])
}

#[test]
fn the_grmhd_rest_mass_integrates_the_meridional_measure() {
    let sim = GrmhdSim::build(Rmhd, IdealGas { gamma: 4.0 / 3.0 }, SchwarzschildKSCylindricalRz { mass: MASS })
        .cells(N)
        .origin([R_LO, -Z_HALF])
        .spacing(spacing())
        .boundaries(boundaries())
        .cyl_plane(CylPlane::Rz)
        .allocate()
        .expect("sim")
        .set_initial(|x| {
            MhdPrim::new(
                Prim::adiabatic(Density(1.0 + 0.1 * x[1] / Z_HALF), Tensor::new(velocity(x)), Pressure(0.1)),
                Tensor::new([0.0; 3]),
            )
        })
        .seed_faces_uniform([0.0, 0.0])
        .build();
    let diag = sim.conservation_diag().expect("host diagnostics");
    let int = &sim.geom.interior;
    let mut want = 0.0;
    for c in int.iter() {
        let (i, j) = ((c[0] - int.spaces[0].lo) as usize, (c[1] - int.spaces[1].lo) as usize);
        want += *sim.fields.cons.den.view().at(c) * proper_volume(i, j);
    }
    assert!(!sim.conserved_is_densitized(), "GR MHD stores the undensitized state");
    assert!((diag.mass - want).abs() <= 1e-12 * want, "reported mass {:e} against {want:e}", diag.mass);
}

#[test]
fn the_densitized_grhd_rest_mass_integrates_the_coordinate_volume() {
    let sim = GrhdSim::build(Rhd, IdealGas { gamma: 4.0 / 3.0 }, SchwarzschildKSCylindricalRz { mass: MASS })
        .cells(N)
        .origin([R_LO, -Z_HALF])
        .spacing(spacing())
        .boundaries(boundaries())
        .cyl_plane(CylPlane::Rz)
        .allocate()
        .expect("sim")
        .set_initial(|x| Prim::adiabatic(Density(1.0 + 0.1 * x[1] / Z_HALF), Tensor::new(velocity(x)), Pressure(0.1)))
        .build();
    assert!(sim.conserved_is_densitized(), "GR hydro stores the densitized state");
    let diag = sim.conservation_diag().expect("host diagnostics");
    let [dr, dz] = spacing();
    let want: f64 = sim.geom.interior.iter().map(|c| *sim.fields.cons.den.view().at(c) * dr * dz * TWO_PI).sum();
    assert!((diag.mass - want).abs() <= 1e-12 * want, "reported mass {:e} against {want:e}", diag.mass);
    // the lorentz factor of the contravariant velocity under the meridional metric, read from
    // the primitive velocity the recovery fills; written here for every cell
    for c in sim.geom.interior.iter() {
        let v = velocity(sim.geom.cell_coord(c));
        for k in 0..3 {
            sim.fields.prim.vel[k].set(c, v[k]);
        }
    }
    let diag = sim.conservation_diag().expect("host diagnostics");
    let mut want_w = 1.0_f64;
    for c in sim.geom.interior.iter() {
        let [big_r, z] = sim.geom.cell_coord(c);
        let r = (big_r * big_r + z * z).sqrt();
        let two_h = 2.0 * MASS / r;
        let v = velocity([big_r, z]);
        // gamma over (R, phi, z): the (R, z) block delta + (2H/r^2) x x, and R^2 on phi
        let l = [big_r, 0.0, z];
        let mut v2 = big_r * big_r * v[1] * v[1];
        for a in [0usize, 2] {
            for b in [0usize, 2] {
                let g = f64::from(u8::from(a == b)) + two_h / (r * r) * l[a] * l[b];
                v2 += g * v[a] * v[b];
            }
        }
        want_w = want_w.max(1.0 / (1.0 - v2).sqrt());
    }
    let got_w = diag.max_w.expect("a relativistic regime reports its lorentz factor");
    assert!((got_w - want_w).abs() <= 1e-12 * want_w, "reported max W {got_w} against {want_w}");
}
