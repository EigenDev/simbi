// =============================================================================
// census_mask_charts.rs
//
// a census on a curvilinear chart leaves out the cells the run's own excision
// and body masks act on, judged in the coordinates each mask is defined in:
// - excision on a spherical chart is the slab whose radial face midpoint lies
//   inside the excision radius, the excision fill's own sampling point;
// - a body masks the cells whose centroid, embedded in cartesian space, lies
//   inside the body, with the body's center read through the chart's
//   body-axis convention (the (r, theta) and (R, z) planes are the x-z plane).
// each gate samples a mass census of unit-density gas and compares the kept
// fraction of the total volume with the fraction an independent per-cell count
// keeps, from analytic cell volumes. each premise asserts that judging the mask
// in chart coordinates would count differently, and the excision radius sits
// between a cell's radial midpoint and its volume centroid, so the sampling
// point decides that cell.
//
// run: cargo test -p symbi --test census_mask_charts
// =============================================================================

use std::f64::consts::PI;
use symbi::prelude::SimSubstrate;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{Cylindrical, Spherical};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection};
use symbi_sim::census::RegisteredCensus;
use symbi_sim::substrate_seam::KernelSet;
use symbi_source_compile::CensusConfig;
use symbi_substrate::census_sample::{NonphysicalCells, level_partials};
use symbi_xpu::{CpuSpace, HostMemory};

type SphSim = SimState<Newtonian, 2, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;
type CylSim = SimState<Newtonian, 2, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory>;

const NR: usize = 16;
const NT: usize = 8;
const R_LO: f64 = 1.0;
const R_HI: f64 = 3.0;
const T_LO: f64 = 0.1;
const T_HI: f64 = PI - 0.1;
const DR: f64 = (R_HI - R_LO) / NR as f64;
const DT: f64 = (T_HI - T_LO) / NT as f64;

fn census() -> RegisteredCensus {
    let cfg = CensusConfig::from_json(
        r#"{"name": "mass", "axes": [], "values": [2], "value_names": ["mass"], "op": "add", "params": [],
            "nodes": [{"op": "VARIABLE_RHO"}, {"op": "VARIABLE_DV"}, {"op": "MULTIPLY", "left": 0, "right": 1}]}"#,
    )
    .expect("census json");
    RegisteredCensus::new(symbi_sim::census::CensusEvaluator::new(&cfg).expect("census compiles"))
}

fn mass_sph(s: &SphSim, nonphysical: NonphysicalCells) -> f64 {
    let regs = vec![census()];
    level_partials(s, &regs, 0, None, None, nonphysical, s.time)[0].0.iter().sum()
}

fn mass_cyl(s: &CylSim, nonphysical: NonphysicalCells) -> f64 {
    let regs = vec![census()];
    level_partials(s, &regs, 0, None, None, nonphysical, s.time)[0].0.iter().sum()
}

fn sph(bodies: Option<BodyCollection<f64, 2>>) -> SphSim {
    let s = SphSim::build(Newtonian, IdealGas { gamma: 1.4 }, Spherical)
        .cells([NR, NT])
        .origin([R_LO, T_LO])
        .spacing([DR, DT])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("spherical sim")
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0]), Pressure(1.0)))
        .build();
    let s = match bodies {
        Some(b) => s.with_bodies(b),
        None => s,
    };
    s.substrate().c2p(&s);
    s
}

/// the radial and polar faces of spherical cell (i, j)
fn sph_faces(i: usize, j: usize) -> (f64, f64, f64, f64) {
    (R_LO + i as f64 * DR, R_LO + (i + 1) as f64 * DR, T_LO + j as f64 * DT, T_LO + (j + 1) as f64 * DT)
}

/// the volume of spherical cell (i, j) up to the azimuthal constant
fn sph_volume(i: usize, j: usize) -> f64 {
    let (rl, rh, tl, th) = sph_faces(i, j);
    (rh.powi(3) - rl.powi(3)) / 3.0 * (tl.cos() - th.cos())
}

/// the volume centroid of spherical cell (i, j), written out from its first moments
fn sph_centroid(i: usize, j: usize) -> (f64, f64) {
    let (rl, rh, tl, th) = sph_faces(i, j);
    let r = 0.75 * (rh.powi(4) - rl.powi(4)) / (rh.powi(3) - rl.powi(3));
    let t = ((th.sin() - th * th.cos()) - (tl.sin() - tl * tl.cos())) / (tl.cos() - th.cos());
    (r, t)
}

/// the fraction of the total volume the cells `kept` keep
fn kept_fraction(cells: impl Iterator<Item = (f64, bool)>) -> f64 {
    let (mut kept, mut total) = (0.0, 0.0);
    for (dv, keep) in cells {
        total += dv;
        if keep {
            kept += dv;
        }
    }
    kept / total
}

fn sph_cells() -> impl Iterator<Item = (usize, usize)> {
    (0..NR).flat_map(|i| (0..NT).map(move |j| (i, j)))
}

#[test]
fn the_spherical_excised_slab_is_left_out() {
    let s = sph(None);
    // between the radial midpoint (1.3125) and the volume centroid (~1.3155) of radial cell 2
    let r_exc = 1.314;
    let whole = mass_sph(&s, NonphysicalCells::default());
    let masked = mass_sph(&s, NonphysicalCells { excision_radius: Some(r_exc) });
    let midpoint_out = |i: usize, j: usize| {
        let (rl, rh, _, _) = sph_faces(i, j);
        0.5 * (rl + rh) < r_exc
    };
    let want = kept_fraction(sph_cells().map(|(i, j)| (sph_volume(i, j), !midpoint_out(i, j))));
    let chart = kept_fraction(sph_cells().map(|(i, j)| {
        let (rl, rh, tl, th) = sph_faces(i, j);
        let (r, t) = (0.5 * (rl + rh), 0.5 * (tl + th));
        (sph_volume(i, j), r * r + t * t >= r_exc * r_exc)
    }));
    let centroid = kept_fraction(sph_cells().map(|(i, j)| (sph_volume(i, j), sph_centroid(i, j).0 >= r_exc)));
    assert!((want - chart).abs() > 1e-3, "the chart-coordinate mask keeps the same cells; the gate is vacuous");
    assert!((want - centroid).abs() > 1e-3, "the radius decides no cell between midpoint and centroid");
    assert!(want < 1.0, "the slab holds no cell");
    let got = masked / whole;
    assert!((got - want).abs() < 1e-12, "the census keeps {got} of the volume, the excised slab {want}");
}

#[test]
fn a_body_on_the_spherical_meridional_plane_masks_its_cartesian_interior() {
    // a rigid sphere centred at cartesian (x, z) = (2.0, 0.6), its 2d position in the plane's
    // (x, z) components
    let (cx, cz, radius) = (2.0, 0.6, 0.45);
    let wall = Body::rigid_sphere(0, Tensor::new([cx, cz]), Tensor::zeros(), 1.0, radius, 1.0, true);
    let s = sph(Some(BodyCollection::new().add(wall)));
    let whole = sph(None);
    let (all, masked) = (mass_sph(&whole, NonphysicalCells::default()), mass_sph(&s, NonphysicalCells::default()));
    let inside = |i: usize, j: usize| {
        let (r, t) = sph_centroid(i, j);
        let (x, z) = (r * t.sin(), r * t.cos());
        ((x - cx).powi(2) + (z - cz).powi(2)).sqrt() < radius
    };
    let want = kept_fraction(sph_cells().map(|(i, j)| (sph_volume(i, j), !inside(i, j))));
    let chart = kept_fraction(sph_cells().map(|(i, j)| {
        let (rl, rh, tl, th) = sph_faces(i, j);
        let (r, t) = (0.5 * (rl + rh), 0.5 * (tl + th));
        (sph_volume(i, j), ((r - cx).powi(2) + (t - cz).powi(2)).sqrt() >= radius)
    }));
    assert!(want < 1.0 - 1e-3, "the body masks no cell");
    assert!((want - chart).abs() > 1e-3, "the chart-coordinate mask keeps the same cells; the gate is vacuous");
    let got = masked / all;
    assert!((got - want).abs() < 1e-12, "the census keeps {got} of the volume, the body's exterior {want}");
}

#[test]
fn a_body_on_the_cylindrical_rz_plane_masks_its_cartesian_interior() {
    const NZ: usize = 12;
    let (z_lo, dz) = (-1.5, 0.25);
    // the (R, z) plane is the x-z plane; the body sits at (x, z) = (2.0, 0.2)
    // the radius puts one cell between its radial midpoint and its centroid relative to the surface
    let (cx, cz, radius) = (2.0, 0.2, 0.321);
    let build = |bodies: Option<BodyCollection<f64, 2>>| {
        let s = CylSim::build(Newtonian, IdealGas { gamma: 1.4 }, Cylindrical)
            .cells([NR, NZ])
            .origin([R_LO, z_lo])
            .spacing([DR, dz])
            .boundaries(Boundaries::uniform(BoundaryType::Outflow))
            .allocate()
            .expect("cylindrical sim")
            .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0]), Pressure(1.0)))
            .build();
        let s = match bodies {
            Some(b) => s.with_bodies(b),
            None => s,
        };
        s.substrate().c2p(&s);
        s
    };
    let wall = Body::rigid_sphere(0, Tensor::new([cx, cz]), Tensor::zeros(), 1.0, radius, 1.0, true);
    let (s, whole) = (build(Some(BodyCollection::new().add(wall))), build(None));
    let (all, masked) = (mass_cyl(&whole, NonphysicalCells::default()), mass_cyl(&s, NonphysicalCells::default()));
    let cells = || (0..NR).flat_map(|i| (0..NZ).map(move |k| (i, k)));
    let volume = |i: usize| {
        let (rl, rh) = (R_LO + i as f64 * DR, R_LO + (i + 1) as f64 * DR);
        (rh * rh - rl * rl) / 2.0 * dz
    };
    let inside = |i: usize, k: usize| {
        let (rl, rh) = (R_LO + i as f64 * DR, R_LO + (i + 1) as f64 * DR);
        let r = 2.0 / 3.0 * (rh.powi(3) - rl.powi(3)) / (rh * rh - rl * rl);
        let z = z_lo + (k as f64 + 0.5) * dz;
        ((r - cx).powi(2) + (z - cz).powi(2)).sqrt() < radius
    };
    let want = kept_fraction(cells().map(|(i, k)| (volume(i), !inside(i, k))));
    // the chart reading: the radial midpoint in place of the centroid
    let chart = kept_fraction(cells().map(|(i, k)| {
        let r = R_LO + (i as f64 + 0.5) * DR;
        let z = z_lo + (k as f64 + 0.5) * dz;
        (volume(i), ((r - cx).powi(2) + (z - cz).powi(2)).sqrt() >= radius)
    }));
    assert!(want < 1.0 - 1e-3, "the body masks no cell");
    assert!((want - chart).abs() > 1e-4, "the midpoint reading keeps the same cells; the gate is vacuous");
    let got = masked / all;
    assert!((got - want).abs() < 1e-12, "the census keeps {got} of the volume, the body's exterior {want}");
}
