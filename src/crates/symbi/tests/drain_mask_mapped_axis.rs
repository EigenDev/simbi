// =============================================================================
// drain_mask_mapped_axis.rs
//
// a draining sink on a spherical (r, theta) grid removes mass from the cells
// whose volume centroid, embedded in cartesian space (the (r, theta) plane is
// the x-z plane), lies inside its accretion radius; beyond the mask's tanh
// transition the loss falls to the tail of the profile. one forward-euler step from gas at rest, with the post-step drain,
// isolates the drain: the density flux vanishes and gravity moves momentum only. the radial axis is
// log-spaced, where the face map reads the decade slope; a uniform radial axis
// is the control. a second gate runs the same drain on a 2D cartesian grid with
// a log-spaced x axis, where the dispatch box follows the mapped cells. the
// expected geometry is computed here from the face formulas.
//
// run: cargo test -p symbi --test drain_mask_mapped_axis
// =============================================================================

use std::f64::consts::PI;
use symbi::prelude::SimSubstrate;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{AxisMap, Cartesian, Spherical};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, SurfaceSpec};
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

type SphSim = SimState<Newtonian, 2, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;

const NR: usize = 64;
const NT: usize = 96;
const R_LO: f64 = 1.0;
const R_HI: f64 = 10.0;
const T_LO: f64 = 0.05;
const T_HI: f64 = PI - 0.05;
// the sink sits at cartesian x = z = 3 (r = 4.24, theta = pi/4). its position components are
// equal, so the drain's chart embedding (r cos theta, r sin theta) and the meridional (x, z) reading
// of a 2D spherical body position place it at the same point
const BODY_X: f64 = 3.0;
const BODY_Z: f64 = 3.0;
const R_ACC: f64 = 1.2;

fn radial_face(log: bool, i: usize) -> f64 {
    if log {
        R_LO * (R_HI / R_LO).powf(i as f64 / NR as f64)
    } else {
        R_LO + i as f64 * (R_HI - R_LO) / NR as f64
    }
}

/// the cartesian (x, z) of cell (i, j)'s volume centroid, from its first moments
fn centroid_xz(log: bool, i: usize, j: usize) -> (f64, f64) {
    let (rl, rh) = (radial_face(log, i), radial_face(log, i + 1));
    let dt = (T_HI - T_LO) / NT as f64;
    let (tl, th) = (T_LO + j as f64 * dt, T_LO + (j + 1) as f64 * dt);
    let r = 0.75 * (rh.powi(4) - rl.powi(4)) / (rh.powi(3) - rl.powi(3));
    let t = ((th.sin() - th * th.cos()) - (tl.sin() - tl * tl.cos())) / (tl.cos() - th.cos());
    (r * t.sin(), r * t.cos())
}

/// the cells' relative density loss over one drain step, with each cell's distance from the sink
/// center and its largest physical width
fn drain_losses(log: bool) -> Vec<(f64, f64, f64)> {
    let dt_theta = (T_HI - T_LO) / NT as f64;
    let maps = log.then(|| {
        [AxisMap::Log { start: R_LO, log_slope: (R_HI / R_LO).log10() / NR as f64 }, AxisMap::Uniform { start: T_LO, dx: dt_theta }]
    });
    let mut s = SphSim::build(Newtonian, IdealGas { gamma: 5.0 / 3.0 }, Spherical)
        .cells([NR, NT])
        .origin([R_LO, T_LO])
        .spacing([(R_HI - R_LO) / NR as f64, dt_theta])
        .coord_maps(maps)
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("spherical sim")
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0]), Pressure(1.0)))
        .build();
    s.timestepping = Timestepping::Euler;
    let sink = Body::black_hole(0, Tensor::new([BODY_X, BODY_Z]), Tensor::zeros(), 1.0, R_ACC, 0.1, 0.0, 1.0, R_ACC)
        .with_surface(SurfaceSpec::Drain);
    let mut s = s.with_bodies(BodyCollection::new().add(sink));
    let kernels = s.substrate();
    kernels.c2p(&s);
    kernels.ghost_fill(&s);
    let before: Vec<f64> = s.geom.interior.iter().map(|c| *s.fields.cons.den.view().at(c)).collect();
    evolve_with_callback(&mut s, &kernels, 1e-3, 1, |_| {}).expect("one drain step");
    assert_eq!(s.time, 1e-3, "the step stopped short of its end time");
    let lo = [s.geom.interior.spaces[0].lo, s.geom.interior.spaces[1].lo];
    s.geom
        .interior
        .iter()
        .zip(before)
        .map(|(c, b)| {
            let (i, j) = ((c[0] - lo[0]) as usize, (c[1] - lo[1]) as usize);
            let (x, z) = centroid_xz(log, i, j);
            let d = ((x - BODY_X).powi(2) + (z - BODY_Z).powi(2)).sqrt();
            let r_mid = 0.5 * (radial_face(log, i) + radial_face(log, i + 1));
            let width = (radial_face(log, i + 1) - radial_face(log, i)).max(r_mid * dt_theta);
            let loss = (b - *s.fields.cons.den.view().at(c)) / b;
            if loss.abs() > 1e-6 && std::env::var("E14_DUMP").is_ok() {
                let (rr, tt) = ((x * x + z * z).sqrt(), x.atan2(z));
                eprintln!("drained cell ({i},{j}) r {rr:.3} theta {tt:.3} x {x:.3} z {z:.3} d {d:.3} loss {loss:.3e}");
            }
            (loss, d, width)
        })
        .collect()
}

#[test]
fn a_drain_acts_inside_its_accretion_radius_on_a_log_radial_axis() {
    for (label, log) in [("uniform", false), ("log", true)] {
        let cells = drain_losses(log);
        // cells well inside drain; cells beyond the mask's transition stay untouched
        let inside: Vec<_> = cells.iter().filter(|(_, d, w)| *d < R_ACC - 2.0 * w).collect();
        let outside: Vec<_> = cells.iter().filter(|(_, d, w)| *d > R_ACC + 2.0 * w).collect();
        assert!(inside.len() >= 4, "{label}: {} cells lie well inside the sink; the gate is vacuous", inside.len());
        let least_inside = inside.iter().map(|(l, _, _)| *l).fold(f64::INFINITY, f64::min);
        let most_outside = outside.iter().map(|(l, _, _)| l.abs()).fold(0.0f64, f64::max);
        eprintln!("{label}: {} cells inside, least loss {least_inside:.3e}; {} outside, largest |loss| {most_outside:.3e}", inside.len(), outside.len());
        assert!(least_inside > 0.0, "{label}: a cell well inside the sink lost no mass ({least_inside:.3e})");
        // the mask is chi = (1 - tanh(phi / w)) / 2 with w the least cell width, so two of the
        // largest widths beyond the accretion radius chi <= (1 - tanh 2) / 2 = 0.018, against chi
        // near one inside
        assert!(
            most_outside < 0.02 * least_inside,
            "{label}: a cell well outside the sink lost {most_outside:.3e}, against {least_inside:.3e} inside"
        );
    }
}

type CartSim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

const NX: usize = 96;
const NY: usize = 64;
const Y_HALF: f64 = 3.0;
const SINK_X: f64 = 4.0;

/// the cells' relative density loss over one drain step on a 2D cartesian grid with an x axis on
/// [R_LO, R_HI], with each cell's distance from a sink at (SINK_X, 0) and its largest width
fn cartesian_drain_losses(log: bool) -> Vec<(f64, f64, f64)> {
    let dy = 2.0 * Y_HALF / NY as f64;
    let xface = |i: usize| if log { R_LO * (R_HI / R_LO).powf(i as f64 / NX as f64) } else { R_LO + i as f64 * (R_HI - R_LO) / NX as f64 };
    let maps = log.then(|| {
        [AxisMap::Log { start: R_LO, log_slope: (R_HI / R_LO).log10() / NX as f64 }, AxisMap::Uniform { start: -Y_HALF, dx: dy }]
    });
    let mut s = CartSim::build(Newtonian, IdealGas { gamma: 5.0 / 3.0 }, Cartesian)
        .cells([NX, NY])
        .origin([R_LO, -Y_HALF])
        .spacing([(R_HI - R_LO) / NX as f64, dy])
        .coord_maps(maps)
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("cartesian sim")
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0]), Pressure(1.0)))
        .build();
    s.timestepping = Timestepping::Euler;
    let sink = Body::black_hole(0, Tensor::new([SINK_X, 0.0]), Tensor::zeros(), 1.0, R_ACC, 0.1, 0.0, 1.0, R_ACC)
        .with_surface(SurfaceSpec::Drain);
    let mut s = s.with_bodies(BodyCollection::new().add(sink));
    let kernels = s.substrate();
    kernels.c2p(&s);
    kernels.ghost_fill(&s);
    let before: Vec<f64> = s.geom.interior.iter().map(|c| *s.fields.cons.den.view().at(c)).collect();
    evolve_with_callback(&mut s, &kernels, 1e-3, 1, |_| {}).expect("one drain step");
    assert_eq!(s.time, 1e-3, "the step stopped short of its end time");
    let lo = [s.geom.interior.spaces[0].lo, s.geom.interior.spaces[1].lo];
    s.geom
        .interior
        .iter()
        .zip(before)
        .map(|(c, b)| {
            let (i, j) = ((c[0] - lo[0]) as usize, (c[1] - lo[1]) as usize);
            let x = 0.5 * (xface(i) + xface(i + 1));
            let y = -Y_HALF + (j as f64 + 0.5) * dy;
            let d = ((x - SINK_X).powi(2) + y * y).sqrt();
            let width = (xface(i + 1) - xface(i)).max(dy);
            ((b - *s.fields.cons.den.view().at(c)) / b, d, width)
        })
        .collect()
}

#[test]
fn a_drain_acts_inside_its_accretion_radius_on_a_log_cartesian_axis() {
    for (label, log) in [("uniform", false), ("log", true)] {
        let cells = cartesian_drain_losses(log);
        let inside: Vec<_> = cells.iter().filter(|(_, d, w)| *d < R_ACC - 2.0 * w).collect();
        let outside: Vec<_> = cells.iter().filter(|(_, d, w)| *d > R_ACC + 2.0 * w).collect();
        assert!(inside.len() >= 4, "{label}: {} cells lie well inside the sink; the gate is vacuous", inside.len());
        let least_inside = inside.iter().map(|(l, _, _)| *l).fold(f64::INFINITY, f64::min);
        let most_outside = outside.iter().map(|(l, _, _)| l.abs()).fold(0.0f64, f64::max);
        eprintln!("cartesian {label}: {} cells inside, least loss {least_inside:.3e}; {} outside, largest |loss| {most_outside:.3e}", inside.len(), outside.len());
        assert!(least_inside > 0.0, "cartesian {label}: a cell well inside the sink lost no mass ({least_inside:.3e})");
        // the tanh tail bound of the spherical gate
        assert!(
            most_outside < 0.02 * least_inside,
            "cartesian {label}: a cell well outside the sink lost {most_outside:.3e}, against {least_inside:.3e} inside"
        );
    }
}
