// =============================================================================
// body_meridional_frame.rs
//
// on the 2D spherical (r, theta) chart the grid plane is the meridional x-z
// plane: a body position (a, b) and a body velocity (u, w) are the cartesian
// pairs (x, z) = (a, b) and (v_x, v_z) = (u, w), with x = r sin(theta) and
// z = r cos(theta). the body sits at (x, z) = (4.5, 1.5), off the symmetry line
// x = z, so the swapped reading (z, x) would place it at the mirrored point
// (x, z) = (1.5, 4.5). judged against the cell centroids written out here from
// their first moments:
//   - a drain removes mass from the cells inside the body and leaves the cells
//     around the mirrored point at the tail of its mask;
//   - a rigid wall drives the gas inside it toward the body velocity, the
//     cartesian (v_x, v_z) seen in each cell's (r, theta) frame;
//   - a census leaves out exactly the cells whose centroid lies inside the body;
//   - a sealed wall that stops gas rotating in the x-z plane books the lost
//     angular momentum about y, L_y = sum dV [(z - z_b) m_x - (x - x_b) m_z],
//     on the body's y torque.
// one forward-euler step from uniform gas at rest isolates the penalization.
//
// run: cargo test -p symbi --test body_meridional_frame
// =============================================================================

use std::f64::consts::PI;
use symbi::prelude::SimSubstrate;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Spherical;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, SurfaceSpec};
use symbi_sim::census::RegisteredCensus;
use symbi_sim::substrate_seam::KernelSet;
use symbi_source_compile::CensusConfig;
use symbi_substrate::census_sample::{NonphysicalCells, level_partials};
use symbi_xpu::{CpuSpace, HostMemory};

type SphSim = SimState<Newtonian, 2, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;

const NR: usize = 64;
const NT: usize = 96;
const R_LO: f64 = 1.0;
const R_HI: f64 = 10.0;
const T_LO: f64 = 0.05;
const T_HI: f64 = PI - 0.05;
const DR: f64 = (R_HI - R_LO) / NR as f64;
const DT: f64 = (T_HI - T_LO) / NT as f64;
const BODY: (f64, f64) = (4.5, 1.5);
const MIRROR: (f64, f64) = (1.5, 4.5);
const RADIUS: f64 = 1.2;

/// the cell's radial and polar faces
fn faces(i: usize, j: usize) -> (f64, f64, f64, f64) {
    (R_LO + i as f64 * DR, R_LO + (i + 1) as f64 * DR, T_LO + j as f64 * DT, T_LO + (j + 1) as f64 * DT)
}

/// the cell's volume centroid (r, theta), from its first moments
fn centroid(i: usize, j: usize) -> (f64, f64) {
    let (rl, rh, tl, th) = faces(i, j);
    let r = 0.75 * (rh.powi(4) - rl.powi(4)) / (rh.powi(3) - rl.powi(3));
    let t = ((th.sin() - th * th.cos()) - (tl.sin() - tl * tl.cos())) / (tl.cos() - th.cos());
    (r, t)
}

/// the distance of the cell centroid's meridional (x, z) from `p`, and the cell's largest width
fn distance_and_width(i: usize, j: usize, p: (f64, f64)) -> (f64, f64) {
    let (r, t) = centroid(i, j);
    let (x, z) = (r * t.sin(), r * t.cos());
    (((x - p.0).powi(2) + (z - p.1).powi(2)).sqrt(), DR.max((R_LO + (i + 1) as f64 * DR) * DT))
}

fn gas(bodies: Option<BodyCollection<f64, 2>>) -> SphSim {
    let mut s = SphSim::build(Newtonian, IdealGas { gamma: 5.0 / 3.0 }, Spherical)
        .cells([NR, NT])
        .origin([R_LO, T_LO])
        .spacing([DR, DT])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("spherical sim")
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0]), Pressure(1.0)))
        .build();
    s.timestepping = Timestepping::Euler;
    match bodies {
        Some(b) => s.with_bodies(b),
        None => s,
    }
}

/// one penalized step; the per-cell (relative density loss, radial momentum, polar momentum)
fn one_step(body: Body<f64, 2>) -> Vec<((usize, usize), (f64, f64, f64))> {
    let mut s = gas(Some(BodyCollection::new().add(body)));
    let kernels = s.substrate();
    kernels.c2p(&s);
    kernels.ghost_fill(&s);
    let before: Vec<f64> = s.geom.interior.iter().map(|c| *s.fields.cons.den.view().at(c)).collect();
    evolve_with_callback(&mut s, &kernels, 1e-3, 1, |_| {}).expect("one step");
    assert_eq!(s.time, 1e-3, "the step stopped short of its end time");
    let lo = [s.geom.interior.spaces[0].lo, s.geom.interior.spaces[1].lo];
    s.geom
        .interior
        .iter()
        .zip(before)
        .map(|(c, b)| {
            let ij = ((c[0] - lo[0]) as usize, (c[1] - lo[1]) as usize);
            let rho = *s.fields.cons.den.view().at(c);
            (ij, ((b - rho) / b, *s.fields.cons.mom[0].view().at(c), *s.fields.cons.mom[1].view().at(c)))
        })
        .collect()
}

#[test]
fn a_drain_acts_at_its_meridional_position() {
    let sink = Body::black_hole(0, Tensor::new([BODY.0, BODY.1]), Tensor::zeros(), 1.0, RADIUS, 0.1, 0.0, 1.0, RADIUS)
        .with_surface(SurfaceSpec::Drain);
    let cells = one_step(sink);
    let well_inside = |p: (f64, f64)| {
        move |&&((i, j), _): &&((usize, usize), (f64, f64, f64))| {
            let (d, w) = distance_and_width(i, j, p);
            d < RADIUS - 2.0 * w
        }
    };
    let inside: Vec<_> = cells.iter().filter(well_inside(BODY)).collect();
    let mirrored: Vec<_> = cells.iter().filter(well_inside(MIRROR)).collect();
    assert!(inside.len() >= 4 && mirrored.len() >= 4, "too few cells inside the body or its mirror; the gate is vacuous");
    let least_inside = inside.iter().map(|(_, (l, _, _))| *l).fold(f64::INFINITY, f64::min);
    let most_mirrored = mirrored.iter().map(|(_, (l, _, _))| l.abs()).fold(0.0f64, f64::max);
    assert!(least_inside > 0.0, "a cell inside the body lost no mass ({least_inside:.3e})");
    // the mask is chi = (1 - tanh(phi / w)) / 2 with w the least cell width; two widths outside
    // the body chi <= 0.018, and the mirrored point lies far beyond that
    assert!(
        most_mirrored < 0.02 * least_inside,
        "the drain acted at the mirrored point: loss {most_mirrored:.3e} against {least_inside:.3e} inside"
    );
}

#[test]
fn a_rigid_wall_drives_the_gas_toward_its_meridional_velocity() {
    // the wall velocity (v_x, v_z) = (0.3, -0.2); inside the wall the gas momentum, starting at rest,
    // relaxes along it, so each cell's (m_r, m_theta) is parallel to the cartesian velocity in its frame
    let (vx, vz) = (0.3, -0.2);
    // a sealed no-slip wall: equal normal and tangential stiffness relax the whole momentum along
    // the wall velocity
    let wall = Body::rigid_sphere(0, Tensor::new([BODY.0, BODY.1]), Tensor::new([vx, vz]), 1.0e12, RADIUS, 1.0, true)
        .with_surface(SurfaceSpec::Porous { porosity: 0.0, k_eta_n: 50.0, k_eta_t: 50.0 });
    let cells = one_step(wall);
    let mut checked = 0;
    for ((i, j), (_, mr, mt)) in &cells {
        let (d, w) = distance_and_width(*i, *j, BODY);
        if d >= RADIUS - 2.0 * w {
            continue;
        }
        let (_, t) = centroid(*i, *j);
        // e_r = (sin t, cos t) and e_theta = (cos t, -sin t) in (x, z)
        let (vr, vt) = (vx * t.sin() + vz * t.cos(), vx * t.cos() - vz * t.sin());
        let (m, v) = ((mr * mr + mt * mt).sqrt(), (vr * vr + vt * vt).sqrt());
        assert!(m > 0.0, "cell ({i}, {j}) inside the wall gained no momentum");
        let cross = (mr * vt - mt * vr) / (m * v);
        let dot = (mr * vr + mt * vt) / (m * v);
        assert!(cross.abs() < 1e-8 && dot > 0.0, "cell ({i}, {j}): momentum ({mr:.3e}, {mt:.3e}) against the wall velocity ({vr:.3e}, {vt:.3e})");
        checked += 1;
    }
    assert!(checked >= 4, "{checked} cells inside the wall; the gate is vacuous");
}

#[test]
fn a_census_leaves_out_the_cells_inside_the_meridional_body() {
    let census = {
        let cfg = CensusConfig::from_json(
            r#"{"name": "mass", "axes": [], "values": [2], "value_names": ["mass"], "op": "add", "params": [],
                "nodes": [{"op": "VARIABLE_RHO"}, {"op": "VARIABLE_DV"}, {"op": "MULTIPLY", "left": 0, "right": 1}]}"#,
        )
        .expect("census json");
        vec![RegisteredCensus::new(symbi_sim::census::CensusEvaluator::new(&cfg).expect("census compiles"))]
    };
    let mass = |s: &SphSim| -> f64 {
        s.substrate().c2p(s);
        level_partials(s, &census, 0, None, None, NonphysicalCells::default(), s.time)[0].0.iter().sum()
    };
    let sink = Body::black_hole(0, Tensor::new([BODY.0, BODY.1]), Tensor::zeros(), 1.0, RADIUS, 0.1, 0.0, 1.0, RADIUS)
        .with_surface(SurfaceSpec::Drain);
    let (kept, all) = (mass(&gas(Some(BodyCollection::new().add(sink)))), mass(&gas(None)));
    let volume = |i: usize, j: usize| {
        let (rl, rh, tl, th) = faces(i, j);
        (rh.powi(3) - rl.powi(3)) / 3.0 * (tl.cos() - th.cos())
    };
    let (mut outside, mut total) = (0.0, 0.0);
    for i in 0..NR {
        for j in 0..NT {
            total += volume(i, j);
            if distance_and_width(i, j, BODY).0 >= RADIUS {
                outside += volume(i, j);
            }
        }
    }
    let want = outside / total;
    assert!(want < 1.0 - 1e-3, "the body covers no cell");
    assert!((kept / all - want).abs() < 1e-12, "the census keeps {} of the volume, the body's exterior {want}", kept / all);
}

#[test]
fn a_wall_books_the_meridional_torque_about_y() {
    // gas inside a sealed static wall rotates rigidly in the x-z plane about the body center,
    // v = omega (z - z_b, -(x - x_b)), whose angular momentum about y is
    // L_y = sum rho dV [(z - z_b) v_x - (x - x_b) v_z]; stopping it books that moment on the body
    let omega = 0.2;
    let wall = Body::rigid_sphere(0, Tensor::new([BODY.0, BODY.1]), Tensor::zeros(), 1.0e12, RADIUS, 1.0, true)
        .with_surface(SurfaceSpec::Porous { porosity: 0.0, k_eta_n: 50.0, k_eta_t: 50.0 });
    let s = SphSim::build(Newtonian, IdealGas { gamma: 5.0 / 3.0 }, Spherical)
        .cells([NR, NT])
        .origin([R_LO, T_LO])
        .spacing([DR, DT])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("spherical sim")
        .set_initial(|p| {
            let (r, t) = (p[0], p[1]);
            let (x, z) = (r * t.sin(), r * t.cos());
            let (vx, vz) = (omega * (z - BODY.1), -omega * (x - BODY.0));
            // the cartesian (v_x, v_z) in the cell's (r, theta) frame
            Prim::adiabatic(Density(1.0), Tensor::new([vx * t.sin() + vz * t.cos(), vx * t.cos() - vz * t.sin()]), Pressure(1.0))
        })
        .build();
    let s = s.with_bodies(BodyCollection::new().add(wall));
    let kernels = s.substrate();
    kernels.c2p(&s);
    kernels.ghost_fill(&s);
    let lo = [s.geom.interior.spaces[0].lo, s.geom.interior.spaces[1].lo];
    let moment = |s: &SphSim| -> f64 {
        let mut l = 0.0;
        for c in s.geom.interior.iter() {
            let (i, j) = ((c[0] - lo[0]) as usize, (c[1] - lo[1]) as usize);
            let (r, t) = centroid(i, j);
            let (x, z) = (r * t.sin(), r * t.cos());
            let (mr, mt) = (*s.fields.cons.mom[0].view().at(c), *s.fields.cons.mom[1].view().at(c));
            let (mx, mz) = (mr * t.sin() + mt * t.cos(), mr * t.cos() - mt * t.sin());
            let (rl, rh, tl, th) = faces(i, j);
            let dv = 2.0 * PI * (rh.powi(3) - rl.powi(3)) / 3.0 * (tl.cos() - th.cos());
            l += dv * ((z - BODY.1) * mx - (x - BODY.0) * mz);
        }
        l
    };
    let l_before = moment(&s);
    let dt = 1e-3;
    symbi::regimes::substrate_kernels::dispatch_penalize(&s, dt, 5.0 / 3.0, 1.0);
    let l_after = moment(&s);
    let booked = s.immersed.as_ref().expect("bodies").diagnostics.consolidate()[0].torque_delta;
    let lost = l_before - l_after;
    assert!(lost > 0.0, "the wall removed no angular momentum ({lost:.3e}); the gate is vacuous");
    assert!(
        (booked[1] * dt - lost).abs() < 1e-9 * lost,
        "the booked y torque {} times dt departs from the gas's lost L_y {lost:.12e}",
        booked[1]
    );
    assert!(booked[0] == 0.0 && booked[2] == 0.0, "the meridional torque leaked off the y axis: {booked:?}");
}
