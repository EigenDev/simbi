// =============================================================================
// magnetized_body_graded_axis.rs
//
// a resistive immersed body adds the edge EMF eta chi J, so inside the body
// (chi = 1) the field diffuses as dB/dt = eta laplacian(B). with B_y = 1 + s x^2
// and no B_x, one euler step from rest changes the field by the ideal scheme's
// own update plus the resistive part; the ideal part does not depend on eta (it
// carries the riemann solver's dissipation of B_y at the limiter-clipped minimum on
// x = 0, which halves with the cell width), so the difference of the steps at eta
// and at eta = 0 is the resistive part alone. it is compared on every interior face
// with the discrete masked update written out here, which is 2 eta s dt where the
// mask is one. the body operators take the current from
// the uniform cell width, as the cartesian constrained-transport curl does
// (admission refuses a magnetized run on a graded cartesian axis), so the
// operators also refuse a graded axis by name for a caller of the library; on a
// uniform axis the resistive part of the interior update is pinned against its
// discrete expectation.
//
// run: cargo test -p symbi --test magnetized_body_graded_axis
// =============================================================================

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{AxisMap, Cartesian};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, MagneticSpec, SurfaceSpec};
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;
type Kern = NewtonianMhdSubstrateKernelSet<HostMemory, f64, 2>;

const N: usize = 64;
const RATIO: f64 = 1.05;
const S: f64 = 0.5;
const ETA: f64 = 0.1;
const R_BODY: f64 = 0.45;
const DT: f64 = 1e-5;

/// the B_y update of every face after one euler step of a sink of resistivity `eta` at the
/// origin on [-1, 1]^2, the x axis geometric when `graded`
fn face_update(graded: bool, eta: f64) -> (std::collections::HashMap<[isize; 2], f64>, std::collections::HashMap<[isize; 2], f64>) {
    let dx = 2.0 / N as f64;
    let w0 = 2.0 * (RATIO - 1.0) / (RATIO.powi(N as i32) - 1.0);
    let maps = graded.then(|| [AxisMap::Geometric { start: -1.0, width: w0, ratio: RATIO }, AxisMap::Uniform { start: -1.0, dx }]);
    let field = |x: f64| 1.0 + S * x * x;
    let s = Sim::build(NewtonianMhd, IdealGas { gamma: 5.0 / 3.0 }, Cartesian)
        .cells([N, N])
        .origin([-1.0, -1.0])
        .spacing([dx, dx])
        .coord_maps(maps)
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim")
        .set_initial(|x| {
            MhdPrim::new(Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)), Tensor::new([0.0, field(x[0]), 0.0]))
        })
        .seed_faces(|d, x| if d == 1 { field(x[0]) } else { 0.0 })
        .build();
    let sink = Body::black_hole(0, Tensor::new([0.0, 0.0]), Tensor::zeros(), 1e-6, R_BODY, 0.05, 0.0, 1.0, R_BODY)
        .with_surface(SurfaceSpec::Drain)
        .with_magnetic(MagneticSpec::Resistive { eta });
    let mut s = s.with_bodies(BodyCollection::new().add(sink));
    s.timestepping = Timestepping::Euler;
    let kernels = Kern::new(5.0 / 3.0, 0.4, 1.0, &s.geom.allocated);
    kernels.c2p(&s);
    kernels.ghost_fill(&s);
    let faces = &s.fields.mhd.as_ref().expect("mhd").bface[1];
    let before: std::collections::HashMap<[isize; 2], f64> = faces.domain().clone().iter().map(|c| (c, *faces.view().at(c))).collect();
    evolve_with_callback(&mut s, &kernels, DT, 1, |_| {}).expect("one step");
    let after = &s.fields.mhd.as_ref().expect("mhd").bface[1];
    let lo = [s.geom.interior.spaces[0].lo, s.geom.interior.spaces[1].lo];
    let initial = before.iter().map(|(c, b)| ([c[0] - lo[0], c[1] - lo[1]], *b)).collect();
    let update = before.iter().map(|(c, b)| ([c[0] - lo[0], c[1] - lo[1]], *after.view().at(*c) - b)).collect();
    (initial, update)
}

/// the largest departure of the resistive part of the interior B_y update from its discrete
/// expectation, relative to 2 eta s dt, with the number of faces checked. the resistive EMF at a
/// cell corner is E_z = eta chi J_z: J_z the difference of the two B_y faces beside the corner over
/// the cell width (B_x = 0), chi = (1 - tanh(phi / w)) / 2 the body mask at the corner, phi the
/// corner's distance from the body center less the body radius, w the least cell width. a y-face
/// changes by dt (E_z right - E_z left) / dx
fn resistive_departure(graded: bool) -> (f64, usize) {
    let (initial, with) = face_update(graded, ETA);
    let (_, without) = face_update(graded, 0.0);
    let dx = 2.0 / N as f64;
    let chi = |x: f64, y: f64| 0.5 * (1.0 - (((x * x + y * y).sqrt() - R_BODY) / dx).tanh());
    // the corner at the lower x face of cell i on the face row j
    let emf = |i: isize, j: isize| {
        let (x, y) = (-1.0 + i as f64 * dx, -1.0 + j as f64 * dx);
        ETA * chi(x, y) * (initial[&[i, j]] - initial[&[i - 1, j]]) / dx
    };
    let scale = 2.0 * ETA * S * DT;
    let (mut worst, mut checked) = (0.0f64, 0usize);
    for i in 2..(N as isize - 2) {
        for j in 2..(N as isize - 1) {
            let (x, y) = (-1.0 + (i as f64 + 0.5) * dx, -1.0 + j as f64 * dx);
            if (x * x + y * y).sqrt() > R_BODY - 6.0 * dx {
                continue;
            }
            let want = DT * (emf(i + 1, j) - emf(i, j)) / dx;
            let resistive = with[&[i, j]] - without[&[i, j]];
            worst = worst.max((resistive - want).abs() / scale);
            checked += 1;
        }
    }
    (worst, checked)
}

#[test]
fn a_resistive_body_diffuses_the_interior_field_on_a_uniform_axis() {
    let (worst, checked) = resistive_departure(false);
    assert!(checked >= 50, "{checked} interior faces; the gate is vacuous");
    // the update is the difference of order-one face values changed by about 1e-6, so round-off
    // enters at about 1e-10 of 2 eta s dt (measured 1.7e-10)
    assert!(worst < 1e-9, "the resistive part of the interior field update departs from its discrete expectation by {worst:.3e} of 2 eta s dt");
}

#[test]
#[should_panic(expected = "runs on uniformly spaced axes")]
fn a_resistive_body_on_a_graded_axis_is_refused() {
    let _ = face_update(true, ETA);
}
