// =============================================================================
// viscous_alpha_iso_closure.rs
//
// the shakura-sunyaev alpha viscosity of a locally isothermal gas,
// nu(x) = alpha cs^2(x) / Omega_K(r) about immersed body 0, evaluates nu at each
// stencil cell from that cell's own closure value cs^2. the stress on a face is
// then one number shared by the two cells it separates. gates:
// - on a cartesian box (2D and 3D) with cs^2 varying across it and a sheared
//   flow compactly supported inside it, every face stress enters its two cells
//   with opposite signs and the boundary faces carry none, so one viscous
//   application leaves the total momentum unchanged to the rounding of the
//   stored momenta; a premise asserts the operator changed the momentum. the
//   box is bounded: Omega_K(r) is evaluated at the stencil cell's position, so
//   across a periodic seam nu takes two values and the continuum itself leaks;
// - on the cartesian and the cylindrical (R, phi) charts, raising cs^2 in one
//   cell changes the update of its face neighbor and leaves a cell outside the
//   3-point stencil bitwise unchanged.
//
// run: cargo test -p symbi --test viscous_alpha_iso_closure
// =============================================================================

use std::f64::consts::PI;
use symbi::regimes::substrate::IsoSubstrateKernelSet;
use symbi::sim::state::*;
use symbi::sim::substrate_seam::{KernelSet, WithViscosity};
use symbi_algebra::Tensor;
use symbi_geometry::{Cartesian, CylindricalRPhi};
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::Isothermal;
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::quantity::Density;
use symbi_hydro::state::PrimG;
use symbi_ib::{Body, BodyCollection};
use symbi_xpu::{CpuSpace, HostMemory};

const CS0: f64 = 0.3;
const ALPHA: f64 = 0.1;
const DT: f64 = 1.0e-4;
const L: f64 = 1.0;

type Sim2 = SimState<IsoNewtonian, 2, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
type Sim3 = SimState<IsoNewtonian, 3, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
type SimRPhi = SimState<IsoNewtonian, 2, CylindricalRPhi, Isothermal<f64>, CpuSpace, HostMemory>;

/// the closure cs^2 = cs0^2 (1 + 0.5 sin(pi x + 0.3) cos(pi y - 0.7) cos(pi z + 1.1)), varying on
/// every axis. the phases break the parity of the box: with the flow's pure sines, a closure of
/// definite parity lets a center-only cs^2 cancel in the momentum sum.
fn cs2_at(x: &[f64]) -> f64 {
    let z = x.get(2).map_or(1.0, |z| (PI * z + 1.1).cos());
    CS0 * CS0 * (1.0 + 0.5 * (PI * x[0] + 0.3).sin() * (PI * x[1] - 0.7).cos() * z)
}

fn body<const D: usize>() -> BodyCollection<f64, D> {
    BodyCollection::new().add(Body::gravitational(0, Tensor::zeros(), Tensor::zeros(), 1.0, 0.05, 0.1))
}

/// writes `cs2(center)` into the closure on the interior and continues it into the ghosts
fn set_closure<const D: usize, G>(
    sim: &SimState<IsoNewtonian, D, G, Isothermal<f64>, CpuSpace, HostMemory>,
    cs2: impl Fn([isize; D]) -> f64,
) where
    G: symbi_geometry::Metric<f64, D>,
{
    let field = sim.fields.cs2.as_ref().expect("the isothermal closure field");
    for c in sim.geom.interior.iter() {
        field.set(c, cs2(c));
    }
    sim.extend_isothermal_cs2_into_ghosts(&sim.geom.interior);
}

fn momenta<const D: usize, G>(sim: &SimState<IsoNewtonian, D, G, Isothermal<f64>, CpuSpace, HostMemory>) -> Vec<Vec<f64>>
where
    G: symbi_geometry::Metric<f64, D>,
{
    (0..D).map(|d| sim.geom.interior.iter().map(|c| *sim.fields.cons.mom[d].view().at(c)).collect()).collect()
}

/// every momentum component sums to its value before the pass, to the rounding of the stored momenta
fn assert_conserved(label: &str, before: &[Vec<f64>], after: &[Vec<f64>]) {
    let mut acted = 0.0f64;
    for (b, a) in before.iter().zip(after) {
        let drift: f64 = a.iter().zip(b).map(|(x, y)| x - y).sum();
        let largest = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        acted = acted.max(largest);
        // each cell's m + dm rounds by eps/2 |m|, and each difference m_after - m_before and each
        // partial sum rounds again: the drift is bounded by a few eps sum |m|; 8 covers those three
        // roundings with margin
        let scale: f64 = a.iter().map(|x| x.abs()).sum();
        assert!(
            drift.abs() <= 8.0 * f64::EPSILON * scale,
            "{label}: the total momentum changed by {drift:.3e} against a largest cell change of \
             {largest:.3e} (sum |m| = {scale:.3e})"
        );
    }
    assert!(acted > 1e-10, "{label}: the alpha viscous pass never changed the momentum; the gate is vacuous");
}

/// the compact weight (1 - s^2)^2 for s = |x - x0| / 0.6 < 1, zero outside; x0 is off the body
/// and off the box center, so no parity of the box cancels the stress
fn bump(x: &[f64]) -> f64 {
    let x0 = [0.2, -0.1, 0.15];
    let s2: f64 = x.iter().zip(x0).map(|(xi, ci)| (xi - ci) * (xi - ci)).sum::<f64>() / 0.36;
    if s2 < 1.0 { (1.0 - s2) * (1.0 - s2) } else { 0.0 }
}

fn boxed_2d() -> (Sim2, f64) {
    let n = 24;
    let dx = 2.0 * L / n as f64;
    let sim = Sim2::build(IsoNewtonian, Isothermal { cs: CS0 }, Cartesian)
        .cells([n, n])
        .origin([-L, -L])
        .spacing([dx; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(0.3)
        .allocate()
        .expect("sim")
        .set_initial(|[x, y]| {
            let w = bump(&[x, y]);
            PrimG::<f64, 2, IsoModel>::isothermal(
                Density(1.0 + 0.2 * (PI * x).cos()),
                Tensor::new([0.3 * w * (PI * y).sin(), -0.2 * w * (PI * x).sin()]),
            )
        })
        .build()
        .with_bodies(body::<2>());
    (sim, dx)
}

fn boxed_3d() -> (Sim3, f64) {
    let n = 12;
    let dx = 2.0 * L / n as f64;
    let sim = Sim3::build(IsoNewtonian, Isothermal { cs: CS0 }, Cartesian)
        .cells([n, n, n])
        .origin([-L, -L, -L])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(0.3)
        .allocate()
        .expect("sim")
        .set_initial(|[x, y, z]| {
            let w = bump(&[x, y, z]);
            PrimG::<f64, 3, IsoModel>::isothermal(
                Density(1.0 + 0.2 * (PI * x).cos()),
                Tensor::new([0.3 * w * (PI * y).sin(), -0.2 * w * (PI * z).sin(), 0.25 * w * (PI * x).sin()]),
            )
        })
        .build()
        .with_bodies(body::<3>());
    (sim, dx)
}

fn periodic_2d() -> (Sim2, f64) {
    let n = 24;
    let dx = 2.0 * L / n as f64;
    let sim = Sim2::build(IsoNewtonian, Isothermal { cs: CS0 }, Cartesian)
        .cells([n, n])
        .origin([-L, -L])
        .spacing([dx; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .allocate()
        .expect("sim")
        .set_initial(|[x, y]| {
            PrimG::<f64, 2, IsoModel>::isothermal(
                Density(1.0 + 0.2 * (PI * x).cos()),
                Tensor::new([0.3 * (PI * y).sin(), -0.2 * (PI * x).sin()]),
            )
        })
        .build()
        .with_bodies(body::<2>());
    (sim, dx)
}

fn periodic_3d() -> (Sim3, f64) {
    let n = 12;
    let dx = 2.0 * L / n as f64;
    let sim = Sim3::build(IsoNewtonian, Isothermal { cs: CS0 }, Cartesian)
        .cells([n, n, n])
        .origin([-L, -L, -L])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .allocate()
        .expect("sim")
        .set_initial(|[x, y, z]| {
            PrimG::<f64, 3, IsoModel>::isothermal(
                Density(1.0 + 0.2 * (PI * x).cos()),
                Tensor::new([0.3 * (PI * y).sin(), -0.2 * (PI * z).sin(), 0.25 * (PI * x).sin()]),
            )
        })
        .build()
        .with_bodies(body::<3>());
    (sim, dx)
}

fn center<const D: usize>(c: [isize; D], dx: f64) -> [f64; D] {
    std::array::from_fn(|ax| -L + (c[ax] as f64 + 0.5) * dx)
}

#[test]
fn locally_isothermal_alpha_viscosity_conserves_momentum_in_2d() {
    let (sim, dx) = boxed_2d();
    set_closure(&sim, |c| cs2_at(&center(c, dx)));
    let k = IsoSubstrateKernelSet::<HostMemory, f64, 2>::new(CS0, 0.3, &sim.geom.allocated).with_alpha(ALPHA);
    k.c2p(&sim);
    k.ghost_fill(&sim);
    let before = momenta(&sim);
    k.viscous(&sim, DT);
    assert_conserved("2D", &before, &momenta(&sim));
}

#[test]
fn locally_isothermal_alpha_viscosity_conserves_momentum_in_3d() {
    let (sim, dx) = boxed_3d();
    set_closure(&sim, |c| cs2_at(&center(c, dx)));
    let k = IsoSubstrateKernelSet::<HostMemory, f64, 3>::new(CS0, 0.3, &sim.geom.allocated).with_alpha(ALPHA);
    k.c2p(&sim);
    k.ghost_fill(&sim);
    let before = momenta(&sim);
    k.viscous(&sim, DT);
    assert_conserved("3D", &before, &momenta(&sim));
}

/// the alpha update at `neighbor` responds to raising cs^2 at `raised`, and the update at `far`
/// stays bitwise: returns (response at the neighbor, largest change at the far cell)
fn closure_response<const D: usize, G>(
    build: impl Fn() -> SimState<IsoNewtonian, D, G, Isothermal<f64>, CpuSpace, HostMemory>,
    raised: [isize; D],
    neighbor: [isize; D],
    far: [isize; D],
) -> (f64, f64)
where
    G: symbi_geometry::Metric<f64, D>,
    IsoSubstrateKernelSet<HostMemory, f64, D>: KernelSet<D, D, HostMemory, f64>,
{
    let update = |raise: bool| -> Vec<[f64; D]> {
        let sim = build();
        set_closure(&sim, |c| if raise && c == raised { 2.0 * CS0 * CS0 } else { CS0 * CS0 });
        let k = IsoSubstrateKernelSet::<HostMemory, f64, D>::new(CS0, 0.3, &sim.geom.allocated).with_alpha(ALPHA);
        k.c2p(&sim);
        k.ghost_fill(&sim);
        let at = |c: [isize; D]| -> [f64; D] { std::array::from_fn(|d| *sim.fields.cons.mom[d].view().at(c)) };
        let before = [at(neighbor), at(far)];
        k.viscous(&sim, DT);
        let after = [at(neighbor), at(far)];
        (0..2).map(|ii| std::array::from_fn(|d| after[ii][d] - before[ii][d])).collect()
    };
    let (plain, raised_run) = (update(false), update(true));
    let response = (0..D).map(|d| (raised_run[0][d] - plain[0][d]).abs()).fold(0.0, f64::max);
    let far_change = (0..D).map(|d| (raised_run[1][d] - plain[1][d]).abs()).fold(0.0, f64::max);
    (response, far_change)
}

#[test]
fn a_raised_closure_reaches_the_face_neighbor_on_the_cartesian_chart() {
    let (response, far) = closure_response(|| periodic_2d().0, [12, 12], [13, 12], [16, 12]);
    assert!(response > 1e-12, "the neighbor's alpha update ignores the raised cs^2 next to it ({response:.3e})");
    assert_eq!(far, 0.0, "a cell outside the stencil felt the raised cs^2");
}

#[test]
fn a_raised_closure_reaches_the_face_neighbor_on_the_cylindrical_chart() {
    let (nr, np) = (24usize, 24usize);
    let (r_lo, r_hi) = (1.0, 3.0);
    let (dr, dp) = ((r_hi - r_lo) / nr as f64, 2.0 * PI / np as f64);
    let build = || -> SimRPhi {
        SimRPhi::build(IsoNewtonian, Isothermal { cs: CS0 }, CylindricalRPhi)
            .cells([nr, np])
            .origin([r_lo, 0.0])
            .spacing([dr, dp])
            .boundaries(Boundaries(std::array::from_fn(|a| {
                if a == 1 { [BoundaryType::Periodic; 2] } else { [BoundaryType::Outflow; 2] }
            })))
            .cfl(0.3)
            .allocate()
            .expect("sim")
            // a sheared, phi-dependent flow so both the radial and the azimuthal faces carry stress
            .set_initial(|[r, phi]| {
                PrimG::<f64, 2, IsoModel>::isothermal(
                    Density(1.0),
                    Tensor::new([0.05 * phi.sin(), 0.5 / r + 0.1 * phi.cos()]),
                )
            })
            .build()
            .with_bodies(body::<2>())
    };
    for (neighbor, label) in [([13, 12], "radial"), ([12, 13], "azimuthal")] {
        let (response, far) = closure_response(build, [12, 12], neighbor, [16, 16]);
        assert!(
            response > 1e-12,
            "{label} neighbor: the alpha update ignores the raised cs^2 next to it ({response:.3e})"
        );
        assert_eq!(far, 0.0, "a cell outside the stencil felt the raised cs^2");
    }
}

#[test]
fn a_raised_closure_reaches_the_face_neighbor_in_3d() {
    for (neighbor, label) in [([7, 6, 6], "x"), ([6, 7, 6], "y"), ([6, 6, 7], "z")] {
        let (response, far) = closure_response(|| periodic_3d().0, [6, 6, 6], neighbor, [9, 9, 9]);
        assert!(response > 1e-12, "{label} neighbor: the alpha update ignores the raised cs^2 next to it ({response:.3e})");
        assert_eq!(far, 0.0, "a cell outside the stencil felt the raised cs^2");
    }
}
