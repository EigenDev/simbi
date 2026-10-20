// =============================================================================
// gradient_wall_mapped_axis.rs
//
// a neumann wall prescribes each ghost primitive as `u_edge + q * dist`, where
// dist is the separation of the ghost and edge cells' face midpoints along the
// wall's axis. on a graded axis those face midpoints come from the axis map:
// log `face(i) = x_lo 10^(i s)`, geometric `face(i) = x_lo + w (r^i - 1)/(r - 1)`.
// the grid stores the nominal linear width `extent / n` as its spacing, the value
// the front door hands the builder; the map carries the slope or first width.
// the expected ghosts are computed here from the face formulas, independent of
// the kernel's binding. a uniform axis is the control.
//
// run: cargo test -p symbi --test gradient_wall_mapped_axis
// =============================================================================

use symbi::prelude::*;
use symbi::regimes::substrate_kernels::GradientBc;
use symbi::sim::evolve::KernelSet;
use symbi_geometry::AxisMap;

type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

const N: usize = 8;
const X_LO: f64 = 1.0;
const X_HI: f64 = 10.0;
const RHO: f64 = 2.0;
const VX: f64 = 0.4;
const PRE: f64 = 3.0;
// per-variable outward gradients in prim order [rho, vel_0, pre]
const Q: [f64; 3] = [0.5, -0.3, 1.0];
const RATIO: f64 = 1.3;

fn log_map() -> AxisMap {
    AxisMap::Log { start: X_LO, log_slope: (X_HI / X_LO).log10() / N as f64 }
}

fn geometric_width() -> f64 {
    (X_HI - X_LO) * (RATIO - 1.0) / (RATIO.powi(N as i32) - 1.0)
}

fn geometric_map() -> AxisMap {
    AxisMap::Geometric { start: X_LO, width: geometric_width(), ratio: RATIO }
}

/// the face positions written out from each map's definition
fn face(map: Option<AxisMap>, i: isize) -> f64 {
    let nominal = (X_HI - X_LO) / N as f64;
    match map {
        None => X_LO + i as f64 * nominal,
        Some(AxisMap::Log { .. }) => X_LO * (X_HI / X_LO).powf(i as f64 / N as f64),
        Some(AxisMap::Geometric { .. }) => X_LO + geometric_width() * (RATIO.powi(i as i32) - 1.0) / (RATIO - 1.0),
        Some(AxisMap::Uniform { .. }) => unreachable!("the control passes no map"),
    }
}

fn midpoint(map: Option<AxisMap>, i: isize) -> f64 {
    0.5 * (face(map, i) + face(map, i + 1))
}

/// fill both neumann walls of a uniform-state grid on `map` and return the largest ghost error
/// against `u_edge + q * dist`, with the number of ghosts checked
fn worst_ghost_error(map: Option<AxisMap>) -> (f64, usize) {
    let boundaries = Boundaries::<1>::per_axis([[BoundaryType::Neumann(0), BoundaryType::Neumann(0)]]);
    let sim = Sim::build(Newtonian, IdealGas { gamma: 1.4 }, Cartesian)
        .cells([N])
        .origin([X_LO])
        .spacing([(X_HI - X_LO) / N as f64])
        .coord_maps(map.map(|m| [m]))
        .boundaries(boundaries)
        .allocate()
        .expect("sim")
        .set_initial(|_| Prim::adiabatic(Density(RHO), Tensor::new([VX]), Pressure(PRE)))
        .build();
    let (sub, id) = sim.substrate().with_gradient_boundary(GradientBc::Neumann(Q.to_vec()));
    assert_eq!(id, 0);
    sub.c2p(&sim);
    sub.ghost_fill(&sim);
    let (lo, hi) = (sim.geom.interior.spaces[0].lo, sim.geom.interior.spaces[0].hi);
    let pre = sim.fields.prim.pre_field().unwrap();
    let (mut worst, mut checked) = (0.0f64, 0usize);
    for c in sim.geom.allocated.iter() {
        let i = c[0];
        let edge = if i < lo { lo } else if i >= hi { hi - 1 } else { continue };
        // whole-grid indices: the first interior cell is index 0
        let dist = (midpoint(map, i - lo) - midpoint(map, edge - lo)).abs();
        let got = [*sim.fields.prim.rho.view().at(c), *sim.fields.prim.vel[0].view().at(c), *pre.view().at(c)];
        let want = [RHO + Q[0] * dist, VX + Q[1] * dist, PRE + Q[2] * dist];
        for k in 0..3 {
            worst = worst.max((got[k] - want[k]).abs() / want[k].abs());
        }
        checked += 1;
    }
    (worst, checked)
}

#[test]
fn a_neumann_wall_extrapolates_over_the_mapped_cell_separation() {
    let cases = [("uniform", None), ("log", Some(log_map())), ("geometric", Some(geometric_map()))];
    let mut failures = Vec::new();
    for (label, map) in cases {
        let (worst, checked) = worst_ghost_error(map);
        assert!(checked >= 4, "{label}: {checked} ghosts checked; the gate is vacuous");
        eprintln!("{label}: worst relative ghost error {worst:.3e} over {checked} ghosts");
        if worst > 1e-12 {
            failures.push(format!("{label}: worst relative ghost error {worst:.3e}"));
        }
    }
    // the premise: on the graded axes the separation differs from the nominal width, so a wall
    // reading the nominal spacing would be caught
    let nominal = (X_HI - X_LO) / N as f64;
    for (label, map) in [("log", Some(log_map())), ("geometric", Some(geometric_map()))] {
        let sep = midpoint(map, N as isize) - midpoint(map, N as isize - 1);
        assert!((sep - nominal).abs() > 0.1 * nominal, "{label}: separation {sep} equals the nominal width");
    }
    assert!(failures.is_empty(), "{failures:?}");
}
