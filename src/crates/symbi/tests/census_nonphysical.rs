// =============================================================================
// census_nonphysical.rs
//
// a census counts physical, uncovered gas. beside its ghosts and the coarse
// cells a finer level covers, it leaves out:
// - the cells whose centroid lies inside the excised core of a kerr-schild
//   horizon, by the excision kernels' own predicate;
// - the cells whose centroid lies inside a body whose surface acts on the gas
//   (an accretor's drain, a rigid wall), by the body's own mask geometry at the
//   position it holds when the census samples.
// a body that only gravitates leaves every cell in. a masked cell is excluded,
// which keeps it apart from a physical cell that falls outside the bin edges:
// the dropped count reports the latter alone.
//
// the gates sample through the production entry on a uniform unit-density
// grid, so a mass census counts cells, and compare against an independent
// count of the cells whose centroid lies in the masked region.
//
// run: cargo test -p symbi --test census_nonphysical
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
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

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

const N: usize = 40;
const DX: f64 = 2.0 / N as f64;
const LO: f64 = -1.0;

fn sim(bodies: Option<BodyCollection<f64, 2>>) -> Sim {
    let sim = Sim::build(Newtonian, IdealGas { gamma: 1.4 }, Cartesian)
        .cells([N, N])
        .origin([LO; 2])
        .spacing([DX; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim")
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0]), Pressure(1.0)))
        .build();
    let sim = match bodies {
        Some(b) => sim.with_bodies(b),
        None => sim,
    };
    AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(1.4, 0.4, &sim.geom.allocated).c2p(&sim.store);
    sim
}

/// the mass census: `rho * dv` summed, binned on x over [-0.5, 1] so the cells left of x = -0.5
/// are physical gas outside the edges
fn census(binned: bool) -> RegisteredCensus {
    let axes = if binned { r#"[{"name": "x", "expr": 3, "edges": [-0.5, 1.0]}]"# } else { "[]" };
    let json = format!(
        r#"{{
            "name": "mass",
            "axes": {axes},
            "values": [2],
            "value_names": ["mass"],
            "op": "add",
            "params": [],
            "nodes": [
                {{"op": "VARIABLE_RHO"}},
                {{"op": "VARIABLE_DV"}},
                {{"op": "MULTIPLY", "left": 0, "right": 1}},
                {{"op": "VARIABLE_X1"}}
            ]
        }}"#
    );
    let cfg = CensusConfig::from_json(&json).expect("census json");
    RegisteredCensus::new(symbi_sim::census::CensusEvaluator::new(&cfg).expect("census compiles"))
}

/// (mass, dropped) of one sample
fn sample(sim: &Sim, binned: bool, nonphysical: NonphysicalCells) -> (f64, u64) {
    let registrations = vec![census(binned)];
    let partial = level_partials(sim, &registrations, 0, None, None, nonphysical, sim.time);
    (partial[0].0.iter().sum(), partial[0].1)
}

/// the cells whose centroid satisfies `inside`
fn count(inside: impl Fn(f64, f64) -> bool) -> usize {
    (0..N)
        .flat_map(|i| (0..N).map(move |j| (LO + (i as f64 + 0.5) * DX, LO + (j as f64 + 0.5) * DX)))
        .filter(|&(x, y)| inside(x, y))
        .count()
}

/// the number of unit-density cells a mass total counts: each weighs dx^2, so the total over n
/// cells is n dx^2 up to the summation's roundoff
fn counted(mass: f64) -> usize {
    let n = mass / (DX * DX);
    assert!((n - n.round()).abs() < 1e-9, "the mass {mass} is no whole number of cells");
    n.round() as usize
}

#[test]
fn the_excised_core_is_left_out() {
    let s = sim(None);
    let (whole, _) = sample(&s, false, NonphysicalCells::default());
    assert_eq!(counted(whole), N * N, "the unmasked census counts every cell");
    let (masked, _) = sample(&s, false, NonphysicalCells { excision_radius: Some(0.33) });
    let excised = count(|x, y| x * x + y * y < 0.33 * 0.33);
    assert!(excised > 20, "the core holds {excised} cells; the gate is vacuous");
    assert_eq!(counted(masked), N * N - excised, "the census counts the excised core");
}

#[test]
fn an_accretor_and_a_rigid_body_leave_their_interiors_out() {
    let sink = Body::black_hole(0, Tensor::new([0.3, 0.2]), Tensor::zeros(), 1.0, 0.25, 0.05, 1.0, 0.0, 0.25);
    let wall = Body::rigid_sphere(1, Tensor::new([-0.4, -0.3]), Tensor::zeros(), 1.0, 0.2, 1.0, true);
    let s = sim(Some(BodyCollection::new().add(sink).add(wall)));
    let (masked, _) = sample(&s, false, NonphysicalCells::default());
    let inside = count(|x, y| ((x - 0.3).powi(2) + (y - 0.2).powi(2)).sqrt() < 0.25 || ((x + 0.4).powi(2) + (y + 0.3).powi(2)).sqrt() < 0.2);
    assert!(inside > 30, "the bodies hold {inside} cells; the gate is vacuous");
    assert_eq!(counted(masked), N * N - inside, "the census counts the bodies' interiors");
}

#[test]
fn a_body_that_only_gravitates_leaves_every_cell_in() {
    let gravity = Body::gravitational(0, Tensor::new([0.0, 0.0]), Tensor::zeros(), 1.0, 0.4, 0.05);
    let s = sim(Some(BodyCollection::new().add(gravity)));
    let (masked, _) = sample(&s, false, NonphysicalCells::default());
    assert_eq!(counted(masked), N * N, "a gravitating body masked the gas it pulls on");
}

#[test]
fn a_moving_body_masks_the_cells_it_covers_when_the_census_samples() {
    let sink = Body::black_hole(0, Tensor::new([-0.5, 0.0]), Tensor::zeros(), 1.0, 0.25, 0.05, 1.0, 0.0, 0.25);
    let mut s = sim(Some(BodyCollection::new().add(sink)));
    let registrations = vec![census(false)];
    let at = |s: &Sim| level_partials(s, &registrations, 0, None, None, NonphysicalCells::default(), s.time)[0].0[0];
    let first = at(&s);
    assert_eq!(counted(first), N * N - count(|x, y| ((x + 0.5).powi(2) + y * y).sqrt() < 0.25));
    // the body moves; the same registration (its segment cached from the first sample) masks the
    // cells at the new position and returns the ones it left
    s.immersed.as_mut().unwrap().bodies.get_mut(0).position = Tensor::new([0.45, 0.1]);
    let second = at(&s);
    let now_inside = count(|x, y| ((x - 0.45).powi(2) + (y - 0.1).powi(2)).sqrt() < 0.25);
    assert_eq!(counted(second), N * N - now_inside, "the census masked the body's old position");
}

#[test]
fn masked_cells_are_excluded_apart_from_physical_cells_outside_the_edges() {
    let sink = Body::black_hole(0, Tensor::new([-0.7, 0.0]), Tensor::zeros(), 1.0, 0.25, 0.05, 1.0, 0.0, 0.25);
    let s = sim(Some(BodyCollection::new().add(sink)));
    let (mass, dropped) = sample(&s, true, NonphysicalCells::default());
    let body = |x: f64, y: f64| ((x + 0.7).powi(2) + y * y).sqrt() < 0.25;
    // the body straddles the left edge at x = -0.5: its cells on either side are excluded, and the
    // physical cells left of the edge are dropped
    let left_physical = count(|x, y| x < -0.5 && !body(x, y));
    let binned_physical = count(|x, y| x >= -0.5 && !body(x, y));
    assert!(count(|x, y| x < -0.5 && body(x, y)) > 0 && count(|x, y| x >= -0.5 && body(x, y)) > 0, "the body straddles no edge; the gate is vacuous");
    assert_eq!(dropped as usize, left_physical, "the dropped count includes masked cells");
    assert_eq!(counted(mass), binned_physical, "the binned mass includes masked cells");
}

#[test]
fn a_shaped_body_masks_its_own_geometry_in_its_own_orientation() {
    // a slab 0.6 by 0.2 in its own frame, turned an eighth about z
    let wall = Body::rigid_sphere(0, Tensor::new([0.1, 0.0]), Tensor::zeros(), 1.0, 0.35, 1.0, true);
    let mut s = sim(Some(BodyCollection::new().add(wall)));
    let (c, n) = (std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2);
    {
        let immersed = s.immersed.as_mut().unwrap();
        immersed.shapes[0] = Some(symbi_ib::sdf::SdfExpr::Cuboid { center: [0.0; 3], half_extents: [0.3, 0.1, 1.0] });
        immersed.bodies.get_mut(0).orientation = [[c, -n, 0.0], [n, c, 0.0], [0.0, 0.0, 1.0]];
    }
    let (masked, _) = sample(&s, false, NonphysicalCells::default());
    // the body-frame point of a world centroid is R^T (x - center)
    let inside = count(|x, y| {
        let (dx, dy) = (x - 0.1, y);
        (c * dx + n * dy).abs() < 0.3 && (-n * dx + c * dy).abs() < 0.1
    });
    let unturned = count(|x, y| (x - 0.1).abs() < 0.3 && y.abs() < 0.1);
    assert!(inside > 0 && inside != unturned, "the turn changes no count; the gate is vacuous");
    assert_eq!(counted(masked), N * N - inside, "the census masked the slab outside its own orientation");
}
