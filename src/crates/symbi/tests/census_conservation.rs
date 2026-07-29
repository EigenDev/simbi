// =============================================================================
// census_conservation.rs
//
// the unification gate for binned reductions: total mass and total energy — the live
// conservation diagnostic — expressed as a census with NO bin axes. a census is a
// pointwise map followed by a segmented reduce, and a global reduction is the case
// with a single bucket. if the mechanism cannot express it, the mechanism is wrong.
//
// and the partition gate: the same extensive quantity binned into radial shells must
// sum back to the global total. that is the leaf-tiling check — it catches a bucket
// assignment that double-counts or loses cells, which no per-bucket value alone would
// reveal.
//
// the grid is SPHERICAL on purpose. the cell measure is r^2 dr, so a census that
// weighted cells uniformly (or by a cartesian dx) would land on a visibly different
// total. a cartesian grid with uniform spacing would pass under a wrong measure and
// prove nothing about the volume weight.
// =============================================================================

use symbi::regimes::substrate_gpu::{ReductionOrder, field_segmented_reduce};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Spherical;
use symbi_grid::Field;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_ir::emit::ReductionOp;
use symbi_sim::census::{BinAxis, CensusSpec};
use symbi_xpu::{CpuSpace, HostMemory};

type SimSph = SimState<Newtonian, 1, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;

const GAMMA: f64 = 1.4;
const N: usize = 256;
const R_LO: f64 = 0.5;
const R_HI: f64 = 4.5;
const DR: f64 = (R_HI - R_LO) / N as f64;

/// a steeply falling atmosphere. the density spans four orders of magnitude across the
/// shell while the cell volume grows as r^2, so the two weightings pull in opposite
/// directions — a census that used the wrong measure could not land on the same total
/// by coincidence.
fn density_at(r: f64) -> f64 {
    r.powi(-3)
}

fn pressure_at(r: f64) -> f64 {
    0.1 * density_at(r)
}

fn build_sim() -> SimSph {
    let sim = SimSph::build(Newtonian, IdealGas { gamma: GAMMA }, Spherical)
        .cells([N])
        .origin([R_LO])
        .spacing([DR])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("spherical sim construction failed")
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([0.0]),
            pre: 1.0,
        })
        .build();

    let cnrg = sim.fields.cons.nrg_field().expect("Newtonian cons.nrg");
    for c in sim.geom.interior.iter() {
        let r = R_LO + (c[0] as f64 + 0.5) * DR;
        sim.fields.cons.den.view_mut().set(c, density_at(r));
        sim.fields.cons.mom[0].view_mut().set(c, 0.0);
        cnrg.view_mut().set(c, pressure_at(r) / (GAMMA - 1.0));
    }
    sim
}

/// the census value fields: mass and energy per cell, each weighted by the cell's
/// lab-frame volume measure. this is what the `dV` expression leaf resolves to, taken
/// from the same block geometry the finite-volume update uses, so the sums are correct
/// on a curvilinear grid rather than assuming dx^3.
fn weighted_values(sim: &SimSph) -> (Field<f64, 1, HostMemory>, Field<f64, 1, HostMemory>) {
    let bg = sim.geom.block_geometry(sim.physics.metric);
    let a = sim.motion.a;
    let mass = Field::<f64, 1, HostMemory>::zeros(&sim.geom.allocated).expect("mass field");
    let energy = Field::<f64, 1, HostMemory>::zeros(&sim.geom.allocated).expect("energy field");
    let nrg = sim.fields.cons.nrg_field().expect("cons.nrg");
    for c in sim.geom.interior.iter() {
        let dv = bg.labframe_volume(c, a);
        mass.view_mut()
            .set(c, *sim.fields.cons.den.view().at(c) * dv);
        energy.view_mut().set(c, *nrg.view().at(c) * dv);
    }
    (mass, energy)
}

/// every interior cell into bucket zero — the zero-axis census.
fn one_bucket(sim: &SimSph) -> Field<f64, 1, HostMemory> {
    Field::<f64, 1, HostMemory>::zeros(&sim.geom.allocated).expect("segment field")
}

#[test]
fn total_mass_and_energy_are_a_census_with_no_bins() {
    // design acceptance: the conservation diagnostic IS a census with no bin axes,
    // agreeing to round-off. the two paths differ only in how they group the addends.
    let sim = build_sim();
    let diag = sim
        .conservation_diag()
        .expect("host-resident sim reports a conservation diagnostic");

    let spec = CensusSpec::new(
        "conservation",
        vec![],
        vec!["mass".into(), "energy".into()],
        ReductionOp::Add,
    )
    .expect("a census with no axes is a global reduction");
    assert_eq!(spec.n_segments(), 1, "no axes means one bucket");

    let (mass, energy) = weighted_values(&sim);
    let census = field_segmented_reduce(
        &[&mass, &energy],
        &one_bucket(&sim),
        &sim.geom.interior,
        spec.n_segments(),
        spec.op(),
    );

    assert_eq!(census.values.len(), 2);
    assert_eq!(census.dropped, 0, "every interior cell bins");
    assert_eq!(census.order, ReductionOrder::Exact);

    let tol = 1.0e-13 * diag.mass.abs();
    assert!(
        (census.values[0] - diag.mass).abs() <= tol,
        "census mass {:e} != conservation diagnostic {:e}",
        census.values[0],
        diag.mass
    );
    let diag_energy = diag.energy.expect("Newtonian carries an energy equation");
    let tol = 1.0e-13 * diag_energy.abs();
    assert!(
        (census.values[1] - diag_energy).abs() <= tol,
        "census energy {:e} != conservation diagnostic {:e}",
        census.values[1],
        diag_energy
    );

    // the measure is load-bearing: on this r^2 grid an unweighted sum is a different
    // number entirely, so the agreement above is not an accident of a flat geometry.
    let unweighted: f64 = sim
        .geom
        .interior
        .iter()
        .map(|c| *sim.fields.cons.den.view().at(c))
        .sum();
    assert!(
        (unweighted - diag.mass).abs() > 0.1 * diag.mass.abs(),
        "the r^2 cell measure must matter here, else this gate proves nothing about dV \
         (unweighted {unweighted:e} vs weighted {:e})",
        diag.mass
    );
}

#[test]
fn shell_bins_sum_back_to_the_global_total() {
    // the partition gate: an extensive quantity summed over every bucket equals its
    // global reduction. a bucket assignment that double-counted or lost cells fails
    // here and passes every per-bucket check.
    let sim = build_sim();
    let diag = sim.conservation_diag().expect("conservation diagnostic");

    // log-spaced radial shells spanning the whole domain, so no interior cell falls
    // outside the edges.
    let n_shells = 16;
    let edges: Vec<f64> = (0..=n_shells)
        .map(|k| {
            let f = k as f64 / n_shells as f64;
            (R_LO.ln() + f * (R_HI.ln() - R_LO.ln())).exp()
        })
        .collect();
    let spec = CensusSpec::new(
        "shells",
        vec![BinAxis::new("r", edges).expect("log-spaced edges")],
        vec!["mass".into()],
        ReductionOp::Add,
    )
    .expect("valid census");
    assert_eq!(spec.n_segments(), n_shells);

    let (mass, _) = weighted_values(&sim);
    let segment = Field::<f64, 1, HostMemory>::zeros(&sim.geom.allocated).expect("segment field");
    for c in sim.geom.interior.iter() {
        let r = R_LO + (c[0] as f64 + 0.5) * DR;
        segment.view_mut().set(c, spec.segment_marker(&[r]) as f64);
    }

    let census = field_segmented_reduce(
        &[&mass],
        &segment,
        &sim.geom.interior,
        spec.n_segments(),
        spec.op(),
    );
    assert_eq!(
        census.dropped, 0,
        "the shell edges span the domain, so no cell is outside the binning"
    );
    // every shell holds gas: an empty bucket would make the partition check weaker than
    // it looks, since a lost cell could hide in a bucket that was empty anyway.
    for (s, v) in census.values.iter().enumerate() {
        assert!(
            *v > 0.0,
            "shell {s} is empty; the binning does not tile the gas"
        );
    }

    let binned: f64 = census.values.iter().sum();
    let tol = 1.0e-12 * diag.mass.abs();
    assert!(
        (binned - diag.mass).abs() <= tol,
        "shell masses sum to {binned:e}, global total is {:e}",
        diag.mass
    );
}

#[test]
fn cells_outside_the_shell_edges_are_reported_not_absorbed() {
    // a binning that does not span the domain must say so. if the shortfall were folded
    // into the outermost shell, an under-covering census would be indistinguishable
    // from a physics result — the failure this counter exists to prevent.
    let sim = build_sim();

    // edges covering only the inner half of the shell.
    let r_cut = 0.5 * (R_LO + R_HI);
    let spec = CensusSpec::new(
        "inner",
        vec![BinAxis::new("r", vec![R_LO, r_cut]).expect("edges")],
        vec!["mass".into()],
        ReductionOp::Add,
    )
    .expect("valid census");

    let (mass, _) = weighted_values(&sim);
    let segment = Field::<f64, 1, HostMemory>::zeros(&sim.geom.allocated).expect("segment field");
    let mut expect_dropped = 0u64;
    let mut expect_inside = 0.0f64;
    for c in sim.geom.interior.iter() {
        let r = R_LO + (c[0] as f64 + 0.5) * DR;
        segment.view_mut().set(c, spec.segment_marker(&[r]) as f64);
        if r > r_cut {
            expect_dropped += 1;
        } else {
            expect_inside += *mass.view().at(c);
        }
    }
    assert!(expect_dropped > 0, "the truncated edges must exclude cells");

    let census = field_segmented_reduce(
        &[&mass],
        &segment,
        &sim.geom.interior,
        spec.n_segments(),
        spec.op(),
    );
    assert_eq!(census.dropped, expect_dropped);
    let tol = 1.0e-13 * expect_inside.abs();
    assert!(
        (census.values[0] - expect_inside).abs() <= tol,
        "the single shell holds only the cells inside its edges: {:e} vs {expect_inside:e}",
        census.values[0]
    );
}
