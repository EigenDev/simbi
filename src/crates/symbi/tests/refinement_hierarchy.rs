// =============================================================================
// refine_hierarchy.rs
//
// 1-level Hierarchy gate: a 1-level Hierarchy must reproduce evolve()
// BIT-FOR-BIT. the hierarchy's advance_level re-sequences the single-level SSP
// stage loop so flux registers can hook into it; this test pins
// the two loops in lockstep — any drift (a reordered kernel call, a missed
// stage hook) shows up as a bit mismatch.
//
// covers: 1D sod (rk2) + 3D smooth pulse (rk3), adiabatic newtonian, every
// field compared by bit pattern over the full allocated domain.
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;

type Sim<const D: usize> = SimState<Newtonian, D, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

fn assert_fields_bit_identical<const D: usize>(a: &Sim<D>, b: &Sim<D>)
where
    Newtonian: symbi_hydro::regime::Regime<f64, D>,
    Cartesian: symbi_geometry::Metric<f64, D>,
{
    assert_eq!(a.time.to_bits(), b.time.to_bits(), "time differs");
    assert_eq!(a.dt.to_bits(), b.dt.to_bits(), "dt differs");
    assert_eq!(a.iteration, b.iteration, "iteration differs");

    let a_nrg = a.fields.cons.nrg_field().expect("newtonian cons.nrg");
    let b_nrg = b.fields.cons.nrg_field().expect("newtonian cons.nrg");
    let a_pre = a.fields.prim.pre_field().expect("newtonian prim.pre");
    let b_pre = b.fields.prim.pre_field().expect("newtonian prim.pre");
    for coord in a.geom.allocated.iter() {
        let bits = |x: f64| x.to_bits();
        assert_eq!(
            bits(*a.fields.cons.den.view().at(coord)),
            bits(*b.fields.cons.den.view().at(coord)),
            "cons.den differs at {coord:?}"
        );
        assert_eq!(
            bits(*a_nrg.view().at(coord)),
            bits(*b_nrg.view().at(coord)),
            "cons.nrg differs at {coord:?}"
        );
        assert_eq!(
            bits(*a.fields.prim.rho.view().at(coord)),
            bits(*b.fields.prim.rho.view().at(coord)),
            "prim.rho differs at {coord:?}"
        );
        assert_eq!(
            bits(*a_pre.view().at(coord)),
            bits(*b_pre.view().at(coord)),
            "prim.pre differs at {coord:?}"
        );
        for dd in 0..D {
            assert_eq!(
                bits(*a.fields.cons.mom[dd].view().at(coord)),
                bits(*b.fields.cons.mom[dd].view().at(coord)),
                "cons.mom[{dd}] differs at {coord:?}"
            );
            assert_eq!(
                bits(*a.fields.prim.vel[dd].view().at(coord)),
                bits(*b.fields.prim.vel[dd].view().at(coord)),
                "prim.vel[{dd}] differs at {coord:?}"
            );
        }
    }
}

#[test]
fn single_level_hierarchy_matches_evolve_sod_1d_rk2() {
    let n = 128usize;
    let dx = 1.0 / n as f64;
    let make_sim = || {
        // raw conserved writes inverted to a prim closure: vel=0, pre from
        // nrg = pre/(gamma-1) at the cell center x = (c[0]+0.5)*dx.
        Sim::<1>::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
            .cells([n])
            .spacing([dx])
            .allocate()
            .unwrap()
            .set_initial(|x: [f64; 1]| {
                let (rho, pre) = if x[0] < 0.5 { (1.0, 1.0) } else { (0.125, 0.1) };
                Prim { rho, vel: symbi_algebra::Tensor::new([0.0]), pre }
            })
            .build()
    };

    let mut reference = make_sim();
    let ref_kernels =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, 0.4, &reference.geom.allocated);
    evolve(&mut reference, &ref_kernels, 0.1).expect("reference evolve failed");

    let sim = make_sim();
    let kernels =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, 0.4, &sim.geom.allocated);
    let mut hier = Hierarchy::single(sim, kernels);
    hier.evolve(0.1).expect("hierarchy evolve failed");

    assert!(reference.iteration > 5, "reference took only {} steps", reference.iteration);
    assert_fields_bit_identical(&reference, &hier.levels[0].state);
}

#[test]
fn single_level_hierarchy_matches_evolve_pulse_3d_rk3() {
    let n = 12usize;
    let dx = 1.0 / n as f64;
    let pi = std::f64::consts::PI;
    let make_sim = || {
        // raw conserved writes inverted to a prim closure: vel=0, pre from
        // nrg = pre/(gamma-1), sampled at the cell center (c+0.5)*dx.
        Sim::<3>::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
            .cells([n, n, n])
            .spacing([dx; 3])
            .boundaries(Boundaries::uniform(BoundaryType::Periodic))
            .timestepping(Timestepping::Rk3)
            .allocate()
            .unwrap()
            .set_initial(|x: [f64; 3]| {
                let rho = 1.0 + 0.1 * (2.0 * pi * x[0]).sin() * (2.0 * pi * x[1]).cos();
                let pre = 1.0 + 0.1 * (2.0 * pi * x[2]).sin();
                Prim { rho, vel: symbi_algebra::Tensor::new([0.0; 3]), pre }
            })
            .build()
    };

    let mut reference = make_sim();
    let ref_kernels =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 3>::new(GAMMA, 0.4, &reference.geom.allocated);
    evolve(&mut reference, &ref_kernels, 0.05).expect("reference evolve failed");

    let sim = make_sim();
    let kernels =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 3>::new(GAMMA, 0.4, &sim.geom.allocated);
    let mut hier = Hierarchy::single(sim, kernels);
    hier.evolve(0.05).expect("hierarchy evolve failed");

    assert!(reference.iteration > 1, "reference took only {} steps", reference.iteration);
    assert_fields_bit_identical(&reference, &hier.levels[0].state);
}
