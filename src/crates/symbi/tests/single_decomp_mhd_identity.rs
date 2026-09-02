// =============================================================================
// single_decomp_mhd_identity.rs
//
// exact constrained-transport driver-equivalence gate. a single-grid rmhd run
// and a one-tile decomposed run start from identical cell and staggered magnetic
// states. every conserved field, cell-centered magnetic component, staggered
// face component, clock, and final timestep must match bitwise after evolution.
// =============================================================================

use symbi::prelude::*;
use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, evolve_decomposed};
use symbi_grid::Field;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::rmhd::Rmhd;
use symbi_sim::tracers::seed_mass_weighted;

const GAMMA: f64 = 5.0 / 3.0;
const T_FINAL: f64 = 0.02;
const B0: [f64; 3] = [0.3, 0.2, 0.1];

type Sim = SimStateGeneric<Rmhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = RmhdSubstrateKernelSet<HostMemory, f64, 2>;

fn make(timestepping: Timestepping) -> (Sim, Kern) {
    let sim = Sim::build(Rmhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([32, 32])
        .bounds([0.0, 0.0], [1.0, 1.0])
        .boundaries(BoundaryType::Outflow)
        .cfl(0.35)
        .timestepping(timestepping)
        .allocate()
        .expect("mhd simulation construction")
        .set_initial(|[x, y]| {
            let radius2 = (x - 0.5).powi(2) + (y - 0.5).powi(2);
            let bump = 0.2 * (-(radius2 / 0.01)).exp();
            MhdPrim::new(
                Prim::adiabatic(
                    Density(1.0 + bump),
                    Tensor::new([0.0, 0.0, 0.0]),
                    Pressure(1.0 + bump),
                ),
                Tensor::new(B0),
            )
        })
        .seed_faces_uniform([B0[0], B0[1]])
        .build();
    let kernels = Kern::new(GAMMA, 0.35, 1.0, &sim.geom.allocated);
    (sim, kernels)
}

fn assert_field_bits(
    single: &Field<f64, 2, HostMemory>,
    decomposed: &Field<f64, 2, HostMemory>,
    name: &str,
) {
    for cell in single.domain().iter() {
        assert_eq!(
            single.view().at(cell).to_bits(),
            decomposed.view().at(cell).to_bits(),
            "{name} differs at {cell:?}: {} vs {}",
            single.view().at(cell),
            decomposed.view().at(cell),
        );
    }
}

fn divergence_max(sim: &Sim) -> f64 {
    let magnetic = sim.fields.mhd.as_ref().expect("mhd fields");
    let mut maximum = 0.0_f64;
    for cell in sim.geom.interior.iter() {
        let mut divergence = 0.0;
        for axis in 0..2 {
            let mut upper = cell;
            upper[axis] += 1;
            divergence += (*magnetic.bface[axis].view().at(upper)
                - *magnetic.bface[axis].view().at(cell))
                / sim.geom.dx[axis];
        }
        maximum = maximum.max(divergence.abs());
    }
    maximum
}

fn assert_mhd_driver_identity(timestepping: Timestepping) {
    let (mut single, single_kernels) = make(timestepping);
    let (mut decomposed, decomposed_kernels) = make(timestepping);

    evolve(&mut single, &single_kernels, T_FINAL).expect("single-grid evolve");
    evolve_decomposed(
        &mut [&mut *decomposed],
        &[&decomposed_kernels],
        [1, 1],
        &[0],
        timestepping,
        0.0,
        T_FINAL,
        u64::MAX,
        &LocalCopy,
        |_, _, _| std::ops::ControlFlow::Continue(()),
    );

    assert_eq!(
        single.time.to_bits(),
        decomposed.time.to_bits(),
        "time differs"
    );
    assert_eq!(single.dt.to_bits(), decomposed.dt.to_bits(), "dt differs");
    assert_field_bits(
        &single.fields.cons.den,
        &decomposed.fields.cons.den,
        "cons.den",
    );
    for component in 0..3 {
        assert_field_bits(
            &single.fields.cons.mom[component],
            &decomposed.fields.cons.mom[component],
            &format!("cons.mom[{component}]"),
        );
    }
    assert_field_bits(
        single.fields.cons.nrg_field().expect("single energy"),
        decomposed
            .fields
            .cons
            .nrg_field()
            .expect("decomposed energy"),
        "cons.nrg",
    );

    let single_mhd = single.fields.mhd.as_ref().expect("single mhd fields");
    let decomposed_mhd = decomposed
        .fields
        .mhd
        .as_ref()
        .expect("decomposed mhd fields");
    for component in 0..3 {
        assert_field_bits(
            &single_mhd.bcell.b[component],
            &decomposed_mhd.bcell.b[component],
            &format!("bcell[{component}]"),
        );
    }
    for axis in 0..2 {
        assert_field_bits(
            &single_mhd.bface[axis],
            &decomposed_mhd.bface[axis],
            &format!("bface[{axis}]"),
        );
    }

    let magnetic_evolved = single.geom.interior.iter().any(|cell| {
        (0..2).any(|axis| single_mhd.bface[axis].view().at(cell).to_bits() != B0[axis].to_bits())
    });
    assert!(
        magnetic_evolved,
        "staggered magnetic field never evolved; ct identity is vacuous",
    );
    let divergence = divergence_max(&single);
    assert!(
        divergence < 1.0e-12,
        "constrained transport produced div(B) = {divergence:e}",
    );
}

fn assert_mhd_tracer_driver_identity(timestepping: Timestepping) {
    let (mut single, single_kernels) = make(timestepping);
    let (mut decomposed, decomposed_kernels) = make(timestepping);
    single.tracers = Some(seed_mass_weighted(&single, 512));
    decomposed.tracers = Some(seed_mass_weighted(&decomposed, 512));

    evolve(&mut single, &single_kernels, T_FINAL).expect("single-grid mhd tracers");
    evolve_decomposed(
        &mut [&mut *decomposed],
        &[&decomposed_kernels],
        [1, 1],
        &[0],
        timestepping,
        0.0,
        T_FINAL,
        u64::MAX,
        &LocalCopy,
        |_, _, _| std::ops::ControlFlow::Continue(()),
    );

    let single_tracers = single.tracers.as_ref().unwrap();
    let decomposed_tracers = decomposed.tracers.as_ref().unwrap();
    assert_eq!(single_tracers.id, decomposed_tracers.id);
    assert_eq!(single_tracers.owner, decomposed_tracers.owner);
    assert_eq!(single_tracers.flags, decomposed_tracers.flags);
}

fn assert_mhd_tracing_inert(timestepping: Timestepping) {
    const TRANSPORT_TIME: f64 = 0.2;

    let (mut untraced, untraced_kernels) = make(timestepping);
    let (mut traced, traced_kernels) = make(timestepping);
    traced.tracers = Some(seed_mass_weighted(&traced, 512));
    let initial_owners = traced.tracers.as_ref().unwrap().owner.clone();

    evolve(&mut untraced, &untraced_kernels, TRANSPORT_TIME).expect("untraced mhd evolve");
    evolve(&mut traced, &traced_kernels, TRANSPORT_TIME).expect("traced mhd evolve");

    let moved = traced
        .tracers
        .as_ref()
        .unwrap()
        .owner
        .iter()
        .zip(initial_owners)
        .filter(|(owner, initial)| **owner != *initial)
        .count();
    assert!(moved > 0, "mhd tracer transport was not exercised");
    assert!(untraced.tracers.is_none());
    assert_eq!(traced.time.to_bits(), untraced.time.to_bits());
    assert_eq!(traced.dt.to_bits(), untraced.dt.to_bits());
    assert_eq!(traced.iteration, untraced.iteration);

    assert_field_bits(
        &traced.fields.cons.den,
        &untraced.fields.cons.den,
        "cons.den",
    );
    for component in 0..3 {
        assert_field_bits(
            &traced.fields.cons.mom[component],
            &untraced.fields.cons.mom[component],
            &format!("cons.mom[{component}]"),
        );
    }
    assert_field_bits(
        traced.fields.cons.nrg_field().expect("traced energy"),
        untraced.fields.cons.nrg_field().expect("untraced energy"),
        "cons.nrg",
    );

    let traced_mhd = traced.fields.mhd.as_ref().expect("traced mhd fields");
    let untraced_mhd = untraced.fields.mhd.as_ref().expect("untraced mhd fields");
    for component in 0..3 {
        assert_field_bits(
            &traced_mhd.bcell.b[component],
            &untraced_mhd.bcell.b[component],
            &format!("bcell[{component}]"),
        );
    }
    for axis in 0..2 {
        assert_field_bits(
            &traced_mhd.bface[axis],
            &untraced_mhd.bface[axis],
            &format!("bface[{axis}]"),
        );
    }
}

#[test]
fn euler_mhd_single_grid_equals_one_tile_decomposed_bitwise() {
    assert_mhd_driver_identity(Timestepping::Euler);
}

#[test]
fn rk2_mhd_single_grid_equals_one_tile_decomposed_bitwise() {
    assert_mhd_driver_identity(Timestepping::Rk2);
}

#[test]
fn rk2_mhd_tracers_single_grid_equal_one_tile_decomposed() {
    assert_mhd_tracer_driver_identity(Timestepping::Rk2);
}

#[test]
fn rk2_mhd_tracing_is_bitwise_inert() {
    assert_mhd_tracing_inert(Timestepping::Rk2);
}
