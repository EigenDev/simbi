// =============================================================================
// single_decomp_excise_identity.rs
//
// exact horizon-excision driver-equivalence gate. cartesian kerr-schild rhd
// evolves an origin-containing grid with an active excision sphere and horizon
// diagnostic body. the fluid state, clock, timestep, and horizon ledger must
// match bitwise between the single-grid and one-tile decomposed drivers.
// =============================================================================

use symbi::prelude::*;
use symbi::regimes::substrate_rhd::RhdSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, evolve_decomposed};
use symbi::sim::substrate_seam::WithExcision;
use symbi_geometry::SchwarzschildKSCartesian;
use symbi_hydro::Rhd;
use symbi_ib::{Body, BodyCollection, BodyKind};

const GAMMA: f64 = 4.0 / 3.0;
const MASS: f64 = 0.3;
const R_EXC: f64 = 0.35;
const R_DIAG: f64 = 0.9;
const T_FINAL: f64 = 0.05;
const RHO_FLOOR: f64 = 1.0e-10;

type Sim = SimState<Rhd, 2, SchwarzschildKSCartesian<f64>, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = RhdSubstrateKernelSet<HostMemory, f64, 2>;

fn make(timestepping: Timestepping) -> (Sim, Kern) {
    let sim = Sim::build(
        Rhd,
        IdealGas { gamma: GAMMA },
        SchwarzschildKSCartesian { mass: MASS },
    )
    .cells([24, 24])
    .bounds([-1.2, -1.2], [1.2, 1.2])
    .boundaries(BoundaryType::Outflow)
    .cfl(0.3)
    .timestepping(timestepping)
    .allocate()
    .expect("kerr-schild simulation construction")
    .set_initial(|_| Prim {
        rho: 1.0,
        vel: Tensor::new([0.0, 0.0]),
        pre: 0.1,
    })
    .build()
    .with_bodies(BodyCollection::new().add(Body::horizon(0, R_EXC, R_DIAG)));
    let kernels = Kern::new(GAMMA, 0.3, &sim.geom.allocated).with_excision(R_EXC, 1.0, 1.0);
    (sim, kernels)
}

fn ledger(sim: &Sim) -> (f64, f64, f64, f64) {
    let body = sim.immersed.as_ref().expect("horizon body").bodies.get(0);
    match body.kind {
        BodyKind::Horizon {
            total_accreted_mass,
            total_accreted_energy,
            mdot,
            edot,
            ..
        } => (total_accreted_mass, total_accreted_energy, mdot, edot),
        _ => panic!("test body is not a horizon"),
    }
}

fn assert_excise_driver_identity(timestepping: Timestepping) {
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
    for cell in single.geom.interior.iter() {
        for (name, single_field, decomposed_field) in [
            (
                "cons.den",
                &single.fields.cons.den,
                &decomposed.fields.cons.den,
            ),
            (
                "cons.mom[0]",
                &single.fields.cons.mom[0],
                &decomposed.fields.cons.mom[0],
            ),
            (
                "cons.mom[1]",
                &single.fields.cons.mom[1],
                &decomposed.fields.cons.mom[1],
            ),
            (
                "cons.nrg",
                single.fields.cons.nrg_field().expect("single energy"),
                decomposed
                    .fields
                    .cons
                    .nrg_field()
                    .expect("decomposed energy"),
            ),
        ] {
            assert_eq!(
                single_field.view().at(cell).to_bits(),
                decomposed_field.view().at(cell).to_bits(),
                "{name} differs at {cell:?}",
            );
        }
    }

    let single_ledger = ledger(&single);
    let decomposed_ledger = ledger(&decomposed);
    for (index, (single_value, decomposed_value)) in [
        single_ledger.0,
        single_ledger.1,
        single_ledger.2,
        single_ledger.3,
    ]
    .into_iter()
    .zip([
        decomposed_ledger.0,
        decomposed_ledger.1,
        decomposed_ledger.2,
        decomposed_ledger.3,
    ])
    .enumerate()
    {
        assert_eq!(
            single_value.to_bits(),
            decomposed_value.to_bits(),
            "horizon ledger component {index} differs",
        );
    }
    assert!(
        single_ledger.0.abs() > 1.0e-12,
        "horizon ledger recorded no mass; driver identity is vacuous",
    );

    let mut excised_cells = 0usize;
    let mut evolved_exterior_cells = 0usize;
    for cell in single.geom.interior.iter() {
        let position = single.geom.cell_coord(cell);
        let radius = (position[0].powi(2) + position[1].powi(2)).sqrt();
        let density = *single.fields.prim.rho.view().at(cell);
        if radius < R_EXC - 0.1 {
            assert_eq!(
                density.to_bits(),
                RHO_FLOOR.to_bits(),
                "excised density is not the vacuum floor at {cell:?}",
            );
            excised_cells += 1;
        } else if radius > R_DIAG {
            let momentum = *single.fields.cons.mom[0].view().at(cell);
            if momentum.to_bits() != 0.0_f64.to_bits() {
                evolved_exterior_cells += 1;
            }
        }
    }
    assert!(excised_cells > 0, "no cell was excised");
    assert!(
        evolved_exterior_cells > 0,
        "exterior fluid never evolved; driver identity is vacuous",
    );
}

#[test]
fn euler_excise_single_grid_equals_one_tile_decomposed_bitwise() {
    assert_excise_driver_identity(Timestepping::Euler);
}

#[test]
fn rk2_excise_single_grid_equals_one_tile_decomposed_bitwise() {
    assert_excise_driver_identity(Timestepping::Rk2);
}

#[test]
fn rk3_excise_single_grid_equals_one_tile_decomposed_bitwise() {
    assert_excise_driver_identity(Timestepping::Rk3);
}
