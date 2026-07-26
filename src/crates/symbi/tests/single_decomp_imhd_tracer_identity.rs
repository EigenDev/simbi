// =============================================================================
// single_decomp_imhd_tracer_identity.rs
//
// isothermal-mhd mass-tracer driver-equivalence gate. a periodic translating
// flow must move tracers through the accepted density flux, and single-grid and
// one-tile decomposed drivers must produce identical ownership.
// =============================================================================

use symbi::prelude::*;
use symbi::regimes::substrate_isothermal_mhd::IsothermalMhdSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, evolve_decomposed};
use symbi_hydro::energy::IsoModel;
use symbi_hydro::mhd_state::MhdPrimG;
use symbi_hydro::state::PrimG;
use symbi_sim::tracers::seed_mass_weighted;

const CS: f64 = 1.0;
const CFL: f64 = 0.35;
const T_FINAL: f64 = 0.05;

type Sim = SimStateGeneric<IsothermalMhd, 2, 3, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
type Kern = IsothermalMhdSubstrateKernelSet<HostMemory, f64, 2>;

fn make() -> (Sim, Kern) {
    let sim = Sim::build(IsothermalMhd, Isothermal { cs: CS }, Cartesian)
        .cells([32, 16])
        .bounds([0.0, 0.0], [1.0, 1.0])
        .boundaries(BoundaryType::Periodic)
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("isothermal mhd simulation construction")
        .set_initial(|_| MhdPrimG::<f64, 3, IsoModel> {
            hydro: PrimG {
                rho: 1.0,
                vel: Tensor::new([0.4, 0.0, 0.0]),
                pre: Default::default(),
            },
            mag: Tensor::new([0.2, 0.0, 0.1]),
        })
        .seed_faces_uniform([0.2, 0.0])
        .build();
    let kernels = Kern::new(CS, CFL, 1.0, &sim.geom.allocated);
    (sim, kernels)
}

#[test]
fn rk2_imhd_tracers_single_grid_equal_one_tile_decomposed() {
    let (mut single, single_kernels) = make();
    let (mut decomposed, decomposed_kernels) = make();
    single.tracers = Some(seed_mass_weighted(&single, 4096));
    decomposed.tracers = Some(seed_mass_weighted(&decomposed, 4096));
    let initial_owner = single.tracers.as_ref().unwrap().owner.clone();

    evolve(&mut single, &single_kernels, T_FINAL).expect("single-grid imhd tracers");
    evolve_decomposed(
        &mut [&mut *decomposed],
        &[&decomposed_kernels],
        [1, 1],
        &[0],
        Timestepping::Rk2,
        0.0,
        T_FINAL,
        u64::MAX,
        &LocalCopy,
        |_, _, _| std::ops::ControlFlow::Continue(()),
    );

    let single_tracers = single.tracers.as_ref().unwrap();
    let decomposed_tracers = decomposed.tracers.as_ref().unwrap();
    assert_ne!(
        single_tracers.owner, initial_owner,
        "the translating flow never transported a tracer"
    );
    assert_eq!(single_tracers.id, decomposed_tracers.id);
    assert_eq!(single_tracers.owner, decomposed_tracers.owner);
    assert_eq!(single_tracers.flags, decomposed_tracers.flags);
}
