// =============================================================================
// driven_boundary_gpu.rs
//
// the driven-boundary pass on-device (GPU). the boundary NVRTC kernel (the
// (Coord, Assign) instance of apply_dag_core_gv) prescribes the x_lo ghost band's prim state, built
// lazily + module-cached. proves the boundary kernel renders + launches on the rtx 2070 and writes
// the prescribed inflow into unified-memory ghosts, matching the CPU twin.
//
// run: cargo test -p symbi --features cuda --test driven_boundary_gpu
// =============================================================================

#![cfg(feature = "cuda")]

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::KernelSet;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::expr_bridge::build_boundary_dag;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_hydro::{NEWTONIAN_SPEC, SourceConfig};
use symbi_xpu::cuda::{CudaSpace, UnifiedMemory};

type DevSim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CudaSpace, UnifiedMemory>;

#[test]
fn driven_inflow_prescribes_ghost_state_on_gpu() {
    let boundaries = Boundaries::<2>::per_axis([
        [BoundaryType::Driven(0), BoundaryType::Outflow],
        [BoundaryType::Periodic, BoundaryType::Periodic],
    ]);
    let sim = DevSim::build(Newtonian, IdealGas { gamma: 1.4 }, Cartesian)
        .cells([8, 8])
        .bounds([0.0, 0.0], [1.0, 1.0])
        .boundaries(boundaries)
        .finish()
        .unwrap();
    sim.seed_cells(|_| Prim {
        rho: 1.0,
        vel: Tensor::new([0.0, 0.0]),
        pre: 1.0,
    });

    let json = r#"{
        "kind": "dirichlet", "dim": 2, "outputs": [0, 1, 2, 3], "params": [],
        "nodes": [ {"op":"CONSTANT","value":2.0}, {"op":"CONSTANT","value":1.0},
                   {"op":"CONSTANT","value":0.0}, {"op":"CONSTANT","value":3.0} ]
    }"#;
    let cfg = SourceConfig::from_json(json).expect("parse");
    let built = build_boundary_dag(&cfg, &NEWTONIAN_SPEC).expect("driven boundary");

    let base =
        AdiabaticSubstrateKernelSet::<UnifiedMemory, f64, 2>::new(1.4, 0.4, &sim.geom.allocated);
    let (sub, id) = base.with_driven_boundary(built, cfg.params.clone());
    assert_eq!(id, 0);

    sub.ghost_fill(&sim);
    symbi_xpu::cuda::ctx_sync();

    let mut checked = 0usize;
    for c in sim.geom.allocated.iter() {
        let x = sim.geom.cell_coord(c);
        if x[0] < 0.0 && x[1] > 0.0 && x[1] < 1.0 {
            let rho = *sim.fields.prim.rho.view().at(c);
            let v0 = *sim.fields.prim.vel[0].view().at(c);
            let v1 = *sim.fields.prim.vel[1].view().at(c);
            let p = *sim.fields.prim.pre_field().unwrap().view().at(c);
            assert!(
                rho.is_finite() && (rho - 2.0).abs() < 1e-12,
                "gpu x_lo ghost rho = {rho}, want 2"
            );
            assert!(
                (v0 - 1.0).abs() < 1e-12,
                "gpu x_lo ghost vel_0 = {v0}, want 1"
            );
            assert!(v1.abs() < 1e-12, "gpu x_lo ghost vel_1 = {v1}, want 0");
            assert!((p - 3.0).abs() < 1e-12, "gpu x_lo ghost pre = {p}, want 3");
            checked += 1;
        }
    }
    assert!(checked > 0, "no x_lo ghost-band cells checked");
}
