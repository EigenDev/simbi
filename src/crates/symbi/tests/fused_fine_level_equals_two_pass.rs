// =============================================================================
// fused_fine_level_equals_two_pass.rs
//
// source/body fusion must hold at refined levels as well as the uni-grid. a 2-level
// hierarchy carrying a central accreting body is evolved two ways — every level's kernel-set fused
// (`with_source_fusion`, body folded into godunov) vs every level two-pass (the standalone
// `body_source`) — and must produce a bit-for-bit identical trajectory on every level. this gates the
// py `$make` closure fusing fine levels (it re-attaches the source / enables fusion per level);
// without it a refined run silently drops to the two-pass on its finest, most-cell-dense levels.
//
// run: cargo test -p symbi --test fused_fine_level_equals_two_pass
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const N: usize = 8;

type Sim = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 3>;
type Hier = Hierarchy<Newtonian, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset>;

fn central_black_hole() -> BodyCollection<f64, 3> {
    BodyCollection::new().add(Body::black_hole(
        0,
        Tensor::new([0.5; 3]),
        Tensor::zeros(),
        1.0,  // mass
        0.05, // radius
        0.15, // softening
        10.0, // sink_rate
        1e-3, // sink_delta
        0.2,  // accretion_radius
    ))
}

// a 2-level hierarchy on [0,1)^3 with the middle half refined, a central accreting body on the finest
// level. `fused` builds every level's kernel-set with source fusion on (body folds into godunov) or
// off (the standalone body_source pass).
fn two_level(fused: bool) -> Hier {
    let dx = 1.0 / N as f64;
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 3])
        .origin([0.0; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(CFL)
        .allocate()
        .unwrap()
        .set_initial(|_x: [f64; 3]| {
            Prim::adiabatic(Density(2.0), Tensor::new([0.0; 3]), Pressure(1.0))
        })
        .build();
    let ck = {
        let ks = Kset::new(GAMMA, CFL, &coarse.geom.allocated);
        if fused { ks.with_source_fusion() } else { ks }
    };
    let regions = [RefinementRegion {
        x_lo: [0.25; 3],
        x_hi: [0.75; 3],
    }];
    let hier = Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, move |s| {
        let ks = Kset::new(GAMMA, CFL, &s.geom.allocated);
        if fused { ks.with_source_fusion() } else { ks }
    })
    .unwrap()
    .with_bodies(central_black_hole());
    // refinement allocates the fine level zeroed; the coarse initial condition has to be
    // prolonged into its interior. without this the fine level carries vacuum, both kernel sets
    // fold a body source into nothing, and the bit-for-bit comparison below holds trivially.
    hier.seed_fine_from_coarse().expect("fine-level seed");
    hier
}

#[test]
fn fused_fine_level_equals_two_pass() {
    // the two-pass is the default; this test pins the fused kernel as live, so
    // opt in before the policy OnceLock latches.
    unsafe { std::env::set_var("SYMBI_FUSE", "1") };
    let t_final = 0.02;
    let mut h_two = two_level(false);
    let mut h_fused = two_level(true);
    h_two.evolve(t_final).expect("two-pass hierarchy evolve");
    h_fused.evolve(t_final).expect("fused hierarchy evolve");

    // guard: the finest fused kernel-set actually compiled the body fold (else two-pass vs two-pass).
    assert_eq!(
        h_fused.levels[1].kernels.body_only_fused_state(),
        Some(true),
        "fine-level body fusion did not compile — fell back to the two-pass",
    );

    assert_eq!(h_two.levels.len(), h_fused.levels.len(), "level count");
    for lvl in 0..h_two.levels.len() {
        let a = &h_two.levels[lvl].state;
        let b = &h_fused.levels[lvl].state;
        for c in a.geom.interior.iter() {
            assert_eq!(
                a.fields.cons.den.view().at(c).to_bits(),
                b.fields.cons.den.view().at(c).to_bits(),
                "den differs at level {lvl} cell {c:?}",
            );
            for k in 0..3 {
                assert_eq!(
                    a.fields.cons.mom[k].view().at(c).to_bits(),
                    b.fields.cons.mom[k].view().at(c).to_bits(),
                    "mom_{k} differs at level {lvl} cell {c:?}",
                );
            }
            let (na, nb) = (
                a.fields.cons.nrg_field().unwrap(),
                b.fields.cons.nrg_field().unwrap(),
            );
            assert_eq!(
                na.view().at(c).to_bits(),
                nb.view().at(c).to_bits(),
                "nrg differs at level {lvl} cell {c:?}",
            );
        }
    }
}
