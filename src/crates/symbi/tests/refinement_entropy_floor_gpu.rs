// =============================================================================
// refinement_entropy_floor_gpu.rs
//
// the device twin of the balanced-seam entropy-floor gate: the same 2-level
// hydrostatic column on CudaSpace/UnifiedMemory, exercising the SAME baked
// wb_cf_lerp_encode / wb_cf_decode kernel pair the cpu gate runs. the transfer
// activates on device because the balance gate is memory-space-blind; this
// asserts the device hierarchy holds the same floor bound the cpu gate pins.
//
// run: cargo test -p symbi --features cuda --test refinement_entropy_floor_gpu
// =============================================================================

#![cfg(feature = "cuda")]

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::cuda::{CudaSpace, UnifiedMemory};

const GAMMA: f64 = 5.0 / 3.0;
const N: usize = 128;
const CFL: f64 = 0.4;
const K0: f64 = 1.0;
/// the gravitating mass sits one domain-width left of `x = 0`, so the gas at `x`
/// feels a bare point mass at radius `x + 1` and the domain covers `r` in [1, 2]
/// with no singularity.
const G_OFFSET: f64 = 1.0;
const GM: f64 = 100.0;
const T_GATE: f64 = 2.0;

/// the isentropic atmosphere in hydrostatic balance against `GM`, from the
/// bernoulli invariant, normalized to `rho = 1` at the outer edge.
fn hydrostatic(x: [f64; 1]) -> Prim<f64, 1> {
    let r = x[0] + G_OFFSET;
    let a = (GAMMA - 1.0) / (GAMMA * K0);
    let c = 1.0 / a - GM / (1.0 + G_OFFSET);
    let rho = (a * (GM / r + c)).powf(1.0 / (GAMMA - 1.0));
    Prim {
        rho,
        vel: symbi_algebra::Tensor::new([0.0]),
        pre: K0 * rho.powf(GAMMA),
    }
}

type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CudaSpace, UnifiedMemory>;
type Kset = AdiabaticSubstrateKernelSet<UnifiedMemory, f64, 1>;
type Hier = Hierarchy<Newtonian, 1, 1, Cartesian, IdealGas<f64>, CudaSpace, UnifiedMemory, Kset>;

fn build(balanced: bool) -> Hier {
    let region = RefinementRegion {
        x_lo: [0.3],
        x_hi: [0.7],
    };
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N])
        .spacing([1.0 / N as f64])
        .boundaries(Boundaries::uniform(BoundaryType::Reflect))
        .cfl(CFL)
        .allocate()
        .expect("sim construction failed")
        .set_initial(hydrostatic)
        .build();
    let make = move |s: &Sim| {
        Kset::new(GAMMA, CFL, &s.geom.allocated).well_balanced_reconstruction(balanced)
    };
    let ck = make(&coarse);
    let hier = Hierarchy::with_refinement(coarse, ck, &[region], ProlongOrder::Ppm, make)
        .unwrap()
        .with_bodies(symbi_ib::BodyCollection::new().add(symbi_ib::Body::gravitational(
            0,
            symbi_algebra::Tensor::new([-G_OFFSET]),
            symbi_algebra::Tensor::zeros(),
            GM,
            1.0e-3,
            0.0,
        )));
    for lvl in 1..hier.levels.len() {
        hier.levels[lvl].state.seed_cells(hydrostatic);
    }
    hier
}

/// the worst `K / K0` over every interior cell of every level.
fn worst_entropy_ratio(hier: &Hier) -> (f64, usize) {
    let (mut worst, mut worst_level) = (f64::INFINITY, 0usize);
    for (lvl, level) in hier.levels.iter().enumerate() {
        let st = &level.state;
        let rho = st.fields.prim.rho.view();
        let pre = st
            .fields
            .prim
            .pre
            .as_ref()
            .expect("adiabatic carries pressure")
            .view();
        for c in st.geom.interior.iter() {
            let r = *rho.at(c);
            if r <= 0.0 {
                continue;
            }
            let k = *pre.at(c) / r.powf(GAMMA) / K0;
            if k < worst {
                worst = k;
                worst_level = lvl;
            }
        }
    }
    (worst, worst_level)
}

#[test]
fn the_balanced_seam_transfer_holds_the_entropy_floor_on_device() {
    // positive control: the PLAIN 2-level seam must vent visibly by this clock
    // (the cpu gate measures 1.2e-2), or the balanced arm below is quiet about a
    // setup that never stressed the seam.
    let mut plain = build(false);
    plain.evolve(T_GATE).unwrap();
    let (floor_plain, _) = worst_entropy_ratio(&plain);
    assert!(
        1.0 - floor_plain > 1.0e-3,
        "the plain device seam stopped venting (deficit {:.2e}); the balanced gate below \
         is vacuous on this setup",
        1.0 - floor_plain
    );

    // the gate: the balanced hierarchy activates the equilibrium transfer on the
    // device memory space and holds the floor inside the cpu gate's bound (the
    // open restriction-side residual, measured 1.8e-5 at t = 2).
    let mut on = build(true);
    on.evolve(T_GATE).unwrap();
    let (floor_on, lvl) = worst_entropy_ratio(&on);
    println!(
        "device balanced 2-level column at t = {T_GATE}: deficit {:.2e} (plain {:.2e})",
        1.0 - floor_on,
        1.0 - floor_plain
    );
    assert!(
        floor_on > 1.0 - 5.0e-5,
        "the balance-aware coarse-fine transfer left a deficit of {:.2e} on level {lvl} \
         on the device hierarchy: beyond the restriction-side residual, so the device \
         kernel path is venting",
        1.0 - floor_on
    );
}
