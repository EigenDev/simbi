// =============================================================================
// mhd_unseeded_face_guardrail.rs
//
// an MHD sim carrying unseeded staggered face B fails at evolve entry with an
// actionable message, ahead of any deep c2p/dt panic. seeding cell-centered B
// (seed_cell/seed_cells) leaves the faces uninitialized; the constrained transport needs the
// staggered `bface` as its divergence-free ground truth.
//
// the setup is also the minimal frontend path: prelude + SimCpuGeneric + builder +
// seed_cells + substrate().
// =============================================================================

use symbi::prelude::*;

type Sim = SimCpuGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>>;

#[test]
#[should_panic(expected = "staggered face B is un-seeded")]
fn evolve_without_seeding_faces_fails_with_guidance() {
    let mut sim = Sim::build(NewtonianMhd, IdealGas { gamma: 1.4 }, Cartesian)
        .cells([16, 16])
        .bounds([0.0, 0.0], [1.0, 1.0])
        .finish()
        .unwrap();
    // seed the cell-centered state (incl. cell B) but forget `seed_face` — the classic first-MHD
    // mistake. the guardrail must catch it at evolve entry.
    sim.seed_cells(|_| {
        MhdPrim::new(
            Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)),
            Tensor::new([0.1, 0.0, 0.0]),
        )
    });
    let sub = sim.substrate();
    evolve(&mut sim, &sub, 0.01).expect("guardrail should panic before evolve runs");
}
