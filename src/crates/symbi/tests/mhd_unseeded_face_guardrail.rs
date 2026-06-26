// =============================================================================
// mhd_unseeded_face_guardrail.rs
//
// new-user guardrail (ergonomics pass, win 4): an MHD sim whose staggered face B was never
// seeded must fail at evolve ENTRY with an actionable message — not march into a deep c2p/dt
// panic. seeding cell-centered B (seed_cell/seed_cells) does NOT initialize the faces; the
// constrained transport needs the staggered `bface` as its divergence-free ground truth.
//
// (also a compact showcase of the ergonomics pass: prelude + SimCpuGeneric + builder +
// seed_cells + substrate().)
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
    // seed the CELL-centered state (incl. cell B) but FORGET `seed_face` — the classic first-MHD
    // mistake. the guardrail must catch it at evolve entry.
    sim.seed_cells(|_| MhdPrim {
        hydro: Prim { rho: 1.0, vel: Tensor::new([0.0, 0.0, 0.0]), pre: 1.0 },
        mag: Tensor::new([0.1, 0.0, 0.0]),
    });
    let sub = sim.substrate();
    evolve(&mut sim, &sub, 0.01).expect("guardrail should panic before evolve runs");
}
