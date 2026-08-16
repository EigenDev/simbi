// =============================================================================
// nmhd_immersed_sink.rs
//
// the immersed-body drain under Newtonian MHD via the 1/2|B|^2 sandwich. the drain
// acts on the hydro conserved state, whose `nrg` is the total energy
// `p/(g-1) + 1/2 rho v^2 + 1/2|B|^2`; stripping `1/2|B|^2` before the drain and
// restoring it after makes the drain see the gas energy alone, so the plasma (mass,
// momentum, gas energy) is removed while the magnetic field is left to constrained
// transport. this pins the well-posedness of a subgrid sink in MHD: mass drops, the
// cell-centered B is bit-unchanged, and 1/2|B|^2 is exactly invariant.
// =============================================================================

use symbi::regimes::substrate_kernels::dispatch_penalize;
use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet3D;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::state::Prim;
use symbi_ib::sdf::SdfExpr;
use symbi_ib::{Body, BodyCollection, SurfaceSpec};
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<NewtonianMhd, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

const N: usize = 24;
const L: f64 = 1.0;
const GAMMA: f64 = 5.0 / 3.0;

fn make_sim(b0: f64) -> Sim {
    let dx = 2.0 * L / N as f64;
    // uniform gas threaded by a uniform Bx = b0 (div-free): nothing evolves the field on its own,
    // so any change to 1/2|B|^2 after the drain is the sink wrongly touching the field.
    Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N, N])
        .origin([-L, -L, -L])
        .spacing([dx, dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("nmhd sim construction failed")
        .set_initial(move |_| MhdPrim {
            hydro: Prim {
                rho: 1.0,
                vel: Tensor::new([0.0, 0.0, 0.0]),
                pre: 1.0,
            },
            mag: Tensor::new([b0, 0.0, 0.0]),
        })
        .seed_faces(move |axis, _| if axis == 0 { b0 } else { 0.0 })
        .build()
}

fn total_mass(s: &Sim) -> f64 {
    s.geom
        .interior
        .iter()
        .map(|c| *s.fields.cons.den.view().at(c))
        .sum()
}

fn magnetic_energy(s: &Sim) -> f64 {
    let m = s.fields.mhd.as_ref().unwrap();
    s.geom
        .interior
        .iter()
        .map(|c| {
            let mut bsq = 0.0;
            for k in 0..3 {
                let b = *m.bcell[k].view().at(c);
                bsq += b * b;
            }
            0.5 * bsq
        })
        .sum()
}

fn bcell_snapshot(s: &Sim) -> Vec<f64> {
    let m = s.fields.mhd.as_ref().unwrap();
    let mut v = Vec::new();
    for c in s.geom.interior.iter() {
        for k in 0..3 {
            v.push(*m.bcell[k].view().at(c));
        }
    }
    v
}

/// per interior cell: density, the total energy `nrg`, and the magnetic energy `1/2|B|^2`
/// from the (drain-invariant) cell-centered B. with `v = 0`, `nrg - 1/2|B|^2` is the gas
/// internal energy, and a uniform drain scales it and the density by the same factor, so
/// the specific internal energy `(nrg - 1/2|B|^2) / den` is invariant. that is the sandwich
/// property: without it the naive drain scales the total energy — magnetic part included —
/// so the recovered gas energy comes out low and the specific internal energy drops.
fn cell_state(s: &Sim) -> Vec<(f64, f64, f64)> {
    let m = s.fields.mhd.as_ref().unwrap();
    let nrg = s.fields.cons.nrg_field().unwrap();
    let mut v = Vec::new();
    for c in s.geom.interior.iter() {
        let den = *s.fields.cons.den.view().at(c);
        let e = *nrg.view().at(c);
        let mut bsq = 0.0;
        for k in 0..3 {
            let b = *m.bcell[k].view().at(c);
            bsq += b * b;
        }
        v.push((den, e, 0.5 * bsq));
    }
    v
}

fn setup(b0: f64) -> Sim {
    let mut sim = make_sim(b0);
    // a shaped subgrid sink at the origin: a pure-drain porous wall (porosity 1 = the drain
    // channel active, wall channels off), so it removes plasma by a uniform scaling. the shape
    // routes it to the runtime-JIT'd kernel.
    sim = sim.with_bodies(BodyCollection::new().add(
        Body::rigid_sphere(0, Tensor::zeros(), Tensor::zeros(), 1.0, 0.3, 1.0, true).with_surface(
            SurfaceSpec::Porous {
                porosity: 1.0,
                k_eta_n: 50.0,
                k_eta_t: 50.0,
            },
        ),
    ));
    sim.immersed.as_mut().unwrap().shapes[0] = Some(SdfExpr::<f64, 3>::cuboid(
        [0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25],
    ));
    sim
}

// how many masked cells (density dropped) a run must touch for the assertion to bite.
const MIN_DRAINED_CELLS: usize = 8;

#[test]
fn nmhd_sink_drains_plasma_and_leaves_the_field_untouched() {
    let sim = setup(0.5);
    let mass0 = total_mass(&sim);
    let mag_e0 = magnetic_energy(&sim);
    let bcell0 = bcell_snapshot(&sim);
    let before = cell_state(&sim);
    assert!(
        mag_e0 > 0.0,
        "the seed field must carry energy for the test to bite"
    );

    // the bracketed penalize (the sandwich) — the exact production path.
    let sub = NewtonianMhdSubstrateKernelSet3D::<HostMemory, f64>::new(
        GAMMA,
        0.3,
        1.0,
        &sim.geom.allocated,
    );
    sub.penalize(&sim, 1e-3);

    assert!(total_mass(&sim) < mass0, "the sink drained no plasma");
    // constrained transport owns B; the drain never touches it (bit-exact).
    assert_eq!(
        bcell_snapshot(&sim),
        bcell0,
        "the drain modified the magnetic field"
    );
    // and the magnetic energy read from B is unchanged (trivially, since B is untouched).
    assert_eq!(magnetic_energy(&sim), mag_e0);

    // the real sandwich property: in every drained cell the specific internal energy is
    // preserved -- the drain removed gas energy in proportion to mass.
    let after = cell_state(&sim);
    let mut drained = 0;
    for ((d0, e0, b0), (d1, e1, b1)) in before.iter().zip(&after) {
        assert_eq!(b0, b1, "1/2|B|^2 changed in a cell (B was touched)");
        if *d1 < *d0 - 1e-12 {
            drained += 1;
            let u0 = (e0 - b0) / d0; // specific internal energy before
            let u1 = (e1 - b1) / d1; // ... and after the drain
            assert!(
                (u1 - u0).abs() < 1e-9 * u0.abs().max(1.0),
                "specific internal energy changed under the drain: {u0} -> {u1} \
                 (the magnetic energy leaked into the gas drain)"
            );
        }
    }
    assert!(
        drained >= MIN_DRAINED_CELLS,
        "the sink barely acted ({drained} cells)"
    );
}

#[test]
fn nmhd_naive_drain_without_the_sandwich_corrupts_the_gas_energy() {
    // bug-injection: the same sink, but the bare hydro drain without the 1/2|B|^2 bracket.
    // it scales the total energy -- magnetic part included -- so the recovered gas energy is
    // low and the specific internal energy drops. this is exactly what the sandwich prevents;
    // if this ever stops dropping, the sandwich has become a silent no-op.
    let sim = setup(0.5);
    let before = cell_state(&sim);
    dispatch_penalize(&sim, 1e-3, GAMMA, 1.0);
    let after = cell_state(&sim);

    let mut drained = 0;
    let mut corrupted = 0;
    for ((d0, e0, b0), (d1, e1, b1)) in before.iter().zip(&after) {
        assert_eq!(b0, b1, "the bare drain still must not touch B");
        if *d1 < *d0 - 1e-12 {
            drained += 1;
            let u0 = (e0 - b0) / d0;
            let u1 = (e1 - b1) / d1;
            if u1 < u0 - 1e-9 {
                corrupted += 1; // gas energy leaked into the field's share -> too low
            }
        }
    }
    assert!(
        drained >= MIN_DRAINED_CELLS,
        "the naive drain barely acted ({drained} cells)"
    );
    // the deeply-masked cells (drain factor well below 1) leak visibly; mask-boundary cells
    // (factor ~1) corrupt below 1e-9. it is enough that a substantial set is corrupted -- that
    // is what the sandwich makes exact for every cell in the companion test.
    assert!(
        corrupted >= MIN_DRAINED_CELLS,
        "the naive drain left the gas energy intact ({corrupted} of {drained} corrupted) -- \
         the sandwich test would not bite"
    );
}

#[test]
fn c_fast_makes_a_low_beta_sink_stiffer() {
    // the fast-magnetosonic rate: the wall relaxes on c_fast = sqrt(c_s^2 + c_a^2), c_a^2 =
    // |B|^2/rho, so the same sink in the same gas drains more per step in a strong (low-beta)
    // field than a weak one. here c_s ~ 1.3; b0 = 5 gives c_a ~ 5, a several-fold stiffer wall.
    // the gas state is identical for both (same rho/p/v), so the only difference is the rate --
    // if c_a^2 were dropped from it, the two would drain identically.
    let drained = |b0: f64| -> f64 {
        let sim = setup(b0);
        let m0 = total_mass(&sim);
        let sub = NewtonianMhdSubstrateKernelSet3D::<HostMemory, f64>::new(
            GAMMA,
            0.3,
            1.0,
            &sim.geom.allocated,
        );
        sub.penalize(&sim, 1e-3);
        m0 - total_mass(&sim)
    };
    let weak = drained(0.01); // c_a ~ 0 -> c_fast ~ c_s
    let strong = drained(5.0); // c_a ~ 5 -> c_fast ~ 4x c_s
    assert!(
        strong > 1.5 * weak,
        "the low-beta sink was not stiffer: weak-field drained {weak}, strong-field drained {strong}"
    );
}
