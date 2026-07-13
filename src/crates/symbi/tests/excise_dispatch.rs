// =============================================================================
// excise_dispatch.rs
//
// the horizon-excision dispatch, end to end on a real sim: an origin-containing
// cartesian kerr-schild box with the black hole (and its coordinate singularity)
// on the grid. the pass overwrites the excised sphere's primitives with outward
// zero-gradient copies + local conserved rebuilds, leaves the far field
// BIT-untouched, and a uniform state passes through as the exact identity.
// =============================================================================

use symbi::regimes::substrate_kernels::dispatch_excise;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::SchwarzschildKSCartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::state::Prim;
use symbi_hydro::Rhd;
use symbi_xpu::{CpuSpace, HostMemory};

const N: usize = 48;
const L: f64 = 1.2;
const GAMMA: f64 = 4.0 / 3.0;
const MASS: f64 = 0.3; // r_+ = 0.6, well inside the box
const R_EXC: f64 = 0.35; // inside r_+, above the metric guard M/2 = 0.15

type Sim = SimState<Rhd, 2, SchwarzschildKSCartesian<f64>, IdealGas<f64>, CpuSpace, HostMemory>;

fn build_sim(init: impl Fn([f64; 2]) -> Prim<f64, 2>) -> Sim {
    let dx = 2.0 * L / N as f64;
    let sim = Sim::build(Rhd, IdealGas { gamma: GAMMA }, SchwarzschildKSCartesian { mass: MASS })
        .cells([N, N])
        .origin([-L, -L])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim")
        .set_initial(|x| init(x))
        .build();
    // the builder stores the CONSERVED state; the primitive fields materialize via
    // c2p during evolution (every stage ends with one), so the excision pass always
    // reads current prims in production. populate them directly here to model the
    // post-c2p state the pass actually sees.
    for c in sim.geom.interior.iter() {
        let lo = sim.geom.interior.spaces[0].lo;
        let x = sim.geom.x_lo[0] + ((c[0] - lo) as f64 + 0.5) * dx;
        let y = sim.geom.x_lo[1] + ((c[1] - lo) as f64 + 0.5) * dx;
        let p = init([x, y]);
        sim.fields.prim.rho.set(c, p.rho);
        sim.fields.prim.vel[0].set(c, p.vel[0]);
        sim.fields.prim.vel[1].set(c, p.vel[1]);
        sim.fields.prim.pre_field().unwrap().set(c, p.pre);
    }
    sim
}

fn snapshot(sim: &Sim) -> Vec<[f64; 8]> {
    sim.geom
        .interior
        .iter()
        .map(|c| {
            [
                *sim.fields.prim.rho.view().at(c),
                *sim.fields.prim.vel[0].view().at(c),
                *sim.fields.prim.vel[1].view().at(c),
                *sim.fields.prim.pre_field().unwrap().view().at(c),
                *sim.fields.cons.den.view().at(c),
                *sim.fields.cons.mom[0].view().at(c),
                *sim.fields.cons.mom[1].view().at(c),
                *sim.fields.cons.nrg_field().unwrap().view().at(c),
            ]
        })
        .collect()
}

#[test]
fn excision_fills_the_sphere_and_leaves_the_far_field_bit_untouched() {
    let sim = build_sim(|[x, y]| Prim {
        rho: 1.0 + 0.2 * (2.0 * x).sin() * (1.5 * y).cos(),
        vel: Tensor::new([0.08 * (x + y).cos(), -0.06 * (x - y).sin()]),
        pre: 0.05 + 0.01 * (x * y).cos(),
    });
    let before = snapshot(&sim);

    dispatch_excise(&sim, GAMMA, R_EXC);

    let after = snapshot(&sim);
    let dx = 2.0 * L / N as f64;
    let (mut n_live, mut n_excised_changed) = (0usize, 0usize);
    for (i, c) in sim.geom.interior.iter().enumerate() {
        // the cell-centre radius; the +-2 dx band dodges the rim's exact
        // centroid-convention edge so the classification is unambiguous.
        let lo = sim.geom.interior.spaces[0].lo;
        let x = sim.geom.x_lo[0] + ((c[0] - lo) as f64 + 0.5) * dx;
        let y = sim.geom.x_lo[1] + ((c[1] - lo) as f64 + 0.5) * dx;
        let r = (x * x + y * y).sqrt();
        for k in 0..8 {
            assert!(after[i][k].is_finite(), "non-finite field {k} at ({x:.3},{y:.3})");
        }
        if r > R_EXC + 2.0 * dx {
            n_live += 1;
            for k in 0..8 {
                assert_eq!(
                    after[i][k].to_bits(),
                    before[i][k].to_bits(),
                    "live cell touched: field {k} at ({x:.3},{y:.3})"
                );
            }
        } else if r < R_EXC - 2.0 * dx && after[i] != before[i] {
            n_excised_changed += 1;
        }
    }
    assert!(n_live > 1000, "the far field must dominate (got {n_live} live cells)");
    assert!(
        n_excised_changed > 20,
        "the fill never rewrote the deep sphere (got {n_excised_changed} changed cells)"
    );
}

#[test]
fn excision_preserves_uniform_prims_and_is_idempotent() {
    // a uniform primitive state: every donor equals every target, so the fill
    // sweeps are the bitwise identity on the primitives. the conserved rebuild
    // recomputes the excised cells' cons from those prims — its arithmetic may
    // differ from the builder's host-side initial conversion by reassociation
    // (allowed, bounded below), but the PASS ITSELF must be idempotent: a second
    // dispatch reads the same prims at the same metric and must change nothing.
    let sim = build_sim(|_| Prim { rho: 1.3, vel: Tensor::new([0.05, -0.04]), pre: 0.02 });
    let before = snapshot(&sim);
    dispatch_excise(&sim, GAMMA, R_EXC);
    let once = snapshot(&sim);

    for (i, (a, b)) in once.iter().zip(before.iter()).enumerate() {
        // primitives: bitwise identity everywhere (copy of a uniform state).
        for k in 0..4 {
            assert_eq!(a[k].to_bits(), b[k].to_bits(), "prim {k} changed at interior cell {i}");
        }
        // conserved: the traced rebuild must agree with the builder's host
        // conversion physically; the spelling difference is reassociation-scale.
        for k in 4..8 {
            let scale = b[k].abs().max(1e-30);
            assert!(
                (a[k] - b[k]).abs() / scale < 1e-4,
                "cons {k} at cell {i}: rebuild {} vs stored {}",
                a[k],
                b[k]
            );
        }
    }

    dispatch_excise(&sim, GAMMA, R_EXC);
    let twice = snapshot(&sim);
    for (i, (a, b)) in twice.iter().zip(once.iter()).enumerate() {
        for k in 0..8 {
            assert_eq!(
                a[k].to_bits(),
                b[k].to_bits(),
                "second dispatch not idempotent: field {k} at interior cell {i}"
            );
        }
    }
}
