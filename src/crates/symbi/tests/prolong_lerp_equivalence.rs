// =============================================================================
// prolong_lerp_equivalence.rs
//
// the gates for the lerp-then-prolong split:
//
//   - BIT EQUALITY: the two-pass path (field_lerp over the coarse parent
//     region + the single-snapshot prolong) reproduces the fused time-pair
//     prolong kernel bit-for-bit on every fine cell, every order, at a
//     nontrivial alpha. the lerp expression and its consumption are unchanged
//     — only where the intermediate lives moves (register -> coarse cell) —
//     so any difference is a wiring bug.
//
//   - LINEAR EXACTNESS: a field linear in the coarse index with dyadic
//     coefficients prolongs to the exact linear value at each fine sub-cell
//     position (plm: the van-leer slope of equal one-sided differences is the
//     difference itself; ppm: the parabola of linear data is the line, the
//     monotonizer is a no-op, the sub-cell average is the midpoint value).
//     every operation is dyadic-exact, so the assert is ==.
//
//   - PPM CONSERVATION: the ratio^D children of a parent average back to the
//     parent value (the sub-cell averages partition the parent integral).
//
// these laws also gate any replacement pipeline (axis-split sweeps, per-parent
// evaluation): it must pass the same three.
// =============================================================================

use symbi::sim::refinement::ProlongOrder;
use symbi::sim::refinement::transfer::{
    ProlongSweepScratch, prolong_prims, prolong_prims_lerped, prolong_prims_swept,
};
use symbi::sim::state::PrimFieldsGeneric;
use symbi_algebra::{Domain, Space};
use symbi_xpu::HostMemory;

type Prims = PrimFieldsGeneric<3, 3, HostMemory>;

const NCOMP: usize = 5;
const ALPHA: f64 = 0.37;

fn cube(lo: isize, hi: isize) -> Domain<3> {
    Domain::new([
        Space { name: "i", lo, hi },
        Space { name: "j", lo, hi },
        Space { name: "k", lo, hi },
    ])
}

fn comps(p: &Prims) -> [&symbi_grid::Field<f64, 3, HostMemory>; NCOMP] {
    [
        &p.rho,
        &p.vel[0],
        &p.vel[1],
        &p.vel[2],
        p.pre_field().unwrap(),
    ]
}

// fill every allocated cell of every component with f(comp, coord).
fn seed(p: &Prims, f: impl Fn(usize, [isize; 3]) -> f64) {
    for (kk, field) in comps(p).iter().enumerate() {
        for c in field.domain().iter() {
            field.set(c, f(kk, c));
        }
    }
}

// coarse allocated -4..20 per axis; fine region = a negative-lo ghost-slab-like
// box (exercises the euclidean parent division) + fine allocated covering it.
fn coarse_prims() -> Prims {
    Prims::zeros_with_pressure(&cube(-4, 20), true).unwrap()
}

fn fine_prims() -> Prims {
    Prims::zeros_with_pressure(&cube(-4, 36), true).unwrap()
}

fn fine_region() -> Domain<3> {
    Domain::new([
        Space {
            name: "i",
            lo: -4,
            hi: 0,
        },
        Space {
            name: "j",
            lo: -4,
            hi: 36,
        },
        Space {
            name: "k",
            lo: -4,
            hi: 36,
        },
    ])
}

// smooth, comp-distinct, non-symmetric data — nothing the limiters can shortcut.
fn wavy(kk: usize, c: [isize; 3]) -> f64 {
    let (x, y, z) = (c[0] as f64, c[1] as f64, c[2] as f64);
    1.5 + 0.3 * (0.4 * x + 0.1 * kk as f64).sin()
        + 0.2 * (0.3 * y - 0.2 * z).cos()
        + 0.05 * kk as f64
}

#[test]
fn lerp_then_prolong_is_bit_identical_to_the_time_pair_kernel() {
    for order in [ProlongOrder::Pcm, ProlongOrder::Plm, ProlongOrder::Ppm] {
        let (old, new) = (coarse_prims(), coarse_prims());
        seed(&old, wavy);
        seed(&new, |kk, c| {
            wavy(kk, c) + 0.1 * ((c[0] + c[1]) as f64 * 0.2).sin()
        });
        let lerp = coarse_prims();
        let (dst_pair, dst_split) = (fine_prims(), fine_prims());
        let region = fine_region();

        prolong_prims(&old, &new, &dst_pair, &region, order, ALPHA);
        prolong_prims_lerped(&lerp, &old, &new, &dst_split, &region, order, ALPHA);

        for (kk, (a, b)) in comps(&dst_pair).iter().zip(comps(&dst_split)).enumerate() {
            for c in region.iter() {
                let (va, vb) = (*a.view().at(c), *b.view().at(c));
                assert_eq!(
                    va.to_bits(),
                    vb.to_bits(),
                    "{order:?} comp {kk} differs at {c:?}: pair={va:?} split={vb:?}",
                );
            }
        }
    }
}

// the axis-split sweep chain against the fused tensor-product
// kernel: bit identity at every order and a nontrivial alpha. the sweeps
// materialize the SAME per-axis composition through f64 intermediates, so any
// bit difference is a wiring bug (pass order, operand order, a frac spelled
// differently) — never loosen this to a tolerance.
#[test]
fn axis_split_sweeps_are_bit_identical_to_the_fused_kernel() {
    for order in [ProlongOrder::Pcm, ProlongOrder::Plm, ProlongOrder::Ppm] {
        let (old, new) = (coarse_prims(), coarse_prims());
        seed(&old, wavy);
        seed(&new, |kk, c| {
            wavy(kk, c) + 0.1 * ((c[0] + c[1]) as f64 * 0.2).sin()
        });
        let lerp = coarse_prims();
        let (dst_pair, dst_sweep) = (fine_prims(), fine_prims());
        let region = fine_region();
        let scratch = ProlongSweepScratch::for_slab(&region, order, true);

        prolong_prims(&old, &new, &dst_pair, &region, order, ALPHA);
        prolong_prims_swept(
            &scratch, &lerp, &old, &new, &dst_sweep, &region, order, ALPHA,
        );

        for (kk, (a, b)) in comps(&dst_pair).iter().zip(comps(&dst_sweep)).enumerate() {
            for c in region.iter() {
                let (va, vb) = (*a.view().at(c), *b.view().at(c));
                assert_eq!(
                    va.to_bits(),
                    vb.to_bits(),
                    "{order:?} comp {kk} differs at {c:?}: fused={va:?} sweep={vb:?}",
                );
            }
        }
    }
}

#[test]
fn linear_coarse_data_prolongs_exactly() {
    // v(i,j,k) = 1 + 0.25 i - 0.5 j + 0.125 k (+ comp offset): dyadic
    // coefficients, so slope extraction, sub-cell evaluation, and the lerp
    // (old == new makes alpha inert) are all exact in f64.
    let lin = |kk: usize, c: [isize; 3]| -> f64 {
        1.0 + 0.25 * c[0] as f64 - 0.5 * c[1] as f64 + 0.125 * c[2] as f64 + kk as f64
    };
    for order in [ProlongOrder::Plm, ProlongOrder::Ppm] {
        let (old, new) = (coarse_prims(), coarse_prims());
        seed(&old, lin);
        seed(&new, lin);
        let lerp = coarse_prims();
        let dst = fine_prims();
        let region = fine_region();
        prolong_prims_lerped(&lerp, &old, &new, &dst, &region, order, ALPHA);

        for (kk, field) in comps(&dst).iter().enumerate() {
            for c in region.iter() {
                // the fine cell's position in COARSE index coordinates:
                // parent + (parity + 1/2)/2 - 1/2 per axis.
                let pos = |f: isize| -> f64 {
                    let p = f.div_euclid(2) as f64;
                    let q = f.rem_euclid(2) as f64;
                    p + (q + 0.5) * 0.5 - 0.5
                };
                let expect =
                    1.0 + 0.25 * pos(c[0]) - 0.5 * pos(c[1]) + 0.125 * pos(c[2]) + kk as f64;
                let got = *field.view().at(c);
                assert_eq!(
                    got, expect,
                    "{order:?} comp {kk} at {c:?}: linear data must prolong exactly",
                );
            }
        }
    }
}

#[test]
fn ppm_children_average_back_to_the_parent() {
    let (old, new) = (coarse_prims(), coarse_prims());
    seed(&old, wavy);
    seed(&new, wavy);
    let lerp = coarse_prims();
    let dst = fine_prims();
    // an interior box whose children all lie in the fine allocated domain.
    let region = cube(0, 32);
    prolong_prims_lerped(&lerp, &old, &new, &dst, &region, ProlongOrder::Ppm, 0.0);

    for (kk, (fine, coarse)) in comps(&dst).iter().zip(comps(&old)).enumerate() {
        for p in cube(0, 16).iter() {
            let mut sum = 0.0;
            for o in 0..8isize {
                let child = [
                    2 * p[0] + (o & 1),
                    2 * p[1] + ((o >> 1) & 1),
                    2 * p[2] + ((o >> 2) & 1),
                ];
                sum += *fine.view().at(child);
            }
            let (avg, parent) = (sum / 8.0, *coarse.view().at(p));
            assert!(
                (avg - parent).abs() <= 1e-13 * parent.abs().max(1.0),
                "comp {kk} parent {p:?}: children average {avg} != parent {parent}",
            );
        }
    }
}
