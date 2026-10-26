// =============================================================================
// flux_register_contribution.rs
//
// a flux-register face split across a cut agrees bit for bit with the same
// face on one grid: the tile holding the fine faces accumulates each fine
// stage's unit into a contributing face, the unit crosses through the plan's
// packing (gather into a wire buffer, scatter into the owner's inbox), and the
// owner folds it into its receiving face after its own coarse accumulations,
// in the stage order of the single grid. the unit carries the stage's unscaled
// fine-face sums and the coefficient `area * w`, and the owner folds
// `R + coefficient * S`: the operation and the operands of the single grid's
// accumulation. the register then corrects the same cell with the same bits. a
// second accumulation into a unit before it is sent is refused.
//
// run: cargo test -p symbi --test flux_register_contribution
// =============================================================================
use symbi::sim::refinement::FluxRegister;
use symbi::sim::state::{ConsFieldsGeneric, axis_name};
use symbi_algebra::{Domain, Space};
use symbi_sim::atlas::Span;
use symbi_sim::plan_exchange::{gather, scatter};
use symbi_xpu::HostMemory;

type Cons = ConsFieldsGeneric<2, 2, HostMemory>;
type Register = FluxRegister<2, 2, HostMemory>;

const N: usize = 32;
const NG: isize = 3;
const RATIO: isize = 2;
/// the patch in root cells; its hi face on axis 0 is the cut of the split arm
const PATCH: [(isize, isize); 2] = [(12, 20), (12, 20)];
const CUT: isize = 20;
const DX: f64 = 1.0 / N as f64;
const DT: f64 = 0.01;

fn domain(lo: [isize; 2], hi: [isize; 2]) -> Domain<2> {
    Domain::new(std::array::from_fn(|a| Space { name: axis_name(a), lo: lo[a], hi: hi[a] }))
}

/// a level's allocated domain: `interior` with `ng` ghosts on every side
fn allocated(interior_lo: [isize; 2], interior_hi: [isize; 2], ng: isize) -> Domain<2> {
    domain([interior_lo[0] - ng, interior_lo[1] - ng], [interior_hi[0] + ng, interior_hi[1] + ng])
}

/// a deterministic flux value at an absolute face, distinct per component and direction; a
/// uniform flux is the same at every face and stage, so the coarse and fine accumulations of
/// a register face cancel to their rounding
fn flux_value(comp: usize, dir: usize, x: isize, y: isize, stage: usize, uniform: bool) -> f64 {
    if uniform {
        return 0.1 + 0.25 * comp as f64 + 0.5 * dir as f64;
    }
    let (x, y) = (x as f64, y as f64);
    0.37 * ((0.13 * x + 0.29 * y + comp as f64 + 0.5 * dir as f64 + 0.71 * stage as f64).sin() + 1.7)
}

/// flux fields over `alloc` for both directions, holding `flux_value` at the absolute face
/// `offset + local`
fn flux_fields(alloc: &Domain<2>, offset: [isize; 2], stage: usize, uniform: bool) -> [Cons; 2] {
    std::array::from_fn(|dir| {
        let f = Cons::zeros_with_energy(alloc, true).expect("flux fields");
        let fields = [&f.den, &f.mom[0], &f.mom[1], f.nrg_field().unwrap()];
        for (comp, field) in fields.iter().enumerate() {
            for c in alloc.iter() {
                field.view_mut().set(c, flux_value(comp, dir, c[0] + offset[0], c[1] + offset[1], stage, uniform));
            }
        }
        f
    })
}

fn face_bits(reg: &Register, ax: usize, side: usize) -> Vec<u64> {
    let (fields, dom) = reg.face(ax, side).expect("the face exists");
    let mut out = Vec::new();
    for field in fields {
        for c in dom.iter() {
            out.push(field.view().at(c).to_bits());
        }
    }
    out
}

/// stage weights of a two-stage scheme: the first stage's flux enters the recombination
/// with half weight, the second with half
const WEIGHTS: [f64; 2] = [0.5, 0.5];

#[test]
fn a_contribution_folded_by_its_owner_equals_the_local_accumulation_bit_for_bit() {
    assert_owner_matches(WEIGHTS, false);
}

/// stages of distinct weights: every unit carries its own coefficient, folded in stage order
#[test]
fn units_of_distinct_coefficients_fold_in_stage_order_as_on_the_single_grid() {
    assert_owner_matches([0.3, 0.7], false);
}

/// uniform fluxes: the coarse and fine accumulations cancel, and the owner keeps the single
/// grid's rounding residue bit for bit
#[test]
fn cancelling_accumulations_leave_the_single_grids_residue_on_the_owner() {
    assert_owner_matches([0.3, 0.7], true);
}

fn assert_owner_matches(weights: [f64; 2], uniform: bool) {
    // one grid: the root interior [0, 32)^2, the patch [12, 20)^2, the fine level [24, 40)^2
    let root = domain([0, 0], [N as isize, N as isize]);
    let coverage = domain([PATCH[0].0, PATCH[1].0], [PATCH[0].1, PATCH[1].1]);
    let mono = Register::new(&coverage, &root, true, false, 0).expect("mono register");
    // the split: the left tile's interior [0, 20) x [0, 32) ends at the patch's hi face;
    // the right tile's interior starts there, local index 0 at absolute 20
    let left_root = domain([0, 0], [CUT, N as isize]);
    let sender = Register::new(&coverage, &left_root, true, false, 0)
        .expect("sender register")
        .with_contributing_faces(&coverage, [[false, true], [false, false]], true, false, 0)
        .expect("contributing face");
    let owner_slab = domain([0, PATCH[1].0], [1, PATCH[1].1]);
    let owner = Register::receiving(&[(0, 1, owner_slab)], true, false, 0).expect("owner register");

    let dx = [DX; 2];
    let fdx = [DX / RATIO as f64; 2];
    // the coarse fluxes at the root's own faces: the same absolute values on the single grid
    // and on the owner, whose local index is offset by the cut
    let mono_coarse: Vec<[Cons; 2]> = (0..2).map(|ii| flux_fields(&allocated([0, 0], [N as isize, N as isize], NG), [0, 0], ii, uniform)).collect();
    let owner_coarse: Vec<[Cons; 2]> =
        (0..2).map(|ii| flux_fields(&allocated([0, 0], [N as isize - CUT, N as isize], NG), [CUT, 0], ii, uniform)).collect();
    // the fine fluxes: the fine level's index space is the same on the single grid and on
    // the left tile
    let fine_alloc = allocated([RATIO * PATCH[0].0, RATIO * PATCH[1].0], [RATIO * PATCH[0].1, RATIO * PATCH[1].1], NG + 1);
    let fine: Vec<[Cons; 2]> = (0..4).map(|s| flux_fields(&fine_alloc, [0, 0], 10 + s, uniform)).collect();

    mono.zero_uniform();
    sender.zero_uniform();
    owner.zero_uniform();
    // the root stages: coarse accumulations on the single grid and on the owner
    for ii in 0..2 {
        for dir in 0..2 {
            mono.accumulate_coarse_uniform(&mono_coarse[ii], &dx, dir, weights[ii] * DT);
            owner.accumulate_coarse_uniform(&owner_coarse[ii], &dx, dir, weights[ii] * DT);
            sender.accumulate_coarse_uniform(&mono_coarse[ii], &dx, dir, weights[ii] * DT);
        }
    }
    // the fine substeps and stages: the single grid accumulates in place; the sender
    // accumulates the stage's unit alone, which crosses the wire and is folded by the owner
    let fine_dt = DT / RATIO as f64;
    let sender_face = sender.face(0, 1).expect("contributing face").1.clone();
    let src_region: [Span; 2] = std::array::from_fn(|a| Span { lo: sender_face.spaces[a].lo, hi: sender_face.spaces[a].hi });
    let dst_region: [Span; 2] = std::array::from_fn(|a| Span { lo: owner.inboxes()[0].2.spaces[a].lo, hi: owner.inboxes()[0].2.spaces[a].hi });
    let mut wire = vec![0.0; sender_face.volume()];
    let mut units_seen = 0usize;
    for (s, stage_flux) in fine.iter().enumerate() {
        let jj = s % 2;
        for dir in 0..2 {
            mono.accumulate_fine_uniform(stage_flux, &fdx, dir, weights[jj] * fine_dt);
            sender.accumulate_fine_uniform(stage_flux, &fdx, dir, weights[jj] * fine_dt);
        }
        let contributing = sender.contributing();
        assert_eq!(contributing.len(), 1);
        // the unit: each gas slot the stage's unscaled fine-face sum, in the kernel's order, and
        // the coefficient slot the stage's `area * w`
        let (slots, face_dom) = (&contributing[0].1, contributing[0].2);
        assert_eq!(slots.len(), 4 + 1, "four gas slots and the coefficient");
        let coefficient = fdx[1] * (weights[jj] * fine_dt);
        for c in face_dom.iter() {
            let (x, y) = (RATIO * c[0], RATIO * c[1]);
            for comp in 0..4 {
                let sum = flux_value(comp, 0, x, y, 10 + s, uniform) + flux_value(comp, 0, x, y + 1, 10 + s, uniform);
                assert_eq!(slots[comp].view().at(c).to_bits(), sum.to_bits(), "stage {s} comp {comp} at {c:?}: the unit is not the unscaled sum");
            }
            assert_eq!(slots[4].view().at(c).to_bits(), coefficient.to_bits(), "stage {s} at {c:?}: the coefficient");
        }
        let inboxes = owner.inboxes();
        for (src, dst) in contributing[0].1.iter().zip(&inboxes[0].1) {
            gather(src, &src_region, &mut wire);
            assert!(wire.iter().any(|v| *v != 0.0), "the stage's unit is zero; the gate is vacuous");
            scatter(dst, &dst_region, &wire);
            units_seen += 1;
        }
        sender.zero_contributions();
        owner.fold_inboxes();
    }
    assert_eq!(units_seen, 4 * 5, "four stages of four components and a coefficient crossed");
    // a cleared unit holds -0.0, the additive identity of the unscaled accumulation
    assert!(face_bits(&sender, 0, 1).iter().all(|b| *b == (-0.0f64).to_bits()), "the contributing face is reset to -0.0 after its unit left");
    // a rejected attempt discards a landed unit: the inbox is cleared without a fold and the
    // register keeps its value
    let before = face_bits(&owner, 0, 1);
    for dst in &owner.inboxes()[0].1 {
        scatter(dst, &dst_region, &wire);
    }
    owner.clear_inboxes();
    for dst in &owner.inboxes()[0].1 {
        assert!(owner.inboxes()[0].2.iter().all(|c| *dst.view().at(c) == 0.0), "the inbox is cleared");
    }
    assert_eq!(before, face_bits(&owner, 0, 1), "clearing the inbox leaves the register as it was");

    let expected = face_bits(&mono, 0, 1);
    let got = face_bits(&owner, 0, 1);
    assert_eq!(expected.len(), got.len());
    if uniform {
        // the cancellation: every register value within a few roundings of the accumulated terms,
        // dt * area * flux <= 0.01 * (1 / 32) * 1.6
        let bound = 8.0 * f64::EPSILON * DT * DX * 1.6;
        assert!(expected.iter().all(|b| f64::from_bits(*b).abs() <= bound), "the uniform fluxes did not cancel; the gate is not a cancellation");
    } else {
        assert!(expected.iter().any(|b| f64::from_bits(*b) != 0.0), "the register is zero; the gate is vacuous");
    }
    let differing = expected.iter().zip(&got).filter(|(a, b)| a != b).count();
    assert_eq!(differing, 0, "{differing} of {} register values differ between the owner and the single grid", got.len());
}

#[test]
#[should_panic(expected = "a second accumulation into the transported unit")]
fn a_second_accumulation_into_an_unsent_unit_is_refused() {
    let root = domain([0, 0], [CUT, N as isize]);
    let coverage = domain([PATCH[0].0, PATCH[1].0], [PATCH[0].1, PATCH[1].1]);
    let sender = Register::new(&coverage, &root, true, false, 0)
        .expect("sender register")
        .with_contributing_faces(&coverage, [[false, true], [false, false]], true, false, 0)
        .expect("contributing face");
    let fine_alloc = allocated([RATIO * PATCH[0].0, RATIO * PATCH[1].0], [RATIO * PATCH[0].1, RATIO * PATCH[1].1], NG + 1);
    let flux = flux_fields(&fine_alloc, [0, 0], 0, false);
    let fdx = [DX / RATIO as f64; 2];
    sender.zero_contributions();
    sender.accumulate_fine_uniform(&flux, &fdx, 0, 0.5 * DT);
    // a second event of the same kind, as an equilibrium flux would add, before the unit is sent
    sender.accumulate_fine_uniform(&flux, &fdx, 0, -0.5 * DT);
}
