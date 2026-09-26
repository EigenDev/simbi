// =============================================================================
// emf_register_contribution.rs
//
// an EMF-register slab split across a cut agrees bit for bit with the same
// slab on one grid, and corrects the same faces with the same bits: the tile
// holding the fine sub-edges accumulates each fine substep's unit into a
// contributing slab, the unit crosses through the plan's packing (gather
// into a wire buffer, scatter into the owner's inbox), and the owner folds it
// after its own coarse sample, in the substep order of the single grid. the
// box-corner edges of the cut plane lie in the transported slab and in the
// transverse planes' local slabs, each copy holding the identical value.
//
// run: cargo test -p symbi --test emf_register_contribution
// =============================================================================
use symbi::sim::refinement::EmfRegister;
use symbi::sim::state::{BfaceFields, EfieldFields, axis_name};
use symbi_algebra::{Domain, Space};
use symbi_grid::Field;
use symbi_sim::atlas::Span;
use symbi_sim::plan_exchange::{gather, scatter};
use symbi_xpu::HostMemory;

type Register = EmfRegister<2, HostMemory>;

const N: isize = 32;
const RATIO: isize = 2;
/// the patch in root cells; its hi face on axis 0 is the cut of the split arm
const PATCH: [(isize, isize); 2] = [(12, 20), (12, 20)];
const CUT: isize = 20;
const DX: f64 = 1.0 / N as f64;
const DT: f64 = 0.01;
/// the corner EMF of a 2.5D grid runs along the missing axis
const EDGE_AXIS: usize = 2;

fn domain(lo: [isize; 2], hi: [isize; 2]) -> Domain<2> {
    Domain::new(std::array::from_fn(|a| Space { name: axis_name(a), lo: lo[a], hi: hi[a] }))
}

/// a deterministic EMF at an absolute node
fn emf_value(x: isize, y: isize, sample: usize) -> f64 {
    let (x, y) = (x as f64, y as f64);
    0.41 * ((0.17 * x + 0.23 * y + 0.61 * sample as f64).sin() + 1.3)
}

/// the corner EMF over the node domain `nodes`, holding `emf_value` at the absolute node
/// `offset + local`
fn efield(nodes: &Domain<2>, offset: [isize; 2], sample: usize) -> EfieldFields<2, HostMemory> {
    let e: [Field<f64, 2, HostMemory>; 2] = std::array::from_fn(|_| {
        let f = Field::zeros(nodes).expect("efield");
        for c in nodes.iter() {
            f.view_mut().set(c, emf_value(c[0] + offset[0], c[1] + offset[1], sample));
        }
        f
    });
    EfieldFields { e }
}

/// staggered faces over the interior `lo..hi` with `axis`-faces one node past the interior,
/// holding a deterministic value at the absolute face
fn bfaces(lo: [isize; 2], hi: [isize; 2], offset: [isize; 2]) -> BfaceFields<2, HostMemory> {
    let b: [Field<f64, 2, HostMemory>; 2] = std::array::from_fn(|axis| {
        let dom = domain(lo, std::array::from_fn(|a| if a == axis { hi[a] + 1 } else { hi[a] }));
        let f = Field::zeros(&dom).expect("bface");
        for c in dom.iter() {
            let (x, y) = ((c[0] + offset[0]) as f64, (c[1] + offset[1]) as f64);
            f.view_mut().set(c, 0.8 + 0.1 * (0.31 * x - 0.19 * y + axis as f64).cos());
        }
        f
    });
    BfaceFields { b }
}

fn slab_bits(reg: &Register, bb: usize, side: usize) -> Vec<u64> {
    let (field, dom) = reg.slab_of(EDGE_AXIS, bb, side).expect("the slab exists");
    dom.iter().map(|c| field.view().at(c).to_bits()).collect()
}

#[test]
fn a_contribution_folded_by_its_owner_equals_the_local_accumulation_and_corrects_the_same_faces() {
    // one grid: the root interior [0, 32)^2, the patch [12, 20)^2
    let root = domain([0, 0], [N, N]);
    let coverage = domain([PATCH[0].0, PATCH[1].0], [PATCH[0].1, PATCH[1].1]);
    let mono = Register::new(&coverage, &root).expect("mono register");
    // the split: the left tile's interior [0, 20) x [0, 32) ends at the patch's hi face on
    // axis 0; the right tile's interior starts there, local index 0 at absolute 20, and the
    // patch lies in its frame at [-8, 0) x [12, 20)
    let left_root = domain([0, 0], [CUT, N]);
    let sender = Register::new(&coverage, &left_root)
        .expect("sender register")
        .with_contributing_planes(&coverage, [[false, true], [false, false]])
        .expect("contributing plane");
    let owner_patch = domain([PATCH[0].0 - CUT, PATCH[1].0], [PATCH[0].1 - CUT, PATCH[1].1]);
    let owner = Register::receiving(&owner_patch, [[false, true], [false, false]]).expect("owner register");

    // the root's corner EMF on the single grid and on the owner, the same absolute values
    let mono_coarse = efield(&domain([0, 0], [N + 1, N + 1]), [0, 0], 0);
    let owner_coarse = efield(&domain([0, 0], [N - CUT + 1, N + 1]), [CUT, 0], 0);
    // the fine level's corner EMF per substep: the fine index space is the same on the
    // single grid and on the left tile
    let fine_nodes = domain([RATIO * PATCH[0].0, RATIO * PATCH[1].0], [RATIO * PATCH[0].1 + 1, RATIO * PATCH[1].1 + 1]);
    let fine: Vec<EfieldFields<2, HostMemory>> = (1..=2).map(|s| efield(&fine_nodes, [0, 0], s)).collect();

    // the root tail: the coarse sample, once per root step
    mono.zero();
    sender.zero();
    owner.zero();
    mono.accumulate_coarse(&mono_coarse, DT);
    owner.accumulate_coarse(&owner_coarse, DT);
    sender.accumulate_coarse(&mono_coarse, DT);
    // the fine substeps: the single grid accumulates in place; the sender's contributing
    // slab holds the substep's unit alone, which crosses the wire and is folded by the owner
    let fine_dt = DT / RATIO as f64;
    let (_, sender_dom) = sender.slab_of(EDGE_AXIS, 0, 1).expect("contributing slab");
    let src_region: [Span; 2] = std::array::from_fn(|a| Span { lo: sender_dom.spaces[a].lo, hi: sender_dom.spaces[a].hi });
    let inbox_dom = owner.inboxes()[0].2.clone();
    let dst_region: [Span; 2] = std::array::from_fn(|a| Span { lo: inbox_dom.spaces[a].lo, hi: inbox_dom.spaces[a].hi });
    let mut wire = vec![0.0; sender_dom.volume()];
    for substep in &fine {
        mono.accumulate_fine(substep, fine_dt);
        sender.accumulate_fine(substep, fine_dt);
        let contributing = sender.contributing();
        let inboxes = owner.inboxes();
        assert_eq!((contributing.len(), inboxes.len()), (1, 1));
        assert_eq!(contributing[0].0, inboxes[0].0, "the sender's slot names the owner's inbox");
        gather(contributing[0].1, &src_region, &mut wire);
        assert!(wire.iter().any(|v| *v != 0.0), "the substep's unit is zero; the gate is vacuous");
        scatter(inboxes[0].1, &dst_region, &wire);
        sender.zero_contributions();
        owner.fold_inboxes();
    }
    assert!(slab_bits(&sender, 0, 1).iter().all(|b| *b == 0), "the contributing slab is cleared after its unit left");
    // a rejected attempt discards a landed unit: the inbox is cleared without a fold and the
    // register keeps its value
    let before = slab_bits(&owner, 0, 1);
    scatter(owner.inboxes()[0].1, &dst_region, &wire);
    owner.clear_inboxes();
    assert!(owner.inboxes()[0].2.iter().all(|c| *owner.inboxes()[0].1.view().at(c) == 0.0), "the inbox is cleared");
    assert_eq!(before, slab_bits(&owner, 0, 1), "clearing the inbox leaves the register as it was");

    // the transported slab against the single grid's, node for node
    let expected = slab_bits(&mono, 0, 1);
    let got = slab_bits(&owner, 0, 1);
    assert_eq!(expected.len(), got.len());
    assert!(expected.iter().any(|b| f64::from_bits(*b) != 0.0), "the register is zero; the gate is vacuous");
    let differing = expected.iter().zip(&got).filter(|(a, b)| a != b).count();
    assert_eq!(differing, 0, "{differing} of {} register values differ between the owner and the single grid", got.len());

    // the box corners of the cut plane lie in the transverse planes' local slabs of the
    // sender with the identical value
    for (corner_y, side) in [(PATCH[1].0, 0usize), (PATCH[1].1, 1usize)] {
        let (xhi, _) = mono.slab_of(EDGE_AXIS, 0, 1).unwrap();
        let (ylocal, _) = sender.slab_of(EDGE_AXIS, 1, side).unwrap();
        let corner = [CUT, corner_y];
        assert_eq!(
            xhi.view().at(corner).to_bits(),
            ylocal.view().at(corner).to_bits(),
            "the corner edge at y = {corner_y} differs between the transported slab and the local transverse slab"
        );
        assert_ne!(*xhi.view().at(corner), 0.0, "the corner edge holds zero; the gate is vacuous");
    }

    // the owner's outside faces: the y-faces of the cells just beyond the plane, corrected
    // with the same bits as the single grid's
    let inv_dx = [1.0 / DX; 2];
    let mono_faces = bfaces([0, 0], [N, N], [0, 0]);
    let owner_faces = bfaces([0, 0], [N - CUT, N], [CUT, 0]);
    mono.apply(&mono_faces, &inv_dx);
    owner.apply(&owner_faces, &inv_dx);
    let mut corrected = 0usize;
    for y in PATCH[1].0..=PATCH[1].1 {
        let want = mono_faces[1].view().at([CUT, y]).to_bits();
        let got = owner_faces[1].view().at([0, y]).to_bits();
        assert_eq!(want, got, "the y-face at absolute ({CUT}, {y}) differs after the apply");
        let before = 0.8 + 0.1 * (0.31 * CUT as f64 - 0.19 * y as f64 + 1.0).cos();
        corrected += usize::from(f64::from_bits(want) != before);
    }
    assert_eq!(corrected, (PATCH[1].1 - PATCH[1].0 + 1) as usize, "every outside face was corrected");
}

/// the same split in 3D, where a coarse edge covers two fine sub-edges of unequal EMF and the
/// unit is their length average times the fine timestep: every edge family with edges in
/// the cut plane crosses, and the transported slabs equal the single grid's
#[test]
fn a_3d_contribution_averages_unequal_sub_edges_like_the_single_grid() {
    type Register3 = EmfRegister<3, HostMemory>;
    const NZ: isize = 8;
    const PZ: (isize, isize) = (2, 6);
    let domain3 = |lo: [isize; 3], hi: [isize; 3]| Domain::new(std::array::from_fn(|a| Space { name: axis_name(a), lo: lo[a], hi: hi[a] }));
    let root = domain3([0, 0, 0], [N, N, NZ]);
    let coverage = domain3([PATCH[0].0, PATCH[1].0, PZ.0], [PATCH[0].1, PATCH[1].1, PZ.1]);
    let mono = Register3::new(&coverage, &root).expect("mono register");
    let left_root = domain3([0, 0, 0], [CUT, N, NZ]);
    let faces = [[false, true], [false, false], [false, false]];
    let sender = Register3::new(&coverage, &left_root).expect("sender").with_contributing_planes(&coverage, faces).expect("planes");
    let owner_patch = domain3([PATCH[0].0 - CUT, PATCH[1].0, PZ.0], [PATCH[0].1 - CUT, PATCH[1].1, PZ.1]);
    let owner = Register3::receiving(&owner_patch, faces).expect("owner");
    // an edge EMF field per direction over the level's edge nodes: distinct along every
    // axis, so the two sub-edges of a coarse edge differ
    let edge_field = |lo: [isize; 3], hi: [isize; 3], offset: [isize; 3], sample: usize| -> EfieldFields<3, HostMemory> {
        let e: [Field<f64, 3, HostMemory>; 3] = std::array::from_fn(|d| {
            let dom = domain3(lo, std::array::from_fn(|a| if a == d { hi[a] } else { hi[a] + 1 }));
            let f = Field::zeros(&dom).expect("efield");
            for c in dom.iter() {
                let (x, y, z) = ((c[0] + offset[0]) as f64, (c[1] + offset[1]) as f64, (c[2] + offset[2]) as f64);
                f.view_mut().set(c, 0.3 * ((0.19 * x + 0.23 * y + 0.37 * z + 0.5 * d as f64 + 0.7 * sample as f64).sin() + 1.2));
            }
            f
        });
        EfieldFields { e }
    };
    let mono_coarse = edge_field([0, 0, 0], [N, N, NZ], [0, 0, 0], 0);
    let owner_coarse = edge_field([0, 0, 0], [N - CUT, N, NZ], [CUT, 0, 0], 0);
    let fine: Vec<EfieldFields<3, HostMemory>> = (1..=2)
        .map(|s| edge_field([RATIO * PATCH[0].0, RATIO * PATCH[1].0, RATIO * PZ.0], [RATIO * PATCH[0].1, RATIO * PATCH[1].1, RATIO * PZ.1], [0, 0, 0], s))
        .collect();
    mono.zero();
    sender.zero();
    owner.zero();
    mono.accumulate_coarse(&mono_coarse, DT);
    owner.accumulate_coarse(&owner_coarse, DT);
    let fine_dt = DT / RATIO as f64;
    for substep in &fine {
        mono.accumulate_fine(substep, fine_dt);
        sender.accumulate_fine(substep, fine_dt);
        let contributing = sender.contributing();
        let inboxes = owner.inboxes();
        assert_eq!(contributing.len(), 2, "two edge families have edges in the cut plane");
        for (slot, field, dom) in &contributing {
            let (_, inbox, inbox_dom) = inboxes.iter().find(|(s, _, _)| s == slot).expect("the owner's inbox of the slot");
            let src: [Span; 3] = std::array::from_fn(|a| Span { lo: dom.spaces[a].lo, hi: dom.spaces[a].hi });
            let dst: [Span; 3] = std::array::from_fn(|a| Span { lo: inbox_dom.spaces[a].lo, hi: inbox_dom.spaces[a].hi });
            let mut wire = vec![0.0; dom.volume()];
            gather(field, &src, &mut wire);
            assert!(wire.iter().any(|v| *v != 0.0), "the unit is zero; the gate is vacuous");
            scatter(inbox, &dst, &wire);
        }
        sender.zero_contributions();
        owner.fold_inboxes();
    }
    // the sub-edges differ: the unit is their average, so it equals neither sub-edge scaled
    let (ef, _) = (&fine[0].e[1], ());
    let (a, b) = (*ef.view().at([2 * CUT, 2 * PATCH[1].0 + 1, 2 * PZ.0]), *ef.view().at([2 * CUT, 2 * PATCH[1].0 + 1, 2 * PZ.0 + 1]));
    assert_ne!(a.to_bits(), b.to_bits(), "the two sub-edges of a coarse edge hold one value; the average is untested");
    // the owner's slab lies in its own index space: the two slabs are walked in lockstep
    for tt in [1usize, 2usize] {
        let (mono_reg, dom) = mono.slab_of(tt, 0, 1).expect("mono slab");
        let (owner_reg, owner_dom) = owner.slab_of(tt, 0, 1).expect("owner slab");
        assert_eq!(dom.volume(), owner_dom.volume(), "edge family {tt}: the slabs differ in shape");
        let differing = dom
            .iter()
            .zip(owner_dom.iter())
            .filter(|(m, o)| mono_reg.view().at(*m).to_bits() != owner_reg.view().at(*o).to_bits())
            .count();
        assert_eq!(differing, 0, "edge family {tt}: {differing} register values differ between the owner and the single grid");
        assert!(dom.iter().any(|c| *mono_reg.view().at(c) != 0.0), "edge family {tt}: the register is zero; the gate is vacuous");
    }
}
