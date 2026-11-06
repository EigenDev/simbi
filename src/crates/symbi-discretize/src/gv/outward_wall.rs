// =============================================================================
// gv/outward_wall.rs
//
// the outward-only wall: a domain face that passes outflow and refuses inflow. it runs after the
// stage's flux, over the face plane of one wall, and acts on each face by the sign of the mass flux
// the riemann solver computed there:
// - an outward mass flux keeps every component of the face flux, so outflow leaves as through the
//   zero-gradient wall the ghost fill also applies
// - an inward mass flux takes the solid-wall flux in its place: no mass, no energy and no
//   tangential momentum cross the face, and the normal momentum flux is the pressure of the
//   interior cell beside it
// the ghost cells keep the zero-gradient fill, so the reconstructions at interior faces read the
// same states as under an outflow wall. the viscous stress at the wall follows its own face
// condition and is independent of this choice.
//
// buffers: the mass, energy and every momentum flux on the wall's axis, read and rewritten in
// place at the face, and the primitive pressure of the adjacent interior cell. the dispatch runs
// over the face index on the wall's axis: the interior's first index on a low wall, the index past
// its last on a high wall, whose adjacent cell lies one index back.
// =============================================================================
use symbi_algebra::algebra::Numeric;
use symbi_carrier::Scalar;
use symbi_ir::{ElementWiseOp, FieldRef, Gv, KernelProgram, KernelWrite, NodeId, trace_kernel};

use super::gv_load_at;

/// the outward-only wall on grid axis `axis`, the high wall when `high`; `normal_slot` is the
/// momentum component along the axis and `ncomp` the momentum count
pub fn outward_wall_gv(ndim: usize, ncomp: usize, axis: usize, high: bool, normal_slot: usize) -> KernelProgram {
    assert!(axis < ndim && normal_slot < ncomp, "outward_wall_gv: axis {axis} of {ndim}, slot {normal_slot} of {ncomp}");
    trace_kernel(|cx| {
        let ax = axis as u8;
        let mass = cx.field("f_rho", FieldRef::MassFlux(ax));
        let nrg = cx.field("f_nrg", FieldRef::NrgFlux(ax));
        let mom: Vec<Gv> = (0..ncomp).map(|k| cx.field(&format!("f_mom_{k}"), FieldRef::MomFlux { comp: k as u8, axis: ax })).collect();
        // the interior cell beside the face: the cell itself on a low wall, one index back on a high one
        let coords: Vec<NodeId> = cx.with_trace(|t| (0..ndim).map(|d| t.coord(d as u8)).collect());
        let mut beside = coords.clone();
        if high {
            let one = cx.with_trace(|t| t.graph().add_const(symbi_ir::ConstValue::I32(1), None));
            beside[axis] = cx.with_trace(|t| t.graph().element_wise(ElementWiseOp::Sub, vec![coords[axis], one], None));
        }
        let pressure = gv_load_at(cx, "pre_beside", FieldRef::PrimPre, &beside);
        let outward = if high { mass } else { -mass };
        let inward = outward.cmp_lt(Gv::ZERO);
        let zero = Gv::ZERO;
        let mut writes = vec![
            KernelWrite::new("f_rho_new", FieldRef::MassFlux(ax), Gv::select(inward, zero, mass).node()),
            KernelWrite::new("f_nrg_new", FieldRef::NrgFlux(ax), Gv::select(inward, zero, nrg).node()),
        ];
        for (k, m) in mom.iter().enumerate() {
            let wall = if k == normal_slot { pressure } else { zero };
            writes.push(KernelWrite::new(format!("f_mom_{k}_new"), FieldRef::MomFlux { comp: k as u8, axis: ax }, Gv::select(inward, wall, *m).node()));
        }
        writes
    })
}
