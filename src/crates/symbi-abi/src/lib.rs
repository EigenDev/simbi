// =============================================================================
// symbi-abi
//
// the typed trace <-> dispatch ABI vocabulary: the born-typed names for kernel
// field buffers (`FieldRef`/`FieldBind`), scalar params (`ScalarRef`/`ScalarBind`/
// `BodyScalar`), and mesh scalars (`MeshScalar`). each name is minted in exactly
// one place (`name()`) and recovered by `parse()`, so a producer (the trace) and a
// consumer (the dispatch) agree on a spelling — drift is the failure mode the
// typed ABI was built to kill.
//
// this crate is a leaf (serde only). it holds the closed domain vocabulary so the
// graph-theoretic IR (symbi-ir) can carry the typed containers (`FieldBind` lives
// in `Prepared`/`GvKernel`) while every hydro field name is spelled here, once.
//
// usage:
//  let den = symbi_abi::FieldRef::cons_den();
//  let bind = symbi_abi::FieldBind::from_path("cons.mom_1");
// =============================================================================

pub mod ct_scratch;
pub mod field_ref;
pub mod scalar_param;
pub mod scalar_ref;

pub use ct_scratch::{
    CtCellCt, CtEdgeCt, CtFaceCt, CtScratch, CtScratchKey, CtWireName, GridAxis, PhysComp,
    PlaneComp, ScratchKey, SweepAxis, Transverse,
};
pub use field_ref::{FieldBind, FieldRef, StateComp, StateSlot};
pub use scalar_param::MeshScalar;
pub use scalar_ref::{BodyScalar, ScalarBind, ScalarRef};
