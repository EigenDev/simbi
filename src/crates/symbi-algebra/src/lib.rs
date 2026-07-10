// =============================================================================
// symbi-algebra
//
// pure mathematical foundations for the symbi framework. zero runtime deps.
// provides algebraic traits, domain geometry, tensor types, and field element
// markers that downstream crates (symbi, symbi-geometry, symbi-hydro) build on.
//
// usage:
//   use symbi_algebra::{Domain, Tensor, FieldElement, domain, index};
//   // production `Scalar` / `Selectable` live in `symbi_ir::algebra`.
// =============================================================================

pub mod algebra;
pub mod block;
pub mod boundary;
pub mod domain;
pub mod element;
pub mod layout;
pub mod matrix;
pub mod tensor;
pub mod variance;

pub use algebra::{Numeric, OrderedNumeric};
pub use block::BlockGrid;
pub use boundary::{self as bc, IndexMap};
pub use element::FieldElement;
pub use domain::{Domain, DomainId, Space, Side, Axis, IntoAxis, domain, index, IndexName, IntoRange, Split};
pub use layout::{flat_offset, nest_order, strides_from_extent, unflatten, Layout, CONTIGUOUS_AXIS};
pub use matrix::{Matrix, outer};
pub use tensor::{Tensor, dot, cross, norm, normalize, vec2, vec3, vec4, VecN, Vec2, Vec3, Vec4};
pub use variance::{
    Indexed, Upper, Lower, Ortho, Cart, Contravariant, Covariant, Physical, Embedded, contract,
};
