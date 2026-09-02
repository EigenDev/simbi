// =============================================================================
// symbi-algebra
//
// pure mathematical foundations for the symbi framework. zero runtime deps.
// provides algebraic traits, domain geometry, tensor types, and field element
// markers that downstream crates (symbi, symbi-geometry, symbi-hydro) build on.
//
// usage:
//   use symbi_algebra::{Domain, Tensor, FieldElement, domain, index};
//   // production `Scalar` / `Selectable` live in `symbi_carrier`.
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
pub use domain::{
    Axis, Domain, DomainId, IndexName, IntoAxis, IntoRange, Side, Space, Split, domain, index,
};
pub use element::FieldElement;
pub use layout::{
    CONTIGUOUS_AXIS, Layout, flat_offset, nest_order, strides_from_extent, unflatten,
};
pub use matrix::{Matrix, outer};
pub use tensor::{Tensor, Vec2, Vec3, Vec4, VecN, cross, dot, norm, normalize, vec2, vec3, vec4};
pub use variance::{
    Cart, Contravariant, Covariant, Embedded, Indexed, Lower, Ortho, Physical, Upper, contract,
};
