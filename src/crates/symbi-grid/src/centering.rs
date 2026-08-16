// =============================================================================
// centering.rs
//
// field-centering marker types for staggered grids.
// the `Centering` trait + `Cell` / `Face` / `Edge` zero-sized markers let
// the type system distinguish cell-centered, face-centered, and edge-centered
// fields; the staggering axis is carried by array indices.
//
// usage:
//   Field<f64, D, Mem>                    -> cell-centered (default)
//   Field<f64, D, Mem, Face>              -> face-centered (axis from array index)
//   Field<f64, D, Mem, Edge>              -> edge-centered (axis from array index)
//
//   [Field<f64, D, Mem, Face>; D]         -> bface group: bface[d] is on axis-d face
//   [Field<f64, D, Mem, Edge>; D]         -> efield group: efield[d] is on axis-d edge
//
// markers are zero-sized phantom tags; no runtime cost. only the type
// signature changes — stencil offsets stay coord-arithmetic.
//
// rationale for axis-erased markers over a per-axis `Face<const AX: usize>`:
// chalkboard kernels need `for cc in 0..D { bface[cc][coord] }` patterns.
// arrays require uniform element types, so `[Field<.., Face<0>>, Field<..,
// Face<1>>, ...]` fails to typecheck. the resolution is one array indexed by
// axis, with no per-axis type tag.
// =============================================================================

/// marker trait for field-centering tags. zero-sized, phantom only.
pub trait Centering: 'static + Copy + Send + Sync {}

/// cell-centered field: value at the geometric center of each cell.
#[derive(Copy, Clone, Debug)]
pub struct Cell;
impl Centering for Cell {}

/// face-centered field: value at the centroid of a cell face.
/// the specific axis of staggering is carried by the array index in the
/// owning FieldGroup.
#[derive(Copy, Clone, Debug)]
pub struct Face;
impl Centering for Face {}

/// edge-centered field: value at the midpoint of a cell edge.
/// the specific axis the edge is parallel to is carried by the array index
/// in the owning FieldGroup.
#[derive(Copy, Clone, Debug)]
pub struct Edge;
impl Centering for Edge {}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_centering<C: Centering>() {}

    #[test]
    fn cell_is_centering() {
        assert_centering::<Cell>();
    }

    #[test]
    fn face_is_centering() {
        assert_centering::<Face>();
    }

    #[test]
    fn edge_is_centering() {
        assert_centering::<Edge>();
    }

    #[test]
    fn markers_are_zero_sized() {
        assert_eq!(std::mem::size_of::<Cell>(), 0);
        assert_eq!(std::mem::size_of::<Face>(), 0);
        assert_eq!(std::mem::size_of::<Edge>(), 0);
    }
}
