// =============================================================================
// embedding.rs
//
// the cartesian embedding of a chart position and the cartesian axes a body's
// position components occupy on a chart. bodies live in cartesian space: a
// body's signed distance and its gravity are evaluated at the cartesian image of
// a cell's position, and a body's d-component position names the cartesian
// axes of the grid's plane (the (r, theta) and (R, z) planes are the x-z plane).
//
// carrier-generic: the same expression serves the host (S = f64, the census
// masks) and the traced immersed-body kernels (S = Gv), so both place a cell and
// a body identically by construction.
//
// usage:
//  let x = chart_to_cartesian(Geometry::Spherical, [r, theta, phi]);
//  let cart = body_cartesian_axes(Geometry::Spherical, 2, &[0, 1]); // [0, 2]
// =============================================================================

use crate::metric::Geometry;
use symbi_carrier::Scalar;

/// the cartesian position of the chart position `coord`, in natural coordinate order: cartesian
/// (x, y, z); cylindrical (R, phi, z); spherical (r, theta, phi)
pub fn chart_to_cartesian<S: Scalar>(chart: Geometry, coord: [S; 3]) -> [S; 3] {
    match chart {
        Geometry::Cartesian => coord,
        Geometry::Cylindrical => {
            let (r, phi) = (coord[0], coord[1]);
            [r * phi.cos(), r * phi.sin(), coord[2]]
        }
        Geometry::Spherical => {
            let (r, th, phi) = (coord[0], coord[1], coord[2]);
            let rst = r * th.sin();
            [rst * phi.cos(), rst * phi.sin(), r * th.cos()]
        }
    }
}

/// the cartesian axes (0 = x, 1 = y, 2 = z) a body's `ndim`-component position occupies on a
/// chart whose grid axes carry the coordinates `axes`: the grid plane's convention
pub fn body_cartesian_axes(chart: Geometry, ndim: usize, axes: &[usize]) -> Vec<usize> {
    match chart {
        Geometry::Cartesian => (0..ndim).collect(),
        Geometry::Cylindrical => axes.to_vec(),
        Geometry::Spherical => {
            if ndim >= 3 {
                vec![0, 1, 2]
            } else {
                vec![0, 2]
            }
        }
    }
}
