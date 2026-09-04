// =============================================================================
// magnetic_slip_ct.rs
//
// the constrained-transport gather/scatter for the tensor magnetic-slip operator
// on a 3D Cartesian staggered grid. the slip electric field couples all three
// current components through the dyad A(B) = coeff (|B|^2 I - B B), but the CT
// current components live on differently oriented edges, so a pointwise tensor has
// no naturally colocated edge vector to act on. the operator is assembled at the
// cell centers as the common quadrature site:
//   R   gathers the three edge current components to the center (J_q = R J),
//   R^* scatters a cell-centered vector back to the oriented edges,
// with `R^*` the exact transpose of `R`. the discrete operator is E = R^* A(B_q) R J,
// so the global edge inner product
//   <J, E>_edge = <R J, A(B_q) R J>_q >= 0
// is a sum of per-cell positive-semidefinite forms. the field gather `B_q` (faces to
// center) forms the frozen dyad coefficient and carries no adjoint.
//
// the grid uses the owned-by-lower-corner convention: a d-edge and a d-face are
// indexed by the cell at their lower corner. periodic in every axis (the initial
// boundary condition under which the adjoint identity is proved).
//
// the unit-measure transpose is already the physically weighted adjoint. the Hodge
// measure of a d-edge is its length dx_d times its dual-face area dx_{t1} dx_{t2},
// which is the common cell volume V = dx dy dz for every orientation; the cell
// quadrature weight is also V. the equal weights cancel from
// <R J, y>_q = <J, R^* y>_edge, so anisotropic spacing (dx != dy != dz) leaves the
// 1/4 scatter unchanged. a later magnetic-energy calibration may refine the
// face-based magnetic norm, but it may not redefine R or R^*.
//
// carrier-generic over the scalar, so one stencil serves the f64 reference/adjoint
// proof and the traced two-pass kernels.
//
// usage:
//   let jq = grid.gather_current(&edge_j);  // R J at cell centers
//   let bq = grid.gather_field(&face_b);    // B_q at cell centers
//   // form F_q = A(B_q) J_q per cell, then:
//   let emf = grid.scatter_emf(&cell_f);    // E = R^* F on edges
// =============================================================================

use symbi_carrier::Scalar;

/// a periodic 3D grid holding the staggered layout's dimensions. fields are flat
/// `Vec<S>` of length `nx*ny*nz`, one entry per cell in the owned-by-lower-corner
/// indexing; a three-component field is `[Vec<S>; 3]`.
#[derive(Clone, Copy, Debug)]
pub struct CtGrid {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

impl CtGrid {
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self { nx, ny, nz }
    }

    pub fn cells(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// the flat index of cell `(i, j, k)` with periodic wrap on every axis.
    fn at(&self, i: isize, j: isize, k: isize) -> usize {
        let wrap = |v: isize, n: usize| -> usize { v.rem_euclid(n as isize) as usize };
        let (ii, jj, kk) = (wrap(i, self.nx), wrap(j, self.ny), wrap(k, self.nz));
        (ii * self.ny + jj) * self.nz + kk
    }

    /// the two transverse axes of axis `d`, in cyclic order.
    fn transverse(d: usize) -> (usize, usize) {
        ((d + 1) % 3, (d + 2) % 3)
    }

    /// the Hodge measure of a primal `d`-edge on a uniform Cartesian grid: its edge length
    /// `dx_d` times its dual-face area `dx_{t1} dx_{t2}`. this is the common cell volume
    /// `V = dx dy dz` for every edge orientation, since the product runs over all three axes.
    pub fn edge_hodge_weight(d: usize, dx: [f64; 3]) -> f64 {
        let (t1, t2) = Self::transverse(d);
        dx[d] * dx[t1] * dx[t2]
    }

    /// the cell quadrature weight: the cell volume `V = dx dy dz`.
    pub fn cell_volume(dx: [f64; 3]) -> f64 {
        dx[0] * dx[1] * dx[2]
    }

    /// `R`: gather the three edge current components to the cell centers. component `d`
    /// averages the four `d`-edges bounding each cell (transverse offsets in `{0, -1}`),
    /// weight `1/4`.
    pub fn gather_current<S: Scalar>(&self, edge: &[Vec<S>; 3]) -> [Vec<S>; 3] {
        let quarter = S::from_f64(0.25);
        std::array::from_fn(|d| {
            let (t1, t2) = Self::transverse(d);
            let mut out = vec![S::ZERO; self.cells()];
            for i in 0..self.nx as isize {
                for j in 0..self.ny as isize {
                    for k in 0..self.nz as isize {
                        let base = [i, j, k];
                        let mut sum = S::ZERO;
                        for &a in &[0isize, -1] {
                            for &b in &[0isize, -1] {
                                let mut off = base;
                                off[t1] += a;
                                off[t2] += b;
                                sum = sum + edge[d][self.at(off[0], off[1], off[2])];
                            }
                        }
                        out[self.at(i, j, k)] = sum * quarter;
                    }
                }
            }
            out
        })
    }

    /// `R^*`: scatter a cell-centered vector to the oriented edges, the exact transpose
    /// of `gather_current`. component `d` sends each `d`-edge the four cells that hold it
    /// (transverse offsets in `{0, +1}`, the negated gather offsets), weight `1/4`.
    pub fn scatter_emf<S: Scalar>(&self, cell: &[Vec<S>; 3]) -> [Vec<S>; 3] {
        let quarter = S::from_f64(0.25);
        std::array::from_fn(|d| {
            let (t1, t2) = Self::transverse(d);
            let mut out = vec![S::ZERO; self.cells()];
            for i in 0..self.nx as isize {
                for j in 0..self.ny as isize {
                    for k in 0..self.nz as isize {
                        let base = [i, j, k];
                        let mut sum = S::ZERO;
                        for &a in &[0isize, 1] {
                            for &b in &[0isize, 1] {
                                let mut off = base;
                                off[t1] += a;
                                off[t2] += b;
                                sum = sum + cell[d][self.at(off[0], off[1], off[2])];
                            }
                        }
                        out[self.at(i, j, k)] = sum * quarter;
                    }
                }
            }
            out
        })
    }

    /// gather the three face field components to the cell centers: component `d` averages
    /// its two `d`-faces (at `c[d] -/+ 1/2`, indices `c - e_d` and `c`), weight `1/2`. this
    /// forms the frozen dyad coefficient `B_q` and carries no adjoint.
    pub fn gather_field<S: Scalar>(&self, face: &[Vec<S>; 3]) -> [Vec<S>; 3] {
        let half = S::HALF;
        std::array::from_fn(|d| {
            let mut out = vec![S::ZERO; self.cells()];
            for i in 0..self.nx as isize {
                for j in 0..self.ny as isize {
                    for k in 0..self.nz as isize {
                        let base = [i, j, k];
                        let mut lo = base;
                        lo[d] -= 1;
                        let hi = face[d][self.at(base[0], base[1], base[2])];
                        let lo = face[d][self.at(lo[0], lo[1], lo[2])];
                        out[self.at(i, j, k)] = half * (hi + lo);
                    }
                }
            }
            out
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbi_algebra::Tensor;
    use symbi_ib::magnetic_slip::slip_apply;

    // a deterministic pseudo-random fill, so the adjoint identity is exercised on generic
    // data without a coincidental cancellation and without an rng dependency.
    fn fill(grid: &CtGrid, seed: u64) -> [Vec<f64>; 3] {
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64 / u64::MAX as f64) * 2.0 - 1.0
        };
        std::array::from_fn(|_| (0..grid.cells()).map(|_| next()).collect())
    }

    fn inner(a: &[Vec<f64>; 3], b: &[Vec<f64>; 3]) -> f64 {
        (0..3)
            .map(|d| a[d].iter().zip(&b[d]).map(|(x, y)| x * y).sum::<f64>())
            .sum()
    }

    // the load-bearing identity: R^* is the exact transpose of R, so the edge and cell
    // inner products of the gather agree, <R J, y>_q = <J, R^* y>_edge, on a periodic grid
    // of coprime extents (so every stencil wrap is exercised).
    #[test]
    fn current_gather_and_emf_scatter_are_adjoint() {
        let grid = CtGrid::new(3, 4, 5);
        let j = fill(&grid, 1);
        let y = fill(&grid, 2);
        let lhs = inner(&grid.gather_current(&j), &y); // <R J, y>_q
        let rhs = inner(&j, &grid.scatter_emf(&y)); // <J, R^* y>_edge
        assert!(
            (lhs - rhs).abs() < 1e-12 * lhs.abs().max(1.0),
            "adjoint identity broken: <R J, y> = {lhs} vs <J, R^* y> = {rhs}"
        );
    }

    // adjointness holds on a second, differently shaped grid: the identity is a property of
    // the transpose stencil, not of one lattice size.
    #[test]
    fn adjointness_is_grid_size_independent() {
        for (nx, ny, nz) in [(2, 2, 2), (5, 3, 2), (4, 4, 4), (7, 3, 5)] {
            let grid = CtGrid::new(nx, ny, nz);
            let j = fill(&grid, 11);
            let y = fill(&grid, 22);
            let lhs = inner(&grid.gather_current(&j), &y);
            let rhs = inner(&j, &grid.scatter_emf(&y));
            assert!(
                (lhs - rhs).abs() < 1e-12 * lhs.abs().max(1.0),
                "adjoint broken on {nx}x{ny}x{nz}: {lhs} vs {rhs}"
            );
        }
    }

    // the Hodge measures of the three edge orientations and the cell all equal the common
    // volume V for uniform Cartesian spacing, including anisotropic dx != dy != dz. the equal
    // weights are why the unit-measure transpose is already the physically weighted adjoint;
    // step 5 may calibrate the magnetic norm but must not reweight R or R^*.
    #[test]
    fn uniform_cartesian_hodge_measures_are_the_common_volume() {
        for dx in [[1.0, 1.0, 1.0], [0.5, 2.0, 0.25], [3.0, 0.1, 7.0]] {
            let v = CtGrid::cell_volume(dx);
            for d in 0..3 {
                assert!(
                    (CtGrid::edge_hodge_weight(d, dx) - v).abs() < 1e-12 * v,
                    "edge {d} Hodge weight {} != cell volume {v} under spacing {dx:?}",
                    CtGrid::edge_hodge_weight(d, dx)
                );
            }
        }
    }

    // the composed operator E = R^* A(B_q) R J realizes a nonnegative global quadratic form:
    // <J, E>_edge = <R J, A(B_q) R J>_q >= 0, per-cell PSD summed. this is the discrete energy
    // dissipation, the correct global assertion in place of an edgewise J.E sign.
    #[test]
    fn the_composed_operator_is_a_nonnegative_quadratic_form() {
        let grid = CtGrid::new(4, 5, 3);
        let face_b = fill(&grid, 7);
        let edge_j = fill(&grid, 8);
        let coeff = 0.6_f64; // a_B chi_B >= 0, uniform here

        let bq = grid.gather_field(&face_b);
        let jq = grid.gather_current(&edge_j);

        // cell pass: F_q = A(B_q) (R J)_q, one PSD dyad per cell.
        let mut f: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; grid.cells()]);
        let mut quad_q = 0.0; // <R J, A(B_q) R J>_q
        for c in 0..grid.cells() {
            let b = Tensor::new([bq[0][c], bq[1][c], bq[2][c]]);
            let jc = Tensor::new([jq[0][c], jq[1][c], jq[2][c]]);
            let fc = slip_apply(coeff, &b, &jc);
            for d in 0..3 {
                f[d][c] = fc[d];
            }
            quad_q += jc.dot(&fc);
        }

        // edge pass: E = R^* F, and the global edge form equals the cell form.
        let e = grid.scatter_emf(&f);
        let quad_edge = inner(&edge_j, &e); // <J, E>_edge

        assert!(quad_q >= -1e-13, "per-cell quadrature form is negative: {quad_q}");
        assert!(
            (quad_edge - quad_q).abs() < 1e-11 * quad_q.abs().max(1.0),
            "global form mismatch: <J,E>_edge = {quad_edge} vs <RJ, A RJ>_q = {quad_q}"
        );
        assert!(quad_edge >= -1e-13, "discrete dissipation is negative: {quad_edge}");
    }
}
