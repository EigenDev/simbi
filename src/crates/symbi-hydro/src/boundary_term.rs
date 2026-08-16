// =============================================================================
// boundary_term.rs
//
// carrier-generic ghost-cell fills for the classical boundary conditions, written
// once over `S: Scalar` — evaluated at S=f64 (the analytical reference) and traced
// at S=Gv (the rendered ghost-fill kernel) from the same definition, the boundary
// analogue of the source lifts in `source_term.rs`.
//
// the existing lattice-map ghost fills are the two degenerate members of this family:
//   - a grade-0 copy at the outflow edge cell is zero-gradient Neumann (q = 0).
//   - a reflect (mirror + normal-sign flip) about the wall is the homogeneous
//     Dirichlet member (the wall value is held at 0 for the normal component).
// `neumann_ghost` / `robin_ghost` generalize these to a prescribed normal gradient
// and a prescribed mixed (Robin) relation, so the ghost value is a function of the
// boundary-adjacent interior ("edge") value, the cell geometry, and the prescribed
// coefficients — no new access pattern beyond the edge read the outflow map already
// performs.
//
// conventions (n = outward unit normal; the ghost cell lies along +n from the edge):
//   - `dist >= 0`  : edge-cell-center -> ghost-cell-center separation along n.
//   - `h    >  0`  : same separation used as the face-centered finite difference
//                    stencil width for the Robin normal derivative.
// both are positive lengths; the outward orientation is already folded into the
// sign of a prescribed gradient `q` (a positive `q` raises the state outward).
// =============================================================================

use crate::Scalar;

/// neumann ghost fill — prescribe the outward normal derivative `dU/dn = q` at the boundary.
/// given the boundary-adjacent interior ("edge") value `u_edge` and the outward edge->ghost
/// separation `dist >= 0`, the linear extrapolation
///   `U_ghost = u_edge + q * dist`
/// reproduces the prescribed gradient exactly: `(U_ghost - u_edge)/dist = q`. `q = 0` recovers
/// the zero-gradient copy (the outflow fill), so outflow is the homogeneous member.
pub fn neumann_ghost<S: Scalar>(u_edge: S, q: S, dist: S) -> S {
    u_edge + q * dist
}

/// robin ghost fill — prescribe the mixed relation `a*U_face + b*(dU/dn) = c` at the boundary
/// face, with the face midway between the edge interior cell and the ghost cell (separation `h`
/// along the outward normal). approximating the face state and normal derivative by the
/// two-point stencil
///   `U_face = (u_edge + U_ghost)/2`,   `dU/dn = (U_ghost - u_edge)/h`
/// and solving the prescribed relation for the ghost value gives
///   `U_ghost = [c - u_edge*(a/2 - b/h)] / (a/2 + b/h)`.
/// this degenerates to Dirichlet `U_face = c/a` at `b = 0` and to Neumann `dU/dn = c/b` at
/// `a = 0`, so a single lift covers all three classical boundary conditions.
pub fn robin_ghost<S: Scalar>(u_edge: S, a: S, b: S, c: S, h: S) -> S {
    let a2 = S::from_f64(0.5) * a;
    let b_h = b / h;
    (c - u_edge * (a2 - b_h)) / (a2 + b_h)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(x: f64, y: f64, ctx: &str) {
        assert!((x - y).abs() < 1e-12, "{ctx}: {x} != {y}");
    }

    #[test]
    fn neumann_reproduces_the_prescribed_gradient() {
        // u_edge = 3, q = 2, dist = 0.5 -> U_ghost = 3 + 1 = 4; the recovered gradient is q.
        let u_g = neumann_ghost(3.0, 2.0, 0.5);
        approx(u_g, 4.0, "neumann ghost value");
        approx((u_g - 3.0) / 0.5, 2.0, "recovered gradient");
    }

    #[test]
    fn neumann_zero_gradient_is_the_outflow_copy() {
        // q = 0 -> the ghost copies the edge cell, exactly the outflow (zero-gradient) fill.
        approx(neumann_ghost(7.25, 0.0, 0.9), 7.25, "zero-gradient copy");
    }

    #[test]
    fn robin_degenerates_to_dirichlet_at_b_zero() {
        // b = 0, a = 2, c = 6 -> Dirichlet U_face = c/a = 3. check U_face = (u_edge+U_ghost)/2 = 3.
        let u_edge = 1.5;
        let u_g = robin_ghost(u_edge, 2.0, 0.0, 6.0, 0.4);
        approx((u_edge + u_g) / 2.0, 3.0, "robin->dirichlet face value");
    }

    #[test]
    fn robin_degenerates_to_neumann_at_a_zero() {
        // a = 0, b = 1, c = 2, h = 0.5 -> Neumann dU/dn = c/b = 2. check (U_ghost-u_edge)/h = 2.
        let u_edge = -0.7;
        let u_g = robin_ghost(u_edge, 0.0, 1.0, 2.0, 0.5);
        approx((u_g - u_edge) / 0.5, 2.0, "robin->neumann gradient");
    }

    #[test]
    fn robin_general_satisfies_the_prescribed_relation() {
        // a general (a,b,c): the reconstructed face state + normal derivative satisfy a*U_face +
        // b*dU/dn = c by construction, for any edge value.
        let (a, b, c, h) = (1.3, 0.7, 2.1, 0.25);
        for &u_edge in &[-2.0, 0.0, 3.4] {
            let u_g = robin_ghost(u_edge, a, b, c, h);
            let u_face = (u_edge + u_g) / 2.0;
            let dudn = (u_g - u_edge) / h;
            approx(a * u_face + b * dudn, c, "robin relation");
        }
    }
}
