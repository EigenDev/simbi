// =============================================================================
// excise.rs
//
// horizon-excision fill: zero-gradient propagation of primitive state into an
// excised sphere, as repeated sweeps of a fixed diagonal-outward stencil.
// each sweep, every excised cell (|x| < r_exc) takes the state of its diagonal
// neighbor in the outward direction (sign(x_0), sign(x_1)); live cells keep
// their own state. `onion_pass_count` sweeps propagate rim values to the
// deepest cell. inside a black-hole horizon every characteristic points
// inward, so the filled values are numerical padding the exterior never sees —
// a first-order nearest-donor copy is sufficient by causality.
//
// usage:
//   let filled = onion_fill_cell(own, pp, pm, mp, mm, x_c, r_exc);
//   let k = onion_pass_count(r_exc, min_dx);
// =============================================================================

use symbi_ir::algebra::Scalar;

/// one sweep of the fill at one cell: the excised-cell state is the diagonal
/// neighbor toward larger |x| per axis; a live cell passes through unchanged.
/// `NF` state components share one selection mask. the four diagonal states are
/// at cell offsets (+1,+1), (+1,-1), (-1,+1), (-1,-1) in axis order (x_0, x_1).
/// the mask is SHARP (a cell either fills or it does not): a partial blend of
/// two primitive states is not itself a valid primitive state.
pub fn onion_fill_cell<S: Scalar, const NF: usize>(
    own: [S; NF],
    diag_pp: [S; NF],
    diag_pm: [S; NF],
    diag_mp: [S; NF],
    diag_mm: [S; NF],
    x_c: [S; 2],
    r_exc: S,
) -> [S; NF] {
    // the outward diagonal: sign(x_0) picks the x-side, sign(x_1) the y-side.
    // stepping (sign(x_0), sign(x_1)) strictly increases |x|, so repeated sweeps
    // are monotone rim -> center and terminate at live donors.
    let x_pos = x_c[0].cmp_ge(S::ZERO);
    let y_pos = x_c[1].cmp_ge(S::ZERO);
    let r = (x_c[0] * x_c[0] + x_c[1] * x_c[1]).sqrt();
    let excised = r.cmp_lt(r_exc);
    std::array::from_fn(|kk| {
        let px = S::select(y_pos, diag_pp[kk], diag_pm[kk]);
        let mx = S::select(y_pos, diag_mp[kk], diag_mm[kk]);
        let outward = S::select(x_pos, px, mx);
        S::select(excised, outward, own[kk])
    })
}

/// the sweep count that guarantees every excised cell holds a rim-propagated
/// value: the deepest cell sits `r_exc / min_dx` cells from the rim along the
/// slowest (single-axis-dominant) path, plus margin for the staircase corners.
pub fn onion_pass_count(r_exc: f64, min_dx: f64) -> usize {
    assert!(r_exc > 0.0 && min_dx > 0.0, "onion_pass_count: positive radius and width");
    (r_exc / min_dx).ceil() as usize + 2
}

#[cfg(test)]
mod tests {
    use super::*;

    // a synthetic even-resolution grid centered on the origin: cell centers at
    // (ii + 0.5 - n/2) dx straddle the origin and never sit on an axis.
    fn centers(n: usize, dx: f64) -> Vec<f64> {
        (0..n).map(|ii| (ii as f64 + 0.5 - n as f64 / 2.0) * dx).collect()
    }

    fn sweep_grid(vals: &[Vec<f64>], xs: &[f64], r_exc: f64) -> Vec<Vec<f64>> {
        let n = xs.len();
        let mut out = vals.to_vec();
        for ii in 1..n - 1 {
            for jj in 1..n - 1 {
                let [v] = onion_fill_cell(
                    [vals[ii][jj]],
                    [vals[ii + 1][jj + 1]],
                    [vals[ii + 1][jj - 1]],
                    [vals[ii - 1][jj + 1]],
                    [vals[ii - 1][jj - 1]],
                    [xs[ii], xs[jj]],
                    r_exc,
                );
                out[ii][jj] = v;
            }
        }
        out
    }

    #[test]
    fn uniform_state_is_preserved_bitwise() {
        // a copy of a uniform state is the uniform state, excised or live.
        let u = [1.3_f64, -0.2, 0.7, 4.0e-3];
        for &(x, y) in &[(0.1_f64, 0.1), (5.0, 5.0), (-0.3, 0.2)] {
            let got = onion_fill_cell(u, u, u, u, u, [x, y], 1.4);
            assert_eq!(got, u, "at ({x},{y})");
        }
    }

    #[test]
    fn excised_cell_takes_the_outward_diagonal_per_quadrant() {
        let own = [0.0_f64];
        let (pp, pm, mp, mm) = ([1.0_f64], [2.0_f64], [3.0_f64], [4.0_f64]);
        let r_exc = 10.0;
        let pick = |x: f64, y: f64| onion_fill_cell(own, pp, pm, mp, mm, [x, y], r_exc)[0];
        assert_eq!(pick(1.0, 1.0), 1.0, "first quadrant -> (+,+)");
        assert_eq!(pick(1.0, -1.0), 2.0, "fourth quadrant -> (+,-)");
        assert_eq!(pick(-1.0, 1.0), 3.0, "second quadrant -> (-,+)");
        assert_eq!(pick(-1.0, -1.0), 4.0, "third quadrant -> (-,-)");
    }

    #[test]
    fn live_cell_keeps_its_own_state() {
        let own = [7.7_f64];
        let got = onion_fill_cell(own, [1.0], [2.0], [3.0], [4.0], [3.0, 4.0], 1.4);
        assert_eq!(got, own, "|x| = 5 > r_exc = 1.4 is live");
    }

    #[test]
    fn sweeps_propagate_rim_values_to_the_deepest_cell() {
        // live cells hold 1, excised cells start at 0; after onion_pass_count sweeps
        // every cell must hold 1 — the pass-count law the dispatch relies on.
        let (n, dx, r_exc) = (32, 0.1, 1.4);
        let xs = centers(n, dx);
        let mut vals: Vec<Vec<f64>> = (0..n)
            .map(|ii| {
                (0..n)
                    .map(|jj| {
                        let r = (xs[ii] * xs[ii] + xs[jj] * xs[jj]).sqrt();
                        if r < r_exc { 0.0 } else { 1.0 }
                    })
                    .collect()
            })
            .collect();
        let n_excised = vals.iter().flatten().filter(|&&v| v == 0.0).count();
        assert!(n_excised > 100, "the excised ball must be deep (got {n_excised} cells)");
        for _ in 0..onion_pass_count(r_exc, dx) {
            vals = sweep_grid(&vals, &xs, r_exc);
        }
        for ii in 1..n - 1 {
            for jj in 1..n - 1 {
                assert_eq!(vals[ii][jj], 1.0, "cell ({ii},{jj}) never received a rim value");
            }
        }
    }

    #[test]
    fn fill_preserves_x_y_symmetry() {
        // a state symmetric under (x, y) -> (y, x) stays symmetric under a sweep:
        // the donor of the mirrored cell is the mirrored donor. even resolution keeps
        // cell centers off the axes, so the sign selects are never on the tie point.
        let (n, dx, r_exc) = (16, 0.2, 1.4);
        let xs = centers(n, dx);
        let vals: Vec<Vec<f64>> = (0..n)
            .map(|ii| (0..n).map(|jj| xs[ii] * xs[jj] + 0.3 * (xs[ii] + xs[jj])).collect())
            .collect();
        let out = sweep_grid(&vals, &xs, r_exc);
        for ii in 0..n {
            for jj in 0..n {
                assert_eq!(out[ii][jj], out[jj][ii], "symmetry broken at ({ii},{jj})");
            }
        }
    }
}
