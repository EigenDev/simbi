// =============================================================================
// excise.rs
//
// horizon-excision fill: zero-gradient propagation of primitive state into the
// excised region, as repeated sweeps of a fixed diagonal-outward stencil.
// the excised region is the sublevel set of the kerr-schild radius,
// r_ks(x; a) < r_exc — the sphere |x| < r_exc at a = 0 and the oblate spheroid
// (x^2 + y^2)/(r_exc^2 + a^2) + z^2/r_exc^2 < 1 at spin a about z (the r = const
// surfaces of the cartesian kerr-schild chart). each sweep, every excised cell
// takes the state of its diagonal neighbor in the outward direction
// (sign(x_0), sign(x_1)[, sign(x_2)]); stepping outward strictly increases
// r_ks (it is monotone along axis-outward rays), so repeated sweeps are
// monotone rim -> center and terminate at live donors. inside a black-hole
// horizon every characteristic points inward, so the filled values are
// numerical padding the exterior never sees — a first-order nearest-donor
// copy is sufficient by causality.
//
// usage:
//   let excised = ks_excised(&x_c, spin, r_exc);
//   let filled = onion_fill_cell(own, pp, pm, mp, mm, x_c, excised);
//   let k = onion_pass_count((r_exc * r_exc + spin * spin).sqrt(), min_dx);
// =============================================================================

use symbi_ir::algebra::Scalar;

/// the excision predicate: kerr-schild radius r_ks(x; a) < r_exc, as a select
/// mask. r_ks solves the oblate-spheroidal quartic,
///   r_ks^2 = (R^2 - a^2)/2 + sqrt(((R^2 - a^2)/2)^2 + a^2 z^2),  R^2 = |x|^2,
/// with the missing axes of a D < 3 position reading z = 0 (the equatorial
/// slice, where the excised disc has coordinate radius sqrt(r_exc^2 + a^2)).
/// compared in the SQUARE (both sides non-negative), so no sqrt of the radius
/// itself enters the mask. a = 0 reduces to |x|^2 < r_exc^2 exactly.
pub fn ks_excised<S: Scalar, const D: usize>(x_c: &[S; D], spin: S, r_exc: S) -> S::Mask {
    let z = if D > 2 { x_c[2] } else { S::ZERO };
    let mut rr2 = S::ZERO;
    for kk in 0..D {
        rr2 = rr2 + x_c[kk] * x_c[kk];
    }
    let half = S::from_f64(0.5);
    let d = half * (rr2 - spin * spin);
    let az = spin * z;
    let r_ks2 = d + (d * d + az * az).sqrt();
    r_ks2.cmp_lt(r_exc * r_exc)
}

/// one sweep of the fill at one cell: the excised-cell state is the diagonal
/// neighbor toward larger |x| per axis; a live cell passes through unchanged.
/// `NF` state components share the one `excised` selection mask (computed by
/// the caller, e.g. [`ks_excised`]). the four diagonal states are at cell
/// offsets (+1,+1), (+1,-1), (-1,+1), (-1,-1) in axis order (x_0, x_1). the
/// mask is SHARP (a cell either fills or it does not): a partial blend of two
/// primitive states is not itself a valid primitive state.
pub fn onion_fill_cell<S: Scalar, const NF: usize>(
    own: [S; NF],
    diag_pp: [S; NF],
    diag_pm: [S; NF],
    diag_mp: [S; NF],
    diag_mm: [S; NF],
    x_c: [S; 2],
    excised: S::Mask,
) -> [S; NF] {
    // the outward diagonal: sign(x_0) picks the x-side, sign(x_1) the y-side.
    // stepping (sign(x_0), sign(x_1)) strictly increases every |x_k|, hence the
    // kerr-schild radius, so repeated sweeps are monotone rim -> center and
    // terminate at live donors.
    let x_pos = x_c[0].cmp_ge(S::ZERO);
    let y_pos = x_c[1].cmp_ge(S::ZERO);
    std::array::from_fn(|kk| {
        let px = S::select(y_pos, diag_pp[kk], diag_pm[kk]);
        let mx = S::select(y_pos, diag_mp[kk], diag_mm[kk]);
        let outward = S::select(x_pos, px, mx);
        S::select(excised, outward, own[kk])
    })
}

/// the 3d onion sweep: every excised cell takes the primitive state of its outward
/// CORNER-diagonal neighbor, selected by (sign(x), sign(y), sign(z)) — the step
/// strictly increases |x| on every axis, hence the kerr-schild radius, so repeated
/// sweeps are monotone rim -> center and terminate at live donors. `excised` is the
/// caller-computed mask ([`ks_excised`]). `diags` are the 8 corner neighbors in
/// z-fastest sign order: [mmm, mmp, mpm, mpp, pmm, pmp, ppm, ppp] (p = +1, m = -1 on
/// the (x, y, z) axes respectively).
pub fn onion_fill_cell_3d<S: Scalar, const NF: usize>(
    own: [S; NF],
    diags: &[[S; NF]; 8],
    x_c: [S; 3],
    excised: S::Mask,
) -> [S; NF] {
    let x_pos = x_c[0].cmp_ge(S::ZERO);
    let y_pos = x_c[1].cmp_ge(S::ZERO);
    let z_pos = x_c[2].cmp_ge(S::ZERO);
    std::array::from_fn(|kk| {
        // three-level sign select down the corner tree: z within each (x, y) pair,
        // then y within each x half, then x.
        let pp = S::select(z_pos, diags[7][kk], diags[6][kk]);
        let pm = S::select(z_pos, diags[5][kk], diags[4][kk]);
        let mp = S::select(z_pos, diags[3][kk], diags[2][kk]);
        let mm = S::select(z_pos, diags[1][kk], diags[0][kk]);
        let px = S::select(y_pos, pp, pm);
        let mx = S::select(y_pos, mp, mm);
        let outward = S::select(x_pos, px, mx);
        S::select(excised, outward, own[kk])
    })
}

/// the sweep count that guarantees every excised cell holds a rim-propagated
/// value: the deepest cell sits `extent / min_dx` cells from the rim along the
/// slowest (single-axis-dominant) path, plus margin for the staircase corners.
/// `extent` is the region's largest semi-axis — r_exc for the a = 0 sphere,
/// sqrt(r_exc^2 + a^2) for the spinning spheroid's equatorial semi-major axis.
pub fn onion_pass_count(extent: f64, min_dx: f64) -> usize {
    assert!(extent > 0.0 && min_dx > 0.0, "onion_pass_count: positive extent and width");
    (extent / min_dx).ceil() as usize + 2
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
                let x_c = [xs[ii], xs[jj]];
                let [v] = onion_fill_cell(
                    [vals[ii][jj]],
                    [vals[ii + 1][jj + 1]],
                    [vals[ii + 1][jj - 1]],
                    [vals[ii - 1][jj + 1]],
                    [vals[ii - 1][jj - 1]],
                    x_c,
                    ks_excised(&x_c, 0.0, r_exc),
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
            let got = onion_fill_cell(u, u, u, u, u, [x, y], ks_excised(&[x, y], 0.0, 1.4));
            assert_eq!(got, u, "at ({x},{y})");
        }
    }

    #[test]
    fn excised_cell_takes_the_outward_diagonal_per_quadrant() {
        let own = [0.0_f64];
        let (pp, pm, mp, mm) = ([1.0_f64], [2.0_f64], [3.0_f64], [4.0_f64]);
        let r_exc = 10.0;
        let pick = |x: f64, y: f64| {
            onion_fill_cell(own, pp, pm, mp, mm, [x, y], ks_excised(&[x, y], 0.0, r_exc))[0]
        };
        assert_eq!(pick(1.0, 1.0), 1.0, "first quadrant -> (+,+)");
        assert_eq!(pick(1.0, -1.0), 2.0, "fourth quadrant -> (+,-)");
        assert_eq!(pick(-1.0, 1.0), 3.0, "second quadrant -> (-,+)");
        assert_eq!(pick(-1.0, -1.0), 4.0, "third quadrant -> (-,-)");
    }

    #[test]
    fn live_cell_keeps_its_own_state() {
        let own = [7.7_f64];
        let got =
            onion_fill_cell(own, [1.0], [2.0], [3.0], [4.0], [3.0, 4.0], ks_excised(&[3.0, 4.0], 0.0, 1.4));
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

#[cfg(test)]
mod tests_3d {
    use super::*;

    #[test]
    fn uniform_state_is_preserved_bitwise_3d() {
        let u = [1.3_f64, -0.2, 0.7, 0.1, 4.0e-3];
        let diags = [u; 8];
        for &(x, y, z) in &[(0.1_f64, 0.1, 0.1), (5.0, 5.0, 5.0), (-0.3, 0.2, -0.4)] {
            let got = onion_fill_cell_3d(u, &diags, [x, y, z], ks_excised(&[x, y, z], 0.0, 1.4));
            assert_eq!(got, u, "at ({x},{y},{z})");
        }
    }

    #[test]
    fn excised_cell_takes_the_outward_corner_per_octant() {
        // donor values encode their corner: 100*sx + 10*sy + sz with s in {1(+), 2(-)}.
        let own = [0.0_f64];
        let val = |sx: i32, sy: i32, sz: i32| -> [f64; 1] {
            [(100 * if sx > 0 { 1 } else { 2 } + 10 * if sy > 0 { 1 } else { 2 }
                + if sz > 0 { 1 } else { 2 }) as f64]
        };
        // z-fastest sign order: mmm, mmp, mpm, mpp, pmm, pmp, ppm, ppp.
        let diags = [
            val(-1, -1, -1),
            val(-1, -1, 1),
            val(-1, 1, -1),
            val(-1, 1, 1),
            val(1, -1, -1),
            val(1, -1, 1),
            val(1, 1, -1),
            val(1, 1, 1),
        ];
        let r_exc = 10.0;
        for &(x, y, z) in &[
            (1.0_f64, 1.0, 1.0),
            (1.0, 1.0, -1.0),
            (1.0, -1.0, 1.0),
            (1.0, -1.0, -1.0),
            (-1.0, 1.0, 1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (-1.0, -1.0, -1.0),
        ] {
            let got =
                onion_fill_cell_3d(own, &diags, [x, y, z], ks_excised(&[x, y, z], 0.0, r_exc))[0];
            let want = val(x.signum() as i32, y.signum() as i32, z.signum() as i32)[0];
            assert_eq!(got, want, "octant ({x},{y},{z}) picked the wrong corner donor");
        }
        // a live cell (outside r_exc) keeps its own state.
        let live = onion_fill_cell_3d(
            own,
            &diags,
            [20.0, 20.0, 20.0],
            ks_excised(&[20.0, 20.0, 20.0], 0.0, r_exc),
        )[0];
        assert_eq!(live, 0.0, "live cell must keep its own value");
    }
}

#[cfg(test)]
mod predicate_tests {
    use super::*;

    // the f64 scalar's mask type is bool.
    fn is_excised_3d(x: [f64; 3], a: f64, r: f64) -> bool {
        ks_excised(&x, a, r)
    }

    #[test]
    fn zero_spin_is_the_sphere() {
        let r = 1.4;
        for &(x, y, z) in &[(0.5_f64, 0.5, 0.5), (1.0, 0.9, 0.1), (1.0, 1.0, 1.0), (0.0, 0.0, 1.39)] {
            let want = (x * x + y * y + z * z).sqrt() < r;
            assert_eq!(is_excised_3d([x, y, z], 0.0, r), want, "at ({x},{y},{z})");
        }
    }

    #[test]
    fn spinning_region_is_the_oblate_spheroid() {
        // r_ks < r_exc is exactly (x^2 + y^2)/(r_exc^2 + a^2) + z^2/r_exc^2 < 1.
        let (a, r) = (0.9, 1.2);
        for &(x, y, z) in &[
            (1.3_f64, 0.0, 0.0),
            (0.0, 1.45, 0.0),
            (0.0, 0.0, 1.1),
            (0.0, 0.0, 1.3),
            (1.0, 1.0, 0.3),
            (0.7, 0.7, 0.8),
        ] {
            let want = (x * x + y * y) / (r * r + a * a) + (z * z) / (r * r) < 1.0;
            assert_eq!(is_excised_3d([x, y, z], a, r), want, "at ({x},{y},{z})");
        }
    }

    #[test]
    fn equatorial_slice_is_the_widened_disc() {
        // the D = 2 predicate (z = 0) excises the coordinate disc R < sqrt(r_exc^2 + a^2).
        let (a, r) = (0.9_f64, 1.2_f64);
        let disc = (r * r + a * a).sqrt();
        for &rr in &[0.5, disc - 1e-9, disc + 1e-9, 2.0] {
            let x = [rr / 2.0_f64.sqrt(), rr / 2.0_f64.sqrt()];
            let got = ks_excised(&x, a, r);
            assert_eq!(got, rr < disc, "coordinate radius {rr}");
        }
    }

    #[test]
    fn outward_diagonal_step_leaves_the_region_monotonically() {
        // the donor step (sign x, sign y, sign z) * dx strictly increases r_ks: an
        // excised cell's donor chain can never re-enter the region once it leaves.
        let (a, r, dx) = (0.9, 1.2, 0.05);
        let mut x = [0.02_f64, -0.03, 0.01];
        let mut inside = true;
        for _ in 0..200 {
            let step: [f64; 3] = std::array::from_fn(|kk| x[kk].signum() * dx);
            let next: [f64; 3] = std::array::from_fn(|kk| x[kk] + step[kk]);
            let was = is_excised_3d(x, a, r);
            let now = is_excised_3d(next, a, r);
            assert!(!(now && !was), "outward step re-entered the excised region at {next:?}");
            inside = now;
            x = next;
        }
        assert!(!inside, "200 outward steps never left the spheroid");
    }
}
