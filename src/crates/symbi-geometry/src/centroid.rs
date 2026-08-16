// =============================================================================
// centroid.rs
//
// the volume-weighted cell centroid per chart: the point at which a cell's
// metric must be evaluated so that the covariant conserved state stored at seed
// time inverts exactly under the metric-aware c2p.
//
// the weight is the chart's volume element, so the centroid is the first moment
// of the coordinate over the cell:
//   cartesian   x_c     = (lo + hi) / 2                    (dV = dx dy dz)
//   spherical   r_c     = int r^3 dr / int r^2 dr          (dV = r^2 sin(t) dr dt dp)
//               theta_c = int t sin t dt / int sin t dt
//               phi_c   = (lo + hi) / 2
//   cylindrical R_c     = int R^2 dR / int R dR            (dV = R dR dp dz)
//               phi_c, z_c = (lo + hi) / 2
//
// carrier-generic: the same expression serves the host (S = f64, cell seeding)
// and the traced kernel (S = Gv, in-kernel c2p + densitization lapse). storage
// and recovery therefore sample the metric at the same point by construction —
// evaluating them at different points leaves a per-cell state outside the range
// of the c2p, which on a curved chart drives the recovered pressure negative
// wherever the gas is cold enough that the (alpha - 1) D term dominates tau.
//
// usage:
//  let r_c = volume_weighted_centroid(Geometry::Spherical, 0, r_lo, r_hi);
// =============================================================================

use crate::metric::Geometry;
use symbi_ir::algebra::Scalar;

/// the volume-weighted centroid of coordinate `axis` over a cell spanning `lo..hi`
/// on `coords`. `axis` is the coordinate slot (0 = r/R/x, 1 = theta/phi/y,
/// 2 = phi/z/z), which the grid axis maps onto.
pub fn volume_weighted_centroid<S: Scalar>(coords: Geometry, axis: usize, lo: S, hi: S) -> S {
    let two = S::from_f64(2.0);
    let midpoint = || (lo + hi) / two;
    match (coords, axis) {
        // dV = r^2 sin(theta) dr dtheta dphi
        (Geometry::Spherical, 0) => {
            let i_r2 = (hi.powi(3) - lo.powi(3)) / S::from_f64(3.0);
            let i_r3 = (hi.powi(4) - lo.powi(4)) / S::from_f64(4.0);
            i_r3 / i_r2
        }
        (Geometry::Spherical, 1) => {
            let (c_lo, c_hi) = (lo.cos(), hi.cos());
            // int sin t dt = cos(lo) - cos(hi); int t sin t dt = [sin t - t cos t]
            let i_sin = c_lo - c_hi;
            let i_tsin = (hi.sin() - hi * c_hi) - (lo.sin() - lo * c_lo);
            i_tsin / i_sin
        }
        // dV = R dR dphi dz
        (Geometry::Cylindrical, 0) => {
            let i_r = (hi.powi(2) - lo.powi(2)) / two;
            let i_r2 = (hi.powi(3) - lo.powi(3)) / S::from_f64(3.0);
            i_r2 / i_r
        }
        // every remaining slot carries a weight independent of its own coordinate
        // (cartesian everywhere, the azimuth of both curvilinear charts, and the
        // cylindrical z), so the first moment is the arithmetic midpoint.
        _ => midpoint(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12 * a.abs().max(1.0)
    }

    #[test]
    fn each_chart_matches_its_first_moment_by_quadrature() {
        // the centroid is int x w(x) dx / int w(x) dx for the chart's radial weight;
        // check against a fine midpoint quadrature, derived independently of the
        // closed forms above.
        // the midpoint rule is O(h^2), so the reference itself carries ~1e-11
        // relative error — the closed forms are exact and an incorrect one lands at
        // the percent level, so this separates them by many orders of magnitude.
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9 * a.abs().max(1.0);
        let quad = |lo: f64, hi: f64, w: fn(f64) -> f64| {
            let n = 200_000;
            let h = (hi - lo) / n as f64;
            let (mut num, mut den) = (0.0, 0.0);
            for ii in 0..n {
                let x = lo + (ii as f64 + 0.5) * h;
                num += x * w(x) * h;
                den += w(x) * h;
            }
            num / den
        };
        let (lo, hi) = (2.0_f64, 2.7_f64);
        assert!(close(
            volume_weighted_centroid(Geometry::Spherical, 0, lo, hi),
            quad(lo, hi, |r| r * r)
        ));
        assert!(close(
            volume_weighted_centroid(Geometry::Cylindrical, 0, lo, hi),
            quad(lo, hi, |r| r)
        ));
        let (tl, th) = (0.4_f64, 1.1_f64);
        assert!(close(
            volume_weighted_centroid(Geometry::Spherical, 1, tl, th),
            quad(tl, th, f64::sin)
        ));
    }

    #[test]
    fn weight_free_slots_are_the_midpoint() {
        // a slot whose volume weight is constant along it: cartesian axes, both
        // azimuths, and the cylindrical z.
        for (coords, axis) in [
            (Geometry::Cartesian, 0),
            (Geometry::Cartesian, 1),
            (Geometry::Spherical, 2),
            (Geometry::Cylindrical, 1),
            (Geometry::Cylindrical, 2),
        ] {
            assert!(close(volume_weighted_centroid(coords, axis, 3.0, 4.0), 3.5));
        }
    }

    #[test]
    fn the_charts_disagree_on_the_radial_slot() {
        // the spherical and cylindrical radial centroids are DIFFERENT points; a
        // chart-blind seed that applies one formula to the other chart mislocates
        // the metric by a finite amount that no resolution increase removes.
        let (lo, hi) = (1.0_f64, 2.0_f64);
        let sph = volume_weighted_centroid(Geometry::Spherical, 0, lo, hi);
        let cyl = volume_weighted_centroid(Geometry::Cylindrical, 0, lo, hi);
        assert!((sph - cyl).abs() > 1e-3, "sph {sph} vs cyl {cyl}");
    }
}
