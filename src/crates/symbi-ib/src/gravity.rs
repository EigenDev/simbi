// =============================================================================
// gravity.rs
//
// mutual gravity between immersed bodies: a softened direct sum over every
// pair, `F = G m_i m_j r_vec / (|r|^2 + eps^2)^{3/2}`, applied at the body
// CENTERS. the force is central and equal-and-opposite, so pair momentum and
// pair angular momentum are conserved identically and no torque arises. a
// kinematic body (two_way_coupling = false) sources gravity but ignores the
// reaction — the fixed-potential convention the prescribed binary already
// uses. O(N^2) host-side; at fragment counts of order 10^3 this is noise next
// to the grid work, so there is no tree and no cutoff.
//
// usage:
//   let grav = MutualGravity { g: 1.0, softening: 0.05 };
//   advance_bonded(&mut coll, &mut bonds, None, Some(&grav), dt, &[]);
// =============================================================================

use crate::collection::BodyCollection;

/// mutual-gravity parameters: the gravitational constant in code units and a
/// plummer softening length shared by every pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MutualGravity {
    pub g: f64,
    pub softening: f64,
}

fn lift<const D: usize>(v: &symbi_algebra::Tensor<f64, D>) -> [f64; 3] {
    let mut out = [0.0; 3];
    for a in 0..D {
        out[a] = v[a];
    }
    out
}

/// add the pairwise gravitational forces to the per-body accumulators.
/// central forces at centers: no torque contribution.
pub(crate) fn accumulate_gravity<const D: usize>(
    force: &mut [[f64; 3]],
    grav: &MutualGravity,
    bodies: &BodyCollection<f64, D>,
) {
    let nb = bodies.len();
    let eps2 = grav.softening * grav.softening;
    for i in 0..nb {
        for j in (i + 1)..nb {
            let (bi, bj) = (bodies.get(i), bodies.get(j));
            if (!bi.two_way_coupling && !bj.two_way_coupling)
                || bi.mass <= 0.0
                || bj.mass <= 0.0
            {
                continue;
            }
            let xi = lift(&bi.position);
            let xj = lift(&bj.position);
            let r = [xj[0] - xi[0], xj[1] - xi[1], xj[2] - xi[2]];
            let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + eps2;
            let inv_r3 = 1.0 / (r2 * r2.sqrt());
            let c = grav.g * bi.mass * bj.mass * inv_r3;
            for a in 0..3 {
                force[i][a] += c * r[a];
                force[j][a] -= c * r[a];
            }
        }
    }
}

/// the shortest pair dynamical time of the current configuration,
/// `2 pi sqrt((|r|^2 + eps^2)^{3/2} / (G (m_i + m_j)))` — the substep law
/// resolves it with the same safety factor as a spring period. pairs with a
/// kinematic member still count (the mobile member falls in its field).
pub(crate) fn min_dynamical_time<const D: usize>(
    grav: &MutualGravity,
    bodies: &BodyCollection<f64, D>,
) -> f64 {
    let nb = bodies.len();
    let eps2 = grav.softening * grav.softening;
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut t_min = f64::INFINITY;
    for i in 0..nb {
        for j in (i + 1)..nb {
            let (bi, bj) = (bodies.get(i), bodies.get(j));
            if (!bi.two_way_coupling && !bj.two_way_coupling)
                || bi.mass <= 0.0
                || bj.mass <= 0.0
            {
                continue;
            }
            let mut r2 = eps2;
            for a in 0..D {
                let d = bj.position[a] - bi.position[a];
                r2 += d * d;
            }
            let gm = grav.g * (bi.mass + bj.mass);
            if gm > 0.0 {
                t_min = t_min.min(two_pi * (r2 * r2.sqrt() / gm).sqrt());
            }
        }
    }
    t_min
}

/// the gravitational potential energy of the configuration (mobile and
/// kinematic members alike), `-G m_i m_j / sqrt(|r|^2 + eps^2)` summed over
/// pairs — the bookkeeping side of the orbit gates.
pub fn gravitational_potential_energy<const D: usize>(
    grav: &MutualGravity,
    bodies: &BodyCollection<f64, D>,
) -> f64 {
    let nb = bodies.len();
    let eps2 = grav.softening * grav.softening;
    let mut pe = 0.0;
    for i in 0..nb {
        for j in (i + 1)..nb {
            let (bi, bj) = (bodies.get(i), bodies.get(j));
            if bi.mass <= 0.0 || bj.mass <= 0.0 {
                continue;
            }
            let mut r2 = eps2;
            for a in 0..D {
                let d = bj.position[a] - bi.position[a];
                r2 += d * d;
            }
            pe -= grav.g * bi.mass * bj.mass / r2.sqrt();
        }
    }
    pe
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::Body;
    use crate::bond::{advance_bonded, Bond, BondMaterial};
    use crate::contact::{ContactMaterial, Contacts};
    use symbi_algebra::Tensor;

    type V2 = Tensor<f64, 2>;

    fn body(x: f64, y: f64, vx: f64, vy: f64, mass: f64, mobile: bool) -> Body<f64, 2> {
        Body::rigid_sphere(0, V2::new([x, y]), V2::new([vx, vy]), mass, 0.5, 0.05, true)
            .with_two_way_coupling(mobile)
    }

    fn total_ke(coll: &BodyCollection<f64, 2>) -> f64 {
        coll.bodies().iter().map(|b| b.mechanical_ke()).sum()
    }

    #[test]
    fn equal_mass_binary_returns_after_one_period() {
        // relative circular orbit: v_rel = sqrt(G M / d), each body carries
        // half; the period is 2 pi d / v_rel.
        let v_half = 0.5 * (2.0_f64 / 1.0).sqrt();
        let mut coll = BodyCollection::<f64, 2>::new()
            .add_fragment(body(-0.5, 0.0, 0.0, -v_half, 1.0, true))
            .add_fragment(body(0.5, 0.0, 0.0, v_half, 1.0, true));
        let grav = MutualGravity { g: 1.0, softening: 0.0 };
        let mut bonds: Vec<Bond> = Vec::new();
        let period = 2.0 * std::f64::consts::PI / (2.0_f64).sqrt();

        let e0 = total_ke(&coll) + gravitational_potential_energy(&grav, &coll);
        let dt = 0.002;
        let steps = (period / dt).round() as usize;
        for _ in 0..steps {
            advance_bonded(&mut coll, &mut bonds, None, Some(&grav), dt, &[]);
        }
        for (b, x_expect) in [(0, -0.5), (1, 0.5)] {
            let pos = coll.get(b).position;
            let err = ((pos[0] - x_expect).powi(2) + pos[1].powi(2)).sqrt();
            assert!(err < 5e-3, "body {b} missed closure by {err}");
        }
        let p: [f64; 2] = coll.bodies().iter().fold([0.0; 2], |mut p, b| {
            p[0] += b.mass * b.velocity[0];
            p[1] += b.mass * b.velocity[1];
            p
        });
        assert!(p[0].abs() < 1e-12 && p[1].abs() < 1e-12, "momentum drift {p:?}");
        let e1 = total_ke(&coll) + gravitational_potential_energy(&grav, &coll);
        assert!((e1 - e0).abs() < 1e-4 * e0.abs(), "orbit energy drift {e0} -> {e1}");
    }

    #[test]
    fn softening_keeps_a_near_collision_finite() {
        let mut coll = BodyCollection::<f64, 2>::new()
            .add_fragment(body(-0.01, 0.0, 0.0, 0.0, 1.0, true))
            .add_fragment(body(0.01, 0.0, 0.0, 0.0, 1.0, true));
        let grav = MutualGravity { g: 1.0, softening: 0.1 };
        let mut bonds: Vec<Bond> = Vec::new();
        for _ in 0..200 {
            advance_bonded(&mut coll, &mut bonds, None, Some(&grav), 0.005, &[]);
        }
        for b in coll.bodies() {
            assert!(b.position[0].is_finite() && b.velocity[0].is_finite());
            assert!(b.velocity[0].abs() < 10.0, "softened pair reached {}", b.velocity[0]);
        }
    }

    // a seven-fragment rubble pile (hexagon + center, touching spheres held by
    // contact + self-gravity, no bonds) sent past a fixed central mass on a
    // parabolic orbit with pericenter q. returns the maximum pairwise
    // fragment separation at the end of the encounter.
    fn parabolic_encounter(q: f64) -> f64 {
        const M_CENTRAL: f64 = 1000.0;
        let grav = MutualGravity { g: 1.0, softening: 0.05 };
        let r0 = 40.0;
        // parabolic speed at r0 with angular momentum sqrt(2 G M q).
        let v2 = 2.0 * M_CENTRAL / r0;
        let v_t = (2.0 * M_CENTRAL * q).sqrt() / r0;
        let v_r = (v2 - v_t * v_t).sqrt();

        let mut coll = BodyCollection::<f64, 2>::new()
            .add_fragment(body(0.0, 0.0, 0.0, 0.0, M_CENTRAL, false));
        let mut positions = vec![(0.0, 0.0)];
        for k in 0..6 {
            let th = std::f64::consts::PI / 3.0 * k as f64;
            positions.push((th.cos(), th.sin()));
        }
        for (px, py) in positions {
            coll = coll.add_fragment(body(r0 + px, py, -v_r, v_t, 1.0, true));
        }
        let mut bonds: Vec<Bond> = Vec::new();
        let _ = BondMaterial::rigid();
        let mut contacts = Contacts::new(ContactMaterial {
            k_n: 1e5,
            k_t: 0.0,
            gamma_n: 20.0,
            mu: 0.0,
        });
        let mut t = 0.0;
        while t < 12.0 {
            advance_bonded(&mut coll, &mut bonds, Some(&mut contacts), Some(&grav), 0.01, &[]);
            t += 0.01;
        }
        let mut max_sep: f64 = 0.0;
        for i in 1..coll.len() {
            for j in (i + 1)..coll.len() {
                let (bi, bj) = (coll.get(i), coll.get(j));
                let dx = bj.position[0] - bi.position[0];
                let dy = bj.position[1] - bi.position[1];
                max_sep = max_sep.max((dx * dx + dy * dy).sqrt());
            }
        }
        max_sep
    }

    #[test]
    fn tidal_disruption_onset_brackets_the_roche_distance() {
        // tidal-vs-self-gravity balance for a pile of size a and mass m about
        // a point mass M: d_roche = a (2 M / m)^{1/3} ~ 6.6 for a ~ 1, m = 7,
        // M = 1000. a pericenter far outside keeps the pile compact; one far
        // inside shreds it.
        let far = parabolic_encounter(20.0);
        assert!(far < 4.0, "pile disrupted outside the roche distance: max sep {far}");
        let close = parabolic_encounter(2.0);
        assert!(close > 8.0, "pile survived deep inside the roche distance: max sep {close}");
    }
}
