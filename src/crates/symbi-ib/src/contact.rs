// =============================================================================
// contact.rs
//
// soft-sphere contact between rigid fragments (cundall & strack 1979): pairs
// whose bounding spheres overlap while free of an intact bond repel
// through a linear spring-dashpot on the overlap, with coulomb-capped
// tangential friction from a slip accumulator at the contact point. an intact
// bond owns its pair's interaction entirely; contact takes over the moment the
// bond breaks.
//
// the normal force is non-attractive by construction: the dashpot may reduce
// the repulsion of a separating pair but the total normal force clamps at
// zero, so a contact stays purely repulsive. the tangential spring force saturates at
// the coulomb cone `|F_t| <= mu F_n` (the accumulator is clipped to the cap,
// the standard cundall-strack sliding rule). the total force is applied as
// +F on body i and -F on body j at the common contact point, the same
// conservation spelling as the bond module: pair linear momentum and pair
// total angular momentum are conserved identically.
//
// slip state persists per contacting pair in a BTreeMap keyed (i, j), so
// iteration and accumulation order are deterministic; entries are dropped on
// separation.
//
// usage:
//   let mut contacts = Contacts::new(ContactMaterial {
//       k_n: 1e3, k_t: 8e2, gamma_n: 4.0, mu: 0.3,
//   });
//   advance_bonded(&mut coll, &mut bonds, Some(&mut contacts), None, dt, &[]);
// =============================================================================

use crate::body::Body;
use crate::bond::{Bond, Kick};
use crate::collection::BodyCollection;
use std::collections::{BTreeMap, BTreeSet};
use symbi_algebra::Tensor;

/// contact-law parameters shared by every fragment pair: normal spring `k_n`,
/// tangential spring `k_t`, normal dashpot `gamma_n`, friction coefficient
/// `mu`. the spring-dashpot restitution is
/// `e = exp(-pi zeta / sqrt(1 - zeta^2))` with
/// `zeta = gamma_n / (2 sqrt(k_n m_eff))`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactMaterial {
    pub k_n: f64,
    pub k_t: f64,
    pub gamma_n: f64,
    pub mu: f64,
}

/// the contact subsystem: the material plus the per-pair tangential slip
/// accumulators of the touching pairs.
#[derive(Clone, Debug)]
pub struct Contacts {
    pub material: ContactMaterial,
    slip: BTreeMap<(usize, usize), Tensor<f64, 3>>,
}

// geometry of one candidate pair: outward data the force and slip phases share.
struct Overlap {
    delta: f64,
    n: [f64; 3],
    point: [f64; 3],
    v_rel: [f64; 3],
}

fn lift<const D: usize>(v: &Tensor<f64, D>) -> [f64; 3] {
    let mut out = [0.0; 3];
    for a in 0..D {
        out[a] = v[a];
    }
    out
}

fn overlap_of<const D: usize>(bi: &Body<f64, D>, bj: &Body<f64, D>) -> Option<Overlap> {
    let xi = lift(&bi.position);
    let xj = lift(&bj.position);
    let s = [xj[0] - xi[0], xj[1] - xi[1], xj[2] - xi[2]];
    let dist = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
    let delta = bi.radius + bj.radius - dist;
    if delta <= 0.0 || dist <= f64::MIN_POSITIVE {
        return None;
    }
    let n = [s[0] / dist, s[1] / dist, s[2] / dist];
    // the center of the overlap lens along the line of centers.
    let reach = bi.radius - 0.5 * delta;
    let point = [
        xi[0] + n[0] * reach,
        xi[1] + n[1] * reach,
        xi[2] + n[2] * reach,
    ];
    let vi = material_velocity(bi, point);
    let vj = material_velocity(bj, point);
    Some(Overlap {
        delta,
        n,
        point,
        v_rel: [vj[0] - vi[0], vj[1] - vi[1], vj[2] - vi[2]],
    })
}

fn material_velocity<const D: usize>(body: &Body<f64, D>, p: [f64; 3]) -> [f64; 3] {
    let x = lift(&body.position);
    let v = lift(&body.velocity);
    let r = [p[0] - x[0], p[1] - x[1], p[2] - x[2]];
    let wxr = crate::body::cross3([body.omega[0], body.omega[1], body.omega[2]], r);
    [v[0] + wxr[0], v[1] + wxr[1], v[2] + wxr[2]]
}

// non-attractive normal force magnitude: spring on the overlap, dashpot on the
// separation rate, clamped at zero so a departing contact pushes or releases.
fn normal_magnitude(mat: &ContactMaterial, delta: f64, v_n: f64) -> f64 {
    (mat.k_n * delta - mat.gamma_n * v_n).max(0.0)
}

/// the set of pairs an intact bond owns, keyed (min, max).
pub(crate) fn bonded_pairs(bonds: &[Bond]) -> BTreeSet<(usize, usize)> {
    bonds
        .iter()
        .filter(|b| b.intact)
        .map(|b| (b.i.min(b.j), b.i.max(b.j)))
        .collect()
}

impl Contacts {
    pub fn new(material: ContactMaterial) -> Self {
        Self {
            material,
            slip: BTreeMap::new(),
        }
    }

    /// number of pairs carrying contact state.
    pub fn active(&self) -> usize {
        self.slip.len()
    }

    // the contact force on body i of the pair, at the contact point. reads the
    // stored slip; saturation against the coulomb cone happens here on the
    // force (the accumulator itself is clipped in `update_slip`).
    pub(crate) fn kick<const D: usize>(
        &self,
        key: (usize, usize),
        bi: &Body<f64, D>,
        bj: &Body<f64, D>,
    ) -> Option<Kick> {
        let ov = overlap_of(bi, bj)?;
        let m = &self.material;
        let v_n = ov.v_rel[0] * ov.n[0] + ov.v_rel[1] * ov.n[1] + ov.v_rel[2] * ov.n[2];
        let f_n = normal_magnitude(m, ov.delta, v_n);
        let mut f_t = [0.0; 3];
        if let Some(slip) = self.slip.get(&key) {
            let sn = slip[0] * ov.n[0] + slip[1] * ov.n[1] + slip[2] * ov.n[2];
            for a in 0..3 {
                f_t[a] = m.k_t * (slip[a] - sn * ov.n[a]);
            }
            let mag = (f_t[0] * f_t[0] + f_t[1] * f_t[1] + f_t[2] * f_t[2]).sqrt();
            let cap = m.mu * f_n;
            if mag > cap && mag > 0.0 {
                let scale = cap / mag;
                for a in 0..3 {
                    f_t[a] *= scale;
                }
            }
        }
        // repulsion pushes i away from j (along -n).
        Some(Kick {
            force_on_i: [
                -f_n * ov.n[0] + f_t[0],
                -f_n * ov.n[1] + f_t[1],
                -f_n * ov.n[2] + f_t[2],
            ],
            midpoint: ov.point,
        })
    }

    /// advance the slip accumulators over one substep: for every touching
    /// unbonded pair, re-project the stored slip perpendicular to the current
    /// normal, add the tangential relative material displacement, and clip to
    /// the coulomb cone; drop the state of separated pairs.
    pub(crate) fn update_slip<const D: usize>(
        &mut self,
        bodies: &BodyCollection<f64, D>,
        bonds: &[Bond],
        h: f64,
    ) {
        let bonded = bonded_pairs(bonds);
        let nb = bodies.len();
        let mut touching: BTreeSet<(usize, usize)> = BTreeSet::new();
        for i in 0..nb {
            for j in (i + 1)..nb {
                let (bi, bj) = (bodies.get(i), bodies.get(j));
                if bonded.contains(&(i, j)) || (!bi.two_way_coupling && !bj.two_way_coupling) {
                    continue;
                }
                let Some(ov) = overlap_of(bi, bj) else {
                    continue;
                };
                touching.insert((i, j));
                let m = self.material;
                let slip = self.slip.entry((i, j)).or_insert_with(Tensor::zeros);
                let sn = slip[0] * ov.n[0] + slip[1] * ov.n[1] + slip[2] * ov.n[2];
                let vn = ov.v_rel[0] * ov.n[0] + ov.v_rel[1] * ov.n[1] + ov.v_rel[2] * ov.n[2];
                for a in 0..3 {
                    slip[a] = (slip[a] - sn * ov.n[a]) + (ov.v_rel[a] - vn * ov.n[a]) * h;
                }
                // sliding: clip the accumulator to the coulomb cone so the
                // stored spring force stays within `mu F_n`.
                if m.k_t > 0.0 && m.mu.is_finite() {
                    let f_n = normal_magnitude(&m, ov.delta, vn);
                    let cap = m.mu * f_n / m.k_t;
                    let mag = (slip[0] * slip[0] + slip[1] * slip[1] + slip[2] * slip[2]).sqrt();
                    if mag > cap && mag > 0.0 {
                        let scale = cap / mag;
                        for a in 0..3 {
                            slip[a] *= scale;
                        }
                    }
                }
            }
        }
        self.slip.retain(|key, _| touching.contains(key));
    }

    // stability contributions for the substep law: any mobile body may enter a
    // contact, whose worst-case pair mass is half its own, so the springs and
    // dashpot count twice in its translational row sums; the tangential spring
    // torques through the body radius.
    pub(crate) fn stability_row(&self, radius: f64) -> (f64, f64, f64) {
        let m = &self.material;
        let k_trans = 2.0 * (m.k_n + m.k_t);
        let g_trans = 2.0 * m.gamma_n;
        let k_rot = m.k_t * radius * radius;
        (k_trans, g_trans, k_rot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::{Bond, BondMaterial, advance_bonded};

    type V2 = Tensor<f64, 2>;

    fn ball(x: f64, y: f64, vx: f64, vy: f64) -> Body<f64, 2> {
        Body::rigid_sphere(0, V2::new([x, y]), V2::new([vx, vy]), 1.0, 0.5, 0.05, true)
            .with_two_way_coupling(true)
    }

    fn total_momentum(coll: &BodyCollection<f64, 2>) -> [f64; 2] {
        let mut p = [0.0; 2];
        for b in coll.bodies() {
            for a in 0..2 {
                p[a] += b.mass * b.velocity[a];
            }
        }
        p
    }

    fn total_lz(coll: &BodyCollection<f64, 2>) -> f64 {
        let mut lz = 0.0;
        for b in coll.bodies() {
            lz += b.mass * (b.position[0] * b.velocity[1] - b.position[1] * b.velocity[0]);
            lz += b.angular_momentum()[2];
        }
        lz
    }

    fn separation(coll: &BodyCollection<f64, 2>) -> f64 {
        let (bi, bj) = (coll.get(0), coll.get(1));
        let dx = bj.position[0] - bi.position[0];
        let dy = bj.position[1] - bi.position[1];
        (dx * dx + dy * dy).sqrt()
    }

    #[test]
    fn elastic_head_on_bounce_reverses_the_velocities() {
        let mut coll = BodyCollection::<f64, 2>::new()
            .add_fragment(ball(0.0, 0.0, 0.5, 0.0))
            .add_fragment(ball(2.0, 0.0, -0.5, 0.0));
        let mut contacts = Contacts::new(ContactMaterial {
            k_n: 1000.0,
            k_t: 0.0,
            gamma_n: 0.0,
            mu: 0.0,
        });
        let mut bonds: Vec<Bond> = Vec::new();
        let ke0: f64 = coll.bodies().iter().map(|b| b.mechanical_ke()).sum();
        for _ in 0..440 {
            advance_bonded(&mut coll, &mut bonds, Some(&mut contacts), None, 0.005, &[]);
        }
        assert!(
            separation(&coll) > 1.5,
            "pair did not separate: {}",
            separation(&coll)
        );
        let (v0, v1) = (coll.get(0).velocity[0], coll.get(1).velocity[0]);
        assert!((v0 + 0.5).abs() < 0.005, "left exit velocity {v0}");
        assert!((v1 - 0.5).abs() < 0.005, "right exit velocity {v1}");
        let p = total_momentum(&coll);
        assert!(
            p[0].abs() < 1e-12 && p[1].abs() < 1e-12,
            "momentum drift {p:?}"
        );
        let ke1: f64 = coll.bodies().iter().map(|b| b.mechanical_ke()).sum();
        assert!(
            (ke1 - ke0).abs() < 1e-2 * ke0,
            "elastic bounce energy {ke0} -> {ke1}"
        );
        assert_eq!(contacts.active(), 0, "slip state leaked past separation");
    }

    #[test]
    fn restitution_matches_the_spring_dashpot_analytic() {
        // zeta = gamma_n / (2 sqrt(k_n m_eff)) = 0.1 for the linear
        // spring-dashpot; e = exp(-pi zeta / sqrt(1 - zeta^2)) ~ 0.729. the
        // non-attraction clamp trims the adhesive tail, so the band is loose.
        let mut coll = BodyCollection::<f64, 2>::new()
            .add_fragment(ball(0.0, 0.0, 0.5, 0.0))
            .add_fragment(ball(2.0, 0.0, -0.5, 0.0));
        let mut contacts = Contacts::new(ContactMaterial {
            k_n: 1000.0,
            k_t: 0.0,
            gamma_n: 4.4721359549995795,
            mu: 0.0,
        });
        let mut bonds: Vec<Bond> = Vec::new();
        for _ in 0..440 {
            advance_bonded(&mut coll, &mut bonds, Some(&mut contacts), None, 0.005, &[]);
        }
        assert!(separation(&coll) > 1.2, "pair did not separate");
        let rel_exit = coll.get(1).velocity[0] - coll.get(0).velocity[0];
        let zeta: f64 = 0.1;
        let expected = (-std::f64::consts::PI * zeta / (1.0 - zeta * zeta).sqrt()).exp();
        assert!(
            (rel_exit - expected).abs() < 0.1 * expected,
            "restitution {rel_exit} vs analytic {expected}"
        );
    }

    fn run_oblique() -> (BodyCollection<f64, 2>, [f64; 2]) {
        let mut coll = BodyCollection::<f64, 2>::new()
            .add_fragment(ball(0.0, 0.0, 0.6, 0.0))
            .add_fragment(ball(2.0, 0.55, 0.0, 0.0));
        let mut contacts = Contacts::new(ContactMaterial {
            k_n: 1000.0,
            k_t: 800.0,
            gamma_n: 4.4721359549995795,
            mu: 0.3,
        });
        let mut bonds: Vec<Bond> = Vec::new();
        // record the line of centers when contact first engages.
        let mut n0 = [0.0f64; 2];
        for _ in 0..1200 {
            advance_bonded(&mut coll, &mut bonds, Some(&mut contacts), None, 0.005, &[]);
            if n0 == [0.0; 2] && separation(&coll) < 1.0 {
                let d = [
                    coll.get(1).position[0] - coll.get(0).position[0],
                    coll.get(1).position[1] - coll.get(0).position[1],
                ];
                let mag = (d[0] * d[0] + d[1] * d[1]).sqrt();
                n0 = [d[0] / mag, d[1] / mag];
            }
        }
        (coll, n0)
    }

    #[test]
    fn oblique_impact_obeys_the_coulomb_cone_and_spins_the_target() {
        let (coll, n0) = run_oblique();
        assert!(n0 != [0.0; 2], "contact never engaged");
        assert!(separation(&coll) > 1.1, "pair did not separate");
        // the struck body's impulse decomposed along the engagement normal:
        // the tangential part is bounded by the coulomb cone.
        let jb = [coll.get(1).velocity[0], coll.get(1).velocity[1]];
        let j_n = jb[0] * n0[0] + jb[1] * n0[1];
        let jt = [jb[0] - j_n * n0[0], jb[1] - j_n * n0[1]];
        let j_t = (jt[0] * jt[0] + jt[1] * jt[1]).sqrt();
        assert!(j_n > 0.05, "no normal impulse transferred");
        assert!(j_t > 1e-4, "friction transferred nothing");
        assert!(
            j_t / j_n <= 0.3 * 1.15,
            "coulomb cone violated: |J_t|/|J_n| = {}",
            j_t / j_n
        );
        assert!(
            coll.get(1).omega[2].abs() > 1e-3,
            "friction imparted no spin"
        );
        let p = total_momentum(&coll);
        assert!(
            (p[0] - 0.6).abs() < 1e-12 && p[1].abs() < 1e-12,
            "momentum drift {p:?}"
        );
    }

    #[test]
    fn oblique_impact_conserves_total_angular_momentum() {
        let mut coll = BodyCollection::<f64, 2>::new()
            .add_fragment(ball(0.0, 0.0, 0.6, 0.0))
            .add_fragment(ball(2.0, 0.55, 0.0, 0.0));
        let lz0 = total_lz(&coll);
        let mut contacts = Contacts::new(ContactMaterial {
            k_n: 1000.0,
            k_t: 800.0,
            gamma_n: 4.4721359549995795,
            mu: 0.3,
        });
        let mut bonds: Vec<Bond> = Vec::new();
        for _ in 0..1200 {
            advance_bonded(&mut coll, &mut bonds, Some(&mut contacts), None, 0.005, &[]);
        }
        let lz1 = total_lz(&coll);
        assert!(
            (lz1 - lz0).abs() < 1e-9 * lz0.abs().max(1.0),
            "angular momentum drift {lz0} -> {lz1}"
        );
    }

    #[test]
    fn separating_overlap_never_attracts() {
        // dashpot pull would exceed the spring push here; the clamp must hold
        // the normal force at zero so the pair coasts apart unimpeded.
        let mut coll = BodyCollection::<f64, 2>::new()
            .add_fragment(ball(0.0, 0.0, -0.4, 0.0))
            .add_fragment(ball(0.9, 0.0, 0.4, 0.0));
        let mut contacts = Contacts::new(ContactMaterial {
            k_n: 1000.0,
            k_t: 0.0,
            gamma_n: 200.0,
            mu: 0.0,
        });
        let mut bonds: Vec<Bond> = Vec::new();
        let mut prev_sep = separation(&coll);
        for _ in 0..400 {
            advance_bonded(&mut coll, &mut bonds, Some(&mut contacts), None, 0.005, &[]);
            let sep = separation(&coll);
            assert!(
                sep >= prev_sep,
                "separating pair pulled back: {prev_sep} -> {sep}"
            );
            prev_sep = sep;
        }
        let rel = coll.get(1).velocity[0] - coll.get(0).velocity[0];
        assert!(rel >= 0.8 - 1e-9, "separation speed decayed: {rel}");
        assert!(prev_sep > 1.2, "pair failed to separate: {prev_sep}");
    }

    #[test]
    fn intact_bond_owns_the_pair_until_breakage() {
        // a bond holds the pair overlapped at rest length 0.8 < r_i + r_j:
        // contact stays silent while the bond is intact, then pushes the
        // fragments apart to touching distance once it breaks.
        let mut coll = BodyCollection::<f64, 2>::new()
            .add_fragment(ball(0.0, 0.0, 0.0, 0.0))
            .add_fragment(ball(0.8, 0.0, 0.0, 0.0));
        let mat = BondMaterial {
            k_n: 500.0,
            gamma: 5.0,
            ..BondMaterial::rigid()
        };
        let mut bonds = vec![Bond::form(0, 1, coll.get(0), coll.get(1), mat)];
        let mut contacts = Contacts::new(ContactMaterial {
            k_n: 1000.0,
            k_t: 0.0,
            gamma_n: 5.0,
            mu: 0.0,
        });
        for _ in 0..200 {
            advance_bonded(&mut coll, &mut bonds, Some(&mut contacts), None, 0.01, &[]);
            let sep = separation(&coll);
            assert!(
                (sep - 0.8).abs() < 0.02,
                "contact fired through an intact bond: sep {sep}"
            );
        }
        bonds[0].intact = false;
        for _ in 0..600 {
            advance_bonded(&mut coll, &mut bonds, Some(&mut contacts), None, 0.01, &[]);
        }
        let sep = separation(&coll);
        assert!(
            sep > 1.0,
            "contact failed to expel the overlap after breakage: {sep}"
        );
    }

    #[test]
    fn frictional_collision_is_deterministic() {
        let run = || {
            let (coll, _) = run_oblique();
            let mut bits = Vec::new();
            for b in coll.bodies() {
                bits.push(b.position[0].to_bits());
                bits.push(b.position[1].to_bits());
                bits.push(b.omega[2].to_bits());
            }
            bits
        };
        assert_eq!(run(), run(), "identical frictional runs diverged");
    }
}
