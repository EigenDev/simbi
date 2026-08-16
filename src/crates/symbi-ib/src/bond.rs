// =============================================================================
// bond.rs
//
// breakable elastic bonds between rigid fragments: the bonded-particle
// discrete element model (cundall & strack 1979; potyondy & cundall 2004),
// host-side and grid-free. a bond joins two bodies along their center line:
// - normal spring on the center-distance extension `e = |x_j - x_i| - L0`
// - tangential spring on the accumulated slip of the surface material points
//   at the bond midpoint (the cundall-strack incremental shear accumulator,
//   re-projected perpendicular to the current normal every substep)
// - linear damper on the relative material velocity at the midpoint
// the total force is applied as +F on body i and -F on body j at the common
// center-line midpoint, so pair linear momentum and pair total angular
// momentum (orbital + spin) are conserved identically for every force direction:
//   d(momentum) = F - F = 0
//   d(L)        = (x_i - x_j) x F + (x_j - x_i) x F = 0
// the bond breaks (tombstoned in place, keeping the list order) when the tensile stress
// `k_n max(e, 0) / area` exceeds `sigma_t` or the shear stress `|F_t| / area`
// exceeds `tau_s`. breakage is a pure function of body state, so identical
// states produce identical break sets on every rank.
//
// time integration is velocity-verlet on translation, subcycled so the
// stiffest intact spring takes >= 10 substeps per period (and a damper at
// least 5 substeps per relaxation time); rotation advances by one explicit
// euler step of the rigid-body equations per substep. bodies with
// two_way_coupling = false are kinematic: they drift at their prescribed
// velocity and ignore bond forces (a clamp at v = 0, a pull at constant v).
//
// usage:
//   let mat = BondMaterial { k_n: 500.0, ..BondMaterial::rigid() };
//   let mut bonds = vec![Bond::form(0, 1, coll.get(0), coll.get(1), mat)];
//   let substeps = advance_bonded(&mut coll, &mut bonds, None, None, dt, &[]);
// =============================================================================

use crate::body::{Body, cross3};
use crate::collection::BodyCollection;
use symbi_algebra::Tensor;

/// material parameters of a bond: stiffnesses, damping, and the strength
/// envelope. `area` is the bond cross-section the stress criteria divide by
/// (pi r_b^2 for a circular bond of radius r_b).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BondMaterial {
    pub k_n: f64,
    pub k_t: f64,
    pub gamma: f64,
    pub area: f64,
    pub sigma_t: f64,
    pub tau_s: f64,
}

impl BondMaterial {
    /// an unbreakable, undamped, normal-only bond; override fields as needed.
    pub fn rigid() -> Self {
        Self {
            k_n: 0.0,
            k_t: 0.0,
            gamma: 0.0,
            area: 1.0,
            sigma_t: f64::MAX,
            tau_s: f64::MAX,
        }
    }
}

/// one bond between bodies `i` and `j` (collection indices). `rest_length` is
/// the center-to-center distance at formation; `slip` is the accumulated
/// tangential displacement of the material points at the bond midpoint
/// (world frame, kept perpendicular to the current normal). broken bonds
/// tombstone in place (`intact = false`) so the bond list order — and with it
/// every accumulation order — is fixed for the whole run.
#[derive(Clone, Copy, Debug)]
pub struct Bond {
    pub i: usize,
    pub j: usize,
    pub rest_length: f64,
    pub material: BondMaterial,
    pub slip: Tensor<f64, 3>,
    pub intact: bool,
}

impl Bond {
    /// form a bond between two bodies at their present separation: the center
    /// distance becomes the rest length (a pre-strained assembly is made by
    /// moving a body after formation).
    pub fn form<const D: usize>(
        i: usize,
        j: usize,
        body_i: &Body<f64, D>,
        body_j: &Body<f64, D>,
        material: BondMaterial,
    ) -> Self {
        assert!(i != j, "a bond joins two distinct bodies");
        let mut d2 = 0.0;
        for a in 0..D {
            let d = body_j.position[a] - body_i.position[a];
            d2 += d * d;
        }
        Self {
            i,
            j,
            rest_length: d2.sqrt(),
            material,
            slip: Tensor::zeros(),
            intact: true,
        }
    }
}

/// per-body load held constant across the subcycle (the gas force/torque of the
/// enclosing fluid step, or a test's applied load). the force acts at the body
/// center, the torque is world-frame.
#[derive(Clone, Copy, Debug)]
pub struct ExternalLoad<const D: usize> {
    pub force: Tensor<f64, D>,
    pub torque: Tensor<f64, 3>,
}

impl<const D: usize> ExternalLoad<D> {
    pub fn zero() -> Self {
        Self {
            force: Tensor::zeros(),
            torque: Tensor::zeros(),
        }
    }
}

/// the elastic energy stored in the intact bonds:
/// `sum 0.5 k_n e^2 + 0.5 k_t |slip|^2`. with the kinetic energy of the
/// bodies this is the mechanical energy a damped network holds flat or sheds.
pub fn bond_potential_energy<const D: usize>(
    bonds: &[Bond],
    bodies: &BodyCollection<f64, D>,
) -> f64 {
    let mut pe = 0.0;
    for bond in bonds.iter().filter(|b| b.intact) {
        let (bi, bj) = (bodies.get(bond.i), bodies.get(bond.j));
        let mut d2 = 0.0;
        for a in 0..D {
            let d = bj.position[a] - bi.position[a];
            d2 += d * d;
        }
        let e = d2.sqrt() - bond.rest_length;
        let mut s2 = 0.0;
        for a in 0..3 {
            s2 += bond.slip[a] * bond.slip[a];
        }
        pe += 0.5 * bond.material.k_n * e * e + 0.5 * bond.material.k_t * s2;
    }
    pe
}

// lift a D-vector into 3-space (zero-padded trailing components).
fn lift<const D: usize>(v: &Tensor<f64, D>) -> [f64; 3] {
    let mut out = [0.0; 3];
    for a in 0..D {
        out[a] = v[a];
    }
    out
}

// the world velocity of the body's material point at world position `p`:
// `v + omega x (p - x)`, all in 3-space.
fn material_velocity<const D: usize>(body: &Body<f64, D>, p: [f64; 3]) -> [f64; 3] {
    let x = lift(&body.position);
    let v = lift(&body.velocity);
    let r = [p[0] - x[0], p[1] - x[1], p[2] - x[2]];
    let wxr = cross3([body.omega[0], body.omega[1], body.omega[2]], r);
    [v[0] + wxr[0], v[1] + wxr[1], v[2] + wxr[2]]
}

// the force a pair interaction exerts on body i (body j receives the exact
// negation) and the common point it acts at — shared by bonds and contacts.
pub(crate) struct Kick {
    pub(crate) force_on_i: [f64; 3],
    pub(crate) midpoint: [f64; 3],
}

// book one pair kick into the per-body force/torque accumulators: +F on i and
// -F on j at the common point, torques from the levers to each center.
fn apply_kick<const D: usize>(
    force: &mut [[f64; 3]],
    torque: &mut [[f64; 3]],
    i: usize,
    j: usize,
    bi: &Body<f64, D>,
    bj: &Body<f64, D>,
    kick: &Kick,
) {
    let f = kick.force_on_i;
    let xi = lift(&bi.position);
    let xj = lift(&bj.position);
    let ri = [
        kick.midpoint[0] - xi[0],
        kick.midpoint[1] - xi[1],
        kick.midpoint[2] - xi[2],
    ];
    let rj = [
        kick.midpoint[0] - xj[0],
        kick.midpoint[1] - xj[1],
        kick.midpoint[2] - xj[2],
    ];
    let ti = cross3(ri, f);
    let tj = cross3(rj, [-f[0], -f[1], -f[2]]);
    for a in 0..3 {
        force[i][a] += f[a];
        force[j][a] -= f[a];
        torque[i][a] += ti[a];
        torque[j][a] += tj[a];
    }
}

fn eval_bond<const D: usize>(
    bond: &mut Bond,
    bi: &Body<f64, D>,
    bj: &Body<f64, D>,
) -> Option<Kick> {
    if !bond.intact {
        return None;
    }
    let m = &bond.material;
    let xi = lift(&bi.position);
    let xj = lift(&bj.position);
    let s = [xj[0] - xi[0], xj[1] - xi[1], xj[2] - xi[2]];
    let dist = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
    // two coincident centers have no defined normal; treat as an overlap the
    // contact layer owns and exert nothing this evaluation.
    if dist <= f64::MIN_POSITIVE {
        return None;
    }
    let n = [s[0] / dist, s[1] / dist, s[2] / dist];
    let e = dist - bond.rest_length;

    // tangential spring force from the slip accumulator, projected
    // perpendicular to the instantaneous normal (the accumulator itself is
    // re-projected during the drift phase; the projection here keeps the
    // force exactly tangential between re-projections).
    let sn = bond.slip[0] * n[0] + bond.slip[1] * n[1] + bond.slip[2] * n[2];
    let st = [
        bond.slip[0] - sn * n[0],
        bond.slip[1] - sn * n[1],
        bond.slip[2] - sn * n[2],
    ];
    let ft = [m.k_t * st[0], m.k_t * st[1], m.k_t * st[2]];
    let ft_mag = (ft[0] * ft[0] + ft[1] * ft[1] + ft[2] * ft[2]).sqrt();

    // strength envelope: tension-only on the normal channel, magnitude on the
    // shear channel. evaluated before the force is applied, so the breaking
    // evaluation itself exerts nothing.
    if m.k_n * e.max(0.0) / m.area > m.sigma_t || ft_mag / m.area > m.tau_s {
        bond.intact = false;
        return None;
    }

    let midpoint = [
        0.5 * (xi[0] + xj[0]),
        0.5 * (xi[1] + xj[1]),
        0.5 * (xi[2] + xj[2]),
    ];
    // damping on the relative material velocity at the midpoint: the force on
    // i follows the relative velocity of j's surface past i's, which makes the
    // pair power `-F_d . v_rel` non-positive.
    let vi = material_velocity(bi, midpoint);
    let vj = material_velocity(bj, midpoint);
    let vrel = [vj[0] - vi[0], vj[1] - vi[1], vj[2] - vi[2]];

    // normal spring pulls i toward j when stretched (e > 0).
    let force_on_i = [
        m.k_n * e * n[0] + ft[0] + m.gamma * vrel[0],
        m.k_n * e * n[1] + ft[1] + m.gamma * vrel[1],
        m.k_n * e * n[2] + ft[2] + m.gamma * vrel[2],
    ];
    Some(Kick {
        force_on_i,
        midpoint,
    })
}

// accumulate tangential slip over the drift: rotate the accumulator
// perpendicular to the current normal, then add the tangential part of the
// relative material displacement `v_rel_t * h` at the midpoint.
fn accumulate_slip<const D: usize>(bond: &mut Bond, bi: &Body<f64, D>, bj: &Body<f64, D>, h: f64) {
    if !bond.intact || bond.material.k_t == 0.0 {
        return;
    }
    let xi = lift(&bi.position);
    let xj = lift(&bj.position);
    let s = [xj[0] - xi[0], xj[1] - xi[1], xj[2] - xi[2]];
    let dist = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
    if dist <= f64::MIN_POSITIVE {
        return;
    }
    let n = [s[0] / dist, s[1] / dist, s[2] / dist];
    let midpoint = [
        0.5 * (xi[0] + xj[0]),
        0.5 * (xi[1] + xj[1]),
        0.5 * (xi[2] + xj[2]),
    ];
    let vi = material_velocity(bi, midpoint);
    let vj = material_velocity(bj, midpoint);
    let vrel = [vj[0] - vi[0], vj[1] - vi[1], vj[2] - vi[2]];
    let vn = vrel[0] * n[0] + vrel[1] * n[1] + vrel[2] * n[2];
    let sn = bond.slip[0] * n[0] + bond.slip[1] * n[1] + bond.slip[2] * n[2];
    for a in 0..3 {
        bond.slip[a] = (bond.slip[a] - sn * n[a]) + (vrel[a] - vn * n[a]) * h;
    }
}

/// the stable substep for the current bond network. every channel a body
/// feels is bounded, with the stiffness/damping of its intact bonds summed —
/// a body inside a lattice sees the row sum, so the bound tightens with its
/// coordination number:
/// - pair spring period `2 pi sqrt(m_eff / (k_n + k_t))` with >= 10 substeps
/// - per-body spring period `2 pi sqrt(m / sum k)` with >= 10 substeps
/// - per-body translational damping time `m / sum gamma` with >= 5 substeps
/// - per-body rotational spring/damping through the midpoint lever arm
///   `lc = rest_length / 2`: period `2 pi sqrt(I_min / sum k_t lc^2)` and
///   relaxation `I_min / sum gamma lc^2`, same safety factors (the damper
///   acts on the material velocity at the midpoint, so it torques spin even
///   when no tangential spring exists).
/// kinematic bodies carry no dynamics and impose nothing. a contact material,
/// when present, adds its worst-case row (any mobile body may touch a
/// neighbor of half its own mass) to every mobile body.
fn stable_substep<const D: usize>(
    bodies: &BodyCollection<f64, D>,
    bonds: &[Bond],
    contacts: Option<&crate::contact::Contacts>,
    gravity: Option<&crate::gravity::MutualGravity>,
) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    let nb = bodies.len();
    let n_src = bodies.source_count();
    // the two-way fragments are the bodies that integrate in the subcycle; the
    // source prefix and kinematic fragments carry prescribed motion instead.
    let dynamic = |idx: usize| idx >= n_src && bodies.get(idx).two_way_coupling;
    let mut k_trans = vec![0.0f64; nb];
    let mut k_rot = vec![0.0f64; nb];
    let mut g_trans = vec![0.0f64; nb];
    let mut g_rot = vec![0.0f64; nb];
    let mut h = f64::INFINITY;
    for bond in bonds.iter().filter(|b| b.intact) {
        let (bi, bj) = (bodies.get(bond.i), bodies.get(bond.j));
        let m = &bond.material;
        let m_eff = match (dynamic(bond.i), dynamic(bond.j)) {
            (true, true) => bi.mass * bj.mass / (bi.mass + bj.mass),
            (true, false) => bi.mass,
            (false, true) => bj.mass,
            (false, false) => continue,
        };
        let k = m.k_n + m.k_t;
        if k > 0.0 {
            h = h.min(two_pi * (m_eff / k).sqrt() / 10.0);
        }
        let lc2 = 0.25 * bond.rest_length * bond.rest_length;
        for (idx, _body) in [(bond.i, bi), (bond.j, bj)] {
            if dynamic(idx) {
                k_trans[idx] += k;
                k_rot[idx] += m.k_t * lc2;
                g_trans[idx] += m.gamma;
                g_rot[idx] += m.gamma * lc2;
            }
        }
    }
    for b in 0..nb {
        let body = bodies.get(b);
        if !dynamic(b) || body.mass <= 0.0 {
            continue;
        }
        if let Some(cts) = contacts {
            let (kt, gt, kr) = cts.stability_row(body.radius);
            k_trans[b] += kt;
            g_trans[b] += gt;
            k_rot[b] += kr;
        }
        if k_trans[b] > 0.0 {
            h = h.min(two_pi * (body.mass / k_trans[b]).sqrt() / 10.0);
        }
        if g_trans[b] > 0.0 {
            h = h.min(body.mass / g_trans[b] / 5.0);
        }
        let i_min = body.inertia_body[0]
            .min(body.inertia_body[1])
            .min(body.inertia_body[2]);
        if i_min > 0.0 {
            if k_rot[b] > 0.0 {
                h = h.min(two_pi * (i_min / k_rot[b]).sqrt() / 10.0);
            }
            if g_rot[b] > 0.0 {
                h = h.min(i_min / g_rot[b] / 5.0);
            }
        }
    }
    // the closest gravitating pair sets a dynamical time the drift must
    // resolve with the same safety factor as a spring period.
    if let Some(grav) = gravity {
        h = h.min(crate::gravity::min_dynamical_time(grav, bodies) / 10.0);
    }
    h
}

/// advance the bonded body system over one enclosing step `dt`: subcycled
/// velocity-verlet translation + per-substep euler rotation, with `external`
/// per-body loads (the enclosing step's gas force/torque) held frozen.
/// mobility tiers:
/// - the source prefix (`idx < bodies.source_count()`) is fully frozen here —
///   the legacy body integrator (prescribed binary / fixed-potential sink)
///   owns its motion; sources still exert bond/contact/gravity forces on
///   fragments and absorb none of the reaction (the fixed-potential
///   convention).
/// - a fragment with `two_way_coupling = false` is kinematic: it drifts at
///   its prescribed velocity and ignores forces (a clamp, a puller).
/// - every other fragment integrates under the accumulated pair forces.
/// `external` is either empty or one load per body. returns the substep count.
/// the pair loop reads only body state and the bond list, both identical on
/// every rank of a decomposed run, so break sets and trajectories are
/// reproduced identically wherever the same state is advanced.
pub fn advance_bonded<const D: usize>(
    bodies: &mut BodyCollection<f64, D>,
    bonds: &mut [Bond],
    mut contacts: Option<&mut crate::contact::Contacts>,
    gravity: Option<&crate::gravity::MutualGravity>,
    dt: f64,
    external: &[ExternalLoad<D>],
) -> usize {
    let nb = bodies.len();
    assert!(
        external.is_empty() || external.len() == nb,
        "external loads: {} for {} bodies",
        external.len(),
        nb,
    );
    let h_stable = stable_substep(bodies, bonds, contacts.as_deref(), gravity);
    let n_sub = if h_stable.is_finite() {
        (dt / h_stable).ceil().max(1.0) as usize
    } else {
        1
    };
    let h = dt / n_sub as f64;

    // per-body force (3-space) + torque accumulators, rebuilt each evaluation.
    let mut force = vec![[0.0f64; 3]; nb];
    let mut torque = vec![[0.0f64; 3]; nb];
    let accumulate = |force: &mut Vec<[f64; 3]>,
                      torque: &mut Vec<[f64; 3]>,
                      bodies: &BodyCollection<f64, D>,
                      bonds: &mut [Bond],
                      contacts: Option<&crate::contact::Contacts>| {
        for f in force.iter_mut() {
            *f = [0.0; 3];
        }
        for t in torque.iter_mut() {
            *t = [0.0; 3];
        }
        for bond in bonds.iter_mut() {
            let (bi, bj) = (bodies.get(bond.i), bodies.get(bond.j));
            if let Some(kick) = eval_bond(bond, bi, bj) {
                apply_kick(force, torque, bond.i, bond.j, bi, bj, &kick);
            }
        }
        // contact acts on every touching pair left unowned by an intact bond.
        if let Some(cts) = contacts {
            let bonded = crate::contact::bonded_pairs(bonds);
            for i in 0..nb {
                for j in (i + 1)..nb {
                    let (bi, bj) = (bodies.get(i), bodies.get(j));
                    if bonded.contains(&(i, j)) || (!bi.two_way_coupling && !bj.two_way_coupling) {
                        continue;
                    }
                    if let Some(kick) = cts.kick((i, j), bi, bj) {
                        apply_kick(force, torque, i, j, bi, bj, &kick);
                    }
                }
            }
        }
        // mutual gravity: central forces at centers, no torque.
        if let Some(grav) = gravity {
            crate::gravity::accumulate_gravity(force, grav, bodies);
        }
        if !external.is_empty() {
            for b in 0..nb {
                let ef = lift(&external[b].force);
                for a in 0..3 {
                    force[b][a] += ef[a];
                    torque[b][a] += external[b].torque[a];
                }
            }
        }
    };

    // one half kick: velocity gains h/2 F/m and the rotation advances by the
    // matching half-impulse of torque. pairing every spin half-impulse with the
    // velocity half-impulse from the same force evaluation is what conserves
    // the pair total angular momentum: per evaluation the orbital change
    // `x x (h/2 F)` and the spin change `(mid - x) x (h/2 F)` sum to
    // `mid x (h/2 F)` on each side of the bond and cancel exactly. a full-step
    // rotation from one evaluation would leak angular momentum at O(h).
    fn half_kick<const D: usize>(
        bodies: &mut BodyCollection<f64, D>,
        force: &[[f64; 3]],
        torque: &[[f64; 3]],
        h: f64,
        n_src: usize,
    ) {
        for b in n_src..bodies.len() {
            let body = bodies.get_mut(b);
            if body.two_way_coupling && body.mass > 0.0 {
                for a in 0..D {
                    body.velocity[a] += 0.5 * h * force[b][a] / body.mass;
                }
            }
            let t = if body.two_way_coupling {
                Tensor::new(torque[b])
            } else {
                Tensor::zeros()
            };
            body.advance_rotation(t, 0.5 * h);
        }
    }

    let n_src = bodies.source_count();
    for _ in 0..n_sub {
        // opening half kick from forces at the current state.
        accumulate(&mut force, &mut torque, bodies, bonds, contacts.as_deref());
        half_kick(bodies, &force, &torque, h, n_src);
        // drift every fragment at its (half-kicked or prescribed) velocity;
        // the source prefix holds its position.
        for b in n_src..nb {
            let body = bodies.get_mut(b);
            for a in 0..D {
                body.position[a] += h * body.velocity[a];
            }
        }
        // slip accumulates over the drift, at the post-drift geometry with the
        // half-step velocities.
        for bond in bonds.iter_mut() {
            let (bi, bj) = (bodies.get(bond.i), bodies.get(bond.j));
            let (bi, bj) = (*bi, *bj);
            accumulate_slip(bond, &bi, &bj, h);
        }
        if let Some(cts) = contacts.as_deref_mut() {
            cts.update_slip(bodies, bonds, h);
        }
        // closing half kick from forces at the drifted state.
        accumulate(&mut force, &mut torque, bodies, bonds, contacts.as_deref());
        half_kick(bodies, &force, &torque, h, n_src);
    }
    n_sub
}

/// the pair-physics carrier a fragment-bearing simulation attaches to its
/// immersed-body side-car: the bond list plus the optional contact and mutual
/// gravity subsystems, advanced together by [`advance_bonded`].
#[derive(Clone, Debug, Default)]
pub struct FragmentPhysics {
    pub bonds: Vec<Bond>,
    pub contacts: Option<crate::contact::Contacts>,
    pub gravity: Option<crate::gravity::MutualGravity>,
}

impl FragmentPhysics {
    /// advance the fragments over one enclosing step with the given frozen
    /// per-body loads; returns the substep count.
    pub fn advance<const D: usize>(
        &mut self,
        bodies: &mut BodyCollection<f64, D>,
        dt: f64,
        external: &[ExternalLoad<D>],
    ) -> usize {
        advance_bonded(
            bodies,
            &mut self.bonds,
            self.contacts.as_mut(),
            self.gravity.as_ref(),
            dt,
            external,
        )
    }

    /// number of intact bonds (the breakage ledger the history can log).
    pub fn intact_bonds(&self) -> usize {
        self.bonds.iter().filter(|b| b.intact).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type V2 = Tensor<f64, 2>;

    fn sphere(idx: usize, x: f64, y: f64, vx: f64, vy: f64, mobile: bool) -> Body<f64, 2> {
        Body::rigid_sphere(
            idx,
            V2::new([x, y]),
            V2::new([vx, vy]),
            1.0,
            0.45,
            0.05,
            true,
        )
        .with_two_way_coupling(mobile)
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

    // total angular momentum about the origin: orbital `x cross m v` plus
    // every body's spin angular momentum (z components; planar motion).
    fn total_lz(coll: &BodyCollection<f64, 2>) -> f64 {
        let mut lz = 0.0;
        for b in coll.bodies() {
            lz += b.mass * (b.position[0] * b.velocity[1] - b.position[1] * b.velocity[0]);
            lz += b.angular_momentum()[2];
        }
        lz
    }

    fn total_ke(coll: &BodyCollection<f64, 2>) -> f64 {
        coll.bodies().iter().map(|b| b.mechanical_ke()).sum()
    }

    fn separation(coll: &BodyCollection<f64, 2>, i: usize, j: usize) -> f64 {
        let (bi, bj) = (coll.get(i), coll.get(j));
        let dx = bj.position[0] - bi.position[0];
        let dy = bj.position[1] - bi.position[1];
        (dx * dx + dy * dy).sqrt()
    }

    #[test]
    fn bonded_pair_oscillates_at_the_analytic_frequency() {
        let mut coll = BodyCollection::<f64, 2>::new()
            .add_fragment(sphere(0, 0.0, 0.0, 0.0, 0.0, true))
            .add_fragment(sphere(0, 1.0, 0.0, 0.0, 0.0, true));
        let mat = BondMaterial {
            k_n: 250.0,
            ..BondMaterial::rigid()
        };
        let mut bonds = vec![Bond::form(0, 1, coll.get(0), coll.get(1), mat)];
        // stretch after formation: rest length stays 1.0, extension 0.1.
        coll.get_mut(1).position[0] = 1.1;

        let e0 = total_ke(&coll) + bond_potential_energy(&bonds, &coll);
        let dt = 0.005;
        let mut crossings: Vec<f64> = Vec::new();
        let mut prev = separation(&coll, 0, 1) - 1.0;
        let mut t = 0.0;
        while t < 3.0 {
            advance_bonded(&mut coll, &mut bonds, None, None, dt, &[]);
            t += dt;
            let e = separation(&coll, 0, 1) - 1.0;
            // upward zero crossing, linearly interpolated inside the step.
            if prev < 0.0 && e >= 0.0 {
                crossings.push(t - dt * e / (e - prev));
            }
            prev = e;
        }
        assert!(
            crossings.len() >= 8,
            "expected many periods, got {}",
            crossings.len()
        );
        let period = (crossings.last().unwrap() - crossings[0]) / (crossings.len() as f64 - 1.0);
        // m_eff = 1/2, omega = sqrt(k_n / m_eff), period = 2 pi / omega
        let expected = 2.0 * std::f64::consts::PI * (0.5 / 250.0_f64).sqrt();
        assert!(
            (period - expected).abs() < 0.01 * expected,
            "period {period} vs analytic {expected}"
        );

        let p = total_momentum(&coll);
        assert!(
            p[0].abs() < 1e-12 && p[1].abs() < 1e-12,
            "momentum drift {p:?}"
        );
        // velocity-verlet energy oscillates within an O((h omega)^2) band and
        // stays inside it over the run; h omega ~ 0.11 puts the band at ~0.3%.
        let e1 = total_ke(&coll) + bond_potential_energy(&bonds, &coll);
        assert!(
            (e1 - e0).abs() < 1e-2 * e0,
            "undamped energy drift {e0} -> {e1}"
        );
    }

    #[test]
    fn damped_pair_decays_at_the_analytic_envelope_rate() {
        let mut coll = BodyCollection::<f64, 2>::new()
            .add_fragment(sphere(0, 0.0, 0.0, 0.0, 0.0, true))
            .add_fragment(sphere(0, 1.0, 0.0, 0.0, 0.0, true));
        let mat = BondMaterial {
            k_n: 250.0,
            gamma: 2.0,
            ..BondMaterial::rigid()
        };
        let mut bonds = vec![Bond::form(0, 1, coll.get(0), coll.get(1), mat)];
        coll.get_mut(1).position[0] = 1.1;

        // successive extension maxima follow exp(-gamma t / (2 m_eff)).
        let dt = 0.002;
        let mut peaks: Vec<(f64, f64)> = Vec::new();
        let (mut prev_e, mut prev_de) = (0.1, 0.0);
        let mut t = 0.0;
        while t < 2.0 {
            advance_bonded(&mut coll, &mut bonds, None, None, dt, &[]);
            t += dt;
            let e = separation(&coll, 0, 1) - 1.0;
            let de = e - prev_e;
            if prev_de > 0.0 && de <= 0.0 && e > 0.0 {
                peaks.push((t, e));
            }
            prev_de = de;
            prev_e = e;
        }
        assert!(
            peaks.len() >= 4,
            "expected several maxima, got {}",
            peaks.len()
        );
        let (t0, a0) = peaks[0];
        let (t1, a1) = *peaks.last().unwrap();
        let rate = (a0 / a1).ln() / (t1 - t0);
        // gamma / (2 m_eff) = 2.0 / (2 * 0.5)
        let expected = 2.0;
        assert!(
            (rate - expected).abs() < 0.05 * expected,
            "decay rate {rate} vs {expected}"
        );
    }

    #[test]
    fn pair_conserves_momentum_and_angular_momentum_with_spin_and_damping() {
        // head-on radial approach plus one spinning body: nearly all the
        // energy is dissipatable (the conserved angular momentum is the small
        // spin-only L, so the rigid co-rotation floor is far below the start).
        let mut coll = BodyCollection::<f64, 2>::new()
            .add_fragment(sphere(0, 0.0, 0.0, 0.3, 0.0, true))
            .add_fragment(sphere(0, 1.05, 0.0, -0.3, 0.0, true));
        coll.get_mut(0).omega = Tensor::new([0.0, 0.0, 1.5]);
        let mat = BondMaterial {
            k_n: 200.0,
            k_t: 150.0,
            gamma: 3.0,
            ..BondMaterial::rigid()
        };
        let mut bonds = vec![Bond::form(0, 1, coll.get(0), coll.get(1), mat)];

        let lz0 = total_lz(&coll);
        let e0 = total_ke(&coll) + bond_potential_energy(&bonds, &coll);
        let mut e_prev = e0;
        for _ in 0..400 {
            advance_bonded(&mut coll, &mut bonds, None, None, 0.005, &[]);
            let e = total_ke(&coll) + bond_potential_energy(&bonds, &coll);
            // the damper dissipates; the integrator adds a bounded
            // O((h omega)^2) oscillation on top, which stays bounded.
            assert!(
                e <= e0 * (1.0 + 5e-3),
                "damped energy exceeded start: {e0} -> {e}"
            );
            e_prev = e;
        }
        assert!(
            e_prev < 0.25 * e0,
            "damping dissipated almost nothing: {e0} -> {e_prev}"
        );
        let p = total_momentum(&coll);
        assert!(
            p[0].abs() < 1e-12 && p[1].abs() < 1e-12,
            "momentum drift {p:?}"
        );
        let lz1 = total_lz(&coll);
        assert!(
            (lz1 - lz0).abs() < 1e-9 * lz0.abs().max(1.0),
            "angular momentum drift {lz0} -> {lz1}"
        );
        assert!(e_prev < e0, "damping dissipated nothing");
    }

    #[test]
    fn braced_lattice_cantilever_matches_euler_bernoulli() {
        // a 2 x 12 pin-jointed strip, x-braced, clamped at column 0. bending
        // stiffness comes from chord stretching alone (k_t = 0): the truss
        // flexural rigidity is ei = 2 (k_n L0)(d/2)^2 with chord spacing d.
        const COLS: usize = 8;
        let idx = |col: usize, row: usize| col * 2 + row;
        let mut coll = BodyCollection::<f64, 2>::new();
        for col in 0..COLS {
            for row in 0..2 {
                let mobile = col > 0;
                coll = coll.add_fragment(sphere(0, col as f64, row as f64, 0.0, 0.0, mobile));
            }
        }
        let mat = BondMaterial {
            k_n: 4000.0,
            gamma: 8.0,
            ..BondMaterial::rigid()
        };
        let mut bonds = Vec::new();
        for col in 0..COLS {
            bonds.push(Bond::form(
                idx(col, 0),
                idx(col, 1),
                coll.get(idx(col, 0)),
                coll.get(idx(col, 1)),
                mat,
            ));
            if col + 1 < COLS {
                for row in 0..2 {
                    bonds.push(Bond::form(
                        idx(col, row),
                        idx(col + 1, row),
                        coll.get(idx(col, row)),
                        coll.get(idx(col + 1, row)),
                        mat,
                    ));
                }
                bonds.push(Bond::form(
                    idx(col, 0),
                    idx(col + 1, 1),
                    coll.get(idx(col, 0)),
                    coll.get(idx(col + 1, 1)),
                    mat,
                ));
                bonds.push(Bond::form(
                    idx(col, 1),
                    idx(col + 1, 0),
                    coll.get(idx(col, 1)),
                    coll.get(idx(col + 1, 0)),
                    mat,
                ));
            }
        }

        // transverse tip load, half on each chord end.
        let p_load = 1.0;
        let mut external = vec![ExternalLoad::<2>::zero(); coll.len()];
        external[idx(COLS - 1, 0)].force = V2::new([0.0, 0.5 * p_load]);
        external[idx(COLS - 1, 1)].force = V2::new([0.0, 0.5 * p_load]);

        // relax to statics by kinetic quenching (the standard dem statics
        // scheme): periodically zero the mobile velocities so the network
        // descends its potential; settled when the ke regained between
        // quenches vanishes and the tip stops moving.
        let tip_of = |coll: &BodyCollection<f64, 2>| {
            0.5 * (coll.get(idx(COLS - 1, 0)).position[1] + coll.get(idx(COLS - 1, 1)).position[1]
                - 1.0)
        };
        let mut settled = false;
        let mut tip_prev = f64::NAN;
        for it in 0..4000 {
            advance_bonded(&mut coll, &mut bonds, None, None, 0.02, &external);
            if it % 25 == 24 {
                let ke = total_ke(&coll);
                for b in 0..coll.len() {
                    let body = coll.get_mut(b);
                    if body.two_way_coupling {
                        body.velocity = Tensor::zeros();
                        body.omega = Tensor::zeros();
                    }
                }
                let tip_now = tip_of(&coll);
                if ke < 1e-10 && (tip_now - tip_prev).abs() < 1e-8 {
                    settled = true;
                    break;
                }
                tip_prev = tip_now;
            }
        }
        assert!(
            settled,
            "cantilever did not reach statics, ke = {}",
            total_ke(&coll)
        );

        let tip = tip_of(&coll);
        // euler-bernoulli tip deflection P L^3 / (3 ei), beam length measured
        // clamp column to tip column.
        let ei = 2.0 * 4000.0 * 1.0 * 0.25;
        let l = (COLS - 1) as f64;
        let expected = p_load * l * l * l / (3.0 * ei);
        assert!(
            (tip - expected).abs() < 0.25 * expected,
            "tip deflection {tip} vs euler-bernoulli {expected}"
        );
    }

    // pull a five-body chain apart at prescribed end velocity; returns the
    // macro step at which the weak bond broke and the final position bits.
    fn run_break_chain() -> (usize, Vec<u64>) {
        let mut coll = BodyCollection::<f64, 2>::new();
        for k in 0..5 {
            let mobile = k != 0 && k != 4;
            let vx = if k == 4 { 0.02 } else { 0.0 };
            coll = coll.add_fragment(sphere(0, k as f64, 0.0, vx, 0.0, mobile));
        }
        let strong = BondMaterial {
            k_n: 500.0,
            gamma: 10.0,
            ..BondMaterial::rigid()
        };
        let weak = BondMaterial {
            sigma_t: 5.0,
            ..strong
        };
        let mut bonds: Vec<Bond> = (0..4)
            .map(|k| {
                let m = if k == 1 { weak } else { strong };
                Bond::form(k, k + 1, coll.get(k), coll.get(k + 1), m)
            })
            .collect();

        let mut break_step = 0;
        for step in 1..=400 {
            advance_bonded(&mut coll, &mut bonds, None, None, 0.01, &[]);
            if !bonds[1].intact {
                break_step = step;
                break;
            }
        }
        assert!(break_step > 0, "weak bond never broke");
        assert!(
            bonds[0].intact && bonds[2].intact && bonds[3].intact,
            "wrong bond broke"
        );
        // quasi-static series chain: every bond carries the same tension, the
        // weak one parts at e = sigma_t area / k_n = 0.01, so the end
        // separation at breakage is 4 (1 + 0.01) to quasi-static accuracy.
        let sep = separation(&coll, 0, 4);
        assert!((4.02..=4.08).contains(&sep), "break separation {sep}");

        let mut bits = Vec::new();
        for b in coll.bodies() {
            bits.push(b.position[0].to_bits());
            bits.push(b.position[1].to_bits());
        }
        (break_step, bits)
    }

    #[test]
    fn stretched_chain_breaks_at_the_weak_bond_deterministically() {
        let (step_a, bits_a) = run_break_chain();
        let (step_b, bits_b) = run_break_chain();
        assert_eq!(step_a, step_b, "break step differs between identical runs");
        assert_eq!(bits_a, bits_b, "trajectories differ between identical runs");
    }

    #[test]
    fn stiff_network_subcycles_to_stability() {
        let mut coll = BodyCollection::<f64, 2>::new()
            .add_fragment(sphere(0, 0.0, 0.0, 0.0, 0.0, true))
            .add_fragment(sphere(0, 1.0, 0.0, 0.0, 0.0, true));
        let mat = BondMaterial {
            k_n: 1e6,
            gamma: 1.0,
            ..BondMaterial::rigid()
        };
        let mut bonds = vec![Bond::form(0, 1, coll.get(0), coll.get(1), mat)];
        coll.get_mut(1).position[0] = 1.001;

        let e0 = total_ke(&coll) + bond_potential_energy(&bonds, &coll);
        let n_sub = advance_bonded(&mut coll, &mut bonds, None, None, 0.05, &[]);
        // pair period 2 pi sqrt(m_eff / k_n) ~ 4.4e-3: the enclosing step must
        // split into >= 10 substeps per period.
        assert!(n_sub > 100, "stiff spring under-subcycled: {n_sub}");
        for _ in 0..10 {
            advance_bonded(&mut coll, &mut bonds, None, None, 0.05, &[]);
        }
        let e1 = total_ke(&coll) + bond_potential_energy(&bonds, &coll);
        assert!(
            e1.is_finite() && e1 <= e0 * 1.001,
            "stiff network gained energy {e0} -> {e1}"
        );
    }
}
