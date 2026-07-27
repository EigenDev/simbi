// =============================================================================
// body.rs
//
// core data structures for immersed boundary bodies. a body represents a
// discrete physical object (black hole, planet, rigid sphere) embedded in
// the fluid grid.
//
// uses a flat enum for capabilities. all state updates are immutable
// (copy-and-modify).
//
// usage:
//   let bh = Body::black_hole(0, pos, vel, 1.0, 0.1, 0.04, 10.0, 0.0, 0.2);
//   let moved = bh.at_position(new_pos);
// =============================================================================

use symbi_algebra::Tensor;
use symbi_ir::algebra::Scalar;

/// the Rodrigues rotation matrix about unit axis `n` by `theta` (row-major):
/// `R = cos(theta) I + sin(theta) [n]_x + (1 - cos(theta)) n n^T`.
fn rodrigues_matrix<S: Scalar>(n: [S; 3], theta: S) -> [[S; 3]; 3] {
    let c = theta.cos();
    let s = theta.sin();
    let u = S::ONE - c;
    let (nx, ny, nz) = (n[0], n[1], n[2]);
    [
        [c + u * nx * nx, u * nx * ny - s * nz, u * nx * nz + s * ny],
        [u * ny * nx + s * nz, c + u * ny * ny, u * ny * nz - s * nx],
        [u * nz * nx - s * ny, u * nz * ny + s * nx, c + u * nz * nz],
    ]
}

/// the 3x3 matrix product `a * b` (row-major).
fn matmul3<S: Scalar>(a: [[S; 3]; 3], b: [[S; 3]; 3]) -> [[S; 3]; 3] {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j])
    })
}

/// the transpose of a 3x3 matrix (row-major).
fn transpose3<S: Scalar>(m: [[S; 3]; 3]) -> [[S; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| m[j][i]))
}

/// the 3x3 matrix-vector product `m * v`.
fn matvec3<S: Scalar>(m: [[S; 3]; 3], v: [S; 3]) -> [S; 3] {
    std::array::from_fn(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

/// the 3-vector cross product `a x b`.
pub(crate) fn cross3<S: Scalar>(a: [S; 3], b: [S; 3]) -> [S; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// capability-specific data for a body.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BodyKind<S: Scalar> {
    Passive,
    Gravitational {
        softening: S,
    },
    BlackHole {
        softening: S,
        sink_rate: S,
        accretion_radius: S,
        total_accreted_mass: S,
        accretion_rate: S,
        sink_delta: S, // 0 = torque-free, 1 = standard
    },
    /// the GR EXCISION HORIZON as a first-class immersed boundary: NOT a Newtonian point mass
    /// (gravity is the fixed metric, the sink is the vacuum excision), but a diagnostic-carrying
    /// absorbing surface. its accretion ledger — `total_accreted_{mass,energy}` + the instantaneous
    /// `mdot`/`edot` — is fed by the shell-flux reduction through a coordinate sphere at
    /// `diagnostic_radius` (OUTSIDE the horizon, where the flux is well-posed). the covariant energy
    /// makes `edot` a conserved, `diagnostic_radius`-invariant rate at steady state.
    Horizon {
        excision_radius: S,
        diagnostic_radius: S,
        total_accreted_mass: S,
        total_accreted_energy: S,
        mdot: S,
        edot: S,
    },
    Planet {
        softening: S,
        inertia: S,
        no_slip: bool,
    },
    RigidSphere {
        inertia: S,
        no_slip: bool,
    },
}

/// the hydrodynamic penalization stack a body's surface runs (the property
/// algebra). config-static — parameters, never state, never checkpointed.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum SurfaceSpec {
    /// the uniform-scaling drain: the validated accretor (p = 1).
    #[default]
    Drain,
    /// the porosity dial: `porosity` scales the drain channel, (1 - porosity)
    /// the wall channels; the wall rates are `k_eta_* c_s / dx`
    /// (multiplicative dials — zero is an exact off switch, so `k_eta_t = 0`
    /// is a free-slip surface).
    Porous {
        porosity: f64,
        k_eta_n: f64,
        k_eta_t: f64,
    },
    /// the torque-free accretor: the drain plus a tangential
    /// anti-relaxation `lambda_t = -xi lambda_rho`, so the accreted mass carries
    /// no net angular momentum to the body (the Dittmann sink). `xi in [0, 1]`:
    /// `xi = 0` is the standard drain, `xi = 1` fully torque-free. isothermal.
    TorqueFree { xi: f64 },
}

/// the magnetic coupling a body's surface runs — the MHD analog of `SurfaceSpec`.
/// config-static, never state, never checkpointed. `None` is transparent to the
/// magnetic field: a hydro run and an MHD run with a non-magnetic body evolve
/// identically, and a subgrid sink drains only the plasma while the flux is left to
/// constrained transport. the flux-anchor / resistive / beta-floor couplings land in
/// later work.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum MagneticSpec {
    /// no magnetic coupling: the body does not act on the magnetic field.
    #[default]
    None,
    /// localized Ohmic resistivity: a diffusivity `eta` masked by the body indicator `chi` adds the
    /// resistive edge EMF `eta*chi*J` before constrained transport, dissipating the magnetic field
    /// THREADING the body while leaving the exterior flux untouched. div-B-clean (the shared curl
    /// consumes it) and unconditionally dissipative (`-C diag(eta*chi) C^T` is negative-definite for
    /// `eta >= 0`), so the body can only shed field, never amplify it.
    Resistive { eta: f64 },
}

/// the full surface-coupling stack a body runs: the hydrodynamic surface physics and,
/// for MHD, the magnetic coupling — one per subsystem, each relaxing toward its own
/// declared target. config-static.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct BodySpec {
    /// the hydrodynamic surface: drain / porous wall / torque-free channels.
    pub surface: SurfaceSpec,
    /// the magnetic surface coupling (MHD); default `None` is transparent to B.
    pub magnetic: MagneticSpec,
}

/// a physical body embedded in the simulation grid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Body<S: Scalar, const D: usize> {
    pub idx: usize,
    pub position: Tensor<S, D>,
    pub velocity: Tensor<S, D>,
    pub force: Tensor<S, D>,
    pub torque: Tensor<S, 3>,
    pub mass: S,
    pub radius: S,
    pub two_way_coupling: bool,
    pub kind: BodyKind<S>,
    /// the surface-coupling stack: the hydrodynamic surface physics (`spec.surface`)
    /// and the magnetic coupling (`spec.magnetic`). kinematics stay on `kind`; this
    /// picks the baked kernel.
    pub spec: BodySpec,
    /// the body's ORIENTATION as a row-major rotation matrix `R` (init identity), advanced each step
    /// by the angular velocity. a shaped rigid wall's mask is `shape.rotated(R)` and its Dual-normal
    /// tracks it. every non-rotating body keeps the identity.
    pub orientation: [[S; 3]; 3],
    /// the ANGULAR VELOCITY vector `omega` (world frame, radians/time). a shaped wall's surface drags
    /// the gas at `omega x r`; a two-way body's omega evolves from the reaction torque via Euler's
    /// equations, so an asymmetric body tumbles. zero for every non-spinning body.
    pub omega: Tensor<S, 3>,
    /// the PRINCIPAL MOMENTS of inertia `(I1, I2, I3)` in the body frame (along the shape's local
    /// axes). anisotropic (unequal) moments make Euler's gyroscopic term nonzero -> torque-free
    /// precession/nutation (an asymmetric body wobbles). default isotropic (1, 1, 1).
    pub inertia_body: [S; 3],
}

// -- factory functions --

impl<S: Scalar, const D: usize> Body<S, D> {
    fn base(
        idx: usize,
        position: Tensor<S, D>,
        velocity: Tensor<S, D>,
        mass: S,
        radius: S,
        kind: BodyKind<S>,
    ) -> Self {
        Self {
            idx,
            position,
            velocity,
            force: Tensor::zeros(),
            torque: Tensor::zeros(),
            mass,
            radius,
            two_way_coupling: false,
            kind,
            spec: BodySpec::default(),
            orientation: [
                [S::ONE, S::ZERO, S::ZERO],
                [S::ZERO, S::ONE, S::ZERO],
                [S::ZERO, S::ZERO, S::ONE],
            ],
            omega: Tensor::zeros(),
            inertia_body: [S::ONE, S::ONE, S::ONE],
        }
    }

    /// declare the hydrodynamic surface stack (fluent; the default is the drain).
    pub fn with_surface(mut self, surface: SurfaceSpec) -> Self {
        self.spec.surface = surface;
        self
    }

    /// declare the magnetic coupling (fluent; the default is `None`, transparent to B).
    pub fn with_magnetic(mut self, magnetic: MagneticSpec) -> Self {
        self.spec.magnetic = magnetic;
        self
    }

    /// set the angular-velocity vector `omega` (world frame; fluent). a nonzero spin makes a shaped
    /// rigid wall rotate its mask + drag the gas at its surface.
    pub fn with_angular_velocity(mut self, omega: Tensor<S, 3>) -> Self {
        self.omega = omega;
        self
    }

    /// convenience: a spin RATE about `axis` (any nonzero vector, normalized here). fluent.
    pub fn with_spin_about(mut self, rate: S, axis: Tensor<S, 3>) -> Self {
        let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let s = rate / n;
        self.omega = Tensor::new([axis[0] * s, axis[1] * s, axis[2] * s]);
        self
    }

    /// convenience: a spin RATE about z. fluent.
    pub fn with_spin(mut self, rate: S) -> Self {
        self.omega = Tensor::new([S::ZERO, S::ZERO, rate]);
        self
    }

    /// set the principal moments of inertia `(I1, I2, I3)` in the body frame (fluent). unequal
    /// moments make an asymmetric body precess/tumble under Euler's gyroscopic term.
    pub fn with_inertia_principal(mut self, moments: [S; 3]) -> Self {
        self.inertia_body = moments;
        self
    }

    /// advance the FULL rigid-body rotation over `dt` under the external torque `torque_world` (world
    /// frame): integrate Euler's equations with the DIAGONAL body-frame inertia —
    /// `dL_body = (tau_body - omega_body x L_body) dt` — then roll the orientation by the updated
    /// angular velocity `R <- Rodrigues(omega_hat, |omega|*dt) R`. the gyroscopic term `omega x L`
    /// drives torque-free precession of an ANISOTROPIC body; ISOTROPIC inertia zeros it and this
    /// reduces to `omega += torque * dt / I`. `omega = 0`, `torque = 0` is a no-op.
    pub fn advance_rotation(&mut self, torque_world: Tensor<S, 3>, dt: S) {
        let r = self.orientation;
        let rt = transpose3(r);
        let i = self.inertia_body;
        // body-frame angular velocity + momentum, and the external torque increment in the body frame.
        let wb = matvec3(rt, [self.omega[0], self.omega[1], self.omega[2]]);
        let lb = [i[0] * wb[0], i[1] * wb[1], i[2] * wb[2]];
        let tb = matvec3(rt, [torque_world[0], torque_world[1], torque_world[2]]);
        // Euler over dt: L_body advances by the external torque minus the gyroscopic moment
        // omega_body x L_body, both integrated over the step: dL_body = (tau_body - omega x L) dt.
        let g = cross3(wb, lb);
        let lb_new = [
            lb[0] + (tb[0] - g[0]) * dt,
            lb[1] + (tb[1] - g[1]) * dt,
            lb[2] + (tb[2] - g[2]) * dt,
        ];
        let wb_new = [lb_new[0] / i[0], lb_new[1] / i[1], lb_new[2] / i[2]];
        self.omega = Tensor::new(matvec3(r, wb_new));
        // roll the orientation by the updated angular velocity.
        let w = [self.omega[0], self.omega[1], self.omega[2]];
        let wmag = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
        let nonzero = wmag.cmp_gt(S::ZERO);
        let divisor = S::select(nonzero, wmag, S::ONE);
        let inv = S::select(nonzero, S::ONE / divisor, S::ZERO);
        let axis = [w[0] * inv, w[1] * inv, w[2] * inv];
        let dr = rodrigues_matrix(axis, wmag * dt);
        self.orientation = matmul3(dr, r);
    }

    /// the body's mechanical kinetic energy: translational `0.5 m |v|^2` plus rotational
    /// `0.5 omega . I . omega`, with the world-frame inertia `I = R diag(inertia_body) R^T` (so
    /// `omega . I . omega = sum_k inertia_body[k] (R^T omega)_k^2`). this is the energy the gas
    /// force/torque deposit in a two-way body; the gas total-energy loss (`BodyDelta::energy_delta`)
    /// equals this KE gain plus the (non-negative) dissipated heat that stays in the gas — the
    /// gas+body conservation ledger for a sealed rigid wall.
    pub fn mechanical_ke(&self) -> S {
        let half = S::from_f64(0.5);
        let mut ke = S::ZERO;
        for a in 0..D {
            ke = ke + half * self.mass * self.velocity[a] * self.velocity[a];
        }
        // omega in the body frame, weighted by the principal moments.
        let wb = matvec3(
            transpose3(self.orientation),
            [self.omega[0], self.omega[1], self.omega[2]],
        );
        for k in 0..3 {
            ke = ke + half * self.inertia_body[k] * wb[k] * wb[k];
        }
        ke
    }

    /// the world-frame angular momentum `L = I omega = R (inertia_body ⊙ (R^T omega))` — the rigid-
    /// body spin state a viz glyph draws, and the conserved quantity a torque-free sink must not gain.
    pub fn angular_momentum(&self) -> Tensor<S, 3> {
        let wb = matvec3(
            transpose3(self.orientation),
            [self.omega[0], self.omega[1], self.omega[2]],
        );
        let lb = [
            self.inertia_body[0] * wb[0],
            self.inertia_body[1] * wb[1],
            self.inertia_body[2] * wb[2],
        ];
        Tensor::new(matvec3(self.orientation, lb))
    }

    /// the translational kinetic energy `0.5 m |v|^2` (the rigid drift).
    pub fn translational_ke(&self) -> S {
        let half = S::from_f64(0.5);
        let mut ke = S::ZERO;
        for a in 0..D {
            ke = ke + half * self.mass * self.velocity[a] * self.velocity[a];
        }
        ke
    }

    /// the rotational kinetic energy `0.5 omega . I . omega = 0.5 sum_k inertia_body[k] (R^T omega)_k^2`.
    pub fn rotational_ke(&self) -> S {
        let half = S::from_f64(0.5);
        let wb = matvec3(
            transpose3(self.orientation),
            [self.omega[0], self.omega[1], self.omega[2]],
        );
        let mut ke = S::ZERO;
        for k in 0..3 {
            ke = ke + half * self.inertia_body[k] * wb[k] * wb[k];
        }
        ke
    }

    pub fn passive(
        idx: usize,
        position: Tensor<S, D>,
        velocity: Tensor<S, D>,
        mass: S,
        radius: S,
    ) -> Self {
        Self::base(idx, position, velocity, mass, radius, BodyKind::Passive)
    }

    pub fn gravitational(
        idx: usize,
        position: Tensor<S, D>,
        velocity: Tensor<S, D>,
        mass: S,
        radius: S,
        softening: S,
    ) -> Self {
        Self::base(
            idx,
            position,
            velocity,
            mass,
            radius,
            BodyKind::Gravitational { softening },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn black_hole(
        idx: usize,
        position: Tensor<S, D>,
        velocity: Tensor<S, D>,
        mass: S,
        radius: S,
        softening: S,
        sink_rate: S,
        sink_delta: S,
        accretion_radius: S,
    ) -> Self {
        Self::base(
            idx,
            position,
            velocity,
            mass,
            radius,
            BodyKind::BlackHole {
                softening,
                sink_rate,
                accretion_radius,
                total_accreted_mass: S::ZERO,
                accretion_rate: S::ZERO,
                sink_delta,
            },
        )
    }

    /// the GR excision horizon: a static, massless (metric-gravity), diagnostic-only immersed
    /// boundary at the chart origin. `radius` is set to `diagnostic_radius` (the measurement shell).
    pub fn horizon(idx: usize, excision_radius: S, diagnostic_radius: S) -> Self {
        Self::base(
            idx,
            Tensor::zeros(),
            Tensor::zeros(),
            S::ZERO,
            diagnostic_radius,
            BodyKind::Horizon {
                excision_radius,
                diagnostic_radius,
                total_accreted_mass: S::ZERO,
                total_accreted_energy: S::ZERO,
                mdot: S::ZERO,
                edot: S::ZERO,
            },
        )
    }

    pub fn planet(
        idx: usize,
        position: Tensor<S, D>,
        velocity: Tensor<S, D>,
        mass: S,
        radius: S,
        inertia: S,
        no_slip: bool,
    ) -> Self {
        Self::base(
            idx,
            position,
            velocity,
            mass,
            radius,
            BodyKind::Planet {
                softening: S::ZERO,
                inertia,
                no_slip,
            },
        )
    }

    pub fn rigid_sphere(
        idx: usize,
        position: Tensor<S, D>,
        velocity: Tensor<S, D>,
        mass: S,
        radius: S,
        inertia: S,
        no_slip: bool,
    ) -> Self {
        let mut b = Self::base(
            idx,
            position,
            velocity,
            mass,
            radius,
            BodyKind::RigidSphere { inertia, no_slip },
        );
        // isotropic by default: the scalar inertia on all three principal axes.
        b.inertia_body = [inertia, inertia, inertia];
        b
    }
}

// -- immutable updates --

impl<S: Scalar, const D: usize> Body<S, D> {
    pub fn at_position(mut self, pos: Tensor<S, D>) -> Self {
        self.position = pos;
        self
    }

    pub fn with_velocity(mut self, vel: Tensor<S, D>) -> Self {
        self.velocity = vel;
        self
    }

    pub fn with_force(mut self, f: Tensor<S, D>) -> Self {
        self.force = f;
        self
    }

    pub fn with_torque(mut self, t: Tensor<S, 3>) -> Self {
        self.torque = t;
        self
    }

    pub fn with_mass(mut self, m: S) -> Self {
        self.mass = m;
        self
    }

    pub fn with_radius(mut self, r: S) -> Self {
        self.radius = r;
        self
    }

    pub fn with_two_way_coupling(mut self, flag: bool) -> Self {
        self.two_way_coupling = flag;
        self
    }
}

// -- capability queries --

impl<S: Scalar, const D: usize> Body<S, D> {
    pub fn has_gravity(&self) -> bool {
        matches!(
            self.kind,
            BodyKind::Gravitational { .. } | BodyKind::BlackHole { .. } | BodyKind::Planet { .. }
        )
    }

    pub fn has_accretion(&self) -> bool {
        matches!(
            self.kind,
            BodyKind::BlackHole { .. } | BodyKind::Horizon { .. }
        )
    }

    pub fn has_rigid(&self) -> bool {
        matches!(
            self.kind,
            BodyKind::RigidSphere { .. } | BodyKind::Planet { .. }
        )
    }

    /// softening length, or None if the body has no gravitational capability.
    pub fn softening(&self) -> Option<S> {
        match self.kind {
            BodyKind::Gravitational { softening, .. } => Some(softening),
            BodyKind::BlackHole { softening, .. } => Some(softening),
            BodyKind::Planet { softening, .. } => Some(softening),
            _ => None,
        }
    }

    /// accretion radius, or None if the body has no accretion capability.
    pub fn accretion_radius(&self) -> Option<S> {
        match self.kind {
            BodyKind::BlackHole {
                accretion_radius, ..
            } => Some(accretion_radius),
            _ => None,
        }
    }

    /// the penalization mask radius: the geometric scale the SDF mask and dispatch support
    /// use. an accretor masks to its accretion radius; a rigid sphere masks to its physical
    /// radius (the wall sits at the body surface). None if the body runs no surface
    /// penalization (passive / purely gravitational), which gates it out of the penalize step.
    pub fn mask_radius(&self) -> Option<S> {
        match self.kind {
            BodyKind::BlackHole {
                accretion_radius, ..
            } => Some(accretion_radius),
            BodyKind::RigidSphere { .. } => Some(self.radius),
            _ => None,
        }
    }

    /// sink rate, or None if the body has no accretion capability.
    pub fn sink_rate(&self) -> Option<S> {
        match self.kind {
            BodyKind::BlackHole { sink_rate, .. } => Some(sink_rate),
            _ => None,
        }
    }

    /// sink delta (torque control), or None if the body has no accretion capability.
    pub fn sink_delta(&self) -> Option<S> {
        match self.kind {
            BodyKind::BlackHole { sink_delta, .. } => Some(sink_delta),
            _ => None,
        }
    }

    /// moment of inertia, or None if the body has no rigid capability.
    pub fn inertia(&self) -> Option<S> {
        match self.kind {
            BodyKind::RigidSphere { inertia, .. } => Some(inertia),
            BodyKind::Planet { inertia, .. } => Some(inertia),
            _ => None,
        }
    }

    /// no-slip flag, or None if the body has no rigid capability.
    pub fn no_slip(&self) -> Option<bool> {
        match self.kind {
            BodyKind::RigidSphere { no_slip, .. } => Some(no_slip),
            BodyKind::Planet { no_slip, .. } => Some(no_slip),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type V2 = Tensor<f64, 2>;
    type V3 = Tensor<f64, 3>;

    #[test]
    fn passive_body() {
        let b = Body::passive(0, V2::new([1.0, 2.0]), V2::zeros(), 1.0, 0.1);
        assert!(!b.has_gravity());
        assert!(!b.has_accretion());
        assert!(!b.has_rigid());
    }

    #[test]
    fn gravitational_body() {
        let b = Body::gravitational(0, V2::new([0.0, 0.0]), V2::zeros(), 1.0, 0.1, 0.04);
        assert!(b.has_gravity());
        assert!(!b.has_accretion());
        assert_eq!(b.softening(), Some(0.04));
    }

    #[test]
    fn black_hole() {
        let b = Body::black_hole(
            0,
            V3::new([1.0, 0.0, 0.0]),
            V3::zeros(),
            1.0,
            0.1,
            0.04,
            10.0,
            0.0,
            0.2,
        );
        assert!(b.has_gravity());
        assert!(b.has_accretion());
        assert_eq!(b.softening(), Some(0.04));
        assert_eq!(b.accretion_radius(), Some(0.2));
        assert_eq!(b.sink_rate(), Some(10.0));
        assert_eq!(b.sink_delta(), Some(0.0));
    }

    #[test]
    fn planet() {
        let b = Body::planet(0, V2::new([1.0, 0.0]), V2::zeros(), 0.001, 0.05, 0.1, true);
        assert!(b.has_gravity());
        assert!(b.has_rigid());
        assert!(!b.has_accretion());
        assert_eq!(b.inertia(), Some(0.1));
        assert_eq!(b.no_slip(), Some(true));
    }

    #[test]
    fn rigid_sphere() {
        let b = Body::rigid_sphere(0, V2::new([0.0, 0.0]), V2::zeros(), 1.0, 0.5, 0.3, false);
        assert!(!b.has_gravity());
        assert!(b.has_rigid());
        assert_eq!(b.inertia(), Some(0.3));
        assert_eq!(b.no_slip(), Some(false));
    }

    #[test]
    fn immutable_updates() {
        let b = Body::passive(0, V2::new([1.0, 2.0]), V2::zeros(), 1.0, 0.1);
        let moved = b.at_position(V2::new([3.0, 4.0]));
        assert_eq!(moved.position, V2::new([3.0, 4.0]));
        assert_eq!(b.position, V2::new([1.0, 2.0])); // original unchanged (copy)

        let fast = b.with_velocity(V2::new([1.0, 0.0]));
        assert_eq!(fast.velocity, V2::new([1.0, 0.0]));

        let heavy = b.with_mass(10.0);
        assert_eq!(heavy.mass, 10.0);
    }

    #[test]
    fn with_two_way_coupling() {
        let b = Body::passive(0, V2::zeros(), V2::zeros(), 1.0, 0.1).with_two_way_coupling(true);
        assert!(b.two_way_coupling);
    }

    #[test]
    fn body_spec_defaults_to_the_drain_and_no_magnetic_coupling() {
        // base() relies on the default being the drain (the accretor) with a magnetic
        // coupling of None -- transparent to B, so an MHD run with a non-magnetic body
        // is bit-identical to the hydro path.
        let b = Body::<f64, 2>::passive(0, V2::zeros(), V2::zeros(), 1.0, 0.1);
        assert_eq!(b.spec, BodySpec::default());
        assert_eq!(b.spec.surface, SurfaceSpec::Drain);
        assert_eq!(b.spec.magnetic, MagneticSpec::None);
        // the builders route through `spec`, and the surface stack is untouched by the
        // magnetic declaration.
        let wall = SurfaceSpec::Porous {
            porosity: 0.0,
            k_eta_n: 50.0,
            k_eta_t: 50.0,
        };
        let b = b.with_surface(wall).with_magnetic(MagneticSpec::None);
        assert_eq!(b.spec.surface, wall);
        assert_eq!(b.spec.magnetic, MagneticSpec::None);
    }

    #[test]
    fn softening_none_on_passive() {
        let b = Body::passive(0, V2::zeros(), V2::zeros(), 1.0, 0.1);
        assert_eq!(b.softening(), None);
    }

    #[test]
    fn accretion_radius_none_on_grav() {
        let b = Body::gravitational(0, V2::zeros(), V2::zeros(), 1.0, 0.1, 0.04);
        assert_eq!(b.accretion_radius(), None);
    }

    #[test]
    fn inertia_none_on_passive() {
        let b = Body::passive(0, V2::zeros(), V2::zeros(), 1.0, 0.1);
        assert_eq!(b.inertia(), None);
    }
}
