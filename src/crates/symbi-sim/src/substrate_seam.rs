// =============================================================================
// substrate_seam.rs
//
// the sim <-> substrate seam: the abstractions the sim core needs to TALK ABOUT
// substrates without depending on any concrete regime KernelSet.
// homing `KernelSet` and the `Solver`/`RegimeKind` enums here keeps
// `FieldStore`/`state.rs` from depending UP into `regimes` (which would be a cycle).
//
// homed here in `symbi-sim`, every reference from the sim core points DOWN; the
// concrete kernelsets in the `symbi` crate's `regimes/` IMPLEMENT `KernelSet` and
// depend DOWN on this module.
//
// contents:
// - KernelSet      — one method per physics operation; the integrator calls only these.
// - Solver / RegimeKind — the Riemann-solver + regime-family classification enums.
//
// the regime -> concrete-KernelSet map (`RegimeSubstrate`) and the `sim.substrate()`
// convenience are NOT here: they name concrete kernelsets, so they live in the substrate
// layer (the `symbi` crate). the orphan rule enforces this — a trait mapping the foreign
// `Regime` types to local kernelsets must be local to the crate that owns the kernelsets.
//
// usage:
//  impl KernelSet<D, D, Mem, Sc> for IsoSubstrateKernelSet<Mem, Sc, D> { .. }
// =============================================================================

use symbi_algebra::OrderedNumeric;
use symbi_hydro::regime::Regime;
use symbi_ir::algebra::Scalar;
use symbi_xpu::MemorySpace;

use crate::state::FieldStore;

// =============================================================================
// KernelSet trait
// =============================================================================

/// one method per physics operation. each regime implements this with
/// concrete kernel dispatches. the driver calls only these methods —
/// never touches sim.fields directly.
///
/// source terms (gravity, geometric) are fused into godunov_euler and
/// godunov_rk2. unused sources produce zero via scalar parameters
/// (e.g., body_mass = 0.0). no separate source passes.
/// `NDIM` = grid dimension, `DOF` = vector (momentum-component) dimension;
/// they coincide for the natural case and diverge for axisymmetric (DOF>NDIM).
// the kernel-set sees ONLY the `FieldStore`: its 4 storage params, never
// the physics tags `R`/`M`/`E` or the executor `S` (the concrete set bakes `R::SPEC` /
// `eos_param` at construction — it does not read them off the sim). this is the keystone
// decoupling: 4 params, and the energy/schema bounds off `R` stay LOCAL to
// `FieldStore`. impls name the `&FieldStore` argument `sim`.
pub trait KernelSet<const NDIM: usize, const DOF: usize, Mem, Sc>
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    fn flux(&self, store: &FieldStore<NDIM, DOF, Mem, Sc>, dir: usize);

    /// stencil halfwidth of the evolution face reconstruction along the sweep:
    /// 2 for the plm family (pcm rides plm at theta = 0), 3 for ppm. the
    /// coarse-fine transfer must prolong one order higher than the evolution
    /// reconstruction or the boundary degrades the interior order; the widest
    /// baked prolongation (the ppm sub-cell average) covers plm evolution only,
    /// so a refinement hierarchy refuses any set reporting a reach above 2.
    fn reconstruction_reach(&self) -> u8 {
        2
    }
    fn c2p(&self, store: &FieldStore<NDIM, DOF, Mem, Sc>);
    // the SSP Shu-Osher stage update `cons = a0*u_n + ac*(cons - dt*div(F) + dt*S)`. one method
    // serves every explicit SSP scheme; the driver feeds the per-stage convex coefficients
    // (a0, ac) from the `Timestepping` table (forward-Euler = (0, 1)).
    fn godunov_stage(&self, store: &FieldStore<NDIM, DOF, Mem, Sc>, dt: f64, a0: f64, ac: f64);
    fn cfl(&self, store: &FieldStore<NDIM, DOF, Mem, Sc>) -> f64;
    fn ghost_fill(&self, store: &FieldStore<NDIM, DOF, Mem, Sc>);
    fn snapshot(&self, store: &FieldStore<NDIM, DOF, Mem, Sc>);

    /// FIRST-ORDER FLUX CORRECTION, run per RK substage AFTER c2p: any zone whose high-order c2p
    /// went unphysical (p <= 0, rho <= 0, NaN) is redone with first-order (PCM + HLLE) fluxes
    /// reconstructed from the PHYSICAL stage-input state (`u_stage`), and the sharp high-order state
    /// is kept everywhere else. a floor-free robustness layer: cells the sharp scheme cannot recover
    /// fall back to the diffusive-but-robust first-order update with no pressure floor. host-gated
    /// on a failure reduction, so a clean substage pays only the scan. default: no-op.
    ///
    /// returns whether the WHOLE STEP must be rejected: GRMHD limits the geometric source against
    /// a source-free low-order anchor, and an anchor that is itself inadmissible is a statement
    /// about the timestep, not the source. the driver then rolls the step back through
    /// `restore_step` and replays it at a smaller dt.
    fn fofc(
        &self,
        _store: &FieldStore<NDIM, DOF, Mem, Sc>,
        _dt: f64,
        _a0: f64,
        _ac: f64,
        _stage: u8,
    ) -> bool {
        false
    }

    /// whether this kernel set runs FOFC (`fofc` is non-trivial). when true the driver also takes the
    /// per-stage `u_stage` snapshot every substage (FOFC restores `cons <- u_stage` to reconstruct the
    /// first-order fluxes from the physical stage input). default: false.
    fn fofc_active(&self) -> bool {
        false
    }

    /// retain the complete state needed to replay a rejected explicit step. paired with
    /// `restore_step`; a kernel set that implements one must implement the other, and a driver
    /// must gate BOTH on `fofc_active` — restoring from a snapshot that was never taken would
    /// overwrite the live state with zeros.
    fn snapshot_retry(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>) {}

    /// restore the complete step-entry state after `fofc` requests a retry: the conserved gas
    /// state, the staggered and cell-centered magnetic field, AND the primitives derived from
    /// them. the retried step re-enters at `wave_speeds`/`flux`, which reconstruct from `prim`,
    /// so an implementation that rolls back only the conserved state would reconstruct the
    /// rejected attempt's primitives.
    fn restore_step(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>) {}

    /// post-godunov hook (e.g., constrained transport for MHD). default: no-op.
    fn post_godunov(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>, _dt: f64, _stage: u8) {}

    /// electric field computation (MHD). default: no-op.
    fn efield(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>) {}

    /// materialize per-cell wave speeds BEFORE the flux reads them (RMHD: the exact quartic
    /// into wave_speed_l/r, so the flux does a cheap Davis fan without re-solving the
    /// quartic per face). runs each stage after the prim is current (post c2p+ghost). default:
    /// no-op — regimes with cheap algebraic wave speeds keep computing them inline in the flux.
    fn wave_speeds(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>) {}

    /// forward immersed-body source (gravity): `cons += dt * S_body`, applied
    /// per RK stage after godunov when the sim has bodies. default: no-op (body-free regimes).
    fn body_source(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>, _dt: f64) {}

    /// the passive-scalar (dye) stage update: `cons.chi` advances on the
    /// materialized mass flux (donor-cell upwind) in the same SSP form as the
    /// gas, then the concentration `prim.chi = cons.chi/den` is recovered. runs
    /// AFTER the fofc phase so a spliced mass flux and the stage-final density
    /// are what the dye rides. default: no-op (regimes without the dye wired).
    fn chi_update(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>, _dt: f64, _a0: f64, _ac: f64) {}

    /// the per-axis interface dye flux `flux[d].chi = mass_flux_d * upwind(prim.chi)`, written
    /// during the FLUX phase so the coarse-fine registers sample it alongside the gas fluxes they
    /// correct. separate from `chi_update` for that reason alone: the divergence that consumes it
    /// runs later, after fofc has settled the mass flux it rides.
    /// default: no-op (regimes without the dye wired).
    fn chi_flux(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>) {}

    /// does this kernel set carry a NON-fused (additive) source overlay? gates the per-stage
    /// `snapshot_stage` + `source_apply` in `step()`. default: false (fused / source-free sets).
    fn has_additive_source(&self) -> bool {
        false
    }

    /// snapshot the stage-INPUT cons into `u_stage`, BEFORE the godunov stage overwrites it.
    /// the additive `source_apply` then evaluates `S` at this state — the same state the fused
    /// stage uses — making `plain + additive == fused` bit-for-bit. default: no-op.
    fn snapshot_stage(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>) {}

    /// the additive source pass: `cons += weight * S(u_stage)`, applied per RK stage AFTER
    /// godunov, with the SSP stage weight `weight = ac*dt`. the general (non-fused) source
    /// execution; default: no-op.
    fn source_apply(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>, _weight: f64) {}

    /// backward immersed-body feedback: reduce the per-body force/torque/
    /// accreted-mass from the fluid into the side-car diagnostics, once per step. default: no-op.
    fn body_feedback(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>, _dt: f64) {}

    /// the GR horizon shell-flux accretion diagnostic: the rest-mass and covariant (killing) energy
    /// rates `(mdot, edot)` crossing a coordinate sphere at `diagnostic_radius`, measured by a
    /// GPU Add-reduction of the per-cell outward boundary flux of `Omega = { r_ks < diagnostic_radius }`
    /// (divergence-theorem-consistent with the godunov; `edot` is `diagnostic_radius`-invariant at
    /// steady state). default `(0, 0)`: flat backgrounds / regimes without the baked shell kernels.
    fn horizon_accretion(
        &self,
        _store: &FieldStore<NDIM, DOF, Mem, Sc>,
        _diagnostic_radius: f64,
    ) -> (f64, f64) {
        (0.0, 0.0)
    }

    /// the immersed-boundary penalization: the property-
    /// algebra surface physics (drain, walls, porosity, thermal surfaces),
    /// applied post-source each substage — the ONE body
    /// mechanism. default: no-op (regimes without a baked penalize envelope).
    fn penalize(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>, _dt: f64) {}

    /// the constant-nu viscous transport pass: the Navier-Stokes
    /// shear operator, applied post-step once the primitive velocity is current.
    /// default: no-op (inviscid regimes, and sets without a baked viscous kernel).
    fn viscous(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>, _dt: f64) {}

    /// horizon excision: overwrite the causally disconnected cells inside the
    /// kerr-schild-radius level set r_ks(x; a) < r_exc (the sphere about the chart
    /// origin at a = 0, the oblate spheroid at spin — strictly inside the horizon)
    /// with a cold vacuum floor on the gas primitives + a local conserved rebuild,
    /// once per step after the RK combination. magnetized
    /// sets rebuild with the cell's own B; the staggered faces stay CT-owned.
    /// default: no-op (flat backgrounds, and regimes without the baked kernels).
    /// composed from the sweep/finalize pieces so the DECOMPOSED loop can drive the
    /// passes itself, exchanging halos around them: the pass count comes from the
    /// store, so both drivers run the same number, and the exchange after the
    /// rebuild publishes the finalized excised state into the neighbors' halos
    /// before the next step's stencils read it.
    fn excise(&self, store: &FieldStore<NDIM, DOF, Mem, Sc>) {
        for _ in 0..self.excise_pass_count(store) {
            self.excise_sweep(store);
        }
        self.excise_finalize(store);
    }

    /// the number of fill passes the excision region needs (0 = excision inert).
    fn excise_pass_count(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>) -> usize {
        0
    }

    /// ONE fill pass (fill + writeback) of the excision region; no-op default.
    fn excise_sweep(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>) {}

    /// the conserved rebuild of the excised cells after the last sweep; no-op default.
    fn excise_finalize(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>) {}
}

/// stash the config's constant-nu viscosity onto a kernel set.
/// a SEPARATE (non-const-generic) trait so the build chain can call it on the
/// concrete set without the `KernelSet<NDIM, DOF, ..>` inference ambiguity. the
/// default is a no-op (regimes with no viscous kernel ignore it);
/// `IsoSubstrateKernelSet` overrides it to store the value.
pub trait WithViscosity: Sized {
    fn with_viscosity(self, _nu: f64) -> Self {
        self
    }
    /// stash the Shakura-Sunyaev alpha. default no-op;
    /// IsoSubstrateKernelSet stores it.
    fn with_alpha(self, _alpha: f64) -> Self {
        self
    }
}

/// stash the config's Ohmic resistivity `eta` onto a kernel set. same non-const-generic shape as
/// [`WithViscosity`]; the default is a no-op (non-MHD regimes have no resistive EMF and ignore it);
/// the MHD kernel sets override it to store the value (the resistive edge EMF + CFL).
pub trait WithResistivity: Sized {
    fn with_resistivity(self, _eta: f64) -> Self {
        self
    }
}

/// stash the horizon-excision radius and initial primitive scales onto a kernel set. same
/// non-const-generic shape as [`WithViscosity`]; the default is a no-op
/// (regimes without the baked excision kernels ignore it);
/// `rhdsubstratekernelset` and the relativistic mhd set use the scales to define a
/// unit-invariant atmosphere below the least dense and least pressurized initial state.
pub trait WithExcision: Sized {
    fn with_excision(self, _r_exc: f64, _rho_scale: f64, _pre_scale: f64) -> Self {
        self
    }
}

// =============================================================================
// classification enums
// =============================================================================

/// Riemann solver selector for the substrate flux dispatch. iso is HLLE-only by
/// physics (no contact wave); the other regimes route between `_hlle` (default,
/// unsuffixed), `_hllc` (contact-resolving, all regimes that have a contact
/// wave), and `_hlld` (RMHD-only 5-wave; the gv-traced version uses
/// `Scalar::iterate_vec` for the 15-step secant on pressure).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Solver {
    Hlle,
    Hllc,
    /// HLLC with the Fleischmann (2020) low-mach / low-dissipation correction (newtonian only).
    HllcLm,
    /// HLLC whose acoustic dissipation is scaled by the ACOUSTIC CONTENT of the face data —
    /// the fraction of the impedance relation `dp = rho c du` the jumps actually carry —
    /// rather than by the local mach number against a reference value. newtonian only.
    /// recovers the `phi ~ Ma` scaling the low-mach asymptotics require, with no tuned
    /// constant, and returns to classical HLLC on any face carrying a real wave.
    HllcAcoustic,
    Hlld,
}

impl Solver {
    /// kernel-name suffix: matches the AOT emit names in `symbi-aot/build.rs`.
    pub fn kernel_suffix(self) -> &'static str {
        match self {
            Solver::Hlle => "",
            Solver::Hllc => "_hllc",
            Solver::HllcLm => "_hllc_lm",
            Solver::HllcAcoustic => "_hllc_acoustic",
            Solver::Hlld => "_hlld",
        }
    }

    /// whether this solver is physically valid for `regime`. ENCODES the runtime `dispatch_flux`
    /// assert as a checkable predicate: HLLE is universal; HLLC resolves a contact wave but no
    /// magnetic structure (non-MHD only); HLLD is the MHD 5-wave solver (MHD only). this is the
    /// matrix `valid_for` validation reads BEFORE building a substrate.
    pub fn valid_for(self, regime: RegimeKind) -> bool {
        match self {
            Solver::Hlle => true,
            // HLLC resolves the gas contact in the normal flux. valid for every regime with a
            // contact-resolving HLLC flux kernel: all hydro/RHD, plus the energy-carrying MHD
            // regimes NMHD and RMHD (the UCT edge EMF reduces to the HLL EMF for B_x != 0 — the
            // contact carries no transverse field, M&DZ p.11 — so HLLC-MHD = HLLC flux + HLL EMF).
            // EXCLUDED: isothermal MHD (no thermal contact, no HLLC flux kernel built).
            // an ISOTHERMAL regime carries no thermal contact wave, so there is no third wave for
            // HLLC to resolve and no HLLC flux kernel is built for it — stated positively here
            // rather than as "not mhd", which admitted isothermal HYDRO and left a selectable
            // solver whose kernel panics at the first dispatch.
            Solver::Hllc => matches!(
                regime,
                RegimeKind::Newtonian
                    | RegimeKind::Rhd
                    | RegimeKind::NewtonianMhd
                    | RegimeKind::Rmhd
            ),
            // HLLC-LM: HLLC with the acoustic dissipation scaled down at low local mach number
            // (Fleischmann, Adami & Adams 2020). available wherever a contact-resolving HLLC flux
            // exists AND the central reformulation its scaling acts on is an identity — which needs
            // the star states to satisfy the jump conditions across both outer waves and the
            // contact. that holds for newtonian euler and for the Mignone-Bodo relativistic star
            // states (both pinned by per-face gates).
            //
            // EXCLUDED: isothermal (no contact wave to resolve, hence no HLLC flux kernel), and the
            // MHD regimes, whose star states carry the null vs non-null normal-field branches — the
            // reformulation has not been shown to be an identity there, and a scaling applied to a
            // non-identity is a different solver rather than a modified one.
            Solver::HllcLm => matches!(regime, RegimeKind::Newtonian | RegimeKind::Rhd),
            // HLLC-ACOUSTIC: the same centralized reformulation as HLLC-LM with a different
            // sensor, so it inherits that arm's requirement exactly. NEWTONIAN ONLY for now:
            // the impedance relation `dp = rho c du` the sensor measures against is the
            // newtonian acoustic one, and its relativistic form carries the specific enthalpy —
            // a different sensor, not the same one on a different state.
            Solver::HllcAcoustic => matches!(regime, RegimeKind::Newtonian),
            Solver::Hlld => regime.is_mhd(),
        }
    }
}

/// the regime FAMILY — the coarse classification a `Regime`'s `SPEC` resolves to, sufficient to
/// validate the (solver, regime) matrix. derived from `Regime::SPEC` via [`RegimeKind::of`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegimeKind {
    Newtonian,
    IsoNewtonian,
    Rhd,
    Rmhd,
    NewtonianMhd,
    IsoMhd,
}

impl RegimeKind {
    /// classify a regime from its compile-time `SPEC`.
    pub fn of<S: Scalar, const D: usize, R: Regime<S, D>>() -> Self {
        match R::SPEC.name {
            "newtonian" => RegimeKind::Newtonian,
            "iso_newtonian" => RegimeKind::IsoNewtonian,
            "rhd" => RegimeKind::Rhd,
            "rmhd" => RegimeKind::Rmhd,
            "newtonian_mhd" => RegimeKind::NewtonianMhd,
            "iso_mhd" => RegimeKind::IsoMhd,
            other => panic!("RegimeKind::of: unknown regime spec name '{other}'"),
        }
    }
    /// whether this family carries a magnetic field (the only distinction the solver matrix needs).
    pub fn is_mhd(self) -> bool {
        matches!(
            self,
            RegimeKind::Rmhd | RegimeKind::NewtonianMhd | RegimeKind::IsoMhd
        )
    }
}

#[cfg(test)]
mod solver_matrix_tests {
    use super::{RegimeKind, Solver};

    // the (solver, regime) validity matrix is what a config is checked against, so it must admit
    // exactly the pairs that have a baked face-flux kernel — no more. a pair it admits without one
    // panics at the first dispatch, after the queue slot is spent;
    // `every_solver_the_matrix_accepts_has_its_face_flux_baked` is the other half of this contract.
    #[test]
    fn valid_for_matrix() {
        // HLLE is universal: two waves, every regime has them.
        for r in [
            RegimeKind::Newtonian,
            RegimeKind::IsoNewtonian,
            RegimeKind::Rhd,
            RegimeKind::Rmhd,
            RegimeKind::NewtonianMhd,
            RegimeKind::IsoMhd,
        ] {
            assert!(Solver::Hlle.valid_for(r), "hlle universal: {r:?}");
        }

        // HLLD is the five-wave MHD solver.
        for r in [
            RegimeKind::Rmhd,
            RegimeKind::NewtonianMhd,
            RegimeKind::IsoMhd,
        ] {
            assert!(Solver::Hlld.valid_for(r), "hlld valid for MHD: {r:?}");
        }
        for r in [
            RegimeKind::Newtonian,
            RegimeKind::IsoNewtonian,
            RegimeKind::Rhd,
        ] {
            assert!(!Solver::Hlld.valid_for(r), "hlld is MHD-only: {r:?}");
        }

        // HLLC resolves a CONTACT wave, so it needs one to exist. an isothermal closure
        // `p = c^2 rho` leaves no independent thermodynamic degree of freedom and hence no entropy
        // wave — the characteristic families are `u +/- c` alone and HLLC degenerates to HLL. that
        // is a property of the closure, not of the magnetic field, so it rules out isothermal HYDRO
        // exactly as it rules out isothermal MHD; no HLLC kernel is baked for either.
        for r in [
            RegimeKind::Newtonian,
            RegimeKind::Rhd,
            RegimeKind::NewtonianMhd,
            RegimeKind::Rmhd,
        ] {
            assert!(Solver::Hllc.valid_for(r), "hllc valid for {r:?}");
        }
        for r in [RegimeKind::IsoNewtonian, RegimeKind::IsoMhd] {
            assert!(
                !Solver::Hllc.valid_for(r),
                "hllc must be refused for {r:?}: an isothermal closure carries no contact wave, and \
                 no isothermal HLLC flux kernel is baked, so admitting it panics at dispatch"
            );
        }

        // HLLC-LM is HLLC plus the low-mach acoustic-dissipation scaling, so it is admissible
        // wherever HLLC is AND the central reformulation the scaling acts on is an identity — which
        // requires the star states to satisfy the jump conditions across both outer waves and the
        // contact. verified for newtonian euler and for the Mignone-Bodo relativistic star states;
        // NOT established for the MHD star states, whose null vs non-null normal-field branches make
        // the reformulation a separate question.
        for r in [RegimeKind::Newtonian, RegimeKind::Rhd] {
            assert!(Solver::HllcLm.valid_for(r), "hllc-lm valid for {r:?}");
            assert!(
                Solver::Hllc.valid_for(r),
                "hllc-lm cannot be admissible where plain hllc is not: {r:?}"
            );
        }
        for r in [
            RegimeKind::IsoNewtonian,
            RegimeKind::IsoMhd,
            RegimeKind::NewtonianMhd,
            RegimeKind::Rmhd,
        ] {
            assert!(
                !Solver::HllcLm.valid_for(r),
                "hllc-lm must be refused for {r:?}"
            );
        }
    }
}
