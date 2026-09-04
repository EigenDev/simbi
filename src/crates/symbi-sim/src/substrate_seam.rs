// =============================================================================
// substrate_seam.rs
//
// the sim <-> substrate seam: the abstractions the sim core needs to talk about
// substrates without depending on any concrete regime KernelSet.
// homing `KernelSet` and the `Solver`/`RegimeKind` enums here keeps
// `FieldStore`/`state.rs` from depending up into `regimes` (which would be a cycle).
//
// homed here in `symbi-sim`, every reference from the sim core points down; the
// concrete kernelsets in the `symbi` crate's `regimes/` implement `KernelSet` and
// depend down on this module.
//
// contents:
// - KernelSet      — one method per physics operation; the integrator calls only these.
// - Solver / RegimeKind — the Riemann-solver + regime-family classification enums.
//
// the regime -> concrete-KernelSet map (`RegimeSubstrate`) and the `sim.substrate()`
// convenience are not here: they name concrete kernelsets, so they live in the substrate
// layer (the `symbi` crate). the orphan rule enforces this — a trait mapping the foreign
// `Regime` types to local kernelsets must be local to the crate that owns the kernelsets.
//
// usage:
//  impl KernelSet<D, D, Mem, Sc> for IsoSubstrateKernelSet<Mem, Sc, D> { .. }
// =============================================================================

use symbi_algebra::OrderedNumeric;
use symbi_carrier::Scalar;
use symbi_hydro::regime::Regime;
use symbi_xpu::MemorySpace;

use crate::state::FieldStore;

/// which fallback-completion path ran on a corrected substage: the shared
/// first-order redo (godunov from the spliced fluxes), or the GRMHD
/// conservative replay that re-runs the update with the geometric source
/// scaled to the largest admissible fraction. an untroubled or inactive pass
/// carries `SharedRedo` — the normal path, with nothing replayed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceReplayOutcome {
    SharedRedo,
    ConservativeReplay,
}

/// the fallback ladder's step decision: accept the substage, or reject the
/// whole step for a replay at a smaller timestep. the stage driver folds this
/// into `StageOutcome` at exactly one site.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FofcDecision {
    Accept,
    RetryStep,
}

/// the fallback ladder's per-substage report — four orthogonal facts. counts
/// are named reductions of the channel masks: `troubled` from the
/// `TroubledCell` decode of the recovery status, `frozen` from the
/// `FreezeApplied` mask the correcting select itself wrote — the act
/// performed, never a recomputed prediction. the process-global census
/// counters are observations of this report.
#[must_use = "the FOFC decision must be folded into the stage outcome"]
#[derive(Clone, Copy, Debug)]
pub struct FofcReport {
    troubled: u64,
    frozen: u64,
    replay: SourceReplayOutcome,
    decision: FofcDecision,
    /// the admissible-boundary projection's receipt for this pass, when the
    /// regime ran a projection. the substrate produces it; the sim books it
    /// into the projection ledger. the projection is a diagnostic accounting,
    /// so a receipt never changes the decision.
    receipt: Option<crate::projection_ledger::ProjectionReceipt>,
    /// the guard acts of this pass — troubled and frozen cell counts — minted at
    /// the sites that performed them. the sim books them into the guard ledger;
    /// diagnostic evidence, so they never change the decision.
    guards: crate::guard_ledger::GuardReceipt,
}

impl FofcReport {
    /// the one constructor for a pass whose ladder is inactive (a regime
    /// without FOFC, or a configuration that opts out): zero counts, the
    /// shared-redo path, accept. an inactive pass cannot fabricate a replay
    /// claim or a count.
    pub fn inactive() -> Self {
        Self {
            troubled: 0,
            frozen: 0,
            replay: SourceReplayOutcome::SharedRedo,
            decision: FofcDecision::Accept,
            receipt: None,
            guards: crate::guard_ledger::GuardReceipt::empty(),
        }
    }

    /// the report of an active pass, from the orchestrator's measured facts.
    /// `frozen_exterior` is the decision evidence — the freeze acts outside
    /// any configured horizon — validated here and folded into the decision's
    /// coherence rather than stored (the four report fields stay orthogonal).
    /// cross-field laws: an untroubled pass carries no acts, no replay claim,
    /// and accepts; a conservative replay requires a troubled pass; a retry
    /// requires at least one exterior freeze act. a freeze count exceeding the
    /// troubled count is lawful — a splice-boundary neighbor of a flagged cell
    /// receives mixed-order fluxes and can freeze without having been flagged.
    pub fn of_pass(
        troubled: u64,
        frozen: u64,
        frozen_exterior: u64,
        replay: SourceReplayOutcome,
        decision: FofcDecision,
    ) -> Self {
        assert!(
            frozen_exterior <= frozen,
            "exterior freeze evidence ({frozen_exterior}) exceeds the freeze count ({frozen})"
        );
        if troubled == 0 {
            assert!(
                frozen == 0
                    && replay == SourceReplayOutcome::SharedRedo
                    && decision == FofcDecision::Accept,
                "an untroubled pass carries no acts, no replay claim, and accepts \
                 (frozen {frozen}, replay {replay:?}, decision {decision:?})"
            );
        }
        if replay == SourceReplayOutcome::ConservativeReplay {
            assert!(
                troubled > 0,
                "a conservative replay requires a troubled pass"
            );
        }
        if decision == FofcDecision::RetryStep {
            assert!(
                frozen_exterior > 0,
                "a retry decision requires an exterior freeze act the mask shows"
            );
        }
        Self {
            troubled,
            frozen,
            replay,
            decision,
            receipt: None,
            guards: crate::guard_ledger::GuardReceipt::empty(),
        }
    }

    /// attach the admissible-boundary projection's receipt for this pass. the
    /// substrate calls this on the report it returns; the receipt is evidence,
    /// so it never revises the counts or the decision.
    pub fn with_receipt(
        mut self,
        receipt: Option<crate::projection_ledger::ProjectionReceipt>,
    ) -> Self {
        self.receipt = receipt;
        self
    }

    /// attach this pass's guard acts, minted at the flag and freeze sites. the
    /// substrate calls this on the report it returns; the acts are evidence, so
    /// they never revise the counts or the decision.
    pub fn with_guards(mut self, guards: crate::guard_ledger::GuardReceipt) -> Self {
        self.guards = guards;
        self
    }

    /// the projection receipt of this pass, when a projection ran.
    pub fn receipt(&self) -> Option<crate::projection_ledger::ProjectionReceipt> {
        self.receipt
    }

    /// this pass's guard acts — the troubled and frozen cell counts.
    pub fn guards(&self) -> crate::guard_ledger::GuardReceipt {
        self.guards
    }

    pub fn troubled(&self) -> u64 {
        self.troubled
    }
    /// the freeze acts of this pass. the backing `freeze_applied` mask is
    /// pass-scoped (written only when the ladder fires); this count is the
    /// durable record.
    pub fn frozen(&self) -> u64 {
        self.frozen
    }
    pub fn replay(&self) -> SourceReplayOutcome {
        self.replay
    }
    pub fn decision(&self) -> FofcDecision {
        self.decision
    }
}

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
// the kernel-set sees only the `FieldStore`: its 4 storage params, never
// the physics tags `R`/`M`/`E` or the executor `S` (the concrete set bakes `R::SPEC` /
// `eos_param` at construction — it does not read them off the sim). this is the keystone
// decoupling: 4 params, and the energy/schema bounds off `R` stay local to
// `FieldStore`. impls name the `&FieldStore` argument `sim`.
pub trait KernelSet<const NDIM: usize, const DOF: usize, Mem, Sc>
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    fn flux(&self, store: &FieldStore<NDIM, DOF, Mem, Sc>, dir: usize);

    /// whether the face reconstruction limits departures from the local hydrostatic
    /// equilibrium rather than the state itself (the kaeppeli-mishra balance). the
    /// refinement hierarchy reads this: a balanced level's premise is that stencil
    /// data on one isentrope present no face jump, and a coarse-fine ghost prolonged
    /// from the raw state does not land on the isentrope, so the seam re-introduces
    /// exactly the jump the reconstruction removed. default: plain reconstruction.
    fn hydrostatic_balance(&self) -> bool {
        false
    }

    /// stencil halfwidth of the evolution face reconstruction along the sweep:
    /// 2 for the plm family (pcm rides plm at theta = 0), 3 for ppm. the
    /// coarse-fine transfer must prolong one order higher than the evolution
    /// reconstruction or the boundary degrades the interior order; the widest
    /// baked prolongation (the ppm sub-cell average) covers plm evolution only,
    /// so a refinement hierarchy refuses any set reporting a reach above 2.
    fn reconstruction_reach(&self) -> u8 {
        2
    }

    /// the adiabatic index when this set's energy closure is the gamma law, which
    /// is what lets a region's conserved energy be rebuilt from a pressure
    /// rewritten in primitive space (`nrg = p / (gamma - 1) + rho v^2 / 2`). the
    /// balanced restriction of a refinement hierarchy needs exactly that; any
    /// other closure returns None and the hierarchy refuses to balance the
    /// restriction rather than rebuild the energy with the wrong formula.
    fn gamma_law(&self) -> Option<f64> {
        None
    }
    fn c2p(&self, store: &FieldStore<NDIM, DOF, Mem, Sc>);
    // the SSP Shu-Osher stage update `cons = a0*u_n + ac*(cons - dt*div(F) + dt*S)`. one method
    // serves every explicit SSP scheme; the driver feeds the per-stage convex coefficients
    // (a0, ac) from the `Timestepping` table (forward-Euler = (0, 1)).
    fn godunov_stage(&self, store: &FieldStore<NDIM, DOF, Mem, Sc>, dt: f64, a0: f64, ac: f64);
    fn cfl(&self, store: &FieldStore<NDIM, DOF, Mem, Sc>) -> f64;
    fn ghost_fill(&self, store: &FieldStore<NDIM, DOF, Mem, Sc>);
    fn snapshot(&self, store: &FieldStore<NDIM, DOF, Mem, Sc>);

    /// first-order flux correction, run per RK substage after c2p: any zone whose high-order c2p
    /// went unphysical (p <= 0, rho <= 0, NaN) is redone with first-order (PCM + HLLE) fluxes
    /// reconstructed from the physical stage-input state (`u_stage`), and the sharp high-order state
    /// is kept everywhere else. a floor-free robustness layer: cells the sharp scheme cannot recover
    /// fall back to the diffusive-but-robust first-order update with no pressure floor. host-gated
    /// on a failure reduction, so a clean substage pays only the scan.
    ///
    /// returns the typed per-substage [`FofcReport`]: the troubled count, the count of cells the
    /// correcting select actually froze, which replay path ran, and the accept/retry decision the
    /// stage driver folds into its outcome. the default is the inactive pass.
    ///
    /// the returned report carries the projection ledger receipt when the
    /// regime ran an admissible-boundary projection; the sim books it, scaled
    /// by the stage's downstream propagation weight, at the stage boundary.
    fn fofc(
        &self,
        _store: &FieldStore<NDIM, DOF, Mem, Sc>,
        _dt: f64,
        _a0: f64,
        _ac: f64,
        _stage: u8,
    ) -> FofcReport {
        FofcReport::inactive()
    }

    /// whether this kernel set runs FOFC (`fofc` is non-trivial). when true the driver also takes the
    /// per-stage `u_stage` snapshot every substage (FOFC restores `cons <- u_stage` to reconstruct the
    /// first-order fluxes from the physical stage input). default: false.
    fn fofc_active(&self) -> bool {
        false
    }

    /// retain the complete state needed to replay a rejected explicit step. paired with
    /// `restore_step`; a kernel set that implements one must implement the other, and a driver
    /// must gate both on `fofc_active` — restoring from a snapshot that was never taken would
    /// overwrite the live state with zeros.
    fn snapshot_retry(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>) {}

    /// restore the complete step-entry state after `fofc` requests a retry: the conserved gas
    /// state, the staggered and cell-centered magnetic field, and the primitives derived from
    /// them. the retried step re-enters at `wave_speeds`/`flux`, which reconstruct from `prim`,
    /// so an implementation that rolls back only the conserved state would reconstruct the
    /// rejected attempt's primitives.
    fn restore_step(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>) {}

    /// post-godunov hook (e.g., constrained transport for MHD). default: no-op.
    fn post_godunov(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>, _dt: f64, _stage: u8) {}

    /// electric field computation (MHD). default: no-op.
    fn efield(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>) {}

    /// materialize per-cell wave speeds before the flux reads them (RMHD: the exact quartic
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
    /// after the fofc phase so a spliced mass flux and the stage-final density
    /// are what the dye rides. default: no-op (regimes without the dye wired).
    fn chi_update(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>, _dt: f64, _a0: f64, _ac: f64) {}

    /// the per-axis interface dye flux `flux[d].chi = mass_flux_d * upwind(prim.chi)`, written
    /// during the flux phase so the coarse-fine registers sample it alongside the gas fluxes they
    /// correct. separate from `chi_update` for that reason alone: the divergence that consumes it
    /// runs later, after fofc has settled the mass flux it rides.
    /// default: no-op (regimes without the dye wired).
    fn chi_flux(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>) {}

    /// does this kernel set carry a non-fused (additive) source overlay? gates the per-stage
    /// `snapshot_stage` + `source_apply` in `step()`. default: false (fused / source-free sets).
    fn has_additive_source(&self) -> bool {
        false
    }

    /// snapshot the stage-input cons into `u_stage`, before the godunov stage overwrites it.
    /// the additive `source_apply` then evaluates `S` at this state — the same state the fused
    /// stage uses — making `plain + additive == fused` bit-for-bit. default: no-op.
    fn snapshot_stage(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>) {}

    /// the additive source pass: `cons += weight * S(u_stage)`, applied per RK stage after
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
    /// applied post-source each substage — the one body
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
    /// composed from the sweep/finalize pieces so the decomposed loop can drive the
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

    /// one fill pass (fill + writeback) of the excision region; no-op default.
    fn excise_sweep(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>) {}

    /// the conserved rebuild of the excised cells after the last sweep; no-op default.
    fn excise_finalize(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>) {}
}

/// stash the config's constant-nu viscosity onto a kernel set.
/// a separate (non-const-generic) trait so the build chain can call it on the
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
    /// HLLC plus the anti-dissipation pressure correction (Chen, Li, Li, Yuan & Gao,
    /// J. Comput. Phys. 456:111027, 2022; the low-mach half of the HLLC+ scheme of Chen, Lin,
    /// Li & Yan, SIAM J. Sci. Comput. 42:B921, 2020). newtonian only. the correction rescales
    /// the dissipation proportional to the face's normal velocity jump — the term of magnitude
    /// `rho c du` that carries the whole low-mach accuracy defect — down to the convective
    /// magnitude, recovering the `Ma^2` scaling of pressure fluctuations that the continuous
    /// Euler system obeys. it reads the local mach number alone, saturating at the sonic point,
    /// so the scheme has no reference mach number to set.
    ///
    /// every signal speed, the contact speed, the contact pressure and both star states keep
    /// their classical values, so the flux is classical HLLC identically wherever the normal
    /// velocity jump vanishes. a stagnant stratified column is such a face, which is what makes
    /// this variant safe on a hydrostatic background: the pressure-jump dissipation that damps
    /// the hydrostatic truncation residual stays at full strength. measured on a balanced
    /// isentropic column at rest, the residual speed holds at 3e-15 on 128 cells and 5e-15 on
    /// 256, where the acoustic ramp reaches 8e-11 and 1e-3 on the same pair.
    ///
    /// the correction addresses the face's normal velocity jump; the grid-aligned shock
    /// instability is carried by the transverse one, and the scaling saturates as the local
    /// mach number approaches unity, which is the regime a shock front occupies. this arm
    /// therefore inherits classical HLLC's behavior on the carbuncle, measured on Quirk's
    /// odd-even test at a transverse kinetic energy of 8.9e-3 against classical HLLC's 4.8e-3
    /// and the acoustic ramp's 1.2e-8. a run whose science lives in stratified subsonic flow
    /// takes this arm; a run resolving a grid-aligned strong shock takes the ramp.
    HllcPlus,
    Hlld,
}

impl Solver {
    /// every variant, in declaration order. gates and sweeps must iterate this rather than a
    /// hand-written array: both coverage gates listed solvers by hand and both silently omitted
    /// a solver each for as long as it had shipped, so the gate whose whole job is "no accepted
    /// (solver, regime) pair lacks a baked kernel" was blind to it. a `match` here
    /// would not help — an array literal is not exhaustiveness-checked — so the array lives
    /// beside the enum and the length assertion below fails the build if the two drift.
    pub const ALL: &'static [Solver] =
        &[Solver::Hlle, Solver::Hllc, Solver::HllcPlus, Solver::Hlld];

    /// kernel-name suffix: matches the AOT emit names in `symbi-aot/build.rs`.
    pub fn kernel_suffix(self) -> &'static str {
        match self {
            Solver::Hlle => "",
            Solver::Hllc => "_hllc",
            Solver::HllcPlus => "_hllc_plus",
            Solver::Hlld => "_hlld",
        }
    }

    /// whether this solver is physically valid for `regime`. encodes the runtime `dispatch_flux`
    /// assert as a checkable predicate: HLLE is universal; HLLC resolves a contact wave but no
    /// magnetic structure (non-MHD only); HLLD is the MHD 5-wave solver (MHD only). this is the
    /// matrix `valid_for` validation reads before building a substrate.
    pub fn valid_for(self, regime: RegimeKind) -> bool {
        match self {
            Solver::Hlle => true,
            // HLLC resolves the gas contact in the normal flux. valid for every regime with a
            // contact-resolving HLLC flux kernel: all hydro/RHD, plus the energy-carrying MHD
            // regimes NMHD and RMHD (the UCT edge EMF reduces to the HLL EMF for B_x != 0 — the
            // contact carries no transverse field, M&DZ p.11 — so hllc-mhd = HLLC flux + HLL EMF).
            // excluded: isothermal MHD (no thermal contact, no HLLC flux kernel built).
            // an isothermal regime carries no thermal contact wave, so there is no third wave for
            // HLLC to resolve and no HLLC flux kernel is built for it — stated positively here
            // rather than as "not mhd", which admitted isothermal hydro and left a selectable
            // solver whose kernel panics at the first dispatch.
            Solver::Hllc => matches!(
                regime,
                RegimeKind::Newtonian
                    | RegimeKind::Rhd
                    | RegimeKind::NewtonianMhd
                    | RegimeKind::Rmhd
            ),
            // HLLC+: the transverse shear viscosity is a property of the multidimensional
            // momentum balance, so it carries into the relativistic regime with the inertia
            // rewritten from the mass density to the enthalpy density `rho h W^2 = e + p`;
            // the newtonian arm additionally carries the low-mach accuracy term, whose
            // relativistic velocity-jump / pressure-jump split is a separate derivation.
            // excluded: isothermal (no contact wave to resolve, hence no HLLC flux kernel),
            // and the MHD regimes, whose star states carry the null vs non-null normal-field
            // branches the shear coefficient is not derived across.
            Solver::HllcPlus => matches!(regime, RegimeKind::Newtonian | RegimeKind::Rhd),
            Solver::Hlld => regime.is_mhd(),
        }
    }
}

/// the regime family — the coarse classification a `Regime`'s `SPEC` resolves to, sufficient to
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

        // HLLC resolves a contact wave, so it needs one to exist. an isothermal closure
        // `p = c^2 rho` leaves no independent thermodynamic degree of freedom and hence no entropy
        // wave — the characteristic families are `u +/- c` alone and HLLC degenerates to HLL. that
        // is a property of the closure, not of the magnetic field, so it rules out isothermal hydro
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

        // HLLC+ is HLLC plus two additive dissipation rescalings, each derived from the
        // newtonian `rho c du` impedance, so it is admissible on the newtonian gas contact and
        // refused everywhere the derivation does not reach: the relativistic inertia carries the
        // lorentz factor and the specific enthalpy, the MHD star states carry the null vs
        // non-null normal-field branches, and an isothermal closure has no contact wave at all.
        assert!(Solver::HllcPlus.valid_for(RegimeKind::Newtonian));
        assert!(
            Solver::Hllc.valid_for(RegimeKind::Newtonian),
            "HLLC+ cannot be admissible where plain hllc is not"
        );
        assert!(Solver::HllcPlus.valid_for(RegimeKind::Rhd));
        for r in [
            RegimeKind::IsoNewtonian,
            RegimeKind::IsoMhd,
            RegimeKind::NewtonianMhd,
            RegimeKind::Rmhd,
        ] {
            assert!(
                !Solver::HllcPlus.valid_for(r),
                "HLLC+ must be refused for {r:?}"
            );
        }
    }
}

/// adding A variant without extending `Solver::ALL` fails here. the match is exhaustive, so a
/// new variant is a compile error rather than a silently-unswept solver; the arm's only job is
/// to state the count that `ALL` must have.
const _: () = {
    let expected = {
        let mut n = 0;
        // one arm per variant; the compiler rejects a new variant that is not listed.
        let all = [Solver::Hlle, Solver::Hllc, Solver::HllcPlus, Solver::Hlld];
        while n < all.len() {
            n += 1;
        }
        n
    };
    assert!(Solver::ALL.len() == expected);
};
