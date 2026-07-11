// =============================================================================
// substrate_seam.rs
//
// the sim <-> substrate seam: the abstractions the sim core needs to TALK ABOUT
// substrates without depending on any concrete regime KernelSet (docs/design/41).
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
/// `NDIM` = grid dimension, `DOF` = vector (momentum-component) dimension (docs/design/18);
/// they coincide for the natural case and diverge for axisymmetric (DOF>NDIM).
// the kernel-set sees ONLY the `FieldStore` (docs/design/35 R3): its 4 storage params, never
// the physics tags `R`/`M`/`E` or the executor `S` (the concrete set bakes `R::SPEC` /
// `eos_param` at construction — it does not read them off the sim). this is the keystone
// decoupling: 4 params instead of 8, and the energy/schema bounds off `R` (R4) stay LOCAL to
// `FieldStore`. impls keep their `sim` param name (now `&FieldStore`) so bodies are unchanged.
pub trait KernelSet<const NDIM: usize, const DOF: usize, Mem, Sc>
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    fn flux(&self, store: &FieldStore<NDIM, DOF, Mem, Sc>, dir: usize);
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
    /// fall back to the diffusive-but-robust first-order update, not to a pressure floor. host-gated
    /// on a failure reduction, so a clean substage pays only the scan. default: no-op.
    fn fofc(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>, _dt: f64, _a0: f64, _ac: f64, _stage: u8) {}

    /// whether this kernel set runs FOFC (`fofc` is non-trivial). when true the driver also takes the
    /// per-stage `u_stage` snapshot every substage (FOFC restores `cons <- u_stage` to reconstruct the
    /// first-order fluxes from the physical stage input). default: false.
    fn fofc_active(&self) -> bool {
        false
    }

    /// post-godunov hook (e.g., constrained transport for MHD). default: no-op.
    fn post_godunov(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>, _dt: f64, _stage: u8) {}

    /// electric field computation (MHD). default: no-op.
    fn efield(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>) {}

    /// materialize per-cell wave speeds BEFORE the flux reads them (RMHD: the exact quartic
    /// into wave_speed_l/r, so the flux does a cheap Davis fan instead of re-solving the
    /// quartic per face). runs each stage after the prim is current (post c2p+ghost). default:
    /// no-op — regimes with cheap algebraic wave speeds keep computing them inline in the flux.
    fn wave_speeds(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>) {}

    /// forward immersed-body source (gravity, docs/design/19): `cons += dt * S_body`, applied
    /// per RK stage after godunov when the sim has bodies. default: no-op (body-free regimes).
    fn body_source(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>, _dt: f64) {}

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

    /// backward immersed-body feedback (docs/design/19): reduce the per-body force/torque/
    /// accreted-mass from the fluid into the side-car diagnostics, once per step. default: no-op.
    fn body_feedback(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>, _dt: f64) {}

    /// the immersed-boundary penalization (docs/design/50): the property-
    /// algebra surface physics (drain today; walls, porosity, thermal surfaces
    /// as they land), applied post-source each substage — the ONE body
    /// mechanism. default: no-op (regimes without a baked penalize envelope).
    fn penalize(&self, _store: &FieldStore<NDIM, DOF, Mem, Sc>, _dt: f64) {}
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
    Hlld,
}

impl Solver {
    /// kernel-name suffix: matches the AOT emit names in `symbi-aot/build.rs`.
    pub fn kernel_suffix(self) -> &'static str {
        match self {
            Solver::Hlle => "",
            Solver::Hllc => "_hllc",
            Solver::HllcLm => "_hllc_lm",
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
            Solver::Hllc => {
                !regime.is_mhd() || matches!(regime, RegimeKind::NewtonianMhd | RegimeKind::Rmhd)
            }
            // HLLC-LM: the Fleischmann (2020) low-mach / low-dissipation HLLC. emitted for the
            // adiabatic (newtonian euler) flux only -- the LM correction is a non-relativistic gas
            // closure; iso has no contact wave, and the relativistic / mhd HLLC bodies ignore it.
            Solver::HllcLm => matches!(regime, RegimeKind::Newtonian),
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

    // the (solver, regime) validity matrix MUST mirror the dispatch_flux runtime assert:
    // hydro/RHD => HLLE + HLLC valid, HLLD invalid; MHD => HLLE + HLLD valid, HLLC invalid.
    #[test]
    fn valid_for_matrix() {
        let hydro = [
            RegimeKind::Newtonian,
            RegimeKind::IsoNewtonian,
            RegimeKind::Rhd,
        ];
        let mhd = [
            RegimeKind::Rmhd,
            RegimeKind::NewtonianMhd,
            RegimeKind::IsoMhd,
        ];
        for r in hydro {
            assert!(Solver::Hlle.valid_for(r), "hlle universal: {r:?}");
            assert!(Solver::Hllc.valid_for(r), "hllc valid for hydro: {r:?}");
            assert!(!Solver::Hlld.valid_for(r), "hlld is MHD-only: {r:?}");
        }
        for r in mhd {
            assert!(Solver::Hlle.valid_for(r), "hlle universal: {r:?}");
            assert!(Solver::Hlld.valid_for(r), "hlld valid for MHD: {r:?}");
        }
        // HLLC is valid for the energy-carrying MHD regimes (contact-resolving flux + HLL EMF)
        // but NOT isothermal MHD (no thermal contact / no HLLC kernel).
        assert!(
            Solver::Hllc.valid_for(RegimeKind::NewtonianMhd),
            "hllc valid for nmhd"
        );
        assert!(
            Solver::Hllc.valid_for(RegimeKind::Rmhd),
            "hllc valid for rmhd"
        );
        assert!(
            !Solver::Hllc.valid_for(RegimeKind::IsoMhd),
            "hllc invalid for iso-mhd (no contact)"
        );
    }
}
